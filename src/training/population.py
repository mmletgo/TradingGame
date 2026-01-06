"""种群管理模块

管理特定类型 Agent 的种群，支持从 NEAT 基因组创建 Agent。
"""

import ctypes
import gc
import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import neat
import numpy as np

from src.bio.agents.base import Agent


def malloc_trim() -> None:
    """调用 glibc 的 malloc_trim 将释放的内存归还给操作系统

    Python 的内存分配器通常不会将释放的内存归还给 OS，而是保留在内部
    供将来使用。这会导致 VmRSS 持续增长。通过调用 malloc_trim(0)，
    可以强制将未使用的内存页归还给操作系统。

    仅在 Linux 系统上有效。
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        # 非 Linux 系统或无法加载 libc，静默跳过
        pass
from src.bio.agents.market_maker import MarketMakerAgent
from src.bio.agents.retail import RetailAgent
from src.bio.agents.retail_pro import RetailProAgent
from src.bio.agents.whale import WhaleAgent
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, AgentType, Config
from src.core.log_engine.logger import get_logger


def _get_memory_mb() -> float:
    """获取当前进程的内存使用量（MB）"""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:
        pass
    return 0.0


class Population:
    """种群管理类

    管理特定类型的 Agent 种群，包括创建、评估、淘汰和繁殖。

    Attributes:
        agent_type: Agent 类型（散户/庄家/做市商）
        agents: Agent 列表
        neat_pop: NEAT 种群对象
        neat_config: NEAT 配置
        agent_config: Agent 配置
        generation: 当前代数
        logger: 日志器
    """

    agent_type: AgentType
    agents: list[Agent]
    neat_pop: neat.Population
    neat_config: neat.Config
    agent_config: AgentConfig
    generation: int
    logger: logging.Logger
    _executor: ThreadPoolExecutor | None
    _num_workers: int

    def _get_executor(self) -> ThreadPoolExecutor:
        """获取实例级别线程池"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._num_workers,
                thread_name_prefix=f"pop_{self.agent_type.value}"
            )
        return self._executor

    def shutdown_executor(self) -> None:
        """关闭实例级别线程池"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __init__(self, agent_type: AgentType, config: Config) -> None:
        """创建种群

        初始化 NEAT 种群，创建初始 Agent 列表。

        Args:
            agent_type: Agent 类型（散户/庄家/做市商）
            config: 全局配置对象
        """
        self.agent_type = agent_type
        self.agent_config = config.agents[agent_type]
        self.generation = 0
        self.logger = get_logger("population")
        self._executor = None
        self._num_workers = 8

        # 根据 Agent 类型选择 NEAT 配置文件
        # 散户使用 67 个输入（10档订单簿 + 10笔成交），庄家使用 607 个输入，做市商使用 634 个输入
        from pathlib import Path

        config_dir = Path(config.training.neat_config_path)
        if agent_type == AgentType.MARKET_MAKER:
            neat_config_path = config_dir / "neat_market_maker.cfg"
        elif agent_type == AgentType.WHALE:
            neat_config_path = config_dir / "neat_whale.cfg"
        elif agent_type == AgentType.RETAIL_PRO:
            neat_config_path = config_dir / "neat_retail_pro.cfg"
        else:
            neat_config_path = config_dir / "neat_retail.cfg"

        # 加载 NEAT 配置
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(neat_config_path),
        )

        # 动态设置 pop_size 为 AgentConfig.count
        self.neat_config.pop_size = self.agent_config.count

        # 创建 NEAT 种群
        self.neat_pop = neat.Population(self.neat_config)

        # 获取初始基因组并创建 Agent
        genomes = list(self.neat_pop.population.items())
        self.agents = self.create_agents(genomes)

        self.logger.info(
            f"创建 {agent_type.value} 种群，初始 Agent 数量: {len(self.agents)}"
        )

    # Agent ID 偏移量，确保不同种群的 agent_id 不冲突
    # 每个种群类型使用不同的高位，每种群最多支持 100 万个 Agent
    _AGENT_ID_OFFSET = {
        AgentType.RETAIL: 0,
        AgentType.RETAIL_PRO: 1_000_000,
        AgentType.WHALE: 2_000_000,
        AgentType.MARKET_MAKER: 3_000_000,
    }

    def _create_single_agent(
        self,
        idx: int,
        genome_id: int,
        genome: neat.DefaultGenome,
        agent_class: type[Agent],
        agent_id_offset: int,
    ) -> tuple[int, Agent]:
        """创建单个 Agent（线程安全）

        Args:
            idx: Agent 在列表中的索引
            genome_id: 基因组 ID
            genome: NEAT 基因组对象
            agent_class: Agent 类（RetailAgent/WhaleAgent/MarketMakerAgent）
            agent_id_offset: Agent ID 偏移量

        Returns:
            (索引, Agent) 元组
        """
        brain = Brain.from_genome(genome, self.neat_config)
        unique_agent_id = agent_id_offset + idx
        agent = agent_class(unique_agent_id, brain, self.agent_config)
        return (idx, agent)

    def create_agents(
        self,
        genomes: list[tuple[int, neat.DefaultGenome]],
    ) -> list[Agent]:
        """从基因组创建 Agent 列表（并行化版本）

        遍历基因组列表，为每个基因组创建对应的 Brain 和 Agent。
        根据种群的 agent_type 创建对应类型的 Agent（散户/庄家/做市商）。
        小批量（<50）串行处理，大批量并行处理以提升性能。

        Args:
            genomes: NEAT 基因组列表，每项为 (genome_id, genome) 元组

        Returns:
            创建的 Agent 列表
        """
        # 确定 Agent 类
        if self.agent_type == AgentType.RETAIL:
            agent_class: type[Agent] = RetailAgent
        elif self.agent_type == AgentType.RETAIL_PRO:
            agent_class = RetailProAgent
        elif self.agent_type == AgentType.WHALE:
            agent_class = WhaleAgent
        elif self.agent_type == AgentType.MARKET_MAKER:
            agent_class = MarketMakerAgent
        else:
            raise ValueError(f"未知的 Agent 类型: {self.agent_type}")

        agent_id_offset = self._AGENT_ID_OFFSET.get(self.agent_type, 0)

        # 小批量直接串行处理，避免线程池开销
        if len(genomes) < 50:
            agents: list[Agent] = []
            for idx, (genome_id, genome) in enumerate(genomes):
                brain = Brain.from_genome(genome, self.neat_config)
                unique_agent_id = agent_id_offset + idx
                agent = agent_class(unique_agent_id, brain, self.agent_config)
                agents.append(agent)
            return agents

        # 大批量并行创建
        executor = self._get_executor()
        futures: dict[Future[tuple[int, Agent]], int] = {}

        for idx, (genome_id, genome) in enumerate(genomes):
            future = executor.submit(
                self._create_single_agent,
                idx,
                genome_id,
                genome,
                agent_class,
                agent_id_offset,
            )
            futures[future] = idx

        # 收集结果，按索引排序
        results: list[tuple[int, Agent]] = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                raise RuntimeError(f"创建 Agent 失败: {e}") from e

        results.sort(key=lambda x: x[0])
        return [agent for _, agent in results]

    def evaluate(self, current_price: float) -> list[tuple[Agent, float]]:
        """评估种群适应度

        使用向量化运算计算所有 Agent 的适应度，并按适应度从高到低排序。

        Args:
            current_price: 当前市场价格，用于计算未实现盈亏

        Returns:
            按适应度从高到低排序的 (Agent, 适应度) 元组列表
        """
        n = len(self.agents)
        if n == 0:
            return []

        # 1. 收集所有 Agent 的账户数据到 numpy 数组
        balances = np.array([a.account.balance for a in self.agents])
        quantities = np.array([a.account.position.quantity for a in self.agents])
        avg_prices = np.array([a.account.position.avg_price for a in self.agents])
        initial_balances = np.array([a.account.initial_balance for a in self.agents])

        # 2. 向量化计算未实现盈亏: (current_price - avg_price) * quantity
        unrealized_pnl = (current_price - avg_prices) * quantities

        # 3. 向量化计算净值: balance + unrealized_pnl
        equities = balances + unrealized_pnl

        # 4. 向量化计算适应度: equity / initial_balance
        fitnesses = equities / initial_balances

        # 5. 获取从高到低的排序索引
        sorted_indices = np.argsort(fitnesses)[::-1]

        # 6. 按排序索引构建结果
        return [(self.agents[i], float(fitnesses[i])) for i in sorted_indices]

    def evolve(self, current_price: float) -> None:
        """进化种群

        使用 NEAT 算法进行一代进化：
        1. 使用向量化 evaluate 计算所有 Agent 适应度
        2. 通过 agent -> genome 映射为基因组设置适应度
        3. 清理旧 Agent 对象（在 neat_pop.run 之前，断开对旧 genome 的引用）
        4. 调用 NEAT 种群的 run 方法进行一代进化
        5. 清理 NEAT 历史数据
        6. 从新基因组重建 Agent 列表

        当 NEAT 进化失败（如种群灭绝）时，自动重置种群并重新开始。

        关键优化：在 neat_pop.run() 之前先清理旧 Agent，这样旧的 genome 对象
        就没有外部引用了，当 neat_pop.run() 替换 population 时，旧的 genome
        可以被 GC 立即回收，避免内存泄漏。

        Args:
            current_price: 当前市场价格，用于计算适应度
        """
        # [MEMORY] 记录进化开始时内存
        mem_start = _get_memory_mb()

        # 1. 使用向量化 evaluate 获取所有 Agent 的适应度
        agent_fitnesses = self.evaluate(current_price)

        # 2. 通过 agent -> genome 映射为基因组设置适应度
        for agent, fitness in agent_fitnesses:
            genome = agent.brain.get_genome()
            genome.fitness = fitness  # type: ignore[assignment]

        # 3. 【关键】在 neat_pop.run() 之前：
        # - 保存旧基因组引用（用于后续清理其内部数据）
        # - 清理旧 Agent 对象（断开 Agent -> Brain -> genome 引用链）
        mem_before_cleanup = _get_memory_mb()

        # 保存旧基因组引用，用于进化后清理其内部数据
        old_genomes = list(self.neat_pop.population.values())

        self._cleanup_old_agents()
        # 强制 GC，回收旧 Agent 和 Brain 对象
        gc.collect()
        gc.collect()
        mem_after_cleanup = _get_memory_mb()

        # 4. 调用 NEAT 进化（eval_genomes 留空，适应度已设置）
        def eval_genomes(
            _genomes: list[tuple[int, neat.DefaultGenome]], _config: neat.Config
        ) -> None:
            # 适应度已经在上面设置好了，这里不需要再计算
            pass

        try:
            self.neat_pop.run(eval_genomes, n=1)
        except RuntimeError as e:
            # NEAT 进化失败（通常是因为种群灭绝或无法繁殖足够后代）
            # 重置种群并重新开始
            self.logger.warning(
                f"{self.agent_type.value} 种群进化失败: {e}，正在重置种群..."
            )
            # 清理旧基因组数据
            self._cleanup_genome_internals(old_genomes)
            del old_genomes
            gc.collect()
            gc.collect()
            malloc_trim()
            self._reset_neat_population()
            return

        # 5. 【关键】清理旧基因组的内部数据
        # NEAT 进化后，新基因组已经创建，旧基因组不再需要
        # 但旧基因组的 connections 和 nodes 字典仍占用大量内存
        # 注意：由于精英保留（elitism），某些旧基因组可能仍在新种群中
        # 只清理不在新种群中的旧基因组
        new_genome_ids = set(self.neat_pop.population.keys())
        old_genomes_to_clean = [
            g for g in old_genomes if g.key not in new_genome_ids
        ]
        self._cleanup_genome_internals(old_genomes_to_clean)
        del old_genomes
        del old_genomes_to_clean

        # 强制 GC 并调用 malloc_trim 将内存归还给操作系统
        gc.collect()
        gc.collect()
        malloc_trim()

        # [MEMORY] 记录 NEAT run 后内存
        mem_after_neat = _get_memory_mb()

        # 6. 增加代数计数
        self.generation += 1

        # 7. 清理 NEAT 种群中的历史数据，防止内存泄漏
        mem_before_neat_cleanup = _get_memory_mb()
        self._cleanup_neat_history()
        # 清理后立即 GC 并 malloc_trim
        gc.collect()
        malloc_trim()
        mem_after_neat_cleanup = _get_memory_mb()

        # 7. 从新基因组重建 Agent 列表
        mem_before_create = _get_memory_mb()
        new_genomes = list(self.neat_pop.population.items())
        self.agents = self.create_agents(new_genomes)
        mem_after_create = _get_memory_mb()

        # [MEMORY] 输出详细的内存变化日志
        mem_end = _get_memory_mb()
        self.logger.info(
            f"[MEMORY_EVOLVE] {self.agent_type.value} gen_{self.generation}: "
            f"agent_cleanup={mem_after_cleanup - mem_before_cleanup:+.1f}MB, "
            f"neat_run=+{mem_after_neat - mem_after_cleanup:.1f}MB, "
            f"neat_cleanup={mem_after_neat_cleanup - mem_before_neat_cleanup:+.1f}MB, "
            f"agent_create=+{mem_after_create - mem_before_create:.1f}MB, "
            f"total={mem_end - mem_start:+.1f}MB"
        )

        self.logger.info(
            f"{self.agent_type.value} 种群完成第 {self.generation} 代进化，"
            f"Agent 数量: {len(self.agents)}"
        )

    def _cleanup_genome_internals(self, genomes: list[neat.DefaultGenome]) -> None:
        """清理基因组内部数据结构

        NEAT 基因组包含 connections 和 nodes 字典，每个都存储大量的 Gene 对象。
        对于大种群（如 10000 个散户），这些数据结构占用大量内存。
        当基因组不再需要时，显式清理这些数据结构可以让 GC 更快回收内存。

        Args:
            genomes: 需要清理的基因组列表
        """
        for genome in genomes:
            # 清理 connections 字典（ConnectionGene 对象）
            if hasattr(genome, 'connections') and genome.connections is not None:
                genome.connections.clear()
            # 清理 nodes 字典（NodeGene 对象）
            if hasattr(genome, 'nodes') and genome.nodes is not None:
                genome.nodes.clear()
            # 清理 fitness（虽然这是个数值，但为了彻底性也清理）
            genome.fitness = None  # type: ignore[assignment]

    def _cleanup_old_agents(self) -> None:
        """清理旧 Agent 对象

        显式打破循环引用，帮助垃圾回收器及时回收内存。
        必须在 neat_pop.run() 之前调用，以确保旧 genome 对象可以被立即回收。

        清理策略：
        1. 先收集所有需要清理的对象引用
        2. 断开 Brain -> genome/network/config 的引用
        3. 断开 Agent -> brain/account 的引用
        4. 清空 agents 列表
        5. 删除本地引用并强制 GC
        """
        # 收集所有需要清理的 Brain 和 Network 对象
        brains_to_cleanup = []
        networks_to_cleanup = []
        accounts_to_cleanup = []

        for agent in self.agents:
            if hasattr(agent, 'brain') and agent.brain is not None:
                brains_to_cleanup.append(agent.brain)
                if hasattr(agent.brain, 'network') and agent.brain.network is not None:
                    networks_to_cleanup.append(agent.brain.network)
            if hasattr(agent, 'account') and agent.account is not None:
                accounts_to_cleanup.append(agent.account)

        # 清理 Network 内部状态（先清理最内层）
        for network in networks_to_cleanup:
            if hasattr(network, 'node_evals'):
                network.node_evals = None  # type: ignore[assignment]
            if hasattr(network, 'values'):
                network.values = None  # type: ignore[assignment]
            if hasattr(network, 'input_nodes'):
                network.input_nodes = None  # type: ignore[assignment]
            if hasattr(network, 'output_nodes'):
                network.output_nodes = None  # type: ignore[assignment]
            # Cython FastFeedForwardNetwork 可能有额外属性
            if hasattr(network, 'active'):
                network.active = None  # type: ignore[assignment]

        # 清理 Brain 引用
        for brain in brains_to_cleanup:
            brain.genome = None  # type: ignore[assignment]
            brain.network = None  # type: ignore[assignment]
            brain.config = None  # type: ignore[assignment]

        # 清理 Account 引用
        for account in accounts_to_cleanup:
            if hasattr(account, 'position'):
                account.position = None  # type: ignore[assignment]

        # 断开 Agent 的引用
        for agent in self.agents:
            agent.brain = None  # type: ignore[assignment]
            agent.account = None  # type: ignore[assignment]
            # 清理 Agent 可能持有的其他引用
            if hasattr(agent, '_input_buffer'):
                agent._input_buffer = None  # type: ignore[assignment]

        # 清空列表
        self.agents.clear()

        # 删除本地引用
        del brains_to_cleanup
        del networks_to_cleanup
        del accounts_to_cleanup

        # 注意：这里不调用 gc.collect()，让调用方决定何时 GC
        # 这样可以批量处理多个种群后再统一 GC，提高效率

    def _cleanup_neat_history(self) -> None:
        """清理 NEAT 种群中的历史数据

        防止 species_set 和其他历史数据无限增长导致内存泄漏。
        这是多竞技场模式内存泄漏的主要修复点。
        """
        current_genome_ids = set(self.neat_pop.population.keys())
        stats = {}  # 用于收集清理统计

        # 0. 清理 best_genome 引用（如果它不在当前种群中）
        # best_genome 会持续持有历史最优基因组的引用，可能导致内存泄漏
        if hasattr(self.neat_pop, 'best_genome') and self.neat_pop.best_genome is not None:
            if self.neat_pop.best_genome.key not in current_genome_ids:
                # best_genome 已经不在当前种群中，可以清理
                # 但为了保持 NEAT 功能正常，我们用当前种群中最优的替代
                best_in_current = max(
                    self.neat_pop.population.values(),
                    key=lambda g: g.fitness if g.fitness is not None else float('-inf')
                )
                self.neat_pop.best_genome = best_in_current
                stats['best_genome'] = 'updated'

        # 1. 清理 species_set 中的历史成员信息
        if hasattr(self.neat_pop, 'species') and self.neat_pop.species is not None:
            species_set = self.neat_pop.species

            # 清理 genome_to_species 映射（关键！这是内存泄漏的主要来源）
            if hasattr(species_set, 'genome_to_species'):
                old_size = len(species_set.genome_to_species)
                species_set.genome_to_species = {
                    gid: sid
                    for gid, sid in species_set.genome_to_species.items()
                    if gid in current_genome_ids
                }
                new_size = len(species_set.genome_to_species)
                stats['genome_to_species'] = f"{old_size}->{new_size}"

            # 清理每个物种的历史成员列表
            if hasattr(species_set, 'species'):
                total_members_old = 0
                total_members_new = 0
                for sid, species in list(species_set.species.items()):
                    # 清理 members 字典中可能积累的旧引用
                    if hasattr(species, 'members') and species.members:
                        total_members_old += len(species.members)
                        species.members = {
                            gid: genome
                            for gid, genome in species.members.items()
                            if gid in current_genome_ids
                        }
                        total_members_new += len(species.members)
                    # 清理 fitness_history（如果存在）
                    if hasattr(species, 'fitness_history'):
                        if len(species.fitness_history) > 5:
                            species.fitness_history = species.fitness_history[-5:]
                    # 清理 representative（如果不在当前种群中）
                    if hasattr(species, 'representative'):
                        if species.representative is not None:
                            if species.representative.key not in current_genome_ids:
                                # 用当前成员中的第一个作为代表
                                if species.members:
                                    species.representative = next(iter(species.members.values()))
                stats['species_members'] = f"{total_members_old}->{total_members_new}"
                stats['species_count'] = len(species_set.species)

        # 2. 清理 stagnation 中的历史数据
        if hasattr(self.neat_pop, 'stagnation') and self.neat_pop.stagnation is not None:
            stagnation = self.neat_pop.stagnation
            # 清理 species_fitness 历史
            if hasattr(stagnation, 'species_fitness'):
                old_len = len(stagnation.species_fitness)
                stagnation.species_fitness = {}
                stats['stagnation_fitness'] = f"{old_len}->0"

        # 3. 清理 reproduction 中的历史数据
        if hasattr(self.neat_pop, 'reproduction') and self.neat_pop.reproduction is not None:
            reproduction = self.neat_pop.reproduction
            # 清理 ancestors（祖先引用，这是另一个内存泄漏来源）
            if hasattr(reproduction, 'ancestors'):
                old_len = len(reproduction.ancestors)
                reproduction.ancestors = {}
                stats['ancestors'] = f"{old_len}->0"
            # 清理 genome_indexer 的历史（如果需要）
            # genome_indexer 只是一个计数器，不需要清理

        # 4. 清理 reporters 中可能积累的数据
        if hasattr(self.neat_pop, 'reporters') and self.neat_pop.reporters is not None:
            reporters = self.neat_pop.reporters
            if hasattr(reporters, 'reporters'):
                for reporter in reporters.reporters:
                    # StdOutReporter 等可能有 generation_statistics
                    if hasattr(reporter, 'generation_statistics'):
                        if len(reporter.generation_statistics) > 5:
                            reporter.generation_statistics = reporter.generation_statistics[-5:]
                    # 清理 species_statistics
                    if hasattr(reporter, 'species_statistics'):
                        if len(reporter.species_statistics) > 5:
                            reporter.species_statistics = reporter.species_statistics[-5:]
                    # 清理 most_fit_genomes
                    if hasattr(reporter, 'most_fit_genomes'):
                        if len(reporter.most_fit_genomes) > 5:
                            reporter.most_fit_genomes = reporter.most_fit_genomes[-5:]

        # 5. 清理 config 中可能缓存的引用
        # neat.Config 对象可能持有一些缓存
        if hasattr(self.neat_pop, 'config') and self.neat_pop.config is not None:
            config = self.neat_pop.config
            # 清理 genome_config 中可能的缓存
            if hasattr(config, 'genome_config'):
                genome_config = config.genome_config
                # 某些自定义实现可能有缓存
                if hasattr(genome_config, '_cache'):
                    genome_config._cache = {}

        # [MEMORY] 输出 NEAT 清理统计
        if stats:
            stats_str = ", ".join(f"{k}={v}" for k, v in stats.items())
            self.logger.info(
                f"[MEMORY_NEAT_CLEANUP] {self.agent_type.value}: {stats_str}"
            )

    def _reset_neat_population(self) -> None:
        """重置 NEAT 种群

        当进化失败时调用，创建一个全新的随机种群。
        """
        # 先清理旧的 Agent 对象
        self._cleanup_old_agents()

        self.neat_pop = neat.Population(self.neat_config)
        genomes = list(self.neat_pop.population.items())
        self.agents = self.create_agents(genomes)
        # 重置后代数不变，表示这是一次意外重置
        self.logger.info(
            f"{self.agent_type.value} 种群已重置，新 Agent 数量: {len(self.agents)}"
        )

    def reset_agents(self) -> None:
        """重置所有 Agent 的账户状态

        在 episode 开始时调用，将所有 Agent 的账户恢复到初始状态。
        """
        for agent in self.agents:
            agent.reset(self.agent_config)

    def replace_worst_agents(
        self,
        new_genomes: list[neat.DefaultGenome],
    ) -> list[tuple[int, Agent, Agent]]:
        """增量替换最差的 Agent（不重建整个种群）

        优化点：只创建/替换需要的 Agent，避免重建整个种群。

        Args:
            new_genomes: 要注入的新 genome 列表（已设置 fitness 和 key）

        Returns:
            被替换的 (索引, 旧Agent, 新Agent) 列表
        """
        if not new_genomes:
            return []

        n_to_replace = len(new_genomes)
        neat_pop = self.neat_pop

        # 1. 按 genome.fitness 找到当前种群中最差的 N 个 agent
        # 构建 (agent_idx, genome_id, fitness) 列表
        agent_genome_fitness: list[tuple[int, int, float]] = []
        for idx, agent in enumerate(self.agents):
            genome = agent.brain.get_genome()
            fitness = genome.fitness if genome.fitness is not None else float('-inf')
            agent_genome_fitness.append((idx, genome.key, fitness))

        # 按 fitness 升序排序（最差的在前面）
        agent_genome_fitness.sort(key=lambda x: x[2])

        # 取最差的 N 个
        worst_entries = agent_genome_fitness[:n_to_replace]

        # 2. 收集这些 agent 的索引和对应的 genome_id
        worst_indices: list[int] = [entry[0] for entry in worst_entries]
        worst_genome_ids: list[int] = [entry[1] for entry in worst_entries]

        # 3. 从 neat_pop.population 中删除这些旧 genome，添加新 genome
        # 先删除旧的
        for old_genome_id in worst_genome_ids:
            if old_genome_id in neat_pop.population:
                del neat_pop.population[old_genome_id]

        # 为新 genome 分配唯一 key 并添加到种群
        max_key = max(neat_pop.population.keys()) if neat_pop.population else 0
        for i, genome in enumerate(new_genomes):
            new_key = max_key + 1 + i
            genome.key = new_key
            neat_pop.population[new_key] = genome

        # 4. 只为新 genome 创建新的 Agent 对象
        # 确定 Agent 类
        if self.agent_type == AgentType.RETAIL:
            agent_class: type[Agent] = RetailAgent
        elif self.agent_type == AgentType.RETAIL_PRO:
            agent_class = RetailProAgent
        elif self.agent_type == AgentType.WHALE:
            agent_class = WhaleAgent
        elif self.agent_type == AgentType.MARKET_MAKER:
            agent_class = MarketMakerAgent
        else:
            raise ValueError(f"未知的 Agent 类型: {self.agent_type}")

        agent_id_offset = self._AGENT_ID_OFFSET.get(self.agent_type, 0)

        # 构建替换信息列表
        replaced_info: list[tuple[int, Agent, Agent]] = []

        for i, genome in enumerate(new_genomes):
            idx = worst_indices[i]
            old_agent = self.agents[idx]

            # 5. 创建新的 Agent 对象
            _, new_agent = self._create_single_agent(
                idx,
                genome.key,
                genome,
                agent_class,
                agent_id_offset,
            )

            # 6. 清理被替换的旧 Agent 对象（参考 _cleanup_old_agents 的实现）
            if hasattr(old_agent, 'brain') and old_agent.brain is not None:
                brain = old_agent.brain
                # 清理 Network 内部状态
                if hasattr(brain, 'network') and brain.network is not None:
                    network = brain.network
                    if hasattr(network, 'node_evals'):
                        network.node_evals = None  # type: ignore[assignment]
                    if hasattr(network, 'values'):
                        network.values = None  # type: ignore[assignment]
                    if hasattr(network, 'input_nodes'):
                        network.input_nodes = None  # type: ignore[assignment]
                    if hasattr(network, 'output_nodes'):
                        network.output_nodes = None  # type: ignore[assignment]
                # 清理 Brain 引用
                brain.genome = None  # type: ignore[assignment]
                brain.network = None  # type: ignore[assignment]
                brain.config = None  # type: ignore[assignment]
                old_agent.brain = None  # type: ignore[assignment]
            if hasattr(old_agent, 'account') and old_agent.account is not None:
                if hasattr(old_agent.account, 'position'):
                    old_agent.account.position = None  # type: ignore[assignment]
                old_agent.account = None  # type: ignore[assignment]
            if hasattr(old_agent, '_input_buffer'):
                old_agent._input_buffer = None  # type: ignore[assignment]

            # 更新 self.agents[idx] 为新创建的 Agent
            self.agents[idx] = new_agent

            replaced_info.append((idx, old_agent, new_agent))

        # 7. 调用 speciate() 重新划分物种
        neat_pop.species.speciate(
            neat_pop.config,
            neat_pop.population,
            neat_pop.generation,
        )

        self.logger.info(
            f"{self.agent_type.value} 种群增量替换了 {n_to_replace} 个最差 Agent"
        )

        # 8. 返回被替换的信息列表
        return replaced_info
