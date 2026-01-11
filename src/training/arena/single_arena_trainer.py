"""单进程多竞技场训练器模块

单进程串行运行多个竞技场，充分利用 OpenMP 多核并行推理。
"""

import gc
import gzip
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.bio.agents.base import AgentType
from src.config.config import Config
from src.core.log_engine.logger import get_logger
from .fitness_aggregator import FitnessAggregator
from .multi_arena_trainer import MultiArenaConfig
from src.training.population import (
    MultiPopulationWorkerPool,
    Population,
    SubPopulationManager,
    WorkerConfig,
    _deserialize_genomes_numpy,
    _unpack_network_params_numpy,
    malloc_trim,
)
from src.training.trainer import Trainer


class SingleArenaTrainer:
    """单进程多竞技场训练器

    复用同一个 Trainer 实例，串行运行 N 个竞技场（每个竞技场 M 个 episode），
    累积适应度后用 FitnessAggregator 汇总，保留多进程进化 Worker 池。

    与 MultiArenaTrainer 的主要区别：
    - 竞技场执行：单进程串行（而非多进程并行）
    - OpenMP 线程：充分利用所有 CPU 核心（而非多进程竞争）
    - 进程通讯：无（而非需要序列化）
    - 内存占用：低（1 份种群而非 N 份）

    Attributes:
        config: 全局配置
        multi_config: 多竞技场配置
        trainer: Trainer 实例（复用）
        evolution_worker_pool: 进化 Worker 池
        generation: 当前代数
        total_episodes: 总 episode 数
    """

    config: Config
    multi_config: MultiArenaConfig
    trainer: Trainer | None
    evolution_worker_pool: MultiPopulationWorkerPool | None
    generation: int
    total_episodes: int

    def __init__(
        self,
        config: Config,
        multi_config: MultiArenaConfig | None = None,
    ) -> None:
        """初始化单进程多竞技场训练器

        Args:
            config: 全局配置
            multi_config: 多竞技场配置，None 使用默认配置
        """
        self.config = config
        self.multi_config = multi_config or MultiArenaConfig()
        self.logger = get_logger("single_arena_trainer")

        self.trainer = None
        self.evolution_worker_pool = None
        self.generation = 0
        self.total_episodes = 0

        self._is_setup = False
        self._is_running = False
        self._worker_pool_synced = False

    @property
    def populations(self) -> dict[AgentType, Population | SubPopulationManager]:
        """获取种群字典（便于与 MultiArenaTrainer 接口兼容）"""
        if self.trainer is None:
            return {}
        return self.trainer.populations

    def setup(self) -> None:
        """初始化训练环境

        1. 创建 Trainer 实例（包含 NEAT 种群、撮合引擎等）
        2. 创建进化 Worker 池（MultiPopulationWorkerPool）
        """
        if self._is_setup:
            self.logger.warning("训练环境已初始化，跳过重复初始化")
            return

        self.logger.info("开始初始化单进程多竞技场训练环境...")

        # 1. 创建 Trainer 实例
        self.trainer = Trainer(self.config)
        self.trainer.setup()
        self.trainer.is_running = True  # 标记 Trainer 运行中

        # 2. 创建进化 Worker 池
        self._create_evolution_worker_pool()

        self._is_setup = True
        self.logger.info(
            f"单进程多竞技场训练环境初始化完成: "
            f"{self.multi_config.num_arenas} 个竞技场, "
            f"每竞技场 {self.multi_config.episodes_per_arena} 个 episode"
        )

    def _create_evolution_worker_pool(self) -> None:
        """创建进化 Worker 池

        Worker 配置：
        - RETAIL: 10 Workers（每个子种群一个）
        - RETAIL_PRO: 1 Worker
        - WHALE: 1 Worker
        - MARKET_MAKER: 多个 Workers（每个子种群一个）
        """
        self.logger.info("正在创建进化 Worker 池...")

        config_dir = self.config.training.neat_config_path
        worker_configs: list[WorkerConfig] = []

        # 断言 trainer 已初始化（由 setup() 保证）
        assert self.trainer is not None, "Trainer must be initialized before creating worker pool"
        populations = self.trainer.populations

        # RETAIL Workers
        retail_pop = populations[AgentType.RETAIL]
        if isinstance(retail_pop, SubPopulationManager):
            for i in range(retail_pop.sub_population_count):
                worker_configs.append(
                    WorkerConfig(
                        AgentType.RETAIL,
                        i,
                        f"{config_dir}/neat_retail.cfg",
                        retail_pop.agents_per_sub,
                    )
                )

        # RETAIL_PRO Worker
        worker_configs.append(
            WorkerConfig(
                AgentType.RETAIL_PRO,
                0,
                f"{config_dir}/neat_retail_pro.cfg",
                self.config.agents[AgentType.RETAIL_PRO].count,
            )
        )

        # WHALE Worker
        worker_configs.append(
            WorkerConfig(
                AgentType.WHALE,
                0,
                f"{config_dir}/neat_whale.cfg",
                self.config.agents[AgentType.WHALE].count,
            )
        )

        # MARKET_MAKER Workers
        mm_pop = populations[AgentType.MARKET_MAKER]
        if isinstance(mm_pop, SubPopulationManager):
            for i in range(mm_pop.sub_population_count):
                worker_configs.append(
                    WorkerConfig(
                        AgentType.MARKET_MAKER,
                        i,
                        f"{config_dir}/neat_market_maker.cfg",
                        mm_pop.agents_per_sub,
                    )
                )

        self.evolution_worker_pool = MultiPopulationWorkerPool(
            config_dir, worker_configs
        )
        self.logger.info(f"进化 Worker 池创建完成: {len(worker_configs)} 个 Worker")

    def _reset_for_arena(self) -> None:
        """重置市场状态（为新竞技场准备）

        重置内容：
        - 订单簿
        - 价格历史
        - EMA 平滑价格
        - 鲶鱼状态
        - 淘汰计数
        - tick 计数
        """
        trainer = self.trainer
        if trainer is None or trainer.matching_engine is None:
            return

        # 重置订单簿并重置最新价
        trainer.matching_engine._orderbook.clear(
            reset_price=self.config.market.initial_price
        )

        # 清空最近成交
        trainer.recent_trades.clear()

        # 重置价格历史
        trainer._price_history.clear()
        trainer._price_history.append(self.config.market.initial_price)

        # 重置 tick 历史数据
        trainer._tick_history_prices.clear()
        trainer._tick_history_volumes.clear()
        trainer._tick_history_amounts.clear()
        trainer._tick_history_prices.append(self.config.market.initial_price)
        trainer._tick_history_volumes.append(0.0)
        trainer._tick_history_amounts.append(0.0)

        # 重置 EMA 平滑价格
        trainer._init_ema_price(self.config.market.initial_price)

        # 重置鲶鱼
        for catfish in trainer.catfish_list:
            catfish.reset()
        trainer._catfish_liquidated = False

        # 重置淘汰计数和重入保护集合
        trainer._pop_liquidated_counts.clear()
        trainer._eliminating_agents.clear()

        # 重置 tick 计数
        trainer.tick = 0

        # 重置 episode 价格统计
        initial_price = self.config.market.initial_price
        trainer._episode_high_price = initial_price
        trainer._episode_low_price = initial_price

        # 做市商重新初始化流动性
        trainer._init_market()

    def _run_arena(self) -> tuple[dict[tuple[AgentType, int], np.ndarray], int]:
        """运行单个竞技场的所有 episode

        Returns:
            (累积适应度字典, 实际运行的 episode 数)
            - 累积适应度字典: {(agent_type, sub_pop_id): 累积适应度数组}
        """
        accumulated_fitness: dict[tuple[AgentType, int], np.ndarray] = {}
        actual_episodes = 0

        trainer = self.trainer
        if trainer is None or trainer.matching_engine is None:
            return accumulated_fitness, actual_episodes

        for _ in range(self.multi_config.episodes_per_arena):
            # 重置 Agent 账户
            for population in trainer.populations.values():
                population.reset_agents()

            # 重置鲶鱼状态
            for catfish in trainer.catfish_list:
                catfish.reset()
            trainer._catfish_liquidated = False

            # 重置市场状态
            trainer._reset_market()

            # 重置 tick 计数和淘汰计数
            trainer.tick = 0
            trainer._pop_liquidated_counts.clear()
            trainer._eliminating_agents.clear()

            # 重置 episode 价格统计
            initial_price = self.config.market.initial_price
            trainer._episode_high_price = initial_price
            trainer._episode_low_price = initial_price

            # 运行 episode
            episode_length = self.config.training.episode_length
            for _ in range(episode_length):
                if not trainer.is_running:
                    break
                trainer.run_tick()

                # 检查鲶鱼是否被强平
                if trainer._catfish_liquidated:
                    break

                # 检查是否满足提前结束条件
                early_end_result = trainer._should_end_episode_early()
                if early_end_result is not None:
                    break

            actual_episodes += 1

            # 累积适应度
            self._accumulate_fitness(accumulated_fitness)

        return accumulated_fitness, actual_episodes

    def _accumulate_fitness(
        self,
        accumulated: dict[tuple[AgentType, int], np.ndarray],
    ) -> None:
        """累积当前 episode 的适应度

        Args:
            accumulated: 累积适应度字典（原地修改）
        """
        trainer = self.trainer
        if trainer is None or trainer.matching_engine is None:
            return

        current_price = trainer.matching_engine._orderbook.last_price

        for agent_type, population in trainer.populations.items():
            if isinstance(population, SubPopulationManager):
                # 子种群管理器
                for sub_pop in population.sub_populations:
                    key = (agent_type, sub_pop.sub_population_id or 0)
                    self._accumulate_population_fitness(
                        sub_pop, key, current_price, accumulated
                    )
            else:
                # 普通种群
                key = (agent_type, 0)
                self._accumulate_population_fitness(
                    population, key, current_price, accumulated
                )

    def _accumulate_population_fitness(
        self,
        population: Population,
        key: tuple[AgentType, int],
        current_price: float,
        accumulated: dict[tuple[AgentType, int], np.ndarray],
    ) -> None:
        """累积单个种群的适应度

        Args:
            population: 种群对象
            key: (agent_type, sub_pop_id) 元组
            current_price: 当前价格
            accumulated: 累积适应度字典（原地修改）
        """
        # 使用 population.evaluate 获取适应度
        agent_fitnesses = population.evaluate(current_price)
        fitness_arr = np.zeros(len(population.agents), dtype=np.float32)

        for idx, (_agent, fitness) in enumerate(agent_fitnesses):
            fitness_arr[idx] = fitness

        if key not in accumulated:
            accumulated[key] = fitness_arr.copy()
        else:
            accumulated[key] += fitness_arr

    def _apply_fitness_to_genomes(
        self,
        avg_fitness: dict[tuple[AgentType, int], np.ndarray],
    ) -> None:
        """将汇总的平均适应度应用到基因组

        Args:
            avg_fitness: 平均适应度字典
                - key: (agent_type, sub_pop_id) 元组
                - value: 平均适应度数组
        """
        populations = self.populations

        for (agent_type, sub_pop_id), fitness_arr in avg_fitness.items():
            population = populations.get(agent_type)
            if population is None:
                continue

            if isinstance(population, SubPopulationManager):
                # 子种群管理器
                if sub_pop_id < len(population.sub_populations):
                    sub_pop = population.sub_populations[sub_pop_id]
                    self._apply_fitness_to_population(sub_pop, fitness_arr)
            else:
                # 普通种群
                if sub_pop_id == 0:
                    self._apply_fitness_to_population(population, fitness_arr)

    def _apply_fitness_to_population(
        self,
        population: Population,
        fitness_arr: np.ndarray,
    ) -> None:
        """将适应度应用到单个种群的基因组

        Args:
            population: 种群对象
            fitness_arr: 适应度数组
        """
        genomes = list(population.neat_pop.population.items())
        for idx, (_genome_id, genome) in enumerate(genomes):
            if idx < len(fitness_arr):
                genome.fitness = float(fitness_arr[idx])

    def _build_fitness_map(
        self,
        avg_fitness: dict[tuple[AgentType, int], np.ndarray],
    ) -> dict[tuple[AgentType, int], np.ndarray]:
        """构建进化所需的适应度映射

        确保所有 Worker 都有对应的适应度数据。

        Args:
            avg_fitness: 从竞技场收集的平均适应度

        Returns:
            适应度映射字典
        """
        fitness_map: dict[tuple[AgentType, int], np.ndarray] = {}

        for agent_type, population in self.populations.items():
            if isinstance(population, SubPopulationManager):
                # 子种群管理器
                for i, sub_pop in enumerate(population.sub_populations):
                    key = (agent_type, i)
                    if key in avg_fitness:
                        fitness_map[key] = avg_fitness[key]
                    else:
                        # 如果竞技场没有返回该子种群的适应度，使用默认值
                        fitness_map[key] = np.zeros(
                            len(sub_pop.agents), dtype=np.float32
                        )
            else:
                # 普通种群
                key = (agent_type, 0)
                if key in avg_fitness:
                    fitness_map[key] = avg_fitness[key]
                else:
                    fitness_map[key] = np.zeros(
                        len(population.agents), dtype=np.float32
                    )

        return fitness_map

    def _update_populations_from_evolution(
        self,
        evolution_results: dict[
            tuple[AgentType, int],
            tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]],
        ],
    ) -> None:
        """从进化结果更新种群

        Args:
            evolution_results: 进化结果字典
                - key: (agent_type, sub_pop_id)
                - value: (genome_data, network_params_data)
        """
        for (agent_type, sub_pop_id), (
            genome_data,
            network_params_data,
        ) in evolution_results.items():
            population = self.populations.get(agent_type)
            if population is None:
                continue

            if isinstance(population, SubPopulationManager):
                if sub_pop_id < len(population.sub_populations):
                    sub_pop = population.sub_populations[sub_pop_id]
                    self._update_single_population(
                        sub_pop, genome_data, network_params_data
                    )
            else:
                if sub_pop_id == 0:
                    self._update_single_population(
                        population, genome_data, network_params_data
                    )

    def _update_single_population(
        self,
        population: Population,
        genome_data: tuple[np.ndarray, ...],
        network_params_data: tuple[np.ndarray, ...],
    ) -> None:
        """更新单个种群

        Args:
            population: 种群对象
            genome_data: 基因组数据
            network_params_data: 网络参数数据
        """
        # 保存旧基因组用于清理
        old_genomes = list(population.neat_pop.population.values())

        # 反序列化基因组
        keys, fitnesses, metadata, nodes, conns = genome_data
        population.neat_pop.population = _deserialize_genomes_numpy(
            keys,
            fitnesses,
            metadata,
            nodes,
            conns,
            population.neat_config.genome_config,
        )

        # 增加代数
        population.generation += 1

        # 清理旧基因组
        new_genome_ids = set(population.neat_pop.population.keys())
        old_to_clean = [g for g in old_genomes if g.key not in new_genome_ids]
        population._cleanup_genome_internals(old_to_clean)

        # 清理 NEAT 历史
        population._cleanup_neat_history()

        # 解包网络参数并更新 Agent Brain
        params_list = _unpack_network_params_numpy(*network_params_data)
        new_genomes = list(population.neat_pop.population.items())
        for idx, (_gid, genome) in enumerate(new_genomes):
            if idx < len(population.agents) and idx < len(params_list):
                population.agents[idx].brain.update_from_network_params(
                    genome, params_list[idx]
                )

    def run_round(self) -> dict[str, Any]:
        """运行一轮训练

        1. 串行运行所有竞技场
        2. 汇总适应度
        3. 应用适应度到基因组
        4. 执行 NEAT 进化
        5. 更新种群

        Returns:
            本轮统计信息
        """
        if not self._is_setup:
            raise RuntimeError("训练环境未初始化，请先调用 setup()")

        round_start = time.perf_counter()
        stats: dict[str, Any] = {}

        # 1. 串行运行所有竞技场
        arena_start = time.perf_counter()
        arena_fitnesses: list[dict[tuple[AgentType, int], np.ndarray]] = []
        episode_counts: list[int] = []

        for _arena_id in range(self.multi_config.num_arenas):
            # 重置市场状态（不重置种群）
            self._reset_for_arena()

            # 运行 episodes_per_arena 个 episode
            arena_fitness, ep_count = self._run_arena()
            arena_fitnesses.append(arena_fitness)
            episode_counts.append(ep_count)

        stats["arena_run_time"] = time.perf_counter() - arena_start

        # 2. 汇总适应度
        aggregate_start = time.perf_counter()
        avg_fitness = FitnessAggregator.aggregate_simple_average(
            arena_fitnesses, episode_counts
        )
        stats["aggregate_time"] = time.perf_counter() - aggregate_start

        # 3. 应用到基因组
        self._apply_fitness_to_genomes(avg_fitness)

        # 4. 执行 NEAT 进化
        evolve_start = time.perf_counter()
        fitness_map = self._build_fitness_map(avg_fitness)

        assert self.evolution_worker_pool is not None

        # 首次进化需要同步基因组
        sync_genomes = not self._worker_pool_synced

        evolution_results = self.evolution_worker_pool.evolve_all_parallel(
            fitness_map, sync_genomes=sync_genomes
        )
        self._worker_pool_synced = True
        stats["evolve_time"] = time.perf_counter() - evolve_start

        # 5. 更新种群
        update_start = time.perf_counter()
        self._update_populations_from_evolution(evolution_results)

        # 进化后重新注册费率、重建映射表和执行顺序
        if self.trainer is not None:
            self.trainer._register_all_agents()
            self.trainer._build_agent_map()
            self.trainer._build_execution_order()
            self.trainer._update_pop_total_counts()
            self.trainer._update_network_caches()

        stats["update_time"] = time.perf_counter() - update_start

        # 增加代数和总 episode 计数
        self.generation += 1
        episodes_this_round = sum(episode_counts)
        self.total_episodes += episodes_this_round
        stats["generation"] = self.generation
        stats["episodes_this_round"] = episodes_this_round
        stats["total_episodes"] = self.total_episodes
        stats["avg_fitnesses"] = avg_fitness

        # 垃圾回收
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        malloc_trim()

        stats["total_time"] = time.perf_counter() - round_start

        self.logger.info(
            f"训练轮次 {self.generation} 完成: "
            f"arena_run={stats['arena_run_time']:.2f}s, "
            f"aggregate={stats['aggregate_time']:.2f}s, "
            f"evolve={stats['evolve_time']:.2f}s, "
            f"update={stats['update_time']:.2f}s, "
            f"total={stats['total_time']:.2f}s"
        )

        return stats

    def train(
        self,
        num_rounds: int | None = None,
        checkpoint_callback: Callable[[int], None] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """主训练循环

        Args:
            num_rounds: 训练轮数，None 表示无限循环
            checkpoint_callback: 检查点回调函数，参数为当前代数
            progress_callback: 进度回调函数，参数为本轮统计信息
        """
        if not self._is_setup:
            self.setup()

        self._is_running = True
        round_count = 0

        try:
            while self._is_running:
                if num_rounds is not None and round_count >= num_rounds:
                    break

                stats = self.run_round()
                round_count += 1

                # 进度回调（每轮调用）
                if progress_callback is not None:
                    progress_callback(stats)

                # 检查点回调（按间隔调用）
                if checkpoint_callback is not None:
                    checkpoint_callback(self.generation)

        except KeyboardInterrupt:
            self.logger.info("收到中断信号，停止训练...")
        finally:
            self._is_running = False
            self.logger.info(f"训练完成，共运行 {round_count} 轮")

    def save_checkpoint(self, path: str) -> None:
        """保存检查点

        格式与 MultiArenaTrainer 兼容。

        Args:
            path: 检查点文件路径
        """
        # 清理 NEAT 历史数据以减少 checkpoint 体积
        for agent_type, population in self.populations.items():
            if isinstance(population, SubPopulationManager):
                for sub_pop in population.sub_populations:
                    sub_pop._cleanup_neat_history()
            else:
                population._cleanup_neat_history()

        checkpoint_data: dict[str, Any] = {
            "generation": self.generation,
            "populations": {},
        }

        # 序列化种群数据
        for agent_type, population in self.populations.items():
            if isinstance(population, SubPopulationManager):
                # 子种群管理器
                pop_data = {
                    "is_sub_population_manager": True,
                    "sub_population_count": population.sub_population_count,
                    "sub_populations": [],
                }
                for sub_pop in population.sub_populations:
                    sub_pop_data = {
                        "generation": sub_pop.generation,
                        "neat_pop": sub_pop.neat_pop,
                    }
                    pop_data["sub_populations"].append(sub_pop_data)
                checkpoint_data["populations"][agent_type] = pop_data
            else:
                # 普通种群
                checkpoint_data["populations"][agent_type] = {
                    "generation": population.generation,
                    "neat_pop": population.neat_pop,
                }

        # 保存到文件
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用 gzip 压缩保存
        with gzip.open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        self.logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str) -> None:
        """加载检查点

        支持 MultiArenaTrainer 的检查点格式。

        Args:
            path: 检查点文件路径
        """
        # 自动检测文件格式
        with open(path, "rb") as f:
            magic = f.read(2)

        # gzip 文件的魔数是 0x1f 0x8b
        if magic == b"\x1f\x8b":
            with gzip.open(path, "rb") as f:
                checkpoint_data = pickle.load(f)
        else:
            with open(path, "rb") as f:
                checkpoint_data = pickle.load(f)

        self.generation = checkpoint_data.get("generation", 0)
        populations_data = checkpoint_data.get("populations", {})

        # 恢复种群数据
        for agent_type, pop_data in populations_data.items():
            if agent_type not in self.populations:
                continue

            population = self.populations[agent_type]

            if isinstance(population, SubPopulationManager):
                # 子种群管理器
                if pop_data.get("is_sub_population_manager"):
                    sub_pops_data = pop_data.get("sub_populations", [])
                    for i, sub_pop_data in enumerate(sub_pops_data):
                        if i < len(population.sub_populations):
                            sub_pop = population.sub_populations[i]
                            sub_pop.generation = sub_pop_data.get("generation", 0)
                            sub_pop.neat_pop = sub_pop_data.get("neat_pop")

                            # 重新创建 Agent
                            genomes = list(sub_pop.neat_pop.population.items())
                            sub_pop.agents = sub_pop.create_agents(genomes)
                else:
                    # 旧格式：单个种群迁移到子种群
                    self.logger.warning(
                        f"{agent_type.value} 检查点为旧格式，需要迁移"
                    )
            else:
                # 普通种群
                population.generation = pop_data.get("generation", 0)
                population.neat_pop = pop_data.get("neat_pop")

                # 重新创建 Agent
                genomes = list(population.neat_pop.population.items())
                population.agents = population.create_agents(genomes)

        # 进化后重新注册费率、重建映射表和执行顺序
        if self.trainer is not None:
            self.trainer._register_all_agents()
            self.trainer._build_agent_map()
            self.trainer._build_execution_order()
            self.trainer._update_pop_total_counts()
            self.trainer._update_network_caches()

        # 重置 Worker 池同步标志，下次进化时需要重新同步
        self._worker_pool_synced = False

        self.logger.info(f"检查点已加载: {path}, generation={self.generation}")

    def stop(self) -> None:
        """停止训练并清理资源"""
        self._is_running = False

        # 关闭进化 Worker 池
        if self.evolution_worker_pool is not None:
            self.evolution_worker_pool.shutdown()
            self.evolution_worker_pool = None

        # 关闭 Trainer 中的 Worker 池
        if self.trainer is not None:
            # 关闭统一 Worker 池
            if self.trainer._unified_worker_pool is not None:
                self.trainer._unified_worker_pool.shutdown()
                self.trainer._unified_worker_pool = None

            # 关闭种群的线程池
            for population in self.trainer.populations.values():
                if isinstance(population, SubPopulationManager):
                    population.shutdown_executor()
                else:
                    population.shutdown_executor()

            self.trainer.is_running = False

        self._is_setup = False
        self.logger.info("单进程多竞技场训练器已停止")

    def __enter__(self) -> "SingleArenaTrainer":
        """上下文管理器入口"""
        self.setup()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """上下文管理器出口"""
        self.stop()
