"""多竞技场训练协调器模块

协调多个竞技场并行训练，管理进化周期。
"""

import gc
import gzip
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.bio.agents.base import AgentType
from src.config.config import Config
from src.core.log_engine.logger import get_logger
from .arena_pool import ArenaPool
from .fitness_aggregator import FitnessAggregator
from src.training.population import (
    MultiPopulationWorkerPool,
    Population,
    SubPopulationManager,
    WorkerConfig,
    _serialize_genomes_numpy,
    _pack_network_params_numpy,
    _extract_network_params_from_genome,
    malloc_trim,
)


@dataclass
class MultiArenaConfig:
    """多竞技场配置"""

    num_arenas: int = 10
    episodes_per_arena: int = 10


class MultiArenaTrainer:
    """多竞技场训练协调器

    协调多个竞技场并行训练，管理进化周期。

    核心流程：
    1. 初始化：创建 NEAT 种群、创建竞技场池
    2. 训练循环：
       a. 序列化并广播基因组到所有竞技场
       b. 并行运行所有竞技场
       c. 汇总适应度
       d. 执行 NEAT 进化
       e. 保存检查点

    Attributes:
        config: 全局配置
        multi_arena_config: 多竞技场配置
        populations: 种群字典 {AgentType -> Population/SubPopulationManager}
        evolution_worker_pool: 进化 Worker 池
        arena_pool: 竞技场池
        generation: 当前代数
        logger: 日志器
    """

    config: Config
    multi_arena_config: MultiArenaConfig
    populations: dict[AgentType, Population | SubPopulationManager]
    evolution_worker_pool: MultiPopulationWorkerPool | None
    arena_pool: ArenaPool | None
    generation: int

    def __init__(
        self,
        config: Config,
        multi_arena_config: MultiArenaConfig | None = None,
    ) -> None:
        """初始化多竞技场训练协调器

        Args:
            config: 全局配置
            multi_arena_config: 多竞技场配置，None 使用默认配置
        """
        self.config = config
        self.multi_arena_config = multi_arena_config or MultiArenaConfig()
        self.logger = get_logger("multi_arena_trainer")

        self.populations = {}
        self.evolution_worker_pool = None
        self.arena_pool = None
        self.generation = 0
        self.total_episodes = 0

        self._is_setup = False
        self._is_running = False

    def setup(self) -> None:
        """初始化训练环境

        1. 创建 NEAT 种群（Population / SubPopulationManager）
        2. 创建进化 Worker 池（MultiPopulationWorkerPool）
        3. 创建竞技场池（ArenaPool）
        """
        if self._is_setup:
            self.logger.warning("训练环境已初始化，跳过重复初始化")
            return

        self.logger.info("开始初始化多竞技场训练环境...")

        # 1. 创建 NEAT 种群
        self._create_populations()

        # 2. 创建进化 Worker 池
        self._create_evolution_worker_pool()

        # 3. 创建竞技场池
        self._create_arena_pool()

        self._is_setup = True
        self.logger.info(
            f"多竞技场训练环境初始化完成: "
            f"{self.multi_arena_config.num_arenas} 个竞技场, "
            f"{len(self.populations)} 个种群"
        )

    def _create_populations(self) -> None:
        """创建 NEAT 种群

        种群配置：
        - RETAIL: SubPopulationManager（10个子种群）
        - RETAIL_PRO: Population
        - WHALE: Population
        - MARKET_MAKER: SubPopulationManager（2个子种群）
        """
        self.logger.info("正在创建 NEAT 种群...")

        # RETAIL: 10个子种群
        retail_sub_count = self.config.training.retail_sub_population_count
        self.populations[AgentType.RETAIL] = SubPopulationManager(
            self.config, AgentType.RETAIL, sub_count=retail_sub_count
        )

        # RETAIL_PRO: 1个种群
        self.populations[AgentType.RETAIL_PRO] = Population(
            AgentType.RETAIL_PRO, self.config
        )

        # WHALE: 1个种群
        self.populations[AgentType.WHALE] = Population(AgentType.WHALE, self.config)

        # MARKET_MAKER: 2个子种群
        mm_sub_count = 2  # 做市商固定为2个子种群
        self.populations[AgentType.MARKET_MAKER] = SubPopulationManager(
            self.config, AgentType.MARKET_MAKER, sub_count=mm_sub_count
        )

        self.logger.info(
            f"NEAT 种群创建完成: "
            f"RETAIL={len(self.populations[AgentType.RETAIL].agents)}, "
            f"RETAIL_PRO={len(self.populations[AgentType.RETAIL_PRO].agents)}, "
            f"WHALE={len(self.populations[AgentType.WHALE].agents)}, "
            f"MARKET_MAKER={len(self.populations[AgentType.MARKET_MAKER].agents)}"
        )

    def _create_evolution_worker_pool(self) -> None:
        """创建进化 Worker 池

        Worker 配置：
        - RETAIL: 10 Workers（每个子种群一个）
        - RETAIL_PRO: 1 Worker
        - WHALE: 1 Worker
        - MARKET_MAKER: 2 Workers（每个子种群一个）
        """
        self.logger.info("正在创建进化 Worker 池...")

        config_dir = self.config.training.neat_config_path
        worker_configs: list[WorkerConfig] = []

        # RETAIL Workers
        retail_pop = self.populations[AgentType.RETAIL]
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
        mm_pop = self.populations[AgentType.MARKET_MAKER]
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

    def _create_arena_pool(self) -> None:
        """创建竞技场池"""
        self.logger.info("正在创建竞技场池...")

        self.arena_pool = ArenaPool(
            self.multi_arena_config.num_arenas,
            self.config,
        )
        self.arena_pool.start()

        self.logger.info(
            f"竞技场池创建完成: {self.multi_arena_config.num_arenas} 个竞技场"
        )

    def _serialize_genomes_for_broadcast(
        self,
    ) -> tuple[dict[AgentType, bytes], dict[AgentType, tuple[np.ndarray, ...]]]:
        """序列化基因组和网络参数用于广播

        为每个种群序列化一份基因组数据，用于广播到所有竞技场。

        Returns:
            (genome_data_dict, network_params_dict) 元组
            - genome_data_dict: 各物种的序列化基因组 {AgentType -> bytes}
            - network_params_dict: 各物种的网络参数 {AgentType -> tuple[np.ndarray, ...]}
        """
        genome_data_dict: dict[AgentType, bytes] = {}
        network_params_dict: dict[AgentType, tuple[np.ndarray, ...]] = {}

        for agent_type, population in self.populations.items():
            # 获取第一个基因组作为代表（竞技场会复制给所有 Agent）
            genomes = population.get_all_genomes()
            if not genomes:
                self.logger.warning(f"{agent_type.value} 没有基因组")
                continue

            # 取第一个基因组作为代表
            representative_genome = genomes[0]

            # 序列化基因组
            genome_data_dict[agent_type] = pickle.dumps(representative_genome)

            # 获取 NEAT 配置
            if isinstance(population, SubPopulationManager):
                neat_config = population.sub_populations[0].neat_config
            else:
                neat_config = population.neat_config

            # 提取所有基因组的网络参数
            params_list: list[dict[str, np.ndarray | int]] = []
            for genome in genomes:
                params = _extract_network_params_from_genome(genome, neat_config)
                params_list.append(params)

            # 打包网络参数
            network_params_dict[agent_type] = _pack_network_params_numpy(params_list)

        return genome_data_dict, network_params_dict

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
        for (agent_type, sub_pop_id), fitness_arr in avg_fitness.items():
            population = self.populations.get(agent_type)
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
        for idx, (genome_id, genome) in enumerate(genomes):
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
        from src.training.population import (
            _deserialize_genomes_numpy,
            _unpack_network_params_numpy,
        )

        for (agent_type, sub_pop_id), (genome_data, network_params_data) in evolution_results.items():
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
        from src.training.population import (
            _deserialize_genomes_numpy,
            _unpack_network_params_numpy,
        )

        # 保存旧基因组用于清理
        old_genomes = list(population.neat_pop.population.values())

        # 反序列化基因组
        keys, fitnesses, metadata, nodes, conns = genome_data
        population.neat_pop.population = _deserialize_genomes_numpy(
            keys, fitnesses, metadata, nodes, conns, population.neat_config.genome_config
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
        for idx, (gid, genome) in enumerate(new_genomes):
            if idx < len(population.agents) and idx < len(params_list):
                population.agents[idx].brain.update_from_network_params(
                    genome, params_list[idx]
                )

    def run_round(self) -> dict[str, Any]:
        """运行一轮训练

        1. 序列化基因组
        2. 广播到所有竞技场
        3. 并行运行
        4. 汇总适应度
        5. 执行进化
        6. 更新种群

        Returns:
            本轮统计信息
        """
        if not self._is_setup:
            raise RuntimeError("训练环境未初始化，请先调用 setup()")

        round_start = time.perf_counter()
        stats: dict[str, Any] = {"generation": self.generation}

        # 1. 序列化基因组和网络参数
        serialize_start = time.perf_counter()
        genome_data_dict, network_params_dict = self._serialize_genomes_for_broadcast()
        stats["serialize_time"] = time.perf_counter() - serialize_start

        # 2. 广播到所有竞技场
        broadcast_start = time.perf_counter()
        assert self.arena_pool is not None
        self.arena_pool.broadcast_genomes(genome_data_dict, network_params_dict)
        stats["broadcast_time"] = time.perf_counter() - broadcast_start

        # 3. 并行运行所有竞技场
        run_start = time.perf_counter()
        avg_fitness = self.arena_pool.run_all(
            self.multi_arena_config.episodes_per_arena
        )
        stats["arena_run_time"] = time.perf_counter() - run_start

        # 4. 将汇总的适应度应用到基因组
        self._apply_fitness_to_genomes(avg_fitness)

        # 5. 执行 NEAT 进化
        evolve_start = time.perf_counter()
        fitness_map = self._build_fitness_map(avg_fitness)

        assert self.evolution_worker_pool is not None

        # 首次进化需要同步基因组
        sync_genomes = self.generation == 0

        evolution_results = self.evolution_worker_pool.evolve_all_parallel(
            fitness_map, sync_genomes=sync_genomes
        )
        stats["evolve_time"] = time.perf_counter() - evolve_start

        # 6. 更新种群
        update_start = time.perf_counter()
        self._update_populations_from_evolution(evolution_results)
        stats["update_time"] = time.perf_counter() - update_start

        # 增加代数和总 episode 计数
        self.generation += 1
        episodes_this_round = (
            self.multi_arena_config.num_arenas *
            self.multi_arena_config.episodes_per_arena
        )
        self.total_episodes += episodes_this_round
        stats["episodes_this_round"] = episodes_this_round
        stats["total_episodes"] = self.total_episodes

        # 垃圾回收
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        malloc_trim()

        stats["total_time"] = time.perf_counter() - round_start

        self.logger.info(
            f"训练轮次 {self.generation} 完成: "
            f"serialize={stats['serialize_time']:.2f}s, "
            f"broadcast={stats['broadcast_time']:.2f}s, "
            f"arena_run={stats['arena_run_time']:.2f}s, "
            f"evolve={stats['evolve_time']:.2f}s, "
            f"update={stats['update_time']:.2f}s, "
            f"total={stats['total_time']:.2f}s"
        )

        return stats

    def train(
        self,
        num_rounds: int | None = None,
        checkpoint_callback: Callable[[int], None] | None = None,
    ) -> None:
        """主训练循环

        Args:
            num_rounds: 训练轮数，None 表示无限循环
            checkpoint_callback: 检查点回调函数，参数为当前代数
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

                # 检查点回调
                if checkpoint_callback is not None:
                    checkpoint_callback(self.generation)

        except KeyboardInterrupt:
            self.logger.info("收到中断信号，停止训练...")
        finally:
            self._is_running = False
            self.logger.info(f"训练完成，共运行 {round_count} 轮")

    def save_checkpoint(self, path: str) -> None:
        """保存检查点

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

        Args:
            path: 检查点文件路径
        """
        # 自动检测文件格式
        with open(path, "rb") as f:
            magic = f.read(2)

        # gzip 文件的魔数是 0x1f 0x8b
        if magic == b'\x1f\x8b':
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

        self.logger.info(f"检查点已加载: {path}, generation={self.generation}")

    def stop(self) -> None:
        """停止训练并清理资源"""
        self._is_running = False

        # 关闭竞技场池
        if self.arena_pool is not None:
            self.arena_pool.shutdown()
            self.arena_pool = None

        # 关闭进化 Worker 池
        if self.evolution_worker_pool is not None:
            self.evolution_worker_pool.shutdown()
            self.evolution_worker_pool = None

        # 关闭种群的线程池
        for population in self.populations.values():
            if isinstance(population, SubPopulationManager):
                population.shutdown_executor()
            else:
                population.shutdown_executor()

        self._is_setup = False
        self.logger.info("多竞技场训练器已停止")

    def __enter__(self) -> "MultiArenaTrainer":
        """上下文管理器入口"""
        self.setup()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """上下文管理器出口"""
        self.stop()
