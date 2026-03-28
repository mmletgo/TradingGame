"""多竞技场并行训练器模块

核心特性：通过 ArenaWorkerPool 将完整 episode 执行委托给 Worker 进程池，
主进程仅负责进化和网络参数同步。

设计原则：
1. 神经网络共享：进化后将新网络参数同步给所有 Worker
2. 账户状态独立：每个 Worker 内部维护独立的 ArenaState
3. Episode 级并行：Worker 独立运行完整 episode tick 循环，消除 tick 级 IPC
4. 订单簿独立：每个 Worker 内部的竞技场有独立的 MatchingEngine 和 OrderBook
"""

import gc
import gzip
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np


@dataclass
class MultiArenaConfig:
    """多竞技场配置"""

    num_arenas: int = 2
    episodes_per_arena: int = 50


if TYPE_CHECKING:
    from src.training._cython.batch_decide_openmp import BatchNetworkCache

from src.bio.agents.base import AgentType
from src.config.config import Config
from src.core.log_engine.logger import get_logger
from src.market.adl.adl_manager import ADLManager
from src.market.noise_trader.noise_trader import NoiseTrader
from src.market.matching.matching_engine import MatchingEngine
from src.training.population import (
    MultiPopulationWorkerPool,
    Population,
    SubPopulationManager,
    WorkerConfig,
    _apply_species_data_to_population,
    _concat_network_params_numpy,
    _deserialize_genomes_numpy,
    _pack_network_params_numpy,
    _serialize_genomes_numpy,
    _serialize_species_data,
    _unpack_network_params_numpy,
    malloc_trim,
)

from .arena_state import (
    AgentAccountState,
    ArenaState,
    NoiseTraderAccountState,
)
from .arena_worker import AgentInfo, ArenaWorkerPool, EpisodeResult
from .fitness_aggregator import FitnessAggregator
from .shared_network_memory import SharedNetworkMemory, SharedNetworkMetadata

# 缓存类型常量
CACHE_TYPE_FULL = 1
CACHE_TYPE_MARKET_MAKER = 2

# 尝试导入 OpenMP 批量决策缓存模块
try:
    from src.training._cython.batch_decide_openmp import BatchNetworkCache

    HAS_OPENMP_DECIDE = True
except ImportError:
    HAS_OPENMP_DECIDE = False
    BatchNetworkCache = None  # type: ignore


class ParallelArenaTrainer:
    """多竞技场并行训练器

    核心特性：通过 ArenaWorkerPool 将完整 episode 执行委托给 Worker 进程池。
    主进程仅负责：创建种群、NEAT 进化、网络参数同步、检查点管理。

    Attributes:
        config: 全局配置
        multi_config: 多竞技场配置
        populations: 共享的种群（神经网络）
        arena_states: N 个独立的竞技场状态（主进程保留用于检查点等）
        network_caches: 共享的网络缓存（主进程用于 MM 初始化等）
        evolution_worker_pool: 进化 Worker 池
        generation: 当前代数
        total_episodes: 总 episode 数
    """

    config: Config
    multi_config: MultiArenaConfig
    populations: dict[AgentType, Population | SubPopulationManager]
    arena_states: list[ArenaState]
    network_caches: Any  # dict[AgentType, BatchNetworkCache] | None
    evolution_worker_pool: MultiPopulationWorkerPool | None
    generation: int
    total_episodes: int

    def __init__(
        self,
        config: Config,
        multi_config: MultiArenaConfig | None = None,
    ) -> None:
        """初始化多竞技场并行训练器

        Args:
            config: 全局配置
            multi_config: 多竞技场配置，None 使用默认配置
        """
        self.config = config
        self.multi_config = multi_config or MultiArenaConfig()
        self.logger = get_logger("parallel_arena_trainer")

        self.populations: dict[AgentType, Population | SubPopulationManager] = {}
        self.arena_states: list[ArenaState] = []
        self.network_caches: Any = None
        self.evolution_worker_pool: MultiPopulationWorkerPool | None = None
        self.generation: int = 0
        self.total_episodes: int = 0

        self._is_setup: bool = False
        self._is_running: bool = False
        self._worker_pool_synced: bool = False

        # Agent ID 到 Agent 对象的映射表（O(1) 查找）
        self.agent_map: dict[int, Any] = {}

        # Agent ID 到网络索引的映射表（用于 decide_multi_arena）
        # 格式: {agent_type: {agent_id: network_index}}
        self._network_index_map: dict[AgentType, dict[int, int]] = {}

        # EMA 参数
        self._ema_alpha: float = 0.1

        # Arena Worker 进程池（episode 级 IPC）
        self._arena_worker_pool: ArenaWorkerPool | None = None

        # 共享内存管理
        self._shared_network_memories: dict[AgentType, SharedNetworkMemory] = {}
        self._prev_shared_network_memories: dict[AgentType, SharedNetworkMemory] = {}

    def setup(self) -> None:
        """初始化：创建种群、竞技场状态、网络缓存、进化 Worker 池、Arena Worker 池

        重要：Worker 进程池的创建必须在内存分配（如创建种群）之前完成。
        这是因为 fork 时子进程会继承父进程的内存空间，如果父进程在 fork 前
        已经分配了大量内存，子进程修改数据时会触发 COW（Copy-On-Write）
        导致内存复制，引发内存泄漏问题。
        """
        if self._is_setup:
            self.logger.warning("训练环境已初始化，跳过重复初始化")
            return

        self.logger.info("开始初始化多竞技场并行训练环境...")

        # 1. 先创建进化 Worker 池（在分配大内存之前 fork）
        self._create_evolution_worker_pool()

        # 读取 EMA 配置
        self._ema_alpha = self.config.market.ema_alpha

        # 2. 创建种群（共享神经网络）- 大量内存分配
        self._create_populations()

        # 3. 创建竞技场状态（主进程保留用于检查点等）
        self._create_arena_states()

        # 4. 初始化网络缓存（主进程保留用于进化后同步）
        self._init_network_caches()

        # 5. 构建 Agent 映射表
        self._build_agent_map()

        # 6. 创建 Arena Worker Pool（替代 Execute Worker Pool）
        agent_infos = self._build_agent_infos()
        num_workers = min(self.multi_config.num_arenas, os.cpu_count() or 16)
        self._arena_worker_pool = ArenaWorkerPool(
            num_workers=num_workers,
            num_arenas=self.multi_config.num_arenas,
            config=self.config,
            agent_infos=agent_infos,
        )
        self._arena_worker_pool.start()

        # 7. 同步网络参数到 Workers
        self._sync_networks_to_workers()

        self._is_setup = True
        self.logger.info(
            f"多竞技场并行训练环境初始化完成: "
            f"{self.multi_config.num_arenas} 个竞技场, "
            f"每竞技场 {self.multi_config.episodes_per_arena} 个 episode, "
            f"Arena Workers: {num_workers}"
        )

    def _create_populations(self) -> None:
        """创建共享的种群（神经网络）"""
        self.logger.info("正在创建种群...")

        # RETAIL_PRO: 10个子种群
        self.populations[AgentType.RETAIL_PRO] = SubPopulationManager(
            self.config, AgentType.RETAIL_PRO, sub_count=10
        )

        # MARKET_MAKER: 6个子种群
        self.populations[AgentType.MARKET_MAKER] = SubPopulationManager(
            self.config, AgentType.MARKET_MAKER, sub_count=6
        )

        self.logger.info(f"种群创建完成: {len(self.populations)} 种类型")

    def _create_arena_states(self) -> None:
        """创建 N 个独立的竞技场状态"""
        self.logger.info(f"正在创建 {self.multi_config.num_arenas} 个竞技场状态...")

        self.arena_states = []
        initial_price = self.config.market.initial_price

        for arena_id in range(self.multi_config.num_arenas):
            # 创建独立的撮合引擎和 ADL 管理器
            matching_engine = MatchingEngine(self.config.market)
            adl_manager = ADLManager()

            # 从 Agent 创建账户状态
            agent_states: dict[int, AgentAccountState] = {}
            for population in self.populations.values():
                for agent in population.agents:
                    state = AgentAccountState.from_agent(agent)
                    agent_states[agent.agent_id] = state

            # 创建噪声交易者状态
            noise_trader_states: dict[int, NoiseTraderAccountState] = {}
            noise_trader_config = self.config.noise_trader
            for i in range(noise_trader_config.count):
                trader_id = -(i + 1)  # 负数 ID: -1, -2, -3, ...
                noise_trader = NoiseTrader(trader_id, noise_trader_config)
                state = NoiseTraderAccountState.from_noise_trader(noise_trader)
                noise_trader_states[trader_id] = state
                # 注册噪声交易者到撮合引擎（手续费为 0）
                matching_engine.register_agent(trader_id, 0.0, 0.0)

            # 注册所有 Agent 到撮合引擎
            for agent_type, population in self.populations.items():
                agent_config = population.agent_config
                for agent in population.agents:
                    matching_engine.register_agent(
                        agent.agent_id,
                        agent_config.maker_fee_rate,
                        agent_config.taker_fee_rate,
                    )

            # 创建竞技场状态
            arena_state = ArenaState(
                arena_id=arena_id,
                matching_engine=matching_engine,
                adl_manager=adl_manager,
                agent_states=agent_states,
                noise_trader_states=noise_trader_states,
                recent_trades=deque(maxlen=100),
                price_history=deque([initial_price], maxlen=1000),
                tick_history_prices=deque([initial_price], maxlen=100),
                tick_history_volumes=deque([0.0], maxlen=100),
                tick_history_amounts=deque([0.0], maxlen=100),
                smooth_mid_price=initial_price,
                tick=0,
                pop_liquidated_counts={agent_type: 0 for agent_type in AgentType},
                eliminating_agents=set(),
                episode_high_price=initial_price,
                episode_low_price=initial_price,
            )

            self.arena_states.append(arena_state)

        self.logger.info(f"竞技场状态创建完成: {len(self.arena_states)} 个竞技场")

    def _build_agent_map(self, force: bool = False) -> None:
        """构建 Agent ID 到 Agent 对象的映射表（O(1) 查找）

        Args:
            force: 是否强制重建（默认 False，已存在时跳过）
        """
        # 快速路径：如果已构建且数量匹配，跳过重建
        if not force and self.agent_map:
            expected_count = sum(len(p.agents) for p in self.populations.values())
            if len(self.agent_map) == expected_count:
                return

        self.agent_map.clear()
        for population in self.populations.values():
            for agent in population.agents:
                self.agent_map[agent.agent_id] = agent

        # 同时构建网络索引映射表
        self._build_network_index_map()

    def _build_network_index_map(self) -> None:
        """构建 Agent ID 到网络索引的映射表

        网络索引是 Agent 在其种群 agents 列表中的位置，
        与 BatchNetworkCache 中的网络顺序一致。
        """
        self._network_index_map.clear()
        for agent_type, population in self.populations.items():
            type_map: dict[int, int] = {}
            for idx, agent in enumerate(population.agents):
                type_map[agent.agent_id] = idx
            self._network_index_map[agent_type] = type_map

    def _get_network_index(self, agent_type: AgentType, agent_id: int) -> int:
        """获取 Agent 在其种群中的网络索引

        Args:
            agent_type: Agent 类型
            agent_id: Agent ID

        Returns:
            网络索引，如果未找到返回 -1
        """
        type_map = self._network_index_map.get(agent_type)
        if type_map is None:
            return -1
        return type_map.get(agent_id, -1)

    def _init_network_caches(self) -> None:
        """初始化共享的网络数据缓存"""
        if not HAS_OPENMP_DECIDE:
            self.logger.warning("OpenMP 批量决策模块不可用，将使用串行推理")
            self.network_caches = None
            return

        if BatchNetworkCache is None:
            self.network_caches = None
            return

        self.network_caches = {}

        for agent_type, population in self.populations.items():
            if not population.agents:
                continue

            # 确定缓存类型
            if agent_type == AgentType.MARKET_MAKER:
                cache_type = CACHE_TYPE_MARKET_MAKER
            else:
                cache_type = CACHE_TYPE_FULL

            # 创建缓存
            num_networks = len(population.agents)
            num_threads = self.config.training.openmp_threads
            cache = BatchNetworkCache(num_networks, cache_type, num_threads)

            # 提取网络数据
            networks = [agent.brain.network for agent in population.agents]
            cache.update_networks(networks)

            self.network_caches[agent_type] = cache
            self.logger.debug(
                f"已为 {agent_type.value} 创建网络缓存，数量={num_networks}"
            )

        self.logger.info(
            f"网络数据缓存初始化完成，共 {len(self.network_caches)} 种类型"
        )

    def _update_network_caches(self) -> None:
        """更新网络数据缓存（进化后调用）

        优先使用 _cached_network_params_data（packed numpy 数组）直接填充 C 结构，
        跳过 Python 对象创建。回退时使用原有的 Python 对象路径。

        注意：不清除 _cached_network_params_data，由后续的 _sync_networks_to_workers() 消费。
        """
        if self.network_caches is None:
            return

        for agent_type, population in self.populations.items():
            cache = self.network_caches.get(agent_type)
            if cache is None:
                continue

            cached = getattr(population, '_cached_network_params_data', None)
            if cached is not None:
                # Phase 2: 直接从 packed numpy 数组更新 C 结构
                cache.update_networks_from_numpy(*cached)
                # 不清除 _cached_network_params_data：
                # 由 _sync_networks_to_workers() 消费，确保进化后的新参数能正确同步到 Workers
            else:
                # 回退：使用原有方式（从 Python 对象提取）
                if not population.agents:
                    continue
                networks = [agent.brain.network for agent in population.agents]
                cache.update_networks(networks)
                networks.clear()
                del networks

    def _create_evolution_worker_pool(self) -> None:
        """创建进化 Worker 池

        重要：此方法在创建种群之前调用，因此不能依赖 self.populations。
        所有参数直接从 self.config 计算。
        """
        self.logger.info("正在创建进化 Worker 池...")

        config_dir = self.config.training.neat_config_path
        worker_configs: list[WorkerConfig] = []

        # 子种群数量硬编码（与 _create_populations 保持一致）
        # RETAIL_PRO: 10 个子种群
        # MARKET_MAKER: 6 个子种群
        retail_pro_sub_count = 10
        mm_sub_count = 6

        # RETAIL_PRO Workers
        retail_pro_total = self.config.agents[AgentType.RETAIL_PRO].count
        retail_pro_per_sub = retail_pro_total // retail_pro_sub_count
        for i in range(retail_pro_sub_count):
            worker_configs.append(
                WorkerConfig(
                    AgentType.RETAIL_PRO,
                    i,
                    f"{config_dir}/neat_retail_pro.cfg",
                    retail_pro_per_sub,
                )
            )

        # MARKET_MAKER Workers
        mm_total = self.config.agents[AgentType.MARKET_MAKER].count
        mm_per_sub = mm_total // mm_sub_count
        for i in range(mm_sub_count):
            worker_configs.append(
                WorkerConfig(
                    AgentType.MARKET_MAKER,
                    i,
                    f"{config_dir}/neat_market_maker.cfg",
                    mm_per_sub,
                )
            )

        self.evolution_worker_pool = MultiPopulationWorkerPool(
            config_dir, worker_configs
        )
        self.logger.info(f"进化 Worker 池创建完成: {len(worker_configs)} 个 Worker")

    # ========================================================================
    # Arena Worker Pool 相关方法
    # ========================================================================

    def _build_agent_infos(self) -> list[AgentInfo]:
        """从 populations 构建 AgentInfo 列表（供 Arena Worker 使用）

        Returns:
            AgentInfo 列表，包含所有种群中所有 Agent 的信息
        """
        agent_infos: list[AgentInfo] = []
        for agent_type, population in self.populations.items():
            if isinstance(population, SubPopulationManager):
                for sub_pop_id, sub_pop in enumerate(population.sub_populations):
                    agent_config = sub_pop.agent_config
                    # 计算 network_index 偏移：前面子种群的 agent 总数
                    offset = sum(
                        len(population.sub_populations[j].agents)
                        for j in range(sub_pop_id)
                    )
                    for idx, agent in enumerate(sub_pop.agents):
                        agent_infos.append(AgentInfo(
                            agent_id=agent.agent_id,
                            agent_type=agent_type,
                            sub_pop_id=sub_pop_id,
                            network_index=offset + idx,
                            initial_balance=agent_config.initial_balance,
                            leverage=agent_config.leverage,
                            maintenance_margin_rate=agent_config.maintenance_margin_rate,
                            maker_fee_rate=agent_config.maker_fee_rate,
                            taker_fee_rate=agent_config.taker_fee_rate,
                        ))
            else:
                agent_config = population.agent_config
                for idx, agent in enumerate(population.agents):
                    agent_infos.append(AgentInfo(
                        agent_id=agent.agent_id,
                        agent_type=agent_type,
                        sub_pop_id=0,
                        network_index=idx,
                        initial_balance=agent_config.initial_balance,
                        leverage=agent_config.leverage,
                        maintenance_margin_rate=agent_config.maintenance_margin_rate,
                        maker_fee_rate=agent_config.maker_fee_rate,
                        taker_fee_rate=agent_config.taker_fee_rate,
                    ))
        return agent_infos

    def _sync_networks_to_workers(self) -> None:
        """将当前网络参数同步到所有 Arena Workers（共享内存模式）

        使用共享内存零拷贝传输网络数据，主进程填充一次，所有 Worker 通过共享内存访问。
        """
        if self._arena_worker_pool is None:
            return

        network_params: dict[AgentType, tuple[np.ndarray, ...]] = {}
        for agent_type, pop in self.populations.items():
            # 优先使用进化后缓存的 packed numpy 数据
            cached = getattr(pop, '_cached_network_params_data', None)
            if cached is not None:
                network_params[agent_type] = cached
                pop._cached_network_params_data = None
                continue

            # 首次同步或回退：从 Agent 的 brain.network 提取参数
            try:
                agents = pop.agents
                params_list: list[dict[str, np.ndarray | int]] = []
                for agent in agents:
                    params_list.append(agent.brain.network.get_params())
                packed = _pack_network_params_numpy(params_list)
                network_params[agent_type] = packed
                params_list.clear()
                del params_list
            except Exception as e:
                self.logger.warning(
                    f"无法提取 {agent_type.value} 的网络参数: {e}"
                )
                continue

        if not network_params:
            return

        # 使用共享内存同步
        new_shm_memories: dict[AgentType, SharedNetworkMemory] = {}
        try:
            metadata_map: dict[AgentType, SharedNetworkMetadata] = {}

            for agent_type, params in network_params.items():
                shm_mem = SharedNetworkMemory()
                metadata = shm_mem.create_and_fill(
                    agent_type=agent_type,
                    network_params=params,
                    generation=self.generation,
                )
                metadata_map[agent_type] = metadata
                new_shm_memories[agent_type] = shm_mem

            # 发送元数据给所有 Worker，等待 ack
            self._arena_worker_pool.attach_shared_networks(metadata_map)

            # Worker 都已 ack，可以安全清理上一代的共享内存
            for shm_mem in self._prev_shared_network_memories.values():
                shm_mem.close_and_unlink()
            self._prev_shared_network_memories.clear()

            # 当前的变为上一代
            self._prev_shared_network_memories = self._shared_network_memories
            self._shared_network_memories = new_shm_memories

        except Exception as e:
            self.logger.warning(
                f"共享内存同步失败，回退到 Queue 模式: {e}"
            )
            # 清理已创建的共享内存
            for shm_mem in new_shm_memories.values():
                shm_mem.close_and_unlink()
            # 回退到原有模式
            self._arena_worker_pool.update_networks(network_params)

    # ========================================================================
    # 训练轮次 (run_round) 和适应度汇总
    # ========================================================================

    def run_round(
        self,
        episode_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """运行一轮训练（所有竞技场的所有 episode + 进化）

        Workers 独立运行完整 episode tick 循环，主进程仅汇总 fitness 并执行进化。

        Args:
            episode_callback: 每个 episode 完成后的回调函数

        Returns:
            本轮统计信息
        """
        if not self._is_setup:
            raise RuntimeError("训练环境未初始化，请先调用 setup()")

        # 确保 _is_running 为 True
        self._is_running = True

        round_start = time.perf_counter()
        stats: dict[str, Any] = {}

        # 1. 运行所有竞技场的所有 episode
        arena_start = time.perf_counter()
        arena_fitnesses: list[dict[tuple[AgentType, int], np.ndarray]] = []
        episode_counts: list[int] = []

        assert self._arena_worker_pool is not None

        # 禁用GC，避免 tick 期间的 GC 停顿导致性能抖动
        gc.disable()

        try:
            for ep_idx in range(self.multi_config.episodes_per_arena):
                # Workers 独立运行一个 episode
                results: list[EpisodeResult] = self._arena_worker_pool.run_episodes(
                    num_episodes=1,
                    episode_length=self.config.training.episode_length,
                )

                # 汇总 fitness
                episode_fitness = self._aggregate_worker_fitness(results)
                arena_fitnesses.append(episode_fitness)
                episode_counts.append(self.multi_config.num_arenas)

                # League 训练钩子：per-arena fitness
                for result in results:
                    for arena_id, per_type_fitness in result.per_arena_fitness.items():
                        for agent_type, fitness_arr in per_type_fitness.items():
                            self._on_arena_fitness_collected(
                                arena_id, agent_type, fitness_arr, 0.0
                            )

                # Episode 回调
                if episode_callback is not None:
                    episode_stats = self._build_episode_stats_from_results(
                        results, ep_idx
                    )
                    episode_callback(episode_stats)

                # GC 策略：每个 episode 清理年轻代，每 10 个 episode 清理全部并释放内存
                if (ep_idx + 1) % 10 == 0:
                    gc.collect(0)
                    gc.collect(1)
                    gc.collect(2)
                    malloc_trim()
                else:
                    gc.collect(0)
        finally:
            # 确保 GC 重新启用
            gc.enable()

        stats["arena_run_time"] = time.perf_counter() - arena_start

        # 2. 汇总适应度
        avg_fitness = self._collect_fitness_all_arenas(arena_fitnesses, episode_counts)

        # 【内存泄漏修复】汇总后清理 arena_fitnesses（包含多个 episode 的适应度数据）
        episodes_this_round = sum(episode_counts)
        del arena_fitnesses
        del episode_counts

        # 3. 执行 NEAT 进化
        evolve_start = time.perf_counter()
        fitness_map = self._build_fitness_map(avg_fitness)

        assert self.evolution_worker_pool is not None

        # 首次进化时需要同步基因组到 Worker
        if not self._worker_pool_synced:
            genomes_map: dict[tuple[AgentType, int], tuple[np.ndarray, ...]] = {}
            for agent_type, pop in self.populations.items():
                if isinstance(pop, SubPopulationManager):
                    for i, sub_pop in enumerate(pop.sub_populations):
                        genome_data = _serialize_genomes_numpy(sub_pop.neat_pop.population)
                        genomes_map[(agent_type, i)] = genome_data
                else:
                    genome_data = _serialize_genomes_numpy(pop.neat_pop.population)
                    genomes_map[(agent_type, 0)] = genome_data
            self.evolution_worker_pool.set_genomes(genomes_map)
            # 【内存泄漏修复】同步后立即清理 genomes_map 并强制释放内存
            del genomes_map
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
            malloc_trim()
            self.logger.info("首次进化：基因组已同步到 Worker 池")

        evolution_results = self.evolution_worker_pool.evolve_all_parallel(
            fitness_map, lite=True
        )
        self._worker_pool_synced = True
        stats["evolve_time"] = time.perf_counter() - evolve_start

        # 【内存泄漏修复】进化完成后清理 fitness_map
        del fitness_map

        # 4. 更新种群
        update_start = time.perf_counter()
        self._update_populations_from_evolution(evolution_results)

        # 【内存泄漏修复】更新完成后清理 evolution_results（包含大量 NumPy 数组）
        del evolution_results

        # 进化后更新网络缓存（主进程保留）
        self._update_network_caches()

        # 重新创建竞技场状态中的 Agent 账户状态
        self._refresh_agent_states()

        # 同步新网络参数到 Arena Workers
        self._sync_networks_to_workers()

        stats["update_time"] = time.perf_counter() - update_start

        # 增加代数和总 episode 计数
        self.generation += 1
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
            f"evolve={stats['evolve_time']:.2f}s, "
            f"update={stats['update_time']:.2f}s, "
            f"total={stats['total_time']:.2f}s"
        )

        return stats

    def _aggregate_worker_fitness(
        self, results: list[EpisodeResult]
    ) -> dict[tuple[AgentType, int], np.ndarray]:
        """汇总所有 Worker 的 fitness 结果

        Args:
            results: 各 Worker 返回的 EpisodeResult 列表

        Returns:
            按 (AgentType, sub_pop_id) 汇总的适应度字典
        """
        accumulated: dict[tuple[AgentType, int], np.ndarray] = {}
        for result in results:
            for key, fitness_arr in result.accumulated_fitness.items():
                if key not in accumulated:
                    accumulated[key] = fitness_arr.copy()
                else:
                    accumulated[key] += fitness_arr
        return accumulated

    def _build_episode_stats_from_results(
        self, results: list[EpisodeResult], ep_idx: int
    ) -> dict[str, Any]:
        """从 Worker 结果构建 episode 统计信息

        Args:
            results: 各 Worker 返回的 EpisodeResult 列表
            ep_idx: 当前轮次内的 episode 索引（0-based）

        Returns:
            Episode 统计信息字典
        """
        high_prices: list[float] = []
        low_prices: list[float] = []
        arena_ticks: list[int] = []
        end_reasons: list[str | None] = []
        end_ticks: list[int] = []

        for result in results:
            for arena_id, arena_stats in result.arena_stats.items():
                high_prices.append(arena_stats.high_price)
                low_prices.append(arena_stats.low_price)
                arena_ticks.append(arena_stats.end_tick)
                end_reasons.append(arena_stats.end_reason)
                end_ticks.append(arena_stats.end_tick)

        global_high = max(high_prices) if high_prices else 0.0
        global_low = min(low_prices) if low_prices else 0.0
        global_episode = (
            self.generation * self.multi_config.episodes_per_arena + ep_idx + 1
        )

        return {
            "episode": global_episode,
            "episode_in_round": ep_idx + 1,
            "generation": self.generation,
            "num_arenas": self.multi_config.num_arenas,
            "high_price": global_high,
            "low_price": global_low,
            "arena_high_prices": high_prices,
            "arena_low_prices": low_prices,
            "arena_ticks": arena_ticks,
            "arena_end_reasons": end_reasons,
            "arena_end_ticks": end_ticks,
        }

    # ========================================================================
    # 适应度汇总和进化
    # ========================================================================

    def _on_arena_fitness_collected(
        self,
        arena_id: int,
        agent_type: AgentType,
        fitness: np.ndarray,
        current_price: float,
    ) -> None:
        """竞技场适应度收集钩子（子类可重写）

        在每个竞技场的每个种群适应度计算完成后调用。

        Args:
            arena_id: 竞技场 ID
            agent_type: Agent 类型
            fitness: 适应度数组
            current_price: 当前价格
        """
        pass  # 默认空实现

    def _collect_fitness_all_arenas(
        self,
        arena_fitnesses: list[dict[tuple[AgentType, int], np.ndarray]],
        episode_counts: list[int],
    ) -> dict[tuple[AgentType, int], np.ndarray]:
        """收集并汇总所有竞技场的适应度"""
        return FitnessAggregator.aggregate_simple_average(
            arena_fitnesses, episode_counts
        )

    def _build_fitness_map(
        self,
        avg_fitness: dict[tuple[AgentType, int], np.ndarray],
    ) -> dict[tuple[AgentType, int], np.ndarray]:
        """构建进化所需的适应度映射"""
        fitness_map: dict[tuple[AgentType, int], np.ndarray] = {}

        for agent_type, population in self.populations.items():
            if isinstance(population, SubPopulationManager):
                for i, sub_pop in enumerate(population.sub_populations):
                    key = (agent_type, i)
                    if key in avg_fitness:
                        fitness_map[key] = avg_fitness[key]
                    else:
                        fitness_map[key] = np.zeros(
                            len(sub_pop.agents), dtype=np.float32
                        )
            else:
                key = (agent_type, 0)
                if key in avg_fitness:
                    fitness_map[key] = avg_fitness[key]
                else:
                    fitness_map[key] = np.zeros(
                        len(population.agents), dtype=np.float32
                    )

        return fitness_map

    # ========================================================================
    # 种群更新
    # ========================================================================

    def _update_populations_from_evolution(
        self,
        evolution_results: dict[
            tuple[AgentType, int],
            tuple[
                tuple[np.ndarray, ...] | None,
                tuple[np.ndarray, ...],
                tuple[np.ndarray, np.ndarray],
            ],
        ],
        deserialize_genomes: bool = False,
    ) -> None:
        """从进化结果更新种群

        Args:
            evolution_results: 进化结果字典，包含 (genome_data, network_params_data, species_data)
                genome_data 在 lite 模式下为 None
            deserialize_genomes: 是否反序列化基因组（默认 False，延迟反序列化）
        """
        # 按 agent_type 收集 network_params_data 用于批量缓存更新
        # 使用 (sub_pop_id, data) 元组确保按子种群顺序拼接
        network_params_by_type: dict[AgentType, list[tuple[int, tuple[np.ndarray, ...]]]] = {}

        for (agent_type, sub_pop_id), (
            genome_data,
            network_params_data,
            species_data,
        ) in evolution_results.items():
            population = self.populations.get(agent_type)
            if population is None:
                continue

            # 收集 network_params_data（带 sub_pop_id 用于排序）
            if agent_type not in network_params_by_type:
                network_params_by_type[agent_type] = []
            network_params_by_type[agent_type].append((sub_pop_id, network_params_data))

            if genome_data is None:
                # Lite 模式：genome_data 为 None，跳过 _update_single_population
                # 仅更新 generation 和标记 dirty
                if isinstance(population, SubPopulationManager):
                    if sub_pop_id < len(population.sub_populations):
                        sub_pop = population.sub_populations[sub_pop_id]
                        sub_pop.generation += 1
                        sub_pop._genomes_dirty = True
                        sub_pop._pending_genome_data = None
                        sub_pop._pending_species_data = species_data
                else:
                    if sub_pop_id == 0:
                        population.generation += 1
                        population._genomes_dirty = True
                        population._pending_genome_data = None
                        population._pending_species_data = species_data
            else:
                if isinstance(population, SubPopulationManager):
                    if sub_pop_id < len(population.sub_populations):
                        sub_pop = population.sub_populations[sub_pop_id]
                        self._update_single_population(
                            sub_pop,
                            genome_data,
                            network_params_data,
                            species_data,
                            deserialize_genomes,
                        )
                else:
                    if sub_pop_id == 0:
                        self._update_single_population(
                            population,
                            genome_data,
                            network_params_data,
                            species_data,
                            deserialize_genomes,
                        )

        # 拼接并缓存到 population 对象上，供 _update_network_caches 使用
        for agent_type, id_params_list in network_params_by_type.items():
            population = self.populations.get(agent_type)
            if population is not None:
                # 按 sub_pop_id 排序确保拼接顺序与 agent 索引一致
                id_params_list.sort(key=lambda x: x[0])
                sorted_params = [data for _, data in id_params_list]
                population._cached_network_params_data = _concat_network_params_numpy(sorted_params)
        del network_params_by_type

        # 所有种群更新完成后统一清理
        gc.collect(0)

    def _update_single_population(
        self,
        population: Population,
        genome_data: tuple[np.ndarray, ...] | None,
        network_params_data: tuple[np.ndarray, ...],
        species_data: tuple[np.ndarray, np.ndarray],
        deserialize_genomes: bool = False,
    ) -> None:
        """更新单个种群

        Args:
            population: 种群对象
            genome_data: 基因组数据元组（lite 模式下可为 None）
            network_params_data: 网络参数数据元组
            species_data: species 数据元组 (genome_ids, species_ids)
            deserialize_genomes: 是否反序列化基因组（默认 False，延迟反序列化）
        """
        # 增加代数
        population.generation += 1

        if deserialize_genomes:
            # 解包网络参数
            params_list = _unpack_network_params_numpy(*network_params_data)

            # 完整反序列化：重建 NEAT 种群
            assert genome_data is not None, "deserialize_genomes=True 时 genome_data 不能为 None"
            old_genomes = list(population.neat_pop.population.values())

            keys, fitnesses, metadata, nodes, conns = genome_data
            population.neat_pop.population = _deserialize_genomes_numpy(
                keys,
                fitnesses,
                metadata,
                nodes,
                conns,
                population.neat_config.genome_config,
            )

            new_genome_ids = set(population.neat_pop.population.keys())
            old_to_clean = [g for g in old_genomes if g.key not in new_genome_ids]
            population._cleanup_genome_internals(old_to_clean)

            # 应用 species 数据（在 cleanup 之前）
            species_genome_ids, species_species_ids = species_data
            _apply_species_data_to_population(
                population.neat_pop,
                species_genome_ids,
                species_species_ids,
                population.generation,
            )

            # 注：NEAT 历史数据清理已移至 save_checkpoint 时统一执行

            # 更新 Agent Brain（使用完整的 genome + params）
            new_genomes = list(population.neat_pop.population.items())
            for idx, (_gid, genome) in enumerate(new_genomes):
                if idx < len(population.agents) and idx < len(params_list):
                    population.agents[idx].brain.update_from_network_params(
                        genome, params_list[idx]
                    )
            # 释放中间变量
            for p in params_list:
                p.clear()
            params_list.clear()
            del params_list
            del old_genomes
            del old_to_clean
            del new_genomes
        else:
            # 并行竞技场模式：跳过 brain 更新
            # ArenaWorker 使用 BatchNetworkCache 推理，主进程不调用 brain.forward()
            population._pending_genome_data = genome_data
            population._pending_species_data = species_data
            population._genomes_dirty = True

    def _refresh_agent_states(self, force: bool = False) -> None:
        """刷新所有竞技场的 Agent 账户状态

        性能优化：进化后 agent_id 保持不变，不需要重新创建 AgentAccountState 对象。
        只在以下情况需要完整重建：
        1. force=True（checkpoint 恢复时）
        2. agent_states 为空（首次创建）

        Args:
            force: 是否强制重建所有状态
        """
        # 快速路径：检查是否需要重建
        needs_rebuild = force
        if not needs_rebuild and self.arena_states:
            # 检查第一个竞技场的状态是否已存在且数量匹配
            first_arena = self.arena_states[0]
            expected_count = sum(len(p.agents) for p in self.populations.values())
            if len(first_arena.agent_states) != expected_count:
                needs_rebuild = True

        if needs_rebuild:
            # 完整路径：重新创建所有状态
            for arena in self.arena_states:
                arena.agent_states.clear()
                for population in self.populations.values():
                    for agent in population.agents:
                        state = AgentAccountState.from_agent(agent)
                        arena.agent_states[agent.agent_id] = state
            # 强制重建映射表
            self._build_agent_map(force=True)
        else:
            # 快速路径：只更新映射表（如果需要）
            self._build_agent_map()

    # ========================================================================
    # 主训练循环
    # ========================================================================

    def train(
        self,
        num_rounds: int | None = None,
        checkpoint_callback: Callable[[int], None] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        episode_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """主训练循环

        Args:
            num_rounds: 训练轮数，None 表示无限循环
            checkpoint_callback: 检查点回调函数，参数为当前代数
            progress_callback: 进度回调函数，参数为本轮统计信息
            episode_callback: Episode 回调函数，每个 episode 完成后调用
        """
        if not self._is_setup:
            self.setup()

        self._is_running = True
        round_count = 0

        try:
            while self._is_running:
                if num_rounds is not None and round_count >= num_rounds:
                    break

                stats = self.run_round(episode_callback=episode_callback)
                round_count += 1

                if progress_callback is not None:
                    progress_callback(stats)

                if checkpoint_callback is not None:
                    checkpoint_callback(self.generation)

        except KeyboardInterrupt:
            self.logger.info("收到中断信号，停止训练...")
        finally:
            self._is_running = False
            self.logger.info(f"训练完成，共运行 {round_count} 轮")

    # ========================================================================
    # 检查点管理
    # ========================================================================

    def save_checkpoint(self, path: str) -> None:
        """保存检查点

        使用精简格式：只保存基因组核心数据和 species 映射，不保存 NEAT 历史数据。
        这样 checkpoint 文件更小，加载时也不会带入任何历史数据。

        Args:
            path: 检查点文件路径
        """
        # Sync genomes from workers if needed (lite 模式下 genome_data 为 None)
        if self.evolution_worker_pool is not None:
            needs_sync: bool = False
            for agent_type, population in self.populations.items():
                if isinstance(population, SubPopulationManager):
                    for sub_pop in population.sub_populations:
                        if sub_pop._genomes_dirty and sub_pop._pending_genome_data is None:
                            needs_sync = True
                            break
                else:
                    if population._genomes_dirty and population._pending_genome_data is None:
                        needs_sync = True
                if needs_sync:
                    break

            if needs_sync:
                self.logger.info("Checkpoint 保存前同步基因组数据...")
                genomes_map = self.evolution_worker_pool.sync_genomes_from_workers()
                # Apply synced genomes to populations
                for (agent_type, sub_pop_id), genome_data in genomes_map.items():
                    population = self.populations.get(agent_type)
                    if population is None:
                        continue
                    if isinstance(population, SubPopulationManager):
                        if sub_pop_id < len(population.sub_populations):
                            sub_pop = population.sub_populations[sub_pop_id]
                            sub_pop._pending_genome_data = genome_data
                    else:
                        if sub_pop_id == 0:
                            population._pending_genome_data = genome_data
                del genomes_map

        checkpoint_data: dict[str, Any] = {
            "checkpoint_version": 2,  # 新版本标识，用于区分精简格式
            "generation": self.generation,
            "populations": {},
        }

        for agent_type, population in self.populations.items():
            if isinstance(population, SubPopulationManager):
                pop_data: dict[str, Any] = {
                    "is_sub_population_manager": True,
                    "sub_population_count": population.sub_population_count,
                    "sub_populations": [],
                }
                for sub_pop in population.sub_populations:
                    # 优先使用 pending 数据（跳过反序列化->序列化往返）
                    if sub_pop._genomes_dirty and sub_pop._pending_genome_data is not None:
                        genome_data = sub_pop._pending_genome_data
                        species_data = sub_pop._pending_species_data or (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
                    else:
                        # 需要先清理 NEAT 历史数据
                        sub_pop._cleanup_neat_history_light()
                        genome_data = _serialize_genomes_numpy(sub_pop.neat_pop.population)
                        species_data = _serialize_species_data(sub_pop.neat_pop.species)
                    sub_pop_data = {
                        "generation": sub_pop.generation,
                        "genome_data": genome_data,
                        "species_data": species_data,
                    }
                    pop_data["sub_populations"].append(sub_pop_data)
                checkpoint_data["populations"][agent_type] = pop_data
            else:
                # 优先使用 pending 数据
                if population._genomes_dirty and population._pending_genome_data is not None:
                    genome_data = population._pending_genome_data
                    species_data = population._pending_species_data or (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
                else:
                    population._cleanup_neat_history_light()
                    genome_data = _serialize_genomes_numpy(population.neat_pop.population)
                    species_data = _serialize_species_data(population.neat_pop.species)
                checkpoint_data["populations"][agent_type] = {
                    "generation": population.generation,
                    "genome_data": genome_data,
                    "species_data": species_data,
                }

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        # 【内存泄漏修复】保存后显式删除 checkpoint_data 及其包含的 numpy 数组
        del checkpoint_data
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        malloc_trim()

        self.logger.info(f"检查点已保存: {path}")

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str | None:
        """查找最新的多竞技场检查点文件

        按 generation 数字从大到小排序，返回最新的检查点路径。
        支持的格式：
        - parallel_arena_gen_*.pkl（多竞技场）
        - ep_*.pkl（单竞技场兼容）

        Args:
            checkpoint_dir: 检查点目录路径

        Returns:
            最新检查点的路径，如果不存在则返回 None
        """
        import re

        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None

        # 优先查找 parallel_arena_gen_*.pkl 文件
        pattern = re.compile(r"parallel_arena_gen_(\d+)\.pkl$")
        checkpoints: list[tuple[int, Path]] = []

        for f in checkpoint_path.glob("parallel_arena_gen_*.pkl"):
            match = pattern.match(f.name)
            if match:
                generation = int(match.group(1))
                checkpoints.append((generation, f))

        # 如果没有多竞技场checkpoint，尝试查找单竞技场checkpoint（兼容性）
        if not checkpoints:
            pattern_single = re.compile(r"ep_(\d+)\.pkl$")
            for f in checkpoint_path.glob("ep_*.pkl"):
                match = pattern_single.match(f.name)
                if match:
                    episode = int(match.group(1))
                    checkpoints.append((episode, f))

        if not checkpoints:
            return None

        # 按 generation/episode 数字降序排序，取最新的
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return str(checkpoints[0][1])

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """加载检查点

        支持多种检查点格式：
        - version 2（精简格式）：只包含基因组核心数据和 species 映射
        - version 1 / 无版本（旧格式）：包含完整的 neat_pop 对象

        Args:
            path: 检查点文件路径
        """
        with open(path, "rb") as f:
            magic = f.read(2)

        if magic == b"\x1f\x8b":
            with gzip.open(path, "rb") as f:
                checkpoint_data = pickle.load(f)
        else:
            with open(path, "rb") as f:
                checkpoint_data = pickle.load(f)

        # 检测 checkpoint 版本
        checkpoint_version = checkpoint_data.get("checkpoint_version", 1)

        # 兼容多训练场和单训练场检查点格式
        self.generation = checkpoint_data.get(
            "generation", checkpoint_data.get("episode", 0)
        )
        populations_data = checkpoint_data.get("populations", {})

        for agent_type, pop_data in populations_data.items():
            if agent_type not in self.populations:
                continue

            population = self.populations[agent_type]

            if isinstance(population, SubPopulationManager):
                if pop_data.get("is_sub_population_manager"):
                    sub_pops_data = pop_data.get("sub_populations", [])
                    for i, sub_pop_data in enumerate(sub_pops_data):
                        if i < len(population.sub_populations):
                            sub_pop = population.sub_populations[i]
                            sub_pop.generation = sub_pop_data.get("generation", 0)

                            if checkpoint_version >= 2 and "genome_data" in sub_pop_data:
                                # 新格式：从精简数据重建
                                self._load_population_from_compact_data(
                                    sub_pop, sub_pop_data
                                )
                            else:
                                # 旧格式：直接使用 neat_pop
                                sub_pop.neat_pop = sub_pop_data.get("neat_pop")
                                # 【关键修复】清理旧格式 checkpoint 中的历史数据，防止内存泄漏
                                sub_pop._cleanup_neat_history_light()
                                genomes = list(sub_pop.neat_pop.population.items())
                                sub_pop.agents = sub_pop.create_agents(genomes)
                else:
                    self.logger.warning(f"{agent_type.value} 检查点为旧格式，需要迁移")
            else:
                population.generation = pop_data.get("generation", 0)

                if checkpoint_version >= 2 and "genome_data" in pop_data:
                    # 新格式：从精简数据重建
                    self._load_population_from_compact_data(population, pop_data)
                else:
                    # 旧格式：直接使用 neat_pop
                    population.neat_pop = pop_data.get("neat_pop")
                    # 【关键修复】清理旧格式 checkpoint 中的历史数据，防止内存泄漏
                    population._cleanup_neat_history_light()
                    genomes = list(population.neat_pop.population.items())
                    population.agents = population.create_agents(genomes)

        # 更新网络缓存
        self._update_network_caches()

        # 刷新竞技场的 Agent 状态（checkpoint 恢复后需要强制重建）
        self._refresh_agent_states(force=True)

        # 重置 Worker 池同步标志
        self._worker_pool_synced = False

        # 同步网络参数到 Arena Workers
        self._sync_networks_to_workers()

        # 【内存泄漏修复】加载完成后释放临时对象
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        malloc_trim()

        self.logger.info(
            f"检查点已加载: {path}, generation={self.generation}, "
            f"version={checkpoint_version}"
        )

        return checkpoint_data

    def _load_population_from_compact_data(
        self, population: Population, pop_data: dict[str, Any]
    ) -> None:
        """从精简格式数据加载种群

        Args:
            population: Population 对象
            pop_data: 包含 genome_data 和 species_data 的字典
        """
        genome_data = pop_data["genome_data"]
        species_data = pop_data.get("species_data", (np.array([]), np.array([])))

        # 1. 反序列化基因组
        keys, fitnesses, metadata, all_nodes, all_conns = genome_data
        population.neat_pop.population = _deserialize_genomes_numpy(
            keys, fitnesses, metadata, all_nodes, all_conns,
            population.neat_config.genome_config,
        )

        # 2. 应用 species 数据
        species_genome_ids, species_species_ids = species_data
        _apply_species_data_to_population(
            population.neat_pop,
            species_genome_ids,
            species_species_ids,
            population.generation,
        )

        # 3. 重建 Agent
        genomes = list(population.neat_pop.population.items())
        population.agents = population.create_agents(genomes)

        # 清理 NEAT 历史数据
        population._cleanup_neat_history_light()

    # ========================================================================
    # 停止和资源清理
    # ========================================================================

    def stop(self) -> None:
        """停止训练并清理资源"""
        self._is_running = False

        # 关闭 Arena Worker Pool
        if self._arena_worker_pool is not None:
            self._arena_worker_pool.shutdown()
            self._arena_worker_pool = None

        # 清理共享内存
        for shm_mem in self._shared_network_memories.values():
            shm_mem.close_and_unlink()
        self._shared_network_memories.clear()
        for shm_mem in self._prev_shared_network_memories.values():
            shm_mem.close_and_unlink()
        self._prev_shared_network_memories.clear()

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
        self.logger.info("多竞技场并行训练器已停止")

    # ========================================================================
    # 测试模式
    # ========================================================================

    def setup_for_testing(
        self, populations_data: dict[AgentType, list[bytes]]
    ) -> None:
        """测试模式初始化：从 genome 数据创建种群

        不创建进化 Worker 池和 Arena Worker Pool，仅用于评估适应度。

        Args:
            populations_data: 各物种的序列化基因组列表字典
        """
        import pickle as _pickle
        import neat
        from src.bio.agents.base import Agent
        from src.bio.brain.brain import Brain
        from src.bio.agents.retail_pro import RetailProAgent
        from src.bio.agents.market_maker import MarketMakerAgent
        from src.training.population import Population, SubPopulationManager

        _AGENT_CLASSES: dict[AgentType, type] = {
            AgentType.RETAIL_PRO: RetailProAgent,
            AgentType.MARKET_MAKER: MarketMakerAgent,
        }
        _AGENT_ID_OFFSET: dict[AgentType, int] = {
            AgentType.RETAIL_PRO: 0,
            AgentType.MARKET_MAKER: 1_000_000,
        }

        # 加载 NEAT 配置
        config_dir = Path(self.config.training.neat_config_path)
        neat_configs: dict[AgentType, neat.Config] = {}
        for agent_type in AgentType:
            if agent_type == AgentType.MARKET_MAKER:
                neat_config_path = config_dir / "neat_market_maker.cfg"
            else:
                neat_config_path = config_dir / "neat_retail_pro.cfg"
            neat_configs[agent_type] = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                str(neat_config_path),
            )

        # 从 genome 数据创建种群
        for agent_type in AgentType:
            genome_data_list = populations_data.get(agent_type, [])
            if not genome_data_list:
                continue

            # 反序列化所有 genome
            genome_list: list[neat.DefaultGenome] = []
            for gd in genome_data_list:
                genome = _pickle.loads(gd)
                if isinstance(genome, neat.DefaultGenome):
                    genome_list.append(genome)

            if not genome_list:
                continue

            agent_config = self.config.agents[agent_type]
            neat_config = neat_configs[agent_type]
            agent_class = _AGENT_CLASSES[agent_type]
            agent_id_offset = _AGENT_ID_OFFSET[agent_type]

            # 创建 Agent 列表
            agents: list[Agent] = []
            for i in range(agent_config.count):
                genome = genome_list[i % len(genome_list)]
                brain = Brain.from_genome(genome, neat_config)
                if agent_type == AgentType.MARKET_MAKER:
                    agent = agent_class(agent_id_offset + i, brain, agent_config, as_config=self.config.as_model)
                else:
                    agent = agent_class(agent_id_offset + i, brain, agent_config)
                agents.append(agent)

            # 创建简化的 Population 对象
            pop = Population.__new__(Population)
            pop.agent_type = agent_type
            pop.agent_config = agent_config
            pop._as_config = self.config.as_model
            pop.generation = 0
            pop.logger = get_logger("population")
            pop._executor = None
            pop._num_workers = 8
            pop.neat_config = neat_config
            pop.neat_pop = None
            pop.agents = agents
            pop.sub_population_id = None
            pop._pending_genome_data = None
            pop._pending_species_data = None
            pop._genomes_dirty = False
            pop._accumulated_fitness = {}
            pop._accumulation_count = 0
            if agent_type == AgentType.MARKET_MAKER:
                pop.neat_config_path = str(config_dir / "neat_market_maker.cfg")
            else:
                pop.neat_config_path = str(config_dir / "neat_retail_pro.cfg")

            self.populations[agent_type] = pop

        # 创建竞技场状态
        self._create_arena_states()

        # 初始化网络缓存
        self._init_network_caches()

        # 构建 Agent 映射
        self._build_agent_map()

        # 刷新竞技场的 Agent 状态
        self._refresh_agent_states()

        # 不创建进化 Worker 池和 Arena Worker Pool
        self.evolution_worker_pool = None

    # ========================================================================
    # 上下文管理器
    # ========================================================================

    def __enter__(self) -> "ParallelArenaTrainer":
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
