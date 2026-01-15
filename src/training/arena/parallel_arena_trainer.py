"""多竞技场并行推理训练器模块

核心特性：将多个竞技场的神经网络推理合并成一个批量操作。

设计原则：
1. 神经网络共享：所有竞技场使用同一套 BatchNetworkCache，进化后统一更新
2. 账户状态独立：每个竞技场维护独立的 ArenaState
3. 批量推理合并：N 个竞技场 x M 个 Agent 的推理合并成单次 OpenMP 并行操作
4. 订单簿独立：每个竞技场有独立的 MatchingEngine 和 OrderBook
"""

import gc
import gzip
import pickle
import random
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
    use_shared_memory_ipc: bool = True  # 默认使用共享内存 IPC（性能提升约 21%）


if TYPE_CHECKING:
    from src.training._cython.batch_decide_openmp import BatchNetworkCache

from src.bio.agents.base import ActionType, AgentType
from src.config.config import CatfishMode, Config
from src.core.log_engine.logger import get_logger
from src.market.adl.adl_manager import ADLCandidate, ADLManager
from src.market.catfish import create_all_catfish, create_catfish
from src.market.market_state import NormalizedMarketState
from src.market.matching.matching_engine import MatchingEngine
from src.market.matching.trade import Trade
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.training.fast_math import log_normalize_signed, log_normalize_unsigned
from src.training.population import (
    MultiPopulationWorkerPool,
    Population,
    SubPopulationManager,
    WorkerConfig,
    _apply_species_data_to_population,
    _deserialize_genomes_numpy,
    _unpack_network_params_numpy,
    malloc_trim,
)

from .arena_state import (
    AgentAccountState,
    AgentStateAdapter,
    ArenaState,
    CatfishAccountState,
    calculate_order_quantity_from_state,
    calculate_skew_factor_from_state,
)
from .fitness_aggregator import FitnessAggregator

# 缓存类型常量
CACHE_TYPE_RETAIL = 0
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
    """多竞技场并行推理训练器

    核心特性：将 N 个竞技场 x M 个 Agent 的神经网络推理合并成单次 OpenMP 并行操作。

    Attributes:
        config: 全局配置
        multi_config: 多竞技场配置
        populations: 共享的种群（神经网络）
        arena_states: N 个独立的竞技场状态
        network_caches: 共享的网络缓存
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
        """初始化多竞技场并行推理训练器

        Args:
            config: 全局配置
            multi_config: 多竞技场配置，None 使用默认配置
        """
        self.config = config
        self.multi_config = multi_config or MultiArenaConfig()
        self.logger = get_logger("parallel_arena_trainer")

        self.populations = {}
        self.arena_states = []
        self.network_caches = None
        self.evolution_worker_pool = None
        self.generation = 0
        self.total_episodes = 0

        self._is_setup = False
        self._is_running = False
        self._worker_pool_synced = False

        # Execute Worker 池相关
        self._execute_worker_pool: Any = None  # ArenaExecuteWorkerPool | None
        self._use_execute_workers: bool = True  # 是否使用 Execute Worker 池

        # Agent ID 到 Agent 对象的映射表（O(1) 查找）
        self.agent_map: dict[int, Any] = {}

        # Agent ID 到网络索引的映射表（用于 decide_multi_arena）
        # 格式: {agent_type: {agent_id: network_index}}
        self._network_index_map: dict[AgentType, dict[int, int]] = {}

        # EMA 参数
        self._ema_alpha: float = 0.1

        # 预分配市场状态缓冲区
        self._market_state_buffers: dict[str, np.ndarray] = {
            "bid_data": np.zeros(200, dtype=np.float32),
            "ask_data": np.zeros(200, dtype=np.float32),
            "trade_prices": np.zeros(100, dtype=np.float32),
            "trade_quantities": np.zeros(100, dtype=np.float32),
            "tick_prices": np.zeros(100, dtype=np.float32),
            "tick_volumes": np.zeros(100, dtype=np.float32),
            "tick_amounts": np.zeros(100, dtype=np.float32),
        }

        # 缓存上次推理的数组结果（用于 Worker 池执行）
        # 格式: {AgentType: {arena_idx: (agent_ids_array, decisions_array)}}
        # decisions_array shape: (num_agents, 4), 列: [action_type, side, price, quantity]
        # agent_ids_array shape: (num_agents,), 与 decisions_array 行对应的 agent_id
        self._last_inference_arrays: dict[
            AgentType, dict[int, tuple[np.ndarray, np.ndarray]]
        ] = {}
        # Worker 池返回的订单簿快照（用于构建真实市场状态）
        # {arena_id: (bid_depth, ask_depth, last_price, mid_price)}
        self._worker_depth_cache: dict[
            int, tuple[np.ndarray, np.ndarray, float, float]
        ] = {}
        # 奇数数量趋势鲶鱼的方向平衡开关（交替多/空以避免系统性偏差）
        self._catfish_balance_bias_to_buy: bool = True

    def setup(self) -> None:
        """初始化：创建种群、竞技场状态、网络缓存、进化 Worker 池"""
        if self._is_setup:
            self.logger.warning("训练环境已初始化，跳过重复初始化")
            return

        self.logger.info("开始初始化多竞技场并行推理训练环境...")

        # 1. 创建种群（共享神经网络）
        self._create_populations()

        # 2. 创建竞技场状态（每个竞技场独立状态）
        self._create_arena_states()

        # 3. 初始化网络缓存（共享）
        self._init_network_caches()

        # 4. 构建 Agent 映射表
        self._build_agent_map()

        # 5. 创建进化 Worker 池
        self._create_evolution_worker_pool()

        # 读取 EMA 配置
        self._ema_alpha = self.config.market.ema_alpha

        # 6. 创建 Execute Worker 池（仅在禁用鲶鱼时使用）
        catfish_enabled = (
            self.config.catfish is not None and self.config.catfish.enabled
        )
        if self._use_execute_workers and not catfish_enabled:
            self._create_execute_worker_pool()

        self._is_setup = True
        self.logger.info(
            f"多竞技场并行推理训练环境初始化完成: "
            f"{self.multi_config.num_arenas} 个竞技场, "
            f"每竞技场 {self.multi_config.episodes_per_arena} 个 episode"
        )

    def _create_populations(self) -> None:
        """创建共享的种群（神经网络）"""
        self.logger.info("正在创建种群...")

        # RETAIL: 10个子种群
        self.populations[AgentType.RETAIL] = SubPopulationManager(
            self.config, AgentType.RETAIL, sub_count=10
        )

        # RETAIL_PRO: 1个种群
        self.populations[AgentType.RETAIL_PRO] = Population(
            AgentType.RETAIL_PRO, self.config
        )

        # WHALE: 1个种群
        self.populations[AgentType.WHALE] = Population(AgentType.WHALE, self.config)

        # MARKET_MAKER: 4个子种群
        self.populations[AgentType.MARKET_MAKER] = SubPopulationManager(
            self.config, AgentType.MARKET_MAKER, sub_count=4
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

            # 创建鲶鱼状态
            catfish_states: dict[int, CatfishAccountState] = {}
            if self.config.catfish and self.config.catfish.enabled:
                catfish_balance = self._calculate_catfish_initial_balance()
                whale_config = self.config.agents[AgentType.WHALE]

                if self.config.catfish.multi_mode:
                    catfish_list = create_all_catfish(
                        self.config.catfish,
                        initial_balance=catfish_balance,
                        leverage=whale_config.leverage,
                        maintenance_margin_rate=whale_config.maintenance_margin_rate,
                    )
                else:
                    catfish = create_catfish(
                        -1,
                        self.config.catfish,
                        initial_balance=catfish_balance,
                        leverage=whale_config.leverage,
                        maintenance_margin_rate=whale_config.maintenance_margin_rate,
                    )
                    catfish_list = [catfish]

                for catfish in catfish_list:
                    state = CatfishAccountState.from_catfish(catfish)
                    catfish_states[catfish.catfish_id] = state
                    # 注册鲶鱼到撮合引擎
                    matching_engine.register_agent(catfish.catfish_id, 0.0, 0.0)

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
                catfish_states=catfish_states,
                recent_trades=deque(maxlen=100),
                price_history=[initial_price],
                tick_history_prices=deque([initial_price], maxlen=100),
                tick_history_volumes=deque([0.0], maxlen=100),
                tick_history_amounts=deque([0.0], maxlen=100),
                smooth_mid_price=initial_price,
                tick=0,
                pop_liquidated_counts={agent_type: 0 for agent_type in AgentType},
                eliminating_agents=set(),
                episode_high_price=initial_price,
                episode_low_price=initial_price,
                catfish_liquidated=False,
            )

            self.arena_states.append(arena_state)

        self.logger.info(f"竞技场状态创建完成: {len(self.arena_states)} 个竞技场")

    def _build_agent_map(self) -> None:
        """构建 Agent ID 到 Agent 对象的映射表（O(1) 查找）"""
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

    def _calculate_catfish_initial_balance(self) -> float:
        """计算每条鲶鱼的初始资金"""
        agents_config = self.config.agents

        mm_config = agents_config[AgentType.MARKET_MAKER]
        mm_fund = mm_config.count * mm_config.initial_balance * mm_config.leverage

        other_fund = 0.0
        for agent_type in [AgentType.RETAIL, AgentType.RETAIL_PRO, AgentType.WHALE]:
            cfg = agents_config[agent_type]
            other_fund += cfg.count * cfg.initial_balance * cfg.leverage

        if other_fund >= mm_fund:
            raise ValueError(
                f"鲶鱼资金配置错误：做市商杠杆后资金({mm_fund})必须大于"
                f"其他物种杠杆后资金({other_fund})"
            )

        catfish_count = 3
        return (mm_fund - other_fund) / catfish_count

    def _balance_catfish_directions(self) -> None:
        """强制平衡趋势创造者鲶鱼的方向（跨所有竞技场）

        收集所有竞技场的趋势创造者鲶鱼，随机分配一半为买方向，一半为卖方向。
        这样可以确保无论有多少个竞技场，鲶鱼的买卖方向总是严格平衡的。
        """
        # 收集所有趋势创造者鲶鱼
        trend_creators: list[CatfishAccountState] = []
        for arena in self.arena_states:
            for catfish_state in arena.catfish_states.values():
                if catfish_state.catfish_mode == CatfishMode.TREND_CREATOR:
                    trend_creators.append(catfish_state)

        if not trend_creators:
            return

        # 随机打乱后分配方向
        random.shuffle(trend_creators)
        total = len(trend_creators)
        buy_target = total // 2
        sell_target = total // 2

        # 奇数数量时，额外的 1 条鲶鱼在多/空之间轮换，避免长期偏向
        if total % 2 == 1:
            if self._catfish_balance_bias_to_buy:
                buy_target += 1
            else:
                sell_target += 1
            self._catfish_balance_bias_to_buy = not self._catfish_balance_bias_to_buy

        for i, catfish in enumerate(trend_creators):
            if i < buy_target:
                catfish.current_direction = 1  # 买方向
            else:
                catfish.current_direction = -1  # 卖方向

        self.logger.debug(f"鲶鱼方向已平衡: 买方向={buy_target}, 卖方向={sell_target}")

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
            if agent_type == AgentType.RETAIL:
                cache_type = CACHE_TYPE_RETAIL
            elif agent_type == AgentType.MARKET_MAKER:
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
        """更新网络数据缓存（进化后调用）"""
        if self.network_caches is None:
            return

        for agent_type, population in self.populations.items():
            cache = self.network_caches.get(agent_type)
            if cache is None or not population.agents:
                continue

            networks = [agent.brain.network for agent in population.agents]
            cache.update_networks(networks)

    def _create_evolution_worker_pool(self) -> None:
        """创建进化 Worker 池"""
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

    def _create_execute_worker_pool(self) -> None:
        """创建 Execute Worker 池

        根据 multi_config.use_shared_memory_ipc 选择使用：
        - ArenaExecuteWorkerPoolShm（共享内存版，性能更优）
        - ArenaExecuteWorkerPool（Queue 版，兼容性更好）
        """
        arena_ids = [arena.arena_id for arena in self.arena_states]
        num_workers = min(4, len(arena_ids))  # 最多 4 个 Worker

        if self.multi_config.use_shared_memory_ipc:
            from .execute_worker import ArenaExecuteWorkerPoolShm

            self._execute_worker_pool = ArenaExecuteWorkerPoolShm(
                num_workers=num_workers,
                arena_ids=arena_ids,
                config=self.config,
            )
            ipc_mode = "共享内存"
        else:
            from .execute_worker import ArenaExecuteWorkerPool

            self._execute_worker_pool = ArenaExecuteWorkerPool(
                num_workers=num_workers,
                arena_ids=arena_ids,
                config=self.config,
            )
            ipc_mode = "Queue"

        self._execute_worker_pool.start()
        self.logger.info(
            f"Execute Worker 池创建完成: {num_workers} 个 Worker, "
            f"{len(arena_ids)} 个竞技场, IPC 模式: {ipc_mode}"
        )

    def _collect_fee_rates(self) -> dict[int, tuple[float, float]]:
        """收集所有 Agent 的费率

        Returns:
            费率映射字典，agent_id -> (maker_rate, taker_rate)
        """
        fee_rates: dict[int, tuple[float, float]] = {}

        for agent_type, population in self.populations.items():
            agent_config = population.agent_config
            maker_rate = agent_config.maker_fee_rate
            taker_rate = agent_config.taker_fee_rate

            for agent in population.agents:
                fee_rates[agent.agent_id] = (maker_rate, taker_rate)

        return fee_rates

    def run_round(
        self,
        episode_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """运行一轮训练（所有竞技场的所有 episode + 进化）

        Args:
            episode_callback: 每个 episode 完成后的回调函数

        Returns:
            本轮统计信息
        """
        if not self._is_setup:
            raise RuntimeError("训练环境未初始化，请先调用 setup()")

        # 确保 _is_running 为 True，否则 _run_episode_all_arenas 会立即返回
        self._is_running = True

        round_start = time.perf_counter()
        stats: dict[str, Any] = {}

        # 1. 运行所有竞技场的所有 episode
        arena_start = time.perf_counter()
        arena_fitnesses: list[dict[tuple[AgentType, int], np.ndarray]] = []
        episode_counts: list[int] = []

        # 禁用GC，避免tick期间的GC停顿导致性能抖动
        gc.disable()

        try:
            for ep_idx in range(self.multi_config.episodes_per_arena):
                # 重置所有竞技场
                self._reset_all_arenas()

                # 做市商初始化（每个竞技场）
                self._init_market_all_arenas()

                # 运行一个 episode（所有竞技场同步推进）
                episode_fitness = self._run_episode_all_arenas()

                # 累积适应度
                arena_fitnesses.append(episode_fitness)
                episode_counts.append(self.multi_config.num_arenas)

                # Episode 回调
                if episode_callback is not None:
                    episode_stats = self._get_episode_stats(ep_idx)
                    episode_callback(episode_stats)

                # 每个episode结束后执行轻量GC（只回收年轻代）
                gc.collect(0)
        finally:
            # 确保GC重新启用
            gc.enable()

        stats["arena_run_time"] = time.perf_counter() - arena_start

        # 2. 汇总适应度
        aggregate_start = time.perf_counter()
        avg_fitness = self._collect_fitness_all_arenas(arena_fitnesses, episode_counts)
        stats["aggregate_time"] = time.perf_counter() - aggregate_start

        # 3. 应用到基因组
        self._apply_fitness_to_genomes(avg_fitness)

        # 4. 执行 NEAT 进化
        evolve_start = time.perf_counter()
        fitness_map = self._build_fitness_map(avg_fitness)

        assert self.evolution_worker_pool is not None

        sync_genomes = not self._worker_pool_synced
        evolution_results = self.evolution_worker_pool.evolve_all_parallel(
            fitness_map, sync_genomes=sync_genomes
        )
        self._worker_pool_synced = True
        stats["evolve_time"] = time.perf_counter() - evolve_start

        # 5. 更新种群
        update_start = time.perf_counter()
        self._update_populations_from_evolution(evolution_results)

        # 进化后更新网络缓存
        self._update_network_caches()

        # 重新创建竞技场状态中的 Agent 账户状态
        self._refresh_agent_states()

        stats["update_time"] = time.perf_counter() - update_start

        # 增加代数和总 episode 计数
        self.generation += 1
        episodes_this_round = sum(episode_counts)
        self.total_episodes += episodes_this_round
        stats["generation"] = self.generation
        stats["episodes_this_round"] = episodes_this_round
        stats["total_episodes"] = self.total_episodes
        stats["avg_fitnesses"] = avg_fitness
        stats["species_fitness_stats"] = self._collect_species_fitness_stats()

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

    def _get_episode_stats(self, ep_idx: int) -> dict[str, Any]:
        """收集当前 episode 的统计信息

        Args:
            ep_idx: 当前轮次内的 episode 索引（0-based）

        Returns:
            Episode 统计信息字典
        """
        # 收集所有竞技场的最高价和最低价
        high_prices: list[float] = []
        low_prices: list[float] = []

        for arena in self.arena_states:
            high_prices.append(arena.episode_high_price)
            low_prices.append(arena.episode_low_price)

        # 计算全局最高/最低价（取各竞技场的极值）
        global_high = max(high_prices) if high_prices else 0.0
        global_low = min(low_prices) if low_prices else 0.0

        # 计算全局 episode 编号
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
        }

    def _collect_species_fitness_stats(self) -> dict[AgentType, dict[str, Any]]:
        """收集各物种的 NEAT species 适应度统计

        遍历各种群的 NEAT population，收集每个 species 的平均适应度。

        Returns:
            {AgentType: {"species_count": int, "species_avg_fitnesses": list[float]}}
        """
        result: dict[AgentType, dict[str, Any]] = {}

        for agent_type, population in self.populations.items():
            species_avg_fitnesses: list[float] = []

            # SubPopulationManager 需要遍历所有子种群
            if isinstance(population, SubPopulationManager):
                for sub_pop in population.sub_populations:
                    species_avg_fitnesses.extend(
                        self._get_species_fitnesses_from_neat_pop(sub_pop.neat_pop)
                    )
            else:
                # Population 直接访问 neat_pop
                species_avg_fitnesses.extend(
                    self._get_species_fitnesses_from_neat_pop(population.neat_pop)
                )

            result[agent_type] = {
                "species_count": len(species_avg_fitnesses),
                "species_avg_fitnesses": species_avg_fitnesses,
            }

        return result

    def _get_species_fitnesses_from_neat_pop(self, neat_pop: Any) -> list[float]:
        """从 NEAT population 获取各 species 的平均适应度

        Args:
            neat_pop: NEAT Population 对象

        Returns:
            各 species 的平均适应度列表
        """
        species_avg_fitnesses: list[float] = []

        if not hasattr(neat_pop, "species") or neat_pop.species is None:
            return species_avg_fitnesses

        species_set = neat_pop.species
        if not hasattr(species_set, "species") or not species_set.species:
            return species_avg_fitnesses

        for species in species_set.species.values():
            member_fitnesses = species.get_fitnesses()
            valid_fitnesses = [f for f in member_fitnesses if f is not None]
            if valid_fitnesses:
                avg = sum(valid_fitnesses) / len(valid_fitnesses)
                species_avg_fitnesses.append(avg)

        return species_avg_fitnesses

    def _reset_all_arenas(self) -> None:
        """重置所有竞技场状态"""
        initial_price = self.config.market.initial_price
        self._worker_depth_cache.clear()

        for arena in self.arena_states:
            # 重置订单簿
            arena.matching_engine._orderbook.clear(reset_price=initial_price)

            # 重置竞技场状态
            arena.reset_episode(initial_price)

            # 重置 Agent 账户状态
            for agent_type, population in self.populations.items():
                agent_config = population.agent_config
                for agent in population.agents:
                    state = arena.agent_states.get(agent.agent_id)
                    if state:
                        state.reset(agent_config)

            # 重置鲶鱼状态
            if self.config.catfish and self.config.catfish.enabled:
                catfish_balance = self._calculate_catfish_initial_balance()
                for catfish_state in arena.catfish_states.values():
                    catfish_state.reset(catfish_balance)

        # 强制平衡趋势创造者鲶鱼的方向（跨所有竞技场）
        if self.config.catfish and self.config.catfish.enabled:
            self._balance_catfish_directions()

        # 重置 Execute Worker 池的订单簿
        if self._execute_worker_pool is not None:
            fee_rates = self._collect_fee_rates()
            self._execute_worker_pool.reset_all(
                initial_price=initial_price,
                fee_rates=fee_rates,
            )

    def _prepare_mm_init_orders(
        self,
    ) -> dict[int, list[tuple[int, list[dict[str, float]], list[dict[str, float]]]]]:
        """准备所有竞技场的做市商初始化订单

        为每个竞技场的每个做市商准备初始挂单数据。

        Returns:
            各竞技场的做市商初始化数据，格式:
            {arena_id: [(agent_id, bid_orders, ask_orders), ...], ...}
        """
        mm_init_orders: dict[
            int, list[tuple[int, list[dict[str, float]], list[dict[str, float]]]]
        ] = {}

        mm_population = self.populations.get(AgentType.MARKET_MAKER)
        if not mm_population:
            return mm_init_orders

        mm_agents = list(mm_population.agents)

        for arena in self.arena_states:
            arena_orders: list[
                tuple[int, list[dict[str, float]], list[dict[str, float]]]
            ] = []
            orderbook = arena.matching_engine._orderbook

            for agent in mm_agents:
                agent_state = arena.agent_states.get(agent.agent_id)
                if agent_state is None:
                    continue

                # 计算市场状态
                market_state = self._compute_market_state_for_arena(arena)

                # 使用 Agent 进行推理，获取神经网络输出
                inputs = agent.observe(market_state, orderbook)  # type: ignore[attr-defined]
                outputs = agent.brain.forward(inputs)  # type: ignore[attr-defined]
                mid_price = market_state.mid_price
                tick_size = (
                    market_state.tick_size if market_state.tick_size > 0 else 0.01
                )

                # 使用 agent_state 解析输出
                _, params = self._parse_market_maker_output(
                    agent_state, np.array(outputs), mid_price, tick_size
                )

                bid_orders = params.get("bid_orders", [])
                ask_orders = params.get("ask_orders", [])
                arena_orders.append((agent.agent_id, bid_orders, ask_orders))

            mm_init_orders[arena.arena_id] = arena_orders

        return mm_init_orders

    def _init_market_all_arenas(self) -> None:
        """所有竞技场做市商初始化（使用批量推理）"""
        mm_population = self.populations.get(AgentType.MARKET_MAKER)
        if not mm_population:
            return

        # 使用 Worker 池初始化
        if self._execute_worker_pool is not None:
            mm_init_orders = self._prepare_mm_init_orders()
            results = self._execute_worker_pool.init_market_makers(mm_init_orders)

            # 处理返回结果，更新主进程中的状态
            for arena_id, result in results.items():
                arena = self.arena_states[arena_id]
                self._worker_depth_cache[arena_id] = (
                    result.bid_depth,
                    result.ask_depth,
                    result.last_price,
                    result.mid_price,
                )

                # 更新 Agent 账户状态（从成交结果）
                for trade_tuple in result.trades:
                    (
                        trade_id,
                        price,
                        qty,
                        buyer_id,
                        seller_id,
                        buyer_fee,
                        seller_fee,
                        is_buyer_taker,
                    ) = trade_tuple
                    # 更新 taker
                    taker_id = buyer_id if is_buyer_taker else seller_id
                    taker_state = arena.agent_states.get(taker_id)
                    if taker_state is not None:
                        fee = buyer_fee if is_buyer_taker else seller_fee
                        taker_state.on_trade(
                            price, qty, is_buyer_taker, fee, is_maker=False
                        )

                    # 更新 maker
                    maker_id = seller_id if is_buyer_taker else buyer_id
                    maker_state = arena.agent_states.get(maker_id)
                    if maker_state is not None:
                        maker_fee = seller_fee if is_buyer_taker else buyer_fee
                        maker_state.on_trade(
                            price, qty, not is_buyer_taker, maker_fee, is_maker=True
                        )

                # 更新做市商挂单 ID
                for agent_id, (bid_ids, ask_ids) in result.mm_order_updates.items():
                    agent_state = arena.agent_states.get(agent_id)
                    if agent_state is not None:
                        agent_state.bid_order_ids = list(bid_ids)
                        agent_state.ask_order_ids = list(ask_ids)

                # 同步订单簿价格到主进程（用于计算市场状态）
                arena.smooth_mid_price = result.mid_price

            return

        # 原来的串行初始化逻辑
        mm_agents = list(mm_population.agents)

        for arena in self.arena_states:
            orderbook = arena.matching_engine._orderbook

            for agent in mm_agents:
                agent_state = arena.agent_states.get(agent.agent_id)
                if agent_state is None:
                    continue

                # 计算市场状态
                market_state = self._compute_market_state_for_arena(arena)

                # 使用 Agent 进行推理，获取神经网络输出
                inputs = agent.observe(market_state, orderbook)  # type: ignore[attr-defined]
                outputs = agent.brain.forward(inputs)  # type: ignore[attr-defined]
                mid_price = market_state.mid_price
                tick_size = (
                    market_state.tick_size if market_state.tick_size > 0 else 0.01
                )

                # 使用 agent_state 解析输出
                _, params = self._parse_market_maker_output(
                    agent_state, np.array(outputs), mid_price, tick_size
                )

                # 使用新方法执行交易
                self._execute_mm_action_in_arena(arena, agent_state, params)

    def _compute_market_state_for_arena(
        self, arena: ArenaState
    ) -> NormalizedMarketState:
        """计算单个竞技场的归一化市场状态

        Args:
            arena: 竞技场状态

        Returns:
            归一化后的市场状态
        """
        orderbook = arena.matching_engine._orderbook
        cached_depth = None
        if (
            self._execute_worker_pool is not None
            and arena.arena_id in self._worker_depth_cache
        ):
            cached_depth = self._worker_depth_cache[arena.arena_id]

        # 获取实时参考价格
        if cached_depth is not None:
            _bid_depth, _ask_depth, last_price, mid_price = cached_depth
            current_mid_price = mid_price if mid_price > 0 else last_price
        else:
            current_mid_price = orderbook.get_mid_price()
            if current_mid_price is None:
                current_mid_price = orderbook.last_price
        if current_mid_price == 0:
            current_mid_price = 100.0

        # 更新 EMA 平滑价格
        arena.smooth_mid_price = (
            self._ema_alpha * current_mid_price
            + (1 - self._ema_alpha) * arena.smooth_mid_price
        )
        smooth_mid_price = arena.smooth_mid_price

        tick_size = orderbook.tick_size

        # 使用 get_depth_numpy 直接获取 NumPy 数组（Worker 池优先使用快照）
        if cached_depth is not None:
            bid_depth, ask_depth = _bid_depth, _ask_depth
        else:
            bid_depth, ask_depth = orderbook.get_depth_numpy(levels=100)

        # 获取并清零缓冲区
        bid_data = self._market_state_buffers["bid_data"]
        ask_data = self._market_state_buffers["ask_data"]
        trade_prices = self._market_state_buffers["trade_prices"]
        trade_quantities = self._market_state_buffers["trade_quantities"]
        tick_prices_normalized = self._market_state_buffers["tick_prices"]
        tick_volumes_normalized = self._market_state_buffers["tick_volumes"]
        tick_amounts_normalized = self._market_state_buffers["tick_amounts"]

        bid_data.fill(0)
        ask_data.fill(0)
        trade_prices.fill(0)
        trade_quantities.fill(0)
        tick_prices_normalized.fill(0)
        tick_volumes_normalized.fill(0)
        tick_amounts_normalized.fill(0)

        # 向量化买盘
        bid_prices = bid_depth[:, 0]
        bid_qtys = bid_depth[:, 1]
        bid_valid_mask = bid_prices > 0
        n_bids = int(np.sum(bid_valid_mask))
        if n_bids > 0 and smooth_mid_price > 0:
            bid_data[0 : n_bids * 2 : 2] = (
                bid_prices[:n_bids] - smooth_mid_price
            ) / smooth_mid_price
            bid_data[1 : n_bids * 2 : 2] = log_normalize_unsigned(bid_qtys[:n_bids])

        # 向量化卖盘
        ask_prices = ask_depth[:, 0]
        ask_qtys = ask_depth[:, 1]
        ask_valid_mask = ask_prices > 0
        n_asks = int(np.sum(ask_valid_mask))
        if n_asks > 0 and smooth_mid_price > 0:
            ask_data[0 : n_asks * 2 : 2] = (
                ask_prices[:n_asks] - smooth_mid_price
            ) / smooth_mid_price
            ask_data[1 : n_asks * 2 : 2] = log_normalize_unsigned(ask_qtys[:n_asks])

        # 向量化成交（直接迭代 deque，避免 list() 转换开销）
        n_trades = len(arena.recent_trades)
        if n_trades > 0:
            prices_arr = np.empty(n_trades, dtype=np.float32)
            qtys_arr = np.empty(n_trades, dtype=np.float32)
            for i, t in enumerate(arena.recent_trades):
                prices_arr[i] = t.price
                qtys_arr[i] = t.quantity if t.is_buyer_taker else -t.quantity

            if smooth_mid_price > 0:
                trade_prices[:n_trades] = (
                    prices_arr - smooth_mid_price
                ) / smooth_mid_price
            trade_quantities[:n_trades] = log_normalize_signed(qtys_arr)

        # Tick 历史价格归一化（使用 deque，maxlen=100，无需切片）
        if arena.tick_history_prices:
            hist_prices = np.array(arena.tick_history_prices, dtype=np.float32)
            volumes = np.array(arena.tick_history_volumes, dtype=np.float32)
            amounts = np.array(arena.tick_history_amounts, dtype=np.float32)
            n = len(hist_prices)

            base_price = hist_prices[0]
            if base_price > 0:
                tick_prices_normalized[-n:] = (hist_prices - base_price) / base_price

            tick_volumes_normalized[-n:] = log_normalize_signed(volumes)
            tick_amounts_normalized[-n:] = log_normalize_signed(amounts, scale=12.0)

        return NormalizedMarketState(
            mid_price=smooth_mid_price,
            tick_size=tick_size,
            bid_data=bid_data.copy(),
            ask_data=ask_data.copy(),
            trade_prices=trade_prices.copy(),
            trade_quantities=trade_quantities.copy(),
            tick_history_prices=tick_prices_normalized.copy(),
            tick_history_volumes=tick_volumes_normalized.copy(),
            tick_history_amounts=tick_amounts_normalized.copy(),
        )

    def _execute_with_worker_pool(
        self,
        all_decisions: dict[
            int, list[tuple[AgentAccountState, ActionType, dict[str, Any]]]
        ],
    ) -> dict[int, Any]:
        """使用 Worker 池执行所有竞技场的决策

        Args:
            all_decisions: 各竞技场的决策数据，格式:
                {arena_idx: [(state, action, params), ...]}

        Returns:
            各竞技场的执行结果
        """
        from .execute_worker import ArenaExecuteData

        arena_commands: dict[int, ArenaExecuteData] = {}

        for arena_idx, decisions in all_decisions.items():
            arena = self.arena_states[arena_idx]

            # 收集需要强平的 Agent
            liquidated_agents: list[tuple[int, int, bool]] = []
            for state in arena.agent_states.values():
                if state.is_liquidated and state.agent_id in arena.eliminating_agents:
                    is_mm = state.agent_type == AgentType.MARKET_MAKER
                    liquidated_agents.append(
                        (state.agent_id, state.position_quantity, is_mm)
                    )

            # 转换决策数据
            mm_decisions: list[
                tuple[int, list[dict[str, float]], list[dict[str, float]]]
            ] = []
            non_mm_decisions: list[tuple[int, int, int, float, int]] = []

            for state, action, params in decisions:
                if state.agent_type == AgentType.MARKET_MAKER:
                    bid_orders = params.get("bid_orders", [])
                    ask_orders = params.get("ask_orders", [])
                    mm_decisions.append((state.agent_id, bid_orders, ask_orders))
                else:
                    # 转换 action 到 int
                    action_int = 0  # HOLD
                    if action == ActionType.PLACE_BID:
                        action_int = 1
                    elif action == ActionType.PLACE_ASK:
                        action_int = 2
                    elif action == ActionType.CANCEL:
                        action_int = 3
                    elif action == ActionType.MARKET_BUY:
                        action_int = 4
                    elif action == ActionType.MARKET_SELL:
                        action_int = 5

                    if action_int == 0:  # HOLD
                        continue

                    # 方向
                    side_int = (
                        1
                        if action in (ActionType.PLACE_BID, ActionType.MARKET_BUY)
                        else 2
                    )

                    price = params.get("price", 0.0)
                    quantity = int(params.get("quantity", 0))

                    non_mm_decisions.append(
                        (state.agent_id, action_int, side_int, price, quantity)
                    )

            # 尝试从缓存的数组结果构建 decisions_array
            decisions_array = self._build_decisions_array_from_cache(arena_idx)

            arena_commands[arena_idx] = ArenaExecuteData(
                liquidated_agents=liquidated_agents,
                decisions=non_mm_decisions,
                mm_decisions=mm_decisions,
                decisions_array=decisions_array,
            )

        # 调用 Worker 池执行
        assert self._execute_worker_pool is not None
        return self._execute_worker_pool.execute_all(arena_commands)

    def _build_decisions_array_from_cache(self, arena_idx: int) -> np.ndarray | None:
        """从缓存的推理结果构建 decisions_array

        将各 AgentType 的缓存数组合并，添加 agent_id 列，
        并过滤掉 HOLD 动作（action_type == 0）。

        注意：Cython 返回的第 4 列是 quantity_ratio (0-1)，需要转换为实际 quantity。

        Args:
            arena_idx: 竞技场索引

        Returns:
            decisions_array: shape (N, 5)，列顺序 [agent_id, action_type, side, price, quantity]
            如果没有缓存或所有动作都是 HOLD，返回 None
        """
        if not self._last_inference_arrays:
            return None

        # 获取竞技场状态和 mid_price（优先使用平滑价，避免主进程 orderbook 不同步）
        arena = self.arena_states[arena_idx]
        mid_price = arena.smooth_mid_price
        if mid_price <= 0:
            mid_price = arena.matching_engine.orderbook.get_mid_price()
            if mid_price is None:
                mid_price = arena.matching_engine.orderbook.last_price
        if mid_price <= 0:
            mid_price = 100.0
        agent_states = arena.agent_states

        # 收集所有非做市商类型的数组
        arrays_to_concat: list[np.ndarray] = []

        for agent_type, arena_data in self._last_inference_arrays.items():
            if arena_idx not in arena_data:
                continue

            agent_ids, decisions = arena_data[arena_idx]
            if len(decisions) == 0:
                continue

            # 过滤掉 HOLD 动作（action_type == 0）
            # decisions 的列顺序是 [action_type, side, price, quantity_ratio]
            non_hold_mask = decisions[:, 0] != 0
            if not np.any(non_hold_mask):
                continue

            filtered_agent_ids = agent_ids[non_hold_mask]
            filtered_decisions = decisions[non_hold_mask].copy()  # 复制以避免修改原数组

            # 将 quantity_ratio (第 4 列，索引 3) 转换为实际 quantity
            for i in range(len(filtered_agent_ids)):
                agent_id = int(filtered_agent_ids[i])
                action_type_int = int(filtered_decisions[i, 0])
                price = float(filtered_decisions[i, 2])
                quantity_ratio = float(filtered_decisions[i, 3])

                state = agent_states.get(agent_id)
                if state is None:
                    filtered_decisions[i, 3] = 0
                    continue

                # 根据 action_type 判断是买还是卖
                # PLACE_BID=1, MARKET_BUY=4 -> is_buy=True
                # PLACE_ASK=2, MARKET_SELL=5 -> is_buy=False
                is_buy = action_type_int in (1, 4)

                # MARKET_SELL 特殊处理：如果有多仓，卖出仓位的比例
                if action_type_int == 5 and state.position_quantity > 0:
                    quantity = max(1, int(state.position_quantity * quantity_ratio))
                else:
                    price_for_qty = (
                        mid_price if action_type_int in (1, 2, 4, 5) else price
                    )
                    if price_for_qty <= 0:
                        price_for_qty = mid_price
                    quantity = calculate_order_quantity_from_state(
                        state,
                        price_for_qty,
                        quantity_ratio,
                        is_buy=is_buy,
                        ref_price=mid_price,
                    )

                filtered_decisions[i, 3] = quantity

            # 构建带 agent_id 的数组: [agent_id, action_type, side, price, quantity]
            full_array = np.column_stack(
                [
                    filtered_agent_ids.reshape(-1, 1),
                    filtered_decisions,
                ]
            )
            arrays_to_concat.append(full_array)

        if not arrays_to_concat:
            return None

        # 合并所有类型的数组
        combined_array = np.vstack(arrays_to_concat)
        return combined_array

    def _process_worker_results(
        self,
        results: dict[int, Any],
    ) -> dict[int, list[Trade]]:
        """处理 Worker 返回的执行结果

        更新各竞技场的 AgentAccountState，包括：
        - 根据成交更新账户余额和持仓
        - 更新挂单 ID

        Args:
            results: 各竞技场的执行结果

        Returns:
            各竞技场的成交列表（用于后续统计）
        """
        arena_tick_trades: dict[int, list[Trade]] = {}

        for arena_id, result in results.items():
            arena = self.arena_states[arena_id]
            tick_trades: list[Trade] = []
            self._worker_depth_cache[arena_id] = (
                result.bid_depth,
                result.ask_depth,
                result.last_price,
                result.mid_price,
            )

            # 处理成交
            for trade_tuple in result.trades:
                (
                    trade_id,
                    price,
                    qty,
                    buyer_id,
                    seller_id,
                    buyer_fee,
                    seller_fee,
                    is_buyer_taker,
                ) = trade_tuple

                # 创建 Trade 对象用于统计
                trade = Trade(
                    trade_id=trade_id,
                    price=price,
                    quantity=qty,
                    buyer_id=buyer_id,
                    seller_id=seller_id,
                    buyer_fee=buyer_fee,
                    seller_fee=seller_fee,
                    is_buyer_taker=is_buyer_taker,
                )
                tick_trades.append(trade)
                arena.recent_trades.append(trade)

                # 更新 taker 账户
                taker_id = buyer_id if is_buyer_taker else seller_id
                taker_state = arena.agent_states.get(taker_id)
                if taker_state is not None and not taker_state.is_liquidated:
                    fee = buyer_fee if is_buyer_taker else seller_fee
                    taker_state.on_trade(
                        price, qty, is_buyer_taker, fee, is_maker=False
                    )

                # 更新 maker 账户
                maker_id = seller_id if is_buyer_taker else buyer_id
                maker_state = arena.agent_states.get(maker_id)
                if maker_state is not None and not maker_state.is_liquidated:
                    maker_fee = seller_fee if is_buyer_taker else buyer_fee
                    maker_state.on_trade(
                        price, qty, not is_buyer_taker, maker_fee, is_maker=True
                    )

            # 更新非做市商挂单 ID
            for agent_id, pending_id in result.pending_updates.items():
                agent_state = arena.agent_states.get(agent_id)
                if agent_state is not None:
                    agent_state.pending_order_id = pending_id

            # 更新做市商挂单 ID
            for agent_id, (bid_ids, ask_ids) in result.mm_order_updates.items():
                agent_state = arena.agent_states.get(agent_id)
                if agent_state is not None:
                    agent_state.bid_order_ids = list(bid_ids)
                    agent_state.ask_order_ids = list(ask_ids)

            # 同步价格
            arena.smooth_mid_price = result.mid_price

            if arena.eliminating_agents:
                cleared_ids = [
                    agent_id
                    for agent_id in arena.eliminating_agents
                    if arena.agent_states.get(agent_id)
                    and arena.agent_states[agent_id].position_quantity == 0
                ]
                for agent_id in cleared_ids:
                    arena.eliminating_agents.discard(agent_id)

            arena_tick_trades[arena_id] = tick_trades

        return arena_tick_trades

    def run_tick_all_arenas(self) -> bool:
        """并行执行所有竞技场的一个 tick

        Returns:
            bool: 是否所有竞技场都应该继续运行
        """
        # 阶段1: 准备（串行）- 强平检查、计算市场状态
        arena_market_states: list[NormalizedMarketState] = []
        arena_active_agents: list[list[AgentAccountState]] = []
        arena_catfish_trades: list[list[Trade]] = [[] for _ in self.arena_states]

        for arena_idx, arena in enumerate(self.arena_states):
            arena.tick += 1

            # Tick 1: 只记录做市商初始挂单后的状态
            if arena.tick == 1:
                # 从 Worker 缓存获取实际价格用于价格统计
                actual_price = arena.smooth_mid_price
                if arena.arena_id in self._worker_depth_cache:
                    _, _, last_price, mid_price = self._worker_depth_cache[
                        arena.arena_id
                    ]
                    if last_price > 0:
                        actual_price = last_price
                    elif mid_price > 0:
                        actual_price = mid_price

                current_price = arena.smooth_mid_price
                arena.price_history.append(current_price)
                arena.tick_history_prices.append(current_price)
                arena.tick_history_volumes.append(0.0)
                arena.tick_history_amounts.append(0.0)
                # 使用实际价格更新 high/low 统计
                if actual_price > arena.episode_high_price:
                    arena.episode_high_price = actual_price
                if actual_price < arena.episode_low_price:
                    arena.episode_low_price = actual_price
                arena_market_states.append(self._compute_market_state_for_arena(arena))
                arena_active_agents.append([])
                continue

            # 获取当前价格
            current_price = (
                arena.smooth_mid_price
                if arena.smooth_mid_price > 0
                else arena.matching_engine._orderbook.last_price
            )

            # 强平检查
            self._handle_liquidations_for_arena(arena, current_price)

            # 鲶鱼行动
            arena_catfish_trades[arena_idx] = self._catfish_action_for_arena(arena)

            # 计算市场状态
            market_state = self._compute_market_state_for_arena(arena)
            arena_market_states.append(market_state)

            # 直接收集活跃的 Agent 状态（无需创建 adapter）
            active_states: list[AgentAccountState] = [
                state
                for state in arena.agent_states.values()
                if not state.is_liquidated
            ]

            # 随机打乱执行顺序
            random.shuffle(active_states)
            arena_active_agents.append(active_states)

        # 阶段2: 批量推理（并行）- 合并所有竞技场的推理（使用 direct 方法）
        all_decisions = self._batch_inference_all_arenas_direct(
            arena_market_states, arena_active_agents
        )

        # 阶段3: 执行
        all_continue = True

        # 使用 Worker 池执行（如果可用）
        if self._execute_worker_pool is not None:
            # 过滤掉 tick=1 的竞技场
            filtered_decisions: dict[
                int, list[tuple[AgentAccountState, ActionType, dict[str, Any]]]
            ] = {
                arena_idx: decisions
                for arena_idx, decisions in all_decisions.items()
                if self.arena_states[arena_idx].tick > 1
            }

            if filtered_decisions:
                # 使用 Worker 池执行
                results = self._execute_with_worker_pool(filtered_decisions)
                arena_tick_trades = self._process_worker_results(results)

                # 后处理：记录价格历史、检查提前结束条件
                for arena_idx in filtered_decisions.keys():
                    arena = self.arena_states[arena_idx]
                    tick_trades = arena_tick_trades.get(arena_idx, [])
                    catfish_trades = arena_catfish_trades[arena_idx]
                    if catfish_trades:
                        tick_trades = catfish_trades + tick_trades

                    # 从 Worker 缓存中获取实际成交价格用于价格统计
                    actual_price = arena.smooth_mid_price
                    if arena_idx in self._worker_depth_cache:
                        _, _, last_price, mid_price = self._worker_depth_cache[
                            arena_idx
                        ]
                        # 优先使用 last_price（实际成交价格），否则使用 mid_price
                        if last_price > 0:
                            actual_price = last_price
                        elif mid_price > 0:
                            actual_price = mid_price

                    # 记录价格历史（使用 smooth_mid_price 供 Agent 决策参考）
                    current_price = arena.smooth_mid_price
                    arena.price_history.append(current_price)
                    # 使用本 tick 成交价格更新 high/low 统计
                    self._update_episode_price_stats_from_trades(
                        arena,
                        tick_trades,
                        fallback_price=actual_price,
                    )

                    # 记录 tick 历史数据
                    arena.tick_history_prices.append(current_price)
                    volume, amount = self._aggregate_tick_trades(tick_trades)
                    arena.tick_history_volumes.append(volume)
                    arena.tick_history_amounts.append(amount)

                    # 鲶鱼强平检查（Worker 池模式不支持鲶鱼，但保持接口一致）
                    self._check_catfish_liquidation_for_arena(arena, current_price)

                    # 检查提前结束条件
                    if self._should_end_episode_early_for_arena(arena) is not None:
                        all_continue = False
                    if arena.catfish_liquidated:
                        all_continue = False

            return all_continue

        # 原来的串行执行逻辑
        for arena_idx, arena in enumerate(self.arena_states):
            if arena.tick == 1:
                continue

            decisions = all_decisions.get(arena_idx, [])
            tick_trades: list[Trade] = []

            # 缓存常用引用（减少属性查找开销）
            matching_engine = arena.matching_engine
            orderbook = matching_engine._orderbook
            process_order = matching_engine.process_order
            cancel_order = matching_engine.cancel_order
            order_map_get = orderbook.order_map.get
            agent_states_get = arena.agent_states.get
            recent_trades = arena.recent_trades
            recent_trades_append = recent_trades.append

            for state, action, params in decisions:
                if state.agent_type == AgentType.MARKET_MAKER:
                    # ========== 做市商执行（内联） ==========
                    # 1. 撤销所有旧挂单
                    for order_id in state.bid_order_ids:
                        cancel_order(order_id)
                    for order_id in state.ask_order_ids:
                        cancel_order(order_id)
                    state.bid_order_ids.clear()
                    state.ask_order_ids.clear()

                    # 2. 挂买单
                    for order_spec in params.get("bid_orders", []):
                        order_id = state.generate_order_id(arena.arena_id)
                        order = Order(
                            order_id=order_id,
                            agent_id=state.agent_id,
                            side=OrderSide.BUY,
                            order_type=OrderType.LIMIT,
                            price=order_spec["price"],
                            quantity=int(order_spec["quantity"]),
                        )
                        trades = process_order(order)
                        # 内联账户更新
                        for trade in trades:
                            is_buyer = trade.is_buyer_taker
                            fee = trade.buyer_fee if is_buyer else trade.seller_fee
                            state.on_trade(
                                trade.price,
                                trade.quantity,
                                is_buyer,
                                fee,
                                is_maker=False,
                            )
                            recent_trades_append(trade)
                            tick_trades.append(trade)
                            # 更新 maker 账户
                            maker_id = trade.seller_id if is_buyer else trade.buyer_id
                            maker_state = agent_states_get(maker_id)
                            if maker_state is not None:
                                maker_fee = (
                                    trade.seller_fee if is_buyer else trade.buyer_fee
                                )
                                maker_state.on_trade(
                                    trade.price,
                                    trade.quantity,
                                    not is_buyer,
                                    maker_fee,
                                    is_maker=True,
                                )
                        if order_map_get(order_id):
                            state.bid_order_ids.append(order_id)

                    # 3. 挂卖单
                    for order_spec in params.get("ask_orders", []):
                        order_id = state.generate_order_id(arena.arena_id)
                        order = Order(
                            order_id=order_id,
                            agent_id=state.agent_id,
                            side=OrderSide.SELL,
                            order_type=OrderType.LIMIT,
                            price=order_spec["price"],
                            quantity=int(order_spec["quantity"]),
                        )
                        trades = process_order(order)
                        for trade in trades:
                            is_buyer = trade.is_buyer_taker
                            fee = trade.buyer_fee if is_buyer else trade.seller_fee
                            state.on_trade(
                                trade.price,
                                trade.quantity,
                                is_buyer,
                                fee,
                                is_maker=False,
                            )
                            recent_trades_append(trade)
                            tick_trades.append(trade)
                            maker_id = trade.seller_id if is_buyer else trade.buyer_id
                            maker_state = agent_states_get(maker_id)
                            if maker_state is not None:
                                maker_fee = (
                                    trade.seller_fee if is_buyer else trade.buyer_fee
                                )
                                maker_state.on_trade(
                                    trade.price,
                                    trade.quantity,
                                    not is_buyer,
                                    maker_fee,
                                    is_maker=True,
                                )
                        if order_map_get(order_id):
                            state.ask_order_ids.append(order_id)
                else:
                    # ========== 非做市商执行（内联） ==========
                    if state.is_liquidated or action == ActionType.HOLD:
                        continue

                    trades = []

                    if action == ActionType.PLACE_BID or action == ActionType.PLACE_ASK:
                        # 先撤旧单
                        if state.pending_order_id is not None:
                            cancel_order(state.pending_order_id)
                            state.pending_order_id = None

                        order_id = state.generate_order_id(arena.arena_id)
                        side = (
                            OrderSide.BUY
                            if action == ActionType.PLACE_BID
                            else OrderSide.SELL
                        )
                        order = Order(
                            order_id=order_id,
                            agent_id=state.agent_id,
                            side=side,
                            order_type=OrderType.LIMIT,
                            price=params["price"],
                            quantity=int(params["quantity"]),
                        )
                        trades = process_order(order)
                        if order_map_get(order_id):
                            state.pending_order_id = order_id

                    elif action == ActionType.CANCEL:
                        if state.pending_order_id is not None:
                            cancel_order(state.pending_order_id)
                            state.pending_order_id = None

                    elif (
                        action == ActionType.MARKET_BUY
                        or action == ActionType.MARKET_SELL
                    ):
                        order_id = state.generate_order_id(arena.arena_id)
                        side = (
                            OrderSide.BUY
                            if action == ActionType.MARKET_BUY
                            else OrderSide.SELL
                        )
                        order = Order(
                            order_id=order_id,
                            agent_id=state.agent_id,
                            side=side,
                            order_type=OrderType.MARKET,
                            price=0.0,
                            quantity=int(params["quantity"]),
                        )
                        trades = process_order(order)

                    # 内联账户更新
                    for trade in trades:
                        is_buyer = trade.is_buyer_taker
                        fee = trade.buyer_fee if is_buyer else trade.seller_fee
                        state.on_trade(
                            trade.price, trade.quantity, is_buyer, fee, is_maker=False
                        )
                        recent_trades_append(trade)
                        tick_trades.append(trade)
                        maker_id = trade.seller_id if is_buyer else trade.buyer_id
                        maker_state = agent_states_get(maker_id)
                        if maker_state is not None:
                            maker_fee = (
                                trade.seller_fee if is_buyer else trade.buyer_fee
                            )
                            maker_state.on_trade(
                                trade.price,
                                trade.quantity,
                                not is_buyer,
                                maker_fee,
                                is_maker=True,
                            )

            catfish_trades = arena_catfish_trades[arena_idx]
            if catfish_trades:
                tick_trades = catfish_trades + tick_trades

            # 记录价格历史
            current_price = orderbook.last_price
            arena.price_history.append(current_price)
            self._update_episode_price_stats_from_trades(
                arena,
                tick_trades,
                fallback_price=current_price,
            )

            # 记录 tick 历史数据（deque maxlen=100 自动管理长度）
            arena.tick_history_prices.append(current_price)
            volume, amount = self._aggregate_tick_trades(tick_trades)
            arena.tick_history_volumes.append(volume)
            arena.tick_history_amounts.append(amount)

            # 鲶鱼强平检查
            self._check_catfish_liquidation_for_arena(arena, current_price)

            # 检查提前结束条件
            if self._should_end_episode_early_for_arena(arena) is not None:
                all_continue = False
            if arena.catfish_liquidated:
                all_continue = False

        return all_continue

    def _batch_inference_all_arenas(
        self,
        arena_market_states: list[NormalizedMarketState],
        arena_active_agents: list[list[AgentStateAdapter]],
    ) -> dict[int, list[tuple[AgentStateAdapter, ActionType, dict[str, Any]]]]:
        """批量推理所有竞技场的所有 Agent（使用 decide_multi_arena 合并推理）

        核心优化：将 N 个竞技场 x M 个 Agent 的推理合并成单次 OpenMP 并行操作

        Args:
            arena_market_states: 各竞技场的市场状态
            arena_active_agents: 各竞技场的活跃 Agent 适配器列表

        Returns:
            dict[arena_id, list[(adapter, action, params)]]
        """
        results: dict[
            int, list[tuple[AgentStateAdapter, ActionType, dict[str, Any]]]
        ] = {}

        # 初始化结果
        for arena_idx in range(len(arena_active_agents)):
            results[arena_idx] = []

        if self.network_caches is None:
            # 回退到串行推理
            for arena_idx, (market_state, adapters) in enumerate(
                zip(arena_market_states, arena_active_agents)
            ):
                results[arena_idx] = self._serial_inference_for_arena(
                    arena_idx, market_state, adapters
                )
            return results

        # 按 AgentType 分组收集数据（跨所有竞技场）
        # 格式: {agent_type: {arena_idx: [(adapter, agent, network_idx), ...]}}
        type_arena_data: dict[
            AgentType, dict[int, list[tuple[AgentStateAdapter, Any, int]]]
        ] = {}

        for arena_idx, adapters in enumerate(arena_active_agents):
            for adapter in adapters:
                agent_type = adapter.agent_type
                agent = self.agent_map.get(adapter.agent_id)
                if agent is None:
                    continue

                network_idx = self._get_network_index(agent_type, adapter.agent_id)
                if network_idx < 0:
                    continue

                if agent_type not in type_arena_data:
                    type_arena_data[agent_type] = {}
                if arena_idx not in type_arena_data[agent_type]:
                    type_arena_data[agent_type][arena_idx] = []

                type_arena_data[agent_type][arena_idx].append(
                    (adapter, agent, network_idx)
                )

        # 对每种类型使用 decide_multi_arena 进行批量推理
        for agent_type, arena_data in type_arena_data.items():
            cache = self.network_caches.get(agent_type)
            if cache is None or not cache.is_valid():
                continue

            # 准备 decide_multi_arena 的参数
            # 按 arena_idx 排序以确保结果映射正确
            sorted_arena_indices = sorted(arena_data.keys())

            agents_per_arena: list[list[Any]] = []
            market_states: list[NormalizedMarketState] = []
            network_indices_per_arena: list[list[int]] = []
            adapter_mapping: list[list[AgentStateAdapter]] = (
                []
            )  # 记录每个竞技场的 adapter 顺序

            for arena_idx in sorted_arena_indices:
                arena_entries = arena_data[arena_idx]

                arena_agents: list[Any] = []
                arena_network_indices: list[int] = []
                arena_adapters: list[AgentStateAdapter] = []

                for adapter, agent, network_idx in arena_entries:
                    arena_agents.append(agent)
                    arena_network_indices.append(network_idx)
                    arena_adapters.append(adapter)

                agents_per_arena.append(arena_agents)
                market_states.append(arena_market_states[arena_idx])
                network_indices_per_arena.append(arena_network_indices)
                adapter_mapping.append(arena_adapters)

            # 调用 decide_multi_arena 进行批量推理
            try:
                raw_results = cache.decide_multi_arena(
                    agents_per_arena,
                    market_states,
                    network_indices_per_arena,
                )
            except Exception as e:
                self.logger.warning(f"批量决策失败 {agent_type.value}: {e}")
                continue

            # 解析结果并填充到 results
            is_market_maker = agent_type == AgentType.MARKET_MAKER

            for result_idx, arena_idx in enumerate(sorted_arena_indices):
                arena_results = raw_results.get(result_idx, [])
                arena_adapters = adapter_mapping[result_idx]
                market_state = arena_market_states[arena_idx]
                mid_price = market_state.mid_price
                tick_size = (
                    market_state.tick_size if market_state.tick_size > 0 else 0.01
                )

                for i, raw_result in enumerate(arena_results):
                    if i >= len(arena_adapters):
                        break

                    adapter = arena_adapters[i]
                    agent = self.agent_map.get(adapter.agent_id)
                    if agent is None:
                        continue

                    try:
                        if is_market_maker:
                            nn_output, _, _ = raw_result
                            action, params = self._parse_market_maker_output(
                                agent, nn_output, mid_price, tick_size
                            )
                        else:
                            # 注意：Cython 返回的 quantity 实际上是 quantity_ratio（0-1浮点数）
                            action_type_int, side_int, price, quantity_ratio = (
                                raw_result
                            )
                            action, params = self._convert_retail_result(
                                agent,
                                action_type_int,
                                side_int,
                                price,
                                quantity_ratio,
                                mid_price,
                            )

                        results[arena_idx].append((adapter, action, params))
                    except Exception:
                        pass

        return results

    def _batch_inference_all_arenas_direct(
        self,
        arena_market_states: list[NormalizedMarketState],
        arena_active_states: list[list[AgentAccountState]],
    ) -> dict[int, list[tuple[AgentAccountState, ActionType, dict[str, Any]]]]:
        """批量推理所有竞技场的所有 Agent（直接使用 AgentAccountState，无需 adapter）

        核心优化：将 N 个竞技场 x M 个 Agent 的推理合并成单次 OpenMP 并行操作。
        直接使用 AgentAccountState 对象，避免创建 adapter 的开销。

        对于非做市商类型，使用 return_array=True 获取 NumPy 数组格式，
        并缓存到 _last_inference_arrays 供 _execute_with_worker_pool 使用。

        Args:
            arena_market_states: 各竞技场的市场状态
            arena_active_states: 各竞技场的活跃 Agent 状态列表

        Returns:
            dict[arena_id, list[(state, action, params)]]
        """
        results: dict[
            int, list[tuple[AgentAccountState, ActionType, dict[str, Any]]]
        ] = {}

        # 初始化结果
        for arena_idx in range(len(arena_active_states)):
            results[arena_idx] = []

        # 清空上次推理的数组缓存
        self._last_inference_arrays.clear()

        if self.network_caches is None:
            # 回退到串行推理
            return results

        # 按 AgentType 分组收集数据（跨所有竞技场）
        # 格式: {agent_type: {arena_idx: [(state, network_idx), ...]}}
        type_arena_data: dict[
            AgentType, dict[int, list[tuple[AgentAccountState, int]]]
        ] = {}

        for arena_idx, states in enumerate(arena_active_states):
            for state in states:
                agent_type = state.agent_type

                network_idx = self._get_network_index(agent_type, state.agent_id)
                if network_idx < 0:
                    continue

                if agent_type not in type_arena_data:
                    type_arena_data[agent_type] = {}
                if arena_idx not in type_arena_data[agent_type]:
                    type_arena_data[agent_type][arena_idx] = []

                type_arena_data[agent_type][arena_idx].append((state, network_idx))

        # 对每种类型使用 decide_multi_arena_direct 进行批量推理
        for agent_type, arena_data in type_arena_data.items():
            cache = self.network_caches.get(agent_type)
            if cache is None or not cache.is_valid():
                continue

            # 准备 decide_multi_arena_direct 的参数
            sorted_arena_indices = sorted(arena_data.keys())

            states_per_arena: list[list[AgentAccountState]] = []
            market_states: list[NormalizedMarketState] = []
            network_indices_per_arena: list[list[int]] = []
            state_mapping: list[list[AgentAccountState]] = []

            for arena_idx in sorted_arena_indices:
                arena_entries = arena_data[arena_idx]

                arena_states_list: list[AgentAccountState] = []
                arena_network_indices: list[int] = []

                for state, network_idx in arena_entries:
                    arena_states_list.append(state)
                    arena_network_indices.append(network_idx)

                states_per_arena.append(arena_states_list)
                market_states.append(arena_market_states[arena_idx])
                network_indices_per_arena.append(arena_network_indices)
                state_mapping.append(arena_states_list)

            # 判断是否为做市商类型
            is_market_maker = agent_type == AgentType.MARKET_MAKER

            # 调用 decide_multi_arena_direct 进行批量推理
            # 非做市商类型使用 return_array=True 以获取 NumPy 数组格式
            try:
                raw_results = cache.decide_multi_arena_direct(
                    states_per_arena,
                    market_states,
                    network_indices_per_arena,
                    return_array=not is_market_maker,
                )
            except Exception as e:
                self.logger.warning(f"批量决策失败 {agent_type.value}: {e}")
                continue

            # 初始化该类型的数组缓存
            if not is_market_maker:
                self._last_inference_arrays[agent_type] = {}

            # 解析结果并填充到 results
            for result_idx, arena_idx in enumerate(sorted_arena_indices):
                arena_results = raw_results.get(result_idx, None)
                arena_states_list = state_mapping[result_idx]
                market_state = arena_market_states[arena_idx]
                mid_price = market_state.mid_price
                tick_size = (
                    market_state.tick_size if market_state.tick_size > 0 else 0.01
                )

                if is_market_maker:
                    # 做市商：raw_results 是 list 格式
                    if arena_results is None:
                        arena_results = []
                    for i, raw_result in enumerate(arena_results):
                        if i >= len(arena_states_list):
                            break

                        state = arena_states_list[i]
                        try:
                            nn_output, _, _ = raw_result
                            action, params = self._parse_market_maker_output(
                                state, nn_output, mid_price, tick_size
                            )
                            results[arena_idx].append((state, action, params))
                        except Exception:
                            pass
                else:
                    # 非做市商：raw_results 是 NumPy 数组 shape=(num_agents, 4)
                    # 列顺序: [action_type, side, price, quantity]
                    if arena_results is None or len(arena_results) == 0:
                        continue

                    decisions_array: np.ndarray = arena_results
                    num_agents = min(len(decisions_array), len(arena_states_list))

                    # 构建 agent_ids 数组（用于后续构建带 agent_id 的完整数组）
                    agent_ids = np.array(
                        [arena_states_list[i].agent_id for i in range(num_agents)],
                        dtype=np.float64,
                    )

                    # 缓存数组结果供 _execute_with_worker_pool 使用
                    self._last_inference_arrays[agent_type][arena_idx] = (
                        agent_ids,
                        decisions_array[:num_agents].copy(),
                    )

                    # 同时填充 list 格式结果（用于兼容性）
                    for i in range(num_agents):
                        state = arena_states_list[i]
                        try:
                            action_type_int = int(decisions_array[i, 0])
                            side_int = int(decisions_array[i, 1])
                            price = float(decisions_array[i, 2])
                            # 注意：Cython 返回的是 quantity_ratio（0-1浮点数），不是实际数量
                            quantity_ratio = float(decisions_array[i, 3])

                            action, params = self._convert_retail_result(
                                state,
                                action_type_int,
                                side_int,
                                price,
                                quantity_ratio,
                                mid_price,
                            )
                            results[arena_idx].append((state, action, params))
                        except Exception:
                            pass

        return results

    def _serial_inference_for_arena(
        self,
        arena_idx: int,
        market_state: NormalizedMarketState,
        adapters: list[AgentStateAdapter],
    ) -> list[tuple[AgentStateAdapter, ActionType, dict[str, Any]]]:
        """串行推理（回退方案）

        使用 AgentAccountState 进行推理，确保账户独立性。
        """
        results: list[tuple[AgentStateAdapter, ActionType, dict[str, Any]]] = []
        arena = self.arena_states[arena_idx]
        orderbook = arena.matching_engine._orderbook
        mid_price = market_state.mid_price
        tick_size = market_state.tick_size if market_state.tick_size > 0 else 0.01

        for adapter in adapters:
            agent = self._get_agent_by_id(adapter.agent_id)
            agent_state = arena.agent_states.get(adapter.agent_id)
            if agent is None or agent_state is None:
                continue

            try:
                # 临时同步 agent_state 到 agent.account，以便 observe() 使用正确的状态
                self._sync_state_to_agent(agent, agent_state)

                # 使用 agent.observe() 获取神经网络输入
                inputs = agent.observe(market_state, orderbook)
                # 使用 agent.brain.forward() 获取输出
                outputs = agent.brain.forward(inputs)

                # 使用独立的解析方法（基于 agent_state 计算订单数量）
                if agent_state.agent_type == AgentType.MARKET_MAKER:
                    action, params = self._parse_market_maker_output(
                        agent_state, np.array(outputs), mid_price, tick_size
                    )
                else:
                    # 解析非做市商输出
                    action, params = self._parse_non_mm_output(
                        agent_state, outputs, mid_price, tick_size
                    )

                results.append((adapter, action, params))
            except Exception:
                pass

        return results

    def _sync_state_to_agent(self, agent: Any, state: AgentAccountState) -> None:
        """临时同步 AgentAccountState 到 Agent.account

        用于回退推理路径，确保 observe() 使用正确的竞技场状态。
        """
        account = agent.account
        account.balance = state.balance
        account.position.quantity = state.position_quantity
        account.position.avg_price = state.position_avg_price
        account.pending_order_id = state.pending_order_id

    def _parse_non_mm_output(
        self,
        agent_state: AgentAccountState,
        outputs: Any,
        mid_price: float,
        tick_size: float,
    ) -> tuple[ActionType, dict[str, Any]]:
        """解析非做市商的神经网络输出

        输出结构（8个值）：
        - [0-5]: 动作得分
        - [6]: 价格偏移 (-1 到 1)
        - [7]: 数量比例 (-1 到 1)
        """
        from src.bio.agents._cython.fast_decide import (
            fast_argmax,
            fast_round_price,
            fast_clip,
        )

        # 解析动作类型
        action_type_int = fast_argmax(outputs, 0, 6)

        # 解析价格和数量
        price_offset_raw = outputs[6] if len(outputs) > 6 else 0.0
        quantity_ratio_raw = outputs[7] if len(outputs) > 7 else 0.5

        # 价格偏移映射到 [-100, 100] ticks
        price_offset = fast_clip(price_offset_raw, -1.0, 1.0) * 100 * tick_size
        price = fast_round_price(mid_price + price_offset, tick_size)

        # 数量比例映射到 [0.1, 1.0]
        quantity_ratio = 0.1 + (fast_clip(quantity_ratio_raw, -1.0, 1.0) + 1) * 0.45

        # 方向：买单=1，卖单=2
        side_int = 1 if action_type_int in (1, 4) else 2

        return self._convert_retail_result(
            agent_state, action_type_int, side_int, price, quantity_ratio, mid_price
        )

    def _get_agent_by_id(self, agent_id: int) -> Any:
        """根据 ID 获取 Agent 对象（O(1) 查找）"""
        return self.agent_map.get(agent_id)

    def _parse_market_maker_output(
        self,
        agent_state: AgentAccountState,
        output: np.ndarray,
        mid_price: float,
        tick_size: float,
    ) -> tuple[ActionType, dict[str, Any]]:
        """解析做市商的神经网络输出

        神经网络输出结构（共 41 个值）：
        - 输出[0-9]: 买单1-10的价格偏移（-1到1，映射到1-100 ticks）
        - 输出[10-19]: 买单1-10的数量权重（-1到1，映射到0-1）
        - 输出[20-29]: 卖单1-10的价格偏移（-1到1，映射到1-100 ticks）
        - 输出[30-39]: 卖单1-10的数量权重（-1到1，映射到0-1）
        - 输出[40]: 总下单比例基准（-1到1，映射到0.01-1）
        """
        from src.bio.agents.base import fast_clip, fast_round_price

        # 正确的索引：与 MarketMakerAgent.decide() 保持一致
        bid_price_offsets = output[0:10]
        bid_qty_weights = output[10:20]
        ask_price_offsets = output[20:30]
        ask_qty_weights = output[30:40]
        total_ratio_raw = output[40] if len(output) > 40 else 0.0

        bid_raw_ratios = np.maximum(
            0.0, (np.clip(bid_qty_weights, -1.0, 1.0) + 1.0) * 0.5
        )
        ask_raw_ratios = np.maximum(
            0.0, (np.clip(ask_qty_weights, -1.0, 1.0) + 1.0) * 0.5
        )

        total_raw_ratio = float(bid_raw_ratios.sum() + ask_raw_ratios.sum())
        if total_raw_ratio > 0:
            bid_ratios = bid_raw_ratios / total_raw_ratio
            ask_ratios = ask_raw_ratios / total_raw_ratio
        else:
            bid_ratios = np.zeros(10, dtype=np.float64)
            ask_ratios = np.zeros(10, dtype=np.float64)

        skew_factor = calculate_skew_factor_from_state(agent_state, mid_price)
        bid_ratios, ask_ratios = self._apply_position_skew(
            bid_ratios, ask_ratios, skew_factor
        )

        # 总下单比例：映射到 [0.01, 1]，与 MarketMakerAgent.decide() 一致
        total_ratio = 0.01 + (fast_clip(total_ratio_raw, -1.0, 1.0) + 1) * 0.5 * 0.99
        bid_ratios = bid_ratios * total_ratio
        ask_ratios = ask_ratios * total_ratio

        bid_orders: list[dict[str, float]] = []
        ask_orders: list[dict[str, float]] = []

        # 循环10次，处理10个买单和10个卖单
        for i in range(10):
            offset = fast_clip(bid_price_offsets[i], -1.0, 1.0)
            ticks = 1 + (offset + 1) * 49.5
            price = mid_price - ticks * tick_size
            price = fast_round_price(price, tick_size)

            ratio = float(bid_ratios[i])
            if ratio > 0:
                qty = calculate_order_quantity_from_state(
                    agent_state, price, ratio, is_buy=True, ref_price=mid_price
                )
                if qty > 0:
                    bid_orders.append({"price": price, "quantity": float(qty)})

            offset = fast_clip(ask_price_offsets[i], -1.0, 1.0)
            ticks = 1 + (offset + 1) * 49.5
            price = mid_price + ticks * tick_size
            price = fast_round_price(price, tick_size)

            ratio = float(ask_ratios[i])
            if ratio > 0:
                qty = calculate_order_quantity_from_state(
                    agent_state, price, ratio, is_buy=False, ref_price=mid_price
                )
                if qty > 0:
                    ask_orders.append({"price": price, "quantity": float(qty)})

        return ActionType.HOLD, {"bid_orders": bid_orders, "ask_orders": ask_orders}

    @staticmethod
    def _apply_position_skew(
        bid_raw_ratios: np.ndarray,
        ask_raw_ratios: np.ndarray,
        skew_factor: float,
        min_side_weight: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """应用仓位倾斜到买卖权重（与 MarketMakerAgent 保持一致）"""
        bid_multiplier = 1.0 + skew_factor
        ask_multiplier = 1.0 - skew_factor

        bid_adjusted = bid_raw_ratios * bid_multiplier
        ask_adjusted = ask_raw_ratios * ask_multiplier

        total_bid = float(bid_adjusted.sum())
        total_ask = float(ask_adjusted.sum())
        total = total_bid + total_ask

        if total <= 0:
            return np.full(10, 0.1, dtype=np.float64), np.full(
                10, 0.1, dtype=np.float64
            )

        bid_side_ratio = total_bid / total
        ask_side_ratio = total_ask / total

        if bid_side_ratio < min_side_weight:
            target_bid_total = min_side_weight
            target_ask_total = 1.0 - min_side_weight
        elif ask_side_ratio < min_side_weight:
            target_ask_total = min_side_weight
            target_bid_total = 1.0 - min_side_weight
        else:
            target_bid_total = bid_side_ratio
            target_ask_total = ask_side_ratio

        if total_bid > 0:
            bid_ratios = bid_adjusted / total_bid * target_bid_total
        else:
            bid_ratios = np.full(10, target_bid_total / 10, dtype=np.float64)

        if total_ask > 0:
            ask_ratios = ask_adjusted / total_ask * target_ask_total
        else:
            ask_ratios = np.full(10, target_ask_total / 10, dtype=np.float64)

        return bid_ratios, ask_ratios

    def _convert_retail_result(
        self,
        agent_state: AgentAccountState,
        action_type_int: int,
        side_int: int,
        price: float,
        quantity_ratio: float,
        mid_price: float,
    ) -> tuple[ActionType, dict[str, Any]]:
        """将 Cython 返回的散户/高级散户/庄家结果转换为 ActionType 和 params

        注意：Cython 返回的 quantity 实际上是 quantity_ratio（0-1浮点数比例），
        需要调用 calculate_order_quantity_from_state 转换为实际订单数量。
        """
        params: dict[str, Any] = {}

        if action_type_int == 0:
            action = ActionType.HOLD
        elif action_type_int == 1:
            action = ActionType.PLACE_BID
            params["price"] = price
            actual_qty = calculate_order_quantity_from_state(
                agent_state, mid_price, quantity_ratio, is_buy=True
            )
            params["quantity"] = actual_qty
        elif action_type_int == 2:
            action = ActionType.PLACE_ASK
            params["price"] = price
            actual_qty = calculate_order_quantity_from_state(
                agent_state, mid_price, quantity_ratio, is_buy=False
            )
            params["quantity"] = actual_qty
        elif action_type_int == 3:
            action = ActionType.CANCEL
        elif action_type_int == 4:
            action = ActionType.MARKET_BUY
            actual_qty = calculate_order_quantity_from_state(
                agent_state, mid_price, quantity_ratio, is_buy=True
            )
            params["quantity"] = actual_qty
        elif action_type_int == 5:
            action = ActionType.MARKET_SELL
            position_qty = agent_state.position_quantity
            if position_qty > 0:
                # 有多仓时卖出持仓的比例
                sell_qty = max(1, int(position_qty * quantity_ratio))
                params["quantity"] = min(sell_qty, int(position_qty))
            else:
                # 空仓或空头持仓，开空仓
                actual_qty = calculate_order_quantity_from_state(
                    agent_state, mid_price, quantity_ratio, is_buy=False
                )
                params["quantity"] = actual_qty
        else:
            action = ActionType.HOLD

        return action, params

    def _handle_liquidations_for_arena(
        self, arena: ArenaState, current_price: float
    ) -> None:
        """处理单个竞技场的强平"""
        # 向量化检查强平条件
        agents_to_liquidate: list[AgentAccountState] = []
        for agent_state in arena.agent_states.values():
            if agent_state.is_liquidated:
                continue
            if agent_state.check_liquidation(current_price):
                agents_to_liquidate.append(agent_state)

        if not agents_to_liquidate:
            return

        # Worker 池模式：仅标记强平，实际撤单/平仓交给 Worker 执行
        if self._execute_worker_pool is not None:
            for agent_state in agents_to_liquidate:
                arena.eliminating_agents.add(agent_state.agent_id)
                agent_state.is_liquidated = True
                arena.mark_agent_liquidated(
                    agent_state.agent_id,
                    agent_state.agent_type,
                )
            return

        # 阶段1: 撤销挂单
        for agent_state in agents_to_liquidate:
            self._cancel_agent_orders_in_arena(arena, agent_state)

        # 阶段2: 市价平仓
        agents_need_adl: list[tuple[AgentAccountState, int, bool]] = []
        for agent_state in agents_to_liquidate:
            remaining_qty, is_long = self._execute_liquidation_in_arena(
                arena, agent_state, current_price
            )
            if remaining_qty > 0:
                agents_need_adl.append((agent_state, remaining_qty, is_long))

            agent_state.is_liquidated = True
            arena.mark_agent_liquidated(agent_state.agent_id, agent_state.agent_type)

            if agent_state.balance < 0:
                agent_state.balance = 0.0

        # 阶段3: ADL
        if agents_need_adl:
            latest_price = arena.matching_engine._orderbook.last_price
            self._execute_adl_in_arena(arena, agents_need_adl, latest_price)

    def _cancel_agent_orders_in_arena(
        self, arena: ArenaState, agent_state: AgentAccountState
    ) -> None:
        """撤销 Agent 在竞技场中的挂单"""
        if agent_state.agent_type == AgentType.MARKET_MAKER:
            for order_id in agent_state.bid_order_ids + agent_state.ask_order_ids:
                arena.matching_engine.cancel_order(order_id)
            agent_state.bid_order_ids.clear()
            agent_state.ask_order_ids.clear()
        else:
            if agent_state.pending_order_id is not None:
                arena.matching_engine.cancel_order(agent_state.pending_order_id)
                agent_state.pending_order_id = None

    def _update_trade_accounts(
        self,
        arena: ArenaState,
        agent_state: AgentAccountState,
        trades: list[Trade],
    ) -> None:
        """更新成交相关的账户状态"""
        for trade in trades:
            is_buyer = trade.is_buyer_taker
            fee = trade.buyer_fee if is_buyer else trade.seller_fee
            agent_state.on_trade(
                trade.price, trade.quantity, is_buyer, fee, is_maker=False
            )
            arena.recent_trades.append(trade)

            # 更新 maker 账户
            maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
            maker_state = arena.agent_states.get(maker_id)
            if maker_state is not None:
                maker_is_buyer = not trade.is_buyer_taker
                maker_fee = (
                    trade.seller_fee if trade.is_buyer_taker else trade.buyer_fee
                )
                maker_state.on_trade(
                    trade.price,
                    trade.quantity,
                    maker_is_buyer,
                    maker_fee,
                    is_maker=True,
                )

    def _execute_mm_action_in_arena(
        self,
        arena: ArenaState,
        agent_state: AgentAccountState,
        params: dict[str, Any],
    ) -> list[Trade]:
        """在竞技场中执行做市商动作（不依赖 Agent 对象）"""
        matching_engine = arena.matching_engine
        all_trades: list[Trade] = []

        # 1. 撤销所有旧挂单（分别遍历，避免列表合并创建新列表）
        cancel_order = matching_engine.cancel_order  # 缓存方法引用
        for order_id in agent_state.bid_order_ids:
            cancel_order(order_id)
        for order_id in agent_state.ask_order_ids:
            cancel_order(order_id)
        agent_state.bid_order_ids.clear()
        agent_state.ask_order_ids.clear()

        # 2. 挂买单
        for order_spec in params.get("bid_orders", []):
            order_id = agent_state.generate_order_id(arena.arena_id)
            order = Order(
                order_id=order_id,
                agent_id=agent_state.agent_id,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=order_spec["price"],
                quantity=int(order_spec["quantity"]),
            )
            trades = matching_engine.process_order(order)
            self._update_trade_accounts(arena, agent_state, trades)
            all_trades.extend(trades)
            # 检查订单是否在订单簿中（未完全成交）
            if matching_engine._orderbook.order_map.get(order_id):
                agent_state.bid_order_ids.append(order_id)

        # 3. 挂卖单
        for order_spec in params.get("ask_orders", []):
            order_id = agent_state.generate_order_id(arena.arena_id)
            order = Order(
                order_id=order_id,
                agent_id=agent_state.agent_id,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=order_spec["price"],
                quantity=int(order_spec["quantity"]),
            )
            trades = matching_engine.process_order(order)
            self._update_trade_accounts(arena, agent_state, trades)
            all_trades.extend(trades)
            if matching_engine._orderbook.order_map.get(order_id):
                agent_state.ask_order_ids.append(order_id)

        return all_trades

    def _execute_non_mm_action_in_arena(
        self,
        arena: ArenaState,
        agent_state: AgentAccountState,
        action: ActionType,
        params: dict[str, Any],
    ) -> list[Trade]:
        """在竞技场中执行非做市商动作（不依赖 Agent 对象）"""
        if agent_state.is_liquidated:
            return []

        matching_engine = arena.matching_engine
        trades: list[Trade] = []

        if action == ActionType.HOLD:
            return []

        elif action == ActionType.PLACE_BID:
            if agent_state.pending_order_id is not None:
                matching_engine.cancel_order(agent_state.pending_order_id)
                agent_state.pending_order_id = None

            order_id = agent_state.generate_order_id(arena.arena_id)
            order = Order(
                order_id=order_id,
                agent_id=agent_state.agent_id,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=params["price"],
                quantity=int(params["quantity"]),
            )
            trades = matching_engine.process_order(order)
            # 如果订单未完全成交，记录挂单ID
            if matching_engine._orderbook.order_map.get(order_id):
                agent_state.pending_order_id = order_id

        elif action == ActionType.PLACE_ASK:
            if agent_state.pending_order_id is not None:
                matching_engine.cancel_order(agent_state.pending_order_id)
                agent_state.pending_order_id = None

            order_id = agent_state.generate_order_id(arena.arena_id)
            order = Order(
                order_id=order_id,
                agent_id=agent_state.agent_id,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=params["price"],
                quantity=int(params["quantity"]),
            )
            trades = matching_engine.process_order(order)
            if matching_engine._orderbook.order_map.get(order_id):
                agent_state.pending_order_id = order_id

        elif action == ActionType.CANCEL:
            if agent_state.pending_order_id is not None:
                matching_engine.cancel_order(agent_state.pending_order_id)
                agent_state.pending_order_id = None

        elif action == ActionType.MARKET_BUY:
            order_id = agent_state.generate_order_id(arena.arena_id)
            order = Order(
                order_id=order_id,
                agent_id=agent_state.agent_id,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                price=0.0,
                quantity=int(params["quantity"]),
            )
            trades = matching_engine.process_order(order)

        elif action == ActionType.MARKET_SELL:
            order_id = agent_state.generate_order_id(arena.arena_id)
            order = Order(
                order_id=order_id,
                agent_id=agent_state.agent_id,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                price=0.0,
                quantity=int(params["quantity"]),
            )
            trades = matching_engine.process_order(order)

        self._update_trade_accounts(arena, agent_state, trades)
        return trades

    def _execute_liquidation_in_arena(
        self,
        arena: ArenaState,
        agent_state: AgentAccountState,
        current_price: float,
    ) -> tuple[int, bool]:
        """在竞技场中执行强平市价单"""
        position_qty = agent_state.position_quantity
        if position_qty == 0:
            return 0, True

        is_long = position_qty > 0
        target_qty = abs(position_qty)

        side = OrderSide.SELL if is_long else OrderSide.BUY
        order_id = agent_state.generate_order_id(arena.arena_id)
        order = Order(
            order_id=order_id,
            agent_id=agent_state.agent_id,
            side=side,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=target_qty,
        )

        trades = arena.matching_engine.process_order(order)

        for trade in trades:
            is_buyer = trade.is_buyer_taker
            # taker fee: 买方是 taker 时用 buyer_fee，否则用 seller_fee
            fee = trade.buyer_fee if trade.is_buyer_taker else trade.seller_fee
            agent_state.on_trade(trade.price, trade.quantity, is_buyer, fee, False)
            arena.recent_trades.append(trade)

            maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
            maker_state = arena.agent_states.get(maker_id)
            if maker_state is not None:
                maker_is_buyer = not trade.is_buyer_taker
                # maker fee: 买方是 taker 时 maker 是卖方用 seller_fee，否则用 buyer_fee
                maker_fee = (
                    trade.seller_fee if trade.is_buyer_taker else trade.buyer_fee
                )
                maker_state.on_trade(
                    trade.price, trade.quantity, maker_is_buyer, maker_fee, True
                )

        remaining_qty = abs(agent_state.position_quantity)
        return remaining_qty, is_long

    def _execute_adl_in_arena(
        self,
        arena: ArenaState,
        agents_need_adl: list[tuple[AgentAccountState, int, bool]],
        current_price: float,
    ) -> None:
        """在竞技场中执行 ADL"""
        adl_price = arena.adl_manager.get_adl_price(current_price)

        # 计算 ADL 候选
        long_candidates: list[ADLCandidate] = []
        short_candidates: list[ADLCandidate] = []

        for agent_state in arena.agent_states.values():
            if agent_state.is_liquidated:
                continue
            if agent_state.position_quantity == 0:
                continue

            equity = agent_state.get_equity(current_price)
            pnl_percent = (
                equity - agent_state.initial_balance
            ) / agent_state.initial_balance

            if pnl_percent <= 0:
                continue

            position_value = abs(agent_state.position_quantity) * current_price
            effective_leverage = position_value / equity if equity > 0 else 0.0
            adl_score = pnl_percent * effective_leverage

            # 创建一个模拟的 participant
            class MockParticipant:
                def __init__(self, state: AgentAccountState):
                    self._state = state
                    self.account = self

                @property
                def position(self):
                    return self

                @property
                def quantity(self):
                    return self._state.position_quantity

            candidate = ADLCandidate(
                participant=MockParticipant(agent_state),  # type: ignore
                position_qty=agent_state.position_quantity,
                pnl_percent=pnl_percent,
                effective_leverage=effective_leverage,
                adl_score=adl_score,
            )

            if agent_state.position_quantity > 0:
                long_candidates.append(candidate)
            else:
                short_candidates.append(candidate)

        long_candidates.sort(key=lambda c: c.adl_score, reverse=True)
        short_candidates.sort(key=lambda c: c.adl_score, reverse=True)

        # 执行 ADL
        for agent_state, remaining_qty, is_long in agents_need_adl:
            candidates = short_candidates if is_long else long_candidates

            for candidate in candidates:
                if remaining_qty <= 0:
                    break

                candidate_available_qty = abs(candidate.position_qty)
                trade_qty = min(candidate_available_qty, remaining_qty)

                if trade_qty <= 0:
                    continue

                # 更新被强平方
                if is_long:
                    agent_state.position_quantity -= trade_qty
                else:
                    agent_state.position_quantity += trade_qty
                agent_state.balance += (
                    (adl_price - agent_state.position_avg_price)
                    * trade_qty
                    * (1 if is_long else -1)
                )

                # 更新对手方
                if candidate.position_qty > 0:
                    candidate.position_qty -= trade_qty
                else:
                    candidate.position_qty += trade_qty

                remaining_qty -= trade_qty

            # 兜底处理
            if agent_state.position_quantity != 0:
                agent_state.position_quantity = 0
                agent_state.position_avg_price = 0.0

    def _catfish_action_for_arena(self, arena: ArenaState) -> list[Trade]:
        """鲶鱼在竞技场中的行动

        实现三种鲶鱼的决策和执行逻辑：
        - TREND_CREATOR: 保持当前方向持续操作
        - MEAN_REVERSION: 价格偏离 EMA 时反向操作
        - RANDOM: 随机概率触发，方向也随机
        """
        if not arena.catfish_states:
            return []

        orderbook = arena.matching_engine._orderbook
        matching_engine = arena.matching_engine
        tick = arena.tick
        price_history = arena.price_history

        catfish_trades: list[Trade] = []
        for catfish_state in arena.catfish_states.values():
            if catfish_state.is_liquidated:
                continue

            # 调用鲶鱼决策方法
            should_act, direction = catfish_state.decide(tick, price_history)

            if not should_act or direction == 0:
                continue

            # 计算下单数量（吃掉前3档）
            quantity = self._calculate_catfish_quantity(orderbook, direction)

            if quantity <= 0:
                continue

            # 创建市价单
            side = OrderSide.BUY if direction > 0 else OrderSide.SELL
            order_id = catfish_state.generate_order_id(arena.arena_id)
            order = Order(
                order_id=order_id,
                agent_id=catfish_state.catfish_id,
                side=side,
                order_type=OrderType.MARKET,
                price=0.0,
                quantity=quantity,
            )

            # 确保鲶鱼已注册（费率为0）
            matching_engine.register_agent(catfish_state.catfish_id, 0.0, 0.0)

            # 执行市价单
            trades = matching_engine.match_market_order(order)

            # 更新鲶鱼账户和 maker 账户
            for trade in trades:
                catfish_trades.append(trade)
                is_buyer = trade.is_buyer_taker
                catfish_state.on_trade(trade.price, trade.quantity, is_buyer)
                arena.recent_trades.append(trade)

                # 更新 maker 的账户状态
                maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
                maker_state = arena.agent_states.get(maker_id)
                if maker_state is not None:
                    maker_is_buyer = not trade.is_buyer_taker
                    maker_fee = (
                        trade.seller_fee if trade.is_buyer_taker else trade.buyer_fee
                    )
                    maker_state.on_trade(
                        trade.price, trade.quantity, maker_is_buyer, maker_fee, True
                    )

        return catfish_trades

    def _calculate_catfish_quantity(self, orderbook: Any, direction: int) -> int:
        """计算鲶鱼下单数量（吃掉前3档）

        Args:
            orderbook: 订单簿
            direction: 方向（1=买，-1=卖）

        Returns:
            下单数量
        """
        target_ticks = 3

        # 获取盘口深度
        depth = orderbook.get_depth(levels=target_ticks)

        if direction > 0:  # 买入，吃卖盘
            levels = depth["asks"]
        else:  # 卖出，吃买盘
            levels = depth["bids"]

        if len(levels) < target_ticks:
            return 0

        # 累加前 target_ticks 档的数量
        total_qty = 0
        for i in range(min(target_ticks, len(levels))):
            price, qty = levels[i]
            total_qty += int(qty)

        return total_qty

    def _check_catfish_liquidation_for_arena(
        self, arena: ArenaState, current_price: float
    ) -> None:
        """检查鲶鱼强平"""
        for catfish_state in arena.catfish_states.values():
            if catfish_state.is_liquidated:
                continue

            if catfish_state.check_liquidation(current_price):
                catfish_state.is_liquidated = True
                arena.catfish_liquidated = True
                break

    def _should_end_episode_early_for_arena(
        self, arena: ArenaState
    ) -> tuple[str, AgentType | None] | None:
        """检查单个竞技场是否应该提前结束"""
        # 检查种群存活数量
        pop_total_counts: dict[AgentType, int] = {}
        for agent_type, population in self.populations.items():
            pop_total_counts[agent_type] = len(population.agents)

        for agent_type, total in pop_total_counts.items():
            if total > 0:
                liquidated = arena.pop_liquidated_counts.get(agent_type, 0)
                alive = total - liquidated
                if alive < total / 4:
                    return ("population_depleted", agent_type)

        # 检查订单簿单边挂单
        if arena.arena_id in self._worker_depth_cache:
            bid_depth, ask_depth, _last_price, _mid_price = self._worker_depth_cache[
                arena.arena_id
            ]
            has_bids = bool(np.any(bid_depth[:, 0] > 0))
            has_asks = bool(np.any(ask_depth[:, 0] > 0))
        else:
            orderbook = arena.matching_engine._orderbook
            has_bids = orderbook.get_best_bid() is not None
            has_asks = orderbook.get_best_ask() is not None
        if has_bids != has_asks:
            return ("one_sided_orderbook", None)

        return None

    def _aggregate_tick_trades(self, tick_trades: list[Trade]) -> tuple[float, float]:
        """聚合本 tick 的成交量和成交额"""
        if not tick_trades:
            return 0.0, 0.0

        buy_volume = sum(t.quantity for t in tick_trades if t.is_buyer_taker)
        sell_volume = sum(t.quantity for t in tick_trades if not t.is_buyer_taker)
        buy_amount = sum(t.price * t.quantity for t in tick_trades if t.is_buyer_taker)
        sell_amount = sum(
            t.price * t.quantity for t in tick_trades if not t.is_buyer_taker
        )

        total_volume = buy_volume + sell_volume
        total_amount = buy_amount + sell_amount

        if buy_amount > sell_amount:
            return float(total_volume), total_amount
        elif sell_amount > buy_amount:
            return float(-total_volume), -total_amount
        return 0.0, 0.0

    def _update_episode_price_stats_from_trades(
        self,
        arena: ArenaState,
        tick_trades: list[Trade],
        fallback_price: float | None = None,
    ) -> None:
        """使用本 tick 成交价格更新 episode high/low"""
        if tick_trades:
            tick_high = max(trade.price for trade in tick_trades)
            tick_low = min(trade.price for trade in tick_trades)
        elif fallback_price is not None:
            tick_high = fallback_price
            tick_low = fallback_price
        else:
            return

        if tick_high > arena.episode_high_price:
            arena.episode_high_price = tick_high
        if tick_low < arena.episode_low_price:
            arena.episode_low_price = tick_low

    def _run_episode_all_arenas(
        self,
    ) -> dict[tuple[AgentType, int], np.ndarray]:
        """运行所有竞技场的一个 episode

        Returns:
            汇总后的适应度字典
        """
        episode_length = self.config.training.episode_length

        for _ in range(episode_length):
            if not self._is_running:
                break

            should_continue = self.run_tick_all_arenas()
            if not should_continue:
                break

        # 收集适应度
        return self._collect_episode_fitness()

    def _collect_episode_fitness(self) -> dict[tuple[AgentType, int], np.ndarray]:
        """收集单个 episode 的适应度（跨所有竞技场求和）

        使用相对收益适应度：Agent 收益率 - 市场平均收益率
        这样可以消除市场整体方向的影响，鼓励 Agent 做出相对于市场的超额收益
        """
        accumulated: dict[tuple[AgentType, int], np.ndarray] = {}

        for arena in self.arena_states:
            if arena.arena_id in self._worker_depth_cache:
                _bid_depth, _ask_depth, last_price, mid_price = (
                    self._worker_depth_cache[arena.arena_id]
                )
                current_price = (
                    mid_price
                    if mid_price > 0
                    else last_price if last_price > 0 else arena.smooth_mid_price
                )
            else:
                current_price = arena.matching_engine._orderbook.last_price

            # 计算该竞技场的市场平均收益率（用于相对适应度）
            market_avg_return = self._calculate_market_avg_return(arena, current_price)

            for agent_type, population in self.populations.items():
                if isinstance(population, SubPopulationManager):
                    for sub_pop in population.sub_populations:
                        key = (agent_type, sub_pop.sub_population_id or 0)
                        fitness_arr = self._calculate_fitness_for_population(
                            sub_pop, arena, current_price, market_avg_return
                        )
                        if key not in accumulated:
                            accumulated[key] = fitness_arr.copy()
                        else:
                            accumulated[key] += fitness_arr
                else:
                    key = (agent_type, 0)
                    fitness_arr = self._calculate_fitness_for_population(
                        population, arena, current_price, market_avg_return
                    )
                    if key not in accumulated:
                        accumulated[key] = fitness_arr.copy()
                    else:
                        accumulated[key] += fitness_arr

        return accumulated

    def _calculate_market_avg_return(
        self, arena: ArenaState, current_price: float
    ) -> float:
        """计算单个竞技场的市场平均收益率

        遍历所有 Agent，按初始资金加权计算平均收益率。

        Args:
            arena: 竞技场状态
            current_price: 当前价格

        Returns:
            市场平均收益率
        """
        total_weighted_return = 0.0
        total_initial = 0.0

        for agent_state in arena.agent_states.values():
            initial = agent_state.initial_balance
            if initial <= 0:
                continue

            equity = agent_state.get_equity(current_price)
            weighted = equity - initial
            total_weighted_return += weighted
            total_initial += initial

        return total_weighted_return / total_initial if total_initial > 0 else 0.0

    def _calculate_fitness_for_population(
        self,
        population: Population,
        arena: ArenaState,
        current_price: float,
        market_avg_return: float = 0.0,
    ) -> np.ndarray:
        """计算单个种群在单个竞技场中的适应度

        使用相对收益适应度：Agent 收益率 - 市场平均收益率
        做市商：0.5 * 相对收益率 + 0.5 * maker_volume 排名归一化
        庄家：0.5 * 相对收益率 + 0.5 * volatility_contribution 排名归一化

        Args:
            population: 种群
            arena: 竞技场状态
            current_price: 当前价格
            market_avg_return: 市场平均收益率

        Returns:
            适应度数组
        """
        n = len(population.agents)
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        # 1. 计算所有 Agent 的相对收益率
        relative_returns = np.zeros(n, dtype=np.float32)
        for idx, agent in enumerate(population.agents):
            agent_state = arena.agent_states.get(agent.agent_id)
            if agent_state is None:
                continue

            equity = agent_state.get_equity(current_price)
            initial = agent_state.initial_balance

            if initial > 0:
                agent_return = (equity - initial) / initial
                relative_returns[idx] = agent_return - market_avg_return
            else:
                relative_returns[idx] = 0.0

        # 2. 根据种群类型计算最终适应度
        if population.agent_type == AgentType.MARKET_MAKER:
            # 做市商：0.5 * 相对收益率 + 0.5 * maker_volume 排名归一化
            maker_volumes = np.array(
                [
                    (
                        arena.agent_states[agent.agent_id].maker_volume
                        if agent.agent_id in arena.agent_states
                        else 0
                    )
                    for agent in population.agents
                ],
                dtype=np.float32,
            )

            # 排名归一化到 [0, 1]
            volume_ranks = np.argsort(np.argsort(maker_volumes))
            if n > 1:
                volume_rank_normalized = volume_ranks / (n - 1)
            else:
                volume_rank_normalized = np.zeros(n, dtype=np.float32)

            fitness_arr = 0.5 * relative_returns + 0.5 * volume_rank_normalized

        elif population.agent_type == AgentType.WHALE:
            # 庄家：0.5 * 相对收益率 + 0.5 * volatility_contribution 排名归一化
            volatility_contributions = np.array(
                [
                    (
                        arena.agent_states[agent.agent_id].volatility_contribution
                        if agent.agent_id in arena.agent_states
                        else 0.0
                    )
                    for agent in population.agents
                ],
                dtype=np.float32,
            )

            # 排名归一化到 [0, 1]
            volatility_ranks = np.argsort(np.argsort(volatility_contributions))
            if n > 1:
                volatility_rank_normalized = volatility_ranks / (n - 1)
            else:
                volatility_rank_normalized = np.zeros(n, dtype=np.float32)

            fitness_arr = 0.5 * relative_returns + 0.5 * volatility_rank_normalized

        else:
            # 散户、高级散户：纯相对收益率
            fitness_arr = relative_returns

        return fitness_arr

    def _collect_fitness_all_arenas(
        self,
        arena_fitnesses: list[dict[tuple[AgentType, int], np.ndarray]],
        episode_counts: list[int],
    ) -> dict[tuple[AgentType, int], np.ndarray]:
        """收集并汇总所有竞技场的适应度"""
        return FitnessAggregator.aggregate_simple_average(
            arena_fitnesses, episode_counts
        )

    def _apply_fitness_to_genomes(
        self,
        avg_fitness: dict[tuple[AgentType, int], np.ndarray],
    ) -> None:
        """将汇总的平均适应度应用到基因组"""
        for (agent_type, sub_pop_id), fitness_arr in avg_fitness.items():
            population = self.populations.get(agent_type)
            if population is None:
                continue

            if isinstance(population, SubPopulationManager):
                if sub_pop_id < len(population.sub_populations):
                    sub_pop = population.sub_populations[sub_pop_id]
                    self._apply_fitness_to_population(sub_pop, fitness_arr)
            else:
                if sub_pop_id == 0:
                    self._apply_fitness_to_population(population, fitness_arr)

    def _apply_fitness_to_population(
        self,
        population: Population,
        fitness_arr: np.ndarray,
    ) -> None:
        """将适应度应用到单个种群的基因组"""
        genomes = list(population.neat_pop.population.items())
        for idx, (_genome_id, genome) in enumerate(genomes):
            if idx < len(fitness_arr):
                genome.fitness = float(fitness_arr[idx])

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

    def _update_populations_from_evolution(
        self,
        evolution_results: dict[
            tuple[AgentType, int],
            tuple[
                tuple[np.ndarray, ...],
                tuple[np.ndarray, ...],
                tuple[np.ndarray, np.ndarray],
            ],
        ],
        deserialize_genomes: bool = False,
    ) -> None:
        """从进化结果更新种群

        Args:
            evolution_results: 进化结果字典，包含 (genome_data, network_params_data, species_data)
            deserialize_genomes: 是否反序列化基因组（默认 False，延迟反序列化）
        """
        for (agent_type, sub_pop_id), (
            genome_data,
            network_params_data,
            species_data,
        ) in evolution_results.items():
            population = self.populations.get(agent_type)
            if population is None:
                continue

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

    def _update_single_population(
        self,
        population: Population,
        genome_data: tuple[np.ndarray, ...],
        network_params_data: tuple[np.ndarray, ...],
        species_data: tuple[np.ndarray, np.ndarray],
        deserialize_genomes: bool = False,
    ) -> None:
        """更新单个种群

        Args:
            population: 种群对象
            genome_data: 基因组数据元组
            network_params_data: 网络参数数据元组
            species_data: species 数据元组 (genome_ids, species_ids)
            deserialize_genomes: 是否反序列化基因组（默认 False，延迟反序列化）
        """
        # 增加代数
        population.generation += 1

        # 解包网络参数
        params_list = _unpack_network_params_numpy(*network_params_data)

        if deserialize_genomes:
            # 完整反序列化：重建 NEAT 种群
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

            population._cleanup_neat_history()

            # 更新 Agent Brain（使用完整的 genome + params）
            new_genomes = list(population.neat_pop.population.items())
            for idx, (_gid, genome) in enumerate(new_genomes):
                if idx < len(population.agents) and idx < len(params_list):
                    population.agents[idx].brain.update_from_network_params(
                        genome, params_list[idx]
                    )
        else:
            # 延迟反序列化：只更新网络参数（不更新 genome 引用）
            for idx, params in enumerate(params_list):
                if idx < len(population.agents):
                    population.agents[idx].brain.update_network_only(params)

            # 存储待反序列化数据（用于保存检查点时）
            population._pending_genome_data = genome_data
            population._genomes_dirty = True

            # 同步 species 数据到主进程（关键修复！）
            # 即使延迟反序列化 genome，也需要先反序列化并同步 species
            # 否则 _collect_species_fitness_stats 会返回空数据
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

            # 清理旧的 genome 对象
            new_genome_ids = set(population.neat_pop.population.keys())
            old_to_clean = [g for g in old_genomes if g.key not in new_genome_ids]
            population._cleanup_genome_internals(old_to_clean)

            # 应用 species 数据
            species_genome_ids, species_species_ids = species_data
            _apply_species_data_to_population(
                population.neat_pop,
                species_genome_ids,
                species_species_ids,
                population.generation,
            )

            # 更新 Agent 的 genome 引用
            new_genomes = list(population.neat_pop.population.items())
            for idx, (_gid, genome) in enumerate(new_genomes):
                if idx < len(population.agents):
                    population.agents[idx].brain.genome = genome

            # 标记数据已同步，不需要再延迟反序列化
            population._pending_genome_data = None
            population._genomes_dirty = False

    def _refresh_agent_states(self) -> None:
        """刷新所有竞技场的 Agent 账户状态

        现在使用 calculate_order_quantity_from_state 和 calculate_skew_factor_from_state
        从 AgentAccountState 计算订单数量，不再依赖 Agent 的原始 account，
        因此不需要重置 Agent 的原始账户。
        """
        for arena in self.arena_states:
            arena.agent_states.clear()
            for population in self.populations.values():
                for agent in population.agents:
                    state = AgentAccountState.from_agent(agent)
                    arena.agent_states[agent.agent_id] = state

        # 重建 Agent 映射表
        self._build_agent_map()

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

    def save_checkpoint(self, path: str) -> None:
        """保存检查点

        格式与 SingleArenaTrainer 和 MultiArenaTrainer 兼容。

        Args:
            path: 检查点文件路径
        """
        # 同步待反序列化的基因组数据（延迟反序列化）
        for population in self.populations.values():
            if isinstance(population, SubPopulationManager):
                for sub_pop in population.sub_populations:
                    sub_pop.sync_genomes_from_pending()
            else:
                population.sync_genomes_from_pending()

        # 清理 NEAT 历史数据
        for population in self.populations.values():
            if isinstance(population, SubPopulationManager):
                for sub_pop in population.sub_populations:
                    sub_pop._cleanup_neat_history()
            else:
                population._cleanup_neat_history()

        checkpoint_data: dict[str, Any] = {
            "generation": self.generation,
            "populations": {},
        }

        for agent_type, population in self.populations.items():
            if isinstance(population, SubPopulationManager):
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
                checkpoint_data["populations"][agent_type] = {
                    "generation": population.generation,
                    "neat_pop": population.neat_pop,
                }

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        self.logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str) -> None:
        """加载检查点

        支持多训练场和单训练场的检查点格式。
        - 多训练场：使用 generation 字段
        - 单训练场：使用 episode 字段（映射到 generation）

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
                            sub_pop.neat_pop = sub_pop_data.get("neat_pop")

                            genomes = list(sub_pop.neat_pop.population.items())
                            sub_pop.agents = sub_pop.create_agents(genomes)
                else:
                    self.logger.warning(f"{agent_type.value} 检查点为旧格式，需要迁移")
            else:
                population.generation = pop_data.get("generation", 0)
                population.neat_pop = pop_data.get("neat_pop")

                genomes = list(population.neat_pop.population.items())
                population.agents = population.create_agents(genomes)

        # 更新网络缓存
        self._update_network_caches()

        # 刷新竞技场的 Agent 状态
        self._refresh_agent_states()

        # 重置 Worker 池同步标志
        self._worker_pool_synced = False

        self.logger.info(f"检查点已加载: {path}, generation={self.generation}")

    def stop(self) -> None:
        """停止训练并清理资源"""
        self._is_running = False

        # 关闭 Execute Worker 池
        if self._execute_worker_pool is not None:
            self._execute_worker_pool.shutdown()
            self._execute_worker_pool = None

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
        self.logger.info("多竞技场并行推理训练器已停止")

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
