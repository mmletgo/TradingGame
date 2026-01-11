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

    num_arenas: int = 10
    episodes_per_arena: int = 10

if TYPE_CHECKING:
    from src.training._cython.batch_decide_openmp import BatchNetworkCache

from src.bio.agents.base import ActionType, AgentType
from src.config.config import Config
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
    _deserialize_genomes_numpy,
    _unpack_network_params_numpy,
    malloc_trim,
)

from .arena_state import (
    AgentAccountState,
    AgentStateAdapter,
    ArenaState,
    CatfishAccountState,
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
                tick_history_prices=[initial_price],
                tick_history_volumes=[0.0],
                tick_history_amounts=[0.0],
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

        self.evolution_worker_pool = MultiPopulationWorkerPool(config_dir, worker_configs)
        self.logger.info(f"进化 Worker 池创建完成: {len(worker_configs)} 个 Worker")

    def run_round(self) -> dict[str, Any]:
        """运行一轮训练（所有竞技场的所有 episode + 进化）

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

        for _ep_idx in range(self.multi_config.episodes_per_arena):
            # 重置所有竞技场
            self._reset_all_arenas()

            # 做市商初始化（每个竞技场）
            self._init_market_all_arenas()

            # 运行一个 episode（所有竞技场同步推进）
            episode_fitness = self._run_episode_all_arenas()

            # 累积适应度
            arena_fitnesses.append(episode_fitness)
            episode_counts.append(self.multi_config.num_arenas)

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

    def _reset_all_arenas(self) -> None:
        """重置所有竞技场状态"""
        initial_price = self.config.market.initial_price

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

    def _init_market_all_arenas(self) -> None:
        """所有竞技场做市商初始化（使用批量推理）"""
        mm_population = self.populations.get(AgentType.MARKET_MAKER)
        if not mm_population:
            return

        # 获取所有做市商 Agent
        mm_agents = list(mm_population.agents)

        for arena in self.arena_states:
            orderbook = arena.matching_engine._orderbook

            for agent in mm_agents:
                agent_state = arena.agent_states.get(agent.agent_id)
                if agent_state is None:
                    continue

                # 计算市场状态
                market_state = self._compute_market_state_for_arena(arena)

                # 使用 Agent 的决策方法
                action, params = agent.decide(market_state, orderbook)  # type: ignore[attr-defined]

                # 执行交易
                self._execute_action_in_arena(
                    arena, agent, agent_state, action, params
                )

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

        # 获取实时参考价格
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

        # 使用 get_depth_numpy 直接获取 NumPy 数组
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

        # 向量化成交
        trades = list(arena.recent_trades)
        if trades:
            n_trades = len(trades)
            prices_arr = np.empty(n_trades, dtype=np.float32)
            qtys_arr = np.empty(n_trades, dtype=np.float32)
            for i, t in enumerate(trades):
                prices_arr[i] = t.price
                qtys_arr[i] = t.quantity if t.is_buyer_taker else -t.quantity

            if smooth_mid_price > 0:
                trade_prices[:n_trades] = (
                    prices_arr - smooth_mid_price
                ) / smooth_mid_price
            trade_quantities[:n_trades] = log_normalize_signed(qtys_arr)

        # Tick 历史价格归一化
        if arena.tick_history_prices:
            hist_prices = np.array(arena.tick_history_prices[-100:], dtype=np.float32)
            volumes = np.array(arena.tick_history_volumes[-100:], dtype=np.float32)
            amounts = np.array(arena.tick_history_amounts[-100:], dtype=np.float32)
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

    def _execute_action_in_arena(
        self,
        arena: ArenaState,
        agent: Any,
        agent_state: AgentAccountState,
        action: ActionType,
        params: dict[str, Any],
    ) -> list[Trade]:
        """在竞技场中执行 Agent 动作

        Args:
            arena: 竞技场状态
            agent: Agent 对象
            agent_state: Agent 账户状态
            action: 动作类型
            params: 动作参数

        Returns:
            成交列表
        """
        trades = agent.execute_action(action, params, arena.matching_engine)

        # 更新账户状态
        for trade in trades:
            is_buyer = trade.is_buyer_taker
            # Trade 类有 buyer_fee 和 seller_fee，不是 taker_fee/maker_fee
            # is_buyer_taker=True 时：buyer=taker(fee=buyer_fee), seller=maker(fee=seller_fee)
            # is_buyer_taker=False 时：seller=taker(fee=seller_fee), buyer=maker(fee=buyer_fee)
            fee = trade.buyer_fee if is_buyer else trade.seller_fee
            agent_state.on_trade(
                trade.price, trade.quantity, is_buyer, fee, is_maker=False
            )
            arena.recent_trades.append(trade)

            # 更新 maker 的账户状态
            maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
            maker_state = arena.agent_states.get(maker_id)
            if maker_state is not None:
                maker_is_buyer = not trade.is_buyer_taker
                # maker 的手续费：is_buyer_taker=True 时 maker 是卖方，用 seller_fee
                #                is_buyer_taker=False 时 maker 是买方，用 buyer_fee
                maker_fee = trade.seller_fee if trade.is_buyer_taker else trade.buyer_fee
                maker_state.on_trade(
                    trade.price, trade.quantity, maker_is_buyer, maker_fee, is_maker=True
                )

        return trades

    def run_tick_all_arenas(self) -> bool:
        """并行执行所有竞技场的一个 tick

        Returns:
            bool: 是否所有竞技场都应该继续运行
        """
        # 阶段1: 准备（串行）- 强平检查、计算市场状态
        arena_market_states: list[NormalizedMarketState] = []
        arena_active_agents: list[list[AgentStateAdapter]] = []

        for arena in self.arena_states:
            arena.tick += 1

            # Tick 1: 只记录做市商初始挂单后的状态
            if arena.tick == 1:
                current_price = arena.smooth_mid_price
                arena.price_history.append(current_price)
                arena.tick_history_prices.append(current_price)
                arena.tick_history_volumes.append(0.0)
                arena.tick_history_amounts.append(0.0)
                arena.update_price_stats(current_price)
                arena_market_states.append(
                    self._compute_market_state_for_arena(arena)
                )
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
            self._catfish_action_for_arena(arena)

            # 计算市场状态
            market_state = self._compute_market_state_for_arena(arena)
            arena_market_states.append(market_state)

            # 收集活跃的 Agent 状态适配器
            active_adapters: list[AgentStateAdapter] = []
            for agent_state in arena.agent_states.values():
                if not agent_state.is_liquidated:
                    active_adapters.append(AgentStateAdapter(agent_state))

            # 随机打乱执行顺序
            random.shuffle(active_adapters)
            arena_active_agents.append(active_adapters)

        # 阶段2: 批量推理（并行）- 合并所有竞技场的推理
        all_decisions = self._batch_inference_all_arenas(
            arena_market_states, arena_active_agents
        )

        # 阶段3: 执行（串行）- 每个竞技场的交易执行
        all_continue = True
        for arena_idx, arena in enumerate(self.arena_states):
            if arena.tick == 1:
                continue

            decisions = all_decisions.get(arena_idx, [])
            tick_trades: list[Trade] = []

            for adapter, action, params in decisions:
                agent = self._get_agent_by_id(adapter.agent_id)
                agent_state = arena.agent_states.get(adapter.agent_id)
                if agent is None or agent_state is None:
                    continue

                trades = self._execute_action_in_arena(
                    arena, agent, agent_state, action, params
                )
                tick_trades.extend(trades)

            # 记录价格历史
            current_price = arena.matching_engine._orderbook.last_price
            arena.price_history.append(current_price)
            arena.update_price_stats(current_price)

            # 记录 tick 历史数据
            arena.tick_history_prices.append(current_price)
            volume, amount = self._aggregate_tick_trades(tick_trades)
            arena.tick_history_volumes.append(volume)
            arena.tick_history_amounts.append(amount)

            # 限制历史长度
            if len(arena.tick_history_prices) > 100:
                arena.tick_history_prices = arena.tick_history_prices[-100:]
                arena.tick_history_volumes = arena.tick_history_volumes[-100:]
                arena.tick_history_amounts = arena.tick_history_amounts[-100:]

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
                    market_state, adapters
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
            adapter_mapping: list[list[AgentStateAdapter]] = []  # 记录每个竞技场的 adapter 顺序

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
                    market_state.tick_size if market_state.tick_size > 0 else 0.1
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
                            action_type_int, side_int, price, quantity = raw_result
                            action, params = self._convert_retail_result(
                                agent,
                                action_type_int,
                                side_int,
                                price,
                                quantity,
                                mid_price,
                            )

                        results[arena_idx].append((adapter, action, params))
                    except Exception:
                        pass

        return results

    def _serial_inference_for_arena(
        self,
        market_state: NormalizedMarketState,
        adapters: list[AgentStateAdapter],
    ) -> list[tuple[AgentStateAdapter, ActionType, dict[str, Any]]]:
        """串行推理（回退方案）"""
        results: list[tuple[AgentStateAdapter, ActionType, dict[str, Any]]] = []

        for adapter in adapters:
            agent = self._get_agent_by_id(adapter.agent_id)
            if agent is None:
                continue

            try:
                # 需要一个临时的 orderbook 引用
                # 这里使用第一个竞技场的订单簿（仅用于获取结构信息）
                orderbook = self.arena_states[0].matching_engine._orderbook
                action, params = agent.decide(market_state, orderbook)
                results.append((adapter, action, params))
            except Exception:
                pass

        return results

    def _get_agent_by_id(self, agent_id: int) -> Any:
        """根据 ID 获取 Agent 对象（O(1) 查找）"""
        return self.agent_map.get(agent_id)

    def _parse_market_maker_output(
        self,
        agent: Any,
        output: np.ndarray,
        mid_price: float,
        tick_size: float,
    ) -> tuple[ActionType, dict[str, Any]]:
        """解析做市商的神经网络输出"""
        from src.bio.agents.base import fast_clip, fast_round_price

        bid_price_offsets = output[0:5]
        bid_qty_weights = output[5:10]
        ask_price_offsets = output[10:15]
        ask_qty_weights = output[15:20]
        total_ratio_raw = output[20] if len(output) > 20 else 0.0

        total_ratio = (fast_clip(total_ratio_raw, -1.0, 1.0) + 1) * 0.5
        skew_factor = agent._calculate_skew_factor(mid_price)

        bid_weights_sum = sum(max(0, (w + 1) * 0.5) for w in bid_qty_weights)
        ask_weights_sum = sum(max(0, (w + 1) * 0.5) for w in ask_qty_weights)
        total_weights = bid_weights_sum + ask_weights_sum

        if total_weights > 0:
            bid_multiplier = 1.0 + skew_factor
            ask_multiplier = 1.0 - skew_factor
            bid_weights_sum *= bid_multiplier
            ask_weights_sum *= ask_multiplier
            total_weights = bid_weights_sum + ask_weights_sum

        bid_orders: list[dict[str, float]] = []
        ask_orders: list[dict[str, float]] = []

        for i in range(5):
            offset = fast_clip(bid_price_offsets[i], -1.0, 1.0)
            ticks = 1 + (offset + 1) * 49.5
            price = mid_price - ticks * tick_size
            price = fast_round_price(price, tick_size)

            weight = max(0, (bid_qty_weights[i] + 1) * 0.5)
            if total_weights > 0 and weight > 0:
                ratio = (weight / total_weights) * total_ratio
                qty = agent._calculate_order_quantity(
                    price, ratio, is_buy=True, ref_price=mid_price
                )
                if qty > 0:
                    bid_orders.append({"price": price, "quantity": float(qty)})

            offset = fast_clip(ask_price_offsets[i], -1.0, 1.0)
            ticks = 1 + (offset + 1) * 49.5
            price = mid_price + ticks * tick_size
            price = fast_round_price(price, tick_size)

            weight = max(0, (ask_qty_weights[i] + 1) * 0.5)
            if total_weights > 0 and weight > 0:
                ratio = (weight / total_weights) * total_ratio
                qty = agent._calculate_order_quantity(
                    price, ratio, is_buy=False, ref_price=mid_price
                )
                if qty > 0:
                    ask_orders.append({"price": price, "quantity": float(qty)})

        return ActionType.HOLD, {"bid_orders": bid_orders, "ask_orders": ask_orders}

    def _convert_retail_result(
        self,
        agent: Any,
        action_type_int: int,
        side_int: int,
        price: float,
        quantity: int,
        mid_price: float,
    ) -> tuple[ActionType, dict[str, Any]]:
        """将 Cython 返回的散户/高级散户/庄家结果转换为 ActionType 和 params"""
        params: dict[str, Any] = {}

        if action_type_int == 0:
            action = ActionType.HOLD
        elif action_type_int == 1:
            action = ActionType.PLACE_BID
            params["price"] = price
            actual_qty = agent._calculate_order_quantity(price, quantity, is_buy=True)
            params["quantity"] = actual_qty
        elif action_type_int == 2:
            action = ActionType.PLACE_ASK
            params["price"] = price
            actual_qty = agent._calculate_order_quantity(price, quantity, is_buy=False)
            params["quantity"] = actual_qty
        elif action_type_int == 3:
            action = ActionType.CANCEL
        elif action_type_int == 4:
            action = ActionType.MARKET_BUY
            actual_qty = agent._calculate_order_quantity(
                mid_price, quantity, is_buy=True
            )
            params["quantity"] = actual_qty
        elif action_type_int == 5:
            action = ActionType.MARKET_SELL
            position_qty = agent.account.position.quantity
            if position_qty > 0:
                sell_qty = max(1, int(position_qty * quantity))
                params["quantity"] = min(sell_qty, int(position_qty))
            else:
                actual_qty = agent._calculate_order_quantity(
                    mid_price, quantity, is_buy=False
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

        # 阶段1: 撤销挂单
        for agent_state in agents_to_liquidate:
            agent = self._get_agent_by_id(agent_state.agent_id)
            if agent:
                self._cancel_agent_orders_in_arena(arena, agent)

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

    def _cancel_agent_orders_in_arena(self, arena: ArenaState, agent: Any) -> None:
        """撤销 Agent 在竞技场中的挂单"""
        if agent.agent_type == AgentType.MARKET_MAKER:
            from src.bio.agents.market_maker import MarketMakerAgent

            if isinstance(agent, MarketMakerAgent):
                agent._cancel_all_orders(arena.matching_engine)
        else:
            agent_state = arena.agent_states.get(agent.agent_id)
            if agent_state and agent_state.pending_order_id is not None:
                arena.matching_engine.cancel_order(agent_state.pending_order_id)
                agent_state.pending_order_id = None

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
                maker_fee = trade.seller_fee if trade.is_buyer_taker else trade.buyer_fee
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
            pnl_percent = (equity - agent_state.initial_balance) / agent_state.initial_balance

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
                agent_state.balance += (adl_price - agent_state.position_avg_price) * trade_qty * (1 if is_long else -1)

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

    def _catfish_action_for_arena(self, arena: ArenaState) -> None:
        """鲶鱼在竞技场中的行动"""
        if not arena.catfish_states:
            return

        orderbook = arena.matching_engine._orderbook

        for catfish_state in arena.catfish_states.values():
            if catfish_state.is_liquidated:
                continue

            # 简化的鲶鱼决策逻辑
            # 这里需要根据实际的鲶鱼类型实现具体逻辑
            # 暂时跳过鲶鱼行动
            pass

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
        """收集单个 episode 的适应度（跨所有竞技场求和）"""
        accumulated: dict[tuple[AgentType, int], np.ndarray] = {}

        for arena in self.arena_states:
            current_price = arena.matching_engine._orderbook.last_price

            for agent_type, population in self.populations.items():
                if isinstance(population, SubPopulationManager):
                    for sub_pop in population.sub_populations:
                        key = (agent_type, sub_pop.sub_population_id or 0)
                        fitness_arr = self._calculate_fitness_for_population(
                            sub_pop, arena, current_price
                        )
                        if key not in accumulated:
                            accumulated[key] = fitness_arr.copy()
                        else:
                            accumulated[key] += fitness_arr
                else:
                    key = (agent_type, 0)
                    fitness_arr = self._calculate_fitness_for_population(
                        population, arena, current_price
                    )
                    if key not in accumulated:
                        accumulated[key] = fitness_arr.copy()
                    else:
                        accumulated[key] += fitness_arr

        return accumulated

    def _calculate_fitness_for_population(
        self,
        population: Population,
        arena: ArenaState,
        current_price: float,
    ) -> np.ndarray:
        """计算单个种群在单个竞技场中的适应度"""
        fitness_arr = np.zeros(len(population.agents), dtype=np.float32)

        for idx, agent in enumerate(population.agents):
            agent_state = arena.agent_states.get(agent.agent_id)
            if agent_state is None:
                continue

            # 计算净值
            equity = agent_state.get_equity(current_price)
            initial = agent_state.initial_balance

            # 计算适应度（收益率）
            if initial > 0:
                fitness = (equity - initial) / initial
            else:
                fitness = 0.0

            fitness_arr[idx] = fitness

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
            tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]],
        ],
        deserialize_genomes: bool = False,
    ) -> None:
        """从进化结果更新种群

        Args:
            evolution_results: 进化结果字典
            deserialize_genomes: 是否反序列化基因组（默认 False，延迟反序列化）
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
                        sub_pop, genome_data, network_params_data, deserialize_genomes
                    )
            else:
                if sub_pop_id == 0:
                    self._update_single_population(
                        population, genome_data, network_params_data, deserialize_genomes
                    )

    def _update_single_population(
        self,
        population: Population,
        genome_data: tuple[np.ndarray, ...],
        network_params_data: tuple[np.ndarray, ...],
        deserialize_genomes: bool = False,
    ) -> None:
        """更新单个种群

        Args:
            population: 种群对象
            genome_data: 基因组数据元组
            network_params_data: 网络参数数据元组
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

    def _refresh_agent_states(self) -> None:
        """刷新所有竞技场的 Agent 账户状态"""
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

        支持 MultiArenaTrainer 和 SingleArenaTrainer 的检查点格式。

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

        self.generation = checkpoint_data.get("generation", 0)
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
                    self.logger.warning(
                        f"{agent_type.value} 检查点为旧格式，需要迁移"
                    )
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
