"""训练器模块

管理训练流程，协调种群和撮合引擎。
"""

import gc
import pickle
import random
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.bio.agents.base import AgentType


def _get_memory_mb() -> float:
    """获取当前进程的内存使用量（MB）"""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:
        pass
    return 0.0


if TYPE_CHECKING:
    from src.bio.agents.base import ActionType, Agent
    from src.market.catfish.catfish_base import CatfishBase
    from src.market.market_state import (
        NormalizedMarketState as NormalizedMarketStateType,
    )
    from src.market.orderbook.orderbook import OrderBook
    from src.training.generation_saver import GenerationSaver
from src.config.config import Config
from src.core.log_engine.logger import get_logger
from src.market.adl.adl_manager import ADLCandidate, ADLManager
from src.market.catfish import CatfishBase, create_all_catfish, create_catfish
from src.market.market_state import NormalizedMarketState
from src.market.matching.matching_engine import MatchingEngine
from src.market.matching.trade import Trade
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.training.fast_math import log_normalize_signed, log_normalize_unsigned
from src.training.population import Population, RetailSubPopulationManager, malloc_trim

# 缓存类型常量 - 直接在 Python 中定义，避免 Cython 导出问题
CACHE_TYPE_RETAIL = 0
CACHE_TYPE_FULL = 1
CACHE_TYPE_MARKET_MAKER = 2

# 尝试导入 OpenMP 批量决策缓存模块
try:
    from src.training._cython.batch_decide_openmp import (
        batch_decide_retail,
        batch_decide_full,
        batch_decide_market_maker,
        BatchNetworkCache,
    )
    HAS_OPENMP_DECIDE = True
except ImportError:
    HAS_OPENMP_DECIDE = False
    BatchNetworkCache = None  # type: ignore


class Trainer:
    """训练器

    管理 NEAT 进化训练流程，协调四种群在模拟市场中竞争。

    Attributes:
        config: 全局配置
        populations: 四种群（散户/高级散户/庄家/做市商）
        matching_engine: 撮合引擎
        tick: 当前 tick
        episode: 当前 episode
        is_running: 是否运行中
        is_paused: 是否暂停
        recent_trades: 最近成交记录
    """

    config: Config
    populations: dict[AgentType, Population | RetailSubPopulationManager]
    matching_engine: MatchingEngine | None
    adl_manager: ADLManager | None
    tick: int
    episode: int
    is_running: bool
    is_paused: bool
    recent_trades: deque[Trade]
    agent_map: dict[int, "Agent"]
    agent_execution_order: list["Agent"]
    _pop_total_counts: dict[AgentType, int]  # 各种群总数
    _pop_liquidated_counts: dict[
        AgentType, int
    ]  # 各种群当前 episode 已淘汰数量（爆仓即淘汰）
    tick_start_price: float  # tick 开始时的价格，供数据采集使用
    _eliminating_agents: set[int]  # 正在强平/淘汰的 Agent ID 集合（防止重入）
    _adl_long_candidates: list[
        ADLCandidate
    ]  # 多头 ADL 候选（已排序，持仓数量会动态更新）
    _adl_short_candidates: list[
        ADLCandidate
    ]  # 空头 ADL 候选（已排序，持仓数量会动态更新）
    _executor: ThreadPoolExecutor | None
    _num_workers: int
    catfish_list: list["CatfishBase"]
    _price_history: list[float]
    _tick_history_prices: list[float]
    _tick_history_volumes: list[float]
    _tick_history_amounts: list[float]
    _episode_high_price: float  # 当前 episode 最高价
    _episode_low_price: float  # 当前 episode 最低价
    arena_id: int | None  # 竞技场 ID（多竞技场场景）
    _generation_saver: "GenerationSaver | None"

    def __init__(self, config: Config, arena_id: int | None = None) -> None:
        """创建训练器

        Args:
            config: 全局配置对象
            arena_id: 竞技场 ID（多竞技场场景，默认 None）
        """
        self.config = config
        self.arena_id = arena_id
        self.logger = get_logger("trainer")

        self.tick = 0
        self.episode = 0
        self.is_running = False
        self.is_paused = False

        self.populations = {}
        self.matching_engine = None
        self.adl_manager = None
        self.recent_trades = deque(maxlen=100)
        self.agent_map = {}
        self.agent_execution_order = []
        self._pop_total_counts = {}
        self._pop_liquidated_counts = {}
        self.tick_start_price = 0.0
        self._eliminating_agents = set()
        self._adl_long_candidates = []
        self._adl_short_candidates = []
        self._executor = None
        self._num_workers = 16

        # EMA 平滑价格相关
        self._smooth_mid_price: float = 0.0
        self._ema_alpha: float = 0.1

        # 鲶鱼相关
        self.catfish_list: list["CatfishBase"] = []
        self._price_history: list[float] = []
        self._catfish_liquidated: bool = False  # 鲶鱼是否被强平（触发 episode 结束）

        # Tick 历史数据（用于 Agent 输入特征）
        self._tick_history_prices: list[float] = []  # 每 tick 价格
        self._tick_history_volumes: list[float] = []  # 每 tick 成交量（带方向）
        self._tick_history_amounts: list[float] = []  # 每 tick 成交额（带方向）

        # Episode 价格统计
        self._episode_high_price: float = 0.0
        self._episode_low_price: float = 0.0

        # 每代保存器
        self._generation_saver: "GenerationSaver | None" = None

        # 预分配市场状态缓冲区（性能优化）
        self._market_state_buffers: dict[str, NDArray[np.float32]] = {
            'bid_data': np.zeros(200, dtype=np.float32),
            'ask_data': np.zeros(200, dtype=np.float32),
            'trade_prices': np.zeros(100, dtype=np.float32),
            'trade_quantities': np.zeros(100, dtype=np.float32),
            'tick_prices': np.zeros(100, dtype=np.float32),
            'tick_volumes': np.zeros(100, dtype=np.float32),
            'tick_amounts': np.zeros(100, dtype=np.float32),
        }

        # 网络数据缓存（OpenMP 优化）
        self._network_caches: dict[AgentType, "BatchNetworkCache"] | None = None
        self._cache_initialized: bool = False
        # 是否使用 OpenMP 决策（可通过配置关闭）
        self.use_openmp_decide: bool = True

    def set_generation_saver(self, saver: "GenerationSaver") -> None:
        """设置每代保存器

        设置后，每次进化完成会自动保存该代的最佳基因组。

        Args:
            saver: GenerationSaver 实例
        """
        self._generation_saver = saver

    def _init_ema_price(self, initial_price: float) -> None:
        """初始化 EMA 平滑价格

        在 episode 开始时调用，使用初始价格作为 EMA 的初始值。

        Args:
            initial_price: 初始价格
        """
        self._smooth_mid_price = initial_price

    def _update_ema_price(self, current_mid_price: float) -> float:
        """更新 EMA 平滑价格

        公式: smooth = alpha * current + (1 - alpha) * prev_smooth

        Args:
            current_mid_price: 当前实时 mid_price

        Returns:
            更新后的 smooth_mid_price
        """
        self._smooth_mid_price = (
            self._ema_alpha * current_mid_price
            + (1 - self._ema_alpha) * self._smooth_mid_price
        )
        return self._smooth_mid_price

    def _aggregate_tick_trades(self, tick_trades: list[Trade]) -> tuple[float, float]:
        """聚合本 tick 的成交量和成交额（符号+总量方式）

        统计本 tick 所有成交的 taker 买入和 taker 卖出。
        如果买入成交额 > 卖出成交额，结果为 +总量
        如果卖出成交额 > 买入成交额，结果为 -总量
        如果相等，结果为 0

        Returns:
            (带方向的成交量, 带方向的成交额)
        """
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

    def _setup_populations_from_checkpoint(
        self, populations_data: dict[AgentType, dict]
    ) -> None:
        """从检查点数据直接创建种群

        直接从检查点的 NEAT 种群数据创建 Agent，避免先创建全新 Agent 再清理的开销。

        Args:
            populations_data: 检查点中的种群数据
                格式: {agent_type: {"generation": int, "neat_pop": neat.Population}}
        """
        for agent_type in AgentType:
            if agent_type not in populations_data:
                # 如果检查点中没有此种群，创建全新种群
                self.populations[agent_type] = Population(agent_type, self.config)
                continue

            pop_data = populations_data[agent_type]

            # 创建 Population 对象但不初始化 NEAT 种群（使用检查点数据）
            pop = Population.__new__(Population)
            pop.agent_type = agent_type
            pop.agent_config = self.config.agents[agent_type]
            pop.generation = pop_data["generation"]
            pop.logger = get_logger("population")
            pop._executor = None
            pop._num_workers = 8

            # 加载 NEAT 配置
            from pathlib import Path
            import neat

            config_dir = Path(self.config.training.neat_config_path)
            if agent_type == AgentType.MARKET_MAKER:
                neat_config_path = config_dir / "neat_market_maker.cfg"
            elif agent_type == AgentType.WHALE:
                neat_config_path = config_dir / "neat_whale.cfg"
            elif agent_type == AgentType.RETAIL_PRO:
                neat_config_path = config_dir / "neat_retail_pro.cfg"
            else:
                neat_config_path = config_dir / "neat_retail.cfg"

            pop.neat_config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                str(neat_config_path),
            )
            pop.neat_config.pop_size = pop.agent_config.count

            # 直接使用检查点中的 NEAT 种群
            pop.neat_pop = pop_data["neat_pop"]

            # 从 NEAT 种群创建 Agent
            genomes = list(pop.neat_pop.population.items())
            pop.agents = pop.create_agents(genomes)

            self.populations[agent_type] = pop

            self.logger.info(
                f"从检查点恢复 {agent_type.value} 种群，"
                f"代数: {pop.generation}，Agent 数量: {len(pop.agents)}"
            )

    def _calculate_catfish_initial_balance(self) -> float:
        """计算每条鲶鱼的初始资金

        公式：鲶鱼总资金 = 做市商杠杆后资金 - 其他物种杠杆后资金
        每条鲶鱼分配 1/3

        Returns:
            每条鲶鱼的初始资金

        Raises:
            ValueError: 如果其他物种资金 >= 做市商资金
        """
        agents_config = self.config.agents

        # 做市商杠杆后资金
        mm_config = agents_config[AgentType.MARKET_MAKER]
        mm_fund = mm_config.count * mm_config.initial_balance * mm_config.leverage

        # 其他物种杠杆后资金
        other_fund = 0.0
        for agent_type in [AgentType.RETAIL, AgentType.RETAIL_PRO, AgentType.WHALE]:
            cfg = agents_config[agent_type]
            other_fund += cfg.count * cfg.initial_balance * cfg.leverage

        # 校验资金配置
        if other_fund >= mm_fund:
            raise ValueError(
                f"鲶鱼资金配置错误：做市商杠杆后资金({mm_fund})必须大于"
                f"其他物种杠杆后资金({other_fund})"
            )

        # 每条鲶鱼资金（3条鲶鱼）
        catfish_count = 3
        per_catfish = (mm_fund - other_fund) / catfish_count
        return per_catfish

    def _get_executor(self) -> ThreadPoolExecutor:
        """获取或创建线程池（惰性初始化）"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._num_workers, thread_name_prefix="neat_worker"
            )
        return self._executor

    def _shutdown_executor(self) -> None:
        """关闭线程池"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _init_network_caches(self) -> None:
        """初始化网络数据缓存

        为每种 Agent 类型创建 BatchNetworkCache，预分配内存和提取网络数据。
        只在 HAS_OPENMP_DECIDE 为 True 且 use_openmp_decide 为 True 时创建。
        """
        if not HAS_OPENMP_DECIDE or not self.use_openmp_decide:
            self._network_caches = None
            self._cache_initialized = False
            return

        if BatchNetworkCache is None:
            self._network_caches = None
            self._cache_initialized = False
            return

        self._network_caches = {}

        # 为每种 Agent 类型创建缓存
        for agent_type, population in self.populations.items():
            if not population.agents:
                continue

            # 确定缓存类型
            if agent_type == AgentType.RETAIL:
                cache_type = CACHE_TYPE_RETAIL
            elif agent_type == AgentType.MARKET_MAKER:
                cache_type = CACHE_TYPE_MARKET_MAKER
            else:  # RETAIL_PRO, WHALE
                cache_type = CACHE_TYPE_FULL

            # 创建缓存（使用配置的线程数）
            num_networks = len(population.agents)
            num_threads = self.config.training.openmp_threads
            cache = BatchNetworkCache(num_networks, cache_type, num_threads)

            # 提取网络数据
            networks = [agent.brain.network for agent in population.agents]
            cache.update_networks(networks)

            self._network_caches[agent_type] = cache
            self.logger.debug(f"已为 {agent_type.value} 创建网络缓存，数量={num_networks}")

        self._cache_initialized = True
        self.logger.info(f"网络数据缓存初始化完成，共 {len(self._network_caches)} 种类型")

    def _update_network_caches(self) -> None:
        """更新网络数据缓存

        在 NEAT 进化后调用，重新提取所有网络数据到缓存。
        """
        if self._network_caches is None:
            return

        for agent_type, population in self.populations.items():
            cache = self._network_caches.get(agent_type)
            if cache is None or not population.agents:
                continue

            networks = [agent.brain.network for agent in population.agents]
            cache.update_networks(networks)

    def setup(self, checkpoint: dict | None = None) -> None:
        """初始化训练环境

        创建四种群、撮合引擎，初始化市场。
        训练模式使用直接调用。

        Args:
            checkpoint: 可选的检查点数据，如果提供则直接从检查点恢复种群
                       而不是创建全新种群（避免不必要的内存分配）
        """
        # 串行创建四种群
        if checkpoint is not None and "populations" in checkpoint:
            # 直接从检查点恢复，不创建全新 Agent
            self._setup_populations_from_checkpoint(checkpoint["populations"])
            self.tick = checkpoint.get("tick", 0)
            self.episode = checkpoint.get("episode", 0)
        else:
            # 创建全新种群（每个种群内部的Agent创建是并行的）
            for agent_type in AgentType:
                if agent_type == AgentType.RETAIL:
                    # 散户使用子种群管理器
                    sub_count = self.config.training.retail_sub_population_count
                    self.populations[agent_type] = RetailSubPopulationManager(
                        self.config, sub_count=sub_count
                    )
                else:
                    self.populations[agent_type] = Population(agent_type, self.config)

        # 创建撮合引擎
        self.matching_engine = MatchingEngine(self.config.market)

        # 创建 ADL 管理器
        self.adl_manager = ADLManager()

        # 初始化鲶鱼（如果配置中启用）
        if self.config.catfish and self.config.catfish.enabled:
            # 计算鲶鱼初始资金
            catfish_initial_balance = self._calculate_catfish_initial_balance()

            # 鲶鱼使用庄家的杠杆和维持保证金率
            whale_config = self.config.agents[AgentType.WHALE]
            catfish_leverage = whale_config.leverage
            catfish_mmr = whale_config.maintenance_margin_rate

            if self.config.catfish.multi_mode:
                # 多模式：同时创建三种鲶鱼
                self.catfish_list = create_all_catfish(
                    self.config.catfish,
                    initial_balance=catfish_initial_balance,
                    leverage=catfish_leverage,
                    maintenance_margin_rate=catfish_mmr,
                )
                for catfish in self.catfish_list:
                    self.matching_engine.register_agent(
                        catfish.catfish_id,
                        0.0,  # maker fee
                        0.0,  # taker fee
                    )
                self.logger.info(
                    f"鲶鱼已启用: 多模式（三种鲶鱼同时运行，相位错开）, "
                    f"每条初始资金={catfish_initial_balance/1e8:.2f}亿"
                )
            else:
                # 单模式：只创建一种鲶鱼
                catfish = create_catfish(
                    -1,
                    self.config.catfish,
                    initial_balance=catfish_initial_balance,
                    leverage=catfish_leverage,
                    maintenance_margin_rate=catfish_mmr,
                )
                self.catfish_list = [catfish]
                self.matching_engine.register_agent(
                    catfish.catfish_id,
                    0.0,  # maker fee
                    0.0,  # taker fee
                )
                self.logger.info(
                    f"鲶鱼已启用: 模式={self.config.catfish.mode.value}, "
                    f"初始资金={catfish_initial_balance/1e8:.2f}亿"
                )

        # 注册所有 Agent 的费率
        self._register_all_agents()

        # 构建 Agent 映射表和执行顺序
        self._build_agent_map()
        self._build_execution_order()

        # 记录各种群总数
        self._update_pop_total_counts()

        # 读取 EMA 配置并初始化
        self._ema_alpha = self.config.market.ema_alpha
        self._init_ema_price(self.config.market.initial_price)

        # 初始化市场（做市商先行动）
        self._init_market()

        # 初始化网络数据缓存（OpenMP 优化）
        self._init_network_caches()

        self.logger.info("训练环境初始化完成")

    def _register_all_agents(self) -> None:
        """注册所有 Agent 的费率到撮合引擎

        遍历所有种群的 Agent，将其费率信息注册到撮合引擎。
        应在 setup() 后和 evolve() 后调用。
        """
        if not self.matching_engine:
            return

        for agent_type, population in self.populations.items():
            agent_config = population.agent_config
            for agent in population.agents:
                self.matching_engine.register_agent(
                    agent.agent_id,
                    agent_config.maker_fee_rate,
                    agent_config.taker_fee_rate,
                )

    def _build_agent_map(self) -> None:
        """构建 Agent ID 到 Agent 对象的映射表"""
        self.agent_map.clear()
        for population in self.populations.values():
            for agent in population.agents:
                self.agent_map[agent.agent_id] = agent

    def _build_execution_order(self) -> None:
        """构建 Agent 执行顺序列表（做市商 -> 庄家 -> 高级散户 -> 散户）"""
        self.agent_execution_order.clear()
        for agent_type in [
            AgentType.MARKET_MAKER,
            AgentType.WHALE,
            AgentType.RETAIL_PRO,
            AgentType.RETAIL,
        ]:
            population = self.populations.get(agent_type)
            if population:
                self.agent_execution_order.extend(population.agents)

    def _update_pop_total_counts(self) -> None:
        """更新各种群总数"""
        for agent_type, population in self.populations.items():
            self._pop_total_counts[agent_type] = len(population.agents)

    def _update_agents_after_migration(
        self,
        replaced_info: dict[AgentType, list[tuple[int, "Agent", "Agent"]]],
    ) -> None:
        """迁移后增量更新内部状态

        仅更新被替换的 Agent 相关状态，避免重建整个 agent_map 和 agent_execution_order。

        Args:
            replaced_info: {agent_type: [(idx, old_agent, new_agent), ...]}
        """
        if not self.matching_engine:
            return

        # 1. 遍历所有被替换的 agent
        for agent_type, replacements in replaced_info.items():
            for idx, old_agent, new_agent in replacements:
                # 2. 更新 agent_map：移除旧 agent 的映射，添加新 agent 的映射
                if old_agent.agent_id in self.agent_map:
                    del self.agent_map[old_agent.agent_id]
                self.agent_map[new_agent.agent_id] = new_agent

                # 3. 更新 agent_execution_order：找到旧 agent 的位置，替换为新 agent
                try:
                    exec_idx = self.agent_execution_order.index(old_agent)
                    self.agent_execution_order[exec_idx] = new_agent
                except ValueError:
                    # 旧 agent 不在执行顺序中（可能已被移除），添加新 agent
                    self.agent_execution_order.append(new_agent)

                # 4. 注册新 agent 的费率到撮合引擎
                agent_config = self.populations[agent_type].agent_config
                self.matching_engine.register_agent(
                    new_agent.agent_id,
                    agent_config.maker_fee_rate,
                    agent_config.taker_fee_rate,
                )

    def _mark_agent_liquidated(self, agent_id: int) -> None:
        """标记 Agent 已被强平

        Args:
            agent_id: 被强平的 Agent ID
        """
        agent = self.agent_map.get(agent_id)
        if agent and not agent.is_liquidated:
            agent.is_liquidated = True
            # 根据类型输出不同的日志
            type_name = {
                AgentType.RETAIL: "散户",
                AgentType.RETAIL_PRO: "高级散户",
                AgentType.WHALE: "庄家",
                AgentType.MARKET_MAKER: "做市商",
            }.get(agent.agent_type, str(agent.agent_type.value))

            self.logger.info(
                f"{type_name} Agent {agent_id} 已被强平，本轮 episode 禁用"
            )

    def _cancel_agent_orders(self, agent: "Agent") -> None:
        """撤销 agent 的所有挂单

        Args:
            agent: 要撤销挂单的 Agent
        """
        if not self.matching_engine:
            return

        # 做市商有多个挂单（bid_order_ids 和 ask_order_ids），需要全部撤销
        if agent.agent_type == AgentType.MARKET_MAKER:
            from src.bio.agents.market_maker import MarketMakerAgent

            if isinstance(agent, MarketMakerAgent):
                agent._cancel_all_orders(self.matching_engine)
        elif agent.account.pending_order_id is not None:
            self.matching_engine.cancel_order(agent.account.pending_order_id)
            agent.account.pending_order_id = None

    def _execute_liquidation_market_order(self, agent: "Agent") -> tuple[int, bool]:
        """执行强平市价单（不含 ADL）

        只执行市价单平仓，不执行 ADL。用于三阶段强平流程中的阶段2。

        Args:
            agent: 被强平的 Agent

        Returns:
            (剩余未平仓数量, 是否为多头)
        """
        position_qty = agent.account.position.quantity
        if position_qty == 0:
            return 0, True

        # 记录原始持仓方向
        is_long = position_qty > 0
        target_qty = abs(position_qty)

        # 创建市价平仓单
        side = OrderSide.SELL if is_long else OrderSide.BUY
        order = Order(
            order_id=agent._generate_order_id(),
            agent_id=agent.agent_id,
            side=side,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=target_qty,
        )

        # 直接撮合
        trades = self.matching_engine.process_order(order)

        # 更新账户（taker 和 maker）
        for trade in trades:
            is_buyer = trade.is_buyer_taker
            agent.account.on_trade(trade, is_buyer)
            self.recent_trades.append(trade)
            # 更新 maker 的账户
            maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
            maker_agent = self.agent_map.get(maker_id)
            if maker_agent is not None:
                maker_is_buyer = not trade.is_buyer_taker
                maker_agent.account.on_trade(trade, maker_is_buyer)

        # 返回剩余未平仓数量和方向
        remaining_qty = abs(agent.account.position.quantity)
        return remaining_qty, is_long

    def _execute_adl(
        self,
        liquidated_agent: "Agent",
        remaining_qty: int,
        current_price: float,
        is_long: bool,
    ) -> None:
        """执行 ADL 自动减仓

        使用 tick 开始时预计算的候选清单，避免重复计算 ADL 分数。
        ADL 成交后更新候选清单中的 position_qty，确保后续 ADL 不会重复使用已减掉的仓位。

        Args:
            liquidated_agent: 被强平的 Agent
            remaining_qty: 剩余需要平仓的数量
            current_price: 当前市场价格
            is_long: 被强平方是否为多头
        """
        if not self.adl_manager:
            return

        # ADL 成交价格：直接使用当前市场价格
        adl_price = self.adl_manager.get_adl_price(current_price)

        # 选择对应方向的预计算候选清单
        # 被强平方是多头（需要卖出平仓），则需要空头对手
        # 被强平方是空头（需要买入平仓），则需要多头对手
        candidates = (
            self._adl_short_candidates if is_long else self._adl_long_candidates
        )

        # self.logger.info(
        #     f"ADL 触发: Agent {liquidated_agent.agent_id} "
        #     f"剩余平仓量 {remaining_qty}, "
        #     f"成交价 {adl_price:.2f}, "
        #     f"候选人数 {len(candidates)}"
        # )

        for candidate in candidates:
            if remaining_qty <= 0:
                break
            # 使用候选清单中的 position_qty（已被之前的 ADL 更新过）
            # 同时也要检查实际仓位，取两者最小值
            candidate_available_qty = abs(candidate.position_qty)
            actual_position = abs(candidate.participant.account.position.quantity)
            available_qty = min(candidate_available_qty, actual_position)

            liquidated_actual_position = abs(liquidated_agent.account.position.quantity)
            trade_qty = min(available_qty, remaining_qty, liquidated_actual_position)

            if trade_qty <= 0:
                continue

            # 更新账户
            liquidated_agent.account.on_adl_trade(trade_qty, adl_price, is_taker=True)
            candidate.participant.account.on_adl_trade(
                trade_qty, adl_price, is_taker=False
            )

            # 更新候选清单中的 position_qty，确保后续 ADL 不会重复使用
            if candidate.position_qty > 0:
                candidate.position_qty -= trade_qty
            else:
                candidate.position_qty += trade_qty

            # self.logger.info(
            #     f"ADL 成交: Agent {liquidated_agent.agent_id} 与候选者 "
            #     f"成交 {trade_qty} @ {adl_price:.2f}, "
            #     f"候选剩余持仓 {candidate.position_qty}"
            # )

            remaining_qty -= trade_qty

        # if remaining_qty > 0:
        #     self.logger.warning(
        #         f"ADL 未能完全平仓: Agent {liquidated_agent.agent_id} "
        #         f"剩余 {remaining_qty} 无法匹配 "
        #         f"(盈利对手候选数={len(candidates)}, 将由系统兜底清零)"
        #     )

        # 兜底处理：确保 liquidated_agent 的仓位清零
        actual_remaining = abs(liquidated_agent.account.position.quantity)
        if actual_remaining > 0:
            # self.logger.warning(
            #     f"ADL 兜底清零: Agent {liquidated_agent.agent_id} "
            #     f"实际剩余仓位 {liquidated_agent.account.position.quantity}, "
            #     f"强制清零"
            # )
            liquidated_agent.account.position.quantity = 0
            liquidated_agent.account.position.avg_price = 0.0
        if liquidated_agent.account.balance < 0:
            liquidated_agent.account.balance = 0.0

    def _should_end_episode_early(self) -> tuple[str, AgentType | None] | None:
        """检查是否满足提前结束 episode 的条件（O(1) 复杂度）

        触发条件：
        - 任意种群的存活数量少于初始值的 1/4
        - 订单簿只有单边挂单（只有 bid 或只有 ask）

        这确保每个种群都有足够的个体用于 NEAT 进化，以及市场流动性正常。

        Returns:
            tuple[str, AgentType | None] | None: (原因, 种群类型)，如果没有则返回 None
        """
        # 检查种群存活数量
        for agent_type, total in self._pop_total_counts.items():
            if total > 0:
                liquidated = self._pop_liquidated_counts.get(agent_type, 0)
                alive = total - liquidated
                # 任意种群存活少于初始值的 1/4 时触发早停
                if alive < total / 4:
                    return ("population_depleted", agent_type)

        # 检查订单簿单边挂单
        orderbook = self.matching_engine._orderbook
        has_bids = orderbook.get_best_bid() is not None
        has_asks = orderbook.get_best_ask() is not None
        if has_bids != has_asks:  # 只有一边有挂单
            return ("one_sided_orderbook", None)

        return None

    def _log_market_maker_status(self) -> None:
        """调试日志：输出做市商状态，帮助排查单边挂单问题"""
        mm_population = self.populations.get(AgentType.MARKET_MAKER)
        if not mm_population:
            self.logger.warning("没有做市商种群")
            return

        orderbook = self.matching_engine._orderbook
        current_price = orderbook.last_price

        self.logger.warning("=== 做市商状态调试 ===")
        alive_count = 0
        total_long_pos = 0
        total_short_pos = 0
        leverage_full_count = 0

        for agent in mm_population.agents:
            if agent.is_liquidated:
                continue

            alive_count += 1
            equity = agent.account.get_equity(current_price)
            position_qty = agent.account.position.quantity
            position_value = abs(position_qty) * current_price
            max_pos_value = equity * agent.account.leverage

            if position_qty > 0:
                total_long_pos += position_qty
            elif position_qty < 0:
                total_short_pos += abs(position_qty)

            # 检查杠杆上限
            leverage_full = (
                position_value >= max_pos_value * 0.99 if max_pos_value > 0 else False
            )
            if leverage_full:
                leverage_full_count += 1

            # 检查订单状态
            bid_count = len(agent.bid_order_ids)
            ask_count = len(agent.ask_order_ids)

            # 只输出问题做市商（杠杆满或无双边订单）
            if leverage_full or bid_count == 0 or ask_count == 0:
                self.logger.warning(
                    f"  MM {agent.agent_id}: "
                    f"pos={position_qty}, equity={equity:.0f}, "
                    f"pos_value={position_value:.0f}, max_pos={max_pos_value:.0f}, "
                    f"杠杆满={leverage_full}, "
                    f"bid_orders={bid_count}, ask_orders={ask_count}"
                )

        # 统计被强平的做市商
        liquidated_count = sum(1 for a in mm_population.agents if a.is_liquidated)

        self.logger.warning(
            f"存活做市商: {alive_count}/{len(mm_population.agents)}, "
            f"被强平: {liquidated_count}, 杠杆满: {leverage_full_count}"
        )
        self.logger.warning(
            f"多头总仓位: {total_long_pos}, 空头总仓位: {total_short_pos}, "
            f"净仓位: {total_long_pos - total_short_pos}"
        )

        # 输出订单簿状态
        best_bid = orderbook.get_best_bid()
        best_ask = orderbook.get_best_ask()
        bid_volume = (
            sum(pl.total_quantity for pl in orderbook.bids.values())
            if orderbook.bids
            else 0
        )
        ask_volume = (
            sum(pl.total_quantity for pl in orderbook.asks.values())
            if orderbook.asks
            else 0
        )
        self.logger.warning(
            f"订单簿: best_bid={best_bid}, best_ask={best_ask}, "
            f"bid_volume={bid_volume}, ask_volume={ask_volume}, "
            f"last_price={current_price}"
        )

        # 输出其他物种的净仓位
        other_long = 0
        other_short = 0
        for agent_type in [AgentType.RETAIL, AgentType.RETAIL_PRO, AgentType.WHALE]:
            pop = self.populations.get(agent_type)
            if pop:
                for agent in pop.agents:
                    if not agent.is_liquidated:
                        qty = agent.account.position.quantity
                        if qty > 0:
                            other_long += qty
                        elif qty < 0:
                            other_short += abs(qty)
        self.logger.warning(
            f"其他物种仓位: 多头={other_long}, 空头={other_short}, "
            f"净仓位={other_long - other_short}"
        )

    def _compute_normalized_market_state(self) -> NormalizedMarketState:
        """预计算归一化的公共市场数据

        在每个 tick 开始时调用，避免每个 Agent 重复计算相同的归一化数据。
        使用 EMA 平滑后的 mid_price 作为参考价格，减缓价格变化传导速度。

        性能优化：
        - 使用预分配缓冲区减少内存分配
        - 使用 get_depth_numpy() 避免列表推导
        - 使用 Numba JIT 加速对数归一化

        Returns:
            NormalizedMarketState: 归一化后的市场状态
        """
        orderbook = self.matching_engine._orderbook

        # 获取实时参考价格
        current_mid_price = orderbook.get_mid_price()
        if current_mid_price is None:
            current_mid_price = orderbook.last_price
        if current_mid_price == 0:
            current_mid_price = 100.0

        # 更新 EMA 平滑价格，用于 Agent 报价和归一化计算
        smooth_mid_price = self._update_ema_price(current_mid_price)

        tick_size = orderbook.tick_size

        # 使用 get_depth_numpy 直接获取 NumPy 数组
        bid_depth, ask_depth = orderbook.get_depth_numpy(levels=100)

        # 获取并清零缓冲区
        bid_data = self._market_state_buffers['bid_data']
        ask_data = self._market_state_buffers['ask_data']
        trade_prices = self._market_state_buffers['trade_prices']
        trade_quantities = self._market_state_buffers['trade_quantities']
        tick_prices_normalized = self._market_state_buffers['tick_prices']
        tick_volumes_normalized = self._market_state_buffers['tick_volumes']
        tick_amounts_normalized = self._market_state_buffers['tick_amounts']

        bid_data.fill(0)
        ask_data.fill(0)
        trade_prices.fill(0)
        trade_quantities.fill(0)
        tick_prices_normalized.fill(0)
        tick_volumes_normalized.fill(0)
        tick_amounts_normalized.fill(0)

        # 向量化买盘：100档 × 2 = 200（使用平滑价格归一化）
        # bid_depth shape: (100, 2), 列0=价格, 列1=数量
        bid_prices = bid_depth[:, 0]
        bid_qtys = bid_depth[:, 1]
        # 找到有效档位数（价格 > 0）
        bid_valid_mask = bid_prices > 0
        n_bids = int(np.sum(bid_valid_mask))
        if n_bids > 0 and smooth_mid_price > 0:
            bid_data[0 : n_bids * 2 : 2] = (
                bid_prices[:n_bids] - smooth_mid_price
            ) / smooth_mid_price
            # 数量使用对数归一化
            bid_data[1 : n_bids * 2 : 2] = log_normalize_unsigned(bid_qtys[:n_bids])

        # 向量化卖盘：100档 × 2 = 200（使用平滑价格归一化）
        ask_prices = ask_depth[:, 0]
        ask_qtys = ask_depth[:, 1]
        ask_valid_mask = ask_prices > 0
        n_asks = int(np.sum(ask_valid_mask))
        if n_asks > 0 and smooth_mid_price > 0:
            ask_data[0 : n_asks * 2 : 2] = (
                ask_prices[:n_asks] - smooth_mid_price
            ) / smooth_mid_price
            # 数量使用对数归一化
            ask_data[1 : n_asks * 2 : 2] = log_normalize_unsigned(ask_qtys[:n_asks])

        # 向量化成交：100笔（数量带方向：正=taker买入，负=taker卖出）
        trades = list(self.recent_trades)  # deque 转换为 list
        if trades:
            n_trades = len(trades)
            # 直接构建数组，避免列表推导
            prices_arr = np.empty(n_trades, dtype=np.float32)
            qtys_arr = np.empty(n_trades, dtype=np.float32)
            for i, t in enumerate(trades):
                prices_arr[i] = t.price
                qtys_arr[i] = t.quantity if t.is_buyer_taker else -t.quantity

            if smooth_mid_price > 0:
                trade_prices[:n_trades] = (prices_arr - smooth_mid_price) / smooth_mid_price
            # 成交数量带方向的对数归一化
            trade_quantities[:n_trades] = log_normalize_signed(qtys_arr)

        # Tick 历史价格归一化（以第一个 tick 价格为基准）
        if self._tick_history_prices:
            hist_prices = np.array(self._tick_history_prices[-100:], dtype=np.float32)
            volumes = np.array(self._tick_history_volumes[-100:], dtype=np.float32)
            amounts = np.array(self._tick_history_amounts[-100:], dtype=np.float32)
            n = len(hist_prices)

            # 价格归一化：以第一个 tick 价格为基准
            base_price = hist_prices[0]
            if base_price > 0:
                tick_prices_normalized[-n:] = (hist_prices - base_price) / base_price

            # 成交量归一化
            tick_volumes_normalized[-n:] = log_normalize_signed(volumes)

            # 成交额归一化（scale=12）
            tick_amounts_normalized[-n:] = log_normalize_signed(amounts, scale=12.0)

        return NormalizedMarketState(
            mid_price=smooth_mid_price,  # 使用 EMA 平滑后的价格
            tick_size=tick_size,
            bid_data=bid_data.copy(),
            ask_data=ask_data.copy(),
            trade_prices=trade_prices.copy(),
            trade_quantities=trade_quantities.copy(),
            tick_history_prices=tick_prices_normalized.copy(),
            tick_history_volumes=tick_volumes_normalized.copy(),
            tick_history_amounts=tick_amounts_normalized.copy(),
        )

    def _evolve_populations_parallel(self, current_price: float) -> None:
        """进化所有种群

        当前实现使用串行进化。
        子种群并行进化方案因 Python GIL 和序列化开销限制而无法提供性能提升。
        保留方法名以保持 API 兼容性。
        """
        # [MEMORY] 记录进化开始前的内存
        mem_before_evolve = _get_memory_mb()

        # 串行进化所有种群
        for pop in self.populations.values():
            try:
                pop.evolve(current_price)
            except Exception as e:
                self.logger.error(f"种群 {pop.agent_type.value} 进化失败: {e}")
                raise

        # [MEMORY] 记录进化后、最终 GC 前的内存
        mem_after_evolve = _get_memory_mb()

        # 所有种群进化完成后，更新网络数据缓存
        self._update_network_caches()

        # 最终全面 GC，确保所有临时对象被回收
        # 包括 Python 2 代和 3 代垃圾
        gc.collect(0)  # 年轻代
        gc.collect(1)  # 中年代
        gc.collect(2)  # 老年代

        # 【关键】调用 malloc_trim 将释放的内存归还给操作系统
        # Python 的内存分配器通常不会主动归还内存，导致 VmRSS 持续增长
        malloc_trim()

        # [MEMORY] 记录 GC 后的内存
        mem_after_gc = _get_memory_mb()

        # 输出内存变化统计
        arena_tag = f"Arena-{self.arena_id}" if self.arena_id is not None else "Trainer"
        self.logger.info(
            f"[MEMORY_EVOLVE_PARALLEL] {arena_tag} ep_{self.episode}: "
            f"evolve={mem_after_evolve - mem_before_evolve:+.1f}MB, "
            f"gc_released={mem_after_evolve - mem_after_gc:.1f}MB, "
            f"net={mem_after_gc - mem_before_evolve:+.1f}MB"
        )

    def _evolve_populations_with_cached_fitness(self) -> None:
        """使用缓存适应度进化所有种群

        当 tick 数不足时调用，跳过适应度计算，使用之前 episode 缓存的适应度进行进化。
        这样可以打破"tick 不足 → 不进化 → 行为不变 → tick 不足"的死循环。
        """
        for agent_type, population in self.populations.items():
            try:
                success = population.evolve_with_cached_fitness()
                if success:
                    self.logger.debug(
                        f"{agent_type.value} 种群使用缓存适应度进化成功"
                    )
                else:
                    self.logger.warning(
                        f"{agent_type.value} 种群没有缓存适应度，跳过本次进化"
                    )
            except Exception as e:
                self.logger.error(
                    f"种群 {agent_type.value} 缓存进化失败: {e}"
                )

            # 每个种群进化后立即 GC
            gc.collect()

        # 所有种群进化完成后，更新网络数据缓存
        self._update_network_caches()

        # 最终全面 GC
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        malloc_trim()

    def _batch_decide_serial(
        self,
        agents: list["Agent"],
        market_state: "NormalizedMarketStateType",
        orderbook: "OrderBook",
    ) -> list[tuple["Agent", "ActionType", dict[str, Any]]]:
        """高效串行决策 - 单线程处理所有 Agent

        经测试，由于 Python GIL 限制，多线程并行决策反而比单线程慢。
        单线程串行处理避免了线程调度开销和 GIL 竞争。

        Args:
            agents: Agent 列表
            market_state: 归一化市场状态
            orderbook: 订单簿

        Returns:
            决策结果列表：[(agent, action, params), ...]
        """
        results: list[tuple["Agent", "ActionType", dict[str, Any]]] = []

        for agent in agents:
            # 跳过已淘汰的 Agent
            if agent.is_liquidated:
                continue
            try:
                action, params = agent.decide(market_state, orderbook)
                results.append((agent, action, params))
            except Exception:
                # 忽略错误，继续处理其他 Agent
                pass

        return results

    def _batch_decide_openmp(
        self,
        agents: list["Agent"],
        market_state: "NormalizedMarketStateType",
        orderbook: "OrderBook",
    ) -> list[tuple["Agent", "ActionType", dict[str, Any]]]:
        """OpenMP 多核并行决策

        使用 OpenMP prange 实现真正的多核 CPU 并行神经网络计算。
        流程：
        1. 按 Agent 类型分组（不同类型有不同输入输出维度）
        2. 串行：收集所有 Agent 的输入向量（observe）
        3. 并行：按类型批量执行神经网络前向传播（OpenMP prange）
        4. 串行：解析输出并返回决策结果

        Args:
            agents: Agent 列表
            market_state: 归一化市场状态
            orderbook: 订单簿

        Returns:
            决策结果列表：[(agent, action, params), ...]
        """
        # 过滤掉已淘汰的 Agent
        active_agents = [a for a in agents if not a.is_liquidated]
        n = len(active_agents)

        if n == 0:
            return []

        # ======= 缓存路径：优先使用缓存版本 =======
        if self._cache_initialized and self._network_caches:
            return self._batch_decide_with_cache(active_agents, market_state)

        # ======= 原有逻辑：非缓存路径 =======
        # 尝试导入 batch_activate_parallel
        try:
            from neat._cython.fast_network import batch_activate_parallel
        except ImportError:
            # 回退到串行版本
            return self._batch_decide_serial(agents, market_state, orderbook)

        # 按 Agent 类型分组（不同类型有不同的输入输出维度）
        type_groups: dict[AgentType, list[tuple[int, "Agent"]]] = {}
        for idx, agent in enumerate(active_agents):
            agent_type = agent.agent_type
            if agent_type not in type_groups:
                type_groups[agent_type] = []
            type_groups[agent_type].append((idx, agent))

        # 为每个类型准备批量数据并处理
        all_results: list[tuple[int, "Agent", "ActionType", dict[str, Any]]] = []

        for agent_type, group in type_groups.items():
            if not group:
                continue

            # 收集输入和网络
            inputs_list: list[np.ndarray] = []
            networks_list: list = []
            valid_items: list[tuple[int, "Agent"]] = []

            for idx, agent in group:
                try:
                    input_vec = agent.observe(market_state, orderbook)
                    network = agent.brain.network
                    inputs_list.append(input_vec)
                    networks_list.append(network)
                    valid_items.append((idx, agent))
                except Exception:
                    pass

            if not valid_items:
                continue

            # 获取该类型的输入输出维度
            num_valid = len(valid_items)
            num_inputs = networks_list[0].num_inputs
            num_outputs = networks_list[0].num_outputs

            # 准备批量数组
            inputs_batch = np.zeros((num_valid, num_inputs), dtype=np.float64)
            outputs_batch = np.zeros((num_valid, num_outputs), dtype=np.float64)

            for i, inp in enumerate(inputs_list):
                inputs_batch[i, :len(inp)] = inp

            # OpenMP 并行执行神经网络前向传播
            batch_activate_parallel(networks_list, inputs_batch, outputs_batch)

            # 解析输出（不同类型使用不同的解析逻辑）
            for i, (idx, agent) in enumerate(valid_items):
                try:
                    output = outputs_batch[i]
                    action, params = self._parse_agent_output(
                        agent, output, market_state, orderbook
                    )
                    all_results.append((idx, agent, action, params))
                except Exception:
                    pass

        # 按原始顺序排序结果
        all_results.sort(key=lambda x: x[0])

        return [(agent, action, params) for _, agent, action, params in all_results]

    def _parse_agent_output(
        self,
        agent: "Agent",
        output: np.ndarray,
        market_state: "NormalizedMarketStateType",
        orderbook: "OrderBook",
    ) -> tuple["ActionType", dict[str, Any]]:
        """解析 Agent 的神经网络输出为动作

        根据 Agent 类型使用不同的解析逻辑。

        Args:
            agent: Agent 对象
            output: 神经网络输出数组
            market_state: 归一化市场状态
            orderbook: 订单簿

        Returns:
            (动作类型, 参数字典)
        """
        from src.bio.agents.base import fast_argmax, fast_round_price, fast_clip

        mid_price = market_state.mid_price
        if mid_price == 0:
            mid_price = 100.0
        tick_size = market_state.tick_size if market_state.tick_size > 0 else 0.1

        agent_type = agent.agent_type

        if agent_type == AgentType.MARKET_MAKER:
            # 做市商：21 个输出，直接返回 QUOTE 动作
            return self._parse_market_maker_output(
                agent, output, mid_price, tick_size
            )
        else:
            # 散户/高级散户/庄家：9 个输出
            return self._parse_retail_output(
                agent, output, mid_price, tick_size, agent_type
            )

    def _parse_retail_output(
        self,
        agent: "Agent",
        output: np.ndarray,
        mid_price: float,
        tick_size: float,
        agent_type: AgentType,
    ) -> tuple["ActionType", dict[str, Any]]:
        """解析散户/高级散户/庄家的神经网络输出"""
        from src.bio.agents.base import ActionType, fast_argmax, fast_round_price, fast_clip

        # 解析动作类型
        num_actions = 7 if agent_type == AgentType.WHALE else 6
        action_idx = fast_argmax(output, 0, num_actions)
        action = ActionType(action_idx)

        # 解析参数
        price_offset_norm = fast_clip(output[7], -1.0, 1.0)
        quantity_ratio_norm = fast_clip(output[8], -1.0, 1.0)
        quantity_ratio = (quantity_ratio_norm + 1) * 0.5

        params: dict[str, Any] = {}

        if action == ActionType.PLACE_BID:
            price_offset_ticks = price_offset_norm * 100
            raw_price = mid_price + price_offset_ticks * tick_size
            params["price"] = fast_round_price(raw_price, tick_size)
            params["quantity"] = agent._calculate_order_quantity(
                mid_price, quantity_ratio, is_buy=True
            )
        elif action == ActionType.PLACE_ASK:
            price_offset_ticks = price_offset_norm * 100
            raw_price = mid_price + price_offset_ticks * tick_size
            params["price"] = fast_round_price(raw_price, tick_size)
            params["quantity"] = agent._calculate_order_quantity(
                mid_price, quantity_ratio, is_buy=False
            )
        elif action == ActionType.MARKET_BUY:
            params["quantity"] = agent._calculate_order_quantity(
                mid_price, quantity_ratio, is_buy=True
            )
        elif action == ActionType.MARKET_SELL:
            position_qty = agent.account.position.quantity
            if position_qty > 0:
                params["quantity"] = max(1, int(position_qty * quantity_ratio))
            else:
                params["quantity"] = agent._calculate_order_quantity(
                    mid_price, quantity_ratio, is_buy=False
                )

        return action, params

    def _parse_market_maker_output(
        self,
        agent: "Agent",
        output: np.ndarray,
        mid_price: float,
        tick_size: float,
    ) -> tuple["ActionType", dict[str, Any]]:
        """解析做市商的神经网络输出"""
        from src.bio.agents.base import ActionType, fast_round_price, fast_clip

        # 做市商输出：21 个值
        # [0-4] 买单价格偏移, [5-9] 买单数量权重
        # [10-14] 卖单价格偏移, [15-19] 卖单数量权重
        # [20] 总下单比例基准

        bid_price_offsets = output[0:5]
        bid_qty_weights = output[5:10]
        ask_price_offsets = output[10:15]
        ask_qty_weights = output[15:20]
        total_ratio_raw = output[20] if len(output) > 20 else 0.0

        # 映射总下单比例
        total_ratio = (fast_clip(total_ratio_raw, -1.0, 1.0) + 1) * 0.5

        # 计算倾斜因子
        skew_factor = agent._calculate_skew_factor(mid_price)

        # 应用倾斜到权重
        bid_weights_sum = sum(max(0, (w + 1) * 0.5) for w in bid_qty_weights)
        ask_weights_sum = sum(max(0, (w + 1) * 0.5) for w in ask_qty_weights)
        total_weights = bid_weights_sum + ask_weights_sum

        if total_weights > 0:
            bid_multiplier = 1.0 + skew_factor
            ask_multiplier = 1.0 - skew_factor
            bid_weights_sum *= bid_multiplier
            ask_weights_sum *= ask_multiplier
            total_weights = bid_weights_sum + ask_weights_sum

        # 构建订单列表（使用字典格式以匹配 _place_quote_orders 期望的输入）
        bid_orders: list[dict[str, float]] = []
        ask_orders: list[dict[str, float]] = []

        for i in range(5):
            # 买单
            offset = fast_clip(bid_price_offsets[i], -1.0, 1.0)
            ticks = 1 + (offset + 1) * 49.5  # 1-100 ticks
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

            # 卖单
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

        return ActionType.QUOTE, {"bid_orders": bid_orders, "ask_orders": ask_orders}

    def _batch_decide_parallel(
        self,
        agents: list["Agent"],
        market_state: "NormalizedMarketStateType",
        orderbook: "OrderBook",
        timeout: float = 60.0,
    ) -> list[tuple["Agent", "ActionType", dict[str, Any]]]:
        """并行决策

        使用 OpenMP 多核并行执行神经网络前向传播。
        如果 OpenMP 版本不可用，回退到串行版本。

        Args:
            agents: Agent 列表
            market_state: 归一化市场状态
            orderbook: 订单簿
            timeout: 超时时间（秒），未使用

        Returns:
            决策结果列表：[(agent, action, params), ...]
        """
        # 尝试使用 OpenMP 并行版本
        return self._batch_decide_openmp(agents, market_state, orderbook)

    def _batch_decide_with_cache(
        self,
        active_agents: list["Agent"],
        market_state: "NormalizedMarketStateType",
    ) -> list[tuple["Agent", "ActionType", dict[str, Any]]]:
        """使用缓存执行批量决策

        按 Agent 类型分组，使用对应的 BatchNetworkCache 执行决策。

        Args:
            active_agents: 活跃的 Agent 列表（已过滤淘汰的）
            market_state: 归一化市场状态

        Returns:
            决策结果列表：[(agent, action, params), ...]
        """
        from src.bio.agents.base import ActionType

        # 按 Agent 类型分组（保持原始索引）
        type_groups: dict[AgentType, list[tuple[int, "Agent"]]] = {}
        for idx, agent in enumerate(active_agents):
            agent_type = agent.agent_type
            if agent_type not in type_groups:
                type_groups[agent_type] = []
            type_groups[agent_type].append((idx, agent))

        all_results: list[tuple[int, "Agent", "ActionType", dict[str, Any]]] = []
        mid_price = market_state.mid_price

        for agent_type, group in type_groups.items():
            cache = self._network_caches.get(agent_type)  # type: ignore
            if cache is None or not cache.is_valid():
                # 回退到非缓存版本（单独处理这组）
                self.logger.warning(
                    f"缓存无效或不存在 {agent_type.value}，跳过"
                )
                continue

            # 收集该类型的所有 Agent
            group_agents = [agent for _, agent in group]

            # 使用缓存执行决策
            try:
                raw_results = cache.decide(group_agents, market_state)
            except Exception as e:
                self.logger.warning(f"缓存决策失败 {agent_type.value}: {e}")
                continue

            # 转换结果
            is_whale = (agent_type == AgentType.WHALE)
            is_market_maker = (agent_type == AgentType.MARKET_MAKER)
            tick_size = market_state.tick_size if market_state.tick_size > 0 else 0.1

            for i, (idx, agent) in enumerate(group):
                try:
                    if is_market_maker:
                        # 做市商返回 (nn_output, mid_price, tick_size) 元组
                        nn_output, _, _ = raw_results[i]
                        action, params = self._parse_market_maker_output(
                            agent, nn_output, mid_price, tick_size
                        )
                    else:
                        # 散户/高级散户/庄家：解析 Cython 返回的结果
                        action_type_int, side_int, price, quantity = raw_results[i]
                        action, params = self._convert_retail_result(
                            agent, action_type_int, side_int, price, quantity,
                            mid_price, is_whale=is_whale
                        )
                    all_results.append((idx, agent, action, params))
                except Exception:
                    pass

        # 按原始顺序排序
        all_results.sort(key=lambda x: x[0])
        return [(agent, action, params) for _, agent, action, params in all_results]

    def _convert_retail_result(
        self,
        agent: "Agent",
        action_type_int: int,
        side_int: int,
        price: float,
        quantity: int,
        mid_price: float,
        is_whale: bool = False,
    ) -> tuple["ActionType", dict[str, Any]]:
        """将 Cython 返回的散户/庄家结果转换为 ActionType 和 params

        Args:
            agent: Agent 对象
            action_type_int: 动作类型整数（与 Cython batch_decide_openmp.pyx 一致）
                - 0: HOLD
                - 1: PLACE_BID
                - 2: PLACE_ASK
                - 3: CANCEL
                - 4: MARKET_BUY
                - 5: MARKET_SELL
                - 6: CLEAR_POSITION（仅庄家）
            side_int: 方向整数（1=买, 2=卖）
            price: 价格
            quantity: 数量（实际上是比例值，需要转换）
            mid_price: 中间价
            is_whale: 是否为庄家

        Returns:
            (动作类型, 参数字典)
        """
        from src.bio.agents.base import ActionType

        params: dict[str, Any] = {}

        # 动作类型映射（与 Cython 的 ACTION_* 常量一致）
        if action_type_int == 0:
            # HOLD - 不动
            action = ActionType.HOLD
        elif action_type_int == 1:
            # PLACE_BID - 挂买单
            action = ActionType.PLACE_BID
            params["price"] = price
            # quantity 是比例值（0-1），需要转换为实际数量
            actual_qty = agent._calculate_order_quantity(
                price, quantity, is_buy=True
            )
            params["quantity"] = actual_qty
        elif action_type_int == 2:
            # PLACE_ASK - 挂卖单
            action = ActionType.PLACE_ASK
            params["price"] = price
            actual_qty = agent._calculate_order_quantity(
                price, quantity, is_buy=False
            )
            params["quantity"] = actual_qty
        elif action_type_int == 3:
            # CANCEL - 撤单
            action = ActionType.CANCEL
        elif action_type_int == 4:
            # MARKET_BUY - 市价买入
            action = ActionType.MARKET_BUY
            actual_qty = agent._calculate_order_quantity(
                mid_price, quantity, is_buy=True
            )
            params["quantity"] = actual_qty
        elif action_type_int == 5:
            # MARKET_SELL - 市价卖出
            action = ActionType.MARKET_SELL
            # 对于市价卖出，需要考虑持仓情况
            position_qty = agent.account.position.quantity
            if position_qty > 0:
                # 有多仓时卖出
                sell_qty = max(1, int(position_qty * quantity))
                params["quantity"] = min(sell_qty, int(position_qty))
            else:
                # 空仓或空头，开空仓
                actual_qty = agent._calculate_order_quantity(
                    mid_price, quantity, is_buy=False
                )
                params["quantity"] = actual_qty
        elif action_type_int == 6 and is_whale:
            # CLEAR_POSITION - 清仓（仅庄家）
            action = ActionType.CLEAR_POSITION
        else:
            # 默认不动
            action = ActionType.HOLD

        return action, params

    def _check_liquidations_vectorized(self, current_price: float) -> list["Agent"]:
        """向量化检查所有 Agent 的强平条件"""
        active_agents = [a for a in self.agent_execution_order if not a.is_liquidated]
        n = len(active_agents)

        if n == 0:
            return []

        balances = np.array(
            [a.account.balance for a in active_agents], dtype=np.float64
        )
        quantities = np.array(
            [a.account.position.quantity for a in active_agents], dtype=np.float64
        )
        avg_prices = np.array(
            [a.account.position.avg_price for a in active_agents], dtype=np.float64
        )
        maintenance_rates = np.array(
            [a.account.maintenance_margin_rate for a in active_agents], dtype=np.float64
        )

        unrealized_pnl = (current_price - avg_prices) * quantities
        equities = balances + unrealized_pnl
        position_values = np.abs(quantities) * current_price

        # 使用 np.divide 的 where 参数避免除零警告
        margin_ratios = np.divide(
            equities,
            position_values,
            out=np.full_like(equities, np.inf),
            where=position_values > 0,
        )

        need_liquidation = margin_ratios < maintenance_rates
        return [active_agents[i] for i in range(n) if need_liquidation[i]]

    def _init_market(self) -> None:
        """初始化市场（直接调用模式）

        只有做市商先行动，建立初始流动性。

        每个做市商按顺序行动，后一个做市商看到的是前一个做市商
        下单后的市场状态。
        """
        if not self.matching_engine:
            return

        orderbook = self.matching_engine._orderbook

        mm_population = self.populations.get(AgentType.MARKET_MAKER)
        if mm_population:
            for agent in mm_population.agents:
                # 每个做市商决策前重新计算市场状态，确保看到最新的订单簿
                market_state = self._compute_normalized_market_state()
                action, params = agent.decide(market_state, orderbook)
                # 直接执行（绕过事件系统）
                trades = agent.execute_action(action, params, self.matching_engine)
                for trade in trades:
                    self.recent_trades.append(trade)
                    # 更新 maker 的账户（即使自成交也需要更新）
                    maker_id = (
                        trade.seller_id if trade.is_buyer_taker else trade.buyer_id
                    )
                    maker_agent = self.agent_map.get(maker_id)
                    if maker_agent is not None:
                        # maker 的方向与 taker 相反
                        is_buyer = not trade.is_buyer_taker
                        maker_agent.account.on_trade(trade, is_buyer)

                # 市场初始化阶段不检查淘汰（此时价格还未稳定）

    def _reset_market(self) -> None:
        """重置市场状态

        清空订单簿，做市商重新建立流动性。
        """
        if not self.matching_engine:
            return

        # 清空订单簿并重置最新价
        self.matching_engine._orderbook.clear(
            reset_price=self.config.market.initial_price
        )

        # 清空最近成交
        self.recent_trades.clear()

        # 重置价格历史
        self._price_history.clear()
        self._price_history.append(self.config.market.initial_price)

        # 重置 tick 历史数据
        self._tick_history_prices.clear()
        self._tick_history_volumes.clear()
        self._tick_history_amounts.clear()
        self._tick_history_prices.append(self.config.market.initial_price)
        self._tick_history_volumes.append(0.0)
        self._tick_history_amounts.append(0.0)

        # 重置 EMA 平滑价格
        self._init_ema_price(self.config.market.initial_price)

        # 做市商重新初始化
        self._init_market()

        # 调试：检查初始化后的仓位平衡
        total_position = 0
        long_qty = 0
        short_qty = 0
        for population in self.populations.values():
            for agent in population.agents:
                qty = agent.account.position.quantity
                total_position += qty
                if qty > 0:
                    long_qty += qty
                elif qty < 0:
                    short_qty += qty
        if total_position != 0:
            self.logger.error(
                f"初始化后仓位不对等！总偏差={total_position}, 多头={long_qty}, 空头={short_qty}"
            )

        # 初始化 tick_start_price，供数据采集使用
        self.tick_start_price = self.matching_engine._orderbook.last_price

    def run_tick(self) -> None:
        """执行一个 tick（直接调用模式）

        时序设计：
        1. Tick 1：只展示做市商初始挂单后的市场状态，其他 agent 不行动
        2. Tick 2+：
           - Tick 开始：用 smooth_mid_price 检查所有 agent 的强平条件（爆仓即淘汰）
           - Tick 过程：Agent 按顺序决策和下单
           - Tick 结束：下单效果（价格变动）在下个 tick 被感知

        这样设计确保：
        - UI 模式下第一个可见 tick 是做市商初始挂单后的状态
        - 强平检查和 Agent 报价使用同一价格基准（smooth_mid_price）
        - 所有 agent 在同一价格基础上被检查，公平
        """
        if not self.matching_engine:
            return

        self.tick += 1
        tick_trades: list[Trade] = []  # 收集本 tick 所有成交
        orderbook = self.matching_engine._orderbook

        # 使用 smooth_mid_price 作为强平检查的价格依据，与 Agent 报价逻辑一致
        # 注意：smooth_mid_price 在 _compute_normalized_market_state() 中更新
        # 这里先获取当前的平滑价格用于强平检查
        current_price = (
            self._smooth_mid_price
            if self._smooth_mid_price > 0
            else orderbook.last_price
        )
        self.tick_start_price = current_price  # 保存 tick 开始时的价格

        # === Tick 1：只记录做市商初始挂单后的状态，不执行 agent 行动 ===
        if self.tick == 1:
            # 只记录价格历史和 tick 数据
            self._price_history.append(current_price)
            self._tick_history_prices.append(current_price)
            self._tick_history_volumes.append(0.0)
            self._tick_history_amounts.append(0.0)
            # 更新 episode 价格统计
            if current_price > self._episode_high_price:
                self._episode_high_price = current_price
            if current_price < self._episode_low_price:
                self._episode_low_price = current_price
            return

        # === Tick 2+：正常执行所有 agent 行动 ===
        # === Tick 开始：检查所有 agent 的强平条件（爆仓即淘汰）===
        # 阶段1：向量化收集需要淘汰的 agent，统一撤销挂单
        agents_to_liquidate = self._check_liquidations_vectorized(current_price)
        agents_to_liquidate_ids: set[int] = set()
        for agent in agents_to_liquidate:
            agents_to_liquidate_ids.add(agent.agent_id)
            # 立即撤销挂单，防止在后续平仓过程中作为 maker 被成交
            self._cancel_agent_orders(agent)

        # === 三阶段强平处理 ===
        # 阶段2：统一执行市价单平仓，收集需要 ADL 的 Agent
        agents_need_adl: list[tuple["Agent", int, bool]] = (
            []
        )  # (agent, remaining_qty, is_long)
        if len(agents_to_liquidate) > 0:
            for agent in agents_to_liquidate:
                remaining_qty, is_long = self._execute_liquidation_market_order(agent)
                if remaining_qty > 0:
                    agents_need_adl.append((agent, remaining_qty, is_long))
                # 标记淘汰
                agent.is_liquidated = True
                self._pop_liquidated_counts[agent.agent_type] = (
                    self._pop_liquidated_counts.get(agent.agent_type, 0) + 1
                )
                # 穿仓兜底
                if agent.account.balance < 0:
                    agent.account.balance = 0.0

            # 阶段3：用最新价格计算 ADL 候选并执行
            if agents_need_adl:
                # 获取订单簿最新价格（强平市价单执行后的价格）
                latest_price = orderbook.last_price

                # 按方向计算 ADL 候选清单
                self._adl_long_candidates = []
                self._adl_short_candidates = []
                for agent in self.agent_execution_order:
                    if agent.is_liquidated:
                        continue
                    if agent.agent_id in agents_to_liquidate_ids:
                        continue

                    # 用最新价格计算 ADL 候选信息
                    candidate = self.adl_manager.calculate_adl_score(
                        agent, latest_price
                    )
                    if candidate is None:
                        continue

                    # 只有盈利的 Agent 才能作为 ADL 对手方
                    if candidate.pnl_percent <= 0:
                        continue

                    # 按持仓方向分类
                    if candidate.position_qty > 0:
                        self._adl_long_candidates.append(candidate)
                    else:
                        self._adl_short_candidates.append(candidate)

                # 将鲶鱼加入 ADL 候选列表
                for catfish in self.catfish_list:
                    if catfish.is_liquidated:
                        continue

                    position_qty = catfish.account.position.quantity
                    if position_qty == 0:
                        continue

                    # 计算 ADL 分数
                    equity = catfish.account.get_equity(latest_price)
                    pnl_percent = (
                        equity - catfish.account.initial_balance
                    ) / catfish.account.initial_balance

                    # 只有盈利的才能作为 ADL 候选
                    if pnl_percent <= 0:
                        continue

                    position_value = abs(position_qty) * latest_price
                    effective_leverage = position_value / equity if equity > 0 else 0.0
                    adl_score = pnl_percent * effective_leverage

                    candidate = ADLCandidate(
                        participant=catfish,
                        position_qty=position_qty,
                        pnl_percent=pnl_percent,
                        effective_leverage=effective_leverage,
                        adl_score=adl_score,
                    )

                    if position_qty > 0:
                        self._adl_long_candidates.append(candidate)
                    else:
                        self._adl_short_candidates.append(candidate)

                # 按 ADL 分数从高到低排序
                self._adl_long_candidates.sort(key=lambda c: c.adl_score, reverse=True)
                self._adl_short_candidates.sort(key=lambda c: c.adl_score, reverse=True)

                # 执行 ADL
                for agent, remaining_qty, is_long in agents_need_adl:
                    self._execute_adl(agent, remaining_qty, latest_price, is_long)
                    # 兜底处理：确保仓位清零
                    if agent.account.position.quantity != 0:
                        self.logger.warning(
                            f"ADL 后仓位未清零: Agent {agent.agent_id}, "
                            f"pos={agent.account.position.quantity}, 强制清零"
                        )
                        agent.account.position.quantity = 0
                        agent.account.position.avg_price = 0.0

        # === 鲶鱼行动（在 Agent 之前）===
        for catfish in self.catfish_list:
            # 检查鲶鱼是否已被强平
            if catfish.is_liquidated:
                continue

            should_act, direction = catfish.decide(
                orderbook,
                self.tick,
                self._price_history,
            )
            if should_act and direction != 0:
                catfish_trades = catfish.execute(direction, self.matching_engine)
                catfish.record_action(self.tick)

                for trade in catfish_trades:
                    self.recent_trades.append(trade)
                    tick_trades.append(trade)  # 收集到本 tick 成交

                    # 更新鲶鱼账户（taker）
                    is_buyer = trade.is_buyer_taker
                    catfish.account.on_trade(trade, is_buyer)

                    # 更新 maker 账户（与鲶鱼成交的对手方）
                    maker_id = (
                        trade.seller_id if trade.is_buyer_taker else trade.buyer_id
                    )
                    if maker_id > 0:
                        maker_agent = self.agent_map.get(maker_id)
                        if maker_agent is not None:
                            maker_is_buyer = not trade.is_buyer_taker
                            maker_agent.account.on_trade(trade, maker_is_buyer)

        # 预计算归一化市场数据
        market_state = self._compute_normalized_market_state()

        # === 随机打乱执行顺序（每个 tick 都不同，模拟真实环境）===
        random.shuffle(self.agent_execution_order)

        # === 并行决策阶段 ===
        decisions = self._batch_decide_parallel(
            self.agent_execution_order, market_state, orderbook
        )

        # === 串行执行阶段 ===
        for agent, action, params in decisions:
            # 庄家下单前记录价格（用于计算波动性贡献）
            pre_trade_price = (
                orderbook.last_price
                if agent.agent_type == AgentType.WHALE else 0.0
            )

            trades = agent.execute_action(action, params, self.matching_engine)

            # 庄家成交后计算波动性贡献
            if agent.agent_type == AgentType.WHALE and trades:
                post_trade_price = orderbook.last_price
                if pre_trade_price > 0 and post_trade_price > 0:
                    price_impact = abs(post_trade_price - pre_trade_price) / pre_trade_price
                    agent.account.volatility_contribution += price_impact

            for trade in trades:
                self.recent_trades.append(trade)
                tick_trades.append(trade)  # 收集到本 tick 成交
                maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
                maker_agent = self.agent_map.get(maker_id)
                if maker_agent is not None:
                    is_buyer = not trade.is_buyer_taker
                    maker_agent.account.on_trade(trade, is_buyer)

        # 记录价格历史（tick 结束时）
        # 使用 last_price 而非 mid_price，确保价格符合 tick_size
        current_price = orderbook.last_price
        self._price_history.append(current_price)

        # 更新 episode 价格统计
        if current_price > self._episode_high_price:
            self._episode_high_price = current_price
        if current_price < self._episode_low_price:
            self._episode_low_price = current_price
        # 限制历史长度（避免内存无限增长）
        if len(self._price_history) > 1000:
            self._price_history = self._price_history[-1000:]

        # 记录 tick 历史数据
        self._tick_history_prices.append(current_price)
        volume, amount = self._aggregate_tick_trades(tick_trades)
        self._tick_history_volumes.append(volume)
        self._tick_history_amounts.append(amount)
        # 限制历史长度（最多 100 条）
        if len(self._tick_history_prices) > 100:
            self._tick_history_prices = self._tick_history_prices[-100:]
            self._tick_history_volumes = self._tick_history_volumes[-100:]
            self._tick_history_amounts = self._tick_history_amounts[-100:]

        # === 鲶鱼强平检查（放在所有 Agent 之后）===
        self._check_catfish_liquidation(current_price)

    def _check_catfish_liquidation(self, current_price: float) -> None:
        """检查鲶鱼强平

        鲶鱼强平后本轮 episode 立即结束，进入进化阶段。

        Args:
            current_price: 当前价格
        """
        for catfish in self.catfish_list:
            if catfish.is_liquidated:
                continue

            if catfish.account.check_liquidation(current_price):
                # 执行鲶鱼强平
                self._execute_catfish_liquidation(catfish, current_price)
                catfish.is_liquidated = True

                # 穿仓兜底
                if catfish.account.balance < 0:
                    catfish.account.balance = 0.0

                # 标记鲶鱼强平（调用方决定是否结束 episode）
                self._catfish_liquidated = True
                self.logger.info(
                    f"鲶鱼 {catfish.__class__.__name__} 被强平 "
                    f"(episode={self.episode}, tick={self.tick})"
                )
                break  # 一条鲶鱼强平就停止检查

    def _execute_catfish_liquidation(
        self, catfish: "CatfishBase", current_price: float
    ) -> None:
        """执行鲶鱼强平

        Args:
            catfish: 被强平的鲶鱼
            current_price: 当前价格
        """
        position_qty = catfish.account.position.quantity
        if position_qty == 0:
            return

        is_long = position_qty > 0
        target_qty = abs(position_qty)

        # 创建市价平仓单
        side = OrderSide.SELL if is_long else OrderSide.BUY
        order = Order(
            order_id=catfish._generate_order_id(),
            agent_id=catfish.catfish_id,
            side=side,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=target_qty,
        )

        # 撮合
        trades = self.matching_engine.match_market_order(order)

        # 更新账户
        for trade in trades:
            is_buyer = trade.is_buyer_taker
            catfish.account.on_trade(trade, is_buyer)
            self.recent_trades.append(trade)

            # 更新 maker 账户
            maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
            if maker_id > 0:
                maker_agent = self.agent_map.get(maker_id)
                if maker_agent is not None:
                    maker_is_buyer = not trade.is_buyer_taker
                    maker_agent.account.on_trade(trade, maker_is_buyer)

    def run_episode(self) -> None:
        """运行一个 episode

        1. 重置所有 Agent 账户
        2. 运行 episode_length 个 tick
        3. 评估适应度并进化
        """
        if not self.matching_engine:
            return

        self.episode += 1
        episode_length = self.config.training.episode_length

        # 1. 重置所有种群的 Agent 账户
        for population in self.populations.values():
            population.reset_agents()

        # 重置鲶鱼
        for catfish in self.catfish_list:
            catfish.reset()
        self._catfish_liquidated = False  # 重置鲶鱼强平标志

        # 重置市场状态
        self._reset_market()

        # 重置 tick 计数和各种群淘汰计数（每个 episode 从 0 开始）
        self.tick = 0
        self._pop_liquidated_counts.clear()
        self._eliminating_agents.clear()  # 清空重入保护集合

        # 初始化 episode 价格统计
        initial_price = self.config.market.initial_price
        self._episode_high_price = initial_price
        self._episode_low_price = initial_price

        # 2. 运行 episode_length 个 tick
        for _ in range(episode_length):
            if not self.is_running or self.is_paused:
                break
            self.run_tick()

            # 检查鲶鱼是否被强平
            if self._catfish_liquidated:
                self.logger.info(
                    f"Episode {self.episode} 因鲶鱼强平提前结束 (tick={self.tick})"
                )
                break

            # 检查是否满足提前结束条件
            early_end_result = self._should_end_episode_early()
            if early_end_result is not None:
                reason, agent_type = early_end_result
                if reason == "population_depleted" and agent_type is not None:
                    total = self._pop_total_counts[agent_type]
                    liquidated = self._pop_liquidated_counts.get(agent_type, 0)
                    alive = total - liquidated
                    self.logger.warning(
                        f"Episode {self.episode} 提前结束：{agent_type.value} "
                        f"存活不足 1/4 ({alive}/{total}) (tick={self.tick})"
                    )
                elif reason == "one_sided_orderbook":
                    orderbook = self.matching_engine._orderbook
                    side = "只有买盘" if orderbook.get_best_bid() else "只有卖盘"
                    self.logger.warning(
                        f"Episode {self.episode} 提前结束：订单簿{side} (tick={self.tick})"
                    )
                    # 调试日志：输出做市商状态
                    # self._log_market_maker_status()
                break

        # 3. 进化
        # - tick 数 >= 10 时：正常进化（重新计算适应度）
        # - tick 数 < 10 时：使用缓存适应度进化（打破死循环）
        min_ticks_for_evolution = 10
        if self.is_running and not self.is_paused:
            if self.tick >= min_ticks_for_evolution:
                # 正常进化：重新计算适应度 + 选择 + 繁殖
                current_price = self.matching_engine._orderbook.last_price
                self._evolve_populations_parallel(current_price)

                # 保存每代的 best_genome（如果设置了保存器）
                if self._generation_saver is not None:
                    generation = next(iter(self.populations.values())).generation
                    self._generation_saver.save_generation(
                        generation=generation,
                        populations=self.populations,
                        current_price=current_price,
                    )

                # 进化后重新注册新 Agent 的费率，重建映射表和执行顺序
                self._register_all_agents()
                self._build_agent_map()
                self._build_execution_order()
                self._update_pop_total_counts()

                self.logger.info(f"Episode {self.episode} 完成，tick={self.tick}")
            else:
                # tick 不足：使用缓存适应度进化，打破死循环
                self.logger.warning(
                    f"Episode {self.episode} tick 数不足（{self.tick} < {min_ticks_for_evolution}），"
                    f"使用缓存适应度进化"
                )
                self._evolve_populations_with_cached_fitness()

                # 进化后重新注册新 Agent 的费率，重建映射表和执行顺序
                self._register_all_agents()
                self._build_agent_map()
                self._build_execution_order()
                self._update_pop_total_counts()

    def train(
        self,
        episodes: int | None = None,
        state_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """主训练循环

        Args:
            episodes: 训练的 episode 数量，None 表示无限训练模式
            state_callback: 可选的状态回调函数（用于 UI 更新）
        """
        self.is_running = True
        checkpoint_interval = self.config.training.checkpoint_interval

        ep = 0
        while self.is_running:
            # 检查是否达到目标 episode 数
            if episodes is not None and ep >= episodes:
                break

            self.run_episode()

            # 状态回调
            if state_callback:
                state_callback(self._get_state())

            # 定期保存检查点（使用全局 episode 计数）
            if checkpoint_interval > 0 and self.episode % checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoints/ep_{self.episode}.pkl")

            ep += 1

        self.is_running = False

    def _get_state(self) -> dict[str, Any]:
        """获取当前状态快照"""
        return {
            "tick": self.tick,
            "episode": self.episode,
            "populations": {
                agent_type.value: {
                    "count": len(pop.agents),
                    "generation": pop.generation,
                }
                for agent_type, pop in self.populations.items()
            },
            "high_price": self._episode_high_price,
            "low_price": self._episode_low_price,
        }

    def get_price_stats(self) -> dict:
        """获取当前 episode 的价格统计

        Returns:
            价格统计字典
        """
        total_volume = 0.0
        if self._tick_history_volumes:
            total_volume = sum(abs(v) for v in self._tick_history_volumes)

        return {
            "tick_count": self.tick,
            "high_price": self._episode_high_price,
            "low_price": self._episode_low_price,
            "final_price": (
                self.matching_engine._orderbook.last_price
                if self.matching_engine
                else 0.0
            ),
            "total_volume": total_volume,
        }

    def get_population_stats(self) -> dict:
        """获取当前种群统计

        Returns:
            种群统计字典，包含淘汰数、平均适应度和最精英 species 的平均适应度
        """
        stats: dict = {
            "liquidations": dict(self._pop_liquidated_counts),
            "avg_fitness": {},
            "elite_species_fitness": {},
        }

        if not self.matching_engine:
            return stats

        current_price = self.matching_engine._orderbook.last_price

        for agent_type, pop in self.populations.items():
            agent_fitnesses = pop.evaluate(current_price)
            if agent_fitnesses:
                stats["avg_fitness"][agent_type] = sum(
                    f for _, f in agent_fitnesses
                ) / len(agent_fitnesses)

            # 获取最精英 species 的平均适应度
            elite_fitness = pop.get_elite_species_avg_fitness()
            if elite_fitness is not None:
                stats["elite_species_fitness"][agent_type] = elite_fitness

        return stats

    def _serialize_population_data(
        self, pop: "Population | RetailSubPopulationManager"
    ) -> dict:
        """序列化单个种群数据

        支持普通 Population 和 RetailSubPopulationManager 两种类型。

        Args:
            pop: 种群对象

        Returns:
            序列化的种群数据字典
        """
        if isinstance(pop, RetailSubPopulationManager):
            # RetailSubPopulationManager: 保存所有子种群
            return {
                "is_sub_population_manager": True,
                "generation": pop.generation,
                "sub_population_count": pop.sub_population_count,
                "sub_populations": [
                    {
                        "neat_pop": sub_pop.neat_pop,
                        "generation": sub_pop.generation,
                    }
                    for sub_pop in pop.sub_populations
                ],
            }
        else:
            # 普通 Population
            return {
                "generation": pop.generation,
                "neat_pop": pop.neat_pop,
            }

    def save_checkpoint_data(self) -> dict:
        """返回检查点数据（不写入文件）

        用于多竞技场场景下由 ArenaManager 统一保存。

        Returns:
            检查点数据字典
        """
        return {
            "tick": self.tick,
            "episode": self.episode,
            "populations": {
                agent_type: self._serialize_population_data(pop)
                for agent_type, pop in self.populations.items()
            },
        }

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str | None:
        """查找最新的检查点文件

        按 episode 数字从大到小排序，返回最新的检查点路径。

        Args:
            checkpoint_dir: 检查点目录路径

        Returns:
            最新检查点的路径，如果不存在则返回 None
        """
        import re

        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None

        # 查找所有 ep_*.pkl 文件
        pattern = re.compile(r"ep_(\d+)\.pkl$")
        checkpoints: list[tuple[int, Path]] = []

        for f in checkpoint_path.glob("ep_*.pkl"):
            match = pattern.match(f.name)
            if match:
                episode = int(match.group(1))
                checkpoints.append((episode, f))

        if not checkpoints:
            return None

        # 按 episode 数字降序排序，取最新的
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return str(checkpoints[0][1])

    def save_checkpoint(self, path: str) -> None:
        """保存检查点

        Args:
            path: 检查点文件路径
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "tick": self.tick,
            "episode": self.episode,
            "populations": {
                agent_type: self._serialize_population_data(pop)
                for agent_type, pop in self.populations.items()
            },
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        self.logger.info(f"检查点已保存: {path}")

    def _load_population_data(
        self,
        pop: "Population | RetailSubPopulationManager",
        pop_data: dict,
        agent_type: AgentType,
    ) -> None:
        """加载单个种群数据

        支持普通 Population 和 RetailSubPopulationManager 两种类型。
        自动检测旧格式并迁移。

        Args:
            pop: 种群对象
            pop_data: checkpoint 中的种群数据
            agent_type: Agent 类型
        """
        if isinstance(pop, RetailSubPopulationManager):
            # 当前代码使用 RetailSubPopulationManager
            if pop_data.get("is_sub_population_manager"):
                # 新格式：直接加载子种群数据
                self._load_sub_population_manager_new_format(pop, pop_data)
            else:
                # 旧格式：单个 neat_pop，需要迁移
                self.logger.info(
                    f"检测到旧格式 checkpoint（单个 RETAIL 种群），自动迁移到子种群格式"
                )
                self._migrate_old_checkpoint_to_sub_populations(pop, pop_data)
        else:
            # 普通 Population
            pop.generation = pop_data["generation"]
            pop.neat_pop = pop_data["neat_pop"]
            genomes = list(pop.neat_pop.population.items())
            pop.agents = pop.create_agents(genomes)

    def _load_sub_population_manager_new_format(
        self,
        manager: "RetailSubPopulationManager",
        pop_data: dict,
    ) -> None:
        """加载新格式的子种群管理器数据

        Args:
            manager: 子种群管理器
            pop_data: checkpoint 中的种群数据
        """
        saved_sub_count = pop_data["sub_population_count"]
        current_sub_count = manager.sub_population_count

        if saved_sub_count != current_sub_count:
            self.logger.warning(
                f"checkpoint 子种群数量 ({saved_sub_count}) 与当前配置 "
                f"({current_sub_count}) 不匹配，将尝试加载"
            )

        # 加载每个子种群
        for i, sub_pop_data in enumerate(pop_data["sub_populations"]):
            if i >= len(manager.sub_populations):
                self.logger.warning(f"跳过多余的子种群 {i}")
                break

            sub_pop = manager.sub_populations[i]
            sub_pop.generation = sub_pop_data["generation"]
            sub_pop.neat_pop = sub_pop_data["neat_pop"]
            genomes = list(sub_pop.neat_pop.population.items())
            sub_pop.agents = sub_pop.create_agents(genomes)

    def _migrate_old_checkpoint_to_sub_populations(
        self,
        manager: "RetailSubPopulationManager",
        pop_data: dict,
    ) -> None:
        """将旧格式 checkpoint 迁移到子种群格式

        将单个大 neat_pop 拆分成多个子种群。

        Args:
            manager: 子种群管理器
            pop_data: checkpoint 中的旧格式种群数据
        """
        old_neat_pop = pop_data["neat_pop"]
        old_generation = pop_data["generation"]
        old_genomes = list(old_neat_pop.population.items())
        total_count = len(old_genomes)

        sub_count = manager.sub_population_count
        per_sub = total_count // sub_count

        self.logger.info(
            f"迁移旧 checkpoint: {total_count} 个基因组 -> "
            f"{sub_count} 个子种群，每个 {per_sub} 个"
        )

        # 按顺序拆分基因组到各子种群
        for i, sub_pop in enumerate(manager.sub_populations):
            start_idx = i * per_sub
            end_idx = start_idx + per_sub if i < sub_count - 1 else total_count
            sub_genomes = old_genomes[start_idx:end_idx]

            # 更新子种群的 neat_pop.population
            sub_pop.neat_pop.population = {gid: genome for gid, genome in sub_genomes}
            sub_pop.generation = old_generation

            # 重新进行物种划分
            sub_pop.neat_pop.species.speciate(
                sub_pop.neat_config,
                sub_pop.neat_pop.population,
                old_generation,
            )

            # 重建 agents
            genomes = list(sub_pop.neat_pop.population.items())
            sub_pop.agents = sub_pop.create_agents(genomes)

            self.logger.debug(
                f"子种群 {i}: 加载 {len(genomes)} 个基因组 "
                f"(索引 {start_idx}-{end_idx-1})"
            )

    def load_checkpoint(self, path: str) -> None:
        """加载检查点

        支持新旧两种格式：
        - 新格式：RetailSubPopulationManager 保存为多个子种群
        - 旧格式：单个 RETAIL 种群，自动迁移到子种群格式

        Args:
            path: 检查点文件路径
        """
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.tick = checkpoint["tick"]
        self.episode = checkpoint["episode"]

        for agent_type, pop_data in checkpoint["populations"].items():
            if agent_type in self.populations:
                pop = self.populations[agent_type]
                self._load_population_data(pop, pop_data, agent_type)

        # 注册恢复的 Agent 费率，重建映射表和执行顺序
        self._register_all_agents()
        self._build_agent_map()
        self._build_execution_order()
        self._update_pop_total_counts()

        # 更新网络数据缓存
        self._update_network_caches()

        self.logger.info(f"检查点已加载: {path}")

    def load_checkpoint_data(self, checkpoint: dict) -> None:
        """从检查点数据恢复（不读取文件）

        用于多竞技场场景下由 ArenaManager 统一加载。
        支持新旧两种格式的 checkpoint。

        Args:
            checkpoint: 检查点数据字典
        """
        self.tick = checkpoint["tick"]
        self.episode = checkpoint["episode"]

        for agent_type, pop_data in checkpoint["populations"].items():
            if agent_type in self.populations:
                pop = self.populations[agent_type]
                # 【关键修复】先清理旧 Agent，防止内存泄漏
                # setup() 时已创建了 Agent，加载检查点时需要先清理
                if isinstance(pop, RetailSubPopulationManager):
                    for sub_pop in pop.sub_populations:
                        sub_pop._cleanup_old_agents()
                else:
                    pop._cleanup_old_agents()
                gc.collect()
                gc.collect()

                # 使用统一的加载方法（支持新旧格式）
                self._load_population_data(pop, pop_data, agent_type)

        self._register_all_agents()
        self._build_agent_map()
        self._build_execution_order()
        self._update_pop_total_counts()

        # 更新网络数据缓存
        self._update_network_caches()

    def pause(self) -> None:
        """暂停训练"""
        self.is_paused = True
        self.logger.info("训练已暂停")

    def resume(self) -> None:
        """恢复训练"""
        self.is_paused = False
        self.logger.info("训练已恢复")

    def stop(self) -> None:
        """停止训练"""
        self.is_running = False
        self.logger.info("训练已停止")
        self._shutdown_executor()

        # 清理网络数据缓存
        self._network_caches = None
        self._cache_initialized = False
