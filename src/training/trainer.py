"""训练器模块

管理训练流程，协调种群和撮合引擎。
"""

import pickle
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.bio.agents.base import AgentType

if TYPE_CHECKING:
    from src.bio.agents.base import ActionType, Agent
    from src.market.market_state import (
        NormalizedMarketState as NormalizedMarketStateType,
    )
    from src.market.orderbook.orderbook import OrderBook
from src.config.config import Config
from src.core.log_engine.logger import get_logger
from src.market.adl.adl_manager import ADLCandidate, ADLManager
from src.market.catfish import CatfishBase, create_all_catfish, create_catfish
from src.market.market_state import NormalizedMarketState
from src.market.matching.matching_engine import MatchingEngine
from src.market.matching.trade import Trade
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.training.population import Population


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
    populations: dict[AgentType, Population]
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

    def __init__(self, config: Config) -> None:
        """创建训练器

        Args:
            config: 全局配置对象
        """
        self.config = config
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
        self.catfish_list = []
        self._price_history = []

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

    def setup(self) -> None:
        """初始化训练环境

        创建四种群、撮合引擎，初始化市场。
        训练模式使用直接调用。
        """
        # 串行创建四种群（每个种群内部的Agent创建是并行的）
        for agent_type in AgentType:
            self.populations[agent_type] = Population(agent_type, self.config)

        # 创建撮合引擎
        self.matching_engine = MatchingEngine(self.config.market)

        # 创建 ADL 管理器
        self.adl_manager = ADLManager()

        # 初始化鲶鱼（如果配置中启用）
        if self.config.catfish and self.config.catfish.enabled:
            if self.config.catfish.multi_mode:
                # 多模式：同时创建三种鲶鱼
                self.catfish_list = create_all_catfish(self.config.catfish)
                for catfish in self.catfish_list:
                    self.matching_engine.register_agent(
                        catfish.catfish_id,
                        0.0,  # maker fee
                        0.0,  # taker fee
                    )
                self.logger.info(
                    f"鲶鱼已启用: 多模式（三种鲶鱼同时运行，相位错开）"
                )
            else:
                # 单模式：只创建一种鲶鱼
                catfish = create_catfish(-1, self.config.catfish)
                self.catfish_list = [catfish]
                self.matching_engine.register_agent(
                    catfish.catfish_id,
                    0.0,  # maker fee
                    0.0,  # taker fee
                )
                self.logger.info(
                    f"鲶鱼已启用: 模式={self.config.catfish.mode.value}"
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
        """构建 Agent 执行顺序列表（做市商 -> 空头庄家 -> 多头庄家 -> 高级散户 -> 散户）"""
        self.agent_execution_order.clear()
        for agent_type in [
            AgentType.MARKET_MAKER,
            AgentType.BEAR_WHALE,
            AgentType.BULL_WHALE,
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

    def _mark_agent_liquidated(self, agent_id: int) -> None:
        """标记 Agent 已被强平

        Args:
            agent_id: 被强平的 Agent ID
        """
        agent = self.agent_map.get(agent_id)
        if agent and not agent.is_liquidated:
            agent.is_liquidated = True
            self.logger.info(f"Agent {agent_id} 已被强平，本轮 episode 禁用")

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

        self.logger.info(
            f"ADL 触发: Agent {liquidated_agent.agent_id} "
            f"剩余平仓量 {remaining_qty}, "
            f"成交价 {adl_price:.2f}, "
            f"候选人数 {len(candidates)}"
        )

        for candidate in candidates:
            if remaining_qty <= 0:
                break
            # 使用候选清单中的 position_qty（已被之前的 ADL 更新过）
            # 同时也要检查实际仓位，取两者最小值
            candidate_available_qty = abs(candidate.position_qty)
            actual_position = abs(candidate.agent.account.position.quantity)
            available_qty = min(candidate_available_qty, actual_position)

            liquidated_actual_position = abs(liquidated_agent.account.position.quantity)
            trade_qty = min(available_qty, remaining_qty, liquidated_actual_position)

            if trade_qty <= 0:
                continue

            # 更新账户
            liquidated_agent.account.on_adl_trade(trade_qty, adl_price, is_taker=True)
            candidate.agent.account.on_adl_trade(trade_qty, adl_price, is_taker=False)

            # 更新候选清单中的 position_qty，确保后续 ADL 不会重复使用
            if candidate.position_qty > 0:
                candidate.position_qty -= trade_qty
            else:
                candidate.position_qty += trade_qty

            # self.logger.info(
            #     f"ADL 成交: Agent {liquidated_agent.agent_id} 与 Agent {candidate.agent.agent_id} "
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

    def _should_end_episode_early(self) -> AgentType | None:
        """检查是否满足提前结束 episode 的条件（O(1) 复杂度）

        触发条件：
        - 任意种群的存活数量少于初始值的 1/4

        这确保每个种群都有足够的个体用于 NEAT 进化。

        Returns:
            AgentType | None: 满足条件的种群类型，如果没有则返回 None
        """
        for agent_type, total in self._pop_total_counts.items():
            if total > 0:
                liquidated = self._pop_liquidated_counts.get(agent_type, 0)
                alive = total - liquidated
                # 任意种群存活少于初始值的 1/4 时触发早停
                if alive < total / 4:
                    return agent_type
        return None

    def _compute_normalized_market_state(self) -> NormalizedMarketState:
        """预计算归一化的公共市场数据

        在每个 tick 开始时调用，避免每个 Agent 重复计算相同的归一化数据。
        使用 EMA 平滑后的 mid_price 作为参考价格，减缓价格变化传导速度。

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
        depth = orderbook.get_depth(levels=100)

        # 向量化买盘：100档 × 2 = 200（使用平滑价格归一化）
        bid_data = np.zeros(200, dtype=np.float32)
        bids = depth["bids"]
        if bids:
            bid_prices = np.array([p for p, _ in bids], dtype=np.float32)
            bid_qtys = np.array([q for _, q in bids], dtype=np.float32)
            n = len(bids)
            if smooth_mid_price > 0:
                bid_data[0 : n * 2 : 2] = (
                    bid_prices - smooth_mid_price
                ) / smooth_mid_price
            # 数量使用对数归一化：log10(qty + 1) / 10，将 1e10 压缩到 ~1.0
            bid_data[1 : n * 2 : 2] = np.log10(bid_qtys + 1) / 10.0

        # 向量化卖盘：100档 × 2 = 200（使用平滑价格归一化）
        ask_data = np.zeros(200, dtype=np.float32)
        asks = depth["asks"]
        if asks:
            ask_prices = np.array([p for p, _ in asks], dtype=np.float32)
            ask_qtys = np.array([q for _, q in asks], dtype=np.float32)
            n = len(asks)
            if smooth_mid_price > 0:
                ask_data[0 : n * 2 : 2] = (
                    ask_prices - smooth_mid_price
                ) / smooth_mid_price
            # 数量使用对数归一化
            ask_data[1 : n * 2 : 2] = np.log10(ask_qtys + 1) / 10.0

        # 向量化成交：100笔（数量带方向：正=taker买入，负=taker卖出）
        trade_prices = np.zeros(100, dtype=np.float32)
        trade_quantities = np.zeros(100, dtype=np.float32)
        trades = list(self.recent_trades)  # deque 转换为 list
        if trades:
            prices = np.array([t.price for t in trades], dtype=np.float32)
            qtys = np.array(
                [t.quantity if t.is_buyer_taker else -t.quantity for t in trades],
                dtype=np.float32,
            )
            n = len(trades)
            if smooth_mid_price > 0:
                trade_prices[:n] = (prices - smooth_mid_price) / smooth_mid_price
            # 成交数量带方向的对数归一化：sign(qty) * log10(|qty| + 1) / 10
            trade_quantities[:n] = np.sign(qtys) * np.log10(np.abs(qtys) + 1) / 10.0

        return NormalizedMarketState(
            mid_price=smooth_mid_price,  # 使用 EMA 平滑后的价格
            tick_size=tick_size,
            bid_data=bid_data,
            ask_data=ask_data,
            trade_prices=trade_prices,
            trade_quantities=trade_quantities,
        )

    def _evolve_populations_parallel(self, current_price: float) -> None:
        """并行进化所有种群"""
        executor = self._get_executor()
        populations = list(self.populations.values())

        futures = {
            executor.submit(pop.evolve, current_price): pop for pop in populations
        }

        for future in as_completed(futures):
            pop = futures[future]
            try:
                future.result()
            except Exception as e:
                self.logger.error(f"种群 {pop.agent_type.value} 进化失败: {e}")
                raise

    def _batch_decide_parallel(
        self,
        agents: list["Agent"],
        market_state: "NormalizedMarketStateType",
        orderbook: "OrderBook",
    ) -> list[tuple["Agent", "ActionType", dict[str, Any]]]:
        """并行执行所有 Agent 的决策"""
        executor = self._get_executor()

        future_to_idx: dict[
            Future[tuple["ActionType", dict[str, Any]]], tuple[int, "Agent"]
        ] = {}
        for idx, agent in enumerate(agents):
            if not agent.is_liquidated:
                future = executor.submit(agent.decide, market_state, orderbook)
                future_to_idx[future] = (idx, agent)

        results: list[tuple["Agent", "ActionType", dict[str, Any]] | None] = [
            None
        ] * len(agents)
        for future in as_completed(future_to_idx):
            idx, agent = future_to_idx[future]
            try:
                action, params = future.result()
                results[idx] = (agent, action, params)
            except Exception as e:
                self.logger.warning(f"Agent {agent.agent_id} 决策异常: {e}")

        return [r for r in results if r is not None]

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
        1. Tick 开始：用 smooth_mid_price 检查所有 agent 的强平条件（爆仓即淘汰）
        2. Tick 过程：Agent 按顺序决策和下单
        3. Tick 结束：下单效果（价格变动）在下个 tick 被感知

        这样设计确保：
        - 强平检查和 Agent 报价使用同一价格基准（smooth_mid_price）
        - 所有 agent 在同一价格基础上被检查，公平
        """
        if not self.matching_engine:
            return

        self.tick += 1
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
                    if agent.account.balance < 0:
                        agent.account.balance = 0.0

        # === 鲶鱼行动（在 Agent 之前）===
        for catfish in self.catfish_list:
            should_act, direction = catfish.decide(
                orderbook,
                self.tick,
                self._price_history,
            )
            if should_act and direction != 0:
                catfish_trades = catfish.execute(direction, self.matching_engine)
                catfish.record_action(self.tick)
                # 鲶鱼无限资金模式：只更新 maker 账户
                for trade in catfish_trades:
                    self.recent_trades.append(trade)
                    # 确定 maker（与鲶鱼成交的对手方）
                    maker_id = (
                        trade.seller_id if trade.is_buyer_taker
                        else trade.buyer_id
                    )
                    # 只有正数 ID 的 maker 才需要更新账户
                    if maker_id > 0:
                        maker_agent = self.agent_map.get(maker_id)
                        if maker_agent is not None:
                            is_buyer = not trade.is_buyer_taker
                            maker_agent.account.on_trade(trade, is_buyer)

        # 预计算归一化市场数据
        market_state = self._compute_normalized_market_state()

        # === 并行决策阶段 ===
        decisions = self._batch_decide_parallel(
            self.agent_execution_order, market_state, orderbook
        )

        # === 串行执行阶段 ===
        for agent, action, params in decisions:
            trades = agent.execute_action(action, params, self.matching_engine)

            for trade in trades:
                self.recent_trades.append(trade)
                maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
                maker_agent = self.agent_map.get(maker_id)
                if maker_agent is not None:
                    is_buyer = not trade.is_buyer_taker
                    maker_agent.account.on_trade(trade, is_buyer)

        # 记录价格历史（tick 结束时）
        current_price = orderbook.get_mid_price()
        if current_price is None:
            current_price = orderbook.last_price
        self._price_history.append(current_price)
        # 限制历史长度（避免内存无限增长）
        if len(self._price_history) > 1000:
            self._price_history = self._price_history[-1000:]

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

        # 重置市场状态
        self._reset_market()

        # 重置 tick 计数和各种群淘汰计数（每个 episode 从 0 开始）
        self.tick = 0
        self._pop_liquidated_counts.clear()
        self._eliminating_agents.clear()  # 清空重入保护集合

        # 2. 运行 episode_length 个 tick
        for _ in range(episode_length):
            if not self.is_running or self.is_paused:
                break
            self.run_tick()

            # 检查是否满足提前结束条件
            early_end_type = self._should_end_episode_early()
            if early_end_type is not None:
                total = self._pop_total_counts[early_end_type]
                liquidated = self._pop_liquidated_counts.get(early_end_type, 0)
                alive = total - liquidated
                self.logger.warning(
                    f"Episode {self.episode} 提前结束：{early_end_type.value} "
                    f"存活不足 1/4 ({alive}/{total}) (tick={self.tick})"
                )
                break

        # 3. 进化（仅在正常完成 episode 时）
        if self.is_running and not self.is_paused:
            current_price = self.matching_engine._orderbook.last_price
            self._evolve_populations_parallel(current_price)

            # 进化后重新注册新 Agent 的费率，重建映射表和执行顺序
            self._register_all_agents()
            self._build_agent_map()
            self._build_execution_order()
            self._update_pop_total_counts()

            self.logger.info(f"Episode {self.episode} 完成，tick={self.tick}")

    def train(
        self,
        episodes: int,
        state_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """主训练循环

        Args:
            episodes: 训练的 episode 数量
            state_callback: 可选的状态回调函数（用于 UI 更新）
        """
        self.is_running = True
        checkpoint_interval = self.config.training.checkpoint_interval

        for ep in range(episodes):
            if not self.is_running:
                break

            self.run_episode()

            # 状态回调
            if state_callback:
                state_callback(self._get_state())

            # 定期保存检查点
            if checkpoint_interval > 0 and (ep + 1) % checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoints/ep_{ep + 1}.pkl")

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
        }

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
                agent_type: {
                    "generation": pop.generation,
                    "neat_pop": pop.neat_pop,
                }
                for agent_type, pop in self.populations.items()
            },
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        self.logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str) -> None:
        """加载检查点

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
                pop.generation = pop_data["generation"]
                pop.neat_pop = pop_data["neat_pop"]
                # 重建 agents
                genomes = list(pop.neat_pop.population.items())
                pop.agents = pop.create_agents(genomes)

        # 注册恢复的 Agent 费率，重建映射表和执行顺序
        self._register_all_agents()
        self._build_agent_map()
        self._build_execution_order()
        self._update_pop_total_counts()

        self.logger.info(f"检查点已加载: {path}")

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
