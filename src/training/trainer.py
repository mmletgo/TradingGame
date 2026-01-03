"""训练器模块

管理训练流程，协调种群、撮合引擎和事件系统。
"""

import pickle
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.bio.agents.base import AgentType

if TYPE_CHECKING:
    from src.bio.agents.base import Agent
from src.config.config import Config
from src.core.event_engine.event_bus import EventBus
from src.core.event_engine.events import Event
from src.core.log_engine.logger import get_logger
from src.market.adl.adl_manager import ADLManager
from src.market.market_state import NormalizedMarketState
from src.market.matching.matching_engine import MatchingEngine
from src.market.matching.trade import Trade
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.training.population import Population


class Trainer:
    """训练器

    管理 NEAT 进化训练流程，协调三种群在模拟市场中竞争。

    Attributes:
        config: 全局配置
        event_bus: 事件总线
        populations: 三种群（散户/庄家/做市商）
        matching_engine: 撮合引擎
        tick: 当前 tick
        episode: 当前 episode
        is_running: 是否运行中
        is_paused: 是否暂停
        recent_trades: 最近成交记录
    """

    config: Config
    event_bus: EventBus
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
    _pop_liquidated_counts: dict[AgentType, int]  # 各种群当前 episode 已淘汰数量
    tick_start_price: float  # tick 开始时的价格，供数据采集使用

    def __init__(self, config: Config) -> None:
        """创建训练器

        Args:
            config: 全局配置对象
        """
        self.config = config
        self.event_bus = EventBus()
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

    def setup(self) -> None:
        """初始化训练环境

        创建三种群、撮合引擎，初始化市场。
        训练模式使用直接调用，不需要订阅事件。
        """
        # 创建三种群
        for agent_type in AgentType:
            self.populations[agent_type] = Population(
                agent_type, self.config, self.event_bus
            )

        # 创建撮合引擎
        self.matching_engine = MatchingEngine(self.event_bus, self.config.market)

        # 创建 ADL 管理器
        self.adl_manager = ADLManager()

        # 训练模式使用直接调用，不需要订阅成交事件和强平事件
        # 保留事件系统用于可能的调试或 UI 模式

        # 注册所有 Agent 的费率
        self._register_all_agents()

        # 构建 Agent 映射表和执行顺序
        self._build_agent_map()
        self._build_execution_order()

        # 记录各种群总数
        self._update_pop_total_counts()

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

    def _on_trade(self, event: Event) -> None:
        """处理成交事件，记录最近成交

        从事件数据中重建 Trade 对象并记录。
        """
        data = event.data
        # 从事件数据重建 Trade 对象
        trade = Trade(
            trade_id=data.get("trade_id", 0),
            price=data.get("price", 0.0),
            quantity=int(data.get("quantity", 0)),  # 确保是 int
            buyer_id=data.get("buyer_id", 0),
            seller_id=data.get("seller_id", 0),
            buyer_fee=data.get("buyer_fee", 0.0),
            seller_fee=data.get("seller_fee", 0.0),
            is_buyer_taker=data.get("is_buyer_taker", True),
        )
        # deque(maxlen=100) 自动丢弃旧数据
        self.recent_trades.append(trade)

    def _on_liquidation(self, event: Event) -> None:
        """处理强平事件，提交市价单平仓"""
        agent_id = event.data.get("agent_id")
        quantity = event.data.get("position_quantity")
        if agent_id is not None and quantity is not None and self.matching_engine:
            # 平仓：如果持仓为正（多头），则卖出；如果为负（空头），则买入
            side = OrderSide.SELL if quantity > 0 else OrderSide.BUY
            order = Order(
                order_id=0,  # 由撮合引擎分配
                agent_id=agent_id,
                side=side,
                order_type=OrderType.MARKET,
                price=0.0,  # 市价单不需要价格
                quantity=abs(int(quantity)),  # 确保是 int
            )
            self.matching_engine.process_order(order)

        # 标记 Agent 已被强平
        self._mark_agent_liquidated(agent_id)

    def _mark_agent_liquidated(self, agent_id: int) -> None:
        """标记 Agent 已被强平

        Args:
            agent_id: 被强平的 Agent ID
        """
        agent = self.agent_map.get(agent_id)
        if agent and not agent.is_liquidated:
            agent.is_liquidated = True
            self.logger.info(f"Agent {agent_id} 已被强平，本轮 episode 禁用")

    def _handle_liquidation_direct(self, agent: "Agent", current_price: float) -> None:
        """直接处理强平（训练模式）

        创建市价平仓单，直接调用撮合引擎处理，更新账户。
        如果市价单无法完全成交，触发 ADL 机制。

        Args:
            agent: 被强平的 Agent
            current_price: 当前价格
        """
        if not self.matching_engine or not self.adl_manager:
            return

        position_qty = agent.account.position.quantity
        if position_qty == 0:
            return

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
        trades = self.matching_engine.process_order_direct(order)

        # 计算已成交数量
        filled_qty = sum(trade.quantity for trade in trades)

        # 更新账户（taker 和 maker）
        for trade in trades:
            # 使用 is_buyer_taker 判断 taker 是买方还是卖方
            # 旧逻辑 `is_buyer = trade.buyer_id == agent.agent_id` 在自成交时会出错
            is_buyer = trade.is_buyer_taker
            agent.account.on_trade(trade, is_buyer)
            self.recent_trades.append(trade)
            # 更新 maker 的账户（即使自成交也需要更新）
            maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
            maker_agent = self.agent_map.get(maker_id)
            if maker_agent is not None:
                # maker 的方向与 taker 相反
                maker_is_buyer = not trade.is_buyer_taker
                maker_agent.account.on_trade(trade, maker_is_buyer)

        # 检查是否需要 ADL
        remaining_qty = target_qty - filled_qty
        if remaining_qty > 0:
            self._execute_adl(agent, remaining_qty, current_price, is_long)

    def _execute_adl(
        self,
        liquidated_agent: "Agent",
        remaining_qty: int,
        current_price: float,
        is_long: bool,
    ) -> None:
        """执行 ADL 自动减仓

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

        # 获取所有 Agent
        all_agents: list["Agent"] = []
        for population in self.populations.values():
            all_agents.extend(population.agents)

        # 确定需要的对手方方向
        # 被强平方是多头（需要卖出平仓），则需要空头对手（买入）
        # 被强平方是空头（需要买入平仓），则需要多头对手（卖出）
        target_side = -1 if is_long else 1

        # 获取 ADL 候选列表
        candidates = self.adl_manager.get_adl_candidates(
            agents=all_agents,
            current_price=current_price,
            target_side=target_side,
            exclude_agent_id=liquidated_agent.agent_id,
        )

        # 执行 ADL（内部直接更新账户，返回剩余未平仓数量）
        remaining_after_adl = self.adl_manager.execute_adl(
            liquidated_agent=liquidated_agent,
            remaining_qty=remaining_qty,
            candidates=candidates,
            adl_price=adl_price,
            current_price=current_price,
        )

        # ADL 成交后检查 candidates 的强平和淘汰条件
        # 注意：已淘汰的 candidate 不需要再检查（is_liquidated=True）
        for candidate in candidates:
            candidate_agent = candidate.agent
            if candidate_agent.is_liquidated:
                continue  # 已淘汰，跳过检查
            # 检查强平条件
            if candidate_agent.account.check_liquidation(current_price):
                self._handle_liquidation_direct(candidate_agent, current_price)
            # 检查淘汰条件
            self._check_elimination(candidate_agent, current_price)

        # 兜底处理：理论上不应该出现 ADL 候选不足的情况（多空对等）
        # 如果出现，说明有其他 bug，记录错误日志并强制清零
        if remaining_after_adl > 0:
            self.logger.error(
                f"ADL 异常: Agent {liquidated_agent.agent_id} "
                f"剩余仓位 {remaining_after_adl} 无法匹配（多空应对等，请检查bug）, "
                f"强制清零, 余额 {liquidated_agent.account.balance:.2f} → 0"
            )
            # 强制清零仓位和余额
            liquidated_agent.account.position.quantity = 0
            liquidated_agent.account.position.avg_price = 0.0
            liquidated_agent.account.balance = 0.0

    def _check_elimination(self, agent: "Agent", current_price: float) -> None:
        """检查并处理个体淘汰（资金不足时淘汰）

        当 当前净值/初始资金 < 0.1 时：
        1. 强平其持有的仓位（市价单 + ADL）
        2. 标记个体为淘汰状态

        Args:
            agent: 要检查的 Agent
            current_price: 当前价格
        """
        if agent.is_liquidated:
            return  # 已淘汰，无需重复检查

        if agent.account.check_elimination(current_price, threshold=0.1):
            # 先撤销挂单（防止淘汰后仍被成交）
            # 做市商有多个挂单（bid_order_ids 和 ask_order_ids），需要全部撤销
            if agent.agent_type == AgentType.MARKET_MAKER and self.matching_engine:
                # 类型检查：确保是 MarketMakerAgent
                from src.bio.agents.market_maker import MarketMakerAgent
                if isinstance(agent, MarketMakerAgent):
                    agent._cancel_all_orders_direct(self.matching_engine)
            elif agent.account.pending_order_id is not None and self.matching_engine:
                self.matching_engine.cancel_order_direct(agent.account.pending_order_id)
                agent.account.pending_order_id = None

            # 再强平仓位（如果有的话）
            if agent.account.position.quantity != 0:
                self._handle_liquidation_direct(agent, current_price)

            # 处理穿仓：如果余额为负，归零（由系统承担损失）
            if agent.account.balance < 0:
                agent.account.balance = 0.0

            # 最后标记为淘汰
            agent.is_liquidated = True
            self._pop_liquidated_counts[agent.agent_type] = (
                self._pop_liquidated_counts.get(agent.agent_type, 0) + 1
            )
            self.logger.info(
                f"Agent {agent.agent_id} 已淘汰（资金不足10%），本轮 episode 禁用"
            )

    def _any_population_eliminated(self) -> AgentType | None:
        """检查是否有任一种群被全部淘汰（O(1) 复杂度）

        Returns:
            AgentType | None: 被淘汰的种群类型，如果没有则返回 None
        """
        for agent_type, total in self._pop_total_counts.items():
            if total > 0:
                liquidated = self._pop_liquidated_counts.get(agent_type, 0)
                if liquidated >= total:
                    return agent_type
        return None

    def _compute_normalized_market_state(self) -> NormalizedMarketState:
        """预计算归一化的公共市场数据

        在每个 tick 开始时调用，避免每个 Agent 重复计算相同的归一化数据。

        Returns:
            NormalizedMarketState: 归一化后的市场状态
        """
        orderbook = self.matching_engine._orderbook

        # 获取参考价格
        mid_price = orderbook.get_mid_price()
        if mid_price is None:
            mid_price = orderbook.last_price
        if mid_price == 0:
            mid_price = 100.0

        tick_size = orderbook.tick_size
        depth = orderbook.get_depth(levels=100)

        # 向量化买盘：100档 × 2 = 200
        bid_data = np.zeros(200, dtype=np.float32)
        bids = depth["bids"]
        if bids:
            bid_prices = np.array([p for p, _ in bids], dtype=np.float32)
            bid_qtys = np.array([q for _, q in bids], dtype=np.float32)
            n = len(bids)
            if mid_price > 0:
                bid_data[0 : n * 2 : 2] = (bid_prices - mid_price) / mid_price
            bid_data[1 : n * 2 : 2] = bid_qtys

        # 向量化卖盘：100档 × 2 = 200
        ask_data = np.zeros(200, dtype=np.float32)
        asks = depth["asks"]
        if asks:
            ask_prices = np.array([p for p, _ in asks], dtype=np.float32)
            ask_qtys = np.array([q for _, q in asks], dtype=np.float32)
            n = len(asks)
            if mid_price > 0:
                ask_data[0 : n * 2 : 2] = (ask_prices - mid_price) / mid_price
            ask_data[1 : n * 2 : 2] = ask_qtys

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
            if mid_price > 0:
                trade_prices[:n] = (prices - mid_price) / mid_price
            trade_quantities[:n] = qtys

        return NormalizedMarketState(
            mid_price=mid_price,
            tick_size=tick_size,
            bid_data=bid_data,
            ask_data=ask_data,
            trade_prices=trade_prices,
            trade_quantities=trade_quantities,
        )

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
                trades = agent.execute_action_direct(
                    action, params, self.matching_engine
                )
                for trade in trades:
                    self.recent_trades.append(trade)
                    # 更新 maker 的账户（即使自成交也需要更新）
                    maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
                    maker_agent = self.agent_map.get(maker_id)
                    if maker_agent is not None:
                        # maker 的方向与 taker 相反
                        is_buyer = not trade.is_buyer_taker
                        maker_agent.account.on_trade(trade, is_buyer)

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
            self.logger.error(f"初始化后仓位不对等！总偏差={total_position}, 多头={long_qty}, 空头={short_qty}")

        # 初始化 tick_start_price，供数据采集使用
        self.tick_start_price = self.matching_engine._orderbook.last_price

    def run_tick(self) -> None:
        """执行一个 tick（直接调用模式）

        时序设计：
        1. Tick 开始：用当前价格检查所有 agent 的强平和淘汰条件
        2. Tick 过程：Agent 按顺序决策和下单
        3. Tick 结束：下单效果（价格变动）在下个 tick 被感知

        这样设计确保：
        - 淘汰检查和数据采集使用同一价格（tick 开始时的价格）
        - 所有 agent 在同一价格基础上被检查，公平
        """
        if not self.matching_engine:
            return

        self.tick += 1
        orderbook = self.matching_engine._orderbook
        current_price = orderbook.last_price
        self.tick_start_price = current_price  # 保存 tick 开始时的价格

        # === Tick 开始：检查所有 agent 的强平和淘汰条件 ===
        for agent in self.agent_execution_order:
            if agent.is_liquidated:
                continue
            # 检查强平
            if agent.account.check_liquidation(current_price):
                self._handle_liquidation_direct(agent, current_price)
            # 检查淘汰
            self._check_elimination(agent, current_price)

        # 预计算归一化市场数据
        market_state = self._compute_normalized_market_state()

        # === Tick 过程：Agent 按顺序决策和下单 ===
        for agent in self.agent_execution_order:
            if agent.is_liquidated:
                continue

            # 决策
            action, params = agent.decide(market_state, orderbook)

            # 直接执行（绕过事件系统）
            trades = agent.execute_action_direct(
                action, params, self.matching_engine
            )

            # 记录成交并更新对手方账户（maker）
            for trade in trades:
                self.recent_trades.append(trade)
                # 更新 maker 的账户（taker 的账户已在 execute_action_direct 中更新）
                maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
                # 注意：即使 maker_id == agent.agent_id（自成交），也需要更新 maker 端
                # 因为 _process_trades_direct 只更新了 taker 端（买或卖其中一方）
                maker_agent = self.agent_map.get(maker_id)
                if maker_agent is not None:
                    # maker 的方向与 taker 相反：taker 买则 maker 卖，taker 卖则 maker 买
                    is_buyer = not trade.is_buyer_taker
                    maker_agent.account.on_trade(trade, is_buyer)

        # 强平和淘汰检查移到下个 tick 开始时执行

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

        # 2. 运行 episode_length 个 tick
        for _ in range(episode_length):
            if not self.is_running or self.is_paused:
                break
            self.run_tick()

            # 检查是否有任一种群被全部淘汰
            eliminated_type = self._any_population_eliminated()
            if eliminated_type is not None:
                self.logger.warning(
                    f"Episode {self.episode} 提前结束：{eliminated_type.value} 已全部淘汰 (tick={self.tick})"
                )
                break

        # 3. 进化（仅在正常完成 episode 时）
        if self.is_running and not self.is_paused:
            current_price = self.matching_engine._orderbook.last_price
            for population in self.populations.values():
                population.evolve(current_price)

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
                pop.agents = pop.create_agents(genomes, self.event_bus)

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
