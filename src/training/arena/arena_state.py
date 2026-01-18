"""竞技场状态模块

本模块定义用于多竞技场并行推理架构的状态类。
将 Agent 账户状态与 Agent 对象解耦，每个竞技场维护独立的状态副本。
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from src.config.config import AgentConfig, AgentType, CatfishConfig, CatfishMode

if TYPE_CHECKING:
    from src.bio.agents.base import Agent
    from src.market.adl.adl_manager import ADLManager
    from src.market.catfish.catfish_base import CatfishBase
    from src.market.matching.matching_engine import MatchingEngine


@dataclass
class AgentAccountState:
    """Agent 账户状态（轻量级，约 200 bytes）

    与 Agent 对象解耦，每个竞技场维护独立副本。
    包含所有交易相关的状态信息，支持独立的持仓更新和盈亏计算。

    Attributes:
        agent_id: Agent ID
        agent_type: Agent 类型
        balance: 当前余额
        position_quantity: 持仓数量（正数为多头，负数为空头）
        position_avg_price: 持仓均价
        realized_pnl: 已实现盈亏
        leverage: 杠杆倍数
        maintenance_margin_rate: 维持保证金率
        initial_balance: 初始余额
        pending_order_id: 当前挂单ID（None 表示无挂单）
        maker_volume: 作为 maker 的累计成交量
        volatility_contribution: 作为 taker 的价格冲击累计（庄家用）
        is_liquidated: 是否已被强平
        order_counter: 订单计数器
        maker_fee_rate: 挂单费率
        taker_fee_rate: 吃单费率
    """

    agent_id: int
    agent_type: AgentType
    balance: float
    position_quantity: int
    position_avg_price: float
    realized_pnl: float
    leverage: float
    maintenance_margin_rate: float
    initial_balance: float
    pending_order_id: int | None
    maker_volume: int
    volatility_contribution: float
    is_liquidated: bool
    order_counter: int
    maker_fee_rate: float
    taker_fee_rate: float
    bid_order_ids: list[int] = field(default_factory=list)
    ask_order_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_agent(cls, agent: "Agent") -> "AgentAccountState":
        """从 Agent 对象创建状态副本

        提取 Agent 账户的核心状态，创建独立副本。

        Args:
            agent: Agent 对象

        Returns:
            新创建的 AgentAccountState 实例
        """
        account = agent.account

        # 获取做市商挂单 ID（如果是做市商）
        bid_order_ids: list[int] = []
        ask_order_ids: list[int] = []
        if hasattr(agent, 'bid_order_ids'):
            bid_order_ids = list(agent.bid_order_ids)
        if hasattr(agent, 'ask_order_ids'):
            ask_order_ids = list(agent.ask_order_ids)

        return cls(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            balance=account.balance,
            position_quantity=account.position.quantity,
            position_avg_price=account.position.avg_price,
            realized_pnl=account.position.realized_pnl,
            leverage=account.leverage,
            maintenance_margin_rate=account.maintenance_margin_rate,
            initial_balance=account.initial_balance,
            pending_order_id=account.pending_order_id,
            maker_volume=account.maker_volume,
            volatility_contribution=account.volatility_contribution,
            is_liquidated=agent.is_liquidated,
            order_counter=agent._order_counter,
            maker_fee_rate=account.maker_fee_rate,
            taker_fee_rate=account.taker_fee_rate,
            bid_order_ids=bid_order_ids,
            ask_order_ids=ask_order_ids,
        )

    def reset(self, config: AgentConfig) -> None:
        """重置到初始状态

        用于 Episode 开始时重置账户状态。

        Args:
            config: Agent 配置对象
        """
        self.balance = config.initial_balance
        self.initial_balance = config.initial_balance
        self.position_quantity = 0
        self.position_avg_price = 0.0
        self.realized_pnl = 0.0
        self.leverage = config.leverage
        self.maintenance_margin_rate = config.maintenance_margin_rate
        self.pending_order_id = None
        self.maker_volume = 0
        self.volatility_contribution = 0.0
        self.is_liquidated = False
        self.order_counter = 0
        self.maker_fee_rate = config.maker_fee_rate
        self.taker_fee_rate = config.taker_fee_rate
        self.bid_order_ids = []
        self.ask_order_ids = []

    def get_equity(self, current_price: float) -> float:
        """计算净值

        净值 = 余额 + 浮动盈亏

        Args:
            current_price: 当前市场价格

        Returns:
            账户净值
        """
        unrealized_pnl = (current_price - self.position_avg_price) * self.position_quantity
        return self.balance + unrealized_pnl

    def get_margin_ratio(self, current_price: float) -> float:
        """计算保证金率

        保证金率 = 净值 / 持仓市值，无持仓时返回无穷大。

        Args:
            current_price: 当前市场价格

        Returns:
            保证金率
        """
        equity = self.get_equity(current_price)
        position_value = abs(self.position_quantity) * current_price
        if position_value == 0:
            return float("inf")
        return equity / position_value

    def check_liquidation(self, current_price: float) -> bool:
        """检查是否需要强平

        当保证金率低于维持保证金率时需要强平。

        Args:
            current_price: 当前市场价格

        Returns:
            True 表示需要强平
        """
        margin_ratio = self.get_margin_ratio(current_price)
        return margin_ratio < self.maintenance_margin_rate

    def on_trade(
        self,
        trade_price: float,
        trade_quantity: int,
        is_buyer: bool,
        fee: float,
        is_maker: bool,
    ) -> float:
        """处理成交，更新持仓和余额

        完整实现持仓更新逻辑：
        - 空仓开仓
        - 加仓（加权平均）
        - 减仓（实现盈亏）
        - 完全平仓
        - 反向开仓

        Args:
            trade_price: 成交价格
            trade_quantity: 成交数量
            is_buyer: 是否为买方
            fee: 手续费
            is_maker: 是否为 maker

        Returns:
            本次成交产生的已实现盈亏
        """
        # 累加 maker 成交量
        if is_maker:
            self.maker_volume += trade_quantity

        # 确定成交方向：1=买入，-1=卖出
        side = 1 if is_buyer else -1

        # 更新持仓，计算已实现盈亏
        realized_pnl = self._update_position(side, trade_quantity, trade_price)

        # 更新余额（已实现盈亏 - 手续费）
        self.balance += realized_pnl - fee

        return realized_pnl

    def _update_position(self, side: int, quantity: int, price: float) -> float:
        """更新持仓，返回已实现盈亏

        实现与 Position.update 相同的逻辑。

        Args:
            side: 成交方向（1=买入，-1=卖出）
            quantity: 成交数量
            price: 成交价格

        Returns:
            本次成交产生的已实现盈亏
        """
        realized: float = 0.0

        # 空仓：直接开仓
        if self.position_quantity == 0:
            self.position_quantity = quantity * side
            self.position_avg_price = price
            return 0.0

        # 持多头
        if self.position_quantity > 0:
            if side == 1:  # 买入加仓
                # 加权平均计算新均价
                total_cost = self.position_quantity * self.position_avg_price + quantity * price
                self.position_quantity += quantity
                self.position_avg_price = total_cost / self.position_quantity
            else:  # 卖出
                if quantity < self.position_quantity:
                    # 减多仓
                    realized = (price - self.position_avg_price) * quantity
                    self.position_quantity -= quantity
                elif quantity == self.position_quantity:
                    # 完全平多
                    realized = (price - self.position_avg_price) * self.position_quantity
                    self.position_quantity = 0
                    self.position_avg_price = 0.0
                else:
                    # 反向开空
                    realized = (price - self.position_avg_price) * self.position_quantity
                    remaining = quantity - self.position_quantity
                    self.position_quantity = -remaining
                    self.position_avg_price = price
        # 持空头
        else:
            if side == -1:  # 卖出加空
                # 加权平均计算新均价
                abs_qty = abs(self.position_quantity)
                total_cost = abs_qty * self.position_avg_price + quantity * price
                self.position_quantity -= quantity
                self.position_avg_price = total_cost / abs(self.position_quantity)
            else:  # 买入
                abs_qty = abs(self.position_quantity)
                if quantity < abs_qty:
                    # 减空仓
                    realized = (self.position_avg_price - price) * quantity
                    self.position_quantity += quantity
                elif quantity == abs_qty:
                    # 完全平空
                    realized = (self.position_avg_price - price) * abs_qty
                    self.position_quantity = 0
                    self.position_avg_price = 0.0
                else:
                    # 反向开多
                    realized = (self.position_avg_price - price) * abs_qty
                    remaining = quantity - abs_qty
                    self.position_quantity = remaining
                    self.position_avg_price = price

        self.realized_pnl += realized
        return realized

    def generate_order_id(self, arena_id: int) -> int:
        """生成跨竞技场唯一的订单 ID

        订单 ID 结构：
        - 高 16 位：arena_id
        - 中 32 位：agent_id
        - 低 16 位：order_counter

        Args:
            arena_id: 竞技场 ID

        Returns:
            唯一的订单 ID
        """
        self.order_counter += 1
        # arena_id 占高 16 位，agent_id 占中 32 位，counter 占低 16 位
        return (arena_id << 48) | (self.agent_id << 16) | (self.order_counter & 0xFFFF)


@dataclass
class CatfishAccountState:
    """鲶鱼账户状态

    与 CatfishBase 对象解耦，每个竞技场维护独立副本。
    包含鲶鱼交易相关的状态信息和特有的策略状态。

    Attributes:
        catfish_id: 鲶鱼 ID（负数）
        catfish_mode: 鲶鱼类型（趋势创造者/均值回归/随机/做市）
        balance: 当前余额
        position_quantity: 持仓数量
        position_avg_price: 持仓均价
        realized_pnl: 已实现盈亏
        leverage: 杠杆倍数
        maintenance_margin_rate: 维持保证金率
        initial_balance: 初始余额
        is_liquidated: 是否已被强平
        order_counter: 订单计数器
        current_direction: 趋势创造者方向（1=买，-1=卖）
        ema: 均值回归 EMA 值
        ema_initialized: EMA 是否已初始化
        ma_period: 均线周期
        deviation_threshold: 偏离阈值
        action_probability: 每个 tick 行动的概率（0-1）
    """

    catfish_id: int
    catfish_mode: CatfishMode
    balance: float
    position_quantity: int
    position_avg_price: float
    realized_pnl: float
    leverage: float
    maintenance_margin_rate: float
    initial_balance: float
    is_liquidated: bool
    order_counter: int
    current_direction: int
    ema: float
    ema_initialized: bool
    ma_period: int
    deviation_threshold: float
    action_probability: float

    @classmethod
    def from_catfish(cls, catfish: "CatfishBase") -> "CatfishAccountState":
        """从 CatfishBase 对象创建状态副本

        提取鲶鱼的核心状态，创建独立副本。

        Args:
            catfish: CatfishBase 对象

        Returns:
            新创建的 CatfishAccountState 实例
        """
        account = catfish.account
        config = catfish.config

        # 获取特有状态（根据鲶鱼类型）
        current_direction = getattr(catfish, "_current_direction", 0)
        ema = getattr(catfish, "_ema", 0.0)
        ema_initialized = getattr(catfish, "_ema_initialized", False)

        # 根据类名推断鲶鱼模式
        class_name = catfish.__class__.__name__

        if "TrendCreator" in class_name or "TrendFollowing" in class_name:
            catfish_mode = CatfishMode.TREND_CREATOR
        elif "MeanReversion" in class_name:
            catfish_mode = CatfishMode.MEAN_REVERSION
        elif "RandomTrading" in class_name:
            catfish_mode = CatfishMode.RANDOM
        else:
            catfish_mode = config.mode

        return cls(
            catfish_id=catfish.catfish_id,
            catfish_mode=catfish_mode,
            balance=account.balance,
            position_quantity=account.position.quantity,
            position_avg_price=account.position.avg_price,
            realized_pnl=account.position.realized_pnl,
            leverage=account.leverage,
            maintenance_margin_rate=account.maintenance_margin_rate,
            initial_balance=account.initial_balance,
            is_liquidated=catfish.is_liquidated,
            order_counter=abs(catfish._next_order_id),
            current_direction=current_direction,
            ema=ema,
            ema_initialized=ema_initialized,
            ma_period=config.ma_period,
            deviation_threshold=config.deviation_threshold,
            action_probability=config.action_probability,
        )

    def reset(self, initial_balance: float) -> None:
        """重置到初始状态

        用于 Episode 开始时重置鲶鱼状态。
        趋势创造者会重新随机选择方向，确保每个竞技场独立随机。

        Args:
            initial_balance: 初始余额
        """
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position_quantity = 0
        self.position_avg_price = 0.0
        self.realized_pnl = 0.0
        self.is_liquidated = False
        self.order_counter = 0
        # 趋势创造者重新随机选择方向（每个竞技场独立）
        if self.catfish_mode == CatfishMode.TREND_CREATOR:
            self.current_direction = random.choice([1, -1])
        else:
            self.current_direction = 0
        self.ema = 0.0
        self.ema_initialized = False

    def get_equity(self, current_price: float) -> float:
        """计算净值

        净值 = 余额 + 浮动盈亏

        Args:
            current_price: 当前市场价格

        Returns:
            账户净值
        """
        unrealized_pnl = (current_price - self.position_avg_price) * self.position_quantity
        return self.balance + unrealized_pnl

    def get_margin_ratio(self, current_price: float) -> float:
        """计算保证金率

        保证金率 = 净值 / 持仓市值

        Args:
            current_price: 当前市场价格

        Returns:
            保证金率
        """
        equity = self.get_equity(current_price)
        position_value = abs(self.position_quantity) * current_price
        if position_value == 0:
            return float("inf")
        return equity / position_value

    def check_liquidation(self, current_price: float) -> bool:
        """检查是否需要强平

        鲶鱼仅在资金归零时强平。

        Args:
            current_price: 当前市场价格

        Returns:
            True 表示需要强平
        """
        equity = self.get_equity(current_price)
        return equity <= 0

    def on_trade(
        self,
        trade_price: float,
        trade_quantity: int,
        is_buyer: bool,
    ) -> float:
        """处理成交，更新持仓和余额

        鲶鱼手续费为 0。

        Args:
            trade_price: 成交价格
            trade_quantity: 成交数量
            is_buyer: 是否为买方

        Returns:
            本次成交产生的已实现盈亏
        """
        side = 1 if is_buyer else -1
        realized_pnl = self._update_position(side, trade_quantity, trade_price)
        self.balance += realized_pnl
        return realized_pnl

    def _update_position(self, side: int, quantity: int, price: float) -> float:
        """更新持仓，返回已实现盈亏

        与 AgentAccountState._update_position 相同的逻辑。

        Args:
            side: 成交方向（1=买入，-1=卖出）
            quantity: 成交数量
            price: 成交价格

        Returns:
            本次成交产生的已实现盈亏
        """
        realized: float = 0.0

        if self.position_quantity == 0:
            self.position_quantity = quantity * side
            self.position_avg_price = price
            return 0.0

        if self.position_quantity > 0:
            if side == 1:
                total_cost = self.position_quantity * self.position_avg_price + quantity * price
                self.position_quantity += quantity
                self.position_avg_price = total_cost / self.position_quantity
            else:
                if quantity < self.position_quantity:
                    realized = (price - self.position_avg_price) * quantity
                    self.position_quantity -= quantity
                elif quantity == self.position_quantity:
                    realized = (price - self.position_avg_price) * self.position_quantity
                    self.position_quantity = 0
                    self.position_avg_price = 0.0
                else:
                    realized = (price - self.position_avg_price) * self.position_quantity
                    remaining = quantity - self.position_quantity
                    self.position_quantity = -remaining
                    self.position_avg_price = price
        else:
            if side == -1:
                abs_qty = abs(self.position_quantity)
                total_cost = abs_qty * self.position_avg_price + quantity * price
                self.position_quantity -= quantity
                self.position_avg_price = total_cost / abs(self.position_quantity)
            else:
                abs_qty = abs(self.position_quantity)
                if quantity < abs_qty:
                    realized = (self.position_avg_price - price) * quantity
                    self.position_quantity += quantity
                elif quantity == abs_qty:
                    realized = (self.position_avg_price - price) * abs_qty
                    self.position_quantity = 0
                    self.position_avg_price = 0.0
                else:
                    realized = (self.position_avg_price - price) * abs_qty
                    remaining = quantity - abs_qty
                    self.position_quantity = remaining
                    self.position_avg_price = price

        self.realized_pnl += realized
        return realized

    def generate_order_id(self, arena_id: int) -> int:
        """生成跨竞技场唯一的订单 ID

        鲶鱼订单 ID 使用负数空间。

        Args:
            arena_id: 竞技场 ID

        Returns:
            唯一的订单 ID（负数）
        """
        self.order_counter += 1
        # 使用负数空间
        return -(arena_id * 1_000_000_000 + abs(self.catfish_id) * 1_000_000 + self.order_counter)

    def update_ema(self, price: float, ma_period: int) -> None:
        """更新 EMA 值

        用于均值回归策略。

        Args:
            price: 当前价格
            ma_period: 均线周期
        """
        if not self.ema_initialized:
            self.ema = price
            self.ema_initialized = True
        else:
            alpha = 2.0 / (ma_period + 1)
            self.ema = alpha * price + (1 - alpha) * self.ema

    def can_act(self) -> bool:
        """检查是否可以行动（随机概率判断）

        Returns:
            是否可以行动
        """
        return random.random() < self.action_probability

    def decide(self, tick: int, price_history: Sequence[float]) -> tuple[bool, int]:
        """决策是否行动以及行动方向

        根据鲶鱼类型执行不同的决策逻辑：
        - TREND_CREATOR: 随机概率决定是否行动，保持当前方向
        - MEAN_REVERSION: 价格偏离 EMA 时反向操作，随机概率决定是否行动
        - RANDOM: 随机概率决定是否行动，方向也随机

        Args:
            tick: 当前 tick
            price_history: 历史价格列表

        Returns:
            (should_act, direction): 是否行动和方向（1=买，-1=卖）
        """
        if self.catfish_mode == CatfishMode.TREND_CREATOR:
            # 趋势创造者：使用更高的固定行动概率（50%），确保趋势能够形成
            # 不使用共用的 action_probability，因为趋势创造者的方向是固定的，需要更高的行动频率
            if random.random() >= 0.5:
                return False, 0
            return True, self.current_direction

        elif self.catfish_mode == CatfishMode.MEAN_REVERSION:
            # 均值回归：检查价格偏离
            if len(price_history) == 0:
                return False, 0

            current_price = price_history[-1]

            # 更新 EMA
            self.update_ema(current_price, self.ma_period)

            # 需要至少 ma_period 个数据点
            if len(price_history) < self.ma_period:
                return False, 0

            if self.ema <= 0:
                return False, 0

            # 计算价格偏离率
            deviation = (current_price - self.ema) / self.ema

            # 检查是否超过阈值
            if abs(deviation) < self.deviation_threshold:
                return False, 0

            # 随机概率判断是否行动
            if not self.can_act():
                return False, 0

            # 逆势操作：价格高于均线则卖出，价格低于均线则买入
            direction = -1 if deviation > 0 else 1
            return True, direction

        elif self.catfish_mode == CatfishMode.RANDOM:
            # 随机交易：随机概率决定是否行动
            if not self.can_act():
                return False, 0

            direction = 1 if random.random() < 0.5 else -1
            return True, direction

        return False, 0


@dataclass
class ArenaState:
    """单个竞技场的独立状态

    封装竞技场运行所需的所有状态，与其他竞技场完全隔离。

    Attributes:
        arena_id: 竞技场 ID
        matching_engine: 撮合引擎实例
        adl_manager: ADL 自动减仓管理器实例
        agent_states: Agent 账户状态字典（agent_id -> AgentAccountState）
        catfish_states: 鲶鱼账户状态字典（catfish_id -> CatfishAccountState）
        recent_trades: 最近成交记录队列
        price_history: 价格历史列表
        tick_history_prices: Tick 历史价格队列（maxlen=100，自动截断）
        tick_history_volumes: Tick 历史成交量队列（maxlen=100，自动截断）
        tick_history_amounts: Tick 历史成交额队列（maxlen=100，自动截断）
        smooth_mid_price: 平滑中间价（EMA）
        tick: 当前 tick 数
        pop_liquidated_counts: 各种群已强平 Agent 数量
        eliminating_agents: 正在被强平的 Agent ID 集合
        episode_high_price: Episode 最高价
        episode_low_price: Episode 最低价
        catfish_liquidated: 鲶鱼是否已被强平
        end_reason: Episode 结束原因（None=正常结束, "catfish"=鲶鱼强平,
            "pop:TYPE"=某种群不足, "orderbook"=订单簿单边）
        end_tick: Episode 结束时的 tick 数
    """

    arena_id: int
    matching_engine: "MatchingEngine"
    adl_manager: "ADLManager"
    agent_states: dict[int, AgentAccountState]
    catfish_states: dict[int, CatfishAccountState]
    recent_trades: deque[object] = field(default_factory=lambda: deque(maxlen=100))
    price_history: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    tick_history_prices: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    tick_history_volumes: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    tick_history_amounts: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    smooth_mid_price: float = 0.0
    tick: int = 0
    pop_liquidated_counts: dict[AgentType, int] = field(default_factory=dict)
    eliminating_agents: set[int] = field(default_factory=set)
    episode_high_price: float = 0.0
    episode_low_price: float = float("inf")
    catfish_liquidated: bool = False
    end_reason: str | None = None
    end_tick: int = 0
    consecutive_one_sided_ticks: int = 0  # 连续单边订单簿 tick 计数

    # 扁平化数组（进化后初始化，用于向量化强平检查）
    _num_agents: int = 0
    _balances: np.ndarray | None = None  # shape (num_agents,)
    _position_quantities: np.ndarray | None = None
    _position_avg_prices: np.ndarray | None = None
    _leverages: np.ndarray | None = None
    _maintenance_margins: np.ndarray | None = None
    _is_liquidated_flags: np.ndarray | None = None
    _agent_id_to_idx: dict[int, int] | None = None
    _idx_to_agent_id: np.ndarray | None = None

    def init_flat_arrays(self) -> None:
        """从 agent_states 初始化扁平化数组

        初始化用于向量化计算的 NumPy 数组，将 agent_states 字典中的状态
        映射到连续内存中，为高效的向量化强平检查做准备。
        """
        n = len(self.agent_states)
        self._num_agents = n
        self._balances = np.zeros(n, dtype=np.float64)
        self._position_quantities = np.zeros(n, dtype=np.float64)
        self._position_avg_prices = np.zeros(n, dtype=np.float64)
        self._leverages = np.zeros(n, dtype=np.float64)
        self._maintenance_margins = np.zeros(n, dtype=np.float64)
        self._is_liquidated_flags = np.zeros(n, dtype=np.bool_)
        self._agent_id_to_idx = {}
        self._idx_to_agent_id = np.zeros(n, dtype=np.int64)

        for idx, (agent_id, state) in enumerate(self.agent_states.items()):
            self._agent_id_to_idx[agent_id] = idx
            self._idx_to_agent_id[idx] = agent_id
            self._sync_to_array(state, idx)

    def _sync_to_array(self, state: AgentAccountState, idx: int) -> None:
        """同步单个状态到数组

        Args:
            state: Agent 账户状态
            idx: 状态在数组中的索引
        """
        if self._balances is None:
            return
        self._balances[idx] = state.balance
        self._position_quantities[idx] = state.position_quantity
        self._position_avg_prices[idx] = state.position_avg_price
        self._leverages[idx] = state.leverage
        self._maintenance_margins[idx] = state.maintenance_margin_rate
        self._is_liquidated_flags[idx] = state.is_liquidated

    def sync_state_to_array(self, agent_id: int) -> None:
        """同步指定 agent 到数组

        Args:
            agent_id: Agent ID
        """
        if self._agent_id_to_idx is None:
            return
        idx = self._agent_id_to_idx.get(agent_id)
        if idx is not None:
            self._sync_to_array(self.agent_states[agent_id], idx)

    def reset_episode(self, initial_price: float) -> None:
        """重置 Episode 状态

        Args:
            initial_price: 初始价格
        """
        self.recent_trades.clear()
        self.price_history.clear()
        self.price_history.append(initial_price)
        self.tick_history_prices.clear()
        self.tick_history_volumes.clear()
        self.tick_history_amounts.clear()
        self.smooth_mid_price = initial_price
        self.tick = 0
        self.pop_liquidated_counts = {agent_type: 0 for agent_type in AgentType}
        self.eliminating_agents.clear()
        self.episode_high_price = initial_price
        self.episode_low_price = initial_price
        self.catfish_liquidated = False
        self.end_reason = None
        self.end_tick = 0
        self.consecutive_one_sided_ticks = 0

    def get_agent_state(self, agent_id: int) -> AgentAccountState | None:
        """获取 Agent 状态

        Args:
            agent_id: Agent ID

        Returns:
            AgentAccountState 或 None
        """
        return self.agent_states.get(agent_id)

    def get_catfish_state(self, catfish_id: int) -> CatfishAccountState | None:
        """获取鲶鱼状态

        Args:
            catfish_id: 鲶鱼 ID

        Returns:
            CatfishAccountState 或 None
        """
        return self.catfish_states.get(catfish_id)

    def update_price_stats(self, price: float) -> None:
        """更新价格统计信息

        Args:
            price: 当前价格
        """
        if price > self.episode_high_price:
            self.episode_high_price = price
        if price < self.episode_low_price:
            self.episode_low_price = price
        self.price_history.append(price)

    def mark_agent_liquidated(self, agent_id: int, agent_type: AgentType) -> None:
        """标记 Agent 已被强平

        Args:
            agent_id: Agent ID
            agent_type: Agent 类型
        """
        agent_state = self.agent_states.get(agent_id)
        if agent_state:
            agent_state.is_liquidated = True
        self.pop_liquidated_counts[agent_type] = (
            self.pop_liquidated_counts.get(agent_type, 0) + 1
        )


# ============================================================================
# 辅助函数：基于 AgentAccountState 计算订单数量和倾斜因子
# ============================================================================


def calculate_order_quantity_from_state(
    state: AgentAccountState,
    price: float,
    ratio: float,
    is_buy: bool = True,
    ref_price: float = 0.0,
) -> int:
    """根据 AgentAccountState 计算订单数量

    与 Agent._calculate_order_quantity() 逻辑完全一致，
    但使用 AgentAccountState 而非 agent.account。

    Args:
        state: Agent 账户状态
        price: 订单价格
        ratio: 数量比例（0.0 到 1.0）
        is_buy: 是否为买入方向
        ref_price: 参考价格（用于计算 equity 和仓位价值，默认 0 使用 price）

    Returns:
        订单数量（整数）
    """
    MAX_ORDER_QUANTITY = 100_000_000

    calc_price = ref_price if ref_price > 0 else price
    equity = state.get_equity(calc_price)

    if equity <= 0:
        return 0

    max_pos_value = equity * state.leverage
    current_pos = state.position_quantity
    current_pos_value = abs(current_pos) * calc_price

    if is_buy:
        if current_pos >= 0:
            available_pos_value = max(0, max_pos_value - current_pos_value)
        else:
            available_pos_value = current_pos_value + max_pos_value
    else:
        if current_pos <= 0:
            available_pos_value = max(0, max_pos_value - current_pos_value)
        else:
            available_pos_value = current_pos_value + max_pos_value

    ratio = min(1.0, ratio)
    quantity = (available_pos_value * ratio) / price if price > 0 else 0.0

    if quantity < 1:
        return 0
    return min(MAX_ORDER_QUANTITY, int(quantity))


def calculate_skew_factor_from_state(
    state: AgentAccountState,
    mid_price: float,
) -> float:
    """根据 AgentAccountState 计算做市商仓位倾斜因子

    与 MarketMakerAgent._calculate_skew_factor() 逻辑完全一致。

    Args:
        state: Agent 账户状态
        mid_price: 中间价

    Returns:
        倾斜因子，范围 [-1, 1]
    """
    equity = state.get_equity(mid_price)
    if equity <= 0:
        return 0.0

    position_qty = state.position_quantity
    if position_qty == 0:
        return 0.0

    position_value = abs(position_qty) * mid_price
    max_position_value = equity * state.leverage
    pos_ratio = (
        min(1.0, position_value / max_position_value)
        if max_position_value > 0
        else 0.0
    )

    if position_qty > 0:
        return -pos_ratio
    else:
        return pos_ratio


# ============================================================================
# 适配器类：使 AgentAccountState 兼容 BatchNetworkCache.decide()
# ============================================================================


class PositionAdapter:
    """将 AgentAccountState 的持仓数据适配为 Position-like 接口

    BatchNetworkCache._extract_agents_to_batch() 需要访问 agent.account.position
    """

    __slots__ = ("quantity", "avg_price")

    def __init__(self, state: AgentAccountState) -> None:
        self.quantity: int = state.position_quantity
        self.avg_price: float = state.position_avg_price


class AccountAdapter:
    """将 AgentAccountState 适配为 Account-like 接口

    BatchNetworkCache._extract_agents_to_batch() 需要访问:
    - agent.account.balance
    - agent.account.position
    - agent.account.get_margin_ratio()
    - agent.account.get_equity()
    """

    __slots__ = ("_state", "balance", "position")

    def __init__(self, state: AgentAccountState) -> None:
        self._state = state
        self.balance: float = state.balance
        self.position = PositionAdapter(state)

    def get_margin_ratio(self, current_price: float) -> float:
        """计算保证金率"""
        return self._state.get_margin_ratio(current_price)

    def get_equity(self, current_price: float) -> float:
        """计算净值"""
        return self._state.get_equity(current_price)


class AgentStateAdapter:
    """将 AgentAccountState 适配为 Agent-like 接口，供 BatchNetworkCache.decide() 使用

    这是一个轻量级适配器，允许在不修改 Cython 代码的情况下，
    使用 AgentAccountState 进行批量推理。

    使用示例:
        adapters = [AgentStateAdapter(state) for state in arena.agent_states.values()]
        decisions = cache.decide(adapters, market_state)
    """

    __slots__ = ("account", "is_liquidated", "agent_type", "agent_id")

    def __init__(self, state: AgentAccountState) -> None:
        self.account = AccountAdapter(state)
        self.is_liquidated: bool = state.is_liquidated
        self.agent_type: AgentType = state.agent_type
        self.agent_id: int = state.agent_id
