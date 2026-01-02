"""ADL (Auto-Deleveraging) 自动减仓管理器"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.bio.agents.base import Agent


@dataclass
class ADLCandidate:
    """ADL 候选者信息"""

    agent: "Agent"
    position_qty: int  # 持仓数量（正=多头，负=空头）
    pnl_percent: float  # 盈亏百分比
    effective_leverage: float  # 有效杠杆
    adl_score: float  # ADL 排名分数（越高越优先）


class ADLManager:
    """ADL 自动减仓管理器

    当强平订单无法完全成交时，触发 ADL 机制。
    按照 ADL 排名（盈利百分比 * 有效杠杆）选择对手方强制减仓。
    """

    def __init__(self) -> None:
        from src.core.log_engine.logger import get_logger

        self.logger = get_logger("adl")

    def calculate_bankruptcy_price(
        self,
        agent: "Agent",
        current_price: float,
    ) -> float:
        """计算破产价格（净值归零时的价格）

        多头：bankruptcy_price = avg_price - balance / |quantity|
        空头：bankruptcy_price = avg_price + balance / |quantity|

        Args:
            agent: Agent 对象
            current_price: 当前市场价格

        Returns:
            破产价格（最小为 0.01）
        """
        position = agent.account.position
        balance = agent.account.balance
        quantity = position.quantity

        # 无持仓时返回当前价格
        if quantity == 0:
            return current_price

        avg_price = position.avg_price
        abs_quantity = abs(quantity)

        if quantity > 0:
            # 多头：价格下跌到破产价时净值归零
            # 净值 = balance + (price - avg_price) * quantity = 0
            # price = avg_price - balance / quantity
            bankruptcy_price = avg_price - balance / abs_quantity
        else:
            # 空头：价格上涨到破产价时净值归零
            # 净值 = balance + (avg_price - price) * |quantity| = 0
            # price = avg_price + balance / |quantity|
            bankruptcy_price = avg_price + balance / abs_quantity

        # 破产价格下限为 0.01
        return max(0.01, bankruptcy_price)

    def calculate_adl_score(
        self,
        agent: "Agent",
        current_price: float,
    ) -> ADLCandidate | None:
        """计算单个 Agent 的 ADL 排名分数

        排名公式：
        - 如果 PnL% > 0：排名 = PnL% * 有效杠杆
        - 如果 PnL% <= 0：排名 = PnL% / 有效杠杆

        其中：
        - PnL% = 浮动盈亏 / |开仓成本|
        - 有效杠杆 = |持仓市值| / 净值

        Args:
            agent: Agent 对象
            current_price: 当前市场价格

        Returns:
            ADLCandidate 对象，无持仓时返回 None
        """
        position = agent.account.position
        quantity = position.quantity

        # 无持仓，不参与 ADL
        if quantity == 0:
            return None

        avg_price = position.avg_price
        abs_quantity = abs(quantity)

        # 计算开仓成本
        entry_cost = abs_quantity * avg_price
        if entry_cost <= 0:
            return None

        # 计算浮动盈亏
        unrealized_pnl = position.get_unrealized_pnl(current_price)

        # 计算盈亏百分比
        pnl_percent = unrealized_pnl / entry_cost

        # 计算净值
        equity = agent.account.get_equity(current_price)
        if equity <= 0:
            # 净值为非正数，有效杠杆设为极大值
            effective_leverage = float("inf")
        else:
            # 有效杠杆 = |持仓市值| / 净值
            position_value = abs_quantity * current_price
            effective_leverage = position_value / equity

        # 计算 ADL 排名分数
        if pnl_percent > 0:
            # 盈利者：分数 = 盈利百分比 * 有效杠杆
            adl_score = pnl_percent * effective_leverage
        else:
            # 亏损者：分数 = 盈利百分比 / 有效杠杆
            # 有效杠杆越高，分数越低（越不优先被选中）
            if effective_leverage == float("inf"):
                adl_score = float("-inf")
            elif effective_leverage > 0:
                adl_score = pnl_percent / effective_leverage
            else:
                adl_score = float("-inf")

        return ADLCandidate(
            agent=agent,
            position_qty=quantity,
            pnl_percent=pnl_percent,
            effective_leverage=effective_leverage,
            adl_score=adl_score,
        )

    def get_adl_candidates(
        self,
        agents: list["Agent"],
        current_price: float,
        target_side: int,  # 1=需要多头对手，-1=需要空头对手
        exclude_agent_id: int | None = None,
    ) -> list[ADLCandidate]:
        """获取 ADL 候选列表，按排名从高到低排序

        Args:
            agents: 所有 Agent 列表
            current_price: 当前市场价格
            target_side: 需要的对手方向（1=多头，-1=空头）
            exclude_agent_id: 要排除的 Agent ID（通常是被强平的那个）

        Returns:
            按 ADL 分数从高到低排序的候选列表
        """
        candidates: list[ADLCandidate] = []

        for agent in agents:
            # 排除被强平的 Agent
            if exclude_agent_id is not None and agent.agent_id == exclude_agent_id:
                continue

            # 排除已淘汰的 Agent
            if agent.is_liquidated:
                continue

            # 计算 ADL 分数
            candidate = self.calculate_adl_score(agent, current_price)
            if candidate is None:
                continue

            # 筛选持有反方向仓位的 Agent
            # target_side=1 表示需要多头对手（即持有多头仓位的 Agent）
            # target_side=-1 表示需要空头对手（即持有空头仓位的 Agent）
            if target_side > 0 and candidate.position_qty <= 0:
                continue
            if target_side < 0 and candidate.position_qty >= 0:
                continue

            candidates.append(candidate)

        # 按 ADL 分数从高到低排序
        candidates.sort(key=lambda c: c.adl_score, reverse=True)

        return candidates

    def execute_adl(
        self,
        liquidated_agent: "Agent",
        remaining_qty: int,
        candidates: list[ADLCandidate],
        bankruptcy_price: float,
        current_price: float,
    ) -> list[tuple["Agent", int, float]]:
        """执行 ADL 成交

        按照候选列表顺序，逐个与被强平仓位成交，直到仓位全部平完。

        Args:
            liquidated_agent: 被强平的 Agent
            remaining_qty: 剩余需要平仓的数量（正数）
            candidates: ADL 候选列表（已按排名排序）
            bankruptcy_price: 成交价格（破产价格）
            current_price: 当前市场价格（仅用于日志）

        Returns:
            ADL 成交列表：[(对手方Agent, 成交数量, 成交价格), ...]
        """
        adl_trades: list[tuple["Agent", int, float]] = []

        if remaining_qty <= 0:
            return adl_trades

        self.logger.info(
            f"ADL 触发: Agent {liquidated_agent.agent_id} "
            f"剩余平仓量 {remaining_qty}, "
            f"破产价 {bankruptcy_price:.2f}, "
            f"当前价 {current_price:.2f}, "
            f"候选人数 {len(candidates)}"
        )

        for candidate in candidates:
            if remaining_qty <= 0:
                break

            # 计算可减仓数量（取对手方持仓绝对值和剩余需求的较小值）
            counter_position = abs(candidate.position_qty)
            trade_qty = min(counter_position, remaining_qty)

            if trade_qty <= 0:
                continue

            # 记录 ADL 成交
            adl_trades.append((candidate.agent, trade_qty, bankruptcy_price))

            self.logger.info(
                f"ADL 成交: Agent {candidate.agent.agent_id} "
                f"被减仓 {trade_qty}, "
                f"成交价 {bankruptcy_price:.2f}, "
                f"原持仓 {candidate.position_qty}, "
                f"ADL分数 {candidate.adl_score:.4f}"
            )

            remaining_qty -= trade_qty

        if remaining_qty > 0:
            self.logger.warning(
                f"ADL 未能完全平仓: Agent {liquidated_agent.agent_id} "
                f"剩余 {remaining_qty} 无法匹配"
            )

        return adl_trades
