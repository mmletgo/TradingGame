"""ADL (Auto-Deleveraging) 模块测试"""

import pytest
from unittest.mock import MagicMock

from src.config.config import AgentConfig, AgentType
from src.market.adl.adl_manager import ADLCandidate, ADLManager
from src.market.account.account import Account
from src.market.orderbook.order import OrderSide


@pytest.fixture
def adl_manager() -> ADLManager:
    return ADLManager()


@pytest.fixture
def retail_config() -> AgentConfig:
    return AgentConfig(
        count=100,
        initial_balance=10000.0,
        leverage=100.0,
        maintenance_margin_rate=0.005,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
    )


def create_mock_agent(
    agent_id: int,
    balance: float,
    position_qty: int,
    avg_price: float,
    config: AgentConfig,
) -> MagicMock:
    """创建 mock Agent"""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.is_liquidated = False

    account = Account(agent_id, AgentType.RETAIL_PRO, config)
    account.balance = balance
    if position_qty != 0:
        side = OrderSide.BUY if position_qty > 0 else OrderSide.SELL
        account.position.update(side.value, abs(position_qty), avg_price)

    agent.account = account
    return agent


class TestCalculateADLScore:
    """测试 ADL 排名分数计算"""

    def test_profitable_long_position(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """测试盈利多头的 ADL 分数"""
        # 多头开仓 100 @ 100，当前价 110，盈利 10%
        agent = create_mock_agent(1, 10000.0, 100, 100.0, retail_config)
        candidate = adl_manager.calculate_adl_score(agent, 110.0)

        assert candidate is not None
        assert candidate.position_qty == 100
        assert candidate.pnl_percent > 0  # 盈利
        assert candidate.adl_score > 0  # 盈利时分数为正

    def test_losing_long_position(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """测试亏损多头的 ADL 分数"""
        # 多头开仓 100 @ 100，当前价 90，亏损 10%
        agent = create_mock_agent(1, 10000.0, 100, 100.0, retail_config)
        candidate = adl_manager.calculate_adl_score(agent, 90.0)

        assert candidate is not None
        assert candidate.pnl_percent < 0  # 亏损
        assert candidate.adl_score < 0  # 亏损时分数为负

    def test_no_position_returns_none(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """无持仓时返回 None"""
        agent = create_mock_agent(1, 10000.0, 0, 0.0, retail_config)
        candidate = adl_manager.calculate_adl_score(agent, 100.0)
        assert candidate is None


class TestAccountOnADLTrade:
    """测试 Account.on_adl_trade"""

    def test_long_position_adl(self, retail_config: AgentConfig):
        """测试多头被 ADL 减仓"""
        account = Account(1, AgentType.RETAIL_PRO, retail_config)
        account.position.update(OrderSide.BUY.value, 100, 100.0)
        initial_balance = account.balance

        # 以 90 的价格被减仓 50
        realized_pnl = account.on_adl_trade(50, 90.0, is_taker=True)

        # 盈亏 = (90 - 100) * 50 = -500
        assert abs(realized_pnl - (-500.0)) < 0.01
        assert account.position.quantity == 50
        assert abs(account.balance - (initial_balance - 500.0)) < 0.01

    def test_short_position_adl(self, retail_config: AgentConfig):
        """测试空头被 ADL 减仓"""
        account = Account(1, AgentType.RETAIL_PRO, retail_config)
        account.position.update(OrderSide.SELL.value, 100, 100.0)
        initial_balance = account.balance

        # 以 110 的价格被减仓 50
        realized_pnl = account.on_adl_trade(50, 110.0, is_taker=True)

        # 盈亏 = (100 - 110) * 50 = -500
        assert abs(realized_pnl - (-500.0)) < 0.01
        assert account.position.quantity == -50
        assert abs(account.balance - (initial_balance - 500.0)) < 0.01

    def test_position_cleared(self, retail_config: AgentConfig):
        """测试仓位完全清零"""
        account = Account(1, AgentType.RETAIL_PRO, retail_config)
        account.position.update(OrderSide.BUY.value, 100, 100.0)

        account.on_adl_trade(100, 90.0, is_taker=True)

        assert account.position.quantity == 0
        assert account.position.avg_price == 0.0
