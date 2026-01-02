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

    account = Account(agent_id, AgentType.RETAIL, config)
    account.balance = balance
    if position_qty != 0:
        side = OrderSide.BUY if position_qty > 0 else OrderSide.SELL
        account.position.update(side.value, abs(position_qty), avg_price)

    agent.account = account
    return agent


class TestCalculateBankruptcyPrice:
    """测试破产价格计算"""

    def test_long_position_bankruptcy_price(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """测试多头破产价格"""
        # 多头：balance=1000, qty=100, avg=100
        # bankruptcy_price = avg_price - balance / qty = 100 - 1000/100 = 90
        agent = create_mock_agent(1, 1000.0, 100, 100.0, retail_config)
        bankruptcy_price = adl_manager.calculate_bankruptcy_price(agent, 100.0)
        assert abs(bankruptcy_price - 90.0) < 0.01

    def test_short_position_bankruptcy_price(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """测试空头破产价格"""
        # 空头：balance=1000, qty=-100, avg=100
        # bankruptcy_price = avg_price + balance / |qty| = 100 + 1000/100 = 110
        agent = create_mock_agent(1, 1000.0, -100, 100.0, retail_config)
        bankruptcy_price = adl_manager.calculate_bankruptcy_price(agent, 100.0)
        assert abs(bankruptcy_price - 110.0) < 0.01

    def test_no_position_returns_current_price(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """无持仓时返回当前价格"""
        agent = create_mock_agent(1, 10000.0, 0, 0.0, retail_config)
        bankruptcy_price = adl_manager.calculate_bankruptcy_price(agent, 100.0)
        assert bankruptcy_price == 100.0

    def test_bankruptcy_price_minimum(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """破产价格最小值为 0.01"""
        # 如果计算出的破产价格为负，应返回 0.01
        agent = create_mock_agent(1, 20000.0, 100, 100.0, retail_config)
        # bankruptcy = 100 - 20000/100 = 100 - 200 = -100 -> 应返回 0.01
        bankruptcy_price = adl_manager.calculate_bankruptcy_price(agent, 100.0)
        assert bankruptcy_price == 0.01


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


class TestGetADLCandidates:
    """测试获取 ADL 候选列表"""

    def test_filter_by_position_direction(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """测试按持仓方向筛选"""
        agents = [
            create_mock_agent(1, 10000.0, 100, 100.0, retail_config),   # 多头
            create_mock_agent(2, 10000.0, -100, 100.0, retail_config),  # 空头
            create_mock_agent(3, 10000.0, 50, 100.0, retail_config),    # 多头
        ]

        # 需要多头对手
        candidates = adl_manager.get_adl_candidates(
            agents=agents, current_price=110.0, target_side=1, exclude_agent_id=None
        )

        assert len(candidates) == 2
        for c in candidates:
            assert c.position_qty > 0

    def test_exclude_agent_id(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """测试排除指定 Agent"""
        agents = [
            create_mock_agent(1, 10000.0, 100, 100.0, retail_config),
            create_mock_agent(2, 10000.0, 100, 100.0, retail_config),
        ]

        candidates = adl_manager.get_adl_candidates(
            agents=agents, current_price=110.0, target_side=1, exclude_agent_id=1
        )

        assert len(candidates) == 1
        assert candidates[0].agent.agent_id == 2

    def test_include_liquidated_agents_with_position(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """测试包含已淘汰但仍持有仓位的 Agent（保持多空对等）"""
        agents = [
            create_mock_agent(1, 10000.0, 100, 100.0, retail_config),
            create_mock_agent(2, 10000.0, 100, 100.0, retail_config),
        ]
        agents[1].is_liquidated = True  # 已淘汰但仍持有仓位

        candidates = adl_manager.get_adl_candidates(
            agents=agents, current_price=110.0, target_side=1, exclude_agent_id=None
        )

        # 应包含两个 Agent，因为已淘汰的 Agent 仍持有仓位
        assert len(candidates) == 2

    def test_sorted_by_adl_score(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """测试按 ADL 分数排序"""
        agents = [
            create_mock_agent(1, 10000.0, 100, 100.0, retail_config),   # 盈利较少
            create_mock_agent(2, 10000.0, 100, 80.0, retail_config),    # 盈利较多（开仓成本低）
        ]

        candidates = adl_manager.get_adl_candidates(
            agents=agents, current_price=110.0, target_side=1, exclude_agent_id=None
        )

        # 应按分数从高到低排序
        assert len(candidates) == 2
        assert candidates[0].adl_score >= candidates[1].adl_score


class TestExecuteADL:
    """测试执行 ADL"""

    def test_full_adl(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """测试完全 ADL（候选足够）"""
        liquidated_agent = create_mock_agent(1, 0.0, 100, 100.0, retail_config)

        candidate = ADLCandidate(
            agent=create_mock_agent(2, 10000.0, -100, 100.0, retail_config),
            position_qty=-100,
            pnl_percent=0.1,
            effective_leverage=1.0,
            adl_score=0.1,
        )

        adl_trades = adl_manager.execute_adl(
            liquidated_agent=liquidated_agent,
            remaining_qty=100,
            candidates=[candidate],
            bankruptcy_price=90.0,
            current_price=100.0,
        )

        assert len(adl_trades) == 1
        assert adl_trades[0][1] == 100  # 成交数量
        assert adl_trades[0][2] == 90.0  # 成交价格

    def test_partial_adl(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """测试部分 ADL（候选不足）"""
        liquidated_agent = create_mock_agent(1, 0.0, 100, 100.0, retail_config)

        candidate = ADLCandidate(
            agent=create_mock_agent(2, 10000.0, -50, 100.0, retail_config),
            position_qty=-50,
            pnl_percent=0.1,
            effective_leverage=1.0,
            adl_score=0.1,
        )

        adl_trades = adl_manager.execute_adl(
            liquidated_agent=liquidated_agent,
            remaining_qty=100,
            candidates=[candidate],
            bankruptcy_price=90.0,
            current_price=100.0,
        )

        assert len(adl_trades) == 1
        assert adl_trades[0][1] == 50  # 只能成交 50

    def test_multiple_candidates(
        self, adl_manager: ADLManager, retail_config: AgentConfig
    ):
        """测试多个候选"""
        liquidated_agent = create_mock_agent(1, 0.0, 100, 100.0, retail_config)

        candidates = [
            ADLCandidate(
                agent=create_mock_agent(2, 10000.0, -40, 100.0, retail_config),
                position_qty=-40,
                pnl_percent=0.2,
                effective_leverage=1.0,
                adl_score=0.2,
            ),
            ADLCandidate(
                agent=create_mock_agent(3, 10000.0, -30, 100.0, retail_config),
                position_qty=-30,
                pnl_percent=0.1,
                effective_leverage=1.0,
                adl_score=0.1,
            ),
        ]

        adl_trades = adl_manager.execute_adl(
            liquidated_agent=liquidated_agent,
            remaining_qty=60,
            candidates=candidates,
            bankruptcy_price=90.0,
            current_price=100.0,
        )

        assert len(adl_trades) == 2
        assert adl_trades[0][1] == 40  # 第一个候选全部减仓
        assert adl_trades[1][1] == 20  # 第二个候选部分减仓


class TestAccountOnADLTrade:
    """测试 Account.on_adl_trade"""

    def test_long_position_adl(self, retail_config: AgentConfig):
        """测试多头被 ADL 减仓"""
        account = Account(1, AgentType.RETAIL, retail_config)
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
        account = Account(1, AgentType.RETAIL, retail_config)
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
        account = Account(1, AgentType.RETAIL, retail_config)
        account.position.update(OrderSide.BUY.value, 100, 100.0)

        account.on_adl_trade(100, 90.0, is_taker=True)

        assert account.position.quantity == 0
        assert account.position.avg_price == 0.0
