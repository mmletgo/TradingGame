"""测试账户模块"""

import pytest

from src.config.config import AgentConfig, AgentType
from src.market.account.account import Account
from src.market.account.position import Position
from src.market.orderbook.order import OrderSide


class TestPositionInit:
    """测试 Position.__init__"""

    def test_create_empty_position(self):
        """测试创建空持仓"""
        position = Position()

        assert position.quantity == 0.0
        assert position.avg_price == 0.0
        assert position.realized_pnl == 0.0


class TestPositionUpdate:
    """测试 Position.update"""

    def test_open_long_position(self):
        """测试开多仓"""
        position = Position()

        pnl = position.update(OrderSide.BUY, 10.0, 100.0)

        assert position.quantity == 10.0
        assert position.avg_price == 100.0
        assert position.realized_pnl == 0.0
        assert pnl == 0.0

    def test_open_short_position(self):
        """测试开空仓"""
        position = Position()

        pnl = position.update(OrderSide.SELL, 10.0, 100.0)

        assert position.quantity == -10.0
        assert position.avg_price == 100.0
        assert position.realized_pnl == 0.0
        assert pnl == 0.0

    def test_add_to_long_position(self):
        """测试加多仓"""
        position = Position()
        position.update(OrderSide.BUY, 10.0, 100.0)

        pnl = position.update(OrderSide.BUY, 5.0, 105.0)

        # 新均价 = (10*100 + 5*105) / 15 = 101.666...
        assert position.quantity == 15.0
        assert abs(position.avg_price - 101.6666667) < 0.0001
        assert position.realized_pnl == 0.0
        assert pnl == 0.0

    def test_add_to_short_position(self):
        """测试加空仓"""
        position = Position()
        position.update(OrderSide.SELL, 10.0, 100.0)

        pnl = position.update(OrderSide.SELL, 5.0, 95.0)

        # 新均价 = (10*100 + 5*95) / 15 = 98.333...
        assert position.quantity == -15.0
        assert abs(position.avg_price - 98.3333333) < 0.0001
        assert position.realized_pnl == 0.0
        assert pnl == 0.0

    def test_reduce_long_position(self):
        """测试减多仓"""
        position = Position()
        position.update(OrderSide.BUY, 10.0, 100.0)

        pnl = position.update(OrderSide.SELL, 3.0, 110.0)

        # 盈亏 = 3 * (110 - 100) = 30
        assert position.quantity == 7.0
        assert position.avg_price == 100.0
        assert abs(pnl - 30.0) < 0.01
        assert abs(position.realized_pnl - 30.0) < 0.01

    def test_reduce_short_position(self):
        """测试减空仓"""
        position = Position()
        position.update(OrderSide.SELL, 10.0, 100.0)

        pnl = position.update(OrderSide.BUY, 3.0, 90.0)

        # 盈亏 = 3 * (100 - 90) = 30
        assert position.quantity == -7.0
        assert position.avg_price == 100.0
        assert abs(pnl - 30.0) < 0.01
        assert abs(position.realized_pnl - 30.0) < 0.01

    def test_close_long_position(self):
        """测试完全平多仓"""
        position = Position()
        position.update(OrderSide.BUY, 10.0, 100.0)

        pnl = position.update(OrderSide.SELL, 10.0, 110.0)

        # 盈亏 = 10 * (110 - 100) = 100
        assert position.quantity == 0.0
        assert position.avg_price == 0.0
        assert abs(pnl - 100.0) < 0.01
        assert abs(position.realized_pnl - 100.0) < 0.01

    def test_close_short_position(self):
        """测试完全平空仓"""
        position = Position()
        position.update(OrderSide.SELL, 10.0, 100.0)

        pnl = position.update(OrderSide.BUY, 10.0, 90.0)

        # 盈亏 = 10 * (100 - 90) = 100
        assert position.quantity == 0.0
        assert position.avg_price == 0.0
        assert abs(pnl - 100.0) < 0.01
        assert abs(position.realized_pnl - 100.0) < 0.01

    def test_reverse_from_long_to_short(self):
        """测试反向开仓：多转空"""
        position = Position()
        position.update(OrderSide.BUY, 10.0, 100.0)

        pnl = position.update(OrderSide.SELL, 15.0, 110.0)

        # 平多仓盈亏 = 10 * (110 - 100) = 100
        # 剩余开空 5 @ 110
        assert position.quantity == -5.0
        assert position.avg_price == 110.0
        assert abs(pnl - 100.0) < 0.01
        assert abs(position.realized_pnl - 100.0) < 0.01

    def test_reverse_from_short_to_long(self):
        """测试反向开仓：空转多"""
        position = Position()
        position.update(OrderSide.SELL, 10.0, 100.0)

        pnl = position.update(OrderSide.BUY, 15.0, 90.0)

        # 平空仓盈亏 = 10 * (100 - 90) = 100
        # 剩余开多 5 @ 90
        assert position.quantity == 5.0
        assert position.avg_price == 90.0
        assert abs(pnl - 100.0) < 0.01
        assert abs(position.realized_pnl - 100.0) < 0.01

    def test_multiple_partial_closes(self):
        """测试多次部分平仓"""
        position = Position()
        position.update(OrderSide.BUY, 100.0, 100.0)

        pnl1 = position.update(OrderSide.SELL, 30.0, 110.0)
        pnl2 = position.update(OrderSide.SELL, 50.0, 105.0)

        # 第一次平仓盈亏 = 30 * (110 - 100) = 300
        # 第二次平仓盈亏 = 50 * (105 - 100) = 250
        assert position.quantity == 20.0
        assert position.avg_price == 100.0
        assert abs(pnl1 - 300.0) < 0.01
        assert abs(pnl2 - 250.0) < 0.01
        assert abs(position.realized_pnl - 550.0) < 0.01


class TestPositionGetUnrealizedPnl:
    """测试 Position.get_unrealized_pnl"""

    def test_long_position_profit(self):
        """测试多仓盈利"""
        position = Position()
        position.update(OrderSide.BUY, 10.0, 100.0)

        # 当前价 110，均价 100，数量 10
        # 浮动盈亏 = (110 - 100) * 10 = 100
        unrealized_pnl = position.get_unrealized_pnl(110.0)

        assert abs(unrealized_pnl - 100.0) < 0.01

    def test_long_position_loss(self):
        """测试多仓亏损"""
        position = Position()
        position.update(OrderSide.BUY, 10.0, 100.0)

        # 当前价 90，均价 100，数量 10
        # 浮动盈亏 = (90 - 100) * 10 = -100
        unrealized_pnl = position.get_unrealized_pnl(90.0)

        assert abs(unrealized_pnl - (-100.0)) < 0.01

    def test_short_position_profit(self):
        """测试空仓盈利"""
        position = Position()
        position.update(OrderSide.SELL, 10.0, 100.0)

        # 当前价 90，均价 100，数量 -10
        # 浮动盈亏 = (90 - 100) * (-10) = 100
        unrealized_pnl = position.get_unrealized_pnl(90.0)

        assert abs(unrealized_pnl - 100.0) < 0.01

    def test_short_position_loss(self):
        """测试空仓亏损"""
        position = Position()
        position.update(OrderSide.SELL, 10.0, 100.0)

        # 当前价 110，均价 100，数量 -10
        # 浮动盈亏 = (110 - 100) * (-10) = -100
        unrealized_pnl = position.get_unrealized_pnl(110.0)

        assert abs(unrealized_pnl - (-100.0)) < 0.01

    def test_empty_position(self):
        """测试空仓"""
        position = Position()

        # 空仓时浮动盈亏为 0
        unrealized_pnl = position.get_unrealized_pnl(110.0)

        assert unrealized_pnl == 0.0

    def test_long_position_break_even(self):
        """测试多仓保本"""
        position = Position()
        position.update(OrderSide.BUY, 10.0, 100.0)

        # 当前价等于均价
        unrealized_pnl = position.get_unrealized_pnl(100.0)

        assert unrealized_pnl == 0.0

    def test_short_position_break_even(self):
        """测试空仓保本"""
        position = Position()
        position.update(OrderSide.SELL, 10.0, 100.0)

        # 当前价等于均价
        unrealized_pnl = position.get_unrealized_pnl(100.0)

        assert unrealized_pnl == 0.0


class TestPositionGetMarginUsed:
    """测试 Position.get_margin_used"""

    def test_long_position_margin(self):
        """测试多仓保证金占用"""
        position = Position()
        position.update(OrderSide.BUY, 10.0, 100.0)

        # 保证金 = 10 * 110 / 10 = 110
        margin = position.get_margin_used(110.0, 10.0)

        assert abs(margin - 110.0) < 0.01

    def test_short_position_margin(self):
        """测试空仓保证金占用"""
        position = Position()
        position.update(OrderSide.SELL, 10.0, 100.0)

        # 保证金 = 10 * 90 / 10 = 90
        margin = position.get_margin_used(90.0, 10.0)

        assert abs(margin - 90.0) < 0.01

    def test_empty_position_margin(self):
        """测试空仓保证金占用"""
        position = Position()

        # 空仓保证金为 0
        margin = position.get_margin_used(100.0, 10.0)

        assert margin == 0.0

    def test_high_leverage_reduces_margin(self):
        """测试高杠杆减少保证金占用"""
        position = Position()
        position.update(OrderSide.BUY, 10.0, 100.0)

        # 10倍杠杆: 10 * 100 / 10 = 100
        margin_10x = position.get_margin_used(100.0, 10.0)
        # 100倍杠杆: 10 * 100 / 100 = 10
        margin_100x = position.get_margin_used(100.0, 100.0)

        assert abs(margin_10x - 100.0) < 0.01
        assert abs(margin_100x - 10.0) < 0.01


class TestAccountInit:
    """测试 Account.__init__"""

    def test_create_retail_account(self):
        """测试创建散户账户"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)

        assert account.agent_id == 1
        assert account.agent_type == AgentType.RETAIL
        assert account.balance == 10000.0
        assert account.position.quantity == 0.0
        assert account.position.avg_price == 0.0
        assert account.position.realized_pnl == 0.0
        assert account.leverage == 100.0
        assert account.maintenance_margin_rate == 0.005
        assert account.maker_fee_rate == 0.0002
        assert account.taker_fee_rate == 0.0005
        assert account.pending_order_id is None

    def test_create_whale_account(self):
        """测试创建庄家账户"""
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        account = Account(agent_id=10001, agent_type=AgentType.WHALE, config=config)

        assert account.agent_id == 10001
        assert account.agent_type == AgentType.WHALE
        assert account.balance == 10000000.0
        assert account.leverage == 10.0
        assert account.maintenance_margin_rate == 0.05
        assert account.maker_fee_rate == 0.0
        assert account.taker_fee_rate == 0.0001

    def test_create_market_maker_account(self):
        """测试创建做市商账户"""
        config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        account = Account(
            agent_id=10011, agent_type=AgentType.MARKET_MAKER, config=config
        )

        assert account.agent_id == 10011
        assert account.agent_type == AgentType.MARKET_MAKER
        assert account.balance == 10000000.0
        assert account.leverage == 10.0
        assert account.maintenance_margin_rate == 0.05
        assert account.maker_fee_rate == 0.0
        assert account.taker_fee_rate == 0.0001
