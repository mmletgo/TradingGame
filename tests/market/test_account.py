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


class TestAccountGetEquity:
    """测试 Account.get_equity"""

    def test_equity_without_position(self):
        """测试无持仓时净值等于余额"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)

        # 无持仓时，净值 = 余额
        equity = account.get_equity(100.0)

        assert equity == 10000.0

    def test_equity_with_long_position_profit(self):
        """测试多仓盈利时的净值"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 10 @ 100
        account.position.update(OrderSide.BUY, 10.0, 100.0)

        # 当前价 110，浮动盈亏 = (110 - 100) * 10 = 100
        # 净值 = 余额 10000 + 浮动盈亏 100 = 10100
        equity = account.get_equity(110.0)

        assert abs(equity - 10100.0) < 0.01

    def test_equity_with_long_position_loss(self):
        """测试多仓亏损时的净值"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 10 @ 100
        account.position.update(OrderSide.BUY, 10.0, 100.0)

        # 当前价 90，浮动盈亏 = (90 - 100) * 10 = -100
        # 净值 = 余额 10000 + 浮动盈亏 -100 = 9900
        equity = account.get_equity(90.0)

        assert abs(equity - 9900.0) < 0.01

    def test_equity_with_short_position_profit(self):
        """测试空仓盈利时的净值"""
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        account = Account(agent_id=10001, agent_type=AgentType.WHALE, config=config)
        # 开空仓 100 @ 100
        account.position.update(OrderSide.SELL, 100.0, 100.0)

        # 当前价 90，浮动盈亏 = (90 - 100) * (-100) = 1000
        # 净值 = 余额 10000000 + 浮动盈亏 1000 = 10001000
        equity = account.get_equity(90.0)

        assert abs(equity - 10001000.0) < 0.01

    def test_equity_with_short_position_loss(self):
        """测试空仓亏损时的净值"""
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        account = Account(agent_id=10001, agent_type=AgentType.WHALE, config=config)
        # 开空仓 100 @ 100
        account.position.update(OrderSide.SELL, 100.0, 100.0)

        # 当前价 110，浮动盈亏 = (110 - 100) * (-100) = -1000
        # 净值 = 余额 10000000 + 浮动盈亏 -1000 = 9999000
        equity = account.get_equity(110.0)

        assert abs(equity - 9999000.0) < 0.01

    def test_equity_after_balance_change(self):
        """测试余额变化后的净值"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 10 @ 100
        account.position.update(OrderSide.BUY, 10.0, 100.0)
        # 余额增加 500（模拟已实现盈亏）
        account.balance += 500.0

        # 当前价 110，浮动盈亏 = (110 - 100) * 10 = 100
        # 净值 = 余额 10500 + 浮动盈亏 100 = 10600
        equity = account.get_equity(110.0)

        assert abs(equity - 10600.0) < 0.01


class TestAccountGetAvailableMargin:
    """测试 Account.get_available_margin"""

    def test_available_margin_without_position(self):
        """测试无持仓时可用保证金等于净值"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)

        # 无持仓时，占用保证金为 0，可用保证金 = 净值 = 余额
        available_margin = account.get_available_margin(100.0)

        assert available_margin == 10000.0

    def test_available_margin_with_long_position(self):
        """测试持有多仓时的可用保证金"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 10 @ 100
        account.position.update(OrderSide.BUY, 10.0, 100.0)

        # 当前价 110
        # 净值 = 余额 10000 + 浮动盈亏 (110-100)*10 = 10100
        # 占用保证金 = 10 * 110 / 100 = 11
        # 可用保证金 = 10100 - 11 = 10089
        available_margin = account.get_available_margin(110.0)

        assert abs(available_margin - 10089.0) < 0.01

    def test_available_margin_with_short_position(self):
        """测试持有空仓时的可用保证金"""
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        account = Account(agent_id=10001, agent_type=AgentType.WHALE, config=config)
        # 开空仓 100 @ 100
        account.position.update(OrderSide.SELL, 100.0, 100.0)

        # 当前价 90
        # 净值 = 余额 10000000 + 浮动盈亏 (90-100)*(-100) = 10001000
        # 占用保证金 = 100 * 90 / 10 = 900
        # 可用保证金 = 10001000 - 900 = 10000100
        available_margin = account.get_available_margin(90.0)

        assert abs(available_margin - 10000100.0) < 0.01

    def test_available_margin_when_loss(self):
        """测试亏损时可用保证金减少"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 100 @ 100（大量持仓）
        account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 当前价 90（亏损）
        # 净值 = 余额 10000 + 浮动盈亏 (90-100)*100 = 9000
        # 占用保证金 = 100 * 90 / 100 = 90
        # 可用保证金 = 9000 - 90 = 8910
        available_margin = account.get_available_margin(90.0)

        assert abs(available_margin - 8910.0) < 0.01

    def test_available_margin_negative(self):
        """测试可用保证金为负数（爆仓风险）"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 1000 @ 100（极大量持仓）
        account.position.update(OrderSide.BUY, 1000.0, 100.0)

        # 当前价 90（大幅亏损）
        # 净值 = 余额 10000 + 浮动盈亏 (90-100)*1000 = 0
        # 占用保证金 = 1000 * 90 / 100 = 900
        # 可用保证金 = 0 - 900 = -900（负数表示已爆仓）
        available_margin = account.get_available_margin(90.0)

        assert abs(available_margin - (-900.0)) < 0.01


class TestAccountGetMarginRatio:
    """测试 Account.get_margin_ratio"""

    def test_margin_ratio_without_position(self):
        """测试无持仓时保证金率返回无穷大"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)

        # 无持仓时，保证金率应为无穷大
        margin_ratio = account.get_margin_ratio(100.0)

        assert margin_ratio == float("inf")

    def test_margin_ratio_with_long_position_profit(self):
        """测试多仓盈利时的保证金率"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 10 @ 100
        account.position.update(OrderSide.BUY, 10.0, 100.0)

        # 当前价 110
        # 净值 = 余额 10000 + 浮动盈亏 (110-100)*10 = 10100
        # 持仓市值 = 10 * 110 = 1100
        # 保证金率 = 10100 / 1100 = 9.1818...
        margin_ratio = account.get_margin_ratio(110.0)

        assert abs(margin_ratio - 9.1818) < 0.01

    def test_margin_ratio_with_long_position_loss(self):
        """测试多仓亏损时的保证金率"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 100 @ 100（大量持仓）
        account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 当前价 90（亏损）
        # 净值 = 余额 10000 + 浮动盈亏 (90-100)*100 = 9000
        # 持仓市值 = 100 * 90 = 9000
        # 保证金率 = 9000 / 9000 = 1.0
        margin_ratio = account.get_margin_ratio(90.0)

        assert abs(margin_ratio - 1.0) < 0.01

    def test_margin_ratio_with_short_position(self):
        """测试空仓时的保证金率"""
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        account = Account(agent_id=10001, agent_type=AgentType.WHALE, config=config)
        # 开空仓 100 @ 100
        account.position.update(OrderSide.SELL, 100.0, 100.0)

        # 当前价 90
        # 净值 = 余额 10000000 + 浮动盈亏 (90-100)*(-100) = 10001000
        # 持仓市值 = 100 * 90 = 9000
        # 保证金率 = 10001000 / 9000 = 1111.222...
        margin_ratio = account.get_margin_ratio(90.0)

        assert abs(margin_ratio - 1111.222) < 0.01

    def test_margin_ratio_negative_equity(self):
        """测试净值为负时的保证金率（爆仓状态）"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 1000 @ 100（极大量持仓）
        account.position.update(OrderSide.BUY, 1000.0, 100.0)

        # 当前价 90（大幅亏损）
        # 净值 = 余额 10000 + 浮动盈亏 (90-100)*1000 = 0
        # 持仓市值 = 1000 * 90 = 90000
        # 保证金率 = 0 / 90000 = 0
        margin_ratio = account.get_margin_ratio(90.0)

        assert margin_ratio == 0.0

    def test_margin_ratio_below_maintenance_margin(self):
        """测试保证金率低于维持保证金率（强平风险）"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 2000 @ 100（超大量持仓）
        account.position.update(OrderSide.BUY, 2000.0, 100.0)

        # 当前价 90（大幅亏损）
        # 净值 = 余额 10000 + 浮动盈亏 (90-100)*2000 = -10000
        # 持仓市值 = 2000 * 90 = 180000
        # 保证金率 = -10000 / 180000 = -0.0556（负数表示已爆仓）
        margin_ratio = account.get_margin_ratio(90.0)

        assert margin_ratio < 0  # 低于维持保证金率
        assert abs(margin_ratio - (-0.0556)) < 0.001


class TestAccountCheckLiquidation:
    """测试 Account.check_liquidation"""

    def test_need_liquidation(self):
        """测试需要平仓的情况"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 2000 @ 100（超大量持仓）
        account.position.update(OrderSide.BUY, 2000.0, 100.0)

        # 当前价 90（大幅亏损）
        # 保证金率 = -10000 / 180000 = -0.0556 < 维持保证金率 0.005
        need_liquidation = account.check_liquidation(90.0)

        assert need_liquidation is True

    def test_no_need_liquidation(self):
        """测试不需要平仓的情况"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 10 @ 100（小量持仓）
        account.position.update(OrderSide.BUY, 10.0, 100.0)

        # 当前价 95（小幅亏损）
        # 净值 = 10000 + (95-100)*10 = 9950
        # 持仓市值 = 10 * 95 = 950
        # 保证金率 = 9950 / 950 = 10.47 > 维持保证金率 0.005
        need_liquidation = account.check_liquidation(95.0)

        assert need_liquidation is False

    def test_no_need_liquidation_without_position(self):
        """测试无持仓时不需要平仓"""
        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)

        # 无持仓时保证金率为无穷大，不需要平仓
        need_liquidation = account.check_liquidation(100.0)

        assert need_liquidation is False

    def test_need_liquidation_short_position(self):
        """测试空仓亏损需要平仓"""
        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        account = Account(agent_id=10001, agent_type=AgentType.WHALE, config=config)
        # 开空仓 200000 @ 100（大量持仓）
        account.position.update(OrderSide.SELL, 200000.0, 100.0)

        # 当前价 145（大幅亏损）
        # 净值 = 10000000 + (145-100)*(-200000) = 10000000 - 9000000 = 1000000
        # 持仓市值 = 200000 * 145 = 29000000
        # 保证金率 = 1000000 / 29000000 = 0.0345 < 维持保证金率 0.05
        need_liquidation = account.check_liquidation(145.0)

        assert need_liquidation is True


class TestAccountLiquidate:
    """测试 Account.liquidate"""

    def test_liquidate_long_position(self):
        """测试多仓强平事件发布"""
        from src.core.event_engine.events import Event, EventType
        from src.core.event_engine.event_bus import EventBus

        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)
        # 开多仓 100 @ 100
        account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 创建事件总线并订阅强平事件
        event_bus = EventBus()
        published_events = []

        def liquidation_handler(event: Event):
            published_events.append(event)

        event_bus.subscribe(EventType.LIQUIDATION, liquidation_handler)

        # 执行强平
        account.liquidate(90.0, event_bus)

        # 验证事件发布
        assert len(published_events) == 1
        event = published_events[0]
        assert event.event_type == EventType.LIQUIDATION
        assert event.data["agent_id"] == 1
        assert event.data["agent_type"] == AgentType.RETAIL
        assert event.data["current_price"] == 90.0
        assert event.data["position_quantity"] == 100.0
        assert event.data["position_avg_price"] == 100.0

    def test_liquidate_short_position(self):
        """测试空仓强平事件发布"""
        from src.core.event_engine.events import Event, EventType
        from src.core.event_engine.event_bus import EventBus

        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        account = Account(agent_id=10001, agent_type=AgentType.WHALE, config=config)
        # 开空仓 50000 @ 100
        account.position.update(OrderSide.SELL, 50000.0, 100.0)

        # 创建事件总线并订阅强平事件
        event_bus = EventBus()
        published_events = []

        def liquidation_handler(event: Event):
            published_events.append(event)

        event_bus.subscribe(EventType.LIQUIDATION, liquidation_handler)

        # 执行强平
        account.liquidate(120.0, event_bus)

        # 验证事件发布
        assert len(published_events) == 1
        event = published_events[0]
        assert event.event_type == EventType.LIQUIDATION
        assert event.data["agent_id"] == 10001
        assert event.data["agent_type"] == AgentType.WHALE
        assert event.data["current_price"] == 120.0
        assert event.data["position_quantity"] == -50000.0
        assert event.data["position_avg_price"] == 100.0

    def test_liquidate_without_position(self):
        """测试无持仓时强平事件发布"""
        from src.core.event_engine.events import Event, EventType
        from src.core.event_engine.event_bus import EventBus

        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)

        # 创建事件总线并订阅强平事件
        event_bus = EventBus()
        published_events = []

        def liquidation_handler(event: Event):
            published_events.append(event)

        event_bus.subscribe(EventType.LIQUIDATION, liquidation_handler)

        # 执行强平
        account.liquidate(100.0, event_bus)

        # 验证事件发布（即使无持仓也发布事件）
        assert len(published_events) == 1
        event = published_events[0]
        assert event.event_type == EventType.LIQUIDATION
        assert event.data["position_quantity"] == 0.0
        assert event.data["position_avg_price"] == 0.0


class TestAccountOnTrade:
    """测试 Account.on_trade"""

    def test_buy_trade(self):
        """测试买入成交"""
        from src.market.matching.trade import Trade

        config = AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

        account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)

        # 创建成交记录：买入 10 @ 100，手续费 5
        trade = Trade(
            trade_id=1,
            price=100.0,
            quantity=10.0,
            buyer_id=1,
            seller_id=2,
            buyer_fee=5.0,
            seller_fee=2.0,
        )

        # 处理买入成交
        account.on_trade(trade, is_buyer=True)

        # 验证持仓：数量 +10，均价 100
        assert account.position.quantity == 10.0
        assert account.position.avg_price == 100.0
        # 验证余额：10000 - 5 = 9995
        assert account.balance == 9995.0

    def test_sell_trade(self):
        """测试卖出成交"""
        from src.market.matching.trade import Trade

        config = AgentConfig(
            count=10,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        account = Account(agent_id=1, agent_type=AgentType.WHALE, config=config)
        # 先开多仓 100 @ 100
        account.position.update(OrderSide.BUY, 100.0, 100.0)

        # 创建成交记录：卖出 50 @ 110，手续费 5.5
        trade = Trade(
            trade_id=1,
            price=110.0,
            quantity=50.0,
            buyer_id=2,
            seller_id=1,
            buyer_fee=5.5,
            seller_fee=0.0,
        )

        # 记录初始余额
        initial_balance = account.balance

        # 处理卖出成交
        account.on_trade(trade, is_buyer=False)

        # 验证持仓：数量减至 50，均价不变
        assert account.position.quantity == 50.0
        assert account.position.avg_price == 100.0
        # 验证余额：扣除手续费
        assert account.balance == initial_balance
        # 验证已实现盈亏：50 * (110 - 100) = 500
        assert abs(account.position.realized_pnl - 500.0) < 0.01
