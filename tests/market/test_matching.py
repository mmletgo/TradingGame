"""
撮合引擎模块测试
"""

from src.config.config import MarketConfig
from src.market.matching.matching_engine import MatchingEngine


def test_matching_engine_init_normal():
    """测试正常创建撮合引擎"""
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )

    engine = MatchingEngine(config)

    # 验证引擎创建成功
    assert engine is not None
    assert engine._config is config
    assert engine._orderbook is not None
    assert engine._next_trade_id == 1


def test_matching_engine_init_different_configs():
    """测试不同配置创建撮合引擎"""
    # 不同的 tick_size
    config1 = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine1 = MatchingEngine(config1)
    assert engine1._orderbook.tick_size == 0.01

    # 不同的 depth
    config2 = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=50,
    )
    engine2 = MatchingEngine(config2)
    assert engine2._config.depth == 50


def test_matching_engine_multiple_engines():
    """测试创建多个撮合引擎实例"""
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )

    engine1 = MatchingEngine(config)
    engine2 = MatchingEngine(config)

    # 两个引擎应该是独立的实例
    assert engine1 is not engine2
    assert engine1._orderbook is not engine2._orderbook
    assert engine1._next_trade_id == 1
    assert engine2._next_trade_id == 1


# ==================== calculate_fee 相关测试 ====================


def test_calculate_fee_retail_maker():
    """测试散户挂单费率（万2）"""
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册散户费率：挂单万2，吃单万5
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)

    # 10000 元成交金额，挂单手续费应该是 2 元
    fee = engine.calculate_fee(agent_id=1, amount=10000.0, is_maker=True)
    assert fee == 2.0

    # 50000 元成交金额，挂单手续费应该是 10 元
    fee = engine.calculate_fee(agent_id=1, amount=50000.0, is_maker=True)
    assert fee == 10.0


def test_calculate_fee_retail_taker():
    """测试散户吃单费率（万5）"""
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册散户费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)

    # 10000 元成交金额，吃单手续费应该是 5 元
    fee = engine.calculate_fee(agent_id=1, amount=10000.0, is_maker=False)
    assert fee == 5.0

    # 50000 元成交金额，吃单手续费应该是 25 元
    fee = engine.calculate_fee(agent_id=1, amount=50000.0, is_maker=False)
    assert fee == 25.0


def test_calculate_fee_whale():
    """测试庄家费率（挂单0，吃单万1）"""
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册庄家费率：挂单0，吃单万1
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)

    # 挂单手续费为 0
    fee = engine.calculate_fee(agent_id=2, amount=10000.0, is_maker=True)
    assert fee == 0.0

    # 吃单手续费应该是 1 元（万1）
    fee = engine.calculate_fee(agent_id=2, amount=10000.0, is_maker=False)
    assert fee == 1.0


def test_calculate_fee_market_maker():
    """测试做市商费率（挂单0，吃单万1）"""
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册做市商费率：挂单0，吃单万1
    engine.register_agent(agent_id=3, maker_rate=0.0, taker_rate=0.0001)

    # 挂单手续费为 0
    fee = engine.calculate_fee(agent_id=3, amount=10000.0, is_maker=True)
    assert fee == 0.0

    # 吃单手续费应该是 1 元（万1）
    fee = engine.calculate_fee(agent_id=3, amount=10000.0, is_maker=False)
    assert fee == 1.0


def test_calculate_fee_unregistered_agent():
    """测试未注册的 Agent 使用默认散户费率"""
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 未注册的 Agent 应该使用默认散户费率
    fee = engine.calculate_fee(agent_id=999, amount=10000.0, is_maker=True)
    assert fee == 2.0  # 万2

    fee = engine.calculate_fee(agent_id=999, amount=10000.0, is_maker=False)
    assert fee == 5.0  # 万5


def test_calculate_fee_zero_amount():
    """测试成交金额为 0 时手续费为 0"""
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)

    # 成交金额为 0，手续费应该也是 0
    fee = engine.calculate_fee(agent_id=1, amount=0.0, is_maker=True)
    assert fee == 0.0

    fee = engine.calculate_fee(agent_id=1, amount=0.0, is_maker=False)
    assert fee == 0.0


def test_register_agent():
    """测试注册 Agent 费率"""
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册 Agent
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)

    # 验证费率已设置
    assert engine._fee_rates[1] == (0.0002, 0.0005)

    # 更新费率
    engine.register_agent(agent_id=1, maker_rate=0.0, taker_rate=0.0001)

    # 验证费率已更新
    assert engine._fee_rates[1] == (0.0, 0.0001)


def test_multiple_agents_different_rates():
    """测试多个 Agent 使用不同费率"""
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册散户
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    # 注册庄家
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)
    # 注册做市商
    engine.register_agent(agent_id=3, maker_rate=0.0, taker_rate=0.0001)

    # 验证不同 Agent 的费率
    amount = 10000.0

    # 散户挂单：2 元
    assert engine.calculate_fee(1, amount, True) == 2.0
    # 散户吃单：5 元
    assert engine.calculate_fee(1, amount, False) == 5.0

    # 庄家挂单：0 元
    assert engine.calculate_fee(2, amount, True) == 0.0
    # 庄家吃单：1 元
    assert engine.calculate_fee(2, amount, False) == 1.0

    # 做市商挂单：0 元
    assert engine.calculate_fee(3, amount, True) == 0.0
    # 做市商吃单：1 元
    assert engine.calculate_fee(3, amount, False) == 1.0


# ==================== match_limit_order 相关测试 ====================


def test_match_limit_order_buy_with_asks():
    """测试买单与卖盘撮合"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)  # 卖方
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)  # 买方（庄家）

    # 先挂卖单（卖盘）
    sell_order1 = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=10.0,
    )
    engine._orderbook.add_order(sell_order1)

    sell_order2 = Order(
        order_id=2,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.6,
        quantity=5.0,
    )
    engine._orderbook.add_order(sell_order2)

    # 买单价格高于卖盘最优价，应该成交
    buy_order = Order(
        order_id=3,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.7,
        quantity=8.0,
    )

    trades = engine.match_limit_order(buy_order)

    # 应该产生 1 笔成交（与 sell_order1 成交 8.0）
    assert len(trades) == 1
    assert trades[0].trade_id == 1
    assert trades[0].price == 100.5  # 成交价格是卖单价格
    assert trades[0].quantity == 8.0
    assert trades[0].buyer_id == 2
    assert trades[0].seller_id == 1
    # 买方是 taker（庄家万1），卖方是 maker（散户万2）
    assert trades[0].buyer_fee == 100.5 * 8.0 * 0.0001  # 庄家吃单万1
    assert trades[0].seller_fee == 100.5 * 8.0 * 0.0002  # 散户挂单万2

    # 买单应该完全成交
    assert buy_order.filled_quantity == 8.0

    # 卖单 1 应该还剩 2.0
    assert sell_order1.filled_quantity == 8.0


def test_match_limit_order_sell_with_bids():
    """测试卖单与买盘撮合"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)  # 买方
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)  # 卖方（庄家）

    # 先挂买单（买盘）
    buy_order1 = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.8,
        quantity=10.0,
    )
    engine._orderbook.add_order(buy_order1)

    buy_order2 = Order(
        order_id=2,
        agent_id=1,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.7,
        quantity=5.0,
    )
    engine._orderbook.add_order(buy_order2)

    # 卖单价格低于买盘最优价，应该成交
    sell_order = Order(
        order_id=3,
        agent_id=2,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=99.6,
        quantity=7.0,
    )

    trades = engine.match_limit_order(sell_order)

    # 应该产生 1 笔成交（与 buy_order1 成交 7.0）
    assert len(trades) == 1
    assert trades[0].trade_id == 1
    assert trades[0].price == 99.8  # 成交价格是买单价格
    assert trades[0].quantity == 7.0
    assert trades[0].buyer_id == 1
    assert trades[0].seller_id == 2
    # 买方是 maker（散户万2），卖方是 taker（庄家万1）
    assert trades[0].buyer_fee == 99.8 * 7.0 * 0.0002  # 散户挂单万2
    assert trades[0].seller_fee == 99.8 * 7.0 * 0.0001  # 庄家吃单万1

    # 卖单应该完全成交
    assert sell_order.filled_quantity == 7.0

    # 买单 1 应该还剩 3.0
    assert buy_order1.filled_quantity == 7.0


def test_match_limit_order_no_match():
    """测试无法撮合时挂单"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)

    # 挂卖单
    sell_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=10.0,
    )
    engine._orderbook.add_order(sell_order)

    # 买单价格低于卖盘最优价，无法撮合，应该挂单
    buy_order = Order(
        order_id=2,
        agent_id=1,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=5.0,
    )

    trades = engine.match_limit_order(buy_order)

    # 不应该有成交
    assert len(trades) == 0

    # 买单应该挂在订单簿上
    assert buy_order.filled_quantity == 0.0
    assert buy_order.order_id in engine._orderbook.order_map
    assert engine._orderbook.get_best_bid() == 100.5


def test_match_limit_order_partial_fill():
    """测试部分成交后挂单"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)

    # 挂卖单，数量只有 3.0
    sell_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=3.0,
    )
    engine._orderbook.add_order(sell_order)

    # 买单数量是 10.0，只能成交 3.0，剩余 7.0 挂单
    buy_order = Order(
        order_id=2,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=10.0,
    )

    trades = engine.match_limit_order(buy_order)

    # 应该产生 1 笔成交
    assert len(trades) == 1
    assert trades[0].quantity == 3.0

    # 买单剩余部分应该挂在订单簿上（新挂单状态，filled_quantity 重置为 0）
    assert buy_order.quantity == 7.0  # 剩余数量
    assert buy_order.filled_quantity == 0.0  # 重置为新挂单状态
    assert buy_order.order_id in engine._orderbook.order_map
    assert engine._orderbook.get_best_bid() == 101.0


def test_match_limit_order_multiple_price_levels():
    """测试跨多个价格档位撮合"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)

    # 挂多个价格档位的卖单
    sell_order1 = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=3.0,
    )
    engine._orderbook.add_order(sell_order1)

    sell_order2 = Order(
        order_id=2,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.6,
        quantity=4.0,
    )
    engine._orderbook.add_order(sell_order2)

    sell_order3 = Order(
        order_id=3,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.7,
        quantity=5.0,
    )
    engine._orderbook.add_order(sell_order3)

    # 买单数量 10.0，应该跨越三个价格档位
    buy_order = Order(
        order_id=4,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=10.0,
    )

    trades = engine.match_limit_order(buy_order)

    # 应该产生 3 笔成交
    assert len(trades) == 3

    # 第一笔：100.5 价格成交 3.0
    assert trades[0].price == 100.5
    assert trades[0].quantity == 3.0

    # 第二笔：100.6 价格成交 4.0
    assert trades[1].price == 100.6
    assert trades[1].quantity == 4.0

    # 第三笔：100.7 价格成交 3.0
    assert trades[2].price == 100.7
    assert trades[2].quantity == 3.0

    # 买单应该完全成交
    assert buy_order.filled_quantity == 10.0

    # 卖单 3 应该还剩 2.0
    assert sell_order3.filled_quantity == 3.0


def test_match_limit_order_empty_orderbook():
    """测试空订单簿时挂单"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)

    # 订单簿为空，挂买单
    buy_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10.0,
    )

    trades = engine.match_limit_order(buy_order)

    # 不应该有成交
    assert len(trades) == 0

    # 订单应该挂在订单簿上
    assert buy_order.order_id in engine._orderbook.order_map
    assert engine._orderbook.get_best_bid() == 100.0


def test_match_limit_order_fully_filled():
    """测试订单完全成交后不挂单"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)

    # 挂卖单
    sell_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=10.0,
    )
    engine._orderbook.add_order(sell_order)

    # 买单数量等于卖单数量
    buy_order = Order(
        order_id=2,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=10.0,
    )

    trades = engine.match_limit_order(buy_order)

    # 应该产生 1 笔成交
    assert len(trades) == 1
    assert trades[0].quantity == 10.0

    # 买单应该完全成交
    assert buy_order.filled_quantity == 10.0

    # 买单不应该在订单簿中（完全成交）
    assert buy_order.order_id not in engine._orderbook.order_map

    # 卖单也应该完全成交，从订单簿移除
    assert sell_order.order_id not in engine._orderbook.order_map
    assert engine._orderbook.get_best_ask() is None


def test_match_limit_order_orderbook_quantity():
    """测试订单簿数量统计正确性（验证挂单时数量为剩余数量）"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)

    # 挂卖单，数量只有 3.0
    sell_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=3.0,
    )
    engine._orderbook.add_order(sell_order)

    # 买单数量是 10.0，只能成交 3.0，剩余 7.0 挂单
    buy_order = Order(
        order_id=2,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=10.0,
    )

    engine.match_limit_order(buy_order)

    # 验证订单簿数量统计正确
    depth = engine._orderbook.get_depth()
    # 买盘应该有 1 档，数量是 7.0（剩余数量，不是 10.0）
    assert len(depth["bids"]) == 1
    assert depth["bids"][0][0] == 101.0  # 价格
    assert depth["bids"][0][1] == 7.0  # 数量应该是剩余数量

    # 验证 price_level 的 total_quantity 也是正确的
    assert engine._orderbook.bids[101.0].total_quantity == 7.0


# ==================== match_market_order 相关测试 ====================


def test_match_market_order_buy_fully_filled():
    """测试市价买单完全成交"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)  # 卖方
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)  # 买方（庄家）

    # 挂卖单（卖盘）
    sell_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=10.0,
    )
    engine._orderbook.add_order(sell_order)

    # 市价买单，数量等于卖盘数量
    market_buy = Order(
        order_id=2,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        price=0.0,  # 市价单价格无意义
        quantity=10.0,
    )

    trades = engine.match_market_order(market_buy)

    # 应该产生 1 笔成交
    assert len(trades) == 1
    assert trades[0].trade_id == 1
    assert trades[0].price == 100.5  # 成交价格是卖单价格
    assert trades[0].quantity == 10.0
    assert trades[0].buyer_id == 2
    assert trades[0].seller_id == 1
    # 买方是 taker（庄家万1），卖方是 maker（散户万2）
    assert trades[0].buyer_fee == 100.5 * 10.0 * 0.0001
    assert trades[0].seller_fee == 100.5 * 10.0 * 0.0002

    # 市价买单应该完全成交
    assert market_buy.filled_quantity == 10.0

    # 卖单应该完全成交，从订单簿移除
    assert sell_order.order_id not in engine._orderbook.order_map
    assert engine._orderbook.get_best_ask() is None

    # 市价单不应该挂在订单簿上
    assert market_buy.order_id not in engine._orderbook.order_map


def test_match_market_order_sell_fully_filled():
    """测试市价卖单完全成交"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)  # 买方
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)  # 卖方（庄家）

    # 挂买单（买盘）
    buy_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.8,
        quantity=10.0,
    )
    engine._orderbook.add_order(buy_order)

    # 市价卖单
    market_sell = Order(
        order_id=2,
        agent_id=2,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        price=0.0,
        quantity=10.0,
    )

    trades = engine.match_market_order(market_sell)

    # 应该产生 1 笔成交
    assert len(trades) == 1
    assert trades[0].price == 99.8  # 成交价格是买单价格
    assert trades[0].quantity == 10.0
    assert trades[0].buyer_id == 1
    assert trades[0].seller_id == 2
    # 买方是 maker（散户万2），卖方是 taker（庄家万1）
    assert trades[0].buyer_fee == 99.8 * 10.0 * 0.0002
    assert trades[0].seller_fee == 99.8 * 10.0 * 0.0001

    # 市价卖单应该完全成交
    assert market_sell.filled_quantity == 10.0

    # 买单应该完全成交，从订单簿移除
    assert buy_order.order_id not in engine._orderbook.order_map
    assert engine._orderbook.get_best_bid() is None

    # 市价单不应该挂在订单簿上
    assert market_sell.order_id not in engine._orderbook.order_map


def test_match_market_order_buy_partial_fill():
    """测试市价买单部分成交（对手盘不足）"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)

    # 挂卖单，数量只有 5.0
    sell_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=5.0,
    )
    engine._orderbook.add_order(sell_order)

    # 市价买单，数量是 10.0，但对手盘只有 5.0
    market_buy = Order(
        order_id=2,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        price=0.0,
        quantity=10.0,
    )

    trades = engine.match_market_order(market_buy)

    # 应该产生 1 笔成交
    assert len(trades) == 1
    assert trades[0].quantity == 5.0

    # 市价买单应该只成交了 5.0
    assert market_buy.filled_quantity == 5.0

    # 卖单应该完全成交，从订单簿移除
    assert sell_order.order_id not in engine._orderbook.order_map
    assert engine._orderbook.get_best_ask() is None

    # 市价单不应该挂在订单簿上，剩余部分直接丢弃
    assert market_buy.order_id not in engine._orderbook.order_map


def test_match_market_order_sell_partial_fill():
    """测试市价卖单部分成交（对手盘不足）"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)

    # 挂买单，数量只有 5.0
    buy_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.8,
        quantity=5.0,
    )
    engine._orderbook.add_order(buy_order)

    # 市价卖单，数量是 10.0，但对手盘只有 5.0
    market_sell = Order(
        order_id=2,
        agent_id=2,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        price=0.0,
        quantity=10.0,
    )

    trades = engine.match_market_order(market_sell)

    # 应该产生 1 笔成交
    assert len(trades) == 1
    assert trades[0].quantity == 5.0

    # 市价卖单应该只成交了 5.0
    assert market_sell.filled_quantity == 5.0

    # 买单应该完全成交，从订单簿移除
    assert buy_order.order_id not in engine._orderbook.order_map
    assert engine._orderbook.get_best_bid() is None

    # 市价单不应该挂在订单簿上
    assert market_sell.order_id not in engine._orderbook.order_map


def test_match_market_order_empty_orderbook():
    """测试市价单在空订单簿时不成交"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)

    # 订单簿为空，下市价买单
    market_buy = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        price=0.0,
        quantity=10.0,
    )

    trades = engine.match_market_order(market_buy)

    # 不应该有成交
    assert len(trades) == 0

    # 市价单不应该成交
    assert market_buy.filled_quantity == 0.0

    # 市价单不应该挂在订单簿上
    assert market_buy.order_id not in engine._orderbook.order_map


def test_match_market_order_multiple_price_levels():
    """测试市价单跨多个价格档位成交"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)

    # 挂多个价格档位的卖单
    sell_order1 = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=3.0,
    )
    engine._orderbook.add_order(sell_order1)

    sell_order2 = Order(
        order_id=2,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.6,
        quantity=4.0,
    )
    engine._orderbook.add_order(sell_order2)

    sell_order3 = Order(
        order_id=3,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.7,
        quantity=5.0,
    )
    engine._orderbook.add_order(sell_order3)

    # 市价买单，数量 10.0，跨越三个价格档位
    market_buy = Order(
        order_id=4,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        price=0.0,
        quantity=10.0,
    )

    trades = engine.match_market_order(market_buy)

    # 应该产生 3 笔成交
    assert len(trades) == 3

    # 第一笔：100.5 价格成交 3.0
    assert trades[0].price == 100.5
    assert trades[0].quantity == 3.0

    # 第二笔：100.6 价格成交 4.0
    assert trades[1].price == 100.6
    assert trades[1].quantity == 4.0

    # 第三笔：100.7 价格成交 3.0
    assert trades[2].price == 100.7
    assert trades[2].quantity == 3.0

    # 市价买单应该完全成交
    assert market_buy.filled_quantity == 10.0

    # 卖单 3 应该还剩 2.0
    assert sell_order3.filled_quantity == 3.0
    assert 3 in engine._orderbook.order_map  # sell_order3 仍在订单簿中
    assert engine._orderbook.get_best_ask() == 100.7


def test_match_market_order_taker_fee():
    """测试市价单永远是 taker，支付吃单费率"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0, taker_rate=0.0001)  # 做市商
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)  # 庄家

    # 做市商挂卖单（maker）
    sell_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=10.0,
    )
    engine._orderbook.add_order(sell_order)

    # 庄家下市价买单（taker）
    market_buy = Order(
        order_id=2,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        price=0.0,
        quantity=10.0,
    )

    trades = engine.match_market_order(market_buy)

    # 市价单（庄家）是 taker，应该支付吃单费率万1
    # 挂单方（做市商）是 maker，应该支付挂单费率0
    assert trades[0].buyer_fee == 100.5 * 10.0 * 0.0001  # 庄家吃单万1
    assert trades[0].seller_fee == 0.0  # 做市商挂单0


# ==================== process_order 相关测试 ====================


def test_process_order_limit_order():
    """测试处理限价单"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)

    # 挂卖单
    sell_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=10.0,
    )
    engine._orderbook.add_order(sell_order)

    # 处理限价买单
    buy_order = Order(
        order_id=2,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=8.0,
    )

    trades = engine.process_order(buy_order)

    # 应该产生 1 笔成交
    assert len(trades) == 1
    assert trades[0].price == 100.5
    assert trades[0].quantity == 8.0
    assert trades[0].buyer_id == 2
    assert trades[0].seller_id == 1


def test_process_order_market_order():
    """测试处理市价单"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)

    # 挂卖单
    sell_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=10.0,
    )
    engine._orderbook.add_order(sell_order)

    # 处理市价买单
    market_buy = Order(
        order_id=2,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        price=0.0,
        quantity=8.0,
    )

    trades = engine.process_order(market_buy)

    # 应该产生 1 笔成交
    assert len(trades) == 1
    assert trades[0].price == 100.5
    assert trades[0].quantity == 8.0


def test_process_order_no_trade():
    """测试处理未成交的订单"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)

    # 订单簿为空，处理限价买单
    buy_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10.0,
    )

    trades = engine.process_order(buy_order)

    # 不应该有成交
    assert len(trades) == 0

    # 订单应该挂在订单簿上
    assert buy_order.order_id in engine._orderbook.order_map


def test_process_order_multiple_trades():
    """测试处理订单产生多笔成交"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)

    # 挂多个价格档位的卖单
    sell_order1 = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=3.0,
    )
    engine._orderbook.add_order(sell_order1)

    sell_order2 = Order(
        order_id=2,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.6,
        quantity=4.0,
    )
    engine._orderbook.add_order(sell_order2)

    # 处理买单，跨越两个价格档位
    buy_order = Order(
        order_id=3,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=7.0,
    )

    trades = engine.process_order(buy_order)

    # 应该产生 2 笔成交
    assert len(trades) == 2
    assert trades[0].price == 100.5
    assert trades[0].quantity == 3.0
    assert trades[1].price == 100.6
    assert trades[1].quantity == 4.0


def test_process_order_partial_fill():
    """测试订单部分成交"""
    from src.market.orderbook.order import Order, OrderSide, OrderType

    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(config)

    # 注册费率
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    engine.register_agent(agent_id=2, maker_rate=0.0, taker_rate=0.0001)

    # 挂卖单，数量只有 3.0
    sell_order = Order(
        order_id=1,
        agent_id=1,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=3.0,
    )
    engine._orderbook.add_order(sell_order)

    # 处理买单，数量是 10.0，只能成交 3.0，剩余 7.0 挂单
    buy_order = Order(
        order_id=2,
        agent_id=2,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=10.0,
    )

    trades = engine.process_order(buy_order)

    # 应该产生 1 笔成交
    assert len(trades) == 1
    assert trades[0].quantity == 3.0

    # 买单剩余部分应该挂在订单簿上
    assert buy_order.quantity == 7.0
    assert buy_order.filled_quantity == 0.0
    assert buy_order.order_id in engine._orderbook.order_map
