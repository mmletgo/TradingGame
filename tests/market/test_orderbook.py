"""
测试订单簿模块
"""

import pytest
from collections import OrderedDict

from src.market.orderbook.orderbook import PriceLevel, OrderBook
from src.market.orderbook.order import Order, OrderSide, OrderType


def test_create_price_level():
    """测试创建价格档位"""
    level = PriceLevel(price=100.5)

    assert level.price == 100.5
    assert isinstance(level.orders, OrderedDict)
    assert len(level.orders) == 0
    assert level.total_quantity == 0


def test_create_price_level_zero():
    """测试创建价格为0的档位"""
    level = PriceLevel(price=0.0)

    assert level.price == 0.0
    assert isinstance(level.orders, OrderedDict)
    assert len(level.orders) == 0
    assert level.total_quantity == 0


def test_create_price_level_negative():
    """测试创建负价格档位（虽然实际场景不会）"""
    level = PriceLevel(price=-10.0)

    assert level.price == -10.0
    assert isinstance(level.orders, OrderedDict)
    assert len(level.orders) == 0
    assert level.total_quantity == 0


def test_add_order_single():
    """测试添加单个订单"""
    level = PriceLevel(price=100.5)
    order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=10,
    )

    level.add_order(order)

    assert len(level.orders) == 1
    assert level.orders[1] is order
    assert level.total_quantity == 10


def test_add_order_multiple():
    """测试添加多个订单"""
    level = PriceLevel(price=100.0)
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=5,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=15,
    )
    order3 = Order(
        order_id=3,
        agent_id=102,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=7,
    )

    level.add_order(order1)
    level.add_order(order2)
    level.add_order(order3)

    assert len(level.orders) == 3
    # 验证 FIFO 顺序
    orders_list = list(level.orders.values())
    assert orders_list[0] is order1
    assert orders_list[1] is order2
    assert orders_list[2] is order3
    assert level.total_quantity == 27


def test_remove_order_exists():
    """测试移除存在的订单"""
    level = PriceLevel(price=100.0)
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=20,
    )

    level.add_order(order1)
    level.add_order(order2)

    # 移除中间的订单
    removed = level.remove_order(order_id=1)

    assert removed is order1
    assert len(level.orders) == 1
    assert 1 not in level.orders
    assert 2 in level.orders
    assert level.total_quantity == 20


def test_remove_order_not_exists():
    """测试移除不存在的订单"""
    level = PriceLevel(price=100.0)
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )

    level.add_order(order1)

    # 移除不存在的订单
    removed = level.remove_order(order_id=999)

    assert removed is None
    assert len(level.orders) == 1
    assert level.total_quantity == 10


def test_remove_order_empty():
    """测试从空价格档位移除订单"""
    level = PriceLevel(price=100.0)

    removed = level.remove_order(order_id=1)

    assert removed is None
    assert len(level.orders) == 0
    assert level.total_quantity == 0


def test_remove_order_fifo_preserved():
    """测试移除订单后 FIFO 顺序保持正确"""
    level = PriceLevel(price=100.0)
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=5,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=15,
    )
    order3 = Order(
        order_id=3,
        agent_id=102,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=7,
    )

    level.add_order(order1)
    level.add_order(order2)
    level.add_order(order3)

    # 移除中间订单
    level.remove_order(order_id=2)

    # 验证剩余订单的 FIFO 顺序
    orders_list = list(level.orders.values())
    assert len(orders_list) == 2
    assert orders_list[0] is order1
    assert orders_list[1] is order3
    assert level.total_quantity == 12  # 5 + 7


def test_get_volume_empty():
    """测试空档位返回0"""
    level = PriceLevel(price=100.0)

    assert level.get_volume() == 0


def test_get_volume_with_orders():
    """测试有订单返回总量"""
    level = PriceLevel(price=100.0)
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=20,
    )

    level.add_order(order1)
    level.add_order(order2)

    assert level.get_volume() == 30


def test_create_orderbook_default():
    """测试创建空订单簿（默认 tick_size）"""
    book = OrderBook()

    assert book.bids == {}
    assert book.asks == {}
    assert book.order_map == {}
    assert book.last_price == 0.0
    assert book.tick_size == 0.01


def test_create_orderbook_custom_tick():
    """测试创建订单簿（自定义 tick_size）"""
    book = OrderBook(tick_size=0.01)

    assert book.bids == {}
    assert book.asks == {}
    assert book.order_map == {}
    assert book.last_price == 0.0
    assert book.tick_size == 0.01


def test_add_order_buy():
    """测试添加买单"""
    book = OrderBook()
    order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )

    book.add_order(order)

    # 验证买盘有该价格档位
    assert 100.0 in book.bids
    assert len(book.bids) == 1
    assert book.bids[100.0].price == 100.0
    assert book.bids[100.0].total_quantity == 10
    # 验证订单映射
    assert 1 in book.order_map
    assert book.order_map[1] is order
    # 验证卖盘为空
    assert len(book.asks) == 0


def test_add_order_sell():
    """测试添加卖单"""
    book = OrderBook()
    order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )

    book.add_order(order)

    # 验证卖盘有该价格档位
    assert 100.0 in book.asks
    assert len(book.asks) == 1
    assert book.asks[100.0].price == 100.0
    assert book.asks[100.0].total_quantity == 10
    # 验证订单映射
    assert 1 in book.order_map
    assert book.order_map[1] is order
    # 验证买盘为空
    assert len(book.bids) == 0


def test_add_order_existing_level():
    """测试添加到已有档位"""
    book = OrderBook()
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=20,
    )

    book.add_order(order1)
    book.add_order(order2)

    # 验证只有一个价格档位
    assert len(book.bids) == 1
    assert 100.0 in book.bids
    # 验证档位总量
    assert book.bids[100.0].total_quantity == 30
    # 验证档位中有两个订单
    assert len(book.bids[100.0].orders) == 2
    # 验证订单映射
    assert 1 in book.order_map
    assert 2 in book.order_map


def test_add_order_different_prices():
    """测试添加不同价格的订单"""
    book = OrderBook()
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.5,
        quantity=20,
    )
    order3 = Order(
        order_id=3,
        agent_id=102,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=15,
    )

    book.add_order(order1)
    book.add_order(order2)
    book.add_order(order3)

    # 验证有三个价格档位
    assert len(book.bids) == 3
    assert 100.0 in book.bids
    assert 99.5 in book.bids
    assert 100.5 in book.bids
    # 验证每个档位的数量
    assert book.bids[100.0].total_quantity == 10
    assert book.bids[99.5].total_quantity == 20
    assert book.bids[100.5].total_quantity == 15


def test_add_order_both_sides():
    """测试同时添加买单和卖单"""
    book = OrderBook()
    buy_order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    sell_order = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=20,
    )

    book.add_order(buy_order)
    book.add_order(sell_order)

    # 验证买卖盘都正确
    assert len(book.bids) == 1
    assert len(book.asks) == 1
    assert 100.0 in book.bids
    assert 100.0 in book.asks
    assert book.bids[100.0].total_quantity == 10
    assert book.asks[100.0].total_quantity == 20
    # 验证订单映射
    assert 1 in book.order_map
    assert 2 in book.order_map


def test_cancel_order_exists_buy():
    """测试撤销存在的买单"""
    book = OrderBook()
    order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    book.add_order(order)

    # 撤销订单
    cancelled = book.cancel_order(order_id=1)

    # 验证返回的订单
    assert cancelled is order
    # 验证买盘空了（档位被删除）
    assert len(book.bids) == 0
    # 验证订单映射被移除
    assert 1 not in book.order_map


def test_cancel_order_exists_sell():
    """测试撤销存在的卖单"""
    book = OrderBook()
    order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    book.add_order(order)

    # 撤销订单
    cancelled = book.cancel_order(order_id=1)

    # 验证返回的订单
    assert cancelled is order
    # 验证卖盘空了（档位被删除）
    assert len(book.asks) == 0
    # 验证订单映射被移除
    assert 1 not in book.order_map


def test_cancel_order_not_exists():
    """测试撤销不存在的订单"""
    book = OrderBook()
    order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    book.add_order(order)

    # 撤销不存在的订单
    cancelled = book.cancel_order(order_id=999)

    # 验证返回 None
    assert cancelled is None
    # 验证原订单还在
    assert 1 in book.order_map
    assert len(book.bids) == 1


def test_cancel_order_multiple_same_price():
    """测试撤销同价格档位中的某个订单"""
    book = OrderBook()
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=20,
    )
    order3 = Order(
        order_id=3,
        agent_id=102,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=15,
    )

    book.add_order(order1)
    book.add_order(order2)
    book.add_order(order3)

    # 撤销中间的订单
    cancelled = book.cancel_order(order_id=2)

    # 验证返回的订单
    assert cancelled is order2
    # 验证档位还存在（还有其他订单）
    assert 100.0 in book.bids
    assert len(book.bids) == 1
    # 验证档位总量正确
    assert book.bids[100.0].total_quantity == 25  # 10 + 15
    # 验证订单映射
    assert 1 in book.order_map
    assert 2 not in book.order_map
    assert 3 in book.order_map


def test_cancel_order_removes_level_if_empty():
    """测试撤销订单后档位变空时删除档位"""
    book = OrderBook()
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=20,
    )

    book.add_order(order1)
    book.add_order(order2)

    # 撤销买单
    book.cancel_order(order_id=1)

    # 验证买盘档位被删除
    assert 100.0 not in book.bids
    assert len(book.bids) == 0
    # 验证卖盘档位还在
    assert 101.0 in book.asks
    assert len(book.asks) == 1


def test_get_best_bid_with_orders():
    """测试有买单时返回最高买价"""
    book = OrderBook()
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.5,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=20,
    )
    order3 = Order(
        order_id=3,
        agent_id=102,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.0,
        quantity=15,
    )

    book.add_order(order1)
    book.add_order(order2)
    book.add_order(order3)

    # 最优买价应该是 100.0（买盘最高价）
    assert book.get_best_bid() == 100.0


def test_get_best_bid_single_order():
    """测试只有一个买单时返回该价格"""
    book = OrderBook()
    order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.5,
        quantity=10,
    )

    book.add_order(order)

    assert book.get_best_bid() == 99.5


def test_get_best_bid_empty():
    """测试无买单时返回 None"""
    book = OrderBook()

    assert book.get_best_bid() is None


def test_get_best_bid_after_cancellation():
    """测试撤销订单后更新最优买价"""
    book = OrderBook()
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.5,
        quantity=20,
    )

    book.add_order(order1)
    book.add_order(order2)

    # 初始最优买价是 100.0
    assert book.get_best_bid() == 100.0

    # 撤销 100.0 的买单
    book.cancel_order(order_id=1)

    # 最优买价应该是 99.5
    assert book.get_best_bid() == 99.5


def test_get_best_bid_ignores_asks():
    """测试最优买价不受卖单影响"""
    book = OrderBook()
    buy_order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.0,
        quantity=10,
    )
    sell_order = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=98.0,
        quantity=20,
    )

    book.add_order(buy_order)
    book.add_order(sell_order)

    # 最优买价应该只看买盘
    assert book.get_best_bid() == 99.0


def test_get_best_ask_with_orders():
    """测试有卖单时返回最低卖价"""
    book = OrderBook()
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=20,
    )
    order3 = Order(
        order_id=3,
        agent_id=102,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=15,
    )

    book.add_order(order1)
    book.add_order(order2)
    book.add_order(order3)

    # 最优卖价应该是 100.0（卖盘最低价）
    assert book.get_best_ask() == 100.0


def test_get_best_ask_single_order():
    """测试只有一个卖单时返回该价格"""
    book = OrderBook()
    order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=10,
    )

    book.add_order(order)

    assert book.get_best_ask() == 100.5


def test_get_best_ask_empty():
    """测试无卖单时返回 None"""
    book = OrderBook()

    assert book.get_best_ask() is None


def test_get_best_ask_after_cancellation():
    """测试撤销订单后更新最优卖价"""
    book = OrderBook()
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=20,
    )

    book.add_order(order1)
    book.add_order(order2)

    # 初始最优卖价是 100.0
    assert book.get_best_ask() == 100.0

    # 撤销 100.0 的卖单
    book.cancel_order(order_id=1)

    # 最优卖价应该是 100.5
    assert book.get_best_ask() == 100.5


def test_get_best_ask_ignores_bids():
    """测试最优卖价不受买单影响"""
    book = OrderBook()
    sell_order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=10,
    )
    buy_order = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=102.0,
        quantity=20,
    )

    book.add_order(sell_order)
    book.add_order(buy_order)

    # 最优卖价应该只看卖盘
    assert book.get_best_ask() == 101.0


def test_get_depth_default_levels():
    """测试获取完整深度（默认100档）"""
    book = OrderBook()
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.5,
        quantity=20,
    )
    order3 = Order(
        order_id=3,
        agent_id=102,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=15,
    )
    order4 = Order(
        order_id=4,
        agent_id=103,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=101.5,
        quantity=25,
    )

    book.add_order(order1)
    book.add_order(order2)
    book.add_order(order3)
    book.add_order(order4)

    depth = book.get_depth()

    # 验证返回结构
    assert "bids" in depth
    assert "asks" in depth
    # 验证买盘：按价格从高到低
    assert depth["bids"] == [(100.0, 10), (99.5, 20)]
    # 验证卖盘：按价格从低到高
    assert depth["asks"] == [(101.0, 15), (101.5, 25)]


def test_get_depth_custom_levels():
    """测试自定义档位数"""
    book = OrderBook()
    # 添加5个买盘档位
    for i in range(5):
        order = Order(
            order_id=i + 1,
            agent_id=100 + i,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=100.0 - i * 0.5,
            quantity=10 + i,
        )
        book.add_order(order)
    # 添加5个卖盘档位
    for i in range(5):
        order = Order(
            order_id=10 + i + 1,
            agent_id=200 + i,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=101.0 + i * 0.5,
            quantity=20 + i,
        )
        book.add_order(order)

    depth = book.get_depth(levels=3)

    # 应该只返回3档
    assert len(depth["bids"]) == 3
    assert len(depth["asks"]) == 3
    # 买盘前3档：100.0, 99.5, 99.0
    assert depth["bids"][0] == (100.0, 10)
    assert depth["bids"][1] == (99.5, 11)
    assert depth["bids"][2] == (99.0, 12)
    # 卖盘前3档：101.0, 101.5, 102.0
    assert depth["asks"][0] == (101.0, 20)
    assert depth["asks"][1] == (101.5, 21)
    assert depth["asks"][2] == (102.0, 22)


def test_get_depth_insufficient_levels():
    """测试档位数不足时返回所有存在的档位"""
    book = OrderBook()
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=15,
    )

    book.add_order(order1)
    book.add_order(order2)

    # 请求100档，但只有1档
    depth = book.get_depth(levels=100)

    assert len(depth["bids"]) == 1
    assert len(depth["asks"]) == 1
    assert depth["bids"] == [(100.0, 10)]
    assert depth["asks"] == [(101.0, 15)]


def test_get_depth_empty_orderbook():
    """测试空订单簿返回空列表"""
    book = OrderBook()

    depth = book.get_depth()

    assert depth["bids"] == []
    assert depth["asks"] == []


def test_get_depth_one_side_empty():
    """测试只有买盘或只有卖盘"""
    book = OrderBook()
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.5,
        quantity=20,
    )

    book.add_order(order1)
    book.add_order(order2)

    depth = book.get_depth()

    # 买盘有数据
    assert len(depth["bids"]) == 2
    assert depth["bids"] == [(100.0, 10), (99.5, 20)]
    # 卖盘为空
    assert depth["asks"] == []


def test_get_depth_aggregates_same_price():
    """测试同一价格档位的数量聚合"""
    book = OrderBook()
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=20,
    )
    order3 = Order(
        order_id=3,
        agent_id=102,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.5,
        quantity=15,
    )

    book.add_order(order1)
    book.add_order(order2)
    book.add_order(order3)

    depth = book.get_depth()

    # 100.0 档位应该有两个订单，数量聚合为 30
    assert len(depth["bids"]) == 2
    assert depth["bids"][0] == (100.0, 30)
    assert depth["bids"][1] == (99.5, 15)


def test_get_mid_price_normal():
    """测试正常计算中间价"""
    book = OrderBook()
    buy_order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    sell_order = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=102.0,
        quantity=20,
    )

    book.add_order(buy_order)
    book.add_order(sell_order)

    # 中间价 = (100.0 + 102.0) / 2 = 101.0
    assert book.get_mid_price() == 101.0


def test_get_mid_price_only_bids():
    """测试只有买盘时返回 None"""
    book = OrderBook()
    buy_order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )

    book.add_order(buy_order)

    assert book.get_mid_price() is None


def test_get_mid_price_only_asks():
    """测试只有卖盘时返回 None"""
    book = OrderBook()
    sell_order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )

    book.add_order(sell_order)

    assert book.get_mid_price() is None


def test_get_mid_price_empty():
    """测试空订单簿返回 None"""
    book = OrderBook()

    assert book.get_mid_price() is None


def test_get_mid_price_same_price():
    """测试买卖价相同时返回该价格"""
    book = OrderBook()
    buy_order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10,
    )
    sell_order = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=20,
    )

    book.add_order(buy_order)
    book.add_order(sell_order)

    # 中间价 = (100.0 + 100.0) / 2 = 100.0
    assert book.get_mid_price() == 100.0


def test_get_mid_price_multiple_levels():
    """测试多档位时使用最优价计算"""
    book = OrderBook()
    # 添加多个买盘档位
    buy_order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99.0,
        quantity=10,
    )
    buy_order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=20,
    )
    # 添加多个卖盘档位
    sell_order1 = Order(
        order_id=3,
        agent_id=102,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=102.0,
        quantity=15,
    )
    sell_order2 = Order(
        order_id=4,
        agent_id=103,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=101.0,
        quantity=25,
    )

    book.add_order(buy_order1)
    book.add_order(buy_order2)
    book.add_order(sell_order1)
    book.add_order(sell_order2)

    # 最优买价 100.0，最优卖价 101.0
    # 中间价 = (100.0 + 101.0) / 2 = 100.5
    assert book.get_mid_price() == 100.5
