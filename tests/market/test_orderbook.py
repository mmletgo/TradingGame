"""
测试订单簿模块
"""

import pytest
from collections import OrderedDict

from src.market.orderbook.orderbook import PriceLevel
from src.market.orderbook.order import Order, OrderSide, OrderType


def test_create_price_level():
    """测试创建价格档位"""
    level = PriceLevel(price=100.5)

    assert level.price == 100.5
    assert isinstance(level.orders, OrderedDict)
    assert len(level.orders) == 0
    assert level.total_quantity == 0.0


def test_create_price_level_zero():
    """测试创建价格为0的档位"""
    level = PriceLevel(price=0.0)

    assert level.price == 0.0
    assert isinstance(level.orders, OrderedDict)
    assert len(level.orders) == 0
    assert level.total_quantity == 0.0


def test_create_price_level_negative():
    """测试创建负价格档位（虽然实际场景不会）"""
    level = PriceLevel(price=-10.0)

    assert level.price == -10.0
    assert isinstance(level.orders, OrderedDict)
    assert len(level.orders) == 0
    assert level.total_quantity == 0.0


def test_add_order_single():
    """测试添加单个订单"""
    level = PriceLevel(price=100.5)
    order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=10.0,
    )

    level.add_order(order)

    assert len(level.orders) == 1
    assert level.orders[1] is order
    assert level.total_quantity == 10.0


def test_add_order_multiple():
    """测试添加多个订单"""
    level = PriceLevel(price=100.0)
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=5.0,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=15.0,
    )
    order3 = Order(
        order_id=3,
        agent_id=102,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=7.5,
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
    assert level.total_quantity == 27.5


def test_remove_order_exists():
    """测试移除存在的订单"""
    level = PriceLevel(price=100.0)
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10.0,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=20.0,
    )

    level.add_order(order1)
    level.add_order(order2)

    # 移除中间的订单
    removed = level.remove_order(order_id=1)

    assert removed is order1
    assert len(level.orders) == 1
    assert 1 not in level.orders
    assert 2 in level.orders
    assert level.total_quantity == 20.0


def test_remove_order_not_exists():
    """测试移除不存在的订单"""
    level = PriceLevel(price=100.0)
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10.0,
    )

    level.add_order(order1)

    # 移除不存在的订单
    removed = level.remove_order(order_id=999)

    assert removed is None
    assert len(level.orders) == 1
    assert level.total_quantity == 10.0


def test_remove_order_empty():
    """测试从空价格档位移除订单"""
    level = PriceLevel(price=100.0)

    removed = level.remove_order(order_id=1)

    assert removed is None
    assert len(level.orders) == 0
    assert level.total_quantity == 0.0


def test_remove_order_fifo_preserved():
    """测试移除订单后 FIFO 顺序保持正确"""
    level = PriceLevel(price=100.0)
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=5.0,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=15.0,
    )
    order3 = Order(
        order_id=3,
        agent_id=102,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=7.5,
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
    assert level.total_quantity == 12.5  # 5.0 + 7.5


def test_get_volume_empty():
    """测试空档位返回0"""
    level = PriceLevel(price=100.0)

    assert level.get_volume() == 0.0


def test_get_volume_with_orders():
    """测试有订单返回总量"""
    level = PriceLevel(price=100.0)
    order1 = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10.0,
    )
    order2 = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=20.5,
    )

    level.add_order(order1)
    level.add_order(order2)

    assert level.get_volume() == 30.5
