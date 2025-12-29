"""
测试订单模块
"""

import pytest
import time

from src.market.orderbook.order import Order, OrderSide, OrderType


def test_create_limit_buy_order():
    """测试创建限价买单"""
    order = Order(
        order_id=1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.5,
        quantity=10.0,
    )

    assert order.order_id == 1
    assert order.agent_id == 100
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.LIMIT
    assert order.price == 100.5
    assert order.quantity == 10.0
    assert order.filled_quantity == 0.0
    assert order.timestamp > 0


def test_create_limit_sell_order():
    """测试创建限价卖单"""
    order = Order(
        order_id=2,
        agent_id=101,
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        price=99.8,
        quantity=5.0,
    )

    assert order.order_id == 2
    assert order.agent_id == 101
    assert order.side == OrderSide.SELL
    assert order.order_type == OrderType.LIMIT
    assert order.price == 99.8
    assert order.quantity == 5.0
    assert order.filled_quantity == 0.0


def test_create_market_buy_order():
    """测试创建市价买单"""
    order = Order(
        order_id=3,
        agent_id=102,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        price=0.0,
        quantity=20.0,
    )

    assert order.order_id == 3
    assert order.agent_id == 102
    assert order.side == OrderSide.BUY
    assert order.order_type == OrderType.MARKET
    assert order.price == 0.0
    assert order.quantity == 20.0
    assert order.filled_quantity == 0.0


def test_create_market_sell_order():
    """测试创建市价卖单"""
    order = Order(
        order_id=4,
        agent_id=103,
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        price=0.0,
        quantity=15.0,
    )

    assert order.order_id == 4
    assert order.agent_id == 103
    assert order.side == OrderSide.SELL
    assert order.order_type == OrderType.MARKET
    assert order.price == 0.0
    assert order.quantity == 15.0
    assert order.filled_quantity == 0.0


def test_timestamp_is_set():
    """测试时间戳自动设置"""
    before = time.time()
    order = Order(
        order_id=5,
        agent_id=104,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=1.0,
    )
    after = time.time()

    assert before <= order.timestamp <= after


def test_filled_quantity_initial_zero():
    """测试已成交数量初始为0"""
    order = Order(
        order_id=6,
        agent_id=105,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=100.0,
        quantity=10.0,
    )

    assert order.filled_quantity == 0.0


def test_order_side_enum_values():
    """测试订单方向枚举值"""
    assert OrderSide.BUY == 1
    assert OrderSide.SELL == -1


def test_order_type_enum_values():
    """测试订单类型枚举值"""
    assert OrderType.LIMIT == 1
    assert OrderType.MARKET == 2
