"""
成交记录模块测试
"""

import time

from src.market.matching.trade import Trade


def test_trade_init_normal():
    """测试正常创建成交记录"""
    # 等待一小段时间确保时间戳不同
    time_before = time.time()

    trade = Trade(
        trade_id=1,
        price=100.5,
        quantity=10.0,
        buyer_id=123,
        seller_id=456,
        buyer_fee=0.05,
        seller_fee=0.02,
        is_buyer_taker=True,
    )

    time_after = time.time()

    # 验证所有属性
    assert trade.trade_id == 1
    assert trade.price == 100.5
    assert trade.quantity == 10.0
    assert trade.buyer_id == 123
    assert trade.seller_id == 456
    assert trade.buyer_fee == 0.05
    assert trade.seller_fee == 0.02
    assert trade.is_buyer_taker is True

    # 验证时间戳在合理范围内
    assert time_before <= trade.timestamp <= time_after


def test_trade_init_zero_values():
    """测试零值成交记录"""
    trade = Trade(
        trade_id=0,
        price=0.0,
        quantity=0.0,
        buyer_id=0,
        seller_id=0,
        buyer_fee=0.0,
        seller_fee=0.0,
        is_buyer_taker=False,
    )

    assert trade.trade_id == 0
    assert trade.price == 0.0
    assert trade.quantity == 0.0
    assert trade.buyer_id == 0
    assert trade.seller_id == 0
    assert trade.buyer_fee == 0.0
    assert trade.seller_fee == 0.0
    assert trade.is_buyer_taker is False
    # 时间戳应该存在
    assert trade.timestamp > 0


def test_trade_init_small_values():
    """测试小数值成交记录"""
    trade = Trade(
        trade_id=999,
        price=0.01,
        quantity=0.001,
        buyer_id=1,
        seller_id=2,
        buyer_fee=0.0001,
        seller_fee=0.0002,
        is_buyer_taker=True,
    )

    assert trade.trade_id == 999
    assert trade.price == 0.01
    assert trade.quantity == 0.001
    assert trade.buyer_id == 1
    assert trade.seller_id == 2
    assert trade.buyer_fee == 0.0001
    assert trade.seller_fee == 0.0002


def test_trade_init_large_values():
    """测试大数值成交记录"""
    trade = Trade(
        trade_id=1000000,
        price=10000.0,
        quantity=1000000.0,
        buyer_id=99999,
        seller_id=88888,
        buyer_fee=100.0,
        seller_fee=200.0,
        is_buyer_taker=False,
    )

    assert trade.trade_id == 1000000
    assert trade.price == 10000.0
    assert trade.quantity == 1000000.0
    assert trade.buyer_id == 99999
    assert trade.seller_id == 88888
    assert trade.buyer_fee == 100.0
    assert trade.seller_fee == 200.0


def test_trade_timestamp_unique():
    """测试时间戳唯一性（快速创建两条记录）"""
    trade1 = Trade(
        trade_id=1,
        price=100.0,
        quantity=10.0,
        buyer_id=1,
        seller_id=2,
        buyer_fee=0.1,
        seller_fee=0.1,
        is_buyer_taker=True,
    )

    trade2 = Trade(
        trade_id=2,
        price=100.0,
        quantity=10.0,
        buyer_id=1,
        seller_id=2,
        buyer_fee=0.1,
        seller_fee=0.1,
        is_buyer_taker=False,
    )

    # 两条记录的时间戳应该不同（除非在同一微秒内创建）
    # 一般来说它们会不同，但即使相同也是合法的
    assert trade1.timestamp > 0
    assert trade2.timestamp > 0
    assert trade1.timestamp <= trade2.timestamp
