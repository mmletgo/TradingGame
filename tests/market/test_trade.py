"""
成交记录模块测试
"""

from src.market.matching.trade import Trade


def test_trade_init_normal():
    """测试正常创建成交记录"""
    trade = Trade(
        trade_id=1,
        price=100.5,
        quantity=10,
        buyer_id=123,
        seller_id=456,
        buyer_fee=0.05,
        seller_fee=0.02,
        is_buyer_taker=True,
    )

    # 验证所有属性
    assert trade.trade_id == 1
    assert trade.price == 100.5
    assert trade.quantity == 10
    assert trade.buyer_id == 123
    assert trade.seller_id == 456
    assert trade.buyer_fee == 0.05
    assert trade.seller_fee == 0.02
    assert trade.is_buyer_taker is True

    # 训练模式下时间戳默认为 0.0（优化性能，避免 time.time() 调用）
    assert trade.timestamp == 0.0


def test_trade_init_zero_values():
    """测试零值成交记录"""
    trade = Trade(
        trade_id=0,
        price=0.0,
        quantity=0,
        buyer_id=0,
        seller_id=0,
        buyer_fee=0.0,
        seller_fee=0.0,
        is_buyer_taker=False,
    )

    assert trade.trade_id == 0
    assert trade.price == 0.0
    assert trade.quantity == 0
    assert trade.buyer_id == 0
    assert trade.seller_id == 0
    assert trade.buyer_fee == 0.0
    assert trade.seller_fee == 0.0
    assert trade.is_buyer_taker is False
    # 训练模式下时间戳默认为 0.0
    assert trade.timestamp == 0.0


def test_trade_init_small_values():
    """测试小数值成交记录"""
    trade = Trade(
        trade_id=999,
        price=0.01,
        quantity=1,
        buyer_id=1,
        seller_id=2,
        buyer_fee=0.0001,
        seller_fee=0.0002,
        is_buyer_taker=True,
    )

    assert trade.trade_id == 999
    assert trade.price == 0.01
    assert trade.quantity == 1
    assert trade.buyer_id == 1
    assert trade.seller_id == 2
    assert trade.buyer_fee == 0.0001
    assert trade.seller_fee == 0.0002


def test_trade_init_large_values():
    """测试大数值成交记录"""
    trade = Trade(
        trade_id=1000000,
        price=10000.0,
        quantity=1000000,
        buyer_id=99999,
        seller_id=88888,
        buyer_fee=100.0,
        seller_fee=200.0,
        is_buyer_taker=False,
    )

    assert trade.trade_id == 1000000
    assert trade.price == 10000.0
    assert trade.quantity == 1000000
    assert trade.buyer_id == 99999
    assert trade.seller_id == 88888
    assert trade.buyer_fee == 100.0
    assert trade.seller_fee == 200.0


def test_trade_timestamp_default():
    """测试时间戳默认值为 0.0（训练模式优化）"""
    trade1 = Trade(
        trade_id=1,
        price=100.0,
        quantity=10,
        buyer_id=1,
        seller_id=2,
        buyer_fee=0.1,
        seller_fee=0.1,
        is_buyer_taker=True,
    )

    trade2 = Trade(
        trade_id=2,
        price=100.0,
        quantity=10,
        buyer_id=1,
        seller_id=2,
        buyer_fee=0.1,
        seller_fee=0.1,
        is_buyer_taker=False,
    )

    # 训练模式下时间戳默认都是 0.0
    assert trade1.timestamp == 0.0
    assert trade2.timestamp == 0.0


def test_trade_timestamp_custom():
    """测试可以传入自定义时间戳"""
    custom_timestamp = 1234567890.123
    trade = Trade(
        trade_id=1,
        price=100.0,
        quantity=10,
        buyer_id=1,
        seller_id=2,
        buyer_fee=0.1,
        seller_fee=0.1,
        is_buyer_taker=True,
        timestamp=custom_timestamp,
    )

    assert trade.timestamp == custom_timestamp
