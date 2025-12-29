"""
撮合引擎模块测试
"""

from src.config.config import MarketConfig
from src.core.event_engine.event_bus import EventBus
from src.core.event_engine.events import Event, EventType
from src.market.matching.matching_engine import MatchingEngine


def test_matching_engine_init_normal():
    """测试正常创建撮合引擎"""
    event_bus = EventBus()
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )

    engine = MatchingEngine(event_bus, config)

    # 验证引擎创建成功
    assert engine is not None
    assert engine._event_bus is event_bus
    assert engine._config is config
    assert engine._orderbook is not None
    assert engine._next_trade_id == 1


def test_matching_engine_init_subscribes_events():
    """测试撮合引擎订阅事件"""
    event_bus = EventBus()
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )

    # 创建引擎应该订阅事件
    engine = MatchingEngine(event_bus, config)

    # 验证事件总线有订阅者（通过发布事件并捕获来验证）
    # 由于我们使用的是占位函数，这里只验证不会报错
    test_event = Event(
        event_type=EventType.ORDER_PLACED,
        timestamp=0.0,
        data={"order_id": 1},
    )

    # 发布事件应该不会抛出异常
    event_bus.publish(test_event)

    cancel_event = Event(
        event_type=EventType.ORDER_CANCELLED,
        timestamp=0.0,
        data={"order_id": 1},
    )
    event_bus.publish(cancel_event)


def test_matching_engine_init_different_configs():
    """测试不同配置创建撮合引擎"""
    event_bus = EventBus()

    # 不同的 tick_size
    config1 = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )
    engine1 = MatchingEngine(event_bus, config1)
    assert engine1._orderbook.tick_size == 0.01

    # 不同的 depth
    config2 = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=50,
    )
    engine2 = MatchingEngine(event_bus, config2)
    assert engine2._config.depth == 50


def test_matching_engine_multiple_engines():
    """测试创建多个撮合引擎实例"""
    event_bus = EventBus()
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )

    engine1 = MatchingEngine(event_bus, config)
    engine2 = MatchingEngine(event_bus, config)

    # 两个引擎应该是独立的实例
    assert engine1 is not engine2
    assert engine1._orderbook is not engine2._orderbook
    assert engine1._next_trade_id == 1
    assert engine2._next_trade_id == 1


# ==================== calculate_fee 相关测试 ====================


def test_calculate_fee_retail_maker():
    """测试散户挂单费率（万2）"""
    event_bus = EventBus()
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(event_bus, config)

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
    event_bus = EventBus()
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(event_bus, config)

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
    event_bus = EventBus()
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(event_bus, config)

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
    event_bus = EventBus()
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(event_bus, config)

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
    event_bus = EventBus()
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(event_bus, config)

    # 未注册的 Agent 应该使用默认散户费率
    fee = engine.calculate_fee(agent_id=999, amount=10000.0, is_maker=True)
    assert fee == 2.0  # 万2

    fee = engine.calculate_fee(agent_id=999, amount=10000.0, is_maker=False)
    assert fee == 5.0  # 万5


def test_calculate_fee_zero_amount():
    """测试成交金额为 0 时手续费为 0"""
    event_bus = EventBus()
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(event_bus, config)

    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)

    # 成交金额为 0，手续费应该也是 0
    fee = engine.calculate_fee(agent_id=1, amount=0.0, is_maker=True)
    assert fee == 0.0

    fee = engine.calculate_fee(agent_id=1, amount=0.0, is_maker=False)
    assert fee == 0.0


def test_register_agent():
    """测试注册 Agent 费率"""
    event_bus = EventBus()
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(event_bus, config)

    # 注册 Agent
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)

    # 验证费率已设置
    assert engine._fee_rates[1] == (0.0002, 0.0005)

    # 更新费率
    engine.register_agent(agent_id=1, maker_rate=0.0, taker_rate=0.0001)

    # 验证费率已更新
    assert engine._fee_rates[1] == (0.0, 0.0001)


def test_unregister_agent():
    """测试注销 Agent 费率"""
    event_bus = EventBus()
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(event_bus, config)

    # 注册 Agent
    engine.register_agent(agent_id=1, maker_rate=0.0002, taker_rate=0.0005)
    assert 1 in engine._fee_rates

    # 注销 Agent
    engine.unregister_agent(agent_id=1)
    assert 1 not in engine._fee_rates

    # 注销不存在的 Agent 不应该报错
    engine.unregister_agent(agent_id=999)  # 不应该抛出异常


def test_multiple_agents_different_rates():
    """测试多个 Agent 使用不同费率"""
    event_bus = EventBus()
    config = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )
    engine = MatchingEngine(event_bus, config)

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
