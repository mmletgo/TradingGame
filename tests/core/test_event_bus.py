"""EventBus 测试"""

import pytest

from src.core.event_engine import EventBus, EventType


class TestEventBusInit:
    """测试 EventBus.__init__"""

    def test_create_instance(self) -> None:
        """测试创建实例"""
        event_bus = EventBus()
        assert event_bus is not None
        assert isinstance(event_bus, EventBus)

    def test_initial_state_no_subscribers(self) -> None:
        """测试初始状态无订阅者"""
        event_bus = EventBus()
        assert event_bus._subscribers == {}
        assert len(event_bus._subscribers) == 0

    def test_subscribers_is_dict(self) -> None:
        """测试 _subscribers 是正确类型的字典"""
        event_bus = EventBus()
        assert isinstance(event_bus._subscribers, dict)


class TestEventBusSubscribe:
    """测试 EventBus.subscribe"""

    def test_subscribe_normal(self) -> None:
        """测试正常订阅"""
        event_bus = EventBus()

        def handler(event: object) -> None:
            pass

        event_bus.subscribe(EventType.TICK_START, handler)

        assert EventType.TICK_START in event_bus._subscribers
        assert len(event_bus._subscribers[EventType.TICK_START]) == 1
        assert event_bus._subscribers[EventType.TICK_START][0] is handler

    def test_subscribe_duplicate_handler(self) -> None:
        """测试重复订阅同一处理函数"""
        event_bus = EventBus()

        def handler(event: object) -> None:
            pass

        event_bus.subscribe(EventType.TICK_START, handler)
        event_bus.subscribe(EventType.TICK_START, handler)

        assert len(event_bus._subscribers[EventType.TICK_START]) == 2

    def test_subscribe_multiple_event_types(self) -> None:
        """测试订阅多个事件类型"""
        event_bus = EventBus()

        def handler1(event: object) -> None:
            pass

        def handler2(event: object) -> None:
            pass

        event_bus.subscribe(EventType.TICK_START, handler1)
        event_bus.subscribe(EventType.TICK_END, handler2)

        assert EventType.TICK_START in event_bus._subscribers
        assert EventType.TICK_END in event_bus._subscribers
        assert event_bus._subscribers[EventType.TICK_START][0] is handler1
        assert event_bus._subscribers[EventType.TICK_END][0] is handler2