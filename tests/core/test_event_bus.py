"""EventBus 测试"""

import pytest

from src.core.event_engine import EventBus, Event, EventType


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


class TestEventBusUnsubscribe:
    """测试 EventBus.unsubscribe"""

    def test_unsubscribe_normal(self) -> None:
        """测试正常取消订阅"""
        event_bus = EventBus()

        def handler(event: object) -> None:
            pass

        event_bus.subscribe(EventType.TICK_START, handler)
        assert len(event_bus._subscribers[EventType.TICK_START]) == 1

        event_bus.unsubscribe(EventType.TICK_START, handler)
        assert len(event_bus._subscribers[EventType.TICK_START]) == 0

    def test_unsubscribe_non_existent_handler(self) -> None:
        """测试取消不存在的订阅"""
        event_bus = EventBus()

        def handler1(event: object) -> None:
            pass

        def handler2(event: object) -> None:
            pass

        event_bus.subscribe(EventType.TICK_START, handler1)
        assert len(event_bus._subscribers[EventType.TICK_START]) == 1

        # 取消一个从未订阅的 handler，应该静默处理
        event_bus.unsubscribe(EventType.TICK_START, handler2)
        # handler1 应该还在
        assert len(event_bus._subscribers[EventType.TICK_START]) == 1
        assert event_bus._subscribers[EventType.TICK_START][0] is handler1

    def test_unsubscribe_non_existent_event_type(self) -> None:
        """测试取消未订阅的事件类型"""
        event_bus = EventBus()

        def handler(event: object) -> None:
            pass

        # 取消一个从未订阅的事件类型，应该静默处理不报错
        event_bus.unsubscribe(EventType.TICK_START, handler)
        assert EventType.TICK_START not in event_bus._subscribers

    def test_unsubscribe_multiple_handlers(self) -> None:
        """测试取消多个 handler 中的一个"""
        event_bus = EventBus()

        def handler1(event: object) -> None:
            pass

        def handler2(event: object) -> None:
            pass

        event_bus.subscribe(EventType.TICK_START, handler1)
        event_bus.subscribe(EventType.TICK_START, handler2)
        assert len(event_bus._subscribers[EventType.TICK_START]) == 2

        # 取消 handler1，handler2 应该还在
        event_bus.unsubscribe(EventType.TICK_START, handler1)
        assert len(event_bus._subscribers[EventType.TICK_START]) == 1
        assert event_bus._subscribers[EventType.TICK_START][0] is handler2


class TestEventBusPublish:
    """测试 EventBus.publish"""

    def test_publish_normal(self) -> None:
        """测试正常发布事件"""
        event_bus = EventBus()
        received_events: list[Event] = []

        def handler(event: Event) -> None:
            received_events.append(event)

        event_bus.subscribe(EventType.TICK_START, handler)
        event = Event(event_type=EventType.TICK_START, timestamp=100.0, data={"tick": 1})
        event_bus.publish(event)

        assert len(received_events) == 1
        assert received_events[0] is event
        assert received_events[0].event_type == EventType.TICK_START
        assert received_events[0].timestamp == 100.0
        assert received_events[0].data == {"tick": 1}

    def test_publish_no_subscribers(self) -> None:
        """测试无订阅者时发布事件"""
        event_bus = EventBus()
        event = Event(event_type=EventType.TICK_START, timestamp=100.0, data={"tick": 1})

        # 没有订阅者，应该静默处理不报错
        event_bus.publish(event)

    def test_publish_multiple_subscribers(self) -> None:
        """测试多个订阅者接收事件"""
        event_bus = EventBus()
        received_events_1: list[Event] = []
        received_events_2: list[Event] = []
        received_events_3: list[Event] = []

        def handler1(event: Event) -> None:
            received_events_1.append(event)

        def handler2(event: Event) -> None:
            received_events_2.append(event)

        def handler3(event: Event) -> None:
            received_events_3.append(event)

        event_bus.subscribe(EventType.TICK_START, handler1)
        event_bus.subscribe(EventType.TICK_START, handler2)
        event_bus.subscribe(EventType.TICK_START, handler3)

        event = Event(event_type=EventType.TICK_START, timestamp=100.0, data={"tick": 1})
        event_bus.publish(event)

        # 所有订阅者都应该收到事件
        assert len(received_events_1) == 1
        assert len(received_events_2) == 1
        assert len(received_events_3) == 1
        # 所有订阅者收到的是同一个事件对象
        assert received_events_1[0] is event
        assert received_events_2[0] is event
        assert received_events_3[0] is event

    def test_publish_different_event_types(self) -> None:
        """测试发布不同类型的事件"""
        event_bus = EventBus()
        received_events: list[Event] = []

        def handler(event: Event) -> None:
            received_events.append(event)

        # 只订阅 TICK_START
        event_bus.subscribe(EventType.TICK_START, handler)

        # 发布 TICK_START 事件
        event1 = Event(event_type=EventType.TICK_START, timestamp=100.0)
        event_bus.publish(event1)

        # 发布 TICK_END 事件（没有订阅者）
        event2 = Event(event_type=EventType.TICK_END, timestamp=200.0)
        event_bus.publish(event2)

        # 应该只收到 TICK_START 事件
        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.TICK_START

    def test_publish_empty_handler_list(self) -> None:
        """测试订阅者列表为空时发布"""
        event_bus = EventBus()

        # 订阅后取消订阅，导致列表为空
        def handler(event: Event) -> None:
            pass

        event_bus.subscribe(EventType.TICK_START, handler)
        event_bus.unsubscribe(EventType.TICK_START, handler)
        assert len(event_bus._subscribers[EventType.TICK_START]) == 0

        # 发布事件应该静默处理
        event = Event(event_type=EventType.TICK_START, timestamp=100.0)
        event_bus.publish(event)


class TestEventBusClear:
    """测试 EventBus.clear"""

    def test_clear_removes_all_subscribers(self) -> None:
        """测试清除后无订阅者"""
        event_bus = EventBus()

        # 添加多个订阅
        def handler1(event: Event) -> None:
            pass

        def handler2(event: Event) -> None:
            pass

        event_bus.subscribe(EventType.TICK_START, handler1)
        event_bus.subscribe(EventType.TICK_END, handler2)
        assert len(event_bus._subscribers) == 2

        # 清除所有订阅
        event_bus.clear()
        assert len(event_bus._subscribers) == 0
        assert EventType.TICK_START not in event_bus._subscribers
        assert EventType.TICK_END not in event_bus._subscribers

    def test_clear_empty_event_bus(self) -> None:
        """测试清除空的事件总线"""
        event_bus = EventBus()
        assert len(event_bus._subscribers) == 0

        # 清空操作应该不会报错
        event_bus.clear()
        assert len(event_bus._subscribers) == 0

    def test_clear_then_subscribe(self) -> None:
        """测试清除后可以重新订阅"""
        event_bus = EventBus()

        # 添加订阅
        def handler(event: Event) -> None:
            pass

        event_bus.subscribe(EventType.TICK_START, handler)
        assert len(event_bus._subscribers) == 1

        # 清除
        event_bus.clear()
        assert len(event_bus._subscribers) == 0

        # 重新订阅
        event_bus.subscribe(EventType.TICK_END, handler)
        assert len(event_bus._subscribers) == 1
        assert EventType.TICK_END in event_bus._subscribers


class TestEventBusSubscribeWithId:
    """测试 EventBus.subscribe_with_id"""

    def test_subscribe_with_id_normal(self) -> None:
        """测试带 ID 正常订阅"""
        event_bus = EventBus()

        def handler(event: Event) -> None:
            pass

        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 1, handler)

        assert EventType.TRADE_EXECUTED in event_bus._subscriber_ids
        assert 1 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]
        assert event_bus._subscriber_ids[EventType.TRADE_EXECUTED][1] is handler

    def test_subscribe_with_id_multiple_agents(self) -> None:
        """测试多个 Agent 订阅同一事件类型"""
        event_bus = EventBus()

        def handler1(event: Event) -> None:
            pass

        def handler2(event: Event) -> None:
            pass

        def handler3(event: Event) -> None:
            pass

        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 1, handler1)
        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 2, handler2)
        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 3, handler3)

        assert EventType.TRADE_EXECUTED in event_bus._subscriber_ids
        assert len(event_bus._subscriber_ids[EventType.TRADE_EXECUTED]) == 3
        assert 1 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]
        assert 2 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]
        assert 3 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]

    def test_subscribe_with_id_replaces_handler(self) -> None:
        """测试同一 ID 重复订阅会替换 handler"""
        event_bus = EventBus()

        def handler1(event: Event) -> None:
            pass

        def handler2(event: Event) -> None:
            pass

        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 1, handler1)
        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 1, handler2)

        assert len(event_bus._subscriber_ids[EventType.TRADE_EXECUTED]) == 1
        assert event_bus._subscriber_ids[EventType.TRADE_EXECUTED][1] is handler2


class TestEventBusUnsubscribeWithId:
    """测试 EventBus.unsubscribe_with_id"""

    def test_unsubscribe_with_id_normal(self) -> None:
        """测试带 ID 正常取消订阅"""
        event_bus = EventBus()

        def handler(event: Event) -> None:
            pass

        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 1, handler)
        assert 1 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]

        event_bus.unsubscribe_with_id(EventType.TRADE_EXECUTED, 1)
        assert 1 not in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]

    def test_unsubscribe_with_id_non_existent_id(self) -> None:
        """测试取消不存在的 ID 订阅"""
        event_bus = EventBus()

        def handler(event: Event) -> None:
            pass

        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 1, handler)
        assert 1 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]

        # 取消不存在的 ID，应静默处理
        event_bus.unsubscribe_with_id(EventType.TRADE_EXECUTED, 999)
        # 原有订阅不受影响
        assert 1 in event_bus._subscriber_ids[EventType.TRADE_EXECUTED]

    def test_unsubscribe_with_id_non_existent_event_type(self) -> None:
        """测试取消未订阅的事件类型"""
        event_bus = EventBus()

        # 取消未订阅的事件类型，应静默处理不报错
        event_bus.unsubscribe_with_id(EventType.TRADE_EXECUTED, 1)
        assert EventType.TRADE_EXECUTED not in event_bus._subscriber_ids


class TestEventBusPublishTargeted:
    """测试 EventBus.publish 定向发送"""

    def test_publish_targeted_normal(self) -> None:
        """测试定向发送事件"""
        event_bus = EventBus()
        received_events_1: list[Event] = []
        received_events_2: list[Event] = []
        received_events_3: list[Event] = []

        def handler1(event: Event) -> None:
            received_events_1.append(event)

        def handler2(event: Event) -> None:
            received_events_2.append(event)

        def handler3(event: Event) -> None:
            received_events_3.append(event)

        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 1, handler1)
        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 2, handler2)
        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 3, handler3)

        # 定向发送给 ID 1 和 2
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=100.0,
            data={"trade_id": 1},
            target_ids={1, 2},
        )
        event_bus.publish(event)

        # 只有 ID 1 和 2 收到事件
        assert len(received_events_1) == 1
        assert len(received_events_2) == 1
        assert len(received_events_3) == 0

    def test_publish_targeted_to_nonexistent(self) -> None:
        """测试发送给不存在的 ID"""
        event_bus = EventBus()
        received_events: list[Event] = []

        def handler(event: Event) -> None:
            received_events.append(event)

        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 1, handler)

        # 定向发送给不存在的 ID
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=100.0,
            data={"trade_id": 1},
            target_ids={999, 1000},  # 不存在的 ID
        )
        event_bus.publish(event)

        # 没有人收到事件
        assert len(received_events) == 0

    def test_publish_targeted_partial_existence(self) -> None:
        """测试定向发送给部分存在的 ID"""
        event_bus = EventBus()
        received_events_1: list[Event] = []
        received_events_2: list[Event] = []

        def handler1(event: Event) -> None:
            received_events_1.append(event)

        def handler2(event: Event) -> None:
            received_events_2.append(event)

        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 1, handler1)
        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 2, handler2)

        # 定向发送给 ID 1 和 999（不存在）
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=100.0,
            data={"trade_id": 1},
            target_ids={1, 999},
        )
        event_bus.publish(event)

        # 只有 ID 1 收到事件
        assert len(received_events_1) == 1
        assert len(received_events_2) == 0

    def test_publish_broadcast_does_not_use_subscriber_ids(self) -> None:
        """测试广播发送不使用 _subscriber_ids"""
        event_bus = EventBus()
        received_subscriber: list[Event] = []
        received_subscriber_id: list[Event] = []

        def broadcast_handler(event: Event) -> None:
            received_subscriber.append(event)

        def targeted_handler(event: Event) -> None:
            received_subscriber_id.append(event)

        # 使用两种不同的订阅方式
        event_bus.subscribe(EventType.TRADE_EXECUTED, broadcast_handler)
        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 1, targeted_handler)

        # 广播发送（target_ids=None）
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=100.0,
            data={"trade_id": 1},
        )
        event_bus.publish(event)

        # 只有 _subscribers 中的 handler 收到事件
        assert len(received_subscriber) == 1
        assert len(received_subscriber_id) == 0

    def test_publish_targeted_uses_subscriber_ids(self) -> None:
        """测试定向发送使用 _subscriber_ids"""
        event_bus = EventBus()
        received_subscriber: list[Event] = []
        received_subscriber_id: list[Event] = []

        def broadcast_handler(event: Event) -> None:
            received_subscriber.append(event)

        def targeted_handler(event: Event) -> None:
            received_subscriber_id.append(event)

        # 使用两种不同的订阅方式
        event_bus.subscribe(EventType.TRADE_EXECUTED, broadcast_handler)
        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 1, targeted_handler)

        # 定向发送（target_ids={1}）
        event = Event(
            event_type=EventType.TRADE_EXECUTED,
            timestamp=100.0,
            data={"trade_id": 1},
            target_ids={1},
        )
        event_bus.publish(event)

        # 只有 _subscriber_ids 中的 handler 收到事件
        assert len(received_subscriber) == 0
        assert len(received_subscriber_id) == 1


class TestEventBusClearWithSubscriberIds:
    """测试 EventBus.clear 同时清除 _subscriber_ids"""

    def test_clear_clears_subscriber_ids(self) -> None:
        """测试 clear 同时清除 _subscriber_ids"""
        event_bus = EventBus()

        def handler1(event: Event) -> None:
            pass

        def handler2(event: Event) -> None:
            pass

        # 添加两种订阅
        event_bus.subscribe(EventType.TICK_START, handler1)
        event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, 1, handler2)

        assert len(event_bus._subscribers) == 1
        assert len(event_bus._subscriber_ids) == 1

        # 清除所有订阅
        event_bus.clear()

        assert len(event_bus._subscribers) == 0
        assert len(event_bus._subscriber_ids) == 0
        assert EventType.TICK_START not in event_bus._subscribers
        assert EventType.TRADE_EXECUTED not in event_bus._subscriber_ids