"""事件总线 - 模块间解耦通信的核心组件"""

from collections.abc import Callable

from .events import Event, EventType


class EventBus:
    """事件总线，支持发布/订阅模式的事件系统"""

    def __init__(self) -> None:
        """初始化事件总线

        创建订阅者字典，键为事件类型，值为回调函数列表
        """
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """订阅指定类型的事件

        Args:
            event_type: 要订阅的事件类型
            handler: 事件处理函数，接收 Event 作为参数
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """取消订阅指定类型的事件

        Args:
            event_type: 要取消订阅的事件类型
            handler: 要移除的事件处理函数

        Note:
            如果 event_type 未被订阅或 handler 不在订阅列表中，静默处理不报错
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                # handler 不在列表中，静默处理
                pass

    def publish(self, event: Event) -> None:
        """发布事件，通知所有订阅者

        Args:
            event: 要发布的事件对象

        Note:
            如果该事件类型没有订阅者，静默处理不报错
        """
        event_type = event.event_type
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                handler(event)

    def clear(self) -> None:
        """清除所有订阅

        清空订阅者字典，移除所有事件类型的订阅关系。
        主要用于系统重置场景，如重新开始训练或单元测试。
        """
        self._subscribers.clear()
