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
