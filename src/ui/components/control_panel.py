"""控制面板组件

提供训练/演示的控制按钮和状态显示。
"""

import dearpygui.dearpygui as dpg
from typing import Callable


class ControlPanel:
    """控制面板"""

    def __init__(
        self,
        on_start: Callable[[], None] | None = None,
        on_pause: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        on_speed_change: Callable[[float], None] | None = None,
    ):
        """初始化控制面板

        组件会自动添加到当前DearPyGui上下文中。

        Args:
            on_start: 开始按钮回调
            on_pause: 暂停/继续按钮回调
            on_stop: 停止按钮回调
            on_speed_change: 速度滑块变化回调
        """
        self.on_start = on_start
        self.on_pause = on_pause
        self.on_stop = on_stop
        self.on_speed_change = on_speed_change
        self._is_paused: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        """创建UI组件"""
        with dpg.group(horizontal=True):
            dpg.add_button(label="开始", callback=self._on_start_click, tag="btn_start",
                          width=60)
            dpg.add_button(label="暂停", callback=self._on_pause_click, tag="btn_pause",
                          width=60)
            dpg.add_button(label="停止", callback=self._on_stop_click, tag="btn_stop",
                          width=60)

            dpg.add_spacer(width=20)

            dpg.add_text("速度:")
            dpg.add_slider_float(label="", default_value=1.0, min_value=0.1, max_value=10.0,
                width=100, callback=self._on_speed_slider, tag="speed_slider",
                format="%.1fx")

            dpg.add_spacer(width=20)

            dpg.add_text("Episode: 0", tag="episode_text")
            dpg.add_spacer(width=10)
            dpg.add_text("Tick: 0/0", tag="tick_text")
            dpg.add_spacer(width=10)
            dpg.add_text("价格: 0.00", tag="price_text")

    def _on_start_click(self, sender: int | str | None = None,
                       app_data: int | None = None) -> None:
        """开始按钮点击处理"""
        if self.on_start:
            self.on_start()
        dpg.configure_item("btn_start", enabled=False)

    def _on_pause_click(self, sender: int | str | None = None,
                       app_data: int | None = None) -> None:
        """暂停/继续按钮点击处理"""
        self._is_paused = not self._is_paused
        if self._is_paused:
            dpg.set_item_label("btn_pause", "继续")
        else:
            dpg.set_item_label("btn_pause", "暂停")

        if self.on_pause:
            self.on_pause()

    def _on_stop_click(self, sender: int | str | None = None,
                      app_data: int | None = None) -> None:
        """停止按钮点击处理"""
        if self.on_stop:
            self.on_stop()
        dpg.configure_item("btn_start", enabled=True)
        dpg.set_item_label("btn_pause", "暂停")
        self._is_paused = False

    def _on_speed_slider(self, sender: int | str, value: float) -> None:
        """速度滑块变化处理"""
        if self.on_speed_change:
            self.on_speed_change(value)

    def update_status(self, episode: int, tick: int, total_ticks: int, price: float) -> None:
        """更新状态显示

        Args:
            episode: 当前episode编号
            tick: 当前tick编号
            total_ticks: 总tick数
            price: 当前价格
        """
        dpg.set_value("episode_text", f"Episode: {episode}")
        dpg.set_value("tick_text", f"Tick: {tick}/{total_ticks}")
        dpg.set_value("price_text", f"价格: {price:.2f}")

    def reset(self) -> None:
        """重置控制面板状态"""
        dpg.configure_item("btn_start", enabled=True)
        dpg.set_item_label("btn_pause", "暂停")
        self._is_paused = False
        self.update_status(0, 0, 0, 0.0)

    @property
    def is_paused(self) -> bool:
        """获取当前暂停状态"""
        return self._is_paused
