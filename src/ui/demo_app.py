"""演示模式UI应用

从检查点加载训练好的模型，仅运行展示，不进化。
支持速度控制。使用DearPyGui作为GUI框架。
"""

import dearpygui.dearpygui as dpg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.trainer import Trainer

from src.ui.data_collector import UIDataCollector, UIDataSnapshot
from src.ui.ui_controller import UIController
from src.ui.components import OrderBookPanel, ChartPanel, TradesPanel, ControlPanel


class DemoUIApp:
    """演示模式UI应用

    从检查点加载训练好的模型，仅运行展示，不进化。
    支持速度控制，适合展示训练效果。

    Attributes:
        trainer: 已初始化的训练器
        episode_length: 每个episode的tick数
        data_collector: UI数据采集器
        controller: UI控制器
    """

    trainer: "Trainer"
    episode_length: int
    data_collector: UIDataCollector
    controller: UIController

    # UI组件
    orderbook_panel: "OrderBookPanel | None"
    chart_panel: "ChartPanel | None"
    trades_panel: "TradesPanel | None"
    control_panel: "ControlPanel | None"

    def __init__(
        self, trainer: "Trainer", checkpoint_path: str | None = None
    ) -> None:
        """初始化演示模式UI应用

        Args:
            trainer: 已初始化的训练器
            checkpoint_path: 检查点路径（可选），用于加载训练好的模型
        """
        self.trainer = trainer
        self.episode_length = trainer.config.training.episode_length

        # 加载检查点
        if checkpoint_path:
            trainer.load_checkpoint(checkpoint_path)

        # 创建数据采集器和控制器
        self.data_collector = UIDataCollector(history_length=self.episode_length)
        self.controller = UIController(
            trainer=trainer,
            data_collector=self.data_collector,
            sample_rate=1,  # 每tick采集
        )

        # UI组件
        self.orderbook_panel = None
        self.chart_panel = None
        self.trades_panel = None
        self.control_panel = None

        self._setup_dpg()
        self._setup_ui()

    def _setup_dpg(self) -> None:
        """初始化DearPyGui上下文和视口"""
        dpg.create_context()
        dpg.create_viewport(
            title="NEAT Trading Simulator - 演示模式",
            width=1400,
            height=900,
        )
        dpg.setup_dearpygui()

    def _setup_ui(self) -> None:
        """创建UI布局

        布局结构：
        - 顶部：控制面板（开始/暂停/停止按钮，速度滑块，状态显示）
        - 中部左侧：订单簿面板
        - 中部右侧：价格/净值图表面板
        - 底部：成交记录面板
        """
        with dpg.window(
            label="主窗口",
            tag="main_window",
            width=-1,
            height=-1,
            no_title_bar=True,
            no_move=True,
            no_resize=True,
        ):
            # 控制面板（顶部）
            self.control_panel = ControlPanel(
                parent="main_window",
                on_start=self._on_start,
                on_pause=self._on_pause,
                on_stop=self._on_stop,
                on_speed_change=self._on_speed_change,
            )

            dpg.add_separator()

            # 主内容区（订单簿 + 图表）
            with dpg.group(horizontal=True):
                # 左侧：订单簿
                self.orderbook_panel = OrderBookPanel(parent="main_window")

                # 右侧：图表
                self.chart_panel = ChartPanel(parent="main_window")

            dpg.add_separator()

            # 底部：成交记录
            with dpg.group(horizontal=True):
                self.trades_panel = TradesPanel(parent="main_window")

        dpg.set_primary_window("main_window", True)

    def _on_start(self) -> None:
        """开始演示按钮回调

        启动演示模式，在后台线程中运行演示循环（不进化）。
        """
        self.controller.start_demo()

    def _on_pause(self) -> None:
        """暂停/继续按钮回调

        切换暂停状态。
        """
        if self.controller.is_paused():
            self.controller.resume()
        else:
            self.controller.pause()

    def _on_stop(self) -> None:
        """停止演示按钮回调

        停止演示并清理资源。
        """
        self.controller.stop()

    def _on_speed_change(self, speed: float) -> None:
        """速度变化回调

        演示模式支持速度控制，调整tick间隔。

        Args:
            speed: 速度倍率（1.0=正常，10.0=10倍速）
        """
        self.controller.set_speed(speed)

    def _update_ui(self) -> None:
        """更新UI（每帧调用）

        从控制器获取最新数据快照，更新各个UI组件。
        """
        # 获取最新数据
        data: UIDataSnapshot | None = self.controller.get_latest_data()
        if data is None:
            return

        # 更新控制面板状态
        if self.control_panel:
            self.control_panel.update_status(
                episode=data.episode,
                tick=data.tick,
                total_ticks=self.episode_length,
                price=data.last_price,
            )

        # 更新订单簿
        if self.orderbook_panel:
            self.orderbook_panel.update(data.bids, data.asks)

        # 更新图表
        if self.chart_panel:
            self.chart_panel.update_price(data.price_history)
            self.chart_panel.update_equity(data.equity_history, data.population_stats)

        # 更新成交记录
        if self.trades_panel:
            self.trades_panel.update(data.recent_trades)

    def run(self) -> None:
        """运行主循环

        显示视口并进入主循环，每帧更新UI。
        循环结束后清理资源。
        """
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            self._update_ui()
            dpg.render_dearpygui_frame()

        # 清理
        self.controller.stop()
        dpg.destroy_context()
