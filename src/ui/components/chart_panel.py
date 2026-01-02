"""图表面板组件

显示价格曲线和种群资产曲线（2x2网格布局）。
"""

import dearpygui.dearpygui as dpg
from typing import TYPE_CHECKING, Any

from src.config.config import AgentType

if TYPE_CHECKING:
    from dataclasses import dataclass

    @dataclass
    class PopulationStats:
        avg_equity: float
        alive_count: int
        total_count: int


# 种群颜色配置
POPULATION_COLORS: dict[AgentType, tuple[int, int, int]] = {
    AgentType.RETAIL: (100, 200, 100),      # 绿色
    AgentType.RETAIL_PRO: (100, 150, 255),  # 蓝色
    AgentType.WHALE: (255, 200, 100),       # 橙色
    AgentType.MARKET_MAKER: (200, 100, 255) # 紫色
}

# 种群中文名称
POPULATION_NAMES: dict[AgentType, str] = {
    AgentType.RETAIL: "散户",
    AgentType.RETAIL_PRO: "高级散户",
    AgentType.WHALE: "庄家",
    AgentType.MARKET_MAKER: "做市商"
}

# 2x2网格布局的种群顺序：左上、右上、左下、右下
GRID_LAYOUT: list[tuple[AgentType, AgentType]] = [
    (AgentType.RETAIL, AgentType.RETAIL_PRO),      # 第一行：散户, 高级散户
    (AgentType.WHALE, AgentType.MARKET_MAKER),     # 第二行：庄家, 做市商
]


class ChartPanel:
    """图表面板

    显示价格曲线和种群资产曲线（2x2网格布局）。
    """

    def __init__(self, parent: int | str):
        self.parent = parent
        self._setup_ui()

    def _setup_ui(self) -> None:
        """创建UI组件"""
        with dpg.child_window(parent=self.parent, width=-1, height=-1):
            # 价格曲线
            dpg.add_text("价格走势", color=(255, 255, 0))
            with dpg.plot(label="", height=200, width=-1, tag="price_plot"):
                dpg.add_plot_axis(dpg.mvXAxis, label="Tick", tag="price_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="价格", tag="price_y_axis")
                dpg.add_line_series([], [], label="价格",
                    parent="price_y_axis", tag="price_series")

            # 设置价格曲线颜色
            self._setup_price_theme()

            dpg.add_separator()

            # 种群资产曲线标题
            dpg.add_text("种群平均资产", color=(255, 255, 0))

            # 2x2网格布局的资产图表
            for row_agents in GRID_LAYOUT:
                with dpg.group(horizontal=True):
                    for agent_type in row_agents:
                        self._create_equity_plot(agent_type)

            # 设置种群曲线颜色
            self._setup_equity_themes()

            dpg.add_separator()

            # 种群统计信息
            dpg.add_text("种群统计", color=(255, 255, 0))
            for agent_type in AgentType:
                color = POPULATION_COLORS.get(agent_type, (200, 200, 200))
                name = POPULATION_NAMES.get(agent_type, agent_type.value)
                dpg.add_text(f"{name}: 均值: 0  存活: 0",
                    tag=f"stat_{agent_type.value}", color=color)

    def _create_equity_plot(self, agent_type: AgentType) -> None:
        """创建单个种群的资产图表

        Args:
            agent_type: Agent类型
        """
        name = POPULATION_NAMES.get(agent_type, agent_type.value)
        color = POPULATION_COLORS.get(agent_type, (200, 200, 200))
        tag_prefix = agent_type.value

        with dpg.group():
            with dpg.plot(label=name, height=160, width=-1, tag=f"equity_plot_{tag_prefix}"):
                dpg.add_plot_axis(dpg.mvXAxis, label="Tick", tag=f"equity_x_axis_{tag_prefix}")
                dpg.add_plot_axis(dpg.mvYAxis, label="资产", tag=f"equity_y_axis_{tag_prefix}")
                dpg.add_line_series([], [],
                    label=name,
                    parent=f"equity_y_axis_{tag_prefix}",
                    tag=f"equity_series_{tag_prefix}")

    def _setup_price_theme(self) -> None:
        """设置价格曲线主题"""
        with dpg.theme() as price_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 100), category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme("price_series", price_theme)

    def _setup_equity_themes(self) -> None:
        """设置种群曲线颜色主题"""
        for agent_type in AgentType:
            color = POPULATION_COLORS.get(agent_type, (200, 200, 200))
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (*color, 255), category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(f"equity_series_{agent_type.value}", theme)

    def update_price(self, price_history: list[float]) -> None:
        """更新价格曲线

        Args:
            price_history: 价格历史列表
        """
        if not price_history:
            return
        ticks = list(range(len(price_history)))
        dpg.set_value("price_series", [ticks, price_history])

        # 自动调整X轴范围
        dpg.set_axis_limits("price_x_axis", 0, max(len(price_history), 1))

        # 自动调整Y轴范围
        min_p = min(price_history)
        max_p = max(price_history)
        margin = (max_p - min_p) * 0.1 or 1
        dpg.set_axis_limits("price_y_axis", min_p - margin, max_p + margin)

    def update_equity(self, equity_history: dict[AgentType, list[float]],
                     population_stats: dict[AgentType, Any]) -> None:
        """更新资产曲线和统计

        Args:
            equity_history: 各种群资产历史，key为AgentType，value为资产列表
            population_stats: 各种群统计信息，key为AgentType，value为统计对象
        """
        for agent_type in AgentType:
            tag_prefix = agent_type.value
            history = equity_history.get(agent_type, [])

            if history:
                ticks = list(range(len(history)))
                dpg.set_value(f"equity_series_{tag_prefix}", [ticks, history])

                # 自动调整坐标轴
                max_tick = len(history)
                min_equity = min(history)
                max_equity = max(history)

                dpg.set_axis_limits(f"equity_x_axis_{tag_prefix}", 0, max_tick)
                margin = (max_equity - min_equity) * 0.1 or 1
                dpg.set_axis_limits(f"equity_y_axis_{tag_prefix}", min_equity - margin, max_equity + margin)

            # 更新统计文本
            stats = population_stats.get(agent_type)
            name = POPULATION_NAMES.get(agent_type, agent_type.value)
            if stats:
                avg_str = self._format_number(stats.avg_equity)
                dpg.set_value(f"stat_{agent_type.value}",
                    f"{name}: 均值: {avg_str}  存活: {stats.alive_count}/{stats.total_count}")
            else:
                dpg.set_value(f"stat_{agent_type.value}",
                    f"{name}: 均值: 0  存活: 0/0")

    def _format_number(self, num: float) -> str:
        """格式化数字（大数字用K/M表示）

        Args:
            num: 要格式化的数字

        Returns:
            格式化后的字符串
        """
        if abs(num) >= 1e6:
            return f"{num/1e6:.1f}M"
        elif abs(num) >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.0f}"
