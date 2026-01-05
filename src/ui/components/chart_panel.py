"""图表面板组件

显示价格曲线、种群资产曲线、资产分布小提琴图和鲶鱼资金曲线。
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import TYPE_CHECKING, Any

from src.config.config import AgentType

if TYPE_CHECKING:
    from dataclasses import dataclass

    @dataclass
    class PopulationStats:
        avg_equity: float
        alive_count: int
        total_count: int
        alive_equities: list[float]

    @dataclass
    class CatfishInfo:
        name: str
        equity: float
        position_qty: int
        position_value: float
        initial_balance: float
        is_liquidated: bool


# 种群颜色配置
POPULATION_COLORS: dict[AgentType, tuple[int, int, int]] = {
    AgentType.RETAIL: (100, 200, 100),       # 绿色
    AgentType.RETAIL_PRO: (100, 150, 255),   # 蓝色
    AgentType.WHALE: (255, 100, 150),        # 粉红色（庄家）
    AgentType.MARKET_MAKER: (200, 100, 255), # 紫色
}

# 鲶鱼颜色配置
CATFISH_COLORS: dict[str, tuple[int, int, int]] = {
    "TrendFollowingCatfish": (255, 165, 0),      # 橙色
    "CycleSwingCatfish": (0, 191, 255),          # 天蓝色
    "MeanReversionCatfish": (255, 105, 180),     # 粉色
    "RandomTradingCatfish": (148, 0, 211),       # 深紫色
}

# 鲶鱼中文名称
CATFISH_NAMES: dict[str, str] = {
    "TrendFollowingCatfish": "趋势追踪",
    "CycleSwingCatfish": "周期摆动",
    "MeanReversionCatfish": "逆势操作",
    "RandomTradingCatfish": "随机买卖",
}

# 种群中文名称
POPULATION_NAMES: dict[AgentType, str] = {
    AgentType.RETAIL: "散户",
    AgentType.RETAIL_PRO: "高级散户",
    AgentType.WHALE: "庄家",
    AgentType.MARKET_MAKER: "做市商",
}

# 纵向布局的种群顺序
VERTICAL_LAYOUT: list[AgentType] = [
    AgentType.RETAIL,
    AgentType.RETAIL_PRO,
    AgentType.WHALE,
    AgentType.MARKET_MAKER,
]


class ChartPanel:
    """图表面板

    显示价格曲线、种群资产曲线、资产分布小提琴图和鲶鱼资金曲线。
    """

    # 图表面板配置
    PANEL_WIDTH: int = 1360  # 总宽度
    EQUITY_PLOT_HEIGHT: int = 140  # 每个资产图表高度
    PRICE_PLOT_HEIGHT: int = 140  # 价格图表高度
    VIOLIN_PLOT_HEIGHT: int = 120  # 小提琴图高度
    VIOLIN_PLOT_WIDTH: int = 340  # 每个小提琴图宽度（4个并排）
    CATFISH_PLOT_HEIGHT: int = 140  # 鲶鱼图表高度
    CATFISH_PLOT_WIDTH: int = 340  # 每个鲶鱼图表宽度（4个并排）
    KDE_POINTS: int = 50  # KDE曲线采样点数

    # 是否显示鲶鱼图表
    _catfish_enabled: bool

    def __init__(self) -> None:
        """初始化图表面板

        组件会自动添加到当前DearPyGui上下文中。
        """
        self._catfish_enabled = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        """创建UI组件"""
        with dpg.child_window(width=self.PANEL_WIDTH, height=-1, no_scrollbar=True):
            # 价格曲线
            dpg.add_text("价格走势", color=(255, 255, 0))
            with dpg.plot(label="", height=self.PRICE_PLOT_HEIGHT, width=-1, tag="price_plot"):
                dpg.add_plot_axis(dpg.mvXAxis, label="Tick", tag="price_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="价格", tag="price_y_axis")
                dpg.add_line_series([], [], label="价格",
                    parent="price_y_axis", tag="price_series")

            # 设置价格曲线颜色
            self._setup_price_theme()

            dpg.add_separator()

            # 种群资产曲线标题
            dpg.add_text("种群存活个体平均资产曲线", color=(255, 255, 0))

            # 纵向4行布局，每行2张图（所有个体 + 存活个体）
            for agent_type in VERTICAL_LAYOUT:
                self._create_equity_row(agent_type)

            # 设置种群曲线颜色
            self._setup_equity_themes()

            dpg.add_separator()

            # 种群统计信息（水平排列节省空间）
            dpg.add_text("种群统计", color=(255, 255, 0))
            with dpg.group(horizontal=True):
                for agent_type in AgentType:
                    color = POPULATION_COLORS.get(agent_type, (200, 200, 200))
                    name = POPULATION_NAMES.get(agent_type, agent_type.value)
                    dpg.add_text(f"{name}: 总计: 0  存活: 0",
                        tag=f"stat_{agent_type.value}", color=color)
                    dpg.add_spacer(width=20)

            # 小提琴图区域（4个并排）
            self._create_violin_plots()

            # 鲶鱼图表区域（3个并排，默认隐藏）
            self._create_catfish_plots()

    def _create_equity_row(self, agent_type: AgentType) -> None:
        """创建单个种群的资产图表行（存活个体平均）

        Args:
            agent_type: Agent类型
        """
        name = POPULATION_NAMES.get(agent_type, agent_type.value)
        tag_prefix = agent_type.value

        # 存活个体平均资产
        with dpg.plot(label=f"{name}", height=self.EQUITY_PLOT_HEIGHT,
                     width=-1, tag=f"equity_plot_{tag_prefix}"):
            dpg.add_plot_axis(dpg.mvXAxis, label="Tick", tag=f"equity_x_axis_{tag_prefix}")
            dpg.add_plot_axis(dpg.mvYAxis, label="平均资产", tag=f"equity_y_axis_{tag_prefix}")
            dpg.add_line_series([], [],
                label=name,
                parent=f"equity_y_axis_{tag_prefix}",
                tag=f"equity_series_{tag_prefix}")

    def _setup_price_theme(self) -> None:
        """设置价格曲线主题"""
        with dpg.theme() as price_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 100),
                                   category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme("price_series", price_theme)

    def _setup_equity_themes(self) -> None:
        """设置种群曲线颜色主题"""
        for agent_type in AgentType:
            color = POPULATION_COLORS.get(agent_type, (200, 200, 200))
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (*color, 255),
                                       category=dpg.mvThemeCat_Plots)
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
                     alive_equity_history: dict[AgentType, list[float]],
                     population_stats: dict[AgentType, Any]) -> None:
        """更新资产曲线和统计

        Args:
            equity_history: 各种群所有个体平均资产历史，key为AgentType，value为平均资产列表
            alive_equity_history: 各种群存活个体平均资产历史，key为AgentType，value为平均资产列表
            population_stats: 各种群统计信息，key为AgentType，value为统计对象
        """
        for agent_type in AgentType:
            tag_prefix = agent_type.value
            alive_history = alive_equity_history.get(agent_type, [])

            # 更新存活个体资产曲线
            if alive_history:
                ticks = list(range(len(alive_history)))
                dpg.set_value(f"equity_series_{tag_prefix}", [ticks, alive_history])

                # 自动调整坐标轴
                max_tick = len(alive_history)
                min_equity = min(alive_history)
                max_equity = max(alive_history)

                dpg.set_axis_limits(f"equity_x_axis_{tag_prefix}", 0, max_tick)
                margin = (max_equity - min_equity) * 0.1 or 1
                dpg.set_axis_limits(f"equity_y_axis_{tag_prefix}",
                                   min_equity - margin, max_equity + margin)

            # 更新统计文本和小提琴图
            stats = population_stats.get(agent_type)
            name = POPULATION_NAMES.get(agent_type, agent_type.value)
            if stats:
                total_str = self._format_number(stats.total_equity)
                dpg.set_value(f"stat_{agent_type.value}",
                    f"{name}: 总计: {total_str}  存活: {stats.alive_count}/{stats.total_count}")
                # 更新小提琴图
                self._update_violin_plot(agent_type, stats.alive_equities)
            else:
                dpg.set_value(f"stat_{agent_type.value}",
                    f"{name}: 总计: 0  存活: 0/0")
                # 清空小提琴图
                self._update_violin_plot(agent_type, [])

    def _format_number(self, num: float) -> str:
        """格式化数字（大数字用K/M/B/亿表示）

        Args:
            num: 要格式化的数字

        Returns:
            格式化后的字符串
        """
        if abs(num) >= 1e8:
            return f"{num/1e8:.2f}亿"
        elif abs(num) >= 1e6:
            return f"{num/1e6:.1f}M"
        elif abs(num) >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.0f}"

    def _create_violin_plots(self) -> None:
        """创建4个并排的小提琴图"""
        with dpg.group(horizontal=True):
            for agent_type in VERTICAL_LAYOUT:
                name = POPULATION_NAMES.get(agent_type, agent_type.value)
                tag_prefix = agent_type.value

                with dpg.plot(
                    label=name,
                    height=self.VIOLIN_PLOT_HEIGHT,
                    width=self.VIOLIN_PLOT_WIDTH,
                    tag=f"violin_plot_{tag_prefix}",
                    no_mouse_pos=True,
                ):
                    dpg.add_plot_axis(
                        dpg.mvXAxis, label="资产", tag=f"violin_x_axis_{tag_prefix}"
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis, label="", tag=f"violin_y_axis_{tag_prefix}"
                    )

                    # 小提琴形状（Area Series）
                    dpg.add_area_series(
                        [],
                        [],
                        parent=f"violin_y_axis_{tag_prefix}",
                        tag=f"violin_area_{tag_prefix}",
                    )

                    # 中位数线
                    dpg.add_line_series(
                        [],
                        [],
                        parent=f"violin_y_axis_{tag_prefix}",
                        tag=f"violin_median_{tag_prefix}",
                    )

                    # 四分位线（Q1和Q3）
                    dpg.add_line_series(
                        [],
                        [],
                        parent=f"violin_y_axis_{tag_prefix}",
                        tag=f"violin_q1_{tag_prefix}",
                    )
                    dpg.add_line_series(
                        [],
                        [],
                        parent=f"violin_y_axis_{tag_prefix}",
                        tag=f"violin_q3_{tag_prefix}",
                    )

        # 设置小提琴图主题
        self._setup_violin_themes()

    def _setup_violin_themes(self) -> None:
        """设置小提琴图颜色主题"""
        for agent_type in AgentType:
            color = POPULATION_COLORS.get(agent_type, (200, 200, 200))
            tag_prefix = agent_type.value

            # 小提琴形状主题（半透明填充）
            with dpg.theme() as area_theme:
                with dpg.theme_component(dpg.mvAreaSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Fill,
                        (*color, 100),
                        category=dpg.mvThemeCat_Plots,
                    )
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Line,
                        (*color, 200),
                        category=dpg.mvThemeCat_Plots,
                    )
            dpg.bind_item_theme(f"violin_area_{tag_prefix}", area_theme)

            # 中位数线主题（白色粗线）
            with dpg.theme() as median_theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Line,
                        (255, 255, 255, 255),
                        category=dpg.mvThemeCat_Plots,
                    )
            dpg.bind_item_theme(f"violin_median_{tag_prefix}", median_theme)

            # 四分位线主题（灰色细线）
            with dpg.theme() as quartile_theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Line,
                        (180, 180, 180, 200),
                        category=dpg.mvThemeCat_Plots,
                    )
            dpg.bind_item_theme(f"violin_q1_{tag_prefix}", quartile_theme)
            dpg.bind_item_theme(f"violin_q3_{tag_prefix}", quartile_theme)

    def _gaussian_kde(
        self, data: np.ndarray, x_grid: np.ndarray, bandwidth: float | None = None
    ) -> np.ndarray:
        """高斯核密度估计（纯NumPy实现）

        使用高斯核函数计算概率密度估计，不依赖scipy。

        Args:
            data: 原始数据数组
            x_grid: 评估点网格
            bandwidth: 带宽，None则使用Silverman法则自动计算

        Returns:
            密度估计值数组
        """
        n = len(data)
        if n == 0:
            return np.zeros_like(x_grid)

        # Silverman法则计算带宽
        if bandwidth is None:
            std = np.std(data, ddof=1) if n > 1 else 1.0
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            # 避免带宽过小
            scale = min(std, iqr / 1.34) if iqr > 0 else std
            bandwidth = 0.9 * scale * (n ** (-0.2)) if scale > 0 else 1.0

        # 确保带宽不为0
        bandwidth = max(bandwidth, 1e-6)

        # 计算核密度估计
        # K(u) = exp(-u^2/2) / sqrt(2*pi)
        diff = x_grid[:, np.newaxis] - data[np.newaxis, :]
        kernel_vals = np.exp(-0.5 * (diff / bandwidth) ** 2)
        density = np.sum(kernel_vals, axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))

        return density

    def _update_violin_plot(
        self, agent_type: AgentType, equities: list[float]
    ) -> None:
        """更新单个种群的小提琴图

        计算KDE密度曲线并绘制对称的小提琴形状，同时添加中位数和四分位线。

        Args:
            agent_type: Agent类型
            equities: 存活个体的资产列表
        """
        tag_prefix = agent_type.value

        # 数据不足时清空图表
        if len(equities) < 2:
            dpg.set_value(f"violin_area_{tag_prefix}", [[], []])
            dpg.set_value(f"violin_median_{tag_prefix}", [[], []])
            dpg.set_value(f"violin_q1_{tag_prefix}", [[], []])
            dpg.set_value(f"violin_q3_{tag_prefix}", [[], []])
            return

        data = np.array(equities)

        # 计算统计量
        median = float(np.median(data))
        q1 = float(np.percentile(data, 25))
        q3 = float(np.percentile(data, 75))
        data_min = float(np.min(data))
        data_max = float(np.max(data))

        # 扩展数据范围
        data_range = data_max - data_min
        margin = data_range * 0.1 if data_range > 0 else 1.0
        x_min = data_min - margin
        x_max = data_max + margin

        # 创建评估网格
        x_grid = np.linspace(x_min, x_max, self.KDE_POINTS)

        # 计算KDE
        density = self._gaussian_kde(data, x_grid)

        # 归一化密度到 [-0.5, 0.5]
        max_density = np.max(density)
        if max_density > 0:
            normalized_density = density / max_density * 0.4
        else:
            normalized_density = density

        # 创建小提琴形状（上半部分 + 下半部分，形成闭合路径）
        upper_y = normalized_density.tolist()
        lower_y = (-normalized_density[::-1]).tolist()

        upper_x = x_grid.tolist()
        lower_x = x_grid[::-1].tolist()

        violin_x = upper_x + lower_x
        violin_y = upper_y + lower_y

        # 更新小提琴形状
        dpg.set_value(f"violin_area_{tag_prefix}", [violin_x, violin_y])

        # 更新中位数线（水平线在y=0）
        dpg.set_value(f"violin_median_{tag_prefix}", [[median, median], [-0.3, 0.3]])

        # 更新四分位线
        dpg.set_value(f"violin_q1_{tag_prefix}", [[q1, q1], [-0.2, 0.2]])
        dpg.set_value(f"violin_q3_{tag_prefix}", [[q3, q3], [-0.2, 0.2]])

        # 调整坐标轴范围
        dpg.set_axis_limits(f"violin_x_axis_{tag_prefix}", x_min, x_max)
        dpg.set_axis_limits(f"violin_y_axis_{tag_prefix}", -0.6, 0.6)

    def _create_catfish_plots(self) -> None:
        """创建四只鲶鱼的图表区域（一行四个，默认隐藏）"""
        # 鲶鱼区域容器（默认隐藏）
        with dpg.group(tag="catfish_container", show=False):
            dpg.add_separator()
            dpg.add_text("鲶鱼资金曲线", color=(255, 255, 0))

            # 四只鲶鱼图表水平排列
            with dpg.group(horizontal=True):
                catfish_types = [
                    "TrendFollowingCatfish",
                    "CycleSwingCatfish",
                    "MeanReversionCatfish",
                    "RandomTradingCatfish",
                ]

                for i, catfish_type in enumerate(catfish_types):
                    name = CATFISH_NAMES.get(catfish_type, catfish_type)
                    color = CATFISH_COLORS.get(catfish_type, (200, 200, 200))

                    with dpg.group():
                        # 图表标题和持仓状态
                        with dpg.group(horizontal=True):
                            dpg.add_text(
                                f"{name}",
                                tag=f"catfish_title_{i}",
                                color=color,
                            )
                            dpg.add_text(
                                " | 持仓: 0",
                                tag=f"catfish_position_{i}",
                                color=(200, 200, 200),
                            )

                        # 净值曲线图
                        with dpg.plot(
                            label="",
                            height=self.CATFISH_PLOT_HEIGHT,
                            width=self.CATFISH_PLOT_WIDTH,
                            tag=f"catfish_plot_{i}",
                        ):
                            dpg.add_plot_axis(
                                dpg.mvXAxis,
                                label="Tick",
                                tag=f"catfish_x_axis_{i}",
                            )
                            dpg.add_plot_axis(
                                dpg.mvYAxis,
                                label="净值",
                                tag=f"catfish_y_axis_{i}",
                            )
                            dpg.add_line_series(
                                [],
                                [],
                                label=name,
                                parent=f"catfish_y_axis_{i}",
                                tag=f"catfish_series_{i}",
                            )

        # 设置鲶鱼曲线颜色主题
        self._setup_catfish_themes()

    def _setup_catfish_themes(self) -> None:
        """设置鲶鱼曲线颜色主题"""
        catfish_types = [
            "TrendFollowingCatfish",
            "CycleSwingCatfish",
            "MeanReversionCatfish",
            "RandomTradingCatfish",
        ]

        for i, catfish_type in enumerate(catfish_types):
            color = CATFISH_COLORS.get(catfish_type, (200, 200, 200))

            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Line,
                        (*color, 255),
                        category=dpg.mvThemeCat_Plots,
                    )
            dpg.bind_item_theme(f"catfish_series_{i}", theme)

    def update_catfish(
        self,
        catfish_data: list[Any],
        catfish_equity_history: list[list[float]],
    ) -> None:
        """更新鲶鱼图表

        Args:
            catfish_data: 鲶鱼信息列表（CatfishInfo对象列表）
            catfish_equity_history: 四只鲶鱼的净值历史
        """
        # 如果没有鲶鱼数据，隐藏鲶鱼区域
        if not catfish_data:
            if self._catfish_enabled:
                dpg.configure_item("catfish_container", show=False)
                self._catfish_enabled = False
            return

        # 如果鲶鱼区域被隐藏，显示它
        if not self._catfish_enabled:
            dpg.configure_item("catfish_container", show=True)
            self._catfish_enabled = True

        # 更新每只鲶鱼的图表
        for i, catfish_info in enumerate(catfish_data):
            if i >= 4:  # 最多显示4只鲶鱼
                break

            # 获取鲶鱼类型和颜色
            catfish_type = catfish_info.name
            name = CATFISH_NAMES.get(catfish_type, catfish_type)
            color = CATFISH_COLORS.get(catfish_type, (200, 200, 200))

            # 更新持仓显示
            position_qty = catfish_info.position_qty
            position_value = catfish_info.position_value
            value_str = self._format_number(position_value)
            if position_qty > 0:
                position_text = f" | 持仓: +{position_qty} (多) | 市值: {value_str}"
                position_color = (100, 200, 100)  # 绿色
            elif position_qty < 0:
                position_text = f" | 持仓: {position_qty} (空) | 市值: {value_str}"
                position_color = (255, 100, 100)  # 红色
            else:
                position_text = " | 持仓: 0 (空仓)"
                position_color = (200, 200, 200)  # 灰色

            # 如果被强平，显示强平状态
            if catfish_info.is_liquidated:
                position_text = " | [已强平]"
                position_color = (255, 50, 50)

            dpg.set_value(f"catfish_position_{i}", position_text)
            dpg.configure_item(f"catfish_position_{i}", color=position_color)

            # 更新净值曲线
            if i < len(catfish_equity_history):
                equity_history = catfish_equity_history[i]
                if equity_history:
                    ticks = list(range(len(equity_history)))
                    dpg.set_value(f"catfish_series_{i}", [ticks, equity_history])

                    # 自动调整坐标轴
                    max_tick = len(equity_history)
                    min_equity = min(equity_history)
                    max_equity = max(equity_history)

                    dpg.set_axis_limits(f"catfish_x_axis_{i}", 0, max_tick)
                    margin = (max_equity - min_equity) * 0.1 or 1
                    dpg.set_axis_limits(
                        f"catfish_y_axis_{i}",
                        min_equity - margin,
                        max_equity + margin,
                    )
