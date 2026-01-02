"""订单簿面板组件

显示订单簿深度图和合并的买卖盘价格列表。
"""

import dearpygui.dearpygui as dpg


class OrderBookPanel:
    """订单簿面板

    显示深度图和合并的买卖盘价格列表（价格从高到低）。
    """

    # 面板配置
    PANEL_WIDTH: int = 280
    DEPTH_CHART_HEIGHT: int = 180
    DISPLAY_LEVELS: int = 10  # 每边显示档数

    def __init__(self) -> None:
        """初始化订单簿面板

        组件会自动添加到当前DearPyGui上下文中。
        """
        self._setup_ui()

    def _setup_ui(self) -> None:
        """创建UI组件"""
        with dpg.child_window(width=self.PANEL_WIDTH, height=-1, no_scrollbar=True):
            dpg.add_text("订单簿", color=(255, 255, 0))
            dpg.add_separator()

            # 深度图
            with dpg.plot(label="深度图", height=self.DEPTH_CHART_HEIGHT, width=-1,
                         tag=self._make_tag("depth_plot")):
                dpg.add_plot_axis(dpg.mvXAxis, label="价格", tag=self._make_tag("depth_x"))
                dpg.add_plot_axis(dpg.mvYAxis, label="累计量", tag=self._make_tag("depth_y"))

                # 买盘（绿色）
                dpg.add_area_series([], [],
                    label="买盘", parent=self._make_tag("depth_y"),
                    tag=self._make_tag("bid_shade"))

                # 卖盘（红色）
                dpg.add_area_series([], [],
                    label="卖盘", parent=self._make_tag("depth_y"),
                    tag=self._make_tag("ask_shade"))

            # 设置深度图颜色主题
            self._setup_depth_theme()

            dpg.add_separator()

            # 合并的买卖盘价格列表（卖盘在上，买盘在下，价格都从高到低）
            dpg.add_text("盘口（价格从高到低）", color=(255, 255, 0))
            with dpg.table(header_row=True, tag=self._make_tag("orderbook_table"),
                          borders_innerH=True, borders_outerH=True,
                          borders_innerV=True, borders_outerV=True,
                          scrollY=False):
                dpg.add_table_column(label="价格", width_fixed=True, init_width_or_weight=90)
                dpg.add_table_column(label="数量", width_fixed=True, init_width_or_weight=90)
                dpg.add_table_column(label="类型", width_fixed=True, init_width_or_weight=50)

    def _setup_depth_theme(self) -> None:
        """设置深度图颜色主题"""
        # 买盘绿色主题
        with dpg.theme() as bid_theme:
            with dpg.theme_component(dpg.mvAreaSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, (100, 255, 100, 100),
                                   category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, (100, 255, 100, 255),
                                   category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(self._make_tag("bid_shade"), bid_theme)

        # 卖盘红色主题
        with dpg.theme() as ask_theme:
            with dpg.theme_component(dpg.mvAreaSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, (255, 100, 100, 100),
                                   category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 100, 100, 255),
                                   category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(self._make_tag("ask_shade"), ask_theme)

    def update(self, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> None:
        """更新订单簿数据

        Args:
            bids: 买盘数据列表，每项为 (价格, 数量)，按价格从高到低排序
            asks: 卖盘数据列表，每项为 (价格, 数量)，按价格从低到高排序
        """
        # 更新深度图
        self._update_depth_chart(bids, asks)

        # 更新合并的价格列表
        self._update_orderbook_table(bids, asks)

    def _update_depth_chart(self, bids: list[tuple[float, float]],
                           asks: list[tuple[float, float]]) -> None:
        """更新深度图"""
        # 计算买盘累计量（按价格从高到低排序）
        bid_prices: list[float] = [p for p, _ in bids[:20]]
        bid_cumsum: list[float] = []
        cumsum: float = 0.0
        for _, qty in bids[:20]:
            cumsum += qty
            bid_cumsum.append(cumsum)

        # 计算卖盘累计量（按价格从低到高排序）
        ask_prices: list[float] = [p for p, _ in asks[:20]]
        ask_cumsum: list[float] = []
        cumsum = 0.0
        for _, qty in asks[:20]:
            cumsum += qty
            ask_cumsum.append(cumsum)

        # 更新area series
        if bid_prices:
            dpg.set_value(self._make_tag("bid_shade"), [bid_prices, bid_cumsum])
        if ask_prices:
            dpg.set_value(self._make_tag("ask_shade"), [ask_prices, ask_cumsum])

        # 自动调整坐标轴
        if bid_prices or ask_prices:
            all_prices = bid_prices + ask_prices
            if all_prices:
                min_price = min(all_prices)
                max_price = max(all_prices)
                dpg.set_axis_limits(self._make_tag("depth_x"),
                    min_price * 0.999, max_price * 1.001)

                all_volumes = bid_cumsum + ask_cumsum
                if all_volumes:
                    max_volume = max(all_volumes)
                    dpg.set_axis_limits(self._make_tag("depth_y"), 0, max_volume * 1.1)

    def _update_orderbook_table(self, bids: list[tuple[float, float]],
                                asks: list[tuple[float, float]]) -> None:
        """更新合并的买卖盘价格列表

        显示顺序：卖盘（价格从高到低）在上，买盘（价格从高到低）在下
        """
        table_tag = self._make_tag("orderbook_table")

        # 清空表格
        children = dpg.get_item_children(table_tag, slot=1)
        if children:
            for child in children:
                dpg.delete_item(child)

        # 获取前10档
        ask_levels = asks[:self.DISPLAY_LEVELS]
        bid_levels = bids[:self.DISPLAY_LEVELS]

        # 卖盘：按价格从高到低显示（原始数据是从低到高，需要反转）
        for price, qty in reversed(ask_levels):
            with dpg.table_row(parent=table_tag):
                dpg.add_text(f"{price:.2f}", color=(255, 100, 100))
                dpg.add_text(f"{qty:.0f}")
                dpg.add_text("卖", color=(255, 100, 100))

        # 添加分隔行（中间价位置）
        if ask_levels or bid_levels:
            with dpg.table_row(parent=table_tag):
                dpg.add_text("───────", color=(128, 128, 128))
                dpg.add_text("───────", color=(128, 128, 128))
                dpg.add_text("──", color=(128, 128, 128))

        # 买盘：按价格从高到低显示（原始数据已经是从高到低）
        for price, qty in bid_levels:
            with dpg.table_row(parent=table_tag):
                dpg.add_text(f"{price:.2f}", color=(100, 255, 100))
                dpg.add_text(f"{qty:.0f}")
                dpg.add_text("买", color=(100, 255, 100))

    def _make_tag(self, suffix: str) -> str:
        """生成唯一的tag名称"""
        return f"orderbook_{suffix}"
