"""订单簿面板组件

显示订单簿深度图和价格列表。
"""

import dearpygui.dearpygui as dpg
from typing import Callable


class OrderBookPanel:
    """订单簿面板

    显示深度图和价格列表。
    """

    def __init__(self, parent: int | str):
        self.parent = parent
        self._setup_ui()

    def _setup_ui(self) -> None:
        """创建UI组件"""
        with dpg.child_window(parent=self.parent, width=350, height=-1):
            dpg.add_text("订单簿", color=(255, 255, 0))
            dpg.add_separator()

            # 深度图（使用 shade series）
            with dpg.plot(label="深度图", height=200, width=-1, tag=self._make_tag("depth_plot")):
                dpg.add_plot_axis(dpg.mvXAxis, label="价格", tag=self._make_tag("depth_x"))
                dpg.add_plot_axis(dpg.mvYAxis, label="累计量", tag=self._make_tag("depth_y"))

                # 买盘（绿色）- 使用 area_series 替代 shade_series
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

            # 价格列表（表格）
            dpg.add_text("卖盘", color=(255, 100, 100))
            with dpg.table(header_row=True, tag=self._make_tag("ask_table"),
                          borders_innerH=True, borders_outerH=True,
                          borders_innerV=True, borders_outerV=True,
                          scrollY=True, height=150):
                dpg.add_table_column(label="价格")
                dpg.add_table_column(label="数量")

            dpg.add_separator()

            dpg.add_text("买盘", color=(100, 255, 100))
            with dpg.table(header_row=True, tag=self._make_tag("bid_table"),
                          borders_innerH=True, borders_outerH=True,
                          borders_innerV=True, borders_outerV=True,
                          scrollY=True, height=150):
                dpg.add_table_column(label="价格")
                dpg.add_table_column(label="数量")

    def _setup_depth_theme(self) -> None:
        """设置深度图颜色主题"""
        # 买盘绿色主题
        with dpg.theme() as bid_theme:
            with dpg.theme_component(dpg.mvAreaSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, (100, 255, 100, 100), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, (100, 255, 100, 255), category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(self._make_tag("bid_shade"), bid_theme)

        # 卖盘红色主题
        with dpg.theme() as ask_theme:
            with dpg.theme_component(dpg.mvAreaSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, (255, 100, 100, 100), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 100, 100, 255), category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(self._make_tag("ask_shade"), ask_theme)

    def update(self, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> None:
        """更新订单簿数据

        Args:
            bids: 买盘数据列表，每项为 (价格, 数量)
            asks: 卖盘数据列表，每项为 (价格, 数量)
        """
        # 更新深度图
        self._update_depth_chart(bids, asks)

        # 更新价格列表
        self._update_price_table(bids, asks)

    def _update_depth_chart(self, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> None:
        """更新深度图"""
        # 计算买盘累计量（按价格从高到低排序）
        bid_prices: list[float] = [p for p, _ in bids[:20]]  # 显示前20档
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

    def _update_price_table(self, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> None:
        """更新价格列表"""
        # 清空并重建卖盘表格
        ask_table = self._make_tag("ask_table")
        children = dpg.get_item_children(ask_table, slot=1)
        if children:
            for child in children:
                dpg.delete_item(child)

        for price, qty in asks[:10]:  # 显示前10档
            with dpg.table_row(parent=ask_table):
                dpg.add_text(f"{price:.2f}", color=(255, 100, 100))
                dpg.add_text(f"{qty:.0f}")

        # 清空并重建买盘表格
        bid_table = self._make_tag("bid_table")
        children = dpg.get_item_children(bid_table, slot=1)
        if children:
            for child in children:
                dpg.delete_item(child)

        for price, qty in bids[:10]:
            with dpg.table_row(parent=bid_table):
                dpg.add_text(f"{price:.2f}", color=(100, 255, 100))
                dpg.add_text(f"{qty:.0f}")

    def _make_tag(self, suffix: str) -> str:
        """生成唯一的tag名称"""
        return f"orderbook_{suffix}"
