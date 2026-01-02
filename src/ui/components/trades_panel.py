"""成交记录面板组件

显示最近的成交记录列表。
"""

import dearpygui.dearpygui as dpg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui.data_collector import TradeInfo


class TradesPanel:
    """成交记录面板"""

    # 面板配置
    PANEL_WIDTH: int = 280
    MAX_DISPLAY_TRADES: int = 30  # 最大显示条数

    def __init__(self) -> None:
        """初始化成交记录面板

        组件会自动添加到当前DearPyGui上下文中。
        """
        self._setup_ui()

    def _setup_ui(self) -> None:
        """创建UI组件"""
        with dpg.child_window(width=self.PANEL_WIDTH, height=-1, no_scrollbar=True):
            dpg.add_text("成交记录", color=(255, 255, 0))
            dpg.add_separator()

            with dpg.table(header_row=True, tag="trades_table",
                          borders_innerH=True, borders_outerH=True,
                          borders_innerV=True, borders_outerV=True,
                          scrollY=False):
                dpg.add_table_column(label="Tick", width_fixed=True, init_width_or_weight=50)
                dpg.add_table_column(label="价格", width_fixed=True, init_width_or_weight=70)
                dpg.add_table_column(label="数量", width_fixed=True, init_width_or_weight=70)
                dpg.add_table_column(label="方向", width_fixed=True, init_width_or_weight=45)

    def update(self, trades: "list[TradeInfo]") -> None:
        """更新成交记录

        Args:
            trades: 成交记录列表
        """
        # 清空表格
        children = dpg.get_item_children("trades_table", slot=1)
        if children:
            for child in children:
                dpg.delete_item(child)

        # 添加最新成交（倒序显示，最新在上）
        for trade in reversed(trades[-self.MAX_DISPLAY_TRADES:]):  # 显示最近N笔
            with dpg.table_row(parent="trades_table"):
                dpg.add_text(str(trade.tick))
                dpg.add_text(f"{trade.price:.2f}")
                dpg.add_text(str(int(trade.quantity)))

                # 买入绿色，卖出红色
                if trade.is_buyer_taker:
                    dpg.add_text("买入", color=(100, 255, 100))
                else:
                    dpg.add_text("卖出", color=(255, 100, 100))
