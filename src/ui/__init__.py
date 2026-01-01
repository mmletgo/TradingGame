"""UI模块

提供训练可视化所需的数据采集和展示组件。
"""

from src.ui.data_collector import (
    PopulationStats,
    TradeInfo,
    UIDataCollector,
    UIDataSnapshot,
)
from src.ui.ui_controller import UIController
from src.ui.training_app import TrainingUIApp
from src.ui.demo_app import DemoUIApp

__all__ = [
    "UIController",
    "UIDataCollector",
    "UIDataSnapshot",
    "TradeInfo",
    "PopulationStats",
    "TrainingUIApp",
    "DemoUIApp",
]
