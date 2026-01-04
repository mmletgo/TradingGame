"""
鲶鱼模块

提供鲶鱼（Catfish）的各种实现。鲶鱼是一种特殊的市场参与者，
用于在训练中引入外部扰动，增加市场动态性。

主要类：
- CatfishBase: 鲶鱼抽象基类
- TrendFollowingCatfish: 趋势追踪型鲶鱼
- CycleSwingCatfish: 周期摆动型鲶鱼
- MeanReversionCatfish: 逆势操作型鲶鱼

工厂函数：
- create_catfish: 根据配置创建鲶鱼实例
"""

from src.config.config import CatfishConfig, CatfishMode
from src.market.catfish.catfish_base import CatfishBase
from src.market.catfish.trend_following import TrendFollowingCatfish
from src.market.catfish.cycle_swing import CycleSwingCatfish
from src.market.catfish.mean_reversion import MeanReversionCatfish


def create_catfish(
    catfish_id: int,
    config: CatfishConfig,
) -> CatfishBase:
    """
    根据配置创建鲶鱼实例

    工厂函数，根据 config.mode 创建对应类型的鲶鱼。

    Args:
        catfish_id: 鲶鱼ID（应为负数）
        config: 鲶鱼配置

    Returns:
        对应类型的鲶鱼实例

    Raises:
        ValueError: 如果 mode 不是有效的 CatfishMode
    """
    if config.mode == CatfishMode.TREND_FOLLOWING:
        return TrendFollowingCatfish(catfish_id, config)
    elif config.mode == CatfishMode.CYCLE_SWING:
        return CycleSwingCatfish(catfish_id, config)
    elif config.mode == CatfishMode.MEAN_REVERSION:
        return MeanReversionCatfish(catfish_id, config)
    else:
        raise ValueError(f"未知的鲶鱼模式: {config.mode}")


__all__ = [
    "CatfishBase",
    "TrendFollowingCatfish",
    "CycleSwingCatfish",
    "MeanReversionCatfish",
    "create_catfish",
]
