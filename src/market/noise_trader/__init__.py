"""噪声交易者模块"""

from src.market.noise_trader.noise_trader import NoiseTrader
from src.market.noise_trader.noise_trader_account import NoiseTraderAccount

__all__ = ["NoiseTrader", "NoiseTraderAccount", "create_noise_traders"]


def create_noise_traders(config: "NoiseTraderConfig") -> list[NoiseTrader]:
    """创建噪声交易者列表

    Args:
        config: 噪声交易者配置

    Returns:
        噪声交易者列表，ID 从 -1 到 -count
    """
    from src.config.config import NoiseTraderConfig
    return [
        NoiseTrader(trader_id=-(i + 1), config=config)
        for i in range(config.count)
    ]
