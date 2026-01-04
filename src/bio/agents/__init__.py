"""AI Agent 模块

定义五种类型的 AI Agent：散户（RetailAgent）、高级散户（RetailProAgent）、
多头庄家（BullWhaleAgent）、空头庄家（BearWhaleAgent）、做市商（MarketMakerAgent）。
"""

from src.bio.agents.base import Agent
from src.bio.agents.bear_whale import BearWhaleAgent
from src.bio.agents.bull_whale import BullWhaleAgent
from src.bio.agents.retail_pro import RetailProAgent
from src.bio.agents.whale import WhaleBaseAgent

__all__ = [
    "Agent",
    "RetailProAgent",
    "WhaleBaseAgent",
    "BullWhaleAgent",
    "BearWhaleAgent",
]
