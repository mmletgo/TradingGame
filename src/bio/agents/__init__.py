"""AI Agent 模块

定义四种类型的 AI Agent：散户（RetailAgent）、高级散户（RetailProAgent）、
庄家（WhaleAgent）、做市商（MarketMakerAgent）。
"""

from src.bio.agents.base import Agent
from src.bio.agents.retail_pro import RetailProAgent
from src.bio.agents.whale import WhaleAgent

__all__ = [
    "Agent",
    "RetailProAgent",
    "WhaleAgent",
]
