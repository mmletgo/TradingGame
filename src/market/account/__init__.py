"""账户模块

本模块包含账户管理相关的类：
- Position: 持仓类（Cython 加速）
- Account: 账户类（Python 实现）
- FastAccount: 快速账户类（Cython 加速）
"""

from src.market.account.position import Position
from src.market.account.account import Account

# 尝试导入 Cython 加速的 FastAccount
try:
    from src.market.account.fast_account import FastAccount
except ImportError:
    FastAccount = None  # type: ignore

__all__ = ["Position", "Account", "FastAccount"]
