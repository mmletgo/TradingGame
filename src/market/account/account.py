"""账户模块

本模块定义账户(Account)类，用于记录 Agent 的交易账户状态。
"""

from src.config.config import AgentConfig, AgentType
from src.market.account.position import Position


class Account:
    """账户类

    记录 Agent 的交易账户状态，包括余额、持仓、杠杆倍数和手续费率等。

    Attributes:
        agent_id: Agent ID
        agent_type: Agent 类型
        balance: 余额
        position: 持仓对象
        leverage: 杠杆倍数
        maintenance_margin_rate: 维持保证金率
        maker_fee_rate: 挂单费率
        taker_fee_rate: 吃单费率
        pending_order_id: 当前挂单ID
    """

    def __init__(self, agent_id: int, agent_type: AgentType, config: AgentConfig) -> None:
        """创建账户

        初始化余额、持仓、杠杆、费率等账户属性。

        Args:
            agent_id: Agent ID
            agent_type: Agent 类型
            config: Agent 配置对象
        """
        self.agent_id: int = agent_id
        self.agent_type: AgentType = agent_type
        self.balance: float = config.initial_balance
        self.position: Position = Position()
        self.leverage: float = config.leverage
        self.maintenance_margin_rate: float = config.maintenance_margin_rate
        self.maker_fee_rate: float = config.maker_fee_rate
        self.taker_fee_rate: float = config.taker_fee_rate
        self.pending_order_id: int | None = None
