"""回放环境配置"""
from dataclasses import dataclass


@dataclass
class ReplayConfig:
    """回放环境配置

    Attributes:
        hftrade_data_dir: HFtrade monitor/data 目录路径
        exchange: 交易所名称，如 "binance_usdt_swap"
        pair: 交易对名称，如 "btc_usdt"
        date_start: 开始日期 "YYYY-MM-DD"
        date_end: 结束日期 "YYYY-MM-DD"
        tick_size: 最小价格变动单位
        contract_size: 合约面值（Binance=1.0, Gate.io 合约需设置）
        agent_type: "RETAIL_PRO" 或 "MARKET_MAKER"
        initial_balance: 初始资金
        leverage: 杠杆倍数
        maintenance_margin_rate: 维持保证金率
        maker_fee_rate: 挂单费率
        taker_fee_rate: 吃单费率
        episode_length: 每 episode 最大步数（0=使用全部数据）
        ob_depth: 订单簿深度
        trade_history_len: 状态中保留的成交历史长度
        tick_history_len: 状态中保留的 tick 历史长度
        position_cost_weight: 持仓成本惩罚权重 lambda
    """

    hftrade_data_dir: str
    exchange: str
    pair: str
    date_start: str
    date_end: str
    tick_size: float
    contract_size: float = 1.0
    agent_type: str = "RETAIL_PRO"
    initial_balance: float = 20_000.0
    leverage: float = 10.0
    maintenance_margin_rate: float = 0.05
    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0005
    episode_length: int = 10_000
    ob_depth: int = 5
    trade_history_len: int = 100
    tick_history_len: int = 100
    position_cost_weight: float = 0.02
