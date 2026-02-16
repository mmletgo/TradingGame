# 噪声交易者模块 (noise_trader)

## 概述
噪声交易者是替代鲶鱼(catfish)的新机制，用于为市场提供随机性流动性和布朗运动价格特征。

## 核心设计
- 100个独立噪声交易者，各自每 tick 以 50% 概率行动
- 行动时 50% 买 / 50% 卖，通过市价单撮合
- 下单量：`max(1, int(lognormvariate(mu=3.0, sigma=1.0)))`
- 由中心极限定理保证净买卖量趋向正态分布 → 价格随机游走

## 文件结构

| 文件 | 说明 |
|------|------|
| `noise_trader.py` | NoiseTrader 类 - 决策和执行 |
| `noise_trader_account.py` | NoiseTraderAccount 类 - 账户和持仓 |
| `__init__.py` | 导出 + `create_noise_traders()` 工厂函数 |

## 账户规则
- 初始余额 1e18（视为无限资金）
- 不触发强平检查
- 手续费为 0
- 可作为 ADL 对手方
- 有完整的持仓和 PnL 跟踪

## 与鲶鱼的差异
| 鲶鱼（旧） | 噪声交易者（新） |
|-----------|----------------|
| 3个，三种策略 | 100个，统一随机 |
| 有限资金，会被强平 | 无限资金，不强平 |
| 强平后episode结束 | 不影响episode |
| 吃盘口前1档 | 按分布采样数量 |

## 接口

### NoiseTrader
- `decide() -> tuple[bool, int, int]`: 返回 (should_act, direction, quantity)
- `execute(direction, quantity, matching_engine) -> list[Trade]`: 执行市价单
- `reset()`: 重置状态

### create_noise_traders(config) -> list[NoiseTrader]
工厂函数，创建 config.count 个噪声交易者，ID 从 -1 到 -count。
