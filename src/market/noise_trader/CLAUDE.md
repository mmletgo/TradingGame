# 噪声交易者模块 (noise_trader)

## 模块概述

噪声交易者是替代鲶鱼(catfish)的新机制，用于为市场提供随机性流动性和布朗运动价格特征。200个独立噪声交易者通过随机买卖行为，由中心极限定理保证净买卖量趋向正态分布，从而产生价格随机游走。

## 设计理念

1. **布朗运动模拟**：200个独立噪声交易者的净买卖量趋向正态分布，价格呈现随机游走特征
2. **无限资金**：初始余额 1e18，不受资金限制，持续提供流动性
3. **零手续费**：maker 和 taker 费率均为 0，降低交易成本
4. **无强平风险**：不触发强平检查，保证市场稳定性
5. **ADL 参与者**：可作为 ADL 对手方，参与风险分摊

## 文件结构

```
noise_trader/
├── __init__.py              # 模块导出 + create_noise_traders() 工厂函数
├── noise_trader.py          # NoiseTrader 类 - 决策和执行
├── noise_trader_account.py  # NoiseTraderAccount 类 - 账户和持仓
└── CLAUDE.md                # 本文档
```

## 核心类

### NoiseTraderAccount (noise_trader_account.py)

噪声交易者账户类，无限资金，不会被强平。保留持仓和 PnL 跟踪，可作为 ADL 对手方。

**属性：**
| 属性 | 类型 | 说明 |
|------|------|------|
| `trader_id` | int | 交易者 ID（负数） |
| `initial_balance` | float | 初始余额（1e18） |
| `balance` | float | 当前余额 |
| `position_qty` | int | 持仓数量（正数=多头，负数=空头） |
| `position_avg_price` | float | 平均持仓价格 |
| `realized_pnl` | float | 已实现盈亏 |

**核心方法：**

#### `on_trade(trade: Trade, is_taker: bool) -> None`

处理成交（无手续费）。

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `trade` | Trade | 成交记录 |
| `is_taker` | bool | 是否为 taker（噪声交易者始终是 taker） |

**执行逻辑：**
1. 根据成交方向确定仓位变化方向
2. 空仓时：直接开仓
3. 同向加仓时：加权平均计算新均价
4. 反向减仓时：实现盈亏，可能反向开仓

#### `on_adl_trade(price: float, quantity: int) -> None`

处理 ADL 成交。

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `price` | float | 成交价格 |
| `quantity` | int | 成交数量 |

**执行逻辑：**
1. 检查是否有持仓
2. 计算平仓数量（取持仓和请求数量的最小值）
3. 计算已实现盈亏
4. 更新持仓和余额

#### `get_equity(current_price: float) -> float`

获取净值。

**公式：** `balance + unrealized_pnl`

**返回值：** 账户净值（浮动盈亏已计入）

#### `reset() -> None`

重置账户。

**执行步骤：**
- 余额恢复到初始值（1e18）
- 持仓数量归零
- 持仓均价归零
- 已实现盈亏归零

### NoiseTrader (noise_trader.py)

噪声交易者类，每 tick 以固定概率行动，行动时随机买/卖，下单量从对数正态分布采样。

**属性：**
| 属性 | 类型 | 说明 |
|------|------|------|
| `trader_id` | int | 交易者 ID（负数，从 -1 开始） |
| `config` | NoiseTraderConfig | 配置对象 |
| `account` | NoiseTraderAccount | 账户对象 |
| `_next_order_id` | int | 下一个订单 ID（负数空间） |

**初始化验证：**
```python
if trader_id >= 0:
    raise ValueError(f"噪声交易者ID必须为负数，当前值: {trader_id}")
```

**核心方法：**

#### `decide(buy_probability: float = 0.5) -> tuple[bool, int, int]`

决策是否行动、方向和数量。

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `buy_probability` | float | 买入概率，默认0.5表示等概率买卖 |

**返回值：** `(should_act, direction, quantity)`
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `should_act` | bool | 是否行动 |
| `direction` | int | 方向（1=买, -1=卖） |
| `quantity` | int | 下单数量 |

**决策逻辑：**
1. 以 `action_probability`（默认 50%）概率决定是否行动
2. 行动时以 `buy_probability` 概率决定买入（默认 50%，可由 Episode 级偏置调整）
3. 下单量从对数正态分布采样：`max(1, int(lognormvariate(mu=12.0, sigma=1.0)))`

#### `execute(direction: int, quantity: int, matching_engine: MatchingEngine) -> list[Trade]`

执行市价单。

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `direction` | int | 方向（1=买, -1=卖） |
| `quantity` | int | 下单数量 |
| `matching_engine` | MatchingEngine | 撮合引擎 |

**返回值：** 成交记录列表

**执行逻辑：**
1. 生成订单 ID（负数空间，递减）
2. 创建市价单
3. 提交给撮合引擎处理
4. 遍历成交记录，更新账户（噪声交易者始终是 taker）

#### `reset() -> None`

重置噪声交易者。

**执行步骤：**
- 重置账户
- 重置订单 ID 生成器

### create_noise_traders() 工厂函数 (__init__.py)

创建噪声交易者列表。

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | NoiseTraderConfig | 噪声交易者配置 |

**返回值：** 噪声交易者列表，ID 从 -1 到 -count

**实现：**
```python
def create_noise_traders(config: "NoiseTraderConfig") -> list[NoiseTrader]:
    from src.config.config import NoiseTraderConfig
    return [
        NoiseTrader(trader_id=-(i + 1), config=config)
        for i in range(config.count)
    ]
```

## 账户规则

| 规则 | 值 | 说明 |
|------|-----|------|
| 初始余额 | 1e18 | 视为无限资金 |
| 强平检查 | 不触发 | 不会被强平 |
| Maker 手续费 | 0 | 无手续费 |
| Taker 手续费 | 0 | 无手续费 |
| ADL 对手方 | 支持 | 可作为 ADL 对手方 |
| 持仓跟踪 | 完整 | 有完整的持仓和 PnL 跟踪 |

## 布朗运动原理

### 中心极限定理

200个独立噪声交易者，各自以 50% 概率行动，行动时 50% 买/50% 卖：

- 单个噪声交易者的期望净买卖量 = 0
- 净买卖量的方差有限
- 由中心极限定理，100个噪声交易者的总净买卖量趋向正态分布

### 价格随机游走

- 净买入 > 净卖出 -> 价格上涨
- 净买入 < 净卖出 -> 价格下跌
- 净买卖量服从正态分布 -> 价格变化服从正态分布 -> 布朗运动

## 与鲶鱼的差异

| 特性 | 鲶鱼（旧） | 噪声交易者（新） |
|------|-----------|----------------|
| 数量 | 3个，三种策略 | 100个，统一随机 |
| 资金 | 有限资金，会被强平 | 无限资金，不强平 |
| 影响 | 强平后 episode 结束 | 不影响 episode |
| 吃单方式 | 吃盘口前1档 | 按分布采样数量 |
| 手续费 | 有 | 0 |

## 模块导出

`__init__.py` 导出以下公共 API：
```python
from src.market.noise_trader.noise_trader import NoiseTrader
from src.market.noise_trader.noise_trader_account import NoiseTraderAccount

__all__ = ["NoiseTrader", "NoiseTraderAccount", "create_noise_traders"]
```

## 依赖关系

### 外部依赖
- `random` - 随机数生成（random.random, random.lognormvariate）

### 内部依赖
- `src.config.config.NoiseTraderConfig` - 噪声交易者配置
- `src.market.orderbook.order.Order` - 订单类
- `src.market.orderbook.order.OrderSide` - 订单方向
- `src.market.orderbook.order.OrderType` - 订单类型
- `src.market.matching.trade.Trade` - 成交记录
- `src.market.matching.matching_engine.MatchingEngine` - 撮合引擎

### 被依赖关系
- `src.training.trainer.Trainer` - 训练引擎
- `src.market.adl.adl_manager.ADLManager` - ADL 管理器

## 使用示例

### 创建噪声交易者

```python
from src.market.noise_trader import create_noise_traders, NoiseTrader
from src.config.config import NoiseTraderConfig

# 使用工厂函数创建
config = NoiseTraderConfig(count=100)
noise_traders = create_noise_traders(config)

# 或手动创建单个噪声交易者
trader = NoiseTrader(trader_id=-1, config=config)
```

### 执行交易

```python
from src.market.matching import MatchingEngine

# 决策
should_act, direction, quantity = trader.decide()

if should_act:
    # 执行市价单
    trades = trader.execute(direction, quantity, matching_engine)
    print(f"成交数量: {len(trades)}")
```

### 在 Trainer 中使用

```python
class Trainer:
    def __init__(self):
        # 创建噪声交易者
        self.noise_traders = create_noise_traders(self.config.noise_trader)

    def run_tick(self):
        # 让噪声交易者行动
        for trader in self.noise_traders:
            should_act, direction, quantity = trader.decide()
            if should_act:
                trader.execute(direction, quantity, self.matching_engine)

        # 预计算 ADL 候选清单（包括噪声交易者）
        for trader in self.noise_traders:
            candidate = self.adl_manager.calculate_adl_score(trader, current_price)
            # ...
```

### Episode 重置

```python
# 每个 Episode 开始时重置所有噪声交易者
for trader in noise_traders:
    trader.reset()
```

## 配置说明

噪声交易者配置来自 `NoiseTraderConfig`：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `count` | int | 200 | 噪声交易者数量 |
| `action_probability` | float | 0.5 | 每个 tick 行动概率 |
| `quantity_mu` | float | 12.0 | 对数正态分布 mu 参数 |
| `quantity_sigma` | float | 1.0 | 对数正态分布 sigma 参数 |
| `episode_bias_range` | float | 0.15 | Episode 级买入概率偏置范围 |
| `ou_theta` | float | 0.035 | OU 过程均值回归速度（每 tick 回归 3.5% 的偏差） |
| `ou_sigma` | float | 0.04 | OU 过程噪声强度 |

**Ornstein-Uhlenbeck 随机过程：**
- buy_prob 本身作为 OU 随机过程演化，防止价格漂移同时避免产生可预测的确定性模式
- 每 tick 更新：`buy_prob[t+1] = buy_prob[t] + θ × (μ - buy_prob[t]) + σ × ε`，其中 μ = episode_buy_prob，ε ~ N(0,1)
- buy_prob 被 clamp 到 [0.1, 0.9] 范围
- 相比确定性均值回归，OU 过程允许临时趋势存在，且回归路径不可预测，PR 无法学到固定的反转模式
- OU 状态在 `trainer.py` 的 `_ou_buy_prob` 和 `arena_state.py` 的 `ArenaState.ou_buy_prob` 中维护
- 每个 episode 开始时 `ou_buy_prob` 初始化为 `episode_buy_prob`

**下单量分布说明：**
- 使用对数正态分布 `lognormvariate(mu, sigma)`
- `mu=12.0, sigma=1.0` 产生的下单量中位数约 162,755，均值约 268,337
- `max(1, int(...))` 确保下单量至少为 1
- 200 个噪声交易者每 tick 总成交量约 2,680 万单位，净力量标准差约 4,400,000 单位，约为散户现实定向力量的 3.4 倍，既主导价格走势又不会击穿做市商双边挂单

## 注意事项

### ID 规则
- 噪声交易者 ID 必须为负数
- 从 -1 开始递减
- 订单 ID 也在负数空间

### 市价单特性
- 噪声交易者只提交市价单
- 始终是 taker 角色
- 未成交部分直接丢弃

### 强平豁免
- 噪声交易者不参与强平检查
- 即使保证金率为负也不会被强平
- 保证市场流动性稳定

### ADL 参与
- 盈利的噪声交易者可以作为 ADL 候选
- 被 ADL 减仓时使用 `on_adl_trade()` 方法
- 不收取手续费

### 资金无限
- 初始余额 1e18，视为无限资金
- 实际交易中不会耗尽资金
- 余额和 PnL 仍然正确跟踪
