# Catfish 鲶鱼模块

## 模块概述

鲶鱼（Catfish）模块提供一种特殊的市场参与者，用于在训练中引入外部扰动，增加市场动态性。鲶鱼不参与 NEAT 进化，而是按照预设的规则进行交易。

## 文件结构

- `__init__.py` - 模块导出和工厂函数
- `catfish_base.py` - 鲶鱼抽象基类
- `catfish_account.py` - 鲶鱼账户类
- `trend_following.py` - 趋势追踪型鲶鱼
- `cycle_swing.py` - 周期摆动型鲶鱼
- `mean_reversion.py` - 逆势操作型鲶鱼

## 核心类

### CatfishAccount (catfish_account.py)

鲶鱼账户类，简化版账户用于管理鲶鱼的余额、持仓和保证金。

**属性：**
- `catfish_id: int` - 鲶鱼 ID（负数）
- `initial_balance: float` - 初始余额
- `balance: float` - 当前余额
- `position: Position` - 持仓对象
- `leverage: float` - 杠杆倍数
- `maintenance_margin_rate: float` - 维持保证金率

**核心方法：**
- `get_equity(current_price)` - 计算净值 = 余额 + 浮动盈亏
- `get_margin_ratio(current_price)` - 计算保证金率
- `check_liquidation(current_price)` - 检查是否需要强平
- `on_trade(trade, is_buyer)` - 处理成交（鲶鱼手续费为0）
- `on_adl_trade(quantity, price, is_taker)` - 处理 ADL 成交
- `reset()` - 重置账户（Episode 开始时调用）

### CatfishBase (catfish_base.py)

鲶鱼抽象基类，定义鲶鱼的通用接口和行为。

**属性：**
- `catfish_id: int` - 鲶鱼ID（使用负数避免与Agent冲突）
- `config: CatfishConfig` - 鲶鱼配置
- `phase_offset: int` - 相位偏移（用于错开多个鲶鱼的触发时间）
- `account: CatfishAccount` - 鲶鱼账户
- `is_liquidated: bool` - 是否已被强平

**核心方法：**
- `decide(orderbook, tick, price_history) -> tuple[bool, int]` - 抽象方法，决策是否行动和方向
- `execute(direction, matching_engine) -> list[Trade]` - 执行市价单
- `can_act(tick) -> bool` - 检查冷却时间（考虑相位偏移）
- `record_action(tick)` - 记录行动时间
- `reset()` - 重置鲶鱼状态（账户、强平标志等）

### TrendFollowingCatfish (trend_following.py)

趋势追踪型鲶鱼，根据历史价格变化率顺势下单。

**策略逻辑：**
1. 回看 `lookback_period` 个 tick 的价格
2. 计算价格变化率
3. 若变化率超过 `trend_threshold`，顺势下单
4. 有冷却时间 `action_cooldown`

### CycleSwingCatfish (cycle_swing.py)

周期摆动型鲶鱼，按固定周期交替买卖。

**策略逻辑：**
1. 每 `half_cycle_length` 个 tick 切换方向
2. 每 `action_interval` 个 tick 行动一次
3. 无视市场状态，按固定周期交替买卖

### MeanReversionCatfish (mean_reversion.py)

逆势操作型鲶鱼，当价格偏离均线时反向操作。

**策略逻辑：**
1. 维护 EMA 均值（周期 `ma_period`）
2. 当当前价格偏离 EMA 超过 `deviation_threshold`，反向操作
3. 有冷却时间 `action_cooldown`

## 工厂函数

### create_catfish(catfish_id, config, phase_offset=0, initial_balance=0.0, leverage=10.0, maintenance_margin_rate=0.05) -> CatfishBase

根据 `config.mode` 创建对应类型的鲶鱼实例。

**参数：**
- `catfish_id: int` - 鲶鱼ID（应为负数）
- `config: CatfishConfig` - 鲶鱼配置
- `phase_offset: int` - 相位偏移（默认0）
- `initial_balance: float` - 初始资金
- `leverage: float` - 杠杆（默认10.0）
- `maintenance_margin_rate: float` - 维持保证金率（默认0.05）

**注意：** 此函数用于单模式创建，当前系统默认使用多模式（三种鲶鱼同时运行）。

### create_all_catfish(config, initial_balance, leverage=10.0, maintenance_margin_rate=0.05) -> list[CatfishBase]

创建所有三种鲶鱼实例，相位错开以避免同时触发。**这是当前系统的默认行为**。

**参数：**
- `config: CatfishConfig` - 鲶鱼配置
- `initial_balance: float` - **必填** 每条鲶鱼的初始资金
- `leverage: float` - 杠杆（默认10.0）
- `maintenance_margin_rate: float` - 维持保证金率（默认0.05）

**返回：**
- 三种鲶鱼实例的列表：[TrendFollowing, CycleSwing, MeanReversion]
- 相位偏移分别为：0, cooldown/3, cooldown*2/3

## 使用示例

**多模式（推荐，当前默认）：**

```python
from src.config.config import CatfishConfig
from src.market.catfish import create_all_catfish

# 创建三种鲶鱼（相位错开）
config = CatfishConfig(
    enabled=True,
    multi_mode=True,  # 三种模式同时运行
    lookback_period=50,
    trend_threshold=0.02,
)
catfish_list = create_all_catfish(config, initial_balance=6_600_000_000.0)

# 在每个 tick 调用
for catfish in catfish_list:
    should_act, direction = catfish.decide(orderbook, tick, price_history)
    if should_act:
        trades = catfish.execute(direction, matching_engine)
        catfish.record_action(tick)
```

**单模式（已弃用）：**

```python
from src.config.config import CatfishConfig, CatfishMode
from src.market.catfish import create_catfish

# 创建趋势追踪型鲶鱼
config = CatfishConfig(
    enabled=True,
    multi_mode=False,
    mode=CatfishMode.TREND_FOLLOWING,
)
catfish = create_catfish(catfish_id=-1, config=config, initial_balance=6_600_000_000.0)
```

## 配置参数

鲶鱼配置在 `src/config/config.py` 的 `CatfishConfig` 中定义：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| enabled | bool | False | 是否启用鲶鱼 |
| multi_mode | bool | True | 是否同时启用三种模式 |
| mode | CatfishMode | TREND_FOLLOWING | 单模式时的鲶鱼行为模式 |
| fund_multiplier | float | 3.0 | 资金乘数（相对于做市商） |
| market_maker_base_fund | float | 20,000,000 | 做市商基础资金 |
| lookback_period | int | 50 | 趋势追踪回看周期 |
| trend_threshold | float | 0.02 | 趋势阈值 |
| half_cycle_length | int | 100 | 周期摆动半周期长度 |
| action_interval | int | 5 | 周期摆动行动间隔 |
| ma_period | int | 200 | 均线周期 |
| deviation_threshold | float | 0.03 | 偏离阈值 |
| action_cooldown | int | 10 | 行动冷却时间 |

## 鲶鱼资金模式

### 有限资金模式（当前）

鲶鱼现在拥有有限资金，参与强平和 ADL 机制：

**初始资金计算：**
```
鲶鱼总资金 = 做市商杠杆后资金 - 其他物种杠杆后资金
每条鲶鱼资金 = 总资金 / 3
```

默认配置下：
- 做市商：100 × 5000万 × 10 = 500亿
- 其他物种：200亿 + 2亿 + 100亿 = 302亿
- 鲶鱼总资金：500亿 - 302亿 = 198亿
- 每条鲶鱼：66亿

**下单量计算：**
- 不按自身资金计算，而是按盘口深度计算
- 目标：吃掉对手盘前 3 档，产生至少 3 tick 的价格波动

**手续费：**
- maker 费率：0
- taker 费率：0

**强平机制：**
- 鲶鱼保证金率低于维持保证金率时触发强平
- 强平检查在所有 Agent 执行完毕之后进行
- **任意一条鲶鱼被强平 → 本轮 Episode 立即结束，进入进化阶段**

**ADL 机制：**
- 鲶鱼可作为 ADL 候选者（对手方）
- 按相同的排名公式计算 ADL 分数

**Episode 重置：**
- 每个 Episode 开始时调用 `catfish.reset()` 重置账户和状态

## 依赖关系

- `src.config.config` - CatfishConfig, CatfishMode
- `src.market.orderbook.order` - Order, OrderSide, OrderType
- `src.market.matching.trade` - Trade
- `src.market.matching.matching_engine` - MatchingEngine (TYPE_CHECKING)
- `src.market.orderbook.orderbook` - OrderBook (TYPE_CHECKING)
- `src.market.account.position` - Position
