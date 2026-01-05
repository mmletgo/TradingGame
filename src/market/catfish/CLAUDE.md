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
- `get_equity(current_price) -> float` - 计算净值 = 余额 + 浮动盈亏
- `get_margin_ratio(current_price) -> float` - 计算保证金率 = 净值 / 持仓市值（无持仓时返回 inf）
- `check_liquidation(current_price) -> bool` - 检查是否需要强平（保证金率 < 维持保证金率）
- `on_trade(trade, is_buyer) -> None` - 处理成交（鲶鱼手续费为0）
  - 根据买卖方向更新持仓
  - 将已实现盈亏加到余额
- `on_adl_trade(quantity, price, is_taker) -> float` - 处理 ADL 成交
  - 计算实际成交数量（min(quantity, 持仓数量)）
  - 多头：`(price - avg_price) * quantity`
  - 空头：`(avg_price - price) * quantity`
  - 仓位清零时重置均价
  - 返回已实现盈亏
- `reset() -> None` - 重置账户（恢复初始余额，重置持仓）

### CatfishBase (catfish_base.py)

鲶鱼抽象基类，定义鲶鱼的通用接口和行为。

**属性：**
- `catfish_id: int` - 鲶鱼ID（使用负数避免与Agent冲突）
- `config: CatfishConfig` - 鲶鱼配置
- `phase_offset: int` - 相位偏移（用于错开多个鲶鱼的触发时间）
- `account: CatfishAccount` - 鲶鱼账户
- `is_liquidated: bool` - 是否已被强平
- `_next_order_id: int` - 下一个订单ID（负数空间，初始值为 catfish_id * 1,000,000）
- `_last_action_tick: int` - 上次行动的tick（初始值为 -1000，确保首次可以行动）

**核心方法：**
- `decide(orderbook, tick, price_history) -> tuple[bool, int]` - 抽象方法，决策是否行动和方向
- `execute(direction, matching_engine) -> list[Trade]` - 执行市价单
  - 计算下单量（调用 `_calculate_quantity`）
  - 创建市价单 Order（order_id 为负数）
  - 注册鲶鱼费率（maker=0, taker=0）
  - 调用撮合引擎执行订单
  - 返回成交列表
- `_calculate_quantity(orderbook, direction) -> int` - 计算下单量
  - 目标：吃掉对手盘前 3 档
  - 获取盘口深度，累加对手盘前3档的订单数量
  - 如果对手盘档位不足3档，返回0（不行动）
- `_generate_order_id() -> int` - 生成订单ID（负数空间，每次递减1）
- `can_act(tick) -> bool` - 检查冷却时间
  - 返回 `tick - _last_action_tick >= action_cooldown`
  - 注意：虽然参数有 phase_offset，但实际并未在冷却时间计算中使用
- `record_action(tick)` - 记录行动时间
- `reset()` - 重置鲶鱼状态（账户、强平标志、_last_action_tick）

### TrendFollowingCatfish (trend_following.py)

趋势追踪型鲶鱼，根据历史价格变化率顺势下单。

**策略逻辑：**
1. 检查冷却时间 `action_cooldown`
2. 检查历史数据是否足够（至少 `lookback_period` 个数据点）
3. 回看 `lookback_period` 个 tick 的价格（获取 `price_history[-lookback_period]` 和当前价格）
4. 计算价格变化率：`(current_price - start_price) / start_price`
5. 若变化率绝对值超过 `trend_threshold`，顺势下单
   - 价格上涨则买入（direction = 1）
   - 价格下跌则卖出（direction = -1）

### CycleSwingCatfish (cycle_swing.py)

周期摆动型鲶鱼，按固定周期交替买卖。

**策略逻辑：**
1. 计算完整周期：`full_cycle = 2 * half_cycle_length`
2. 计算当前周期位置：`position_in_cycle = tick % full_cycle`
3. 根据周期位置确定方向：
   - 前半周期（position_in_cycle < half_cycle_length）：买入（direction = 1）
   - 后半周期（position_in_cycle >= half_cycle_length）：卖出（direction = -1）
4. 检查是否到达行动间隔：`tick % action_interval == 0`
5. 无视市场状态，按固定周期交替买卖

**额外属性：**
- `_current_direction: int` - 当前方向（1=买，-1=卖）

**注意：**
- 此策略不使用 `can_act()` 方法，不检查 `action_cooldown`
- 仅通过 `tick % action_interval` 控制行动时机

### MeanReversionCatfish (mean_reversion.py)

逆势操作型鲶鱼，当价格偏离均线时反向操作。

**策略逻辑：**
1. 检查冷却时间 `action_cooldown`
2. 检查历史数据是否存在
3. 使用当前价格更新 EMA：`EMA = alpha * price + (1 - alpha) * prev_EMA`
   - 其中 `alpha = 2 / (ma_period + 1)`
   - 首次调用时直接初始化为当前价格
4. 检查是否有足够数据（至少 `ma_period` 个数据点）
5. 计算价格偏离率：`(current_price - EMA) / EMA`
6. 若偏离率绝对值超过 `deviation_threshold`，反向操作
   - 价格高于均线则卖出（direction = -1）
   - 价格低于均线则买入（direction = 1）

**额外属性：**
- `_ema: float` - 当前 EMA 值
- `_ema_initialized: bool` - EMA 是否已初始化

**reset 方法：**
- 重置 EMA 为 0.0
- 重置 EMA 初始化标志为 False

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
- ID 分配：-1 (TrendFollowing), -2 (CycleSwing), -3 (MeanReversion)
- 相位偏移分别为：0, cooldown//3, cooldown*2//3（整数除法）

**相位偏移说明：**
- 相位偏移用于错开三种鲶鱼的触发时间
- 虽然 `can_act` 方法中没有直接使用 `phase_offset`，但不同的 `phase_offset` 值会影响鲶鱼的初始状态和触发时机

## 使用示例

**多模式（推荐，当前默认）：**

```python
from src.config.config import CatfishConfig
from src.market.catfish import create_all_catfish

# 创建三种鲶鱼（相位错开）
config = CatfishConfig(
    enabled=True,
    multi_mode=True,  # 三种模式同时运行
    lookback_period=10,      # 默认值
    trend_threshold=0.002,   # 默认值
    half_cycle_length=10,    # 默认值
    action_interval=5,       # 默认值
    ma_period=20,            # 默认值
    deviation_threshold=0.003, # 默认值
    action_cooldown=10,      # 默认值
)
catfish_list = create_all_catfish(config, initial_balance=6_600_000_000.0)

# 在每个 tick 调用
for catfish in catfish_list:
    should_act, direction = catfish.decide(orderbook, tick, price_history)
    if should_act:
        trades = catfish.execute(direction, matching_engine)
        catfish.record_action(tick)
```

**单模式（已弃用，不推荐使用）：**

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
| fund_multiplier | float | 3.0 | 资金乘数（相对于做市商基础资金） |
| market_maker_base_fund | float | 20,000,000 | 做市商基础资金 |
| lookback_period | int | 10 | 趋势追踪回看周期（tick数） |
| trend_threshold | float | 0.002 | 趋势阈值（价格变化率） |
| half_cycle_length | int | 10 | 周期摆动半周期长度（tick数） |
| action_interval | int | 5 | 周期摆动行动间隔（tick数） |
| ma_period | int | 20 | 均线周期（EMA计算周期） |
| deviation_threshold | float | 0.003 | 偏离阈值（价格与EMA的偏离率） |
| action_cooldown | int | 10 | 行动冷却时间（tick数） |

**参数说明：**
- `lookback_period`: 趋势追踪鲶鱼回看的历史价格数量
- `trend_threshold`: 价格变化率超过此值时触发交易（0.002 = 0.2%）
- `half_cycle_length`: 周期摆动的半周期长度，完整周期 = 2 × half_cycle_length
- `action_interval`: 周期摆动鲶鱼的行动间隔（每N个tick行动一次）
- `ma_period`: EMA均线周期，用于逆势操作鲶鱼
- `deviation_threshold`: 价格偏离EMA的阈值（0.003 = 0.3%）
- `action_cooldown`: 趋势追踪和逆势操作鲶鱼的冷却时间（周期摆动鲶鱼不使用此参数）

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
- 目标：吃掉对手盘前 3 档（调用 `_calculate_quantity` 方法）
- 累加对手盘前3档的订单数量作为下单量

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
