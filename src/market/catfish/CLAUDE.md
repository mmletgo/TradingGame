# Catfish 鲶鱼模块

## 模块概述

鲶鱼（Catfish）模块提供一种特殊的市场参与者，用于在训练中引入外部扰动，增加市场动态性。鲶鱼不参与 NEAT 进化，而是按照预设的规则进行交易。

## 文件结构

- `__init__.py` - 模块导出和工厂函数
- `catfish_base.py` - 鲶鱼抽象基类
- `catfish_account.py` - 鲶鱼账户类
- `trend_following.py` - 趋势创造者鲶鱼
- `mean_reversion.py` - 逆势操作型鲶鱼
- `random_trading.py` - 随机买卖型鲶鱼

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
- `check_liquidation(current_price) -> bool` - 检查是否需要强平（净值 <= 0 时强平）
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
- `account: CatfishAccount` - 鲶鱼账户
- `is_liquidated: bool` - 是否已被强平
- `_next_order_id: int` - 下一个订单ID（负数空间，初始值为 catfish_id * 1,000,000）

**核心方法：**
- `decide(orderbook, tick, price_history) -> tuple[bool, int]` - 抽象方法，决策是否行动和方向
- `execute(direction, matching_engine) -> list[Trade]` - 执行市价单
  - 计算下单量（调用 `_calculate_quantity`）
  - 创建市价单 Order（order_id 为负数）
  - 注册鲶鱼费率（maker=0, taker=0）
  - 调用撮合引擎执行订单
  - 返回成交列表
- `_calculate_quantity(orderbook, direction) -> int` - 计算下单量
  - 目标：吃掉对手盘前 1 档
  - 获取盘口深度，累加对手盘前1档的订单数量
  - 如果对手盘档位不足1档，返回0（不行动）
- `_generate_order_id() -> int` - 生成订单ID（负数空间，每次递减1）
- `can_act() -> bool` - 随机概率判断是否行动
  - 返回 `random.random() < config.action_probability`
- `reset()` - 重置鲶鱼状态（账户、强平标志）

### TrendCreatorCatfish (trend_following.py)

趋势创造者鲶鱼，每个 Episode 开始时随机选择方向，整个 Episode 保持该方向持续操作。

**策略逻辑：**
1. Episode 开始时随机选择方向（买或卖）
2. 整个 Episode 保持该方向持续操作
3. 每个 tick 随机概率 `action_probability` 决定是否行动

**额外属性：**
- `_current_direction: int` - 当前方向（1=买，-1=卖）

**reset 方法：**
- 调用基类的 reset 方法
- 重新随机选择方向

### MeanReversionCatfish (mean_reversion.py)

逆势操作型鲶鱼，当价格偏离均线时反向操作。

**策略逻辑：**
1. 检查历史数据是否存在
2. 使用当前价格更新 EMA：`EMA = alpha * price + (1 - alpha) * prev_EMA`
   - 其中 `alpha = 2 / (ma_period + 1)`
   - 首次调用时直接初始化为当前价格
3. 检查是否有足够数据（至少 `ma_period` 个数据点）
4. 计算价格偏离率：`(current_price - EMA) / EMA`
5. 若偏离率绝对值超过 `deviation_threshold`，随机概率 `action_probability` 决定是否行动
   - 价格高于均线则卖出（direction = -1）
   - 价格低于均线则买入（direction = 1）

**额外属性：**
- `_ema: float` - 当前 EMA 值
- `_ema_initialized: bool` - EMA 是否已初始化

**reset 方法：**
- 重置 EMA 为 0.0
- 重置 EMA 初始化标志为 False

### RandomTradingCatfish (random_trading.py)

随机买卖型鲶鱼，以随机概率进行买卖操作。

**策略逻辑：**
1. 每个 tick 随机概率 `action_probability` 决定是否行动
2. 若决定交易，随机选择方向：
   - 50% 概率买入（direction = 1）
   - 50% 概率卖出（direction = -1）

**reset 方法：**
- 调用基类的 reset 方法（重置账户、强平标志）


## 工厂函数

### create_catfish(catfish_id, config, initial_balance=0.0, leverage=10.0, maintenance_margin_rate=0.05) -> CatfishBase

根据 `config.mode` 创建对应类型的鲶鱼实例。

**参数：**
- `catfish_id: int` - 鲶鱼ID（应为负数）
- `config: CatfishConfig` - 鲶鱼配置
- `initial_balance: float` - 初始资金
- `leverage: float` - 杠杆（默认10.0）
- `maintenance_margin_rate: float` - 维持保证金率（默认0.05）

**注意：** 此函数用于单模式创建，当前系统默认使用多模式（三种鲶鱼同时运行）。

### create_all_catfish(config, initial_balance, leverage=10.0, maintenance_margin_rate=0.05) -> list[CatfishBase]

创建所有三种鲶鱼实例。**这是当前系统的默认行为**。

**参数：**
- `config: CatfishConfig` - 鲶鱼配置
- `initial_balance: float` - **必填** 每条鲶鱼的初始资金
- `leverage: float` - 杠杆（默认10.0）
- `maintenance_margin_rate: float` - 维持保证金率（默认0.05）

**返回：**
- 三种鲶鱼实例的列表：[TrendCreator, MeanReversion, RandomTrading]
- ID 分配：-1 (TrendCreator), -2 (MeanReversion), -3 (RandomTrading)
- 三种鲶鱼同时运行，每个 tick 各自独立决定是否行动

## 使用示例

**多模式（推荐，当前默认）：**

```python
from src.config.config import CatfishConfig
from src.market.catfish import create_all_catfish

# 创建三种鲶鱼
config = CatfishConfig(
    enabled=True,
    multi_mode=True,  # 四种模式同时运行
    ma_period=20,            # 默认值
    deviation_threshold=0.003, # 默认值
    action_probability=0.3,  # 每个 tick 30% 概率行动（吃单鲶鱼）
)
catfish_list = create_all_catfish(config, initial_balance=6_600_000_000.0)

# 在每个 tick 调用
for catfish in catfish_list:
    should_act, direction = catfish.decide(orderbook, tick, price_history)
    if should_act:
        trades = catfish.execute(direction, matching_engine)
```

**单模式：**

```python
from src.config.config import CatfishConfig, CatfishMode
from src.market.catfish import create_catfish

# 创建趋势创造者鲶鱼
config = CatfishConfig(
    enabled=True,
    multi_mode=False,
    mode=CatfishMode.TREND_CREATOR,
)
catfish = create_catfish(catfish_id=-1, config=config, initial_balance=6_600_000_000.0)
```

## 配置参数

鲶鱼配置在 `src/config/config.py` 的 `CatfishConfig` 中定义：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| enabled | bool | False | 是否启用鲶鱼 |
| multi_mode | bool | True | 是否同时启用四种模式 |
| mode | CatfishMode | TREND_CREATOR | 单模式时的鲶鱼行为模式 |
| ma_period | int | 20 | 均线周期（EMA计算周期） |
| deviation_threshold | float | 0.003 | 偏离阈值（价格与EMA的偏离率） |
| action_probability | float | 0.3 | 每个 tick 行动的概率（0-1） |

**参数说明：**
- `ma_period`: EMA均线周期，用于逆势操作鲶鱼
- `deviation_threshold`: 价格偏离EMA的阈值（0.003 = 0.3%）
- `action_probability`: 每个 tick 所有鲶鱼独立随机决定是否行动的概率（0.3 = 30%）

## 鲶鱼资金模式

### 有限资金模式（当前）

鲶鱼现在拥有有限资金，参与强平和 ADL 机制：

**初始资金计算：**
```
鲶鱼总资金 = 做市商杠杆后资金 - 其他物种杠杆后资金
每条鲶鱼资金 = 总资金 / 4
```

默认配置下（create_config.py）：
- 做市商：400 × 2M × 1.0 = 800M
- 其他物种：200M + 2M + 300M = 502M
- 鲶鱼总资金：800M - 502M = 298M
- 每条鲶鱼：约 74.5M

**下单量计算（吃单鲶鱼）：**
- 不按自身资金计算，而是按盘口深度计算
- 目标：吃掉对手盘前 1 档（调用 `_calculate_quantity` 方法）
- 累加对手盘前1档的订单数量作为下单量

- 每档挂单量固定为 100
- 在盘口外侧 1~3 档各挂一单
- 每个 tick 共挂 6 单（买卖各 3 单）

**手续费：**
- maker 费率：0
- taker 费率：0

**强平机制：**
- 鲶鱼净值 <= 0 时触发强平（即资金归零才爆仓，不按保证金率判断）
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
