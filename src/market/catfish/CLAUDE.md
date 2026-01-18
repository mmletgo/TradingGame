# Catfish 鲶鱼模块

## 模块概述

鲶鱼（Catfish）模块提供一种特殊的市场参与者，用于在训练中引入外部扰动，增加市场动态性。鲶鱼不参与 NEAT 进化，而是按照预设的规则进行交易。

## 核心职责

- **市场扰动**：通过规则驱动的交易行为为市场引入外部冲击
- **趋势创造**：通过持续单向操作创造价格趋势
- **均值回归**：当价格偏离均值时反向操作，抑制过度波动
- **随机噪音**：通过随机交易增加市场不确定性
- **资金约束**：鲶鱼拥有有限资金，可参与强平和 ADL 机制

## 文件结构

- `__init__.py` - 模块导出和工厂函数
- `catfish_base.py` - 鲶鱼抽象基类
- `catfish_account.py` - 鲶鱼账户类（简化版账户）
- `trend_following.py` - 趋势创造者鲶鱼
- `mean_reversion.py` - 逆势操作型鲶鱼
- `random_trading.py` - 随机买卖型鲶鱼

## 核心类

### CatfishAccount (catfish_account.py)

鲶鱼账户类，简化版账户用于管理鲶鱼的余额、持仓和保证金。

**属性：**
- `catfish_id: int` - 鲶鱼 ID（负数，避免与 Agent 冲突）
- `initial_balance: float` - 初始余额
- `balance: float` - 当前余额（已实现盈亏已计入）
- `position: Position` - 持仓对象（复用市场模块的 Position）
- `leverage: float` - 杠杆倍数
- `maintenance_margin_rate: float` - 维持保证金率

**核心方法：**

#### get_equity(current_price: float) -> float
计算净值 = 余额 + 浮动盈亏

#### get_margin_ratio(current_price: float) -> float
计算保证金率 = 净值 / 持仓市值
- 无持仓时返回 `float("inf")`

#### check_liquidation(current_price: float) -> bool
检查是否需要强平（净值 <= 0 时强平）

#### on_trade(trade: Trade, is_buyer: bool) -> None
处理成交回报（鲶鱼手续费为 0）
- 根据买卖方向更新持仓
- 将已实现盈亏加到余额

#### on_adl_trade(quantity: int, price: float, is_taker: bool) -> float
处理 ADL 成交
- 计算实际成交数量：`min(quantity, |持仓数量|)`
- 多头平仓：`(price - avg_price) * quantity`
- 空头平仓：`(avg_price - price) * quantity`
- 仓位清零时重置均价
- 返回已实现盈亏

#### reset() -> None
重置账户状态（恢复初始余额，重置持仓）

### CatfishBase (catfish_base.py)

鲶鱼抽象基类，定义鲶鱼的通用接口和行为。

**属性：**
- `catfish_id: int` - 鲶鱼ID（使用负数避免与Agent冲突，必须 < 0）
- `config: CatfishConfig` - 鲶鱼配置
- `account: CatfishAccount` - 鲶鱼账户
- `is_liquidated: bool` - 是否已被强平
- `_next_order_id: int` - 下一个订单ID（负数空间，初始值为 `catfish_id * 1,000,000`）

**核心方法：**

#### decide(orderbook, tick, price_history) -> tuple[bool, int]
抽象方法，决策是否行动和方向
- **参数：**
  - `orderbook: OrderBook` - 订单簿
  - `tick: int` - 当前 tick
  - `price_history: Sequence[float]` - 历史价格序列（支持 list 和 deque）
- **返回：** `(should_act, direction)` - 是否行动和方向（1=买，-1=卖）

#### execute(direction: int, matching_engine: MatchingEngine) -> list[Trade]
执行市价单
- 计算下单量（调用 `_calculate_quantity`）
- 创建市价单 Order（order_id 为负数）
- 注册鲶鱼费率（maker=0, taker=0）
- 调用撮合引擎执行订单
- 返回成交列表

#### _calculate_quantity(orderbook: OrderBook, direction: int) -> int
计算下单量（私有方法）
- 目标：吃掉对手盘前 1 档
- 获取盘口深度，累加对手盘前 1 档的订单数量
- 如果对手盘档位不足 1 档，返回 0（不行动）

#### _generate_order_id() -> int
生成订单ID（私有方法）
- 返回负数空间的订单ID
- 每次调用递减 1

#### can_act() -> bool
随机概率判断是否行动
- 返回 `random.random() < config.action_probability`

#### reset() -> None
重置鲶鱼状态（账户、强平标志）

### TrendCreatorCatfish (trend_following.py)

趋势创造者鲶鱼，每个 Episode 开始时随机选择方向，整个 Episode 保持该方向持续操作。

**策略逻辑：**
1. Episode 开始时随机选择方向（买或卖）
2. 整个 Episode 保持该方向持续操作
3. 每个 tick 随机概率 `action_probability` 决定是否行动

**额外属性：**
- `_current_direction: int` - 当前方向（1=买，-1=卖）

**decide 方法实现：**
- 调用 `can_act()` 判断是否行动
- 若行动，返回当前方向

**reset 方法：**
- 调用基类的 reset 方法
- 重新随机选择方向

### MeanReversionCatfish (mean_reversion.py)

逆势操作型鲶鱼，当价格偏离均线时反向操作。

**策略逻辑：**
1. 检查历史数据是否存在（若为空，不行动）
2. 使用当前价格更新 EMA：`EMA = alpha * price + (1 - alpha) * prev_EMA`
   - 其中 `alpha = 2 / (ma_period + 1)`
   - 首次调用时直接初始化为当前价格
3. 检查是否有足够数据（至少 `ma_period` 个数据点）
4. 避免 EMA 为 0 的情况
5. 计算价格偏离率：`(current_price - EMA) / EMA`
6. 若偏离率绝对值超过 `deviation_threshold`，随机概率 `action_probability` 决定是否行动
   - 价格高于均线则卖出（direction = -1）
   - 价格低于均线则买入（direction = 1）

**额外属性：**
- `_ema: float` - 当前 EMA 值
- `_ema_initialized: bool` - EMA 是否已初始化

**reset 方法：**
- 调用基类的 reset 方法
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

**返回类型映射：**
- `CatfishMode.TREND_CREATOR` → `TrendCreatorCatfish`
- `CatfishMode.TREND_FOLLOWING` → `TrendCreatorCatfish`（向后兼容别名）
- `CatfishMode.MEAN_REVERSION` → `MeanReversionCatfish`
- `CatfishMode.RANDOM` → `RandomTradingCatfish`

### create_all_catfish(config, initial_balance, leverage=10.0, maintenance_margin_rate=0.05) -> list[CatfishBase]

创建所有三种鲶鱼实例。**这是当前系统的默认行为**。

**参数：**
- `config: CatfishConfig` - 鲶鱼配置
- `initial_balance: float` - **必填** 每条鲶鱼的初始资金
- `leverage: float` - 杠杆（默认10.0）
- `maintenance_margin_rate: float` - 维持保证金率（默认0.05）

**返回：**
- 三种鲶鱼实例的列表：`[TrendCreatorCatfish, MeanReversionCatfish, RandomTradingCatfish]`
- ID 分配：
  - -1: TrendCreatorCatfish（趋势创造者）
  - -2: MeanReversionCatfish（均值回归）
  - -3: RandomTradingCatfish（随机交易）
- 三种鲶鱼同时运行，每个 tick 各自独立决定是否行动

## 使用示例

### 多模式（推荐，当前默认）

```python
from src.config.config import CatfishConfig
from src.market.catfish import create_all_catfish

# 创建三种鲶鱼
config = CatfishConfig(
    enabled=True,
    ma_period=20,                # EMA 周期
    deviation_threshold=0.003,    # 偏离阈值 0.3%
    action_probability=0.3,       # 每个 tick 30% 概率行动
)
catfish_list = create_all_catfish(config, initial_balance=1_166_000_000.0)

# 在每个 tick 调用
for catfish in catfish_list:
    should_act, direction = catfish.decide(orderbook, tick, price_history)
    if should_act:
        trades = catfish.execute(direction, matching_engine)
```

### 单模式

```python
from src.config.config import CatfishConfig, CatfishMode
from src.market.catfish import create_catfish

# 创建趋势创造者鲶鱼
config = CatfishConfig(
    enabled=True,
    mode=CatfishMode.TREND_CREATOR,
    action_probability=0.3,
)
catfish = create_catfish(
    catfish_id=-1,
    config=config,
    initial_balance=1_166_000_000.0
)
```

## 配置参数

鲶鱼配置在 `src/config/config.py` 的 `CatfishConfig` 中定义：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | False | 是否启用鲶鱼 |
| `multi_mode` | bool | True | 是否同时启用三种模式（默认 True） |
| `mode` | CatfishMode | TREND_CREATOR | 单模式时的鲶鱼行为模式 |
| `ma_period` | int | 20 | EMA 均线周期（用于均值回归鲶鱼） |
| `deviation_threshold` | float | 0.003 | 价格偏离 EMA 的阈值（0.003 = 0.3%） |
| `action_probability` | float | 0.3 | 每个 tick 行动的概率（0-1） |

**CatfishMode 枚举：**
- `TREND_CREATOR` - 趋势创造者
- `TREND_FOLLOWING` - 趋势跟随者（向后兼容别名，同 TREND_CREATOR）
- `MEAN_REVERSION` - 均值回归
- `RANDOM` - 随机交易

**参数说明：**
- `ma_period`: EMA 均线周期，用于均值回归鲶鱼判断价格偏离
- `deviation_threshold`: 价格偏离 EMA 的阈值，超过此值时触发反向操作
- `action_probability`: 每个 tick 所有鲶鱼独立随机决定是否行动的概率

## 三种鲶鱼行为模式详解

### 1. 趋势创造者（TrendCreatorCatfish）

**目标：** 通过持续单向操作创造价格趋势

**行为特征：**
- Episode 开始时随机选择方向（多头或空头）
- 整个 Episode 保持该方向持续操作
- 每个 tick 以 `action_probability` 概率行动
- 行动时始终朝同一方向下单

**市场影响：**
- 创造持续性趋势
- 测试 Agent 的趋势跟踪能力
- 可能导致价格剧烈波动

### 2. 均值回归（MeanReversionCatfish）

**目标：** 当价格偏离均值时反向操作，抑制过度波动

**行为特征：**
- 维护 EMA 均线（周期 `ma_period`）
- 计算价格偏离率：`(price - EMA) / EMA`
- 当偏离率超过 `deviation_threshold` 时触发反向操作
- 价格高于均线 → 卖出
- 价格低于均线 → 买入

**EMA 计算公式：**
```
alpha = 2 / (ma_period + 1)
EMA = alpha * price + (1 - alpha) * prev_EMA
```

**市场影响：**
- 抑制价格过度波动
- 提供均值回归动力
- 测试 Agent 的均值回归策略

### 3. 随机交易（RandomTradingCatfish）

**目标：** 通过随机交易增加市场不确定性

**行为特征：**
- 每个 tick 以 `action_probability` 概率行动
- 行动时随机选择方向（50% 买，50% 卖）
- 纯粹的随机漫步行为

**市场影响：**
- 增加市场噪音
- 提高市场不确定性
- 测试 Agent 的抗干扰能力

## 鲶鱼资金与约束

### 初始资金计算

鲶鱼拥有有限资金，可参与强平和 ADL 机制：

**默认配置下的计算：**
```
做市商杠杆后资金 = 400 × 10M × 1.0 = 4000M
其他物种杠杆后资金 = 200M + 2M + 300M = 502M
鲶鱼总资金 = 4000M - 502M = 3498M
每条鲶鱼资金 = 3498M / 3 ≈ 1166M
```

**设计理念：**
- 鲶鱼资金略大于其他物种总和
- 确保鲶鱼有足够力量影响市场
- 但仍受资金约束，可被强平

### 下单量计算

鲶鱼不按自身资金计算下单量，而是按盘口深度计算：

- 目标：吃掉对手盘前 1 档
- 方法：累加对手盘前 1 档的订单数量
- 若对手盘档位不足，返回 0（不行动）

**代码实现：**
```python
def _calculate_quantity(self, orderbook, direction):
    target_ticks = 1
    depth = orderbook.get_depth(levels=target_ticks)
    levels = depth["asks"] if direction > 0 else depth["bids"]
    if len(levels) < target_ticks:
        return 0
    total_qty = sum(int(qty) for price, qty in levels[:target_ticks])
    return total_qty
```

### 手续费

- Maker 费率：0
- Taker 费率：0
- 鲶鱼所有交易均无手续费

### 强平机制

**触发条件：**
- 鲶鱼净值 <= 0（资金归零）
- 不按保证金率判断，仅资金归零时爆仓

**强平后果：**
- **任意一条鲶鱼被强平 → Episode 立即结束**
- 进入进化阶段
- 所有 Agent 的适应度基于当前状态计算

### ADL 机制

鲶鱼可作为 ADL 候选者（对手方）：
- 按相同的排名公式计算 ADL 分数
- 盈利方：`排名 = PnL% × 有效杠杆`
- 亏损方：`排名 = PnL% / 有效杠杆`
- 使用当前市场价格成交

### Episode 重置

每个 Episode 开始时：
1. 调用 `catfish.reset()` 重置账户和状态
2. 趋势创造者重新随机选择方向
3. 均值回归重置 EMA
4. 所有鲶鱼恢复初始资金

## 订单 ID 空间

鲶鱼使用负数空间的订单 ID，避免与 Agent 冲突：

- 初始值：`catfish_id * 1,000,000`
  - 鲶鱼 -1：从 -1,000,000 开始
  - 鲶鱼 -2：从 -2,000,000 开始
  - 鲶鱼 -3：从 -3,000,000 开始
- 每次下单递减 1

## 依赖关系

### 外部依赖
- `random` - 随机数生成
- `collections.abc.Sequence` - 历史价格序列类型
- `abc.ABC` - 抽象基类

### 内部依赖
- `src.config.config` - `CatfishConfig`, `CatfishMode`
- `src.market.orderbook.order` - `Order`, `OrderSide`, `OrderType`
- `src.market.matching.trade` - `Trade`
- `src.market.matching.matching_engine` - `MatchingEngine` (TYPE_CHECKING)
- `src.market.orderbook.orderbook` - `OrderBook` (TYPE_CHECKING)
- `src.market.account.position` - `Position` (Cython 加速)

## 类型注解

所有模块都使用严格的 Python 类型注解：
- 类属性：明确类型标注
- 方法参数和返回值：完整类型注解
- TYPE_CHECKING：避免循环导入

## 向后兼容性

- `TrendFollowingCatfish` 是 `TrendCreatorCatfish` 的别名
- 支持旧的 `CatfishMode.TREND_FOLLOWING` 枚举值
- 两者完全等价，指向同一个类
