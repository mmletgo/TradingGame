# Catfish 鲶鱼模块

## 模块概述

鲶鱼（Catfish）模块提供一种特殊的市场参与者，用于在训练中引入外部扰动，增加市场动态性。鲶鱼不参与 NEAT 进化，而是按照预设的规则进行交易。

## 文件结构

- `__init__.py` - 模块导出和工厂函数
- `catfish_base.py` - 鲶鱼抽象基类
- `trend_following.py` - 趋势追踪型鲶鱼
- `cycle_swing.py` - 周期摆动型鲶鱼
- `mean_reversion.py` - 逆势操作型鲶鱼

## 核心类

### CatfishBase (catfish_base.py)

鲶鱼抽象基类，定义鲶鱼的通用接口和行为。

**属性：**
- `catfish_id: int` - 鲶鱼ID（使用负数避免与Agent冲突）
- `config: CatfishConfig` - 鲶鱼配置
- `phase_offset: int` - 相位偏移（用于错开多个鲶鱼的触发时间）

**核心方法：**
- `decide(orderbook, tick, price_history) -> tuple[bool, int]` - 抽象方法，决策是否行动和方向
- `execute(direction, matching_engine) -> list[Trade]` - 执行市价单
- `can_act(tick) -> bool` - 检查冷却时间（考虑相位偏移）
- `record_action(tick)` - 记录行动时间

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

### create_catfish(catfish_id, config, phase_offset=0) -> CatfishBase

根据 `config.mode` 创建对应类型的鲶鱼实例。

**参数：**
- `catfish_id: int` - 鲶鱼ID（应为负数）
- `config: CatfishConfig` - 鲶鱼配置
- `phase_offset: int` - 相位偏移（默认0）

**注意：** 此函数用于单模式创建，当前系统默认使用多模式（三种鲶鱼同时运行）。

### create_all_catfish(config) -> list[CatfishBase]

创建所有三种鲶鱼实例，相位错开以避免同时触发。**这是当前系统的默认行为**。

**参数：**
- `config: CatfishConfig` - 鲶鱼配置

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
catfish_list = create_all_catfish(config)

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
catfish = create_catfish(catfish_id=-1, config=config)
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

## 依赖关系

- `src.config.config` - CatfishConfig, CatfishMode
- `src.market.orderbook.order` - Order, OrderSide, OrderType
- `src.market.matching.trade` - Trade
- `src.market.matching.matching_engine` - MatchingEngine (TYPE_CHECKING)
- `src.market.orderbook.orderbook` - OrderBook (TYPE_CHECKING)
