# Agents 模块

## 模块概述

Agent 模块定义了三种类型的 AI 交易代理：散户、庄家、做市商。所有 Agent 继承自 `Agent` 基类，通过 NEAT 神经网络进化学习交易策略。

## 文件结构

- `__init__.py` - 模块导出
- `base.py` - Agent 基类和动作类型定义
- `retail_agent.py` - 散户 Agent（继承基类）
- `whale_agent.py` - 庄家 Agent（继承基类）
- `market_maker_agent.py` - 做市商 Agent（重写部分方法）

## 核心类

### ActionType (base.py)

动作类型枚举，定义 Agent 可执行的所有交易动作：

- `HOLD = 0` - 不动
- `PLACE_BID = 1` - 挂买单
- `PLACE_ASK = 2` - 挂卖单
- `CANCEL = 3` - 撤单
- `MARKET_BUY = 4` - 市价买入
- `MARKET_SELL = 5` - 市价卖出
- `CLEAR_POSITION = 6` - 清仓
- `QUOTE = 7` - 做市商双边挂单

### Agent (base.py)

Agent 基类，提供通用属性和方法。

**属性：**
- `agent_id: int` - Agent 唯一标识
- `agent_type: AgentType` - Agent 类型（散户/庄家/做市商）
- `brain: Brain` - NEAT 神经网络
- `account: Account` - 交易账户
- `event_bus: EventBus` - 事件总线

**核心方法：**

#### `__init__(agent_id, agent_type, brain, config, event_bus)`
初始化 Agent，使用 `subscribe_with_id` 订阅 TRADE_EXECUTED 事件，支持定向发送。

#### `_on_trade_event(event: Event) -> None`
处理成交事件。由于已使用定向发送机制，事件必然与本 Agent 相关，无需过滤。

#### `observe(market_state: NormalizedMarketState, orderbook: OrderBook) -> list[float]`
从预计算的市场状态构建神经网络输入：
1. 买盘数据 - 200 个（100档 x 2：价格归一化 + 数量）
2. 卖盘数据 - 200 个（100档 x 2：价格归一化 + 数量）
3. 成交数据 - 300 个（100笔 x 3：价格归一化 + 数量 + 方向）
4. 持仓信息 - 4 个（持仓归一化、均价归一化、余额归一化、净值归一化）
5. 挂单信息 - 散户/庄家 3 个，做市商 30 个（见下方说明）

#### `_get_position_inputs(mid_price: float) -> list[float]`
获取持仓信息输入（4 个值）。

#### `_get_pending_order_inputs(mid_price: float, orderbook: OrderBook) -> list[float]`
获取挂单信息输入。基类返回 3 个值（单挂单），做市商重写返回 30 个值（5 买单 + 5 卖单，每单 3 个值）。

**做市商挂单信息（30 个值）：**
- 买单 5 个位置 x 3 = 15（价格归一化、数量、有效标志）
- 卖单 5 个位置 x 3 = 15（价格归一化、数量、有效标志）

#### `decide(market_state: NormalizedMarketState, orderbook: OrderBook) -> tuple[ActionType, dict[str, Any]]`
决策下一步动作。接收预计算的市场状态，调用 `observe` 获取神经网络输入，前向传播得到输出，解析为动作类型和参数。

神经网络输出结构：
- 输出[0-6]: 7 种动作类型的得分（选择最大值）
- 输出[7]: 价格偏移（-1 到 1，映射到 ±100 个 tick）
- 输出[8]: 数量比例（-1 到 1，映射到 0.1-1.0 的购买力比例）

#### `execute_action(action, params, event_bus) -> None`
执行动作，根据动作类型发布订单事件到事件总线。

#### `reset(config: AgentConfig) -> None`
重置 Agent 状态。使用 `unsubscribe_with_id` 取消订阅，重置账户，然后重新订阅。

#### `get_fitness(current_price: float) -> float`
计算适应度（净值 / 初始净值），用于 NEAT 进化。

## 事件机制

Agent 使用带 ID 的订阅机制处理成交事件：
- 初始化时调用 `event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, agent_id, handler)`
- 重置时调用 `event_bus.unsubscribe_with_id(EventType.TRADE_EXECUTED, agent_id)`

这样撮合引擎可以通过 `target_ids` 定向发送成交事件给相关 Agent，避免广播给所有 Agent 后各自过滤。

## 输入输出规范

### 散户/庄家神经网络输入（707 个值）
| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘 100 档（价格归一化 + 数量）|
| 200-399 | 200 | 卖盘 100 档（价格归一化 + 数量）|
| 400-699 | 300 | 成交 100 笔（价格归一化 + 数量 + 方向）|
| 700-703 | 4 | 持仓信息 |
| 704-706 | 3 | 挂单信息（价格归一化、数量、方向）|

### 做市商神经网络输入（734 个值）
| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘 100 档（价格归一化 + 数量）|
| 200-399 | 200 | 卖盘 100 档（价格归一化 + 数量）|
| 400-699 | 300 | 成交 100 笔（价格归一化 + 数量 + 方向）|
| 700-703 | 4 | 持仓信息 |
| 704-733 | 30 | 挂单信息（5买单 + 5卖单，每单：价格归一化、数量、有效标志）|

### 散户/庄家神经网络输出（9 个值）
| 索引 | 说明 |
|------|------|
| 0-6 | 动作类型得分 |
| 7 | 价格偏移（-1 到 1）|
| 8 | 数量比例（-1 到 1）|

### 做市商神经网络输出（22 个值）
| 索引 | 说明 |
|------|------|
| 0 | QUOTE 动作得分 |
| 1 | CLEAR_POSITION 动作得分 |
| 2-6 | 买单 1-5 价格偏移 |
| 7-11 | 买单 1-5 数量权重 |
| 12-16 | 卖单 1-5 价格偏移 |
| 17-21 | 卖单 1-5 数量权重 |

## 依赖关系

- `src.config.config` - 配置类
- `src.core.event_engine` - 事件系统
- `src.bio.brain` - 神经网络
- `src.market.account` - 账户管理
- `src.market.matching` - 成交记录
- `src.market.market_state` - 市场状态数据
- `src.market.orderbook` - 订单簿
