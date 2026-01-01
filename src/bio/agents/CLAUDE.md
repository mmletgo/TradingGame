# Agents 模块

## 模块概述

Agent 模块定义了四种类型的 AI 交易代理：散户、高级散户、庄家、做市商。所有 Agent 继承自 `Agent` 基类，通过 NEAT 神经网络进化学习交易策略。

## 文件结构

- `__init__.py` - 模块导出
- `base.py` - Agent 基类和动作类型定义
- `retail.py` - 散户 Agent（继承基类，只能看到10档订单簿和10笔成交）
- `retail_pro.py` - 高级散户 Agent（继承基类，可以看到完整100档订单簿和100笔成交）
- `whale.py` - 庄家 Agent（继承基类）
- `market_maker.py` - 做市商 Agent（重写部分方法）

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
- `is_liquidated: bool` - 强平标志，True 表示已被强平，本轮 episode 禁用
- `_action_handlers: dict[ActionType, Callable[[dict[str, Any], EventBus], None]]` - 动作分发表，将动作类型映射到处理函数
- `_input_buffer: np.ndarray` - 预分配的神经网络输入缓冲区（散户/庄家 607，做市商 634）

**核心方法：**

#### `__init__(agent_id, agent_type, brain, config, event_bus)`
初始化 Agent，使用 `subscribe_with_id` 订阅 TRADE_EXECUTED 事件，支持定向发送。初始化 `is_liquidated` 为 False，预分配神经网络输入缓冲区 `_input_buffer`（607 个 float64），并调用 `_init_action_handlers()` 初始化动作分发表。

#### `_on_trade_event(event: Event) -> None`
处理成交事件。由于已使用定向发送机制，事件必然与本 Agent 相关，无需过滤。

#### `observe(market_state: NormalizedMarketState, orderbook: OrderBook) -> np.ndarray`
从预计算的市场状态构建神经网络输入。使用预分配的 `_input_buffer` 数组，通过切片赋值直接复制数据，避免 `np.concatenate` 创建新数组的开销：
1. 买盘数据 - 200 个（100档 x 2：价格归一化 + 数量）
2. 卖盘数据 - 200 个（100档 x 2：价格归一化 + 数量）
3. 成交数据 - 200 个（100笔价格 + 100笔数量带方向）
4. 持仓信息 - 4 个（持仓归一化、均价归一化、余额归一化、净值归一化）
5. 挂单信息 - 散户/庄家 3 个，做市商 30 个（见下方说明）

做市商重写此方法使用更大的缓冲区（634）并填充 30 个挂单信息。

#### `_get_position_inputs(mid_price: float) -> np.ndarray`
获取持仓信息输入（4 个值），返回 NumPy 数组以支持高效拼接。

#### `_get_pending_order_inputs(mid_price: float, orderbook: OrderBook) -> np.ndarray`
获取挂单信息输入，返回 NumPy 数组。基类返回 3 个值（单挂单），做市商重写返回 30 个值（5 买单 + 5 卖单，每单 3 个值）。做市商实现使用预分配数组 `np.zeros(30)` 避免动态扩展。

**做市商挂单信息（30 个值）：**
- 买单 5 个位置 x 3 = 15（价格归一化、数量、有效标志）
- 卖单 5 个位置 x 3 = 15（价格归一化、数量、有效标志）

#### `_fill_order_inputs(inputs, order_ids, offset, mid_price, orderbook) -> None` (做市商私有)
填充订单输入数组的辅助方法。将指定订单列表的信息填充到输入数组的指定偏移位置。

#### `_cancel_all_orders(event_bus) -> None` (做市商私有)
撤销做市商所有买卖挂单，清空 `bid_order_ids` 和 `ask_order_ids` 列表。

#### `_place_quote_orders(orders, side, order_ids, event_bus) -> None` (做市商私有)
挂限价单并记录订单ID到指定列表。被 `execute_action` 的 QUOTE 动作调用。

#### `decide(market_state: NormalizedMarketState, orderbook: OrderBook) -> tuple[ActionType, dict[str, Any]]`
决策下一步动作。如果已被强平（`is_liquidated=True`），直接返回 HOLD。否则接收预计算的市场状态，调用 `observe` 获取神经网络输入，前向传播得到输出，解析为动作类型和参数。

神经网络输出结构：
- 输出[0-6]: 7 种动作类型的得分（选择最大值）
- 输出[7]: 价格偏移（-1 到 1，映射到 ±100 个 tick）
- 输出[8]: 数量比例（-1 到 1，映射到 0.1-1.0 的购买力比例）

#### `_place_limit_order(side: OrderSide, price: float, quantity: float, event_bus: EventBus) -> None`
创建并发布限价单的私有辅助方法。被 `execute_action` 中的 PLACE_BID 和 PLACE_ASK 动作调用。

#### `_place_market_order(side: OrderSide, quantity: float, event_bus: EventBus) -> None`
创建并发布市价单的私有辅助方法。被 `execute_action` 中的 MARKET_BUY、MARKET_SELL 和 CLEAR_POSITION 动作调用。

#### `_init_action_handlers() -> None`
初始化动作分发表。将动作类型映射到对应的处理函数：
- `PLACE_BID` -> 调用 `_place_limit_order(OrderSide.BUY, ...)`
- `PLACE_ASK` -> 调用 `_place_limit_order(OrderSide.SELL, ...)`
- `CANCEL` -> 调用 `_handle_cancel`
- `MARKET_BUY` -> 调用 `_place_market_order(OrderSide.BUY, ...)`
- `MARKET_SELL` -> 调用 `_place_market_order(OrderSide.SELL, ...)`
- `CLEAR_POSITION` -> 调用 `_handle_clear_position`
- `HOLD` 不在表中，会被自然忽略

子类可重写此方法扩展或替换分发表。散户重写此方法调用 `super()._init_action_handlers()` 后覆盖 PLACE_BID/PLACE_ASK 为特定实现（先撤旧单再挂新单）。

#### `_handle_cancel(params: dict[str, Any], event_bus: EventBus) -> None`
处理撤单动作。从 params 中获取 order_id（可选，默认使用账户的 pending_order_id），发布撤单事件。

#### `_handle_clear_position(params: dict[str, Any], event_bus: EventBus) -> None`
处理清仓动作。根据当前持仓方向发送市价单平仓（多仓卖出，空仓买入）。

#### `execute_action(action, params, event_bus) -> None`
执行动作（事件模式）。使用字典分发模式查找对应的处理函数。如果已被强平（`is_liquidated=True`），不执行任何动作直接返回。否则从 `_action_handlers` 获取处理函数并调用。

#### `execute_action_direct(action, params, matching_engine) -> list[Trade]`
直接执行动作（训练模式，绕过事件系统）。直接调用撮合引擎处理订单，成交后直接更新账户。返回成交列表。

各子类重写此方法以实现特定行为：
- **RetailAgent**: PLACE_BID/PLACE_ASK 先撤旧单再挂新单
- **RetailProAgent**: PLACE_BID/PLACE_ASK 先撤旧单再挂新单
- **WhaleAgent**: 所有动作都先撤旧单
- **MarketMakerAgent**: QUOTE 先撤所有旧单再双边挂单，CLEAR_POSITION 先撤单再平仓

#### `_place_limit_order_direct(side, price, quantity, matching_engine) -> list[Trade]`
直接下限价单（训练模式）。创建订单，调用撮合引擎处理，更新账户，返回成交列表。

#### `_place_market_order_direct(side, quantity, matching_engine) -> list[Trade]`
直接下市价单（训练模式）。创建订单，调用撮合引擎处理，更新账户，返回成交列表。

#### `_handle_clear_position_direct(matching_engine) -> list[Trade]`
直接处理清仓（训练模式）。根据持仓方向下市价单平仓。

#### `_process_trades_direct(trades: list[Trade]) -> None`
直接处理成交列表，更新账户（训练模式）。遍历成交列表，调用 `account.on_trade` 更新账户。

#### `reset(config: AgentConfig) -> None`
重置 Agent 状态。使用 `unsubscribe_with_id` 取消订阅，重置账户，然后重新订阅，并将 `is_liquidated` 重置为 False。

## 事件机制

Agent 使用带 ID 的订阅机制处理成交事件：
- 初始化时调用 `event_bus.subscribe_with_id(EventType.TRADE_EXECUTED, agent_id, handler)`
- 重置时调用 `event_bus.unsubscribe_with_id(EventType.TRADE_EXECUTED, agent_id)`

这样撮合引擎可以通过 `target_ids` 定向发送成交事件给相关 Agent，避免广播给所有 Agent 后各自过滤。

## 直接调用模式（训练优化）

训练模式绕过事件系统，直接调用撮合引擎以提高性能：

### 基类方法
- `execute_action_direct(action, params, matching_engine)` - 直接执行动作，返回成交列表
- `_place_limit_order_direct(side, price, quantity, matching_engine)` - 直接下限价单
- `_place_market_order_direct(side, quantity, matching_engine)` - 直接下市价单
- `_handle_clear_position_direct(matching_engine)` - 直接处理清仓
- `_process_trades_direct(trades)` - 直接处理成交，更新账户

### 做市商特有方法
- `_cancel_all_orders_direct(matching_engine)` - 直接撤销所有挂单
- `_place_quote_orders_direct(orders, side, order_ids, matching_engine)` - 直接挂多个限价单
- `_handle_quote_direct(params, matching_engine)` - 直接处理 QUOTE 动作
- `_handle_clear_position_direct_mm(matching_engine)` - 做市商直接清仓（先撤单再平仓）

## 输入输出规范

### 散户神经网络输入（67 个值）
散户只能看到买卖各10档订单簿和最近10笔成交。

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-19 | 20 | 买盘 10 档（价格归一化 + 数量）|
| 20-39 | 20 | 卖盘 10 档（价格归一化 + 数量）|
| 40-49 | 10 | 成交价格归一化 |
| 50-59 | 10 | 成交数量（正=taker买入，负=taker卖出）|
| 60-63 | 4 | 持仓信息 |
| 64-66 | 3 | 挂单信息（价格归一化、数量、方向）|

### 高级散户神经网络输入（607 个值）
高级散户可以看到完整的100档订单簿和最近100笔成交，与庄家相同的输入维度。
使用父类 Agent 的默认 observe 方法。

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘 100 档（价格归一化 + 数量）|
| 200-399 | 200 | 卖盘 100 档（价格归一化 + 数量）|
| 400-499 | 100 | 成交价格归一化 |
| 500-599 | 100 | 成交数量（正=taker买入，负=taker卖出）|
| 600-603 | 4 | 持仓信息 |
| 604-606 | 3 | 挂单信息（价格归一化、数量、方向）|

### 庄家神经网络输入（607 个值）
庄家可以看到完整的100档订单簿和最近100笔成交。

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘 100 档（价格归一化 + 数量）|
| 200-399 | 200 | 卖盘 100 档（价格归一化 + 数量）|
| 400-499 | 100 | 成交价格归一化 |
| 500-599 | 100 | 成交数量（正=taker买入，负=taker卖出）|
| 600-603 | 4 | 持仓信息 |
| 604-606 | 3 | 挂单信息（价格归一化、数量、方向）|

### 做市商神经网络输入（634 个值）
| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘 100 档（价格归一化 + 数量）|
| 200-399 | 200 | 卖盘 100 档（价格归一化 + 数量）|
| 400-499 | 100 | 成交价格归一化 |
| 500-599 | 100 | 成交数量（正=taker买入，负=taker卖出）|
| 600-603 | 4 | 持仓信息 |
| 604-633 | 30 | 挂单信息（5买单 + 5卖单，每单：价格归一化、数量、有效标志）|

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

做市商的 `decide` 方法对输出解析进行了向量化优化，使用 NumPy 批量处理数量比例和价格偏移的计算，减少 Python 循环开销。

## 依赖关系

- `src.config.config` - 配置类
- `src.core.event_engine` - 事件系统
- `src.bio.brain` - 神经网络
- `src.market.account` - 账户管理
- `src.market.matching` - 撮合引擎和成交记录
- `src.market.market_state` - 市场状态数据
- `src.market.orderbook` - 订单簿
