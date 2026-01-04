# Agents 模块

## 模块概述

Agent 模块定义了五种类型的 AI 交易代理：散户、高级散户、多头庄家、空头庄家、做市商。通过 NEAT 神经网络进化学习交易策略。

## 继承结构

```
Agent (base.py)
├── RetailProAgent (retail_pro.py) - 高级散户，实现散户动作空间的 decide 方法
│   └── RetailAgent (retail.py) - 散户，仅重写 observe 方法
├── WhaleBaseAgent (whale.py) - 庄家抽象基类
│   ├── BullWhaleAgent (bull_whale.py) - 多头庄家
│   └── BearWhaleAgent (bear_whale.py) - 空头庄家
└── MarketMakerAgent (market_maker.py) - 做市商
```

## 文件结构

- `__init__.py` - 模块导出
- `base.py` - Agent 基类和动作类型定义（不含 decide 方法）
- `retail_pro.py` - 高级散户 Agent（继承基类，实现散户/高级散户通用的 decide 方法）
- `retail.py` - 散户 Agent（继承 RetailProAgent，仅重写 observe 方法限制可见市场数据）
- `whale.py` - 庄家 Agent 抽象基类（定义庄家公共逻辑）
- `bull_whale.py` - 多头庄家 Agent（继承 WhaleBaseAgent，只做买入方向）
- `bear_whale.py` - 空头庄家 Agent（继承 WhaleBaseAgent，只做卖出方向）
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
- `is_liquidated: bool` - 强平标志，True 表示已被强平，本轮 episode 禁用
- `_input_buffer: np.ndarray` - 预分配的神经网络输入缓冲区（散户/庄家 607，做市商 634）

**核心方法：**

#### `__init__(agent_id, agent_type, brain, config)`
初始化 Agent。初始化 `is_liquidated` 为 False，预分配神经网络输入缓冲区 `_input_buffer`（607 个 float64）。

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

#### `_cancel_all_orders(matching_engine) -> None` (做市商私有)
撤销做市商所有买卖挂单，清空 `bid_order_ids` 和 `ask_order_ids` 列表。

#### `_place_quote_orders(orders, side, order_ids, matching_engine) -> list[Trade]` (做市商私有)
挂限价单并记录订单ID到指定列表，返回成交列表。被 `execute_action` 的 QUOTE 动作调用。

**注意**: Agent 基类不包含 `decide` 方法。`decide` 方法由各子类根据自己的动作空间实现：
- **RetailProAgent**: 实现散户/高级散户通用的 decide 方法（9个输出节点，7种动作）
- **RetailAgent**: 继承 RetailProAgent 的 decide 方法
- **WhaleBaseAgent**: 实现庄家专用的 decide 方法（6个输出节点：HOLD/限价/市价/CANCEL得分 + 价格偏移 + 数量比例）
- **MarketMakerAgent**: 实现做市商专用的 decide 方法（22个输出节点）

**订单数量约束：** 所有订单数量（quantity）均为 int 类型，最小单位为 1。`_calculate_order_quantity` 方法会将计算结果取整并确保至少为 1。

**价格舍入**：所有订单价格都会舍入到 `tick_size` 的整数倍，避免浮点数精度问题导致订单簿数据不一致。

**价格保护**：所有订单价格都会确保至少为一个 `tick_size`，防止神经网络输出极端负偏移时产生负价格或零价格。当计算出的价格小于 `tick_size` 时，自动调整为 `tick_size`。

#### `_place_limit_order(side: OrderSide, price: float, quantity: int, matching_engine: MatchingEngine) -> list[Trade]`
创建并处理限价单的私有辅助方法。被 `execute_action` 中的 PLACE_BID 和 PLACE_ASK 动作调用。quantity 参数为 int 类型。返回成交列表。

#### `_place_market_order(side: OrderSide, quantity: int, matching_engine: MatchingEngine) -> list[Trade]`
创建并处理市价单的私有辅助方法。被 `execute_action` 中的 MARKET_BUY、MARKET_SELL 和 CLEAR_POSITION 动作调用。quantity 参数为 int 类型。返回成交列表。

#### `_handle_cancel(matching_engine: MatchingEngine) -> None`
处理撤单动作。使用账户的 pending_order_id，直接调用撮合引擎撤单。

#### `_handle_clear_position(matching_engine: MatchingEngine) -> list[Trade]`
处理清仓动作。根据当前持仓方向下市价单平仓（多仓卖出，空仓买入）。返回成交列表。

#### `execute_action(action, params, matching_engine) -> list[Trade]`
执行动作。直接调用撮合引擎处理订单，成交后直接更新账户。如果已被强平（`is_liquidated=True`），不执行任何动作直接返回空列表。返回成交列表。

各子类重写此方法以实现特定行为：
- **RetailAgent**: PLACE_BID/PLACE_ASK 先撤旧单再挂新单
- **RetailProAgent**: PLACE_BID/PLACE_ASK 先撤旧单再挂新单
- **BullWhaleAgent/BearWhaleAgent**: 所有动作都先撤旧单
- **MarketMakerAgent**: QUOTE 先撤所有旧单再双边挂单，CLEAR_POSITION 先撤单再平仓

#### `_process_trades(trades: list[Trade]) -> None`
处理成交列表，更新账户。遍历成交列表，调用 `account.on_trade` 更新账户。

#### `reset(config: AgentConfig) -> None`
重置 Agent 状态。重置账户，并将 `is_liquidated` 重置为 False。

## 做市商特有方法

- `_cancel_all_orders(matching_engine)` - 撤销所有挂单
- `_place_quote_orders(orders, side, order_ids, matching_engine)` - 挂多个限价单
- `_handle_quote(params, matching_engine)` - 处理 QUOTE 动作
- `_handle_clear_position_mm(matching_engine)` - 做市商清仓（先撤单再平仓）

## 输入输出规范

### 归一化方法说明

所有 Agent 的神经网络输入都经过归一化处理，确保输入值在合理的数值范围内：

| 数据类型 | 归一化公式 | 数值范围 | 说明 |
|---------|-----------|---------|------|
| 订单簿价格 | `(price - mid_price) / mid_price` | ≈ [-0.1, 0.1] | 相对中间价的价格偏移 |
| 订单簿数量 | `log10(quantity + 1) / 10` | ≈ [0, 1] | 对数归一化，1e10 → 1.0 |
| 成交价格 | `(price - mid_price) / mid_price` | ≈ [-0.1, 0.1] | 相对中间价的价格偏移 |
| 成交数量 | `sign(qty) * log10(\|qty\| + 1) / 10` | ≈ [-1, 1] | 带方向的对数归一化 |
| 持仓价值 | `position_value / (equity * leverage)` | [0, 1] | 相对可用杠杆的比例 |
| 持仓均价 | `(avg_price - mid_price) / mid_price` | ≈ [-0.1, 0.1] | 相对中间价的价格偏移 |
| 余额 | `balance / initial_balance` | [0, +∞) | 相对初始余额的比例 |
| 净值 | `equity / initial_balance` | [0, +∞) | 相对初始余额的比例 |
| 挂单数量 | `log10(quantity + 1) / 10` | ≈ [0, 1] | 对数归一化 |

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

### 多头庄家/空头庄家神经网络输入（607 个值）
多头庄家和空头庄家可以看到完整的100档订单簿和最近100笔成交。

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

### 散户/高级散户神经网络输出（9 个值）
| 索引 | 说明 |
|------|------|
| 0-6 | 动作类型得分 |
| 7 | 价格偏移（-1 到 1）|
| 8 | 数量比例（-1 到 1）|

### 多头庄家/空头庄家神经网络输出（6 个值）
| 索引 | 说明 |
|------|------|
| 0 | HOLD 动作得分 |
| 1 | 限价单动作得分 |
| 2 | 市价单动作得分 |
| 3 | CANCEL 动作得分 |
| 4 | 价格偏移（-1 到 1，映射到 ±100 个 tick）|
| 5 | 数量比例（-1 到 1，映射到 0.1-1.0）|

多头庄家只做买入方向（PLACE_BID/MARKET_BUY/HOLD/CANCEL），空头庄家只做卖出方向（PLACE_ASK/MARKET_SELL/HOLD/CANCEL）。

### 做市商神经网络输出（22 个值）
| 索引 | 说明 |
|------|------|
| 0 | QUOTE 动作得分 |
| 1 | CLEAR_POSITION 动作得分 |
| 2-6 | 买单 1-5 价格偏移（-1 到 1，映射到 1-100 ticks）|
| 7-11 | 买单 1-5 数量权重 |
| 12-16 | 卖单 1-5 价格偏移（-1 到 1，映射到 1-100 ticks）|
| 17-21 | 卖单 1-5 数量权重 |

**价格偏移映射**：神经网络输出 [-1, 1] 完全控制价格偏移，映射到 [1, 100] ticks。买单价格 = mid_price - offset * tick_size，卖单价格 = mid_price + offset * tick_size。

做市商的 `decide` 方法对输出解析进行了向量化优化，使用 NumPy 批量处理数量比例和价格偏移的计算，减少 Python 循环开销。

## 依赖关系

- `src.config.config` - 配置类
- `src.bio.brain` - 神经网络
- `src.market.account` - 账户管理
- `src.market.matching` - 撮合引擎和成交记录
- `src.market.market_state` - 市场状态数据
- `src.market.orderbook` - 订单簿
