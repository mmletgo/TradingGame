# Agents 模块

## 模块概述

Agent 模块定义了四种类型的 AI 交易代理：散户、高级散户、庄家、做市商。通过 NEAT 神经网络进化学习交易策略。

## 继承结构

```
Agent (base.py) - 基类，不包含 decide 方法
├── RetailProAgent (retail_pro.py) - 高级散户，实现散户动作空间的 decide 方法
│   └── RetailAgent (retail.py) - 散户，继承 RetailProAgent，仅重写 observe 方法
├── WhaleAgent (whale.py) - 庄家，实现与散户相同的 decide 方法
└── MarketMakerAgent (market_maker.py) - 做市商，实现专用的 decide 方法
```

## 文件结构

- `__init__.py` - 模块导出
- `base.py` - Agent 基类和动作类型定义（不含 decide 方法）
- `retail_pro.py` - 高级散户 Agent（继承基类，实现散户/高级散户通用的 decide 方法）
- `retail.py` - 散户 Agent（继承 RetailProAgent，仅重写 observe 方法限制可见市场数据）
- `whale.py` - 庄家 Agent（继承基类，可做多也可做空）
- `market_maker.py` - 做市商 Agent（重写部分方法）
- `_cython/` - Cython 加速模块
  - `__init__.py` - Cython 模块导出
  - `fast_decide.pyx` - 决策辅助函数的 Cython 实现（fast_argmax, fast_round_price, fast_clip）
  - `fast_observe.pyx` - observe 方法的 Cython 实现（fast_observe_retail, fast_observe_full, fast_observe_market_maker）

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
- `QUOTE = 7` - 做市商双边挂单（每边1-5单）

### Agent (base.py)

Agent 基类，提供通用属性和方法。

**属性：**
- `agent_id: int` - Agent 唯一标识
- `agent_type: AgentType` - Agent 类型（散户/庄家/做市商）
- `brain: Brain` - NEAT 神经网络
- `account: Account` - 交易账户
- `config: AgentConfig` - Agent 配置对象
- `is_liquidated: bool` - 强平标志，True 表示已被强平，本轮 episode 禁用
- `_input_buffer: np.ndarray` - 预分配的神经网络输入缓冲区（散户 127，高级散户/庄家 607，做市商 934）
- `_position_buffer: np.ndarray` - 预分配的持仓信息缓冲区（4个值）
- `_pending_order_buffer: np.ndarray` - 预分配的挂单信息缓冲区（3个值）
- `_order_counter: int` - 订单计数器，用于生成唯一订单ID

**核心方法：**

#### `__init__(agent_id, agent_type, brain, config)`
初始化 Agent。初始化 `is_liquidated` 为 False，预分配多个缓冲区（`_input_buffer` 607个float64、`_position_buffer` 4个float64、`_pending_order_buffer` 3个float64），初始化订单计数器 `_order_counter` 为 0。

#### `observe(market_state: NormalizedMarketState, orderbook: OrderBook) -> np.ndarray`
从预计算的市场状态构建神经网络输入。使用预分配的 `_input_buffer` 数组，通过切片赋值直接复制数据，避免 `np.concatenate` 创建新数组的开销：
1. 买盘数据 - 200 个（100档 x 2：价格归一化 + 数量）
2. 卖盘数据 - 200 个（100档 x 2：价格归一化 + 数量）
3. 成交数据 - 200 个（100笔价格 + 100笔数量带方向）
4. 持仓信息 - 4 个（持仓归一化、均价归一化、余额归一化、净值归一化）
5. 挂单信息 - 散户/庄家 3 个，做市商 30 个（见下方说明）

做市商重写此方法使用更大的缓冲区（934）并填充 30 个挂单信息和 300 个 tick 历史数据。

#### `_get_position_inputs(mid_price: float) -> np.ndarray`
获取持仓信息输入（4 个值），返回 NumPy 数组以支持高效拼接。

#### `_get_pending_order_inputs(mid_price: float, orderbook: OrderBook) -> np.ndarray`
获取挂单信息输入，返回 NumPy 数组。基类使用预分配的 `_pending_order_buffer` 返回 3 个值（单挂单：价格归一化、数量、方向），做市商重写返回 30 个值（5 买单 + 5 卖单，每单 3 个值）。做市商实现使用 `np.zeros(30)` 避免动态扩展。

**基类挂单信息（3 个值）：**
- 价格归一化（相对 mid_price 的偏移）
- 数量（对数归一化）
- 方向（买单 1.0，卖单 -1.0）

**做市商挂单信息（30 个值）：**
- 买单 5 个位置 x 3 = 15（价格归一化、数量、有效标志）
- 卖单 5 个位置 x 3 = 15（价格归一化、数量、有效标志）

#### `_fill_order_inputs(inputs, order_ids, offset, mid_price, orderbook) -> None` (做市商私有)
填充订单输入数组的辅助方法。将指定订单列表的信息填充到输入数组的指定偏移位置。每个订单填充 3 个值：价格归一化、数量、有效标志。

#### `_calculate_order_quantity(price: float, ratio: float, is_buy: bool = True, ref_price: float = 0.0) -> int`
计算订单数量。根据账户净值、杠杆倍数、当前持仓和数量比例计算订单数量，确保下单后的总持仓市值不超过 equity * leverage。参数 `ratio` 范围为 0.1 到 1.0，表示使用可用空间的比例。`ref_price` 参数用于计算仓位价值（默认为 0 则使用 price）。返回整数数量，如果净值为负或可用空间不足则返回 0。

#### `_generate_order_id() -> int`
生成唯一订单ID。使用 agent_id 和递增计数器组合（agent_id 占高 32 位，计数器占低 32 位），确保多 Agent 时的唯一性。

**注意**: Agent 基类不包含 `decide` 方法。`decide` 方法由各子类根据自己的动作空间实现：
- **RetailProAgent**: 实现散户/高级散户通用的 decide 方法（9个输出节点，6种动作：HOLD/PLACE_BID/PLACE_ASK/CANCEL/MARKET_BUY/MARKET_SELL）
- **RetailAgent**: 继承 RetailProAgent 的 decide 方法
- **WhaleAgent**: 实现庄家专用的 decide 方法（9个输出节点，7种动作，比散户多一个 CLEAR_POSITION）
- **MarketMakerAgent**: 实现做市商专用的 decide 方法（21个输出节点，直接输出买卖双边订单参数）

**订单数量约束：** 所有订单数量（quantity）均为 int 类型，最小单位为 1，最大为 `MAX_ORDER_QUANTITY`（100,000,000）。`_calculate_order_quantity` 方法会将计算结果取整，小于 1 时返回 0。

**价格舍入**：所有订单价格都会舍入到 `tick_size` 的整数倍，避免浮点数精度问题导致订单簿数据不一致。

**价格保护**：所有订单价格都会确保至少为一个 `tick_size`，防止神经网络输出极端负偏移时产生负价格或零价格。当计算出的价格小于 `tick_size` 时，自动调整为 `tick_size`。

#### `_place_limit_order(side: OrderSide, price: float, quantity: int, matching_engine: MatchingEngine) -> list[Trade]`
创建并处理限价单的私有辅助方法。被 `execute_action` 中的 PLACE_BID 和 PLACE_ASK 动作调用。quantity 参数为 int 类型。返回成交列表。如果订单未完全成交，更新账户的 `pending_order_id`。

#### `_place_market_order(side: OrderSide, quantity: int, matching_engine: MatchingEngine) -> list[Trade]`
创建并处理市价单的私有辅助方法。被 `execute_action` 中的 MARKET_BUY、MARKET_SELL 和 CLEAR_POSITION 动作调用。quantity 参数为 int 类型。返回成交列表。

#### `_handle_clear_position(matching_engine: MatchingEngine) -> list[Trade]`
处理清仓动作。先撤掉挂单（`pending_order_id`），再根据当前持仓方向下市价单平仓（多仓卖出，空仓买入）。返回成交列表。做市商重写此方法以处理多个挂单（调用 `_cancel_all_orders`）。

#### `execute_action(action, params, matching_engine) -> list[Trade]`
执行动作。直接调用撮合引擎处理订单，成交后直接更新账户。如果已被强平（`is_liquidated=True`），不执行任何动作直接返回空列表。返回成交列表。

各子类重写此方法以实现特定行为：
- **RetailAgent**: PLACE_BID/PLACE_ASK 先撤旧单再挂新单，其他动作使用父类实现
- **RetailProAgent**: PLACE_BID/PLACE_ASK 先撤旧单再挂新单，其他动作使用父类实现
- **WhaleAgent**: PLACE_BID/PLACE_ASK 先撤旧单再挂新单，其他动作（包括 CLEAR_POSITION）使用父类实现
- **MarketMakerAgent**: 仅处理 QUOTE 动作，调用 `_handle_quote` 先撤所有旧单再挂新单

#### `_process_trades(trades: list[Trade]) -> None`
处理成交列表，更新账户。遍历成交列表，使用 `trade.is_buyer_taker` 判断 taker 方向，调用 `account.on_trade` 更新账户。

#### `reset(config: AgentConfig) -> None`
重置 Agent 状态。重置账户（创建新的 Account 对象），将 `is_liquidated` 重置为 False，重置订单计数器 `_order_counter` 为 0。

## 子类特有方法

### RetailProAgent

#### `get_action_space() -> list[ActionType]`
返回高级散户可用动作空间（6种动作：HOLD/PLACE_BID/PLACE_ASK/CANCEL/MARKET_BUY/MARKET_SELL）。

#### `decide(market_state, orderbook) -> tuple[ActionType, dict]`
决策下一步动作。观察市场状态，通过神经网络前向传播（9个输出节点），解析输出为动作类型和参数。输出结构：[0-6]动作得分、[7]价格偏移、[8]数量比例。如果已被强平，直接返回 HOLD。

### WhaleAgent

#### `get_action_space() -> list[ActionType]`
返回庄家可用动作空间（7种动作：比高级散户多一个 CLEAR_POSITION）。

#### `decide(market_state, orderbook) -> tuple[ActionType, dict]`
决策下一步动作。与 RetailProAgent 类似（9个输出节点），但动作空间包含 CLEAR_POSITION。

### MarketMakerAgent

**属性：**
- `bid_order_ids: list[int]` - 买单订单ID列表（最多5个）
- `ask_order_ids: list[int]` - 卖单订单ID列表（最多5个）

#### `get_action_space() -> list[ActionType]`
返回空列表（做市商不使用动作选择，默认每 tick 双边挂单）。

#### `observe(market_state, orderbook) -> np.ndarray`
重写基类方法，使用更大的输入缓冲区（934 = 604 + 30 挂单信息 + 300 tick 历史数据）。

#### `_calculate_skew_factor(mid_price: float) -> float`
计算仓位倾斜因子。根据当前仓位计算倾斜因子，范围 [-1, 1]。多头仓位返回负值（减少买单权重，增加卖单权重），空头仓位返回正值（增加买单权重，减少卖单权重）。

#### `_apply_position_skew(bid_raw_ratios, ask_raw_ratios, skew_factor, min_side_weight=0.1) -> tuple`
应用仓位倾斜到买卖权重。确保单边最小权重为 10%，总和为 1.0。

#### `decide(market_state, orderbook) -> tuple[ActionType, dict]`
决策下一步动作。做市商默认每 tick 双边挂单，神经网络直接输出价格和数量参数（21个输出节点）。使用向量化优化处理数量比例和价格偏移的计算。如果已被强平，返回空订单列表。

#### `_cancel_all_orders(matching_engine) -> None`
撤销做市商所有买卖挂单，清空 `bid_order_ids` 和 `ask_order_ids` 列表。

#### `_place_quote_orders(orders, side, order_ids, matching_engine) -> list[Trade]`
挂限价单并记录订单ID到指定列表，返回成交列表。被 `_handle_quote` 方法调用。

#### `_handle_quote(params, matching_engine) -> list[Trade]`
处理 QUOTE 动作。先撤掉所有旧挂单，然后双边各挂 1-5 单（每单价格和数量由神经网络决定）。

#### `_handle_clear_position(matching_engine) -> list[Trade]`
处理做市商清仓。先撤掉所有挂单（调用 `_cancel_all_orders`），再根据持仓方向市价平仓。

## 做市商仓位倾斜挂单机制

做市商在 `decide()` 方法中实现仓位倾斜逻辑，根据当前持仓动态调整买卖双边的挂单权重比例。

**核心思路**：
- 多头仓位 → 卖单权重增加，买单权重减少（倾向平仓）
- 空头仓位 → 买单权重增加，卖单权重减少（倾向平仓）
- 仓位越大，倾斜程度越大
- 始终保持双边挂单，只是比例不同

**倾斜因子计算**：
```python
pos_ratio = position_value / (equity * leverage)  # 0 = 无仓位，1 = 杠杆满
skew_factor = -pos_ratio if position_qty > 0 else pos_ratio  # [-1, 1]
```

**权重调整**：
```python
bid_multiplier = 1.0 + skew_factor  # 多头时减少买单
ask_multiplier = 1.0 - skew_factor  # 多头时增加卖单
```

**效果示例**：

| 仓位状态 | 倾斜因子 | 买单权重 | 卖单权重 |
|----------|----------|----------|----------|
| 无仓位   | 0.0      | 0.50     | 0.50     |
| 多头 50% | -0.5     | 0.25     | 0.75     |
| 多头 100%| -1.0     | 0.10     | 0.90     |
| 空头 50% | +0.5     | 0.75     | 0.25     |
| 空头 100%| +1.0     | 0.90     | 0.10     |

**保护机制**：单边最小权重为 10%（`min_side_weight=0.1`），确保任何情况下都保持双边挂单。

**做市商默认挂单**：做市商每 tick 必然双边挂单，无需动作选择。风险管理完全通过 skew_factor 调整买卖权重来实现。这是因为：
- 做市商的核心职责是提供流动性
- 通过仓位倾斜机制，做市商可以在保持双边挂单的情况下逐步减少不利方向的仓位

**订单数量计算**：做市商调用 `_calculate_order_quantity` 时使用 `ref_price=mid_price` 作为参考价格，确保计算时使用一致的市场价格而非订单价格，避免因买卖价差导致的计算误差。

**向量化优化**：做市商的 `decide` 方法对输出解析进行了向量化优化，使用 NumPy 批量处理数量比例和价格偏移的计算，减少 Python 循环开销。

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

### 散户神经网络输入（127 个值）
散户只能看到买卖各10档订单簿、最近10笔成交和最近20个tick历史数据。

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-19 | 20 | 买盘 10 档（价格归一化 + 数量）|
| 20-39 | 20 | 卖盘 10 档（价格归一化 + 数量）|
| 40-49 | 10 | 成交价格归一化 |
| 50-59 | 10 | 成交数量（正=taker买入，负=taker卖出）|
| 60-63 | 4 | 持仓信息 |
| 64-66 | 3 | 挂单信息（价格归一化、数量、方向）|
| 67-86 | 20 | tick 历史价格（最近20个tick）|
| 87-106 | 20 | tick 历史成交量（最近20个tick）|
| 107-126 | 20 | tick 历史成交额（最近20个tick）|

### 高级散户神经网络输入（907 个值）
高级散户可以看到完整的100档订单簿、最近100笔成交和最近100个tick历史数据，与庄家相同的输入维度。
使用父类 Agent 的默认 observe 方法。

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘 100 档（价格归一化 + 数量）|
| 200-399 | 200 | 卖盘 100 档（价格归一化 + 数量）|
| 400-499 | 100 | 成交价格归一化 |
| 500-599 | 100 | 成交数量（正=taker买入，负=taker卖出）|
| 600-603 | 4 | 持仓信息 |
| 604-606 | 3 | 挂单信息（价格归一化、数量、方向）|
| 607-706 | 100 | tick 历史价格（最近 100 个 tick 的价格归一化）|
| 707-806 | 100 | tick 历史成交量（最近 100 个 tick 的成交量归一化）|
| 807-906 | 100 | tick 历史成交额（最近 100 个 tick 的成交额归一化）|

### 庄家神经网络输入（907 个值）
庄家可以看到完整的100档订单簿、最近100笔成交和最近100个tick历史数据，与高级散户相同的输入维度。

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘 100 档（价格归一化 + 数量）|
| 200-399 | 200 | 卖盘 100 档（价格归一化 + 数量）|
| 400-499 | 100 | 成交价格归一化 |
| 500-599 | 100 | 成交数量（正=taker买入，负=taker卖出）|
| 600-603 | 4 | 持仓信息 |
| 604-606 | 3 | 挂单信息（价格归一化、数量、方向）|
| 607-706 | 100 | tick 历史价格（最近 100 个 tick 的价格归一化）|
| 707-806 | 100 | tick 历史成交量（最近 100 个 tick 的成交量归一化）|
| 807-906 | 100 | tick 历史成交额（最近 100 个 tick 的成交额归一化）|

### 做市商神经网络输入（934 个值）
| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘 100 档（价格归一化 + 数量）|
| 200-399 | 200 | 卖盘 100 档（价格归一化 + 数量）|
| 400-499 | 100 | 成交价格归一化 |
| 500-599 | 100 | 成交数量（正=taker买入，负=taker卖出）|
| 600-603 | 4 | 持仓信息 |
| 604-633 | 30 | 挂单信息（5买单 + 5卖单，每单：价格归一化、数量、有效标志）|
| 634-733 | 100 | tick 历史价格（最近 100 个 tick 的价格归一化）|
| 734-833 | 100 | tick 历史成交量（最近 100 个 tick 的成交量归一化）|
| 834-933 | 100 | tick 历史成交额（最近 100 个 tick 的成交额归一化）|

### 散户/高级散户神经网络输出（9 个值）
| 索引 | 说明 |
|------|------|
| 0-6 | 动作类型得分（HOLD/PLACE_BID/PLACE_ASK/CANCEL/MARKET_BUY/MARKET_SELL）|
| 7 | 价格偏移（-1 到 1）|
| 8 | 数量比例（-1 到 1）|

散户和高级散户使用相同的动作空间（6种动作）和神经网络输出结构。

### 庄家神经网络输出（9 个值）
| 索引 | 说明 |
|------|------|
| 0-6 | 动作类型得分（HOLD/PLACE_BID/PLACE_ASK/CANCEL/MARKET_BUY/MARKET_SELL/CLEAR_POSITION）|
| 7 | 价格偏移（-1 到 1）|
| 8 | 数量比例（-1 到 1）|

庄家与散户/高级散户的区别在于动作空间包含 CLEAR_POSITION（7种动作）。

### 做市商神经网络输出（21 个值）
| 索引 | 说明 |
|------|------|
| 0-4 | 买单 1-5 价格偏移（-1 到 1，映射到 1-100 ticks）|
| 5-9 | 买单 1-5 数量权重（-1 到 1，映射到 0-1）|
| 10-14 | 卖单 1-5 价格偏移（-1 到 1，映射到 1-100 ticks）|
| 15-19 | 卖单 1-5 数量权重（-1 到 1，映射到 0-1）|
| 20 | 总下单比例基准（-1 到 1，映射到 0-1，控制使用多少可用资金下单）|

**价格偏移映射**：神经网络输出 [-1, 1] 映射到 [1, 100] ticks。买单价格 = mid_price - offset * tick_size，卖单价格 = mid_price + offset * tick_size。

**数量权重处理**：先归一化确保 10 个订单的总比例 = 1.0，然后应用仓位倾斜调整，最后乘以总下单比例基准。

**做市商默认行为**：做市商每 tick 必然双边挂单（每边 1-5 单），无需动作选择。神经网络直接输出价格和数量参数。

## 依赖关系

- `src.config.config` - 配置类（AgentConfig, AgentType）
- `src.bio.brain` - 神经网络（Brain）
- `src.market.account` - 账户管理（Account）
- `src.market.matching` - 撮合引擎和成交记录（MatchingEngine, Trade）
- `src.market.market_state` - 市场状态数据（NormalizedMarketState）
- `src.market.orderbook` - 订单簿（OrderBook, Order, OrderSide, OrderType）

## Agent 类型差异总结

| 特性 | 散户 | 高级散户 | 庄家 | 做市商 |
|------|------|----------|------|--------|
| 初始资金 | 10万 | 10万 | 1000万 | 1000万 |
| 杠杆倍数 | 100x | 100x | 10x | 10x |
| 订单簿深度 | 10档 | 100档 | 100档 | 100档 |
| 成交历史 | 10笔 | 100笔 | 100笔 | 100笔 |
| 输入维度 | 127 | 907 | 907 | 934 |
| 输出维度 | 9 | 9 | 9 | 21 |
| 动作空间 | 6种 | 6种 | 7种（多CLEAR_POSITION） | QUOTE（双边挂单）|
| 同时挂单数 | 1个 | 1个 | 1个 | 10个（买卖各5个）|
| 撤单再挂 | 是 | 是 | 是 | 是（每tick全撤全挂）|

## 重要实现细节

### 预分配缓冲区优化
所有 Agent 都使用预分配的 NumPy 数组作为输入缓冲区，避免在每次 observe 时创建新数组。基类 Agent 预分配了三个缓冲区：
- `_input_buffer`: 神经网络输入（607维，做市商934维，散户127维）
- `_position_buffer`: 持仓信息（4维）
- `_pending_order_buffer`: 挂单信息（3维，基类专用）

### 订单ID生成
订单ID由 `agent_id`（高32位）和 `_order_counter`（低32位）组合而成，确保多 Agent 环境下的唯一性，比 MD5 哈希更高效。

### 成交处理
`_process_trades` 方法使用 `trade.is_buyer_taker` 判断 taker 方向，而不是比较 `buyer_id` 或 `seller_id`，这样可以正确处理自成交场景。

### 挂单信息格式差异
- **基类（散户/高级散户/庄家）**: 3个值（价格归一化、数量、方向）
- **做市商**: 30个值（5买单 x 3 + 5卖单 x 3，包含价格归一化、数量、有效标志）

### 清仓行为差异
- **基类**: 先撤单（`pending_order_id`），再市价平仓
- **做市商**: 先撤所有单（`_cancel_all_orders`），再市价平仓

### MARKET_SELL 的特殊逻辑
对于散户/庄家的 MARKET_SELL 动作：
- 如果持有多仓（quantity > 0）：卖出持仓的一定比例
- 如果空仓或持有空仓（quantity <= 0）：开空仓，使用 `_calculate_order_quantity` 计算数量

这与 MARKET_BUY 不同，MARKET_BUY 总是使用 `_calculate_order_quantity` 计算数量（无论是平空仓还是开多仓）。

## Cython 加速模块

### fast_observe.pyx

提供 `observe()` 方法的 Cython 加速实现，用于构建神经网络输入向量。每次调用仅需 1-2 微秒。

**核心函数：**

#### `fast_observe_retail(output_buffer, bid_data, ask_data, trade_prices, trade_quantities, tick_history_prices, tick_history_volumes, tick_history_amounts, position_inputs..., pending_inputs...)`
构建散户的神经网络输入向量（127 维）。使用 `nogil` 释放 GIL，支持多线程并行。

#### `fast_observe_full(output_buffer, ...)`
构建高级散户/庄家的神经网络输入向量（907 维）。

#### `fast_observe_market_maker(output_buffer, ..., pending_order_inputs)`
构建做市商的神经网络输入向量（934 维）。做市商的挂单信息通过 30 维数组传入。

#### `get_position_inputs(equity, leverage, position_quantity, position_avg_price, balance, initial_balance, mid_price) -> tuple`
计算持仓信息输入（4 个值）。返回 `(position_value_normalized, position_avg_price_normalized, balance_normalized, equity_normalized)`。

#### `get_pending_order_inputs(order_price, order_quantity, order_side, mid_price) -> tuple`
计算挂单信息输入（3 个值）。返回 `(pending_price_normalized, pending_qty_normalized, pending_side)`。

**使用方式：**
Agent 类在 `observe()` 方法中自动检测 Cython 模块是否可用，如果可用则使用加速版本，否则回退到纯 Python 实现。

```python
# 自动检测并使用 Cython 加速
if _HAS_CYTHON_OBSERVE:
    # 使用 Cython 实现
    fast_observe_full(self._input_buffer, ...)
else:
    # 回退到纯 Python 实现
    self._input_buffer[:200] = market_state.bid_data
    ...
```

### fast_decide.pyx

提供决策辅助函数的 Cython 加速实现。

**核心函数：**

#### `fast_argmax(arr, start, end) -> int`
在指定范围内查找最大值索引。使用 `nogil` 释放 GIL。

#### `fast_round_price(price, tick_size) -> float`
将价格取整到 tick_size 的整数倍，确保最小值为 tick_size。

#### `fast_clip(value, min_val, max_val) -> float`
将值裁剪到指定范围。
