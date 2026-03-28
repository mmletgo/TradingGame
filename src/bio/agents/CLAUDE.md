# Agents 模块

## 模块概述

Agent 模块定义了两种类型的 AI 交易代理：高级散户、做市商。通过 NEAT 神经网络进化学习交易策略。

**代码统计：**
- Python 代码：约 1,347 行（3 个核心文件）
- Cython 代码：约 398 行（2 个加速模块）
- 总计：约 1,745 行

## 继承结构

```
Agent (base.py) - 基类，提供通用属性和方法，不包含 decide 方法
├── RetailProAgent (retail_pro.py) - 高级散户，实现散户动作空间的 decide 方法
└── MarketMakerAgent (market_maker.py) - 做市商，实现专用的双边挂单 decide 方法
```

**设计原则：**
- Agent 基类提供通用功能（观察市场、执行订单、计算数量）
- decide 方法由各子类实现，每种 Agent 有自己的决策逻辑

## 文件结构

| 文件 | 行数 | 功能 |
|------|------|------|
| `__init__.py` | 14 | 模块导出 |
| `base.py` | 550 | Agent 基类和动作类型定义 |
| `retail_pro.py` | 220 | 高级散户 Agent（实现高级散户通用的 decide 方法）|
| `market_maker.py` | 577 | 做市商 Agent（实现双边挂单的 decide 方法）|
| `_cython/fast_observe.pyx` | 303 | observe 方法的 Cython 加速实现 |
| `_cython/fast_decide.pyx` | 95 | decide 辅助函数的 Cython 加速实现 |

## 核心组件

### ActionType 枚举 (base.py)

定义高级散户可执行的所有交易动作。做市商不使用 ActionType，直接执行双边挂单操作。

```python
class ActionType(Enum):
    HOLD = 0        # 不动
    PLACE_BID = 1   # 挂买单
    PLACE_ASK = 2   # 挂卖单
    CANCEL = 3      # 撤单
    MARKET_BUY = 4  # 市价买入
    MARKET_SELL = 5 # 市价卖出
```

### Agent 基类 (base.py)

Agent 基类是两种 Agent 类型的父类，提供通用的属性和方法。

**核心属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `agent_id` | int | Agent 唯一标识 |
| `agent_type` | AgentType | Agent 类型（RETAIL_PRO/MARKET_MAKER）|
| `brain` | Brain | NEAT 神经网络 |
| `account` | Account | 交易账户 |
| `config` | AgentConfig | Agent 配置对象 |
| `is_liquidated` | bool | 强平标志，True 表示已被强平，本轮 episode 禁用 |
| `_input_buffer` | np.ndarray | 预分配的神经网络输入缓冲区（高级散户 527，做市商 592）|
| `_position_buffer` | np.ndarray | 预分配的持仓信息缓冲区（4个值）|
| `_pending_order_buffer` | np.ndarray | 预分配的挂单信息缓冲区（3个值）|
| `_order_counter` | int | 订单计数器，用于生成唯一订单ID |
| `_cumulative_spread_score` | float | 累积 spread 归一化得分（做市商单竞技场模式使用） |
| `_quote_tick_count` | int | 有效报价 tick 数（做市商单竞技场模式使用） |

**核心方法：**

#### `__init__(agent_id, agent_type, brain, config)`
初始化 Agent。预分配三个缓冲区避免内存频繁分配：
- `_input_buffer`: 527 维（做市商 592）
- `_position_buffer`: 4 维
- `_pending_order_buffer`: 3 维

#### `observe(market_state, orderbook) -> np.ndarray`
从预计算的市场状态构建神经网络输入。使用预分配的 `_input_buffer` 数组，通过切片赋值直接复制数据，避免 `np.concatenate` 创建新数组的开销。

**输入向量结构（527 维）：**
- 0-9: 买盘 5 档（价格归一化 + 数量）
- 10-19: 卖盘 5 档（价格归一化 + 数量）
- 20-119: 成交价格 100 笔
- 120-219: 成交数量 100 笔
- 220-223: 持仓信息（4 个值）
- 224-226: 挂单信息（3 个值）
- 227-526: tick 历史数据（100 价格 + 100 成交量 + 100 成交额）

做市商重写此方法使用 592 维缓冲区（包含 60 个挂单信息和 8 个 AS 特征）。

#### `_get_position_inputs(mid_price) -> np.ndarray`
获取持仓信息输入（4 个值）。使用预分配的 `_position_buffer` 避免创建新数组。

**持仓信息结构：**
- `[0]`: 持仓价值归一化 `position_value / (equity * leverage)`
- `[1]`: 持仓均价归一化 `(avg_price - mid_price) / mid_price`
- `[2]`: 余额归一化 `balance / initial_balance`
- `[3]`: 净值归一化 `equity / initial_balance`

#### `_get_pending_order_inputs(mid_price, orderbook) -> np.ndarray`
获取挂单信息输入（3 个值）。使用预分配的 `_pending_order_buffer` 避免创建新数组。

**挂单信息结构（基类）：**
- `[0]`: 价格归一化 `(order_price - mid_price) / mid_price`
- `[1]`: 数量归一化 `log10(quantity + 1) / 10`
- `[2]`: 方向（买单 1.0，卖单 -1.0，无挂单 0.0）

做市商重写此方法返回 60 个值（10 买单 x 3 + 10 卖单 x 3）。

#### `_calculate_order_quantity(price, ratio, is_buy, ref_price) -> int`
计算订单数量。根据账户净值、杠杆倍数、当前持仓和数量比例计算订单数量，确保下单后的总持仓市值不超过 `equity * leverage`。

**参数：**
- `price`: 订单价格（用于计算最终数量）
- `ratio`: 数量比例（0.0 到 1.0，表示使用可用空间的比例）
- `is_buy`: 是否为买入方向
- `ref_price`: 参考价格（用于计算 equity 和仓位价值，默认为 0 则使用 price）

**返回值：** 整数数量，如果净值为负或可用空间不足则返回 0。

**数量约束：** 所有订单数量（quantity）均为 int 类型，最小单位为 1，最大为 `MAX_ORDER_QUANTITY`（100,000,000）。

#### `_generate_order_id() -> int`
生成唯一订单ID。使用高效的位组合方式：
```python
(agent_id << 32) | _order_counter
```
- agent_id 占高 32 位
- 计数器占低 32 位
- 比 MD5 哈希更高效，适合高频交易场景

#### `_place_limit_order(side, price, quantity, matching_engine) -> list[Trade]`
创建并处理限价单的私有辅助方法。被 `execute_action` 中的 PLACE_BID 和 PLACE_ASK 动作调用。如果订单未完全成交，更新账户的 `pending_order_id`。

#### `_place_market_order(side, quantity, matching_engine) -> list[Trade]`
创建并处理市价单的私有辅助方法。被 `execute_action` 中的 MARKET_BUY 和 MARKET_SELL 动作调用。

#### `execute_action(action, params, matching_engine) -> list[Trade]`
执行动作。直接调用撮合引擎处理订单，成交后直接更新账户。

**注意：** Agent 基类不包含 `decide` 方法。`decide` 方法由各子类根据自己的动作空间实现：
- **RetailProAgent**: 3 个输出节点，6 种动作
- **MarketMakerAgent**: 43 个输出节点，直接输出买卖双边订单参数（含 2 个 AS 调整输出）

各子类重写 `execute_action` 实现特定行为：
- **RetailProAgent**: PLACE_BID/PLACE_ASK 先撤旧单再挂新单
- **MarketMakerAgent**: 始终执行双边挂单，先撤所有旧单再挂新单

#### `_process_trades(trades) -> None`
处理成交列表，更新账户。使用 `trade.is_buyer_taker` 判断 taker 方向（正确处理自成交场景），调用 `account.on_trade` 更新账户。

#### `reset(config) -> None`
重置 Agent 状态。重置账户（创建新的 Account 对象），将 `is_liquidated` 重置为 False，重置订单计数器 `_order_counter` 为 0。

#### `update_brain(genome, config) -> None`
原地更新 brain，复用 Agent 对象。避免在进化时频繁创建销毁 Agent 对象。

## 两种 Agent 类型详解

### 1. RetailProAgent - 高级散户

高级散户代表市场中具有更多信息优势的散户交易者。可以看到 5 档订单簿和 100 笔成交。

**decide 方法：** 实现高级散户的决策逻辑。

**神经网络输出（3 个值）：**
- `[0]`: 动作选择（-1 到 1，等宽分 6 个 bin 选择动作类型）
- `[1]`: 价格偏移（-1 到 1，映射到 ±100 ticks）
- `[2]`: 数量比例（-1 到 1，映射到 [0, 1.0]）

**动作映射公式：**
```python
action_value = clip(output[0], -1.0, 1.0)
action_idx = min(5, int((action_value + 1.0) * 3.0))
# [-1.0, -0.667): HOLD(0)
# [-0.667, -0.333): PLACE_BID(1)
# [-0.333, 0.0): PLACE_ASK(2)
# [0.0, 0.333): CANCEL(3)
# [0.333, 0.667): MARKET_BUY(4)
# [0.667, 1.0]: MARKET_SELL(5)
```

**决策流程：**
1. 如果已被强平，直接返回 HOLD
2. 观察市场，获取神经网络输入
3. 神经网络前向传播
4. 单输出等宽分 6 bin 选择动作类型
5. 解析参数（价格偏移、数量比例）
6. 根据动作类型计算具体参数

**动作参数计算：**
- **PLACE_BID**: 价格由神经网络决定（相对 mid_price 的偏移），数量由神经网络决定（买入方向，限制总持仓）
- **PLACE_ASK**: 价格由神经网络决定，数量由神经网络决定（卖出方向，限制总持仓）
- **MARKET_BUY**: 数量由神经网络决定（买入方向，限制总持仓）
- **MARKET_SELL**: 如果持有多仓则卖出持仓的一定比例，否则开空仓
- **CANCEL/HOLD**: 无参数

**execute_action 方法：** PLACE_BID/PLACE_ASK 先撤旧单再挂新单，其他动作使用父类实现。

### 2. MarketMakerAgent - 做市商

做市商代表市场流动性的提供者，初始资金 10M。通过同时维护买卖双边挂单（每边 1-10 单）为市场提供深度。

**关键属性：**
- `bid_order_ids: list[int]` - 买单订单ID列表（最多 10 个）
- `ask_order_ids: list[int]` - 卖单订单ID列表（最多 10 个）
- `MIN_ORDER_QUANTITY = 1` - 最小订单数量
- `MIN_RATIO_THRESHOLD = 0.001` - 最小权重阈值（0.1%）
- `as_config: ASConfig` - AS 模型配置参数（gamma 范围、kappa、sigma 窗口等，通过 `__init__` 的 `as_config` 参数注入）

**observe 方法：** 重写基类方法，使用更大的输入缓冲区（592 维）。

**输入向量结构（592 维）：**
- 0-9: 买盘 5 档（价格归一化 + 数量）
- 10-19: 卖盘 5 档（价格归一化 + 数量）
- 20-119: 成交价格 100 笔
- 120-219: 成交数量 100 笔
- 220-223: 持仓信息（4 个值）
- 224-283: 挂单信息（60 个值：10 买单 x 3 + 10 卖单 x 3）
- 284-383: tick 历史价格（100 个）
- 384-483: tick 历史成交量（100 个）
- 484-583: tick 历史成交额（100 个）
- 584-591: AS 模型特征（8 个值，见下表）

**AS 模型特征（索引 584-591）：**

| 索引 | 名称 | 说明 |
|------|------|------|
| 584 | reservation_offset | 保留价格偏移（`(r - mid) / mid`，Avellaneda-Stoikov 公式计算的保留价格与中间价之差的相对值）|
| 585 | optimal_half_spread / mid_price | 最优半点差比率（AS 公式给出的最优买卖半点差除以中间价）|
| 586 | sigma | 已实现波动率（近期成交价格的标准差，用于 AS 公式输入）|
| 587 | tau | 剩余时间比例（当前 tick 距 episode 结束的时间比例，`[0, 1]`）|
| 588 | kappa | 订单到达率（对数归一化：`log(kappa + 1) / log(kappa_max + 1)`）|
| 589 | inventory_risk | 库存风险（`gamma * sigma^2 * tau * inventory_ratio`，AS 模型的库存惩罚项）|
| 590 | gamma / max_gamma | 风险厌恶系数比率（当前 gamma 除以种群最大 gamma）|
| 591 | spread / (sigma + 1e-8) | 点差/波动率比率（当前报价点差除以已实现波动率，衡量点差相对波动率的水平）|

**decide 方法：** 实现 AS + NN 双边挂单的决策逻辑（43 个输出节点）。

**神经网络输出（43 个值）：**
- `[0-9]`: 买单 1-10 价格偏移（-1 到 1，映射到 [1, 100] ticks，相对 reservation_price）
- `[10-19]`: 买单 1-10 数量权重（-1 到 1，映射到 [0, 1]）
- `[20-29]`: 卖单 1-10 价格偏移（-1 到 1，映射到 [1, 100] ticks，相对 reservation_price）
- `[30-39]`: 卖单 1-10 数量权重（-1 到 1，映射到 [0, 1]）
- `[40]`: 总下单比例基准（-1 到 1，映射到 [0.01, 1.0]）
- `[41]`: gamma_adjustment（风险厌恶调整乘数，映射到 [0.1, 10.0]，用于缩放 AS 的 gamma 参数）
- `[42]`: spread_adjustment（点差调整乘数，映射到 [0.5, 2.0]，用于缩放 AS 给出的最优点差）

**决策流程：**
1. 如果已被强平，返回空订单列表
2. 观察市场，获取神经网络输入（含 AS 特征）
3. 神经网络前向传播
4. AS 模型计算 reservation_price（保留价格）和 optimal_spread（最优点差）：
   - reservation_price = mid_price - gamma * sigma^2 * tau * inventory_ratio
   - optimal_spread = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/kappa)
5. 解析 AS 调整输出（索引 41-42），计算 gamma 调整和点差调整系数
6. 向量化解析数量比例（使用 NumPy 批量处理）
7. 归一化确保 20 个订单的总比例 = 1.0（**不再**根据仓位 skew 修改数量权重）
8. 应用总下单比例基准
9. 解析价格偏移（买单价格 = reservation_price - offset * tick_size，卖单价格 = reservation_price + offset * tick_size，报价中心为 reservation_price 而非 mid_price）
10. 计算每个订单的数量（使用 `_calculate_order_quantity`，统一以 mid_price 作为价格参数）

**execute_action 方法：** 始终执行双边挂单，调用 `_handle_quote` 先撤所有旧单再挂新单。

## 做市商 AS + NN 混合报价机制

做市商在 `decide()` 方法中实现 Avellaneda-Stoikov（AS）模型与神经网络（NN）的混合报价逻辑。AS 模型负责计算理论最优报价中心和点差，NN 通过学习决定如何调整 AS 参数并决定具体的价格分布和数量分布。

**核心思路：**
- AS 模型根据当前库存、波动率、剩余时间等计算 reservation_price（保留价格）作为报价中心
- 报价点差由 AS 模型的 optimal_spread 乘以 NN 输出的 spread_adjustment 决定
- 数量权重由 NN 直接输出，**不再**通过仓位 skew 修改（库存管理已由 AS 的 reservation_price 内在化）

**AS 模型公式：**
```python
# 保留价格（报价中心偏离 mid_price，随库存增大而偏离以激励减仓）
reservation_price = mid_price - gamma * sigma**2 * tau * inventory_ratio

# 最优点差（随风险厌恶系数 gamma、波动率、剩余时间增大而扩大）
optimal_spread = gamma * sigma**2 * tau + (2 / gamma) * math.log(1 + gamma / kappa)
```

**NN 调整参数：**

| 输出索引 | 名称 | 映射范围 | 作用 |
|----------|------|---------|------|
| 41 | gamma_adjustment | [0.1, 10.0] | 缩放 AS 的风险厌恶系数 gamma |
| 42 | spread_adjustment | [0.5, 2.0] | 缩放 AS 给出的最优点差 |

**数量权重：** 由 NN 输出的索引 10-19（买单）和 30-39（卖单）直接决定，归一化后作为各档订单的比例。仓位对数量的影响已通过 AS 的 reservation_price 内在化（库存偏大时 reservation_price 远离 mid_price，使得该方向的挂单更难成交），而不再通过外部乘数调整权重。

**做市商默认行为：** 做市商每 tick 必然双边挂单，无需动作选择。库存风险管理通过 AS 模型的 reservation_price 偏移实现，神经网络通过调整 gamma 和 spread 来进化出最优做市策略。

## 输入输出规范

### 归一化方法

所有 Agent 的神经网络输入都经过归一化处理，确保输入值在合理的数值范围内：

| 数据类型 | 归一化公式 | 数值范围 | 说明 |
|---------|-----------|---------|------|
| 订单簿价格 | `(price - mid_price) / mid_price` | [-0.1, 0.1] | 相对中间价的价格偏移 |
| 订单簿数量 | `log10(quantity + 1) / 10` | [0, 1] | 对数归一化，1e10 -> 1.0 |
| 成交价格 | `(price - mid_price) / mid_price` | [-0.1, 0.1] | 相对中间价的价格偏移 |
| 成交数量 | `sign(qty) * log10(\|qty\| + 1) / 10` | [-1, 1] | 带方向的对数归一化 |
| 持仓价值 | `position_value / (equity * leverage)` | [0, 1] | 相对可用杠杆的比例 |
| 持仓均价 | `(avg_price - mid_price) / mid_price` | [-0.1, 0.1] | 相对中间价的价格偏移 |
| 余额 | `balance / initial_balance` | [0, +inf) | 相对初始余额的比例 |
| 净值 | `equity / initial_balance` | [0, +inf) | 相对初始余额的比例 |
| 挂单数量 | `log10(quantity + 1) / 10` | [0, 1] | 对数归一化 |

### 神经网络输入向量结构对比

| 区间 | 高级散户（527）| 做市商（592）|
|------|------------------|-------------|
| 买盘 | 0-9（5档）| 0-9（5档）|
| 卖盘 | 10-19（5档）| 10-19（5档）|
| 成交价格 | 20-119（100笔）| 20-119（100笔）|
| 成交数量 | 120-219（100笔）| 120-219（100笔）|
| 持仓信息 | 220-223（4）| 220-223（4）|
| 挂单信息 | 224-226（3）| 224-283（60）|
| tick 历史价格 | 227-326（100）| 284-383（100）|
| tick 历史成交量 | 327-426（100）| 384-483（100）|
| tick 历史成交额 | 427-526（100）| 484-583（100）|
| AS 特征 | -（无）| 584-591（8）|

### 神经网络输出向量结构对比

| 索引 | 高级散户（3）| 做市商（43）|
|------|---------------------|-------------|
| 动作选择/价格 | 0: 动作选择（等宽分 6 bin）| 0-9: 买单价格偏移（相对 reservation_price）|
| 价格偏移 | 1: 价格偏移 | 10-19: 买单数量权重 |
| 数量比例 | 2: 数量比例 | 20-29: 卖单价格偏移（相对 reservation_price）|
| - | - | 30-39: 卖单数量权重 |
| - | - | 40: 总下单比例 |
| - | - | 41: gamma_adjustment（[0.1, 10.0]）|
| - | - | 42: spread_adjustment（[0.5, 2.0]）|

## Cython 加速模块

### fast_observe.pyx

提供 `observe()` 方法的 Cython 加速实现。

**核心函数：**

| 函数名 | 功能 | 输入维度 |
|--------|------|---------|
| `fast_observe_full()` | 构建高级散户的神经网络输入向量 | 527 |
| `fast_observe_market_maker()` | 构建做市商的神经网络输入向量 | 592 |
| `get_position_inputs()` | 计算持仓信息输入 | 4 |
| `get_pending_order_inputs()` | 计算挂单信息输入 | 3 |

**优化点：**
- 使用 `nogil` 释放 GIL，支持多线程并行
- 直接操作 NumPy 数组，避免 Python 开销
- 使用 memoryview 避免数组拷贝

### fast_decide.pyx

提供决策辅助函数的 Cython 加速实现。

**核心函数：**

| 函数名 | 功能 |
|--------|------|
| `fast_argmax(arr, start, end)` | 在指定范围内查找最大值索引 |
| `fast_round_price(price, tick_size)` | 将价格取整到 tick_size 的整数倍 |
| `fast_clip(value, min_val, max_val)` | 将值裁剪到指定范围 |
| `fast_copy_to_buffer(buffer, source, offset, length)` | 快速数据复制 |

**使用方式：**
Agent 类自动检测 Cython 模块是否可用，如果可用则使用加速版本，否则回退到纯 Python 实现。

```python
# 自动检测并使用 Cython 加速
if _HAS_CYTHON_OBSERVE:
    fast_observe_full(self._input_buffer, ...)
else:
    # 回退到纯 Python 实现
    self._input_buffer[:10] = market_state.bid_data
    ...
```

## Agent 类型差异总结

| 特性 | 高级散户 | 做市商 |
|------|----------|--------|
| 初始资金 | 2万 | 10M |
| 杠杆倍数 | 10.0x | 10.0x |
| 订单簿深度 | 5档 | 5档 |
| 成交历史 | 100笔 | 100笔 |
| tick 历史 | 100个 | 100个 |
| 输入维度 | 527 | 592（含 8 个 AS 特征）|
| 输出维度 | 3 | 43（含 2 个 AS 调整输出）|
| 动作空间 | 6种 | 双边挂单（无动作选择）|
| 同时挂单数 | 1个 | 20个（买卖各10个）|
| 撤单再挂 | 是 | 是（每tick全撤全挂）|
| 报价中心 | mid_price（偏移）| reservation_price（AS 模型计算）|

## 重要实现细节

### 预分配缓冲区优化

所有 Agent 都使用预分配的 NumPy 数组作为输入缓冲区，避免在每次 observe 时创建新数组。这是性能优化的关键，避免频繁的内存分配和垃圾回收。

| Agent 类型 | 缓冲区大小 |
|-----------|-----------|
| 高级散户 | 527 维 |
| 做市商 | 592 维（含 8 个 AS 特征）|

### 订单 ID 生成

使用高效的位组合方式生成唯一订单 ID：
```python
(agent_id << 32) | _order_counter
```
- agent_id 占高 32 位
- 计数器占低 32 位
- 比 MD5 哈希更高效，适合高频交易场景

### 成交处理

`_process_trades` 方法使用 `trade.is_buyer_taker` 判断 taker 方向，而不是比较 `buyer_id` 或 `seller_id`，这样可以正确处理自成交场景。

### 挂单信息格式差异

- **基类（高级散户）**: 3 个值（价格归一化、数量、方向）
- **做市商**: 60 个值（10 买单 x 3 + 10 卖单 x 3，包含价格归一化、数量、有效标志）

### 做市商双边挂单

做市商每 tick 必然执行双边挂单，先撤销所有旧挂单（调用 `_cancel_all_orders`），然后双边各挂 1-10 单。神经网络直接输出 20 个订单的价格和数量参数。

### MARKET_BUY / MARKET_SELL 对称设计

MARKET_BUY 和 MARKET_SELL 均统一使用 `_calculate_order_quantity` 计算数量，无论当前持仓状态：
- **MARKET_BUY**: `_calculate_order_quantity(mid_price, ratio, is_buy=True)` — 可以平空仓+开多仓
- **MARKET_SELL**: `_calculate_order_quantity(mid_price, ratio, is_buy=False)` — 可以平多仓+开空仓

这种对称设计确保买卖方向没有结构性偏差，防止价格系统性单向漂移。

### 价格舍入与保护

所有订单价格都会：
1. 舍入到 `tick_size` 的整数倍，避免浮点数精度问题
2. 确保至少为一个 `tick_size`，防止神经网络输出极端负偏移时产生负价格或零价格

### 向量化优化

做市商的 `decide` 方法对输出解析进行了向量化优化：
```python
outputs_arr = np.array(outputs)
bid_raw_ratios = np.maximum(0.0, (np.clip(outputs_arr[10:20], -1, 1) + 1) / 2)
ask_raw_ratios = np.maximum(0.0, (np.clip(outputs_arr[30:40], -1, 1) + 1) / 2)
```

使用 NumPy 批量处理数量比例和价格偏移的计算，减少 Python 循环开销。

## 依赖关系

### 上游依赖

- `src.config.config` - 配置类（AgentConfig, AgentType）
- `src.bio.brain` - 神经网络（Brain）

### 下游依赖

- `src.training.population` - 种群管理（从基因组创建 Agent）

### 内部依赖

- `src.market.account` - 账户管理（Account）
- `src.market.matching` - 撮合引擎和成交记录（MatchingEngine, Trade）
- `src.market.market_state` - 市场状态数据（NormalizedMarketState）
- `src.market.orderbook` - 订单簿（OrderBook, Order, OrderSide, OrderType）

## 相关文档

- `src/bio/brain/CLAUDE.md` - 神经网络模块详细文档
- `src/config/config/CLAUDE.md` - 配置管理文档
- `src/training/CLAUDE.md` - 训练引擎文档
