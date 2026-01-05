# Market 模块

## 模块概述

市场模块负责交易市场引擎的核心功能，包括订单簿管理、撮合引擎、账户管理和市场状态数据。

## 文件结构

- `__init__.py` - 模块导出
- `market_state.py` - 归一化市场状态数据类
- `orderbook/` - 订单簿实现（Cython 加速）
- `matching/` - 撮合引擎
- `account/` - 账户管理（持仓、余额、保证金、强平）
- `adl/` - ADL 自动减仓模块
- `catfish/` - 鲶鱼模块（市场扰动器）

## 核心类

### NormalizedMarketState (market_state.py)

预计算的归一化市场数据，用于缓存每个 tick 的公共市场数据，避免每个 Agent 重复计算。

**属性：**
- `mid_price: float` - 中间价（用于归一化计算的参考价格）
- `tick_size: float` - 最小价格变动单位
- `bid_data: NDArray[np.float32]` - 买盘数据，shape (200,)，100档 × 2（价格归一化 + 数量归一化）
- `ask_data: NDArray[np.float32]` - 卖盘数据，shape (200,)，100档 × 2（价格归一化 + 数量归一化）
- `trade_prices: NDArray[np.float32]` - 成交价格归一化，shape (100,)
- `trade_quantities: NDArray[np.float32]` - 成交数量归一化（带方向），shape (100,)，正数表示 taker 是买方，负数表示 taker 是卖方

**归一化方法：**
- 价格归一化：`(price - mid_price) / mid_price`，范围约 [-0.1, 0.1]
- 数量归一化：`log10(quantity + 1) / 10`，将 1e10 压缩到约 1.0
- 成交数量带方向归一化：`sign(qty) * log10(|qty| + 1) / 10`

**使用场景：**
在每个 tick 开始时由 Trainer 预计算一次，然后传递给所有 Agent 使用，避免重复计算订单簿数据的归一化。

## 子模块

### orderbook/

订单簿实现，使用 Cython 加速，支持买卖各 100 档。

**核心类：**

- `Order` - 订单数据类（Python）
  - `order_id: int` - 订单唯一标识
  - `agent_id: int` - 所属 Agent ID
  - `side: OrderSide` - 买卖方向（BUY=1, SELL=-1）
  - `order_type: OrderType` - 订单类型（LIMIT=1, MARKET=2）
  - `price: float` - 委托价格
  - `quantity: int` - 委托数量（整数）
  - `filled_quantity: int` - 已成交数量（整数，初始为 0）
  - `timestamp: float` - 时间戳（训练模式下默认为 0）

- `OrderSide` - 订单方向枚举（IntEnum）
  - `BUY = 1` - 买单
  - `SELL = -1` - 卖单

- `OrderType` - 订单类型枚举（IntEnum）
  - `LIMIT = 1` - 限价单
  - `MARKET = 2` - 市价单

- `PriceLevel` (Cython cdef class) - 价格档位类
  - `price: double` - 档位价格
  - `orders: OrderedDict[int, Order]` - 订单映射（order_id -> Order），保持 FIFO 顺序
  - `total_quantity: long long` - 64位整数，档位总数量，避免大量订单累积时溢出
  - 核心方法：
    - `add_order(order) -> None` - 添加订单到档位，O(1) 操作
    - `remove_order(order_id) -> Order | None` - 从档位移除订单，O(1) 操作
    - `get_volume() -> int` - 获取档位总挂单量

- `OrderBook` (Cython cdef class) - 订单簿类
  - `bids: SortedDict[float, PriceLevel]` - 买盘，升序排列
  - `asks: SortedDict[float, PriceLevel]` - 卖盘，升序排列
  - `order_map: dict[int, Order]` - 全局订单索引（order_id -> Order）
  - `last_price: double` - 最新成交价
  - `tick_size: double` - 最小变动单位
  - `_depth_dirty: bint` - 深度缓存失效标志
  - `_cached_depth: object` - 缓存的深度数据
  - 核心方法：
    - `add_order(order) -> None` - 向订单簿添加订单
    - `cancel_order(order_id) -> Order | None` - 撤销订单
    - `get_best_bid() -> float | None` - 获取最优买价（买盘最高价）
    - `get_best_ask() -> float | None` - 获取最优卖价（卖盘最低价）
    - `get_mid_price() -> float | None` - 获取中间价
    - `get_depth(levels=100) -> dict` - 获取盘口深度数据
    - `clear(reset_price=None) -> None` - 清空订单簿

**关键优化：**
- 价格归一化：消除浮点精度问题，确保档位键一致性
- O(1) 订单查找：order_map 提供全局订单查找
- O(1) 档位删除：OrderedDict 提供 O(1) 订单删除，同时保持 FIFO
- O(1) 最优价格：SortedDict.peekitem() 提供 O(1) 最优价格查询
- O(levels) 深度查询：利用 SortedDict 有序性，避免排序
- 缓存机制：深度数据缓存，避免重复计算

**核心类型说明：**
- `Order.quantity: int` - 订单数量（整数）
- `Order.filled_quantity: int` - 已成交数量（整数）
- `PriceLevel.total_quantity: long long` - 价格档位总数量（64位整数，支持大量订单累积）
- `get_volume() -> int` - 返回订单簿总成交量（整数）

### matching/

撮合引擎，实现价格优先、时间优先的撮合规则。

**核心类：**
- `MatchingEngine` - 撮合引擎类
  - `_config: MarketConfig` - 市场配置
  - `_orderbook: OrderBook` - 订单簿实例
  - `_next_trade_id: int` - 下一个成交 ID
  - `_fee_rates: dict[int, tuple[float, float]]` - Agent 费率缓存
  - 核心方法：
    - `register_agent(agent_id, maker_rate, taker_rate)` - 注册 Agent 费率配置
    - `calculate_fee(agent_id, amount, is_maker)` - 计算手续费
    - `process_order(order)` - 处理订单入口
    - `match_limit_order(order)` - 限价单撮合
    - `match_market_order(order)` - 市价单撮合
    - `cancel_order(order_id)` - 撤单
    - `_match_orders(order, price_check)` - 通用撮合逻辑（私有方法）

- `Trade` - 成交记录数据类
  - `trade_id: int` - 成交 ID
  - `price: float` - 成交价格
  - `quantity: int` - 成交数量（整数）
  - `buyer_id: int` - 买方 Agent ID
  - `seller_id: int` - 卖方 Agent ID
  - `buyer_fee: float` - 买方手续费
  - `seller_fee: float` - 卖方手续费
  - `is_buyer_taker: bool` - 买方是否为 taker
  - `timestamp: float` - 成交时间戳

**撮合规则：**
- 价格优先：最优价格优先成交
- 时间优先：同价格订单按时间顺序成交（FIFO）
- 限价单：未成交部分挂在订单簿上
- 市价单：吃对手盘直到完全成交或对手盘为空，未成交部分丢弃

**费率配置：**
| Agent 类型 | 挂单费率 | 吃单费率 |
|-----------|----------|----------|
| 散户 | 0.0002 (万2) | 0.0005 (万5) |
| 庄家 | -0.0001 (负万1, maker rebate) | 0.0001 (万1) |
| 做市商 | -0.0001 (负万1, maker rebate) | 0.0001 (万1) |
| 鲶鱼 | 0 | 0 |

**性能优化：**
- 预计算 taker 费率到循环外
- 内联费率计算，避免函数调用开销
- 部分成交时同步更新 `PriceLevel.total_quantity`
- 僵尸订单检测与清理
- 空档位清理
- 训练模式使用固定时间戳 0.0

**核心类型说明：**
- `Order.quantity: int` - 订单数量（整数）
- `Order.filled_quantity: int` - 已成交数量（整数）
- `Trade.quantity: int` - 成交数量（整数）

### account/

账户管理，包括持仓管理、余额管理、保证金计算和强平逻辑。

**核心类：**

- `Position` (Cython cdef class) - 持仓类（Cython 加速）
  - `quantity: int` - 持仓数量（正数=多头，负数=空头，0=空仓）
  - `avg_price: double` - 平均开仓价格
  - `realized_pnl: double` - 已实现盈亏累计
  - 核心方法：
    - `update(side, quantity, price) -> float` - 更新持仓，返回已实现盈亏
    - `get_unrealized_pnl(current_price) -> float` - 计算浮动盈亏
    - `get_margin_used(current_price, leverage) -> float` - 计算占用保证金

- `Account` - 账户类（Python 实现）
  - `agent_id: int` - Agent ID
  - `agent_type: AgentType` - Agent 类型
  - `initial_balance: float` - 初始余额
  - `balance: float` - 当前余额（已实现盈亏已计入）
  - `position: Position` - 持仓对象
  - `leverage: float` - 杠杆倍数（散户/高级散户=100，庄家/做市商=10）
  - `maintenance_margin_rate: float` - 维持保证金率
  - `maker_fee_rate: float` - 挂单手续费率
  - `taker_fee_rate: float` - 吃单手续费率
  - `pending_order_id: int | None` - 当前挂单 ID
  - 核心方法：
    - `get_equity(current_price) -> float` - 计算净值 = 余额 + 浮动盈亏
    - `get_margin_ratio(current_price) -> float` - 计算保证金率
    - `check_liquidation(current_price) -> bool` - 检查是否需要强平
    - `on_trade(trade, is_buyer) -> None` - 处理成交回报
    - `on_adl_trade(quantity, price, is_taker) -> float` - 处理 ADL 成交

**持仓更新逻辑：**
- 空仓开仓（多/空）
- 加仓（多加多/空加空）- 使用加权平均法计算新均价
- 减仓（多减/空减）- 实现盈亏
- 完全平仓 - 实现盈亏，重置均价
- 反向开仓（多转空/空转多）- 先平仓实现盈亏，再开新仓

**保证金和强平机制：**
- 占用保证金：`margin_used = |quantity| × current_price / leverage`
- 净值：`equity = balance + unrealized_pnl`
- 保证金率：`margin_ratio = equity / (|quantity| × current_price)`
- 强平触发条件：`margin_ratio < maintenance_margin_rate`

**核心类型说明：**
- `Position.quantity: int` - 持仓数量（整数，正数为多仓，负数为空仓）

### adl/

ADL（Auto-Deleveraging）自动减仓模块，在强平订单无法完全成交时触发。

**核心类：**
- `ADLCandidate` - ADL 候选者信息（包含持仓、盈亏百分比、有效杠杆、ADL 分数）
  - `participant: Union[Agent, CatfishBase]` - 支持代理或鲶鱼作为候选者
  - `position_qty: int` - 持仓数量（正=多头，负=空头）
  - `pnl_percent: float` - 盈亏百分比
  - `effective_leverage: float` - 有效杠杆
  - `adl_score: float` - ADL 排名分数（越高越优先）
  - `agent: Agent` (property) - 兼容属性，仅当 participant 是 Agent 时可用
  - `is_catfish: bool` (property) - 检查候选者是否为鲶鱼

- `ADLManager` - ADL 管理器，负责计算排名和执行减仓

**核心方法：**
- `get_adl_price(current_price) -> float` - 获取 ADL 成交价格（直接使用当前市场价格）
- `calculate_adl_score(agent, current_price) -> ADLCandidate | None` - 计算 ADL 排名分数

**ADL 成交价格：**
- ADL 直接使用当前市场价格成交，不使用破产价格
- 强平 ≠ 破产：被强平时 Agent 可能还有正的净值
- 使用当前市场价格的好处：简单公平、避免异常、符合直觉

### catfish/

鲶鱼（Catfish）模块，提供市场扰动机制。鲶鱼不参与 NEAT 进化，而是按照预设的规则进行交易。

**核心类：**

- `CatfishAccount` - 鲶鱼账户类（简化版账户）
  - `catfish_id: int` - 鲶鱼 ID（负数）
  - `initial_balance: float` - 初始余额
  - `balance: float` - 当前余额
  - `position: Position` - 持仓对象
  - `leverage: float` - 杠杆倍数
  - `maintenance_margin_rate: float` - 维持保证金率
  - 核心方法：`get_equity()`, `get_margin_ratio()`, `check_liquidation()`, `on_trade()`, `on_adl_trade()`, `reset()`

- `CatfishBase` - 鲶鱼抽象基类
  - `catfish_id: int` - 鲶鱼 ID（使用负数避免与 Agent 冲突）
  - `config: CatfishConfig` - 鲶鱼配置
  - `phase_offset: int` - 相位偏移（用于错开多个鲶鱼的触发时间）
  - `account: CatfishAccount` - 鲶鱼账户
  - `is_liquidated: bool` - 是否已被强平
  - 抽象方法：`decide(orderbook, tick, price_history) -> tuple[bool, int]` - 决策是否行动和方向
  - 核心方法：`execute()`, `can_act()`, `record_action()`, `reset()`

- `TrendFollowingCatfish` - 趋势追踪型鲶鱼
  - 根据历史价格变化率顺势下单
  - 配置：`lookback_period`, `trend_threshold`, `action_cooldown`

- `CycleSwingCatfish` - 周期摆动型鲶鱼
  - 按固定周期交替买卖
  - 配置：`half_cycle_length`, `action_interval`
  - 内部状态：`_current_direction` - 当前方向

- `MeanReversionCatfish` - 逆势操作型鲶鱼
  - 当价格偏离 EMA 均线时反向操作
  - 配置：`ma_period`, `deviation_threshold`, `action_cooldown`
  - 内部状态：`_ema`, `_ema_initialized`

**工厂函数：**
- `create_catfish(catfish_id, config, phase_offset=0, initial_balance=0.0, leverage=10.0, maintenance_margin_rate=0.05)` - 根据配置创建单个鲶鱼实例
- `create_all_catfish(config, initial_balance, leverage=10.0, maintenance_margin_rate=0.05)` - 创建所有三种鲶鱼实例（相位错开）

**鲶鱼特点：**
- 手续费为 0（maker 和 taker 费率均为 0）
- 拥有有限资金，可参与强平和 ADL 机制
- 任意鲶鱼被强平则 Episode 立即结束
- 下单量按盘口深度计算（目标：吃掉前 3 档）

## 依赖关系

### 外部依赖
- `numpy` - 数值计算
- `sortedcontainers.SortedDict` - 高性能排序字典（订单簿）
- `collections.OrderedDict` - 保持插入顺序的字典（价格档位）
- `dataclasses` - 数据类装饰器（NormalizedMarketState, ADLCandidate）

### 内部依赖
- `src.config.config` - 配置类（AgentConfig, MarketConfig, CatfishConfig, AgentType, CatfishMode）
- `src.core.log_engine.logger` - 日志系统

### 模块间依赖关系
```
NormalizedMarketState (market_state.py)
    ↓ 无依赖，独立数据类

OrderBook (orderbook/)
    ↓ 依赖
MatchingEngine (matching/)
    ↓ 依赖
Account/Position (account/)
    ↓ 依赖
ADLManager (adl/)

CatfishAccount/CatfishBase (catfish/)
    ↓ 依赖
Position (account/)
OrderBook (orderbook/)
MatchingEngine (matching/)
```
