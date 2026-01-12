# Matching 模块

## 模块概述

撮合引擎模块，负责订单撮合的核心逻辑，管理订单簿，处理订单提交、撤销和撮合。

## 文件结构

- `matching_engine.py` - 撮合引擎类（调用 Cython 实现的核心撮合逻辑）
- `fast_matching.pyx` - Cython 加速的撮合核心逻辑
- `fast_matching.pyi` - 类型存根文件
- `trade.py` - 成交记录数据类

## 核心类

### MatchingEngine (matching_engine.py)

撮合引擎，交易市场的核心组件。

**属性：**
- `_config: MarketConfig` - 市场配置
- `_orderbook: OrderBook` - 订单簿实例
- `_next_trade_id: int` - 下一个成交ID
- `_fee_rates: dict[int, tuple[float, float]]` - Agent费率缓存

**核心方法：**

#### `register_agent(agent_id: int, maker_rate: float, taker_rate: float) -> None`
注册 Agent 的费率配置。由 Trainer 在以下时机调用：
- `setup()` 初始化后
- `evolve()` 进化后（新 Agent 需要注册）
- `load_checkpoint()` 加载检查点后

#### `calculate_fee(agent_id: int, amount: float, is_maker: bool) -> float`
计算手续费。根据 Agent 类型（maker/taker）和费率配置计算手续费金额。
- 未注册的 Agent 使用默认散户费率 (0.0002, 0.0005)
- 返回手续费金额（可能为负数，表示 maker rebate）

#### `_match_orders(order: Order, price_check: Callable | None) -> tuple[list[Trade], float]`
通用撮合逻辑（私有方法）。根据价格优先、时间优先原则与对手盘进行撮合。
- `price_check`: 价格检查函数，接收 `(order_price, best_price, order_side)`，返回是否可以继续撮合。为 `None` 时不进行价格检查（市价单行为）。
- 返回值: `(trades, remaining)` - 成交列表和剩余数量。

**撮合流程：**
1. 获取对手盘最优价格（买单取卖一价，卖单取买一价）
2. 价格检查（限价单）或跳过（市价单）
3. 遍历该价格档位的所有订单（按时间优先）
4. 对每个订单进行撮合，直到完全成交或对手盘为空
5. 成交价格 = maker 订单价格（对手盘价格）
6. 更新订单簿状态（包括 `total_quantity` 和 `last_price`）

**价格优先、时间优先原则：**
- **价格优先**：优先撮合最优价格的订单（买单优先卖一价，卖单优先买一价）
- **时间优先**：同一价格档位内，按订单提交时间先后顺序撮合（`order_map` 中按时间排序）

#### `match_limit_order(order: Order) -> list[Trade]`
限价单撮合。内部调用 `_match_orders` 并传入价格检查函数（买单价格 >= 卖一价，卖单价格 <= 买一价），未成交部分挂在订单簿上。

**价格检查规则：**
- 买单：`order_price >= best_price`（允许以更好价格成交）
- 卖单：`order_price <= best_price`（允许以更好价格成交）

**剩余数量处理：**
- 如果有剩余数量，更新订单的 `quantity` 为剩余值，重置 `filled_quantity` 为 0
- 将更新后的订单挂入订单簿

#### `match_market_order(order: Order) -> list[Trade]`
市价单撮合。内部调用 `_match_orders` 且不进行价格检查，吃对手盘直到完全成交或对手盘为空，未成交部分直接丢弃。

#### `process_order(order: Order) -> list[Trade]`
处理订单入口，根据订单类型调用对应撮合函数，返回成交列表。

#### `cancel_order(order_id: int) -> bool`
撤单。从订单簿中撤销订单，返回是否成功。

#### `orderbook` (property)
获取订单簿实例。

### Trade (trade.py)

成交记录数据类。使用 `__slots__` 优化内存占用，减少属性访问开销。

**属性：**
- `trade_id: int` - 成交ID
- `price: float` - 成交价格
- `quantity: int` - 成交数量（整数）
- `buyer_id: int` - 买方Agent ID
- `seller_id: int` - 卖方Agent ID
- `buyer_fee: float` - 买方手续费
- `seller_fee: float` - 卖方手续费
- `is_buyer_taker: bool` - 买方是否为taker
- `timestamp: float` - 成交时间戳（训练模式使用固定值 `0.0`）

### FastTrade (fast_matching.pyx)

Cython 实现的快速成交记录类，与 `Trade` 类保持相同的属性接口。

**cdef 属性：**
- `trade_id: int` - 成交ID
- `price: double` - 成交价格
- `quantity: int` - 成交数量
- `buyer_id: int` - 买方Agent ID
- `seller_id: int` - 卖方Agent ID
- `buyer_fee: double` - 买方手续费
- `seller_fee: double` - 卖方手续费
- `is_buyer_taker: bint` - 买方是否为taker
- `timestamp: double` - 成交时间戳（固定为 0.0）

### fast_match_orders (fast_matching.pyx)

Cython 实现的快速撮合核心函数。

**签名：**
```python
cpdef tuple fast_match_orders(
    object orderbook,
    object order,
    dict fee_rates,
    int next_trade_id,
    bint is_limit_order
) -> tuple[list[FastTrade], int, int]
```

**参数：**
- `orderbook`: OrderBook 实例
- `order`: 待撮合订单
- `fee_rates`: dict[int, tuple[float, float]] - agent_id -> (maker_rate, taker_rate)
- `next_trade_id`: 下一个成交 ID
- `is_limit_order`: 是否为限价单（True=需要价格检查，False=市价单不检查）

**返回值：**
- `(trades, remaining, next_trade_id)`: 成交列表、剩余数量、更新后的 trade_id

**优化点：**
- 使用 C 类型变量减少 Python 对象开销
- 内联价格检查逻辑（避免 callable 调用）
- 预计算 taker 费率到循环外

## 费率配置

| Agent类型 | 挂单费率 | 吃单费率 |
|-----------|----------|----------|
| 散户 | 0.0002 (万2) | 0.0005 (万5) |
| 高级散户 | 0.0002 (万2) | 0.0005 (万5) |
| 庄家 | -0.0001 (负万1, maker rebate) | 0.0001 (万1) |
| 做市商 | -0.0001 (负万1, maker rebate) | 0.0001 (万1) |

**注意：** 庄家和做市商的挂单费率为负数（maker rebate），表示其挂单成交时会获得交易额万分之一的手续费奖励。

**手续费计算规则：**
- 成交金额 = 成交价格 × 成交数量
- taker 手续费 = 成交金额 × taker 费率（taker 是主动发起成交的一方）
- maker 手续费 = 成交金额 × maker 费率（maker 是被撮合的挂单方）
- 买单时：taker = 买方，maker = 卖方
- 卖单时：taker = 卖方，maker = 买方

## 性能优化

### Cython 撮合引擎

核心撮合逻辑已迁移到 Cython 实现（`fast_matching.pyx`），主要优化：

1. **C 类型变量**：使用 `cdef` 声明 C 级别变量，减少 Python 对象开销
2. **内联价格检查**：将限价单/市价单的价格检查逻辑内联，避免 callable 调用
3. **FastTrade cdef class**：使用 Cython cdef class 替代 Python Trade 类
4. **预计算费率**：taker 费率在循环外预计算

**性能提升**：
- 串行执行时间从 ~99ms 降到 ~91ms，减少约 **8ms (8%)**
- 总 tick 时间从 ~165ms 降到 ~161ms

### `_match_orders` 优化
- 预计算 taker 费率到循环外，避免重复查找
- 内联费率计算，避免 `calculate_fee()` 函数调用开销
- **订单数量为整数**：所有订单数量（quantity）均为 int 类型，消除了浮点精度问题，简化了撮合逻辑

### 数据一致性修复
- **部分成交时同步更新 `PriceLevel.total_quantity`**：确保 UI 显示的档位数量与实际剩余数量一致
- **僵尸订单检测与清理**：在遍历档位订单时，检测订单是否在 `order_map` 中。如果订单在 `price_level.orders` 中但不在 `order_map` 中（数据不一致），则：
  - 直接从 `price_level.orders` 中移除该订单
  - 从 `total_quantity` 中减去该订单的未成交数量
- **空档位清理**：在处理完一个价格档位后，检查档位是否为空（`len(orders) == 0`），如果为空则从 `side_book` 中删除并标记深度为脏
- **价格档位不一致保护**：如果 `get_best_ask/bid()` 返回的价格不在 `side_book` 中，记录警告日志并终止撮合，避免无限循环

### Trade 时间戳优化
训练模式使用固定时间戳 `0.0`，避免 `time.time()` 调用开销。

## 依赖关系

- `src.config.config` - 市场配置
- `src.core.log_engine` - 日志系统
- `src.market.orderbook` - 订单簿
