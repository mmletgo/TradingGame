# Matching 模块

## 模块概述

撮合引擎模块是交易市场的核心组件，负责订单撮合的完整生命周期管理。模块实现了价格优先、时间优先的撮合规则，支持限价单和市价单两种订单类型，并针对不同 Agent 类型提供差异化的手续费策略。

模块包含 Python 和 Cython 两种实现：
- **Python 实现**（`matching_engine.py`）：提供清晰的业务逻辑，包含日志记录
- **Cython 实现**（`fast_matching.pyx`）：高性能撮合引擎，用于生产环境训练

## 文件结构

```
matching/
├── __init__.py              # 模块导出，尝试导入 Cython 加速版本
├── trade.py                 # 成交记录数据类（Python）
├── matching_engine.py       # 撮合引擎类（Python 实现）
├── fast_matching.pxd        # Cython 声明文件
└── fast_matching.pyx        # Cython 加速实现（FastTrade + FastMatchingEngine）
```

## 核心类

### Trade (trade.py)

成交记录数据类，记录订单撮合成功后的交易信息。

**使用 `__slots__` 优化内存占用**，避免动态属性字典的开销。

**属性：**
- `trade_id: int` - 成交ID，全局唯一递增
- `price: float` - 成交价格
- `quantity: int` - 成交数量（整数）
- `buyer_id: int` - 买方Agent ID
- `seller_id: int` - 卖方Agent ID
- `buyer_fee: float` - 买方手续费
- `seller_fee: float` - 卖方手续费
- `is_buyer_taker: bool` - 买方是否为taker（主动吃单方）
- `timestamp: float` - 成交时间戳（训练模式使用固定值 `0.0`）

**设计要点：**
- 使用整数类型的 `quantity` 避免浮点精度问题
- 训练模式下使用固定时间戳避免 `time.time()` 系统调用开销
- 区分 buyer/seller 的 taker/maker 角色，支持不对称手续费计算

### FastTrade (fast_matching.pyx)

Cython 实现的快速成交记录类，与 `Trade` 类保持完全相同的属性接口。

**cdef 属性：**
- `trade_id: int` - 成交ID
- `price: double` - 成交价格
- `quantity: int` - 成交数量
- `buyer_id: int` - 买方Agent ID
- `seller_id: int` - 卖方Agent ID
- `buyer_fee: double` - 买方手续费
- `seller_fee: double` - 卖方手续费
- `is_buyer_taker: bint` - 买方是否为taker
- `timestamp: double` - 固定为 `0.0`

**Cython 优化：**
- 使用 `cdef public` 声明属性，既提供 C 级别访问速度，又允许 Python 代码访问
- 使用 `double` 和 `bint` 类型减少 Python 对象开销
- 所有属性在 `.pxd` 文件中声明，供其他 Cython 模块使用

### MatchingEngine (matching_engine.py)

Python 实现的撮合引擎，提供清晰的业务逻辑和完整的日志记录。

**属性：**
- `_config: MarketConfig` - 市场配置
- `_orderbook: OrderBook` - 订单簿实例
- `_logger: Logger` - 日志器
- `_next_trade_id: int` - 下一个成交ID
- `_fee_rates: dict[int, tuple[float, float]]` - Agent费率缓存

**核心方法：**

#### `__init__(config: MarketConfig) -> None`
初始化撮合引擎，创建订单簿实例。

#### `register_agent(agent_id: int, maker_rate: float, taker_rate: float) -> None`
注册 Agent 的费率配置。

**调用时机：**
- Trainer 初始化后（`setup()`）
- 进化后创建新 Agent（`evolve()`）
- 加载检查点后（`load_checkpoint()`）

#### `calculate_fee(agent_id: int, amount: float, is_maker: bool) -> float`
计算手续费金额。

**费率查找逻辑：**
1. 优先使用 `agent_id` 对应的费率配置
2. 未注册的 Agent 使用默认散户费率 `(0.0002, 0.0005)`
3. 根据 `is_maker` 选择 maker 或 taker 费率
4. 返回 `amount × rate`（可能为负数，表示 maker rebate）

#### `_match_orders(order: Order, price_check: Callable | None) -> tuple[list[FastTrade], int]`
通用撮合逻辑（私有方法），委托给 Cython 的 `fast_match_orders` 实现。

**参数：**
- `order`: 待撮合订单
- `price_check`: 价格检查函数，接收 `(order_price, best_price, order_side)`，返回是否可以继续撮合。`None` 表示不检查（市价单）

**返回值：**
- `(trades, remaining, next_trade_id)`: 成交列表、剩余数量、更新后的 trade_id

#### `match_limit_order(order: Order) -> list[Trade]`
限价单撮合。

**价格检查规则：**
- 买单：`order_price >= best_price`（允许以更好价格成交）
- 卖单：`order_price <= best_price`（允许以更好价格成交）

**剩余数量处理：**
```python
if remaining > 0:
    order.quantity = remaining          # 更新为剩余数量
    order.filled_quantity = 0          # 重置已成交数量
    self._orderbook.add_order(order)   # 挂入订单簿
```

#### `match_market_order(order: Order) -> list[Trade]`
市价单撮合。

**特点：**
- 不进行价格检查，`price_check = None`
- 吃对手盘直到完全成交或对手盘为空
- 未成交部分直接丢弃，不挂入订单簿

#### `process_order(order: Order) -> list[Trade]`
处理订单入口，根据订单类型分派：
- `OrderType.LIMIT` -> `match_limit_order`
- `OrderType.MARKET` -> `match_market_order`

#### `cancel_order(order_id: int) -> bool`
撤单，委托给订单簿的 `cancel_order` 方法。

#### `orderbook` (property)
获取订单簿实例，用于访问深度数据。

### FastMatchingEngine (fast_matching.pyx)

完全 Cython 化的撮合引擎，与 `MatchingEngine` 保持相同接口，但核心路径无日志以避免性能开销。

**cdef 属性：**
- `orderbook: object` - OrderBook 实例
- `_next_trade_id: int` - 下一个成交ID
- `_fee_rates: dict` - Agent 费率映射 (agent_id -> (maker_rate, taker_rate))
- `_tick_size: double` - 最小变动单位

**cpdef 方法（可从 Python 调用）：**

#### `__init__(config: MarketConfig) -> None`
初始化撮合引擎，延迟导入 `OrderBook` 避免循环依赖。

#### `register_agent(agent_id: int, maker_rate: double, taker_rate: double) -> void`
注册/更新 Agent 的费率配置。

#### `calculate_fee(agent_id: int, amount: double, is_maker: bint) -> double`
计算手续费，未注册 Agent 使用默认散户费率 `(0.0002, 0.0005)`。

#### `match_limit_order(order) -> list`
限价单撮合，调用 `fast_match_orders(is_limit_order=True)`。

#### `match_market_order(order) -> list`
市价单撮合，调用 `fast_match_orders(is_limit_order=False)`。

#### `process_order(order) -> list`
处理订单入口，根据 `order.order_type` 分派：
- `1 (LIMIT)` -> `match_limit_order`
- `2 (MARKET)` -> `match_market_order`

#### `cancel_order(order_id: int) -> bint`
撤单，返回是否成功。

**Properties：**
- `tick_size: float` - 最小变动单位
- `next_trade_id: int` - 下一个成交ID
- `fee_rates: dict` - 费率配置

### fast_match_orders (fast_matching.pyx)

Cython 实现的快速撮合核心函数，是整个撮合引擎的性能关键路径。

**函数签名：**
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
- `is_limit_order`: 是否为限价单（True=需要价格检查，False=市价单）

**返回值：**
- `(trades, remaining, next_trade_id)`: 成交列表、剩余数量、更新后的 trade_id

**核心撮合流程：**

1. **初始化**
   ```python
   trades = []
   remaining = order.quantity - order.filled_quantity
   taker_rates = fee_rates.get(taker_agent_id, (0.0002, 0.0005))
   taker_rate = taker_rates[1]  # 预计算 taker 费率
   ```

2. **主撮合循环**
   ```python
   while remaining > 0:
       # 获取对手盘最优价格
       if is_buy:
           best_price = orderbook.get_best_ask()
           if is_limit_order and order_price < best_price:
               break  # 价格不满足，停止撮合
           side_book = orderbook.asks
       else:
           best_price = orderbook.get_best_bid()
           if is_limit_order and order_price > best_price:
               break  # 价格不满足，停止撮合
           side_book = orderbook.bids
   ```

3. **遍历价格档位（时间优先）**
   ```python
   price_level = side_book[best_price]
   for maker_order in list(price_level.orders.values()):
           # 僵尸订单检测与清理
           if maker_order.order_id not in orderbook.order_map:
               del price_level.orders[maker_order.order_id]
               price_level.total_quantity -= zombie_remaining
               continue

           # 计算成交数量
           maker_remaining = maker_order.quantity - maker_order.filled_quantity
           trade_qty = min(remaining, maker_remaining)

           # 成交价格 = maker 订单价格（对手盘价格）
           trade_price = maker_order.price
           trade_amount = trade_price * trade_qty

           # 计算手续费
           taker_fee = trade_amount * taker_rate
           maker_fee = trade_amount * maker_rates[0]

           # 创建成交记录
           trade = FastTrade(...)
           trades.append(trade)

           # 更新订单状态
           order.filled_quantity += trade_qty
           maker_order.filled_quantity += trade_qty
           remaining -= trade_qty
           price_level.total_quantity -= trade_qty

           # Maker 订单完全成交则移除
           if maker_remaining_after == 0:
               orderbook.cancel_order(maker_order.order_id)
   ```

4. **空档位清理**
   ```python
   if len(price_level.orders) == 0:
       del side_book[best_price]
       orderbook._depth_dirty = True
   ```

**Cython 优化点：**
- 使用 `cdef` 声明 C 类型变量，减少 Python 对象开销
- 预计算 taker 费率到循环外
- 内联价格检查逻辑，避免 callable 调用
- 使用 `bint` 代替 `bool` 提升性能
- 使用 `int` 代替 Python 的 `int`（无限制整数）

## 撮合规则

### 价格优先、时间优先原则

**价格优先**：
- 买单优先撮合卖一价（最低卖价）
- 卖单优先撮合买一价（最高买价）

**时间优先**：
- 同一价格档位内，按订单提交时间先后顺序撮合
- 使用 `OrderedDict` 保持 FIFO 顺序

### 限价单撮合规则

**价格检查：**
- 买单：`order_price >= best_ask`（允许以更好价格成交）
- 卖单：`order_price <= best_bid`（允许以更好价格成交）

**剩余数量处理：**
- 无法完全成交的剩余部分挂在订单簿上
- 重置 `filled_quantity = 0`，将 `quantity` 更新为剩余值

### 市价单撮合规则

**特点：**
- 不进行价格检查，直接吃对手盘
- 吃到完全成交或对手盘为空
- 未成交部分直接丢弃，不挂入订单簿

### 成交价格确定

**成交价格 = maker 订单价格（对手盘价格）**

示例：
- 买单以 100 吃单，卖一价为 99 → 成交价 99
- 卖单以 98 吃单，买一价为 99 → 成交价 99

## 手续费模型

### 费率配置

| Agent类型 | 挂单费率 | 吃单费率 | 说明 |
|-----------|----------|----------|------|
| 散户 | 0.0002 (万2) | 0.0005 (万5) | 标准费率 |
| 高级散户 | 0.0002 (万2) | 0.0005 (万5) | 标准费率 |
| 庄家 | -0.0001 (负万1) | 0.0001 (万1) | Maker rebate |
| 做市商 | -0.0001 (负万1) | 0.0001 (万1) | Maker rebate |
| 鲶鱼 | 0 | 0 | 免手续费 |

**Maker Rebate**：庄家和做市商的挂单费率为负数，表示其挂单成交时会获得交易额万分之一的手续费奖励，鼓励提供流动性。

### 手续费计算规则

**基本公式：**
```
成交金额 = 成交价格 × 成交数量
taker 手续费 = 成交金额 × taker 费率
maker 手续费 = 成交金额 × maker 费率
```

**角色确定：**
- **买单时**：taker = 买方（主动发起），maker = 卖方（挂单方）
- **卖单时**：taker = 卖方（主动发起），maker = 买方（挂单方）

**示例：**
- 散户 A 买单 100 @ 10000，吃掉做市商 B 的挂单
- 成交金额 = 100 × 10000 = 1,000,000
- 买方（散户 A）手续费 = 1,000,000 × 0.0005 = 500（taker）
- 卖方（做市商 B）手续费 = 1,000,000 × (-0.0001) = -100（maker，获得奖励）

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

### 数据一致性优化

#### 部分成交时同步更新 `PriceLevel.total_quantity`
确保撮合过程中订单簿的档位数量与实际剩余数量一致，避免 UI 显示错误。

#### 僵尸订单检测与清理
在遍历档位订单时，检测订单是否在 `order_map` 中：

**数据不一致场景**：
- 订单在 `price_level.orders` 中但不在 `orderbook.order_map` 中

**处理方式**：
```python
if maker_order.order_id not in orderbook.order_map:
    del price_level.orders[maker_order.order_id]
    zombie_remaining = maker_order.quantity - maker_order.filled_quantity
    price_level.total_quantity = max(0, price_level.total_quantity - zombie_remaining)
    continue
```

#### 空档位清理
在处理完一个价格档位后，检查档位是否为空（`len(orders) == 0`），如果为空则：
- 从 `side_book` 中删除该价格档位
- 设置 `orderbook._depth_dirty = True` 标记深度缓存失效

### Trade 时间戳优化

训练模式使用固定时间戳 `0.0`，避免 `time.time()` 系统调用开销。

```python
self.timestamp = timestamp if timestamp is not None else 0.0
```

### 整数数量优化

所有订单数量（`quantity`）均为 `int` 类型，消除浮点精度问题：

**优势：**
- 避免浮点累积误差
- 简化撮合逻辑（无需处理浮点比较）
- 提升计算性能
- 档位数量累加更精确

## 依赖关系

### 外部依赖
- `sortedcontainers.SortedDict` - 订单簿使用的高性能排序字典
- `collections.OrderedDict` - 价格档位保持 FIFO 顺序
- `typing` - 类型提示

### 内部依赖
- `src.config.config.MarketConfig` - 市场配置
- `src.core.log_engine.logger` - 日志系统（仅 Python 版本）
- `src.market.orderbook.OrderBook` - 订单簿
- `src.market.orderbook.Order` - 订单类
- `src.market.orderbook.OrderSide` - 订单方向枚举
- `src.market.orderbook.OrderType` - 订单类型枚举

### 模块导出

`__init__.py` 导出以下公共 API：
```python
from src.market.matching.matching_engine import MatchingEngine
from src.market.matching.trade import Trade
from src.market.matching.fast_matching import FastTrade, fast_match_orders

try:
    from src.market.matching.fast_matching import FastMatchingEngine
except ImportError:
    FastMatchingEngine = None  # Cython 未编译时降级

__all__ = [
    "MatchingEngine",
    "FastMatchingEngine",
    "Trade",
    "FastTrade",
    "fast_match_orders",
]
```

**降级策略**：如果 Cython 模块未编译，`FastMatchingEngine` 为 `None`，允许代码继续运行（性能较低）。
