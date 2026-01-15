# Orderbook 模块

## 模块概述

订单簿模块是交易市场的核心组件，负责维护买卖盘的价格档位和订单管理。使用 Cython 加速关键数据结构，提供高性能的订单操作接口。

## 文件结构

- `order.py` - 订单数据模型（Python）
- `orderbook.pyx` - 订单簿实现（Cython）

## 核心数据结构

### Order (order.py)

订单数据类，使用 `__slots__` 优化内存占用。

**属性：**
- `order_id: int` - 订单唯一标识
- `agent_id: int` - 所属 Agent ID
- `side: OrderSide` - 买卖方向（BUY=1, SELL=-1）
- `order_type: OrderType` - 订单类型（LIMIT=1, MARKET=2）
- `price: float` - 委托价格
- `quantity: int` - 委托数量（整数）
- `filled_quantity: int` - 已成交数量（整数，初始为 0）
- `timestamp: float` - 时间戳（训练模式下默认为 0）

### OrderSide (order.py)

订单方向枚举（IntEnum）：
- `BUY = 1` - 买单
- `SELL = -1` - 卖单

### OrderType (order.py)

订单类型枚举（IntEnum）：
- `LIMIT = 1` - 限价单
- `MARKET = 2` - 市价单

### PriceLevel (orderbook.pyx)

价格档位类（Cython cdef class），维护同一价格的所有订单，按 FIFO 顺序排列。

**Cython 属性：**
- `price: double` - 档位价格
- `orders: OrderedDict[int, Order]` - 订单映射（order_id -> Order），保持 FIFO 顺序
- `total_quantity: long long` - 64位整数，档位总数量，避免大量订单累积时溢出

**核心方法：**
- `add_order(order: Order) -> None` - 添加订单到档位，O(1) 操作
- `remove_order(order_id: int) -> Order | None` - 从档位移除订单，O(1) 操作
  - 减去订单剩余数量（quantity - filled_quantity）
  - 返回被移除的订单，不存在则返回 None
- `get_volume() -> int` - 获取档位总挂单量

**设计要点：**
1. 使用 OrderedDict 保证 O(1) 删除操作，同时保持 FIFO 顺序（时间优先原则）
2. 使用 64 位整数存储 total_quantity，支持大量订单累积而不溢出
3. 部分成交时，total_quantity 已经减去了成交数量，因此撤单时减去的是剩余数量

### OrderBook (orderbook.pyx)

订单簿类（Cython cdef class），维护买卖盘口和全局订单索引。

**Cython 属性：**
- `bids: SortedDict[float, PriceLevel]` - 买盘，升序排列
- `asks: SortedDict[float, PriceLevel]` - 卖盘，升序排列
- `order_map: dict[int, Order]` - 全局订单索引（order_id -> Order）
- `last_price: double` - 最新成交价
- `tick_size: double` - 最小变动单位
- `_depth_dirty: bint` - 深度缓存失效标志
- `_cached_depth: object` - 缓存的深度数据
- `_cached_levels: int` - 缓存的档位数

**核心方法：**

#### add_order(order: Order) -> None

向订单簿添加订单。

**执行流程：**
1. 根据订单方向选择买盘或卖盘
2. 价格归一化：`round(price / tick_size) * tick_size`，再次舍入到 10 位小数
   - 消除浮点精度误差（如 91.30000000000001 -> 91.3）
3. 检查价格档位是否存在，不存在则创建新的 PriceLevel
4. 将订单添加到价格档位
5. 添加到全局订单索引
6. 标记深度缓存失效

**关键优化：**
- 价格归一化避免浮点精度问题导致的档位键查找失败
- SortedDict 自动维护有序性，无需手动排序

#### cancel_order(order_id: int) -> Order | None

撤销订单。

**执行流程：**
1. 从 order_map 查找订单，不存在则返回 None
2. 根据订单方向选择买盘或卖盘
3. 使用与 add_order 相同的价格归一化逻辑
4. 从价格档位移除订单
5. 如果档位变空（订单数为 0），删除整个档位
6. 从 order_map 移除订单
7. 标记深度缓存失效
8. 返回被撤销的订单

**容错处理：**
- 如果价格档位不存在（数据不一致），仍从 order_map 移除并返回订单

#### get_best_bid() -> float | None

获取最优买价（买盘最高价）。

**实现：**
- 使用 `SortedDict.peekitem(-1)` 获取最大键，O(1) 操作
- 买盘为空时返回 None

#### get_best_ask() -> float | None

获取最优卖价（卖盘最低价）。

**实现：**
- 使用 `SortedDict.peekitem(0)` 获取最小键，O(1) 操作
- 卖盘为空时返回 None

#### get_mid_price() -> float | None

获取中间价（最优买卖价的平均值）。

**实现：**
- 分别获取最优买价和最优卖价
- 返回平均值，任一为空时返回 None

#### get_depth(levels: int = 100) -> dict[str, list[tuple[float, float]]]

获取盘口深度数据。

**返回格式：**
```python
{
    "bids": [(price, quantity), ...],  # 买盘，价格降序
    "asks": [(price, quantity), ...]   # 卖盘，价格升序
}
```

**执行流程：**
1. 检查缓存：如果缓存有效且档位数匹配，直接返回缓存
2. 买盘：从 SortedDict 升序键中取最后 levels 个并反转，得到降序
3. 卖盘：从 SortedDict 升序键中取前 levels 个
4. 构建结果并缓存
5. 标记缓存有效

**关键优化：**
- 利用 SortedDict 已排序特性，时间复杂度 O(levels)，远低于 O(n log n) 排序
- 缓存机制避免重复计算
- 使用切片而不是循环，提高性能

#### get_depth_numpy(levels: int = 100) -> tuple[NDArray, NDArray]

获取盘口深度数据（NumPy 格式）。

**返回格式：**
```python
(bid_data, ask_data)  # 各 shape (levels, 2)
# 列0=价格，列1=数量
# 未填充的档位为 0
```

**执行流程：**
1. 创建两个 shape (levels, 2) 的零数组，dtype=np.float32
2. 买盘：从后向前取价格键，填充降序排列的买盘数据
3. 卖盘：从前向后取价格键，填充升序排列的卖盘数据
4. 返回 NumPy 数组元组

**与 get_depth 的区别：**
- 直接返回 NumPy 数组，避免在 Python 层转换
- 不使用缓存机制（因为返回可变数组）
- 适用于需要直接进行数值计算的场景

#### clear(reset_price: float | None = None) -> None

清空订单簿。

**执行流程：**
1. 重新初始化 bids 和 asks 为新的 SortedDict
2. 清空 order_map
3. 可选地重置 last_price
4. 清空深度缓存

## Cython 优化说明

### 类型声明

使用 Cython 的 `cdef` 关键字声明 C 级别类型：
- `cdef class PriceLevel` - C 扩展类，减少 Python 开销
- `cdef public double price` - C 级别 double，避免 Python 浮点对象开销
- `cdef public long long total_quantity` - 64 位整数，支持大数量累积
- `cdef public object orders` - Python 对象，保留灵活性

### 性能优化点

1. **O(1) 订单查找：** order_map 提供 O(1) 全局订单查找
2. **O(1) 档位删除：** OrderedDict 提供 O(1) 订单删除，同时保持 FIFO
3. **O(1) 最优价格：** SortedDict.peekitem() 提供 O(1) 最优价格查询
4. **O(levels) 深度查询：** 利用 SortedDict 有序性，避免排序
5. **缓存机制：** 深度数据缓存，避免重复计算
6. **价格归一化：** 消除浮点精度问题，确保档位键一致性

### 浮点精度处理

**问题：** 浮点运算可能产生精度误差（如 91.30000000000001），导致档位键查找失败。

**解决方案：**
```python
normalized_price = round(price / tick_size) * tick_size
normalized_price = round(normalized_price, 10)
```

1. 第一轮：归一化到 tick_size 的整数倍
2. 第二轮：舍入到 10 位小数，消除乘法引入的微小误差

## 与撮合引擎的交互

### 接口契约

订单簿向撮合引擎提供以下接口：

1. **添加订单：** `add_order(order)` - 撮合引擎将新订单加入订单簿
2. **撤销订单：** `cancel_order(order_id)` - 撮合引擎处理撤单请求
3. **获取最优价格：** `get_best_bid()`, `get_best_ask()` - 撮合引擎判断是否触发撮合
4. **获取深度：** `get_depth(levels)` - Agent 观察市场状态
5. **清空订单簿：** `clear()` - Episode 重置时清空

### 数据流向

```
Agent -> 撮合引擎 -> 订单簿
         |
         v
      撮合逻辑
         |
         v
      更新订单簿
```

### 撮合引擎使用流程

1. 接收 Agent 的订单请求
2. 检查订单簿最优价格，判断是否触发撮合
3. 执行撮合逻辑（价格优先、时间优先）
4. 更新订单簿（添加/删除订单，更新成交数量）
5. 返回成交结果

## 约束与限制

1. **整数数量：** 所有数量字段使用整数类型（int/long long），避免浮点累积误差
2. **价格精度：** 价格自动归一化到 tick_size 的整数倍
3. **FIFO 顺序：** 同一价格档位的订单严格按照时间优先原则成交
4. **无重复 ID：** order_id 必须全局唯一，否则会覆盖已有订单

## 使用示例

```python
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.market.orderbook.orderbook import OrderBook

# 创建订单簿
ob = OrderBook(tick_size=0.01)

# 添加买单
order = Order(
    order_id=1,
    agent_id=100,
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    price=100.5,
    quantity=1000
)
ob.add_order(order)

# 获取最优买价
best_bid = ob.get_best_bid()  # 100.5

# 获取深度
depth = ob.get_depth(levels=5)
# {"bids": [(100.5, 1000)], "asks": []}

# 撤销订单
removed = ob.cancel_order(1)

# 清空订单簿
ob.clear()
```

## 依赖关系

- `numpy` - 数值计算（用于 get_depth_numpy 返回 NumPy 数组）
- `sortedcontainers.SortedDict` - 高性能排序字典
- `collections.OrderedDict` - 保持插入顺序的字典
- `typing.TYPE_CHECKING` - 类型检查时的类型提示
