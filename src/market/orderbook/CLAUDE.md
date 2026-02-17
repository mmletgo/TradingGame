# Orderbook 模块 - 订单簿

## 模块概述

订单簿模块是交易市场的核心组件，负责维护买卖盘的价格档位和订单管理。使用 Cython 加速关键数据结构，提供高性能的订单操作接口。本模块实现了一个完整的限价订单簿（Limit Order Book，LOB），支持价格优先、时间优先的撮合原则。

## 设计理念

1. **高性能：** 使用 Cython 编写核心数据结构，减少 Python 开销
2. **低延迟：** 关键操作均为 O(1) 或 O(levels) 时间复杂度
3. **内存优化：** 使用 `__slots__` 和 C 级别类型减少内存占用
4. **浮点精度：** 价格归一化机制确保档位键的一致性
5. **缓存机制：** 深度数据缓存避免重复计算

## 文件结构

- `order.py` - 订单数据模型（Python）
- `orderbook.pyx` - 订单簿实现（Cython）
- `CLAUDE.md` - 本文档

## 核心数据结构

### OrderSide (order.py)

订单方向枚举（IntEnum），使用整数类型便于计算：

```python
class OrderSide(IntEnum):
    BUY = 1    # 买单（做多）
    SELL = -1  # 卖单（做空）
```

**使用场景：**
- 订单方向判断
- 持仓方向计算（乘积）
- 撮合方向确定

### OrderType (order.py)

订单类型枚举（IntEnum）：

```python
class OrderType(IntEnum):
    LIMIT = 1   # 限价单，挂在订单簿上等待撮合
    MARKET = 2  # 市价单，立即吃对手盘，未成交部分丢弃
```

**使用场景：**
- 撮合引擎根据订单类型选择不同的撮合逻辑
- 限价单可能部分成交，市价单可能完全无法成交

### Order (order.py)

订单数据类，使用 `__slots__` 优化内存占用和访问速度。

**属性：**
| 属性 | 类型 | 说明 |
|------|------|------|
| `order_id` | int | 订单唯一标识，全局唯一 |
| `agent_id` | int | 所属 Agent ID |
| `side` | OrderSide | 买卖方向（BUY=1, SELL=-1） |
| `order_type` | OrderType | 订单类型（LIMIT=1, MARKET=2） |
| `price` | float | 委托价格 |
| `quantity` | int | 委托数量（整数，必须 > 0） |
| `filled_quantity` | int | 已成交数量（整数，初始为 0） |
| `timestamp` | float | 时间戳（训练模式下默认为 0.0） |

**设计要点：**
- 使用 `__slots__` 避免创建 `__dict__`，减少内存占用
- 所有数量字段使用整数类型，避免浮点累积误差
- `filled_quantity` 从 0 开始，撮合引擎负责更新

### PriceLevel (orderbook.pyx)

价格档位类（Cython cdef class），维护同一价格的所有订单，按 FIFO 顺序排列。

**Cython 属性：**
| 属性 | 类型 | 说明 |
|------|------|------|
| `price` | double | 档位价格（C 级别 double） |
| `orders` | OrderedDict[int, Order] | 订单映射（order_id -> Order），保持 FIFO 顺序 |
| `total_quantity` | long long | 64位整数，档位总数量 |

**核心方法：**

#### `add_order(order: Order) -> None`

向价格档位添加订单，O(1) 操作。

**执行流程：**
1. 将订单添加到 OrderedDict，键为 order_id
2. 订单数量累加到 total_quantity

**复杂度：** O(1)

#### `remove_order(order_id: int) -> Order | None`

从价格档位移除订单，O(1) 操作。

**执行流程：**
1. 检查订单是否存在于档位中
2. 使用 OrderedDict.pop() 删除订单，O(1) 操作
3. 计算剩余数量：`remaining = order.quantity - order.filled_quantity`
4. 从 total_quantity 中减去剩余数量（不是原始数量）
5. 返回被删除的订单

**关键要点：**
- 减去的是剩余数量，因为部分成交时 total_quantity 已经减去了成交数量
- 如果订单不存在，返回 None

**复杂度：** O(1)

#### `get_volume() -> int`

获取档位总挂单量。

**返回：** total_quantity 的值（整数）

**设计要点：**
- 使用 64 位整数（long long），避免大量订单累积时溢出
- 适用于高并发场景，同一档位可能有数千个订单

**设计要点：**
1. **OrderedDict 的选择：**
   - 普通 dict 在 Python 3.7+ 才保持插入顺序
   - OrderedDict 提供明确的 FIFO 语义和 O(1) 删除
   - 时间优先原则是撮合的核心规则

2. **64 位整数的必要性：**
   - 单个订单数量最大可达数百万
   - 同一档位可能有数千个订单
   - 32 位整数（最大 2^31-1 约 21亿）可能溢出

3. **部分成交的处理：**
   - 成交时撮合引擎会同步更新 total_quantity
   - 撤单时只需减去剩余数量
   - 避免重复减去已成交部分

### OrderBook (orderbook.pyx)

订单簿类（Cython cdef class），维护买卖盘口和全局订单索引。

**Cython 属性：**
| 属性 | 类型 | 说明 |
|------|------|------|
| `bids` | SortedDict[float, PriceLevel] | 买盘，升序排列（最大键在末尾） |
| `asks` | SortedDict[float, PriceLevel] | 卖盘，升序排列（最小键在开头） |
| `order_map` | dict[int, Order] | 全局订单索引（order_id -> Order） |
| `last_price` | double | 最新成交价 |
| `tick_size` | double | 最小变动单位（如 0.01） |
| `_depth_dirty` | bint | 深度缓存失效标志（C 级别 bool） |
| `_cached_depth` | object | 缓存的深度数据 |
| `_cached_levels` | int | 缓存的档位数 |

**核心方法：**

#### `add_order(order: Order) -> None`

向订单簿添加订单。

**执行流程：**
1. 根据订单方向选择买盘（bids）或卖盘（asks）
2. 价格归一化：
   ```python
   normalized_price = round(price / tick_size) * tick_size
   normalized_price = round(normalized_price, 10)
   ```
   - 第一轮：归一化到 tick_size 的整数倍
   - 第二轮：舍入到 10 位小数，消除乘法引入的微小误差
3. 更新订单价格：`order.price = normalized_price`
4. 检查价格档位是否存在，不存在则创建新的 PriceLevel
5. 将订单添加到价格档位
6. 添加到全局订单索引 order_map
7. 标记深度缓存失效：`_depth_dirty = True`

**关键优化：**
- 价格归一化避免浮点精度问题（如 91.30000000000001 -> 91.3）
- SortedDict 自动维护有序性，无需手动排序
- 档位按需创建，避免预分配浪费

**复杂度：** O(log N)，N 为档位数（SortedDict 插入）

**浮点精度问题示例：**
```python
# 问题：浮点运算产生精度误差
price = 91.3
tick_size = 0.01
result1 = price / tick_size * tick_size  # 可能是 91.30000000000001

# 解决：双重舍入
normalized = round(price / tick_size) * tick_size  # 91.30000000000001
normalized = round(normalized, 10)  # 91.3
```

#### `cancel_order(order_id: int) -> Order | None`

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
- 确保即使数据不一致也不会导致内存泄漏

**复杂度：** O(log N)，N 为档位数（SortedDict 查找）

#### `get_best_bid() -> float | None`

获取最优买价（买盘最高价）。

**实现：**
```python
if self.bids:
    return self.bids.peekitem(-1)[0]  # 获取最大键
return None
```

**关键优化：**
- 使用 `SortedDict.peekitem(-1)` 获取最大键，O(1) 操作
- 无需遍历或排序

**复杂度：** O(1)

#### `get_best_ask() -> float | None`

获取最优卖价（卖盘最低价）。

**实现：**
```python
if self.asks:
    return self.asks.peekitem(0)[0]  # 获取最小键
return None
```

**关键优化：**
- 使用 `SortedDict.peekitem(0)` 获取最小键，O(1) 操作

**复杂度：** O(1)

#### `get_mid_price() -> float | None`

获取中间价（最优买卖价的平均值）。

**实现：**
```python
best_bid = self.get_best_bid()
best_ask = self.get_best_ask()
if best_bid is not None and best_ask is not None:
    return (best_bid + best_ask) / 2.0
return None
```

**使用场景：**
- 计算市场合理价格
- 作为归一化的参考价格（NormalizedMarketState）

**复杂度：** O(1)

#### `get_depth(levels: int = 100) -> dict[str, list[tuple[float, float]]]`

获取盘口深度数据，Python 字典格式。

**返回格式：**
```python
{
    "bids": [(price, quantity), ...],  # 买盘，价格降序
    "asks": [(price, quantity), ...]   # 卖盘，价格升序
}
```

**执行流程：**
1. 检查缓存：
   ```python
   if not self._depth_dirty and self._cached_depth is not None and self._cached_levels == levels:
       return self._cached_depth
   ```
2. 买盘处理：
   ```python
   bid_keys = self.bids.keys()
   bid_count = len(bid_keys)
   start_idx = max(0, bid_count - levels)
   bid_prices = list(bid_keys[start_idx:])[::-1]  # 反转为降序
   bids = [(price, self.bids[price].get_volume()) for price in bid_prices]
   ```
3. 卖盘处理：
   ```python
   ask_keys = self.asks.keys()
   ask_prices = list(ask_keys[:levels])
   asks = [(price, self.asks[price].get_volume()) for price in ask_prices]
   ```
4. 构建结果并缓存
5. 标记缓存有效

**关键优化：**
- 利用 SortedDict 已排序特性，时间复杂度 O(levels)
- 避免排序操作，直接切片
- 缓存机制避免重复计算

**复杂度：** O(levels)

**缓存策略：**
- 缓存失效标志：`_depth_dirty`
- 缓存匹配条件：档位数相同
- 添加订单、撤销订单时标记失效

#### `get_depth_numpy(levels: int = 100) -> tuple[NDArray, NDArray]`

获取盘口深度数据，NumPy 格式。

**返回格式：**
```python
(bid_data, ask_data)  # 各 shape (levels, 2), dtype=np.float32
# 列0=价格，列1=数量
# 未填充的档位为 0
```

**执行流程：**
1. 创建两个 shape (levels, 2) 的零数组
2. 买盘：从后向前取价格键，填充降序排列的买盘数据
3. 卖盘：从前向后取价格键，填充升序排列的卖盘数据
4. 返回 NumPy 数组元组

**与 get_depth 的区别：**
- 直接返回 NumPy 数组，避免在 Python 层转换
- 不使用缓存机制（因为返回可变数组）
- 适用于需要直接进行数值计算的场景（如归一化）

**复杂度：** O(levels)

**使用场景：**
- NormalizedMarketState 直接使用 NumPy 数组
- 避免重复的类型转换

#### `clear(reset_price: float | None = None) -> None`

清空订单簿。

**执行流程：**
1. 重新初始化 bids 和 asks 为新的 SortedDict
2. 清空 order_map
3. 可选地重置 last_price
4. 清空深度缓存

**使用场景：**
- Episode 重置时清空订单簿
- 确保完全清空，避免内存泄漏

**设计要点：**
- 重新创建 SortedDict 而不是调用 clear()，确保彻底清空
- 缓存失效标记为 True

## Cython 优化说明

### 类型声明

使用 Cython 的 `cdef` 关键字声明 C 级别类型：

**cdef class：**
- `cdef class PriceLevel` - C 扩展类，减少 Python 开销
- `cdef class OrderBook` - C 扩展类

**C 级别属性：**
- `cdef public double price` - C 级别 double，避免 Python 浮点对象开销
- `cdef public long long total_quantity` - 64 位整数
- `cdef public bint _depth_dirty` - C 级别 bool
- `cdef public object orders` - Python 对象，保留灵活性

**局部变量类型声明：**
```cython
cdef double normalized_price = round(order.price / self.tick_size) * self.tick_size
cdef int i
```

### 性能优化点

1. **O(1) 订单查找：**
   - order_map 提供全局订单查找
   - 避免遍历档位和订单

2. **O(1) 档位删除：**
   - OrderedDict 提供 O(1) 订单删除
   - 同时保持 FIFO 顺序

3. **O(1) 最优价格：**
   - SortedDict.peekitem() 提供 O(1) 最优价格查询
   - 无需遍历

4. **O(levels) 深度查询：**
   - 利用 SortedDict 有序性，避免排序
   - 时间复杂度从 O(n log n) 降为 O(levels)

5. **缓存机制：**
   - 深度数据缓存，避免重复计算
   - 失效标志确保数据一致性

6. **价格归一化：**
   - 消除浮点精度问题
   - 确保档位键一致性

7. **C 级别类型：**
   - 减少 Python 对象开销
   - 提高数值计算速度

### 浮点精度处理

**问题来源：**
- 浮点运算可能产生精度误差（如 91.30000000000001）
- 导致档位键查找失败

**解决方案：**
```python
# 第一轮：归一化到 tick_size 的整数倍
normalized_price = round(price / tick_size) * tick_size
# 第二轮：舍入到 10 位小数，消除乘法引入的微小误差
normalized_price = round(normalized_price, 10)
```

**为什么需要两轮：**
1. 第一轮：确保价格是 tick_size 的整数倍
2. 第二轮：消除浮点乘法引入的精度误差

**示例：**
```python
tick_size = 0.01
price = 91.3

# 问题：
result1 = price / tick_size * tick_size  # 91.30000000000001

# 解决：
result2 = round(price / tick_size) * tick_size  # 91.30000000000001
result3 = round(result2, 10)  # 91.3
```

## 与撮合引擎的交互

### 接口契约

订单簿向撮合引擎提供以下接口：

1. **添加订单：** `add_order(order)`
   - 撮合引擎将新订单加入订单簿
   - 限价单未成交部分挂在订单簿上

2. **撤销订单：** `cancel_order(order_id)`
   - 撮合引擎处理撤单请求
   - Agent 主动撤单或强平时统一撤单

3. **获取最优价格：** `get_best_bid()`, `get_best_ask()`
   - 撮合引擎判断是否触发撮合
   - 市价单直接吃对手盘最优价格

4. **获取深度：** `get_depth(levels)` 或 `get_depth_numpy(levels)`
   - Agent 观察市场状态
   - 用于归一化市场数据

5. **清空订单簿：** `clear(reset_price)`
   - Episode 重置时清空
   - 确保新 Episode 开始时订单簿为空

### 数据流向

```
Agent -> 撮合引擎 -> 订单簿
         |
         v
      撮合逻辑（价格优先、时间优先）
         |
         v
      更新订单簿（添加/删除订单，更新成交数量）
         |
         v
      返回成交结果
```

### 撮合引擎使用流程

1. 接收 Agent 的订单请求
2. 检查订单簿最优价格，判断是否触发撮合
3. 执行撮合逻辑（价格优先、时间优先）
4. 更新订单簿（添加/删除订单，更新成交数量）
5. 返回成交结果

### 撮合规则

- **价格优先：** 最优价格优先成交
- **时间优先：** 同价格订单按时间顺序成交（FIFO）
- **限价单：** 未成交部分挂在订单簿上
- **市价单：** 吃对手盘直到完全成交或对手盘为空，未成交部分丢弃

## 数据结构图示

### OrderBook 结构图

```
OrderBook
|-- bids: SortedDict[float, PriceLevel]  # 买盘，升序排列
|   |-- 100.50: PriceLevel
|   |   |-- orders: OrderedDict{order_id -> Order}
|   |   |   |-- order_1: Order
|   |   |   |-- order_2: Order
|   |   |-- total_quantity: 2000
|   |-- 100.49: PriceLevel
|       |-- ...
|-- asks: SortedDict[float, PriceLevel]  # 卖盘，升序排列
|   |-- 100.51: PriceLevel
|   |-- ...
|   |-- 100.52: PriceLevel
|       |-- ...
|-- order_map: dict[int, Order]  # 全局订单索引
|   |-- order_1: Order
|   |-- order_2: Order
|-- last_price: double
```

### 价格档位示意图

```
买盘（降序显示）    卖盘（升序显示）
100.50 (2000)    |    100.51 (1500)
100.49 (1800)    |    100.52 (1200)
100.48 (2500)    |    100.53 (1800)
...              |    ...
```

## 约束与限制

1. **整数数量：**
   - 所有数量字段使用整数类型（int/long long）
   - 避免浮点累积误差
   - 确保撮合精度

2. **价格精度：**
   - 价格自动归一化到 tick_size 的整数倍
   - 使用双重舍入消除浮点误差

3. **FIFO 顺序：**
   - 同一价格档位的订单严格按照时间优先原则成交
   - 使用 OrderedDict 保证插入顺序

4. **无重复 ID：**
   - order_id 必须全局唯一
   - 否则会覆盖已有订单

5. **线程安全：**
   - 订单簿不是线程安全的
   - 多线程环境需要外部加锁

## 使用示例

### 基本使用

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

### 多档位示例

```python
# 创建订单簿
ob = OrderBook(tick_size=0.01)

# 添加多个买单
for i, price in enumerate([100.5, 100.49, 100.48]):
    order = Order(
        order_id=i+1,
        agent_id=100,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=price,
        quantity=1000
    )
    ob.add_order(order)

# 获取深度
depth = ob.get_depth(levels=5)
# {
#     "bids": [(100.5, 1000), (100.49, 1000), (100.48, 1000)],
#     "asks": []
# }

# 获取最优买价
best_bid = ob.get_best_bid()  # 100.5
```

### NumPy 格式示例

```python
# 获取 NumPy 格式的深度
bid_data, ask_data = ob.get_depth_numpy(levels=5)

# bid_data shape: (5, 2)
# 列0=价格，列1=数量
# [[100.5 , 1000.],
#  [100.49, 1000.],
#  [100.48, 1000.],
#  [0.   ,    0.],
#  [0.   ,    0.]]
```

## 性能特征

### 时间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| add_order | O(log N) | N 为档位数，SortedDict 插入 |
| cancel_order | O(log N) | N 为档位数，SortedDict 查找 |
| get_best_bid | O(1) | peekitem(-1) |
| get_best_ask | O(1) | peekitem(0) |
| get_mid_price | O(1) | 两次 O(1) 查询 |
| get_depth | O(levels) | 利用有序性，避免排序 |
| get_depth_numpy | O(levels) | 直接填充数组 |
| clear | O(1) | 重新创建对象 |

### 空间复杂度

- 订单簿：O(N + M)，N 为档位数，M 为订单数
- 深度缓存：O(levels)，固定大小
- 全局索引：O(M)，M 为订单数

### 性能优化建议

1. **批量操作：**
   - 尽量批量添加订单，减少缓存失效次数
   - 批量获取深度，减少重复计算

2. **合理设置档位数：**
   - 根据实际需求设置 levels 参数
   - 避免获取不必要的深度数据

3. **使用 NumPy 格式：**
   - 需要数值计算时使用 get_depth_numpy
   - 避免重复的类型转换

## 依赖关系

### 外部依赖

- `numpy` - 数值计算（用于 get_depth_numpy 返回 NumPy 数组）
- `sortedcontainers.SortedDict` - 高性能排序字典
- `collections.OrderedDict` - 保持插入顺序的字典
- `typing.TYPE_CHECKING` - 类型检查时的类型提示

### 内部依赖

- `src.market.orderbook.order` - 订单数据模型

### 被依赖关系

- `src.market.matching` - 撮合引擎
- `src.market.market_state` - 市场状态数据
- `src.bio.agents` - Agent 观察

## 常见问题

### Q1: 为什么使用 SortedDict 而不是普通 dict？

**A:** SortedDict 自动维护键的有序性，提供：
- O(1) 最优价格查询（peekitem）
- O(log N) 插入和删除
- 避免 O(n log n) 排序操作

### Q2: 为什么使用 OrderedDict 而不是普通 dict？

**A:** OrderedDict 提供：
- 明确的 FIFO 语义
- O(1) 删除操作（pop）
- 时间优先原则的保证

### Q3: 为什么使用 long long 而不是 int？

**A:** 64 位整数支持：
- 单个订单数量最大可达数百万
- 同一档位可能有数千个订单
- 避免 32 位整数溢出（最大 2^31-1 约 21亿）

### Q4: 价格归一化为什么需要两轮？

**A:** 两轮舍入：
1. 第一轮：归一化到 tick_size 的整数倍
2. 第二轮：消除浮点乘法引入的精度误差

### Q5: 为什么 get_depth_numpy 不使用缓存？

**A:** NumPy 数组是可变的：
- 缓存可能被外部修改
- 导致数据不一致
- 仅 get_depth 使用缓存
