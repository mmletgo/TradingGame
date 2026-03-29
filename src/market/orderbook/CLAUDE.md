# Orderbook 模块 - 订单簿

## 模块概述

订单簿模块是交易市场的核心组件，负责维护买卖盘的价格档位和订单管理。使用 Cython + C++ 加速关键数据结构，实现全 C 级别的订单簿操作。本模块实现了一个完整的限价订单簿（Limit Order Book，LOB），支持价格优先、时间优先的撮合原则。

## 设计理念

1. **全 C++ 数据结构：** 使用 std::map 替代 SortedDict，侵入式双向链表替代 OrderedDict，unordered_map 替代 Python dict
2. **零 Python 对象开销：** 订单使用 COrder 结构体（malloc/free），避免 Python 对象创建
3. **手动引用计数：** PriceLevel 作为 cdef class 在 std::map 中以 PyObject* 存储，手动 Py_INCREF/Py_DECREF
4. **低延迟：** 关键操作均为 O(1) 或 O(log N) 时间复杂度
5. **浮点精度：** 价格归一化机制确保档位键的一致性
6. **缓存机制：** 深度数据缓存避免重复计算

## 文件结构

- `order.py` - 订单数据模型（Python，非热路径使用）
- `orderbook.pxd` - 订单簿 C 级别类型声明（供其他 .pyx 文件 cimport）
- `orderbook.pyx` - 订单簿实现（Cython + C++）
- `CLAUDE.md` - 本文档

## 核心数据结构

### COrder 结构体 (orderbook.pxd)

C 级别的订单结构体，作为侵入式双向链表节点：

```cython
cdef struct COrder:
    long long order_id
    long long agent_id
    int side          # 1=BUY, -1=SELL
    int order_type    # 1=LIMIT, 2=MARKET
    double price
    int quantity
    int filled_quantity
    COrder* prev      # 双向链表前驱
    COrder* next      # 双向链表后继
```

**内存管理：**
- 使用 `malloc` 分配，`free` 释放
- OrderBook 统一管理所有 COrder 的生命周期
- 在以下场景释放：cancel_order_fast、撮合中 maker 完全成交、clear()、__dealloc__

### OrderSide / OrderType / Order (order.py)

Python 级别的订单模型，仅用于非热路径（强平、ADL、演示等）。

### PriceLevel (orderbook.pyx)

价格档位类（Cython cdef class），使用侵入式双向链表维护同一价格的所有订单。

**Cython 属性（声明在 .pxd 中）：**
| 属性 | 类型 | 说明 |
|------|------|------|
| `price` | double | 档位价格 |
| `total_quantity` | long long | 档位总挂单量（未成交部分） |
| `head` | COrder* | 链表头（最早的订单） |
| `tail` | COrder* | 链表尾（最新的订单） |
| `order_lookup` | unordered_map[long long, COrder*] | O(1) 订单查找 |

**核心方法（cdef，C 级别）：**

#### `add_order(COrder* order)`
追加订单到链表尾部（FIFO），更新 order_lookup 和 total_quantity。O(1)。

#### `remove_order(long long order_id) -> COrder*`
从链表中摘除订单，返回 COrder*。调用者不应 free 返回的指针（OrderBook 统一管理）。O(1)。

#### `is_empty() -> bint`
判断链表是否为空（head == NULL）。O(1)。

### OrderBook (orderbook.pyx)

订单簿类（Cython cdef class），使用 C++ std::map 维护买卖盘价格档位的有序性。

**Cython 属性（声明在 .pxd 中）：**
| 属性 | 类型 | 说明 |
|------|------|------|
| `bids_map` | cppmap[double, PyObject*] | 买盘，std::map 升序排列（最大键 = 最优买价） |
| `asks_map` | cppmap[double, PyObject*] | 卖盘，std::map 升序排列（最小键 = 最优卖价） |
| `order_map_cpp` | unordered_map[long long, COrder*] | 全局订单索引 |
| `last_price` | double | 最新成交价 |
| `tick_size` | double | 最小变动单位 |
| `_depth_dirty` | bint | 深度缓存失效标志 |
| `_cached_depth` | object | 缓存的深度数据 |
| `_cached_levels` | int | 缓存的档位数 |

**热路径方法（cdef，C 级别，无 Python 对象开销）：**

#### `add_order_raw(order_id, agent_id, side, order_type, price, quantity) -> COrder*`
创建 COrder（malloc）并加入订单簿。O(log N)。

**执行流程：**
1. 价格归一化（round 到 tick_size 整数倍）
2. malloc 分配 COrder
3. 在 std::map 中查找或创建 PriceLevel（Py_INCREF）
4. 将 COrder 追加到 PriceLevel 链表尾部
5. 加入 order_map_cpp

#### `cancel_order_fast(order_id) -> bint`
快速撤单，从 order_map_cpp 查到 COrder，从 PriceLevel 移除，free COrder。O(log N)。

**执行流程：**
1. 从 order_map_cpp 查找 COrder
2. 从 PriceLevel 链表中摘除
3. 如果 PriceLevel 变空，从 std::map 中删除并 Py_DECREF
4. free COrder

#### `get_best_bid_price() -> double`
获取最优买价。使用 `--end()` 获取 std::map 最大键。O(1)。无买单返回 NAN。

#### `get_best_ask_price() -> double`
获取最优卖价。使用 `begin()` 获取 std::map 最小键。O(1)。无卖单返回 NAN。

#### `has_order(order_id) -> bint`
检查订单是否存在。使用 unordered_map.count()。O(1)。

**Python 兼容方法（供非热路径使用）：**

#### `add_order(order) -> None`
接受 Python Order 对象，内部调用 add_order_raw。

#### `cancel_order(order_id) -> True | None`
Python 兼容撤单。成功返回 True，失败返回 None。

#### `get_best_bid() -> float | None`
#### `get_best_ask() -> float | None`
#### `get_mid_price() -> float | None`
Python 级别的价格查询方法。

#### `get_depth(levels=100) -> dict`
获取盘口深度（Python dict 格式），带缓存机制。

#### `get_depth_numpy(levels=100) -> tuple[NDArray, NDArray]`
获取盘口深度（NumPy 格式），直接遍历 C++ map 迭代器填充数组。

#### `clear(reset_price=None) -> None`
清空订单簿：释放所有 COrder（free），释放所有 PriceLevel 引用（Py_DECREF），清空 C++ map。

#### `order_map` (property)
返回 Python dict 快照（order_id -> Order），供非热路径兼容使用。每次调用都重建，不缓存。

## 内存管理

### COrder 生命周期

| 场景 | 分配 | 释放 |
|------|------|------|
| 添加订单 | add_order_raw: malloc | - |
| 撤单 | - | cancel_order_fast: free |
| 撮合完全成交 | - | process_order_raw: free |
| 清空订单簿 | - | clear / __dealloc__: 遍历 order_map_cpp free |
| 僵尸订单清理 | - | process_order_raw: free |

### PriceLevel 引用计数

| 场景 | 操作 |
|------|------|
| 创建新档位 | Py_INCREF，存入 std::map |
| 档位变空 | Py_DECREF，从 std::map 删除 |
| 清空订单簿 | 遍历所有档位 Py_DECREF |

## 性能特征

### 时间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| add_order_raw | O(log N) | std::map 插入/查找 |
| cancel_order_fast | O(log N) | std::map 查找 + 链表 O(1) 摘除 |
| get_best_bid_price | O(1) | std::map --end() |
| get_best_ask_price | O(1) | std::map begin() |
| has_order | O(1) | unordered_map.count() |
| get_depth_numpy | O(levels) | 遍历 C++ map 迭代器 |
| clear | O(N + M) | 遍历释放所有资源 |

### 与旧实现对比

| 数据结构 | 旧实现 | 新实现 |
|----------|--------|--------|
| 价格档位排序 | Python SortedDict | C++ std::map |
| 同档位订单管理 | Python OrderedDict | 侵入式双向链表 + C++ unordered_map |
| 全局订单索引 | Python dict | C++ unordered_map |
| 订单对象 | Python Order 类 | C 结构体 COrder（malloc/free） |

## 依赖关系

### 外部依赖

- `numpy` - 数值计算（get_depth_numpy）
- `libcpp` - C++ STL 容器（map, unordered_map）
- `cpython.ref` - Python 引用计数管理（Py_INCREF, Py_DECREF）

### 内部依赖

- `src.market.orderbook.order` - Python 订单数据模型（非热路径）

### 被依赖关系

- `src.market.matching.fast_matching` - 撮合引擎（cimport OrderBook, PriceLevel, COrder）
- `src.training._cython.fast_tick_execution` - tick 执行模块（cimport OrderBook）
