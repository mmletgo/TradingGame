# Matching 模块 - 撮合引擎

## 模块概述

撮合引擎模块是交易市场的核心组件，负责订单撮合的完整生命周期管理。模块实现了价格优先、时间优先的撮合规则，支持限价单和市价单两种订单类型，并针对不同 Agent 类型提供差异化的手续费策略。

核心撮合逻辑使用 Cython + C++ 实现（`FastMatchingEngine`），直接操作 C++ 数据结构和 COrder 结构体，避免 Python 对象创建开销。`MatchingEngine` 作为兼容层委托给 `FastMatchingEngine`。

## 文件结构

```
matching/
├── __init__.py              # 模块导出（MatchingEngine = FastMatchingEngine 别名）
├── trade.py                 # 成交记录数据类（Trade，仅非热路径使用）
├── matching_engine.py       # 撮合引擎兼容层（Python，内部委托 FastMatchingEngine）
├── fast_matching.pxd        # Cython + C++ 声明文件
└── fast_matching.pyx        # 撮合引擎核心实现（Cython + C++）
```

## 核心类

### FastTrade (fast_matching.pyx)

Cython 实现的快速成交记录类。

**cdef 属性（声明于 fast_matching.pxd）：**
| 属性 | 类型 | 说明 |
|------|------|------|
| `trade_id` | int | 成交ID |
| `price` | double | 成交价格 |
| `quantity` | int | 成交数量 |
| `buyer_id` | long long | 买方Agent ID（64位） |
| `seller_id` | long long | 卖方Agent ID（64位） |
| `buyer_fee` | double | 买方手续费 |
| `seller_fee` | double | 卖方手续费 |
| `is_buyer_taker` | bint | 买方是否为taker |
| `timestamp` | double | 固定为 `0.0` |

### FastMatchingEngine (fast_matching.pyx)

Cython + C++ 实现的撮合引擎，是整个系统的主撮合引擎。

**cdef 属性（声明于 fast_matching.pxd）：**
| 属性 | 类型 | 说明 |
|------|------|------|
| `_orderbook` | object (OrderBook) | 订单簿实例 |
| `_next_trade_id` | int | 下一个成交ID |
| `_fee_rates` | dict | Agent 费率映射 (agent_id -> (maker_rate, taker_rate)) |
| `_tick_size` | double | 最小变动单位 |

**核心方法：**

#### `process_order_raw(order_id, agent_id, side, order_type, price, quantity) -> list`

**热路径核心方法**。不创建 Python Order 对象，直接操作 C++ 数据结构进行撮合。

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `order_id` | long long | 订单 ID |
| `agent_id` | long long | Agent ID |
| `side` | int | 1=BUY, -1=SELL |
| `order_type` | int | 1=LIMIT, 2=MARKET |
| `price` | double | 价格（市价单传 0.0） |
| `quantity` | int | 数量 |

**核心撮合流程：**

1. **价格归一化** - round 到 tick_size 整数倍
2. **主撮合循环** - 遍历对手盘 C++ std::map
   - 买单：从 asks_map.begin()（最低卖价）开始
   - 卖单：从 bids_map.end()-1（最高买价）开始
3. **链表遍历** - 遍历 PriceLevel 的侵入式双向链表（COrder）
   - 僵尸订单检测（不在 order_map_cpp 中的 COrder）
   - 计算成交数量、手续费
   - 创建 FastTrade 成交记录
   - 更新 COrder.filled_quantity、PriceLevel.total_quantity
   - 完全成交的 maker：从 order_map_cpp 移除，从链表移除，free
4. **空档位清理** - PriceLevel 为空时 Py_DECREF 并从 std::map 中删除
5. **剩余挂单** - 限价单剩余部分通过 add_order_raw 挂入订单簿

#### `process_order(order) -> list`

Python 兼容方法。接受 Python Order 对象，内部委托给 process_order_raw。

#### `cancel_order(order_id) -> bint`

撤单。直接调用 OrderBook.cancel_order_fast()。

#### `register_agent(agent_id, maker_rate, taker_rate) -> void`

注册 Agent 费率配置。

#### `calculate_fee(agent_id, amount, is_maker) -> double`

计算手续费。

**Properties：**
- `orderbook` - 获取订单簿实例（兼容旧接口）
- `tick_size` - 最小变动单位
- `next_trade_id` - 下一个成交ID
- `fee_rates` - 费率配置

### MatchingEngine (matching_engine.py)

Python 兼容层，内部包装 FastMatchingEngine。保留 `_config`、`_logger` 等属性供非热路径代码使用。

**属性映射：**
- `_orderbook` -> `FastMatchingEngine._orderbook`（property）
- `_next_trade_id` -> `FastMatchingEngine._next_trade_id`（property）
- `_fee_rates` -> `FastMatchingEngine._fee_rates`（property）
- `_config` -> 直接存储

### Trade (trade.py)

Python 成交记录类，保留供非热路径使用。结构与 FastTrade 相同。

## 撮合规则

### 价格优先、时间优先原则

**价格优先**：
- 买单优先撮合卖一价（最低卖价）
- 卖单优先撮合买一价（最高买价）

**时间优先**：
- 同一价格档位内，按订单提交时间先后顺序撮合
- 使用侵入式双向链表保持 FIFO 顺序

### 限价单撮合规则

**价格检查：**
- 买单：`normalized_price >= best_ask_price`
- 卖单：`normalized_price <= best_bid_price`

**剩余数量处理：**
- 通过 add_order_raw 挂入订单簿（C 级别，无 Python 对象）

### 市价单撮合规则

- 不进行价格检查
- 吃对手盘直到完全成交或对手盘为空
- 未成交部分直接丢弃

### 成交价格确定

**成交价格 = maker 订单价格（对手盘价格）**

## 手续费模型

| Agent类型 | 挂单费率 | 吃单费率 | 说明 |
|-----------|----------|----------|------|
| 散户/高级散户 | 0.0002 (万2) | 0.0005 (万5) | 标准费率 |
| 做市商 | -0.0001 (负万1) | 0.0001 (万1) | Maker rebate |
| 噪声交易者 | 0 | 0 | 免手续费 |

## 模块导出

`__init__.py` 导出：

```python
from src.market.matching.fast_matching import FastTrade, FastMatchingEngine
MatchingEngine = FastMatchingEngine  # 别名，保持向后兼容
```

所有 `from src.market.matching import MatchingEngine` 或 `from src.market.matching.matching_engine import MatchingEngine` 的代码无需修改即可工作。

## 依赖关系

### 外部依赖
- `libcpp` - C++ STL 容器（map, unordered_map）
- `cpython.ref` - Python 引用计数管理

### 内部依赖
- `src.market.orderbook.orderbook` - cimport OrderBook, PriceLevel, COrder
- `src.config.config.MarketConfig` - 市场配置

### 被依赖关系
- `src.training._cython.fast_tick_execution` - cimport FastMatchingEngine
- `src.training.trainer` - 训练引擎
- `src.training.arena` - 竞技场
- `src.bio.agents` - Agent 订单提交
