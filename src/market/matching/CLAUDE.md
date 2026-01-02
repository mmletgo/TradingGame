# Matching 模块

## 模块概述

撮合引擎模块，负责订单撮合的核心逻辑，管理订单簿，处理订单提交、撤销和撮合。

## 文件结构

- `__init__.py` - 模块导出
- `matching_engine.py` - 撮合引擎类
- `trade.py` - 成交记录数据类

## 核心类

### MatchingEngine (matching_engine.py)

撮合引擎，交易市场的核心组件。

**属性：**
- `_event_bus: EventBus` - 事件总线
- `_config: MarketConfig` - 市场配置
- `_orderbook: OrderBook` - 订单簿实例
- `_next_trade_id: int` - 下一个成交ID
- `_fee_rates: dict[int, tuple[float, float]]` - Agent费率缓存

**初始化时订阅的事件：**
- `ORDER_PLACED` → `_handle_order_placed`：处理订单提交
- `ORDER_CANCELLED` → `_handle_order_cancelled`：处理订单撤销

**核心方法：**

#### `_handle_order_placed(event: Event) -> None`
处理订单提交事件，从事件中获取订单对象，调用 `process_order` 进行撮合。

#### `_handle_order_cancelled(event: Event) -> None`
处理订单撤销事件，从事件中获取订单ID，从订单簿中撤销该订单。

#### `register_agent(agent_id: int, maker_rate: float, taker_rate: float) -> None`
注册 Agent 的费率配置。由 Trainer 在以下时机调用：
- `setup()` 初始化后
- `evolve()` 进化后（新 Agent 需要注册）
- `load_checkpoint()` 加载检查点后

#### `calculate_fee(agent_id: int, amount: float, is_maker: bool) -> float`
计算手续费。未注册的 Agent 使用默认散户费率。

#### `_match_orders(order: Order, price_check: Callable | None) -> tuple[list[Trade], float]`
通用撮合逻辑（私有方法）。根据价格优先、时间优先原则与对手盘进行撮合。
- `price_check`: 价格检查函数，接收 `(order_price, best_price, order_side)`，返回是否可以继续撮合。为 `None` 时不进行价格检查（市价单行为）。
- 返回值: `(trades, remaining)` - 成交列表和剩余数量。

#### `match_limit_order(order: Order) -> list[Trade]`
限价单撮合。内部调用 `_match_orders` 并传入价格检查函数（买单价格 >= 卖一价，卖单价格 <= 买一价），未成交部分挂在订单簿上。

#### `match_market_order(order: Order) -> list[Trade]`
市价单撮合。内部调用 `_match_orders` 且不进行价格检查，吃对手盘直到完全成交或对手盘为空，未成交部分直接丢弃。

#### `process_order(order: Order) -> list[Trade]`
处理订单入口，根据订单类型调用对应撮合函数，并发布成交事件。

#### `process_order_direct(order: Order) -> list[Trade]`
直接处理订单（训练模式）。不发布成交事件，直接返回成交列表。用于训练模式绕过事件系统，减少开销。

#### `cancel_order_direct(order_id: int) -> bool`
直接撤单（训练模式）。不发布任何事件，直接从订单簿中撤销订单。

#### `orderbook` (property)
获取订单簿实例（供直接调用模式使用）。

### Trade (trade.py)

成交记录数据类。

**属性：**
- `trade_id: int` - 成交ID
- `price: float` - 成交价格
- `quantity: float` - 成交数量
- `buyer_id: int` - 买方Agent ID
- `seller_id: int` - 卖方Agent ID
- `buyer_fee: float` - 买方手续费
- `seller_fee: float` - 卖方手续费
- `is_buyer_taker: bool` - 买方是否为taker
- `timestamp: float` - 成交时间戳

## 事件流程

```
Agent.execute_action()
    ↓ 发布 ORDER_PLACED 事件
MatchingEngine._handle_order_placed()
    ↓ 调用 process_order()
MatchingEngine.match_limit_order() / match_market_order()
    ↓ 产生成交
MatchingEngine 发布 TRADE_EXECUTED 事件（定向发送给买卖双方）
    ↓
Agent._on_trade_event() 处理成交
```

## 费率配置

| Agent类型 | 挂单费率 | 吃单费率 |
|-----------|----------|----------|
| 散户 | 0.0002 (万2) | 0.0005 (万5) |
| 庄家 | 0.0 | 0.0001 (万1) |
| 做市商 | 0.0 | 0.0001 (万1) |

## 性能优化

### 直接调用模式（训练优化）
训练模式下绕过事件系统，直接调用撮合方法：
- `process_order_direct(order)` - 直接处理订单
- `cancel_order_direct(order_id)` - 直接撤单

### `_match_orders` 优化
- 预计算 taker 费率到循环外，避免重复查找
- 内联费率计算，避免 `calculate_fee()` 函数调用开销
- **部分成交时同步更新 `PriceLevel.total_quantity`**：确保 UI 显示的档位数量与实际剩余数量一致
- **防无限循环保护**：
  1. **最小数量阈值 `MIN_QTY=1e-9`**：用于检测浮点精度问题导致的微量订单，小于此阈值的数量视为 0
  2. **僵尸订单检测**：在遍历档位订单时，检测订单是否在 `order_map` 中，如果不在则直接从 `price_level.orders` 中移除
  3. **残单清理**：使用 `maker_remaining <= MIN_QTY` 检测残单并调用 `cancel_order` 移除
  4. **完全成交判断优化**：使用 `maker_remaining_after <= MIN_QTY` 判断订单是否完全成交，避免浮点精度问题产生残单
  5. **空档位清理**：在 for 循环结束后检查档位是否为空，如果为空则从 `side_book` 中删除
  6. **改进的进展检测**：只有当没有任何状态变化（无成交、无清理订单）时才计为无进展，连续 100 次无进展则退出撮合
  7. **最大迭代次数限制**：撮合循环最多迭代 100000 次，超过后强制终止并记录错误日志
  8. **价格档位不一致检测**：当 `get_best_ask/bid` 返回的价格不在对应盘口中时，终止撮合

### Trade 时间戳优化
训练模式使用固定时间戳 `0.0`，避免 `time.time()` 调用开销。

## 依赖关系

- `src.config.config` - 市场配置
- `src.core.event_engine` - 事件系统
- `src.core.log_engine` - 日志系统
- `src.market.orderbook` - 订单簿
