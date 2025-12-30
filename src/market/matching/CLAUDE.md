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

#### `match_limit_order(order: Order) -> list[Trade]`
限价单撮合。价格优先、时间优先原则，未成交部分挂在订单簿上。

#### `match_market_order(order: Order) -> list[Trade]`
市价单撮合。吃对手盘直到完全成交或对手盘为空，未成交部分直接丢弃。

#### `process_order(order: Order) -> list[Trade]`
处理订单入口，根据订单类型调用对应撮合函数，并发布成交事件。

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

## 依赖关系

- `src.config.config` - 市场配置
- `src.core.event_engine` - 事件系统
- `src.core.log_engine` - 日志系统
- `src.market.orderbook` - 订单簿
