# Replay 模块 - 实盘回放环境

## 模块概述

实盘回放环境模块将真实交易所的历史订单簿快照和逐笔成交数据作为市场驱动源，让单个 AI Agent 在其中执行交易策略。与训练竞技场中多 Agent 互相撮合不同，回放环境中市场价格由真实数据驱动，Agent 的订单通过保守成交模型（排在真实队列之后）模拟成交。

## 文件结构

```
src/replay/
├── __init__.py               # 模块导出
├── config.py                 # ReplayConfig 配置数据类
├── data_loader.py            # HFtrade Monitor parquet 数据加载器
├── fill_model.py             # 保守成交模拟模型
├── market_state_builder.py   # NormalizedMarketState 构建器
├── replay_engine.py          # 核心回放引擎
├── replay_env.py             # Gymnasium 环境封装
└── CLAUDE.md                 # 本文档
```

## 核心组件

### 1. ReplayConfig (config.py)

回放环境的全部可配置参数。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| hftrade_data_dir | str | 必填 | HFtrade monitor/data 目录路径 |
| exchange | str | 必填 | 交易所名称 |
| pair | str | 必填 | 交易对名称 |
| date_start / date_end | str | 必填 | 日期范围 YYYY-MM-DD |
| tick_size | float | 必填 | 最小价格变动单位 |
| contract_size | float | 1.0 | 合约面值 |
| agent_type | str | "RETAIL_PRO" | "RETAIL_PRO" 或 "MARKET_MAKER" |
| initial_balance | float | 20000 | 初始资金 |
| leverage | float | 10.0 | 杠杆倍数 |
| maintenance_margin_rate | float | 0.05 | 维持保证金率 |
| maker_fee_rate | float | 0.0002 | 挂单费率 |
| taker_fee_rate | float | 0.0005 | 吃单费率 |
| episode_length | int | 10000 | 最大步数 (0=使用全部数据) |
| ob_depth | int | 5 | 订单簿深度 |
| trade_history_len | int | 100 | 成交历史缓冲区长度 |
| tick_history_len | int | 100 | tick 历史缓冲区长度 |
| position_cost_weight | float | 0.02 | 持仓成本惩罚权重 lambda |

### 2. DataLoader (data_loader.py)

加载 HFtrade Monitor 录制的 parquet 数据。

**数据结构:**
- `OrderbookSnapshot`: 5 档订单簿快照 (timestamp_ms, mid_price, bid/ask prices/amounts)
- `MarketTrade`: 逐笔成交 (timestamp_ms, price, amount, side)

**目录约定:**
- 订单簿: `{data_dir}/orderbooks/{exchange}/{pair}/{date}/*.parquet`
- 成交: `{data_dir}/trades/{exchange}/{pair}/{date}/*.parquet`

### 3. FillModel (fill_model.py)

保守成交模型: agent 排在真实队列之后。

- **被动成交** (`check_passive_fills`): 当真实成交穿越 agent 挂单价格时触发
  - agent 买单: 真实卖方成交价 <= agent 挂单价
  - agent 卖单: 真实买方成交价 >= agent 挂单价
- **主动成交** (`check_active_fill`): 市价单按对手方最优一档成交，量限于一档深度

### 4. MarketStateBuilder (market_state_builder.py)

从真实订单簿快照和成交数据构建 `NormalizedMarketState` 对象，使用与 TradingGame_origin 完全相同的归一化公式。

**内部缓冲区:**
- `_trade_prices` / `_trade_quantities`: 成交历史 (归一化后, deque maxlen=100)
- `_tick_prices`: tick 级 mid_price 历史 (deque maxlen=100)
- `_tick_volumes` / `_tick_amounts`: tick 级成交量/额 (原始值, deque maxlen=100)
- `_raw_tick_prices`: AS 模型用的原始价格 (deque maxlen=1000)
- `_base_price`: tick 历史价格归一化基准 (第一个 tick 的 mid_price)

**归一化公式 (与 TradingGame_origin 一致):**
| 数据类型 | 公式 |
|---------|------|
| 订单簿价格 | `(price - mid_price) / mid_price` |
| 订单簿数量 | `log10(quantity + 1) / 10` |
| 成交价格 | `(price - mid_price) / mid_price` |
| 成交数量 | `sign(qty) * log10(\|qty\| + 1) / 10` |
| tick 历史价格 | `(price - base_price) / base_price` |
| tick 历史成交量 | `sign(vol) * log10(\|vol\| + 1) / 10` |
| tick 历史成交额 | `sign(amt) * log10(\|amt\| + 1) / 12` |

**调用顺序:** `add_trades()` -> `update_tick()` -> `build()`

### 5. ReplayEngine (replay_engine.py)

核心回放引擎。在真实市场数据流上模拟单个 agent 的交易。

**生命周期:**
```
load_data(ob_snapshots, trades)
  -> reset(start_idx)
    -> step(action) -> StepResult
    -> step(action) -> StepResult
    -> ... -> done=True
```

**每步流程:**
1. 解析 agent 动作并执行 (挂单/市价单)
2. 推进到下一个订单簿快照
3. 处理期间的成交事件 (被动成交检查)
4. 构建新 NormalizedMarketState
5. 计算 reward = delta_equity/initial - lambda * |pos_value|/initial
6. 检查终止条件 (episode_length / 强平 / 数据耗尽)
7. 结束时强制平仓

**散户动作解析 (3 输出):**
复用 RetailProAgent.decide() 的逻辑:
- output[0]: 动作选择 [-1,1] -> 等宽分 6 bin (HOLD/BID/ASK/CANCEL/MKT_BUY/MKT_SELL)
- output[1]: 价格偏移 [-1,1] -> +/-100 ticks
- output[2]: 数量比例 [-1,1] -> [0, 1.0]

**做市商动作解析 (43 输出):**
复用 MarketMakerAgent.decide() 的逻辑 (简化版，不含 AS 模型):
- [0:10] 买单价格偏移, [10:20] 买单数量权重
- [20:30] 卖单价格偏移, [30:40] 卖单数量权重
- [40] 总下单比例, [41] gamma_adj, [42] spread_adj
- 报价中心使用 mid_price (而非 reservation_price)

**订单量计算:**
复用 Agent._calculate_order_quantity 逻辑:
- max_pos_value = equity * leverage
- 根据当前持仓方向计算可用空间
- quantity = available_pos_value * ratio / price

**StepResult 数据类:**
- `market_state: NormalizedMarketState`
- `reward: float`
- `done: bool`
- `info: dict` (equity, balance, position, step, mid_price)

### 6. ReplayEnv (replay_env.py)

标准 Gymnasium 环境封装，将 ReplayEngine 适配为 RL 训练接口。

**职责:**
- 加载市场数据并初始化 ReplayEngine
- 定义观测空间和动作空间
- 将 NormalizedMarketState 转换为 observation 向量（与原始 Agent.observe() 布局完全一致）
- 从 engine 的 Account 状态获取持仓/挂单信息填充到 observation 中

**空间定义:**

| Agent 类型 | 观测维度 | 动作维度 | 动作含义 |
|-----------|---------|---------|---------|
| RETAIL_PRO | 67 | 3 | 动作选择 + 价格偏移 + 数量比例 |
| MARKET_MAKER | 132 | 43 | 双边挂单参数 |

**Observation 向量结构:**
- 与 Agent.observe() 完全一致的布局（见 src/bio/agents/CLAUDE.md）
- 持仓信息（4 值）从 engine._account 实时获取
- 散户挂单信息（3 值）从 engine._pending_orders 获取
- 做市商挂单信息（60 值）从 engine._pending_orders 获取
- 做市商 AS 特征（8 值）简化填充（sigma/tau 从 state 获取，其余为 0）

**reset 逻辑:**
- 随机起始点: `np_random.integers(0, max_start)`，留出 episode_length + 200 warmup 的余量
- 调用 engine.reset(start_idx)

**step 逻辑:**
- 将动作传递给 engine.step(action)
- 返回 Gymnasium 标准 5 元组 (obs, reward, terminated, truncated=False, info)

## 依赖关系

### 内部依赖 (复用 TradingGame_origin 的市场引擎)
- `src.market.market_state.NormalizedMarketState` - 市场状态数据类
- `src.market.account.account.Account` - 账户类 (余额、持仓、保证金)
- `src.market.matching.trade.Trade` - 成交记录类
- `src.config.config.AgentConfig, AgentType` - 配置类

### 外部依赖
- `numpy` - 数值计算
- `pandas` - parquet 数据加载
- `pyarrow` - parquet 文件读取后端

## 设计决策

1. **保守成交模型**: agent 排在真实队列之后，只有真实成交穿越挂单价格才成交，避免过度乐观的回测偏差
2. **归一化公式复用**: 与 TradingGame_origin 完全一致，确保训练好的 agent 在回放环境中的输入分布相同
3. **做市商简化**: 回放环境中做市商不使用 AS 模型计算 reservation_price，以 mid_price 为报价中心
4. **强制平仓**: episode 结束时以 mid_price 强制清零持仓，确保 PnL 完整
