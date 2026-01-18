# Arena 模块

## 模块概述

竞技场模块提供多竞技场并行推理训练功能。核心特性是将多个竞技场的神经网络推理合并成一个批量操作，交易配对和账户更新串行执行以保证正确性，同时支持通过 Worker 池将执行阶段并行化以提升性能。

## 设计理念

1. **神经网络共享**：所有竞技场使用同一套 `BatchNetworkCache`，进化后统一更新
2. **账户状态独立**：每个竞技场维护独立的 `ArenaState`，包含所有 Agent 的账户状态
3. **批量推理合并**：N 个竞技场 × M 个 Agent 的推理合并成单次 OpenMP 并行操作
4. **订单簿独立**：每个竞技场有独立的 `MatchingEngine` 和 `OrderBook`
5. **执行阶段并行化**：通过 `ArenaExecuteWorkerPool` 将执行阶段分布到多个 Worker 进程

## 文件结构

- `__init__.py` - 模块导出
- `arena_state.py` - 竞技场状态类（AgentAccountState、CatfishAccountState、ArenaState）
- `execute_worker.py` - Execute Worker 池（并行化 Execute 阶段）
- `fitness_aggregator.py` - 适应度汇总器
- `parallel_arena_trainer.py` - 多竞技场并行推理训练器
- `shared_memory_ipc.py` - 共享内存 IPC 机制（零拷贝优化）

## 核心类

### AgentAccountState (arena_state.py)

Agent 账户状态类，将 Agent 账户状态与 Agent 对象解耦，每个竞技场维护独立副本。

**数据结构：**
```python
@dataclass
class AgentAccountState:
    """Agent 账户状态（轻量级，约 200 bytes）"""
    agent_id: int
    agent_type: AgentType
    balance: float
    position_quantity: int
    position_avg_price: float
    realized_pnl: float
    leverage: float
    maintenance_margin_rate: float
    initial_balance: float
    pending_order_id: int | None
    maker_volume: int
    volatility_contribution: float
    is_liquidated: bool
    order_counter: int
    maker_fee_rate: float
    taker_fee_rate: float
    bid_order_ids: list[int]   # 做市商买单挂单 ID 列表
    ask_order_ids: list[int]   # 做市商卖单挂单 ID 列表
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `from_agent(agent)` | 类方法，从 Agent 对象创建状态副本 |
| `reset(config)` | 重置到初始状态 |
| `get_equity(current_price)` | 计算净值（余额 + 浮动盈亏） |
| `get_margin_ratio(current_price)` | 计算保证金率 |
| `check_liquidation(current_price)` | 检查是否需要强平 |
| `on_trade(trade_price, trade_quantity, is_buyer, fee, is_maker)` | 处理成交，返回已实现盈亏 |
| `generate_order_id(arena_id)` | 生成跨竞技场唯一的订单 ID |

### CatfishAccountState (arena_state.py)

鲶鱼账户状态类，包含鲶鱼特有的策略状态。

**数据结构：**
```python
@dataclass
class CatfishAccountState:
    """鲶鱼账户状态"""
    catfish_id: int
    catfish_mode: CatfishMode      # 鲶鱼类型
    balance: float
    position_quantity: int
    position_avg_price: float
    realized_pnl: float
    leverage: float
    maintenance_margin_rate: float
    initial_balance: float
    is_liquidated: bool
    order_counter: int
    current_direction: int         # 趋势创造者方向
    ema: float                     # 均值回归 EMA
    ema_initialized: bool
    ma_period: int                 # 均线周期
    deviation_threshold: float     # 偏离阈值
    action_probability: float      # 随机交易触发概率
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `from_catfish(catfish)` | 类方法，从 CatfishBase 对象创建状态副本 |
| `reset(initial_balance)` | 重置到初始状态，趋势创造者会重新随机选择方向 |
| `decide(tick, price_history)` | 决策是否行动和方向，根据鲶鱼类型执行不同逻辑 |
| `can_act()` | 检查随机概率判断是否可以行动 |
| `update_ema(price, ma_period)` | 更新 EMA 值（均值回归用） |

**鲶鱼类型决策逻辑：**

| 类型 | 决策逻辑 |
|------|---------|
| TREND_CREATOR | 固定 50% 行动概率，保持 Episode 开始时随机选择的方向 |
| MEAN_REVERSION | 价格偏离 EMA 超过阈值时反向操作，使用配置的行动概率 |
| RANDOM | 随机概率触发（使用配置的行动概率），方向也随机 |

### ArenaState (arena_state.py)

单个竞技场的独立状态，封装竞技场运行所需的所有状态。

**数据结构：**
```python
@dataclass
class ArenaState:
    """单个竞技场的独立状态"""
    arena_id: int
    matching_engine: MatchingEngine
    adl_manager: ADLManager
    agent_states: dict[int, AgentAccountState]
    catfish_states: dict[int, CatfishAccountState]
    recent_trades: deque
    price_history: deque[float]          # maxlen=1000，自动管理长度
    tick_history_prices: deque[float]     # maxlen=100
    tick_history_volumes: deque[float]    # maxlen=100
    tick_history_amounts: deque[float]    # maxlen=100
    smooth_mid_price: float
    tick: int
    pop_liquidated_counts: dict[AgentType, int]
    eliminating_agents: set[int]
    episode_high_price: float
    episode_low_price: float
    catfish_liquidated: bool
    end_reason: str | None               # Episode 结束原因
    end_tick: int                        # 结束时的 tick 数
    consecutive_one_sided_ticks: int     # 连续单边订单簿 tick 计数
```

**Episode 结束原因（end_reason）：**

| 值 | 含义 |
|---|------|
| `None` | Episode 正常运行完所有 tick |
| `"population_depleted:RETAIL"` | 散户种群存活少于初始的 1/4 |
| `"population_depleted:RETAIL_PRO"` | 高级散户种群存活少于初始的 1/4 |
| `"population_depleted:WHALE"` | 庄家种群存活少于初始的 1/4 |
| `"population_depleted:MARKET_MAKER"` | 做市商种群存活少于初始的 1/4 |
| `"one_sided_orderbook"` | 订单簿只有单边挂单 |
| `"catfish"` | 鲶鱼被强平 |

### FitnessAggregator (fitness_aggregator.py)

适应度汇总器，用于汇总多个竞技场、多个 episode 的适应度数据。

**方法：**
- `aggregate_simple_average(arena_fitnesses, episode_counts)` - 简单加权平均

**公式：**
```
avg_fitness = sum(arena_fitness) / total_episodes
```

### ParallelArenaTrainer (parallel_arena_trainer.py)

多竞技场并行推理训练器，核心特性是将多个竞技场的神经网络推理合并成一个批量操作。

**设计理念：**
1. **神经网络共享**：所有竞技场使用同一套 `BatchNetworkCache`，进化后统一更新
2. **账户状态独立**：每个竞技场维护独立的 `ArenaState`，包含所有 Agent 的账户状态
3. **批量推理合并**：N 个竞技场 × M 个 Agent 的推理合并成单次 OpenMP 并行操作
4. **订单簿独立**：每个竞技场有独立的 `MatchingEngine` 和 `OrderBook`

**核心流程：**
1. **初始化（setup 阶段）** - 执行顺序很重要：
   a. 先创建进化 Worker 池（在大内存分配之前 fork，避免 COW 内存泄漏）
   b. 创建共享种群（大量内存分配）
   c. 创建 N 个独立竞技场状态
   d. 初始化共享网络缓存
   e. 构建 Agent 映射表
   f. 创建 Execute Worker 池（可选，依赖竞技场状态）

2. **训练循环**：
   a. 重置所有竞技场（及 Execute Worker 池的订单簿）
   b. 同步推进所有竞技场的 tick（批量推理 + 并行/串行执行）
   c. 汇总适应度
   d. 执行 NEAT 进化
   e. 更新网络缓存和 Agent 状态

**类定义：**
```python
@dataclass
class MultiArenaConfig:
    """多竞技场配置"""
    num_arenas: int = 2
    episodes_per_arena: int = 50
    use_shared_memory_ipc: bool = True  # 默认启用共享内存 IPC（性能提升约 21%）

class ParallelArenaTrainer:
    """多竞技场并行推理训练器

    Attributes:
        config: 全局配置
        multi_config: 多竞技场配置
        populations: 共享的种群（神经网络）
        arena_states: N 个独立的竞技场状态
        network_caches: 共享的网络缓存
        evolution_worker_pool: 进化 Worker 池
        _execute_worker_pool: Execute Worker 池（可选，禁用鲶鱼时启用）
        _use_execute_workers: 是否使用 Execute Worker 池
        generation: 当前代数
        total_episodes: 总 episode 数
    """
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `setup()` | 初始化：先创建 Worker 池（避免 COW），再创建种群、竞技场状态、网络缓存、Execute Worker 池 |
| `run_round()` | 运行一轮训练（所有竞技场的所有 episode + 进化） |
| `run_tick_all_arenas()` | 并行执行所有竞技场的一个 tick（支持 Worker 池并行执行） |
| `_batch_inference_all_arenas()` | 批量推理所有竞技场的所有 Agent |
| `_run_episode_all_arenas()` | 运行所有竞技场的一个 episode |
| `_collect_fitness_all_arenas()` | 收集并汇总所有竞技场的适应度 |
| `_refresh_agent_states()` | 进化后刷新所有竞技场的 Agent 账户状态副本 |
| `_create_execute_worker_pool()` | 创建 Execute Worker 池 |
| `_collect_fee_rates()` | 收集所有 Agent 的费率 |
| `_prepare_mm_init_orders()` | 准备做市商初始化订单 |
| `_execute_with_worker_pool()` | 使用 Worker 池执行决策 |
| `_process_worker_results()` | 处理 Worker 返回的执行结果 |
| `train(num_rounds, checkpoint_callback, progress_callback)` | 主训练循环 |
| `save_checkpoint(path)` | 保存检查点 |
| `load_checkpoint(path)` | 加载检查点 |
| `stop()` | 停止训练并清理资源（包括关闭 Execute Worker 池） |
| `_calculate_market_avg_return()` | 计算单个竞技场的市场平均收益率 |
| `_balance_catfish_directions()` | 强制平衡趋势创造者鲶鱼的方向 |

**相对收益适应度：**

为避免市场整体方向偏移导致的正反馈循环，适应度计算使用相对收益而非绝对收益：

```python
# 计算每个竞技场的市场平均收益率
market_avg_return = mean([(equity - initial) / initial for all agents])

# Agent 适应度 = Agent 收益率 - 市场平均收益率
fitness = agent_return - market_avg_return
```

这样可以：
- 消除市场整体方向（上涨/下跌）的影响
- 鼓励 Agent 做出相对于市场的超额收益
- 即使市场整体下跌，表现好于平均的 Agent 仍能获得正向适应度

**鲶鱼方向平衡：**

趋势创造者鲶鱼的方向在所有竞技场间严格平衡：
- 每个 Episode 开始时，收集所有竞技场的趋势创造者鲶鱼
- 随机打乱后，前一半设为买方向（1），后一半设为卖方向（-1）
- 确保不同竞技场的行情有双向差异

配合 60% 的行动概率（`CatfishConfig.action_probability = 0.6`），鲶鱼能有效增加市场双向波动。

**账户完全独立设计：**

各竞技场之间只共享 Agent 的基因（神经网络），账户状态完全独立：
- **资金/余额**：每个竞技场维护独立的 `balance`
- **持仓**：每个竞技场维护独立的 `position_quantity` 和 `position_avg_price`
- **挂单 ID**：做市商的 `bid_order_ids` 和 `ask_order_ids` 各竞技场独立
- **行情**：每个竞技场有独立的订单簿和价格走势

订单数量和倾斜因子计算使用独立的辅助函数，直接基于 `AgentAccountState` 计算：
- `calculate_order_quantity_from_state(state, price, ratio, is_buy, ref_price)` - 基于 AgentAccountState 计算订单数量
- `calculate_skew_factor_from_state(state, mid_price)` - 基于 AgentAccountState 计算做市商仓位倾斜因子

**tick 执行流程（run_tick_all_arenas）：**
```python
# 阶段1: 准备（串行）
for arena in arena_states:
    handle_liquidations(arena)  # 强平检查
    catfish_action(arena)       # 鲶鱼行动（在 Agent 之前）
    market_state = compute_market_state(arena)
    active_agents = get_active_agents(arena)

# 阶段2: 批量推理（并行）- 一次性推理所有竞技场的所有 Agent
all_decisions = _batch_inference_all_arenas(market_states, active_agents)

# 阶段3: 执行
if _execute_worker_pool is not None:
    # 使用 Worker 池并行执行
    results = _execute_with_worker_pool(all_decisions)
    _process_worker_results(results)
else:
    # 串行执行
    for arena in arena_states:
        execute_trades(arena, all_decisions[arena.arena_id])
```

**Execute Worker 池集成：**

当 `_use_execute_workers=True` 时，Execute 阶段会使用 `ArenaExecuteWorkerPool` 并行化：

1. **setup() 阶段**：创建 Execute Worker 池
2. **_reset_all_arenas() 阶段**：重置 Worker 池中各竞技场的订单簿
3. **_init_market_all_arenas() 阶段**：使用 Worker 池执行做市商初始化挂单
4. **run_tick_all_arenas() 阶段3**：
   - 将决策数据转换为 `ArenaExecuteData` 格式
   - 调用 `_execute_worker_pool.execute_all()` 并行执行
   - 处理返回结果，更新主进程中的 `AgentAccountState`
5. **stop() 阶段**：关闭 Worker 池

**批量推理合并（_batch_inference_all_arenas）：**
- 使用 `AgentStateAdapter` 将 `AgentAccountState` 适配为 Agent-like 接口
- 按 AgentType 分组（跨所有竞技场）
- 使用 `_network_index_map` 获取每个 Agent 在其种群中的网络索引
- 调用 `BatchNetworkCache.decide_multi_arena()` 一次性处理所有竞技场的推理
- 将结果重组为 `dict[arena_id, list[decision]]`

**检查点格式：**

Version 2（精简格式，推荐）：
```python
{
    "checkpoint_version": 2,
    "generation": int,
    "populations": {
        AgentType.RETAIL: {
            "is_sub_population_manager": True,
            "sub_population_count": int,
            "sub_populations": [
                {
                    "generation": int,
                    "genome_data": (keys, fitnesses, metadata, nodes, conns),
                    "species_data": (genome_ids, species_ids),
                },
                ...
            ]
        },
        ...
    }
}
```

Version 1（旧格式，向后兼容）：
```python
{
    "generation": int,
    "populations": {
        AgentType.RETAIL: {
            "is_sub_population_manager": True,
            "sub_population_count": int,
            "sub_populations": [
                {"generation": int, "neat_pop": neat.Population},
                ...
            ]
        },
        ...
    }
}
```

### ArenaExecuteWorkerPool (execute_worker.py)

竞技场执行 Worker 池，将 Execute 阶段并行化。每个 Worker 维护若干个竞技场的 OrderBook 和 MatchingEngine。

**设计目标：**
- 将 Execute 阶段并行化
- 通过持久 Worker 池实现竞技场级别的并行执行

**架构：**
```
┌─────────────────────────────────────────────────────────────────┐
│                        主进程                                    │
│  BatchNetworkCache + AgentAccountState + 协调                    │
│                                                                 │
│  每个 tick:                                                      │
│  1. 从 Worker 接收：订单簿深度 + 上一 tick 成交结果              │
│  2. 更新 AgentAccountState                                      │
│  3. 强平检测 + 推理 + 生成决策                                  │
│  4. 发送给 Worker：决策 + 强平 Agent 列表                       │
└─────────────────────────────────────────────────────────────────┘
                              ↕ Queue
┌─────────────────────────────────────────────────────────────────┐
│                        Worker 池                                 │
│  每个 Worker 维护若干个竞技场的 OrderBook                        │
│                                                                 │
│  每个 tick:                                                      │
│  1. 接收：决策 + 强平 Agent 列表                                │
│  2. 执行强平（撤单 + 市价单）                                   │
│  3. 执行 Agent 决策                                             │
│  4. 返回：订单簿深度 + 成交列表                                 │
└─────────────────────────────────────────────────────────────────┘
```

**数据类定义：**

```python
@dataclass
class ExecuteCommand:
    """主进程 -> Worker 的命令"""
    cmd_type: str  # "reset", "init_mm", "execute", "get_depth", "shutdown"
    arena_id: int
    data: Any

@dataclass
class CatfishDecision:
    """吃单鲶鱼决策数据"""
    catfish_id: int        # 鲶鱼 ID（负数）
    direction: int         # 1=买, -1=卖
    quantity_ticks: int    # 吃多少档（默认1）

@dataclass
class ArenaExecuteData:
    """execute 命令的数据"""
    liquidated_agents: list[tuple[int, int, bool]]  # (agent_id, position_qty, is_mm)
    decisions: list[tuple[int, int, int, float, int]]  # (agent_id, action_int, side_int, price, quantity)
    mm_decisions: list[tuple[int, list, list]]  # (agent_id, bid_orders, ask_orders)
    catfish_decisions: list[CatfishDecision]  # 吃单鲶鱼决策列表

@dataclass
class ArenaExecuteResult:
    """Worker -> 主进程的结果"""
    arena_id: int
    bid_depth: np.ndarray  # shape (100, 2) - (price, quantity)
    ask_depth: np.ndarray  # shape (100, 2)
    last_price: float
    mid_price: float
    trades: list[tuple]  # (trade_id, price, qty, buyer_id, seller_id, buyer_fee, seller_fee, is_buyer_taker)
    pending_updates: dict[int, int | None]  # agent_id -> pending_order_id
    mm_order_updates: dict[int, tuple[list, list]]  # agent_id -> (bid_ids, ask_ids)
    catfish_results: list[CatfishTradeResult]  # 吃单鲶鱼成交结果列表
    error: str | None
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `start()` | 启动所有 Worker 进程 |
| `reset_all(initial_price, fee_rates, catfish_ids)` | 重置所有竞技场的订单簿 |
| `init_market_makers(mm_init_orders)` | 初始化做市商挂单 |
| `execute_all(arena_commands)` | 执行所有竞技场的决策（每个 tick 调用） |
| `get_all_depths()` | 获取所有竞技场的订单簿深度 |
| `shutdown()` | 关闭所有 Worker |

**原子动作执行机制：**

Worker 端的 `_handle_execute` 函数采用"原子动作随机打乱"执行策略：

**原子动作类型：**
```python
class AtomicActionType(IntEnum):
    CANCEL = 1       # 撤单
    LIMIT_BUY = 2    # 限价买单
    LIMIT_SELL = 3   # 限价卖单
    MARKET_BUY = 4   # 市价买单
    MARKET_SELL = 5  # 市价卖单
```

**执行流程：**

1. **强平处理（不参与打乱，优先执行）**
   - 撤销被强平 Agent 的所有挂单
   - 执行市价单平仓

2. **收集原子动作**
   - 吃单鲶鱼：转换为 MARKET_BUY/MARKET_SELL 动作
   - 做市商：拆分为 CANCEL（撤旧单）+ LIMIT_BUY/LIMIT_SELL（挂新单）动作
   - 非做市商：拆分为 CANCEL（如需撤旧单）+ 对应动作

3. **随机打乱并执行**
   - 使用 `random.shuffle()` 打乱所有原子动作
   - 逐个执行，调用 `_execute_atomic_action()` 函数

4. **构建返回结果**
   - 聚合鲶鱼成交结果到 `CatfishTradeResult`

### ArenaExecuteWorkerPoolShm (execute_worker.py)

共享内存版竞技场执行 Worker 池，使用 `SharedMemoryIPC` 替代 Queue 通信，减少序列化开销。

**设计目标：**
- 通过共享内存避免 Queue 的 pickle 序列化开销
- 使用无锁轮询实现低延迟通信
- 保持与 `ArenaExecuteWorkerPool` 相同的接口

**与 Queue 版的对比：**

| 特性 | ArenaExecuteWorkerPool (Queue) | ArenaExecuteWorkerPoolShm (Shm) |
|------|-------------------------------|--------------------------------|
| 通信方式 | Queue + pickle | SharedMemory + 轮询 |
| 序列化开销 | 高（每次通信都需要序列化） | 零（直接内存访问） |
| 同步机制 | Queue.put/get 阻塞 | 无锁轮询 + 状态标志 |
| 内存拷贝 | 每次通信需要拷贝 | 读取结果时需要 copy() |
| 适用场景 | 数据量小、通信不频繁 | 高频通信、大数据量 |
| 鲶鱼支持 | ✓ 完整支持 | ✓ 完整支持 |

**主要方法（与 ArenaExecuteWorkerPool 接口兼容）：**

| 方法 | 描述 |
|------|------|
| `start()` | 启动所有 Worker 进程，等待 ready_event |
| `reset_all(initial_price, fee_rates)` | 重置所有竞技场的订单簿 |
| `init_market_makers(mm_init_orders)` | 初始化做市商挂单 |
| `execute_all(arena_commands)` | 执行所有竞技场的决策 |
| `get_all_depths()` | 获取所有竞技场的订单簿深度 |
| `shutdown()` | 关闭所有 Worker 并清理共享内存 |

### SharedMemoryIPC (shared_memory_ipc.py)

共享内存 IPC 管理器，提供零拷贝的进程间通信机制。

**核心组件：**

- `CommandStatus`: 命令状态枚举（IDLE, PENDING, PROCESSING, DONE）
- `CommandType`: 命令类型枚举（RESET, INIT_MM, EXECUTE, GET_DEPTH, SHUTDOWN）
- `ArenaCommandView`: 单个竞技场命令区域的零拷贝视图
- `ArenaResultView`: 单个竞技场结果区域的零拷贝视图
- `SharedMemoryIPC`: 共享内存 IPC 管理器
- `ShmSynchronizer`: 共享内存同步器（无锁轮询）

**内存布局：**

```
Command Region (COMMAND_REGION_SIZE bytes per arena):
- Header (64 bytes): status, cmd_type, counts, padding
- liquidated_data: int64[MAX_LIQUIDATED × 3]
- decisions_data: float64[MAX_DECISIONS × 5]
- mm_agent_ids: int64[MAX_MM_AGENTS]
- mm_decisions_data: float64[MAX_MM_AGENTS × MAX_ORDERS_PER_MM × 2 × 2]
- catfish_decisions: int64[MAX_CATFISH × 3]

Result Region (RESULT_REGION_SIZE bytes per arena):
- Header (64 bytes): status, counts, padding
- bid_depth: float64[100, 2]
- ask_depth: float64[100, 2]
- trades: float64[MAX_TRADES × 8]
- pending_updates: int64[MAX_PENDING_UPDATES × 2]
- mm_order_ids: int64[MAX_MM_AGENTS × (1 + MAX_ORDERS_PER_MM × 2)]
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `get_command_view(arena_id)` | 获取指定竞技场的命令视图 |
| `get_result_view(arena_id)` | 获取指定竞技场的结果视图 |
| `close()` | 关闭并释放共享内存 |

## 使用示例

### 基本使用

```python
from src.training.arena import ParallelArenaTrainer, MultiArenaConfig

# 方式1：使用上下文管理器
multi_config = MultiArenaConfig(num_arenas=2, episodes_per_arena=50)
with ParallelArenaTrainer(config, multi_config) as trainer:
    trainer.train(
        num_rounds=100,
        checkpoint_callback=lambda gen: print(f"Gen {gen}"),
        progress_callback=lambda stats: print(stats)
    )

# 方式2：手动管理
trainer = ParallelArenaTrainer(config, multi_config)
trainer.setup()
try:
    for _ in range(100):
        stats = trainer.run_round()
        if trainer.generation % 10 == 0:
            trainer.save_checkpoint(f"checkpoints/gen_{trainer.generation}.pkl")
finally:
    trainer.stop()
```

### 使用 Execute Worker 池

```python
from src.training.arena import ArenaExecuteWorkerPool, ArenaExecuteData

# 创建 Worker 池
pool = ArenaExecuteWorkerPool(
    num_workers=4,
    arena_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    config=config,
)

# 使用上下文管理器
with pool:
    # 重置所有竞技场
    pool.reset_all(initial_price=100.0, fee_rates={...})

    # 初始化做市商
    results = pool.init_market_makers({
        0: [(mm_id, bid_orders, ask_orders), ...],
        ...
    })

    # 每个 tick 执行
    results = pool.execute_all({
        0: ArenaExecuteData(
            liquidated_agents=[...],
            decisions=[...],
            mm_decisions=[...],
        ),
        ...
    })
```

### 启动脚本

```bash
# 默认训练
python scripts/train_parallel_arena.py --rounds 100

# 自定义参数
python scripts/train_parallel_arena.py --num-arenas 8 --episodes-per-arena 5 --rounds 200

# 从检查点恢复
python scripts/train_parallel_arena.py --resume checkpoints/parallel_arena_gen_50.pkl
```

## 性能优化

### 批量推理合并
- N 个竞技场 × M 个 Agent 合并成单次 OpenMP 并行操作
- 减少 Python 调用开销

### 并行进化
- 使用 MultiPopulationWorkerPool 在多个进程中并行执行 NEAT 进化
- RETAIL: 10 个子种群
- RETAIL_PRO: 1 个种群
- WHALE: 1 个种群
- MARKET_MAKER: 4 个子种群

### Execute Worker 池并行化
- 将执行阶段分布到多个 Worker 进程
- 每个 Worker 维护独立的订单簿
- 原子动作随机打乱执行，避免流动性枯竭

### 共享内存 IPC
- 零拷贝进程间通信
- 避免 pickle 序列化开销
- 无锁轮询同步机制

### 内存管理
- 每轮训练后进行垃圾回收和 malloc_trim
- 进化后调用 `_cleanup_neat_history()` 清理 NEAT 历史数据
- `price_history` 限制最大长度为 1000
- 精简格式 checkpoint 减小文件体积
- AgentAccountState 复用机制

## 依赖关系

- `src.bio.agents` - Agent 类
- `src.config.config` - 配置类
- `src.core.log_engine` - 日志系统
- `src.market.adl` - ADL 管理器
- `src.market.catfish` - 鲶鱼模块
- `src.market.matching` - 撮合引擎
- `src.market.orderbook` - 订单簿
- `src.training.population` - 种群管理
- `src.training._cython.batch_decide_openmp` - OpenMP 批量决策
