# Arena 模块

## 模块概述

竞技场模块提供多竞技场并行推理训练功能。核心特性是将多个竞技场的神经网络推理合并成一个批量操作，交易配对和账户更新串行执行以保证正确性，同时支持通过 Worker 池将执行阶段并行化以提升性能。

## 设计理念

1. **神经网络共享**：所有竞技场使用同一套 `BatchNetworkCache`，进化后统一更新
2. **账户状态独立**：每个竞技场维护独立的 `ArenaState`，包含所有 Agent 的账户状态
3. **批量推理合并**：N 个竞技场 x M 个 Agent 的推理合并成单次 OpenMP 并行操作
4. **订单簿独立**：每个竞技场有独立的 `MatchingEngine` 和 `OrderBook`
5. **执行阶段并行化**：通过 `ArenaExecuteWorkerPool` 将执行阶段分布到多个 Worker 进程
6. **噪声交易者**：每个竞技场包含独立的噪声交易者，提供市场背景噪音

## 文件结构

- `__init__.py` - 模块导出
- `arena_state.py` - 竞技场状态类（AgentAccountState、NoiseTraderAccountState、ArenaState）
- `arena_worker.py` - Arena Worker 进程池架构（Worker 独立运行完整 tick 循环，消除 tick 级 IPC）
- `execute_worker.py` - Execute Worker 池（并行化 Execute 阶段，tick 级 IPC 模式）
- `fitness_aggregator.py` - 适应度汇总器
- `parallel_arena_trainer.py` - 多竞技场并行推理训练器
- `shared_memory_ipc.py` - 共享内存 IPC 机制（零拷贝优化）

## 核心类

### AgentAccountState (arena_state.py)

Agent 账户状态类，将 Agent 账户状态与 Agent 对象解耦，每个竞技场维护独立副本。

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

### NoiseTraderAccountState (arena_state.py)

噪声交易者账户状态类，轻量级，不参与进化。每个 tick 以一定概率下市价单，产生近似布朗运动的价格随机游走。

**数据结构：**
```python
@dataclass
class NoiseTraderAccountState:
    trader_id: int              # 负数 ID
    balance: float              # 无限资金（不会被强平）
    position_quantity: int
    position_avg_price: float
    order_counter: int
    config_quantity_mu: float   # 对数正态分布参数
    config_quantity_sigma: float
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `from_noise_trader(noise_trader)` | 类方法，从 NoiseTrader 对象创建状态副本 |
| `reset()` | 重置到初始状态 |
| `get_equity(current_price)` | 计算净值 |
| `on_trade(price, quantity, is_buyer)` | 处理成交（无手续费） |
| `generate_order_id(arena_id)` | 生成唯一订单 ID |
| `decide(action_probability)` | 决策：返回 (should_act, direction, quantity) |

### ArenaState (arena_state.py)

单个竞技场的独立状态，封装竞技场运行所需的所有状态。

**数据结构：**
```python
@dataclass
class ArenaState:
    arena_id: int
    matching_engine: MatchingEngine
    adl_manager: ADLManager
    agent_states: dict[int, AgentAccountState]
    noise_trader_states: dict[int, NoiseTraderAccountState]
    recent_trades: deque
    price_history: deque[float]          # maxlen=1000
    tick_history_prices: deque[float]     # maxlen=100
    tick_history_volumes: deque[float]    # maxlen=100
    tick_history_amounts: deque[float]    # maxlen=100
    smooth_mid_price: float
    tick: int
    pop_liquidated_counts: dict[AgentType, int]
    eliminating_agents: set[int]
    episode_high_price: float
    episode_low_price: float
    end_reason: str | None
    end_tick: int
    # 扁平化数组（用于向量化强平检查）
    _balances: np.ndarray | None
    _position_quantities: np.ndarray | None
    ...
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `init_flat_arrays()` | 从 agent_states 初始化扁平化数组 |
| `sync_state_to_array(agent_id)` | 同步指定 agent 的状态到数组 |
| `reset_episode(initial_price)` | 重置 Episode 状态 |
| `get_agent_state(agent_id)` | 获取 Agent 状态 |
| `get_noise_trader_state(trader_id)` | 获取噪声交易者状态 |
| `mark_agent_liquidated(agent_id, agent_type)` | 标记 Agent 已被强平 |

**Episode 结束原因（end_reason）：**

| 值 | 含义 |
|---|------|
| `None` | Episode 正常运行完所有 tick |
| `"population_depleted:RETAIL_PRO"` | 高级散户种群存活少于初始的 1/4 |
| `"population_depleted:MARKET_MAKER"` | 做市商种群存活少于初始的 1/4 |
| `"one_sided_orderbook"` | 订单簿只有单边挂单 |

### ParallelArenaTrainer (parallel_arena_trainer.py)

多竞技场并行推理训练器。

**种群配置（2 种类型）：**
- RETAIL_PRO: 12 个子种群（SubPopulationManager）
- MARKET_MAKER: 4 个子种群（SubPopulationManager）

**噪声交易者：**
- 每个竞技场独立创建 N 个噪声交易者（由 `NoiseTraderConfig.count` 配置，默认 100）
- 噪声交易者 ID 为负数：-1, -2, -3, ...
- 每个 tick，噪声交易者以 `action_probability` 概率行动
- 行动时随机买/卖，下单量从对数正态分布采样
- 手续费为 0，不参与强平和进化

**主要方法：**

| 方法 | 描述 |
|------|------|
| `setup()` | 初始化：先创建 Worker 池（避免 COW），再创建种群、竞技场状态、网络缓存、Execute Worker 池 |
| `run_round()` | 运行一轮训练（所有竞技场的所有 episode + 进化） |
| `run_tick_all_arenas()` | 并行执行所有竞技场的一个 tick |
| `_compute_noise_trader_decisions(arena)` | 计算噪声交易者决策 |
| `_build_arena_commands()` | 构建 ArenaExecuteData 命令数据 |
| `_execute_with_worker_pool()` | 使用 Worker 池执行决策 |
| `_process_single_arena_result()` | 处理单个竞技场的 Worker 执行结果 |
| `_prepare_arena_for_tick()` | 准备单个竞技场的 tick 数据（线程安全） |
| `setup_for_testing(populations_data)` | 测试模式初始化 |
| `save_checkpoint(path)` / `load_checkpoint(path)` | 检查点管理 |
| `stop()` | 停止训练并清理资源 |

**tick 执行流程（run_tick_all_arenas）：**
```
阶段1: 准备（并行/串行）
  - 强平检查
  - 噪声交易者决策
  - 计算市场状态
  - 收集活跃 Agent

阶段2: 批量推理（OpenMP 并行）
  - 所有竞技场 x 所有 Agent 合并推理

阶段3: 执行
  - Worker 池并行执行（增量 submit + poll）
  - 处理结果，更新状态
  - 检查提前结束条件
```

### ArenaExecuteWorkerPool / ArenaExecuteWorkerPoolShm (execute_worker.py)

竞技场执行 Worker 池，将 Execute 阶段并行化。

**数据类定义：**

```python
@dataclass
class NoiseTraderDecision:
    trader_id: int      # 噪声交易者 ID（负数）
    direction: int      # 1=买, -1=卖
    quantity: int       # 下单数量

@dataclass
class NoiseTraderTradeResult:
    trader_id: int
    trades: list[tuple]  # 成交列表

@dataclass
class ArenaExecuteData:
    liquidated_agents: list[tuple[int, int, bool]]
    decisions: list[tuple[int, int, int, float, int]]
    mm_decisions: list[tuple[int, list, list]]
    noise_trader_decisions: list[NoiseTraderDecision]

@dataclass
class ArenaExecuteResult:
    arena_id: int
    bid_depth: np.ndarray
    ask_depth: np.ndarray
    last_price: float
    mid_price: float
    trades: list[tuple]
    pending_updates: dict[int, int | None]
    mm_order_updates: dict[int, tuple[list, list]]
    noise_trader_results: list[NoiseTraderTradeResult]
    error: str | None
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `start()` | 启动所有 Worker 进程 |
| `reset_all(initial_price, fee_rates, noise_trader_ids)` | 重置所有竞技场的订单簿 |
| `init_market_makers(mm_init_orders)` | 初始化做市商挂单 |
| `execute_all(arena_commands)` | 执行所有竞技场的决策 |
| `submit_all(arena_commands)` | 仅发送命令（增量处理） |
| `poll_result(arena_id)` | 非阻塞检查结果 |
| `shutdown()` | 关闭所有 Worker |

**原子动作执行流程：**

1. 强平处理（优先执行，不参与打乱）
2. 收集原子动作（噪声交易者市价单、做市商挂单、非做市商订单）
3. 随机打乱并执行
4. 构建返回结果（聚合噪声交易者成交）

### ArenaWorkerPool / arena_worker_main (arena_worker.py)

Arena Worker 进程池架构。每个 Worker 进程独立运行若干竞技场的完整 tick 循环，消除 tick 级 IPC 同步开销。Worker 内部持有独立的 BatchNetworkCache、ArenaState、MatchingEngine。

**数据类定义：**

```python
@dataclass
class AgentInfo:
    agent_id: int
    agent_type: AgentType
    sub_pop_id: int
    network_index: int        # 在 BatchNetworkCache 中的索引
    initial_balance: float
    leverage: float
    maintenance_margin_rate: float
    maker_fee_rate: float
    taker_fee_rate: float

@dataclass
class EpisodeResult:
    worker_id: int
    accumulated_fitness: dict[tuple[AgentType, int], NDArray[np.float32]]
    per_arena_fitness: dict[int, dict[AgentType, NDArray[np.float32]]]
    arena_stats: dict[int, ArenaEpisodeStats]
```

**通信协议：**

| 命令 | 方向 | 描述 |
|------|------|------|
| `update_networks` | 主进程 -> Worker | 进化后发送新网络参数 |
| `run_episode` | 主进程 -> Worker | Worker 独立运行 episodes 并返回适应度 |
| `shutdown` | 主进程 -> Worker | 关闭 Worker |

**ArenaWorkerPool 主要方法：**

| 方法 | 描述 |
|------|------|
| `start()` | 启动所有 Worker 进程 |
| `update_networks(network_params)` | 发送新网络参数给所有 Worker |
| `run_episodes(num_episodes, episode_length)` | 所有 Worker 独立运行 episodes |
| `shutdown()` | 关闭所有 Worker |

**Worker 内部 Episode 执行流程：**

```
_run_episode_local()
├─ 1. 重置所有竞技场 (_reset_arena)
├─ 2. MM 初始化 (_init_mm_all_arenas，所有竞技场复用同一推理结果)
├─ 3. Tick 循环
│   ├─ Tick 1: 仅记录初始状态
│   └─ Tick 2+:
│       ├─ handle_liquidations() - 三阶段强平（撤单→市价平仓→ADL）
│       ├─ compute_noise_trader_decisions() - 噪声交易者决策
│       ├─ compute_market_state() - 归一化市场状态（直接用本地 orderbook）
│       ├─ 批量推理（per-arena 独立，OMP_THREADS=1）
│       ├─ execute_tick_local() - 原子动作模式本地执行
│       ├─ 记录价格/tick 历史
│       └─ check_early_end() - 检查提前结束
└─ 4. _collect_fitness_all_arenas() - 收集适应度
```

**模块级核心函数（从 ParallelArenaTrainer 提取）：**

| 函数 | 描述 |
|------|------|
| `check_liquidations_vectorized()` | 向量化强平检查 |
| `handle_liquidations()` | 三阶段强平处理 |
| `cancel_agent_orders()` | 撤销 Agent 挂单 |
| `execute_liquidation()` | 强平市价平仓 |
| `execute_adl()` | ADL 自动减仓 |
| `update_trade_accounts()` | 更新成交账户状态 |
| `compute_noise_trader_decisions()` | 噪声交易者决策 |
| `compute_market_state()` | 归一化市场状态计算 |
| `check_early_end()` | 检查 Episode 提前结束 |
| `aggregate_tick_trades()` | 聚合 tick 成交量/额 |
| `update_episode_price_stats_from_trades()` | 更新价格统计 |
| `execute_tick_local()` | 本地原子动作执行 |

**与 execute_worker.py 的区别：**

- execute_worker.py: tick 级 IPC，每个 tick 主进程发送命令、Worker 执行、返回结果
- arena_worker.py: episode 级 IPC，Worker 独立运行完整 episode，仅在进化时同步网络参数

### SharedMemoryIPC (shared_memory_ipc.py)

共享内存 IPC 管理器，提供零拷贝的进程间通信机制。

**内存布局：**

```
Command Region:
- Header (64 bytes): status, cmd_type, counts, padding
- liquidated_data: int64[MAX_LIQUIDATED x 3]
- decisions_data: float64[MAX_DECISIONS x 5]
- mm_agent_ids: int64[MAX_MM_AGENTS]
- mm_decisions_data: float64[...]
- noise_trader_decisions: int64[MAX_NOISE_TRADERS x 3]

Result Region:
- Header (64 bytes)
- bid/ask depth, trades, pending_updates, mm_order_ids
```

## 性能优化

- **批量推理合并**：N 竞技场 x M Agent 合并成单次 OpenMP 并行操作
- **向量化强平检查**：NumPy 批量计算保证金率
- **并行进化**：RETAIL_PRO 12 子种群 + MARKET_MAKER 4 子种群
- **Phase 1 跨竞技场并行**：ThreadPoolExecutor 并行准备 tick 数据
- **增量 Worker 结果处理**：submit_all + poll_result 减少延迟
- **Agent 类型分组缓存**：episode 级预构建，tick 级仅过滤
- **延迟反序列化 + numpy->C 管线**：跳过中间 Python 对象
- **共享内存 IPC**：零拷贝、无锁轮询
- **做市商初始化批量推理复用**：Episode 开始时所有竞技场状态相同，`_prepare_mm_init_orders` 只对一个竞技场进行一次 BatchNetworkCache 批量推理，将结果复用到所有 64 个竞技场，避免 64x400=25600 次串行 forward() 调用

## 依赖关系

- `src.bio.agents` - Agent 类（RetailProAgent, MarketMakerAgent）
- `src.config.config` - 配置类（Config, NoiseTraderConfig）
- `src.core.log_engine` - 日志系统
- `src.market.adl` - ADL 管理器
- `src.market.noise_trader` - 噪声交易者模块
- `src.market.matching` - 撮合引擎
- `src.market.orderbook` - 订单簿
- `src.training.population` - 种群管理
- `src.training._cython.batch_decide_openmp` - OpenMP 批量决策
