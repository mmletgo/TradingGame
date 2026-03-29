# Arena 模块

## 模块概述

竞技场模块提供多竞技场并行训练功能。核心架构为 **Episode 级 Worker 进程池**（`ArenaWorkerPool`）：每个 Worker 进程独立运行若干竞技场的完整 tick 循环（包括推理、撮合、强平等），主进程仅负责 NEAT 进化和网络参数同步。

## 设计理念

1. **Episode 级并行**：Worker 进程独立运行完整 episode tick 循环，消除 tick 级 IPC 同步开销
2. **神经网络同步**：进化后主进程将新的网络参数（packed numpy）发送给所有 Worker（支持共享内存零拷贝）
3. **账户状态独立**：每个 Worker 内部维护独立的 `ArenaState`
4. **订单簿独立**：每个 Worker 内部的竞技场有独立的 `MatchingEngine` 和 `OrderBook`
5. **噪声交易者**：每个 Worker 的竞技场包含独立的噪声交易者，提供市场背景噪音
6. **主进程保留竞技场状态**：用于检查点保存/加载和 Agent 状态刷新

## 文件结构

```
src/training/arena/
├── __init__.py                # 模块导出
├── arena_state.py             # 竞技场状态类（AgentAccountState、NoiseTraderAccountState、ArenaState）
├── arena_worker.py            # Arena Worker 进程池架构（Worker 独立运行完整 tick 循环）
├── execute_worker.py          # Execute Worker 池（tick 级 IPC 模式，已弃用）
├── fitness_aggregator.py      # 适应度汇总器
├── parallel_arena_trainer.py  # 多竞技场并行推理训练器
├── shared_memory_ipc.py       # 共享内存 IPC 机制（零拷贝优化）
└── shared_network_memory.py   # BatchNetworkData 共享内存生命周期管理
```

---

## 核心类

### AgentAccountState (arena_state.py)

Agent 账户状态类，将 Agent 账户状态与 Agent 对象解耦，每个竞技场维护独立副本。

**数据结构：**
```python
@dataclass
class AgentAccountState:
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
    bid_order_ids: list[int]      # 做市商专用
    ask_order_ids: list[int]      # 做市商专用
    cumulative_spread_score: float
    quote_tick_count: int
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
    config_quantity_mu: float = 12.0    # 对数正态分布 mu 参数（从配置复制）
    config_quantity_sigma: float = 1.0 # 对数正态分布 sigma 参数（从配置复制）
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `from_noise_trader(noise_trader)` | 类方法，从 NoiseTrader 对象创建状态副本（包括 quantity_mu/sigma） |
| `reset()` | 重置到初始状态 |
| `get_equity(current_price)` | 计算净值 |
| `on_trade(price, quantity, is_buyer)` | 处理成交（无手续费） |
| `generate_order_id(arena_id)` | 生成唯一订单 ID |
| `decide(action_probability, buy_probability=0.5)` | 决策：返回 (should_act, direction, quantity)，buy_probability 支持 Episode 级方向偏置 |

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
    tick_history_prices: deque[float]    # maxlen=100
    tick_history_volumes: deque[float]   # maxlen=100
    tick_history_amounts: deque[float]   # maxlen=100
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

---

### ParallelArenaTrainer (parallel_arena_trainer.py)

多竞技场并行训练器。主进程职责：创建种群、NEAT 进化、网络参数同步、检查点管理。tick 循环完全委托给 `ArenaWorkerPool`。

**种群配置（2 种类型）：**
- RETAIL_PRO: 10 个子种群（SubPopulationManager）
- MARKET_MAKER: 6 个子种群（SubPopulationManager）

**主要方法：**

| 方法 | 描述 |
|------|------|
| `setup()` | 初始化：先创建进化 Worker 池（避免 COW），再创建种群、竞技场状态、网络缓存、Arena Worker Pool |
| `run_round()` | 运行一轮训练：调用 ArenaWorkerPool.run_episodes() 运行所有 episode，汇总 fitness，执行 NEAT 进化，同步新网络 |
| `_build_agent_infos()` | 从 populations 构建 AgentInfo 列表（供 ArenaWorkerPool 使用） |
| `_sync_networks_to_workers()` | 将网络参数（packed numpy）同步到所有 Arena Workers |
| `_aggregate_worker_fitness()` | 汇总所有 Worker 的 fitness 和参与计数，返回 `(fitness_dict, participation_dict)` |
| `_build_episode_stats_from_results()` | 从 Worker 结果构建 episode 统计信息 |
| `setup_for_testing(populations_data)` | 测试模式初始化（不使用 ArenaWorkerPool） |
| `save_checkpoint(path)` / `load_checkpoint(path)` | 检查点管理（保存前自动从 Worker 同步基因组数据） |
| `stop()` | 停止训练并清理资源（关闭 ArenaWorkerPool + 进化 Worker 池） |

**训练轮次流程（run_round）：**
```
1. ArenaWorkerPool.run_episodes(num_episodes=episodes_per_arena)
   ├─ 所有 Worker 批量运行全部 episode（消除 episode 间同步屏障）
   ├─ 每个 Worker 内部连续运行 episodes_per_arena 个 episode
   ├─ 每个 episode 独立生成随机参数（episode_buy_prob、OU 初始状态等）
   └─ Worker 内部合并多 episode 结果后一次性返回
2. _aggregate_worker_fitness() - 汇总 Worker 返回的 fitness
3. _on_arena_fitness_collected() - League 训练钩子
4. _collect_fitness_all_arenas() - 按实际参与数平均 fitness（防适应度稀释）
5. NEAT 进化（lite 模式：跳过基因组序列化，仅返回网络参数和 species 数据）
6. _update_populations_from_evolution() - 更新种群（跳过 brain 更新，仅缓存网络参数）
7. _update_network_caches() - 更新主进程网络缓存（不清除 _cached_network_params_data，由后续 _sync_networks_to_workers 消费）
8. _refresh_agent_states() - 刷新 Agent 状态
9. _sync_networks_to_workers() - 同步新网络到 Workers
```

**共享内存同步机制：**
1. 主进程创建 `SharedNetworkMemory`，填充网络参数
2. 生成 `SharedNetworkMetadata`（~200 字节，包含 shm_name、维度信息）
3. 通过 Queue 发送 metadata 给所有 Worker
4. Worker 调用 `attach()` 挂载共享内存，`BatchNetworkCache` 直接使用共享内存中的网络数据
5. 旧共享内存在所有 Worker ack 后自动清理

---

### AgentInfo (arena_worker.py)

Worker 进程需要的轻量级 Agent 结构信息。

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
    is_historical: bool = False        # 是否为历史代 Agent
    historical_entry_id: str = ""      # 历史 entry ID（用于胜率更新）
```

**历史 Agent 说明：**
- 联盟训练中，历史代精英 Agent 通过 `is_historical=True` 标记
- 历史 Agent 的 `sub_pop_id >= 1000`，自动不参与 NEAT 进化
- 历史 Agent 的 `network_index` 在当前代之后偏移，对应合并后的 BatchNetworkCache 中的位置

### WorkerArenaAssignment (arena_worker.py)

Worker 进程的竞技场分配信息，描述每个竞技场应包含哪些历史 Agent 以及排除哪些当代 Agent 类型。

```python
@dataclass
class WorkerArenaAssignment:
    arena_historical_ids: dict[int, set[int]]         # arena_id → 历史 agent_id 集合
    arena_excluded_current_types: dict[int, set[AgentType]]  # arena_id → 排除的当代类型集合
```

**竞技场类型映射：**
- 纯竞技场：`excluded = set()` — 不排除任何当代类型
- 散户挑战赛：`excluded = {AgentType.MARKET_MAKER}` — 排除当代做市商，使用历史做市商
- MM挑战赛：`excluded = {AgentType.RETAIL_PRO}` — 排除当代散户，使用历史散户

### EpisodeResult (arena_worker.py)

Worker 返回给主进程的结果。

```python
@dataclass
class EpisodeResult:
    worker_id: int
    accumulated_fitness: dict[tuple[AgentType, int], NDArray[np.float32]]  # 累积适应度
    per_arena_fitness: dict[int, dict[AgentType, NDArray[np.float32]]]    # 每竞技场适应度
    arena_stats: dict[int, ArenaEpisodeStats]                             # 竞技场统计
    participation_counts: dict[tuple[AgentType, int], int]                # 每 sub_pop 实际参与的竞技场数
```

### ArenaWorkerPool (arena_worker.py)

管理持久化 Arena Worker 进程池。

**通信协议：**

| 命令 | 方向 | 描述 |
|------|------|------|
| `update_networks` | 主进程 -> Worker | 进化后发送新网络参数 |
| `attach_shared_networks` | 主进程 -> Worker | 发送共享内存元数据，Worker 零拷贝挂载 |
| `update_agent_infos` | 主进程 -> Worker | 更新 agent_infos 列表（联盟训练：每轮动态变化的历史 Agent） |
| `run_episode` | 主进程 -> Worker | Worker 独立运行 episodes 并返回适应度 |
| `shutdown` | 主进程 -> Worker | 关闭 Worker |

**主要方法：**

| 方法 | 描述 |
|------|------|
| `start()` | 启动所有 Worker 进程，支持 CPU 亲和性绑定（`enable_cpu_affinity`） |
| `update_networks(network_params)` | 发送新网络参数给所有 Worker |
| `attach_shared_networks(metadata_map)` | 发送共享内存元数据给所有 Worker |
| `update_agent_infos(agent_infos, per_arena_allocation=None)` | 更新所有 Worker 的 agent_infos（重建内部缓存），支持 per-arena 分配（不同竞技场包含不同 Agent 组合） |
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

**模块级核心函数：**

| 函数 | 描述 |
|------|------|
| `check_liquidations_vectorized()` | 向量化强平检查 |
| `handle_liquidations()` | 三阶段强平处理 |
| `cancel_agent_orders()` | 撤销 Agent 挂单 |
| `execute_liquidation()` | 强平市价平仓 |
| `execute_adl()` | ADL 自动减仓 |
| `update_trade_accounts()` | 更新成交账户状态 |
| `compute_noise_trader_decisions()` | 噪声交易者决策（支持 buy_probability 参数） |
| `compute_market_state()` | 归一化市场状态计算 |
| `_build_per_arena_data()` | 根据 WorkerArenaAssignment 为每个竞技场独立构建 type_groups、pop_total_counts 和 agent_infos |
| `check_early_end()` | 检查 Episode 提前结束 |
| `aggregate_tick_trades()` | 聚合 tick 成交量/额 |
| `update_episode_price_stats_from_trades()` | 更新价格统计 |
| `execute_tick_local()` | 本地原子动作执行 |
| `_get_physical_core_ids()` | 获取每个物理核心对应的第一个逻辑 CPU ID（用于 CPU 亲和性绑定） |

---

### SharedNetworkMetadata / SharedNetworkMemory (shared_network_memory.py)

管理 BatchNetworkData 的共享内存生命周期。主进程创建共享内存并填充网络数据，Worker 进程通过共享内存名称附着到同一块内存实现零拷贝访问。

**SharedNetworkMetadata**：轻量级数据类（~200 字节），通过 Queue 发送给 Worker。

```python
@dataclass
class SharedNetworkMetadata:
    agent_type: AgentType
    shm_name: str
    num_networks: int
    max_nodes: int
    max_connections: int
    max_inputs: int
    max_outputs: int
    total_nodes: int
    total_connections: int
    total_outputs: int
    generation: int
```

**SharedNetworkMemory 主要方法：**

| 方法 | 描述 |
|------|------|
| `compute_buffer_size(num_networks, ...)` | 静态方法，调用 Cython 计算所需共享内存大小 |
| `create_and_fill(agent_type, network_params, generation)` | 主进程调用：创建 SharedMemory，用 Cython fill_shared_memory_buffer 填充数据，返回 SharedNetworkMetadata |
| `attach(metadata)` | Worker 调用：附着到已有共享内存，返回 memoryview |
| `close()` | 关闭共享内存映射（不删除底层共享内存段） |
| `unlink()` | 删除底层共享内存段（仅创建者调用） |
| `close_and_unlink()` | 关闭并删除共享内存（先 unlink 再 close） |

**生命周期：**
- 主进程：`create_and_fill()` -> 发送 metadata 给 Worker -> Worker 使用完成 -> `close_and_unlink()`
- Worker：收到 metadata -> `attach()` 新共享内存 -> `attach_shared_memory()` 切换缓存到新 buffer -> `close()` 旧共享内存（延迟关闭，确保缓存已切换后再释放旧 buffer）

---

### FitnessAggregator (fitness_aggregator.py)

适应度汇总器，汇总多个竞技场、多个 episode 的适应度数据。

**主要方法：**

| 方法 | 描述 |
|------|------|
| `aggregate_simple_average(arena_fitnesses, episode_counts)` | 简单加权平均：`avg = sum(fitness * count) / total_episodes` |

---

## 性能优化

- **Episode 级并行**：Worker 进程独立运行完整 episode，消除 tick 级 IPC 同步开销
- **向量化强平检查**：Worker 内部使用 NumPy 批量计算保证金率
- **并行进化**：RETAIL_PRO 10 子种群 + MARKET_MAKER 6 子种群并行进化
- **Lite 进化模式**：Worker 跳过基因组序列化，仅在 checkpoint 保存时按需同步
- **跳过 brain 更新**：并行竞技场模式下主进程不更新 Agent brain（ArenaWorker 使用 BatchNetworkCache）
- **合并 extract+pack**：网络参数提取和打包合并为 `_extract_and_pack_all_network_params`，减少中间对象
- **优化拓扑计算**：使用 `fast_feed_forward_layers_optimized`（邻接表），大网络加速显著
- **延迟反序列化 + numpy->C 管线**：跳过中间 Python 对象
- **做市商初始化批量推理复用**：Worker 内部 Episode 开始时所有竞技场状态相同，只对一个竞技场进行一次 BatchNetworkCache 批量推理，将结果复用到所有竞技场。复用时每个竞技场的每个 MM 独立添加 ±5 ticks 的随机价格扰动，确保各竞技场初始订单簿结构不同
- **Worker 内部 OpenMP 推理**：每个 Worker 使用独立的 BatchNetworkCache 进行 per-arena 批量推理
- **共享内存零拷贝**：网络参数通过共享内存同步，Worker 直接访问共享内存中的网络数据

---

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
