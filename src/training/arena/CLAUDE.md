# Arena 模块

## 模块概述

竞技场模块提供多竞技场并行训练功能，支持在多个独立进程中运行 episode 并汇总适应度。

## 文件结构

- `__init__.py` - 模块导出
- `arena_pool.py` - 竞技场进程池管理
- `arena_state.py` - 竞技场状态类（AgentAccountState、CatfishAccountState、ArenaState）
- `arena_worker.py` - 竞技场工作进程
- `fitness_aggregator.py` - 适应度汇总器
- `multi_arena_trainer.py` - 多竞技场训练协调器（多进程并行）
- `single_arena_trainer.py` - 单进程多竞技场训练器（串行执行，OpenMP 并行）

## 核心类和函数

### AgentAccountState (arena_state.py)

Agent 账户状态类，用于多竞技场并行推理架构。将 Agent 账户状态与 Agent 对象解耦，每个竞技场维护独立副本。

**类定义：**
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

**持仓更新逻辑（_update_position）：**
- 空仓开仓：直接设置持仓和均价
- 加仓：加权平均计算新均价
- 减仓：计算已实现盈亏
- 完全平仓：清零持仓和均价
- 反向开仓：先平仓再反向开仓

### CatfishAccountState (arena_state.py)

鲶鱼账户状态类，类似 AgentAccountState，但包含鲶鱼特有的策略状态。

**类定义：**
```python
@dataclass
class CatfishAccountState:
    """鲶鱼账户状态"""
    catfish_id: int
    balance: float
    position_quantity: int
    position_avg_price: float
    realized_pnl: float
    leverage: float
    maintenance_margin_rate: float
    initial_balance: float
    is_liquidated: bool
    order_counter: int
    current_direction: int  # 趋势创造者方向
    ema: float              # 均值回归 EMA
    ema_initialized: bool
    last_action_tick: int
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `from_catfish(catfish)` | 类方法，从 CatfishBase 对象创建状态副本 |
| `reset(initial_balance)` | 重置到初始状态 |
| `get_equity(current_price)` | 计算净值 |
| `get_margin_ratio(current_price)` | 计算保证金率 |
| `check_liquidation(current_price)` | 检查是否需要强平（仅资金归零时） |
| `on_trade(trade_price, trade_quantity, is_buyer)` | 处理成交（无手续费） |
| `generate_order_id(arena_id)` | 生成跨竞技场唯一的订单 ID（负数空间） |
| `update_ema(price, ma_period)` | 更新 EMA 值（均值回归策略用） |

### ArenaState (arena_state.py)

单个竞技场的独立状态，封装竞技场运行所需的所有状态。

**类定义：**
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
    price_history: list[float]
    tick_history_prices: list[float]
    tick_history_volumes: list[float]
    tick_history_amounts: list[float]
    smooth_mid_price: float
    tick: int
    pop_liquidated_counts: dict[AgentType, int]
    eliminating_agents: set[int]
    episode_high_price: float
    episode_low_price: float
    catfish_liquidated: bool
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `reset_episode(initial_price)` | 重置 Episode 状态 |
| `get_agent_state(agent_id)` | 获取 Agent 状态 |
| `get_catfish_state(catfish_id)` | 获取鲶鱼状态 |
| `update_price_stats(price)` | 更新价格统计信息（最高/最低价） |
| `mark_agent_liquidated(agent_id, agent_type)` | 标记 Agent 已被强平 |

---

### ArenaPool (arena_pool.py)

竞技场进程池，管理多个 ArenaWorker 进程，协调并行训练。

**类定义：**
```python
class ArenaPool:
    """竞技场进程池

    Attributes:
        num_arenas: 竞技场数量
        config: 全局配置
        workers: 工作进程列表
        cmd_queues: 命令队列（每个竞技场一个）
        result_queue: 结果队列（共享）
        shutdown_event: 共享的 shutdown 事件，用于通知工作进程退出
    """
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `__init__(num_arenas, config)` | 初始化进程池，创建队列 |
| `start()` | 启动所有工作进程 |
| `broadcast_genomes(genome_data, network_params)` | 广播基因组到所有竞技场 |
| `run_all(episodes_per_arena)` | 并行运行所有竞技场，返回汇总后的平均适应度 |
| `shutdown()` | 关闭所有工作进程 |
| `__enter__()` / `__exit__()` | 上下文管理器支持 |

**使用示例：**
```python
from src.training.arena import ArenaPool

# 方式1：使用上下文管理器
with ArenaPool(num_arenas=4, config=config) as pool:
    # 广播基因组
    pool.broadcast_genomes(genome_data, network_params)

    # 运行并获取汇总适应度
    avg_fitness = pool.run_all(episodes_per_arena=10)

# 方式2：手动管理
pool = ArenaPool(num_arenas=4, config=config)
pool.start()
try:
    pool.broadcast_genomes(genome_data, network_params)
    avg_fitness = pool.run_all(episodes_per_arena=10)
finally:
    pool.shutdown()
```

**错误处理：**
- `RuntimeError`: 当任意竞技场 setup 或运行失败时抛出
- `TimeoutError`: 当等待 setup/运行完成超时时抛出

**超时设置：**
- setup 超时：60 秒
- run 超时：动态计算，最大 10 分钟

### ArenaConfig (arena_worker.py)

竞技场配置数据类。

```python
@dataclass
class ArenaConfig:
    arena_id: int                     # 竞技场 ID
    episodes_per_round: int = 10      # 每轮运行的 episode 数量
    episode_length: int = 1000        # 每个 episode 的 tick 数量
```

### arena_worker_process (arena_worker.py)

竞技场工作进程主函数，在独立进程中运行。

**函数签名：**
```python
def arena_worker_process(
    arena_id: int,
    config: Config,
    cmd_queue: Queue[tuple[str, Any]],
    result_queue: Queue[tuple[str, int, Any]],
    shutdown_event: Any = None,  # multiprocessing.Event，用于通知进程退出
) -> None
```

**进程退出机制：**
- 工作进程使用带超时的 `cmd_queue.get(timeout=1.0)` 等待命令
- 每次循环检查 `shutdown_event.is_set()` 状态
- 收到 `shutdown` 命令或检测到 shutdown 事件时退出主循环

**命令格式：**

| 命令 | 参数 | 描述 |
|------|------|------|
| `setup` | `(genome_data_dict, network_params_dict)` | 从基因组创建 Agent |
| `update_networks` | `network_params_dict` | 热更新网络参数（不重建 Agent） |
| `run` | `num_episodes` | 运行 N 个 episode，返回累积适应度 |
| `shutdown` | 无 | 关闭进程 |

**结果格式：**

| 结果类型 | 数据 | 描述 |
|----------|------|------|
| `setup_done` | `(arena_id, None)` | setup 完成 |
| `update_done` | `(arena_id, None)` | 网络参数更新完成 |
| `run_done` | `(arena_id, (fitness_data, episode_count))` | 运行完成 |
| `error` | `(arena_id, error_message)` | 发生错误 |

**适应度数据格式：**
```python
fitness_data: dict[tuple[AgentType, int], np.ndarray]
# key: (agent_type, sub_pop_id)  # sub_pop_id 竞技场模式固定为 0
# value: 累积适应度数组，shape=(pop_size,)
```

### FitnessAggregator (fitness_aggregator.py)

适应度汇总器，用于汇总多个竞技场、多个 episode 的适应度数据。

**方法：**
- `aggregate_simple_average(arena_fitnesses, episode_counts)` - 简单加权平均

**公式：**
```
avg_fitness = sum(arena_fitness) / total_episodes
```

## 使用模式

### 基本使用流程

```python
from multiprocessing import Process, Queue
from src.training.arena import arena_worker_process, ArenaConfig, FitnessAggregator

# 1. 创建队列
cmd_queue = Queue()
result_queue = Queue()

# 2. 启动工作进程
p = Process(
    target=arena_worker_process,
    args=(arena_id, config, cmd_queue, result_queue)
)
p.start()

# 3. 发送 setup 命令
cmd_queue.put(("setup", genome_data_dict, network_params_dict))
result = result_queue.get()  # ("setup_done", arena_id, None)

# 4. 发送 run 命令
cmd_queue.put(("run", num_episodes))
result = result_queue.get()  # ("run_done", arena_id, (fitness_data, episode_count))

# 5. 关闭进程
cmd_queue.put(("shutdown",))
p.join()

# 6. 汇总多个竞技场的适应度
avg_fitness = FitnessAggregator.aggregate_simple_average(
    arena_fitnesses=[fitness_data1, fitness_data2],
    episode_counts=[10, 10]
)
```

### 网络参数格式

`network_params_dict` 是预计算的网络参数，用于快速更新 Brain 而无需完整反序列化基因组。

```python
network_params_dict: dict[AgentType, tuple[np.ndarray, ...]]
# 元组内容：
# (headers, all_input_keys, all_output_keys, all_node_ids,
#  all_biases, all_responses, all_act_types,
#  all_conn_indptr, all_conn_sources, all_conn_weights,
#  all_output_indices)
```

使用 `_unpack_network_params_numpy()` 解包后，通过 `Brain.update_network_only()` 快速更新网络。

## 性能优化

1. **Agent 复用**：首次调用 `broadcast_genomes` 发送 `setup` 命令创建 Agent，后续调用只发送 `update_networks` 命令更新网络参数，避免重复创建 Agent 对象
2. **快速网络更新**：使用 `Brain.update_network_only()` 跳过完整基因组反序列化
3. **进程隔离**：每个竞技场在独立进程中运行，充分利用多核 CPU
4. **累积适应度**：运行多个 episode 后一次性返回累积值，减少进程间通信

**Agent 复用机制：**
- `ArenaPool` 内部维护 `_agents_initialized` 状态
- 首次调用：发送 `setup` 命令，创建完整的 Agent 对象
- 后续调用：发送 `update_networks` 命令，仅更新网络参数（~30秒 → ~1秒）
- `shutdown()` 时重置状态，下次启动时重新初始化

## 注意事项

1. 竞技场模式下 `sub_pop_id` 固定为 0（不支持子种群）
2. 工作进程会忽略 SIGINT，由主进程统一处理退出
3. 鲶鱼强平会触发 episode 提前结束（与训练模式一致）

---

### MultiArenaTrainer (multi_arena_trainer.py)

多竞技场训练协调器，协调多个竞技场并行训练，管理进化周期。

**核心流程：**
1. 初始化：创建 NEAT 种群、创建进化 Worker 池、创建竞技场池
2. 训练循环：
   a. 序列化并广播基因组到所有竞技场
   b. 并行运行所有竞技场
   c. 汇总适应度
   d. 执行 NEAT 进化
   e. 保存检查点

**类定义：**
```python
@dataclass
class MultiArenaConfig:
    """多竞技场配置"""
    num_arenas: int = 10
    episodes_per_arena: int = 10


class MultiArenaTrainer:
    """多竞技场训练协调器

    Attributes:
        config: 全局配置
        multi_arena_config: 多竞技场配置
        populations: 种群字典 {AgentType -> Population/SubPopulationManager}
        evolution_worker_pool: 进化 Worker 池
        arena_pool: 竞技场池
        generation: 当前代数
    """
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `setup()` | 初始化训练环境，创建种群、Worker 池和竞技场池 |
| `run_round()` | 运行一轮训练（广播->运行->汇总->进化） |
| `train(num_rounds, checkpoint_callback)` | 主训练循环 |
| `save_checkpoint(path)` | 保存检查点 |
| `load_checkpoint(path)` | 加载检查点 |
| `stop()` | 停止训练并清理资源 |

**种群配置：**
- RETAIL: SubPopulationManager（10个子种群）
- RETAIL_PRO: Population
- WHALE: Population
- MARKET_MAKER: SubPopulationManager（2个子种群）

**使用示例：**
```python
from src.training.arena import MultiArenaTrainer, MultiArenaConfig

# 方式1：使用上下文管理器
multi_config = MultiArenaConfig(num_arenas=4, episodes_per_arena=10)
with MultiArenaTrainer(config, multi_config) as trainer:
    trainer.train(num_rounds=100, checkpoint_callback=lambda gen: print(f"Gen {gen}"))

# 方式2：手动管理
trainer = MultiArenaTrainer(config, multi_config)
trainer.setup()
try:
    for _ in range(100):
        stats = trainer.run_round()
        if trainer.generation % 10 == 0:
            trainer.save_checkpoint(f"checkpoints/gen_{trainer.generation}.pkl")
finally:
    trainer.stop()
```

**检查点格式：**
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
        AgentType.RETAIL_PRO: {
            "generation": int,
            "neat_pop": neat.Population,
        },
        ...
    }
}
```

**性能优化：**
1. **并行进化**：使用 MultiPopulationWorkerPool 在多个进程中并行执行 NEAT 进化
2. **并行训练**：使用 ArenaPool 在多个进程中并行运行 episode
3. **网络参数传输**：只传输网络参数而非完整基因组，减少序列化开销
4. **内存管理**：每轮训练后进行垃圾回收和 malloc_trim
5. **Checkpoint 体积优化**：
   - 保存前清理 NEAT 历史数据（`_cleanup_neat_history()`）
   - 使用 gzip 压缩保存检查点文件
   - 加载时自动检测文件格式（gzip 或普通 pickle），支持向后兼容

---

### SingleArenaTrainer (single_arena_trainer.py)

单进程多竞技场训练器，串行运行多个竞技场，充分利用 OpenMP 多核并行推理。

**与 MultiArenaTrainer 的对比：**

| 方面 | MultiArenaTrainer | SingleArenaTrainer |
|------|------------------|-------------------|
| 竞技场执行 | 多进程并行 | 单进程串行 |
| OpenMP 线程 | 8×10=80（竞争 32 核） | 32（充分利用） |
| 进程通讯 | 有（序列化） | 无 |
| 内存占用 | 高（N 份种群） | 低（1 份种群） |
| 进化 | 多进程并行（保留） | 多进程并行（保留） |
| 适应度汇总 | FitnessAggregator | FitnessAggregator（相同） |

**设计理念：**
当 OpenMP 已经在单进程内实现了多核并行推理时，多进程竞技场会导致 OpenMP 线程竞争（80 线程抢 32 核）。SingleArenaTrainer 通过单进程串行执行竞技场，让每个竞技场都能充分利用所有 CPU 核心进行 OpenMP 并行推理。

**核心流程：**
1. 初始化：创建单个 Trainer 实例（包含种群、撮合引擎等），创建进化 Worker 池
2. 训练循环：
   a. 串行运行所有竞技场（每个竞技场重置市场状态后运行 M 个 episode）
   b. 累积并汇总适应度
   c. 执行 NEAT 进化（多进程并行）
   d. 更新种群网络参数
   e. 保存检查点

**类定义：**
```python
class SingleArenaTrainer:
    """单进程多竞技场训练器

    Attributes:
        config: 全局配置
        multi_config: 多竞技场配置
        trainer: Trainer 实例（复用）
        evolution_worker_pool: 进化 Worker 池
        generation: 当前代数
        total_episodes: 总 episode 数
    """
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `setup()` | 初始化训练环境，创建 Trainer 和 Worker 池 |
| `run_round()` | 运行一轮训练（串行运行竞技场->汇总->进化） |
| `_run_arena()` | 运行单个竞技场的 M 个 episode |
| `_reset_for_arena()` | 重置市场状态（不重置种群） |
| `_accumulate_fitness()` | 累积单个 episode 的适应度 |
| `train(num_rounds, checkpoint_callback, progress_callback)` | 主训练循环 |
| `save_checkpoint(path)` | 保存检查点 |
| `load_checkpoint(path)` | 加载检查点 |
| `stop()` | 停止训练并清理资源 |

**使用示例：**
```python
from src.training.arena import SingleArenaTrainer, MultiArenaConfig

# 方式1：使用上下文管理器
multi_config = MultiArenaConfig(num_arenas=10, episodes_per_arena=10)
with SingleArenaTrainer(config, multi_config) as trainer:
    trainer.train(
        num_rounds=100,
        checkpoint_callback=lambda gen: print(f"Gen {gen}"),
        progress_callback=lambda stats: print(stats)
    )

# 方式2：手动管理
trainer = SingleArenaTrainer(config, multi_config)
trainer.setup()
try:
    for _ in range(100):
        stats = trainer.run_round()
        if trainer.generation % 10 == 0:
            trainer.save_checkpoint(f"checkpoints/gen_{trainer.generation}.pkl")
finally:
    trainer.stop()
```

**检查点兼容性：**
- 检查点格式与 MultiArenaTrainer 完全相同
- 可以互相加载检查点
- 支持 gzip 压缩和普通 pickle 格式

**启动脚本：**
```bash
# 默认训练
python scripts/train_single_arena.py --rounds 100

# 自定义参数
python scripts/train_single_arena.py --num-arenas 8 --episodes-per-arena 5 --rounds 200

# 从检查点恢复
python scripts/train_single_arena.py --resume checkpoints/single_arena_gen_50.pkl
```

---

### ParallelArenaTrainer (parallel_arena_trainer.py)

多竞技场并行推理训练器，核心特性是将多个竞技场的神经网络推理合并成一个批量操作。

**与 SingleArenaTrainer 的对比：**

| 方面 | SingleArenaTrainer | ParallelArenaTrainer |
|------|-------------------|---------------------|
| 竞技场执行 | 串行（一个接一个） | 同步推进（tick 对齐） |
| 神经网络 | 共享 | 共享 |
| 账户状态 | 复用 Trainer | 独立 ArenaState |
| 推理方式 | 每竞技场单独推理 | 跨竞技场批量推理 |
| 订单簿 | 复用（重置） | 每竞技场独立 |

**设计理念：**
1. **神经网络共享**：所有竞技场使用同一套 `BatchNetworkCache`，进化后统一更新
2. **账户状态独立**：每个竞技场维护独立的 `ArenaState`，包含所有 Agent 的账户状态
3. **批量推理合并**：N 个竞技场 × M 个 Agent 的推理合并成单次 OpenMP 并行操作
4. **订单簿独立**：每个竞技场有独立的 `MatchingEngine` 和 `OrderBook`

**核心流程：**
1. 初始化：创建共享种群、N 个独立竞技场状态、共享网络缓存、进化 Worker 池
2. 训练循环：
   a. 重置所有竞技场
   b. 同步推进所有竞技场的 tick（批量推理）
   c. 汇总适应度
   d. 执行 NEAT 进化
   e. 更新网络缓存和 Agent 状态

**类定义：**
```python
class ParallelArenaTrainer:
    """多竞技场并行推理训练器

    Attributes:
        config: 全局配置
        multi_config: 多竞技场配置
        populations: 共享的种群（神经网络）
        arena_states: N 个独立的竞技场状态
        network_caches: 共享的网络缓存
        evolution_worker_pool: 进化 Worker 池
        generation: 当前代数
        total_episodes: 总 episode 数
    """
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `setup()` | 初始化：创建种群、竞技场状态、网络缓存、进化 Worker 池 |
| `run_round()` | 运行一轮训练（所有竞技场的所有 episode + 进化） |
| `run_tick_all_arenas()` | 并行执行所有竞技场的一个 tick |
| `_batch_inference_all_arenas()` | 批量推理所有竞技场的所有 Agent |
| `_run_episode_all_arenas()` | 运行所有竞技场的一个 episode |
| `_collect_fitness_all_arenas()` | 收集并汇总所有竞技场的适应度 |
| `train(num_rounds, checkpoint_callback, progress_callback)` | 主训练循环 |
| `save_checkpoint(path)` | 保存检查点 |
| `load_checkpoint(path)` | 加载检查点 |
| `stop()` | 停止训练并清理资源 |

**tick 执行流程（run_tick_all_arenas）：**
```python
# 阶段1: 准备（串行）
for arena in arena_states:
    handle_liquidations(arena)  # 强平检查
    market_state = compute_market_state(arena)
    active_agents = get_active_agents(arena)

# 阶段2: 批量推理（并行）- 一次性推理所有竞技场的所有 Agent
all_decisions = _batch_inference_all_arenas(market_states, active_agents)

# 阶段3: 执行（串行）
for arena in arena_states:
    execute_trades(arena, all_decisions[arena.arena_id])
```

**批量推理合并（_batch_inference_all_arenas）：**
- 使用 `AgentStateAdapter` 将 `AgentAccountState` 适配为 Agent-like 接口
- 按 AgentType 分组（跨所有竞技场）
- 对每种类型调用对应的 `BatchNetworkCache.decide()`
- 将结果重组为 `dict[arena_id, list[decision]]`

**使用示例：**
```python
from src.training.arena import ParallelArenaTrainer, MultiArenaConfig

# 方式1：使用上下文管理器
multi_config = MultiArenaConfig(num_arenas=10, episodes_per_arena=10)
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

**检查点兼容性：**
- 检查点格式与 MultiArenaTrainer 和 SingleArenaTrainer 完全相同
- 可以互相加载检查点
- 支持 gzip 压缩和普通 pickle 格式
