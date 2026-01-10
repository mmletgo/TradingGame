# Arena 模块

## 模块概述

竞技场模块提供多竞技场并行训练功能，支持在多个独立进程中运行 episode 并汇总适应度。

## 文件结构

- `__init__.py` - 模块导出
- `arena_pool.py` - 竞技场进程池管理
- `arena_worker.py` - 竞技场工作进程
- `fitness_aggregator.py` - 适应度汇总器
- `multi_arena_trainer.py` - 多竞技场训练协调器

## 核心类和函数

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
) -> None
```

**命令格式：**

| 命令 | 参数 | 描述 |
|------|------|------|
| `setup` | `(genome_data_dict, network_params_dict)` | 从基因组创建 Agent |
| `run` | `num_episodes` | 运行 N 个 episode，返回累积适应度 |
| `shutdown` | 无 | 关闭进程 |

**结果格式：**

| 结果类型 | 数据 | 描述 |
|----------|------|------|
| `setup_done` | `(arena_id, None)` | setup 完成 |
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

1. **快速网络更新**：使用 `Brain.update_network_only()` 跳过完整基因组反序列化
2. **进程隔离**：每个竞技场在独立进程中运行，充分利用多核 CPU
3. **累积适应度**：运行多个 episode 后一次性返回累积值，减少进程间通信

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
