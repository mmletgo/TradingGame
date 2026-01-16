# Training 模块

## 模块概述

训练模块负责管理 NEAT 进化训练流程，包括种群管理和训练协调。

## 文件结构

- `__init__.py` - 模块导出
- `population.py` - 种群管理类
- `trainer.py` - 训练器类
- `checkpoint_loader.py` - checkpoint 加载器
- `fast_math.py` - Numba JIT 加速的数学函数（对数归一化等）
- `arena/` - 竞技场模块（详见 `arena/CLAUDE.md`）
  - `__init__.py` - 模块导出
  - `arena_worker.py` - 竞技场工作进程
  - `fitness_aggregator.py` - 适应度汇总器
- `_cython/` - Cython 加速模块
  - `__init__.py` - 模块导出
  - `batch_decide_openmp.pyx` - OpenMP 并行批量决策
  - `batch_decide_openmp.pxd` - 头文件
  - `fast_execution.pyx` - 非做市商批量订单执行

## 核心类

### fast_execution 模块 (_cython/fast_execution.pyx)

Cython 加速的非做市商批量订单执行模块，优化散户/高级散户/庄家的订单处理性能。

**设计目标：**
- 减少串行执行阶段的 Python 对象开销
- 内联订单 ID 生成和账户更新逻辑
- 缓存常用方法引用减少属性查找

**核心函数：**

#### `execute_non_mm_batch(decisions, matching_engine, orderbook, recent_trades) -> list`

批量执行非做市商的订单决策。

**参数：**
- `decisions: list[(agent, action, params)]` - 决策列表
  - `agent`: Agent 实例（非做市商）
  - `action`: ActionType 枚举值（HOLD=0, PLACE_BID=1, PLACE_ASK=2, CANCEL=3, MARKET_BUY=4, MARKET_SELL=5）
  - `params`: 参数字典 `{"price": float, "quantity": int}`
- `matching_engine`: MatchingEngine 实例
- `orderbook`: OrderBook 实例
- `recent_trades`: deque，用于记录成交

**返回值：**
- `list[Trade]` - 所有成交记录列表

**优化点：**
- 使用 `cdef` 声明 C 类型变量
- 缓存 `process_order`、`cancel_order`、`order_map.get` 等方法引用
- 内联订单 ID 生成：`(agent_id << 32) | order_counter`
- 使用 64 位整数避免位移溢出

#### `execute_non_mm_batch_with_maker_update(decisions, matching_engine, orderbook, recent_trades, agent_map, tick_trades) -> list`

批量执行非做市商的订单决策（包含 maker 账户更新）。

在 `execute_non_mm_batch` 基础上额外处理：
- 更新 maker 账户
- 记录到 tick_trades 列表
- 计算庄家波动性贡献

**额外参数：**
- `agent_map: dict[int, Agent]` - Agent ID 到 Agent 对象的映射
- `tick_trades: list` - 用于记录本 tick 的成交

**注意事项：**
- 不使用 OpenMP 并行，因为订单执行有顺序依赖
- 做市商不在此函数中处理（逻辑复杂，内联收益低）

#### `execute_non_mm_batch_raw(raw_decisions, matching_engine, orderbook, recent_trades, agent_map, tick_trades, mid_price) -> list`

批量执行非做市商的原始决策数据（内联数量计算逻辑）。

此函数直接接受原始决策数据，避免创建 Python dict 和调用 Python 方法的开销。数量计算逻辑内联到此函数中，避免调用 `agent._calculate_order_quantity()`。

**参数：**
- `raw_decisions: list[(agent, action_type_int, side_int, price, quantity_ratio)]` - 原始决策列表
  - `agent`: Agent 实例（非做市商）
  - `action_type_int`: 动作类型整数（0=HOLD, 1=PLACE_BID, 2=PLACE_ASK, 3=CANCEL, 4=MARKET_BUY, 5=MARKET_SELL）
  - `side_int`: 方向整数（1=买, 2=卖），用于限价单
  - `price`: 订单价格
  - `quantity_ratio`: 数量比例（0.0-1.0）
- `matching_engine`: MatchingEngine 实例
- `orderbook`: OrderBook 实例
- `recent_trades`: deque，用于记录成交
- `agent_map: dict[int, Agent]` - Agent ID 到 Agent 对象的映射
- `tick_trades: list` - 用于记录本 tick 的成交
- `mid_price: float` - 中间价（用于市价单的数量计算）

**返回值：**
- `list[Trade]` - 所有成交记录列表

**内联的数量计算逻辑：**
来自 `Agent._calculate_order_quantity`，根据账户净值、杠杆倍数、当前持仓和数量比例计算订单数量。

**MARKET_SELL 特殊逻辑：**
- 如果持有多仓（position.quantity > 0）：卖出持仓的比例（quantity_ratio）
- 否则（空仓或空头）：开空仓，使用内联数量计算逻辑

**优化点：**
- 避免创建 Python dict（params）
- 避免调用 Python 方法（`_calculate_order_quantity`）
- 所有数量计算内联到 Cython 中
- 保持与 `execute_non_mm_batch_with_maker_update` 相同的成交记录和账户更新逻辑

### fast_math 模块 (fast_math.py)

Numba JIT 加速的高频数学函数模块，提供对数归一化等运算的加速实现。

**特性：**
- 自动检测 Numba 是否可用
- Numba 可用时使用 `@njit(cache=True, fastmath=True)` 装饰器加速
- Numba 不可用时自动降级为纯 NumPy 实现

**提供的函数：**
- `log_normalize_unsigned(arr, scale=10.0)` - 无符号对数归一化：`log10(x + 1) / scale`
- `log_normalize_signed(arr, scale=10.0)` - 带符号对数归一化：`sign(x) * log10(|x| + 1) / scale`

**常量：**
- `HAS_NUMBA: bool` - 标识 Numba 是否可用

### CheckpointLoader (checkpoint_loader.py)

checkpoint 加载器，用于加载训练检查点。

**支持的格式：**
- `checkpoints/ep_*.pkl` 格式（单训练场）
- `checkpoints/parallel_arena_gen_*.pkl` 格式（多训练场）

**主要方法：**
- `load(path)` - 加载 checkpoint，返回统一格式
- `detect_type(path)` - 自动检测 checkpoint 类型

**返回格式：**
```python
{
    "type": CheckpointType,            # SINGLE_ARENA 或 PARALLEL_ARENA
    "tick": int,                       # 当前 tick（多训练场时为 0）
    "episode": int,                    # 当前 episode（多训练场时为 generation）
    "populations": {                   # 种群数据
        AgentType: {
            "generation": int,
            "neat_pop": neat.Population
        },
        ...
    },
    "source_arena_id": int | None,     # 源竞技场 ID（通常为 None）
}
```

**Checkpoint 互相读取兼容性：**

| 源格式 | 目标模式 | 兼容性 |
|-------|---------|-------|
| 单训练场 | 单训练场 (Trainer) | ✓ 完全兼容 |
| 单训练场 | 多训练场 (ParallelArenaTrainer) | ✓ 兼容，generation=episode |
| 单训练场 | 演示模式 | ✓ 通过 CheckpointLoader 转换 |
| 多训练场 | 单训练场 (Trainer) | ✓ 兼容，tick=0, episode=generation |
| 多训练场 | 多训练场 (ParallelArenaTrainer) | ✓ 完全兼容 |
| 多训练场 | 演示模式 | ✓ 通过 CheckpointLoader 转换 |

### FitnessAggregator (arena/fitness_aggregator.py)

适应度汇总器，用于汇总多个竞技场、多个 episode 的适应度数据。

**主要方法：**
- `aggregate_simple_average(arena_fitnesses, episode_counts)` - 简单加权平均

**汇总公式：**
```
avg_fitness = sum(arena_fitness) / total_episodes
```

其中：
- `arena_fitness` 是每个竞技场返回的累积适应度（已累加多个 episode）
- `total_episodes = sum(episode_counts)`

**参数说明：**
- `arena_fitnesses: list[dict[tuple[AgentType, int], np.ndarray]]` - 每个竞技场返回的适应度累积值
  - key: `(agent_type, sub_pop_id)` 元组
  - value: 累积适应度数组，`shape=(pop_size,)`
- `episode_counts: list[int]` - 每个竞技场运行的 episode 数量

**返回值：**
- `dict[tuple[AgentType, int], np.ndarray]` - 平均适应度字典

**示例：**
```python
# 假设有2个竞技场，每个运行了10个episode
arena_fitnesses = [
    {(AgentType.RETAIL, 0): np.array([10.0, 20.0, 30.0])},  # 竞技场0累积值
    {(AgentType.RETAIL, 0): np.array([20.0, 30.0, 40.0])},  # 竞技场1累积值
]
episode_counts = [10, 10]

# 结果：(10 + 20) / 20 = 1.5, (20 + 30) / 20 = 2.5, (30 + 40) / 20 = 3.5
result = FitnessAggregator.aggregate_simple_average(arena_fitnesses, episode_counts)
# result = {(AgentType.RETAIL, 0): np.array([1.5, 2.5, 3.5])}
```

### Population (population.py)

管理特定类型 Agent 的种群，支持从 NEAT 基因组创建 Agent。

**主要功能：**
- 创建和管理 NEAT 种群
- 从基因组创建对应类型的 Agent（散户/高级散户/庄家/做市商）
- 向量化评估适应度
- 执行 NEAT 进化算法
- 重置 Agent 账户状态

**适应度计算公式（相对收益）：**

为消除市场整体方向的影响，使用相对收益率而非绝对收益率：
- 市场平均收益率 = mean((equity - initial) / initial) for all agents
- Agent 相对收益率 = Agent 收益率 - 市场平均收益率

| 种群类型 | 适应度公式 |
|---------|-----------|
| 散户 | 相对收益率 |
| 高级散户 | 相对收益率 |
| 庄家 | 0.5 × 相对收益率 + 0.5 × 波动性贡献排名归一化 |
| 做市商 | 0.5 × 相对收益率 + 0.5 × maker_volume 排名归一化 |

**相对收益的优势：**
- 消除市场整体涨跌的影响
- 即使市场整体下跌，表现优于平均的 Agent 仍获得正向适应度
- 避免正反馈循环导致的单边行情

**庄家波动性贡献说明：**
- 每次作为 taker 成交时，累加价格冲击
- 价格冲击 = |成交后价格 - 成交前价格| / 成交前价格
- 波动性贡献越高，适应度越高（正向激励）
- 排名计算仅限庄家种群内部

**关键方法：**
- `create_agents()` - 从基因组列表创建 Agent（小批量串行，大批量并行）
- `_create_single_agent()` - 创建单个 Agent（线程安全）
- `evaluate()` - 评估种群适应度并排序
- `evolve()` - 执行一代 NEAT 进化，捕获 RuntimeError/CompleteExtinctionException 并在进化失败时自动重置种群，记录完整异常堆栈
- `evolve_with_cached_fitness()` - 使用缓存的适应度进行进化（不重新计算适应度），用于 tick 数不足时打破死循环
- `get_elite_species_avg_fitness()` - 获取最精英 species 的平均适应度（遍历所有 NEAT species，返回平均适应度最高的那个）
- `_cleanup_old_agents()` - 清理旧 Agent 对象，打破循环引用，帮助垃圾回收
- `_cleanup_neat_history()` - 清理 NEAT 种群中的历史数据，防止内存泄漏
- `_reset_neat_population()` - 当 NEAT 进化失败时，创建全新的随机种群
- `reset_agents()` - 重置所有 Agent 账户
- `_get_executor()` - 获取实例级别线程池
- `shutdown_executor()` - 关闭实例级别线程池

**多核并行化：**
- 使用实例级别 ThreadPoolExecutor（8个worker）
- 小批量（<50）串行创建，避免线程池开销
- 大批量并行调用 `Brain.from_genome`，按索引排序保证顺序
- **实例级别线程池设计**：支持多进程架构，每个进程有独立的线程池

### SubPopulationManager (population.py)

通用子种群管理器，支持将任意类型的种群拆分为多个独立子种群进行进化。

**兼容别名：** `RetailSubPopulationManager = SubPopulationManager`（保持向后兼容）

**支持的种群类型：**
- `AgentType.RETAIL` - 散户
- `AgentType.MARKET_MAKER` - 做市商

**设计目标：**
- 减少单个NEAT种群的规模，优化进化性能
- 每个子种群独立进化，共享市场环境
- 清理时间显著减少（从~5s降至~0.04s）

**Agent ID 分配：**
- 每个子种群预留100K ID空间
- RETAIL_SUB_0: 0 ~ 99,999
- RETAIL_SUB_1: 100,000 ~ 199,999
- ...
- RETAIL_SUB_9: 900,000 ~ 999,999
- MARKET_MAKER_SUB_0: 3,000,000 ~ 3,099,999
- MARKET_MAKER_SUB_1: 3,100,000 ~ 3,199,999
- ...

**主要属性：**
- `sub_populations: list[Population]` - 子种群列表
- `sub_population_count: int` - 子种群数量（默认10）
- `agents_per_sub: int` - 每个子种群的Agent数量（1000）
- `agents` - 所有子种群Agent的合并视图

**主要方法：**
- `reset_agents()` - 重置所有子种群的Agent
- `evaluate(current_price)` - 评估所有Agent适应度
- `evolve(current_price)` - 进化所有子种群（串行）
- `evolve_parallel_simple(current_price, max_workers)` - **推荐**：简化并行进化（使用 ThreadPoolExecutor + update_brain）
- `evolve_parallel(current_price, max_workers)` - 轻量级并行进化（使用 ProcessPoolExecutor）
- `evolve_parallel_with_network_params(current_price, worker_pool, sync_genomes)` - 使用持久 Worker 池并行进化，返回网络参数
- `get_all_genomes()` - 获取所有子种群的基因组

**配置：**
- `TrainingConfig.retail_sub_population_count: int = 10` - 子种群数量

**性能提升（Benchmark）：**
- Episode 总耗时: 110s → 78s (-30%)
- RETAIL Agent 创建: 10.3s → 4.2s (-60%)
- RETAIL cleanup: 21.3s → 0s (-100%)

**简化并行进化（evolve_parallel_simple）- 推荐：**

使用 `ThreadPoolExecutor` + `update_brain` 实现的简化并行进化方法：

1. **优化原理**：
   - 使用线程池而非进程池，避免序列化开销
   - 使用 `Agent.update_brain()` 原地更新网络，避免重建 Agent 对象
   - 虽然受 GIL 限制，但 NEAT 的 C 扩展操作可以释放 GIL

2. **实现流程**：
   ```python
   # 1. 串行评估所有子种群的适应度
   for pop in sub_populations:
       agent_fitnesses = pop.evaluate(current_price)
       for agent, fitness in agent_fitnesses:
           genome.fitness = fitness

   # 2. 使用线程池并行进化
   with ThreadPoolExecutor(max_workers=10) as executor:
       futures = [executor.submit(evolve_sub_pop, sp) for sp in sub_populations]
       for future in as_completed(futures):
           future.result()
   ```

3. **性能数据（10个子种群，每个1000基因组）**：
   - 进化总耗时：~37s（vs 串行 ~27s）
   - Agent 更新：~4.2s（vs create_agents ~10.3s，-60%）
   - Cleanup：0s（vs 原来 ~21s，-100%）

**轻量级并行进化实现：**

`evolve_parallel()` 使用轻量级序列化实现真正的多进程并行进化：

1. **序列化格式**（模块级函数 `_serialize_genomes`/`_deserialize_genomes`）：
   - 只序列化基因组核心数据：key, fitness, nodes, connections
   - 使用紧凑的元组列表格式，减少 pickle 开销
   - 单个子种群（1000基因组）序列化大小：~25 MB

2. **进化流程**：
   - 主进程：评估适应度 → 序列化基因组 → 启动子进程
   - 子进程：重建 NEAT Population → 执行一代进化 → 序列化返回
   - 主进程：反序列化 → 更新 Agent Brain

3. **性能分析（10个子种群，每个1000基因组，1086连接/基因组）**：
   - 序列化总时间：~3s
   - pickle/unpickle：~5s
   - 反序列化总时间：~5.5s
   - 数据传输量：~246 MB
   - 总耗时：~44s（vs 串行 ~33s）

4. **当前限制**：
   - 由于每个基因组有大量连接（~1086个），序列化开销较大
   - 进程间传输 246MB 数据有显著开销
   - 当前配置下并行进化略慢于串行

5. **优化方向**：
   - 减少 NEAT 配置中的初始连接数
   - 使用共享内存（multiprocessing.shared_memory）避免数据拷贝
   - 使用 numpy 数组进一步压缩序列化格式

**网络参数传输优化（evolve_parallel_with_network_params）：**

使用 `PersistentWorkerPool` + 网络参数传输实现更高效的并行进化：

1. **优化原理**：
   - Worker 进程持久化，避免每次创建新进程的开销
   - 只传输适应度数组（~40KB），不传输基因组数据（~25MB）
   - Worker 进程返回网络参数，主进程使用 `Brain.update_from_network_params()` 直接创建网络

2. **性能数据（10个子种群，每个1000基因组）**：
   - Worker 进化 + 传输：7.77s
   - 反序列化基因组：31.88s（仍需维护 NEAT 状态）
   - 解包网络参数：0.05s
   - 更新 Agent Brain：0.45s
   - 总耗时：~40s（vs 原始 ~80s，提升 1.68x）

3. **使用方式**：
   ```python
   # 创建持久 Worker 池
   worker_pool = PersistentWorkerPool(
       num_workers=10,
       neat_config_path=neat_config_path,
       pop_size=1000,
   )

   # 首次调用需同步基因组
   manager.evolve_parallel_with_network_params(
       current_price=10000.0,
       worker_pool=worker_pool,
       sync_genomes=True,  # 首次需要同步
   )

   # 后续调用
   manager.evolve_parallel_with_network_params(
       current_price=10000.0,
       worker_pool=worker_pool,
       sync_genomes=False,
   )

   # 关闭 Worker 池
   worker_pool.shutdown()
   ```

4. **剩余瓶颈**：
   - 基因组反序列化仍需 31.88s（Python 对象创建）
   - 需维护 NEAT 状态用于检查点
   - 未来可考虑延迟反序列化或仅在需要时反序列化

### PersistentWorkerPool (population.py)

持久 Worker 进程池，维护多个子进程，每个子进程维护自己的 NEAT 种群。

**主要方法：**
- `evolve_all(fitnesses_list)` - 并行进化所有 Worker 的种群，返回基因组数据
- `evolve_all_return_params(fitnesses_list)` - 并行进化并返回基因组数据 + 网络参数
- `set_all_genomes(genomes_list)` - 同步所有 Worker 的基因组数据
- `get_all_genomes()` - 获取所有 Worker 的基因组数据
- `shutdown()` - 关闭所有 Worker

**Worker 命令：**
- `evolve` - 执行进化，返回新基因组（NumPy 格式）
- `evolve_return_params` - 执行进化，返回基因组 + 网络参数
- `set_genomes` - 设置基因组数据
- `get_genomes` - 获取基因组数据
- `shutdown` - 关闭 Worker

### WorkerConfig (population.py)

Worker 配置数据类（dataclass），用于配置 `MultiPopulationWorkerPool` 中的单个 Worker。

**属性：**
- `agent_type: AgentType` - Agent 类型
- `sub_pop_id: int` - 子种群 ID
- `neat_config_path: str` - NEAT 配置文件路径
- `pop_size: int` - 种群大小

### MultiPopulationWorkerPool (population.py)

多种群统一 Worker 池，管理多个不同配置的 Worker，每个 Worker 可以有不同的 NEAT 配置和种群大小。

**设计特点：**
- 每个 Worker 使用独立的 `cmd_queue`，但共享统一的 `result_queue`
- Worker ID 编码为 `(agent_type, sub_pop_id)` 元组
- 支持一次性发送/收集所有进化结果（真正并行）

**主要方法：**
- `evolve_all_parallel(fitness_map, sync_genomes)` - 同时进化所有 Worker 的种群
  - 参数 `fitness_map: dict[(AgentType, int), np.ndarray]` - 每个 Worker 的适应度数组
  - 返回 `dict[(AgentType, int), (genome_data, network_params_data)]`
- `set_genomes(genomes_map)` - 同步基因组到所有 Worker
- `shutdown()` - 关闭所有 Worker

**使用示例：**
```python
# 创建 Worker 配置
configs = [
    WorkerConfig(AgentType.RETAIL, 0, "config/neat_retail.cfg", 1000),
    WorkerConfig(AgentType.RETAIL, 1, "config/neat_retail.cfg", 1000),
    WorkerConfig(AgentType.MARKET_MAKER, 0, "config/neat_market_maker.cfg", 100),
]

# 创建多种群 Worker 池
pool = MultiPopulationWorkerPool("config", configs)

# 并行进化
fitness_map = {
    (AgentType.RETAIL, 0): np.array([...]),
    (AgentType.RETAIL, 1): np.array([...]),
    (AgentType.MARKET_MAKER, 0): np.array([...]),
}
results = pool.evolve_all_parallel(fitness_map)

# 关闭
pool.shutdown()
```

### Trainer (trainer.py)

管理整体训练流程，协调种群和撮合引擎。

**主要功能：**
- 初始化训练环境（创建种群、撮合引擎）
- 管理训练生命周期（tick、episode）
- 处理成交和强平
- 保存/加载检查点
- 支持暂停/恢复/停止控制

**关键方法：**
- `setup()` - 初始化训练环境，创建种群、撮合引擎、ADL 管理器，初始化鲶鱼（如启用），初始化 EMA 平滑价格
- `_init_ema_price()` - 初始化 EMA 平滑价格（在 episode 开始时调用）
- `_update_ema_price()` - 更新 EMA 平滑价格（每 tick 调用）
- `_aggregate_tick_trades()` - 聚合本 tick 的成交量和成交额（符号+总量方式）
- `_calculate_catfish_initial_balance()` - 计算鲶鱼初始资金（做市商杠杆后资金 - 其他物种杠杆后资金）/ 3
- `_register_all_agents()` - 注册所有 Agent 的费率到撮合引擎
- `_build_agent_map()` - 构建 Agent ID 到 Agent 对象的映射表（O(1) 查找）
- `_build_execution_order()` - 构建 Agent 执行顺序列表（初始化时收集所有 Agent，实际执行时每 tick 随机打乱）
- `_cancel_agent_orders()` - 撤销指定 Agent 的所有挂单（做市商撤多单，普通 Agent 撤单个挂单）
- `_execute_liquidation_market_order()` - 执行强平市价单，提交市价单平仓，若市价单无法完全成交则返回剩余数量（调用前必须先撤销挂单）
- `_execute_adl()` - 执行 ADL 自动减仓，在循环中处理 ADL 成交（使用预计算的候选清单、更新账户、更新 position_qty）
- `_check_catfish_liquidation()` - 检查鲶鱼强平，鲶鱼强平后设置 `_catfish_liquidated` 标志（触发 episode 结束）
- `_execute_catfish_liquidation()` - 执行鲶鱼强平市价单
- `_update_pop_total_counts()` - 更新各种群总数（在 setup/evolve/load_checkpoint 后调用）
- `_should_end_episode_early()` - O(1) 检查是否满足提前结束条件，返回 `tuple[str, AgentType | None] | None`
  - 触发条件1：任意种群存活少于初始值的 1/4，返回 `("population_depleted", agent_type)`
  - 触发条件2：订单簿只有单边挂单（只有 bid 或只有 ask），返回 `("one_sided_orderbook", None)`
  - 不触发时返回 `None`
- `_compute_normalized_market_state()` - 向量化计算归一化市场状态，使用 EMA 平滑后的 mid_price
- `run_tick()` - 执行单个 tick，鲶鱼行动在Agent决策之前执行
- `run_episode()` - 运行完整 episode（重置、运行、进化），若满足提前结束条件则提前结束
- `train()` - 主训练循环
- `save_checkpoint()` / `load_checkpoint()` - 检查点管理，支持新旧两种格式
- `find_latest_checkpoint(checkpoint_dir)` - 静态方法，查找指定目录下最新的检查点文件（按 episode 数字排序）
- `_serialize_population_data(pop)` - 序列化种群数据，支持 Population 和 SubPopulationManager
- `_load_population_data(pop, pop_data, agent_type)` - 加载种群数据，自动检测格式并迁移
- `_migrate_old_checkpoint_to_sub_populations(manager, pop_data)` - 将旧格式 checkpoint 迁移到子种群格式

**Checkpoint 格式兼容性：**

支持多种 checkpoint 格式，加载时自动检测并迁移：

**压缩格式（自动检测）：**
- gzip 压缩格式（新）：使用 gzip 压缩 pickle 数据，显著减小文件体积
- 普通 pickle 格式（旧）：向后兼容，自动检测 gzip 魔数（0x1f 0x8b）

**保存时优化：**
- 保存前自动清理 NEAT 历史数据（调用 `_cleanup_neat_history()`），进一步减小体积
- 使用 gzip 压缩保存

**种群数据格式：**

1. **新格式**（SubPopulationManager）：
   ```python
   {
       "is_sub_population_manager": True,
       "generation": int,
       "sub_population_count": int,
       "sub_populations": [
           {"neat_pop": neat_pop, "generation": int},
           ...
       ]
   }
   ```

2. **旧格式**（单个 RETAIL 种群）：
   ```python
   {
       "generation": int,
       "neat_pop": neat_pop,
   }
   ```

**自动迁移**：加载旧格式 checkpoint 时，自动将单个大 neat_pop 拆分成多个子种群：
- 按顺序将基因组分配到各子种群
- 每个子种群重新进行物种划分
- 支持不同子种群数量的配置变更
- `get_price_stats()` - 获取价格统计（高/低/最终价格）
- `get_population_stats()` - 获取种群统计（适应度、淘汰数、最精英 species 平均适应度等）

**性能优化：**
- 使用 `deque(maxlen=100)` 自动管理成交记录，避免列表切片开销
- 使用 `agent_map` 映射表实现 O(1) Agent 查找
- 使用 `_pop_total_counts` 和 `_pop_liquidated_counts` 计数器实现 O(1) 种群淘汰检查，避免每 tick 遍历
- 使用 `agent_execution_order` 预构建执行顺序，合并决策/执行和强平检查循环
- 向量化市场状态计算，使用 NumPy 数组操作替代 Python 循环

**价格稳定机制（EMA 平滑）：**
- 使用 EMA（指数移动平均）平滑 mid_price，减缓价格变化传导速度
- 公式：`smooth_mid_price = α × current_mid_price + (1-α) × prev_smooth_mid_price`
- 参数 `ema_alpha`（默认 0.1）可通过 `MarketConfig.ema_alpha` 配置
- Agent 报价、归一化计算、强平检查使用 `smooth_mid_price`（保持一致性）
- ADL 计算使用实时 `last_price`（实际市场操作需要准确价格）
- 解决了价格在短时间内极端波动的正反馈循环问题

**多核并行化优化：**
- 种群初始化：串行创建种群，但每个种群内部的 Agent 创建是并行的（Agent维度并行）
- `_evolve_populations_parallel()` - 使用 `MultiPopulationWorkerPool` 真正并行进化所有种群（16个 Worker 并行）
  - RETAIL: 10 个子种群
  - RETAIL_PRO: 1 个种群
  - WHALE: 1 个种群
  - MARKET_MAKER: 4 个子种群
  - **延迟反序列化优化**：默认只更新网络参数，不重建完整 genome 对象
  - **性能提升**：~20秒/episode（比旧版本 40秒快一倍）
- `_update_population_from_worker(pop, genome_data, network_params, deserialize_genomes=False)` - 从 Worker 结果更新种群
  - `deserialize_genomes=False`（默认）：只调用 `brain.update_network_only(params)`，不反序列化 genome
  - `deserialize_genomes=True`：完整反序列化 genome，用于保存 checkpoint 时
- `_batch_decide_serial()` - Agent 决策阶段**串行执行**（经测试，由于 GIL 限制，串行比并行快 ~30%）
- `_check_liquidations_vectorized()` - 向量化强平检查（NumPy 批量计算）
- 决策阶段串行，执行阶段串行（保证订单簿一致性，避免 GIL 竞争）
- 线程池惰性初始化，`stop()` 时自动清理
- Population 类使用实例级别线程池（8个worker）处理 Agent 创建，支持多进程架构

**统一 Worker 池架构：**
- `_unified_worker_pool: MultiPopulationWorkerPool` - 统一管理所有 16 个 Worker 进程
- `_worker_pool_synced: bool` - Worker 池是否已同步基因组
- 在 `setup()` 中创建 Worker 池，在 `stop()` 中关闭
- 在 `load_checkpoint()` 和 `load_checkpoint_data()` 后重置同步标志

**决策阶段优化说明：**
- 原 `_batch_decide_parallel()` 使用 16 线程并行决策，但由于 Python GIL 限制，实际并行效果差
- 经测试：10300 Agent 决策，并行耗时 ~200ms，串行耗时 ~110ms
- 现已改为 `_batch_decide_serial()` 串行处理，性能提升约 30%
- 神经网络前向传播（`FastFeedForwardNetwork._forward_pass`）虽已释放 GIL，但 Python 层调用开销仍受 GIL 限制

**串行执行阶段优化（P0）：**

串行执行阶段占单个 tick 耗时的 56.8%（约 94ms），是最大的性能瓶颈。已实施以下优化：

1. **跳过 HOLD 动作**（`_batch_decide_with_cache`/`_batch_decide_openmp`）：
   - 在决策结果转换时直接过滤掉 HOLD 动作（`action_type_int == 0`）
   - 减少串行执行阶段需要处理的 agent 数量
   - 预期减少 20-40% 的迭代次数（取决于 HOLD 占比）

2. **跳过无效 CANCEL 动作**：
   - 如果 agent 没有挂单（`pending_order_id is None`），直接跳过
   - 避免无效的订单簿操作

3. **本地变量缓存**：
   - 缓存 `agent_map.get`、`recent_trades.append`、`tick_trades.append` 方法引用
   - 减少属性查找开销（Python 中每次 `.` 操作都是字典查找）

4. **内联优化**：
   - 将 `maker_id` 计算和 `is_buyer` 判断内联
   - 减少函数调用开销

**优化效果**（基准测试）：
- P0 优化前：166.31ms/tick（并行决策 53.79ms，串行执行 94.52ms）
- P0 优化后：162.93ms/tick（并行决策 51.24ms，串行执行 94.59ms）
- P0 提升约 2%（主要来自减少决策结果数量）

**串行执行内联优化（P1）：**

对散户/高级散户/庄家的 `execute_action` 进行内联优化，跳过函数调用包装：

1. **直接调用撮合引擎**：
   - 跳过 `execute_action` → `_place_limit_order` → `process_order` 的调用链
   - 直接在循环中调用 `matching_engine.process_order`

2. **内联订单 ID 生成**：
   - 跳过 `_generate_order_id()` 函数调用
   - 直接计算：`(agent_id << 32) | order_counter`

3. **内联成交处理**：
   - 跳过 `_process_trades()` 函数调用
   - 直接调用 `account.on_trade()`

4. **做市商保持原逻辑**：
   - 做市商逻辑复杂（需撤销多单、挂多单）
   - 做市商数量少（400个），内联收益有限

**P1 优化效果**（基准测试）：
- P1 优化前：162.93ms/tick（串行执行 94.59ms）
- P1 优化后：151.82ms/tick（串行执行 86.78ms）
- **总体提升约 6.8%，串行执行提升 8.3%**

**累计优化效果**：
- 原始基准：166.31ms/tick
- P0+P1 优化后：151.82ms/tick
- **累计提升约 8.7%**

**Cython 撮合引擎优化（P2）**：

核心撮合逻辑已迁移到 Cython 实现（`src/market/matching/fast_matching.pyx`）：

1. **实现内容**：
   - `FastTrade` cdef class：Cython 版成交记录，使用 C 类型属性
   - `fast_match_orders()` cpdef 函数：核心撮合逻辑，内联价格检查

2. **优化点**：
   - 使用 C 类型变量减少 Python 对象开销
   - 内联限价单/市价单价格检查逻辑（避免 callable 调用）
   - 预计算 taker 费率到循环外

3. **性能提升**：
   - 串行执行时间从 ~99ms 降到 ~91ms，减少约 **8ms (8%)**
   - 总 tick 时间从 ~165ms 降到 ~161ms

**进一步优化方向**：
- 批量订单簿操作
- 减少 Order 对象创建（对象池）

**BatchNetworkCache 缓存优化（OpenMP）：**

为彻底解决 GIL 竞争问题，实现了 `BatchNetworkCache` 缓存机制，将网络数据提取从每次决策移到进化后一次性完成。

**核心组件** (`_cython/batch_decide_openmp.pyx`)：
- `BatchNetworkCache` - Cython cdef class，预分配内存和缓存网络数据
- 支持三种缓存类型：`CACHE_TYPE_RETAIL`(0)、`CACHE_TYPE_FULL`(1)、`CACHE_TYPE_MARKET_MAKER`(2)

**缓存类方法：**
- `update_networks(networks)` - 从 Python 网络对象提取数据到 C 结构（进化后调用一次）
- `decide(inputs, orderbooks, mid_prices)` - 使用缓存数据执行批量决策（每 tick 调用）
- `decide_multi_arena_direct(states, markets, indices, return_array)` - 跨竞技场批量推理，做市商返回解析后的订单数据
- `is_valid()` - 检查缓存是否有效
- `clear()` - 清空缓存

**做市商解析优化（P3）：**

将做市商神经网络输出解析从 Python 端移到 Cython 端，直接在 `nogil` 块中完成订单生成：

1. **新增数据结构**：
   - `MarketMakerOrdersResult` - 存储解析后的订单数据（最多10个买单+10个卖单）
   - 预分配缓冲区 `multi_arena_mm_orders`，避免每次调用分配内存

2. **新增函数**：
   - `_calculate_skew_factor()` - 计算仓位倾斜因子（与 Python 端逻辑一致）
   - `_apply_position_skew_nogil()` - 应用仓位倾斜到权重（nogil 兼容）
   - `_parse_market_maker_full_single()` - 单个做市商完整解析（直接生成订单）
   - `batch_parse_market_maker_full_multi_market_nogil()` - 批量做市商解析（OpenMP 并行）

3. **返回格式变更**：
   - `return_array=False`：返回 `list[dict]`，dict 包含 `{"bid_orders": [...], "ask_orders": [...]}`
   - `return_array=True`：返回 `np.ndarray` shape=(num_agents, 42)

4. **性能提升**：
   - 做市商解析耗时从 ~456ms/tick 降到 ~6.7ms/tick
   - **加速比：约 68x**

**Trainer 集成：**
- `_init_network_caches()` - 在 `setup()` 末尾初始化缓存
- `_update_network_caches()` - 在 `evolve()` 后、`load_checkpoint()` 后调用
- `_batch_decide_with_cache()` - 使用缓存路径执行决策

**性能提升：**
- 优化前：~823ms/tick（每次决策都从 Python 对象提取网络数据）
- 优化后：~160ms/tick（网络数据只在进化后提取一次，决策时直接使用 C 结构）
- **加速比：约 5.1x**

**OpenMP 线程数配置：**
- 通过 `TrainingConfig.openmp_threads` 配置（默认 8）
- 经基准测试，8 线程为最优值，过多线程反而更慢：
  - 1 线程：8.0 ms
  - 8 线程：3.5 ms（最优，2.3x 加速）
  - 32 线程：26.4 ms（比单线程更慢）
- 原因：线程调度开销、内存带宽瓶颈、缓存竞争

**缓存更新时机：**
1. `setup()` - 初始化时
2. `evolve()` - 每次进化后（新基因组 → 新网络）
3. `load_checkpoint()` - 加载检查点后

## 训练流程

1. **初始化阶段** (`setup`)
   - 创建四个种群（散户/高级散户/庄家/做市商）
   - 创建撮合引擎
   - 创建 ADL 管理器
   - 注册所有 Agent 的费率到撮合引擎
   - 构建 Agent 映射表和执行顺序
   - 初始化 EMA 平滑价格（使用 initial_price）
   - 做市商建立初始流动性

2. **Episode 循环** (`run_episode`)
   - 重置所有 Agent 账户
   - 重置鲶鱼状态和强平标志
   - 重置市场状态（包括重置 EMA 平滑价格、价格历史和 tick 历史数据）
   - 重置各种群淘汰计数和重入保护集合
   - 运行 episode_length 个 tick
   - **提前结束条件**：
     - 鲶鱼被强平（立即结束 episode）
     - 任意种群存活少于初始值的 1/4（确保有足够的幸存者用于 NEAT 进化）
     - 订单簿只有单边挂单（确保市场流动性正常）
   - **适应度累积与进化**：
     - 每个 episode 结束时累积当前适应度到各 Agent 的 `genome.fitness`
     - 累积使用平均值：`genome.fitness = 累积适应度总和 / 累积次数`
     - 每 N 个 episode 进化一次（N 由 `TrainingConfig.evolution_interval` 配置，默认 10）
     - 进化时应用累积的平均适应度，然后执行 NEAT 选择和繁殖
     - 进化后清空累积数据，重置 `_episodes_since_evolution` 计数器
   - 进化后重新注册新 Agent 的费率，重建映射表和执行顺序

3. **Tick 执行** (`run_tick`)

   **时序设计**：
   - **Tick 1**：只展示做市商初始挂单后的市场状态，其他 agent 不行动
   - **Tick 2+**：Agent 的下单操作影响的是下一个 tick，确保强平检查和数据采集使用同一价格

   **Tick 1（做市商初始化展示 tick）**：
   - 只记录价格历史和 tick 数据，不执行任何 agent 行动
   - 用于 UI 模式展示做市商初始挂单后形成的市场状态

   **Tick 2+（正常行动 tick）**：
   - **Tick 开始（强平处理分三阶段）**：
     - 保存 tick 开始时的价格到 `tick_start_price`（供数据采集使用）
     - **阶段1（统一撤单）**：遍历所有 Agent 检查强平条件，收集需要淘汰的 Agent，**统一撤销这些 Agent 的所有挂单**
     - **阶段2（统一市价单平仓）**：遍历需要淘汰的 Agent，执行市价单平仓（不触发 ADL），收集需要 ADL 的 Agent
     - **阶段3（用最新价格计算 ADL 候选并执行）**：获取订单簿最新价格，计算 ADL 候选清单，执行 ADL
   - **Tick 过程（原子动作随机执行）**：
     - 向量化计算归一化市场状态
     - **随机打乱 Agent 执行顺序**（每 tick 完全随机，模拟真实环境）
     - Agent 并行决策
     - **收集所有原子动作**：将鲶鱼动作和 Agent 动作拆分为原子操作
     - **随机打乱原子动作顺序**：`random.shuffle(atomic_actions)`
     - **逐个执行原子动作**：通过 `_execute_atomic_action()` 执行
     - 记录成交到 `recent_trades`，更新 maker 账户
   - **Tick 结束**：
     - 下单产生的价格变动效果在下个 tick 被感知
     - 记录当前价格到 `_price_history`（鲶鱼决策使用，最多保留1000个历史价格）
     - 记录 tick 历史数据（价格、成交量、成交额，最多保留100条）
     - 检查鲶鱼强平（鲶鱼强平则立即结束 episode）
     - 数据采集使用 `tick_start_price` 计算资产，与强平检查一致

   **原子动作机制**：

   为了模拟真实市场环境，所有订单操作被拆分为原子动作后随机打乱执行，避免分阶段执行导致的流动性枯竭问题。

   **原子动作类型（AtomicActionType）**：
   - `CANCEL (1)` - 撤单
   - `LIMIT_BUY (2)` - 限价买单
   - `LIMIT_SELL (3)` - 限价卖单
   - `MARKET_BUY (4)` - 市价买单
   - `MARKET_SELL (5)` - 市价卖单

   **原子动作数据结构（AtomicAction）**：
   ```python
   @dataclass
   class AtomicAction:
       action_type: AtomicActionType  # 动作类型
       agent_id: int                  # Agent 或鲶鱼 ID
       order_id: int = 0              # 撤单时的订单 ID
       price: float = 0.0             # 限价单价格
       quantity: int = 0              # 订单数量
       is_market_maker: bool = False  # 是否为做市商
       is_catfish: bool = False       # 是否为鲶鱼
       agent_ref: Any = None          # Agent 或 CatfishBase 引用
   ```

   **动作收集规则**：
   - **吃单鲶鱼**：direction != 0 时，收集一个市价单动作
   - **做市鲶鱼**：direction == 0 时，收集撤旧单动作 + 新挂单动作
   - **做市商**：收集所有撤单动作 + 所有新挂单动作
   - **非做市商限价单**：收集撤旧单动作（如有）+ 新挂单动作
   - **非做市商撤单**：收集撤单动作
   - **非做市商市价单**：收集市价单动作

   **执行方法（`_execute_atomic_action`）**：
   - 检查 Agent/鲶鱼是否已被强平，已强平则跳过
   - 根据动作类型执行对应操作
   - 更新 taker 账户和 maker 账户
   - 记录成交到 `recent_trades` 和 `tick_trades`
   - 庄家市价单/限价单成交时计算波动性贡献

## 强平与淘汰机制（爆仓即淘汰）

**强平即淘汰（Liquidation = Elimination）**：保证金率低于维持保证金率时触发，Agent 直接被淘汰

**Tick 开始时的三阶段强平处理**：
- **阶段1（统一撤单）**：遍历所有 Agent 检查强平条件，收集需要淘汰的 Agent，调用 `_cancel_agent_orders()` 统一撤销这些 Agent 的所有挂单
- **阶段2（统一市价单平仓）**：遍历需要淘汰的 Agent，调用 `_execute_liquidation_market_order()` 执行市价单平仓（不触发 ADL），标记淘汰并穿仓兜底，收集需要 ADL 的 Agent
- **阶段3（用最新价格计算 ADL 候选并执行）**：获取订单簿最新价格（强平市价单执行后的价格），用最新价格计算 ADL 候选清单，执行 ADL

**设计原因**：
- 先统一撤单可防止被淘汰的 Agent 在平仓过程中作为 maker 被成交，导致仓位增加
- **用最新价格计算 ADL 候选**：强平市价单执行后订单簿价格已变化，用最新价格计算候选确保 ADL 候选在当前价格下确实盈利，避免 candidate 因 ADL 出现负 balance

**`_execute_liquidation_market_order()` 执行流程**（阶段2）：
1. 获取 Agent 当前持仓方向和数量
2. 创建市价平仓单，调用撮合引擎处理
3. 成交后更新 Agent 账户（taker）和 maker 账户
4. 返回剩余未平仓数量和持仓方向

**ADL（自动减仓）**：市价强平无法完全成交时触发（阶段3）
1. **获取最新价格**：强平市价单执行后，订单簿价格已变化，获取最新价格
2. **用最新价格计算候选清单**：遍历所有存活 Agent，用最新价格计算 ADL 分数，筛选盈利候选者，按多头/空头分类并排序
3. **将鲶鱼加入候选清单**：遍历所有未被强平的鲶鱼，计算其 ADL 分数（使用庄家的杠杆率），盈利的鲶鱼按持仓方向加入对应候选清单
4. 使用候选清单（`_adl_long_candidates` 或 `_adl_short_candidates`）
   - 被强平方是多头 → 使用空头候选清单
   - 被强平方是空头 → 使用多头候选清单
5. 在 `_execute_adl()` 中直接循环处理：依次与候选对手方以最新市场价格成交，直至剩余仓位清零
6. 通过 `account.on_adl_trade()` 更新双方账户（含穿仓兜底）
7. **更新候选清单的 position_qty**：确保后续 ADL 不会重复使用已减掉的仓位
8. **兜底处理**：如果 ADL 无法完全清零仓位，强制清零被淘汰者的仓位

**重入保护与 ADL 候选管理**：
- 使用 `_eliminating_agents` 集合跟踪正在强平/淘汰过程中的 Agent
- 使用 `_adl_long_candidates` 和 `_adl_short_candidates` 存储预计算的 ADL 候选清单
  - 已提前筛选：未淘汰、本 tick 不会被淘汰、有持仓、盈利
  - 已计算 ADL 分数并排序
  - **包含 Agent 和鲶鱼**：盈利的鲶鱼也会被加入候选清单
  - ADL 成交后动态更新 `position_qty`，避免重复使用已减掉的仓位
- 防止在递归 ADL 过程中同一 Agent 被多次处理
- 防止正在强平的 Agent 作为 maker 更新仓位（导致仓位增加）
- 强平完成后从集合中移除

**淘汰后状态**：
- 被淘汰的 Agent 在本轮 episode 剩余时间内无法执行任何动作（`run_tick` 跳过，`execute_action` 返回空列表）
- 在下一轮 episode 开始时，`reset_agents()` 会重置 `is_liquidated` 标志

## 依赖关系

- `src.bio.agents` - Agent 类
- `src.config.config` - 配置类
- `src.core.log_engine` - 日志系统
- `src.market.adl` - ADL 管理器
- `src.market.matching` - 撮合引擎
- `src.market.orderbook` - 订单簿

## NEAT 配置

不同 Agent 类型使用不同的 NEAT 配置文件（由 Population 自动选择）：
- `config/neat_retail.cfg` - 散户（127 个输入节点，8 个输出节点）
- `config/neat_retail_pro.cfg` - 高级散户（907 个输入节点，8 个输出节点）
- `config/neat_whale.cfg` - 庄家（907 个输入节点，8 个输出节点）
- `config/neat_market_maker.cfg` - 做市商（934 个输入节点，21 个输出节点）

散户只能看到买卖各10档订单簿和最近10笔成交，高级散户和庄家可以看到完整的100档订单簿和100笔成交。

**关键配置参数（防止种群灭绝）：**
- `reset_on_extinction = True` - 当所有物种灭绝时自动重置种群
- `survival_threshold = 0.5` - 每个物种的前 50% 个体可以参与繁殖
- `compatibility_threshold = 4.0` - 物种兼容性阈值，较高的值减少物种数量
- `elitism = 1` - 每个物种保留 1 个最优个体不变
- `species_elitism = 2` - 保留 2 个最优物种不被移除

**注意：**
- NEAT 配置文件中的 `pop_size` 会被 `AgentConfig.count` 动态覆盖，即种群数量由脚本中的配置决定，而非 NEAT 配置文件。
- **Agent ID 唯一性**：每个种群类型使用不同的 ID 偏移量（散户=0，高级散户=1M，庄家=2M，做市商=3M），确保不同种群的 agent_id 全局唯一，避免订单 ID 冲突。

## 启动脚本

训练脚本位于 `scripts/` 目录：
- `train_noui.py` - 无 UI 高性能训练模式
- `train_ui.py` - 带 UI 训练模式

**自动恢复检查点**：`train_noui.py` 和 `train_ui.py` 启动时会自动查找 `checkpoints/` 目录下最新的检查点（按 episode 数字排序），如果存在则自动恢复训练。可通过 `--resume` 参数手动指定检查点覆盖此行为。

### train_noui.py 使用方法

```bash
# 基本训练（自动从最新检查点恢复，若存在）
python scripts/train_noui.py --episodes 100

# 无限训练模式（Ctrl+C 停止）
python scripts/train_noui.py --infinite

# 自定义参数
python scripts/train_noui.py --episodes 500 --episode-length 1000 --checkpoint-interval 50

# 从指定检查点恢复训练
python scripts/train_noui.py --resume checkpoints/ep_50.pkl --episodes 100
```

命令行参数：
- `--episodes`: 训练的 episode 数量（默认: 4000），与 `--infinite` 互斥
- `--infinite`: 无限训练模式，直到手动中断（Ctrl+C），与 `--episodes` 互斥
- `--episode-length`: 每个 episode 的 tick 数量（默认: 100）
- `--checkpoint-interval`: 检查点保存间隔（默认: 10，0 表示不保存）
- `--resume`: 从指定检查点恢复训练
- `--config-dir`: 配置文件目录（默认: config）
- `--log-dir`: 日志目录（默认: logs）
- `--catfish`: 启用鲶鱼机制
- `--catfish-fund-multiplier`: 鲶鱼资金倍数（默认: 3.0）

**检查点格式**：`checkpoints/ep_*.pkl`

## 鲶鱼机制

鲶鱼（Catfish）是规则驱动的市场参与者，用于增加市场波动性。

**特点：**
- 不使用神经网络，规则驱动
- **资金计算**：每条鲶鱼初始资金 = (做市商杠杆后资金 - 其他物种杠杆后资金) / 3，支持多模式（3条鲶鱼同时运行）和单模式（1条鲶鱼）
- 参与强平和 ADL 机制（作为盈利方可作为 ADL 候选）
- 鲶鱼被强平后 Episode 立即结束
- 下单量按盘口计算（吃掉前3档），不按自身资金计算
- 手续费为 0（maker 和 taker）
- 在所有 Agent 之前行动，强平检查在所有 Agent 之后
- 使用庄家的杠杆率和维持保证金率
- 使用 `_price_history`（最多1000个历史价格）进行决策

**行为模式：**
- `trend_creator`：趋势创造者，Episode 开始时随机选择方向，整个 Episode 保持该方向持续操作
- `mean_reversion`：逆势操作，均值回归
- `random`：随机买卖

**配置：**
通过 `CatfishConfig` 配置，包括触发阈值、模式选择、多模式开关等参数。

详见：`src/market/catfish/CLAUDE.md`

## 内存管理

为防止内存泄漏，实现了以下机制：

**1. NEAT 种群历史清理 (`Population._cleanup_neat_history()`)**

每次进化后彻底清理 NEAT 库内部积累的历史数据：
- `genome_to_species` 字典：只保留当前代基因组的映射
- `species.members`：清理已不存在的基因组引用
- `stagnation.species_fitness`：清空物种适应度历史
- `reproduction.ancestors`：清空祖先引用
- `reporters` 统计数据：只保留最近 5 代

**2. Agent 对象清理 (`Population._cleanup_old_agents()`)**

在创建新 Agent 之前，显式打破循环引用：
- 清理 `Brain.network` 内部状态（node_evals, values 等）
- 置空 `Brain.genome`, `Brain.network`, `Brain.config`
- 置空 `Agent.brain`, `Agent.account`
- 多次调用 `gc.collect()` 处理循环引用

**3. 进化后垃圾回收 (`Trainer._evolve_populations_parallel()`)**

所有种群进化完成后，统一调用三代 GC（gc.collect(0/1/2)）和 malloc_trim()，确保释放旧对象。

**GC 策略优化**：
- `Population.evolve()` 和 `Population.evolve_with_cached_fitness()` 正常流程中不再调用 gc.collect()
- 异常处理中保留 gc.collect() + malloc_trim()（异常后清理）
- 由调用方 `_evolve_populations_parallel()` 统一负责 GC，减少 GC 调用次数

### 死锁防护

**1. 多线程导入锁问题修复**

`FastFeedForwardNetwork.create()` 原先在方法内部导入 `fast_graphs` 模块，在多线程环境下（`create_agents()` 使用线程池并行创建 Agent）可能导致 Python 导入锁死锁。已将导入移到模块级别（`neat/_cython/fast_network.pyx`）。

**2. 线程池超时机制**

- `create_agents()`: 添加 120 秒整体超时，防止并行创建 Agent 时死锁
- `_batch_decide_parallel()`: 添加 60 秒超时，防止并行决策时死锁
- 超时后会取消未完成的任务并记录错误日志

### 内存泄漏排查

如果仍然出现内存增长，可能的原因：
1. NEAT 配置文件中启用了过多的 reporter
2. 检查是否有其他未被清理的循环引用
