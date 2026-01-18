# Training 模块

## 模块概述

训练模块负责管理 NEAT 进化训练流程，包括种群管理、训练协调、检查点加载和高性能数学计算。

## 文件结构

- `__init__.py` - 模块导出
- `population.py` - 种群管理类（Population、SubPopulationManager、Worker 池）
- `trainer.py` - 训练器类（Trainer、原子动作机制）
- `checkpoint_loader.py` - checkpoint 加载器（统一接口）
- `fast_math.py` - Numba JIT 加速的数学函数（对数归一化等）
- `arena/` - 竞技场模块（详见 `arena/CLAUDE.md`）
- `_cython/` - Cython 加速模块（批量决策、订单执行）

---

## 核心类和接口

### fast_execution 模块 (_cython/fast_execution.pyx)

Cython 加速的非做市商批量订单执行模块，优化散户/高级散户/庄家的订单处理性能。

**核心函数：**

- `execute_non_mm_batch(decisions, matching_engine, orderbook, recent_trades) -> list`
  - 批量执行非做市商的订单决策
  - 优化点：使用 C 类型变量、缓存方法引用、内联订单 ID 生成

- `execute_non_mm_batch_with_maker_update(decisions, matching_engine, orderbook, recent_trades, agent_map, tick_trades) -> list`
  - 批量执行订单（包含 maker 账户更新、tick_trades 记录、庄家波动性贡献计算）

- `execute_non_mm_batch_raw(raw_decisions, matching_engine, orderbook, recent_trades, agent_map, tick_trades, mid_price) -> list`
  - 批量执行原始决策数据（内联数量计算逻辑，避免 Python 调用开销）

### fast_math 模块 (fast_math.py)

Numba JIT 加速的高频数学函数模块。

**提供的函数：**
- `log_normalize_unsigned(arr, scale=10.0)` - 无符号对数归一化：`log10(x + 1) / scale`
- `log_normalize_signed(arr, scale=10.0)` - 带符号对数归一化：`sign(x) * log10(|x| + 1) / scale`
- `HAS_NUMBA` - 标识 Numba 是否可用

### CheckpointLoader (checkpoint_loader.py)

统一的 checkpoint 加载接口，支持单训练场和多竞技场两种格式。

**主要方法：**
- `detect_type(path)` - 自动检测 checkpoint 类型
- `load(path)` - 加载 checkpoint，返回统一格式

**返回格式：**
```python
{
    "type": CheckpointType,            # SINGLE_ARENA 或 PARALLEL_ARENA
    "tick": int,
    "episode": int,
    "populations": {AgentType: {...}},
    "source_arena_id": int | None,
}
```

**支持的压缩格式：**
- gzip 压缩格式（新）：自动检测 gzip 魔数（0x1f 0x8b）
- 普通 pickle 格式（旧）：向后兼容

---

### Population (population.py)

管理特定类型 Agent 的种群，支持从 NEAT 基因组创建 Agent。

**主要职责：**
- 创建和管理 NEAT 种群
- 从基因组创建对应类型的 Agent
- 向量化评估适应度
- 执行 NEAT 进化算法
- 重置 Agent 账户状态

**适应度计算公式（相对收益）：**

| 种群类型 | 适应度公式 |
|---------|-----------|
| 散户 | 相对收益率 |
| 高级散户 | 相对收益率 |
| 庄家 | 0.5 × 相对收益率 + 0.5 × 波动性贡献排名归一化 |
| 做市商 | 0.5 × 相对收益率 + 0.5 × maker_volume 排名归一化 |

**相对收益率 = Agent 收益率 - 市场平均收益率**

**关键方法：**
- `create_agents(genomes)` - 从基因组列表创建 Agent（小批量串行，大批量并行）
- `evaluate(current_price, market_avg_return)` - 评估种群适应度并排序
- `evolve(current_price)` - 执行一代 NEAT 进化，异常时自动重置种群
- `evolve_with_cached_fitness(current_price)` - 使用缓存的适应度进行进化
- `get_elite_species_avg_fitness()` - 获取最精英 species 的平均适应度
- `reset_agents()` - 重置所有 Agent 账户
- `accumulate_fitness(current_price, market_avg_return)` - 累积当前 episode 的适应度
- `apply_accumulated_fitness()` - 将累积的平均适应度应用到基因组
- `clear_accumulated_fitness()` - 清空累积的适应度数据

**内存管理方法：**
- `_cleanup_old_agents()` - 清理旧 Agent 对象，打破循环引用
- `_cleanup_neat_history()` - 清理 NEAT 种群中的历史数据
- `_reset_neat_population()` - 当 NEAT 进化失败时，创建全新的随机种群
- `_cleanup_genome_internals(genomes)` - 清理基因组内部数据
- `sync_genomes_from_pending()` - 从待处理数据同步基因组（延迟反序列化）

**多核并行化：**
- 使用实例级别 ThreadPoolExecutor（8个worker）
- 小批量（<50）串行创建，避免线程池开销
- 大批量并行调用 `Brain.from_genome`

**Agent ID 偏移量：**
- RETAIL: 0（子种群每 100K 偏移）
- RETAIL_PRO: 1,000,000
- WHALE: 2,000,000
- MARKET_MAKER: 3,000,000（子种群每 100K 偏移）

### SubPopulationManager (population.py)

通用子种群管理器，将种群拆分为多个独立子种群进行进化。

**支持的种群类型：**
- `AgentType.RETAIL` - 散户
- `AgentType.MARKET_MAKER` - 做市商

**设计目标：**
- 减少单个 NEAT 种群的规模，优化进化性能
- 每个子种群独立进化，共享市场环境
- 清理时间显著减少（从~5s降至~0.04s）

**主要属性：**
- `sub_populations: list[Population]` - 子种群列表
- `sub_population_count: int` - 子种群数量（默认10）
- `agents_per_sub: int` - 每个子种群的Agent数量
- `agents` - 所有子种群Agent的合并视图

**主要方法：**
- `reset_agents()` - 重置所有子种群的Agent
- `evaluate(current_price)` - 评估所有Agent适应度
- `evolve(current_price)` - 进化所有子种群（串行）
- `get_all_genomes()` - 获取所有子种群的基因组

### WorkerConfig (population.py)

Worker 配置数据类。

**属性：**
- `agent_type: AgentType` - Agent 类型
- `sub_pop_id: int` - 子种群 ID
- `neat_config_path: str` - NEAT 配置文件路径
- `pop_size: int` - 种群大小

### MultiPopulationWorkerPool (population.py)

多种群统一 Worker 池，管理多个不同配置的 Worker。

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
configs = [
    WorkerConfig(AgentType.RETAIL, 0, "config/neat_retail.cfg", 1000),
    WorkerConfig(AgentType.RETAIL, 1, "config/neat_retail.cfg", 1000),
    WorkerConfig(AgentType.MARKET_MAKER, 0, "config/neat_market_maker.cfg", 100),
]
pool = MultiPopulationWorkerPool("config", configs)
results = pool.evolve_all_parallel(fitness_map)
pool.shutdown()
```

---

### Trainer (trainer.py)

管理整体训练流程，协调种群和撮合引擎。

**主要职责：**
- 初始化训练环境（创建种群、撮合引擎、ADL 管理器）
- 管理训练生命周期（tick、episode）
- 处理成交和强平
- 保存/加载检查点
- 支持暂停/恢复/停止控制

**关键方法：**
- `setup()` - 初始化训练环境
- `_init_ema_price(initial_price)` - 初始化 EMA 平滑价格
- `_update_ema_price(current_mid_price)` - 更新 EMA 平滑价格
- `_aggregate_tick_trades(tick_trades)` - 聚合本 tick 的成交量和成交额
- `_calculate_catfish_initial_balance()` - 计算鲶鱼初始资金
- `_register_all_agents()` - 注册所有 Agent 的费率到撮合引擎
- `_build_agent_map()` - 构建 Agent ID 到 Agent 对象的映射表
- `_build_execution_order()` - 构建 Agent 执行顺序列表
- `_cancel_agent_orders(agent)` - 撤销指定 Agent 的所有挂单
- `_execute_liquidation_market_order(agent)` - 执行强平市价单
- `_execute_adl(liquidated_agent, remaining_qty, current_price, is_long)` - 执行 ADL 自动减仓
- `_check_catfish_liquidation()` - 检查鲶鱼强平
- `_should_end_episode_early()` - O(1) 检查是否满足提前结束条件
- `_compute_normalized_market_state()` - 向量化计算归一化市场状态
- `run_tick()` - 执行单个 tick
- `run_episode()` - 运行完整 episode
- `train()` - 主训练循环
- `save_checkpoint()` / `load_checkpoint()` - 检查点管理
- `find_latest_checkpoint(checkpoint_dir)` - 查找最新检查点文件

**Checkpoint 格式：**

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

2. **旧格式**（单个 Population）：
   ```python
   {
       "generation": int,
       "neat_pop": neat_pop,
   }
   ```

**自动迁移**：加载旧格式 checkpoint 时，自动将单个大 neat_pop 拆分成多个子种群。

**性能优化：**
- 使用 `deque(maxlen=100)` 自动管理成交记录
- 使用 `agent_map` 映射表实现 O(1) Agent 查找
- 使用 `_pop_total_counts` 和 `_pop_liquidated_counts` 计数器实现 O(1) 种群淘汰检查
- 向量化市场状态计算

**价格稳定机制（EMA 平滑）：**
- 使用 EMA（指数移动平均）平滑 mid_price
- 公式：`smooth_mid_price = α × current_mid_price + (1-α) × prev_smooth_mid_price`
- 参数 `ema_alpha`（默认 0.1）可通过 `MarketConfig.ema_alpha` 配置

**统一 Worker 池架构：**
- `_unified_worker_pool: MultiPopulationWorkerPool` - 统一管理所有 Worker 进程
- `_worker_pool_synced: bool` - Worker 池是否已同步基因组

---

### AtomicAction 和 AtomicActionType (trainer.py)

原子动作机制，将复合操作拆分后随机打乱执行顺序。

**AtomicActionType 枚举：**
- `CANCEL (1)` - 撤单
- `LIMIT_BUY (2)` - 限价买单
- `LIMIT_SELL (3)` - 限价卖单
- `MARKET_BUY (4)` - 市价买单
- `MARKET_SELL (5)` - 市价卖单

**AtomicAction 数据类：**
```python
@dataclass
class AtomicAction:
    action_type: AtomicActionType
    agent_id: int
    order_id: int = 0
    price: float = 0.0
    quantity: int = 0
    is_market_maker: bool = False
    is_catfish: bool = False
    agent_ref: Any = None
```

**动作收集规则：**
- 吃单鲶鱼：direction != 0 时，收集一个市价单动作
- 做市鲶鱼：direction == 0 时，收集撤旧单动作 + 新挂单动作
- 做市商：收集所有撤单动作 + 所有新挂单动作
- 非做市商限价单：收集撤旧单动作（如有）+ 新挂单动作
- 非做市商撤单：收集撤单动作
- 非做市商市价单：收集市价单动作

**执行方法：** `_execute_atomic_action(atomic_action)`

---

## 训练流程

### 1. 初始化阶段 (`setup`)
- 创建四个种群（散户/高级散户/庄家/做市商）
- 创建撮合引擎和 ADL 管理器
- 注册所有 Agent 的费率到撮合引擎
- 构建 Agent 映射表和执行顺序
- 初始化 EMA 平滑价格
- 做市商建立初始流动性

### 2. Episode 循环 (`run_episode`)
- 重置所有 Agent 账户
- 重置鲶鱼状态和强平标志
- 重置市场状态
- 运行 episode_length 个 tick

**提前结束条件：**
- 鲶鱼被强平（立即结束 episode）
- 任意种群存活少于初始值的 1/4
- 订单簿只有单边挂单

**适应度累积与进化：**
- 每个 episode 结束时累积当前适应度
- 累积使用平均值：`genome.fitness = 累积适应度总和 / 累积次数`
- 每 N 个 episode 进化一次（默认 10）
- 进化后清空累积数据

### 3. Tick 执行 (`run_tick`)

**时序设计：**
- **Tick 1**：只展示做市商初始挂单后的市场状态
- **Tick 2+**：正常行动 tick

**Tick 2+ 流程：**

1. **Tick 开始（强平处理分三阶段）**：
   - 保存 tick 开始时的价格到 `tick_start_price`
   - 阶段1：统一撤单
   - 阶段2：统一市价单平仓
   - 阶段3：用最新价格计算 ADL 候选并执行

2. **Tick 过程（原子动作随机执行）**：
   - 向量化计算归一化市场状态
   - 随机打乱 Agent 执行顺序
   - Agent 决策
   - 收集所有原子动作
   - 随机打乱原子动作顺序
   - 逐个执行原子动作

3. **Tick 结束**：
   - 记录当前价格到 `_price_history`
   - 记录 tick 历史数据
   - 检查鲶鱼强平

---

## 强平与淘汰机制

**强平即淘汰（Liquidation = Elimination）**

**三阶段强平处理：**
1. **阶段1（统一撤单）**：统一撤销需要淘汰的 Agent 的所有挂单
2. **阶段2（统一市价单平仓）**：执行市价单平仓（不触发 ADL）
3. **阶段3（ADL）**：用最新价格计算 ADL 候选清单，执行 ADL

**设计原因：**
- 先统一撤单可防止被淘汰的 Agent 在平仓过程中作为 maker 被成交
- 用最新价格计算 ADL 候选确保候选在当前价格下确实盈利

**ADL（自动减仓）：**
1. 获取最新价格
2. 用最新价格计算候选清单（包含 Agent 和鲶鱼）
3. 依次与候选对手方成交
4. 更新候选清单的 position_qty
5. 兜底处理：强制清零被淘汰者的仓位

**重入保护：**
- 使用 `_eliminating_agents` 集合跟踪正在强平的 Agent
- 使用 `_adl_long_candidates` 和 `_adl_short_candidates` 存储预计算的候选清单

---

## BatchNetworkCache 缓存优化 (_cython/batch_decide_openmp.pyx)

将网络数据提取从每次决策移到进化后一次性完成。

**缓存类方法：**
- `update_networks(networks)` - 从 Python 网络对象提取数据到 C 结构
- `decide(inputs, orderbooks, mid_prices)` - 使用缓存数据执行批量决策
- `decide_multi_arena_direct(states, markets, indices, return_array)` - 跨竞技场批量推理
- `is_valid()` - 检查缓存是否有效
- `clear()` - 清空缓存

**性能提升：**
- 优化前：~823ms/tick
- 优化后：~160ms/tick
- 加速比：约 5.1x

**OpenMP 线程数配置：**
- 通过 `TrainingConfig.openmp_threads` 配置（默认 8）
- 8 线程为最优值（2.3x 加速）

---

## 内存管理

**1. NEAT 种群历史清理**
- `genome_to_species` 字典：只保留当前代基因组的映射
- `species.members`：清理已不存在的基因组引用
- `stagnation.species_fitness`：清空物种适应度历史
- `reproduction.ancestors`：清空祖先引用
- `reporters` 统计数据：只保留最近 5 代

**2. Agent 对象清理**
- 清理 `Brain.network` 内部状态
- 置空 `Brain.genome`, `Brain.network`, `Brain.config`
- 置空 `Agent.brain`, `Agent.account`

**3. 进化后垃圾回收**
- 所有种群进化完成后，统一调用三代 GC 和 malloc_trim()

**4. Worker 进程内存清理**
- 在 `evolve`、`evolve_return_params` 和 `set_genomes` 命令处理后调用
- 完全清空 `ancestors` 字典
- 完全清空 `stagnation.species_fitness` 字典
- 限制 `fitness_history` 长度为 5
- 删除空物种

**5. 临时网络对象清理**
- 清理 `network.node_evals`、`network.values` 等属性

**6. 基因组内部数据清理**
- `genome.connections.clear()` 后置为空字典
- `genome.nodes.clear()` 后置为空字典

---

## 依赖关系

- `src.bio.agents` - Agent 类
- `src.config.config` - 配置类
- `src.core.log_engine` - 日志系统
- `src.market.adl` - ADL 管理器
- `src.market.matching` - 撮合引擎
- `src.market.orderbook` - 订单簿

---

## NEAT 配置

不同 Agent 类型使用不同的 NEAT 配置文件：
- `config/neat_retail.cfg` - 散户（127 个输入节点，8 个输出节点）
- `config/neat_retail_pro.cfg` - 高级散户（907 个输入节点，8 个输出节点）
- `config/neat_whale.cfg` - 庄家（907 个输入节点，8 个输出节点）
- `config/neat_market_maker.cfg` - 做市商（934 个输入节点，41 个输出节点）

**关键配置参数（防止种群灭绝）：**
- `reset_on_extinction = True`
- `survival_threshold = 0.5`
- `compatibility_threshold = 4.0`
- `elitism = 1`
- `species_elitism = 2`

---

## 启动脚本

训练脚本位于 `scripts/` 目录：
- `train_noui.py` - 无 UI 高性能训练模式
- `train_ui.py` - 带 UI 训练模式

```bash
# 基本训练
python scripts/train_noui.py --episodes 100

# 无限训练模式
python scripts/train_noui.py --infinite

# 从指定检查点恢复
python scripts/train_noui.py --resume checkpoints/ep_50.pkl --episodes 100
```
