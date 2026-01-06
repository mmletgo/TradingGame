# Arena 模块（多竞技场并行训练）

## 模块概述

Arena 模块实现多竞技场并行训练系统，使用多进程架构实现真正的并行化，绕过 Python GIL 限制。支持定期迁移 Agent 基因组，促进跨竞技场的策略交流。

## 文件结构

- `__init__.py` - 模块导出
- `config.py` - 配置类（ArenaConfig, MultiArenaConfig, MigrationStrategy）
- `arena.py` - 单个竞技场封装类
- `arena_manager.py` - 竞技场管理器和工作进程
- `migration.py` - 迁移系统（MigrationPacket, MigrationSystem）
- `metrics.py` - 指标收集和聚合
- `shared_checkpoint.py` - 共享检查点管理（并发安全的跨进程检查点）

## 核心类

### MigrationStrategy (config.py)

迁移策略枚举：
- `RING` - 环形迁移：Arena[i] -> Arena[(i+1) % N]
- `RANDOM` - 随机迁移
- `BEST_TO_WORST` - 最好竞技场迁移到最差竞技场

### ArenaConfig (config.py)

单个竞技场配置：
- `arena_id: int` - 竞技场唯一标识
- `config: Config` - 训练配置（所有竞技场共用）
- `seed: int | None` - 随机种子（不同种子产生不同市场特征）
- `migration_interval: int` - 竞技场内部迁移间隔（episode 数）
- `checkpoint_interval: int` - 检查点保存间隔（episode 数）
- `max_episodes: int` - 最大 episode 数
- `shared_checkpoint_path: str` - 共享检查点路径

### MultiArenaConfig (config.py)

多竞技场配置：
- `num_arenas: int = 10` - 竞技场数量
- `base_config: Config` - 基础配置（所有竞技场共用）
- `migration_interval: int = 10` - 迁移间隔（episode 数）
- `migration_count: int = 5` - 每次迁移的 Agent 数量（每种群）
- `migration_best_ratio: float = 0.5` - 迁移最好个体的比例
- `migration_strategy: MigrationStrategy = RING` - 迁移策略
- `checkpoint_interval: int = 50` - 检查点保存间隔
- `seed_offset: int = 0` - 随机种子偏移量
- `shared_checkpoint_path: str` - 共享检查点路径
- `max_episodes: int` - 最大 episode 数

### Arena (arena.py)

单个竞技场封装，管理一个独立的训练环境。

**主要方法：**
- `setup()` - 初始化竞技场（创建 Trainer、设置种子、设置 `is_running=True` 使 tick 循环能正常运行）
- `run_episode()` - 运行一个 episode，返回 EpisodeMetrics
- `get_migration_candidates(count, select_best)` - 获取迁移候选者的基因组
- `get_best_genomes(top_n)` - 获取各种群的最佳基因组（直接使用 genome.fitness 缓存，用于共享检查点）
- `inject_genomes(packets)` - 批量注入迁入的基因组，按 agent_type 分组处理，调用 `population.replace_worst_agents()` 增量替换最差个体，然后调用 `trainer._update_agents_after_migration()` 增量更新内部状态
- `get_checkpoint_data()` - 获取检查点数据
- `load_checkpoint_data(checkpoint)` - 加载检查点数据
- `stop()` - 停止竞技场（关闭线程池）

### ArenaProcessInfo (arena_manager.py) [推荐]

竞技场进程信息数据类（异步监控模式）：
- `arena_id: int` - 竞技场 ID
- `process: Process` - 进程对象
- `status_queue: Queue` - 状态报告队列（子进程 -> 主进程）
- `control_queue: Queue` - 控制命令队列（主进程 -> 子进程）
- `episode: int` - 当前 episode 编号
- `is_finished: bool` - 是否已完成

### ArenaProcess (arena_manager.py) [已弃用]

竞技场进程信息数据类（同步模式，保留用于兼容）：
- `arena_id: int` - 竞技场 ID
- `process: Process` - 进程对象
- `cmd_queue: Queue` - 命令队列（主进程 -> 子进程）
- `result_queue: Queue` - 结果队列（子进程 -> 主进程）

### arena_worker_autonomous() (arena_manager.py) [推荐]

自治竞技场工作进程入口函数，竞技场按自己的节奏独立运行：
- 自动执行 episode 循环
- 自动触发迁移（从共享检查点获取其他竞技场的最佳个体）
- 自动保存检查点
- 通过 status_queue 报告状态
- 通过 control_queue 接收停止命令

### arena_worker() (arena_manager.py) [已弃用]

竞技场工作进程入口函数（同步模式，保留用于兼容），处理以下命令：
- `setup` - 初始化竞技场
- `run_episode` - 运行一个 episode
- `get_migration_candidates` - 获取迁移候选者
- `inject_genomes` - 注入迁移基因组
- `get_checkpoint` - 获取检查点数据
- `load_checkpoint` - 加载检查点数据
- `stop` - 停止进程

### ArenaManager (arena_manager.py)

竞技场管理器（监控者模式），协调多个竞技场的训练。

**主要方法：**
- `setup()` - 创建所有竞技场进程，初始化共享检查点
- `start()` - 启动所有竞技场进程
- `monitor(progress_callback, check_interval)` - 监控所有竞技场的运行状态（非阻塞）
- `train(episodes, progress_callback)` - 训练（兼容旧接口，内部调用 monitor）
- `stop()` - 停止所有竞技场
- `get_summary()` - 获取训练状态汇总

**辅助函数（模块级）：**
- `_execute_migration_from_checkpoint(arena, checkpoint_manager, arena_id)` - 从共享检查点执行迁移
- `_save_arena_to_checkpoint(arena, checkpoint_manager, arena_id, episode)` - 保存竞技场状态到共享检查点

### MigrationPacket (migration.py)

迁移数据包，用于在竞技场之间传输基因组：
- `source_arena: int` - 来源竞技场 ID
- `agent_type: AgentType` - Agent 类型
- `genome_data: bytes` - 序列化的基因组数据
- `fitness: float` - 适应度
- `generation: int` - 代数

### MigrationSystem (migration.py)

迁移系统，规划基因组的迁移方向。

**主要方法：**
- `plan_migrations(candidates)` - 规划迁移方向，返回每个竞技场应接收的数据包
- `serialize_genome(genome)` - 序列化基因组
- `deserialize_genome(data)` - 反序列化基因组

**迁移策略实现：**
- `_plan_ring_migration()` - 环形迁移：Arena[i] -> Arena[(i+1) % N]
- `_plan_random_migration()` - 随机迁移
- `_plan_best_to_worst_migration()` - 按适应度排名迁移

### EpisodeMetrics (metrics.py)

单个 episode 的指标：
- `arena_id: int` - 竞技场 ID
- `episode: int` - episode 编号
- `tick_count: int` - tick 数量
- `high_price: float` - 最高价
- `low_price: float` - 最低价
- `final_price: float` - 最终价格
- `volatility: float` - 波动率
- `total_volume: float` - 总成交量
- `liquidation_count: dict[AgentType, int]` - 各种群强平数量
- `avg_fitness: dict[AgentType, float]` - 各种群平均适应度

### ArenaMetrics (metrics.py)

单个竞技场的指标收集器：
- `record_episode(metrics)` - 记录 episode 指标
- `get_latest()` - 获取最新指标
- `get_summary(window)` - 获取滑动窗口统计

### MetricsAggregator (metrics.py)

多竞技场指标聚合器（每个竞技场最多保留 1000 条历史记录，防止内存泄漏）：
- `__init__(max_history=1000)` - 初始化聚合器，可配置最大历史记录数
- `update(metrics)` - 更新单个竞技场的指标
- `update_batch(metrics_list)` - 批量更新
- `get_summary()` - 获取各竞技场汇总
- `get_global_summary()` - 获取全局汇总
- `get_history()` - 获取历史数据（用于检查点）

### ArenaCheckpointData (shared_checkpoint.py)

单个竞技场的检查点数据：
- `arena_id: int` - 竞技场唯一标识
- `episode: int` - 当前 episode 编号
- `best_genomes: dict[str, list[tuple[bytes, float]]]` - 各 Agent 类型的最佳基因组
- `populations: dict` - 完整种群数据用于恢复
- `updated_at: float` - 最后更新时间戳

### SharedCheckpointData (shared_checkpoint.py)

共享检查点数据：
- `version: int` - 检查点版本号（每次更新递增）
- `arenas: dict[int, ArenaCheckpointData]` - 各竞技场的检查点数据
- `config: dict` - 训练配置

### SharedCheckpointManager (shared_checkpoint.py)

共享检查点管理器（独立文件模式），每个竞技场使用独立的 checkpoint 文件，完全避免锁竞争。

**目录结构：**
```
checkpoints/multi_arena/
    config.pkl              # 全局配置
    arena_0/
        checkpoint.pkl      # 完整检查点（populations + episode 等，用于恢复）
        best_genomes.pkl    # 轻量级数据（仅 best_genomes，用于迁移）
    arena_1/
        checkpoint.pkl
        best_genomes.pkl
    ...
```

**初始化参数：**
- `checkpoint_path: str = "checkpoints/multi_arena_shared.pkl"` - 检查点路径（兼容旧接口，自动转换为目录模式）

**主要方法：**
- `initialize(config, num_arenas)` - 初始化检查点目录结构
- `update_arena(arena_id, episode, populations, best_genomes)` - 同时保存 `checkpoint.pkl` 和 `best_genomes.pkl`
- `get_migration_candidates(requesting_arena_id, count_per_arena)` - 优先读取轻量级的 `best_genomes.pkl`，避免加载完整 populations
- `get_full_checkpoint()` - 聚合所有竞技场的检查点（用于恢复）
- `_get_arena_best_genomes_path(arena_id)` - 获取 `best_genomes.pkl` 路径
- `_read_arena_best_genomes(path)` - 读取轻量级 best_genomes 文件

**无锁设计：**
- 每个竞技场只写自己的 checkpoint 文件，无需锁
- 读取其他竞技场时直接读取文件（读取是原子的）
- 使用原子写入（先写临时文件 `.pkl.tmp`，再用 `os.replace()` 重命名）确保读取安全
- 完全避免锁竞争和超时问题

**分离存储优化：**
- `checkpoint.pkl`（~100-200MB）：包含完整 `neat_pop` 对象，仅用于恢复训练
- `best_genomes.pkl`（~1-2MB）：仅包含迁移需要的基因组数据
- 迁移时只读取轻量级文件，大幅减少内存占用
- 注意：旧格式的 checkpoint 需要先使用 `scripts/migrate_checkpoints.py` 迁移

## 架构设计

### 异步监控者模式（推荐）

```
                    +-------------------+
                    |   ArenaManager    |
                    |   (监控者)        |
                    +--------+----------+
                             |
                     monitor() 非阻塞轮询
                             |
         +-------------------+-------------------+
         |                   |                   |
    +----v----+         +----v----+         +----v----+
    | Arena 0 |         | Arena 1 |         | Arena N |
    | (自治)   |         | (自治)   |         | (自治)   |
    +----+----+         +----+----+         +----+----+
         |                   |                   |
    独立运行              独立运行              独立运行
    自动迁移              自动迁移              自动迁移
    自动保存              自动保存              自动保存
         |                   |                   |
         +-------------------+-------------------+
                             |
                    +--------v----------+
                    | SharedCheckpoint  |
                    | (共享检查点)       |
                    +-------------------+
```

**异步模式特点：**
1. 每个竞技场独立进化，不等待其他竞技场
2. 主进程只负责监控和收集状态，不协调同步
3. 迁移通过共享检查点实现，各竞技场自主读写
4. 竞技场可以按各自节奏运行，互不阻塞

### 进程间通信

**异步模式（推荐）：**
- `status_queue`: 子进程 -> 主进程（状态报告）
  - `("setup_done", arena_id, None)` - 初始化完成
  - `("episode_done", arena_id, metrics)` - episode 完成
  - `("finished", arena_id, episode_count)` - 训练完成
- `control_queue`: 主进程 -> 子进程（控制命令）
  - `"stop"` - 停止运行

**同步模式（已弃用）：**
- `cmd_queue`: 主进程 -> 子进程（命令）
- `result_queue`: 子进程 -> 主进程（结果）

### 训练流程（异步模式）

1. **初始化阶段**
   - ArenaManager 初始化共享检查点
   - 创建 N 个竞技场进程（使用 arena_worker_autonomous）
   - 启动所有进程，等待 setup_done 信号

2. **自治运行阶段**
   - 每个竞技场独立运行 episode 循环
   - 到达 migration_interval 时，从共享检查点获取其他竞技场的最佳个体并注入
   - 到达 checkpoint_interval 时，保存自身状态到共享检查点
   - 通过 status_queue 报告进度

3. **监控阶段**
   - ArenaManager.monitor() 非阻塞轮询各竞技场状态
   - 收集 EpisodeMetrics 更新到 MetricsAggregator
   - 调用 progress_callback 报告进度
   - 检测所有竞技场完成后结束

4. **停止阶段**
   - 通过 control_queue 发送 stop 命令
   - 等待进程退出，超时则强制终止

### 迁移流程（异步模式）

1. 竞技场 A 到达 checkpoint_interval，调用 `_save_arena_to_checkpoint()`：
   - 获取自身最佳基因组（`get_best_genomes()`）
   - 写入共享检查点

2. 竞技场 B 到达 migration_interval，调用 `_execute_migration_from_checkpoint()`：
   - 从共享检查点读取其他竞技场的最佳基因组
   - 构建 MigrationPacket 列表
   - 注入到自身种群（替换最差个体）

## 使用方法

### 命令行

```bash
# 启用多竞技场训练
python scripts/train_multi_arena.py --num-arenas 10 --episodes 100

# 自定义迁移参数
python scripts/train_multi_arena.py --num-arenas 10 \
    --migration-interval 10 \
    --checkpoint-interval 50 \
    --episodes 100
```

### 程序化使用

```python
from src.training.arena import ArenaManager, MultiArenaConfig, MigrationStrategy

# 创建配置
multi_config = MultiArenaConfig(
    num_arenas=10,
    base_config=config,
    migration_interval=10,
    checkpoint_interval=50,
    max_episodes=100,
    migration_strategy=MigrationStrategy.RING,
)

# 创建管理器
manager = ArenaManager(multi_config)

# 初始化和启动
manager.setup()
manager.start()

# 监控（非阻塞轮询）
manager.monitor(progress_callback=callback)

# 或使用兼容接口
# manager.train(episodes=100, progress_callback=callback)

# 停止
manager.stop()
```

## 资源估算

### 内存

单竞技场估算：~430MB
- 10,000 散户：~200MB
- 100 高级散户 + 100 庄家 + 200 做市商：~80MB
- 订单簿和引擎：~50MB
- NEAT 种群：~100MB

10 个竞技场：~4.3GB + 主进程开销

**建议**：16GB 内存系统运行 10-15 个竞技场

### 线程池

每个竞技场进程内：
- Population 线程池：8 个 worker（实例级别）
- Trainer 线程池：16 个 worker

## 依赖关系

### 上游依赖

- `src.training.trainer` - 训练器类
- `src.training.population` - 种群管理
- `src.config.config` - 配置类
- `src.core.log_engine` - 日志系统

### 下游依赖

- `scripts/train_multi_arena.py` - 命令行入口

## 注意事项

1. **进程安全**：每个竞技场完全独立，无共享状态（除共享检查点文件）
2. **序列化**：基因组通过 pickle 序列化传输，注意版本兼容性
3. **资源清理**：异常退出时确保调用 `manager.stop()` 清理子进程
4. **调试模式**：可通过减少 num_arenas 到 1-3 进行调试
5. **日志分离**：每个竞技场使用独立的日志前缀便于追踪
6. **共享检查点并发**：多个进程可能同时读写，使用独立文件模式避免锁竞争

## 内存管理

多竞技场模式下每个竞技场运行在独立进程中，为防止内存泄漏，实现了以下机制：

**1. 检查点恢复优化 (`ArenaConfig.should_resume`)**

子进程直接从文件读取检查点，然后在 setup 时直接从检查点创建 Agent：
- 主进程只检测检查点文件是否存在，不读取数据
- 子进程使用 `should_resume` 标志决定是否从文件恢复
- 避免在主进程中占用大量内存，避免跨进程序列化大对象

**2. 直接从检查点创建种群 (`Trainer.setup(checkpoint)`)**

如果提供了检查点数据，直接从检查点创建种群，跳过创建全新 Agent：
- `setup(checkpoint=...)` 接受可选检查点参数
- 调用 `_setup_populations_from_checkpoint()` 直接从 NEAT 种群数据创建 Agent
- 避免先创建全新 Agent 再清理的内存开销

**3. 检查点加载时清理旧 Agent (`Trainer.load_checkpoint_data()`)**

加载检查点时先清理 `setup()` 创建的 Agent（兼容旧接口）：
- 调用 `pop._cleanup_old_agents()` 清理旧 Agent
- 执行 `gc.collect()` 回收内存
- 然后才从检查点创建新的 Agent

**4. Agent 对象清理 (`Population._cleanup_old_agents()`)**

在进化和迁移注入前清理旧 Agent 对象：
- 清理 `Brain.network` 内部状态
- 置空 `Brain.genome`, `Brain.network`, `Brain.config`
- 置空 `Agent.brain`, `Agent.account`
- 多次调用 `gc.collect()` 处理循环引用

**5. NEAT 种群历史清理 (`Population._cleanup_neat_history()`)**

每次进化后彻底清理 NEAT 库内部积累的历史数据：
- `genome_to_species` 字典：只保留当前代基因组的映射
- `species.members`：清理已不存在的基因组引用
- `stagnation.species_fitness`：清空物种适应度历史
- `reproduction.ancestors`：清空祖先引用
- `reporters` 统计数据：只保留最近 5 代

**6. 迁移注入后增量更新 trainer 状态 (`Arena.inject_genomes()`)**

注入迁移基因组后，调用 `trainer._update_agents_after_migration()` 增量更新内部状态：
- 只更新被替换的 Agent 在 `agent_map` 和 `agent_execution_order` 中的引用
- 不需要重建整个映射表，避免遍历所有 Agent
- 新 genome 的 fitness 从 MigrationPacket 带入

**7. 迁移后 GC (`_execute_migration_from_checkpoint()`)**

执行迁移后立即清理临时对象并强制多轮（3次）垃圾回收 + `malloc_trim()`。

**8. 检查点保存后 GC (`_save_arena_to_checkpoint()`)**

保存检查点后分步释放内存：
- 先释放 `best_genomes` 并 GC
- 再释放 `checkpoint_data` 和 `populations`，执行多轮 GC
- 最后调用 `malloc_trim()` 将释放的内存归还给操作系统

**9. 每 episode 垃圾回收 (`arena_worker_autonomous()`)**

每个 episode 结束后都强制执行两次 `gc.collect()`，确保进化阶段产生的对象被及时回收。

**10. 分离存储减少迁移内存峰值**

- `get_migration_candidates()` 优先读取轻量级的 `best_genomes.pkl`（~1-2MB）
- 避免加载完整的 `checkpoint.pkl`（~100-200MB）
- 12 个竞技场同时迁移时，内存峰值从 ~1.1GB 降低到 ~20MB

**11. Checkpoint 格式迁移脚本**

使用 `scripts/migrate_checkpoints.py` 将旧格式转换为新格式：
```bash
# 预览
python scripts/migrate_checkpoints.py --dry-run

# 执行迁移
python scripts/migrate_checkpoints.py --checkpoint-dir checkpoints/multi_arena
```
