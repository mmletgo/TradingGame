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

## 核心类

### MigrationStrategy (config.py)

迁移策略枚举：
- `RING` - 环形迁移：Arena[i] → Arena[(i+1) % N]
- `RANDOM` - 随机迁移
- `BEST_TO_WORST` - 最好竞技场迁移到最差竞技场

### ArenaConfig (config.py)

单个竞技场配置：
- `arena_id: int` - 竞技场唯一标识
- `config: Config` - 训练配置（所有竞技场共用）
- `seed: int | None` - 随机种子（不同种子产生不同市场特征）

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

### Arena (arena.py)

单个竞技场封装，管理一个独立的训练环境。

**主要方法：**
- `setup()` - 初始化竞技场（创建 Trainer、设置种子）
- `run_episode()` - 运行一个 episode，返回 EpisodeMetrics
- `get_migration_candidates(count, select_best)` - 获取迁移候选者的基因组
- `inject_genomes(packets)` - 注入迁入的基因组（替换最差个体）
- `get_checkpoint_data()` - 获取检查点数据
- `load_checkpoint_data(checkpoint)` - 加载检查点数据
- `stop()` - 停止竞技场（关闭线程池）

### ArenaProcess (arena_manager.py)

竞技场进程信息数据类：
- `arena_id: int` - 竞技场 ID
- `process: Process` - 进程对象
- `cmd_queue: Queue` - 命令队列（主进程 → 子进程）
- `result_queue: Queue` - 结果队列（子进程 → 主进程）

### arena_worker() (arena_manager.py)

竞技场工作进程入口函数，处理以下命令：
- `setup` - 初始化竞技场
- `run_episode` - 运行一个 episode
- `get_migration_candidates` - 获取迁移候选者
- `inject_genomes` - 注入迁移基因组
- `get_checkpoint` - 获取检查点数据
- `load_checkpoint` - 加载检查点数据
- `stop` - 停止进程

### ArenaManager (arena_manager.py)

竞技场管理器，协调多个竞技场的训练。

**主要方法：**
- `setup()` - 创建所有竞技场进程
- `start()` - 启动所有竞技场进程
- `train(episodes, progress_callback)` - 训练主循环
- `_execute_migration()` - 执行一次迁移
- `stop()` - 停止所有竞技场
- `save_checkpoint(path)` - 保存所有竞技场的检查点
- `load_checkpoint(path)` - 加载检查点
- `get_summary()` - 获取训练状态汇总

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
- `_plan_ring_migration()` - 环形迁移：Arena[i] → Arena[(i+1) % N]
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

多竞技场指标聚合器：
- `update(metrics)` - 更新单个竞技场的指标
- `update_batch(metrics_list)` - 批量更新
- `get_summary()` - 获取各竞技场汇总
- `get_global_summary()` - 获取全局汇总
- `get_history()` - 获取历史数据（用于检查点）

## 架构设计

### 多进程架构

```
                    +-------------------+
                    |   ArenaManager    |
                    |   (主进程)        |
                    +--------+----------+
                             |
         +-------------------+-------------------+
         |                   |                   |
    +----v----+         +----v----+         +----v----+
    | Arena 0 |         | Arena 1 |         | Arena N |
    | (子进程) |         | (子进程) |         | (子进程) |
    +----+----+         +----+----+         +----+----+
         |                   |                   |
    +----v----+         +----v----+         +----v----+
    | Trainer |         | Trainer |         | Trainer |
    | +Engine |         | +Engine |         | +Engine |
    +---------+         +---------+         +---------+
```

**选择多进程的原因：**
1. Python GIL 限制多线程 CPU 密集型计算
2. 10+ 个竞技场需要真正的并行
3. 每个竞技场内部已有线程池优化（决策并行、进化并行）
4. 市场状态完全可独立实例化，无共享全局状态

### 进程间通信

使用 `multiprocessing.Queue` 进行命令和结果传递：
- 主进程通过 `cmd_queue` 发送命令
- 子进程通过 `result_queue` 返回结果
- 迁移数据通过 pickle 序列化传输

### 训练流程

1. **初始化阶段**
   - ArenaManager 创建 N 个竞技场进程
   - 每个进程启动并初始化 Arena
   - 等待所有 Arena setup 完成

2. **Episode 循环**
   - 所有竞技场并行运行 episode
   - 收集各竞技场的 EpisodeMetrics
   - 定期执行迁移（每 migration_interval 个 episode）
   - 定期保存检查点

3. **迁移流程**
   - 从各竞技场收集迁移候选者（最好和最差的 Agent）
   - MigrationSystem 规划迁移方向
   - 分发基因组到目标竞技场
   - 目标竞技场用迁入的基因组替换最差个体

## 使用方法

### 命令行

```bash
# 启用多竞技场训练
python scripts/train_noui.py --multi-arena --num-arenas 10 --episodes 100

# 自定义迁移参数
python scripts/train_noui.py --multi-arena --num-arenas 10 \
    --migration-interval 10 \
    --migration-count 5 \
    --migration-strategy ring \
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
    migration_count=5,
    migration_strategy=MigrationStrategy.RING,
)

# 创建管理器
manager = ArenaManager(multi_config)

# 初始化和启动
manager.setup()
manager.start()

# 训练
manager.train(episodes=100, progress_callback=callback)

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

- `scripts/train_noui.py` - 命令行入口

## 注意事项

1. **进程安全**：每个竞技场完全独立，无共享状态
2. **序列化**：基因组通过 pickle 序列化传输，注意版本兼容性
3. **资源清理**：异常退出时确保调用 `manager.stop()` 清理子进程
4. **调试模式**：可通过减少 num_arenas 到 1-3 进行调试
5. **日志分离**：每个竞技场使用独立的日志前缀便于追踪
