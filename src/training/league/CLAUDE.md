# League Training Module (联盟训练模块)

基于 AlphaStar 联盟训练思路实现的多代对手池训练机制。

## 核心概念

### 问题背景

传统训练中，Agent 只学会克制同一代对手，可能出现"循环克制"（A克B，B克C，C克A），缺乏泛化能力。

### 解决方案

1. **历史对手池**：存储不同代的优秀策略
2. **Main Agents**：主要进化的 Agent，与混合对手竞争
3. **League Exploiter**：专门针对历史对手池训练，发现历史策略的弱点
4. **Main Exploiter**：专门针对当前 Main Agents 训练，发现当前策略的弱点

### 按类型分离架构

每种 Agent 类型（RETAIL, RETAIL_PRO, WHALE, MARKET_MAKER）都有：
- 独立的对手池（存储其他三种类型的历史版本）
- 独立的 League Exploiter 和 Main Exploiter

## 模块结构

```
src/training/league/
├── __init__.py              # 模块导出
├── CLAUDE.md                # 本文档
├── config.py                # 联盟训练配置
├── opponent_entry.py        # 对手条目数据结构
├── opponent_pool.py         # 单类型对手池管理
├── opponent_pool_manager.py # 多类型对手池管理器
├── multi_gen_cache.py       # 多代网络缓存
├── arena_allocator.py       # 竞技场分配器
├── exploiter_manager.py     # Exploiter 管理器
├── league_fitness.py        # 适应度汇总器
└── league_trainer.py        # 联盟训练器主类
```

## 核心类说明

### LeagueTrainingConfig (config.py)

联盟训练配置类，关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pool_dir` | `checkpoints/league_training/opponent_pools` | 对手池存储目录 |
| `max_pool_size_per_type` | 20 | 每种类型最多保存的历史版本数 |
| `milestone_interval` | 50 | 里程碑保存间隔（代数） |
| `exploiter_population_ratio` | 0.1 | Exploiter 种群占 Main 的比例 |
| `exploiter_win_rate_threshold` | 0.55 | 注入对手池的胜率阈值 |
| `sampling_strategy` | `pfsp` | 采样策略：uniform/pfsp/diverse |

### OpponentEntry (opponent_entry.py)

对手条目数据结构，包含：
- `OpponentMetadata`：元数据（entry_id, agent_type, source, win_rates 等）
- `genome_data`：基因组数据（序列化的 NEAT 基因组）
- `network_data`：网络参数（可选，延迟加载）

存储格式：
```
entry_dir/
├── metadata.json    # 元数据
├── genomes.npz      # 基因组数据
└── networks.npz     # 网络参数（可选）
```

### OpponentPool (opponent_pool.py)

单个 Agent 类型的对手池管理器。

主要方法：
- `add_entry(entry)`：添加对手条目
- `sample_opponents(n, strategy, target_type)`：采样对手
- `update_win_rate(entry_id, opponent_type, won)`：更新胜率统计
- `cleanup(current_generation)`：清理旧条目

采样策略：
- **uniform**：均匀随机采样
- **pfsp**：优先选择难以战胜的对手，权重 = (1 - win_rate)²
- **diverse**：随机选择，与 uniform 类似

### OpponentPoolManager (opponent_pool_manager.py)

管理四种 Agent 类型的独立对手池。

主要方法：
- `add_snapshot(generation, populations, source, add_reason)`：批量保存快照
- `sample_opponents_for_arena(target_type, strategy)`：为训练采样对手
- `cleanup_all(current_generation)`：清理所有类型的旧条目

### MultiGenerationNetworkCache (multi_gen_cache.py)

多代网络缓存管理器，按 Agent 类型和 entry_id 管理网络缓存。

缓存类型：
- `current_caches`：当前代网络缓存
- `historical_caches`：历史代网络缓存（LRU 淘汰）
- `league_exploiter_caches`：League Exploiter 网络缓存
- `main_exploiter_caches`：Main Exploiter 网络缓存

### ArenaAllocator (arena_allocator.py)

竞技场分配器，将竞技场按训练目的分配。

竞技场类型：
- **baseline**：全当前代对战（基准）
- **generalization_test**：某类型 Main vs 其他类型历史版本
- **league_exploiter_training**：League Exploiter vs 历史对手（或当前代 Main）
- **main_exploiter_training**：Main Exploiter vs 当前代 Main

主要方法：
- `allocate(pool_manager)`：有历史对手时使用，分配完整的竞技场方案
- `allocate_no_historical()`：无历史对手时使用，仍然分配 Exploiter 竞技场
  - 基准竞技场：全当前代
  - 泛化测试竞技场：转为额外的基准竞技场
  - League Exploiter 竞技场：与当前代 Main 对战（代替历史对手）
  - Main Exploiter 竞技场：正常分配

分配结果数据结构：
```python
@dataclass
class ArenaAllocation:
    assignments: list[ArenaAssignment]
    baseline_arena_ids: list[int]
    generalization_arena_ids: dict[AgentType, list[int]]
    league_exploiter_arena_ids: dict[AgentType, list[int]]
    main_exploiter_arena_ids: list[int]
```

### ExploiterManager (exploiter_manager.py)

Exploiter 管理器，管理每种类型的 League Exploiter 和 Main Exploiter。

主要方法：
- `setup(main_population_sizes)`：初始化 Exploiter 种群
- `evolve_league_exploiter(agent_type, fitness, current_price)`：进化 League Exploiter
- `should_inject_to_pool(agent_type, exploiter_type)`：检查是否应注入对手池
- `update_win_rates(agent_type, exploiter_type, opponent_entry_id, won)`：更新胜率

### LeagueFitnessAggregator (league_fitness.py)

适应度汇总器，从多个竞技场收集并汇总适应度。

汇总策略：
- **simple**：简单平均
- **weighted_average**：加权平均（基准权重 1.0，泛化权重 0.8）
- **min**：取最小值

### LeagueTrainer (league_trainer.py)

联盟训练器主类，继承自 `ParallelArenaTrainer`。

主要流程：
```python
def run_round(self):
    # 1. 分配竞技场
    if self.pool_manager.has_any_historical_opponents():
        self._current_allocation = self.arena_allocator.allocate(self.pool_manager)
    else:
        # 对手池为空时，仍然分配 Exploiter 竞技场
        self._current_allocation = self.arena_allocator.allocate_no_historical()

    # 2. 确保历史对手网络已缓存
    self._ensure_historical_networks_cached()

    # 3. 运行 episodes
    round_stats = self._run_episodes()

    # 4. 汇总适应度
    fitness_results = self.fitness_aggregator.compute_all_fitness(...)

    # 5. 分别进化 Main 和 Exploiter
    self._evolve_populations(fitness_results)
    self._evolve_exploiters(fitness_results)

    # 6. 检查是否注入对手池
    self._check_and_inject_to_pool()

    # 7. 保存检查点
    if self.generation % self.checkpoint_interval == 0:
        self._save_checkpoint()
```

## 竞技场分配方案

默认 54 个独立竞技场，每个竞技场运行 1 个 episode：

| 竞技场范围 | 数量 | 用途 |
|-----------|------|------|
| Arena 0-9 | 10 | 基准对战（全当前代） |
| Arena 10-17 | 8 | 散户泛化测试 |
| Arena 18-25 | 8 | 高级散户泛化测试 |
| Arena 26-33 | 8 | 庄家泛化测试 |
| Arena 34-41 | 8 | 做市商泛化测试 |
| Arena 42-49 | 8 | League Exploiter 训练（每类型 2 个） |
| Arena 50-53 | 4 | Main Exploiter 训练（每类型 1 个） |

## 适应度计算

以散户为例：

```python
散户 Main 适应度 = 加权平均 {
    基准竞技场(Arena 0-9) × 权重 1.0,
    散户泛化测试竞技场(Arena 10-17) × 权重 0.8
}
```

总权重 = 10×1.0 + 8×0.8 = 16.4（与原方案相同）

## 对手池注入条件

| 来源 | 注入条件 |
|------|---------|
| Main Agents | 里程碑间隔（每 50 代） |
| League Exploiter | 对历史对手平均胜率 > 55% |
| Main Exploiter | 对 Main Agents 胜率 > 55% |

## 存储结构

```
checkpoints/league_training/
├── opponent_pools/
│   ├── RETAIL/
│   │   ├── pool_index.json
│   │   ├── gen_050/
│   │   │   ├── metadata.json
│   │   │   ├── genomes.npz
│   │   │   └── networks.npz
│   │   └── ...
│   ├── RETAIL_PRO/
│   ├── WHALE/
│   └── MARKET_MAKER/
├── checkpoints/
│   └── gen_100.pkl
└── league_exploiters/
    ├── RETAIL_gen_100.pkl
    └── ...
```

## 使用方法

### 启动训练

```bash
# 基本训练
python scripts/train_league.py --rounds 200

# 指定竞技场数量
python scripts/train_league.py --num-arenas 16 --rounds 200

# 从检查点恢复
python scripts/train_league.py --resume checkpoints/league_training/checkpoints/gen_100.pkl

# 禁用 Exploiter
python scripts/train_league.py --no-league-exploiter
python scripts/train_league.py --no-main-exploiter

# 指定采样策略
python scripts/train_league.py --sampling-strategy uniform
```

### 代码调用

```python
from src.training.league import LeagueTrainer, LeagueTrainingConfig
from src.config.config import Config
from src.training.arena.config import MultiArenaConfig

config = Config()
multi_config = MultiArenaConfig(num_arenas=27, episodes_per_arena=2)
league_config = LeagueTrainingConfig(
    sampling_strategy='pfsp',
    enable_league_exploiter=True,
    enable_main_exploiter=True,
)

trainer = LeagueTrainer(config, multi_config, league_config)
trainer.setup()

for _ in range(200):
    stats = trainer.run_round()
    print(f"Gen {trainer.generation}: {stats}")

trainer.stop()
```

## 依赖模块

- `src/training/population.py`：种群管理、基因组序列化
- `src/training/arena/`：多竞技场并行训练
- `src/bio/brain/batch_network_cache.py`：批量网络缓存
- `src/config/config.py`：基础配置

## 性能预估

| 操作 | 时间影响 |
|------|---------|
| 竞技场分配 | < 1ms |
| 网络缓存加载（单类型） | 首次 50-200ms/条目 |
| 批量推理（多来源）| 增加约 30-50% |
| 对手池 I/O（单类型） | 散户 75MB ~500ms |
| Exploiter 进化 | 额外 10-20% |
| 总内存占用 | +500MB（缓存 5 代历史对手） |
