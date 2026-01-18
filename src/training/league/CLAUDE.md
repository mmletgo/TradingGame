# League Training Module (联盟训练模块)

基于 AlphaStar 联盟训练思路实现的多代对手池训练机制。

## 核心概念

### 问题背景

传统训练中，Agent 只学会克制同一代对手，可能出现"循环克制"（A克B，B克C，C克A），缺乏泛化能力。

### 解决方案

1. **历史对手池**：存储不同代的优秀策略
2. **Main Agents**：主要进化的 Agent，与混合对手竞争
3. **泛化测试**：Main Agents 与历史对手池中的对手对战，测试泛化能力

### 按类型分离架构

每种 Agent 类型（RETAIL, RETAIL_PRO, WHALE, MARKET_MAKER）都有独立的对手池（存储其他三种类型的历史版本）。

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
| `num_baseline_arenas` | 16 | 基准竞技场数量 |
| `num_generalization_arenas_per_type` | 12 | 每类型泛化测试竞技场数量 |
| `sampling_strategy` | `recency` | 采样策略：uniform/recency/diverse |

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
- `cleanup(current_generation)`：清理旧条目

采样策略：
- **uniform**：均匀随机采样
- **recency**（默认）：时间加权采样，权重 = 代数，越新的对手采样概率越高
- **diverse**：多样性采样，均匀间隔选择不同适应度的对手

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

### ArenaAllocator (arena_allocator.py)

竞技场分配器，将竞技场按训练目的分配。

竞技场类型：
- **baseline**：全当前代对战（基准）
- **generalization_test**：某类型 Main vs 其他类型历史版本

主要方法：
- `allocate(pool_manager)`：有历史对手时使用，分配完整的竞技场方案
- `allocate_no_historical()`：无历史对手时使用，全部分配为基准竞技场

分配结果数据结构：
```python
@dataclass
class ArenaAllocation:
    assignments: list[ArenaAssignment]
    baseline_arena_ids: list[int]
    generalization_arena_ids: dict[AgentType, list[int]]
```

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
        # 对手池为空时，全部为基准竞技场
        self._current_allocation = self.arena_allocator.allocate_no_historical()

    # 2. 确保历史对手网络已缓存
    self._ensure_historical_networks_cached()

    # 3. 运行 episodes
    round_stats = self._run_episodes()

    # 4. 汇总适应度并进化
    self._evolve_populations()

    # 5. 检查里程碑保存
    if generation % milestone_interval == 0:
        self._save_milestone()

    # 6. 清理对手池
    self.pool_manager.cleanup_all()
```

## 竞技场分配方案

默认 64 个独立竞技场，每个竞技场运行 1 个 episode：

| 竞技场范围 | 数量 | 用途 |
|-----------|------|------|
| Arena 0-15 | 16 | 基准对战（全当前代） |
| Arena 16-27 | 12 | 散户泛化测试 |
| Arena 28-39 | 12 | 高级散户泛化测试 |
| Arena 40-51 | 12 | 庄家泛化测试 |
| Arena 52-63 | 12 | 做市商泛化测试 |

总计：16 + 12×4 = 64 个竞技场

## 适应度计算

以散户为例：

```python
散户 Main 适应度 = 加权平均 {
    基准竞技场(Arena 0-15) × 权重 1.0,
    散户泛化测试竞技场(Arena 16-27) × 权重 0.8
}
```

总权重 = 16×1.0 + 12×0.8 = 25.6

## 对手池注入条件

| 来源 | 注入条件 |
|------|---------|
| Main Agents | 里程碑间隔（每 50 代） |

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
└── checkpoints/
    └── gen_100.pkl
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

# 指定采样策略
python scripts/train_league.py --sampling-strategy uniform
```

### 代码调用

```python
from src.training.league import LeagueTrainer, LeagueTrainingConfig
from src.config.config import Config
from src.training.arena.config import MultiArenaConfig

config = Config()
multi_config = MultiArenaConfig(num_arenas=64, episodes_per_arena=1)
league_config = LeagueTrainingConfig(
    sampling_strategy='recency',
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
| 总内存占用 | +500MB（缓存 5 代历史对手） |
