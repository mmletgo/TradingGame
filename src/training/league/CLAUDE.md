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

每种 Agent 类型（RETAIL_PRO, MARKET_MAKER）都有独立的对手池（存储另一种类型的历史版本）。

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

---

## 核心类说明

### LeagueTrainingConfig (config.py)

联盟训练配置类，关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pool_dir` | `checkpoints/league_training/opponent_pools` | 对手池存储目录 |
| `checkpoint_dir` | `checkpoints/league_training/checkpoints` | 检查点存储目录 |
| `max_pool_size_per_type` | 20 | 每种类型最多保存的历史版本数 |
| `milestone_interval` | 50 | 里程碑保存间隔（代数），CLI 默认 1（每代保存） |
| `num_baseline_arenas` | 16 | 基准竞技场数量 |
| `num_generalization_arenas_per_type` | 24 | 每类型泛化测试竞技场数量 |
| `sampling_strategy` | `pfsp` | 采样策略：uniform/recency/diverse/pfsp |
| `recency_decay_lambda` | 2.0 | 指数衰减速率，越大衰减越快 |
| `pfsp_exponent` | 2.0 | 败率加权指数，越大越集中于难对手 |
| `pfsp_explore_bonus` | 2.0 | 未交战对手的探索奖励系数 |
| `pfsp_win_rate_ema_alpha` | 0.3 | 胜率 EMA 平滑因子，越大越重视近期 |
| `generalization_advantage_window` | 20 | 泛化优势比历史窗口大小 |
| `convergence_threshold` | 0.005 | 收敛判断阈值 |
| `convergence_generations` | 10 | 连续满足收敛条件的代数 |
| `elite_ratio` | 0.1 | 精英比例，用于计算精英适应度 |
| `freeze_on_convergence` | `True` | 收敛时是否冻结进化 |
| `freeze_thaw_threshold` | 0.05 | 基准适应度下降超过 5% 则解冻 |
| `min_freeze_generation` | 30 | 最早允许冻结的代数，防止早期误冻结 |
| `fitness_strategy` | `weighted_average` | 适应度汇总策略：simple/weighted_average/min |
| `baseline_weight` | 1.0 | 基准竞技场权重 |
| `generalization_weight` | 0.8 | 泛化竞技场权重 |

#### 泛化优势比参数详解

**`generalization_advantage_window`（历史窗口大小）**

控制 `_advantage_history` 队列的最大长度。该队列存储最近 N 代的泛化优势比统计数据（`GeneralizationAdvantageStats`）。

- **作用**：为收敛判断提供历史数据基础
- **建议**：设为 `convergence_generations` 的 1.5~2 倍

**`convergence_threshold`（收敛阈值）**

判断单代是否"趋于收敛"的阈值。当某类型的泛化优势比绝对值 <= 该阈值时，认为该类型在这一代趋于收敛。

- **计算**：`|泛化平均适应度 - 基准平均适应度| <= threshold`
- **示例**：
  - `0.01` 表示泛化表现与基准表现的差距在 1% 以内视为收敛
  - `0.005` 表示允许 0.5% 的差距

**`convergence_generations`（连续收敛代数）**

判断最终收敛需要连续满足条件的代数。只有最近连续 N 代都满足条件，才判定为真正收敛。

- **前置条件**：`len(_advantage_history) >= convergence_generations` 时才开始判断

**三者协作关系**

```
第 90 代: 泛化优势比 = +0.025  [不满足 <= 0.01]
第 91 代: 泛化优势比 = +0.012  [不满足]
第 92 代: 泛化优势比 = +0.008  [满足]
第 93 代: 泛化优势比 = -0.005  [满足]
第 94 代: 泛化优势比 = +0.003  [满足]
...
第 101 代: 泛化优势比 = +0.002 [满足]
→ 最近 10 代（92-101）都满足 <= 0.01，判定为收敛
```

**`elite_ratio`（精英比例）**

用于计算精英平均适应度的比例。取种群中适应度排名前 N% 的个体计算平均值。

- **计算**：`n_elite = max(1, int(len(fitness_array) * elite_ratio))`
- **示例**：`0.1` 表示取 top 10% 的个体

**调参建议**

| 场景 | window | threshold | generations | elite_ratio | 说明 |
|------|--------|-----------|-------------|-------------|------|
| 快速验证 | 15 | 0.02 | 5 | 0.2 | 宽松条件，快速得到结论 |
| 默认配置 | 20 | 0.01 | 10 | 0.1 | 平衡稳健性和效率 |
| 严格收敛 | 30 | 0.005 | 15 | 0.05 | 严格条件，确保充分收敛 |

---

### OpponentEntry (opponent_entry.py)

对手条目数据结构，包含：
- `OpponentMetadata`：元数据（entry_id, agent_type, source, win_rates 等）
- `genome_data`：基因组数据（序列化的 NEAT 基因组）
- `network_data`：网络参数（可选，延迟加载）

存储格式（使用 `np.savez` 无压缩写入，避免 gzip 压缩开销）：
```
entry_dir/
├── metadata.json    # 元数据
├── genomes.npz      # 基因组数据（无压缩）
└── networks.npz     # 网络参数（可选，无压缩）
```

---

### OpponentPool (opponent_pool.py)

单个 Agent 类型的对手池管理器。

**主要方法：**
- `add_entry(entry)`：添加对手条目
- `sample_opponents(n, strategy, target_type, current_generation)`：采样对手
- `sample_opponents_batch(n, strategy, target_type, current_generation)`：批量采样 n 个不重复对手（用于多个泛化竞技场）
- `update_entry_win_rate(entry_id, target_type, outcome, ema_alpha)`：更新条目胜率（EMA 平滑）
- `cleanup(current_generation)`：清理旧条目（优先删非里程碑，不够时删最旧的里程碑）
- `clear_memory_cache()`：清理所有条目的内存缓存（genome_data, network_data 置 None）

**采样策略：**

| 策略 | 描述 | 权重公式 |
|------|------|---------|
| `uniform` | 均匀随机采样 | 1/N |
| `recency` | 指数衰减时间加权 | `exp(-λ × Δgen / milestone_interval)` |
| `diverse` | 多样性采样 | 按适应度均匀间隔 |
| `pfsp` | Prioritized Fictitious Self-Play | `f(win_rate) × recency × exploration_bonus` |

**PFSP 采样详解：**
```
p(opponent) ∝ f(win_rate) × recency_factor × exploration_bonus

- f(win_rate) = (1 - win_rate)^p（败率越高权重越大）
- recency_factor = exp(-λ × Δgen / milestone_interval)
- exploration_bonus = max(1.0, bonus / sqrt(match_count + 1))（未交战对手高权重）
```

**胜率跟踪：**
- `win_rates` 和 `match_counts` 存储在 `pool_index.json` 的每个条目中
- key 格式为 `vs_{AgentType.value}`（如 `vs_RETAIL_PRO`）
- 更新方式：EMA 平滑，`win_rate = (1 - α) × old + α × outcome`
- outcome 定义：目标物种在泛化竞技场中的平均适应度 > 0 即为"赢"（1.0）

**批量采样：**
- 对手池 >= N 时：无放回采样 N 个不同对手
- 对手池 < N 时：每个对手至少出现 `N // pool_size` 次，剩余按策略权重补足

---

### OpponentPoolManager (opponent_pool_manager.py)

管理两种 Agent 类型的独立对手池。

**主要方法：**
- `add_snapshot(generation, populations, source, add_reason)`：批量保存快照（多线程并行写入各 AgentType）
- `sample_opponents_for_arena(target_type, strategy, current_generation)`：为训练采样对手
- `sample_opponents_batch_for_type(target_type, n_arenas, strategy, current_generation)`：批量采样保证多样性
- `update_win_rates_from_round(allocation, arena_fitnesses, ema_alpha)`：从一轮训练结果批量更新所有对手胜率
- `cleanup_all(current_generation)`：清理所有类型的旧条目
- `load_all()` / `save_all()`：加载/保存所有对手池索引

---

### MultiGenerationNetworkCache (multi_gen_cache.py)

多代网络缓存管理器，按 Agent 类型和 entry_id 管理网络缓存。

**缓存类型：**
- `current_caches`：当前代网络缓存
- `historical_caches`：历史代网络缓存（LRU 淘汰）

**主要方法：**
- `ensure_cached(agent_type, entry_id, pool_manager)`：确保指定条目的网络已缓存
- `clear_all()`：清理所有缓存

**历史缓存创建流程：**
- 使用 `_concat_network_params_numpy` 合并子种群参数
- 通过 `BatchNetworkCache(num_networks, cache_type, num_threads)` 正确构造缓存
- 使用 `update_networks_from_numpy` 直接填充 C 结构

---

### ArenaAllocator (arena_allocator.py)

竞技场分配器，将竞技场按训练目的分配。

**数据结构：**

```python
@dataclass
class AgentSourceConfig:
    source: Literal['current', 'historical']
    entry_id: str | None = None  # historical 时需要指定

@dataclass
class ArenaAssignment:
    arena_id: int
    purpose: Literal['baseline', 'generalization_test']
    agent_sources: dict[AgentType, AgentSourceConfig]
    target_type: AgentType | None  # 泛化测试时的目标类型

@dataclass
class ArenaAllocation:
    assignments: list[ArenaAssignment]
    baseline_arena_ids: list[int]
    generalization_arena_ids: dict[AgentType, list[int]]
```

**竞技场类型：**
- **baseline**：全当前代对战（基准）
- **generalization_test**：某类型 Main vs 其他类型历史版本

**主要方法：**
- `allocate(pool_manager, frozen_types, current_generation)`：有历史对手时使用，冻结物种的泛化竞技场转为额外的 baseline 竞技场。使用批量采样保证对手多样性。
- `allocate_no_historical()`：无历史对手时使用，全部分配为基准竞技场

---

### LeagueFitnessAggregator (league_fitness.py)

适应度汇总器，从多个竞技场收集并汇总适应度。

**主要方法：**
- `aggregate_main_fitness(agent_type, baseline_fitnesses, generalization_fitnesses)`：计算 Main Agents 最终适应度
- `collect_fitness_by_role(allocation, arena_fitnesses)`：按角色收集适应度
- `compute_generalization_advantage(generation, allocation, arena_fitnesses)`：计算泛化优势比
- `check_convergence()`：检查是否收敛（双重收敛）
- `get_advantage_history()`：获取历史记录
- `clear_history()`：清空历史

**汇总策略：**
- **simple**：简单平均
- **weighted_average**：加权平均（基准权重 1.0，泛化权重 0.8）
- **min**：取最小值

### GeneralizationAdvantageStats (league_fitness.py)

泛化优势比统计数据类（支持双重收敛判断）：

```python
@dataclass
class GeneralizationAdvantageStats:
    generation: int
    # 种群级别
    advantages: dict[AgentType, float]          # 种群泛化优势比
    baseline_avg: dict[AgentType, float]        # 种群基准平均适应度
    generalization_avg: dict[AgentType, float]  # 种群泛化平均适应度
    # 精英级别
    elite_advantages: dict[AgentType, float]          # 精英泛化优势比
    elite_baseline_avg: dict[AgentType, float]        # 精英基准平均适应度
    elite_generalization_avg: dict[AgentType, float]  # 精英泛化平均适应度
```

---

### LeagueTrainer (league_trainer.py)

联盟训练器主类，继承自 `ParallelArenaTrainer`。

**主要流程（run_round）：**

```python
def run_round(self):
    # 1. 清理旧 allocation
    # 2. 分配竞技场（冻结物种的泛化竞技场转为 baseline）
    if self.pool_manager.has_any_historical_opponents():
        self._current_allocation = self.arena_allocator.allocate(
            self.pool_manager, frozen_types, current_generation=self.generation
        )
    else:
        self._current_allocation = self.arena_allocator.allocate_no_historical()

    # 3. 确保历史对手网络已缓存
    self._ensure_historical_networks_cached()

    # 4. 运行 episodes（复用父类逻辑）
    round_stats = super().run_round(episode_callback=episode_callback)

    # 5. 计算泛化优势比
    self._compute_and_log_generalization_advantage(round_stats, has_historical)

    # 6. 更新历史对手胜率（EMA 平滑，用于 PFSP 采样）
    self.pool_manager.update_win_rates_from_round(allocation, arena_fitnesses, ema_alpha)

    # 7. 检查冻结/解冻
    self._check_freeze_thaw(round_stats)

    # 8. 同步基因组 + 保存里程碑
    self._sync_genomes_if_needed()
    if generation > 0:
        self._save_milestone()

    # 9. 清理对手池
    self.pool_manager.cleanup_all()
```

**冻结相关方法：**
- `_build_fitness_map()` - 覆写父类方法，排除冻结物种的适应度（使其不参与进化）
- `_check_freeze_thaw(round_stats)` - 检查未冻结物种的双重收敛→冻结，已冻结物种的定期复评→解冻
- `_reevaluate_frozen_species(agent_type, round_stats)` - 复评单个冻结物种，比较 baseline 适应度下降比例

**冻结状态数据类：**

```python
@dataclass
class SpeciesFreezeState:
    is_frozen: bool = False
    freeze_generation: int = 0           # 冻结时的代数
    freeze_baseline_fitness: float = 0.0  # 冻结时的 baseline 平均适应度
    freeze_elite_fitness: float = 0.0     # 冻结时的精英 baseline 平均适应度
    thaw_count: int = 0                   # 解冻次数
```

**基因组同步：**
- `_sync_genomes_if_needed()`：lite 模式下进化后基因组留在 Worker 进程中（`_pending_genome_data=None`），此方法从 Worker 同步基因组数据并设置 `_pending_genome_data`，使 `_save_milestone()` 和 `save_checkpoint()` 可直接引用

**检查点系统：**
- `save_checkpoint()`：先调用 `_sync_genomes_if_needed()`，内联父类序列化逻辑 + league 数据，主线程序列化到内存字节，后台守护线程异步写盘
- `load_checkpoint()`：父类已有 magic bytes 检测（兼容 gzip 和 plain pickle），league 数据使用 plain pickle 读取
- `train()` 中每代都调用 `checkpoint_callback`

---

## 竞技场分配方案

默认 64 个独立竞技场，每个竞技场运行 1 个 episode：

| 竞技场范围 | 数量 | 用途 |
|-----------|------|------|
| Arena 0-15 | 16 | 基准对战（全当前代） |
| Arena 16-39 | 24 | 高级散户泛化测试 |
| Arena 40-63 | 24 | 做市商泛化测试 |

总计：16 + 24×2 = 64 个竞技场

**冻结物种的竞技场重分配**：冻结物种的泛化竞技场转为额外的 baseline 竞技场。例如 MARKET_MAKER 冻结后：16+24=40 个 baseline，24 个泛化（RETAIL_PRO）。

---

## 适应度计算

以高级散户为例：

```python
高级散户 Main 适应度 = 加权平均 {
    基准竞技场(Arena 0-15) × 权重 1.0,
    高级散户泛化测试竞技场(Arena 16-39) × 权重 0.8
}
```

总权重 = 16×1.0 + 24×0.8 = 35.2

**权重分布分析：**
- 总权重 = 16 × 1.0 + 24 × 0.8 × 2 = 16 + 38.4 = 54.4
- 基准占比: 16/54.4 ≈ 29%
- 泛化占比: 38.4/54.4 ≈ 71%

---

## 泛化优势比（Generalization Advantage）

用于评估训练效果和判断收敛的指标。

### 计算公式

```
泛化优势比 = 泛化平均适应度 - 基准平均适应度
```

- **泛化平均适应度**：Main Agents 在泛化测试竞技场（与历史对手对战）中的平均适应度
- **基准平均适应度**：Main Agents 在基准竞技场（全当前代对战）中的平均适应度

### 含义解读

| 泛化优势比 | 含义 |
|-----------|------|
| > 0 | Main 能击败历史对手（继续提升中） |
| < 0 | Main 不如历史对手（需继续训练） |
| ≈ 0 | 可能收敛 |

### 双重收敛判断

收敛采用"双重收敛"机制，同时监控种群和精英的收敛状态：

1. **种群收敛**：最近 `convergence_generations` 代的种群泛化优势比绝对值都 <= `convergence_threshold`
2. **精英收敛**：最近 `convergence_generations` 代的精英泛化优势比绝对值都 <= `convergence_threshold`

只有当种群和精英都收敛时，才判定为真正收敛。

**设计原因**：
- 防止假阳性：平均已收敛但精英还在提升
- 防止假阴性：精英已收敛但探索个体拉低平均值

**收敛状态：**

| 种群收敛 | 精英收敛 | 状态 |
|---------|---------|------|
| 是 | 是 | 双重收敛（真正收敛） |
| 是 | 否 | 种群收敛（精英仍在提升） |
| 否 | 是 | 精英收敛（探索个体拉低平均） |
| 否 | 否 | 未收敛 |

---

### 物种冻结与定期复评

物种达到双重收敛后，冻结其 NEAT 进化（基因组不再变异/交叉），但仍作为对手参与所有竞技场。

**冻结流程：**
1. 某物种达到双重收敛 → 记录当前 baseline 适应度作为基准 → 冻结
2. 冻结物种的泛化竞技场转为额外的 baseline 竞技场（更稳定的适应度评估）
3. 冻结物种不参与 NEAT 进化（从 `_build_fitness_map` 中排除）

**每代复评：**
- 冻结物种每代都在 baseline 竞技场参与对战，因此每代都复评
- 复评指标：当前 baseline 平均适应度 vs 冻结时的 baseline 平均适应度
- 下降比例 `(freeze_fitness - current_fitness) / |freeze_fitness|` 超过 `freeze_thaw_threshold`（默认 5%）则解冻

**隐式冷却期**：解冻后需要 `convergence_generations`（10 代）连续收敛才会重新冻结，天然防止快速反复冻结/解冻。

**训练完成**：所有 2 种物种均冻结时，训练自动完成。

**冻结状态持久化**：`SpeciesFreezeState` 随检查点保存/恢复。

---

### 日志输出示例

```
INFO - 第 100 代泛化优势比:
INFO -   RETAIL_PRO: 种群=-0.0087(基准=0.0893,泛化=0.0806) | 精英=-0.0045(基准=0.1234,泛化=0.1189) [不如历史表现]
INFO -   MARKET_MAKER: 种群=+0.0012(基准=0.0567,泛化=0.0579) | 精英=+0.0003(基准=0.0789,泛化=0.0792) [双重收敛]
```

冻结后：
```
INFO - 物种 MARKET_MAKER 已冻结 (第 120 代, baseline=0.0567, elite_baseline=0.0789)
INFO - 第 140 代泛化优势比:
INFO -   RETAIL_PRO: 种群=+0.0023(基准=0.0512,泛化=0.0535) | 精英=+0.0015(基准=0.0823,泛化=0.0838) [双重收敛]
INFO -   MARKET_MAKER: 种群=+0.0005(基准=0.0567,泛化=0.0572) | 精英=+0.0008(基准=0.0789,泛化=0.0797) [已冻结(第120代起)]
```

复评日志：
```
INFO - 物种 MARKET_MAKER 复评 (冻结于第 120 代): 冻结时baseline=0.0567, 当前baseline=0.0550, 下降比例=0.0300, 阈值=0.0500
INFO - 物种 MARKET_MAKER 保持冻结 (下降 3.00% <= 5.00%)
```

所有物种冻结时：
```
INFO - >>> 所有物种已冻结，训练即将完成 <<<
INFO - 所有物种已冻结，训练完成
```

---

## 对手池注入条件

| 来源 | 注入条件 |
|------|---------|
| Main Agents | 每代保存里程碑（CLI 默认 `--milestone-interval 1`） |

---

## 存储结构

```
checkpoints/league_training/
├── opponent_pools/
│   ├── RETAIL_PRO/
│   │   ├── pool_index.json
│   │   ├── gen_050/
│   │   │   ├── metadata.json
│   │   │   ├── genomes.npz
│   │   │   └── networks.npz
│   │   └── ...
│   └── MARKET_MAKER/
│       ├── pool_index.json
│       └── ...
└── checkpoints/
    └── gen_00100.pkl
```

---

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

# 指定里程碑保存间隔
python scripts/train_league.py --milestone-interval 10 --rounds 200
```

### 代码调用

```python
from src.training.league import LeagueTrainer, LeagueTrainingConfig
from src.training.arena import MultiArenaConfig
from src.config.config import Config

config = Config()
multi_config = MultiArenaConfig(num_arenas=64, episodes_per_arena=1)
league_config = LeagueTrainingConfig(
    sampling_strategy='pfsp',
    freeze_on_convergence=True,
)

trainer = LeagueTrainer(config, multi_config, league_config)
trainer.setup()

for _ in range(200):
    stats = trainer.run_round()
    print(f"Gen {trainer.generation}: {stats}")
    if stats.get('all_species_frozen', False):
        break

trainer.stop()
```

---

## 内存管理

联盟训练涉及大量历史对手数据，需要特别注意内存管理。

### OpponentEntry 内存优化

`OpponentEntry` 的 `genome_data` 和 `network_data` 字段可以为 `None`：

```python
genome_data: dict[int, tuple[...]] | None  # 保存后可置为 None
network_data: dict[int, tuple] | None      # 延迟加载
```

**优化策略**：保存到磁盘后，清理内存中的大数据字段，只保留元数据。需要时从磁盘重新加载。

**NpzFile 管理**：`load()` 方法使用 `with np.load() as f:` 上下文管理器确保 NpzFile 及时关闭文件描述符和内存映射，并用 `np.array()` 拷贝独立数组。

### OpponentPool 清理

- `add_entry()`：保存后自动清理 `genome_data` 和 `network_data`
- `remove_entry()`：删除前显式清理大数据字段
- `get_entry(load_networks=True)`：赋值字段到已有对象（而非替换对象），确保外部引用的清理能生效
- `clear_memory_cache()`：批量清理所有条目的内存缓存

### MultiGenerationNetworkCache 清理

- LRU 淘汰时显式调用 `cache.clear()` 并删除引用
- 创建缓存后清理中间变量（`all_params_list`、`networks` 列表）
- `ensure_cached()` 创建缓存后清理 entry 的 `genome_data` 和 `network_data`

### LeagueFitnessAggregator 清理

- `np.concatenate()` 产生的临时数组在使用后立即删除
- `collected` 字典在使用后删除

### LeagueTrainer 清理

- `_current_round_arena_fitnesses` 清空后调用 `gc.collect(0)`
- `_sync_genomes_if_needed()` 在 `run_round()` 中里程碑保存前统一从 Worker 同步基因组
- `_save_milestone()` 数据收集在主线程完成，I/O 操作在后台守护线程异步执行
- `save_checkpoint()` 先调用 `_sync_genomes_if_needed()` 兜底，主线程序列化后删除中间变量，文件写入在后台线程执行
- 每代执行轻量级 NEAT 历史清理（`_cleanup_neat_history_light`）
- 每 5 代清理对手池内存缓存
- `run_round()` 开始时显式清理旧 `_current_allocation`

### 清理时机

| 时机 | 清理操作 |
|------|---------|
| 里程碑保存后 | 删除序列化中间变量 |
| 检查点序列化后 | 删除 checkpoint_data + gc.collect，文件写入异步 |
| 进化完成后 | 清空 arena_fitnesses + gc.collect(0) |
| 每代 | 轻量级 NEAT 历史清理（genome_to_species、stagnation、ancestors） |
| 每 5 代 | 对手池 clear_memory_cache + gc.collect + malloc_trim |
| 缓存 LRU 淘汰 | 显式 clear() + del |
| ensure_cached 后 | 清理 entry 的 genome_data/network_data |

---

## 性能预估

| 操作 | 时间影响 |
|------|---------|
| 竞技场分配 | < 1ms |
| 网络缓存加载（单类型） | 首次 50-200ms/条目 |
| 批量推理（多来源） | 增加约 30-50% |
| 对手池 I/O（单类型） | 无压缩 npz，文件更大但写入更快 |
| 总内存占用 | +500MB（缓存 5 代历史对手） |

### 保存性能优化

| 优化项 | 方式 | 效果 |
|--------|------|------|
| 去掉 gzip 压缩 | `np.savez` 替代 `np.savez_compressed` | 写入速度提升 3-5 倍 |
| 多线程并行写入 | `ThreadPoolExecutor` 并行各 AgentType | 多类型写入时间趋近最慢的单类型 |
| checkpoint 异步写盘 | 主线程 `pickle.dumps` 后后台线程写文件 | 训练主循环不再阻塞于磁盘 I/O |

---

## 依赖模块

- `src.training.population`：种群管理、基因组序列化
- `src.training.arena`：多竞技场并行训练
- `src.bio.brain.batch_network_cache`：批量网络缓存
- `src.config.config`：基础配置
