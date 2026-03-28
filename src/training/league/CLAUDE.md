# League Training Module (联盟训练模块)

基于 AlphaStar 联盟训练思路实现的混合竞技场训练机制。

## 核心概念

### 问题背景

传统训练中，Agent 只学会克制同一代对手，可能出现"循环克制"（A克B，B克C，C克A），缺乏泛化能力。

### 解决方案：混合竞技场

**所有竞技场中当前代和历史代精英 Agent 同场交易。** 历史对手从对手池中 PFSP 采样多代（默认6代），每代仅取 Top 5% 精英 Agent。NEAT 进化仅使用当前代适应度，代际适应度对比作为监控指标。

**关键设计**：
1. **历史对手池**：存储不同代的优秀策略
2. **混合竞技场**：当前代 + 历史代精英在同一市场中交易
3. **代际对比**：监控当前代相对于历史代的适应度提升

### 按类型分离架构

每种 Agent 类型（RETAIL_PRO, MARKET_MAKER）都有独立的对手池。

### 竞技场 Agent 组成

| 类型 | 当前代 | 历史代（6代×Top 5%精英）| 总计 | 历史占比 |
|------|--------|-------------------------|------|---------|
| 高级散户 | 2,400 | 720（6×5%×2400） | 3,120 | 23% |
| 做市商 | 600 | 180（6×5%×600） | 780 | 23% |
| 噪声交易者 | 300（增强） | - | 300 | - |

历史占比控制在约 23%，保持当前代对市场价格的主导影响力。`historical_elite_ratio` 可按需调整。

## 模块结构

```
src/training/league/
├── __init__.py              # 模块导出
├── CLAUDE.md                # 本文档
├── config.py                # 联盟训练配置
├── opponent_entry.py        # 对手条目数据结构
├── opponent_pool.py         # 单类型对手池管理
├── opponent_pool_manager.py # 多类型对手池管理器
├── arena_allocator.py       # 混合竞技场历史对手采样器
├── league_fitness.py        # 适应度汇总与代际对比
└── league_trainer.py        # 联盟训练器主类
```

---

## 核心类说明

### LeagueTrainingConfig (config.py)

联盟训练配置类，`__post_init__` 自动调用 `validate()` 校验所有参数合法性（包括 `pool_dir`/`checkpoint_dir` 非空、`sampling_strategy` 通过 `Literal` 类型约束、`convergence_fitness_std_threshold > 0`、`recency_decay_lambda > 0`、`pfsp_exponent > 0`、`pfsp_win_rate_ema_alpha ∈ (0, 1]`、`elite_ratio ∈ (0, 1]`、`min_freeze_generation >= 0`、`convergence_generations >= 1`、`generational_comparison_window >= 1`、`hybrid_noise_trader_count >= 0`、`convergence_generations <= generational_comparison_window`）。关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pool_dir` | `/mnt/work/TradingGame/league_training/opponent_pools` | 对手池存储目录 |
| `checkpoint_dir` | `/mnt/work/TradingGame/league_training/checkpoints` | 检查点存储目录 |
| `max_pool_size_per_type` | 100 | 每种类型最多保存的历史版本数 |
| `milestone_interval` | 1 | 里程碑保存间隔（代数） |
| `num_arenas` | 16 | 竞技场数量（对应物理核心数） |
| `episodes_per_arena` | 4 | 每竞技场 episode 数 |
| `num_historical_generations` | 6 | 每轮采样历史代数 |
| `historical_elite_ratio` | 0.05 | 每代取 Top 5% 精英 |
| `historical_freshness_ratio` | 0.5 | 采样中最近历史的最低占比 |
| `hybrid_noise_trader_count` | 300 | 混合竞技场噪声交易者数 |
| `hybrid_noise_trader_quantity_mu` | 10.0 | 噪声交易者下单量 mu |
| `sampling_strategy` | `pfsp` | 采样策略：uniform/recency/diverse/pfsp |
| `recency_decay_lambda` | 2.0 | 指数衰减速率 |
| `pfsp_exponent` | 2.0 | 败率加权指数 |
| `pfsp_explore_bonus` | 2.0 | 未交战对手探索奖励系数 |
| `pfsp_win_rate_ema_alpha` | 0.3 | 胜率 EMA 平滑因子 |
| `generational_comparison_window` | 20 | 代际对比历史窗口 |
| `convergence_fitness_std_threshold` | 0.005 | 适应度标准差收敛阈值 |
| `elite_ratio` | 0.1 | 精英比例（代际对比中精英平均适应度计算） |
| `convergence_generations` | 10 | 连续满足收敛条件的代数 |
| `freeze_on_convergence` | `True` | 收敛时是否冻结进化 |
| `freeze_thaw_threshold` | 0.05 | 适应度下降超过 5% 则解冻 |
| `min_freeze_generation` | 30 | 最早允许冻结的代数 |

---

### OpponentEntry (opponent_entry.py)

对手条目数据结构，包含：
- `OpponentMetadata`：元数据（entry_id, agent_type, source, win_rates 等），`from_dict()` 容忍未知字段实现向前兼容
- `genome_data`：基因组数据（序列化的 NEAT 基因组），`load()` 时 `genomes.npz` 不存在则为 `None`
- `network_data: dict[int, tuple[np.ndarray, ...]] | None`：网络参数（可选，延迟加载）
- `pre_evolution_fitness`：预进化适应度数据（可选，`{sub_pop_id: fitness_array}`）

**原子写入**：`save()` 先写入 `{entry_id}.tmp` 临时目录，全部完成后 `shutil.move` 替换目标目录，保存失败时清理临时目录。保存时自动更新 `metadata.agent_count`。sub_pop_id 解析统一使用正则 `re.match(r"^sub_(\d+)_...$", key)`。

存储格式（使用 `np.savez` 无压缩写入）：
```
entry_dir/
├── metadata.json                # 元数据
├── genomes.npz                  # 基因组数据（无压缩）
├── networks.npz                 # 网络参数（可选，无压缩）
└── pre_evolution_fitness.npz    # 预进化适应度（可选，无压缩）
```

---

### OpponentPool (opponent_pool.py)

单个 Agent 类型的对手池管理器。

**主要方法：**
- `add_entry(entry)`：添加对手条目
- `sample_opponents(n, strategy, target_type, current_generation)`：采样对手
- `sample_opponents_batch(n, strategy, target_type, current_generation)`：批量采样 n 个不重复对手
- `compute_weights(entries, strategy, target_type, current_generation)`：统一计算各策略的采样权重（公开方法，供 arena_allocator 调用）
- `update_entry_win_rate(entry_id, target_type, outcome, ema_alpha)`：更新条目胜率（EMA 平滑）
- `save_index()`：保存索引文件（原子写入：先写 `.tmp` 再 `os.rename` 替换）
- `load_index()`：加载索引文件，捕获 `json.JSONDecodeError`/`OSError`/`ValueError` 异常并备份损坏文件（`.json.bak`）
- `cleanup(current_generation)`：清理旧条目（优先删非里程碑，不够时删最旧的里程碑），操作前浅拷贝 entries 列表，批量删除后统一写入索引一次
- `list_entries()`：返回索引 entries 的浅拷贝（防止外部修改）
- `get_entry()`：从磁盘加载时记录 WARNING 日志
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
- outcome 定义：当前代同类型 Agent 平均适应度 > 0 即为"赢"（1.0）
- 设计意图：采用二元 outcome（0/1）是 PFSP 的标准做法，简化胜率计算同时保持有效的优先级排序

---

### OpponentPoolManager (opponent_pool_manager.py)

管理两种 Agent 类型的独立对手池。通过模块级常量 `LEAGUE_AGENT_TYPES = [AgentType.RETAIL_PRO, AgentType.MARKET_MAKER]` 显式枚举参与联盟训练的类型。

**主要方法：**
- `add_snapshot(generation, genome_data_map, ..., pre_evolution_fitness_map=None)`：批量保存快照（多线程并行写入各 AgentType，futures 结果收集带异常捕获和日志），可附带预进化适应度数据
- `get_pool(agent_type)`：获取指定类型的对手池
- `has_any_historical_opponents()`：检查是否有任何历史对手
- `cleanup_all(current_generation)`：清理所有类型的旧条目
- `load_all()` / `save_all()`：加载/保存所有对手池索引

---

### HybridArenaAllocator (arena_allocator.py)

混合竞技场历史对手采样器。负责从对手池中采样历史对手 entries（带新鲜度约束的 PFSP 采样）。

**数据结构：**

```python
@dataclass
class HybridSamplingResult:
    sampled_entries: dict[AgentType, list[str]]  # {AgentType: [entry_id, ...]}
    elite_networks: dict[AgentType, dict[str, tuple[int, tuple[np.ndarray, ...]]]]
    total_elite_counts: dict[AgentType, int]
```

**采样策略（带新鲜度约束的 PFSP 加权采样）：**
1. 仅遍历 `LEAGUE_AGENT_TYPES`，对每种类型独立采样 `num_historical_generations` 个历史 entries
2. 将对手池按代数排序，分为"最近 1/3（向上取整）"和"全池"
3. 使用 `pool.compute_weights()` 获取配置策略（默认 PFSP）的采样权重
4. `freshness_ratio <= 0` 时 `n_recent = 0`（全部从全池采样）；否则 `n_recent = max(1, n_total * freshness_ratio)`
5. 采样不足补偿：若最近池实际采样数不足 n_recent，差额补偿到全池采样数
6. 从全池（排除已选）中按 PFSP 权重无放回采样剩余数量

**主要方法：**
- `sample_historical(pool_manager, current_generation)` -> `HybridSamplingResult | None`：带新鲜度约束的历史对手采样，所有类型对手池为空时返回 None
- `_sample_with_freshness(pool, n_total, freshness_ratio, current_generation)` -> `list[str]`：单个对手池的新鲜度约束采样，统一使用 `pool.list_entries()` 索引作为数据源

**精英网络提取**由 LeagueTrainer 负责（因为需要加载 entry 数据），allocator 仅返回采样的 entry IDs，`elite_networks` 和 `total_elite_counts` 由调用方填充。

---

### HybridFitnessAggregator (league_fitness.py)

混合竞技场适应度汇总器，负责代际适应度对比和收敛判断。

**数据类：**

```python
@dataclass
class GenerationalComparisonStats:
    generation: int
    current_avg_fitness: dict[AgentType, float]   # 本代平均适应度
    previous_avg_fitness: dict[AgentType, float]   # 上一代平均适应度
    improvement: dict[AgentType, float]            # 提升量
    elite_current_avg: dict[AgentType, float]      # 本代精英平均
    elite_previous_avg: dict[AgentType, float]     # 上一代精英平均
    elite_improvement: dict[AgentType, float]      # 精英提升量
```

**主要方法：**
- `compute_generational_comparison(generation, fitness_arrays)` -> `GenerationalComparisonStats | None`：计算代际适应度对比（第一代返回 None），内部对 `previous_avg`/`elite_previous` 做浅拷贝防止引用共享
- `check_convergence()` -> `(bool, dict[AgentType, bool])`：双重收敛判断（种群和精英的 std 使用 `ddof=1` 样本标准差 ≤ 阈值）
- `get_first_convergence_generation()` -> `int | None`：获取首次收敛代数
- `get_state()` -> `dict`：获取可序列化状态（fitness_history, elite_fitness_history, first_convergence_generation），用于 checkpoint 持久化
- `set_state(state)` -> `None`：从 checkpoint 恢复状态
- `clear_history()`：清空历史

---

### LeagueTrainer (league_trainer.py)

联盟训练器主类，继承自 `ParallelArenaTrainer`。

**混合竞技场流程（run_round）：**

```
0. 等待上一轮 checkpoint 线程完成（防竞态）
1. 清理旧采样结果和历史数据
2. PFSP 采样历史对手（带新鲜度约束）
3. 提取精英网络，构建历史 AgentInfo 列表
4. 更新 Workers 的 agent_infos 和合并网络
   ├─ 有历史 Agent → 发送 combined agent_infos + 合并网络
   └─ 上轮有本轮无 → 发送仅当前代 agent_infos + 重新同步网络（update_agent_infos 会重建空缓存）
5. 缓存进化前代数 → super().run_round()（运行 episodes + NEAT 进化）
   ├─ 历史 Agent（sub_pop_id >= 1000）自动不参与进化
   └─ _sync_networks_to_workers() override 合并历史参数
6. 计算代际适应度对比（使用进化前代数）
7. 检查冻结/解冻（使用进化前代数）
8. 同步基因组 + 按 milestone_interval 间隔保存里程碑
8.5. 等待里程碑后台线程完成（防止与步骤9/10竞态）
9. 更新历史对手胜率（含 avg_fitnesses 回退逻辑）
10. 清理对手池
```

**Override 方法：**

| 方法 | 描述 |
|------|------|
| `_build_agent_infos()` | 追加历史 Agent 到父类结果 |
| `_sync_networks_to_workers()` | 合并当前代+历史精英网络后通过 SharedMemory 同步 |
| `_build_fitness_map()` | 缓存预进化适应度 + 排除冻结物种 |

**核心方法：**

| 方法 | 描述 |
|------|------|
| `_prepare_historical_agents()` | 加载历史 entries → 按需从基因组重建网络参数 → extract_elite_networks → 构建 AgentInfo 列表 |
| `_compute_generational_comparison()` | 过滤当前代适应度（sub_pop_id < 1000）→ 代际对比 |
| `_update_populations_from_evolution()` | Override: 保存 per-sub-pop 网络参数到 `_per_subpop_network_params` 后调用父类 |
| `_update_historical_win_rates()` | 根据当前代平均适应度 > 0 判断胜负，更新对手池 |
| `_check_freeze_thaw()` | 未冻结→收敛检查→冻结；已冻结→每代复评（双维度）→解冻 |
| `_save_milestone()` | 异步保存里程碑（附带预进化适应度） |
| `_collect_genome_data()` | 收集所有种群的基因组数据，供 `_save_milestone()` 和 `save_checkpoint()` 复用 |
| `_sync_genomes_if_needed()` | 从 Worker 同步基因组数据（checkpoint/milestone 保存前） |

**历史 Agent ID 方案：**
- 散户: `10,000,000 + entry_index × 1,000,000 + local_index`
- 做市商: `20,000,000 + entry_index × 1,000,000 + local_index`
- sub_pop_id: `1000 + entry_index`（自动不参与 NEAT 进化）

**合并后的 BatchNetworkCache 布局（以散户为例）：**
```
┌───────────────────────────┬──────────┬──────────┬───┬──────────┐
│ 当前代 (index 0..2399)    │ Entry_A  │ Entry_B  │...│ Entry_F  │
│   10子种群 × 240          │ 120精英  │ 120精英  │   │ 120精英  │
└───────────────────────────┴──────────┴──────────┴───┴──────────┘
总共 2400 + 6×120 = 3120 个网络
```

**预进化适应度机制：**
- `_build_fitness_map()` 在 NEAT 进化前被调用，在此缓存 `avg_fitness` 到 `self._pre_evolution_fitness`
- `_save_milestone()` 保存里程碑时，将 `_pre_evolution_fitness` 按 AgentType 分组后传给 `OpponentPoolManager.add_snapshot()`
- 预进化适应度通过 `OpponentEntry.pre_evolution_fitness` 字段持久化到 `pre_evolution_fitness.npz`
- 解决问题：里程碑保存的是进化后基因组，新生成 offspring fitness 为 None，仅幸存精英有有效 fitness，导致精英选择样本池被严重缩小

**精英网络提取（模块级函数）：**
- `extract_elite_networks(pre_evolution_fitness, network_data, elite_ratio, genome_data)` - 从历史 entry 中提取 Top 精英的网络参数
  - 按 `pre_evolution_fitness` 排序取 Top `elite_ratio`（默认 5%）
  - 回退机制：`pre_evolution_fitness` 不可用时从 `genome_data` 的 fitnesses 提取（过滤 NaN），精英数量基于有效样本数计算（而非含 NaN 的总数）
  - 返回 `(精英总数, packed_network_params_tuple)`

**网络参数重建（模块级函数，回退路径）：**
- `_reconstruct_network_data(genome_data, agent_type, config)` - 从基因组数据重建网络参数
  - 当历史 entry 没有 `networks.npz`（旧版 entry）时的回退路径，从 `genomes.npz` 反序列化基因组 → 创建 FastFeedForwardNetwork → 提取并打包网络参数
  - 正常流程里程碑保存时已写入 `networks.npz`，此函数仅处理旧数据兼容
  - 使用 `_deserialize_genomes_numpy` + `_extract_and_pack_all_network_params` 完成重建
  - 返回 `{sub_pop_id: packed_network_params_tuple}` 或 `None`
  - 性能：约 2.9s / 200 个网络（单子种群）

**冻结状态数据类：**

```python
@dataclass
class SpeciesFreezeState:
    is_frozen: bool = False
    freeze_generation: int = 0           # 冻结时的代数
    freeze_baseline_fitness: float = 0.0  # 冻结时的平均适应度
    freeze_elite_fitness: float = 0.0     # 冻结时的精英平均适应度
    thaw_count: int = 0                   # 解冻次数
```

**检查点系统：**
- `save_checkpoint()`：先调用 `_sync_genomes_if_needed()`，内联父类序列化逻辑 + league 数据（含 `fitness_aggregator_state`），主线程序列化到内存字节，后台守护线程仅负责写盘（不再调用 `pool_manager.save_all()`，避免线程竞态）
- `load_checkpoint()`：父类 `load_checkpoint()` 返回 `checkpoint_data` 字典，子类直接复用（避免重复反序列化），恢复冻结状态和 `fitness_aggregator` 适应度历史，加载后自动设置 `_had_historical_last_round = True`
- `train()` 中每代都调用 `checkpoint_callback`
- `train()` 中 `_libc` 句柄在模块级缓存，避免每次 `CDLL("libc.so.6")` 开销

---

## 竞技场方案

默认 16 个竞技场 × 4 个 episode（对应 16 个 CPU 物理核心），总计 64 个 episode/轮。

所有竞技场使用**相同的 Agent 集合**（当前代 + 历史代精英），复用现有"所有竞技场共享 agent_infos"架构。

**Agent 数量动态变化：**
- 第1代：无历史对手，竞技场仅有当前代 Agent（2,800 个）
- 第2代起：历史对手数量逐渐增加（对手池条目数 < 6 时采样可用数量）
- ArenaWorkerPool 通过 `update_agent_infos` 命令支持每轮动态更新

---

## 适应度计算

混合竞技场中适应度计算流程：

1. **所有竞技场** 简单平均（不再区分 baseline/generalization）
2. **当前代 Agent**（sub_pop_id < 1000）参与 NEAT 进化
3. **历史代 Agent**（sub_pop_id >= 1000）自动被排除（不在 populations 的 sub_pop_id 范围内）

---

## 代际适应度对比

替代原有"泛化优势比"的监控指标，用于评估训练进展和判断收敛。

### 计算方式

每代计算当前代 Agent 的种群平均和精英平均适应度，与上一代对比。

### 双重收敛判断

收敛采用"双重收敛"机制，同时监控种群和精英的收敛状态：

1. **种群收敛**：最近 `convergence_generations` 代的种群平均适应度标准差 ≤ `convergence_fitness_std_threshold`
2. **精英收敛**：最近 `convergence_generations` 代的精英平均适应度标准差 ≤ `convergence_fitness_std_threshold`

只有当种群和精英都收敛时，才判定为真正收敛。

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
1. 某物种达到双重收敛 → 记录当前平均适应度作为基准 → 冻结
2. 冻结物种不参与 NEAT 进化（从 `_build_fitness_map` 中排除）

**每代复评（双维度）：**
- 冻结物种每代都在混合竞技场参与交易，因此每代都复评
- 复评指标：种群平均下降 `pop_drop` 和精英平均下降 `elite_drop`，取 `max(pop_drop, elite_drop)` 作为最终下降比例
- 下降比例公式：`(freeze_value - current_value) / max(|freeze_value|, 0.01)`（分母使用 `max(abs, 0.01)` 避免除零）
- 下降比例超过 `freeze_thaw_threshold`（默认 5%）则解冻

**隐式冷却期**：解冻后需要 `convergence_generations`（10 代）连续收敛才会重新冻结，天然防止快速反复冻结/解冻。

**训练完成**：所有 2 种物种均冻结时，训练自动完成。

**冻结状态持久化**：`SpeciesFreezeState` 随检查点保存/恢复。

---

### 日志输出示例

```
INFO - 第 100 代代际对比:
INFO -   RETAIL_PRO: 种群=0.0893(上代=0.0806,变化=+0.0087) | 精英=0.1234(上代=0.1189,变化=+0.0045) [提升中]
INFO -   MARKET_MAKER: 种群=0.0567(上代=0.0565,变化=+0.0002) | 精英=0.0789(上代=0.0786,变化=+0.0003) [已收敛]
```

冻结后：
```
INFO - 物种 MARKET_MAKER 已冻结 (第 120 代, avg=0.0567, elite=0.0789)
```

复评日志（每10代 INFO，其余 DEBUG；解冻事件始终 INFO）：
```
INFO - 物种 MARKET_MAKER 复评 (冻结于第 120 代): 种群avg: 冻结=0.0567 当前=0.0550 下降=0.0300 | 精英avg: 冻结=0.0789 当前=0.0770 下降=0.0241 | 取max=0.0300, 阈值=0.0500
INFO - 物种 MARKET_MAKER 保持冻结 (下降 3.00%, 未下降超过阈值 5.00%)
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
/mnt/work/TradingGame/league_training/
├── opponent_pools/
│   ├── RETAIL_PRO/
│   │   ├── pool_index.json
│   │   ├── gen_050/
│   │   │   ├── metadata.json
│   │   │   ├── genomes.npz
│   │   │   ├── networks.npz
│   │   │   └── pre_evolution_fitness.npz
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
# 基本训练（默认 16 竞技场 × 4 episode）
python scripts/train_league.py --rounds 200

# 从检查点恢复
python scripts/train_league.py --resume /mnt/work/TradingGame/league_training/checkpoints/gen_100.pkl

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
multi_config = MultiArenaConfig(num_arenas=16, episodes_per_arena=4)
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
genome_data: dict[int, tuple[...]] | None              # 保存后可置为 None
network_data: dict[int, tuple[np.ndarray, ...]] | None  # 延迟加载
```

**优化策略**：保存到磁盘后，清理内存中的大数据字段，只保留元数据。需要时从磁盘重新加载。

**NpzFile 管理**：`load()` 方法使用 `with np.load() as f:` 上下文管理器确保 NpzFile 及时关闭文件描述符和内存映射，并用 `np.array()` 拷贝独立数组。

### OpponentPool 清理

- `add_entry()`：保存后自动清理 `genome_data` 和 `network_data`
- `remove_entry()`：删除前显式清理大数据字段
- `get_entry(load_networks=True)`：赋值字段到已有对象（而非替换对象），确保外部引用的清理能生效
- `clear_memory_cache()`：批量清理所有条目的内存缓存

### LeagueTrainer 清理

- `_prepare_historical_agents()`：加载 entry 后立即清理 `genome_data`/`network_data`/`pre_evolution_fitness`
- `_sync_genomes_if_needed()` 在 `run_round()` 中里程碑保存前统一从 Worker 同步基因组
- `_save_milestone()` 数据收集在主线程完成，I/O 操作在后台守护线程异步执行。里程碑保存时传入 `_per_subpop_network_params` 作为 `network_data_map`，写入 `networks.npz`，后续加载历史对手时直接使用，无需 `_reconstruct_network_data` 重建
- `save_checkpoint()` 先调用 `_sync_genomes_if_needed()` 兜底，主线程序列化后删除中间变量，文件写入在后台线程执行
- 每代执行轻量级 NEAT 历史清理（`_cleanup_neat_history_light`）
- 每 5 代清理对手池内存缓存 + gc.collect + malloc_trim
- `run_round()` 开始时显式清理旧 `_current_sampling_result` 和历史数据

### 清理时机

| 时机 | 清理操作 |
|------|---------|
| 精英提取后 | 清理 entry 的 genome_data/network_data/pre_evolution_fitness |
| 里程碑保存后 | 删除序列化中间变量 |
| 检查点序列化后 | 删除 checkpoint_data + gc.collect，文件写入异步 |
| 每代 | 轻量级 NEAT 历史清理（genome_to_species、stagnation、ancestors） |
| 每 5 代 | 对手池 clear_memory_cache + gc.collect + malloc_trim |

---

## 性能预估

| 操作 | 时间影响 |
|------|---------|
| 历史对手采样 | < 1ms |
| 精英网络提取（单 entry） | 50-200ms |
| 网络参数拼接 | < 10ms |
| 对手池 I/O（单类型） | 无压缩 npz，文件更大但写入更快 |

### 保存性能优化

| 优化项 | 方式 | 效果 |
|--------|------|------|
| 去掉 gzip 压缩 | `np.savez` 替代 `np.savez_compressed` | 写入速度提升 3-5 倍 |
| 多线程并行写入 | `ThreadPoolExecutor` 并行各 AgentType | 多类型写入时间趋近最慢的单类型 |
| checkpoint 异步写盘 | 主线程 `pickle.dumps` 后后台线程写文件 | 训练主循环不再阻塞于磁盘 I/O |

---

## 依赖模块

- `src.training.population`：种群管理、基因组序列化/反序列化（`_deserialize_genomes_numpy`）、网络参数提取（`_extract_and_pack_all_network_params`）、网络参数拼接（`_concat_network_params_numpy`）
- `src.training.arena`：多竞技场并行训练（`ParallelArenaTrainer`、`AgentInfo`、`SharedNetworkMemory`）
- `src.config.config`：基础配置
