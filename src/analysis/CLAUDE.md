# Analysis 模块

## 模块概述

分析模块提供三个核心功能：
1. **演示模式分析** - 分析演示结束时各物种存活个体的分布，生成可视化图表和终端摘要
2. **进化效果测试** - 通过对比测试评估 NEAT 进化算法的有效性
3. **Checkpoint 加载** - 从训练检查点加载基因组数据用于分析和测试

## 模块架构

```
src/analysis/
├── __init__.py              # 模块导出
├── checkpoint_loader.py     # Checkpoint 加载器
├── demo_analyzer.py         # 演示模式分析器
└── evolution_tester.py      # 进化效果测试器
```

## 模块导出

```python
from src.analysis import (
    CheckpointLoader,    # Checkpoint 加载器
    DemoAnalyzer,        # 演示模式分析器
    EvolutionTester,     # 进化效果测试器
)
```

---

## 核心类

### CheckpointLoader (checkpoint_loader.py)

Checkpoint 加载器，从主 checkpoint 文件加载所有基因组数据。

**支持的 Checkpoint 格式：**
- `multi_arena_gen_{N}.pkl`: 文件名直接对应代数
- `ep_{episode}.pkl`: 代数 = episode / evolution_interval（默认10）

**自动压缩检测：**
- 通过检测 magic bytes（0x1f 0x8b）自动识别 gzip 压缩文件

**属性：**
- `checkpoint_dir: Path` - checkpoint 文件目录
- `evolution_interval: int` - 进化间隔（默认 10）
- `logger` - 日志器

**常量：**
- `DEFAULT_EVOLUTION_INTERVAL: int = 10` - 默认进化间隔

**构造参数：**
- `checkpoint_dir: str` - checkpoint 文件目录（默认 "checkpoints"）
- `evolution_interval: int | None` - 进化间隔，默认使用 `DEFAULT_EVOLUTION_INTERVAL`（10）

**核心方法：**

#### `list_generations() -> list[int]`
列出所有可用代数。

扫描两种格式的 checkpoint 文件，返回排序后的代数列表。

返回：排序后的代数列表

#### `find_checkpoint_for_generation(generation: int) -> Path | None`
查找对应代数的 checkpoint。

优先使用 multi_arena 格式。ep_{N}.pkl 的代数 = N / evolution_interval。如果精确匹配不到，尝试找到最接近的 ep_*.pkl。

参数：
- `generation: int` - 代数

返回：checkpoint 文件路径，不存在返回 None

#### `load_genomes(generation: int) -> dict[AgentType, list[bytes]] | None`
加载某代的全部基因组。

参数：
- `generation: int` - 代数

返回：
- `dict[AgentType, list[bytes]]` - 每物种的序列化基因组列表
- `None` - 文件不存在或加载失败

#### `load_checkpoint_data(path: str | Path) -> dict[str, Any] | None`
直接加载 checkpoint 数据。

参数：
- `path: str | Path` - checkpoint 文件路径

返回：checkpoint 数据字典，加载失败返回 None

#### `get_generation_from_checkpoint_path(path: str | Path) -> int | None`
从 checkpoint 路径解析代数。

参数：
- `path: str | Path` - checkpoint 文件路径

返回：代数，无法解析返回 None

**内部方法：**

#### `_load_checkpoint(path: Path) -> dict[str, Any] | None`
加载 checkpoint（自动检测 gzip 压缩格式）。

通过检测 magic bytes（0x1f 0x8b）自动识别 gzip 压缩文件。

#### `_extract_genomes_from_pop_data(pop_data: dict) -> list[bytes]`
从种群数据提取全部基因组。

支持两种格式：
1. SubPopulationManager: is_sub_population_manager=True，有 sub_populations 字段
2. 普通 Population: 有 neat_pop 字段

#### `_extract_genomes_from_neat_pop(neat_pop) -> list[bytes]`
从 neat.Population 对象提取基因组。

使用 `pickle.dumps(genome)` 序列化每个基因组，从 `neat_pop.population` 字典中遍历所有基因组。

**Checkpoint 数据结构：**

普通训练 checkpoint (ep_{N}.pkl):
```python
{
    "tick": int,
    "episode": int,
    "populations": {
        AgentType.RETAIL_PRO: {...},
        AgentType.MARKET_MAKER: {...},
    }
}
```

多竞技场 checkpoint (multi_arena_gen_{N}.pkl):
```python
{
    "generation": int,
    "populations": {...}
}
```

**种群数据格式：**

Population:
```python
{
    "generation": int,
    "neat_pop": neat.Population,
}
```

SubPopulationManager:
```python
{
    "is_sub_population_manager": True,
    "sub_populations": [
        {"neat_pop": neat.Population, ...},
        ...
    ]
}
```

**使用示例：**

```python
from src.analysis.checkpoint_loader import CheckpointLoader
from src.config.config import AgentType

# 创建加载器
loader = CheckpointLoader(checkpoint_dir="checkpoints")

# 列出所有可用代数
generations = loader.list_generations()
print(f"可用代数: {generations}")

# 加载某代的全部基因组
genomes = loader.load_genomes(generation=10)
if genomes:
    for agent_type, genome_list in genomes.items():
        print(f"{agent_type.value}: {len(genome_list)} 个基因组")

# 直接加载 checkpoint 数据
data = loader.load_checkpoint_data("checkpoints/ep_100.pkl")
if data:
    print(f"Episode: {data.get('episode')}")

# 从路径解析代数
gen = loader.get_generation_from_checkpoint_path("checkpoints/multi_arena_gen_50.pkl")
print(f"代数: {gen}")  # 输出: 50
```

**依赖关系：**
- `src.config.config.AgentType` - Agent 类型枚举
- `src.core.log_engine.logger` - 日志器
- `gzip` - gzip 压缩文件支持
- `pickle` - 序列化支持
- `re` - 正则表达式匹配
- `pathlib.Path` - 路径处理

---

### DemoAnalyzer (demo_analyzer.py)

演示模式分析器，分析演示结束时各物种存活个体的分布，生成分析图和终端摘要。

**属性：**
- `_output_dir: str` - 分析结果输出目录
- `_chinese_font: str | None` - 中文字体路径

**构造参数：**
- `output_dir: str` - 分析结果输出目录（默认 "analysis_output"）

**常量：**

`AGENT_TYPE_NAMES: dict[str, str]` - 中文名称映射：
```python
{
    "RETAIL_PRO": "高级散户",
    "MARKET_MAKER": "做市商",
}
```

**核心方法：**

#### `analyze(trainer, end_reason, end_agent_type) -> None`
执行分析并输出结果。

参数：
- `trainer: Trainer` - 训练器实例
- `end_reason: str` - 结束原因（"population_depleted" 或 "one_sided_orderbook"）
- `end_agent_type: AgentType | None` - 触发结束的 Agent 类型

执行流程：
1. 收集各物种存活个体的数据
2. 打印终端摘要
3. 生成分析图

#### `_collect_data(trainer) -> dict[str, Any]`
收集各物种存活个体的数据。

收集内容（每个存活 Agent）：
- `equity: float` - 净值
- `balance: float` - 余额
- `unrealized_pnl: float` - 浮动盈亏
- `position_qty: int` - 持仓量
- `leverage: float` - 杠杆率（position_value / equity）

返回格式：
```python
{
    "episode": int,
    "tick": int,
    "final_price": float,
    "high_price": float,
    "low_price": float,
    "populations": {
        AgentType: {
            "total_count": int,
            "alive_count": int,
            "agents": [
                {
                    "equity": float,
                    "balance": float,
                    "unrealized_pnl": float,
                    "position_qty": int,
                    "leverage": float
                },
                ...
            ]
        },
        ...
    }
}
```

#### `_print_summary(data, end_reason, end_agent_type) -> None`
终端打印摘要。

输出格式：
```
==================================================
演示模式分析结果
==================================================

基本信息:
  Episode: 1 | Tick: 847
  结束原因: 高级散户种群存活不足 1/4

价格统计:
  最终价格: 105.20 | 最高: 112.50 | 最低: 95.30

种群统计:
  高级散户  存活 85/100 (85.0%)      平均净值 523万  盈利/亏损 60/25
  做市商    存活 145/150 (96.7%)     平均净值 2.1亿  盈利/亏损 130/15

分析图已保存到: analysis_output/demo_analysis_20260107_123456.png
==================================================
```

#### `_generate_plots(data) -> str`
生成分析图。

使用 matplotlib 生成 2x2 子图布局：
- 第一行：资产分布图 - 2 个子图（每种群一个），箱线图显示净值分布
- 第二行：持仓分布图 - 2 个子图，条形图显示多头/空仓/空头数量分布

返回保存的图片路径。

**图表特性：**
- 自动查找中文字体（Linux/Windows/macOS）
- 箱线图显示中位数和四分位数
- 红色虚线标注均值
- 持仓分布使用颜色区分：红色（多头）、灰色（空仓）、青色（空头）
- 图片分辨率 150 DPI
- 使用 `bbox_inches="tight"` 优化布局

#### `_format_money(amount) -> str`
格式化金额显示。

格式规则：
- >= 1亿：显示为 "X.XX亿"
- >= 1万：显示为 "X.XX万"
- < 1万：显示为 "X.XX"

#### `_find_chinese_font() -> str | None`
查找可用的中文字体路径。

支持的字体（按优先级）：
- Linux: NotoSansCJK, uming, wqy-microhei, DroidSansFallback
- Windows: msyh, simhei
- macOS: PingFang, STHeiti Light

**使用示例**

```python
from src.training.trainer import Trainer
from src.config.config import Config, AgentType
from src.analysis.demo_analyzer import DemoAnalyzer

# 创建训练器并运行演示
config = Config(...)
trainer = Trainer(config)
trainer.setup()

# ... 运行演示 ...

# 演示结束后自动分析
analyzer = DemoAnalyzer(output_dir="analysis_output")
analyzer.analyze(
    trainer=trainer,
    end_reason="population_depleted",
    end_agent_type=AgentType.RETAIL_PRO
)
```

**输出文件**

分析图保存路径格式：`{output_dir}/demo_analysis_{YYYYMMDD_HHMMSS}.png`

例如：`analysis_output/demo_analysis_20260107_123456.png`

**依赖关系：**

- `src.training.trainer.Trainer` - 训练器（用于获取数据）
- `src.config.config.AgentType` - Agent 类型枚举
- `matplotlib` - 图表生成
- `matplotlib.font_manager` - 字体管理
- `numpy` - 数值计算
- `datetime` - 时间戳生成
- `os` - 文件系统操作

---

### EvolutionTester (evolution_tester.py)

进化效果测试器，通过对比测试评估 NEAT 进化算法的有效性。使用 `ParallelArenaTrainer` 多竞技场测试，通过 OpenMP 实现并行推理。

**属性：**
- `config: Config` - 全局配置
- `checkpoint_dir: str` - checkpoint 文件目录
- `results_dir: str` - 测试结果保存目录
- `logger` - 日志器

**构造参数：**
- `config: Config` - 全局配置对象
- `checkpoint_dir: str` - checkpoint 文件目录（默认 "checkpoints"）
- `results_dir: str` - 测试结果保存目录（默认 "checkpoints/test_results"）

**目录结构：**

初始化时自动创建以下目录：
- `{results_dir}/` - 测试结果根目录
- `{results_dir}/baseline/` - 基准测试结果
- `{results_dir}/comparison/` - 比较测试结果

**核心方法：**

#### `_load_generation_data(generation: int) -> dict[AgentType, list[bytes]] | None`
加载某一代的全部基因组数据。

使用 CheckpointLoader 加载指定代数的基因组数据。

#### `_run_test_with_multi_arena(populations_data, num_arenas, episodes_per_run, episode_length, test_type, run_idx) -> dict`
使用多竞技场运行单次测试。

创建 `ParallelArenaTrainer`，从 genome 数据初始化，创建 `ArenaWorkerPool` 运行 episodes 并收集适应度。

参数：
- `populations_data: dict[AgentType, list[bytes]]` - 各物种的序列化基因组列表
- `num_arenas: int` - 竞技场数量
- `episodes_per_run: int` - episode 数量
- `episode_length: int` - 每个 episode 的 tick 数量
- `test_type: str` - 测试类型（"baseline" 或 "comparison"）
- `run_idx: int` - 运行索引

返回：单次运行结果字典，包含：
```python
{
    "test_type": str,
    "run_idx": int,
    "episodes_per_run": int,
    "total_ticks": int,
    "avg_ticks_per_episode": float,
    "species_results": {
        AgentType: {
            "total_count": int,
            "avg_fitness": float,
            "std_fitness": float,
            "avg_survival_rate": float,
            "std_survival_rate": float,
        },
        ...
    }
}
```

**测试流程（每次运行）：**
1. 创建 `ParallelArenaTrainer` 并调用 `setup_for_testing()` 初始化
2. 创建临时 `ArenaWorkerPool` 并同步网络参数
3. 循环运行 episodes_per_run 个 episode：
   - 调用 `ArenaWorkerPool.run_episodes()` 运行一个 episode
   - 从 `EpisodeResult.per_arena_fitness` 收集各竞技场各物种的适应度
4. 汇总多个 episode 的平均适应度和存活率
5. 关闭 ArenaWorkerPool 并调用 `trainer.stop()` 清理资源

**存活率判断：**
- 使用适应度阈值判断：fitness > -0.99 视为存活
- 全部爆仓时 equity 接近 0，fitness = (0 - initial) / initial 接近 -1

#### `run_baseline_test(generation, num_runs, episode_length, episodes_per_run, num_arenas, force) -> dict`
基准测试：使用第 N 代全部物种的基因组竞技。

参数：
- `generation: int` - 代数
- `num_runs: int` - 运行次数（默认 3）
- `episode_length: int` - 每个 episode 的 tick 数量（默认 1000）
- `episodes_per_run: int` - 每次运行的 episode 数量（默认 10）
- `num_arenas: int` - 竞技场数量（默认 2）
- `force: bool` - 是否强制重新运行（默认 False）

返回格式：
```python
{
    "test_type": "baseline",
    "generation": int,
    "num_runs": int,
    "episodes_per_run": int,
    "species_summary": {
        AgentType: {
            "avg_fitness": float,
            "std_fitness": float,
            "avg_survival_rate": float,
            "std_survival_rate": float,
            "runs": int
        },
        ...
    }
}
```

**适应度计算方式：**
- 所有物种统一使用实际收益率 = (equity - initial) / initial

#### `run_comparison_test(target_generation, base_generation, target_species, ...) -> dict`
比较测试：第 N 代某物种 + 第 N-1 代其他物种。

参数：
- `target_generation: int` - 目标代数（新进化的代）
- `base_generation: int` - 基准代数（旧的代）
- `target_species: AgentType` - 目标物种（使用新代）
- `num_runs: int` - 运行次数（默认 3）
- `episode_length: int` - 每个 episode 的 tick 数量（默认 1000）
- `episodes_per_run: int` - 每次运行的 episode 数量（默认 10）
- `num_arenas: int` - 竞技场数量（默认 2）
- `force: bool` - 是否强制重新运行（默认 False）

返回格式同基准测试，额外包含 `base_generation` 和 `target_species` 字段。

#### `evaluate_evolution_effectiveness(generation, num_runs, episode_length, episodes_per_run, num_arenas, force) -> dict`
评估进化有效性（串行运行各场景测试，PAT 通过 OpenMP 实现并行推理）。

参数：
- `generation: int` - 要评估的代数（N）
- `num_runs: int` - 每个测试的运行次数（默认 3）
- `episode_length: int` - 每个 episode 的 tick 数量（默认 1000）
- `episodes_per_run: int` - 每次运行的 episode 数量（默认 10）
- `num_arenas: int` - 竞技场数量（默认 2）
- `force: bool` - 是否强制重新运行（默认 False）

返回格式：
```python
{
    "generation": int,
    "base_generation": int,
    "baseline": {...},  # 基准测试结果
    "comparisons": {AgentType: {...}, ...},  # 各物种比较测试结果
    "effectiveness": {
        AgentType: {
            "baseline_fitness": float,      # 基准适应度
            "comparison_fitness": float,    # 比较测试适应度
            "absolute_improvement": float,  # 绝对改善
            "relative_improvement_pct": float,  # 相对改善百分比
            "is_effective": bool            # 进化是否有效
        },
        ...
    },
    "summary": {
        "effective_species": [str, ...],
        "ineffective_species": [str, ...]
    }
}
```

**测试架构：**

使用 `ParallelArenaTrainer` 多竞技场测试：
- 评估一代需要：1 个基准测试 + 2 个比较测试 = 3 个场景
- 每个场景串行运行 num_runs 次，每次通过 PAT 使用多竞技场并行推理
- 通过 `setup_for_testing(populations_data)` 初始化测试模式（不创建进化 Worker 池）

**结果缓存：**

- 基准测试：`{results_dir}/baseline/gen_{N}.pkl`
- 比较测试：`{results_dir}/comparison/gen_{N}_vs_gen_{M}_{species}.pkl`

如果结果已存在且 force=False，直接加载返回。

#### `_summarize_results(results, test_type, generation, base_generation, target_species) -> dict`
汇总测试结果。

从多次运行的结果中提取各物种的平均适应度、标准差、存活率等统计信息。

#### `_compile_effectiveness_report(generation, base_generation) -> dict`
编译进化有效性报告。

从缓存的测试结果中读取基准测试和比较测试结果，计算各物种的进化有效性。

**进化有效性判断：**
- `is_effective = comparison_fitness > baseline_fitness`
- 即：比较测试中目标物种的适应度高于基准测试中的适应度

**使用方式：**

通过命令行脚本 `scripts/tools/test_evolution.py` 调用：

```bash
# 列出所有代及其测试状态
python scripts/tools/test_evolution.py --list

# 自动测试所有未完成的代（默认行为）
python scripts/tools/test_evolution.py

# 测试指定代
python scripts/tools/test_evolution.py --generation 100

# 强制重新测试所有代
python scripts/tools/test_evolution.py --force

# 指定测试参数
python scripts/tools/test_evolution.py --num-runs 5 --episode-length 2000 --episodes-per-run 10

# 指定竞技场数量
python scripts/tools/test_evolution.py --num-arenas 4
```

**命令行参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--generation` | None | 只测试指定代（不指定则测试所有未完成的代） |
| `--num-runs` | 3 | 测试运行次数 |
| `--episode-length` | 1000 | 每次测试的 tick 数 |
| `--episodes-per-run` | 10 | 每次测试运行的 episode 数量 |
| `--num-arenas` | 2 | 多竞技场数量 |
| `--list` | - | 列出所有代及其测试状态 |
| `--force` | - | 强制重新运行（忽略已完成的测试结果） |
| `--checkpoint-dir` | checkpoints | checkpoint 文件目录（联盟训练: checkpoints/league_training/checkpoints） |
| `--results-dir` | checkpoints/test_results | 测试结果保存目录 |

**输出示例：**

```
============================================================
进化效果测试 - 第 10 代
============================================================

基准测试（3 次运行）:
  高级散户  平均适应度: +0.2345 (±0.0678)  存活率: 92.1%
  做市商    平均适应度: +0.4567 (±0.0890)  存活率: 98.3%

比较测试（新物种 vs 第 9 代对手）:
  高级散户  适应度: +0.2567 (基准: +0.2345)  变化: +9.5%   ↑ 有效
  做市商    适应度: +0.4789 (基准: +0.4567)  变化: +4.9%   ↑ 有效

总结: 2/2 个物种进化有效
============================================================
```

**关键依赖：**

- `src.training.arena.ParallelArenaTrainer` - 多竞技场并行推理训练器
- `src.training.arena.MultiArenaConfig` - 多竞技场配置
- `src.training.arena.arena_worker.ArenaWorkerPool` - Worker 池
- `src.training.arena.arena_worker.EpisodeResult` - Episode 结果
- `src.training.population.SubPopulationManager` - 子种群管理器
- `src.analysis.checkpoint_loader.CheckpointLoader` - Checkpoint 加载器
- `src.config.config.AgentType, Config` - 配置和枚举
- `src.core.log_engine.logger` - 日志器
- `numpy` - 数值计算
- `pickle` - 结果缓存
- `pathlib.Path` - 路径处理
- `os` - 文件系统操作

---

## 模块使用场景

### 场景 1：演示模式分析

在 `scripts/demo_ui.py` 中使用，演示结束后自动生成分析报告：

```python
from src.analysis.demo_analyzer import DemoAnalyzer

# 演示结束时调用
analyzer = DemoAnalyzer()
analyzer.analyze(
    trainer=trainer,
    end_reason=end_reason,
    end_agent_type=depleted_species
)
```

### 场景 2：进化效果评估

训练完成后使用脚本评估进化有效性：

```bash
# 列出所有代及其测试状态
python scripts/tools/test_evolution.py --list

# 输出示例：
# ================================================================
# 代列表及测试状态
# ================================================================
#   代数 |       保存时间       | 物种数 |       测试状态
# ----------------------------------------------------------------
#      1 | 2026-01-07 10:23:45 |      2 | ✓ 已完成
#      2 | 2026-01-07 10:45:32 |      2 | ◐ 部分完成 (1/2)
#     10 | 2026-01-07 15:30:15 |      2 | ○ 未测试
# ================================================================
# 共 3 代: 1 已完成, 2 待测试

# 自动测试所有未完成的代
python scripts/tools/test_evolution.py

# 测试指定代
python scripts/tools/test_evolution.py --generation 10 --num-runs 5

# 根据输出判断是否需要继续训练
# 如果 2/2 个物种进化有效 → 继续训练
# 如果部分物种进化无效 → 考虑调整参数
```

### 场景 3：Checkpoint 数据加载

从历史 checkpoint 加载基因组数据用于分析：

```python
from src.analysis.checkpoint_loader import CheckpointLoader

loader = CheckpointLoader("checkpoints")

# 列出所有可用代数
generations = loader.list_generations()
print(f"共有 {len(generations)} 代数据")

# 加载某代基因组
genomes = loader.load_genomes(generation=100)
for agent_type, genome_list in genomes.items():
    print(f"{agent_type.value}: {len(genome_list)} 个基因组")
```

---

## 设计要点

### Checkpoint 自动检测

- 支持两种 checkpoint 格式：`multi_arena_gen_{N}.pkl` 和 `ep_{M}.pkl`
- 自动检测 gzip 压缩（通过 magic bytes 0x1f 0x8b）
- 统一接口加载不同格式的 checkpoint
- 支持子种群管理器（SubPopulationManager）格式的种群数据

### 多竞技场并行测试

- 使用 `ParallelArenaTrainer` + OpenMP 实现并行推理
- 各场景串行运行，PAT 内部通过多竞技场实现样本并行
- `EpisodeResult.per_arena_fitness` 包含每个竞技场每个物种的适应度数组，汇总后取平均
- 使用 `ArenaWorkerPool` 管理多进程 Worker

### 适应度计算

所有物种统一使用实际收益率：`fitness = (equity - initial) / initial`

### 结果缓存机制

- 测试结果自动保存到 `{results_dir}/baseline/` 和 `{results_dir}/comparison/`
- 避免重复运行相同测试
- 使用 `force=True` 强制重新运行

---

## 技术约束

- 严格的 Python 类型定义
- 所有模块使用 UTF-8 编码
- 支持跨平台中文字体查找（Linux/Windows/macOS）
- 图表分辨率 150 DPI，使用 `bbox_inches="tight"` 优化布局
- 使用 TYPE_CHECKING 避免循环导入

---

## 相关文档

- `CLAUDE.md` - 项目根文档（训练流程、Agent 类型）
- `src/training/CLAUDE.md` - 训练引擎文档
- `src/training/arena/CLAUDE.md` - 多竞技场并行训练文档
- `src/bio/CLAUDE.md` - 生物系统文档
- `src/config/CLAUDE.md` - 配置管理文档
