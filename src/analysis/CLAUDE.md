# Analysis 模块

## 模块概述

分析模块提供两个核心功能：
1. **演示模式分析** - 分析演示结束时各物种存活个体的分布，生成可视化图表和终端摘要
2. **进化效果测试** - 通过对比测试评估 NEAT 进化算法的有效性

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

**属性：**
- `checkpoint_dir: Path` - checkpoint 文件目录
- `evolution_interval: int` - 进化间隔（默认 10）
- `logger` - 日志器

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

优先使用 multi_arena 格式。ep_{N}.pkl 的代数 = N / evolution_interval。

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
1. SubPopulationManager: `is_sub_population_manager=True`
2. 普通 Population: 有 `neat_pop` 字段

#### `_extract_genomes_from_neat_pop(neat_pop) -> list[bytes]`
从 neat.Population 对象提取基因组。

使用 `pickle.dumps(genome)` 序列化每个基因组。

**Checkpoint 数据结构：**

普通训练 checkpoint (ep_{N}.pkl):
```python
{
    "tick": int,
    "episode": int,
    "populations": {
        AgentType.RETAIL: {...},
        AgentType.RETAIL_PRO: {...},
        AgentType.WHALE: {...},
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

SubPopulationManager (RETAIL/MARKET_MAKER):
```python
{
    "is_sub_population_manager": True,
    "sub_population_count": int,
    "sub_populations": [
        {"neat_pop": neat.Population, "generation": int},
        ...
    ]
}
```

普通 Population (RETAIL_PRO/WHALE):
```python
{
    "generation": int,
    "neat_pop": neat.Population,
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

---

### DemoAnalyzer

演示模式分析器，分析演示结束时各物种存活个体的分布，生成分析图和终端摘要。

**属性：**
- `_output_dir: str` - 分析结果输出目录
- `_chinese_font: str | None` - 中文字体路径

**构造参数：**
- `output_dir: str` - 分析结果输出目录（默认 "analysis_output"）

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
  结束原因: 散户种群存活不足 1/4

价格统计:
  最终价格: 105.20 | 最高: 112.50 | 最低: 95.30

种群统计:
  散户      存活 2450/10000 (24.5%)  平均净值 485万  盈利/亏损 850/1600
  高级散户  存活 85/100 (85.0%)      平均净值 523万  盈利/亏损 60/25
  庄家      存活 180/200 (90.0%)     平均净值 1.2亿  盈利/亏损 150/30
  做市商    存活 145/150 (96.7%)     平均净值 2.1亿  盈利/亏损 130/15

分析图已保存到: analysis_output/demo_analysis_20260107_123456.png
==================================================
```

#### `_generate_plots(data) -> str`
生成分析图。

使用 matplotlib 生成 2x4 子图布局：
- 第一行：资产分布图 - 4 个子图（每种群一个），箱线图显示净值分布
- 第二行：持仓分布图 - 4 个子图，条形图显示多头/空仓/空头数量分布

返回保存的图片路径。

**图表特性：**
- 自动查找中文字体
- 箱线图显示中位数和四分位数
- 红色虚线标注均值
- 持仓分布使用颜色区分：红色（多头）、灰色（空仓）、青色（空头）
- 图片分辨率 150 DPI

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

### 使用示例

```python
from src.training.trainer import Trainer
from src.config.config import Config, AgentType
from src.analysis.demo_analyzer import DemoAnalyzer

# 创建训练器并运行演示
config = Config()
trainer = Trainer(config)
trainer.setup()

# ... 运行演示 ...

# 演示结束后自动分析
analyzer = DemoAnalyzer(output_dir="analysis_output")
analyzer.analyze(
    trainer=trainer,
    end_reason="population_depleted",
    end_agent_type=AgentType.RETAIL
)
```

### 输出文件

分析图保存路径格式：`{output_dir}/demo_analysis_{YYYYMMDD_HHMMSS}.png`

例如：`analysis_output/demo_analysis_20260107_123456.png`

### 依赖关系

- `src.training.trainer.Trainer` - 训练器（用于获取数据）
- `src.config.config.AgentType` - Agent 类型枚举
- `matplotlib` - 图表生成
- `numpy` - 数值计算

---

## 核心类

### EvolutionTester (evolution_tester.py)

进化效果测试器，通过对比测试评估 NEAT 进化算法的有效性。使用 `ParallelArenaTrainer` 多竞技场测试，通过 OpenMP 实现并行推理。

---

**属性：**
- `config: Config` - 全局配置
- `checkpoint_dir: str` - checkpoint 文件目录
- `results_dir: str` - 测试结果保存目录
- `logger` - 日志器

**构造参数：**
- `config: Config` - 全局配置对象
- `checkpoint_dir: str` - checkpoint 文件目录（默认 "checkpoints"）
- `results_dir: str` - 测试结果保存目录（默认 "checkpoints/test_results"）

**核心方法：**

#### `_run_test_with_multi_arena(populations_data, num_arenas, episodes_per_run, test_type, run_idx) -> dict`
使用多竞技场运行单次测试。创建 `ParallelArenaTrainer`，从 genome 数据初始化，运行 episodes 并收集适应度。

参数：
- `populations_data: dict[AgentType, list[bytes]]` - 各物种的序列化基因组列表
- `num_arenas: int` - 竞技场数量
- `episodes_per_run: int` - episode 数量
- `test_type: str` - 测试类型（"baseline" 或 "comparison"）
- `run_idx: int` - 运行索引

返回：单次运行结果字典

#### `run_baseline_test(generation, num_runs, episode_length, episodes_per_run, num_arenas, force) -> dict`
基准测试：使用第 N 代 4 个物种的全部基因组竞技。

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
- 评估一代需要：1 个基准测试 + 4 个比较测试 = 5 个场景
- 每个场景串行运行 num_runs 次，每次通过 PAT 使用多竞技场并行推理
- 通过 `setup_for_testing(populations_data)` 初始化测试模式（不创建进化 Worker 池）

测试流程（每次运行）：
1. 创建 `ParallelArenaTrainer` 并调用 `setup_for_testing()` 初始化
2. 循环运行 episodes_per_run 个 episode：
   - 调用 `_reset_all_arenas()` 和 `_init_market_all_arenas()` 重置状态
   - 逐 tick 执行 `run_tick_all_arenas()`（批量推理 + 执行）
   - 调用 `_collect_episode_fitness()` 收集跨竞技场的适应度（累加值除以竞技场数量）
3. 汇总多个 episode 的平均适应度和存活率
4. 调用 `trainer.stop()` 清理资源

**存活率判断：**
- 使用适应度阈值判断：fitness > -0.99 视为存活
- 全部爆仓时 equity 接近 0，fitness = (0 - initial) / initial 接近 -1

**鲶鱼机制：**
- 测试默认启用鲶鱼，可通过 `--no-catfish` 参数禁用
- 鲶鱼的作用是打破"不交易"僵局，增加市场波动
- episode 结束条件由 PAT 的 `run_tick_all_arenas()` 控制

**结果缓存：**

- 基准测试：`{results_dir}/baseline/gen_{N}.pkl`
- 比较测试：`{results_dir}/comparison/gen_{N}_vs_gen_{M}_{species}.pkl`

如果结果已存在且 force=False，直接加载返回。

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

# 禁用鲶鱼机制
python scripts/tools/test_evolution.py --no-catfish
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
| `--catfish` | True | 启用鲶鱼机制 |
| `--no-catfish` | - | 禁用鲶鱼机制 |

**输出示例：**

```
============================================================
进化效果测试 - 第 10 代
============================================================

基准测试（3 次运行）:
  散户      平均适应度: +0.1234 (±0.0456)  存活率: 85.2%
  高级散户  平均适应度: +0.2345 (±0.0678)  存活率: 92.1%
  庄家      平均适应度: +0.3456 (±0.0789)  存活率: 95.5%
  做市商    平均适应度: +0.4567 (±0.0890)  存活率: 98.3%

比较测试（新物种 vs 第 9 代对手）:
  散户      适应度: +0.1456 (基准: +0.1234)  变化: +18.0%  ↑ 有效
  高级散户  适应度: +0.2567 (基准: +0.2345)  变化: +9.5%   ↑ 有效
  庄家      适应度: +0.3678 (基准: +0.3456)  变化: +6.4%   ↑ 有效
  做市商    适应度: +0.4789 (基准: +0.4567)  变化: +4.9%   ↑ 有效

总结: 4/4 个物种进化有效
============================================================
```

**关键依赖：**

- `src.training.arena.ParallelArenaTrainer` - 多竞技场并行推理训练器
- `src.training.arena.MultiArenaConfig` - 多竞技场配置
- `src.training.population.SubPopulationManager` - 子种群管理器
- `src.analysis.checkpoint_loader.CheckpointLoader` - Checkpoint 加载器

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
#      1 | 2026-01-07 10:23:45 |      4 | ✓ 已完成
#      2 | 2026-01-07 10:45:32 |      4 | ◐ 部分完成 (2/4)
#     10 | 2026-01-07 15:30:15 |      4 | ○ 未测试
# ================================================================
# 共 3 代: 1 已完成, 2 待测试

# 自动测试所有未完成的代
python scripts/tools/test_evolution.py

# 测试指定代
python scripts/tools/test_evolution.py --generation 10 --num-runs 5

# 根据输出判断是否需要继续训练
# 如果 4/4 个物种进化有效 → 继续训练
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

### 多竞技场并行测试

- 使用 `ParallelArenaTrainer` + OpenMP 实现并行推理
- 各场景串行运行，PAT 内部通过多竞技场实现样本并行
- `_collect_episode_fitness()` 返回跨竞技场的累加值，需除以竞技场数量得到平均适应度

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

---

## 相关文档

- `CLAUDE.md` - 项目根文档（训练流程、Agent 类型）
- `src/training/CLAUDE.md` - 训练引擎文档
- `src/bio/CLAUDE.md` - 生物系统文档
- `src/config/CLAUDE.md` - 配置管理文档

