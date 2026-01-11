# Analysis 模块

## 模块概述

分析模块负责对演示模式和进化效果的数据进行分析和可视化。

## 文件结构

- `__init__.py` - 模块导出
- `checkpoint_loader.py` - Checkpoint 加载器
- `demo_analyzer.py` - 演示模式分析器
- `evolution_tester.py` - 进化效果测试器

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

## 使用示例

```python
from src.training.trainer import Trainer
from src.config.config import AgentType
from src.analysis.demo_analyzer import DemoAnalyzer

# 创建训练器并运行演示
trainer = Trainer(config)
trainer.setup()

# ... 运行演示 ...

# 分析结束后调用
analyzer = DemoAnalyzer(output_dir="analysis_output")
analyzer.analyze(
    trainer=trainer,
    end_reason="population_depleted",
    end_agent_type=AgentType.RETAIL
)
```

## 输出文件

分析图保存路径格式：`{output_dir}/demo_analysis_{YYYYMMDD_HHMMSS}.png`

例如：`analysis_output/demo_analysis_20260107_123456.png`

## 依赖关系

- `src.training.trainer` - 训练器（用于获取数据）
- `src.config.config` - 配置类（AgentType）
- `matplotlib` - 图表生成
- `numpy` - 数值计算

## 常量定义

### AGENT_TYPE_NAMES
Agent 类型中文名称映射：
- `RETAIL` → "散户"
- `RETAIL_PRO` → "高级散户"
- `WHALE` → "庄家"
- `MARKET_MAKER` → "做市商"

---

### EvolutionTester

进化效果测试器，通过对比测试评估进化是否有效。使用多进程并行运行所有测试。

**属性：**
- `config: Config` - 全局配置
- `checkpoint_dir: str` - checkpoint 文件目录
- `results_dir: str` - 测试结果保存目录

**构造参数：**
- `config: Config` - 全局配置对象
- `checkpoint_dir: str` - checkpoint 文件目录（默认 "checkpoints"）
- `results_dir: str` - 测试结果保存目录（默认 "checkpoints/test_results"）

**核心方法：**

#### `create_agents_from_genome(agent_type, genome_data, count) -> list[Agent]`
从单个 genome 复制创建完整种群（每个 Agent 有独立账户）。

参数：
- `agent_type: AgentType` - Agent 类型
- `genome_data: bytes` - 序列化的 genome 数据
- `count: int` - 要创建的 Agent 数量

返回：Agent 列表

#### `run_baseline_test(generation, num_runs, episode_length, episodes_per_run, force) -> dict`
基准测试：使用第 N 代 4 个物种的全部基因组竞技。

参数：
- `generation: int` - 代数
- `num_runs: int` - 运行次数（默认 3）
- `episode_length: int` - 每个 episode 的 tick 数量（默认 1000）
- `episodes_per_run: int` - 每次运行的 episode 数量（默认 10）
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
            "avg_fitness": float,      # 平均适应度（不同物种计算方式不同）
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
- 散户/高级散户：适应度 = 收益率 = equity / initial_balance
- 做市商：适应度 = 0.5 × 收益率 + 0.5 × maker_volume 排名归一化
- 庄家：适应度 = 0.5 × 收益率 + 0.5 × volatility_contribution 排名归一化

#### `run_comparison_test(target_generation, base_generation, target_species, ...) -> dict`
比较测试：第 N 代某物种 + 第 N-1 代其他物种。

参数：
- `target_generation: int` - 目标代数（新进化的代）
- `base_generation: int` - 基准代数（旧的代）
- `target_species: AgentType` - 目标物种（使用新代）
- `num_runs: int` - 运行次数（默认 3）
- `episode_length: int` - 每个 episode 的 tick 数量（默认 1000）
- `episodes_per_run: int` - 每次运行的 episode 数量（默认 10）
- `force: bool` - 是否强制重新运行（默认 False）

返回格式同基准测试，额外包含 `base_generation` 和 `target_species` 字段。

#### `evaluate_evolution_effectiveness(generation, num_runs, episode_length, episodes_per_run, force) -> dict`
评估进化有效性（并行运行所有测试）。

参数：
- `generation: int` - 要评估的代数（N）
- `num_runs: int` - 每个测试的运行次数（默认 3）
- `episode_length: int` - 每个 episode 的 tick 数量（默认 1000）
- `episodes_per_run: int` - 每次运行的 episode 数量（默认 10）
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

**并行测试架构：**

使用 `concurrent.futures.ProcessPoolExecutor` 实现真正并行：
- 评估一代需要：1 个基准测试 + 4 个比较测试 = 5 个场景
- 每个场景运行 num_runs 次 = 共 5 * num_runs 个任务
- 进程池大小 = CPU 核心数

Worker 函数 `_run_single_test_worker(params)` 在独立进程中执行：
1. 创建 Trainer
2. 从 genome_data 创建各物种的 Agent 种群
3. 初始化鲶鱼（如果 config.catfish.enabled=True）
4. 运行多个 episode（由 episodes_per_run 参数控制，默认 10）
   - 每个 episode 前重置 Agent 和市场状态
   - 鲶鱼爆仓不结束 episode，物种淘汰到 1/4 时结束
5. 每个 episode 结束时调用 `population.evaluate()` 计算各物种的适应度
6. 汇总多个 episode 的平均适应度和存活率并返回结果

**鲶鱼机制：**
- 测试默认启用鲶鱼，可通过 `--no-catfish` 参数禁用
- 鲶鱼的作用是打破"不交易"僵局，增加市场波动
- **测试模式下鲶鱼爆仓不结束 episode**（与训练模式不同）
- episode 结束条件：tick 达到上限 或 任一物种淘汰到 1/4
- 没有鲶鱼时，进化后的 Agent 可能学会"不交易"策略，导致收益率为 0

**评估指标：**

```python
return_rate = (final_equity - initial_balance) / initial_balance
```

收集数据：
- `avg_return_rate`: 平均收益率
- `survival_rate`: 存活率
- `position_distribution`: 持仓分布（long/short/flat）

**结果缓存：**

- 基准测试：`{results_dir}/baseline/gen_{N}.pkl`
- 比较测试：`{results_dir}/comparison/gen_{N}_vs_gen_{M}_{species}.pkl`

如果结果已存在且 force=False，直接加载返回。

**使用示例：**

```python
from src.config.config import Config
from src.analysis.evolution_tester import EvolutionTester

# 加载配置
config = Config(...)

# 创建测试器
tester = EvolutionTester(
    config,
    generations_dir="checkpoints/generations",
    results_dir="checkpoints/test_results"
)

# 评估第 10 代的进化有效性
# 每次运行执行 10 个 episode，取平均表现
report = tester.evaluate_evolution_effectiveness(
    generation=10,
    num_runs=3,
    episode_length=1000,
    episodes_per_run=10  # 每次运行 10 个 episode
)

# 查看结果
print(f"有效物种: {report['summary']['effective_species']}")
print(f"无效物种: {report['summary']['ineffective_species']}")

for agent_type, eff in report['effectiveness'].items():
    print(f"{agent_type.value}: 改善 {eff['relative_improvement_pct']:.1f}%")
```

**关键依赖：**

- `src.training.trainer.Trainer` - 训练器
- `src.training.population.Population` - 种群管理
- `src.training.arena.migration.MigrationSystem` - genome 序列化/反序列化
- `src.bio.brain.brain.Brain` - 神经网络
- `src.bio.agents.*` - 各类型 Agent
- `src.market.matching.matching_engine.MatchingEngine` - 撮合引擎
- `src.market.adl.adl_manager.ADLManager` - ADL 管理器
