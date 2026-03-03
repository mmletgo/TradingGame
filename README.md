# TradingGame - AI 交易模拟竞技场

基于 NEAT (NeuroEvolution of Augmenting Topologies) 算法的 AI 交易模拟竞技场。两种类型的 AI Agent（高级散户、做市商）通过神经网络进化算法学习交易策略，在模拟订单簿市场中竞争。噪声交易者提供随机性流动性和布朗运动价格特征。

## 项目简介

TradingGame 是一个高性能的 AI 交易模拟系统，模拟真实金融市场的交易机制。AI 代理通过强化学习进化交易策略，在动态变化的市场环境中竞争生存。

### 核心特性

- **NEAT 神经进化算法** - 自动进化神经网络拓扑结构和权重
- **完整的交易机制** - 订单簿、撮合引擎、保证金、强平、ADL 自动减仓
- **两种 AI 代理** - 高级散户、做市商，各有独特的策略和约束
- **噪声交易者** - 100个随机交易者，提供市场流动性和价格随机游走
- **Cython 加速** - 关键路径使用 Cython 优化，支持高频交易模拟
- **多竞技场并行训练** - 批量推理合并，OpenMP 并行执行
- **可视化界面** - 实时监控订单簿、价格走势、资产变化
- **灵活配置** - 所有参数可通过配置文件调整

## 项目结构

```
TradingGame_refactor/
├── CLAUDE.md                 # 项目总览（开发者指南）
├── README.md                 # 本文档（用户指南）
├── setup.py                  # Cython 模块构建配置
├── rebuild.sh                # 一键清理缓存和重新编译
│
├── config/                   # NEAT 配置文件目录
│   ├── neat_retail_pro.cfg   # 高级散户 NEAT 配置（907输入，8输出）
│   └── neat_market_maker.cfg # 做市商 NEAT 配置（964输入，41输出）
│
├── scripts/                  # 启动脚本
│   ├── train_noui.py         # 无 UI 高性能训练
│   ├── train_ui.py           # 带 UI 训练
│   ├── train_parallel_arena.py  # 多竞技场并行训练
│   ├── demo_ui.py            # 演示模式
│   └── tools/                # 工具脚本集合
│       ├── plot_evolution_curve.py    # 绘制进化曲线
│       ├── memory_profiler.py         # 内存分析
│       └── ...
│
├── src/                      # 源代码
│   ├── core/                 # 核心引擎（日志系统）
│   │
│   ├── market/               # 交易市场引擎
│   │   ├── orderbook/        # 订单簿（Cython，买卖各100档）
│   │   ├── matching/         # 撮合引擎（价格优先、时间优先）
│   │   ├── account/          # 账户管理（持仓、余额、保证金、强平）
│   │   ├── adl/              # ADL 自动减仓机制
│   │   └── noise_trader/     # 噪声交易者（市场随机性提供者）
│   │
│   ├── bio/                  # 生物系统
│   │   ├── brain/            # NEAT 神经网络封装
│   │   └── agents/           # 两种 Agent 类型实现
│   │
│   ├── training/             # 训练引擎
│   │   ├── population.py     # 种群管理
│   │   ├── trainer.py        # 训练协调器
│   │   ├── arena/            # 多竞技场并行训练
│   │   └── league/           # 联盟训练
│   │
│   ├── ui/                   # 可视化界面
│   │   ├── components/       # UI 组件
│   │   ├── data_collector.py # 数据采集器
│   │   ├── training_app.py   # 训练模式 UI
│   │   └── demo_app.py       # 演示模式 UI
│   │
│   ├── analysis/             # 分析模块
│   │   ├── demo_analyzer.py  # 演示分析器
│   │   └── evolution_tester.py  # 进化效果测试器
│   │
│   └── config/               # 配置类定义
│
├── tests/                    # 测试代码
├── logs/                     # 日志文件目录（自动创建）
├── checkpoints/              # 检查点目录（自动创建）
└── docs/                     # 项目文档
```

## AI 代理类型

| 类型 | 数量 | 初始资金 | 杠杆 | 输入维度 | 输出维度 | 特点 |
|------|------|----------|------|----------|----------|------|
| **高级散户** | 2,400 | 2万 | 1.0x | 907 | 8 | 完整信息（100档+100笔成交），12子种群×200 |
| **做市商** | 400 | 1000万 | 10.0x | 964 | 41 | 提供流动性，双边挂单，4子种群×100 |
| **噪声交易者** | 100 | 1e18 | - | - | - | 随机买卖，50%行动概率，不参与进化 |

### 神经网络输出说明

**高级散户（8个输出）：**
- 0: HOLD（不动）
- 1: PLACE_BID（挂买单）
- 2: PLACE_ASK（挂卖单）
- 3: CANCEL（撤单）
- 4: MARKET_BUY（市价买入）
- 5: MARKET_SELL（市价卖出）
- 6-7: 保留

**做市商（41个输出）：**
- 0: 清仓所有持仓
- 1-10: 买单档位选择（10档）
- 11-20: 卖单档位选择（10档）
- 21-40: 仓位倾斜（0=全卖，40=全买，20=中性）

## 快速开始

### 环境要求

- Python 3.10+
- NumPy
- Cython
- neat-python (NEAT 算法库)
- dearpygui (可选，用于 UI 界面)

### 安装步骤

1. **克隆项目**
```bash
cd /home/rongheng/python_project/TradingGame_refactor
```

2. **安装依赖**
```bash
pip install numpy cython neat-python dearpygui pytest matplotlib seaborn
```

3. **编译 Cython 模块**
```bash
./rebuild.sh
```

或手动编译：
```bash
python setup.py build_ext --inplace
```

## 使用指南

### 训练模式

训练模式用于训练 NEAT 模型，Agent 会通过进化不断优化策略。

#### 1. 无 UI 高性能训练（推荐）

适用于大规模训练，无界面开销，速度最快。

```bash
# 基础训练：100 个 episode
python scripts/train_noui.py --episodes 100

# 从检查点恢复训练
python scripts/train_noui.py --resume checkpoints/ep_50.pkl --episodes 100

# 自定义参数
python scripts/train_noui.py --episodes 500 --episode-length 1000 --checkpoint-interval 50
```

**训练过程：**
1. 创建两个 NEAT 种群（高级散户、做市商）和 100 个噪声交易者
2. 每个 episode 运行指定数量的 tick
3. 每个 tick 执行：强平检查 → 噪声交易者行动 → Agent 观察 → Agent 决策 → 订单撮合
4. 每个 episode 结束后根据 PnL 计算适应度并执行 NEAT 进化
5. 定期保存检查点

**训练输出：**
- 终端实时显示训练进度
- 日志保存在 `logs/` 目录
- 检查点保存在 `checkpoints/` 目录

#### 2. 带 UI 训练

适用于小规模训练和实时观察训练过程。

```bash
# 启动可视化训练界面
python scripts/train_ui.py
```

**UI 功能：**
- 实时显示订单簿深度图
- 价格走势曲线
- 各种群资产变化
- 成交记录
- 开始/暂停/停止控制

#### 3. 多竞技场并行训练（高级）

通过多个独立的竞技场并行训练，提高适应度评估的稳定性。

```bash
# 默认：2个竞技场，每个50个episode，无限轮
python scripts/train_parallel_arena.py

# 指定轮数
python scripts/train_parallel_arena.py --rounds 100

# 自定义参数
python scripts/train_parallel_arena.py --num-arenas 8 --episodes-per-arena 5 --rounds 200

# 从检查点恢复
python scripts/train_parallel_arena.py --resume checkpoints/parallel_arena_gen_50.pkl
```

**多竞技场特点：**
- 多个竞技场的神经网络推理合并成一个批量，OpenMP 并行执行
- 交易配对和账户更新串行执行（保证正确性）
- 所有竞技场完成后汇总适应度（简单平均）
- 每轮进行一次进化
- 2竞技场×50episode = 每轮100个样本，提高适应度评估稳定性

### 演示模式

演示模式用于展示训练好的模型效果，不进行进化。

```bash
# 从检查点加载模型并演示
python scripts/demo_ui.py --checkpoint checkpoints/ep_100.pkl

# 调整演示速度（通过UI滑块，范围0.1x-100x）
```

**演示模式特点：**
- 不进行 NEAT 进化
- 无限循环运行 episode
- 支持速度控制（0.1x - 100x）
- 结束条件：任意物种淘汰到只剩 1/4
- 结束后自动生成分析图和终端摘要（如启用分析器）

**训练模式 vs 演示模式：**

| 特性 | 训练模式 | 演示模式 |
|------|---------|---------|
| 进化 | 是 | 否 |
| Episode数量 | 有限 | 无限（直到满足结束条件） |
| 速度控制 | 不支持 | 支持（0.1x - 100x） |
| 结束条件 | 订单簿单边、物种淘汰 | 仅物种淘汰 |
| 检查点加载 | 不支持（从头训练） | 支持（加载已有模型） |
| 用途 | 训练新的NEAT模型 | 展示已有模型效果 |

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行单个测试文件
pytest tests/training/test_population.py

# 查看测试覆盖率
pytest tests/ --cov=src --cov-report=html
```

## 核心机制

### 训练生命周期

```
初始化阶段
├─ 创建2个NEAT种群（高级散户、做市商）
├─ 创建100个噪声交易者
├─ 从基因组创建Agent对象
├─ 创建撮合引擎
└─ 做市商建立初始流动性（Tick 1）

Episode循环（每episode N个tick）
├─ 重置所有Agent账户
├─ 重置噪声交易者状态
├─ 重置市场状态和订单簿
└─ Tick循环
    ├─ Tick 1: 仅做市商行动
    └─ Tick 2+:
        ├─ 强平检查（三阶段）
        ├─ 噪声交易者行动
        ├─ 计算归一化市场状态（所有Agent共用）
        ├─ 随机打乱Agent顺序
        ├─ 并行决策（神经网络推理）
        └─ 串行执行（订单提交）

NEAT进化阶段
├─ 计算适应度（基于PnL）
├─ 执行NEAT进化算法
├─ 清理历史数据防止内存泄漏
└─ 从新基因组创建Agent
```

### 强平机制（三阶段）

触发条件：`保证金率 < 维持保证金率`

| 阶段 | 操作 | 目的 |
|------|------|------|
| 阶段1 | 统一撤销被淘汰Agent的所有挂单 | 防止撤单时作为maker产生反向仓位 |
| 阶段2 | 统一提交市价单平仓 | 尝试在订单簿中平仓 |
| 阶段3 | ADL自动减仓 | 处理无法完全成交的情况 |

**ADL排名公式：**
- 盈利方：`排名 = PnL% × 有效杠杆`
- 亏损方：`排名 = PnL% / 有效杠杆`
- 成交价格：使用当前市场价格（非破产价格）

### 噪声交易者

为市场提供随机性流动性和布朗运动价格特征（不参与NEAT进化）：

- 200个独立噪声交易者，各自每 tick 以 50% 概率行动
- 行动时 50% 买 / 50% 卖，通过市价单撮合
- 下单量：`max(1, int(lognormvariate(mu=14.5, sigma=1.0)))`
- 无限资金（1e18），不触发强平
- 手续费为 0，可作为 ADL 对手方

### 手续费模型

| 角色 | Maker费 | Taker费 |
|------|---------|---------|
| 高级散户 | 万2 (0.02%) | 万5 (0.05%) |
| 做市商 | -万1（返佣） | 万1 (0.01%) |
| 噪声交易者 | 0 | 0 |

## 配置说明

### 命令行参数

**train_noui.py / train_ui.py:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 100 | 训练的 episode 数量 |
| `--episode-length` | 1000 | 每个 episode 的 tick 数量 |
| `--checkpoint-interval` | 10 | 检查点保存间隔（0=不保存） |
| `--resume` | None | 从检查点恢复训练 |

**train_parallel_arena.py:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-arenas` | 2 | 竞技场数量 |
| `--episodes-per-arena` | 50 | 每个竞技场的 episode 数量 |
| `--rounds` | 无限 | 训练轮数 |
| `--resume` | None | 从检查点恢复训练 |

**demo_ui.py:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必需 | 检查点文件路径 |

### 代码配置

在 `src/config/config.py` 中配置：

```python
# Agent 配置
retail_pro_count = 2_400              # 高级散户数量（12子种群×200）
retail_pro_initial_balance = 20_000   # 高级散户初始资金

market_maker_count = 400              # 做市商数量（4子种群×100）
market_maker_initial_balance = 10_000_000  # 做市商初始资金

# 噪声交易者配置
noise_trader_count = 100              # 噪声交易者数量
noise_trader_action_probability = 0.5 # 行动概率

# 市场配置
initial_price = 100.0                 # 初始价格
tick_size = 0.01                      # 最小价格变动单位
```

## 工具脚本

项目提供了一系列工具脚本用于分析和调优：

```bash
# 绘制进化曲线
python scripts/tools/plot_evolution_curve.py checkpoints/ep_100.pkl

# 内存分析
python scripts/tools/memory_profiler.py

# 推理性能分析
python scripts/tools/profile_inference_breakdown.py

# Tick 级别性能分析
python scripts/tools/profile_tick_detailed.py

# 竞技场数量基准测试
python scripts/tools/benchmark_arena_count.py

# OpenMP 线程数调优
python scripts/tools/benchmark_openmp_threads.py

# 兼容性阈值调优
python scripts/tools/tune_compatibility_threshold.py
```

## 技术栈

- **Python 3.10+** - 主要开发语言
- **NumPy** - 高性能数值计算
- **Cython** - 关键路径优化
- **neat-python** - NEAT 神经进化算法
- **DearPyGui** - 可视化界面
- **OpenMP** - 多核并行推理
- **pytest** - 单元测试框架

## 性能优化

项目采用了多项性能优化措施：

1. **Cython 加速** - 订单簿、持仓、神经网络推理、撮合引擎
2. **OpenMP 并行** - 批量决策推理，多竞技场并行训练
3. **向量化计算** - 使用 NumPy 批量操作替代 Python 循环
4. **内存预分配** - 预分配 NumPy 数组避免重复内存分配
5. **高效数据结构** - 使用 deque 实现固定长度的历史数据缓冲区
6. **共享内存 IPC** - 多竞技场零拷贝进程间通信

## 文档导航

详细的模块文档请参考各子目录的 `CLAUDE.md`：

- [项目总览](CLAUDE.md) - 项目概述和架构说明
- [市场模块](src/market/CLAUDE.md) - 订单簿、撮合引擎、账户管理
- [Agent 模块](src/bio/agents/CLAUDE.md) - 两种 AI 代理实现
- [训练模块](src/training/CLAUDE.md) - 种群管理和训练流程
- [多竞技场](src/training/arena/CLAUDE.md) - 并行训练详解
- [噪声交易者模块](src/market/noise_trader/CLAUDE.md) - 噪声交易者机制详解
- [UI 模块](src/ui/CLAUDE.md) - 可视化界面说明

## 常见问题

### Q: 修改代码后运行报错怎么办？

A: 必须执行清理缓存和重新编译：

```bash
./rebuild.sh
```

### Q: 如何调整训练速度？

A:
- 无 UI 模式已经是最快速度
- 带 UI 训练模式以最大速度运行，不支持速度控制
- 演示模式支持速度控制（0.1x - 100x），通过 UI 滑块调整

### Q: 检查点文件在哪里？

A: 默认保存在 `checkpoints/` 目录：
- 单竞技场：`ep_{episode}.pkl`
- 多竞技场：`parallel_arena_gen_{generation}.pkl`

### Q: 如何查看训练日志？

A: 日志文件保存在 `logs/` 目录，按日期命名。

### Q: 种群灭绝了怎么办？

A: NEAT 配置已设置 `reset_on_extinction = True`，会自动重置种群。

### Q: 内存占用过高怎么办？

A:
1. 减少 Agent 数量
2. 减少 episode 长度
3. 降低 OpenMP 线程数
4. 使用内存分析工具排查：`python scripts/tools/memory_profiler.py`

### Q: 如何评估训练效果？

A:
1. 使用演示模式观察行为：`python scripts/demo_ui.py --checkpoint checkpoints/ep_100.pkl`
2. 使用进化测试器：`python -m src.analysis.evolution_tester checkpoints/ep_100.pkl`
3. 绘制进化曲线：`python scripts/tools/plot_evolution_curve.py checkpoints/ep_100.pkl`

## 贡献指南

欢迎提交 Issue 和 Pull Request！

开发前请确保：

1. 阅读相关模块的 `CLAUDE.md` 了解代码逻辑
2. 运行 `./rebuild.sh` 清理缓存并重新编译
3. 通过所有测试：`pytest tests/`
4. 更新相关文档（代码逻辑 → CLAUDE.md，功能说明 → README.md）

## 许可证

MIT License

---

**祝训练愉快！让 AI 在市场中进化出独特的交易策略。**
