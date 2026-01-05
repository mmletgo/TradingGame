# TradingGame - AI 交易模拟竞技场

基于 NEAT (NeuroEvolution of Augmenting Topologies) 算法的 AI 交易模拟竞技场。四种类型的 AI Agent（散户、高级散户、庄家、做市商）通过神经网络进化算法学习交易策略，在模拟订单簿市场中竞争。

## 项目简介

TradingGame 是一个高性能的 AI 交易模拟系统，模拟真实金融市场的交易机制。AI 代理通过强化学习进化交易策略，在动态变化的市场环境中竞争生存。

### 核心特性

- **NEAT 神经进化算法** - 自动进化神经网络拓扑结构和权重
- **完整的交易机制** - 订单簿、撮合引擎、保证金、强平、ADL 自动减仓
- **四种 AI 代理** - 散户、高级散户、庄家、做市商，各有独特的策略和约束
- **Cython 加速** - 关键路径使用 Cython 优化，支持高频交易模拟
- **鲶鱼机制** - 规则驱动的市场扰动器，增加市场动态性
- **可视化界面** - 实时监控订单簿、价格走势、资产变化
- **灵活配置** - 所有参数可通过配置文件调整

## 项目结构

```
TradingGame_origin/
├── CLAUDE.md                 # 项目总览（开发者指南）
├── README.md                 # 本文档（用户指南）
├── setup.py                  # Cython 模块构建配置
├── rebuild.sh                # 一键清理缓存和重新编译
│
├── config/                   # 配置文件目录
│   ├── neat_retail.cfg       # 散户 NEAT 配置（67输入，9输出）
│   ├── neat_retail_pro.cfg   # 高级散户 NEAT 配置（607输入，9输出）
│   ├── neat_whale.cfg        # 庄家 NEAT 配置（607输入，9输出）
│   └── neat_market_maker.cfg # 做市商 NEAT 配置（634输入，22输出）
│
├── scripts/                  # 启动脚本
│   ├── train_noui.py         # 无 UI 高性能训练
│   ├── train_ui.py           # 带 UI 训练
│   ├── demo_ui.py            # 演示模式
│   └── create_config.py      # 配置生成工具
│
├── src/                      # 源代码
│   ├── core/                 # 核心引擎
│   │   └── log_engine/       # 统一日志管理
│   │
│   ├── market/               # 交易市场引擎
│   │   ├── orderbook/        # 订单簿（Cython 实现，买卖各100档）
│   │   ├── matching/         # 撮合引擎（价格优先、时间优先）
│   │   ├── account/          # 账户管理（持仓、余额、保证金、强平）
│   │   ├── adl/              # ADL 自动减仓机制
│   │   └── catfish/          # 鲶鱼机制（市场扰动器）
│   │
│   ├── bio/                  # 生物系统
│   │   ├── brain/            # NEAT 神经网络封装
│   │   └── agents/           # 四种 Agent 类型实现
│   │
│   ├── training/             # 训练引擎
│   │   ├── population.py     # 种群管理
│   │   └── trainer.py        # 训练协调器
│   │
│   ├── ui/                   # 可视化界面
│   │   ├── components/       # UI 组件（订单簿、图表、控制面板）
│   │   ├── data_collector.py # 数据采集器
│   │   ├── training_app.py   # 训练模式 UI
│   │   └── demo_app.py       # 演示模式 UI
│   │
│   └── config/               # 配置类定义
│
├── tests/                    # 测试代码
│   ├── bio/                  # 生物系统测试
│   ├── core/                 # 核心引擎测试
│   ├── market/               # 市场引擎测试
│   └── training/             # 训练引擎测试
│
├── logs/                     # 日志文件目录（自动创建）
├── checkpoints/              # 检查点目录（自动创建）
└── docs/                     # 项目文档
```

## AI 代理类型

| 类型 | 观察能力 | 动作空间 | 杠杆 | 特点 |
|------|---------|---------|------|------|
| **散户** | 买卖各10档 + 10笔成交 | 挂单/撤单/吃单/不动 | 100倍 | 信息有限，高风险高收益 |
| **高级散户** | 完整100档 + 100笔成交 | 挂单/撤单/吃单/不动 | 100倍 | 完整市场信息，高杠杆 |
| **庄家** | 完整100档 + 100笔成交 | 挂单/撤单/吃单/不动 | 10倍 | 大资金，低杠杆，趋势跟随 |
| **做市商** | 完整100档 + 100笔成交 | 双边挂单/清仓 | 10倍 | 提供流动性，每tick必然挂单 |

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
cd /home/rongheng/python_project/TradingGame_origin
```

2. **安装依赖**
```bash
pip install numpy cython neat-python dearpygui pytest
```

3. **编译 Cython 模块**
```bash
python setup.py build_ext --inplace
```

或者使用一键脚本：
```bash
./rebuild.sh
```

### 运行训练

#### 无 UI 高性能训练（推荐）

```bash
# 训练 100 个 episode
python scripts/train_noui.py --episodes 100

# 从检查点恢复训练
python scripts/train_noui.py --resume checkpoints/ep_50.pkl --episodes 100

# 启用鲶鱼机制训练
python scripts/train_noui.py --episodes 100 --catfish

# 自定义参数
python scripts/train_noui.py --episodes 500 --episode-length 1000 --checkpoint-interval 50
```

#### 带 UI 训练

```bash
# 启动可视化训练界面
python scripts/train_ui.py
```

#### 演示模式

```bash
# 从检查点加载模型并演示
python scripts/demo_ui.py --checkpoint checkpoints/ep_100.pkl

# 调整演示速度（1.0=正常，10.0=10倍速）
python scripts/demo_ui.py --checkpoint checkpoints/ep_100.pkl --speed 2.0
```

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

### 订单簿与撮合

- **100档深度** - 买卖各100档订单簿
- **价格优先、时间优先** - 标准撮合规则
- **Cython 加速** - 支持高频交易模拟

### 保证金与强平

- **维持保证金率** - 默认 5%，可配置
- **强平触发** - 保证金率低于维持保证金率时自动强平
- **爆仓即淘汰** - 被强平的 Agent 在本轮 episode 剩余时间内禁用

### ADL 自动减仓

当强平订单无法完全成交时，系统自动触发 ADL 机制：

1. **触发条件** - 强平市价单无法完全成交（流动性不足）
2. **排名公式** - 盈利时 `排名 = PnL% × 有效杠杆`，亏损时 `排名 = PnL% / 有效杠杆`
3. **执行流程** - 按排名从高到低选择对手方，以破产价格强制成交

### 鲶鱼机制

鲶鱼是规则驱动的市场参与者，用于增加市场波动性：

- **趋势追踪** - 顺势推动价格
- **周期摆动** - 交替买卖形成波动
- **逆势操作** - 均值回归策略

### NEAT 进化

- **自动拓扑进化** - 神经网络结构自动优化
- **物种形成** - 自动区分不同策略类型
- **精英保留** - 每个物种保留最优个体

## 配置说明

### 训练参数

在 `scripts/train_noui.py` 中或通过命令行参数配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 100 | 训练的 episode 数量 |
| `--episode-length` | 1000 | 每个 episode 的 tick 数量 |
| `--checkpoint-interval` | 10 | 检查点保存间隔（0=不保存） |
| `--catfish` | False | 是否启用鲶鱼机制 |
| `--catfish-mode` | trend_creator | 鲶鱼行为模式 |

### Agent 配置

在 `src/config/config.py` 中配置：

```python
# 散户配置
retail_count = 200              # 种群数量
retail_initial_balance = 1_000_000  # 初始资金
retail_leverage = 100           # 杠杆倍数

# 庄家配置
whale_count = 2
whale_initial_balance = 10_000_000_000
whale_leverage = 10

# 做市商配置
market_maker_count = 100
market_maker_initial_balance = 50_000_000
market_maker_leverage = 10
```

### 市场配置

```python
initial_price = 100.0           # 初始价格
tick_size = 0.01                # 最小价格变动单位
maker_fee_rate = 0.0002         # 挂单手续费率（0.02%）
taker_fee_rate = 0.0005         # 吃单手续费率（0.05%）
maintenance_margin_rate = 0.05  # 维持保证金率（5%）
```

## 使用示例

### 示例 1：基础训练

```bash
# 训练 100 个 episode，每 episode 1000 个 tick
python scripts/train_noui.py --episodes 100 --episode-length 1000
```

训练过程中会：
1. 创建四个种群（散户、高级散户、庄家、做市商）
2. 每个 episode 运行 1000 个 tick
3. 每个 tick 执行：强平检查 → 鲶鱼行动 → Agent 决策 → 撮合成交
4. 每个 episode 结束后进行 NEAT 进化
5. 每 10 个 episode 保存一次检查点

### 示例 2：启用鲶鱼机制

```bash
# 使用趋势追踪型鲶鱼训练
python scripts/train_noui.py --episodes 100 --catfish --catfish-mode trend_following
```

鲶鱼会：
- 每隔几个 tick 下大额市价单
- 产生至少 3 tick 的价格波动
- 参与强平和 ADL 机制
- 被强平时立即结束 episode

### 示例 3：从检查点恢复

```bash
# 从第 50 个 episode 的检查点继续训练
python scripts/train_noui.py --resume checkpoints/ep_50.pkl --episodes 100
```

### 示例 4：可视化演示

```bash
# 加载训练好的模型并演示
python scripts/demo_ui.py --checkpoint checkpoints/ep_100.pkl --speed 2.0
```

界面会显示：
- 实时订单簿深度图
- 价格走势曲线
- 各种群资产变化
- 成交记录
- 鲶鱼状态（如果启用）

## 技术栈

- **Python 3.10+** - 主要开发语言
- **NumPy** - 高性能数值计算
- **Cython** - 关键路径优化
- **neat-python** - NEAT 神经进化算法
- **DearPyGui** - 可视化界面
- **pytest** - 单元测试框架

## 性能优化

项目采用了多项性能优化措施：

1. **Cython 加速** - 订单簿、持仓等关键模块使用 Cython 实现
2. **向量化计算** - 使用 NumPy 批量操作替代 Python 循环
3. **多核并行** - Agent 决策阶段并行执行（16个worker线程池）
4. **内存预分配** - 预分配 NumPy 数组避免重复内存分配
5. **高效数据结构** - 使用 deque 实现固定长度的历史数据缓冲区

## 文档导航

详细的模块文档请参考各子目录的 `CLAUDE.md`：

- [项目总览](CLAUDE.md) - 项目概述和架构说明
- [市场模块](src/market/CLAUDE.md) - 订单簿、撮合引擎、账户管理
- [Agent 模块](src/bio/agents/CLAUDE.md) - 四种 AI 代理实现
- [训练模块](src/training/CLAUDE.md) - 种群管理和训练流程
- [鲶鱼模块](src/market/catfish/CLAUDE.md) - 鲶鱼机制详解
- [UI 模块](src/ui/CLAUDE.md) - 可视化界面说明

## 常见问题

### Q: 修改代码后运行卡住怎么办？

A: 必须执行清理缓存和重新编译：

```bash
./rebuild.sh
```

### Q: 如何调整训练速度？

A: 无 UI 模式已经是最快速度。如需减速观察，使用带 UI 的演示模式并调整速度参数。

### Q: 检查点文件在哪里？

A: 默认保存在 `checkpoints/` 目录，文件名格式为 `ep_{episode}.pkl`。

### Q: 如何查看训练日志？

A: 日志文件保存在 `logs/` 目录，按日期命名。

### Q: 种群灭绝了怎么办？

A: NEAT 配置已设置 `reset_on_extinction = True`，会自动重置种群。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

开发前请确保：

1. 阅读相关模块的 `CLAUDE.md` 了解代码逻辑
2. 运行 `./rebuild.sh` 清理缓存并重新编译
3. 通过所有测试：`pytest tests/`
4. 更新相关文档（代码逻辑 → CLAUDE.md，功能说明 → README.md）

## 许可证

MIT License

## 联系方式

项目主页：[GitHub 链接]

---

**祝训练愉快！让 AI 在市场中进化出独特的交易策略。**
