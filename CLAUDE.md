# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 NEAT (NeuroEvolution of Augmenting Topologies) 算法的 AI 交易模拟竞技场。四类型的 AI Agent（散户、高级散户、庄家、做市商）通过神经网络进化算法学习交易策略，在模拟订单簿市场中竞争。

## 常用命令

### 构建 Cython 模块
```bash
python setup.py build_ext --inplace
```

### 运行测试
```bash
# 运行所有测试
pytest tests/

# 运行单个测试文件
pytest tests/training/test_population.py

# 运行特定测试
pytest tests/training/test_population.py::TestPopulation::test_create_agents
```

### 训练
```bash
# 无 UI 高性能训练
python scripts/train_noui.py --episodes 100

# 从检查点恢复
python scripts/train_noui.py --resume checkpoints/ep_50.pkl --episodes 100

# 启用鲶鱼机制训练
python scripts/train_noui.py --episodes 100 --catfish

# 指定鲶鱼模式
python scripts/train_noui.py --episodes 100 --catfish --catfish-mode trend_creator
```

### 代码修改后必须执行（清理缓存 + 重新编译）

修改代码后必须执行以下命令，否则可能因缓存问题导致运行时卡住或行为异常：

**一键执行（推荐）：**
```bash
./rebuild.sh
```

## 核心架构

### 模块职责

**src/core/** - 核心引擎
- `log_engine/` - 统一日志管理（命令行只显示启动信息和错误，其余输出到日志文件）

**src/market/** - 交易市场引擎
- `orderbook/` - 订单簿（Cython 实现，买卖各100档）
- `matching/` - 撮合引擎（价格优先、时间优先）
- `account/` - 账户管理（持仓、余额、保证金、强平）
- `catfish/` - 鲶鱼机制（规则驱动的市场波动制造者）

**src/bio/** - 生物系统
- `brain/` - NEAT 神经网络封装
- `agents/` - 四种 Agent 类型，继承自 `Agent` 基类

**src/training/** - 训练引擎
- `Population` - 种群管理，从 NEAT 基因组创建 Agent
- `Trainer` - 训练协调器，管理 tick/episode 生命周期

**src/config/** - 配置管理

### Agent 行为规则

| 类型 | 动作 | 约束 |
|------|------|------|
| 散户 | 挂单/撤单/吃单/不动 | 同时只挂一单，100倍杠杆 |
| 高级散户 | 挂单/撤单/吃单/不动 | 同时只挂一单，100倍杠杆（可观察完整100档订单簿） |
| 庄家 | 挂单/撤单/吃单/不动 | 同时只挂一单，10倍杠杆 |
| 做市商 | 双边挂单/清仓 | 每 tick 必然双边挂单（每边1-5单），先撤旧单再挂新单，10倍杠杆 |

### ADL (自动减仓) 机制

当强平订单无法在订单簿中完全成交时，系统自动触发 ADL 机制：

1. **触发条件**：强平市价单无法完全成交（流动性不足）
2. **排名公式**：
   - 盈利时：`排名 = PnL% × 有效杠杆`
   - 亏损时：`排名 = PnL% / 有效杠杆`
3. **执行流程**：按排名从高到低选择对手方，以当前市场价格强制成交
4. **成交价格**：直接使用当前市场价格（不使用破产价格）
   - 设计原则：强平 ≠ 破产，被强平时 Agent 可能还有正净值（仅保证金率过低）
   - 优点：简单公平、避免异常、符合直觉

相关模块：`src/market/adl/`

### 训练流程
1. **初始化**: 创建四种群、撮合引擎、做市商建立初始流动性
2. **Episode 循环**: 重置账户/市场 → 运行 N 个 tick → NEAT 进化
3. **Tick 执行**: 检查强平 → 鲶鱼行动 → 随机打乱所有 Agent 顺序 → 并行决策 → 串行执行

### NEAT 配置
- `config/neat_retail.cfg` - 散户（127 个输入节点，9 个输出节点）
- `config/neat_retail_pro.cfg` - 高级散户（907 个输入节点，9 个输出节点）
- `config/neat_whale.cfg` - 庄家（907 个输入节点，9 个输出节点）
- `config/neat_market_maker.cfg` - 做市商（934 个输入节点，21 个输出节点）

## 目录级 CLAUDE.md 系统

各子目录包含独立的 CLAUDE.md 文件，详细描述该目录下的代码逻辑和接口。阅读代码时优先查阅对应目录的 CLAUDE.md。

## 技术约束

- 优先使用 NumPy 向量化操作
- 无法向量化的场景使用 Cython 加速（如订单簿）
- 严格的 Python 类型定义
- 所有参数可通过配置文件配置
