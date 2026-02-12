# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 NEAT (NeuroEvolution of Augmenting Topologies) 算法的 AI 交易模拟竞技场。四种类型的 AI Agent（散户、高级散户、庄家、做市商）通过神经网络进化算法学习交易策略，在模拟订单簿市场中竞争。

## 常用命令

### 构建 Cython 模块
```bash
./rebuild.sh  # 推荐：清理缓存 + 重新编译
python setup.py build_ext --inplace  # 仅编译
```

### 运行测试
```bash
pytest tests/                                    # 运行所有测试
pytest tests/training/test_population.py         # 运行单个文件
pytest tests/training/test_population.py::TestPopulation::test_create_agents  # 运行特定测试
```

### 训练
```bash
python scripts/train_noui.py --episodes 100                    # 无UI高性能训练
python scripts/train_noui.py --resume checkpoints/ep_50.pkl    # 从检查点恢复
python scripts/train_noui.py --episodes 100 --catfish          # 启用鲶鱼机制
python scripts/train_ui.py                                      # 带UI训练
```

### 联盟训练
```bash
python scripts/train_league.py --rounds 200
python scripts/train_league.py --resume checkpoints/league_training/checkpoints/gen_100.pkl
```

### 演示模式
```bash
python scripts/demo_ui.py --checkpoint checkpoints/ep_100.pkl           # 加载检查点
python scripts/demo_ui.py --checkpoint checkpoints/ep_100.pkl --catfish # 启用鲶鱼
```

**演示模式特点：**
- 鲶鱼默认禁用，启用时鲶鱼爆仓不结束 episode
- 结束条件：任意物种淘汰到只剩 1/4
- 结束后自动生成分析图和终端摘要

---

## 核心业务逻辑

### 训练生命周期

```
初始化阶段
├─ 创建4个NEAT种群
├─ 从基因组创建Agent对象
├─ 创建撮合引擎
└─ 做市商建立初始流动性（Tick 1）

Episode循环（每episode N个tick）
├─ 重置所有Agent账户
├─ 重置市场状态和订单簿
└─ Tick循环
    ├─ Tick 1: 仅做市商行动
    └─ Tick 2+:
        ├─ 强平检查（三阶段）
        ├─ 鲶鱼行动（如启用）
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

### 四种Agent类型

| 类型 | 数量 | 初始资金 | 杠杆 | 订单簿深度 | 动作 |
|------|------|----------|------|-----------|------|
| 散户 | 10,000 | 2万 | 1.0x | 10档 | 挂单/撤单/吃单/不动 |
| 高级散户 | 100 | 2万 | 1.0x | 100档 | 挂单/撤单/吃单/不动 |
| 庄家 | 100 | 3M | 1.0x | 100档 | 挂单/撤单/吃单/不动 |
| 做市商 | 400 | 10M | 1.0x | 100档 | 双边挂单（每边1-10单） |

**约束规则：**
- 散户/高级散户/庄家：同时只能挂一单
- 做市商：每tick必须双边挂单，先撤旧单再挂新单
- 所有操作在最新价 ±100 个最小变动单位内

### 鲶鱼机制

规则驱动的市场扰动器（不参与NEAT进化）：

| 模式 | 行为 |
|------|------|
| 趋势创造者 | Episode开始随机选方向，持续该方向操作 |
| 均值回归 | 价格偏离EMA时反向操作 |
| 随机交易 | 随机买卖制造噪音 |

**特殊规则：**
- 初始资金 = (做市商杠杆后资金 - 其他物种杠杆后资金) / 3
  - 做市商：400 × 10M × 1.0 = 4000M
  - 其他物种：200M + 2M + 300M = 502M
  - 每条鲶鱼：(4000M - 502M) / 3 ≈ 1166M
- 手续费为0
- 鲶鱼被强平 → Episode立即结束（训练模式）

### 手续费模型

| 角色 | Maker费 | Taker费 |
|------|---------|---------|
| 散户/高级散户 | 万2 | 万5 |
| 庄家/做市商 | -万1（返佣） | 万1 |
| 鲶鱼 | 0 | 0 |

---

## 模块架构

### 目录结构

```
src/
├── core/           # 核心引擎
│   └── log_engine/ # 统一日志管理
├── market/         # 交易市场引擎
│   ├── orderbook/  # 订单簿（Cython，买卖各100档）
│   ├── matching/   # 撮合引擎（价格优先、时间优先）
│   ├── account/    # 账户管理（持仓、余额、保证金、强平）
│   ├── adl/        # ADL自动减仓
│   └── catfish/    # 鲶鱼模块
├── bio/            # 生物系统
│   ├── brain/      # NEAT神经网络封装
│   └── agents/     # 四种Agent类型
├── training/       # 训练引擎
│   ├── population.py   # 种群管理
│   ├── trainer.py      # 训练协调器
│   └── arena/          # 多竞技场并行训练模块
├── ui/             # DearPyGui可视化
├── analysis/       # 演示分析器
└── config/         # 配置管理
```

### NEAT配置文件

| 配置 | Agent类型 | 输入节点 | 输出节点 |
|------|----------|----------|----------|
| neat_retail.cfg | 散户 | 127 | 8 |
| neat_retail_pro.cfg | 高级散户 | 907 | 8 |
| neat_whale.cfg | 庄家 | 907 | 8 |
| neat_market_maker.cfg | 做市商 | 964 | 41 |

---

## 目录级 CLAUDE.md 系统

各子目录包含独立的 CLAUDE.md 文件，详细描述该目录下的代码逻辑和接口。阅读代码时优先查阅对应目录的 CLAUDE.md：

| 模块 | 文件 | 内容 |
|------|------|------|
| 市场引擎 | src/market/CLAUDE.md | 订单簿、撮合、账户、ADL |
| Agent | src/bio/agents/CLAUDE.md | 四种Agent行为与输入输出 |
| 神经网络 | src/bio/brain/CLAUDE.md | NEAT封装与前向传播 |
| 训练引擎 | src/training/CLAUDE.md | Episode循环、强平处理 |
| 多竞技场 | src/training/arena/CLAUDE.md | 并行训练、适应度汇总 |
| 鲶鱼 | src/market/catfish/CLAUDE.md | 三种模式详解 |
| ADL | src/market/adl/CLAUDE.md | 自动减仓机制 |

---

## 技术约束

- 优先使用 NumPy 向量化操作
- 无法向量化的场景使用 Cython 加速（如订单簿、持仓计算、神经网络前向传播）
- 严格的 Python 类型定义
- 所有参数可通过配置文件配置
- 代码修改后必须执行 `./rebuild.sh` 清理缓存并重新编译

---

## 性能优化要点

### Cython加速模块
- `OrderBook` - 订单簿操作
- `Position` - 持仓计算
- `FastFeedForwardNetwork` - 神经网络推理（10-100倍提升）

### 并行化策略
- Agent创建：8个worker线程池
- Agent决策：16个worker线程池

### 内存管理
- 预分配输入缓冲区
- NEAT历史数据定期清理
- Episode后强制垃圾回收
- `_price_history` 使用 `deque(maxlen=1000)` 自动管理长度，避免列表切片创建新对象
- `_tick_history_prices/volumes/amounts` 使用 `deque(maxlen=100)` 自动管理长度
- 推理缓存 `_last_inference_arrays` 在使用完成后立即释放
- 每 10 个 episode 执行完整 GC + `malloc_trim()` 释放内存给操作系统
