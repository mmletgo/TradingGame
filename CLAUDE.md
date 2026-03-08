# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 NEAT (NeuroEvolution of Augmenting Topologies) 算法的 AI 交易模拟竞技场。两种类型的 AI Agent（高级散户、做市商）通过神经网络进化算法学习交易策略，在模拟订单簿市场中竞争。噪声交易者提供随机性流动性和布朗运动价格特征。

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
```

**演示模式特点：**
- 结束条件：任意物种淘汰到只剩 1/4
- 结束后自动生成分析图和终端摘要

---

## 核心业务逻辑

### 训练生命周期

```
初始化阶段
├─ 创建2个NEAT种群（高级散户、做市商）
├─ 创建噪声交易者（200个）
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
├─ 计算适应度（高级散户基于PnL，做市商使用复合适应度）
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

### 两种Agent类型 + 噪声交易者

| 类型 | 数量 | 初始资金 | 杠杆 | 订单簿深度 | 动作 |
|------|------|----------|------|-----------|------|
| 高级散户 | 2,400 (12子种群×200) | 2万 | 10.0x | 100档 | 挂单/撤单/吃单/不动 |
| 做市商 | 400 (4子种群×100) | 10M | 10.0x | 100档 | 双边挂单（每边1-10单），报价中心使用 AS 模型的 reservation price 而非 mid_price |
| 噪声交易者 | 200 | 1e18（无限资金） | - | - | 50%概率行动，市价单随机买卖 |

**约束规则：**
- 高级散户：同时只能挂一单
- 做市商：每tick必须双边挂单，先撤旧单再挂新单
- 所有操作在最新价 ±100 个最小变动单位内
- 噪声交易者：不触发强平，零手续费，下单量服从对数正态分布

### 适应度计算（纯已实现 PnL + 对称持仓成本）

所有 Agent 的适应度使用多空对称公式，防止进化产生方向性偏好导致价格单边漂移：

```
fitness = (balance - initial) / initial - λ × |position_qty × current_price| / initial
```

- **纯 balance**：仅基于已实现 PnL（已完成的交易），多空完全对称
- **持仓成本**：`λ × |position_qty × price| / initial`，对多头和空头施加相同惩罚，激励关闭持仓
- **散户** `position_cost_weight`（λ）默认 0.02
- **做市商** `mm_position_cost_weight`（λ）默认 0.005（做市商需持仓做市，惩罚更小）

### 做市商复合适应度

做市商使用双组件加权复合适应度，激励做市商盈利并实际提供流动性：

```
mm_fitness = α × pnl + γ × volume_score
```

| 组件 | 权重 | 计算方式 | 范围 | 激励方向 |
|------|------|---------|------|---------|
| `pnl` | α=0.7 | `(balance - initial) / initial - λ × \|qty × price\| / initial` | [-1, 0] | 盈利能力（纯已实现 + 持仓成本） |
| `volume_score` | γ=0.3 | `maker_volume / (max_maker_volume_in_pop + 1)` | [0, 1) | 做市成交量 |

权重可通过 `TrainingConfig` 的 `mm_fitness_pnl_weight` 和 `mm_fitness_volume_weight` 参数配置。

### 噪声交易者

市场随机性提供者（不参与NEAT进化）：

- 200个独立噪声交易者，各自每 tick 以 50% 概率行动
- 行动时按 `tick_buy_prob` 概率买入，通过市价单撮合
- 每个 Episode 开始时，每个竞技场独立生成随机偏置：`episode_buy_prob = 0.5 + uniform(-episode_bias_range, episode_bias_range)`
- `episode_bias_range` 默认 0.15，即 episode_buy_prob ∈ [0.35, 0.65]
- **Ornstein-Uhlenbeck 随机过程**：buy_prob 本身作为随机过程演化，防止价格漂移同时避免产生可预测的确定性模式
  - 每 tick 更新：`buy_prob[t+1] = buy_prob[t] + θ × (μ - buy_prob[t]) + σ × ε`，其中 μ = episode_buy_prob，ε ~ N(0,1)
  - `ou_theta` 默认 0.035（每 tick 回归 3.5% 的偏差），`ou_sigma` 默认 0.04（噪声强度）
  - buy_prob 被 clamp 到 [0.1, 0.9] 范围
  - 相比确定性均值回归，OU 过程允许临时趋势存在，且回归路径不可预测
- 下单量：`max(1, int(lognormvariate(mu=12.0, sigma=1.0)))`
- 无限资金（1e18），不触发强平检查
- 手续费为 0
- 可作为 ADL 对手方

### 手续费模型

| 角色 | Maker费 | Taker费 |
|------|---------|---------|
| 高级散户 | 万2 | 万5 |
| 做市商 | -万1（返佣） | 万1 |
| 噪声交易者 | 0 | 0 |

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
│   └── noise_trader/ # 噪声交易者模块
├── bio/            # 生物系统
│   ├── brain/      # NEAT神经网络封装
│   └── agents/     # 两种Agent类型（高级散户、做市商）
├── training/       # 训练引擎
│   ├── population.py   # 种群管理
│   ├── trainer.py      # 训练协调器
│   ├── arena/          # 多竞技场并行训练模块
│   └── league/         # 联盟训练模块
├── ui/             # DearPyGui可视化
├── analysis/       # 演示分析器
└── config/         # 配置管理
```

### NEAT配置文件

| 配置 | Agent类型 | 输入节点 | 输出节点 |
|------|----------|----------|----------|
| neat_retail_pro.cfg | 高级散户 | 907 | 8 |
| neat_market_maker.cfg | 做市商 | 972 | 43 |

---

## 目录级 CLAUDE.md 系统

各子目录包含独立的 CLAUDE.md 文件，详细描述该目录下的代码逻辑和接口。阅读代码时优先查阅对应目录的 CLAUDE.md：

| 模块 | 文件 | 内容 |
|------|------|------|
| 市场引擎 | src/market/CLAUDE.md | 订单簿、撮合、账户、ADL；market_state.py 新增 9 个 AS 模型预计算字段（reservation_price、optimal_spread、volatility、as_bid_price、as_ask_price、as_bid_depth、as_ask_depth、inventory_ratio、time_ratio） |
| Agent | src/bio/agents/CLAUDE.md | 两种Agent行为与输入输出 |
| 神经网络 | src/bio/brain/CLAUDE.md | NEAT封装与前向传播 |
| 训练引擎 | src/training/CLAUDE.md | Episode循环、强平处理 |
| 多竞技场 | src/training/arena/CLAUDE.md | 并行训练、适应度汇总 |
| 联盟训练 | src/training/league/CLAUDE.md | 联盟训练机制 |
| 噪声交易者 | src/market/noise_trader/CLAUDE.md | 噪声交易者机制详解 |
| ADL | src/market/adl/CLAUDE.md | 自动减仓机制 |

---

## 技术约束

- 优先使用 NumPy 向量化操作
- 无法向量化的场景使用 Cython 加速（如订单簿、持仓计算、神经网络前向传播）
- 严格的 Python 类型定义
- 所有参数可通过配置文件配置
- AS 模型参数通过 `ASConfig` 配置，包括库存风险厌恶系数 `gamma`、波动率估计窗口 `volatility_window`、做市时间范围 `T` 等，用于预计算 reservation price 和最优价差，并作为输入特征提供给做市商神经网络
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
