# PRD: NEAT AI 交易模拟竞技场

Created: 2025-12-29
Last Updated: 2026-03-06

## 项目概述

基于 NEAT (NeuroEvolution of Augmenting Topologies) 算法的 AI 交易模拟竞技场。两种类型的 AI Agent（高级散户、做市商）通过神经网络进化算法学习交易策略，在模拟订单簿市场中竞争。噪声交易者提供随机性流动性和布朗运动价格特征。

## 背景与目标

- **背景**: 研究不同类型市场参与者在进化压力下的交易策略演化，观察市场生态的动态平衡
- **目标**:
  - 构建真实的订单簿撮合市场模拟环境
  - 实现两种群独立进化的 NEAT 训练系统
  - 支持多竞技场并行训练，加速进化过程
  - 支持联盟训练（AlphaStar 式历史对手池 + PFSP 采样）
  - 提供可视化的 UI 演示界面

---

## 核心业务逻辑

### 1. 市场模拟机制

#### 1.1 订单簿系统
- **深度**: 买卖各100档
- **撮合规则**: 价格优先、时间优先（FIFO）
- **订单类型**: 限价单、市价单
- **价格约束**: 所有操作在最新价 ±100 个最小变动单位内
- **实现**: Cython 加速的 OrderBook（SortedDict 有序字典）

#### 1.2 杠杆与保证金
| 角色 | 杠杆倍数 | 维持保证金率 | 说明 |
|------|----------|--------------|------|
| 高级散户 | 10.0x | 5%（0.05） | 高杠杆交易 |
| 做市商 | 10.0x | 5%（0.05） | 高杠杆，提供深厚流动性 |

**保证金公式：**
- 占用保证金：`margin_used = |quantity| × current_price / leverage`
- 净值：`equity = balance + unrealized_pnl`
- 保证金率：`margin_ratio = equity / (|quantity| × current_price)`
- 强平触发条件：`margin_ratio < maintenance_margin_rate`

#### 1.3 手续费模型
| 角色 | 挂单费(Maker) | 吃单费(Taker) | 说明 |
|------|---------------|---------------|------|
| 高级散户 | 万2 (0.0002) | 万5 (0.0005) | 普通费率 |
| 做市商 | -万1 (-0.0001) | 万1 (0.0001) | Maker返佣 |
| 噪声交易者 | 0 | 0 | 免手续费 |

### 2. 三种市场参与者

#### 2.1 高级散户 (RetailPro)
- **数量**: 2,400（12子种群×200）
- **初始资金**: 2万
- **杠杆**: 10.0x
- **维持保证金率**: 0.05（5%）
- **信息获取**: 买卖各100档订单簿 + 最近100笔成交 + 最近100个tick历史
- **可用动作**: 挂单买入、挂单卖出、撤单、市价买入、市价卖出、不动（6种）
- **约束**: 同时只能挂一单

**神经网络规格：**
- NEAT 配置文件：`neat_retail_pro.cfg`
- 输入维度：907
- 输出维度：8
- 初始隐藏节点：10
- 种群大小（每子种群）：200

**神经网络输入（907维）：**

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘100档（价格归一化 + 数量归一化） |
| 200-399 | 200 | 卖盘100档（价格归一化 + 数量归一化） |
| 400-499 | 100 | 成交价格归一化 |
| 500-599 | 100 | 成交数量（带方向） |
| 600-603 | 4 | 持仓信息 |
| 604-606 | 3 | 挂单信息（价格归一化、数量、方向） |
| 607-706 | 100 | tick历史价格 |
| 707-806 | 100 | tick历史成交量 |
| 807-906 | 100 | tick历史成交额 |

**神经网络输出（8个）：**
- `[0-5]`: 动作类型得分（argmax选择：HOLD/PLACE_BID/PLACE_ASK/CANCEL/MARKET_BUY/MARKET_SELL）
- `[6]`: 价格偏移（-1到1，映射到 ±100 ticks）
- `[7]`: 数量比例（-1到1，映射到 [0, 1.0]）

#### 2.2 做市商 (MarketMaker)
- **数量**: 400（4子种群×100）
- **初始资金**: 1,000万
- **杠杆**: 10.0x
- **维持保证金率**: 0.05（5%）
- **行为模式**: 每tick必须双边挂单（买卖各1-10单），先撤旧单再挂新单
- **特殊**: 报价中心使用 AS 模型的 reservation price，而非 mid_price

**神经网络规格：**
- NEAT 配置文件：`neat_market_maker.cfg`
- 输入维度：972（964基础 + 8个AS特征）
- 输出维度：43（40基础 + 1总比例 + 2个AS调整）
- 初始隐藏节点：10
- 种群大小（每子种群）：150

**神经网络输入（972维）：**

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘100档（价格归一化 + 数量归一化） |
| 200-399 | 200 | 卖盘100档（价格归一化 + 数量归一化） |
| 400-499 | 100 | 成交价格归一化 |
| 500-599 | 100 | 成交数量（带方向） |
| 600-603 | 4 | 持仓信息 |
| 604-663 | 60 | 挂单信息（10买单×3 + 10卖单×3） |
| 664-763 | 100 | tick历史价格 |
| 764-863 | 100 | tick历史成交量 |
| 864-963 | 100 | tick历史成交额 |
| 964-971 | 8 | AS模型特征（见下表） |

**AS模型特征（索引964-971）：**

| 索引 | 名称 | 说明 |
|------|------|------|
| 964 | reservation_offset | 保留价格偏移 `(r - mid) / mid` |
| 965 | optimal_half_spread | 最优半点差比率 |
| 966 | sigma | 已实现波动率 |
| 967 | tau | 剩余时间比例 [0, 1] |
| 968 | kappa | 订单到达率（对数归一化） |
| 969 | inventory_risk | 库存风险 `gamma × sigma² × tau × inventory_ratio` |
| 970 | gamma_ratio | 风险厌恶系数比率 |
| 971 | spread_vol_ratio | 点差/波动率比率 |

**神经网络输出（43个）：**
- `[0-9]`: 买单1-10价格偏移（相对 reservation_price，映射到 [1, 100] ticks）
- `[10-19]`: 买单1-10数量权重
- `[20-29]`: 卖单1-10价格偏移（相对 reservation_price）
- `[30-39]`: 卖单1-10数量权重
- `[40]`: 总下单比例基准（映射到 [0.01, 1.0]）
- `[41]`: gamma_adjustment（映射到 [0.1, 10.0]，缩放AS的gamma参数）
- `[42]`: spread_adjustment（映射到 [0.5, 2.0]，缩放AS最优点差）

#### 2.3 Avellaneda-Stoikov (AS) 做市模型

做市商集成 AS 模型进行混合报价：

**核心公式：**
```
reservation_price = mid_price × (1 - q_norm × γ × σ² × τ)
optimal_spread = γσ²τ + (2/γ) × ln(1 + γ/κ)
```

其中：
- `q_norm` = `inventory / max_inventory`，归一化库存 [-1, 1]
- `γ`（gamma）= 风险厌恶系数（基础值0.1，NN可动态调整 [0.1, 10.0] 倍）
- `σ`（sigma）= 价格波动率（对数收益率标准差，窗口50 ticks）
- `τ`（tau）= 剩余时间（归一化到 [0, 1]）
- `κ`（kappa）= 订单到达强度代理（基础值1.5）

**AS配置参数（ASConfig）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| gamma | 0.1 | 基础风险厌恶系数 |
| kappa_base | 1.5 | 基础订单到达率 |
| vol_window | 50 | 波动率回看窗口 |
| gamma_adj_min | 0.1 | NN gamma调整乘数下限 |
| gamma_adj_max | 10.0 | NN gamma调整乘数上限 |
| spread_adj_min | 0.5 | NN spread调整乘数下限 |
| spread_adj_max | 2.0 | NN spread调整乘数上限 |
| max_reservation_offset | 0.05 | reservation price最大偏移（±5%） |

#### 2.4 噪声交易者 (NoiseTrader)
- **数量**: 200
- **初始资金**: 1e18（视为无限资金）
- **行为模式**: 每tick以50%概率行动，行动时按 `tick_buy_prob` 概率买入，市价单
- **下单量**: `max(1, int(lognormvariate(mu=12.0, sigma=1.0)))`
- **特殊规则**: 不触发强平，零手续费，可作为ADL对手方

**Ornstein-Uhlenbeck 随机过程：**

噪声交易者的买入概率 `buy_prob` 作为 OU 随机过程演化，防止价格漂移同时避免产生可预测的确定性模式：

- 每个 Episode 开始时生成随机偏置：`episode_buy_prob = 0.5 + uniform(-0.15, 0.15)`
- 每 tick 更新：`buy_prob[t+1] = buy_prob[t] + θ × (μ - buy_prob[t]) + σ × ε`
  - μ = episode_buy_prob，ε ~ N(0,1)
  - θ = 0.035（每tick回归3.5%的偏差）
  - σ = 0.04（噪声强度）
- buy_prob 被 clamp 到 [0.1, 0.9] 范围
- 相比确定性均值回归，OU 过程允许临时趋势存在，且回归路径不可预测

**布朗运动原理：**
200个独立噪声交易者的净买卖量由中心极限定理保证趋向正态分布，价格呈现随机游走特征。

### 3. 强平与ADL机制

#### 3.1 强平流程（三阶段）
强平条件：`保证金率 < 维持保证金率`

**阶段1 - 统一撤单**：
1. 遍历所有Agent检查强平条件
2. 收集需要淘汰的Agent列表
3. 统一撤销这些Agent的所有挂单（防止撤单时作为maker产生反向仓位）

**阶段2 - 统一平仓**：
1. 遍历需要淘汰的Agent
2. 提交市价单平仓
3. 收集无法完全成交的Agent（需要ADL）

**阶段3 - ADL自动减仓**：
1. 用最新价格计算ADL候选清单（包含盈利的噪声交易者）
2. 按排名从高到低选择对手方强制成交
3. 循环直至仓位清零
4. 兜底处理：强制清零被淘汰者仓位

#### 3.2 ADL排名公式
- 盈利方（PnL% > 0）：`排名 = PnL% × 有效杠杆`
- 亏损方（PnL% ≤ 0）：`排名 = PnL% / 有效杠杆`
- 成交价格：使用当前市场价格（非破产价格）

#### 3.3 淘汰规则
- 被强平的Agent在当前episode剩余时间禁用
- 不参与决策和执行
- Episode结束后通过NEAT进化繁殖新个体

### 4. 适应度计算

#### 4.1 纯已实现 PnL + 对称持仓成本

所有 Agent 的适应度使用多空对称公式，防止进化产生方向性偏好导致价格单边漂移：

```
fitness = (balance - initial) / initial - λ × |position_qty × current_price| / initial
```

- **纯 balance**：仅基于已实现 PnL（已完成的交易），多空完全对称
- **持仓成本**：对多头和空头施加相同惩罚，激励关闭持仓
- 散户 `position_cost_weight`（λ）默认 0.02
- 做市商 `mm_position_cost_weight`（λ）默认 0.005（做市商需持仓做市，惩罚更小）

#### 4.2 做市商复合适应度

做市商使用双组件加权复合适应度：

```
mm_fitness = α × pnl + γ × volume_score
```

| 组件 | 权重 | 计算方式 | 范围 | 激励方向 |
|------|------|---------|------|---------|
| pnl | α=0.7 | `(balance - initial) / initial - λ × \|qty × price\| / initial` | [-1, 0] | 盈利能力（纯已实现 + 持仓成本） |
| volume_score | γ=0.3 | `maker_volume / (max_maker_volume_in_pop + 1)` | [0, 1) | 做市成交量 |

### 5. 训练模式

#### 5.1 单竞技场训练（Trainer）

```
初始化
├─ 创建2个种群（从NEAT基因组创建Agent）
├─ 创建200个噪声交易者
├─ 创建撮合引擎
├─ 做市商建立初始流动性（Tick 1）
└─ 初始化EMA平滑价格

Episode循环（N个tick）
├─ 重置所有Agent账户
├─ 重置噪声交易者状态
├─ 重置市场状态
└─ Tick循环
    ├─ Tick 1: 仅做市商行动
    └─ Tick 2+:
        ├─ 强平检查（三阶段）
        ├─ 收集所有原子动作（噪声交易者+做市商+散户）
        ├─ 随机打乱原子动作顺序
        ├─ 逐个执行原子动作
        └─ 记录tick历史数据

NEAT进化
├─ 累积适应度（每episode累积，每N个episode平均后进化）
├─ 执行NEAT进化算法
├─ 清理历史数据防止内存泄漏
└─ 从新基因组更新Agent
```

**提前结束条件：**
- 任意种群存活少于初始值的 1/4
- 订单簿只有单边挂单

#### 5.2 多竞技场并行训练（ParallelArenaTrainer）
- 每个竞技场独立 Worker 进程
- 通过 SharedNetworkMemory 零拷贝共享网络参数
- 所有竞技场完成后汇总适应度（简单平均）
- 支持物种迁移

#### 5.3 联盟训练（LeagueTrainer）

基于 AlphaStar 联盟训练思路，混合竞技场中当前代和历史代精英Agent同场交易。

**核心概念：**

| 组件 | 说明 |
|------|------|
| 对手池 | 按 Agent 类型独立存储历史代精英（RETAIL_PRO、MARKET_MAKER 各一个池） |
| PFSP采样 | 优先级虚拟自我博弈，败率越高的对手权重越大 |
| 混合竞技场 | 当前代 + 历史代精英在同一市场中交易 |
| 代际对比 | 监控当前代相对于历史代的适应度提升 |
| 物种冻结 | 收敛的物种冻结进化，但仍参与交易 |

**竞技场 Agent 组成：**

| 类型 | 当前代 | 历史代（6代×Top 5%精英）| 总计 | 历史占比 |
|------|--------|-------------------------|------|---------|
| 高级散户 | 2,400 | 720（6×120） | 3,120 | 23% |
| 做市商 | 400 | 120（6×20） | 520 | 23% |
| 噪声交易者 | 300（增强） | - | 300 | - |

**联盟训练流程（每轮 run_round）：**

```
0. 等待上一轮 checkpoint 线程完成
1. 清理旧采样结果和历史数据
2. PFSP 采样历史对手（带新鲜度约束）
3. 提取精英网络，构建历史 AgentInfo 列表
4. 更新 Workers 的 agent_infos 和合并网络
5. 运行 episodes + NEAT 进化
   ├─ 历史 Agent（sub_pop_id >= 1000）自动不参与进化
   └─ 当前代 Agent（sub_pop_id < 1000）参与 NEAT 进化
6. 计算代际适应度对比
7. 检查冻结/解冻
8. 保存里程碑（含预进化适应度）
9. 更新历史对手胜率
10. 清理对手池
```

**PFSP 采样权重：**
```
p(opponent) ∝ f(win_rate) × recency_factor × exploration_bonus

f(win_rate) = (1 - win_rate)^p     # 败率越高权重越大
recency_factor = exp(-λ × Δgen)    # 近期对手优先
exploration_bonus = max(1, b/√(match_count+1))  # 未交战对手高权重
```

**双重收敛判断：**
1. **种群收敛**：最近10代的种群平均适应度标准差 ≤ 0.005
2. **精英收敛**：最近10代的精英平均适应度标准差 ≤ 0.005

只有种群和精英都收敛时，才判定为真正收敛。

**物种冻结与定期复评：**
- 收敛后冻结 NEAT 进化（基因组不再变异/交叉），仍作为对手参与交易
- 每代复评：种群平均或精英平均适应度下降超过 5% 则解冻
- 解冻后需要连续10代收敛才会重新冻结（隐式冷却期）
- 所有物种均冻结时训练自动完成

**历史 Agent ID 方案：**
- 散户: `10,000,000 + entry_index × 1,000,000 + local_index`
- 做市商: `20,000,000 + entry_index × 1,000,000 + local_index`
- sub_pop_id: `1000 + entry_index`（自动不参与 NEAT 进化）

**联盟训练默认配置：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_arenas | 16 | 竞技场数量 |
| episodes_per_arena | 4 | 每竞技场episode数 |
| num_historical_generations | 6 | 每轮采样历史代数 |
| historical_elite_ratio | 0.05 | 每代取Top 5%精英 |
| historical_freshness_ratio | 0.5 | 采样中最近历史的最低占比 |
| hybrid_noise_trader_count | 300 | 混合竞技场噪声交易者数 |
| sampling_strategy | pfsp | 采样策略 |
| milestone_interval | 1 | 里程碑保存间隔（代数） |
| convergence_fitness_std_threshold | 0.005 | 收敛阈值 |
| convergence_generations | 10 | 连续收敛代数 |
| freeze_thaw_threshold | 0.05 | 解冻阈值（5%） |
| min_freeze_generation | 30 | 最早允许冻结的代数 |
| pfsp_exponent | 2.0 | 败率加权指数 |
| pfsp_explore_bonus | 2.0 | 未交战对手探索奖励系数 |
| pfsp_win_rate_ema_alpha | 0.3 | 胜率EMA平滑因子 |

### 6. 归一化方法

| 数据类型 | 归一化公式 | 范围 |
|---------|-----------|------|
| 订单簿价格 | `(price - mid_price) / mid_price` | [-0.1, 0.1] |
| 订单簿数量 | `log10(quantity + 1) / 10` | [0, 1] |
| 成交数量（带方向） | `sign(qty) × log10(|qty| + 1) / 10` | [-1, 1] |
| tick历史价格 | `(price - base_price) / base_price` | - |
| tick历史成交量 | `sign(vol) × log10(|vol| + 1) / 10` | - |
| tick历史成交额 | `sign(amt) × log10(|amt| + 1) / 12` | - |

---

## 用户场景

### 场景1: 单竞技场训练（无UI）
- **操作**: `python scripts/train_noui.py --episodes 100`
- **预期结果**: 高效后台训练，定期保存检查点

### 场景2: 单竞技场训练（带UI）
- **操作**: `python scripts/train_ui.py`
- **预期结果**: 实时可视化训练过程

### 场景3: 联盟训练
- **操作**: `python scripts/train_league.py --rounds 200`
- **预期结果**: 16竞技场并行，PFSP采样历史对手，双重收敛检测

### 场景4: 演示模式
- **操作**: `python scripts/demo_ui.py --checkpoint checkpoints/ep_100.pkl`
- **预期结果**: 实时展示价格曲线、订单簿、成交记录、资产分布
- **结束条件**: 任意物种淘汰到只剩1/4

### 场景5: 恢复训练
- **操作**: `python scripts/train_noui.py --resume checkpoints/ep_50.pkl --episodes 100`
- **预期结果**: 继续之前的进化进程

### 场景6: 恢复联盟训练
- **操作**: `python scripts/train_league.py --resume checkpoints/league_training/checkpoints/gen_100.pkl`
- **预期结果**: 恢复联盟训练状态（含冻结状态、适应度历史、对手池）

---

## 项目结构

```
TradingGame/
├── src/                          # 源代码
│   ├── core/                     # 核心引擎
│   │   └── log_engine/           # 日志引擎
│   ├── market/                   # 交易市场引擎
│   │   ├── orderbook/            # 订单簿（Cython）
│   │   ├── matching/             # 撮合引擎
│   │   ├── account/              # 账户管理
│   │   ├── adl/                  # ADL自动减仓
│   │   ├── noise_trader/         # 噪声交易者模块
│   │   ├── market_state.py       # 归一化市场状态数据
│   │   └── as_calculator.py      # AS模型计算器
│   ├── bio/                      # 生物系统
│   │   ├── agents/               # 两种Agent（高级散户、做市商）
│   │   └── brain/                # NEAT神经网络
│   ├── training/                 # 训练引擎
│   │   ├── population.py         # 种群管理
│   │   ├── trainer.py            # 训练协调器
│   │   ├── arena/                # 多竞技场模块
│   │   └── league/               # 联盟训练模块
│   ├── ui/                       # DearPyGui UI
│   ├── analysis/                 # 分析模块
│   └── config/                   # 配置管理
├── scripts/                      # 启动脚本
│   ├── train_noui.py             # 单竞技场无UI训练
│   ├── train_ui.py               # 单竞技场UI训练
│   ├── train_league.py           # 联盟训练
│   ├── demo_ui.py                # 演示模式
│   └── create_config.py          # 默认配置创建函数
├── config/                       # NEAT配置文件
│   ├── neat_retail_pro.cfg       # 高级散户（527输入，3输出，pop_size=200）
│   └── neat_market_maker.cfg     # 做市商（972输入，43输出，pop_size=150）
├── checkpoints/                  # 检查点
├── logs/                         # 日志
├── docs/                         # 文档
└── tests/                        # 测试
```

---

## 配置参数

### 市场参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| 初始价格 | 100 | 交易品种初始价格 |
| 最小变动单位 | 0.01 | 价格 tick size |
| 最小交易单位 | 1 | 数量 lot size |
| 盘口深度 | 100 | 买卖各100档 |
| EMA Alpha | 0.9 | 价格平滑系数 |

### Agent配置
| 角色 | 数量 | 初始资金 | 杠杆 | 维持保证金率 | Maker费 | Taker费 |
|------|------|----------|------|--------------|---------|---------|
| 高级散户 | 2,400 | 2万 | 10.0x | 0.05 | 万2 | 万5 |
| 做市商 | 400 | 1,000万 | 10.0x | 0.05 | -万1 | 万1 |
| 噪声交易者 | 200 | 1e18 | - | 不强平 | 0 | 0 |

### 噪声交易者配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| count | 200 | 噪声交易者数量 |
| action_probability | 0.5 | 每tick行动概率 |
| quantity_mu | 12.0 | 对数正态分布mu |
| quantity_sigma | 1.0 | 对数正态分布sigma |
| episode_bias_range | 0.15 | 买入概率偏置范围 |
| ou_theta | 0.035 | OU过程均值回归速度 |
| ou_sigma | 0.04 | OU过程噪声强度 |

### 训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| episode_length | 1000 tick | 每episode的tick数量 |
| checkpoint_interval | 10 | 检查点保存间隔 |
| evolution_interval | 10 | 进化间隔（episode数） |
| retail_pro_sub_population_count | 12 | 高级散户子种群数量 |
| mm_fitness_pnl_weight | 0.7 | 做市商PnL权重 |
| mm_fitness_volume_weight | 0.3 | 做市商成交量权重 |

### NEAT配置
| 配置文件 | Agent类型 | 输入节点 | 输出节点 | 隐藏节点 | 种群大小 |
|---------|----------|---------|---------|---------|---------|
| neat_retail_pro.cfg | 高级散户 | 907 | 8 | 10 | 200 |
| neat_market_maker.cfg | 做市商 | 972 | 43 | 10 | 150 |

---

## 技术栈

- **语言**: Python 3.10+
- **UI**: DearPyGui
- **NEAT**: neat-python
- **加速**: NumPy, Cython, Numba (JIT), OpenMP
- **持久化**: pickle, numpy (npz)

---

## 性能优化

### Cython加速
- 订单簿操作（OrderBook）
- 持仓计算（Position）
- 账户管理（FastAccount）
- 撮合引擎（FastMatchingEngine）
- 神经网络前向传播（FastFeedForwardNetwork，10-100x提升）
- 批量决策（batch_decide_openmp，OpenMP并行）
- 批量订单执行（fast_execution）

### 并行化
- 多竞技场：独立 Worker 进程并行（SharedNetworkMemory 零拷贝共享）
- Agent创建：8个worker线程池
- Agent决策：OpenMP多线程并行推理

### 内存优化
- 预分配输入缓冲区
- NEAT历史数据定期清理
- 显式垃圾回收 + `malloc_trim()`
- `deque(maxlen)` 自动管理历史数据长度
- 对手池内存缓存定期清理
- 每10个episode完整GC

---

## 变更日志

- 2025-12-29: 初始PRD创建
- 2025-12-30: 添加强制平仓机制、做市商行为修改
- 2025-12-30: 移除周期性淘汰繁殖，改为episode结束后NEAT进化
- 2026-01-08: 添加高级散户、ADL、鲶鱼机制、多竞技场并行训练
- 2026-02-16: 三角色重构（移除散户/庄家/鲶鱼，新增噪声交易者）
- 2026-03-06: 全面重写PRD
  - 所有参数以代码为准（高级散户杠杆10.0x、维持保证金率0.05）
  - 添加 AS 模型集成详细说明
  - 添加 OU 随机过程机制
  - 添加联盟训练完整业务逻辑（PFSP、对手池、冻结/解冻）
  - 更新神经网络输入输出规格（做市商972输入/43输出）
  - 噪声交易者数量统一为200
