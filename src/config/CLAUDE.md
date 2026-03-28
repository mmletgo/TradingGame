# Config 配置模块

## 模块概述

配置模块是整个交易竞技场系统的配置中心，定义了所有核心组件的配置数据结构。所有配置项都使用 Python dataclass 定义，提供类型安全、清晰的默认值和 IDE 友好的代码补全支持。

## 模块职责

1. **市场配置管理** - 定义交易市场的核心参数（价格、深度、EMA 等）
2. **Agent 配置管理** - 定义两种 Agent 类型的交易参数（资金、杠杆、费率等）
3. **训练配置管理** - 定义 NEAT 进化训练的参数（episode 长度、并行化配置等）
4. **演示配置管理** - 定义 WebUI 演示服务器的参数
5. **噪声交易者配置管理** - 定义噪声交易者的行为参数

## 文件结构

- `config.py` - 核心配置数据类定义
- `CLAUDE.md` - 本文档

## 核心枚举

### AgentType

定义系统中两种 AI Agent 的类型枚举，继承自 `str` 和 `Enum`，支持字符串比较。

**枚举值：**
- `RETAIL_PRO` - 高级散户：数量 2,400（12子种群×200），资金 2万，可看到深层订单簿（5档）
- `MARKET_MAKER` - 做市商：数量 400（4子种群×100），资金 1,000万，必须双边挂单提供流动性

**使用示例：**
```python
from src.config.config import AgentType

# 遍历所有 Agent 类型
for agent_type in AgentType:
    print(f"Agent类型: {agent_type.name}, 值: {agent_type.value}")

# 判断 Agent 类型
if agent_type == AgentType.MARKET_MAKER:
    print("这是做市商")

# 字符串比较（继承 str）
if agent_type.value == "RETAIL_PRO":
    print("这是高级散户")
```

## 核心配置类

### MarketConfig

市场配置类，定义交易市场的核心参数。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| initial_price | float | 必填 | 初始价格 |
| tick_size | float | 必填 | 最小变动单位 |
| lot_size | float | 必填 | 最小交易单位 |
| depth | int | 必填 | 盘口深度，买卖各多少档 |
| ema_alpha | float | 0.5 | EMA 平滑系数（0-1），值越小价格变化越平滑 |

**详细说明：**

- **initial_price**：市场的初始价格，所有 Agent 的初始资产估值基于此价格
- **tick_size**：价格的最小变动单位，所有订单价格必须是 tick_size 的整数倍
- **lot_size**：订单数量的最小单位，所有订单数量必须是 lot_size 的整数倍
- **depth**：订单簿深度，决定买盘和卖盘各保留多少档位，同时影响 Agent 的输入维度
  - 高级散户/做市商：使用 5 档订单簿数据
- **ema_alpha**：EMA（指数移动平均）平滑系数，用于平滑价格波动
  - 公式：`smooth_price = alpha * current_price + (1-alpha) * prev_smooth_price`
  - 训练时通常设为 0.9（较强平滑）
  - 演示时可以设为更小的值（0.1-0.3）以展示更平滑的价格变化

**使用示例：**
```python
from src.config.config import MarketConfig

market = MarketConfig(
    initial_price=100.0,
    tick_size=0.01,
    lot_size=1.0,
    depth=5,
    ema_alpha=0.9,
)
```

### AgentConfig

Agent 配置类，定义特定类型 Agent 的交易参数。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| count | int | 必填 | 该类型 Agent 的数量 |
| initial_balance | float | 必填 | 初始资产 |
| leverage | float | 必填 | 杠杆倍数 |
| maintenance_margin_rate | float | 必填 | 维持保证金率 |
| maker_fee_rate | float | 必填 | 挂单费率（负数表示返佣） |
| taker_fee_rate | float | 必填 | 吃单费率 |

**默认配置示例：**

| Agent 类型 | 数量 | 初始资金 | 杠杆 | 维持保证金率 | 挂单费率 | 吃单费率 |
|-----------|------|---------|------|------------|---------|---------|
| 高级散户 (RETAIL_PRO) | 2,400 | 2万 | 10.0 | 0.05 | 0.0002 (万2) | 0.0005 (万5) |
| 做市商 (MARKET_MAKER) | 400 | 1,000万 (10M) | 10.0 | 0.05 | -0.0001 (负万1) | 0.0001 (万1) |

**详细说明：**

- **count**：该类型 Agent 的种群数量
  - 高级散户 2,400（12子种群×200），代表交易者群体
  - 做市商 400（4子种群×100），提供足够的流动性

- **initial_balance**：初始资金（单位：元）
  - 高级散户：2万
  - 做市商：1,000万，最大资金群体，提供充足流动性

- **leverage**：杠杆倍数
  - 当前配置：高级散户 10.0x，做市商 10.0x

- **maintenance_margin_rate**：维持保证金率
  - 公式：`强平触发条件 = 净值 / 持仓市值 < 维持保证金率`
  - 当前配置：高级散户 0.05，做市商 0.05
  - 做市商的计算：`0.5 / leverage`（当 leverage=10.0 时为 0.05）

- **maker_fee_rate**：挂单手续费率
  - 高级散户：万2（0.0002）
  - 做市商：负万1（-0.0001），表示返佣，鼓励提供流动性
  - 负费率：成交后交易所返还手续费，降低做市成本

- **taker_fee_rate**：吃单手续费率
  - 高级散户：万5（0.0005）
  - 做市商：万1（0.0001），享受机构费率

**使用示例：**
```python
from src.config.config import AgentConfig, AgentType

retail_pro_config = AgentConfig(
    count=2400,
    initial_balance=20000.0,
    leverage=10.0,
    maintenance_margin_rate=0.05,
    maker_fee_rate=0.0002,
    taker_fee_rate=0.0005,
)
```

### TrainingConfig

训练配置类，定义 NEAT 进化训练的参数。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| episode_length | int | 必填 | 每个 episode 的 tick 数量（默认 1000） |
| checkpoint_interval | int | 必填 | 检查点保存间隔（episode 数） |
| neat_config_path | str | 必填 | NEAT 配置文件目录路径 |
| parallel_workers | int | 16 | 并行工作进程数 |
| enable_parallel_evolution | bool | True | 是否启用并行进化 |
| enable_parallel_decision | bool | True | 是否启用并行决策 |
| enable_parallel_creation | bool | True | 是否启用并行创建 |
| openmp_threads | int | 16 | OpenMP 并行线程数（Cython 神经网络推理，建议设为物理核心数） |
| random_seed | int \| None | None | 随机种子（None 表示不固定） |
| retail_pro_sub_population_count | int | 12 | 高级散户子种群数量 |
| evolution_interval | int | 10 | 每多少个 episode 进化一次 |
| num_arenas | int | 2 | 竞技场数量（多竞技场模式） |
| episodes_per_arena | int | 50 | 每个竞技场运行的 episode 数 |
| mm_fitness_pnl_weight | float | 0.7 | 做市商复合适应度中 PnL 收益率权重 alpha |
| mm_fitness_volume_weight | float | 0.3 | 做市商复合适应度中 Maker 成交量权重 gamma |
| position_cost_weight | float | 0.02 | 散户持仓成本权重（对称持仓惩罚） |
| mm_position_cost_weight | float | 0.005 | 做市商持仓成本权重（做市商需持仓做市，权重更小） |
| enable_cpu_affinity | bool | True | 是否将 Arena Worker 进程绑定到独立的物理 CPU 核心 |

**详细说明：**

- **episode_length**：每个 episode 包含的 tick 数量
  - 快速测试：100-500
  - 正常训练：1,000-5,000
  - 长期训练：10,000+

- **checkpoint_interval**：检查点保存间隔（episode 数）
  - 默认 10：每 10 个 episode 保存一次
  - 设为 0：不保存检查点（不推荐）

- **neat_config_path**：NEAT 配置文件目录路径
  - Population 会自动根据 AgentType 选择对应的配置文件
  - 例如：`config` 目录下应包含 `neat_retail_pro.cfg`、`neat_market_maker.cfg`

- **并行化配置**：
  - `parallel_workers`：并行工作进程数，建议 `CPU 核心数 - 2`
  - `enable_parallel_evolution`：启用并行进化，使用多进程同时进化多个种群
  - `enable_parallel_decision`：启用并行决策（**已废弃**，当前使用串行决策，因 GIL 限制串行更快）
  - `enable_parallel_creation`：启用并行创建 Agent

- **openmp_threads**：Cython 神经网络推理的 OpenMP 线程数
  - 默认 16，建议设为物理核心数
  - 过多线程反而更慢（线程调度开销、内存带宽瓶颈）

- **random_seed**：随机种子
  - 设为整数：固定随机种子，结果可复现
  - 设为 None：不固定随机种子（默认）

- **retail_pro_sub_population_count**：高级散户子种群数量
  - 将 2,400 个高级散户拆分为 12 个子种群，每个 200 个
  - 减少单个 NEAT 种群的规模，优化进化性能
  - 每个子种群独立进化，共享市场环境

- **evolution_interval**：进化间隔
  - 默认 10：每 10 个 episode 进化一次
  - 进化时使用累积的平均适应度

- **多竞技场配置**（多竞技场并行训练模式）：
  - `num_arenas`：竞技场数量（默认 2）
  - `episodes_per_arena`：每个竞技场运行的 episode 数（默认 50）
  - 每轮总 episode 数 = `num_arenas * episodes_per_arena`
  - 所有竞技场完成后汇总适应度（简单平均）
  - 2 竞技场 * 50 episode = 每轮 100 个样本，提高适应度评估稳定性

- **做市商复合适应度权重**（mm_fitness_*_weight）：
  - `mm_fitness_pnl_weight`（alpha=0.7）：PnL 收益率，激励盈利
  - `mm_fitness_volume_weight`（gamma=0.3）：Maker 成交量，激励实际做市
  - 两个权重之和应为 1.0
  - 公式：`mm_fitness = alpha * pnl + gamma * volume_score`

- **持仓成本权重**（position_cost_weight / mm_position_cost_weight）：
  - 适应度公式：`fitness = (balance - initial) / initial - λ × |qty × price| / initial`
  - `position_cost_weight`（λ=0.02）：散户持仓成本，惩罚持仓不平仓
  - `mm_position_cost_weight`（λ=0.005）：做市商持仓成本，权重更小（做市商需持仓做市）
  - 完全多空对称：多头和空头施加相同惩罚，防止进化产生方向性偏好

- **CPU 亲和性**（enable_cpu_affinity）：
  - 启用后，ArenaWorkerPool 启动时将每个 Worker 进程绑定到独立的物理 CPU 核心
  - 通过读取 `/sys/devices/system/cpu/cpuN/topology/` 区分物理核心和超线程逻辑核心
  - 当物理核心数少于 Worker 数时自动跳过绑定并输出警告
  - 避免核心迁移带来的 CPU 缓存失效开销

**使用示例：**
```python
from src.config.config import TrainingConfig

training = TrainingConfig(
    episode_length=1000,
    checkpoint_interval=10,
    neat_config_path="config",
    evolution_interval=10,
    num_arenas=2,
    episodes_per_arena=50,
)
```

### DemoConfig

演示配置类，定义 WebUI 演示服务器的参数。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| host | str | 必填 | 服务器地址（默认 "localhost"） |
| port | int | 必填 | 服务器端口（默认 8000） |
| tick_interval | int | 必填 | tick 间隔（毫秒，默认 100） |

**详细说明：**

- **host**：WebUI 服务器监听地址
  - `localhost`：仅本机访问
  - `0.0.0.0`：允许外部访问

- **port**：WebUI 服务器监听端口
  - 默认 8000
  - 访问地址：`http://localhost:8000`

- **tick_interval**：每个 tick 的时间间隔（毫秒）
  - 100ms：较快节奏（默认）
  - 500ms：中等节奏
  - 1000ms：慢节奏，便于观察

**使用示例：**
```python
from src.config.config import DemoConfig

demo = DemoConfig(
    host="localhost",
    port=8000,
    tick_interval=100,
)
```

### NoiseTraderConfig

噪声交易者配置类，定义噪声交易者的行为参数。噪声交易者用于在市场中提供随机性流动性和布朗运动价格特征。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| count | int | 200 | 噪声交易者数量 |
| action_probability | float | 0.5 | 每个 tick 行动的概率（0-1） |
| quantity_mu | float | 12.0 | 对数正态分布的 mu 参数 |
| quantity_sigma | float | 1.0 | 对数正态分布的 sigma 参数 |
| episode_bias_range | float | 0.15 | Episode 级买入概率偏置范围，buy_prob ∈ [0.5-range, 0.5+range] |
| ou_theta | float | 0.035 | OU 过程均值回归速度（每 tick 回归 3.5% 的偏差） |
| ou_sigma | float | 0.04 | OU 过程噪声强度 |

**噪声交易者行为说明：**

- 每个 tick 每个噪声交易者以 `action_probability`（50%）概率决定是否行动
- 行动时 50% 买 / 50% 卖，通过市价单撮合
- 下单量：`max(1, int(lognormvariate(mu, sigma)))`
- 由中心极限定理保证净买卖量趋向正态分布，从而实现价格随机游走

**噪声交易者特殊规则：**
- 初始资金 1e18（视为无限资金）
- 不触发强平检查
- 手续费为 0（maker 和 taker 费率均为 0）
- 可作为 ADL 对手方
- 有完整的持仓和 PnL 跟踪

**使用示例：**
```python
from src.config.config import NoiseTraderConfig

noise_trader = NoiseTraderConfig(
    count=200,
    action_probability=0.5,
    quantity_mu=12.0,
    quantity_sigma=1.0,
    episode_bias_range=0.15,
)
```

### ASConfig

Avellaneda-Stoikov 做市商模型配置类，定义基于 AS 模型的做市策略参数。该配置供 NN-AS 混合做市商使用，神经网络输出乘数对 AS 模型的核心参数（gamma 和 spread）进行动态调整。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| gamma | float | 0.1 | 基础风险厌恶系数，影响 reservation price 偏离幅度 |
| kappa_base | float | 1.5 | 基础订单到达率，影响最优报价 spread 宽度 |
| vol_window | int | 50 | 波动率回看窗口（tick 数） |
| min_sigma | float | 1e-6 | 波动率下限，防止 sigma=0 导致除零 |
| max_sigma | float | 0.1 | 波动率上限，防止极端波动导致报价失控 |
| gamma_adj_min | float | 0.1 | NN 输出的 gamma 调整乘数下限 |
| gamma_adj_max | float | 10.0 | NN 输出的 gamma 调整乘数上限 |
| spread_adj_min | float | 0.5 | NN 输出的 spread 调整乘数下限 |
| spread_adj_max | float | 2.0 | NN 输出的 spread 调整乘数上限 |
| max_reservation_offset | float | 0.05 | reservation price 相对中间价的最大偏移（±5%） |

**详细说明：**

- **gamma**：AS 模型中的风险厌恶系数（γ），控制做市商对库存风险的敏感度
  - 值越大，reservation price 偏离中间价越多，倾向于更快减仓
  - 神经网络通过 `gamma_adj` 乘数对其动态调整：`effective_gamma = gamma × gamma_adj`
  - `gamma_adj` 被映射到 `[gamma_adj_min, gamma_adj_max]` 区间

- **kappa_base**：基础订单到达率（κ），影响最优报价 spread 的计算
  - AS 模型最优 spread 公式（简化版）：`spread = γ × σ² × T + (2/γ) × ln(1 + γ/κ)`
  - 值越大表示订单越容易成交，spread 可设得更窄

- **vol_window**：计算实时波动率 σ 的回看窗口
  - 使用最近 `vol_window` 个 tick 的对数收益率标准差估计 σ
  - 波动率被 clamp 到 `[min_sigma, max_sigma]`

- **max_reservation_offset**：reservation price 相对当前中间价的最大偏移比例
  - reservation price = 中间价 - γ × σ² × q × T（q 为库存量，T 为剩余时间）
  - 偏移超出 ±5% 时被截断，防止极端库存下报价失真

- **NN 调整乘数范围**（gamma_adj_* / spread_adj_*）：
  - 神经网络输出经 sigmoid 激活后映射到对应区间
  - `gamma_adj_min=0.1`：最保守，gamma 缩小至 10%
  - `gamma_adj_max=10.0`：最激进，gamma 放大 10 倍
  - `spread_adj_min=0.5`：spread 收窄至 50%
  - `spread_adj_max=2.0`：spread 放宽至 200%

**使用示例：**
```python
from src.config.config import ASConfig

as_cfg = ASConfig(
    gamma=0.1,
    kappa_base=1.5,
    vol_window=50,
    min_sigma=1e-6,
    max_sigma=0.1,
    gamma_adj_min=0.1,
    gamma_adj_max=10.0,
    spread_adj_min=0.5,
    spread_adj_max=2.0,
    max_reservation_offset=0.05,
)
```

### Config

全局配置类，汇总所有配置项。这是配置系统的顶层入口。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| market | MarketConfig | 必填 | 市场配置 |
| agents | dict[AgentType, AgentConfig] | 必填 | Agent 配置（按类型） |
| training | TrainingConfig | 必填 | 训练配置 |
| demo | DemoConfig | 必填 | 演示配置 |
| noise_trader | NoiseTraderConfig | NoiseTraderConfig() | 噪声交易者配置 |
| as_model | ASConfig | ASConfig() | Avellaneda-Stoikov 做市商模型配置 |

**使用示例：**
```python
from src.config.config import (
    AgentConfig, AgentType, Config,
    DemoConfig, MarketConfig, NoiseTraderConfig, TrainingConfig
)

market = MarketConfig(
    initial_price=100.0,
    tick_size=0.01,
    lot_size=1.0,
    depth=5,
)

agents = {
    AgentType.RETAIL_PRO: AgentConfig(
        count=2400,
        initial_balance=20000.0,
        leverage=10.0,
        maintenance_margin_rate=0.05,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
    ),
    AgentType.MARKET_MAKER: AgentConfig(
        count=400,
        initial_balance=10_000_000.0,
        leverage=10.0,
        maintenance_margin_rate=0.05,
        maker_fee_rate=-0.0001,
        taker_fee_rate=0.0001,
    ),
}

training = TrainingConfig(
    episode_length=1000,
    checkpoint_interval=10,
    neat_config_path="config",
)

demo = DemoConfig(
    host="localhost",
    port=8000,
    tick_interval=100,
)

noise_trader = NoiseTraderConfig()

config = Config(
    market=market,
    agents=agents,
    training=training,
    demo=demo,
    noise_trader=noise_trader,
)
```

## NEAT 配置文件

NEAT 配置文件位于项目根目录的 `config/` 文件夹下，每种 Agent 类型对应一个配置文件：

| 文件名 | 对应 Agent | 输入节点 | 输出节点 | 隐藏节点 | 种群大小 |
|--------|----------|---------|---------|---------|---------|
| neat_retail_pro.cfg | 高级散户 | 527 | 8 | 10 | 200 |
| neat_market_maker.cfg | 做市商 | 592 | 43 | 10 | 150 |

### 输入输出节点说明

**高级散户 (RETAIL_PRO) - 527 输入，8 输出：**
- 输入：
  - 5档买盘价格归一化(5) + 5档买盘数量归一化(5) = 10
  - 5档卖盘价格归一化(5) + 5档卖盘数量归一化(5) = 10
  - 100笔成交价格归一化(100)
  - 100笔成交数量归一化(100)
  - 持仓信息(4)
  - 挂单信息(3)
  - tick历史价格归一化(100)
  - tick历史成交量归一化(100)
  - tick历史成交额归一化(100)
  - 总计：10 + 10 + 100 + 100 + 4 + 3 + 100 + 100 + 100 = 527
- 输出：与动作空间对应，8个

**做市商 (MARKET_MAKER) - 592 输入，43 输出：**
- 输入：
  - 5档买盘价格归一化(5) + 5档买盘数量归一化(5) = 10
  - 5档卖盘价格归一化(5) + 5档卖盘数量归一化(5) = 10
  - 100笔成交价格归一化(100)
  - 100笔成交数量归一化(100)
  - 持仓信息(4)
  - 挂单信息(60)：10买单×3 + 10卖单×3
  - tick历史价格归一化(100)
  - tick历史成交量归一化(100)
  - tick历史成交额归一化(100)
  - AS模型特征(8)
  - 总计：10 + 10 + 100 + 100 + 4 + 60 + 100 + 100 + 100 + 8 = 592
- 输出：
  - 买单价格偏移(10)：10个买单的价格偏移（相对 reservation_price）
  - 买单数量权重(10)：10个买单的数量权重
  - 卖单价格偏移(10)：10个卖单的价格偏移（相对 reservation_price）
  - 卖单数量权重(10)：10个卖单的数量权重
  - 总下单比例基准(1)：整体下单量比例
  - gamma_adjustment(1)：AS模型gamma调整乘数 [0.1, 10.0]
  - spread_adjustment(1)：AS模型spread调整乘数 [0.5, 2.0]
  - 总计：10 + 10 + 10 + 10 + 1 + 1 + 1 = 43

### NEAT 配置文件格式

NEAT 配置文件使用 INI 格式，包含以下主要部分：

**[NEAT] 部分：**
- `fitness_criterion` - 适应度标准（max 或 min）
- `fitness_threshold` - 适应度阈值（达到此值时终止训练）
- `pop_size` - 种群大小（会被 AgentConfig.count 动态覆盖）
- `reset_on_extinction` - 物种灭绝时是否重置（True）
- `no_fitness_termination` - 是否禁用基于适应度的终止（True）

**[DefaultGenome] 部分：**
- `activation_default` - 默认激活函数（tanh）
- `num_inputs` - 输入节点数
- `num_outputs` - 输出节点数
- `num_hidden` - 隐藏节点数
- `feed_forward` - 是否为前馈网络（True）
- `initial_connection` - 初始连接方式（partial_direct 0.7）
- `weight_init_mean/stdev` - 权重初始化参数
- `bias_init_mean/stdev` - 偏置初始化参数
- `node_add_prob/delete_prob` - 节点添加/删除概率（0.2）
- `conn_add_prob/delete_prob` - 连接添加/删除概率（0.5）

**[DefaultSpeciesSet] 部分：**
- `compatibility_threshold` - 物种兼容性阈值（1.8）

**[DefaultStagnation] 部分：**
- `species_fitness_func` - 物种适应度函数（max）
- `max_stagnation` - 最大停滞代数（20）
- `species_elitism` - 物种精英保留数量（2）

**[DefaultReproduction] 部分：**
- `elitism` - 精英保留数量（1）
- `survival_threshold` - 存活阈值（0.5）
- `min_species_size` - 最小物种大小（1）

## 使用方式

### 1. 通过 scripts/create_config.py 创建默认配置

这是推荐的方式，所有训练脚本都使用此方法创建配置。

```python
from scripts.create_config import create_default_config

# 创建默认配置
config = create_default_config(
    episode_length=1000,
    checkpoint_interval=10,
)
```

**create_default_config 函数签名：**
```python
def create_default_config(
    episode_length: int = 1000,
    checkpoint_interval: int = 10,
    config_dir: str = "config",
    evolution_interval: int = 10,
) -> Config:
```

### 2. 直接创建配置对象

如果需要自定义更多参数，可以直接创建配置对象。

```python
from src.config.config import (
    AgentConfig, AgentType, Config,
    DemoConfig, MarketConfig, NoiseTraderConfig, TrainingConfig
)

market = MarketConfig(
    initial_price=100.0,
    tick_size=0.01,
    lot_size=1.0,
    depth=5,
    ema_alpha=1.0,
)

agents = {
    AgentType.RETAIL_PRO: AgentConfig(
        count=2400,
        initial_balance=20000.0,
        leverage=10.0,
        maintenance_margin_rate=0.05,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
    ),
    AgentType.MARKET_MAKER: AgentConfig(
        count=400,
        initial_balance=10_000_000.0,
        leverage=10.0,
        maintenance_margin_rate=0.05,
        maker_fee_rate=-0.0001,
        taker_fee_rate=0.0001,
    ),
}

training = TrainingConfig(
    episode_length=1000,
    checkpoint_interval=10,
    neat_config_path="config",
)

demo = DemoConfig(
    host="localhost",
    port=8000,
    tick_interval=100,
)

config = Config(
    market=market,
    agents=agents,
    training=training,
    demo=demo,
    noise_trader=NoiseTraderConfig(),
)
```

## 配置参数调优建议

### 市场参数调优

- **ema_alpha**：
  - 训练时：0.9（较强平滑，减少噪音）
  - 演示时：0.1-0.3（更平滑的价格变化）
  - 值越小，价格变化越平滑，但可能滞后

- **depth**：
  - 增大 depth 会增加输入维度
  - 需同步修改 NEAT 配置文件中的 `num_inputs`
  - 高级散户和做市商均使用 5 档

### Agent 参数调优

- **initial_balance**：
  - 高级散户：2万
  - 做市商：1,000万 - 5,000万

- **leverage**：
  - 高级散户：10.0（高杠杆交易）
  - 做市商：10.0（提供深厚流动性）

- **maintenance_margin_rate**：
  - 通常设为 `0.5 / leverage`
  - 做市商必须如此设置（当前杠杆10.0时为0.05）

### 训练参数调优

- **episode_length**：
  - 快速测试：100-500
  - 正常训练：1,000-5,000
  - 长期训练：10,000+

- **parallel_workers**：
  - 根据 CPU 核心数调整
  - 建议 `CPU 核心数 - 2`

- **openmp_threads**：
  - 默认 16，建议设为物理核心数
  - 过多线程反而更慢

### 噪声交易者参数调优

- **count**：
  - 200：默认数量，提供充足的随机流动性

- **action_probability**：
  - 0.5：50% 概率行动（默认）

- **quantity_mu / quantity_sigma**：
  - mu=12.0, sigma=1.0：默认参数，单笔均值约 268,337 单位，200个噪声交易者每 tick 总成交量约 2,680 万单位，噪声力量约为散户现实定向力量的 3.4 倍，既主导价格走势又不会击穿做市商双边挂单

- **episode_bias_range**：
  - 0.15：每个 Episode 开始时，每个竞技场独立生成买入概率偏置，范围 [0.35, 0.65]
  - 每个竞技场独立偏置，防止 NEAT 进化形成单向交易的羊群效应
  - 设为 0 则退化为无偏置的等概率买卖

## 依赖关系

### 外部依赖
- `dataclasses` - 数据类支持（Python 3.7+），使用 `dataclass` 装饰器和 `field` 函数
- `enum` - 枚举类型支持，`AgentType` 继承 `str` 和 `Enum`

### 内部依赖
- 无（配置模块是独立的基础模块）

### 被依赖
- `src/training` - 训练模块使用配置创建种群和训练器
- `src/market/noise_trader` - 噪声交易者模块使用 NoiseTraderConfig
- `src/bio/agents` - Agent 模块使用 AgentConfig 和 MarketConfig
- `src/analysis` - 分析模块使用 Config 和 AgentType

## 相关文档

- `../../CLAUDE.md` - 项目根目录文档
- `../training/CLAUDE.md` - 训练模块，使用配置创建种群和训练器
- `../market/noise_trader/CLAUDE.md` - 噪声交易者模块详细说明
- `../../config/*.cfg` - NEAT 配置文件
- `../../scripts/create_config.py` - 默认配置创建函数
