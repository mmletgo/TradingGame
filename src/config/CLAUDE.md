# Config 配置模块

## 模块概述

配置模块是整个交易竞技场系统的配置中心，定义了所有核心组件的配置数据结构。所有配置项都使用 Python dataclass 定义，提供类型安全、清晰的默认值和 IDE 友好的代码补全支持。

## 模块职责

1. **市场配置管理** - 定义交易市场的核心参数（价格、深度、EMA 等）
2. **Agent 配置管理** - 定义四种 Agent 类型的交易参数（资金、杠杆、费率等）
3. **训练配置管理** - 定义 NEAT 进化训练的参数（episode 长度、并行化配置等）
4. **演示配置管理** - 定义 WebUI 演示服务器的参数
5. **鲶鱼配置管理** - 定义鲶鱼机制的行为参数

## 文件结构

- `config.py` - 核心配置数据类定义
- `CLAUDE.md` - 本文档

## 核心枚举

### AgentType

定义系统中四种 AI Agent 的类型枚举，用于标识和区分不同类型的 Agent。

**枚举值：**
- `RETAIL` - 散户：数量大（10,000），资金小（2万），只能看到浅层订单簿（10档）
- `RETAIL_PRO` - 高级散户：数量少（100），与散户相同资金，可看到深层订单簿（100档）
- `WHALE` - 庄家：数量少（100），资金大（300万），享有低费率返佣，可看到深层订单簿
- `MARKET_MAKER` - 做市商：数量中等（400），资金最大（1,000万），必须双边挂单提供流动性

**使用示例：**
```python
from src.config.config import AgentType

# 遍历所有 Agent 类型
for agent_type in AgentType:
    print(f"Agent类型: {agent_type.name}, 值: {agent_type.value}")

# 判断 Agent 类型
if agent_type == AgentType.MARKET_MAKER:
    print("这是做市商")
```

### CatfishMode

定义鲶鱼的行为模式枚举，用于配置鲶鱼的交易策略。

**枚举值：**
- `TREND_CREATOR` - 趋势创造者：Episode 开始时随机选择方向，整个 Episode 保持该方向持续操作
- `TREND_FOLLOWING` - 趋势追踪（向后兼容别名，等同于 TREND_CREATOR）
- `MEAN_REVERSION` - 逆势操作（均值回归）：当价格偏离 EMA 均线时反向操作
- `RANDOM` - 随机买卖：以随机概率进行买卖操作

**使用示例：**
```python
from src.config.config import CatfishConfig, CatfishMode

# 配置单模式鲶鱼
config = CatfishConfig(
    enabled=True,
    multi_mode=False,
    mode=CatfishMode.MEAN_REVERSION,
    ma_period=20,
    deviation_threshold=0.003,
)
```

## 核心配置类

### MarketConfig

市场配置类，定义交易市场的核心参数。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| initial_price | float | 必填 | 初始价格（默认 100.0） |
| tick_size | float | 必填 | 最小变动单位（默认 0.01） |
| lot_size | float | 必填 | 最小交易单位（默认 1.0） |
| depth | int | 必填 | 盘口深度，买卖各多少档（默认 100） |
| ema_alpha | float | 0.9 | EMA 平滑系数（0-1），值越小价格变化越平滑 |

**详细说明：**

- **initial_price**：市场的初始价格，所有 Agent 的初始资产估值基于此价格
- **tick_size**：价格的最小变动单位，所有订单价格必须是 tick_size 的整数倍
- **lot_size**：订单数量的最小单位，所有订单数量必须是 lot_size 的整数倍
- **depth**：订单簿深度，决定买盘和卖盘各保留多少档位，同时影响 Agent 的输入维度
  - 散户：使用 10 档订单簿数据
  - 高级散户/庄家/做市商：使用 100 档订单簿数据
- **ema_alpha**：EMA（指数移动平均）平滑系数，用于平滑价格波动
  - 公式：`smooth_price = α × current_price + (1-α) × prev_smooth_price`
  - 训练时通常设为 0.9（较强平滑）
  - 演示时可以设为更小的值（0.1-0.3）以展示更平滑的价格变化

**使用示例：**
```python
from src.config.config import MarketConfig

market = MarketConfig(
    initial_price=100.0,
    tick_size=0.01,
    lot_size=1.0,
    depth=100,
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

**默认配置示例（来自 scripts/create_config.py）：**

| Agent 类型 | 数量 | 初始资金 | 杠杆 | 维持保证金率 | 挂单费率 | 吃单费率 |
|-----------|------|---------|------|------------|---------|---------|
| 散户 (RETAIL) | 10,000 | 2万 | 1.0 | 0.5 | 0.0002 (万2) | 0.0005 (万5) |
| 高级散户 (RETAIL_PRO) | 100 | 2万 | 1.0 | 0.5 | 0.0002 (万2) | 0.0005 (万5) |
| 庄家 (WHALE) | 100 | 300万 (3M) | 1.0 | 0.5 | -0.0001 (负万1) | 0.0001 (万1) |
| 做市商 (MARKET_MAKER) | 400 | 1,000万 (10M) | 1.0 | 0.5 | -0.0001 (负万1) | 0.0001 (万1) |

**详细说明：**

- **count**：该类型 Agent 的种群数量
  - 散户数量大（10,000），模拟散户群体的多样性
  - 高级散户和庄家各 100，代表精英交易者
  - 做市商 400，提供足够的流动性

- **initial_balance**：初始资金（单位：元）
  - 散户：2万，模拟小资金散户
  - 高级散户：与散户相同，区别在于能看到更深的订单簿
  - 庄家：300万，大资金交易者
  - 做市商：1,000万，最大资金群体，提供充足流动性

- **leverage**：杠杆倍数
  - 当前配置：所有 Agent 杠杆均为 1.0（无杠杆）
  - 可调整：庄家和做市商可以设置更高杠杆（如 10.0）

- **maintenance_margin_rate**：维持保证金率
  - 公式：`强平触发条件 = 净值 / 持仓市值 < 维持保证金率`
  - 当前配置：所有 Agent 均为 0.5
  - 做市商的特殊计算：`0.5 / leverage`（确保杠杆调整后强平价格合理）

- **maker_fee_rate**：挂单手续费率
  - 散户/高级散户：万2（0.0002）
  - 庄家/做市商：负万1（-0.0001），表示返佣，鼓励提供流动性
  - 负费率：成交后交易所返还手续费，降低做市成本

- **taker_fee_rate**：吃单手续费率
  - 散户/高级散户：万5（0.0005）
  - 庄家/做市商：万1（0.0001），享受机构费率

**使用示例：**
```python
from src.config.config import AgentConfig, AgentType

retail_config = AgentConfig(
    count=10000,
    initial_balance=20000.0,
    leverage=1.0,
    maintenance_margin_rate=0.5,
    maker_fee_rate=0.0002,
    taker_fee_rate=0.0005,
)

# 计算实际占用保证金
# 假设持仓 1000 股，价格 100 元，杠杆 1.0
# 占用保证金 = (1000 × 100) / 1.0 = 100,000 元
# 维持保证金 = 100,000 × 0.5 = 50,000 元
# 如果净值 < 50,000，则触发强平
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
| openmp_threads | int | 12 | OpenMP 并行线程数（Cython 神经网络推理） |
| random_seed | int \| None | None | 随机种子（None 表示不固定） |
| retail_sub_population_count | int | 10 | 散户子种群数量 |
| evolution_interval | int | 10 | 每多少个 episode 进化一次 |
| num_arenas | int | 2 | 竞技场数量（多竞技场模式） |
| episodes_per_arena | int | 50 | 每个竞技场运行的 episode 数 |
| mm_fitness_pnl_weight | float | 0.4 | 做市商复合适应度中 PnL 收益率权重 α |
| mm_fitness_spread_weight | float | 0.3 | 做市商复合适应度中盘口价差质量权重 β |
| mm_fitness_volume_weight | float | 0.2 | 做市商复合适应度中 Maker 成交量权重 γ |
| mm_fitness_survival_weight | float | 0.1 | 做市商复合适应度中存活权重 δ |

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
  - 例如：`config` 目录下应包含 `neat_retail.cfg`、`neat_whale.cfg` 等

- **并行化配置**：
  - `parallel_workers`：并行工作进程数，建议 `CPU 核心数 - 2`
  - `enable_parallel_evolution`：启用并行进化，使用多进程同时进化多个种群
  - `enable_parallel_decision`：启用并行决策（**已废弃**，当前使用串行决策，因 GIL 限制串行更快）
  - `enable_parallel_creation`：启用并行创建 Agent

- **openmp_threads**：Cython 神经网络推理的 OpenMP 线程数
  - 经基准测试，8-12 线程为最优值
  - 过多线程反而更慢（线程调度开销、内存带宽瓶颈）

- **random_seed**：随机种子
  - 设为整数：固定随机种子，结果可复现
  - 设为 None：不固定随机种子（默认）

- **retail_sub_population_count**：散户子种群数量
  - 将 10,000 个散户拆分为 10 个子种群，每个 1,000 个
  - 减少单个 NEAT 种群的规模，优化进化性能
  - 每个子种群独立进化，共享市场环境

- **evolution_interval**：进化间隔
  - 默认 10：每 10 个 episode 进化一次
  - 进化时使用累积的平均适应度

- **多竞技场配置**（多竞技场并行训练模式）：
  - `num_arenas`：竞技场数量（默认 2）
  - `episodes_per_arena`：每个竞技场运行的 episode 数（默认 50）
  - 每轮总 episode 数 = `num_arenas × episodes_per_arena`
  - 所有竞技场完成后汇总适应度（简单平均）
  - 2 竞技场 × 50 episode = 每轮 100 个样本，提高适应度评估稳定性

- **做市商复合适应度权重**（mm_fitness_*_weight）：
  - `mm_fitness_pnl_weight`（α=0.4）：PnL 收益率，激励盈利
  - `mm_fitness_spread_weight`（β=0.3）：盘口价差质量，激励提供紧盘口
  - `mm_fitness_volume_weight`（γ=0.2）：Maker 成交量，激励实际做市
  - `mm_fitness_survival_weight`（δ=0.1）：存活奖励，激励风控
  - 四个权重之和应为 1.0
  - 权重调整建议：如果做市商仍然消极，增加 β 和 γ；如果亏损过大，增加 α

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

### CatfishConfig

鲶鱼配置类，定义鲶鱼（Catfish）的行为参数。鲶鱼是一种特殊的市场参与者，用于在训练中引入外部扰动，增加市场动态性。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| enabled | bool | False | 是否启用鲶鱼 |
| multi_mode | bool | True | 是否同时启用三种模式 |
| mode | CatfishMode | TREND_CREATOR | 单模式时的鲶鱼行为模式 |
| ma_period | int | 20 | 均线周期（EMA计算） |
| deviation_threshold | float | 0.003 | 价格偏离EMA的阈值（0.003 = 0.3%） |
| action_probability | float | 0.3 | 每个 tick 行动的概率（0-1） |

**鲶鱼行为模式说明：**

1. **趋势创造者 (TREND_CREATOR)**
   - Episode 开始时随机选择方向（买或卖）
   - 整个 Episode 保持该方向持续操作
   - 行动概率由 `action_probability` 控制

2. **逆势操作 (MEAN_REVERSION)**
   - 维护 EMA 均值（周期 `ma_period`）
   - 当当前价格偏离 EMA 超过 `deviation_threshold`，反向操作
   - 行动概率由 `action_probability` 控制

3. **随机买卖 (RANDOM)**
   - 以 `action_probability` 概率触发交易
   - 若触发，随机选择买或卖

**多模式 vs 单模式：**

- **多模式（multi_mode=True，默认）**：
  - 三种鲶鱼同时运行
  - 每个 tick 各自独立决定是否行动
  - 提供更丰富的市场扰动

- **单模式（multi_mode=False）**：
  - 只运行一种模式的鲶鱼
  - 由 `mode` 参数指定模式

**鲶鱼资金计算（当前默认）：**

```
鲶鱼总资金 = 做市商杠杆后资金 - 其他物种杠杆后资金
每条鲶鱼资金 = 总资金 / 3
```

默认配置下：
- 做市商：400 × 10M × 1.0 = 4,000M
- 其他物种：200M + 2M + 300M = 502M
- 鲶鱼总资金：4,000M - 502M = 3,498M
- 每条鲶鱼：约 1,166M

**使用示例：**
```python
from src.config.config import CatfishConfig, CatfishMode

# 多模式（推荐）
catfish = CatfishConfig(
    enabled=True,
    multi_mode=True,
    action_probability=0.3,
)

# 单模式
catfish = CatfishConfig(
    enabled=True,
    multi_mode=False,
    mode=CatfishMode.MEAN_REVERSION,
    ma_period=20,
    deviation_threshold=0.003,
    action_probability=0.3,
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
| catfish | CatfishConfig \| None | None | 鲶鱼配置（可选） |

**使用示例：**
```python
from src.config.config import (
    AgentConfig, AgentType, CatfishConfig, Config,
    DemoConfig, MarketConfig, TrainingConfig
)

market = MarketConfig(
    initial_price=100.0,
    tick_size=0.01,
    lot_size=1.0,
    depth=100,
)

agents = {
    AgentType.RETAIL: AgentConfig(
        count=10000,
        initial_balance=20000.0,
        leverage=1.0,
        maintenance_margin_rate=0.5,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
    ),
    AgentType.RETAIL_PRO: AgentConfig(
        count=100,
        initial_balance=20000.0,
        leverage=1.0,
        maintenance_margin_rate=0.5,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
    ),
    AgentType.WHALE: AgentConfig(
        count=100,
        initial_balance=3_000_000.0,
        leverage=1.0,
        maintenance_margin_rate=0.5,
        maker_fee_rate=-0.0001,
        taker_fee_rate=0.0001,
    ),
    AgentType.MARKET_MAKER: AgentConfig(
        count=400,
        initial_balance=10_000_000.0,
        leverage=1.0,
        maintenance_margin_rate=0.5,
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

catfish = CatfishConfig(
    enabled=True,
    multi_mode=True,
)

config = Config(
    market=market,
    agents=agents,
    training=training,
    demo=demo,
    catfish=catfish,
)
```

## NEAT 配置文件

NEAT 配置文件位于项目根目录的 `config/` 文件夹下，每种 Agent 类型对应一个配置文件：

| 文件名 | 对应 Agent | 输入节点 | 输出节点 | 隐藏节点 | 种群大小 |
|--------|----------|---------|---------|---------|---------|
| neat_retail.cfg | 散户 | 127 | 8 | 3 | 150 |
| neat_retail_pro.cfg | 高级散户 | 907 | 8 | 10 | 100 |
| neat_whale.cfg | 庄家 | 907 | 8 | 10 | 200 |
| neat_market_maker.cfg | 做市商 | 964 | 41 | 10 | 150 |

### 输入输出节点说明

**散户 (RETAIL) - 127 输入，8 输出：**
- 输入：
  - 10档买盘价格归一化(10) + 10档买盘数量归一化(10) = 20
  - 10档卖盘价格归一化(10) + 10档卖盘数量归一化(10) = 20
  - 10笔成交价格归一化(10)
  - 10笔成交数量归一化(10)
  - 持仓信息(4)：持仓数量、持仓均价、浮动盈亏、净值归一化
  - 挂单信息(3)：挂单价格归一化、挂单数量归一化、是否有挂单
  - tick历史价格归一化(20)
  - tick历史成交量归一化(20)
  - tick历史成交额归一化(20)
  - 总计：20 + 20 + 10 + 10 + 4 + 3 + 20 + 20 + 20 = 127
- 输出：
  - 动作选择(6)：HOLD、挂买单、挂卖单、撤单、市价买入、市价卖出
  - 价格偏移(1)：相对于中间价的偏移量
  - 数量比例(1)：相对于净值的数量比例
  - 总计：6 + 1 + 1 = 8

**高级散户 (RETAIL_PRO) - 907 输入，8 输出：**
- 输入：
  - 100档买盘价格归一化(100) + 100档买盘数量归一化(100) = 200
  - 100档卖盘价格归一化(100) + 100档卖盘数量归一化(100) = 200
  - 100笔成交价格归一化(100)
  - 100笔成交数量归一化(100)
  - 持仓信息(4)
  - 挂单信息(3)
  - tick历史价格归一化(100)
  - tick历史成交量归一化(100)
  - tick历史成交额归一化(100)
  - 总计：200 + 200 + 100 + 100 + 4 + 3 + 100 + 100 + 100 = 907
- 输出：与散户相同，8个

**庄家 (WHALE) - 907 输入，8 输出：**
- 输入：与高级散户完全相同，907个
- 输出：与散户相同，8个
- 区别：适应度计算包含波动性贡献，激励庄家制造市场波动

**做市商 (MARKET_MAKER) - 964 输入，41 输出：**
- 输入：
  - 100档买盘价格归一化(100) + 100档买盘数量归一化(100) = 200
  - 100档卖盘价格归一化(100) + 100档卖盘数量归一化(100) = 200
  - 100笔成交价格归一化(100)
  - 100笔成交数量归一化(100)
  - 持仓信息(4)
  - 挂单信息(60)：当前挂单信息 + 历史挂单记录
  - tick历史价格归一化(100)
  - tick历史成交量归一化(100)
  - 总计：200 + 200 + 100 + 100 + 4 + 60 + 100 + 100 = 964
- 输出：
  - 买单价格权重(5)：5个价格档位的权重
  - 买单数量权重(5)：5个数量档位的权重
  - 卖单价格权重(5)：5个价格档位的权重
  - 卖单数量权重(5)：5个数量档位的权重
  - 总下单比例基准(1)：整体下单量比例
  - 总计：5 + 5 + 5 + 5 + 1 = 21（注释错误，实际应为21，非41）

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

# 创建默认配置（禁用鲶鱼）
config = create_default_config(
    episode_length=1000,
    checkpoint_interval=10,
    catfish_enabled=False,
)

# 创建启用鲶鱼的配置
config = create_default_config(
    episode_length=100,
    checkpoint_interval=10,
    catfish_enabled=True,
)
```

**create_default_config 函数签名：**
```python
def create_default_config(
    episode_length: int = 1000,
    checkpoint_interval: int = 10,
    config_dir: str = "config",
    catfish_enabled: bool = False,
    evolution_interval: int = 10,
) -> Config:
```

### 2. 直接创建配置对象

如果需要自定义更多参数，可以直接创建配置对象。

```python
from src.config.config import (
    AgentConfig, AgentType, CatfishConfig, Config,
    DemoConfig, MarketConfig, TrainingConfig
)

market = MarketConfig(
    initial_price=100.0,
    tick_size=0.01,
    lot_size=1.0,
    depth=100,
    ema_alpha=1.0,
)

agents = {
    AgentType.RETAIL: AgentConfig(
        count=10000,
        initial_balance=20000.0,
        leverage=1.0,
        maintenance_margin_rate=0.5,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
    ),
    # ... 其他 Agent 类型
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
)
```

### 3. 训练时启用鲶鱼

```bash
# 命令行启用鲶鱼（三种模式同时运行）
python scripts/train_noui.py --episodes 100 --catfish

# 指定单模式
python scripts/train_noui.py --episodes 100 --catfish --catfish-mode trend_creator
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
  - 散户使用 10，其他 Agent 使用 100

### Agent 参数调优

- **initial_balance**：
  - 散户：2万左右
  - 高级散户：与散户相同
  - 庄家：300万 - 1,000万
  - 做市商：1,000万 - 5,000万

- **leverage**：
  - 散户/高级散户：1.0（无杠杆）
  - 庄家：1.0-10.0
  - 做市商：1.0-10.0

- **maintenance_margin_rate**：
  - 通常设为 `0.5 / leverage`
  - 做市商必须如此设置（当前杠杆1.0时为0.5）

### 训练参数调优

- **episode_length**：
  - 快速测试：100-500
  - 正常训练：1,000-5,000
  - 长期训练：10,000+

- **parallel_workers**：
  - 根据 CPU 核心数调整
  - 建议 `CPU 核心数 - 2`

- **openmp_threads**：
  - 经基准测试，8-12 线程为最优值
  - 过多线程反而更慢

### 鲶鱼参数调优

- **action_probability**：
  - 0.3：30% 概率行动（默认，中等扰动）
  - 0.6：60% 概率行动（强扰动）
  - 0.1：10% 概率行动（轻微扰动）

- **ma_period**：
  - 20-200，均线周期（逆势操作鲶鱼使用）

- **deviation_threshold**：
  - 0.001-0.01，偏离阈值（逆势操作鲶鱼使用）
  - 0.003 = 0.3%

## 依赖关系

### 外部依赖
- `dataclasses` - 数据类支持（Python 3.7+）
- `enum` - 枚举类型支持

### 内部依赖
- 无（配置模块是独立的基础模块）

### 被依赖
- `src/training` - 训练模块使用配置创建种群和训练器
- `src/market/catfish` - 鲶鱼模块使用 CatfishConfig
- `src/bio/agents` - Agent 模块使用 AgentConfig 和 MarketConfig

## 相关文档

- `../../CLAUDE.md` - 项目根目录文档
- `../training/CLAUDE.md` - 训练模块，使用配置创建种群和训练器
- `../market/catfish/CLAUDE.md` - 鲶鱼模块详细说明
- `../../config/*.cfg` - NEAT 配置文件
- `../../scripts/create_config.py` - 默认配置创建函数
