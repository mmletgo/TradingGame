# Config 配置模块

## 模块概述

配置模块定义系统的各类配置数据类，提供类型安全的配置管理。所有配置项都使用 Python dataclass 定义，支持清晰的类型注解和默认值。

## 文件结构

- `config.py` - 核心配置数据类定义

## 核心枚举

### AgentType

定义系统中四种 AI Agent 的类型。

**枚举值：**
- `RETAIL` - 散户
- `RETAIL_PRO` - 高级散户
- `WHALE` - 庄家
- `MARKET_MAKER` - 做市商

### CatfishMode

定义鲶鱼的行为模式。

**枚举值：**
- `TREND_FOLLOWING` - 趋势追踪：根据历史价格变化率顺势下单
- `CYCLE_SWING` - 周期摆动：按固定周期交替买卖
- `MEAN_REVERSION` - 逆势操作：当价格偏离均线时反向操作

## 核心配置类

### MarketConfig

市场配置，定义交易市场的核心参数。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| initial_price | float | 必填 | 初始价格（默认 100.0） |
| tick_size | float | 必填 | 最小变动单位（默认 0.1） |
| lot_size | float | 必填 | 最小交易单位（默认 1.0） |
| depth | int | 必填 | 盘口深度（买卖各多少档，默认 100） |
| ema_alpha | float | 0.1 | EMA 平滑系数，0-1 之间，值越小价格变化越平滑（训练时默认 1.0） |

**注意事项：**
- `ema_alpha` 在训练时通常设为 1.0（不使用 EMA），仅在需要平滑价格时设置更小的值
- `depth` 决定了订单簿的档位数量，同时也影响各 Agent 的输入维度

### AgentConfig

Agent 配置，定义特定类型 Agent 的交易参数。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| count | int | 必填 | 该类型 Agent 的数量 |
| initial_balance | float | 必填 | 初始资产 |
| leverage | float | 必填 | 杠杆倍数 |
| maintenance_margin_rate | float | 必填 | 维持保证金率 |
| maker_fee_rate | float | 必填 | 挂单费率（负数表示返佣） |
| taker_fee_rate | float | 必填 | 吃单费率 |

**默认配置示例（来自 create_config.py）：**

| Agent 类型 | 数量 | 初始资金 | 杠杆 | 维持保证金率 | 挂单费率 | 吃单费率 |
|-----------|------|---------|------|------------|---------|---------|
| 散户 (RETAIL) | 10,000 | 20万 | 10.0 | 0.05 | 0.0002 (万2) | 0.0005 (万5) |
| 高级散户 (RETAIL_PRO) | 100 | 20万 | 10.0 | 0.05 | 0.0002 (万2) | 0.0005 (万5) |
| 庄家 (WHALE) | 100 | 1,000万 | 10.0 | 0.05 | -0.0001 (负万1) | 0.0001 (万1) |
| 做市商 (MARKET_MAKER) | 100 | 2,000万 | 10.0 | 0.05 | -0.0001 (负万1) | 0.0001 (万1) |

**注意：**
- 庄家和做市商享有负的挂单费率（rebate），鼓励提供流动性
- 做市商的维持保证金率会动态计算：`0.5 / leverage`
- 散户数量大（10,000），其他类型各 100

### TrainingConfig

训练配置，定义 NEAT 进化训练的参数。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| episode_length | int | 必填 | 每个 episode 的 tick 数量（默认 1000） |
| checkpoint_interval | int | 必填 | 检查点间隔（episode 数，默认 10） |
| neat_config_path | str | 必填 | NEAT 配置文件目录路径 |
| parallel_workers | int | 16 | 并行工作进程数 |
| enable_parallel_evolution | bool | True | 是否启用并行进化 |
| enable_parallel_decision | bool | True | 是否启用并行决策 |
| enable_parallel_creation | bool | True | 是否启用并行创建 |
| random_seed | int \| None | None | 随机种子（None 表示不固定） |

**并行化说明：**
- `enable_parallel_evolution` - NEAT 进化阶段并行评估
- `enable_parallel_decision` - 每个 tick 中 Agent 决策并行化
- `enable_parallel_creation` - Agent 创建过程并行化

### DemoConfig

演示配置，定义 WebUI 演示的参数。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| host | str | 必填 | 服务器地址（默认 "localhost"） |
| port | int | 必填 | 服务器端口（默认 8000） |
| tick_interval | int | 必填 | tick 间隔（毫秒，默认 100） |

### CatfishConfig

鲶鱼配置，定义鲶鱼（Catfish）的行为参数。鲶鱼是一种特殊的市场参与者，用于在训练中引入外部扰动，增加市场动态性。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| enabled | bool | False | 是否启用鲶鱼 |
| multi_mode | bool | True | 是否同时启用三种模式 |
| mode | CatfishMode | TREND_FOLLOWING | 单模式时的鲶鱼行为模式 |
| fund_multiplier | float | 3.0 | 资金乘数（相对于做市商基础资金） |
| market_maker_base_fund | float | 20,000,000 | 做市商基础资金 |
| lookback_period | int | 10 | 趋势追踪回看周期（tick数） |
| trend_threshold | float | 0.002 | 趋势阈值（价格变化率） |
| half_cycle_length | int | 10 | 周期摆动半周期长度（tick数） |
| action_interval | int | 5 | 周期摆动行动间隔（tick数） |
| ma_period | int | 20 | 逆势操作均线周期 |
| deviation_threshold | float | 0.003 | 逆势操作偏离阈值 |
| action_cooldown | int | 10 | 行动冷却时间（tick数） |

**鲶鱼行为模式说明：**

1. **趋势追踪 (TREND_FOLLOWING)**
   - 回看 `lookback_period` 个 tick 的价格
   - 计算价格变化率，若超过 `trend_threshold` 则顺势下单
   - 有冷却时间 `action_cooldown`

2. **周期摆动 (CYCLE_SWING)**
   - 每 `half_cycle_length` 个 tick 切换方向
   - 每 `action_interval` 个 tick 行动一次
   - 无视市场状态，按固定周期交替买卖

3. **逆势操作 (MEAN_REVERSION)**
   - 维护 EMA 均值（周期 `ma_period`）
   - 当当前价格偏离 EMA 超过 `deviation_threshold`，反向操作
   - 有冷却时间 `action_cooldown`

**鲶鱼资金计算（当前默认）：**
```
鲶鱼总资金 = 做市商杠杆后资金 - 其他物种杠杆后资金
每条鲶鱼资金 = 总资金 / 3
```

默认配置下（fund_multiplier=3.0）：
- 做市商：100 × 2,000万 × 10 = 200亿
- 其他物种：100亿 + 2亿 + 100亿 = 202亿
- 鲶鱼总资金：200亿 × 3 - 202亿 = 398亿
- 每条鲶鱼：约 133亿

### Config

全局配置，汇总所有配置项。

**属性：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| market | MarketConfig | 必填 | 市场配置 |
| agents | dict[AgentType, AgentConfig] | 必填 | Agent 配置（按类型） |
| training | TrainingConfig | 必填 | 训练配置 |
| demo | DemoConfig | 必填 | 演示配置 |
| catfish | CatfishConfig \| None | None | 鲶鱼配置（可选） |

## NEAT 配置文件

NEAT 配置文件位于项目根目录的 `config/` 文件夹下，每种 Agent 类型对应一个配置文件：

| 文件名 | 对应 Agent | 输入节点 | 输出节点 | 隐藏节点 | 种群大小 |
|--------|----------|---------|---------|---------|---------|
| neat_retail.cfg | 散户 | 67 | 9 | 3 | 150 |
| neat_retail_pro.cfg | 高级散户 | 607 | 9 | 10 | 100 |
| neat_whale.cfg | 庄家 | 607 | 9 | 10 | 200 |
| neat_market_maker.cfg | 做市商 | 634 | 21 | 10 | 150 |

### 输入输出节点说明

**散户 (RETAIL) - 67 输入，9 输出：**
- 输入：10档买盘(20) + 10档卖盘(20) + 10笔成交价格(10) + 10笔成交数量(10) + 持仓(4) + 挂单(3) = 67
- 输出：动作选择(7) + 价格偏移(1) + 数量比例(1) = 9

**高级散户 (RETAIL_PRO) - 607 输入，9 输出：**
- 输入：100档买盘(200) + 100档卖盘(200) + 100笔成交价格(100) + 100笔成交数量(100) + 持仓(4) + 挂单(3) = 607
- 输出：动作选择(7) + 价格偏移(1) + 数量比例(1) = 9

**庄家 (WHALE) - 607 输入，9 输出：**
- 输入：100档买盘(200) + 100档卖盘(200) + 100笔成交价格(100) + 100笔成交数量(100) + 持仓(4) + 挂单(3) = 607
- 输出：HOLD得分(1) + 限价单得分(1) + 市价单得分(1) + CANCEL得分(1) + 限价做多得分(1) + 限价做空得分(1) + 市价做多得分(1) + 价格偏移(1) + 数量比例(1) = 9

**做市商 (MARKET_MAKER) - 634 输入，21 输出：**
- 输入：100档买盘(200) + 100档卖盘(200) + 100笔成交价格(100) + 100笔成交数量(100) + 持仓(4) + 挂单(3) + 保留(27) = 634
- 输出：买单价格(5) + 买单数量(5) + 卖单价格(5) + 卖单数量(5) + 总下单比例基准(1) = 21

### NEAT 配置文件格式

NEAT 配置文件使用 INI 格式，包含以下主要部分：

**[NEAT] 部分：**
- `fitness_criterion` - 适应度标准（max 或 min）
- `fitness_threshold` - 适应度阈值
- `pop_size` - 种群大小
- `reset_on_extinction` - 物种灭绝时是否重置
- `no_fitness_termination` - 是否禁用基于适应度的终止

**[DefaultGenome] 部分：**
- `activation_default` - 默认激活函数
- `num_inputs` - 输入节点数
- `num_outputs` - 输出节点数
- `num_hidden` - 隐藏节点数
- `feed_forward` - 是否为前馈网络
- `initial_connection` - 初始连接方式
- `weight_init_mean/stdev` - 权重初始化参数
- `bias_init_mean/stdev` - 偏置初始化参数
- `node_add_prob/delete_prob` - 节点添加/删除概率
- `conn_add_prob/delete_prob` - 连接添加/删除概率

**[DefaultSpeciesSet] 部分：**
- `compatibility_threshold` - 物种兼容性阈值

**[DefaultStagnation] 部分：**
- `species_fitness_func` - 物种适应度函数
- `max_stagnation` - 最大停滞代数
- `species_elitism` - 物种精英保留数量

**[DefaultReproduction] 部分：**
- `elitism` - 精英保留数量
- `survival_threshold` - 存活阈值
- `min_species_size` - 最小物种大小

## 使用方式

### 1. 通过 create_config.py 创建默认配置

```python
from scripts.create_config import create_default_config

# 创建默认配置（禁用鲶鱼）
config = create_default_config(
    episode_length=1000,
    checkpoint_interval=10,
    catfish_enabled=False
)

# 创建启用鲶鱼的配置
config = create_default_config(
    episode_length=100,
    checkpoint_interval=10,
    catfish_enabled=True,
    catfish_fund_multiplier=3.0
)
```

### 2. 直接创建配置对象

```python
from src.config.config import (
    AgentConfig, AgentType, CatfishConfig, Config,
    DemoConfig, MarketConfig, TrainingConfig
)

market = MarketConfig(
    initial_price=100.0,
    tick_size=0.1,
    lot_size=1.0,
    depth=100,
    ema_alpha=1.0
)

agents = {
    AgentType.RETAIL: AgentConfig(
        count=10000,
        initial_balance=200000.0,
        leverage=10.0,
        maintenance_margin_rate=0.05,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005
    ),
    # ... 其他 Agent 类型
}

training = TrainingConfig(
    episode_length=1000,
    checkpoint_interval=10,
    neat_config_path="config"
)

demo = DemoConfig(
    host="localhost",
    port=8000,
    tick_interval=100
)

config = Config(
    market=market,
    agents=agents,
    training=training,
    demo=demo
)
```

### 3. 训练时启用鲶鱼

```bash
# 命令行启用鲶鱼（三种模式同时运行）
python scripts/train_noui.py --episodes 100 --catfish

# 指定单模式（已弃用）
python scripts/train_noui.py --episodes 100 --catfish --catfish-mode cycle_swing
```

## 配置参数调优建议

### 市场参数

- `ema_alpha`：
  - 训练时设为 1.0（无平滑）
  - 需要平滑价格时设为 0.1-0.3
  - 值越小，价格变化越平滑，但可能滞后

- `depth`：
  - 增大 depth 会增加输入维度
  - 需要同步修改 NEAT 配置文件中的 `num_inputs`

### Agent 参数

- `initial_balance`：
  - 散户：20万左右
  - 高级散户：与散户相同
  - 庄家：1,000万 - 1亿
  - 做市商：2,000万 - 5,000万

- `leverage`：
  - 散户/高级散户：10-100
  - 庄家：10-20
  - 做市商：10-20

- `maintenance_margin_rate`：
  - 通常设为 `0.5 / leverage` 以确保强平价格合理
  - 做市商必须如此设置

### 训练参数

- `episode_length`：
  - 快速测试：100-500
  - 正常训练：1,000-5,000
  - 长期训练：10,000+

- `parallel_workers`：
  - 根据 CPU 核心数调整
  - 建议 `CPU 核心数 - 2`

### 鲶鱼参数

- `fund_multiplier`：
  - 3.0：中等扰动（默认）
  - 1.0-2.0：轻微扰动
  - 5.0+：强烈扰动

- 模式特定参数：
  - `lookback_period`：10-100，越大越关注长期趋势
  - `trend_threshold`：0.001-0.01，越小越敏感
  - `half_cycle_length`：10-200，摆动周期
  - `ma_period`：20-200，均线周期
  - `deviation_threshold`：0.001-0.01，偏离阈值

## 依赖关系

- `dataclasses` - 数据类支持
- `enum` - 枚举类型支持
- `pathlib` - 路径处理（create_config.py）
- `neat` - NEAT 库（Population 中使用）

## 相关文档

- `../training/CLAUDE.md` - 训练模块，使用配置创建种群和训练器
- `../market/catfish/CLAUDE.md` - 鲶鱼模块详细说明
- `../../config/*.cfg` - NEAT 配置文件
