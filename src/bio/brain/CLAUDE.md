# Brain 模块

## 模块概述

Brain 模块是 NEAT (NeuroEvolution of Augmenting Topologies) 神经网络的封装层，为 AI Agent 提供决策能力。该模块自动使用 Cython 优化的快速前向传播网络（FastFeedForwardNetwork），如果不可用则回退到 neat-python 的原生实现。

Brain 是 Agent 的"大脑"，负责将市场状态输入转换为交易动作输出。通过 NEAT 进化算法，Brain 的网络结构（拓扑）和权重会不断优化，使 Agent 逐渐学会更优的交易策略。

**代码统计：**
- `brain.py`: 158 行

## 文件结构

```
src/bio/brain/
├── __init__.py       # 模块导出
├── brain.py          # Brain 类定义（158行）
└── CLAUDE.md         # 本文档
```

## 核心类

### Brain

NEAT 神经网络封装类，提供简洁的前向传播接口。

**类型注解属性：**
```python
genome: neat.DefaultGenome                     # NEAT 基因组对象，包含网络结构和权重
network: FastFeedForwardNetwork | FeedForwardNetwork  # 实际的神经网络实例
config: neat.Config                            # NEAT 配置对象
```

**关键方法：**

#### `from_genome(genome: neat.DefaultGenome, config: neat.Config) -> Brain` (类方法)

从基因组创建 Brain 实例的工厂方法。

**参数：**
- `genome` - NEAT 基因组对象
- `config` - NEAT 配置对象

**返回：**
- Brain 实例

**使用示例：**
```python
brain = Brain.from_genome(genome, neat_config)
outputs = brain.forward(inputs)
```

#### `__init__(genome: neat.DefaultGenome, config: neat.Config) -> None`

构造函数，创建神经网络封装。

**参数：**
- `genome` - NEAT 基因组对象
- `config` - NEAT 配置对象

**实现细节：**
- 自动使用 `FastFeedForwardNetwork.create()` 创建网络（Cython 优化版本）
- 如果 Cython 版本不可用，会回退到纯 Python 版本

#### `forward(inputs: list[float] | np.ndarray) -> np.ndarray`

执行前向传播，将输入向量转换为输出向量。

**参数：**
- `inputs` - 输入向量，可以是 list 或 numpy ndarray
  - 高级散户：67 个值
  - 做市商：132 个值

**返回：**
- 神经网络输出向量（numpy ndarray）
  - 高级散户：3 个值
  - 做市商：43 个值

**使用示例：**
```python
# Agent 决策流程
inputs = agent.observe(market_state, orderbook)  # 构建输入
outputs = agent.brain.forward(inputs)            # 前向传播
action = agent.decide(outputs, mid_price)        # 解析输出为动作
```

#### `get_genome() -> neat.DefaultGenome`

获取 NEAT 基因组对象。

**返回：**
- NEAT 基因组对象

**使用场景：**
- 进化评估时设置适应度：`genome.fitness = fitness_value`
- 序列化保存网络结构
- 分析进化过程中的网络变化

#### `update_from_genome(genome: neat.DefaultGenome, config: neat.Config) -> None`

原地更新 genome 和 network，避免重建对象。

**参数：**
- `genome` - 新的 NEAT 基因组对象
- `config` - NEAT 配置对象

**使用场景：**
- 进化后更新 Brain，复用原有对象
- 减少对象创建销毁开销

#### `update_from_network_params(genome: neat.DefaultGenome, network_params: dict[str, np.ndarray | int]) -> None`

从预计算的网络参数原地更新 Brain，跳过基因组解析。

**参数：**
- `genome` - 新的 NEAT 基因组对象（用于 get_genome() 返回）
- `network_params` - 网络参数字典，包含：
  - `num_inputs`: int - 输入节点数
  - `num_outputs`: int - 输出节点数
  - `input_keys`: ndarray[int32] - 输入节点 ID
  - `output_keys`: ndarray[int32] - 输出节点 ID
  - `num_nodes`: int - 隐藏+输出节点数
  - `node_ids`: ndarray[int32] - 节点 ID
  - `biases`: ndarray[float32] - 偏置
  - `responses`: ndarray[float32] - 响应
  - `act_types`: ndarray[int32] - 激活函数类型
  - `conn_indptr`: ndarray[int32] - CSR 连接指针
  - `conn_sources`: ndarray[int32] - 连接源节点
  - `conn_weights`: ndarray[float32] - 连接权重
  - `output_indices`: ndarray[int32] - 输出节点索引

**使用场景：**
- 并行进化后快速更新网络，避免从基因组重建网络的开销
- Worker 进程直接返回网络参数，主进程使用此方法创建网络

**性能提升：**
- 从基因组重建网络：需要遍历 nodes/connections 字典，构建拓扑
- 从网络参数创建：直接使用预计算的数组，显著提升速度

#### `update_network_only(network_params: dict[str, np.ndarray | int]) -> None`

仅更新网络，不更新基因组引用。

**参数：**
- `network_params` - 网络参数字典

**使用场景：**
- 延迟反序列化优化：进化时只更新网络用于决策
- 基因组引用保持不变（指向旧的基因组对象）
- 检查点保存时再反序列化并更新基因组

## NEAT 网络前向传播机制

### FastFeedForwardNetwork (Cython 优化)

**特点：**
- 使用 Cython 编译，性能显著优于纯 Python 版本
- 前向传播速度快，适合高频交易场景
- 自动激活（如果可用）

**性能提升：**
- 约 10-100 倍速度提升（取决于网络大小）
- 减少 GIL（全局解释器锁）竞争
- 支持并行化决策

**创建方法：**
1. `FastFeedForwardNetwork.create(genome, config)` - 从基因组创建
2. `FastFeedForwardNetwork.create_from_params(...)` - 从网络参数创建（更快）

### FeedForwardNetwork (纯 Python)

**特点：**
- neat-python 原生实现
- 兼容性好，无需编译
- 作为 FastFeedForwardNetwork 的后备方案

## NEAT 配置文件

不同 Agent 类型使用不同的 NEAT 配置文件，决定输入输出维度和网络参数：

| Agent 类型 | 配置文件 | 输入节点 | 输出节点 | 初始隐藏节点 |
|-----------|---------|---------|---------|------------|
| 高级散户 | neat_retail_pro.cfg | 67 | 3 | 10 |
| 做市商 | neat_market_maker.cfg | 132 | 43 | 10 |

**关键配置参数：**
- `activation_default = tanh` - 激活函数（双曲正切，输出范围 [-1, 1]）
- `aggregation_default = sum` - 聚合函数（求和）
- `feed_forward = True` - 前馈网络（无循环连接）
- `initial_connection = partial_direct 0.7` - 初始部分直接连接（70% 连接率）
- `num_hidden` - 初始隐藏节点数量

## 输入输出接口规范

### 输入向量（由 Agent.observe() 构建）

输入向量是归一化的市场数据，确保神经网络在不同价格尺度下都能稳定学习。

**归一化方法：**

| 数据类型 | 归一化公式 | 数值范围 | 说明 |
|---------|-----------|---------|------|
| 订单簿价格 | `(price - mid_price) / mid_price` | [-0.1, 0.1] | 相对中间价的价格偏移 |
| 订单簿数量 | `log10(quantity + 1) / 10` | [0, 1] | 对数归一化，1e10 -> 1.0 |
| 成交价格 | `(price - mid_price) / mid_price` | [-0.1, 0.1] | 相对中间价的价格偏移 |
| 成交数量 | `sign(qty) * log10(\|qty\| + 1) / 10` | [-1, 1] | 带方向的对数归一化 |
| 持仓价值 | `position_value / (equity * leverage)` | [0, 1] | 相对可用杠杆的比例 |
| 持仓均价 | `(avg_price - mid_price) / mid_price` | [-0.1, 0.1] | 相对中间价的价格偏移 |
| 余额 | `balance / initial_balance` | [0, +inf) | 相对初始余额的比例 |
| 净值 | `equity / initial_balance` | [0, +inf) | 相对初始余额的比例 |
| 挂单数量 | `log10(quantity + 1) / 10` | [0, 1] | 对数归一化 |

### 高级散户神经网络输入（67 个值）

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-9 | 10 | 买盘 5 档（价格归一化 + 数量）|
| 10-19 | 10 | 卖盘 5 档（价格归一化 + 数量）|
| 20-23 | 4 | 持仓信息 |
| 24-26 | 3 | 挂单信息（价格归一化、数量、方向）|
| 27-46 | 20 | tick 历史价格（最近 20 个）|
| 47-66 | 20 | tick 历史成交量（最近 20 个）|

### 做市商神经网络输入（132 个值）

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-9 | 10 | 买盘 5 档（价格归一化 + 数量）|
| 10-19 | 10 | 卖盘 5 档（价格归一化 + 数量）|
| 20-23 | 4 | 持仓信息 |
| 24-83 | **60** | 挂单信息（10买单 + 10卖单，每单 3 个值）|
| 84-103 | 20 | tick 历史价格（最近 20 个）|
| 104-123 | 20 | tick 历史成交量（最近 20 个）|
| 124-131 | 8 | AS 模型特征 |

### 高级散户神经网络输出（3 个值）

| 索引 | 说明 | 值域 | 解析方法 |
|------|------|------|---------|
| 0 | 动作选择 | [-1, 1] | 等宽分 6 bin 选择动作 |
| 1 | 价格偏移 | [-1, 1] | 映射到 +/-100 ticks |
| 2 | 数量比例 | [-1, 1] | 映射到 [0, 1.0] |

**动作类型（6 种）：**
- 0: HOLD（不动）
- 1: PLACE_BID（挂买单）
- 2: PLACE_ASK（挂卖单）
- 3: CANCEL（撤单）
- 4: MARKET_BUY（市价买入）
- 5: MARKET_SELL（市价卖出）

### 做市商神经网络输出（43 个值）

| 索引 | 说明 | 值域 | 解析方法 |
|------|------|------|---------|
| 0-9 | 买单价格偏移 | [-1, 1] | 映射到 [1, 100] ticks（相对 reservation_price） |
| 10-19 | 买单数量权重 | [-1, 1] | 映射到 [0, 1]，归一化 |
| 20-29 | 卖单价格偏移 | [-1, 1] | 映射到 [1, 100] ticks（相对 reservation_price） |
| 30-39 | 卖单数量权重 | [-1, 1] | 映射到 [0, 1]，归一化 |
| 40 | 总下单比例基准 | [-1, 1] | 映射到 [0.01, 1.0] |
| 41 | gamma_adjustment | [-1, 1] | 映射到 [0.1, 10.0]，缩放 AS gamma |
| 42 | spread_adjustment | [-1, 1] | 映射到 [0.5, 2.0]，缩放 AS 最优点差 |

## 依赖关系

### 上游依赖

- `neat` - NEAT 进化算法库（neat-python 或 fork 版本）
  - `neat.DefaultGenome` - 基因组类
  - `neat.Config` - 配置类
  - `neat.nn.FastFeedForwardNetwork` - Cython 优化网络
  - `neat.nn.FeedForwardNetwork` - 原生网络
- `numpy` - 数值计算库（用于输入输出数组）

### 下游依赖

- `src.bio.agents.base` - Agent 基类（使用 Brain 做决策）
- `src.training.population` - 种群管理（从基因组创建 Brain）

## 使用流程

### 1. 创建 Brain（由 Population 自动完成）

```python
# Population.create_agents() 中
brain = Brain.from_genome(genome, neat_config)
agent = RetailProAgent(agent_id, brain, config)
```

### 2. Agent 决策流程

```python
# 1. 观察市场，构建输入
inputs = agent.observe(market_state, orderbook)

# 2. 前向传播，获取输出
outputs = agent.brain.forward(inputs)

# 3. 解析输出，执行动作
action, params = agent.decide(outputs, mid_price, tick_size)
trades = agent.execute_action(action, params, matching_engine)
```

### 3. 进化评估

```python
# Population.evolve() 中
agent_fitnesses = population.evaluate(current_price)

# 设置基因组适应度
for agent, fitness in agent_fitnesses:
    genome = agent.brain.get_genome()
    genome.fitness = fitness

# NEAT 进化
neat_pop.run(eval_genomes, n=1)
```

## 性能优化

### Cython 加速

- **FastFeedForwardNetwork**：使用 Cython 编译，前向传播速度提升 10-100 倍
- **并行决策**：Cython 代码释放 GIL，支持多线程并行决策
- **零拷贝**：输入输出使用 NumPy 数组，避免 Python list 转换开销

### 网络创建优化

- **从参数创建**：`update_from_network_params()` 直接使用预计算数组，避免基因组解析
- **延迟反序列化**：`update_network_only()` 在进化时只更新网络，基因组延迟更新

### 内存优化

- **预分配缓冲区**：Agent 使用预分配的 `_input_buffer`，避免每次创建新数组
- **原地更新**：使用 `update_from_*` 方法复用 Brain 对象

## 设计原则

### 简洁封装

- Brain 只负责前向传播，不涉及动作解析
- 动作解析由 Agent 子类的 `decide()` 方法完成
- 保持 Brain 与交易逻辑解耦

### 自动优化

- 优先使用 Cython 优化版本
- 自动回退到兼容版本
- 对用户透明，无需手动选择

### 灵活配置

- 支持不同输入输出维度
- 通过 NEAT 配置文件灵活调整
- 支持动态种群大小

### 性能优先

- 提供多种更新方法，适应不同场景
- 支持并行决策和批量推理
- 优化内存分配和对象创建

## 注意事项

1. **输入归一化**：必须确保输入数据经过正确的归一化处理，否则神经网络无法学习
2. **输出解析**：神经网络输出需要经过适当的映射才能转换为交易动作
3. **网络进化**：Brain 的网络结构和权重由 NEAT 算法自动进化，无需手动调整
4. **线程安全**：FastFeedForwardNetwork 的前向传播是线程安全的，可以并行调用
5. **对象复用**：进化时优先使用 `update_from_*` 方法复用 Brain 对象
6. **参数更新**：`update_from_network_params()` 适用于并行进化场景，需要预计算参数
