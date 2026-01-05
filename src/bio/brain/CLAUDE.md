# Brain 模块

## 模块概述

Brain 模块是 NEAT (NeuroEvolution of Augmenting Topologies) 神经网络的封装层，为 AI Agent 提供决策能力。该模块自动使用 Cython 优化的快速前向传播网络（FastFeedForwardNetwork），如果不可用则回退到 neat-python 的原生实现。

Brain 是 Agent 的"大脑"，负责将市场状态输入转换为交易动作输出。通过 NEAT 进化算法，Brain 的网络结构（拓扑）和权重会不断优化，使 Agent 逐渐学会更优的交易策略。

## 文件结构

- `__init__.py` - 模块导出
- `brain.py` - Brain 类定义

## 核心类

### Brain

NEAT 神经网络封装类，提供简洁的前向传播接口。

**类型注解属性：**
- `genome: neat.DefaultGenome` - NEAT 基因组对象，包含网络结构和权重
- `network: FastFeedForwardNetwork | FeedForwardNetwork` - 实际的神经网络实例
- `config: neat.Config` - NEAT 配置对象

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

#### `forward(inputs: list[float] | np.ndarray) -> list[float]`

执行前向传播，将输入向量转换为输出向量。

**参数：**
- `inputs` - 输入向量，可以是 list 或 numpy ndarray
  - 散户：67 个值（10档订单簿 + 10笔成交 + 持仓信息）
  - 高级散户/庄家：607 个值（100档订单簿 + 100笔成交 + 持仓信息）
  - 做市商：634 个值（100档订单簿 + 100笔成交 + 持仓信息 + 多挂单）

**返回：**
- 神经网络输出向量（list of float）
  - 散户/高级散户/庄家：9 个值（动作选择 + 价格偏移 + 数量比例）
  - 做市商：22 个值（双边挂单参数）

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

## 网络类型

### FastFeedForwardNetwork (Cython 优化)

**特点：**
- 使用 Cython 编译，性能显著优于纯 Python 版本
- 前向传播速度快，适合高频交易场景
- 自动激活（如果可用）

**性能提升：**
- 约 10-100 倍速度提升（取决于网络大小）
- 减少 GIL（全局解释器锁）竞争
- 支持并行化决策

### FeedForwardNetwork (纯 Python)

**特点：**
- neat-python 原生实现
- 兼容性好，无需编译
- 作为 FastFeedForwardNetwork 的后备方案

## NEAT 配置

不同 Agent 类型使用不同的 NEAT 配置文件，决定输入输出维度和网络参数：

| Agent 类型 | 配置文件 | 输入节点 | 输出节点 | 初始隐藏节点 |
|-----------|---------|---------|---------|------------|
| 散户 | neat_retail.cfg | 67 | 9 | 3 |
| 高级散户 | neat_retail_pro.cfg | 607 | 9 | 3 |
| 庄家 | neat_whale.cfg | 607 | 9 | 3 |
| 做市商 | neat_market_maker.cfg | 634 | 22 | 3 |

**关键配置参数：**
- `activation_default = tanh` - 激活函数（双曲正切，输出范围 [-1, 1]）
- `aggregation_default = sum` - 聚合函数（求和）
- `feed_forward = True` - 前馈网络（无循环连接）
- `initial_connection = full_nodirect` - 初始全连接（输入到输出，无直接连接）
- `num_hidden = 3` - 初始隐藏节点数量
- `pop_size` - 种群大小（由 AgentConfig.count 动态覆盖）

## 输入输出接口

### 输入向量（由 Agent.observe() 构建）

输入向量是归一化的市场数据，确保神经网络在不同价格尺度下都能稳定学习。

#### 散户输入（67 个值）

| 区间 | 数量 | 说明 | 归一化方法 |
|------|------|------|-----------|
| 0-19 | 20 | 买盘 10 档 | 价格: (p-mid)/mid, 数量: log10(q+1)/10 |
| 20-39 | 20 | 卖盘 10 档 | 价格: (p-mid)/mid, 数量: log10(q+1)/10 |
| 40-49 | 10 | 成交价格 | (p-mid)/mid |
| 50-59 | 10 | 成交数量 | sign(q)·log10(\|q\|+1)/10 |
| 60-63 | 4 | 持仓信息 | 持仓价值、均价、余额、净值 |
| 64-66 | 3 | 挂单信息 | 价格、数量、方向 |

#### 高级散户/庄家输入（607 个值）

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘 100 档 |
| 200-399 | 200 | 卖盘 100 档 |
| 400-499 | 100 | 成交价格 |
| 500-599 | 100 | 成交数量 |
| 600-603 | 4 | 持仓信息 |
| 604-606 | 3 | 挂单信息 |

#### 做市商输入（634 个值）

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘 100 档 |
| 200-399 | 200 | 卖盘 100 档 |
| 400-499 | 100 | 成交价格 |
| 500-599 | 100 | 成交数量 |
| 600-603 | 4 | 持仓信息 |
| 604-633 | 30 | 多挂单信息（5买+5卖，每单3值） |

### 输出向量（由 Agent.decide() 解析）

神经网络输出需要经过解析才能转换为具体的交易动作。

#### 散户/高级散户/庄家输出（9 个值）

| 索引 | 说明 | 值域 | 解析方法 |
|------|------|------|---------|
| 0-6 | 动作得分 | [-∞, +∞] | argmax 选择动作 |
| 7 | 价格偏移 | [-1, 1] | 映射到 tick 偏移 |
| 8 | 数量比例 | [-1, 1] | 映射到 [0.1, 1.0] |

**动作类型（7种）：**
- 0: HOLD（不动）
- 1: PLACE_BID（挂买单）
- 2: PLACE_ASK（挂卖单）
- 3: CANCEL（撤单）
- 4: MARKET_BUY（市价买入）
- 5: MARKET_SELL（市价卖出）
- 6: CLEAR_POSITION（清仓）

#### 做市商输出（22 个值）

| 索引 | 说明 | 值域 | 解析方法 |
|------|------|------|---------|
| 0-4 | 买单价格偏移 | [-1, 1] | 映射到 [1, 100] ticks |
| 5-9 | 买单数量权重 | [-∞, +∞] | softmax 归一化 |
| 10-14 | 卖单价格偏移 | [-1, 1] | 映射到 [1, 100] ticks |
| 15-19 | 卖单数量权重 | [-∞, +∞] | softmax 归一化 |
| 20 | 总下单比例 | [-1, 1] | 映射到 [0, 1] |

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
agent = RetailAgent(agent_id, brain, config)
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

### 内存优化

- **预分配缓冲区**：Agent 使用预分配的 `_input_buffer`，避免每次创建新数组
- **视图复制**：使用 NumPy 切片而非 concatenate，减少内存分配

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

## 注意事项

1. **输入归一化**：必须确保输入数据经过正确的归一化处理，否则神经网络无法学习
2. **输出解析**：神经网络输出需要经过适当的映射才能转换为交易动作
3. **网络进化**：Brain 的网络结构和权重由 NEAT 算法自动进化，无需手动调整
4. **线程安全**：FastFeedForwardNetwork 的前向传播是线程安全的，可以并行调用
5. **类型检查**：使用 `TYPE_CHECKING` 避免循环导入问题
