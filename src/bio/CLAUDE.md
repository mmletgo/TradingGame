# Bio 模块

## 模块概述

Bio 模块是项目的生物系统核心，实现了基于 NEAT (NeuroEvolution of Augmenting Topologies) 神经网络进化的四种 AI 交易 Agent。该模块采用清晰的分层架构，将神经网络、Agent 行为和性能优化有机地结合在一起。

生物系统是连接训练引擎与市场引擎的桥梁：
- **训练引擎** 通过 Population 类从基因组创建和管理 Agent，执行进化算法
- **市场引擎** 提供订单簿、撮合引擎和账户管理，Agent 在其中进行交易
- **生物系统** 封装神经网络决策逻辑，提供观察-决策-执行的完整闭环

**代码统计：**
- Python 代码：约 2,100 行（9 个核心文件）
- Cython 代码：约 492 行（2 个加速模块）
- 总计：约 2,592 行

## 目录结构

```
src/bio/
├── brain/                        # 神经网络封装模块
│   ├── __init__.py               # 模块导出
│   ├── brain.py                  # Brain 类定义（159行）
│   └── CLAUDE.md                 # 神经网络模块详细文档
├── agents/                       # Agent 模块
│   ├── __init__.py               # 模块导出（19行）
│   ├── base.py                   # Agent 基类（542行）
│   ├── retail.py                 # 散户 Agent（175行）
│   ├── retail_pro.py             # 高级散户 Agent（220行）
│   ├── whale.py                  # 庄家 Agent（232行）
│   ├── market_maker.py           # 做市商 Agent（577行）
│   ├── CLAUDE.md                 # Agent 模块详细文档
│   └── _cython/                  # Cython 加速模块
│       ├── __init__.py           # 模块导出
│       ├── fast_observe.pyx      # observe 加速（397行）
│       └── fast_decide.pyx       # decide 辅助加速（95行）
└── CLAUDE.md                     # 本文档（生物系统总览）
```

## 核心组件

### Brain 模块 (brain/)

Brain 是 NEAT 神经网络的封装层，为 AI Agent 提供决策能力。该模块自动使用 Cython 优化的快速前向传播网络（FastFeedForwardNetwork），如果不可用则回退到 neat-python 的原生实现。

**核心类：**

| 类名 | 文件 | 功能 |
|------|------|------|
| Brain | brain.py | NEAT 神经网络封装类，提供前向传播接口 |

**主要方法：**

- `from_genome(genome, config)` - 从基因组创建 Brain 实例的工厂方法
- `forward(inputs)` - 执行前向传播，将输入向量转换为输出向量
- `get_genome()` - 获取 NEAT 基因组对象
- `update_from_network_params(genome, network_params)` - 从预计算的网络参数原地更新 Brain
- `update_network_only(network_params)` - 仅更新网络，不更新基因组引用（延迟反序列化优化）

**性能提升：**
- FastFeedForwardNetwork 使用 Cython 编译，速度提升 10-100 倍
- 减少 GIL（全局解释器锁）竞争
- 支持并行化决策

详见：`src/bio/brain/CLAUDE.md`

### Agents 模块 (agents/)

Agent 模块定义了四种类型的 AI 交易代理：散户、高级散户、庄家、做市商。通过 NEAT 神经网络进化学习交易策略。

**继承结构：**

```
Agent (base.py) - 基类，不包含 decide 方法
├── RetailProAgent (retail_pro.py) - 高级散户，实现散户动作空间的 decide 方法
│   └── RetailAgent (retail.py) - 散户，仅重写 observe 方法（限制可见市场数据）
├── WhaleAgent (whale.py) - 庄家，实现与散户相同的 decide 方法
└── MarketMakerAgent (market_maker.py) - 做市商，实现专用的 decide 方法
```

**核心类：**

| 类名 | 文件 | 功能 |
|------|------|------|
| Agent | base.py | Agent 基类，提供通用属性和方法 |
| RetailProAgent | retail_pro.py | 高级散户 Agent |
| RetailAgent | retail.py | 散户 Agent（继承 RetailProAgent） |
| WhaleAgent | whale.py | 庄家 Agent |
| MarketMakerAgent | market_maker.py | 做市商 Agent |

详见：`src/bio/agents/CLAUDE.md`

## 四种 Agent 类型对比

| 特性 | 散户 | 高级散户 | 庄家 | 做市商 |
|------|------|----------|------|--------|
| 初始资金 | 2万 | 2万 | 3M | 10M |
| 杠杆倍数 | 1.0x | 1.0x | 1.0x | 1.0x |
| 订单簿深度 | 10档 | 100档 | 100档 | 100档 |
| 成交历史 | 10笔 | 100笔 | 100笔 | 100笔 |
| **输入维度** | **127** | **907** | **907** | **964** |
| **输出维度** | **8** | **8** | **8** | **41** |
| 动作空间 | 6种 | 6种 | 6种 | 双边挂单（无动作选择）|
| 同时挂单数 | 1个 | 1个 | 1个 | 20个（买卖各10个）|
| 撤单再挂 | 是 | 是 | 是 | 是（每tick全撤全挂）|

**约束规则：**
- 散户/高级散户/庄家：同时只能挂一单
- 做市商：每 tick 必须双边挂单，先撤旧单再挂新单
- 所有操作在最新价 ±100 个最小变动单位内

## 神经网络输入输出规范

### 输入向量（由 Agent.observe() 构建）

输入向量是归一化的市场数据，确保神经网络在不同价格尺度下都能稳定学习。

**归一化方法：**

| 数据类型 | 归一化公式 | 数值范围 | 说明 |
|---------|-----------|---------|------|
| 订单簿价格 | `(price - mid_price) / mid_price` | ≈ [-0.1, 0.1] | 相对中间价的价格偏移 |
| 订单簿数量 | `log10(quantity + 1) / 10` | ≈ [0, 1] | 对数归一化，1e10 → 1.0 |
| 成交价格 | `(price - mid_price) / mid_price` | ≈ [-0.1, 0.1] | 相对中间价的价格偏移 |
| 成交数量 | `sign(qty) * log10(\|qty\| + 1) / 10` | ≈ [-1, 1] | 带方向的对数归一化 |
| 持仓价值 | `position_value / (equity * leverage)` | [0, 1] | 相对可用杠杆的比例 |
| 持仓均价 | `(avg_price - mid_price) / mid_price` | ≈ [-0.1, 0.1] | 相对中间价的价格偏移 |
| 余额 | `balance / initial_balance` | [0, +∞) | 相对初始余额的比例 |
| 净值 | `equity / initial_balance` | [0, +∞) | 相对初始余额的比例 |
| 挂单数量 | `log10(quantity + 1) / 10` | ≈ [0, 1] | 对数归一化 |

### 散户神经网络输入（127 个值）

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-19 | 20 | 买盘 10 档（价格归一化 + 数量）|
| 20-39 | 20 | 卖盘 10 档（价格归一化 + 数量）|
| 40-49 | 10 | 成交价格归一化 |
| 50-59 | 10 | 成交数量（正=taker买入，负=taker卖出）|
| 60-63 | 4 | 持仓信息 |
| 64-66 | 3 | 挂单信息（价格归一化、数量、方向）|
| 67-86 | 20 | tick 历史价格（最近 20 个）|
| 87-106 | 20 | tick 历史成交量（最近 20 个）|
| 107-126 | 20 | tick 历史成交额（最近 20 个）|

### 高级散户/庄家神经网络输入（907 个值）

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘 100 档（价格归一化 + 数量）|
| 200-399 | 200 | 卖盘 100 档（价格归一化 + 数量）|
| 400-499 | 100 | 成交价格归一化 |
| 500-599 | 100 | 成交数量（正=taker买入，负=taker卖出）|
| 600-603 | 4 | 持仓信息 |
| 604-606 | 3 | 挂单信息（价格归一化、数量、方向）|
| 607-706 | 100 | tick 历史价格（最近 100 个）|
| 707-806 | 100 | tick 历史成交量（最近 100 个）|
| 807-906 | 100 | tick 历史成交额（最近 100 个）|

### 做市商神经网络输入（964 个值）

| 区间 | 数量 | 说明 |
|------|------|------|
| 0-199 | 200 | 买盘 100 档（价格归一化 + 数量）|
| 200-399 | 200 | 卖盘 100 档（价格归一化 + 数量）|
| 400-499 | 100 | 成交价格归一化 |
| 500-599 | 100 | 成交数量（正=taker买入，负=taker卖出）|
| 600-603 | 4 | 持仓信息 |
| 604-663 | **60** | 挂单信息（10买单 + 10卖单，每单 3 个值）|
| 664-763 | 100 | tick 历史价格（最近 100 个）|
| 764-863 | 100 | tick 历史成交量（最近 100 个）|
| 864-963 | 100 | tick 历史成交额（最近 100 个）|

### 散户/高级散户/庄家神经网络输出（8 个值）

| 索引 | 说明 | 值域 | 解析方法 |
|------|------|------|---------|
| 0-5 | 动作得分 | [-∞, +∞] | argmax 选择动作 |
| 6 | 价格偏移 | [-1, 1] | 映射到 ±100 ticks |
| 7 | 数量比例 | [-1, 1] | 映射到 [0.1, 1.0] |

**动作类型（6 种）：**
- 0: HOLD（不动）
- 1: PLACE_BID（挂买单）
- 2: PLACE_ASK（挂卖单）
- 3: CANCEL（撤单）
- 4: MARKET_BUY（市价买入）
- 5: MARKET_SELL（市价卖出）

### 做市商神经网络输出（41 个值）

| 索引 | 说明 | 值域 | 解析方法 |
|------|------|------|---------|
| 0-9 | 买单价格偏移 | [-1, 1] | 映射到 [1, 100] ticks |
| 10-19 | 买单数量权重 | [-∞, +∞] | softmax 归一化 |
| 20-29 | 卖单价格偏移 | [-1, 1] | 映射到 [1, 100] ticks |
| 30-39 | 卖单数量权重 | [-∞, +∞] | softmax 归一化 |
| 40 | 总下单比例 | [-1, 1] | 映射到 [0.01, 1.0] |

**做市商默认行为：**
做市商每 tick 必然双边挂单（每边 1-10 单），无需动作选择。神经网络直接输出价格和数量参数。

## Cython 加速模块

### fast_observe.pyx

提供 `observe()` 方法的 Cython 加速实现，用于构建神经网络输入向量。每次调用仅需 1-2 微秒。

**核心函数：**

| 函数名 | 功能 |
|--------|------|
| `fast_observe_retail()` | 构建散户的神经网络输入向量（127 维）|
| `fast_observe_full()` | 构建高级散户/庄家的神经网络输入向量（907 维）|
| `fast_observe_market_maker()` | 构建做市商的神经网络输入向量（964 维）|
| `get_position_inputs()` | 计算持仓信息输入（4 个值）|
| `get_pending_order_inputs()` | 计算挂单信息输入（3 个值）|

**优化点：**
- 使用 `nogil` 释放 GIL，支持多线程并行
- 直接操作 NumPy 数组，避免 Python 开销

### fast_decide.pyx

提供决策辅助函数的 Cython 加速实现。

**核心函数：**

| 函数名 | 功能 |
|--------|------|
| `fast_argmax()` | 在指定范围内查找最大值索引 |
| `fast_round_price()` | 将价格取整到 tick_size 的整数倍 |
| `fast_clip()` | 将值裁剪到指定范围 |

## 性能优化要点

### 预分配缓冲区优化

所有 Agent 都使用预分配的 NumPy 数组作为输入缓冲区，避免在每次 observe 时创建新数组。

| Agent 类型 | 缓冲区大小 |
|-----------|-----------|
| 散户 | 127 维 |
| 高级散户/庄家 | 907 维 |
| 做市商 | 964 维 |

### Cython 加速

- **FastFeedForwardNetwork**：前向传播速度提升 10-100 倍
- **fast_observe 函数**：每次调用仅需 1-2 微秒
- **并行决策**：Cython 代码释放 GIL，支持多线程并行

### 订单 ID 生成

使用高效的位组合方式生成唯一订单 ID：
```python
(agent_id << 32) | _order_counter
```
- agent_id 占高 32 位
- 计数器占低 32 位
- 比 MD5 哈希更高效，适合高频交易场景

## 依赖关系

### 上游依赖

- `neat` - NEAT 进化算法库（neat-python 或 fork 版本）
  - `neat.DefaultGenome` - 基因组类
  - `neat.Config` - 配置类
  - `neat.nn.FastFeedForwardNetwork` - Cython 优化网络
  - `neat.nn.FeedForwardNetwork` - 原生网络
- `numpy` - 数值计算库（用于输入输出数组）

### 下游依赖

- `src.training.population` - 种群管理（从基因组创建 Brain 和 Agent）
- `src.training.trainer` - 训练器（管理 Agent 的生命周期和决策）

### 内部依赖

- `src.config.config` - 配置类（AgentConfig, AgentType）
- `src.market.account` - 账户管理（Account）
- `src.market.matching` - 撮合引擎和成交记录（MatchingEngine, Trade）
- `src.market.market_state` - 市场状态数据（NormalizedMarketState）
- `src.market.orderbook` - 订单簿（OrderBook, Order, OrderSide, OrderType）

## 与训练引擎的交互

### Agent 创建流程

训练引擎通过 `Population.create_agents()` 从基因组创建 Agent：

```python
# src/training/population.py
def create_agents(self, genomes: list[tuple[int, neat.DefaultGenome]]) -> list[Agent]:
    # 1. 根据 agent_type 选择 Agent 类
    if self.agent_type == AgentType.RETAIL:
        agent_class = RetailAgent
    elif self.agent_type == AgentType.RETAIL_PRO:
        agent_class = RetailProAgent
    elif self.agent_type == AgentType.WHALE:
        agent_class = WhaleAgent
    elif self.agent_type == AgentType.MARKET_MAKER:
        agent_class = MarketMakerAgent

    # 2. 为每个基因组创建 Brain 和 Agent
    for idx, (genome_id, genome) in enumerate(genomes):
        brain = Brain.from_genome(genome, self.neat_config)
        agent = agent_class(agent_id, brain, self.agent_config)
        agents.append(agent)

    return agents
```

### 进化更新流程

进化完成后，训练引擎通过 `Agent.update_brain()` 原地更新 Agent 的 Brain，避免重建对象：

```python
# src/training/population.py
def set_genomes(self, genomes: list[tuple[int, neat.DefaultGenome]]) -> None:
    """原地更新 Agent 的 genome，复用 Agent 对象"""
    for agent, (_, new_genome) in zip(self.agents, genomes):
        agent.update_brain(new_genome, self.neat_config)
```

### 适应度评估流程

训练引擎通过 `Population.evaluate()` 计算 Agent 适应度并设置到基因组：

```python
# src/training/population.py
def evaluate(self, current_price: float, market_avg_return: float = 0.0) -> list[tuple[Agent, float]]:
    # 1. 计算 Agent 的 PnL 和收益率
    # 2. 计算相对收益率（Agent 收益率 - 市场平均收益率）
    # 3. 按适应度排序
    for agent, fitness in agent_fitnesses:
        genome = agent.brain.get_genome()
        genome.fitness = fitness

    return sorted(agent_fitnesses, key=lambda x: x[1], reverse=True)
```

## 与市场引擎的交互

### Agent 决策流程

在每个 tick 中，Agent 通过观察-决策-执行循环与市场交互：

```python
# 1. 观察市场，构建输入
inputs = agent.observe(market_state, orderbook)

# 2. 前向传播，获取输出
outputs = agent.brain.forward(inputs)

# 3. 解析输出，执行动作
action, params = agent.decide(outputs, mid_price, tick_size)
trades = agent.execute_action(action, params, matching_engine)
```

### 订单执行流程

Agent 通过 `execute_action()` 方法与市场引擎交互：

```python
# src/bio/agents/base.py
def execute_action(self, action: ActionType, params: dict, matching_engine: MatchingEngine) -> list[Trade]:
    # 1. 根据动作类型创建订单
    if action == ActionType.PLACE_BID:
        order = Order(side=OrderSide.BID, price=price, quantity=quantity)
    elif action == ActionType.MARKET_BUY:
        order = Order(side=OrderSide.BID, order_type=OrderType.MARKET, quantity=quantity)

    # 2. 提交到撮合引擎
    trades = matching_engine.process_order(order)

    # 3. 处理成交，更新账户
    self._process_trades(trades)

    return trades
```

### 账户更新流程

成交后通过 `account.on_trade()` 更新账户状态：

```python
# src/bio/agents/base.py
def _process_trades(self, trades: list[Trade]) -> None:
    for trade in trades:
        # 判断自己是 taker 还是 maker
        if trade.is_buyer_taker:
            self.account.on_trade(
                side=OrderSide.BID,
                price=trade.price,
                quantity=trade.quantity,
                is_taker=True,
            )
```

## NEAT 配置文件

| 配置文件 | Agent类型 | 输入节点 | 输出节点 | 初始隐藏节点 | 种群大小 |
|---------|----------|----------|----------|-------------|---------|
| neat_retail.cfg | 散户 | 127 | 8 | 3 | 150 |
| neat_retail_pro.cfg | 高级散户 | 907 | 8 | 10 | 100 |
| neat_whale.cfg | 庄家 | 907 | 8 | 10 | 200 |
| neat_market_maker.cfg | 做市商 | 964 | 41 | 10 | 150 |

### 关键配置参数

**神经网络结构：**
- `activation_default = tanh` - 激活函数（双曲正切，输出范围 [-1, 1]）
- `aggregation_default = sum` - 聚合函数（求和）
- `feed_forward = True` - 前馈网络（无循环连接）
- `initial_connection = partial_direct 0.7` - 初始部分直接连接（70% 连接率）
- `num_hidden = 3/10` - 初始隐藏节点数量

**进化参数：**
- `pop_size` - 种群大小
- `compatibility_threshold` - 物种兼容性阈值（1.8-2.8）
- `max_stagnation = 20` - 最大停滞代数
- `elitism = 1` - 精英保留数量
- `survival_threshold = 0.5` - 生存阈值（前 50%）

**突变参数：**
- `weight_init_stdev = 0.3/1.0` - 权重初始化标准差
- `weight_mutate_rate = 0.8` - 权重突变率
- `conn_add_prob = 0.5` - 连接添加概率
- `node_add_prob = 0.2` - 节点添加概率

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

### 继承复用

- RetailAgent 继承 RetailProAgent，仅重写 observe 方法
- RetailProAgent、WhaleAgent、MarketMakerAgent 共享相同的基类
- 代码复用率高，易于维护

### 性能优先

- 预分配缓冲区避免频繁内存分配
- Cython 加速关键路径
- 原地更新对象减少 GC 压力
- 并行创建 Agent 提升启动速度

## 相关文档

- `src/bio/brain/CLAUDE.md` - 神经网络模块详细文档
- `src/bio/agents/CLAUDE.md` - Agent 模块详细文档
- `src/config/config/CLAUDE.md` - 配置管理文档
- `src/training/CLAUDE.md` - 训练引擎文档
- `src/market/CLAUDE.md` - 市场引擎文档
