# Arena 模块

## 模块概述

竞技场模块提供多竞技场并行推理训练功能。核心特性是将多个竞技场的神经网络推理合并成一个批量操作，只有交易配对和账户更新串行执行。

## 文件结构

- `__init__.py` - 模块导出
- `arena_state.py` - 竞技场状态类（AgentAccountState、CatfishAccountState、ArenaState）
- `fitness_aggregator.py` - 适应度汇总器
- `parallel_arena_trainer.py` - 多竞技场并行推理训练器

## 核心类

### AgentAccountState (arena_state.py)

Agent 账户状态类，将 Agent 账户状态与 Agent 对象解耦，每个竞技场维护独立副本。

```python
@dataclass
class AgentAccountState:
    """Agent 账户状态（轻量级，约 200 bytes）"""
    agent_id: int
    agent_type: AgentType
    balance: float
    position_quantity: int
    position_avg_price: float
    realized_pnl: float
    leverage: float
    maintenance_margin_rate: float
    initial_balance: float
    pending_order_id: int | None
    maker_volume: int
    volatility_contribution: float
    is_liquidated: bool
    order_counter: int
    maker_fee_rate: float
    taker_fee_rate: float
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `from_agent(agent)` | 类方法，从 Agent 对象创建状态副本 |
| `reset(config)` | 重置到初始状态 |
| `get_equity(current_price)` | 计算净值（余额 + 浮动盈亏） |
| `get_margin_ratio(current_price)` | 计算保证金率 |
| `check_liquidation(current_price)` | 检查是否需要强平 |
| `on_trade(trade_price, trade_quantity, is_buyer, fee, is_maker)` | 处理成交，返回已实现盈亏 |
| `generate_order_id(arena_id)` | 生成跨竞技场唯一的订单 ID |

### CatfishAccountState (arena_state.py)

鲶鱼账户状态类，包含鲶鱼特有的策略状态。

```python
@dataclass
class CatfishAccountState:
    """鲶鱼账户状态"""
    catfish_id: int
    balance: float
    position_quantity: int
    position_avg_price: float
    realized_pnl: float
    leverage: float
    maintenance_margin_rate: float
    initial_balance: float
    is_liquidated: bool
    order_counter: int
    current_direction: int  # 趋势创造者方向
    ema: float              # 均值回归 EMA
    ema_initialized: bool
    last_action_tick: int
```

### ArenaState (arena_state.py)

单个竞技场的独立状态，封装竞技场运行所需的所有状态。

```python
@dataclass
class ArenaState:
    """单个竞技场的独立状态"""
    arena_id: int
    matching_engine: MatchingEngine
    adl_manager: ADLManager
    agent_states: dict[int, AgentAccountState]
    catfish_states: dict[int, CatfishAccountState]
    recent_trades: deque
    price_history: list[float]
    tick_history_prices: list[float]
    tick_history_volumes: list[float]
    tick_history_amounts: list[float]
    smooth_mid_price: float
    tick: int
    pop_liquidated_counts: dict[AgentType, int]
    eliminating_agents: set[int]
    episode_high_price: float
    episode_low_price: float
    catfish_liquidated: bool
```

### FitnessAggregator (fitness_aggregator.py)

适应度汇总器，用于汇总多个竞技场、多个 episode 的适应度数据。

**方法：**
- `aggregate_simple_average(arena_fitnesses, episode_counts)` - 简单加权平均

**公式：**
```
avg_fitness = sum(arena_fitness) / total_episodes
```

---

### ParallelArenaTrainer (parallel_arena_trainer.py)

多竞技场并行推理训练器，核心特性是将多个竞技场的神经网络推理合并成一个批量操作。

**设计理念：**
1. **神经网络共享**：所有竞技场使用同一套 `BatchNetworkCache`，进化后统一更新
2. **账户状态独立**：每个竞技场维护独立的 `ArenaState`，包含所有 Agent 的账户状态
3. **批量推理合并**：N 个竞技场 × M 个 Agent 的推理合并成单次 OpenMP 并行操作
4. **订单簿独立**：每个竞技场有独立的 `MatchingEngine` 和 `OrderBook`

**核心流程：**
1. 初始化：创建共享种群、N 个独立竞技场状态、共享网络缓存、进化 Worker 池
2. 训练循环：
   a. 重置所有竞技场
   b. 同步推进所有竞技场的 tick（批量推理）
   c. 汇总适应度
   d. 执行 NEAT 进化
   e. 更新网络缓存和 Agent 状态

**类定义：**
```python
@dataclass
class MultiArenaConfig:
    """多竞技场配置"""
    num_arenas: int = 2
    episodes_per_arena: int = 50


class ParallelArenaTrainer:
    """多竞技场并行推理训练器

    Attributes:
        config: 全局配置
        multi_config: 多竞技场配置
        populations: 共享的种群（神经网络）
        arena_states: N 个独立的竞技场状态
        network_caches: 共享的网络缓存
        evolution_worker_pool: 进化 Worker 池
        generation: 当前代数
        total_episodes: 总 episode 数
    """
```

**主要方法：**

| 方法 | 描述 |
|------|------|
| `setup()` | 初始化：创建种群、竞技场状态、网络缓存、进化 Worker 池 |
| `run_round()` | 运行一轮训练（所有竞技场的所有 episode + 进化） |
| `run_tick_all_arenas()` | 并行执行所有竞技场的一个 tick |
| `_batch_inference_all_arenas()` | 批量推理所有竞技场的所有 Agent |
| `_run_episode_all_arenas()` | 运行所有竞技场的一个 episode |
| `_collect_fitness_all_arenas()` | 收集并汇总所有竞技场的适应度 |
| `train(num_rounds, checkpoint_callback, progress_callback)` | 主训练循环 |
| `save_checkpoint(path)` | 保存检查点 |
| `load_checkpoint(path)` | 加载检查点 |
| `stop()` | 停止训练并清理资源 |

**tick 执行流程（run_tick_all_arenas）：**
```python
# 阶段1: 准备（串行）
for arena in arena_states:
    handle_liquidations(arena)  # 强平检查
    market_state = compute_market_state(arena)
    active_agents = get_active_agents(arena)

# 阶段2: 批量推理（并行）- 一次性推理所有竞技场的所有 Agent
all_decisions = _batch_inference_all_arenas(market_states, active_agents)

# 阶段3: 执行（串行）
for arena in arena_states:
    execute_trades(arena, all_decisions[arena.arena_id])
```

**批量推理合并（_batch_inference_all_arenas）：**
- 使用 `AgentStateAdapter` 将 `AgentAccountState` 适配为 Agent-like 接口
- 按 AgentType 分组（跨所有竞技场）
- 使用 `_network_index_map` 获取每个 Agent 在其种群中的网络索引
- 调用 `BatchNetworkCache.decide_multi_arena()` 一次性处理所有竞技场的推理
- 将结果重组为 `dict[arena_id, list[decision]]`

**辅助方法：**
- `_build_network_index_map()` - 构建 Agent ID 到网络索引的映射表
- `_get_network_index(agent_type, agent_id)` - 获取 Agent 在其种群中的网络索引
- `_parse_market_maker_output(agent, output, mid_price, tick_size)` - 解析做市商神经网络输出（41个值）为订单列表

**做市商输出解析（_parse_market_maker_output）：**
神经网络输出结构（共 41 个值），与 `MarketMakerAgent.decide()` 保持一致：
- 输出[0-9]: 买单1-10的价格偏移（-1到1，映射到1-100 ticks）
- 输出[10-19]: 买单1-10的数量权重（-1到1，映射到0-1）
- 输出[20-29]: 卖单1-10的价格偏移（-1到1，映射到1-100 ticks）
- 输出[30-39]: 卖单1-10的数量权重（-1到1，映射到0-1）
- 输出[40]: 总下单比例基准（-1到1，映射到0.01-1）

**使用示例：**
```python
from src.training.arena import ParallelArenaTrainer, MultiArenaConfig

# 方式1：使用上下文管理器
multi_config = MultiArenaConfig(num_arenas=2, episodes_per_arena=50)
with ParallelArenaTrainer(config, multi_config) as trainer:
    trainer.train(
        num_rounds=100,
        checkpoint_callback=lambda gen: print(f"Gen {gen}"),
        progress_callback=lambda stats: print(stats)
    )

# 方式2：手动管理
trainer = ParallelArenaTrainer(config, multi_config)
trainer.setup()
try:
    for _ in range(100):
        stats = trainer.run_round()
        if trainer.generation % 10 == 0:
            trainer.save_checkpoint(f"checkpoints/gen_{trainer.generation}.pkl")
finally:
    trainer.stop()
```

**启动脚本：**
```bash
# 默认训练
python scripts/train_parallel_arena.py --rounds 100

# 自定义参数
python scripts/train_parallel_arena.py --num-arenas 8 --episodes-per-arena 5 --rounds 200

# 从检查点恢复
python scripts/train_parallel_arena.py --resume checkpoints/parallel_arena_gen_50.pkl
```

**检查点格式：**
```python
{
    "generation": int,
    "populations": {
        AgentType.RETAIL: {
            "is_sub_population_manager": True,
            "sub_population_count": int,
            "sub_populations": [
                {"generation": int, "neat_pop": neat.Population},
                ...
            ]
        },
        AgentType.RETAIL_PRO: {
            "generation": int,
            "neat_pop": neat.Population,
        },
        ...
    }
}
```

**性能优化：**
1. **批量推理合并**：N 个竞技场 × M 个 Agent 合并成单次 OpenMP 并行操作
2. **并行进化**：使用 MultiPopulationWorkerPool 在多个进程中并行执行 NEAT 进化
3. **网络参数传输**：只传输网络参数而非完整基因组，减少序列化开销
4. **内存管理**：每轮训练后进行垃圾回收和 malloc_trim
5. **Checkpoint 体积优化**：使用 gzip 压缩保存检查点文件
