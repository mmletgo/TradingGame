# ADL (Auto-Deleveraging) 自动减仓模块

## 模块概述

ADL（Auto-Deleveraging，自动减仓）模块负责在强平订单无法完全成交时，强制选择盈利对手方进行减仓，确保市场风险可控。

## 文件结构

- `__init__.py` - 模块导出
- `adl_manager.py` - ADL 管理器核心实现

## 触发条件

ADL 在以下情况触发：
1. Agent 触发强平（保证金率低于维持保证金率）
2. 强平市价单无法在订单簿中完全成交（流动性不足）
3. 剩余未平仓位需要通过 ADL 机制强制减仓

## 核心类

### ADLCandidate

ADL 候选者信息的数据类。

**属性：**
- `agent: Agent` - Agent 对象
- `position_qty: int` - 持仓数量（正=多头，负=空头）
- `pnl_percent: float` - 盈亏百分比
- `effective_leverage: float` - 有效杠杆
- `adl_score: float` - ADL 排名分数（越高越优先被选中）

### ADLManager

ADL 自动减仓管理器。

**方法：**

#### `calculate_bankruptcy_price(agent, current_price) -> float`
计算破产价格（净值归零时的价格）。

公式：
- 多头：`bankruptcy_price = avg_price - balance / |quantity|`
- 空头：`bankruptcy_price = avg_price + balance / |quantity|`

破产价格下限为 0.01。

#### `calculate_adl_score(agent, current_price) -> ADLCandidate | None`
计算单个 Agent 的 ADL 排名分数。

排名公式：
- 盈利时（PnL% > 0）：`排名 = PnL% * 有效杠杆`
- 亏损时（PnL% <= 0）：`排名 = PnL% / 有效杠杆`

其中：
- `PnL% = 浮动盈亏 / |开仓成本|`
- `有效杠杆 = |持仓市值| / 净值`

无持仓时返回 None。

#### `get_adl_candidates(agents, current_price, target_side, exclude_agent_id) -> list[ADLCandidate]`
获取 ADL 候选列表。

筛选条件：
- 排除被强平的 Agent（exclude_agent_id）
- 排除已淘汰的 Agent（is_liquidated=True）
- 只筛选持有反方向仓位的 Agent

返回按 ADL 分数从高到低排序的候选列表。

#### `execute_adl(liquidated_agent, remaining_qty, candidates, bankruptcy_price, current_price) -> list[tuple[Agent, int, float]]`
执行 ADL 成交。

按候选列表顺序逐个减仓，直到剩余需求为零。返回 ADL 成交列表。

## 排名算法说明

ADL 排名算法的设计目标是优先选择高杠杆高盈利的交易者：

1. **盈利交易者**：盈利越多、杠杆越高，排名越高
   - 这类交易者承担了更高的市场风险，应该首先被减仓

2. **亏损交易者**：亏损越多、杠杆越低，排名越高
   - 亏损交易者的排名总是低于盈利交易者

这种设计确保了：
- 系统风险由高杠杆高盈利者优先承担
- 低杠杆保守交易者受到保护

## 执行流程

1. 强平触发后，撮合引擎尝试用市价单平仓
2. 如果市价单未能完全成交，计算剩余需平仓数量
3. 计算被强平 Agent 的破产价格
4. 获取持有反方向仓位的所有候选者，计算其 ADL 分数
5. 按分数排序，从高到低依次减仓
6. ADL 成交价格为破产价格（不是当前市场价格）

## 与其他模块的关系

### 依赖模块
- `src.bio.agents.base.Agent` - 访问 Agent 的账户和持仓信息
- `src.market.account.Account` - 获取余额、净值等账户信息
- `src.market.account.Position` - 获取持仓数量和均价
- `src.core.log_engine.logger` - 日志记录

### 被依赖模块
- `src.training.trainer.Trainer` - 在强平处理中调用 ADL 机制

## 使用示例

```python
from src.market.adl import ADLManager

# 创建管理器
adl_manager = ADLManager()

# 计算破产价格
bankruptcy_price = adl_manager.calculate_bankruptcy_price(agent, current_price)

# 获取候选列表（被强平者持有空头，需要找多头对手）
candidates = adl_manager.get_adl_candidates(
    agents=all_agents,
    current_price=100.0,
    target_side=1,  # 需要多头对手
    exclude_agent_id=liquidated_agent.agent_id,
)

# 执行 ADL
adl_trades = adl_manager.execute_adl(
    liquidated_agent=liquidated_agent,
    remaining_qty=100,
    candidates=candidates,
    bankruptcy_price=bankruptcy_price,
    current_price=100.0,
)
```

## 注意事项

1. ADL 成交价格是破产价格，不是当前市场价格
2. ADL 成交不经过订单簿撮合，直接在对手方账户上执行减仓
3. ADL 会记录详细日志，便于追踪和调试
4. 如果候选者不足以完全平仓，会记录警告日志
