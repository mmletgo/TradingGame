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

#### `get_adl_price(current_price) -> float`
获取 ADL 成交价格。

**设计原则**：ADL 直接使用当前市场价格成交。

强平 ≠ 破产：被强平时 Agent 可能还有正的净值（只是保证金率过低），因此不使用"破产价格"来计算 ADL 成交价。

使用当前市场价格的好处：
- 简单公平：双方都以市场价成交
- 避免异常：不会因为穿仓导致价格异常
- 符合直觉：流动性不足时强制以当前价成交

#### `calculate_adl_score(agent, current_price) -> ADLCandidate | None`
计算单个 Agent 的 ADL 排名分数。

排名公式：
- 盈利时（PnL% > 0）：`排名 = PnL% * 有效杠杆`
- 亏损时（PnL% <= 0）：`排名 = PnL% / 有效杠杆`

其中：
- `PnL% = 浮动盈亏 / |开仓成本|`
- `有效杠杆 = |持仓市值| / 净值`

无持仓时返回 None。

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

1. **Tick 开始时预计算**：Trainer 遍历所有 Agent，筛选盈利的候选者，计算 ADL 分数，按多头/空头分类并排序
2. 强平触发后，撮合引擎尝试用市价单平仓
3. 如果市价单未能完全成交，计算剩余需平仓数量
4. 在 Trainer 的 `_execute_adl()` 中直接处理：
   - 选择对应方向的预计算候选清单（被强平方是多头则使用空头候选，反之使用多头候选）
   - 按排序顺序依次减仓，同时更新候选清单中的 `position_qty` 防止重复使用
   - 通过 `account.on_adl_trade()` 更新账户
5. **ADL 成交价格为当前市场价格**（简单公平，双方都以市价成交）

**注**：由于只有盈利的 Agent 才能作为 ADL 对手方，当所有对手方都亏损时（如特殊入场价导致），ADL 可能无法匹配。此时由系统兜底处理，强制清零被淘汰者的仓位。

## 与其他模块的关系

### 依赖模块
- `src.bio.agents.base.Agent` - 访问 Agent 的账户和持仓信息
- `src.market.account.Account` - 获取余额、净值等账户信息
- `src.market.account.Position` - 获取持仓数量和均价
- `src.core.log_engine.logger` - 日志记录

### 被依赖模块
- `src.training.trainer.Trainer` - 在强平处理中调用 ADL 机制

## 使用示例

Trainer 在 tick 开始时预计算 ADL 候选清单（`_adl_long_candidates` 和 `_adl_short_candidates`）：

```python
from src.market.adl import ADLManager

# 创建管理器
adl_manager = ADLManager()

# 预计算候选列表（在 tick 开始时执行）
long_candidates = []
short_candidates = []
for agent in agents:
    candidate = adl_manager.calculate_adl_score(agent, current_price)
    if candidate and candidate.pnl_percent > 0:
        if candidate.position_qty > 0:
            long_candidates.append(candidate)
        else:
            short_candidates.append(candidate)

# 排序
long_candidates.sort(key=lambda c: c.adl_score, reverse=True)
short_candidates.sort(key=lambda c: c.adl_score, reverse=True)

# 然后在 Trainer 的 _execute_adl 中直接使用这些预计算的候选清单，
# 直接在循环中处理 ADL 成交（更新账户、更新 position_qty）
```

## 注意事项

1. **ADL 成交价格是当前市场价格**，简单公平
2. ADL 成交不经过订单簿撮合，直接在对手方账户上执行减仓
3. ADL 执行逻辑已完全移到 Trainer 的 `_execute_adl()` 中
4. **ADL 候选清单在 Trainer 的 tick 开始时预计算**（`calculate_adl_score()`），避免重复计算 ADL 分数
5. **候选清单中的 position_qty 会被动态更新**，防止多次 ADL 中的同一候选被重复使用
6. 由于多空对等，理论上不会出现候选不足的情况；如果出现则说明有其他 bug
7. ADL 会记录详细日志，便于追踪和调试
