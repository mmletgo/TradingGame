# ADL (Auto-Deleveraging) 自动减仓模块

## 模块概述

ADL（Auto-Deleveraging，自动减仓）模块是交易系统的风险控制组件，负责在强平订单无法完全成交时，强制选择盈利对手方进行减仓，确保市场风险可控。

## 核心职责

1. **候选排名计算**：根据盈亏百分比和有效杠杆计算 ADL 排名分数
2. **成交价格确定**：提供 ADL 成交价格（直接使用当前市场价格）
3. **支持 Agent 和噪声交易者**：噪声交易者也可以作为 ADL 候选者参与减仓

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

ADL 候选者信息的数据类，封装了参与 ADL 减仓的市场参与者相关信息。

**属性：**
- `participant: Union[Agent, NoiseTrader]` - 参与者对象，支持 Agent 或噪声交易者
- `position_qty: int` - 持仓数量（正数=多头，负数=空头）
- `pnl_percent: float` - 盈亏百分比
- `effective_leverage: float` - 有效杠杆
- `adl_score: float` - ADL 排名分数（越高越优先被选中）

**属性方法：**
- `agent: Agent` - 兼容旧代码的属性访问器，仅当 participant 是 Agent 时可用
- `is_noise_trader: bool` - 检查候选者是否为噪声交易者

### ADLManager

ADL 自动减仓管理器，负责计算 ADL 排名分数和确定成交价格。

**初始化：**
```python
def __init__(self) -> None:
    from src.core.log_engine.logger import get_logger
    self.logger = get_logger("adl")
```

**核心方法：**

#### `get_adl_price(current_price) -> float`

获取 ADL 成交价格。

**设计原则**：ADL 直接使用当前市场价格成交。

**参数：**
- `current_price: float` - 当前市场价格

**返回值：**
- `float` - ADL 成交价格（即当前市场价格）

**设计理由：**
- 强平 ≠ 破产：被强平时 Agent 可能还有正的净值（只是保证金率过低）
- 不应该用"破产价格"来计算 ADL 成交价
- 使用当前市场价格的好处：
  - 简单公平：双方都以市场价成交
  - 避免异常：不会因为穿仓导致价格异常
  - 符合直觉：流动性不足时强制以当前价成交

#### `calculate_adl_score(agent, current_price) -> ADLCandidate | None`

计算单个 Agent 的 ADL 排名分数。

**参数：**
- `agent: Agent` - Agent 对象
- `current_price: float` - 当前市场价格

**返回值：**
- `ADLCandidate` - 包含排名信息的候选对象
- `None` - Agent 无持仓时返回 None

**排名公式：**

| 情况 | 公式 |
|------|------|
| 盈利时（PnL% > 0） | `adl_score = PnL% × 有效杠杆` |
| 亏损时（PnL% <= 0） | `adl_score = PnL% / 有效杠杆` |

**计算步骤：**

1. **获取持仓信息**：
   - `quantity = position.quantity` - 持仓数量
   - `avg_price = position.avg_price` - 平均开仓价格
   - 无持仓时直接返回 None

2. **计算开仓成本**：
   ```
   entry_cost = |quantity| × avg_price
   ```

3. **计算浮动盈亏**：
   ```
   unrealized_pnl = position.get_unrealized_pnl(current_price)
   ```

4. **计算盈亏百分比**：
   ```
   pnl_percent = unrealized_pnl / entry_cost
   ```

5. **计算净值**：
   ```
   equity = account.get_equity(current_price)
   ```

6. **计算有效杠杆**：
   ```
   if equity <= 0:
       effective_leverage = inf
   else:
       position_value = |quantity| × current_price
       effective_leverage = position_value / equity
   ```

7. **计算 ADL 分数**：
   - 盈利者（pnl_percent > 0）：
     ```
     adl_score = pnl_percent × effective_leverage
     ```
   - 亏损者（pnl_percent <= 0）：
     ```
     if effective_leverage == inf:
         adl_score = -inf
     elif effective_leverage > 0:
         adl_score = pnl_percent / effective_leverage
     else:
         adl_score = -inf
     ```

## 排名算法设计

ADL 排名算法的设计目标是优先选择高杠杆高盈利的交易者作为减仓对象：

1. **盈利交易者**：盈利越多、杠杆越高，排名越高
   - 这类交易者承担了更高的市场风险，应该首先被减仓
   - 盈利百分比 × 有效杠杆，使得高杠杆高盈利者优先

2. **亏损交易者**：亏损越多、杠杆越低，排名越高
   - 亏损交易者的排名总是低于盈利交易者
   - 盈亏百分比 / 有效杠杆，使得低杠杆亏损者排名较高

3. **净值为负或零的情况**：
   - 有效杠杆设为无穷大
   - 盈利者分数为正无穷（最高优先级）
   - 亏损者分数为负无穷（最低优先级）

**设计理念**：
- 系统风险由高杠杆高盈利者优先承担
- 低杠杆保守交易者受到保护
- 确保市场风险公平分配

## 执行流程

### Trainer 中的 ADL 执行流程

ADL 的执行流程在 Trainer 的 `run_tick()` 方法中实现，分为三个阶段：

#### 阶段 1：预计算 ADL 候选清单（Tick 开始时）

在强平处理之前，预先计算所有 Agent 和噪声交易者的 ADL 分数：

```python
# 遍历所有 Agent
for agent in self.agents:
    candidate = self.adl_manager.calculate_adl_score(agent, latest_price)
    if candidate and candidate.pnl_percent > 0:  # 只选择盈利者
        if candidate.position_qty > 0:
            self._adl_long_candidates.append(candidate)
        else:
            self._adl_short_candidates.append(candidate)

# 遍历所有噪声交易者
for noise_trader in self.noise_traders:
    candidate = self.adl_manager.calculate_adl_score(noise_trader, latest_price)
    if candidate and candidate.pnl_percent > 0:
        if candidate.position_qty > 0:
            self._adl_long_candidates.append(candidate)
        else:
            self._adl_short_candidates.append(candidate)

# 按分数降序排序
self._adl_long_candidates.sort(key=lambda c: c.adl_score, reverse=True)
self._adl_short_candidates.sort(key=lambda c: c.adl_score, reverse=True)
```

**预计算的优势**：
- 避免在每次 ADL 执行时重复计算分数
- 提前筛选掉无持仓和亏损的参与者
- 按分数排序，确保优先选择最优候选

#### 阶段 2：强平市价单执行

对触发强平的 Agent 执行市价单平仓：

```python
# 1. 统一撤销被淘汰 Agent 的所有挂单
self._cancel_agent_orders(liquidated_agent)

# 2. 执行市价单平仓
remaining_qty, is_long = self._execute_liquidation_market_order(
    liquidated_agent, latest_price
)
```

如果市价单完全成交，`remaining_qty` 为 0，ADL 流程结束。

#### 阶段 3：ADL 自动减仓

如果市价单未完全成交（`remaining_qty > 0`），触发 ADL：

```python
if remaining_qty > 0:
    self._execute_adl(liquidated_agent, remaining_qty, latest_price, is_long)
```

**`_execute_adl()` 方法详解**：

1. **获取 ADL 成交价格**：
   ```python
   adl_price = self.adl_manager.get_adl_price(latest_price)
   ```

2. **选择对应方向的候选清单**：
   ```python
   # 被强平方是多头（需要卖出平仓），则需要空头对手
   # 被强平方是空头（需要买入平仓），则需要多头对手
   candidates = (
       self._adl_short_candidates if is_long else self._adl_long_candidates
   )
   ```

3. **循环执行 ADL 成交**：
   ```python
   for candidate in candidates:
       if remaining_qty <= 0:
           break

       # 计算可成交数量（取候选持仓、实际持仓、剩余数量的最小值）
       candidate_available_qty = abs(candidate.position_qty)
       actual_position = abs(candidate.participant.account.position.quantity)
       available_qty = min(candidate_available_qty, actual_position)
       liquidated_actual_position = abs(liquidated_agent.account.position.quantity)
       trade_qty = min(available_qty, remaining_qty, liquidated_actual_position)

       if trade_qty <= 0:
           continue

       # 更新双方账户（不收手续费）
       liquidated_agent.account.on_adl_trade(trade_qty, adl_price, is_taker=True)
       candidate.participant.account.on_adl_trade(trade_qty, adl_price, is_taker=False)

       # 更新候选清单中的 position_qty，防止重复使用
       if candidate.position_qty > 0:
           candidate.position_qty -= trade_qty
       else:
           candidate.position_qty += trade_qty

       remaining_qty -= trade_qty
   ```

4. **兜底处理**：
   ```python
   # 确保被淘汰者的仓位清零
   actual_remaining = abs(liquidated_agent.account.position.quantity)
   if actual_remaining > 0:
       liquidated_agent.account.position.quantity = 0
       liquidated_agent.account.position.avg_price = 0.0
   if liquidated_agent.account.balance < 0:
       liquidated_agent.account.balance = 0.0
   ```

**关键设计要点**：
- 使用预计算的候选清单，避免重复计算
- 动态更新 `position_qty`，防止多次 ADL 中同一候选被重复使用
- 同时检查候选清单和实际持仓，取最小值
- 兜底处理确保被淘汰者仓位清零

### ADL 与账户更新的交互

ADL 成交通过 `Account.on_adl_trade()` 方法更新账户：

**方法签名**：
```python
def on_adl_trade(self, quantity: int, price: float, is_taker: bool) -> float
```

**参数**：
- `quantity: int` - 成交数量（正数）
- `price: float` - 成交价格（当前市场价格）
- `is_taker: bool` - 是否为被强平方（True=被强平方，False=ADL对手方）

**返回值**：
- `float` - 已实现盈亏

**特点**：
- 不收取手续费
- 自动防止过度减仓（内部检查持仓数量）
- 仓位清零时重置均价

## 与其他模块的关系

### 依赖模块

**内部依赖**：
- `src.bio.agents.base.Agent` - 访问 Agent 的账户和持仓信息
- `src.market.noise_trader.noise_trader.NoiseTrader` - 噪声交易者，可作为 ADL 候选者
- `src.market.account.Account` - 获取余额、净值等账户信息
- `src.market.account.Position` - 获取持仓数量和均价
- `src.core.log_engine.logger` - 日志记录

**被依赖模块**：
- `src.training.trainer.Trainer` - 在强平处理中调用 ADL 机制
- `src.training.arena.parallel_arena_trainer.ParallelArenaTrainer` - 多竞技场并行训练中的 ADL 处理

### 数据流

```
Trainer.run_tick()
  ↓
阶段1: 预计算 ADL 候选清单
  ├─ ADLManager.calculate_adl_score(agent) → ADLCandidate
  ├─ 筛选盈利者（pnl_percent > 0）
  ├─ 按持仓方向分类（多头/空头）
  └─ 按分数降序排序
  ↓
阶段2: 强平市价单执行
  ├─ 撤销被淘汰 Agent 的挂单
  ├─ 提交市价平仓单
  └─ 返回剩余未平仓数量
  ↓
阶段3: ADL 自动减仓（如需要）
  ├─ ADLManager.get_adl_price() → 成交价格
  ├─ Trainer._execute_adl()
  │   ├─ 选择对应方向候选清单
  │   ├─ 循环与候选成交
  │   │   ├─ Account.on_adl_trade(被强平方) → 更新账户
  │   │   ├─ Account.on_adl_trade(候选方) → 更新账户
  │   │   └─ 更新 candidate.position_qty
  │   └─ 兜底处理（强制清零）
  └─ ADL 完成
```

## 使用示例

### 基本使用

```python
from src.market.adl import ADLManager

# 创建 ADL 管理器
adl_manager = ADLManager()

# 获取 ADL 成交价格
current_price = 10000.0
adl_price = adl_manager.get_adl_price(current_price)
# adl_price == 10000.0（直接使用当前市场价格）

# 计算 Agent 的 ADL 分数
candidate = adl_manager.calculate_adl_score(agent, current_price)
if candidate:
    print(f"持仓: {candidate.position_qty}")
    print(f"盈亏%: {candidate.pnl_percent:.4f}")
    print(f"有效杠杆: {candidate.effective_leverage:.2f}")
    print(f"ADL 分数: {candidate.adl_score:.4f}")
    print(f"是否为噪声交易者: {candidate.is_noise_trader}")
```

### Trainer 集成示例

```python
class Trainer:
    def __init__(self):
        self.adl_manager = ADLManager()
        self._adl_long_candidates = []
        self._adl_short_candidates = []

    def _prepare_adl_candidates(self, latest_price):
        """预计算 ADL 候选清单"""
        self._adl_long_candidates.clear()
        self._adl_short_candidates.clear()

        # 处理 Agent
        for agent in self.agents:
            candidate = self.adl_manager.calculate_adl_score(agent, latest_price)
            if candidate and candidate.pnl_percent > 0:
                if candidate.position_qty > 0:
                    self._adl_long_candidates.append(candidate)
                else:
                    self._adl_short_candidates.append(candidate)

        # 处理噪声交易者
        for noise_trader in self.noise_traders:
            candidate = self.adl_manager.calculate_adl_score(noise_trader, latest_price)
            if candidate and candidate.pnl_percent > 0:
                if candidate.position_qty > 0:
                    self._adl_long_candidates.append(candidate)
                else:
                    self._adl_short_candidates.append(candidate)

        # 排序
        self._adl_long_candidates.sort(key=lambda c: c.adl_score, reverse=True)
        self._adl_short_candidates.sort(key=lambda c: c.adl_score, reverse=True)

    def _execute_adl(self, liquidated_agent, remaining_qty, current_price, is_long):
        """执行 ADL 减仓"""
        adl_price = self.adl_manager.get_adl_price(current_price)
        candidates = (
            self._adl_short_candidates if is_long else self._adl_long_candidates
        )

        for candidate in candidates:
            if remaining_qty <= 0:
                break

            candidate_available_qty = abs(candidate.position_qty)
            actual_position = abs(candidate.participant.account.position.quantity)
            available_qty = min(candidate_available_qty, actual_position)
            liquidated_actual_position = abs(liquidated_agent.account.position.quantity)
            trade_qty = min(available_qty, remaining_qty, liquidated_actual_position)

            if trade_qty <= 0:
                continue

            # 更新账户
            liquidated_agent.account.on_adl_trade(trade_qty, adl_price, is_taker=True)
            candidate.participant.account.on_adl_trade(trade_qty, adl_price, is_taker=False)

            # 更新候选清单中的持仓数量
            if candidate.position_qty > 0:
                candidate.position_qty -= trade_qty
            else:
                candidate.position_qty += trade_qty

            remaining_qty -= trade_qty

        # 兜底处理
        actual_remaining = abs(liquidated_agent.account.position.quantity)
        if actual_remaining > 0:
            liquidated_agent.account.position.quantity = 0
            liquidated_agent.account.position.avg_price = 0.0
        if liquidated_agent.account.balance < 0:
            liquidated_agent.account.balance = 0.0
```

## 注意事项

### 重要设计原则

1. **ADL 成交价格是当前市场价格**
   - 不使用破产价格
   - 简单公平，双方都以市价成交

2. **ADL 成交不经过订单簿撮合**
   - 直接在对手方账户上执行减仓
   - 不产生 Trade 记录

3. **ADL 不收取手续费**
   - 通过 `on_adl_trade()` 方法执行，而非 `on_trade()`

4. **预计算优化**
   - 候选清单在 tick 开始时预计算
   - 避免重复计算 ADL 分数

5. **动态更新持仓**
   - 候选清单中的 `position_qty` 会被动态更新
   - 防止多次 ADL 中同一候选被重复使用

6. **多空对等**
   - 理论上不会出现候选不足的情况
   - 如果出现则说明有其他 bug

7. **噪声交易者参与**
   - 盈利的噪声交易者也可以作为 ADL 候选
   - 噪声交易者拥有无限资金（1e18），不触发强平，但可被 ADL 减仓

### 边界情况处理

1. **净值为零或负数**
   - 有效杠杆设为无穷大
   - 盈利者优先级最高
   - 亏损者优先级最低

2. **无持仓**
   - `calculate_adl_score()` 返回 None
   - 不参与 ADL

3. **持仓数量不匹配**
   - 同时检查候选清单和实际持仓
   - 取最小值确保安全

4. **兜底处理**
   - 确保 ADL 后被淘汰者仓位清零
   - 负余额清零

### 性能考虑

1. **预计算开销**
   - 每个 tick 计算一次所有 Agent 和噪声交易者的 ADL 分数
   - 时间复杂度：O(n)，n 为 Agent 和噪声交易者总数

2. **ADL 执行开销**
   - 循环候选清单，逐个减仓
   - 平均情况：少数候选即可完成减仓
   - 最坏情况：遍历所有候选

3. **内存开销**
   - 候选清单存储引用，不复制 Agent 对象
   - 每个候选约 100 字节（数据类 + 引用）

## 配置相关

ADL 相关配置来自以下模块：

- `AgentConfig.maintenance_margin_rate` - 维持保证金率，触发强平的阈值
- `AgentConfig.leverage` - 杠杆倍数，影响有效杠杆计算
- `NoiseTraderConfig` - 噪声交易者配置

## 测试建议

测试 ADL 系统时应覆盖以下场景：

1. **正常 ADL 流程**
   - 强平市价单未完全成交
   - 有足够盈利对手方
   - ADL 成功清零仓位

2. **边界情况**
   - 净值为零或负数
   - 持仓数量不匹配
   - 无盈利对手方

3. **噪声交易者参与**
   - 噪声交易者作为 ADL 候选
   - 噪声交易者被 ADL 减仓

4. **多次 ADL**
   - 同一个 tick 内多次触发 ADL
   - 候选持仓数量动态更新

5. **性能测试**
   - 大量 Agent 的预计算性能
   - 候选清单排序性能

## 总结

ADL 模块是交易系统风险控制的关键组件，通过优先选择高杠杆高盈利的交易者进行减仓，确保市场风险公平分配。其设计简洁高效，与强平机制无缝集成，是保障市场稳定运行的重要防线。
