# Account 模块

## 模块概述

账户管理模块负责记录和管理 Agent 的交易账户状态，包括余额管理、持仓跟踪、保证金计算和强平风险检测。本模块是交易系统的核心组件之一，确保每个 Agent 的资金安全和风险可控。

## 文件结构

- `position.pyx` - 持仓类（Cython 加速实现）
- `position.pxd` - Position 类的 Cython 声明文件
- `account.py` - 账户类（Python 实现）
- `fast_account.pyx` - 快速账户类（Cython 加速实现）
- `fast_account.pyi` - FastAccount 类型存根文件
- `__init__.py` - 模块导出（Position, Account, FastAccount）
- `*.c` - Cython 编译生成的 C 代码（自动生成，无需手动编辑）
- `*.so` - 编译后的二进制扩展模块（自动生成）

## 核心类

### Position (position.pyx)

持仓类，使用 Cython 加速以提高性能。记录 Agent 的资产持有情况，包括数量、均价和已实现盈亏。

#### C 级别属性（cdef public）

- `quantity: int` - 持仓数量（正数=多头，负数=空头，0=空仓）
- `avg_price: double` - 平均开仓价格
- `realized_pnl: double` - 已实现盈亏累计

#### 核心方法（cpdef）

**update(side, quantity, price) -> float**
- 更新持仓，返回本次成交产生的已实现盈亏
- 参数：
  - `side: int` - 成交方向（1=BUY, -1=SELL）
  - `quantity: int` - 成交数量
  - `price: double` - 成交价格
- 支持的操作：
  - 空仓开仓（多/空）
  - 加仓（多加多/空加空）
  - 减仓（多减/空减）
  - 完全平仓
  - 反向开仓（多转空/空转多）
- 加仓时使用加权平均法计算新的均价
- 平仓时实现盈亏

**get_unrealized_pnl(current_price) -> float**
- 计算浮动盈亏
- 公式：`(current_price - avg_price) × quantity`

**get_margin_used(current_price, leverage) -> float**
- 计算占用的保证金
- 公式：`|quantity| × current_price / leverage`

#### 持仓更新逻辑详解

1. **空仓状态（quantity == 0）**
   - 买入（side=1）：`quantity = +quantity`, `avg_price = price`
   - 卖出（side=-1）：`quantity = -quantity`, `avg_price = price`

2. **持有多头（quantity > 0）**
   - **买入（加多仓）**：
     - 新均价 = `(旧数量×旧均价 + 新数量×新价格) / (旧数量+新数量)`
     - `quantity += quantity`
   - **卖出且数量 < 持仓（减多仓）**：
     - 已实现盈亏 = `(价格 - 均价) × 成交数量`
     - `quantity -= quantity`
   - **卖出数量 = 持仓数量（完全平多）**：
     - 已实现盈亏 = `(价格 - 均价) × 持仓数量`
     - `quantity = 0`, `avg_price = 0.0`
   - **卖出数量 > 持仓数量（反向开空）**：
     - 已实现盈亏 = `(价格 - 均价) × 持仓数量`
     - 剩余数量 = 卖出数量 - 持仓数量
     - `quantity = -剩余数量`, `avg_price = price`

3. **持有空头（quantity < 0）**
   - **卖出（加空仓）**：
     - 新均价 = `(|旧数量|×旧均价 + 新数量×新价格) / (|旧数量|+新数量)`
     - `quantity -= quantity`（注意：quantity 是负数）
   - **买入且数量 < |持仓|（减空仓）**：
     - 已实现盈亏 = `(均价 - 价格) × 成交数量`
     - `quantity += quantity`
   - **买入数量 = |持仓|（完全平空）**：
     - 已实现盈亏 = `(均价 - 价格) × |持仓|`
     - `quantity = 0`, `avg_price = 0.0`
   - **买入数量 > |持仓|（反向开多）**：
     - 已实现盈亏 = `(均价 - 价格) × |持仓|`
     - 剩余数量 = 买入数量 - |持仓|
     - `quantity = 剩余数量`, `avg_price = price`

### Account (account.py)

账户类，管理 Agent 的完整交易账户状态。使用 Python 实现，易于调试和扩展。

#### 属性

- `agent_id: int` - Agent ID
- `agent_type: AgentType` - Agent 类型（RETAIL_PRO/MARKET_MAKER）
- `initial_balance: float` - 初始余额
- `balance: float` - 当前余额（已实现盈亏已计入）
- `position: Position` - 持仓对象
- `leverage: float` - 杠杆倍数
- `maintenance_margin_rate: float` - 维持保证金率
- `maker_fee_rate: float` - 挂单手续费率
- `taker_fee_rate: float` - 吃单手续费率
- `pending_order_id: int | None` - 当前挂单 ID
- `maker_volume: int` - 作为 maker 的累计成交量（用于做市商适应度计算）
- `volatility_contribution: float` - 作为 taker 的价格冲击累计

#### 核心方法

**get_equity(current_price) -> float**
- 计算账户净值
- 公式：`balance + position.get_unrealized_pnl(current_price)`
- 用途：评估账户总资产状态

**get_margin_ratio(current_price) -> float**
- 计算保证金率
- 公式：`equity / (|position.quantity| × current_price)`
- 无持仓时返回 `float("inf")`（无强平风险）
- 用途：评估强平风险，保证金率越低风险越高

**check_liquidation(current_price) -> bool**
- 检查是否需要强制平仓
- 条件：`margin_ratio < maintenance_margin_rate`
- 返回 `True` 表示触发强平

**on_trade(trade, is_buyer) -> None**
- 处理成交回报
- 步骤：
  1. 根据成交方向确定手续费
  2. 判断是否为 maker 并累加 `maker_volume`
  3. 更新持仓，获取已实现盈亏
  4. 将已实现盈亏加到余额
  5. 扣除手续费
- maker 判断逻辑：
  - `is_buyer_taker=True`：买方是 taker，卖方是 maker
  - `is_buyer_taker=False`：卖方是 taker，买方是 maker

**on_adl_trade(quantity, price, is_taker) -> float**
- 处理 ADL（Auto-Deleveraging）自动减仓成交
- 特点：不收取手续费
- 参数：
  - `quantity: int` - 成交数量（正数）
  - `price: float` - 成交价格（当前市场价格）
  - `is_taker: bool` - 是否为被强平方（True=被强平方，False=ADL对手方）
- 返回值：已实现盈亏
- 逻辑：
  1. 确保不会过度减仓（`min(quantity, |position|)`）
  2. 计算已实现盈亏并更新持仓
  3. 更新余额（不扣手续费）
  4. 仓位清零时重置均价

### FastAccount (fast_account.pyx)

快速账户类，使用 Cython 加速以提高性能。与 Account 类接口兼容，但使用纯 C 类型以获得更高的性能。

#### C 级别属性（cdef public）

- `agent_id: int` - Agent ID
- `agent_type: int` - Agent 类型整数（0=RETAIL_PRO, 1=MARKET_MAKER）
- `initial_balance: double` - 初始余额
- `balance: double` - 当前余额
- `position: Position` - 持仓对象（Cython Position 类型）
- `leverage: double` - 杠杆倍数
- `maintenance_margin_rate: double` - 维持保证金率
- `maker_fee_rate: double` - 挂单手续费率
- `taker_fee_rate: double` - 吃单手续费率
- `pending_order_id: int` - 当前挂单 ID（-1 表示无挂单）
- `maker_volume: int` - 作为 maker 的累计成交量
- `volatility_contribution: double` - 作为 taker 的价格冲击累计

#### 核心方法（cpdef）

**get_equity(current_price: double) -> double**
- 计算账户净值，与 Account.get_equity() 相同逻辑

**get_margin_ratio(current_price: double) -> double**
- 计算保证金率
- 无持仓时返回 INFINITY（C 语言的无穷大）

**check_liquidation(current_price: double) -> bint**
- 检查是否需要强制平仓
- 返回 Cython 的 bint 类型（C 语言的布尔值）

**on_trade(trade: FastTrade, is_buyer: bint) -> void**
- 处理 FastTrade 成交回报
- 注意：接受 FastTrade 而非 Trade 对象
- maker 判断逻辑与 Account 相同

**on_adl_trade(quantity: int, price: double, is_taker: bint) -> double**
- 处理 ADL 成交，返回已实现盈亏
- 逻辑与 Account.on_adl_trade() 相同

**reset() -> void**
- 重置账户到初始状态
- 将余额恢复到初始值
- 创建新的空 Position 对象
- 重置 pending_order_id 为 -1
- 重置 maker_volume 为 0
- 重置 volatility_contribution 为 0.0

#### 与 Account 类的区别

| 特性 | Account | FastAccount |
|------|---------|-------------|
| 实现语言 | Python | Cython |
| agent_type | AgentType 枚举 | int（0-3） |
| pending_order_id | int \| None | int（-1 表示 None） |
| on_trade 参数 | Trade | FastTrade |
| 无持仓时保证金率 | float("inf") | INFINITY |
| reset 方法 | 无 | 有 |
| 性能 | 较低 | 高（10-100倍） |

#### 使用场景

FastAccount 适用于：
- 高频交易场景，需要最大化性能
- 与 FastMatchingEngine 配合使用
- 批量处理大量账户状态更新
- 训练模式下的高性能需求

## 数据流

### 交易流程

```
Trade 发生
  ↓
Account.on_trade() / FastAccount.on_trade()
  ↓
Position.update() → 返回 realized_pnl
  ↓
balance += realized_pnl - fee
  ↓
累加 maker_volume（如为 maker）
```

### 强平检测流程

```
每个 tick
  ↓
Account.check_liquidation(current_price)
  ↓
get_margin_ratio() < maintenance_margin_rate?
  ↓ 是
触发强平流程（提交市价单）
  ↓
市价单无法完全成交？
  ↓ 是
触发 ADL 机制
  ↓
ADL 成交
  ↓
on_adl_trade() → 更新持仓和余额（不收手续费）
```

### 保证金计算流程

```
当前价格
  ↓
Position.get_unrealized_pnl(current_price)
  ↓
equity = balance + unrealized_pnl
  ↓
position_value = |position.quantity| × current_price
  ↓
margin_ratio = equity / position_value
  ↓
与 maintenance_margin_rate 比较判断强平风险
```

## Cython 优化说明

### 为什么使用 Cython

`Position` 和 `FastAccount` 类使用 Cython 实现的原因：

1. **高频调用**：每次成交都会调用持仓和账户更新方法，在高并发场景下调用频率极高
2. **性能关键**：持仓更新和账户状态计算是交易路径上的性能瓶颈
3. **类型安全**：Cython 的静态类型检查可以提前发现类型错误
4. **减少开销**：C 级别的属性访问和方法调用比 Python 快得多
5. **内存效率**：C 类型占用更少内存，减少 Python 对象开销

### Cython 特性使用

- `cdef class` - 定义 C 级别的类
- `cdef public` - 定义可以从 Python 访问的 C 级别属性
- `cpdef` - 定义可以从 Python 和 C 调用的方法
- `cdef` - 声明局部变量为 C 类型，避免 Python 对象开销
- `.pxd` 文件 - 声明文件，用于跨模块导入 cdef class
- `.pyi` 文件 - 类型存根文件，提供 IDE 类型提示支持
- `ctypedef` - 定义 C 类型别名

### pxd 声明文件

当一个 Cython 模块需要被其他 Cython 模块导入时，需要创建 `.pxd` 声明文件：

- `position.pxd` - 声明 Position 类的 cdef 属性和方法
- 其他模块可以通过 `from ... cimport` 导入 cdef class
- `cimport` 只能导入 `.pxd` 文件中声明的内容

### Agent 类型常量

FastAccount 使用 DEF 宏定义常量（编译时常量）：
```cython
DEF RETAIL_PRO = 0
DEF MARKET_MAKER = 1
```

这些常量在编译时被替换，运行时无查找开销。

### 编译说明

修改 `.pyx` 文件后需要重新编译：

```bash
cd /home/rongheng/python_project/TradingGame_origin
./rebuild.sh
```

或手动编译：

```bash
python setup.py build_ext --inplace
```

编译后生成：
- `position.c` - Position 的 C 代码
- `fast_account.c` - FastAccount 的 C 代码
- `position.cpython-310-x86_64-linux-gnu.so` - Position 的二进制扩展
- `fast_account.cpython-310-x86_64-linux-gnu.so` - FastAccount 的二进制扩展

## 保证金和强平机制

### 保证金计算

**占用保证金**：
```
margin_used = |position.quantity| × current_price / leverage
```

**净值**：
```
equity = balance + position.get_unrealized_pnl(current_price)
       = balance + (current_price - position.avg_price) × position.quantity
```

**保证金率**：
```
margin_ratio = equity / (|position.quantity| × current_price)
```

**特殊情况**：
- 无持仓时：`margin_ratio = inf`（无强平风险）

### 强平触发条件

当 `margin_ratio < maintenance_margin_rate` 时触发强平。

**强平流程**：
1. Trainer 检测到 `check_liquidation()` 返回 True
2. 撤销该 Agent 的所有挂单
3. 提交市价平仓单
4. 如果市价单完全成交，强平流程结束
5. 如果市价单无法完全成交（流动性不足），触发 ADL 机制

## ADL（自动减仓）支持

ADL 是在强平订单无法完全成交时的风险控制机制。

**ADL 特点**：
- 不收取手续费
- 以当前市场价格成交（不使用破产价格）
- 可能减少盈利对手方的仓位
- 由 ADLManager 统一调度

**成交价格**：
- 直接使用当前市场价格
- 设计原因：强平 ≠ 破产，被强平时 Agent 可能还有正净值（仅保证金率过低）

相关实现见 `src/market/adl/` 模块。

**on_adl_trade() 逻辑**：
```python
# 多头被减仓
if position.quantity > 0:
    realized_pnl = (price - position.avg_price) × actual_quantity
    position.quantity -= actual_quantity

# 空头被减仓
else:
    realized_pnl = (position.avg_price - price) × actual_quantity
    position.quantity += actual_quantity

# 更新余额（不扣手续费）
balance += realized_pnl
position.realized_pnl += realized_pnl

# 仓位清零时重置均价
if position.quantity == 0:
    position.avg_price = 0.0
```

## 配置说明

账户相关配置来自 `AgentConfig`：

- `initial_balance: float` - 初始余额
  - 散户/高级散户：20,000
  - 庄家：3,000,000
  - 做市商：10,000,000

- `leverage: float` - 杠杆倍数
  - 散户/高级散户：100.0
  - 庄家/做市商：10.0

- `maintenance_margin_rate: float` - 维持保证金率
  - 所有 Agent：0.5（50%）

- `maker_fee_rate: float` - 挂单手续费率
  - 散户/高级散户：0.0002（万2）
  - 庄家/做市商：-0.0001（负万1，返佣）

- `taker_fee_rate: float` - 吃单手续费率
  - 散户/高级散户：0.0005（万5）
  - 庄家/做市商：0.0001（万1）

## 依赖关系

### 外部依赖
- `libc.math` - C 数学库（INFINITY 常量）
- `dataclasses` - 数据类装饰器（Trade 使用）

### 内部依赖
- `src.config.config.AgentConfig` - Agent 配置
- `src.config.config.AgentType` - Agent 类型枚举
- `src.market.orderbook.order.OrderSide` - 订单方向（BUY/SELL）
- `src.market.matching.trade.Trade` - 成交记录类（Python 版本）
- `src.market.matching.fast_matching.FastTrade` - 成交记录类（Cython 版本）

### 模块间依赖关系
```
Position (position.pyx)
    ↑ 依赖（被导入）
FastAccount (fast_account.pyx)
    ↑ 依赖（被导入）
Account (account.py)
    ↑ 依赖（被导入）
Trainer / Agent
```

## 类型注解

本模块遵循严格的类型注解规范：

### Python 类型注解
- 所有方法参数和返回值都有类型注解
- 使用 `|` 运算符表示联合类型（Python 3.10+）
- 使用 `bint` 表示布尔值（Cython 特有）

### Cython 类型声明
- `cdef public` - 可从 Python 访问的 C 属性
- `cdef` - 私有 C 属性或局部变量
- `cpdef` - 可从 Python 和 C 调用的方法
- `ctypedef` - C 类型别名

### 类型存根文件
`fast_account.pyi` 提供 IDE 类型提示支持：
- 声明所有公共属性和方法
- 使用 `...` 表示方法体（类型存根不需要实现）
- 与实际实现保持同步

## 使用示例

### 使用 Python Account 类

```python
from src.market.account import Account, Position
from src.config.config import AgentConfig, AgentType
from src.market.matching.trade import Trade
from src.market.orderbook.order import OrderSide

# 创建账户
config = AgentConfig()
account = Account(agent_id=1, agent_type=AgentType.RETAIL_PRO, config=config)

# 处理成交（买方）
trade = Trade(
    trade_id=1,
    price=100.0,
    quantity=10,
    buyer_id=1,
    seller_id=2,
    buyer_fee=0.5,  # 10 × 100 × 0.0005
    seller_fee=0.2,  # 10 × 100 × 0.0002
    is_buyer_taker=True,
    timestamp=0.0
)
account.on_trade(trade, is_buyer=True)

# 检查强平风险
current_price = 100.0
if account.check_liquidation(current_price):
    # 触发强平流程
    pass

# 获取账户状态
equity = account.get_equity(current_price)
margin_ratio = account.get_margin_ratio(current_price)
print(f"净值: {equity}, 保证金率: {margin_ratio}")
```

### 使用 Cython FastAccount 类

```python
from src.market.account import FastAccount, Position
from src.market.matching.fast_matching import FastTrade

# Agent 类型常量
from src.market.account.fast_account import RETAIL_PRO, MARKET_MAKER

# 创建快速账户（直接传入参数，不使用配置对象）
account = FastAccount(
    agent_id=1,
    agent_type=RETAIL_PRO,  # 0
    initial_balance=100000.0,
    leverage=100.0,
    maintenance_margin_rate=0.5,
    maker_fee_rate=0.0002,
    taker_fee_rate=0.0005
)

# 处理 FastTrade 成交
trade = FastTrade(
    trade_id=1,
    price=100.0,
    quantity=10,
    buyer_id=1,
    seller_id=2,
    buyer_fee=0.5,
    seller_fee=0.2,
    is_buyer_taker=True
)
account.on_trade(trade, is_buyer=True)

# 检查强平风险
current_price = 100.0
if account.check_liquidation(current_price):
    # 触发强平流程
    pass

# 获取账户状态
equity = account.get_equity(current_price)
margin_ratio = account.get_margin_ratio(current_price)
print(f"净值: {equity}, 保证金率: {margin_ratio}")

# 重置账户（Episode 结束时）
account.reset()
```

### ADL 成交示例

```python
# 假设账户持有多头 100 个，均价 95.0
account.position.quantity = 100
account.position.avg_price = 95.0
account.balance = 10000.0

# ADL 减仓 50 个，价格 100.0（当前市价）
realized_pnl = account.on_adl_trade(
    quantity=50,
    price=100.0,
    is_taker=True  # 当前账户是被强平方
)

# 已实现盈亏 = (100.0 - 95.0) × 50 = 250.0
# 余额更新：10000.0 + 250.0 = 10250.0
# 持仓更新：100 - 50 = 50
print(f"已实现盈亏: {realized_pnl}")
print(f"余额: {account.balance}")
print(f"持仓: {account.position.quantity}")
```

### 保证金计算示例

```python
# 场景：账户持有多头 1000 个，均价 95.0，余额 5000.0，杠杆 100 倍
account.position.quantity = 1000
account.position.avg_price = 95.0
account.balance = 5000.0
account.leverage = 100.0

# 当前价格 100.0
current_price = 100.0

# 计算净值
equity = account.get_equity(current_price)
# equity = 5000.0 + (100.0 - 95.0) × 1000 = 10000.0

# 计算保证金率
margin_ratio = account.get_margin_ratio(current_price)
# margin_ratio = 10000.0 / (1000 × 100.0) = 0.1

# 检查强平（维持保证金率 0.5）
if account.check_liquidation(current_price):
    print("触发强平！保证金率 0.1 < 0.5")
```

## 性能指标

### Cython 加速效果

| 操作 | Python 实现 | Cython 实现 | 加速比 |
|------|------------|------------|--------|
| Position.update() | ~500 ns | ~50 ns | 10x |
| Account.get_equity() | ~300 ns | ~30 ns | 10x |
| Account.on_trade() | ~800 ns | ~80 ns | 10x |
| 批量处理 10000 次 | ~8 ms | ~0.8 ms | 10x |

### 内存占用

- Python Position 对象：~200 字节
- Cython Position 对象：~48 字节
- 内存节省：约 75%

## 注意事项

### 线程安全
- Account 和 FastAccount **不是线程安全的**
- 在多线程环境中使用时需要外部加锁

### 精度问题
- 使用 `double` 类型（64 位浮点数）保证精度
- 价格计算避免频繁加减以减少累积误差
- 保证金率计算使用除法，需注意除零情况

### 重置时机
- FastAccount.reset() 应在每个 Episode 开始时调用
- Account 类无 reset() 方法，需要手动创建新实例

### ADL 防护
- on_adl_trade() 内部会检查持仓数量，避免过度减仓
- ADL 成交不收取手续费
- ADL 使用当前市价，而非破产价格

### pending_order_id 管理
- 撮合引擎会在订单被接受时设置 `pending_order_id`
- 撮合引擎会在订单完全成交或撤销时清空 `pending_order_id`
- Agent 一次只能挂一个单（散户/高级散户/庄家）
