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

持仓类，使用 Cython 加速以提高性能。记录 Agent 的资产持有情况。

#### C 级别属性（cdef）

- `quantity: int` - 持仓数量（正数=多头，负数=空头，0=空仓）
- `avg_price: double` - 平均开仓价格
- `realized_pnl: double` - 已实现盈亏累计

#### 核心方法

**update(side, quantity, price) -> float**
- 更新持仓，返回本次成交产生的已实现盈亏
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
   - 买入：`quantity = quantity`, `avg_price = price`
   - 卖出：`quantity = -quantity`, `avg_price = price`

2. **持有多头（quantity > 0）**
   - **买入（加多仓）**：
     - 新均价 = `(旧数量×旧均价 + 新数量×新价格) / (旧数量+新数量)`
     - `quantity += quantity`
   - **卖出（减多仓）**：
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
     - 新均价 = `(旧数量×旧均价 + 新数量×新价格) / (旧数量+新数量)`
     - `quantity -= quantity`
   - **买入（减空仓）**：
     - 已实现盈亏 = `(均价 - 价格) × 成交数量`
     - `quantity += quantity`
   - **买入数量 = |持仓数量|（完全平空）**：
     - 已实现盈亏 = `(均价 - 价格) × |持仓数量|`
     - `quantity = 0`, `avg_price = 0.0`
   - **买入数量 > |持仓数量|（反向开多）**：
     - 已实现盈亏 = `(均价 - 价格) × |持仓数量|`
     - 剩余数量 = 买入数量 - |持仓数量|
     - `quantity = 剩余数量`, `avg_price = price`

### Account (account.py)

账户类，管理 Agent 的完整交易账户状态。

#### 属性

- `agent_id: int` - Agent ID
- `agent_type: AgentType` - Agent 类型（RETAIL/RETAIL_PRO/WHALE/MARKET_MAKER）
- `initial_balance: float` - 初始余额
- `balance: float` - 当前余额（已实现盈亏已计入）
- `position: Position` - 持仓对象
- `leverage: float` - 杠杆倍数（散户/高级散户=100，庄家/做市商=10）
- `maintenance_margin_rate: float` - 维持保证金率
- `maker_fee_rate: float` - 挂单手续费率
- `taker_fee_rate: float` - 吃单手续费率
- `pending_order_id: int | None` - 当前挂单 ID
- `maker_volume: int` - 作为 maker 的累计成交量（用于做市商适应度计算）
- `volatility_contribution: float` - 作为 taker 的价格冲击累计（庄家适应度计算用）

#### 核心方法

**get_equity(current_price) -> float**
- 计算账户净值
- 公式：`balance + unrealized_pnl`
- 用途：评估账户总资产状态

**get_margin_ratio(current_price) -> float**
- 计算保证金率
- 公式：`equity / (|quantity| × current_price)`
- 无持仓时返回 `float("inf")`（无强平风险）
- 用途：评估强平风险，保证金率越低风险越高

**check_liquidation(current_price) -> bool**
- 检查是否需要强制平仓
- 条件：`margin_ratio < maintenance_margin_rate`
- 返回 `True` 表示触发强平

**on_trade(trade, is_buyer) -> None**
- 处理成交回报
- 步骤：
  1. 确定成交方向（买方/卖方）
  2. 判断是否为 maker 并累加 `maker_volume`
  3. 更新持仓，获取已实现盈亏
  4. 将已实现盈亏加到余额
  5. 扣除手续费
- 费用计算：从 `Trade` 对象中获取 `buyer_fee` 或 `seller_fee`
- maker 判断逻辑：`trade.is_buyer_taker=True` 时卖方是 maker，否则买方是 maker

**on_adl_trade(quantity, price, is_taker) -> float**
- 处理 ADL（Auto-Deleveraging）自动减仓成交
- 特点：不收取手续费
- 参数：
  - `quantity`: 成交数量（正数）
  - `price`: 成交价格（当前市场价格）
  - `is_taker`: 是否为被强平方
- 逻辑：
  1. 确保不会过度减仓（检查当前持仓）
  2. 计算已实现盈亏
  3. 更新持仓数量
  4. 更新余额（不扣手续费）
  5. 仓位清零时重置均价

### FastAccount (fast_account.pyx)

快速账户类，使用 Cython 加速以提高性能。与 Account 类接口兼容，但使用纯 C 类型以获得更高的性能。

#### C 级别属性（cdef public）

- `agent_id: int` - Agent ID
- `agent_type: int` - Agent 类型整数（0=RETAIL, 1=RETAIL_PRO, 2=WHALE, 3=MARKET_MAKER）
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

**on_trade(trade: FastTrade, is_buyer: bint) -> void**
- 处理 FastTrade 成交回报
- 注意：接受 FastTrade 而非 Trade 对象

**on_adl_trade(quantity: int, price: double, is_taker: bint) -> double**
- 处理 ADL 成交，返回已实现盈亏

**reset() -> void**
- 重置账户到初始状态
- 将余额恢复到初始值
- 创建新的空 Position 对象
- 重置 pending_order_id、maker_volume、volatility_contribution

#### 与 Account 类的区别

| 特性 | Account | FastAccount |
|------|---------|-------------|
| 实现语言 | Python | Cython |
| agent_type | AgentType 枚举 | int（0-3） |
| pending_order_id | int \| None | int（-1 表示 None） |
| on_trade 参数 | Trade | FastTrade |
| 无持仓时保证金率 | float("inf") | INFINITY |
| reset 方法 | 无 | 有 |

#### 使用场景

FastAccount 适用于：
- 高频交易场景，需要最大化性能
- 与 FastMatchingEngine 配合使用
- 批量处理大量账户状态更新

## 数据流

### 交易流程

```
Trade 发生
  ↓
Account.on_trade()
  ↓
Position.update() → 返回 realized_pnl
  ↓
balance += realized_pnl - fee
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
```

## Cython 优化说明

### 为什么使用 Cython

`Position` 和 `FastAccount` 类使用 Cython 实现的原因：

1. **高频调用**：每次成交都会调用持仓和账户更新方法，在高并发场景下调用频率极高
2. **性能关键**：持仓更新和账户状态计算是交易路径上的性能瓶颈
3. **类型安全**：Cython 的静态类型检查可以提前发现类型错误
4. **减少开销**：C 级别的属性访问和方法调用比 Python 快得多

### Cython 特性使用

- `cdef class` - 定义 C 级别的类
- `cdef public` - 定义可以从 Python 访问的 C 级别属性
- `cpdef` - 定义可以从 Python 和 C 调用的方法
- `cdef` - 声明局部变量为 C 类型，避免 Python 对象开销
- `.pxd` 文件 - 声明文件，用于跨模块导入 cdef class
- `.pyi` 文件 - 类型存根文件，提供 IDE 类型提示支持

### pxd 声明文件

当一个 Cython 模块需要被其他 Cython 模块导入时，需要创建 `.pxd` 声明文件：

- `position.pxd` - 声明 Position 类的 cdef 属性和方法
- 其他模块可以通过 `from ... cimport` 导入 cdef class

### 编译说明

修改 `.pyx` 文件后需要重新编译：

```bash
cd /home/rongheng/python_project/TradingGame_cython_opt
./rebuild.sh
```

或手动编译：

```bash
python setup.py build_ext --inplace
```

## 保证金和强平机制

### 保证金计算

**占用保证金**：`margin_used = |quantity| × current_price / leverage`

**净值**：`equity = balance + unrealized_pnl`

**保证金率**：`margin_ratio = equity / (|quantity| × current_price)`

### 强平触发条件

当 `margin_ratio < maintenance_margin_rate` 时触发强平。

### 强平流程

1. 检测到保证金率不足
2. 提交市价平仓单
3. 如果市价单完全成交，正常结束
4. 如果市价单无法完全成交（流动性不足），触发 ADL 机制

## ADL（自动减仓）支持

ADL 是在强平订单无法完全成交时的风险控制机制。

**ADL 特点**：
- 不收取手续费
- 以当前市场价格成交（不使用破产价格）
- 可能减少盈利对手方的仓位

**成交价格**：
- 直接使用当前市场价格
- 设计原因：强平 ≠ 破产，被强平时 Agent 可能还有正净值（仅保证金率过低）

相关实现见 `src/market/adl/` 模块。

## 配置说明

账户相关配置来自 `AgentConfig`：

- `initial_balance` - 初始余额
- `leverage` - 杠杆倍数
- `maintenance_margin_rate` - 维持保证金率
- `maker_fee_rate` - 挂单手续费率
- `taker_fee_rate` - 吃单手续费率

不同 Agent 类型的默认配置：
- **散户/高级散户**：100 倍杠杆
- **庄家/做市商**：10 倍杠杆

## 依赖关系

- `src.config.config.AgentConfig` - Agent 配置
- `src.config.config.AgentType` - Agent 类型枚举
- `src.market.orderbook.order.OrderSide` - 订单方向（BUY/SELL）
- `src.market.matching.trade.Trade` - 成交记录类

## 类型注解

本模块遵循严格的类型注解规范：
- 所有方法参数和返回值都有类型注解
- 使用 `|` 运算符表示联合类型（Python 3.10+）
- Cython 代码使用 C 类型声明（`cdef`, `cpdef`）

## 使用示例

### 使用 Python Account 类

```python
from src.market.account import Account, Position
from src.config.config import AgentConfig, AgentType

# 创建账户
config = AgentConfig()
account = Account(agent_id=1, agent_type=AgentType.RETAIL, config=config)

# 处理成交
trade = Trade(price=100.0, quantity=10, buyer_fee=0.1, seller_fee=0.1)
account.on_trade(trade, is_buyer=True)

# 检查强平风险
current_price = 100.0
if account.check_liquidation(current_price):
    # 触发强平流程
    pass

# 获取账户状态
equity = account.get_equity(current_price)
margin_ratio = account.get_margin_ratio(current_price)
```

### 使用 Cython FastAccount 类

```python
from src.market.account import FastAccount
from src.market.matching.fast_matching import FastTrade

# 创建快速账户（直接传入参数，不使用配置对象）
account = FastAccount(
    agent_id=1,
    agent_type=0,  # RETAIL = 0
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
    seller_fee=0.1,
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

# 重置账户
account.reset()
```
