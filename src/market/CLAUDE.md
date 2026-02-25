# Market 模块 - 市场引擎

## 模块概述

市场引擎是交易模拟竞技场的核心基础设施，负责订单簿管理、订单撮合、账户状态跟踪、风险控制和市场随机性提供。该模块采用高性能的 Cython 加速技术，实现了完整的限价订单簿（LOB）交易机制，支持多种 Agent 类型的差异化手续费策略和复杂的风险控制机制。

## 设计理念

1. **高性能优先**：关键路径使用 Cython 加速，订单簿、持仓计算、神经网络推理等模块性能提升 10-100 倍
2. **数据一致性**：严格的状态同步机制，确保订单簿、账户、成交记录的一致性
3. **风险控制**：多层次风险控制机制，包括保证金监控、强制平仓、ADL 自动减仓
4. **可扩展性**：模块化设计，支持多种订单类型、Agent 类型
5. **市场真实性**：模拟真实交易所的撮合规则、手续费模型、价格发现机制

## 文件结构

```
src/market/
├── __init__.py              # 模块导出（NormalizedMarketState）
├── market_state.py          # 归一化市场状态数据类
├── orderbook/               # 订单簿模块（Cython 加速）
│   ├── order.py            # 订单数据模型（Order, OrderSide, OrderType）
│   ├── orderbook.pyx       # 订单簿实现（PriceLevel, OrderBook）
│   └── CLAUDE.md           # 订单簿模块文档
├── matching/               # 撮合引擎模块
│   ├── __init__.py         # 模块导出（尝试导入 Cython 版本）
│   ├── trade.py            # 成交记录数据类（Trade）
│   ├── matching_engine.py  # 撮合引擎（Python 实现）
│   ├── fast_matching.pxd   # Cython 声明文件
│   ├── fast_matching.pyx   # 撮合引擎（Cython 实现：FastTrade, FastMatchingEngine, fast_match_orders）
│   └── CLAUDE.md           # 撮合引擎模块文档
├── account/                # 账户管理模块
│   ├── __init__.py         # 模块导出（Position, Account, FastAccount）
│   ├── account.py          # 账户类（Python 实现）
│   ├── position.pxd        # Position Cython 声明文件
│   ├── position.pyx        # 持仓类（Cython 实现）
│   ├── fast_account.pyx    # 快速账户类（Cython 实现）
│   └── CLAUDE.md           # 账户管理模块文档
├── adl/                    # ADL 自动减仓模块
│   ├── __init__.py         # 模块导出
│   ├── adl_manager.py      # ADL 管理器（ADLCandidate, ADLManager）
│   └── CLAUDE.md           # ADL 模块文档
└── noise_trader/           # 噪声交易者模块
    ├── __init__.py         # 模块导出 + create_noise_traders 工厂函数
    ├── noise_trader.py     # 噪声交易者类
    ├── noise_trader_account.py # 噪声交易者账户
    └── CLAUDE.md           # 噪声交易者模块文档
```

## 核心功能模块

### 1. 市场状态数据（market_state.py）

**NormalizedMarketState** - 预计算的归一化市场数据类

在每个 tick 开始时由 Trainer 预计算一次，然后传递给所有 Agent 使用，避免重复计算订单簿数据的归一化。

**属性：**
- `mid_price: float` - 中间价（用于归一化计算的参考价格）
- `tick_size: float` - 最小价格变动单位
- `bid_data: NDArray[np.float32]` - 买盘数据，shape (200,)，100档 x 2（价格归一化 + 数量归一化）
- `ask_data: NDArray[np.float32]` - 卖盘数据，shape (200,)，100档 x 2（价格归一化 + 数量归一化）
- `trade_prices: NDArray[np.float32]` - 成交价格归一化，shape (100,)
- `trade_quantities: NDArray[np.float32]` - 成交数量归一化（带方向），shape (100,)，正数表示 taker 是买方，负数表示 taker 是卖方
- `tick_history_prices: NDArray[np.float32]` - Tick 历史价格归一化，shape (100,)，以第一个 tick 价格为基准，最新数据在末尾
- `tick_history_volumes: NDArray[np.float32]` - Tick 历史成交量归一化（带方向），shape (100,)，正=taker 买入为主，最新数据在末尾
- `tick_history_amounts: NDArray[np.float32]` - Tick 历史成交额归一化（带方向），shape (100,)，正=taker 买入为主，最新数据在末尾

**归一化方法：**
- 价格归一化：`(price - mid_price) / mid_price`，范围约 [-0.1, 0.1]
- 数量归一化：`log10(quantity + 1) / 10`，将 1e10 压缩到约 1.0
- 成交数量带方向归一化：`sign(qty) * log10(|qty| + 1) / 10`
- Tick 历史价格归一化：`(price - base_price) / base_price`，base_price 为第一个 tick 价格
- Tick 历史成交量归一化：`sign(vol) * log10(|vol| + 1) / 10`
- Tick 历史成交额归一化：`sign(amt) * log10(|amt| + 1) / 12`

### 2. 订单簿模块（orderbook/）

详见 [src/market/orderbook/CLAUDE.md](orderbook/CLAUDE.md)

**核心类：**
- `Order` - 订单数据类（Python）
- `OrderSide` - 订单方向枚举（BUY=1, SELL=-1）
- `OrderType` - 订单类型枚举（LIMIT=1, MARKET=2）
- `PriceLevel` - 价格档位类（Cython）
- `OrderBook` - 订单簿类（Cython）

**性能特征：**
| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| add_order | O(log N) | N 为档位数，SortedDict 插入 |
| cancel_order | O(log N) | N 为档位数，SortedDict 查找 |
| get_best_bid | O(1) | peekitem(-1) |
| get_best_ask | O(1) | peekitem(0) |
| get_mid_price | O(1) | 两次 O(1) 查询 |
| get_depth | O(levels) | 利用有序性，避免排序 |

### 3. 撮合引擎模块（matching/）

详见 [src/market/matching/CLAUDE.md](matching/CLAUDE.md)

**核心类：**
- `Trade` - 成交记录数据类（Python）
- `FastTrade` - 快速成交记录类（Cython）
- `MatchingEngine` - 撮合引擎（Python 实现）
- `FastMatchingEngine` - 快速撮合引擎（Cython 实现）
- `fast_match_orders` - 快速撮合核心函数（Cython）

**撮合规则：**
- **价格优先**：最优价格优先成交
- **时间优先**：同价格订单按时间顺序成交（FIFO）
- **限价单**：未成交部分挂在订单簿上
- **市价单**：吃对手盘直到完全成交或对手盘为空，未成交部分丢弃

**手续费模型：**
| Agent 类型 | 挂单费率 | 吃单费率 | 说明 |
|-----------|----------|----------|------|
| 高级散户 | 0.0002 (万2) | 0.0005 (万5) | 标准费率 |
| 做市商 | -0.0001 (负万1) | 0.0001 (万1) | Maker rebate |
| 噪声交易者 | 0 | 0 | 免手续费 |

### 4. 账户管理模块（account/）

详见 [src/market/account/CLAUDE.md](account/CLAUDE.md)

**核心类：**
- `Position` - 持仓类（Cython 加速）
- `Account` - 账户类（Python 实现）
- `FastAccount` - 快速账户类（Cython 实现）

**持仓更新逻辑：**
- 空仓开仓（多/空）
- 加仓（多加多/空加空）- 使用加权平均法计算新均价
- 减仓（多减/空减）- 实现盈亏
- 完全平仓 - 实现盈亏，重置均价
- 反向开仓（多转空/空转多）- 先平仓实现盈亏，再开新仓

**保证金和强平机制：**
- 占用保证金：`margin_used = |quantity| x current_price / leverage`
- 净值：`equity = balance + unrealized_pnl`
- 保证金率：`margin_ratio = equity / (|quantity| x current_price)`
- 强平触发条件：`margin_ratio < maintenance_margin_rate`

### 5. ADL 自动减仓模块（adl/）

详见 [src/market/adl/CLAUDE.md](adl/CLAUDE.md)

**核心类：**
- `ADLCandidate` - ADL 候选者信息
- `ADLManager` - ADL 管理器

**排名公式：**
| 情况 | 公式 |
|------|------|
| 盈利时（PnL% > 0） | `adl_score = PnL% x 有效杠杆` |
| 亏损时（PnL% <= 0） | `adl_score = PnL% / 有效杠杆` |

**ADL 成交价格：** 直接使用当前市场价格，不使用破产价格

**执行流程（三阶段）：**
1. **阶段1：预计算 ADL 候选清单**（Tick 开始时）
   - 遍历所有 Agent 和噪声交易者，计算 ADL 分数
   - 筛选盈利者（pnl_percent > 0）
   - 按持仓方向分类（多头/空头）
   - 按分数降序排序

2. **阶段2：强平市价单执行**
   - 撤销被淘汰 Agent 的所有挂单
   - 提交市价平仓单
   - 返回剩余未平仓数量

3. **阶段3：ADL 自动减仓**（如需要）
   - 选择对应方向的候选清单
   - 循环与候选成交
   - 兜底处理（强制清零）

### 6. 噪声交易者模块（noise_trader/）

详见 [src/market/noise_trader/CLAUDE.md](noise_trader/CLAUDE.md)

**核心类：**
- `NoiseTraderAccount` - 噪声交易者账户类
- `NoiseTrader` - 噪声交易者类
- `create_noise_traders()` - 工厂函数

**噪声交易者行为：**
- 200个独立噪声交易者
- 每 tick 以 50% 概率行动
- 行动时 50% 买 / 50% 卖，通过市价单撮合
- 下单量：`max(1, int(lognormvariate(mu=9.5, sigma=1.0)))`

**噪声交易者特点：**
- 初始资金 1e18（视为无限资金）
- 不触发强平检查
- 手续费为 0（maker 和 taker 费率均为 0）
- 可作为 ADL 对手方
- 有完整的持仓和 PnL 跟踪

## 市场引擎与训练引擎的交互

### 数据流向

```
Trainer（训练引擎）
  |
1. 创建市场组件
  |-- OrderBook（订单簿）
  |-- MatchingEngine（撮合引擎）
  |-- Account/Position（账户和持仓）
  |-- ADLManager（ADL 管理器）
  |-- NoiseTrader（噪声交易者）
  |
2. Episode 循环
  |-- 重置所有 Agent 账户
  |-- 重置噪声交易者状态
  |-- 重置市场状态和订单簿
  |-- Tick 循环
      |-- 强平检查（三阶段）
      |   |-- 阶段1：预计算 ADL 候选清单
      |   |-- 阶段2：强平市价单执行
      |   |-- 阶段3：ADL 自动减仓（如需要）
      |-- 噪声交易者行动
      |-- 计算归一化市场状态（所有 Agent 共用）
      |-- 随机打乱 Agent 顺序
      |-- 并行决策（神经网络推理）
      |-- 串行执行（订单提交）
          |-- MatchingEngine.process_order()
          |   |-- OrderBook.add_order()
          |   |-- fast_match_orders()
          |   |-- 返回 Trade 列表
          |-- Account.on_trade()
              |-- Position.update()
              |-- 更新余额
              |-- 累加 maker_volume
  |
3. NEAT 进化阶段
  |-- 计算适应度（基于 PnL）
  |-- 执行 NEAT 进化算法
  |-- 从新基因组创建 Agent
```

### 接口契约

**市场引擎向训练引擎提供：**
1. **OrderBook** - 订单簿实例，用于深度查询
2. **MatchingEngine** - 撮合引擎，用于订单提交和撮合
3. **Account/Position** - 账户和持仓，用于状态查询和更新
4. **ADLManager** - ADL 管理器，用于风险控制
5. **NoiseTrader** - 噪声交易者实例，用于市场随机性
6. **NormalizedMarketState** - 归一化市场状态，用于 Agent 观察

**训练引擎向市场引擎提供：**
1. **Agent 订单** - Agent 提交的订单请求
2. **当前价格** - 用于强平检测、ADL 计算
3. **配置参数** - Agent 类型、费率、杠杆等

## 性能优化

### Cython 加速模块

| 模块 | Python 实现 | Cython 实现 | 加速比 |
|------|------------|------------|--------|
| OrderBook | - | orderbook.pyx | 10-100x |
| Position | - | position.pyx | 10x |
| Account | account.py | fast_account.pyx | 10x |
| MatchingEngine | matching_engine.py | fast_matching.pyx | 1.1x |

### 并行化策略

- **Agent 决策**：16 个 worker 线程池（神经网络推理）
- **Agent 创建**：8 个 worker 线程池

### 内存管理

- 预分配输入缓冲区
- 订单簿深度缓存
- 使用 `__slots__` 优化内存占用
- 定期清理历史数据

## 数据类型约定

### 整数类型

- `Order.quantity: int` - 订单数量（整数）
- `Order.filled_quantity: int` - 已成交数量（整数）
- `Trade.quantity: int` - 成交数量（整数）
- `Position.quantity: int` - 持仓数量（整数，正数为多仓，负数为空仓）
- `PriceLevel.total_quantity: long long` - 价格档位总数量（64 位整数）

**设计原因：**
- 避免浮点累积误差
- 确保撮合精度
- 支持大量订单累积

### 浮点类型

- `Order.price: float` - 委托价格
- `Trade.price: float` - 成交价格
- `Position.avg_price: double` - 平均开仓价格（Cython）
- `Account.balance: float` - 余额
- `Account.initial_balance: float` - 初始余额

## 配置依赖

市场引擎依赖以下配置：

### MarketConfig

- `tick_size: float` - 最小价格变动单位（默认 0.01）

### AgentConfig

- `initial_balance: float` - 初始余额
- `leverage: float` - 杠杆倍数
- `maintenance_margin_rate: float` - 维持保证金率
- `maker_fee_rate: float` - 挂单手续费率
- `taker_fee_rate: float` - 吃单手续费率

### NoiseTraderConfig

- `count: int` - 噪声交易者数量（默认 100）
- `action_probability: float` - 每个 tick 行动概率（默认 0.5）
- `quantity_mu: float` - 对数正态分布 mu 参数
- `quantity_sigma: float` - 对数正态分布 sigma 参数

## 依赖关系

### 外部依赖

- `numpy` - 数值计算
- `sortedcontainers.SortedDict` - 高性能排序字典
- `collections.OrderedDict` - 保持插入顺序的字典
- `dataclasses` - 数据类装饰器
- `random` - 随机数生成（噪声交易者）

### 内部依赖

- `src.config.config` - 配置类
- `src.core.log_engine.logger` - 日志系统

## 模块导出

```python
from src.market import NormalizedMarketState
from src.market.orderbook import Order, OrderSide, OrderType, OrderBook
from src.market.matching import MatchingEngine, FastMatchingEngine, Trade, FastTrade, fast_match_orders
from src.market.account import Account, FastAccount, Position
from src.market.adl import ADLManager, ADLCandidate
from src.market.noise_trader import NoiseTrader, NoiseTraderAccount, create_noise_traders
```

## 注意事项

### 线程安全

- 所有市场组件**不是线程安全的**
- 多线程环境需要外部加锁
- Trainer 串行执行订单提交和账户更新

### 浮点精度

- 价格使用双重舍入归一化到 tick_size 的整数倍
- 数量使用整数类型避免累积误差
- 保证金计算注意除零情况

### 内存管理

- Episode 结束时调用 `OrderBook.clear()` 清空订单簿
- Episode 结束时调用 `Account.reset()` 或 `FastAccount.reset()` 重置账户
- 定期清理历史数据防止内存泄漏

### ADL 边界情况

- 净值为零或负数时，有效杠杆设为无穷大
- 无持仓时不参与 ADL
- 兜底处理确保被淘汰者仓位清零

### 噪声交易者约束

- 噪声交易者不触发强平检查
- 噪声交易者手续费为 0
- 噪声交易者订单 ID 使用负数空间
- 噪声交易者可作为 ADL 对手方

## 子模块文档

各子模块的详细文档：

- [订单簿模块](orderbook/CLAUDE.md) - OrderBook、PriceLevel、Order
- [撮合引擎模块](matching/CLAUDE.md) - MatchingEngine、Trade、撮合规则
- [账户管理模块](account/CLAUDE.md) - Account、Position、保证金计算
- [ADL 模块](adl/CLAUDE.md) - ADLManager、ADLCandidate、排名算法
- [噪声交易者模块](noise_trader/CLAUDE.md) - NoiseTrader、NoiseTraderAccount
