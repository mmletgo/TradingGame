# Market 模块

## 模块概述

市场模块负责交易市场引擎的核心功能，包括订单簿管理、撮合引擎、账户管理和市场状态数据。

## 文件结构

- `__init__.py` - 模块导出
- `market_state.py` - 归一化市场状态数据类
- `orderbook/` - 订单簿实现（Cython 加速）
- `matching/` - 撮合引擎
- `account/` - 账户管理（持仓、余额、保证金、强平）

## 核心类

### NormalizedMarketState (market_state.py)

预计算的归一化市场数据，用于缓存每个 tick 的公共市场数据，避免每个 Agent 重复计算。

**属性：**
- `mid_price: float` - 中间价（用于归一化计算的参考价格）
- `tick_size: float` - 最小价格变动单位
- `bid_data: NDArray[np.float32]` - 买盘数据，shape (200,)，100档 × 2（价格归一化 + 数量）
- `ask_data: NDArray[np.float32]` - 卖盘数据，shape (200,)，100档 × 2（价格归一化 + 数量）
- `trade_prices: NDArray[np.float32]` - 成交价格归一化，shape (100,)
- `trade_quantities: NDArray[np.float32]` - 成交数量（带方向），shape (100,)，正数表示 taker 是买方，负数表示 taker 是卖方

**使用场景：**
在每个 tick 开始时由 Trainer 预计算一次，然后传递给所有 Agent 使用，避免重复计算订单簿数据的归一化。

## 子模块

### orderbook/

订单簿实现，使用 Cython 加速，支持买卖各 100 档。

### matching/

撮合引擎，实现价格优先、时间优先的撮合规则。

### account/

账户管理，包括：
- 持仓管理
- 余额管理
- 保证金计算
- 强平逻辑

## 依赖关系

- `numpy` - 数值计算
