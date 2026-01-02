# Training 模块

## 模块概述

训练模块负责管理 NEAT 进化训练流程，包括种群管理和训练协调。

## 文件结构

- `__init__.py` - 模块导出
- `population.py` - 种群管理类
- `trainer.py` - 训练器类

## 核心类

### Population (population.py)

管理特定类型 Agent 的种群，支持从 NEAT 基因组创建 Agent。

**主要功能：**
- 创建和管理 NEAT 种群
- 从基因组创建对应类型的 Agent（散户/高级散户/庄家/做市商）
- 向量化评估适应度
- 执行 NEAT 进化算法
- 重置 Agent 账户状态

**关键方法：**
- `create_agents()` - 从基因组列表创建 Agent
- `evaluate()` - 评估种群适应度并排序
- `evolve()` - 执行一代 NEAT 进化
- `reset_agents()` - 重置所有 Agent 账户

### Trainer (trainer.py)

管理整体训练流程，协调种群和撮合引擎。训练模式使用直接调用，绕过事件系统。

**主要功能：**
- 初始化训练环境（创建种群、撮合引擎）
- 管理训练生命周期（tick、episode）
- 直接调用模式处理成交和强平
- 保存/加载检查点
- 支持暂停/恢复/停止控制

**关键方法：**
- `setup()` - 初始化训练环境（训练模式不订阅事件），创建 ADL 管理器
- `_register_all_agents()` - 注册所有 Agent 的费率到撮合引擎
- `_build_agent_map()` - 构建 Agent ID 到 Agent 对象的映射表（O(1) 查找）
- `_build_execution_order()` - 构建 Agent 执行顺序列表（做市商->庄家->高级散户->散户）
- `_handle_liquidation_direct()` - 直接处理强平（训练模式），提交市价单平仓；若市价单无法完全成交则触发 ADL
- `_execute_adl()` - 执行 ADL 自动减仓，计算破产价格、获取候选对手方、执行减仓并更新账户
- `_check_elimination()` - 检查个体淘汰条件（净值/初始资金 < 10%），淘汰时标记 Agent 并增加种群淘汰计数
- `_update_pop_total_counts()` - 更新各种群总数（在 setup/evolve/load_checkpoint 后调用）
- `_any_population_eliminated()` - O(1) 检查是否有任一种群被全部淘汰，返回被淘汰的种群类型
- `_compute_normalized_market_state()` - 向量化计算归一化市场状态
- `run_tick()` - 执行单个 tick（直接调用模式），绕过事件系统
- `run_episode()` - 运行完整 episode（重置、运行、进化），若任一种群全部被淘汰则提前结束
- `train()` - 主训练循环
- `save_checkpoint()` / `load_checkpoint()` - 检查点管理

**性能优化：**
- 使用 `deque(maxlen=100)` 自动管理成交记录，避免列表切片开销
- 使用 `agent_map` 映射表实现 O(1) Agent 查找
- 使用 `_pop_total_counts` 和 `_pop_liquidated_counts` 计数器实现 O(1) 种群淘汰检查，避免每 tick 遍历
- 使用 `agent_execution_order` 预构建执行顺序，合并决策/执行和强平检查循环
- 向量化市场状态计算，使用 NumPy 数组操作替代 Python 循环
- **直接调用模式**：训练时绕过事件系统，直接调用撮合引擎和 Agent 方法

## 训练流程

1. **初始化阶段** (`setup`)
   - 创建三个种群（散户/庄家/做市商）
   - 创建撮合引擎
   - 创建 ADL 管理器
   - 注册所有 Agent 的费率到撮合引擎
   - 构建 Agent 映射表和执行顺序
   - 做市商建立初始流动性（直接调用模式）

2. **Episode 循环** (`run_episode`)
   - 重置所有 Agent 账户
   - 重置市场状态
   - 运行 episode_length 个 tick
   - **提前结束条件**：若任一种群（散户/庄家/做市商）全部被淘汰，则立即结束当前 episode
   - 各种群进化
   - 进化后重新注册新 Agent 的费率，重建映射表和执行顺序

3. **Tick 执行** (`run_tick` - 直接调用模式)
   - 向量化计算归一化市场状态
   - 使用预构建的执行顺序列表遍历所有 Agent：
     - 跳过已淘汰的 Agent
     - 决策（传入市场状态和订单簿）
     - 直接执行动作（`execute_action_direct`，绕过事件系统）
     - 记录成交到 `recent_trades`
     - **对 maker 检查强平/淘汰条件**（成交可能导致 maker 亏损）
     - 检查 taker 强平条件，触发 `_handle_liquidation_direct`（仅平仓）
     - 检查 taker 淘汰条件，触发 `_check_elimination`（资金不足10%时淘汰）

## 直接调用模式

训练模式绕过事件系统，直接调用以提高性能：

### Agent 方法
- `execute_action_direct(action, params, matching_engine)` - 直接执行动作，返回成交列表
- `_place_limit_order_direct()` - 直接下限价单
- `_place_market_order_direct()` - 直接下市价单
- `_process_trades_direct()` - 直接处理成交，更新账户

### 撮合引擎方法
- `process_order_direct(order)` - 直接处理订单，不发布事件
- `cancel_order_direct(order_id)` - 直接撤单

### Trainer 方法
- `_init_market()` - 直接调用做市商初始化市场
- `run_tick()` - 直接调用 Agent 和撮合引擎
- `_handle_liquidation_direct()` - 直接处理强平

## 强平与淘汰机制

**强平（Liquidation）**：保证金率低于维持保证金率时触发
1. `_handle_liquidation_direct()` 创建市价平仓单，直接调用撮合引擎处理
2. 成交后直接更新 Agent 账户
3. **若市价单无法完全成交**（订单簿流动性不足），剩余仓位触发 ADL 机制
4. 强平后 Agent **可以继续交易**（不自动淘汰）

**ADL（自动减仓）**：市价强平无法完全成交时触发
1. 计算被强平方的破产价格
2. 筛选对手方候选（持有反向持仓的 Agent，**包括已淘汰的**），按盈利比例排序
3. 依次与候选对手方以破产价格成交，直至剩余仓位清零
4. 通过 `account.on_adl_trade()` 更新双方账户
5. 由于多空仓位完全对等，理论上不会出现候选不足的情况

**淘汰（Elimination）**：净值/初始资金 < 10% 时触发
1. `_check_elimination()` 检查淘汰条件
2. 满足条件时**先强平其持有的仓位**（通过 `_handle_liquidation_direct`，包含市价单和 ADL）
3. 然后将 Agent 的 `is_liquidated` 标志设为 True
4. 被淘汰的 Agent 在本轮 episode 剩余时间内无法执行任何动作（`run_tick` 跳过，`execute_action_direct` 返回空列表）
5. 在下一轮 episode 开始时，`reset_agents()` 会重置 `is_liquidated` 标志

## 依赖关系

- `src.bio.agents` - Agent 类
- `src.config.config` - 配置类
- `src.core.event_engine` - 事件系统（保留用于调试/UI模式）
- `src.core.log_engine` - 日志系统
- `src.market.adl` - ADL 管理器
- `src.market.matching` - 撮合引擎
- `src.market.orderbook` - 订单簿

## NEAT 配置

不同 Agent 类型使用不同的 NEAT 配置文件（由 Population 自动选择）：
- `config/neat_retail.cfg` - 散户（67 个输入节点，9 个输出节点）
- `config/neat_retail_pro.cfg` - 高级散户（607 个输入节点，9 个输出节点）
- `config/neat_whale.cfg` - 庄家（607 个输入节点，9 个输出节点）
- `config/neat_market_maker.cfg` - 做市商（634 个输入节点，22 个输出节点）

散户只能看到买卖各10档订单簿和最近10笔成交，高级散户和庄家可以看到完整的100档订单簿和100笔成交。

**注意：** NEAT 配置文件中的 `pop_size` 会被 `AgentConfig.count` 动态覆盖，即种群数量由脚本中的配置决定，而非 NEAT 配置文件。

## 启动脚本

训练脚本位于 `scripts/` 目录：
- `train_noui.py` - 无 UI 高性能训练模式

### train_noui.py 使用方法

```bash
# 基本训练（100 个 episode）
python scripts/train_noui.py --episodes 100

# 自定义参数
python scripts/train_noui.py --episodes 500 --episode-length 1000 --checkpoint-interval 50

# 从检查点恢复训练
python scripts/train_noui.py --resume checkpoints/ep_50.pkl --episodes 100
```

命令行参数：
- `--episodes`: 训练的 episode 数量（默认: 100）
- `--episode-length`: 每个 episode 的 tick 数量（默认: 1000）
- `--checkpoint-interval`: 检查点保存间隔（默认: 10，0 表示不保存）
- `--resume`: 从指定检查点恢复训练
- `--config-dir`: 配置文件目录（默认: config）
- `--log-dir`: 日志目录（默认: logs）
