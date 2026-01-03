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

管理整体训练流程，协调种群和撮合引擎。

**主要功能：**
- 初始化训练环境（创建种群、撮合引擎）
- 管理训练生命周期（tick、episode）
- 处理成交和强平
- 保存/加载检查点
- 支持暂停/恢复/停止控制

**关键方法：**
- `setup()` - 初始化训练环境，创建 ADL 管理器
- `_register_all_agents()` - 注册所有 Agent 的费率到撮合引擎
- `_build_agent_map()` - 构建 Agent ID 到 Agent 对象的映射表（O(1) 查找）
- `_build_execution_order()` - 构建 Agent 执行顺序列表（做市商->庄家->高级散户->散户）
- `_cancel_agent_orders()` - 撤销指定 Agent 的所有挂单（做市商撤多单，普通 Agent 撤单个挂单）
- `_handle_liquidation()` - 处理强平，提交市价单平仓，若市价单无法完全成交则触发 ADL，强平完成后淘汰 Agent（调用前必须先撤销挂单）
- `_execute_adl()` - 执行 ADL 自动减仓，在循环中处理 ADL 成交（使用预计算的候选清单、更新账户、更新 position_qty），ADL 后检查参与者的强平条件
- `_update_pop_total_counts()` - 更新各种群总数（在 setup/evolve/load_checkpoint 后调用）
- `_any_population_eliminated()` - O(1) 检查是否有任一种群被全部淘汰，返回被淘汰的种群类型
- `_compute_normalized_market_state()` - 向量化计算归一化市场状态
- `run_tick()` - 执行单个 tick
- `run_episode()` - 运行完整 episode（重置、运行、进化），若任一种群全部被淘汰则提前结束
- `train()` - 主训练循环
- `save_checkpoint()` / `load_checkpoint()` - 检查点管理

**性能优化：**
- 使用 `deque(maxlen=100)` 自动管理成交记录，避免列表切片开销
- 使用 `agent_map` 映射表实现 O(1) Agent 查找
- 使用 `_pop_total_counts` 和 `_pop_liquidated_counts` 计数器实现 O(1) 种群淘汰检查，避免每 tick 遍历
- 使用 `agent_execution_order` 预构建执行顺序，合并决策/执行和强平检查循环
- 向量化市场状态计算，使用 NumPy 数组操作替代 Python 循环

## 训练流程

1. **初始化阶段** (`setup`)
   - 创建三个种群（散户/庄家/做市商）
   - 创建撮合引擎
   - 创建 ADL 管理器
   - 注册所有 Agent 的费率到撮合引擎
   - 构建 Agent 映射表和执行顺序
   - 做市商建立初始流动性

2. **Episode 循环** (`run_episode`)
   - 重置所有 Agent 账户
   - 重置市场状态
   - 运行 episode_length 个 tick
   - **提前结束条件**：若任一种群（散户/庄家/做市商）全部被淘汰，则立即结束当前 episode
   - 各种群进化
   - 进化后重新注册新 Agent 的费率，重建映射表和执行顺序

3. **Tick 执行** (`run_tick`)

   **时序设计**：Agent 的下单操作影响的是下一个 tick，确保强平检查和数据采集使用同一价格

   - **Tick 开始（强平处理分两阶段）**：
     - 保存 tick 开始时的价格到 `tick_start_price`（供数据采集使用）
     - **阶段1**：遍历所有 Agent 检查强平条件，收集需要淘汰的 Agent，**统一撤销这些 Agent 的所有挂单**
     - **预计算 ADL 候选清单**：生成多头/空头两个候选清单（`_adl_long_candidates`、`_adl_short_candidates`），提前筛选（未淘汰、本 tick 不会被淘汰、有持仓、盈利）、计算 ADL 分数并排序，避免后续重复计算
     - **阶段2**：遍历需要淘汰的 Agent，执行平仓（因挂单已撤，不会作为 maker 被成交）
   - **Tick 过程**：
     - 向量化计算归一化市场状态
     - Agent 按顺序决策和下单（做市商→庄家→高级散户→散户）
     - 记录成交到 `recent_trades`，更新 maker 账户（**跳过已淘汰的 agent**，防止资产异常增加）
   - **Tick 结束**：
     - 下单产生的价格变动效果在下个 tick 被感知
     - 数据采集使用 `tick_start_price` 计算资产，与强平检查一致

## 强平与淘汰机制（爆仓即淘汰）

**强平即淘汰（Liquidation = Elimination）**：保证金率低于维持保证金率时触发，Agent 直接被淘汰

**Tick 开始时的两阶段强平处理**：
- **阶段1（统一撤单）**：遍历所有 Agent 检查强平条件，收集需要淘汰的 Agent，调用 `_cancel_agent_orders()` 统一撤销这些 Agent 的所有挂单
- **阶段2（统一平仓）**：遍历需要淘汰的 Agent，调用 `_handle_liquidation(skip_cancel_orders=True)` 执行平仓

**设计原因**：先统一撤单可防止被淘汰的 Agent 在平仓过程中作为 maker 被成交，导致仓位增加

**`_handle_liquidation()` 执行流程**：
1. 重入保护检查（防止同一 Agent 被多次处理）
2. 若 `skip_cancel_orders=False`，撤销所有挂单（ADL 递归调用时需要）：
   - 普通 Agent（散户/庄家）：撤销 `pending_order_id`
   - **做市商**：调用 `_cancel_all_orders()` 撤销所有买卖挂单
3. 创建市价平仓单，调用撮合引擎处理
4. 成交后更新 Agent 账户
5. **若市价单无法完全成交**（订单簿流动性不足），剩余仓位触发 ADL 机制
6. 强平完成后**标记 `is_liquidated = True`**，Agent 被淘汰
7. 处理穿仓（余额为负时归零），验证仓位已清零
8. 移除重入保护标记

**ADL（自动减仓）**：市价强平无法完全成交时触发
1. **Tick 开始时预计算**：Trainer 遍历所有 Agent，筛选盈利的候选者，计算 ADL 分数，按多头/空头分类并排序
2. 强平时使用预计算的候选清单（`_adl_long_candidates` 或 `_adl_short_candidates`）
   - 被强平方是多头 → 使用空头候选清单
   - 被强平方是空头 → 使用多头候选清单
3. 在 `_execute_adl()` 中直接循环处理：依次与候选对手方以当前市场价格成交，直至剩余仓位清零
4. 通过 `account.on_adl_trade()` 更新双方账户
5. **更新候选清单的 position_qty**：确保后续 ADL 不会重复使用已减掉的仓位
6. **ADL 成交后处理**：
   - 检查参与者的强平条件（ADL 可能导致 candidate 爆仓）
   - 如果 candidate 触发强平条件，调用 `_handle_liquidation()` 淘汰该 candidate
   - 如果 ADL 无法完全清零仓位，强制清零被淘汰者的仓位（兜底处理）
7. 由于多空仓位完全对等，理论上不会出现候选不足的情况

**重入保护与 ADL 候选管理**：
- 使用 `_eliminating_agents` 集合跟踪正在强平/淘汰过程中的 Agent
- 使用 `_adl_long_candidates` 和 `_adl_short_candidates` 存储预计算的 ADL 候选清单
  - 已提前筛选：未淘汰、本 tick 不会被淘汰、有持仓、盈利
  - 已计算 ADL 分数并排序
  - ADL 成交后动态更新 `position_qty`，避免重复使用已减掉的仓位
- 防止在递归 ADL 过程中同一 Agent 被多次处理
- 防止正在强平的 Agent 作为 maker 更新仓位（导致仓位增加）
- 强平完成后从集合中移除

**淘汰后状态**：
- 被淘汰的 Agent 在本轮 episode 剩余时间内无法执行任何动作（`run_tick` 跳过，`execute_action` 返回空列表）
- 在下一轮 episode 开始时，`reset_agents()` 会重置 `is_liquidated` 标志

## 依赖关系

- `src.bio.agents` - Agent 类
- `src.config.config` - 配置类
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

**注意：**
- NEAT 配置文件中的 `pop_size` 会被 `AgentConfig.count` 动态覆盖，即种群数量由脚本中的配置决定，而非 NEAT 配置文件。
- **Agent ID 唯一性**：每个种群类型使用不同的 ID 偏移量（散户=0，高级散户=1M，庄家=2M，做市商=3M），确保不同种群的 agent_id 全局唯一，避免订单 ID 冲突。

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
