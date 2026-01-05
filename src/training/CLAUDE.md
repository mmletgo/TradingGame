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
- `create_agents()` - 从基因组列表创建 Agent（小批量串行，大批量并行）
- `_create_single_agent()` - 创建单个 Agent（线程安全）
- `evaluate()` - 评估种群适应度并排序
- `evolve()` - 执行一代 NEAT 进化，捕获 RuntimeError 并在进化失败时自动重置种群
- `_reset_neat_population()` - 当 NEAT 进化失败时，创建全新的随机种群
- `reset_agents()` - 重置所有 Agent 账户
- `_get_shared_executor()` - 获取类级别共享线程池（所有 Population 实例共享）
- `shutdown_executor()` - 关闭共享线程池（类方法）

**多核并行化：**
- 使用类级别共享 ThreadPoolExecutor（16个worker）
- 小批量（<50）串行创建，避免线程池开销
- 大批量并行调用 `Brain.from_genome`，按索引排序保证顺序

### Trainer (trainer.py)

管理整体训练流程，协调种群和撮合引擎。

**主要功能：**
- 初始化训练环境（创建种群、撮合引擎）
- 管理训练生命周期（tick、episode）
- 处理成交和强平
- 保存/加载检查点
- 支持暂停/恢复/停止控制

**关键方法：**
- `setup()` - 初始化训练环境，创建种群、撮合引擎、ADL 管理器，初始化鲶鱼（如启用），初始化 EMA 平滑价格
- `_init_ema_price()` - 初始化 EMA 平滑价格（在 episode 开始时调用）
- `_update_ema_price()` - 更新 EMA 平滑价格（每 tick 调用）
- `_aggregate_tick_trades()` - 聚合本 tick 的成交量和成交额（符号+总量方式）
- `_calculate_catfish_initial_balance()` - 计算鲶鱼初始资金（做市商杠杆后资金 - 其他物种杠杆后资金）/ 3
- `_register_all_agents()` - 注册所有 Agent 的费率到撮合引擎
- `_build_agent_map()` - 构建 Agent ID 到 Agent 对象的映射表（O(1) 查找）
- `_build_execution_order()` - 构建 Agent 执行顺序列表（初始化时收集所有 Agent，实际执行时每 tick 随机打乱）
- `_cancel_agent_orders()` - 撤销指定 Agent 的所有挂单（做市商撤多单，普通 Agent 撤单个挂单）
- `_execute_liquidation_market_order()` - 执行强平市价单，提交市价单平仓，若市价单无法完全成交则返回剩余数量（调用前必须先撤销挂单）
- `_execute_adl()` - 执行 ADL 自动减仓，在循环中处理 ADL 成交（使用预计算的候选清单、更新账户、更新 position_qty）
- `_check_catfish_liquidation()` - 检查鲶鱼强平，鲶鱼强平后设置 `_catfish_liquidated` 标志（触发 episode 结束）
- `_execute_catfish_liquidation()` - 执行鲶鱼强平市价单
- `_update_pop_total_counts()` - 更新各种群总数（在 setup/evolve/load_checkpoint 后调用）
- `_should_end_episode_early()` - O(1) 检查是否满足提前结束条件，返回 `tuple[str, AgentType | None] | None`
  - 触发条件1：任意种群存活少于初始值的 1/4，返回 `("population_depleted", agent_type)`
  - 触发条件2：订单簿只有单边挂单（只有 bid 或只有 ask），返回 `("one_sided_orderbook", None)`
  - 不触发时返回 `None`
- `_compute_normalized_market_state()` - 向量化计算归一化市场状态，使用 EMA 平滑后的 mid_price
- `run_tick()` - 执行单个 tick，鲶鱼行动在Agent决策之前执行
- `run_episode()` - 运行完整 episode（重置、运行、进化），若满足提前结束条件则提前结束
- `train()` - 主训练循环
- `save_checkpoint()` / `load_checkpoint()` - 检查点管理

**性能优化：**
- 使用 `deque(maxlen=100)` 自动管理成交记录，避免列表切片开销
- 使用 `agent_map` 映射表实现 O(1) Agent 查找
- 使用 `_pop_total_counts` 和 `_pop_liquidated_counts` 计数器实现 O(1) 种群淘汰检查，避免每 tick 遍历
- 使用 `agent_execution_order` 预构建执行顺序，合并决策/执行和强平检查循环
- 向量化市场状态计算，使用 NumPy 数组操作替代 Python 循环

**价格稳定机制（EMA 平滑）：**
- 使用 EMA（指数移动平均）平滑 mid_price，减缓价格变化传导速度
- 公式：`smooth_mid_price = α × current_mid_price + (1-α) × prev_smooth_mid_price`
- 参数 `ema_alpha`（默认 0.1）可通过 `MarketConfig.ema_alpha` 配置
- Agent 报价、归一化计算、强平检查使用 `smooth_mid_price`（保持一致性）
- ADL 计算使用实时 `last_price`（实际市场操作需要准确价格）
- 解决了价格在短时间内极端波动的正反馈循环问题

**多核并行化优化：**
- 种群初始化：串行创建种群，但每个种群内部的 Agent 创建是并行的（Agent维度并行）
- `_evolve_populations_parallel()` - 4个种群并行进化
- `_batch_decide_parallel()` - Agent 决策阶段并行执行（NEAT 的 Cython 代码释放 GIL）
- `_check_liquidations_vectorized()` - 向量化强平检查（NumPy 批量计算）
- 决策阶段并行，执行阶段串行（保证订单簿一致性）
- 线程池惰性初始化，`stop()` 时自动清理
- Population 类使用类级别共享线程池（16个worker，所有Population实例共享）处理 Agent 创建
- Trainer 类使用独立线程池（16个worker）处理进化和决策并行

## 训练流程

1. **初始化阶段** (`setup`)
   - 创建四个种群（散户/高级散户/庄家/做市商）
   - 创建撮合引擎
   - 创建 ADL 管理器
   - 注册所有 Agent 的费率到撮合引擎
   - 构建 Agent 映射表和执行顺序
   - 初始化 EMA 平滑价格（使用 initial_price）
   - 做市商建立初始流动性

2. **Episode 循环** (`run_episode`)
   - 重置所有 Agent 账户
   - 重置鲶鱼状态和强平标志
   - 重置市场状态（包括重置 EMA 平滑价格、价格历史和 tick 历史数据）
   - 重置各种群淘汰计数和重入保护集合
   - 运行 episode_length 个 tick
   - **提前结束条件**：
     - 鲶鱼被强平（立即结束 episode）
     - 任意种群存活少于初始值的 1/4（确保有足够的幸存者用于 NEAT 进化）
     - 订单簿只有单边挂单（确保市场流动性正常）
   - 各种群进化
   - 进化后重新注册新 Agent 的费率，重建映射表和执行顺序

3. **Tick 执行** (`run_tick`)

   **时序设计**：Agent 的下单操作影响的是下一个 tick，确保强平检查和数据采集使用同一价格

   - **Tick 开始（强平处理分三阶段）**：
     - 保存 tick 开始时的价格到 `tick_start_price`（供数据采集使用）
     - **阶段1（统一撤单）**：遍历所有 Agent 检查强平条件，收集需要淘汰的 Agent，**统一撤销这些 Agent 的所有挂单**
     - **阶段2（统一市价单平仓）**：遍历需要淘汰的 Agent，执行市价单平仓（不触发 ADL），收集需要 ADL 的 Agent
     - **阶段3（用最新价格计算 ADL 候选并执行）**：获取订单簿最新价格，计算 ADL 候选清单，执行 ADL
   - **Tick 过程**：
     - 鲶鱼行动（如果启用）
     - 向量化计算归一化市场状态
     - **随机打乱 Agent 执行顺序**（每 tick 完全随机，模拟真实环境）
     - Agent 并行决策 → 串行执行下单
     - 记录成交到 `recent_trades`，更新 maker 账户
   - **Tick 结束**：
     - 下单产生的价格变动效果在下个 tick 被感知
     - 记录当前价格到 `_price_history`（鲶鱼决策使用，最多保留1000个历史价格）
     - 记录 tick 历史数据（价格、成交量、成交额，最多保留100条）
     - 检查鲶鱼强平（鲶鱼强平则立即结束 episode）
     - 数据采集使用 `tick_start_price` 计算资产，与强平检查一致

## 强平与淘汰机制（爆仓即淘汰）

**强平即淘汰（Liquidation = Elimination）**：保证金率低于维持保证金率时触发，Agent 直接被淘汰

**Tick 开始时的三阶段强平处理**：
- **阶段1（统一撤单）**：遍历所有 Agent 检查强平条件，收集需要淘汰的 Agent，调用 `_cancel_agent_orders()` 统一撤销这些 Agent 的所有挂单
- **阶段2（统一市价单平仓）**：遍历需要淘汰的 Agent，调用 `_execute_liquidation_market_order()` 执行市价单平仓（不触发 ADL），标记淘汰并穿仓兜底，收集需要 ADL 的 Agent
- **阶段3（用最新价格计算 ADL 候选并执行）**：获取订单簿最新价格（强平市价单执行后的价格），用最新价格计算 ADL 候选清单，执行 ADL

**设计原因**：
- 先统一撤单可防止被淘汰的 Agent 在平仓过程中作为 maker 被成交，导致仓位增加
- **用最新价格计算 ADL 候选**：强平市价单执行后订单簿价格已变化，用最新价格计算候选确保 ADL 候选在当前价格下确实盈利，避免 candidate 因 ADL 出现负 balance

**`_execute_liquidation_market_order()` 执行流程**（阶段2）：
1. 获取 Agent 当前持仓方向和数量
2. 创建市价平仓单，调用撮合引擎处理
3. 成交后更新 Agent 账户（taker）和 maker 账户
4. 返回剩余未平仓数量和持仓方向

**ADL（自动减仓）**：市价强平无法完全成交时触发（阶段3）
1. **获取最新价格**：强平市价单执行后，订单簿价格已变化，获取最新价格
2. **用最新价格计算候选清单**：遍历所有存活 Agent，用最新价格计算 ADL 分数，筛选盈利候选者，按多头/空头分类并排序
3. **将鲶鱼加入候选清单**：遍历所有未被强平的鲶鱼，计算其 ADL 分数（使用庄家的杠杆率），盈利的鲶鱼按持仓方向加入对应候选清单
4. 使用候选清单（`_adl_long_candidates` 或 `_adl_short_candidates`）
   - 被强平方是多头 → 使用空头候选清单
   - 被强平方是空头 → 使用多头候选清单
5. 在 `_execute_adl()` 中直接循环处理：依次与候选对手方以最新市场价格成交，直至剩余仓位清零
6. 通过 `account.on_adl_trade()` 更新双方账户（含穿仓兜底）
7. **更新候选清单的 position_qty**：确保后续 ADL 不会重复使用已减掉的仓位
8. **兜底处理**：如果 ADL 无法完全清零仓位，强制清零被淘汰者的仓位

**重入保护与 ADL 候选管理**：
- 使用 `_eliminating_agents` 集合跟踪正在强平/淘汰过程中的 Agent
- 使用 `_adl_long_candidates` 和 `_adl_short_candidates` 存储预计算的 ADL 候选清单
  - 已提前筛选：未淘汰、本 tick 不会被淘汰、有持仓、盈利
  - 已计算 ADL 分数并排序
  - **包含 Agent 和鲶鱼**：盈利的鲶鱼也会被加入候选清单
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
- `config/neat_retail.cfg` - 散户（127 个输入节点，9 个输出节点）
- `config/neat_retail_pro.cfg` - 高级散户（907 个输入节点，9 个输出节点）
- `config/neat_whale.cfg` - 庄家（907 个输入节点，9 个输出节点）
- `config/neat_market_maker.cfg` - 做市商（934 个输入节点，21 个输出节点）

散户只能看到买卖各10档订单簿和最近10笔成交，高级散户和庄家可以看到完整的100档订单簿和100笔成交。

**关键配置参数（防止种群灭绝）：**
- `reset_on_extinction = True` - 当所有物种灭绝时自动重置种群
- `survival_threshold = 0.5` - 每个物种的前 50% 个体可以参与繁殖
- `compatibility_threshold = 4.0` - 物种兼容性阈值，较高的值减少物种数量
- `elitism = 1` - 每个物种保留 1 个最优个体不变
- `species_elitism = 2` - 保留 2 个最优物种不被移除

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

## 鲶鱼机制

鲶鱼（Catfish）是规则驱动的市场参与者，用于增加市场波动性。

**特点：**
- 不使用神经网络，规则驱动
- **资金计算**：每条鲶鱼初始资金 = (做市商杠杆后资金 - 其他物种杠杆后资金) / 3，支持多模式（3条鲶鱼同时运行）和单模式（1条鲶鱼）
- 参与强平和 ADL 机制（作为盈利方可作为 ADL 候选）
- 鲶鱼被强平后 Episode 立即结束
- 下单量按盘口计算（吃掉前3档），不按自身资金计算
- 手续费为 0（maker 和 taker）
- 在所有 Agent 之前行动，强平检查在所有 Agent 之后
- 使用庄家的杠杆率和维持保证金率
- 使用 `_price_history`（最多1000个历史价格）进行决策

**行为模式：**
- `trend_following`：趋势追踪，顺势推动价格
- `cycle_swing`：周期摆动，交替买卖形成波动
- `mean_reversion`：逆势操作，均值回归

**配置：**
通过 `CatfishConfig` 配置，包括触发阈值、模式选择、多模式开关等参数。

详见：`src/market/catfish/CLAUDE.md`
