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
- 从基因组创建对应类型的 Agent（散户/庄家/做市商）
- 向量化评估适应度
- 执行 NEAT 进化算法
- 重置 Agent 账户状态

**关键方法：**
- `create_agents()` - 从基因组列表创建 Agent
- `evaluate()` - 评估种群适应度并排序
- `evolve()` - 执行一代 NEAT 进化
- `reset_agents()` - 重置所有 Agent 账户

### Trainer (trainer.py)

管理整体训练流程，协调种群、撮合引擎和事件系统。

**主要功能：**
- 初始化训练环境（创建种群、撮合引擎）
- 管理训练生命周期（tick、episode）
- 处理成交和强平事件
- 保存/加载检查点
- 支持暂停/恢复/停止控制

**关键方法：**
- `setup()` - 初始化训练环境
- `run_tick()` - 执行单个 tick（按做市商->庄家->散户顺序）
- `run_episode()` - 运行完整 episode（重置、运行、进化）
- `train()` - 主训练循环
- `save_checkpoint()` / `load_checkpoint()` - 检查点管理

## 训练流程

1. **初始化阶段** (`setup`)
   - 创建三个种群（散户/庄家/做市商）
   - 创建撮合引擎
   - 订阅成交和强平事件
   - 做市商建立初始流动性

2. **Episode 循环** (`run_episode`)
   - 重置所有 Agent 账户
   - 重置市场状态
   - 运行 episode_length 个 tick
   - 各种群进化

3. **Tick 执行** (`run_tick`)
   - 发布 TICK_START 事件
   - 按顺序执行：做市商 -> 庄家 -> 散户
   - 检查强平条件
   - 发布 TICK_END 事件

## 依赖关系

- `src.bio.agents` - Agent 类
- `src.config.config` - 配置类
- `src.core.event_engine` - 事件系统
- `src.core.log_engine` - 日志系统
- `src.market.matching` - 撮合引擎
- `src.market.orderbook` - 订单簿
