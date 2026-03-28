# RL 训练模块

## 概述

将 NEAT 进化出的最优基因组转为 PyTorch 模型，使用 PPO 算法在真实市场数据回放环境中进行微调训练。

## 文件说明

### `__init__.py`
模块入口。

### `config.py` — RLConfig
RL 训练的全部配置参数，使用 dataclass 定义：
- **训练超参**：total_timesteps, learning_rate, gamma, gae_lambda, clip_range, n_steps, batch_size, n_epochs
- **正则化**：ent_coef, vf_coef, max_grad_norm
- **策略网络**：initial_log_std, critic_hidden_sizes
- **微调策略**：freeze_actor_steps（先冻结 actor 只训练 critic）, actor_lr_scale（解冻后 actor 学习率缩放）
- **IO**：checkpoint_path, agent_type, output_dir, log_interval, save_interval

### `converter.py` — NEAT -> PyTorch 转换器

#### NEATNetwork(nn.Module)
精确复刻 NEAT 前馈网络的 PyTorch 模块。

**计算公式**：`output_i = tanh(sum(w_ij * x_j) + bias_i) * response_i`

**关键设计**：
- 所有节点按拓扑排序依次计算，确保依赖关系正确
- 节点映射：input_ids 在前，node_order（隐藏+输出）在后，统一编入 values 数组
- weights/biases/responses 均为 `nn.Parameter`，支持梯度反传用于 RL 微调
- 预计算 `_node_src_indices` 和 `_node_weight_indices` 张量，避免 forward 时重复构建
- 支持单样本 (n_inputs,) 和批量 (batch, n_inputs) 输入

#### NEATtoPyTorchConverter
静态工具类，提供三个核心方法：

1. **convert(genome, neat_config)** — 单基因组转换
   - 提取 input_ids/output_ids、enabled 连接、节点参数
   - 调用 `_compute_topological_order` 计算节点计算顺序
   - 构建 NEATNetwork

2. **convert_from_checkpoint(checkpoint_path, agent_type, neat_config_path)** — 批量转换
   - 支持 gzip 和普通 pickle 两种 checkpoint 格式
   - 支持标准格式和 SubPopulationManager 格式
   - 返回 `[(genome_id, NEATNetwork, fitness)]` 按 fitness 降序排列

3. **verify(genome, neat_config, pytorch_net)** — 一致性验证
   - 使用 `neat.nn.FeedForwardNetwork`（纯 Python）作为参考基准
   - 随机生成 n_samples 个输入，比较两个网络输出的绝对误差
   - 不使用 FastFeedForwardNetwork（Cython）以避免 float32 精度差异

#### _compute_topological_order (内部方法)
- 从输出节点反向追踪可达节点，排除不可达的孤立节点
- 分层拓扑排序：每轮找出所有输入已就绪的节点
- 确定性排序（同层节点按 ID 排序），保证结果可复现

### `policy.py` — ActorCriticPolicy

Actor-Critic 策略模块，将转换后的 NEAT 网络包装为 RL 兼容的策略。

**结构:**
- **Actor**: NEAT 网络 (NEATNetwork)，输出动作均值
- **log_std**: 可学习的动作标准差参数（nn.Parameter），创建随机策略
- **Critic**: 独立的 MLP (nn.Sequential)，估计状态价值 V(s)

**核心方法:**
- `get_action_and_value(obs, action?, deterministic?)` -> (action, log_prob, entropy, value)
- `get_value(obs)` -> value（GAE 计算时使用）
- `get_action_mean(obs)` -> 确定性动作均值（部署时使用）
- `get_param_groups(actor_lr_scale, base_lr)` -> 分组学习率参数列表
- `freeze_actor()` / `unfreeze_actor()` — 控制 actor 参数是否参与梯度更新

**Actor 冻结调度:**
- 初始阶段冻结 actor（NEAT 权重不更新），只训练 critic 和 log_std
- 解冻后 actor 使用更小学习率 (base_lr * actor_lr_scale)

### `trainer.py` — PPOTrainer

CleanRL 风格的 PPO-Clip 训练器。

**训练流程:**
1. 收集 rollout: n_steps 步环境交互，存储 (obs, action, log_prob, reward, done, value)
2. 计算 GAE 优势和 returns
3. Mini-batch PPO 更新: n_epochs 个 epoch，每个 epoch 随机打乱后按 batch_size 切分
4. 定期保存 checkpoint 和输出日志

**PPO 更新细节:**
- **策略损失**: PPO-Clip，`-min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)`
- **价值损失**: Clipped value loss，`max((V-R)^2, (V_clip-R)^2) * 0.5`
- **熵奖励**: `-entropy.mean() * ent_coef`（鼓励探索）
- **梯度裁剪**: `clip_grad_norm_(params, max_grad_norm)`
- **优势归一化**: `(adv - mean) / (std + 1e-8)`

**Actor 解冻调度:**
- 当 global_step >= freeze_actor_steps 时自动解冻 actor
- 解冻时重建优化器，actor 使用 base_lr * actor_lr_scale 的更小学习率

**Checkpoint 包含:**
- policy_state_dict, optimizer_state_dict
- global_step, num_updates, actor_unfrozen
- config 摘要 (agent_type, total_timesteps 等)

**指标记录:**
- episode_reward, episode_length (每 episode 结束时)
- policy_loss, value_loss, entropy, approx_kl (每次 update)

## 训练入口脚本

`scripts/train_rl.py` — 命令行训练入口。

**流程:**
1. 从 NEAT checkpoint 加载并转换指定排名的基因组为 PyTorch 网络
2. 创建回放环境 (ReplayEnv)
3. 构建 ActorCriticPolicy (actor=NEATNetwork + critic=MLP)
4. 创建 PPOTrainer 并执行训练
5. 支持 --resume 从 RL checkpoint 恢复训练

**关键参数:**
- `--checkpoint`: NEAT checkpoint 路径（必填）
- `--data-dir`: HFtrade monitor/data 路径（必填）
- `--date-start` / `--date-end`: 数据日期范围（必填）
- `--genome-rank`: 使用第几名基因组（默认 0 = 最优）
- `--freeze-actor-steps`: Actor 冻结步数（默认 50000）
- `--total-timesteps`: 总训练步数（默认 1000000）
- `--resume`: RL checkpoint 路径（可选，恢复训练）

## 依赖关系

### 内部依赖
- `src.replay.config.ReplayConfig` — 回放环境配置
- `src.replay.replay_env.ReplayEnv` — Gymnasium 环境
- `src.config.config.AgentType` — Agent 类型枚举

### 外部依赖
- `torch` — PyTorch 深度学习框架
- `gymnasium` — 标准 RL 环境接口
- `neat` — NEAT 库（加载 checkpoint 时使用）
- `numpy` — 数值计算
