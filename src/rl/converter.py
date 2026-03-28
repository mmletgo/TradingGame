"""NEAT 基因组到 PyTorch 模块的转换器

提供将 NEAT 进化出的基因组精确转换为等价 PyTorch 模块的功能，
用于后续 RL 微调。转换后的网络保持与原始 NEAT 网络完全一致的前向传播结果。
"""
from __future__ import annotations

import gzip
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NEATNetwork(nn.Module):
    """精确复刻 NEAT 前馈网络的 PyTorch 模块

    网络按拓扑排序的层级依次计算每个节点：
    output_i = activation(sum(w_ij * x_j) + bias_i) * response_i

    其中 activation 统一为 tanh（项目中所有 NEAT 配置的 activation_default = tanh，
    activation_mutate_rate = 0.0）。
    """

    def __init__(
        self,
        input_ids: list[int],
        output_ids: list[int],
        node_order: list[int],
        node_biases: dict[int, float],
        node_responses: dict[int, float],
        connections: list[tuple[int, int, float]],
    ) -> None:
        """初始化 NEATNetwork

        Args:
            input_ids: 输入节点 ID 列表
            output_ids: 输出节点 ID 列表
            node_order: 拓扑排序后的节点列表（隐藏+输出，不含输入）
            node_biases: 节点 ID → bias 值的映射
            node_responses: 节点 ID → response 值的映射
            connections: (src, dst, weight) 所有 enabled 连接
        """
        super().__init__()

        self._input_ids = input_ids
        self._output_ids = output_ids
        self._node_order = node_order

        # 构建节点 -> 索引映射：输入节点在前，拓扑排序节点在后
        all_node_ids: list[int] = input_ids + node_order
        self._node_to_idx: dict[int, int] = {
            nid: i for i, nid in enumerate(all_node_ids)
        }
        self._num_nodes: int = len(all_node_ids)

        # 输出节点在 values 数组中的索引
        self._output_indices: list[int] = [
            self._node_to_idx[oid] for oid in output_ids
        ]

        # 存储偏置和响应为 Parameter（支持梯度）
        biases = torch.zeros(len(node_order))
        responses = torch.ones(len(node_order))
        for i, nid in enumerate(node_order):
            biases[i] = node_biases.get(nid, 0.0)
            responses[i] = node_responses.get(nid, 1.0)
        self.biases = nn.Parameter(biases)
        self.responses = nn.Parameter(responses)

        # 构建连接：对每个目标节点，存储 (src_idx, weight_param_idx) 列表
        # 用于 forward 时按拓扑顺序依次计算
        self._node_connections: list[list[tuple[int, int]]] = []
        weights_list: list[float] = []
        for i, nid in enumerate(node_order):
            node_conns: list[tuple[int, int]] = []
            for src, dst, w in connections:
                if dst == nid:
                    src_idx = self._node_to_idx.get(src)
                    if src_idx is not None:
                        node_conns.append((src_idx, len(weights_list)))
                        weights_list.append(w)
            self._node_connections.append(node_conns)

        self.weights = nn.Parameter(
            torch.tensor(weights_list, dtype=torch.float32)
            if weights_list
            else torch.empty(0, dtype=torch.float32)
        )

        # 预计算用于向量化 forward 的索引
        self._precompute_indices()

    def _precompute_indices(self) -> None:
        """预计算索引张量，用于批量 forward 中高效索引"""
        self._node_src_indices: list[torch.Tensor] = []
        self._node_weight_indices: list[torch.Tensor] = []
        for node_conns in self._node_connections:
            if node_conns:
                src_idxs = torch.tensor(
                    [c[0] for c in node_conns], dtype=torch.long
                )
                w_idxs = torch.tensor(
                    [c[1] for c in node_conns], dtype=torch.long
                )
            else:
                src_idxs = torch.tensor([], dtype=torch.long)
                w_idxs = torch.tensor([], dtype=torch.long)
            self._node_src_indices.append(src_idxs)
            self._node_weight_indices.append(w_idxs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: shape (batch, n_inputs) 或 (n_inputs,)

        Returns:
            shape (batch, n_outputs) 或 (n_outputs,)
        """
        single: bool = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        batch_size: int = x.shape[0]
        device: torch.device = x.device

        # 初始化所有节点值
        values = torch.zeros(batch_size, self._num_nodes, device=device)
        # 填充输入节点
        values[:, : len(self._input_ids)] = x

        # 按拓扑顺序计算每个节点
        input_count: int = len(self._input_ids)
        for i, _nid in enumerate(self._node_order):
            node_idx: int = input_count + i
            src_idxs = self._node_src_indices[i].to(device)
            w_idxs = self._node_weight_indices[i].to(device)

            if len(src_idxs) > 0:
                # 加权求和：收集源节点值并乘以对应权重
                src_vals = values[:, src_idxs]  # (batch, n_connections)
                ws = self.weights[w_idxs]  # (n_connections,)
                agg = (src_vals * ws).sum(dim=1)  # (batch,)
            else:
                agg = torch.zeros(batch_size, device=device)

            # activation(agg + bias) * response
            values[:, node_idx] = (
                torch.tanh(agg + self.biases[i]) * self.responses[i]
            )

        # 提取输出节点
        output_idxs = torch.tensor(
            self._output_indices, dtype=torch.long, device=device
        )
        result = values[:, output_idxs]

        if single:
            result = result.squeeze(0)

        return result


class NEATtoPyTorchConverter:
    """NEAT 基因组到 PyTorch 的转换器

    支持从单个基因组转换，也支持从 checkpoint 文件批量加载转换。
    提供验证方法确保转换后的网络输出与原始 NEAT 网络一致。
    """

    @staticmethod
    def convert(genome: Any, neat_config: Any) -> NEATNetwork:
        """将单个 NEAT 基因组转为 PyTorch 模块

        Args:
            genome: neat.DefaultGenome 对象
            neat_config: neat.Config 对象

        Returns:
            等价的 PyTorch NEATNetwork
        """
        # 1. 获取输入/输出节点 ID
        input_ids: list[int] = list(neat_config.genome_config.input_keys)
        output_ids: list[int] = list(neat_config.genome_config.output_keys)

        # 2. 提取所有 enabled 连接
        connections: list[tuple[int, int, float]] = []
        for (src, dst), conn in genome.connections.items():
            if conn.enabled:
                connections.append((src, dst, conn.weight))

        # 3. 计算拓扑排序
        node_order: list[int] = NEATtoPyTorchConverter._compute_topological_order(
            input_ids, output_ids, connections
        )

        # 4. 提取节点参数
        node_biases: dict[int, float] = {}
        node_responses: dict[int, float] = {}
        for nid, node in genome.nodes.items():
            node_biases[nid] = node.bias
            node_responses[nid] = node.response

        return NEATNetwork(
            input_ids=input_ids,
            output_ids=output_ids,
            node_order=node_order,
            node_biases=node_biases,
            node_responses=node_responses,
            connections=connections,
        )

    @staticmethod
    def _compute_topological_order(
        input_ids: list[int],
        output_ids: list[int],
        connections: list[tuple[int, int, float]],
    ) -> list[int]:
        """计算前馈网络的拓扑排序

        使用 NEAT 的 feed_forward_layers 算法：
        1. 从输入节点开始，标记为已计算
        2. 找到所有输入已满足（所有源节点都已计算）的节点作为下一层
        3. 重复直到没有新节点可以加入

        Returns:
            拓扑排序后的节点 ID 列表（不含输入节点）
        """
        input_set: set[int] = set(input_ids)

        # 对每个节点，记录其所有输入源
        node_inputs: dict[int, set[int]] = {}
        for src, dst, _ in connections:
            if dst not in node_inputs:
                node_inputs[dst] = set()
            node_inputs[dst].add(src)

        # 找到所有需要计算的节点（从输出反向可达）
        required: set[int] = set()
        queue: list[int] = list(output_ids)
        while queue:
            nid = queue.pop()
            if nid in required or nid in input_set:
                continue
            required.add(nid)
            for src in node_inputs.get(nid, set()):
                queue.append(src)

        # 分层拓扑排序
        computed: set[int] = set(input_ids)
        ordered: list[int] = []
        remaining: set[int] = required.copy()

        while remaining:
            # 找到所有输入已就绪的节点
            ready: list[int] = []
            for nid in remaining:
                inputs = node_inputs.get(nid, set())
                if inputs.issubset(computed):
                    ready.append(nid)

            if not ready:
                # 有些节点无法计算（循环依赖或孤立），跳过
                logger.warning(f"无法计算的节点: {remaining}")
                break

            ready.sort()  # 确定性排序
            ordered.extend(ready)
            computed.update(ready)
            remaining -= set(ready)

        return ordered

    @staticmethod
    def convert_from_checkpoint(
        checkpoint_path: str,
        agent_type: str,
        neat_config_path: str = "config",
    ) -> list[tuple[int, NEATNetwork, float]]:
        """从 NEAT checkpoint 加载并转换所有基因组

        支持两种 checkpoint 格式：
        - 标准格式：populations[AgentType] 含 neat_pop
        - SubPopulationManager 格式：populations[AgentType] 含 sub_populations 列表

        Args:
            checkpoint_path: checkpoint 文件路径（.pkl 或 .pkl.gz）
            agent_type: "RETAIL_PRO" 或 "MARKET_MAKER"
            neat_config_path: NEAT 配置文件目录

        Returns:
            [(genome_id, pytorch_network, fitness), ...] 按 fitness 降序排列
        """
        # 1. 加载 checkpoint（支持 gzip）
        path = Path(checkpoint_path)
        with open(path, "rb") as f:
            header = f.read(2)

        if header[:2] == b"\x1f\x8b":  # gzip magic
            with gzip.open(path, "rb") as f:
                checkpoint: dict[str, Any] = pickle.load(f)
        else:
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)

        # 2. 提取对应类型的种群数据
        from src.config.config import AgentType

        atype: AgentType = (
            AgentType.RETAIL_PRO
            if agent_type == "RETAIL_PRO"
            else AgentType.MARKET_MAKER
        )

        pop_data: dict[str, Any] = checkpoint.get("populations", {}).get(
            atype, {}
        )

        # 3. 提取所有基因组（处理 SubPopulationManager 格式）
        all_genomes: list[tuple[int, Any, float]] = []
        if pop_data.get("is_sub_population_manager"):
            for sub in pop_data.get("sub_populations", []):
                neat_pop = sub["neat_pop"]
                for gid, genome in neat_pop.population.items():
                    fitness: float = (
                        genome.fitness
                        if genome.fitness is not None
                        else float("-inf")
                    )
                    all_genomes.append((gid, genome, fitness))
        else:
            neat_pop = pop_data.get("neat_pop")
            if neat_pop is not None:
                for gid, genome in neat_pop.population.items():
                    fitness = (
                        genome.fitness
                        if genome.fitness is not None
                        else float("-inf")
                    )
                    all_genomes.append((gid, genome, fitness))

        # 4. 加载 NEAT 配置
        import neat

        config_file: str = (
            "neat_retail_pro.cfg"
            if agent_type == "RETAIL_PRO"
            else "neat_market_maker.cfg"
        )
        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(Path(neat_config_path) / config_file),
        )

        # 5. 按 fitness 降序排列
        all_genomes.sort(key=lambda x: x[2], reverse=True)

        # 6. 转换每个基因组
        results: list[tuple[int, NEATNetwork, float]] = []
        for gid, genome, fitness in all_genomes:
            try:
                net: NEATNetwork = NEATtoPyTorchConverter.convert(
                    genome, neat_config
                )
                results.append((gid, net, fitness))
            except Exception as e:
                logger.warning(f"转换基因组 {gid} 失败: {e}")

        logger.info(
            f"从 checkpoint 转换了 {len(results)}/{len(all_genomes)} 个"
            f" {agent_type} 基因组"
        )

        return results

    @staticmethod
    def verify(
        genome: Any,
        neat_config: Any,
        pytorch_net: NEATNetwork,
        n_samples: int = 100,
        atol: float = 1e-6,
    ) -> bool:
        """验证 PyTorch 网络与原始 NEAT 网络输出一致

        使用随机输入比较两个网络的输出，确保转换后的 PyTorch 网络
        精确复刻了 NEAT 网络的计算逻辑。

        注意：使用 neat.nn.FeedForwardNetwork（纯 Python）作为参考，
        而非 FastFeedForwardNetwork（Cython），以避免 float32 精度差异。

        Args:
            genome: NEAT 基因组
            neat_config: NEAT 配置
            pytorch_net: 转换后的 PyTorch 网络
            n_samples: 随机测试样本数
            atol: 容忍的绝对误差

        Returns:
            True 如果所有样本输出一致
        """
        # 创建原始 NEAT 网络（纯 Python 版本）
        from neat.nn import FeedForwardNetwork

        neat_net = FeedForwardNetwork.create(genome, neat_config)

        n_inputs: int = len(neat_config.genome_config.input_keys)
        pytorch_net.eval()

        all_close: bool = True
        with torch.no_grad():
            for sample_idx in range(n_samples):
                # 随机输入
                inputs: np.ndarray = np.random.randn(n_inputs).astype(
                    np.float64
                )

                # NEAT 前向传播
                neat_output: np.ndarray = np.array(
                    neat_net.activate(inputs.tolist())
                )

                # PyTorch 前向传播
                torch_input = torch.tensor(inputs, dtype=torch.float32)
                torch_output: np.ndarray = pytorch_net(torch_input).numpy()

                if not np.allclose(neat_output, torch_output, atol=atol):
                    max_diff: float = float(
                        np.max(np.abs(neat_output - torch_output))
                    )
                    logger.warning(
                        f"样本 {sample_idx} 输出不一致: max_diff={max_diff:.8f}"
                    )
                    all_close = False
                    break

        if all_close:
            logger.info(f"验证通过: {n_samples} 个样本全部一致 (atol={atol})")
        else:
            logger.warning("验证失败: PyTorch 网络输出与 NEAT 网络不一致")

        return all_close
