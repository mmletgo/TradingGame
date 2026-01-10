"""
NEAT 神经网络封装模块

使用 Cython 优化的快速前向传播网络。
优化版本已迁移到 fork 的 neat-python 库中。
"""
import neat
import numpy as np
from neat.nn import FastFeedForwardNetwork, FeedForwardNetwork


class Brain:
    """NEAT 神经网络封装

    自动使用 Cython 优化的快速网络（如果可用），
    否则回退到 neat-python 原生实现。
    """

    genome: neat.DefaultGenome
    network: FastFeedForwardNetwork | FeedForwardNetwork
    config: neat.Config

    @classmethod
    def from_genome(cls, genome: neat.DefaultGenome, config: neat.Config) -> "Brain":
        """
        从基因组创建 Brain

        Args:
            genome: NEAT 基因组
            config: NEAT 配置

        Returns:
            Brain 实例
        """
        return cls(genome, config)

    def __init__(self, genome: neat.DefaultGenome, config: neat.Config) -> None:
        """
        创建神经网络封装

        Args:
            genome: NEAT 基因组
            config: NEAT 配置
        """
        self.genome = genome
        self.config = config
        # 使用 FastFeedForwardNetwork（如果可用则为 Cython 优化版本）
        self.network = FastFeedForwardNetwork.create(genome, config)

    def forward(self, inputs: list[float] | np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            inputs: 输入向量（list 或 ndarray）

        Returns:
            神经网络输出向量（numpy 数组）
        """
        return self.network.activate(inputs)

    def get_genome(self) -> neat.DefaultGenome:
        """
        获取基因组

        Returns:
            NEAT 基因组
        """
        return self.genome

    def update_from_genome(
        self, genome: neat.DefaultGenome, config: neat.Config
    ) -> None:
        """原地更新 genome 和 network，避免重建对象

        Args:
            genome: 新的 NEAT 基因组
            config: NEAT 配置
        """
        self.genome = genome
        self.config = config
        # 重建网络（必须的，因为拓扑可能变化）
        self.network = FastFeedForwardNetwork.create(genome, config)

    def update_from_network_params(
        self,
        genome: neat.DefaultGenome,
        network_params: dict[str, np.ndarray | int],
    ) -> None:
        """从网络参数原地更新 Brain，跳过基因组解析

        用于并行进化后快速更新网络，避免从 NEAT 基因组重建网络的开销。
        直接使用预计算的网络参数创建 FastFeedForwardNetwork。

        Args:
            genome: 新的 NEAT 基因组（用于 get_genome() 返回）
            network_params: 网络参数字典，包含：
                - num_inputs: int - 输入节点数
                - num_outputs: int - 输出节点数
                - input_keys: ndarray[int32] - 输入节点 ID
                - output_keys: ndarray[int32] - 输出节点 ID
                - num_nodes: int - 隐藏+输出节点数
                - node_ids: ndarray[int32] - 节点 ID
                - biases: ndarray[float32] - 偏置
                - responses: ndarray[float32] - 响应
                - act_types: ndarray[int32] - 激活函数类型
                - conn_indptr: ndarray[int32] - CSR 连接指针
                - conn_sources: ndarray[int32] - 连接源节点
                - conn_weights: ndarray[float32] - 连接权重
                - output_indices: ndarray[int32] - 输出节点索引
        """
        self.genome = genome
        # 直接从参数创建网络，跳过基因组解析
        self.network = FastFeedForwardNetwork.create_from_params(
            num_inputs=network_params['num_inputs'],
            num_outputs=network_params['num_outputs'],
            input_keys=network_params['input_keys'],
            output_keys=network_params['output_keys'],
            num_nodes=network_params['num_nodes'],
            node_ids=network_params['node_ids'],
            biases=network_params['biases'],
            responses=network_params['responses'],
            act_types=network_params['act_types'],
            conn_indptr=network_params['conn_indptr'],
            conn_sources=network_params['conn_sources'],
            conn_weights=network_params['conn_weights'],
            output_indices=network_params['output_indices'],
        )

    def update_network_only(
        self,
        network_params: dict[str, np.ndarray | int],
    ) -> None:
        """仅更新网络，不更新基因组引用

        用于延迟反序列化优化：进化时只更新网络用于决策，
        基因组引用保持不变（指向旧的基因组对象）。
        检查点保存时再反序列化并更新基因组。

        Args:
            network_params: 网络参数字典
        """
        # 只更新网络，不更新 genome
        self.network = FastFeedForwardNetwork.create_from_params(
            num_inputs=network_params['num_inputs'],
            num_outputs=network_params['num_outputs'],
            input_keys=network_params['input_keys'],
            output_keys=network_params['output_keys'],
            num_nodes=network_params['num_nodes'],
            node_ids=network_params['node_ids'],
            biases=network_params['biases'],
            responses=network_params['responses'],
            act_types=network_params['act_types'],
            conn_indptr=network_params['conn_indptr'],
            conn_sources=network_params['conn_sources'],
            conn_weights=network_params['conn_weights'],
            output_indices=network_params['output_indices'],
        )
