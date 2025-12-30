"""
NEAT 神经网络封装模块

使用 Cython 优化的快速前向传播网络。
"""
import neat
from typing import TYPE_CHECKING

# 尝试导入 Cython 优化版本，失败则回退到原生实现
try:
    from src.bio.brain.fast_network import FastFeedForwardNetwork
    _USE_FAST_NETWORK = True
except ImportError:
    from neat.nn import FeedForwardNetwork
    _USE_FAST_NETWORK = False

if TYPE_CHECKING:
    from src.bio.brain.fast_network import FastFeedForwardNetwork
    from neat.nn import FeedForwardNetwork


class Brain:
    """NEAT 神经网络封装

    自动使用 Cython 优化的快速网络（如果可用），
    否则回退到 neat-python 原生实现。
    """

    genome: neat.DefaultGenome
    network: "FastFeedForwardNetwork | FeedForwardNetwork"
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
        if _USE_FAST_NETWORK:
            self.network = FastFeedForwardNetwork.create(genome, config)
        else:
            self.network = neat.nn.FeedForwardNetwork.create(genome, config)

    def forward(self, inputs: list[float]) -> list[float]:
        """
        前向传播

        Args:
            inputs: 输入向量

        Returns:
            神经网络输出向量
        """
        return self.network.activate(inputs)

    def get_genome(self) -> neat.DefaultGenome:
        """
        获取基因组

        Returns:
            NEAT 基因组
        """
        return self.genome
