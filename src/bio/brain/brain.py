"""
NEAT 神经网络封装模块
"""
import neat
from neat.nn import FeedForwardNetwork


class Brain:
    """NEAT 神经网络封装"""

    genome: neat.DefaultGenome
    network: FeedForwardNetwork
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
