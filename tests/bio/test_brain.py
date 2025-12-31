"""测试 Brain 模块

由于 FastFeedForwardNetwork 是 Cython 扩展类型（不可变），
无法直接 mock 其 create 方法。因此采用以下策略：
1. 单元测试使用实际的 NEAT genome 进行测试
2. 集成测试验证 FastFeedForwardNetwork 正确工作
"""

import pytest
import neat

from src.bio.brain.brain import Brain, _USE_FAST_NETWORK


def create_test_genome_and_config() -> tuple[neat.DefaultGenome, neat.Config]:
    """创建测试用的 genome 和 config"""
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        'config/neat_retail.cfg'
    )
    pop = neat.Population(config)
    genome = list(pop.population.values())[0]
    return genome, config


class TestBrainFromGenome:
    """测试 Brain.from_genome"""

    def test_from_genome_creates_brain(self) -> None:
        """测试从基因组创建 Brain"""
        genome, config = create_test_genome_and_config()

        brain = Brain.from_genome(genome, config)

        assert isinstance(brain, Brain)
        assert brain.genome is genome
        assert brain.config is config
        assert brain.network is not None

    def test_from_genome_with_different_genomes(self) -> None:
        """测试使用不同基因组创建 Brain"""
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            'config/neat_retail.cfg'
        )
        pop = neat.Population(config)
        genomes = list(pop.population.values())[:2]

        brain1 = Brain.from_genome(genomes[0], config)
        brain2 = Brain.from_genome(genomes[1], config)

        assert brain1 is not brain2
        assert brain1.genome is genomes[0]
        assert brain2.genome is genomes[1]

    def test_from_genome_equivalent_to_constructor(self) -> None:
        """测试 from_genome 与直接调用构造函数等价"""
        genome, config = create_test_genome_and_config()

        brain1 = Brain.from_genome(genome, config)
        brain2 = Brain(genome, config)

        assert isinstance(brain1, Brain)
        assert isinstance(brain2, Brain)
        assert type(brain1) is type(brain2)
        assert brain1.genome is brain2.genome
        assert brain1.config is brain2.config


class TestBrainInit:
    """测试 Brain.__init__"""

    def test_create_brain(self) -> None:
        """测试创建 Brain"""
        genome, config = create_test_genome_and_config()

        brain = Brain(genome, config)

        assert brain.genome is genome
        assert brain.config is config
        assert brain.network is not None

    def test_create_brain_with_different_genome(self) -> None:
        """测试使用不同基因组创建 Brain"""
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            'config/neat_retail.cfg'
        )
        pop = neat.Population(config)
        genomes = list(pop.population.values())[:2]

        brain1 = Brain(genomes[0], config)
        brain2 = Brain(genomes[1], config)

        assert brain1.genome is genomes[0]
        assert brain2.genome is genomes[1]
        assert brain1.genome is not brain2.genome
        assert brain1.network is not brain2.network

    def test_brain_network_has_activate(self) -> None:
        """测试创建的网络有 activate 方法"""
        genome, config = create_test_genome_and_config()

        brain = Brain(genome, config)

        assert brain.network is not None
        assert hasattr(brain.network, 'activate')


class TestBrainForward:
    """测试 Brain.forward"""

    def test_forward_normal_input(self) -> None:
        """测试正常输入的前向传播"""
        genome, config = create_test_genome_and_config()
        brain = Brain(genome, config)

        num_inputs = config.genome_config.num_inputs
        inputs = [0.5] * num_inputs
        result = brain.forward(inputs)

        assert len(result) == config.genome_config.num_outputs
        for val in result:
            assert isinstance(val, float)

    def test_forward_different_inputs(self) -> None:
        """测试不同输入产生不同输出"""
        genome, config = create_test_genome_and_config()
        brain = Brain(genome, config)

        num_inputs = config.genome_config.num_inputs

        result1 = brain.forward([0.0] * num_inputs)
        result2 = brain.forward([1.0] * num_inputs)
        result3 = brain.forward([-1.0] * num_inputs)

        # 确保输出格式正确
        assert len(result1) == config.genome_config.num_outputs
        assert len(result2) == config.genome_config.num_outputs
        assert len(result3) == config.genome_config.num_outputs

    def test_forward_boundary_values(self) -> None:
        """测试边界值输入"""
        genome, config = create_test_genome_and_config()
        brain = Brain(genome, config)

        num_inputs = config.genome_config.num_inputs

        # 测试零输入
        result = brain.forward([0.0] * num_inputs)
        assert len(result) == config.genome_config.num_outputs

        # 测试大值输入
        result = brain.forward([1000.0] * num_inputs)
        assert len(result) == config.genome_config.num_outputs

        # 测试负值输入
        result = brain.forward([-1000.0] * num_inputs)
        assert len(result) == config.genome_config.num_outputs


class TestBrainGetGenome:
    """测试 Brain.get_genome"""

    def test_get_genome_returns_stored_genome(self) -> None:
        """测试获取基因组返回保存的基因组"""
        genome, config = create_test_genome_and_config()

        brain = Brain(genome, config)
        result = brain.get_genome()

        assert result is genome

    def test_get_genome_multiple_calls(self) -> None:
        """测试多次调用 get_genome 返回同一个基因组"""
        genome, config = create_test_genome_and_config()

        brain = Brain(genome, config)

        result1 = brain.get_genome()
        result2 = brain.get_genome()
        result3 = brain.get_genome()

        assert result1 is genome
        assert result2 is genome
        assert result3 is genome
        assert result1 is result2
        assert result2 is result3


class TestFastFeedForwardNetworkIntegration:
    """测试 FastFeedForwardNetwork 集成（实际网络测试）"""

    def test_use_fast_network_flag(self) -> None:
        """测试是否使用快速网络"""
        assert _USE_FAST_NETWORK is True, "应该使用 Cython 优化的快速网络"

    def test_real_network_forward(self) -> None:
        """测试实际网络的前向传播"""
        import neat
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            'config/neat_retail.cfg'
        )
        pop = neat.Population(config)
        genome = list(pop.population.values())[0]

        brain = Brain.from_genome(genome, config)

        # 验证网络类型
        assert type(brain.network).__name__ == 'FastFeedForwardNetwork'

        # 测试前向传播
        num_inputs = config.genome_config.num_inputs
        inputs = [0.5] * num_inputs
        outputs = brain.forward(inputs)

        assert len(outputs) == config.genome_config.num_outputs
        for out in outputs:
            assert isinstance(out, float)
