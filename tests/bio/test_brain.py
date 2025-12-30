"""测试 Brain 模块"""

import pytest
from unittest.mock import MagicMock, Mock, patch
from neat.nn import FeedForwardNetwork

from src.bio.brain.brain import Brain


class TestBrainFromGenome:
    """测试 Brain.from_genome"""

    @patch('src.bio.brain.brain.neat.nn.FeedForwardNetwork.create')
    def test_from_genome_creates_brain(self, mock_create):
        """测试从基因组创建 Brain"""
        # 创建 mock genome 和 config
        mock_genome = MagicMock()
        mock_config = MagicMock()

        # 创建 mock network
        mock_network = MagicMock(spec=FeedForwardNetwork)
        mock_create.return_value = mock_network

        # 调用 from_genome 类方法
        brain = Brain.from_genome(mock_genome, mock_config)

        # 验证返回的是 Brain 实例
        assert isinstance(brain, Brain)
        # 验证属性设置正确
        assert brain.genome is mock_genome
        assert brain.config is mock_config
        assert brain.network is mock_network
        # 验证调用过 FeedForwardNetwork.create
        mock_create.assert_called_once_with(mock_genome, mock_config)

    @patch('src.bio.brain.brain.neat.nn.FeedForwardNetwork.create')
    def test_from_genome_with_different_params(self, mock_create):
        """测试使用不同参数创建 Brain"""
        # 创建两对不同的 genome 和 config
        mock_genome1 = MagicMock()
        mock_config1 = MagicMock()
        mock_genome2 = MagicMock()
        mock_config2 = MagicMock()

        # 创建两个不同的 mock network
        mock_network1 = MagicMock(spec=FeedForwardNetwork)
        mock_network2 = MagicMock(spec=FeedForwardNetwork)
        mock_create.side_effect = [mock_network1, mock_network2]

        # 创建两个 Brain
        brain1 = Brain.from_genome(mock_genome1, mock_config1)
        brain2 = Brain.from_genome(mock_genome2, mock_config2)

        # 验证两个 Brain 不同
        assert brain1 is not brain2
        assert brain1.genome is mock_genome1
        assert brain2.genome is mock_genome2
        assert brain1.config is mock_config1
        assert brain2.config is mock_config2

    @patch('src.bio.brain.brain.neat.nn.FeedForwardNetwork.create')
    def test_from_genome_equivalent_to_constructor(self, mock_create):
        """测试 from_genome 与直接调用构造函数等价"""
        mock_genome = MagicMock()
        mock_config = MagicMock()

        mock_network = MagicMock(spec=FeedForwardNetwork)
        mock_create.return_value = mock_network

        # 使用 from_genome 创建
        brain1 = Brain.from_genome(mock_genome, mock_config)
        # 重置 mock
        mock_create.reset_mock()
        # 使用构造函数创建
        brain2 = Brain(mock_genome, mock_config)

        # 验证两者都是 Brain 实例
        assert isinstance(brain1, Brain)
        assert isinstance(brain2, Brain)
        # 验证属性相同
        assert type(brain1) is type(brain2)
        assert brain1.genome is brain2.genome
        assert brain1.config is brain2.config


class TestBrainInit:
    """测试 Brain.__init__"""

    @patch('src.bio.brain.brain.neat.nn.FeedForwardNetwork.create')
    def test_create_brain(self, mock_create):
        """测试创建 Brain"""
        # 创建 mock genome 和 config
        mock_genome = MagicMock()
        mock_config = MagicMock()

        # 创建 mock network
        mock_network = MagicMock(spec=FeedForwardNetwork)
        mock_create.return_value = mock_network

        # 创建 Brain
        brain = Brain(mock_genome, mock_config)

        # 验证属性设置正确
        assert brain.genome is mock_genome
        assert brain.config is mock_config
        assert brain.network is mock_network
        # 验证调用过 FeedForwardNetwork.create
        mock_create.assert_called_once_with(mock_genome, mock_config)

    @patch('src.bio.brain.brain.neat.nn.FeedForwardNetwork.create')
    def test_create_brain_with_different_genome(self, mock_create):
        """测试使用不同基因组创建 Brain"""
        # 创建两个不同的 mock genome
        mock_genome1 = MagicMock()
        mock_genome2 = MagicMock()
        mock_config = MagicMock()

        # 创建两个不同的 mock network
        mock_network1 = MagicMock(spec=FeedForwardNetwork)
        mock_network2 = MagicMock(spec=FeedForwardNetwork)
        mock_create.side_effect = [mock_network1, mock_network2]

        # 创建两个 Brain
        brain1 = Brain(mock_genome1, mock_config)
        brain2 = Brain(mock_genome2, mock_config)

        # 验证两个 Brain 的基因组不同
        assert brain1.genome is mock_genome1
        assert brain2.genome is mock_genome2
        assert brain1.genome is not brain2.genome
        # 网络也应该不同
        assert brain1.network is mock_network1
        assert brain2.network is mock_network2
        assert brain1.network is not brain2.network

    @patch('src.bio.brain.brain.neat.nn.FeedForwardNetwork.create')
    def test_brain_network_is_feedforward(self, mock_create):
        """测试创建的网络是前馈网络类型"""
        # 创建 mock genome 和 config
        mock_genome = MagicMock()
        mock_config = MagicMock()

        # 创建 mock network (使用 FeedForwardNetwork 作为 spec)
        mock_network = MagicMock(spec=FeedForwardNetwork)
        mock_create.return_value = mock_network

        # 创建 Brain
        brain = Brain(mock_genome, mock_config)

        # 验证 network 属性存在
        assert brain.network is mock_network
        # 验证 network 有 activate 方法（FeedForwardNetwork 的特征）
        assert hasattr(brain.network, 'activate')


class TestBrainForward:
    """测试 Brain.forward"""

    @patch('src.bio.brain.brain.neat.nn.FeedForwardNetwork.create')
    def test_forward_normal_input(self, mock_create):
        """测试正常输入的前向传播"""
        # 创建 mock genome 和 config
        mock_genome = MagicMock()
        mock_config = MagicMock()

        # 创建 mock network，设置 activate 返回值
        mock_network = MagicMock(spec=FeedForwardNetwork)
        mock_network.activate.return_value = [0.7, 0.3, 0.5]
        mock_create.return_value = mock_network

        # 创建 Brain
        brain = Brain(mock_genome, mock_config)

        # 调用 forward
        inputs = [1.0, 0.5, -0.3]
        result = brain.forward(inputs)

        # 验证返回值
        assert result == [0.7, 0.3, 0.5]
        # 验证 activate 被正确调用
        mock_network.activate.assert_called_once_with(inputs)

    @patch('src.bio.brain.brain.neat.nn.FeedForwardNetwork.create')
    def test_forward_different_inputs(self, mock_create):
        """测试不同输入的前向传播"""
        mock_genome = MagicMock()
        mock_config = MagicMock()

        mock_network = MagicMock(spec=FeedForwardNetwork)
        mock_network.activate.side_effect = [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]]
        mock_create.return_value = mock_network

        brain = Brain(mock_genome, mock_config)

        # 测试三次不同的输入
        result1 = brain.forward([1.0, 2.0])
        result2 = brain.forward([0.0, 0.0])
        result3 = brain.forward([-1.0, -2.0])

        assert result1 == [0.1, 0.9]
        assert result2 == [0.5, 0.5]
        assert result3 == [0.8, 0.2]
        assert mock_network.activate.call_count == 3

    @patch('src.bio.brain.brain.neat.nn.FeedForwardNetwork.create')
    def test_forward_boundary_values(self, mock_create):
        """测试边界值输入"""
        mock_genome = MagicMock()
        mock_config = MagicMock()

        mock_network = MagicMock(spec=FeedForwardNetwork)
        mock_network.activate.return_value = [0.0]
        mock_create.return_value = mock_network

        brain = Brain(mock_genome, mock_config)

        # 测试零输入
        result = brain.forward([0.0])
        assert result == [0.0]

        # 测试大值输入
        mock_network.activate.return_value = [1.0]
        result = brain.forward([1000.0])
        assert result == [1.0]

        # 测试负值输入
        mock_network.activate.return_value = [-1.0]
        result = brain.forward([-1000.0])
        assert result == [-1.0]


class TestBrainGetGenome:
    """测试 Brain.get_genome"""

    @patch('src.bio.brain.brain.neat.nn.FeedForwardNetwork.create')
    def test_get_genome_returns_stored_genome(self, mock_create):
        """测试获取基因组返回保存的基因组"""
        # 创建 mock genome 和 config
        mock_genome = MagicMock()
        mock_config = MagicMock()
        mock_network = MagicMock(spec=FeedForwardNetwork)
        mock_create.return_value = mock_network

        # 创建 Brain
        brain = Brain(mock_genome, mock_config)

        # 调用 get_genome
        result = brain.get_genome()

        # 验证返回的是传入的 genome
        assert result is mock_genome
        # 验证调用 FeedForwardNetwork.create 仍然只被调用一次（在 __init__ 中）
        mock_create.assert_called_once_with(mock_genome, mock_config)

    @patch('src.bio.brain.brain.neat.nn.FeedForwardNetwork.create')
    def test_get_genome_multiple_calls(self, mock_create):
        """测试多次调用 get_genome 返回同一个基因组"""
        mock_genome = MagicMock()
        mock_config = MagicMock()
        mock_network = MagicMock(spec=FeedForwardNetwork)
        mock_create.return_value = mock_network

        brain = Brain(mock_genome, mock_config)

        # 多次调用 get_genome
        result1 = brain.get_genome()
        result2 = brain.get_genome()
        result3 = brain.get_genome()

        # 验证每次都返回同一个 genome 对象
        assert result1 is mock_genome
        assert result2 is mock_genome
        assert result3 is mock_genome
        # 验证它们都是同一个对象
        assert result1 is result2
        assert result2 is result3
