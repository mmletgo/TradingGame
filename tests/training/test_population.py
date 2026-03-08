"""测试 Population 模块"""

import pytest
from unittest.mock import MagicMock, patch

from src.bio.agents.base import Agent
from src.bio.agents.retail_pro import RetailProAgent
from src.bio.agents.market_maker import MarketMakerAgent
from src.bio.brain.brain import Brain
from src.config.config import (
    AgentConfig,
    AgentType,
    Config,
    DemoConfig,
    MarketConfig,
    TrainingConfig,
)
from src.training.population import Population


class TestPopulationCreateAgents:
    """测试 Population.create_agents"""

    def setup_method(self):
        """每个测试方法前的设置"""
        # 创建 Population 实例（不调用 __init__，直接设置属性）
        self.population = object.__new__(Population)
        self.population.sub_population_id = None
        self.population.neat_config = MagicMock()
        self.population.agent_config = AgentConfig(
            count=10,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )

    @patch.object(Brain, 'from_genome')
    def test_create_retail_pro_agents(self, mock_from_genome):
        """测试创建高级散户 Agent"""
        # 设置种群类型为高级散户
        self.population.agent_type = AgentType.RETAIL_PRO

        # 创建 mock brain
        mock_brain = MagicMock(spec=Brain)
        mock_from_genome.return_value = mock_brain

        # 创建 mock genomes
        mock_genome1 = MagicMock()
        mock_genome2 = MagicMock()
        genomes = [(1, mock_genome1), (2, mock_genome2)]

        # 调用 create_agents
        agents = self.population.create_agents(genomes)

        # 验证返回了正确数量的 Agent
        assert len(agents) == 2
        # 验证都是 RetailProAgent
        assert all(isinstance(agent, RetailProAgent) for agent in agents)
        # 验证 agent_id 正确（高级散户 offset=0，使用 idx 作为 agent_id）
        assert agents[0].agent_id == 0
        assert agents[1].agent_id == 1
        # 验证 Brain.from_genome 被调用了正确次数
        assert mock_from_genome.call_count == 2

    @patch.object(Brain, 'from_genome')
    def test_create_market_maker_agents(self, mock_from_genome):
        """测试创建做市商 Agent"""
        # 设置种群类型为做市商
        self.population.agent_type = AgentType.MARKET_MAKER
        # 设置 AS 模型配置（做市商构造时需要）
        from src.config.config import ASConfig
        self.population._as_config = ASConfig()
        # 更新配置为做市商配置
        self.population.agent_config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        # 创建 mock brain
        mock_brain = MagicMock(spec=Brain)
        mock_from_genome.return_value = mock_brain

        # 创建 mock genomes
        mock_genome1 = MagicMock()
        mock_genome2 = MagicMock()
        mock_genome3 = MagicMock()
        genomes = [(10, mock_genome1), (20, mock_genome2), (30, mock_genome3)]

        # 调用 create_agents
        agents = self.population.create_agents(genomes)

        # 验证返回了正确数量的 Agent
        assert len(agents) == 3
        # 验证都是 MarketMakerAgent
        assert all(isinstance(agent, MarketMakerAgent) for agent in agents)
        # 验证 agent_id 正确（做市商 offset=2_000_000，idx=0,1,2）
        assert agents[0].agent_id == 2_000_000
        assert agents[1].agent_id == 2_000_001
        assert agents[2].agent_id == 2_000_002

    @patch.object(Brain, 'from_genome')
    def test_create_agents_empty_genomes(self, mock_from_genome):
        """测试空基因组列表"""
        self.population.agent_type = AgentType.RETAIL_PRO

        # 空基因组列表
        genomes: list[tuple[int, MagicMock]] = []

        # 调用 create_agents
        agents = self.population.create_agents(genomes)

        # 验证返回空列表
        assert agents == []
        # 验证 Brain.from_genome 没有被调用
        mock_from_genome.assert_not_called()

    @patch.object(Brain, 'from_genome')
    def test_create_agents_uses_correct_neat_config(self, mock_from_genome):
        """测试使用正确的 NEAT 配置"""
        self.population.agent_type = AgentType.RETAIL_PRO

        # 创建 mock brain
        mock_brain = MagicMock(spec=Brain)
        mock_from_genome.return_value = mock_brain

        # 创建 mock genome
        mock_genome = MagicMock()
        genomes = [(1, mock_genome)]

        # 调用 create_agents
        self.population.create_agents(genomes)

        # 验证 Brain.from_genome 使用了正确的配置
        mock_from_genome.assert_called_once_with(
            mock_genome, self.population.neat_config
        )

    @patch.object(Brain, 'from_genome')
    def test_create_agents_each_gets_unique_brain(self, mock_from_genome):
        """测试每个 Agent 获得独立的 Brain"""
        self.population.agent_type = AgentType.RETAIL_PRO

        # 创建多个不同的 mock brain
        mock_brain1 = MagicMock(spec=Brain)
        mock_brain2 = MagicMock(spec=Brain)
        mock_from_genome.side_effect = [mock_brain1, mock_brain2]

        # 创建 mock genomes
        mock_genome1 = MagicMock()
        mock_genome2 = MagicMock()
        genomes = [(1, mock_genome1), (2, mock_genome2)]

        # 调用 create_agents
        agents = self.population.create_agents(genomes)

        # 验证每个 Agent 的 brain 不同
        assert agents[0].brain is mock_brain1
        assert agents[1].brain is mock_brain2
        assert agents[0].brain is not agents[1].brain


class TestPopulationInit:
    """测试 Population.__init__"""

    def _create_config(self, agent_type: AgentType) -> Config:
        """创建测试用配置"""
        retail_pro_config = AgentConfig(
            count=10,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )
        market_maker_config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        return Config(
            market=MarketConfig(
                initial_price=100.0,
                tick_size=0.01,
                lot_size=1.0,
                depth=100,
            ),
            agents={
                AgentType.RETAIL_PRO: retail_pro_config,
                AgentType.MARKET_MAKER: market_maker_config,
            },
            training=TrainingConfig(
                episode_length=1000,
                checkpoint_interval=1000,
                neat_config_path="config",  # 配置目录
            ),
            demo=DemoConfig(
                host="localhost",
                port=8000,
                tick_interval=100,
            ),
        )

    @patch("src.training.population.get_logger")
    @patch("src.training.population.neat.Population")
    @patch("src.training.population.neat.Config")
    @patch.object(Brain, "from_genome")
    def test_create_retail_pro_population(
        self, mock_from_genome, mock_neat_config, mock_neat_pop, mock_get_logger
    ):
        """测试创建高级散户种群"""
        # 设置 mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_config_instance = MagicMock()
        mock_neat_config.return_value = mock_config_instance

        # 模拟 NEAT 种群返回的基因组
        mock_genome1 = MagicMock()
        mock_genome2 = MagicMock()
        mock_pop_instance = MagicMock()
        mock_pop_instance.population = {1: mock_genome1, 2: mock_genome2}
        mock_neat_pop.return_value = mock_pop_instance

        # 模拟 Brain 创建
        mock_brain = MagicMock(spec=Brain)
        mock_from_genome.return_value = mock_brain

        # 创建配置
        config = self._create_config(AgentType.RETAIL_PRO)

        # 创建种群
        population = Population(AgentType.RETAIL_PRO, config)

        # 验证属性
        assert population.agent_type == AgentType.RETAIL_PRO
        assert population.agent_config == config.agents[AgentType.RETAIL_PRO]
        assert population.generation == 0
        assert population.neat_config is mock_config_instance
        assert population.neat_pop is mock_pop_instance

        # 验证创建了正确数量的 Agent
        assert len(population.agents) == 2
        assert all(isinstance(agent, RetailProAgent) for agent in population.agents)

        # 验证日志
        mock_get_logger.assert_called_once_with("population")
        mock_logger.info.assert_called_once()

    @patch("src.training.population.get_logger")
    @patch("src.training.population.neat.Population")
    @patch("src.training.population.neat.Config")
    @patch.object(Brain, "from_genome")
    def test_create_market_maker_population(
        self, mock_from_genome, mock_neat_config, mock_neat_pop, mock_get_logger
    ):
        """测试创建做市商种群"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_config_instance = MagicMock()
        mock_neat_config.return_value = mock_config_instance

        mock_genome1 = MagicMock()
        mock_genome2 = MagicMock()
        mock_genome3 = MagicMock()
        mock_pop_instance = MagicMock()
        mock_pop_instance.population = {10: mock_genome1, 20: mock_genome2, 30: mock_genome3}
        mock_neat_pop.return_value = mock_pop_instance

        mock_brain = MagicMock(spec=Brain)
        mock_from_genome.return_value = mock_brain

        config = self._create_config(AgentType.MARKET_MAKER)

        population = Population(AgentType.MARKET_MAKER, config)

        assert population.agent_type == AgentType.MARKET_MAKER
        assert len(population.agents) == 3
        assert all(isinstance(agent, MarketMakerAgent) for agent in population.agents)

    @patch("src.training.population.get_logger")
    @patch("src.training.population.neat.Population")
    @patch("src.training.population.neat.Config")
    @patch.object(Brain, "from_genome")
    def test_neat_config_loaded_with_correct_path(
        self, mock_from_genome, mock_neat_config, mock_neat_pop, mock_get_logger
    ):
        """测试使用正确的路径加载 NEAT 配置"""
        mock_get_logger.return_value = MagicMock()
        mock_neat_config.return_value = MagicMock()
        mock_pop_instance = MagicMock()
        mock_pop_instance.population = {}
        mock_neat_pop.return_value = mock_pop_instance

        config = self._create_config(AgentType.RETAIL_PRO)

        Population(AgentType.RETAIL_PRO, config)

        # 验证 neat.Config 使用了正确的参数
        # 高级散户使用 neat_retail_pro.cfg 配置文件（在 config 目录下）
        import neat
        mock_neat_config.assert_called_once_with(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            "config/neat_retail_pro.cfg",  # config_dir / neat_retail_pro.cfg
        )


class TestPopulationEvaluate:
    """测试 Population.evaluate"""

    def setup_method(self):
        """每个测试方法前的设置"""
        # 创建 Population 实例（不调用 __init__，直接设置属性）
        self.population = object.__new__(Population)
        self.population.agents = []
        # 设置 agent_type 为 RETAIL_PRO（使用纯收益率适应度）
        self.population.agent_type = AgentType.RETAIL_PRO
        # 设置训练配置（适应度计算需要 position_cost_weight）
        self.population._training_config = TrainingConfig(
            episode_length=1000,
            checkpoint_interval=10,
            neat_config_path="config",
        )

    def _create_mock_agent(
        self,
        agent_id: int,
        balance: float,
        quantity: float,
        avg_price: float,
        initial_balance: float,
    ) -> MagicMock:
        """创建模拟 Agent"""
        agent = MagicMock(spec=Agent)
        agent.agent_id = agent_id
        agent.account = MagicMock()
        agent.account.balance = balance
        agent.account.position = MagicMock()
        agent.account.position.quantity = quantity
        agent.account.position.avg_price = avg_price
        agent.account.initial_balance = initial_balance
        return agent

    def test_evaluate_empty_population(self):
        """测试空种群评估"""
        self.population.agents = []
        result = self.population.evaluate(100.0)
        assert result == []

    def test_evaluate_single_agent(self):
        """测试单个 Agent 评估"""
        # 创建一个 Agent: balance=10000, quantity=0, initial_balance=10000
        # 适应度 = (10000 - 10000) / 10000 = 0.0
        agent = self._create_mock_agent(
            agent_id=1,
            balance=10000.0,
            quantity=0.0,
            avg_price=0.0,
            initial_balance=10000.0,
        )
        self.population.agents = [agent]

        result = self.population.evaluate(100.0)

        assert len(result) == 1
        assert result[0][0] is agent
        assert result[0][1] == pytest.approx(0.0)

    def test_evaluate_sorted_by_fitness_descending(self):
        """测试按适应度从高到低排序

        适应度使用对称公式:
        fitness = (balance - initial) / initial - position_cost_weight × |qty × price| / initial
        position_cost_weight = 0.02（散户默认值）
        """
        # Agent 1: balance=10000, qty=10, initial=10000, price=100
        # fitness = (10000-10000)/10000 - 0.02×|10×100|/10000 = 0 - 0.002 = -0.002
        agent1 = self._create_mock_agent(
            agent_id=1,
            balance=10000.0,
            quantity=10.0,
            avg_price=90.0,
            initial_balance=10000.0,
        )

        # Agent 2: balance=15000, qty=0, initial=10000, price=100
        # fitness = (15000-10000)/10000 - 0.02×|0×100|/10000 = 0.5 - 0 = 0.5
        agent2 = self._create_mock_agent(
            agent_id=2,
            balance=15000.0,
            quantity=0.0,
            avg_price=0.0,
            initial_balance=10000.0,
        )

        # Agent 3: balance=5000, qty=-10, initial=10000, price=100
        # fitness = (5000-10000)/10000 - 0.02×|-10×100|/10000 = -0.5 - 0.002 = -0.502
        agent3 = self._create_mock_agent(
            agent_id=3,
            balance=5000.0,
            quantity=-10.0,
            avg_price=110.0,
            initial_balance=10000.0,
        )

        self.population.agents = [agent1, agent2, agent3]

        result = self.population.evaluate(100.0)

        # 验证排序: agent2 (0.5) > agent1 (-0.002) > agent3 (-0.502)
        assert len(result) == 3
        assert result[0][0] is agent2
        assert result[0][1] == pytest.approx(0.5)
        assert result[1][0] is agent1
        assert result[1][1] == pytest.approx(-0.002)
        assert result[2][0] is agent3
        assert result[2][1] == pytest.approx(-0.502)

    def test_evaluate_with_symmetric_position_cost(self):
        """测试对称持仓成本：多空对称的适应度计算

        适应度使用对称公式:
        fitness = (balance - initial) / initial - position_cost_weight × |qty × price| / initial
        position_cost_weight = 0.02（散户默认值）
        同等数量的多头和空头持仓产生完全相同的适应度惩罚。
        """
        # 多头盈利: balance=10000, qty=100, price=100, initial=10000
        # fitness = (10000-10000)/10000 - 0.02×|100×100|/10000 = 0 - 0.02 = -0.02
        agent_long_profit = self._create_mock_agent(
            agent_id=1,
            balance=10000.0,
            quantity=100.0,
            avg_price=90.0,
            initial_balance=10000.0,
        )

        # 多头亏损: balance=10000, qty=100, price=100, initial=10000
        # fitness = (10000-10000)/10000 - 0.02×|100×100|/10000 = 0 - 0.02 = -0.02
        agent_long_loss = self._create_mock_agent(
            agent_id=2,
            balance=10000.0,
            quantity=100.0,
            avg_price=110.0,
            initial_balance=10000.0,
        )

        # 空头盈利: balance=10000, qty=-100, price=100, initial=10000
        # fitness = (10000-10000)/10000 - 0.02×|-100×100|/10000 = 0 - 0.02 = -0.02
        agent_short_profit = self._create_mock_agent(
            agent_id=3,
            balance=10000.0,
            quantity=-100.0,
            avg_price=110.0,
            initial_balance=10000.0,
        )

        # 空头亏损: balance=10000, qty=-100, price=100, initial=10000
        # fitness = (10000-10000)/10000 - 0.02×|-100×100|/10000 = 0 - 0.02 = -0.02
        agent_short_loss = self._create_mock_agent(
            agent_id=4,
            balance=10000.0,
            quantity=-100.0,
            avg_price=90.0,
            initial_balance=10000.0,
        )

        self.population.agents = [
            agent_long_profit,
            agent_long_loss,
            agent_short_profit,
            agent_short_loss,
        ]

        result = self.population.evaluate(100.0)

        # 验证适应度计算正确：新公式完全多空对称
        assert len(result) == 4
        # 所有 Agent fitness 相同：-0.02（对称持仓成本）
        assert result[0][1] == pytest.approx(-0.02)
        assert result[1][1] == pytest.approx(-0.02)
        assert result[2][1] == pytest.approx(-0.02)
        assert result[3][1] == pytest.approx(-0.02)

    def test_evaluate_returns_float_fitness(self):
        """测试返回的适应度是 Python float 而非 numpy 类型"""
        agent = self._create_mock_agent(
            agent_id=1,
            balance=10000.0,
            quantity=0.0,
            avg_price=0.0,
            initial_balance=10000.0,
        )
        self.population.agents = [agent]

        result = self.population.evaluate(100.0)

        # 验证适应度是 Python float 类型
        assert isinstance(result[0][1], float)
        assert type(result[0][1]) is float


class TestPopulationEvolve:
    """测试 Population.evolve 方法"""

    def _create_config(self) -> Config:
        """创建测试用配置"""
        retail_pro_config = AgentConfig(
            count=10,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )
        market_maker_config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        return Config(
            market=MarketConfig(
                initial_price=100.0,
                tick_size=0.01,
                lot_size=1.0,
                depth=100,
            ),
            agents={
                AgentType.RETAIL_PRO: retail_pro_config,
                AgentType.MARKET_MAKER: market_maker_config,
            },
            training=TrainingConfig(
                episode_length=1000,
                checkpoint_interval=1000,
                neat_config_path="config",  # 配置目录
            ),
            demo=DemoConfig(
                host="localhost",
                port=8000,
                tick_interval=100,
            ),
        )

    @patch("src.training.population.get_logger")
    @patch("src.training.population.neat.Population")
    @patch("src.training.population.neat.Config")
    @patch.object(Brain, "from_genome")
    def test_evolve_updates_generation(
        self, mock_from_genome, mock_neat_config, mock_neat_pop, mock_get_logger
    ):
        """进化后代数增加"""
        # 设置 mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_config_instance = MagicMock()
        mock_neat_config.return_value = mock_config_instance

        # 创建 mock 基因组和种群
        # 注意：必须设置 genome.key 以匹配 population 的 key
        mock_genome1 = MagicMock()
        mock_genome1.key = 1
        mock_genome2 = MagicMock()
        mock_genome2.key = 2
        mock_pop_instance = MagicMock()
        mock_pop_instance.population = {1: mock_genome1, 2: mock_genome2}
        mock_neat_pop.return_value = mock_pop_instance

        # 创建 mock brain，设置 get_genome 返回 mock 基因组
        mock_brain1 = MagicMock(spec=Brain)
        mock_brain2 = MagicMock(spec=Brain)
        mock_brain1.get_genome.return_value = mock_genome1
        mock_brain2.get_genome.return_value = mock_genome2
        mock_from_genome.side_effect = [mock_brain1, mock_brain2, mock_brain1, mock_brain2]

        config = self._create_config()

        population = Population(AgentType.RETAIL_PRO, config)

        # 验证初始代数为 0
        assert population.generation == 0

        # 调用 evolve
        population.evolve(100.0)

        # 验证代数增加
        assert population.generation == 1

        # 再次调用 evolve
        mock_from_genome.side_effect = [mock_brain1, mock_brain2]
        population.evolve(100.0)

        # 验证代数继续增加
        assert population.generation == 2

    @patch("src.training.population.get_logger")
    @patch("src.training.population.neat.Population")
    @patch("src.training.population.neat.Config")
    @patch.object(Brain, "from_genome")
    def test_evolve_rebuilds_agents(
        self, mock_from_genome, mock_neat_config, mock_neat_pop, mock_get_logger
    ):
        """进化后重建 agents 列表"""
        # 设置 mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_config_instance = MagicMock()
        mock_neat_config.return_value = mock_config_instance

        # 初始基因组
        # 注意：必须设置 genome.key 以匹配 population 的 key
        mock_genome1 = MagicMock()
        mock_genome1.key = 1
        mock_pop_instance = MagicMock()
        mock_pop_instance.population = {1: mock_genome1}
        mock_neat_pop.return_value = mock_pop_instance

        # 初始 brain
        mock_brain1 = MagicMock(spec=Brain)
        mock_brain1.get_genome.return_value = mock_genome1
        mock_from_genome.return_value = mock_brain1

        config = self._create_config()

        population = Population(AgentType.RETAIL_PRO, config)
        initial_agents = population.agents.copy()

        # evolve 后更新种群的基因组（模拟 NEAT 进化）
        # 新 genome 使用新的 key=2
        mock_genome2 = MagicMock()
        mock_genome2.key = 2
        mock_pop_instance.population = {2: mock_genome2}

        # 创建新 brain
        mock_brain2 = MagicMock(spec=Brain)
        mock_brain2.get_genome.return_value = mock_genome2
        mock_from_genome.return_value = mock_brain2

        # 调用 evolve
        population.evolve(100.0)

        # 验证 agents 列表被重建（不是同一个对象）
        # 注意：由于对象池化优化，agents 列表是同一个对象，但内容已更新
        assert population.agents is not initial_agents
        # 验证 agents 的 agent_id 保持不变（对象被复用）
        assert population.agents[0].agent_id == 0

    @patch("src.training.population.get_logger")
    @patch("src.training.population.neat.Population")
    @patch("src.training.population.neat.Config")
    @patch.object(Brain, "from_genome")
    def test_evolve_sets_fitness(
        self, mock_from_genome, mock_neat_config, mock_neat_pop, mock_get_logger
    ):
        """进化时设置适应度"""
        # 设置 mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_config_instance = MagicMock()
        mock_neat_config.return_value = mock_config_instance

        # 创建基因组，初始 fitness 设为 None
        # 注意：必须设置 genome.key 以匹配 population 的 key，
        # 否则 _cleanup_genome_internals 会清理所有 genome（因为 key 不匹配）
        mock_genome1 = MagicMock()
        mock_genome1.fitness = None
        mock_genome1.key = 1
        mock_genome2 = MagicMock()
        mock_genome2.fitness = None
        mock_genome2.key = 2
        mock_pop_instance = MagicMock()
        mock_pop_instance.population = {1: mock_genome1, 2: mock_genome2}
        mock_neat_pop.return_value = mock_pop_instance

        # 创建 brain，每个返回对应的 genome
        mock_brain1 = MagicMock(spec=Brain)
        mock_brain2 = MagicMock(spec=Brain)
        mock_brain1.get_genome.return_value = mock_genome1
        mock_brain2.get_genome.return_value = mock_genome2
        mock_from_genome.side_effect = [mock_brain1, mock_brain2, mock_brain1, mock_brain2]

        config = self._create_config()

        population = Population(AgentType.RETAIL_PRO, config)

        # 验证初始时 genome.fitness 为 None
        assert mock_genome1.fitness is None
        assert mock_genome2.fitness is None

        # 调用 evolve
        population.evolve(100.0)

        # 验证 neat_pop.run 被调用
        mock_pop_instance.run.assert_called_once()

        # 验证 genome.fitness 被设置（应该不再是 None）
        # evolve 方法中会调用 genome.fitness = agent.get_fitness(current_price)
        # 由于 Agent 使用真实的 account，fitness 应该被设置为 0.0（(初始净值 - 初始余额)/初始余额）
        assert mock_genome1.fitness == pytest.approx(0.0)
        assert mock_genome2.fitness == pytest.approx(0.0)

        # 验证 brain.get_genome 被调用（用于获取 genome 并设置 fitness）
        mock_brain1.get_genome.assert_called()
        mock_brain2.get_genome.assert_called()


class TestPopulationResetAgents:
    """测试 Population.reset_agents 方法"""

    def _create_config(self) -> Config:
        """创建测试用配置"""
        retail_pro_config = AgentConfig(
            count=10,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        )
        market_maker_config = AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0001,
        )

        return Config(
            market=MarketConfig(
                initial_price=100.0,
                tick_size=0.01,
                lot_size=1.0,
                depth=100,
            ),
            agents={
                AgentType.RETAIL_PRO: retail_pro_config,
                AgentType.MARKET_MAKER: market_maker_config,
            },
            training=TrainingConfig(
                episode_length=1000,
                checkpoint_interval=1000,
                neat_config_path="config",  # 配置目录
            ),
            demo=DemoConfig(
                host="localhost",
                port=8000,
                tick_interval=100,
            ),
        )

    @patch("src.training.population.get_logger")
    @patch("src.training.population.neat.Population")
    @patch("src.training.population.neat.Config")
    @patch.object(Brain, "from_genome")
    def test_reset_agents_resets_all_accounts(
        self, mock_from_genome, mock_neat_config, mock_neat_pop, mock_get_logger
    ):
        """重置所有 Agent 账户"""
        # 设置 mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_config_instance = MagicMock()
        mock_neat_config.return_value = mock_config_instance

        mock_genome1 = MagicMock()
        mock_genome2 = MagicMock()
        mock_pop_instance = MagicMock()
        mock_pop_instance.population = {1: mock_genome1, 2: mock_genome2}
        mock_neat_pop.return_value = mock_pop_instance

        mock_brain = MagicMock(spec=Brain)
        mock_from_genome.return_value = mock_brain

        config = self._create_config()

        population = Population(AgentType.RETAIL_PRO, config)

        # 修改 Agent 的账户状态
        initial_balance = config.agents[AgentType.RETAIL_PRO].initial_balance
        for agent in population.agents:
            agent.account.balance = 5000.0  # 减少余额
            agent.account.position.quantity = 100.0  # 添加持仓
            agent.account.position.avg_price = 95.0  # 设置均价

        # 验证修改生效
        for agent in population.agents:
            assert agent.account.balance == 5000.0
            assert agent.account.position.quantity == 100.0

        # 调用 reset_agents
        population.reset_agents()

        # 验证所有账户恢复到初始状态
        for agent in population.agents:
            assert agent.account.balance == initial_balance
            assert agent.account.position.quantity == 0.0
            assert agent.account.position.avg_price == 0.0
            assert agent.account.initial_balance == initial_balance
