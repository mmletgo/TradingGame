"""Trainer 测试模块

测试训练器的初始化、控制方法、tick/episode 执行和检查点管理。
"""

import pickle
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.bio.agents.base import AgentType
from src.config.config import (
    AgentConfig,
    Config,
    DemoConfig,
    MarketConfig,
    TrainingConfig,
)
from src.training.population import SubPopulationManager
from src.training.trainer import Trainer


def _mock_sub_pop_init(
    self: SubPopulationManager,
    config: Config,
    agent_type: AgentType,
    sub_count: int = 10,
) -> None:
    """SubPopulationManager 的 mock __init__，设置最小属性以通过 setup()"""
    self.agent_type = agent_type
    self.sub_population_count = sub_count
    total_count = config.agents[agent_type].count
    self.agents_per_sub = total_count // sub_count
    self._pending_genome_data = None
    self._genomes_dirty = False
    self.logger = MagicMock()
    # 创建一个 mock 子种群以支持 agent_config property
    mock_sub_pop = MagicMock()
    mock_sub_pop.agents = []
    mock_sub_pop.agent_config = config.agents[agent_type]
    mock_sub_pop.generation = 0
    self.sub_populations = [mock_sub_pop]


@pytest.fixture
def mock_config() -> Config:
    """创建测试用配置"""
    market = MarketConfig(
        initial_price=100.0,
        tick_size=0.01,
        lot_size=1.0,
        depth=100,
    )

    agent_config = AgentConfig(
        count=5,  # 测试时使用较少数量
        initial_balance=10000.0,
        leverage=10.0,
        maintenance_margin_rate=0.05,
        maker_fee_rate=0.0,
        taker_fee_rate=0.0001,
    )

    training = TrainingConfig(
        episode_length=10,  # 测试时使用较短的 episode
        checkpoint_interval=5,
        neat_config_path="config",
    )

    demo = DemoConfig(
        host="localhost",
        port=8000,
        tick_interval=100,
    )

    return Config(
        market=market,
        agents={
            AgentType.RETAIL: agent_config,
            AgentType.RETAIL_PRO: agent_config,
            AgentType.WHALE: agent_config,
            AgentType.MARKET_MAKER: agent_config,
        },
        training=training,
        demo=demo,
    )


class TestTrainerInit:
    """测试 Trainer.__init__"""

    @patch("src.training.trainer.get_logger")
    def test_init_sets_config(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """初始化设置配置"""
        mock_get_logger.return_value = MagicMock()
        trainer = Trainer(mock_config)
        assert trainer.config == mock_config

    @patch("src.training.trainer.get_logger")
    def test_init_sets_default_state(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """初始化设置默认状态"""
        mock_get_logger.return_value = MagicMock()
        trainer = Trainer(mock_config)
        assert trainer.tick == 0
        assert trainer.episode == 0
        assert trainer.is_running is False
        assert trainer.is_paused is False

    @patch("src.training.trainer.get_logger")
    def test_init_empty_populations(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """初始化时种群为空（需要调用 setup 才创建）"""
        mock_get_logger.return_value = MagicMock()
        trainer = Trainer(mock_config)
        assert trainer.populations == {}

    @patch("src.training.trainer.get_logger")
    def test_init_matching_engine_none(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """初始化时撮合引擎为 None"""
        mock_get_logger.return_value = MagicMock()
        trainer = Trainer(mock_config)
        assert trainer.matching_engine is None

    @patch("src.training.trainer.get_logger")
    def test_init_empty_recent_trades(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """初始化时最近成交列表为空"""
        mock_get_logger.return_value = MagicMock()
        trainer = Trainer(mock_config)
        assert len(trainer.recent_trades) == 0


class TestTrainerSetup:
    """测试 Trainer.setup"""

    @patch("src.training.trainer.MultiPopulationWorkerPool")
    @patch("src.training.trainer.Trainer._init_network_caches")
    @patch("src.training.trainer.Trainer._init_market")
    @patch("src.training.trainer.get_logger")
    @patch.object(SubPopulationManager, "__init__", _mock_sub_pop_init)
    @patch("src.training.trainer.Population")
    @patch("src.training.trainer.MatchingEngine")
    def test_setup_creates_populations(
        self,
        mock_matching_engine_class: MagicMock,
        mock_population_class: MagicMock,
        mock_get_logger: MagicMock,
        mock_init_market: MagicMock,
        mock_init_network_caches: MagicMock,
        mock_worker_pool_class: MagicMock,
        mock_config: Config,
    ) -> None:
        """setup 创建四种群"""
        mock_get_logger.return_value = MagicMock()
        mock_pop_instance = MagicMock()
        mock_pop_instance.agents = []
        mock_population_class.return_value = mock_pop_instance
        mock_engine_instance = MagicMock()
        mock_engine_instance._orderbook = MagicMock()
        mock_matching_engine_class.return_value = mock_engine_instance

        trainer = Trainer(mock_config)
        trainer.setup()

        # 验证 Population 被直接调用了 2 次（RETAIL_PRO 和 WHALE）
        assert mock_population_class.call_count == 2

        # 验证 Population 直接创建的种群类型
        pop_call_args_list = mock_population_class.call_args_list
        pop_agent_types = [call_args[0][0] for call_args in pop_call_args_list]
        assert AgentType.RETAIL_PRO in pop_agent_types
        assert AgentType.WHALE in pop_agent_types

        # 验证四个种群都已创建（2 个 SubPopulationManager + 2 个 Population）
        assert len(trainer.populations) == 4
        assert AgentType.RETAIL in trainer.populations
        assert AgentType.RETAIL_PRO in trainer.populations
        assert AgentType.WHALE in trainer.populations
        assert AgentType.MARKET_MAKER in trainer.populations

        # 验证 RETAIL 和 MARKET_MAKER 是 SubPopulationManager 实例
        assert isinstance(trainer.populations[AgentType.RETAIL], SubPopulationManager)
        assert isinstance(trainer.populations[AgentType.MARKET_MAKER], SubPopulationManager)

    @patch("src.training.trainer.MultiPopulationWorkerPool")
    @patch("src.training.trainer.Trainer._init_network_caches")
    @patch("src.training.trainer.Trainer._init_market")
    @patch("src.training.trainer.get_logger")
    @patch.object(SubPopulationManager, "__init__", _mock_sub_pop_init)
    @patch("src.training.trainer.Population")
    @patch("src.training.trainer.MatchingEngine")
    def test_setup_creates_matching_engine(
        self,
        mock_matching_engine_class: MagicMock,
        mock_population_class: MagicMock,
        mock_get_logger: MagicMock,
        mock_init_market: MagicMock,
        mock_init_network_caches: MagicMock,
        mock_worker_pool_class: MagicMock,
        mock_config: Config,
    ) -> None:
        """setup 创建撮合引擎"""
        mock_get_logger.return_value = MagicMock()
        mock_pop_instance = MagicMock()
        mock_pop_instance.agents = []
        mock_population_class.return_value = mock_pop_instance
        mock_engine_instance = MagicMock()
        mock_engine_instance._orderbook = MagicMock()
        mock_matching_engine_class.return_value = mock_engine_instance

        trainer = Trainer(mock_config)
        trainer.setup()

        # 验证撮合引擎被创建
        mock_matching_engine_class.assert_called_once()
        assert trainer.matching_engine is mock_engine_instance


class TestTrainerControl:
    """测试训练控制方法"""

    @patch("src.training.trainer.get_logger")
    def test_pause_sets_flag(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """pause 设置暂停标志"""
        mock_get_logger.return_value = MagicMock()
        trainer = Trainer(mock_config)
        trainer.pause()
        assert trainer.is_paused is True

    @patch("src.training.trainer.get_logger")
    def test_resume_clears_flag(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """resume 清除暂停标志"""
        mock_get_logger.return_value = MagicMock()
        trainer = Trainer(mock_config)
        trainer.is_paused = True
        trainer.resume()
        assert trainer.is_paused is False

    @patch("src.training.trainer.get_logger")
    def test_stop_clears_running(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """stop 清除运行标志"""
        mock_get_logger.return_value = MagicMock()
        trainer = Trainer(mock_config)
        trainer.is_running = True
        trainer.stop()
        assert trainer.is_running is False

    @patch("src.training.trainer.get_logger")
    def test_pause_logs_message(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """pause 记录日志"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        trainer = Trainer(mock_config)
        trainer.pause()
        mock_logger.info.assert_called()

    @patch("src.training.trainer.get_logger")
    def test_resume_logs_message(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """resume 记录日志"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        trainer = Trainer(mock_config)
        trainer.resume()
        mock_logger.info.assert_called()

    @patch("src.training.trainer.get_logger")
    def test_stop_logs_message(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """stop 记录日志"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        trainer = Trainer(mock_config)
        trainer.stop()
        mock_logger.info.assert_called()


class TestTrainerRunTick:
    """测试 run_tick 方法"""

    @patch("src.training.trainer.get_logger")
    def test_run_tick_without_matching_engine(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """没有撮合引擎时 run_tick 直接返回"""
        mock_get_logger.return_value = MagicMock()
        trainer = Trainer(mock_config)
        original_tick = trainer.tick

        trainer.run_tick()

        # tick 不应该增加
        assert trainer.tick == original_tick

    @patch("src.training.trainer.MultiPopulationWorkerPool")
    @patch("src.training.trainer.Trainer._init_network_caches")
    @patch("src.training.trainer.Trainer._init_market")
    @patch("src.training.trainer.get_logger")
    @patch.object(SubPopulationManager, "__init__", _mock_sub_pop_init)
    @patch("src.training.trainer.Population")
    @patch("src.training.trainer.MatchingEngine")
    def test_run_tick_increments_tick(
        self,
        mock_matching_engine_class: MagicMock,
        mock_population_class: MagicMock,
        mock_get_logger: MagicMock,
        mock_init_market: MagicMock,
        mock_init_network_caches: MagicMock,
        mock_worker_pool_class: MagicMock,
        mock_config: Config,
    ) -> None:
        """run_tick 增加 tick 计数"""
        mock_get_logger.return_value = MagicMock()

        # 设置 mock 种群
        mock_population = MagicMock()
        mock_population.agents = []
        mock_population_class.return_value = mock_population

        # 设置 mock 撮合引擎
        mock_engine = MagicMock()
        mock_orderbook = MagicMock()
        mock_orderbook.last_price = 100.0
        mock_orderbook.tick_size = 0.1
        mock_orderbook.get_mid_price.return_value = 100.0
        mock_orderbook.get_depth.return_value = {"bids": [], "asks": []}
        mock_engine._orderbook = mock_orderbook
        mock_matching_engine_class.return_value = mock_engine

        trainer = Trainer(mock_config)
        trainer.setup()

        initial_tick = trainer.tick
        trainer.run_tick()

        assert trainer.tick == initial_tick + 1


class TestTrainerRunEpisode:
    """测试 run_episode 方法"""

    @patch("src.training.trainer.get_logger")
    def test_run_episode_without_matching_engine(
        self, mock_get_logger: MagicMock, mock_config: Config
    ) -> None:
        """没有撮合引擎时 run_episode 直接返回"""
        mock_get_logger.return_value = MagicMock()
        trainer = Trainer(mock_config)
        original_episode = trainer.episode

        trainer.run_episode()

        # episode 不应该增加
        assert trainer.episode == original_episode

    @patch("src.training.trainer.MultiPopulationWorkerPool")
    @patch("src.training.trainer.Trainer._init_network_caches")
    @patch("src.training.trainer.Trainer._init_market")
    @patch("src.training.trainer.Trainer.run_tick")
    @patch("src.training.trainer.get_logger")
    @patch.object(SubPopulationManager, "__init__", _mock_sub_pop_init)
    @patch("src.training.trainer.Population")
    @patch("src.training.trainer.MatchingEngine")
    def test_run_episode_increments_episode(
        self,
        mock_matching_engine_class: MagicMock,
        mock_population_class: MagicMock,
        mock_get_logger: MagicMock,
        mock_run_tick: MagicMock,
        mock_init_market: MagicMock,
        mock_init_network_caches: MagicMock,
        mock_worker_pool_class: MagicMock,
        mock_config: Config,
    ) -> None:
        """run_episode 增加 episode 计数"""
        mock_get_logger.return_value = MagicMock()

        mock_population = MagicMock()
        mock_population.agents = []
        mock_population_class.return_value = mock_population

        mock_engine = MagicMock()
        mock_orderbook = MagicMock()
        mock_orderbook.last_price = 100.0
        mock_orderbook.tick_size = 0.1
        mock_orderbook.bids = {}
        mock_orderbook.asks = {}
        mock_orderbook.order_map = {}
        mock_orderbook.get_mid_price.return_value = 100.0
        mock_orderbook.get_depth.return_value = {"bids": [], "asks": []}
        mock_engine._orderbook = mock_orderbook
        mock_matching_engine_class.return_value = mock_engine

        trainer = Trainer(mock_config)
        trainer.setup()
        trainer.is_running = True

        initial_episode = trainer.episode
        trainer.run_episode()

        assert trainer.episode == initial_episode + 1

    @patch("src.training.trainer.MultiPopulationWorkerPool")
    @patch("src.training.trainer.Trainer._init_network_caches")
    @patch("src.training.trainer.Trainer._init_market")
    @patch("src.training.trainer.Trainer.run_tick")
    @patch("src.training.trainer.get_logger")
    @patch.object(SubPopulationManager, "__init__", _mock_sub_pop_init)
    @patch("src.training.trainer.Population")
    @patch("src.training.trainer.MatchingEngine")
    def test_run_episode_resets_agents(
        self,
        mock_matching_engine_class: MagicMock,
        mock_population_class: MagicMock,
        mock_get_logger: MagicMock,
        mock_run_tick: MagicMock,
        mock_init_market: MagicMock,
        mock_init_network_caches: MagicMock,
        mock_worker_pool_class: MagicMock,
        mock_config: Config,
    ) -> None:
        """run_episode 重置所有 Agent"""
        mock_get_logger.return_value = MagicMock()

        mock_population = MagicMock()
        mock_population.agents = []
        mock_population_class.return_value = mock_population

        mock_engine = MagicMock()
        mock_orderbook = MagicMock()
        mock_orderbook.last_price = 100.0
        mock_orderbook.tick_size = 0.1
        mock_orderbook.bids = {}
        mock_orderbook.asks = {}
        mock_orderbook.order_map = {}
        mock_orderbook.get_mid_price.return_value = 100.0
        mock_orderbook.get_depth.return_value = {"bids": [], "asks": []}
        mock_engine._orderbook = mock_orderbook
        mock_matching_engine_class.return_value = mock_engine

        trainer = Trainer(mock_config)
        trainer.setup()
        trainer.is_running = True

        trainer.run_episode()

        # 验证种群的 reset_agents 被调用（每个种群调用一次，共 4 个种群）
        # Population mock 的 reset_agents 有 call_count
        # SubPopulationManager 的 reset_agents 是真实方法，需要单独验证
        pop_reset_count = mock_population.reset_agents.call_count
        # SubPopulationManager 有 2 个实例（RETAIL, MARKET_MAKER），验证它们也被调用
        # 总共应该调用 4 次（2 次 Population + 2 次 SubPopulationManager）
        assert pop_reset_count == 2  # RETAIL_PRO 和 WHALE 各调用一次

    @patch("src.training.trainer.MultiPopulationWorkerPool")
    @patch("src.training.trainer.Trainer._init_network_caches")
    @patch("src.training.trainer.Trainer._init_market")
    @patch("src.training.trainer.get_logger")
    @patch.object(SubPopulationManager, "__init__", _mock_sub_pop_init)
    @patch("src.training.trainer.Population")
    @patch("src.training.trainer.MatchingEngine")
    def test_run_episode_stops_when_paused(
        self,
        mock_matching_engine_class: MagicMock,
        mock_population_class: MagicMock,
        mock_get_logger: MagicMock,
        mock_init_market: MagicMock,
        mock_init_network_caches: MagicMock,
        mock_worker_pool_class: MagicMock,
        mock_config: Config,
    ) -> None:
        """run_episode 在暂停时停止执行 tick"""
        mock_get_logger.return_value = MagicMock()

        mock_population = MagicMock()
        mock_population.agents = []
        mock_population_class.return_value = mock_population

        mock_engine = MagicMock()
        mock_orderbook = MagicMock()
        mock_orderbook.last_price = 100.0
        mock_orderbook.bids = {}
        mock_orderbook.asks = {}
        mock_orderbook.order_map = {}
        mock_engine._orderbook = mock_orderbook
        mock_matching_engine_class.return_value = mock_engine

        trainer = Trainer(mock_config)
        trainer.setup()
        trainer.is_running = True
        trainer.is_paused = True

        trainer.run_episode()

        # 暂停时 tick 应该为 0（没有执行任何 tick）
        assert trainer.tick == 0


class TestTrainerTrain:
    """测试 train 主循环"""

    @patch("src.training.trainer.MultiPopulationWorkerPool")
    @patch("src.training.trainer.Trainer._init_network_caches")
    @patch("src.training.trainer.Trainer._init_market")
    @patch("src.training.trainer.get_logger")
    @patch.object(SubPopulationManager, "__init__", _mock_sub_pop_init)
    @patch("src.training.trainer.Population")
    @patch("src.training.trainer.MatchingEngine")
    def test_train_sets_is_running(
        self,
        mock_matching_engine_class: MagicMock,
        mock_population_class: MagicMock,
        mock_get_logger: MagicMock,
        mock_init_market: MagicMock,
        mock_init_network_caches: MagicMock,
        mock_worker_pool_class: MagicMock,
        mock_config: Config,
    ) -> None:
        """train 设置 is_running 标志"""
        mock_get_logger.return_value = MagicMock()

        mock_population = MagicMock()
        mock_population.agents = []
        mock_population_class.return_value = mock_population

        mock_engine = MagicMock()
        mock_orderbook = MagicMock()
        mock_orderbook.last_price = 100.0
        mock_orderbook.bids = {}
        mock_orderbook.asks = {}
        mock_orderbook.order_map = {}
        mock_engine._orderbook = mock_orderbook
        mock_matching_engine_class.return_value = mock_engine

        trainer = Trainer(mock_config)
        trainer.setup()

        # 使用 0 个 episode 测试，只检查 is_running 的设置和清除
        trainer.train(episodes=0)

        # 训练结束后 is_running 应该为 False
        assert trainer.is_running is False

    @patch("src.training.trainer.MultiPopulationWorkerPool")
    @patch("src.training.trainer.Trainer._init_network_caches")
    @patch("src.training.trainer.Trainer._init_market")
    @patch("src.training.trainer.Trainer.run_tick")
    @patch("src.training.trainer.get_logger")
    @patch.object(SubPopulationManager, "__init__", _mock_sub_pop_init)
    @patch("src.training.trainer.Population")
    @patch("src.training.trainer.MatchingEngine")
    def test_train_calls_state_callback(
        self,
        mock_matching_engine_class: MagicMock,
        mock_population_class: MagicMock,
        mock_get_logger: MagicMock,
        mock_run_tick: MagicMock,
        mock_init_market: MagicMock,
        mock_init_network_caches: MagicMock,
        mock_worker_pool_class: MagicMock,
        mock_config: Config,
    ) -> None:
        """train 调用状态回调函数"""
        mock_get_logger.return_value = MagicMock()

        mock_population = MagicMock()
        mock_population.agents = []
        mock_population.generation = 0
        mock_population_class.return_value = mock_population

        mock_engine = MagicMock()
        mock_orderbook = MagicMock()
        mock_orderbook.last_price = 100.0
        mock_orderbook.tick_size = 0.1
        mock_orderbook.bids = {}
        mock_orderbook.asks = {}
        mock_orderbook.order_map = {}
        mock_orderbook.get_mid_price.return_value = 100.0
        mock_orderbook.get_depth.return_value = {"bids": [], "asks": []}
        mock_engine._orderbook = mock_orderbook
        mock_matching_engine_class.return_value = mock_engine

        trainer = Trainer(mock_config)
        trainer.setup()

        callback_states: list[dict[str, Any]] = []

        def state_callback(state: dict[str, Any]) -> None:
            callback_states.append(state)

        trainer.train(episodes=2, state_callback=state_callback)

        # 验证回调被调用了 2 次
        assert len(callback_states) == 2

    @patch("src.training.trainer.MultiPopulationWorkerPool")
    @patch("src.training.trainer.Trainer._init_network_caches")
    @patch("src.training.trainer.Trainer._init_market")
    @patch("src.training.trainer.Trainer.run_tick")
    @patch("src.training.trainer.get_logger")
    @patch.object(SubPopulationManager, "__init__", _mock_sub_pop_init)
    @patch("src.training.trainer.Population")
    @patch("src.training.trainer.MatchingEngine")
    def test_train_stops_when_stopped(
        self,
        mock_matching_engine_class: MagicMock,
        mock_population_class: MagicMock,
        mock_get_logger: MagicMock,
        mock_run_tick: MagicMock,
        mock_init_market: MagicMock,
        mock_init_network_caches: MagicMock,
        mock_worker_pool_class: MagicMock,
        mock_config: Config,
    ) -> None:
        """train 在 stop() 被调用后停止"""
        mock_get_logger.return_value = MagicMock()

        mock_population = MagicMock()
        mock_population.agents = []
        mock_population.generation = 0
        mock_population_class.return_value = mock_population

        mock_engine = MagicMock()
        mock_orderbook = MagicMock()
        mock_orderbook.last_price = 100.0
        mock_orderbook.tick_size = 0.1
        mock_orderbook.bids = {}
        mock_orderbook.asks = {}
        mock_orderbook.order_map = {}
        mock_orderbook.get_mid_price.return_value = 100.0
        mock_orderbook.get_depth.return_value = {"bids": [], "asks": []}
        mock_engine._orderbook = mock_orderbook
        mock_matching_engine_class.return_value = mock_engine

        trainer = Trainer(mock_config)
        trainer.setup()

        episodes_run = 0

        def state_callback(state: dict[str, Any]) -> None:
            nonlocal episodes_run
            episodes_run += 1
            if episodes_run >= 2:
                trainer.stop()

        # 请求 10 个 episode，但在第 2 个后停止
        trainer.train(episodes=10, state_callback=state_callback)

        # 验证只运行了 2 个 episode
        assert episodes_run == 2


class TestTrainerCheckpoint:
    """测试检查点保存和加载"""

    @patch("src.training.trainer.get_logger")
    @patch("src.training.trainer.pickle.dump")
    def test_save_checkpoint_creates_file(
        self,
        mock_pickle_dump: MagicMock,
        mock_get_logger: MagicMock,
        mock_config: Config,
    ) -> None:
        """save_checkpoint 创建检查点文件（使用 mock pickle 避免序列化问题）"""
        mock_get_logger.return_value = MagicMock()

        trainer = Trainer(mock_config)
        trainer.tick = 100
        trainer.episode = 10

        # 使用 MagicMock 作为种群
        mock_population = MagicMock()
        mock_population.generation = 5
        mock_population.neat_pop = MagicMock()

        trainer.populations = {
            AgentType.RETAIL: mock_population,
            AgentType.WHALE: mock_population,
            AgentType.MARKET_MAKER: mock_population,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pkl"
            trainer.save_checkpoint(str(checkpoint_path))

            # 验证 pickle.dump 被调用
            mock_pickle_dump.assert_called_once()
            # 验证传递的检查点数据包含正确的 tick 和 episode
            call_args = mock_pickle_dump.call_args[0][0]
            assert call_args["tick"] == 100
            assert call_args["episode"] == 10

    @patch("src.training.trainer.get_logger")
    @patch("src.training.trainer.pickle.dump")
    def test_save_checkpoint_creates_directory(
        self,
        mock_pickle_dump: MagicMock,
        mock_get_logger: MagicMock,
        mock_config: Config,
    ) -> None:
        """save_checkpoint 自动创建目录"""
        mock_get_logger.return_value = MagicMock()

        trainer = Trainer(mock_config)

        # 使用 MagicMock 作为种群
        mock_population = MagicMock()
        mock_population.generation = 0
        mock_population.neat_pop = MagicMock()

        trainer.populations = {
            AgentType.RETAIL: mock_population,
            AgentType.WHALE: mock_population,
            AgentType.MARKET_MAKER: mock_population,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # 使用嵌套目录路径
            checkpoint_path = Path(tmpdir) / "nested" / "dir" / "checkpoint.pkl"
            trainer.save_checkpoint(str(checkpoint_path))

            # 验证目录已创建
            assert checkpoint_path.parent.exists()

    @patch("src.training.trainer.MultiPopulationWorkerPool")
    @patch("src.training.trainer.Trainer._init_network_caches")
    @patch("src.training.trainer.Trainer._init_market")
    @patch("src.training.trainer.get_logger")
    @patch.object(SubPopulationManager, "__init__", _mock_sub_pop_init)
    @patch("src.training.trainer.Population")
    @patch("src.training.trainer.MatchingEngine")
    def test_load_checkpoint_restores_state(
        self,
        mock_matching_engine_class: MagicMock,
        mock_population_class: MagicMock,
        mock_get_logger: MagicMock,
        mock_init_market: MagicMock,
        mock_init_network_caches: MagicMock,
        mock_worker_pool_class: MagicMock,
        mock_config: Config,
    ) -> None:
        """load_checkpoint 恢复状态"""
        mock_get_logger.return_value = MagicMock()

        mock_population = MagicMock()
        mock_population.agents = []
        mock_population.generation = 0
        mock_population.create_agents = MagicMock(return_value=[])
        mock_population_class.return_value = mock_population

        mock_engine = MagicMock()
        mock_engine._orderbook = MagicMock()
        mock_matching_engine_class.return_value = mock_engine

        trainer = Trainer(mock_config)
        trainer.setup()

        # 准备检查点数据 - 使用字典作为 neat_pop 的模拟
        # 创建一个简单的对象来模拟 neat_pop
        mock_neat_pop = type("MockNeatPop", (), {"population": {1: "genome1"}})()

        # 准备检查点数据
        checkpoint = {
            "tick": 500,
            "episode": 50,
            "populations": {
                AgentType.RETAIL: {
                    "generation": 10,
                    "neat_pop": mock_neat_pop,
                },
                AgentType.WHALE: {
                    "generation": 10,
                    "neat_pop": mock_neat_pop,
                },
                AgentType.MARKET_MAKER: {
                    "generation": 10,
                    "neat_pop": mock_neat_pop,
                },
            },
        }

        # 使用 patch 来 mock pickle.load
        with patch("src.training.trainer.pickle.load", return_value=checkpoint):
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = Path(tmpdir) / "checkpoint.pkl"
                # 创建一个空文件（load 会被 mock）
                checkpoint_path.touch()

                # 重置状态
                trainer.tick = 0
                trainer.episode = 0

                trainer.load_checkpoint(str(checkpoint_path))

                # 验证状态已恢复
                assert trainer.tick == 500
                assert trainer.episode == 50


class TestTrainerGetState:
    """测试 _get_state 方法"""

    @patch("src.training.trainer.get_logger")
    def test_get_state_returns_correct_structure(
        self,
        mock_get_logger: MagicMock,
        mock_config: Config,
    ) -> None:
        """_get_state 返回正确的状态结构"""
        mock_get_logger.return_value = MagicMock()

        trainer = Trainer(mock_config)
        trainer.tick = 100
        trainer.episode = 10

        # 直接设置 mock 种群（不调用 setup，避免 _init_market 的复杂性）
        mock_population = MagicMock()
        mock_population.agents = [MagicMock(), MagicMock()]  # 2 个 agent
        mock_population.generation = 5

        trainer.populations = {
            AgentType.RETAIL: mock_population,
            AgentType.WHALE: mock_population,
            AgentType.MARKET_MAKER: mock_population,
        }

        state = trainer._get_state()

        assert "tick" in state
        assert "episode" in state
        assert "populations" in state
        assert state["tick"] == 100
        assert state["episode"] == 10

    @patch("src.training.trainer.get_logger")
    def test_get_state_includes_population_info(
        self,
        mock_get_logger: MagicMock,
        mock_config: Config,
    ) -> None:
        """_get_state 包含种群信息"""
        mock_get_logger.return_value = MagicMock()

        trainer = Trainer(mock_config)

        # 设置不同的种群
        retail_pop = MagicMock()
        retail_pop.agents = [MagicMock(), MagicMock(), MagicMock()]  # 3 个 agent
        retail_pop.generation = 10

        whale_pop = MagicMock()
        whale_pop.agents = [MagicMock()]  # 1 个 agent
        whale_pop.generation = 5

        mm_pop = MagicMock()
        mm_pop.agents = [MagicMock(), MagicMock()]  # 2 个 agent
        mm_pop.generation = 7

        trainer.populations = {
            AgentType.RETAIL: retail_pop,
            AgentType.WHALE: whale_pop,
            AgentType.MARKET_MAKER: mm_pop,
        }

        state = trainer._get_state()

        # 验证种群信息
        assert AgentType.RETAIL.value in state["populations"]
        assert state["populations"][AgentType.RETAIL.value]["count"] == 3
        assert state["populations"][AgentType.RETAIL.value]["generation"] == 10

        assert AgentType.WHALE.value in state["populations"]
        assert state["populations"][AgentType.WHALE.value]["count"] == 1
        assert state["populations"][AgentType.WHALE.value]["generation"] == 5

        assert AgentType.MARKET_MAKER.value in state["populations"]
        assert state["populations"][AgentType.MARKET_MAKER.value]["count"] == 2
        assert state["populations"][AgentType.MARKET_MAKER.value]["generation"] == 7
