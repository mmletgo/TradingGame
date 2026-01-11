"""竞技场工作进程模块

在独立进程中运行，接收命令执行 episode，返回适应度。
"""

import logging
import pickle
import queue
import signal
from dataclasses import dataclass
from multiprocessing import Event, Queue
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import neat
import numpy as np

from src.bio.agents.base import Agent, AgentType

if TYPE_CHECKING:
    from src.training.trainer import Trainer
from src.bio.agents.market_maker import MarketMakerAgent
from src.bio.agents.retail import RetailAgent
from src.bio.agents.retail_pro import RetailProAgent
from src.bio.agents.whale import WhaleAgent
from src.bio.brain.brain import Brain
from src.config.config import Config
from src.core.log_engine.logger import get_logger
from src.market.adl.adl_manager import ADLManager
from src.market.matching.matching_engine import MatchingEngine
from src.training.population import (
    Population,
    _unpack_network_params_numpy,
)


# Agent ID 偏移量，与 Population 保持一致
_AGENT_ID_OFFSET: dict[AgentType, int] = {
    AgentType.RETAIL: 0,
    AgentType.RETAIL_PRO: 1_000_000,
    AgentType.WHALE: 2_000_000,
    AgentType.MARKET_MAKER: 3_000_000,
}

# Agent 类映射
_AGENT_CLASSES: dict[AgentType, type[Agent]] = {
    AgentType.RETAIL: RetailAgent,
    AgentType.RETAIL_PRO: RetailProAgent,
    AgentType.WHALE: WhaleAgent,
    AgentType.MARKET_MAKER: MarketMakerAgent,
}


@dataclass
class ArenaConfig:
    """竞技场配置"""

    arena_id: int
    episodes_per_round: int = 10
    episode_length: int = 1000


def arena_worker_process(
    arena_id: int,
    config: Config,
    cmd_queue: "Queue[Any]",
    result_queue: "Queue[tuple[str, int, Any]]",
    shutdown_event: Any = None,
) -> None:
    """竞技场工作进程主函数

    在独立进程中运行，接收命令执行 episode，返回适应度。

    命令格式：
    - ("setup", genome_data, network_params): 从基因组创建 Agent
    - ("run", num_episodes): 运行 N 个 episode，返回累积适应度
    - ("shutdown",): 关闭进程

    结果格式：
    - ("setup_done", arena_id, None)
    - ("run_done", arena_id, (fitness_data, episode_count))
    - ("error", arena_id, error_message)

    Args:
        arena_id: 竞技场 ID
        config: 全局配置对象
        cmd_queue: 命令队列
        result_queue: 结果队列
        shutdown_event: 可选的 shutdown 事件，用于立即通知进程退出
    """
    # 延迟导入，避免循环依赖
    from src.training.trainer import Trainer
    from src.market.catfish import create_all_catfish, create_catfish

    # 忽略 Ctrl+C，由主进程处理
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    logger = get_logger(f"arena_worker_{arena_id}")
    logger.info(f"Arena {arena_id} worker 进程启动")

    # 初始化 Trainer
    trainer: Trainer | None = None
    neat_configs: dict[AgentType, neat.Config] = {}

    # 加载 NEAT 配置
    config_dir = Path(config.training.neat_config_path)
    for agent_type in AgentType:
        if agent_type == AgentType.MARKET_MAKER:
            neat_config_path = config_dir / "neat_market_maker.cfg"
        elif agent_type == AgentType.WHALE:
            neat_config_path = config_dir / "neat_whale.cfg"
        elif agent_type == AgentType.RETAIL_PRO:
            neat_config_path = config_dir / "neat_retail_pro.cfg"
        else:
            neat_config_path = config_dir / "neat_retail.cfg"

        neat_configs[agent_type] = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(neat_config_path),
        )

    # 辅助函数：检查是否需要退出
    def should_exit() -> bool:
        return shutdown_event is not None and shutdown_event.is_set()

    # 主循环
    while True:
        # 检查 shutdown 事件
        if should_exit():
            logger.info(f"Arena {arena_id} worker 检测到 shutdown 事件")
            break

        try:
            # 使用超时的 get，这样可以定期检查 shutdown 事件
            try:
                cmd_data = cmd_queue.get(timeout=1.0)
            except queue.Empty:
                # 超时，继续循环以检查 shutdown 事件
                continue

            if cmd_data is None:
                break

            cmd = cmd_data[0]

            if cmd == "shutdown":
                logger.info(f"Arena {arena_id} worker 收到关闭命令")
                break

            elif cmd == "setup":
                # 解包参数：("setup", genome_data_dict, network_params_dict)
                genome_data_dict = cmd_data[1]
                network_params_dict = cmd_data[2]
                # genome_data_dict: dict[AgentType, bytes]
                # network_params_dict: dict[AgentType, tuple[...]] (network params numpy arrays)

                try:
                    trainer = _setup_trainer(
                        config=config,
                        genome_data_dict=genome_data_dict,
                        network_params_dict=network_params_dict,
                        neat_configs=neat_configs,
                        create_catfish=create_catfish,
                        create_all_catfish=create_all_catfish,
                        logger=logger,
                    )
                    result_queue.put(("setup_done", arena_id, None))
                    logger.debug(f"Arena {arena_id} setup 完成")
                except Exception as e:
                    logger.error(f"Arena {arena_id} setup 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    result_queue.put(("error", arena_id, str(e)))

            elif cmd == "run":
                # 解包参数：("run", num_episodes)
                num_episodes = cmd_data[1]

                if trainer is None:
                    result_queue.put(("error", arena_id, "Trainer 未初始化，请先执行 setup"))
                    continue

                try:
                    fitness_data, episode_count = _run_episodes(
                        trainer=trainer,
                        num_episodes=num_episodes,
                        episode_length=config.training.episode_length,
                        logger=logger,
                        arena_id=arena_id,
                    )
                    result_queue.put(("run_done", arena_id, (fitness_data, episode_count)))
                    logger.debug(f"Arena {arena_id} 运行 {episode_count} 个 episode 完成")
                except Exception as e:
                    logger.error(f"Arena {arena_id} 运行失败: {e}")
                    import traceback
                    traceback.print_exc()
                    result_queue.put(("error", arena_id, str(e)))

            else:
                logger.warning(f"Arena {arena_id} 收到未知命令: {cmd}")

        except Exception as e:
            logger.error(f"Arena {arena_id} worker 异常: {e}")
            import traceback
            traceback.print_exc()
            result_queue.put(("error", arena_id, str(e)))

    logger.info(f"Arena {arena_id} worker 进程退出")


def _setup_trainer(
    config: Config,
    genome_data_dict: dict[AgentType, bytes],
    network_params_dict: dict[AgentType, tuple[Any, ...]],
    neat_configs: dict[AgentType, neat.Config],
    create_catfish: Callable[..., Any],
    create_all_catfish: Callable[..., Any],
    logger: logging.Logger,
) -> "Trainer":
    """设置 Trainer，从基因组数据创建 Agent

    Args:
        config: 全局配置
        genome_data_dict: 各物种的基因组序列化数据
        network_params_dict: 各物种的网络参数（NumPy 格式）
        neat_configs: 各物种的 NEAT 配置
        create_catfish: 鲶鱼创建函数
        create_all_catfish: 多模式鲶鱼创建函数
        logger: 日志器

    Returns:
        配置好的 Trainer 对象
    """
    from src.training.trainer import Trainer

    # 创建 Trainer
    trainer = Trainer(config)
    trainer.matching_engine = MatchingEngine(config.market)
    trainer.adl_manager = ADLManager()

    # 从基因组创建 Agent
    for agent_type in AgentType:
        genome_data = genome_data_dict.get(agent_type)
        network_params_tuple = network_params_dict.get(agent_type)

        if genome_data is None:
            logger.warning(f"缺少 {agent_type.value} 的基因组数据")
            continue

        # 反序列化基因组
        genome = pickle.loads(genome_data)
        if not isinstance(genome, neat.DefaultGenome):
            logger.warning(f"{agent_type.value} 基因组数据无效")
            continue

        # 获取配置
        agent_config = config.agents[agent_type]
        neat_config = neat_configs[agent_type]
        agent_class = _AGENT_CLASSES[agent_type]
        agent_id_offset = _AGENT_ID_OFFSET[agent_type]

        # 解包网络参数（如果有）
        params_list: list[dict[str, np.ndarray | int]] | None = None
        if network_params_tuple is not None:
            params_list = _unpack_network_params_numpy(*network_params_tuple)

        # 创建 Agent 列表
        agents: list[Agent] = []
        for i in range(agent_config.count):
            brain = Brain.from_genome(genome, neat_config)

            # 如果有预计算的网络参数，使用快速更新
            if params_list is not None and i < len(params_list):
                brain.update_network_only(params_list[i])

            agent = agent_class(agent_id_offset + i, brain, agent_config)
            agents.append(agent)

        # 创建简化的 Population 对象
        pop = Population.__new__(Population)
        pop.agent_type = agent_type
        pop.agent_config = agent_config
        pop.generation = 0
        pop.logger = get_logger("population")
        pop._executor = None
        pop._num_workers = 8
        pop.neat_config = neat_config
        pop.neat_pop = None  # 竞技场模式不需要 NEAT 种群
        pop.agents = agents
        # 初始化适应度累积相关属性
        pop._accumulated_fitness = {}
        pop._accumulation_count = 0

        trainer.populations[agent_type] = pop

    # 注册 Agent 费率
    trainer._register_all_agents()
    trainer._build_agent_map()
    trainer._build_execution_order()
    trainer._update_pop_total_counts()

    # 初始化鲶鱼（如果配置中启用）
    if config.catfish and config.catfish.enabled:
        catfish_initial_balance = trainer._calculate_catfish_initial_balance()
        whale_config = config.agents[AgentType.WHALE]
        catfish_leverage = whale_config.leverage
        catfish_mmr = whale_config.maintenance_margin_rate

        if config.catfish.multi_mode:
            trainer.catfish_list = create_all_catfish(
                config.catfish,
                initial_balance=catfish_initial_balance,
                leverage=catfish_leverage,
                maintenance_margin_rate=catfish_mmr,
            )
            for catfish in trainer.catfish_list:
                trainer.matching_engine.register_agent(catfish.catfish_id, 0.0, 0.0)
        else:
            catfish = create_catfish(
                -1,
                config.catfish,
                initial_balance=catfish_initial_balance,
                leverage=catfish_leverage,
                maintenance_margin_rate=catfish_mmr,
            )
            trainer.catfish_list = [catfish]
            trainer.matching_engine.register_agent(catfish.catfish_id, 0.0, 0.0)

    # 初始化市场
    trainer._ema_alpha = config.market.ema_alpha
    trainer._init_ema_price(config.market.initial_price)
    trainer._init_market()

    return trainer


def _run_episodes(
    trainer: "Trainer",
    num_episodes: int,
    episode_length: int,
    logger: logging.Logger,
    arena_id: int,
) -> tuple[dict[tuple[AgentType, int], np.ndarray], int]:
    """运行多个 episode，返回累积适应度

    Args:
        trainer: Trainer 对象
        num_episodes: 要运行的 episode 数量
        episode_length: 每个 episode 的 tick 数量
        logger: 日志器
        arena_id: 竞技场 ID

    Returns:
        (fitness_data, episode_count) 元组
        - fitness_data: 累积适应度字典
            - key: (agent_type, sub_pop_id) 元组，竞技场模式 sub_pop_id 固定为 0
            - value: 累积适应度数组，shape=(pop_size,)
        - episode_count: 实际运行的 episode 数量
    """
    # 累积适应度存储
    accumulated_fitness: dict[tuple[AgentType, int], np.ndarray] = {}

    # 运行 episode
    actual_episodes = 0
    for ep_idx in range(num_episodes):
        # 重置所有 Agent
        for population in trainer.populations.values():
            population.reset_agents()

        # 重置鲶鱼
        for catfish in trainer.catfish_list:
            catfish.reset()

        # 重置市场状态
        trainer._reset_market()
        trainer.tick = 0
        trainer._pop_liquidated_counts.clear()
        trainer._catfish_liquidated = False

        # 初始化 episode 价格统计
        initial_price = trainer.config.market.initial_price
        trainer._episode_high_price = initial_price
        trainer._episode_low_price = initial_price

        # 运行单个 episode
        trainer.is_running = True
        trainer.episode = ep_idx + 1

        for _ in range(episode_length):
            if not trainer.is_running:
                break
            trainer.run_tick()

            # 检查鲶鱼是否被强平（训练模式下提前结束）
            if trainer._catfish_liquidated:
                break

            # 检查提前结束条件（物种淘汰到 1/4 或订单簿单边）
            early_end_result = trainer._should_end_episode_early()
            if early_end_result is not None:
                break

        trainer.is_running = False
        actual_episodes += 1

        # 计算并累积适应度
        assert trainer.matching_engine is not None, "MatchingEngine 未初始化"
        orderbook = trainer.matching_engine._orderbook
        current_price = orderbook.last_price

        for agent_type, population in trainer.populations.items():
            # 调用 evaluate 获取每个 Agent 的适应度
            agent_fitnesses = population.evaluate(current_price)

            # 构建 agent_id -> 索引映射（避免线性查找）
            agent_id_to_idx = {
                agent.agent_id: idx for idx, agent in enumerate(population.agents)
            }

            # 构建适应度数组
            fitness_arr = np.zeros(len(population.agents), dtype=np.float32)
            for agent, fitness in agent_fitnesses:
                idx = agent_id_to_idx.get(agent.agent_id)
                if idx is not None:
                    fitness_arr[idx] = fitness

            # 累积适应度
            key = (agent_type, 0)  # 竞技场模式 sub_pop_id 固定为 0
            if key not in accumulated_fitness:
                accumulated_fitness[key] = fitness_arr.copy()
            else:
                accumulated_fitness[key] += fitness_arr

    logger.debug(f"Arena {arena_id} 完成 {actual_episodes} 个 episode")
    return accumulated_fitness, actual_episodes


def _evaluate_population_fitness(
    population: Population,
    current_price: float,
) -> np.ndarray:
    """评估种群适应度，返回数组

    Args:
        population: 种群对象
        current_price: 当前价格

    Returns:
        适应度数组，shape=(pop_size,)
    """
    agent_fitnesses = population.evaluate(current_price)

    # 构建适应度数组（按 agent 顺序）
    fitness_arr = np.zeros(len(population.agents), dtype=np.float32)
    agent_to_idx = {agent.agent_id: idx for idx, agent in enumerate(population.agents)}

    for agent, fitness in agent_fitnesses:
        idx = agent_to_idx.get(agent.agent_id)
        if idx is not None:
            fitness_arr[idx] = fitness

    return fitness_arr
