"""种群管理模块

管理特定类型 Agent 的种群，支持从 NEAT 基因组创建 Agent。
"""

import logging

import neat
import numpy as np

from src.bio.agents.base import Agent
from src.bio.agents.market_maker import MarketMakerAgent
from src.bio.agents.retail import RetailAgent
from src.bio.agents.retail_pro import RetailProAgent
from src.bio.agents.whale import WhaleAgent
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, AgentType, Config
from src.core.log_engine.logger import get_logger


class Population:
    """种群管理类

    管理特定类型的 Agent 种群，包括创建、评估、淘汰和繁殖。

    Attributes:
        agent_type: Agent 类型（散户/庄家/做市商）
        agents: Agent 列表
        neat_pop: NEAT 种群对象
        neat_config: NEAT 配置
        agent_config: Agent 配置
        generation: 当前代数
        logger: 日志器
    """

    agent_type: AgentType
    agents: list[Agent]
    neat_pop: neat.Population
    neat_config: neat.Config
    agent_config: AgentConfig
    generation: int
    logger: logging.Logger

    def __init__(self, agent_type: AgentType, config: Config) -> None:
        """创建种群

        初始化 NEAT 种群，创建初始 Agent 列表。

        Args:
            agent_type: Agent 类型（散户/庄家/做市商）
            config: 全局配置对象
        """
        self.agent_type = agent_type
        self.agent_config = config.agents[agent_type]
        self.generation = 0
        self.logger = get_logger("population")

        # 根据 Agent 类型选择 NEAT 配置文件
        # 散户使用 67 个输入（10档订单簿 + 10笔成交），庄家使用 607 个输入，做市商使用 634 个输入
        from pathlib import Path

        config_dir = Path(config.training.neat_config_path)
        if agent_type == AgentType.MARKET_MAKER:
            neat_config_path = config_dir / "neat_market_maker.cfg"
        elif agent_type == AgentType.WHALE:
            neat_config_path = config_dir / "neat_whale.cfg"
        elif agent_type == AgentType.RETAIL_PRO:
            neat_config_path = config_dir / "neat_retail_pro.cfg"
        else:
            neat_config_path = config_dir / "neat_retail.cfg"

        # 加载 NEAT 配置
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(neat_config_path),
        )

        # 动态设置 pop_size 为 AgentConfig.count
        self.neat_config.pop_size = self.agent_config.count

        # 创建 NEAT 种群
        self.neat_pop = neat.Population(self.neat_config)

        # 获取初始基因组并创建 Agent
        genomes = list(self.neat_pop.population.items())
        self.agents = self.create_agents(genomes)

        self.logger.info(
            f"创建 {agent_type.value} 种群，初始 Agent 数量: {len(self.agents)}"
        )

    # Agent ID 偏移量，确保不同种群的 agent_id 不冲突
    # 每个种群类型使用不同的高位，每种群最多支持 100 万个 Agent
    _AGENT_ID_OFFSET = {
        AgentType.RETAIL: 0,
        AgentType.RETAIL_PRO: 1_000_000,
        AgentType.WHALE: 2_000_000,
        AgentType.MARKET_MAKER: 3_000_000,
    }

    def create_agents(
        self,
        genomes: list[tuple[int, neat.DefaultGenome]],
    ) -> list[Agent]:
        """从基因组创建 Agent 列表

        遍历基因组列表，为每个基因组创建对应的 Brain 和 Agent。
        根据种群的 agent_type 创建对应类型的 Agent（散户/庄家/做市商）。

        Args:
            genomes: NEAT 基因组列表，每项为 (genome_id, genome) 元组

        Returns:
            创建的 Agent 列表
        """
        # 根据 agent_type 确定 Agent 类（避免循环内重复判断）
        if self.agent_type == AgentType.RETAIL:
            agent_class = RetailAgent
        elif self.agent_type == AgentType.RETAIL_PRO:
            agent_class = RetailProAgent
        elif self.agent_type == AgentType.WHALE:
            agent_class = WhaleAgent
        elif self.agent_type == AgentType.MARKET_MAKER:
            agent_class = MarketMakerAgent
        else:
            raise ValueError(f"未知的 Agent 类型: {self.agent_type}")

        # 获取该种群类型的 agent_id 偏移量，确保全局唯一
        agent_id_offset = self._AGENT_ID_OFFSET.get(self.agent_type, 0)

        agents: list[Agent] = []

        for idx, (genome_id, genome) in enumerate(genomes):
            brain = Brain.from_genome(genome, self.neat_config)
            # 使用 offset + 索引 作为 agent_id，确保全局唯一
            unique_agent_id = agent_id_offset + idx
            agent = agent_class(unique_agent_id, brain, self.agent_config)
            agents.append(agent)

        return agents

    def evaluate(self, current_price: float) -> list[tuple[Agent, float]]:
        """评估种群适应度

        使用向量化运算计算所有 Agent 的适应度，并按适应度从高到低排序。

        Args:
            current_price: 当前市场价格，用于计算未实现盈亏

        Returns:
            按适应度从高到低排序的 (Agent, 适应度) 元组列表
        """
        n = len(self.agents)
        if n == 0:
            return []

        # 1. 收集所有 Agent 的账户数据到 numpy 数组
        balances = np.array([a.account.balance for a in self.agents])
        quantities = np.array([a.account.position.quantity for a in self.agents])
        avg_prices = np.array([a.account.position.avg_price for a in self.agents])
        initial_balances = np.array([a.account.initial_balance for a in self.agents])

        # 2. 向量化计算未实现盈亏: (current_price - avg_price) * quantity
        unrealized_pnl = (current_price - avg_prices) * quantities

        # 3. 向量化计算净值: balance + unrealized_pnl
        equities = balances + unrealized_pnl

        # 4. 向量化计算适应度: equity / initial_balance
        fitnesses = equities / initial_balances

        # 5. 获取从高到低的排序索引
        sorted_indices = np.argsort(fitnesses)[::-1]

        # 6. 按排序索引构建结果
        return [(self.agents[i], float(fitnesses[i])) for i in sorted_indices]

    def evolve(self, current_price: float) -> None:
        """进化种群

        使用 NEAT 算法进行一代进化：
        1. 使用向量化 evaluate 计算所有 Agent 适应度
        2. 通过 agent -> genome 映射为基因组设置适应度
        3. 调用 NEAT 种群的 run 方法进行一代进化
        4. 从新基因组重建 Agent 列表

        Args:
            current_price: 当前市场价格，用于计算适应度
        """
        # 1. 使用向量化 evaluate 获取所有 Agent 的适应度
        agent_fitnesses = self.evaluate(current_price)

        # 2. 通过 agent -> genome 映射为基因组设置适应度
        for agent, fitness in agent_fitnesses:
            genome = agent.brain.get_genome()
            genome.fitness = fitness  # type: ignore[assignment]

        # 3. 调用 NEAT 进化（eval_genomes 留空，适应度已设置）
        def eval_genomes(
            _genomes: list[tuple[int, neat.DefaultGenome]], _config: neat.Config
        ) -> None:
            # 适应度已经在上面设置好了，这里不需要再计算
            pass

        self.neat_pop.run(eval_genomes, n=1)

        # 4. 增加代数计数
        self.generation += 1

        # 5. 从新基因组重建 Agent 列表
        new_genomes = list(self.neat_pop.population.items())
        self.agents = self.create_agents(new_genomes)

        self.logger.info(
            f"{self.agent_type.value} 种群完成第 {self.generation} 代进化，"
            f"Agent 数量: {len(self.agents)}"
        )

    def reset_agents(self) -> None:
        """重置所有 Agent 的账户状态

        在 episode 开始时调用，将所有 Agent 的账户恢复到初始状态。
        """
        for agent in self.agents:
            agent.reset(self.agent_config)
