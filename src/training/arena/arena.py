"""竞技场模块

封装单个竞技场的训练逻辑。
"""

import pickle
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import neat
    from src.bio.agents.base import Agent, AgentType
    from src.training.arena.config import ArenaConfig
    from src.training.arena.metrics import ArenaMetrics, EpisodeMetrics
    from src.training.arena.migration import MigrationPacket
    from src.training.population import Population


class Arena:
    """单个竞技场封装

    包装 Trainer，添加竞技场 ID、指标收集和迁移接口。
    设计为在独立进程中运行。

    Attributes:
        arena_id: 竞技场 ID
        trainer: 训练器实例
        metrics: 指标收集器
        _seed: 随机种子
    """

    arena_id: int
    trainer: Any  # Trainer 类型，避免循环导入
    metrics: "ArenaMetrics"
    _seed: int | None

    def __init__(self, arena_config: "ArenaConfig") -> None:
        """初始化竞技场

        Args:
            arena_config: 竞技场配置
        """
        from src.training.arena.metrics import ArenaMetrics
        from src.training.trainer import Trainer

        self.arena_id = arena_config.arena_id
        self._seed = arena_config.seed

        # 设置随机种子（如果提供）
        if self._seed is not None:
            random.seed(self._seed)

        # 创建训练器
        self.trainer = Trainer(arena_config.config, arena_id=arena_config.arena_id)

        # 创建指标收集器
        self.metrics = ArenaMetrics(arena_config.arena_id)

    def setup(self, checkpoint: dict | None = None) -> None:
        """初始化训练环境

        Args:
            checkpoint: 可选的检查点数据，如果提供则直接从检查点恢复种群
        """
        # 传递检查点数据到 Trainer.setup()
        # 如果有检查点，Trainer 会直接从检查点创建种群，避免先创建再清理
        trainer_checkpoint = None
        if checkpoint is not None and "trainer" in checkpoint:
            trainer_checkpoint = checkpoint["trainer"]

        self.trainer.setup(checkpoint=trainer_checkpoint)
        # 设置 is_running 标志，否则 run_episode 中的 tick 循环会立即退出
        self.trainer.is_running = True

    def run_episode(self) -> "EpisodeMetrics":
        """运行单个 episode 并收集指标

        Returns:
            Episode 指标
        """
        # 运行 episode
        self.trainer.run_episode()

        # 收集指标
        price_stats = self.trainer.get_price_stats()
        population_stats = self.trainer.get_population_stats()

        metrics = self.metrics.record_episode(
            episode=self.trainer.episode,
            price_stats=price_stats,
            population_stats=population_stats,
        )

        return metrics

    def get_migration_candidates(
        self,
        count: int,
        select_best: bool,
    ) -> list["MigrationPacket"]:
        """获取迁移候选者

        Args:
            count: 每种群选择的数量
            select_best: True 选择最好的，False 选择最差的

        Returns:
            迁移数据包列表
        """
        from src.training.arena.migration import MigrationPacket, MigrationSystem

        candidates: list[MigrationPacket] = []

        if not self.trainer.matching_engine:
            return candidates

        current_price = self.trainer.matching_engine._orderbook.last_price

        for agent_type, population in self.trainer.populations.items():
            # 评估适应度并排序
            agent_fitnesses = population.evaluate(current_price)

            if not agent_fitnesses:
                continue

            # 选择最好或最差的
            if select_best:
                selected = agent_fitnesses[:count]
            else:
                selected = agent_fitnesses[-count:]

            # 创建迁移数据包
            for agent, fitness in selected:
                genome = agent.brain.get_genome()
                packet = MigrationPacket(
                    source_arena=self.arena_id,
                    agent_type=agent_type,
                    genome_data=MigrationSystem.serialize_genome(genome),
                    fitness=fitness,
                    generation=population.generation,
                )
                candidates.append(packet)

        return candidates

    def get_best_genomes(self, top_n: int = 10) -> dict[str, list[tuple[bytes, float]]]:
        """获取各种群的最佳个体基因组

        用于保存到共享检查点，供其他竞技场迁移使用。

        Args:
            top_n: 每个种群获取的最佳个体数量，默认 10

        Returns:
            各 Agent 类型的最佳基因组
            格式: {agent_type.value: [(genome_data, fitness), ...]}
        """
        from src.training.arena.migration import MigrationSystem

        best_genomes: dict[str, list[tuple[bytes, float]]] = {}

        if not self.trainer.matching_engine:
            return best_genomes

        current_price = self.trainer.matching_engine._orderbook.last_price

        for agent_type, population in self.trainer.populations.items():
            # 评估适应度并排序
            agent_fitnesses = population.evaluate(current_price)

            if not agent_fitnesses:
                continue

            # 取 top_n 个最佳个体
            top_agents = agent_fitnesses[:top_n]
            genomes_list: list[tuple[bytes, float]] = []

            for agent, fitness in top_agents:
                genome = agent.brain.get_genome()
                genome_data = MigrationSystem.serialize_genome(genome)
                genomes_list.append((genome_data, fitness))

            best_genomes[agent_type.value] = genomes_list

        return best_genomes

    def inject_genomes(self, packets: list["MigrationPacket"]) -> None:
        """注入迁入的 genome

        将迁入的 genome 注入到对应种群中，替换最差个体。

        Args:
            packets: 迁移数据包列表
        """
        from src.training.arena.migration import MigrationSystem

        for packet in packets:
            population = self.trainer.populations.get(packet.agent_type)
            if population is None:
                continue

            # 反序列化 genome
            genome = MigrationSystem.deserialize_genome(packet.genome_data)

            # 注入到种群中
            self._inject_genome_to_population(population, genome)

    def _inject_genome_to_population(
        self,
        population: "Population",
        genome: "neat.DefaultGenome",
    ) -> None:
        """将 genome 注入到种群中（替换最差个体）

        Args:
            population: 目标种群
            genome: 要注入的 genome
        """
        neat_pop = population.neat_pop

        # 找到最差个体的 genome_id
        worst_id = min(
            neat_pop.population.keys(),
            key=lambda gid: neat_pop.population[gid].fitness or 0.0,
        )

        # 生成新 ID
        new_id = max(neat_pop.population.keys()) + 1
        genome.key = new_id

        # 替换
        neat_pop.population[new_id] = genome
        del neat_pop.population[worst_id]

        # 清理旧 Agent 对象，防止内存泄漏
        population._cleanup_old_agents()

        # 重建 agents
        genomes = list(neat_pop.population.items())
        population.agents = population.create_agents(genomes)

    def get_checkpoint_data(self) -> dict:
        """获取检查点数据

        Returns:
            检查点数据
        """
        return {
            "arena_id": self.arena_id,
            "seed": self._seed,
            "trainer": self.trainer.save_checkpoint_data(),
            "metrics": list(self.metrics.episode_history),
        }

    def load_checkpoint_data(self, checkpoint: dict) -> None:
        """加载检查点数据

        Args:
            checkpoint: 检查点数据
        """
        self.trainer.load_checkpoint_data(checkpoint["trainer"])

        # 恢复指标历史
        if "metrics" in checkpoint:
            self.metrics.episode_history.clear()
            self.metrics.episode_history.extend(checkpoint["metrics"])

    def stop(self) -> None:
        """停止竞技场"""
        self.trainer.stop()

        # 关闭种群的线程池
        for population in self.trainer.populations.values():
            population.shutdown_executor()
