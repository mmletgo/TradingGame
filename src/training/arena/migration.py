"""迁移系统模块

负责管理竞技场之间的 Agent 基因组迁移。
"""

import pickle
import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.config import AgentType
    from src.training.arena.config import MigrationStrategy


@dataclass
class MigrationPacket:
    """迁移数据包

    封装迁移过程中需要传输的 genome 数据。

    Attributes:
        source_arena: 来源竞技场 ID
        agent_type: Agent 类型
        genome_data: 序列化的 genome（pickle bytes）
        fitness: 适应度
        generation: 代数
    """
    source_arena: int
    agent_type: "AgentType"
    genome_data: bytes
    fitness: float
    generation: int


class MigrationSystem:
    """迁移系统

    负责决定迁移方向和执行迁移策略。

    Attributes:
        num_arenas: 竞技场数量
        strategy: 迁移策略
    """

    num_arenas: int
    strategy: "MigrationStrategy"
    _ring_offset: int

    def __init__(
        self,
        num_arenas: int,
        strategy: "MigrationStrategy",
    ) -> None:
        """初始化迁移系统

        Args:
            num_arenas: 竞技场数量
            strategy: 迁移策略
        """
        from src.training.arena.config import MigrationStrategy

        self.num_arenas = num_arenas
        self.strategy = strategy
        self._ring_offset = 0

    def plan_migrations(
        self,
        candidates: list[MigrationPacket],
    ) -> dict[int, list[MigrationPacket]]:
        """规划迁移

        根据迁移策略决定每个候选者应该迁移到哪个竞技场。

        Args:
            candidates: 所有候选者

        Returns:
            {目标竞技场ID: [要迁入的数据包]}
        """
        from src.training.arena.config import MigrationStrategy

        if self.strategy == MigrationStrategy.RING:
            return self._plan_ring_migration(candidates)
        elif self.strategy == MigrationStrategy.RANDOM:
            return self._plan_random_migration(candidates)
        elif self.strategy == MigrationStrategy.BEST_TO_WORST:
            return self._plan_best_to_worst_migration(candidates)
        else:
            return self._plan_ring_migration(candidates)

    def _plan_ring_migration(
        self,
        candidates: list[MigrationPacket],
    ) -> dict[int, list[MigrationPacket]]:
        """环形迁移：Arena[i] 的候选者迁移到 Arena[(i+1) % N]

        Args:
            candidates: 所有候选者

        Returns:
            迁移分配
        """
        migrations: dict[int, list[MigrationPacket]] = {
            i: [] for i in range(self.num_arenas)
        }

        for packet in candidates:
            # 目标 = 来源 + 1 (mod N)
            target_arena = (packet.source_arena + 1) % self.num_arenas
            migrations[target_arena].append(packet)

        return migrations

    def _plan_random_migration(
        self,
        candidates: list[MigrationPacket],
    ) -> dict[int, list[MigrationPacket]]:
        """随机迁移：随机选择目标竞技场（排除来源）

        Args:
            candidates: 所有候选者

        Returns:
            迁移分配
        """
        migrations: dict[int, list[MigrationPacket]] = {
            i: [] for i in range(self.num_arenas)
        }

        arena_ids = list(range(self.num_arenas))
        for packet in candidates:
            # 随机选择目标（排除来源）
            possible_targets = [
                aid for aid in arena_ids if aid != packet.source_arena
            ]
            if possible_targets:
                target_arena = random.choice(possible_targets)
                migrations[target_arena].append(packet)

        return migrations

    def _plan_best_to_worst_migration(
        self,
        candidates: list[MigrationPacket],
    ) -> dict[int, list[MigrationPacket]]:
        """最好到最差：高适应度竞技场的个体迁移到低适应度竞技场

        Args:
            candidates: 所有候选者

        Returns:
            迁移分配
        """
        # 按来源竞技场分组
        by_arena: dict[int, list[MigrationPacket]] = {}
        for packet in candidates:
            by_arena.setdefault(packet.source_arena, []).append(packet)

        # 计算每个竞技场的平均适应度
        arena_fitness: dict[int, float] = {}
        for arena_id, packets in by_arena.items():
            if packets:
                arena_fitness[arena_id] = sum(p.fitness for p in packets) / len(packets)
            else:
                arena_fitness[arena_id] = 0.0

        # 按适应度排序（高到低）
        sorted_arenas = sorted(
            arena_fitness.keys(),
            key=lambda aid: arena_fitness.get(aid, 0.0),
            reverse=True,
        )

        # 初始化迁移分配
        migrations: dict[int, list[MigrationPacket]] = {
            i: [] for i in range(self.num_arenas)
        }

        # 前半部分（好的）迁移到后半部分（差的）
        n = len(sorted_arenas)
        for i in range(n // 2):
            source = sorted_arenas[i]
            target = sorted_arenas[n - 1 - i]
            for packet in by_arena.get(source, []):
                migrations[target].append(packet)

        return migrations

    @staticmethod
    def serialize_genome(genome: object) -> bytes:
        """序列化 genome

        Args:
            genome: NEAT genome 对象

        Returns:
            序列化的 bytes
        """
        return pickle.dumps(genome)

    @staticmethod
    def deserialize_genome(data: bytes) -> object:
        """反序列化 genome

        Args:
            data: 序列化的 bytes

        Returns:
            NEAT genome 对象
        """
        return pickle.loads(data)
