"""竞技场分配器"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.config.config import AgentType
from src.training.league.config import LeagueTrainingConfig
from src.training.league.opponent_pool_manager import OpponentPoolManager


@dataclass
class AgentSourceConfig:
    """单个 Agent 类型在竞技场中的来源配置"""
    source: Literal['current', 'historical']
    entry_id: str | None = None  # historical 时需要指定


@dataclass
class ArenaAssignment:
    """单个竞技场的分配配置"""
    arena_id: int
    purpose: Literal['baseline', 'generalization_test']
    # 四种类型各自的来源
    agent_sources: dict[AgentType, AgentSourceConfig] = field(default_factory=dict)
    # 泛化测试时，哪种类型使用当前代 Main
    target_type: AgentType | None = None


@dataclass
class ArenaAllocation:
    """完整的竞技场分配方案"""
    assignments: list[ArenaAssignment]

    # 按目的分类的竞技场 ID
    baseline_arena_ids: list[int] = field(default_factory=list)
    # 按类型的泛化测试竞技场 ID
    generalization_arena_ids: dict[AgentType, list[int]] = field(default_factory=dict)

    def get_assignment(self, arena_id: int) -> ArenaAssignment | None:
        """获取指定竞技场的分配"""
        for assignment in self.assignments:
            if assignment.arena_id == arena_id:
                return assignment
        return None


class ArenaAllocator:
    """竞技场分配器"""

    def __init__(self, config: LeagueTrainingConfig, num_arenas: int) -> None:
        """初始化

        Args:
            config: 联盟训练配置
            num_arenas: 竞技场数量
        """
        self.config = config
        self.num_arenas = num_arenas

    def allocate(self, pool_manager: OpponentPoolManager, frozen_types: set[AgentType] | None = None) -> ArenaAllocation:
        """分配竞技场

        分配策略：
        1. 基准竞技场：全当前代 Main vs Main
        2. 泛化测试竞技场：按类型轮流测试，一种类型用 Main，其他用历史
           - 冻结物种不需要泛化测试，改为额外的 baseline 竞技场

        Args:
            pool_manager: 对手池管理器
            frozen_types: 已冻结的物种集合

        Returns:
            竞技场分配方案
        """
        assignments: list[ArenaAssignment] = []
        baseline_ids: list[int] = []
        generalization_ids: dict[AgentType, list[int]] = {t: [] for t in AgentType}

        arena_id = 0

        # 1. 基准竞技场：全当前代
        for _ in range(self.config.num_baseline_arenas):
            if arena_id >= self.num_arenas:
                break
            assignment = ArenaAssignment(
                arena_id=arena_id,
                purpose='baseline',
                agent_sources={
                    agent_type: AgentSourceConfig(source='current')
                    for agent_type in AgentType
                },
            )
            assignments.append(assignment)
            baseline_ids.append(arena_id)
            arena_id += 1

        # 2. 泛化测试竞技场：按类型轮流
        for agent_type in AgentType:
            for _ in range(self.config.num_generalization_arenas_per_type):
                if arena_id >= self.num_arenas:
                    break

                # 冻结物种不需要泛化测试，改为额外的 baseline 竞技场
                if frozen_types and agent_type in frozen_types:
                    assignment = ArenaAssignment(
                        arena_id=arena_id,
                        purpose='baseline',
                        agent_sources={
                            t: AgentSourceConfig(source='current')
                            for t in AgentType
                        },
                    )
                    assignments.append(assignment)
                    baseline_ids.append(arena_id)
                    arena_id += 1
                    continue

                # 采样历史对手
                historical = pool_manager.sample_opponents_for_arena(
                    target_type=agent_type,
                    strategy=self.config.sampling_strategy,
                )

                agent_sources: dict[AgentType, AgentSourceConfig] = {}
                for t in AgentType:
                    if t == agent_type:
                        # 目标类型使用当前代 Main
                        agent_sources[t] = AgentSourceConfig(source='current')
                    else:
                        # 其他类型使用历史版本
                        entry_id = historical.get(t)
                        if entry_id:
                            agent_sources[t] = AgentSourceConfig(
                                source='historical',
                                entry_id=entry_id,
                            )
                        else:
                            # 如果没有历史对手，使用当前代
                            agent_sources[t] = AgentSourceConfig(source='current')

                assignment = ArenaAssignment(
                    arena_id=arena_id,
                    purpose='generalization_test',
                    agent_sources=agent_sources,
                    target_type=agent_type,
                )
                assignments.append(assignment)
                generalization_ids[agent_type].append(arena_id)
                arena_id += 1

        return ArenaAllocation(
            assignments=assignments,
            baseline_arena_ids=baseline_ids,
            generalization_arena_ids=generalization_ids,
        )

    def allocate_no_historical(self) -> ArenaAllocation:
        """分配竞技场（无历史对手时使用）

        - 基准竞技场：全当前代
        - 泛化测试竞技场：不分配（因为没有历史对手），改为额外的基准竞技场

        Returns:
            竞技场分配方案
        """
        assignments: list[ArenaAssignment] = []
        baseline_ids: list[int] = []

        arena_id = 0

        # 1. 基准竞技场：全当前代
        for _ in range(self.config.num_baseline_arenas):
            if arena_id >= self.num_arenas:
                break
            assignment = ArenaAssignment(
                arena_id=arena_id,
                purpose='baseline',
                agent_sources={
                    agent_type: AgentSourceConfig(source='current')
                    for agent_type in AgentType
                },
            )
            assignments.append(assignment)
            baseline_ids.append(arena_id)
            arena_id += 1

        # 2. 泛化测试：跳过（因为没有历史对手）
        # 将这些竞技场也分配给基准
        generalization_total = self.config.num_generalization_arenas_per_type * len(AgentType)
        for _ in range(generalization_total):
            if arena_id >= self.num_arenas:
                break
            assignment = ArenaAssignment(
                arena_id=arena_id,
                purpose='baseline',
                agent_sources={
                    agent_type: AgentSourceConfig(source='current')
                    for agent_type in AgentType
                },
            )
            assignments.append(assignment)
            baseline_ids.append(arena_id)
            arena_id += 1

        return ArenaAllocation(
            assignments=assignments,
            baseline_arena_ids=baseline_ids,
            generalization_arena_ids={t: [] for t in AgentType},
        )
