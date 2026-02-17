"""联盟训练器"""
from __future__ import annotations

import gc
import pickle
import threading

import numpy as np
from ctypes import CDLL, c_int
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass

from src.config.config import AgentType, Config
from src.core.log_engine.logger import get_logger
from src.training.arena import MultiArenaConfig, ParallelArenaTrainer
from src.training.league.arena_allocator import ArenaAllocator, ArenaAllocation
from src.training.league.config import LeagueTrainingConfig
from src.training.league.league_fitness import LeagueFitnessAggregator, GeneralizationAdvantageStats
from src.training.league.multi_gen_cache import MultiGenerationNetworkCache
from src.training.league.opponent_pool_manager import OpponentPoolManager
from src.training.population import SubPopulationManager, _serialize_genomes_numpy, _serialize_species_data


@dataclass
class SpeciesFreezeState:
    """物种冻结状态"""
    is_frozen: bool = False
    freeze_generation: int = 0           # 冻结时的代数
    freeze_baseline_fitness: float = 0.0  # 冻结时的 baseline 平均适应度
    freeze_elite_fitness: float = 0.0     # 冻结时的精英 baseline 平均适应度
    thaw_count: int = 0                   # 解冻次数


class LeagueTrainer(ParallelArenaTrainer):
    """联盟训练器

    管理按类型分离的联盟训练，包括：
    - 四种 Agent 类型的 Main Agents
    - 四种类型各自的对手池

    继承 ParallelArenaTrainer，复用多竞技场并行推理能力。
    """

    def __init__(
        self,
        config: Config,
        multi_config: MultiArenaConfig,
        league_config: LeagueTrainingConfig,
    ) -> None:
        """初始化

        Args:
            config: 全局配置
            multi_config: 多竞技场配置
            league_config: 联盟训练配置
        """
        # 调用父类初始化
        super().__init__(config, multi_config)

        self.league_config = league_config
        self.logger = get_logger("league_trainer")

        # 联盟训练组件（延迟初始化）
        self.pool_manager: OpponentPoolManager | None = None
        self.multi_cache: MultiGenerationNetworkCache | None = None
        self.arena_allocator: ArenaAllocator | None = None
        self.fitness_aggregator: LeagueFitnessAggregator | None = None

        # 当前竞技场分配
        self._current_allocation: ArenaAllocation | None = None

        # 统计
        self._last_injection_generation: int = 0

        # 当前轮次各竞技场的适应度（用于泛化优势比计算）
        self._current_round_arena_fitnesses: dict[int, dict[AgentType, np.ndarray]] = {}

        # 冻结状态
        self._freeze_states: dict[AgentType, SpeciesFreezeState] = {
            agent_type: SpeciesFreezeState() for agent_type in AgentType
        }

        # 里程碑异步保存
        self._milestone_thread: threading.Thread | None = None
        self._checkpoint_thread: threading.Thread | None = None

    def setup(self) -> None:
        """初始化

        顺序（避免 COW 内存泄漏）：
        1. 调用父类 setup（创建 Worker 池、种群、竞技场状态等）
        2. 创建联盟训练组件
        3. 加载对手池
        """
        # 1. 调用父类 setup
        super().setup()

        # 2. 创建联盟训练组件
        self.pool_manager = OpponentPoolManager(self.league_config)
        self.multi_cache = MultiGenerationNetworkCache(self.config)
        self.arena_allocator = ArenaAllocator(
            self.league_config,
            self.multi_config.num_arenas,
        )
        self.fitness_aggregator = LeagueFitnessAggregator(self.league_config)

        # 3. 加载对手池
        self.pool_manager.load_all()

        self.logger.info("联盟训练器初始化完成")
        self.logger.info(f"对手池大小: {self.pool_manager.get_pool_sizes()}")

    def _on_arena_fitness_collected(
        self,
        arena_id: int,
        agent_type: AgentType,
        fitness: np.ndarray,
        current_price: float,
    ) -> None:
        """收集各竞技场的适应度用于泛化优势比计算"""
        if arena_id not in self._current_round_arena_fitnesses:
            self._current_round_arena_fitnesses[arena_id] = {}

        # 如果同一轮有多个 episode，累加后在计算时平均
        if agent_type in self._current_round_arena_fitnesses[arena_id]:
            self._current_round_arena_fitnesses[arena_id][agent_type] += fitness
        else:
            self._current_round_arena_fitnesses[arena_id][agent_type] = fitness.copy()

    def _build_fitness_map(
        self,
        avg_fitness: dict[tuple[AgentType, int], np.ndarray],
    ) -> dict[tuple[AgentType, int], np.ndarray]:
        """构建进化所需的适应度映射（排除冻结物种）

        冻结物种不参与进化，从 fitness_map 中排除。
        Worker 不收到进化命令 → 基因组保持不变。
        """
        fitness_map = super()._build_fitness_map(avg_fitness)

        # 排除冻结物种
        if self.league_config.freeze_on_convergence:
            frozen_keys = set()
            for agent_type, state in self._freeze_states.items():
                if state.is_frozen:
                    # 删除该类型的所有 sub_pop 的 key
                    for key in list(fitness_map.keys()):
                        if key[0] == agent_type:
                            frozen_keys.add(key)
            for key in frozen_keys:
                del fitness_map[key]

        return fitness_map

    def run_round(
        self,
        episode_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """运行一轮训练

        流程：
        1. 分配竞技场（确定每个竞技场中各类型的来源）
        2. 确保所需的历史对手网络已缓存
        3. 运行所有竞技场的 episodes
        4. 按类型和角色收集适应度
        5. 检查里程碑保存
        6. 清理对手池

        Args:
            episode_callback: Episode 回调函数

        Returns:
            统计信息字典
        """
        assert self.pool_manager is not None
        assert self.arena_allocator is not None
        assert self.fitness_aggregator is not None

        # 1. 清理旧 allocation
        if self._current_allocation is not None:
            self._current_allocation.assignments.clear()
            self._current_allocation = None

        # 分配竞技场
        if self.pool_manager.has_any_historical_opponents():
            frozen_types = {t for t, s in self._freeze_states.items() if s.is_frozen}
            self._current_allocation = self.arena_allocator.allocate(
                self.pool_manager, frozen_types if frozen_types else None,
                current_generation=self.generation,
            )
        else:
            # 对手池为空时，使用默认分配
            self._current_allocation = self.arena_allocator.allocate_no_historical()

        self.logger.debug(f"竞技场分配完成: {len(self._current_allocation.assignments)} 个竞技场")

        # 清空当前轮次适应度数据
        self._current_round_arena_fitnesses.clear()

        # 记录是否有历史对手
        has_historical = self.pool_manager.has_any_historical_opponents()

        # 2. 确保历史对手网络已缓存
        self._ensure_historical_networks_cached()

        # 3-4. 运行 episodes 并收集适应度（复用父类逻辑）
        round_stats = super().run_round(episode_callback=episode_callback)

        # 计算泛化优势比
        self._compute_and_log_generalization_advantage(round_stats, has_historical)

        # 更新历史对手胜率
        if has_historical and self._current_allocation is not None:
            self.pool_manager.update_win_rates_from_round(
                allocation=self._current_allocation,
                arena_fitnesses=self._current_round_arena_fitnesses,
                ema_alpha=self.league_config.pfsp_win_rate_ema_alpha,
            )
            self.pool_manager.save_all()

        # 清空适应度数据（释放内存）
        self._current_round_arena_fitnesses.clear()
        gc.collect(0)

        # 检查冻结/解冻
        if self.league_config.freeze_on_convergence:
            self._check_freeze_thaw(round_stats)

        # 5. 检查里程碑保存
        if self.generation > 0:
            self._save_milestone()

        # 6. 清理对手池
        self.pool_manager.cleanup_all(self.generation)

        # 添加联盟训练统计
        round_stats['pool_sizes'] = self.pool_manager.get_pool_sizes()

        return round_stats

    def _ensure_historical_networks_cached(self) -> None:
        """确保所有需要的历史对手网络已缓存"""
        if self._current_allocation is None or self.multi_cache is None:
            return

        for assignment in self._current_allocation.assignments:
            for agent_type, source_config in assignment.agent_sources.items():
                if source_config.source == 'historical' and source_config.entry_id:
                    self.multi_cache.ensure_cached(
                        agent_type,
                        source_config.entry_id,
                        self.pool_manager,
                    )

    def _compute_and_log_generalization_advantage(
        self,
        round_stats: dict[str, Any],
        has_historical: bool,
    ) -> None:
        """计算并记录泛化优势比"""
        round_stats['has_historical_opponents'] = has_historical

        if not has_historical or self._current_allocation is None:
            round_stats['generalization_advantage'] = None
            round_stats['baseline_avg_fitness'] = None
            round_stats['generalization_avg_fitness'] = None
            round_stats['is_converged'] = False
            round_stats['converged_by_type'] = None
            round_stats['first_convergence_generation'] = None
            return

        # 多 episode 时取平均
        episodes = self.multi_config.episodes_per_arena
        if episodes > 1:
            for arena_id in self._current_round_arena_fitnesses:
                for agent_type in self._current_round_arena_fitnesses[arena_id]:
                    self._current_round_arena_fitnesses[arena_id][agent_type] /= episodes

        # 计算泛化优势比
        stats = self.fitness_aggregator.compute_generalization_advantage(
            generation=self.generation,
            allocation=self._current_allocation,
            arena_fitnesses=self._current_round_arena_fitnesses,
        )

        if stats is not None:
            # 种群级别
            round_stats['generalization_advantage'] = stats.advantages
            round_stats['baseline_avg_fitness'] = stats.baseline_avg
            round_stats['generalization_avg_fitness'] = stats.generalization_avg

            # 精英级别
            round_stats['elite_generalization_advantage'] = stats.elite_advantages
            round_stats['elite_baseline_avg_fitness'] = stats.elite_baseline_avg
            round_stats['elite_generalization_avg_fitness'] = stats.elite_generalization_avg

            is_converged, pop_converged_by_type, elite_converged_by_type = (
                self.fitness_aggregator.check_convergence()
            )
            round_stats['is_converged'] = is_converged
            round_stats['pop_converged_by_type'] = pop_converged_by_type
            round_stats['elite_converged_by_type'] = elite_converged_by_type
            round_stats['first_convergence_generation'] = (
                self.fitness_aggregator.get_first_convergence_generation()
            )

            self._log_generalization_advantage(
                stats, is_converged, pop_converged_by_type, elite_converged_by_type
            )
        else:
            # 种群级别
            round_stats['generalization_advantage'] = None
            round_stats['baseline_avg_fitness'] = None
            round_stats['generalization_avg_fitness'] = None
            # 精英级别
            round_stats['elite_generalization_advantage'] = None
            round_stats['elite_baseline_avg_fitness'] = None
            round_stats['elite_generalization_avg_fitness'] = None
            # 收敛状态
            round_stats['is_converged'] = False
            round_stats['pop_converged_by_type'] = None
            round_stats['elite_converged_by_type'] = None
            round_stats['first_convergence_generation'] = None

    def _check_freeze_thaw(self, round_stats: dict[str, Any]) -> None:
        """检查并执行冻结/解冻逻辑

        1. 未冻结物种：若双重收敛 → 冻结
        2. 已冻结物种：每 N 代复评一次
        3. 所有物种冻结 → 标记训练完成

        Args:
            round_stats: 当前轮次统计信息
        """
        pop_converged = round_stats.get('pop_converged_by_type')
        elite_converged = round_stats.get('elite_converged_by_type')
        stats_available = pop_converged is not None and elite_converged is not None

        # 1. 检查未冻结物种是否需要冻结
        if stats_available:
            for agent_type in AgentType:
                state = self._freeze_states[agent_type]
                if state.is_frozen:
                    continue

                # 双重收敛检查
                is_dual_converged = (
                    pop_converged.get(agent_type, False) and
                    elite_converged.get(agent_type, False)
                )
                if is_dual_converged and self.generation >= self.league_config.min_freeze_generation:
                    # 获取当前 baseline 适应度作为基准
                    baseline_avg = round_stats.get('baseline_avg_fitness', {})
                    elite_baseline_avg = round_stats.get('elite_baseline_avg_fitness', {})

                    state.is_frozen = True
                    state.freeze_generation = self.generation
                    state.freeze_baseline_fitness = baseline_avg.get(agent_type, 0.0)
                    state.freeze_elite_fitness = elite_baseline_avg.get(agent_type, 0.0)

                    self.logger.info(
                        f"物种 {agent_type.name} 已冻结 (第 {self.generation} 代, "
                        f"baseline={state.freeze_baseline_fitness:.4f}, "
                        f"elite_baseline={state.freeze_elite_fitness:.4f})"
                    )

        # 2. 每代复评已冻结物种（baseline 数据每代都有）
        for agent_type in AgentType:
            state = self._freeze_states[agent_type]
            if not state.is_frozen:
                continue
            if self.generation > state.freeze_generation:
                self._reevaluate_frozen_species(agent_type, round_stats)

        # 3. 检查是否所有物种都已冻结
        all_frozen = all(s.is_frozen for s in self._freeze_states.values())
        round_stats['all_species_frozen'] = all_frozen
        if all_frozen:
            self.logger.info(">>> 所有物种已冻结，训练即将完成 <<<")

    def _reevaluate_frozen_species(
        self,
        agent_type: AgentType,
        round_stats: dict[str, Any],
    ) -> None:
        """复评冻结物种

        比较当前 baseline 适应度与冻结时的基准：
        - 下降超过阈值 → 解冻
        - 否则 → 保持冻结

        Args:
            agent_type: 要复评的物种类型
            round_stats: 当前轮次统计信息
        """
        state = self._freeze_states[agent_type]
        threshold = self.league_config.freeze_thaw_threshold

        # 获取当前 baseline 适应度
        baseline_avg = round_stats.get('baseline_avg_fitness', {})
        current_fitness = baseline_avg.get(agent_type, 0.0)

        # 计算下降比例
        freeze_fitness = state.freeze_baseline_fitness
        if abs(freeze_fitness) > 1e-10:
            drop_ratio = (freeze_fitness - current_fitness) / abs(freeze_fitness)
        else:
            drop_ratio = 0.0

        self.logger.info(
            f"物种 {agent_type.name} 复评 (冻结于第 {state.freeze_generation} 代): "
            f"冻结时baseline={freeze_fitness:.4f}, 当前baseline={current_fitness:.4f}, "
            f"下降比例={drop_ratio:.4f}, 阈值={threshold:.4f}"
        )

        if drop_ratio > threshold:
            # 解冻
            state.is_frozen = False
            state.thaw_count += 1
            self.logger.info(
                f"物种 {agent_type.name} 已解冻 (下降 {drop_ratio:.2%} > {threshold:.2%}), "
                f"累计解冻 {state.thaw_count} 次"
            )
        else:
            self.logger.info(
                f"物种 {agent_type.name} 保持冻结 (下降 {drop_ratio:.2%} <= {threshold:.2%})"
            )

    def _log_generalization_advantage(
        self,
        stats: GeneralizationAdvantageStats,
        is_converged: bool,
        pop_converged_by_type: dict[AgentType, bool],
        elite_converged_by_type: dict[AgentType, bool],
    ) -> None:
        """输出泛化优势比日志"""
        # 获取首次收敛代数
        first_conv_gen = self.fitness_aggregator.get_first_convergence_generation()

        # 构建标题行
        if first_conv_gen is not None:
            lines = [f"第 {stats.generation} 代泛化优势比 (首次收敛于第 {first_conv_gen} 代):"]
        else:
            lines = [f"第 {stats.generation} 代泛化优势比:"]

        for agent_type in AgentType:
            # 种群级别
            adv = stats.advantages[agent_type]
            baseline = stats.baseline_avg[agent_type]
            gen = stats.generalization_avg[agent_type]
            pop_conv = pop_converged_by_type.get(agent_type, False)

            # 精英级别
            elite_adv = stats.elite_advantages[agent_type]
            elite_baseline = stats.elite_baseline_avg[agent_type]
            elite_gen = stats.elite_generalization_avg[agent_type]
            elite_conv = elite_converged_by_type.get(agent_type, False)

            # 状态判断
            freeze_state = self._freeze_states.get(agent_type)
            if freeze_state and freeze_state.is_frozen:
                status = f"已冻结(第{freeze_state.freeze_generation}代起)"
            elif pop_conv and elite_conv:
                status = "双重收敛"
            elif pop_conv:
                status = "种群收敛"
            elif elite_conv:
                status = "精英收敛"
            elif adv > 0.01:
                status = "击败历史对手"
            elif adv < -0.01:
                status = "不如历史表现"
            else:
                status = "趋于收敛"

            lines.append(
                f"  {agent_type.name}: "
                f"种群={adv:+.4f}(基准={baseline:.4f},泛化={gen:.4f}) | "
                f"精英={elite_adv:+.4f}(基准={elite_baseline:.4f},泛化={elite_gen:.4f}) "
                f"[{status}]"
            )

        if is_converged and first_conv_gen == stats.generation:
            # 首次收敛
            lines.append("  >>> 所有物种已双重收敛，可以考虑结束训练 <<<")

        self.logger.info("\n".join(lines))

    def _save_milestone(self) -> None:
        """保存里程碑到对手池（异步）

        数据收集在主线程完成（访问 populations），
        实际 I/O 操作在后台线程执行。
        """
        if self.pool_manager is None:
            return

        # 等待上一次异步保存完成
        if self._milestone_thread is not None and self._milestone_thread.is_alive():
            self._milestone_thread.join()

        self.logger.info(f"保存里程碑: 第 {self.generation} 代")

        # === 数据收集（主线程，快速） ===
        genome_data_map: dict[AgentType, dict[int, tuple]] = {}
        fitness_map: dict[AgentType, float] = {}
        agent_counts: dict[AgentType, int] = {}
        sub_pop_counts: dict[AgentType, int] = {}

        for agent_type, pop_or_manager in self.populations.items():
            genome_data: dict[int, tuple] = {}

            # 检查是否是 SubPopulationManager
            if hasattr(pop_or_manager, 'sub_populations'):
                # SubPopulationManager
                for i, sub_pop in enumerate(pop_or_manager.sub_populations):
                    # 优先使用 pending 数据
                    if getattr(sub_pop, '_genomes_dirty', False) and getattr(sub_pop, '_pending_genome_data', None) is not None:
                        genome_data[i] = sub_pop._pending_genome_data
                    elif hasattr(sub_pop, 'neat_pop'):
                        genomes = sub_pop.neat_pop.population
                        genome_data[i] = _serialize_genomes_numpy(genomes)
                sub_pop_counts[agent_type] = len(pop_or_manager.sub_populations)
                agent_counts[agent_type] = sum(
                    len(sp.agents) for sp in pop_or_manager.sub_populations
                )
            else:
                # 单个 Population - 优先使用 pending 数据
                if getattr(pop_or_manager, '_genomes_dirty', False) and getattr(pop_or_manager, '_pending_genome_data', None) is not None:
                    genome_data[0] = pop_or_manager._pending_genome_data
                elif hasattr(pop_or_manager, 'neat_pop'):
                    genomes = pop_or_manager.neat_pop.population
                    genome_data[0] = _serialize_genomes_numpy(genomes)
                sub_pop_counts[agent_type] = 1
                agent_counts[agent_type] = len(pop_or_manager.agents)

            genome_data_map[agent_type] = genome_data

            # 计算平均适应度
            agents = (
                pop_or_manager.agents
                if hasattr(pop_or_manager, 'agents')
                else [a for sp in pop_or_manager.sub_populations for a in sp.agents]
            )
            fitnesses = [
                a.brain.get_genome().fitness
                for a in agents
                if a.brain.get_genome().fitness is not None
            ]
            fitness_map[agent_type] = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0

        # === 异步 I/O（后台线程） ===
        generation = self.generation
        pool_manager = self.pool_manager

        def _do_save() -> None:
            try:
                pool_manager.add_snapshot(
                    generation=generation,
                    genome_data_map=genome_data_map,
                    network_data_map=None,
                    fitness_map=fitness_map,
                    source='main_agents',
                    add_reason='milestone',
                    sub_population_counts=sub_pop_counts,
                    agent_counts=agent_counts,
                )
            except Exception:
                import traceback
                traceback.print_exc()

        self._milestone_thread = threading.Thread(
            target=_do_save,
            daemon=True,
            name=f"milestone_save_gen{generation}",
        )
        self._milestone_thread.start()

    def save_checkpoint(self, path: str) -> None:
        """保存检查点

        直接构建完整 checkpoint_data（内联父类序列化逻辑 + league 数据），
        序列化到内存字节后在后台线程异步写盘，避免阻塞训练主循环。

        Args:
            path: 检查点文件路径
        """
        # 等待上一次 checkpoint 线程完成
        if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
            self._checkpoint_thread.join()

        # 等待异步里程碑保存完成（避免与检查点保存冲突）
        if self._milestone_thread is not None and self._milestone_thread.is_alive():
            self._milestone_thread.join()

        # 内联父类序列化逻辑，直接构建 checkpoint_data
        checkpoint_data: dict[str, Any] = {
            "checkpoint_version": 2,
            "generation": self.generation,
            "populations": {},
        }

        for agent_type, population in self.populations.items():
            if isinstance(population, SubPopulationManager):
                pop_data: dict[str, Any] = {
                    "is_sub_population_manager": True,
                    "sub_population_count": population.sub_population_count,
                    "sub_populations": [],
                }
                for sub_pop in population.sub_populations:
                    if sub_pop._genomes_dirty and sub_pop._pending_genome_data is not None:
                        genome_data = sub_pop._pending_genome_data
                        species_data = sub_pop._pending_species_data or (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
                    else:
                        sub_pop._cleanup_neat_history()
                        genome_data = _serialize_genomes_numpy(sub_pop.neat_pop.population)
                        species_data = _serialize_species_data(sub_pop.neat_pop.species)
                    sub_pop_data = {
                        "generation": sub_pop.generation,
                        "genome_data": genome_data,
                        "species_data": species_data,
                    }
                    pop_data["sub_populations"].append(sub_pop_data)
                checkpoint_data["populations"][agent_type] = pop_data
            else:
                if population._genomes_dirty and population._pending_genome_data is not None:
                    genome_data = population._pending_genome_data
                    species_data = population._pending_species_data or (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
                else:
                    population._cleanup_neat_history()
                    genome_data = _serialize_genomes_numpy(population.neat_pop.population)
                    species_data = _serialize_species_data(population.neat_pop.species)
                checkpoint_data["populations"][agent_type] = {
                    "generation": population.generation,
                    "genome_data": genome_data,
                    "species_data": species_data,
                }

        # 添加联盟训练数据
        checkpoint_data['league_training'] = {
            'generation': self.generation,
            'pool_sizes': self.pool_manager.get_pool_sizes() if self.pool_manager else {},
            'last_injection_generation': self._last_injection_generation,
            'freeze_states': {
                agent_type.value: {
                    'is_frozen': state.is_frozen,
                    'freeze_generation': state.freeze_generation,
                    'freeze_baseline_fitness': state.freeze_baseline_fitness,
                    'freeze_elite_fitness': state.freeze_elite_fitness,
                    'thaw_count': state.thaw_count,
                }
                for agent_type, state in self._freeze_states.items()
            },
        }

        # 序列化到内存字节
        checkpoint_bytes: bytes = pickle.dumps(checkpoint_data, protocol=pickle.HIGHEST_PROTOCOL)

        # 释放原始数据
        del checkpoint_data
        gc.collect(0)

        # 捕获闭包变量
        logger = self.logger
        pool_manager = self.pool_manager

        def _write_checkpoint_background(
            data: bytes,
            checkpoint_path: str,
        ) -> None:
            """后台线程：写入检查点文件并保存对手池索引"""
            p = Path(checkpoint_path)
            p.parent.mkdir(parents=True, exist_ok=True)

            with open(p, 'wb') as f:
                f.write(data)

            # 保存对手池索引
            if pool_manager is not None:
                pool_manager.save_all()

            logger.info(f"检查点已保存: {checkpoint_path}")

        # 启动后台线程写盘
        self._checkpoint_thread = threading.Thread(
            target=_write_checkpoint_background,
            args=(checkpoint_bytes, path),
            daemon=True,
        )
        self._checkpoint_thread.start()

    def load_checkpoint(self, path: str) -> None:
        """加载检查点

        Args:
            path: 检查点文件路径
        """
        # 加载父类检查点（父类已有 magic bytes 检测，支持 plain pickle 和 gzip）
        super().load_checkpoint(path)

        # 加载联盟训练数据（plain pickle）
        with open(path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        if 'league_training' in checkpoint_data:
            league_data = checkpoint_data['league_training']
            self._last_injection_generation = league_data.get('last_injection_generation', 0)

            # 恢复冻结状态
            freeze_states_data = league_data.get('freeze_states', {})
            for agent_type in AgentType:
                data = freeze_states_data.get(agent_type.value, {})
                if data:
                    state = self._freeze_states[agent_type]
                    state.is_frozen = data.get('is_frozen', False)
                    state.freeze_generation = data.get('freeze_generation', 0)
                    state.freeze_baseline_fitness = data.get('freeze_baseline_fitness', 0.0)
                    state.freeze_elite_fitness = data.get('freeze_elite_fitness', 0.0)
                    state.thaw_count = data.get('thaw_count', 0)

            if any(s.is_frozen for s in self._freeze_states.values()):
                frozen_names = [t.name for t, s in self._freeze_states.items() if s.is_frozen]
                self.logger.info(f"已恢复冻结状态: {', '.join(frozen_names)}")

        # 加载对手池
        if self.pool_manager:
            self.pool_manager.load_all()

        self.logger.info(f"检查点已加载: {path}")

    def stop(self) -> None:
        """停止训练并清理资源"""
        # 等待异步里程碑保存完成
        if self._milestone_thread is not None and self._milestone_thread.is_alive():
            self.logger.info("等待里程碑保存完成...")
            self._milestone_thread.join(timeout=30.0)

        # 等待异步检查点保存完成
        if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
            self.logger.info("等待检查点保存完成...")
            self._checkpoint_thread.join(timeout=30.0)

        # 保存对手池索引
        if self.pool_manager:
            self.pool_manager.save_all()

        # 清理缓存
        if self.multi_cache:
            self.multi_cache.clear_all()

        # 调用父类停止
        super().stop()

        self.logger.info("联盟训练器已停止")

    def train(
        self,
        num_rounds: int | None = None,
        checkpoint_callback: Callable[[int], None] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        episode_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """主训练循环

        Args:
            num_rounds: 训练轮数，None 表示无限训练
            checkpoint_callback: 检查点回调
            progress_callback: 进度回调
            episode_callback: Episode 回调，每个 episode 完成后调用
        """
        self.logger.info(f"开始联盟训练，轮数: {num_rounds or '无限'}")

        self._is_running = True
        round_count = 0
        try:
            while num_rounds is None or round_count < num_rounds:
                if not self._is_running:
                    break

                stats = self.run_round(episode_callback=episode_callback)

                # 所有物种冻结时退出训练
                if stats.get('all_species_frozen', False):
                    self.logger.info("所有物种已冻结，训练完成")
                    # 执行回调（round_count 会在下面自然递增）
                    if checkpoint_callback:
                        checkpoint_callback(self.generation)
                    if progress_callback:
                        progress_callback(stats)
                    round_count += 1
                    break
                round_count += 1

                # 检查点回调
                if checkpoint_callback:
                    checkpoint_callback(self.generation)

                # 进度回调
                if progress_callback:
                    progress_callback(stats)

                # 每代执行轻量级 NEAT 历史清理
                for population in self.populations.values():
                    if isinstance(population, SubPopulationManager):
                        for sub_pop in population.sub_populations:
                            sub_pop._cleanup_neat_history_light()
                    else:
                        population._cleanup_neat_history_light()

                # 内存清理（增强版）
                if self.generation % 5 == 0:
                    # 【内存泄漏修复】定期清理 NEAT 历史数据（每 5 代）
                    for population in self.populations.values():
                        if isinstance(population, SubPopulationManager):
                            for sub_pop in population.sub_populations:
                                sub_pop._cleanup_neat_history()
                        else:
                            population._cleanup_neat_history()

                    # 【内存泄漏修复】清理对手池中 get_entry() 积累的大数据
                    if self.pool_manager is not None:
                        for pool in self.pool_manager.pools.values():
                            pool.clear_memory_cache()

                    gc.collect()
                    try:
                        libc = CDLL("libc.so.6")
                        libc.malloc_trim(c_int(0))
                    except Exception:
                        pass
        finally:
            self._is_running = False

        self.logger.info(f"联盟训练完成，共 {round_count} 轮")
