"""联盟训练器"""
from __future__ import annotations

import gc
import gzip
import pickle

import numpy as np
from ctypes import CDLL, c_int
from pathlib import Path
from typing import Any, Callable

from src.config.config import AgentType, Config
from src.core.log_engine.logger import get_logger
from src.training.arena import MultiArenaConfig, ParallelArenaTrainer
from src.training.league.arena_allocator import ArenaAllocator, ArenaAllocation
from src.training.league.config import LeagueTrainingConfig
from src.training.league.league_fitness import LeagueFitnessAggregator, GeneralizationAdvantageStats
from src.training.league.multi_gen_cache import MultiGenerationNetworkCache
from src.training.league.opponent_pool_manager import OpponentPoolManager
from src.training.population import SubPopulationManager, _serialize_genomes_numpy


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
        market_avg_return: float,
    ) -> None:
        """收集各竞技场的适应度用于泛化优势比计算"""
        if arena_id not in self._current_round_arena_fitnesses:
            self._current_round_arena_fitnesses[arena_id] = {}

        # 如果同一轮有多个 episode，累加后在计算时平均
        if agent_type in self._current_round_arena_fitnesses[arena_id]:
            self._current_round_arena_fitnesses[arena_id][agent_type] += fitness
        else:
            self._current_round_arena_fitnesses[arena_id][agent_type] = fitness.copy()

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

        # 1. 分配竞技场
        if self.pool_manager.has_any_historical_opponents():
            self._current_allocation = self.arena_allocator.allocate(self.pool_manager)
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

        # 5. 检查里程碑保存
        if self.generation > 0 and self.generation % self.league_config.milestone_interval == 0:
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
            round_stats['generalization_advantage'] = stats.advantages
            round_stats['baseline_avg_fitness'] = stats.baseline_avg
            round_stats['generalization_avg_fitness'] = stats.generalization_avg

            is_converged, converged_by_type = self.fitness_aggregator.check_convergence()
            round_stats['is_converged'] = is_converged
            round_stats['converged_by_type'] = converged_by_type
            round_stats['first_convergence_generation'] = (
                self.fitness_aggregator.get_first_convergence_generation()
            )

            self._log_generalization_advantage(stats, is_converged, converged_by_type)
        else:
            round_stats['generalization_advantage'] = None
            round_stats['baseline_avg_fitness'] = None
            round_stats['generalization_avg_fitness'] = None
            round_stats['is_converged'] = False
            round_stats['converged_by_type'] = None
            round_stats['first_convergence_generation'] = None

        # 清空（释放内存）
        self._current_round_arena_fitnesses.clear()

    def _log_generalization_advantage(
        self,
        stats: GeneralizationAdvantageStats,
        is_converged: bool,
        converged_by_type: dict[AgentType, bool],
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
            adv = stats.advantages[agent_type]
            baseline = stats.baseline_avg[agent_type]
            gen = stats.generalization_avg[agent_type]
            conv = converged_by_type.get(agent_type, False)

            if conv:
                status = "已收敛"
            elif adv > 0.01:
                status = "击败历史对手"
            elif adv < -0.01:
                status = "不如历史表现"
            else:
                status = "趋于收敛"

            lines.append(
                f"  {agent_type.name}: 泛化优势={adv:+.4f} "
                f"(基准={baseline:.4f}, 泛化={gen:.4f}) [{status}]"
            )

        if is_converged and first_conv_gen == stats.generation:
            # 首次收敛
            lines.append("  >>> 所有物种已收敛，可以考虑结束训练 <<<")

        self.logger.info("\n".join(lines))

    def _save_milestone(self) -> None:
        """保存里程碑到对手池"""
        if self.pool_manager is None:
            return

        self.logger.info(f"保存里程碑: 第 {self.generation} 代")

        # 收集所有类型的 Main Agents 基因组数据
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
                    if hasattr(sub_pop, 'neat_pop'):
                        genomes = sub_pop.neat_pop.population
                        genome_data[i] = _serialize_genomes_numpy(genomes)
                sub_pop_counts[agent_type] = len(pop_or_manager.sub_populations)
                agent_counts[agent_type] = sum(
                    len(sp.agents) for sp in pop_or_manager.sub_populations
                )
            else:
                # 单个 Population
                if hasattr(pop_or_manager, 'neat_pop'):
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

        # 保存到对手池
        self.pool_manager.add_snapshot(
            generation=self.generation,
            genome_data_map=genome_data_map,
            network_data_map=None,  # 暂不保存网络参数
            fitness_map=fitness_map,
            source='main_agents',
            add_reason='milestone',
            sub_population_counts=sub_pop_counts,
            agent_counts=agent_counts,
        )

    def save_checkpoint(self, path: str) -> None:
        """保存检查点

        保存内容：
        - 父类检查点数据
        - 对手池索引
        - 统计信息

        Args:
            path: 检查点文件路径
        """
        # 获取父类检查点数据
        # 临时调用父类保存，然后修改
        parent_path = path + ".parent.tmp"
        super().save_checkpoint(parent_path)

        # 加载父类数据
        with gzip.open(parent_path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        # 删除临时文件
        Path(parent_path).unlink()

        # 添加联盟训练数据
        checkpoint_data['league_training'] = {
            'generation': self.generation,
            'pool_sizes': self.pool_manager.get_pool_sizes() if self.pool_manager else {},
            'last_injection_generation': self._last_injection_generation,
        }

        # 保存
        with gzip.open(path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        # 保存对手池索引
        if self.pool_manager:
            self.pool_manager.save_all()

        self.logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str) -> None:
        """加载检查点

        Args:
            path: 检查点文件路径
        """
        # 加载父类检查点
        super().load_checkpoint(path)

        # 加载联盟训练数据
        with gzip.open(path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        if 'league_training' in checkpoint_data:
            league_data = checkpoint_data['league_training']
            self._last_injection_generation = league_data.get('last_injection_generation', 0)

        # 加载对手池
        if self.pool_manager:
            self.pool_manager.load_all()

        self.logger.info(f"检查点已加载: {path}")

    def stop(self) -> None:
        """停止训练并清理资源"""
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
                round_count += 1

                # 检查点回调
                if checkpoint_callback and self.generation % self.config.training.checkpoint_interval == 0:
                    checkpoint_callback(self.generation)

                # 进度回调
                if progress_callback:
                    progress_callback(stats)

                # 内存清理
                if self.generation % 10 == 0:
                    gc.collect()
                    try:
                        libc = CDLL("libc.so.6")
                        libc.malloc_trim(c_int(0))
                    except Exception:
                        pass
        finally:
            self._is_running = False

        self.logger.info(f"联盟训练完成，共 {round_count} 轮")
