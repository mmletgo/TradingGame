"""联盟训练器"""
from __future__ import annotations

import gc
import gzip
import pickle
from ctypes import CDLL, c_int
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.config.config import AgentType, Config
from src.core.log_engine.logger import get_logger
from src.training.arena import MultiArenaConfig, ParallelArenaTrainer
from src.training.league.arena_allocator import ArenaAllocator, ArenaAllocation
from src.training.league.config import LeagueTrainingConfig
from src.training.league.exploiter_manager import ExploiterManager
from src.training.league.league_fitness import LeagueFitnessAggregator
from src.training.league.multi_gen_cache import MultiGenerationNetworkCache
from src.training.league.opponent_entry import OpponentEntry, OpponentMetadata
from src.training.league.opponent_pool_manager import OpponentPoolManager
from src.training.population import _serialize_genomes_numpy


class LeagueTrainer(ParallelArenaTrainer):
    """联盟训练器

    管理按类型分离的联盟训练，包括：
    - 四种 Agent 类型的 Main Agents
    - 四种类型各自的 League Exploiter 和 Main Exploiter
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
        self.exploiter_manager: ExploiterManager | None = None
        self.fitness_aggregator: LeagueFitnessAggregator | None = None

        # 当前竞技场分配
        self._current_allocation: ArenaAllocation | None = None

        # 统计
        self._last_injection_generation: int = 0

    def setup(self) -> None:
        """初始化

        顺序（避免 COW 内存泄漏）：
        1. 调用父类 setup（创建 Worker 池、种群、竞技场状态等）
        2. 创建联盟训练组件
        3. 初始化 Exploiter
        4. 加载对手池
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
        self.exploiter_manager = ExploiterManager(self.league_config, self.config)
        self.fitness_aggregator = LeagueFitnessAggregator(self.league_config)

        # 3. 初始化 Exploiter
        main_pop_sizes = {
            agent_type: self.config.agents[agent_type].count
            for agent_type in AgentType
        }
        self.exploiter_manager.setup(main_pop_sizes)

        # 4. 加载对手池
        self.pool_manager.load_all()

        self.logger.info("联盟训练器初始化完成")
        self.logger.info(f"对手池大小: {self.pool_manager.get_pool_sizes()}")
        self.logger.info(f"Exploiter 大小: {self.exploiter_manager.get_exploiter_sizes()}")

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
        5. 分别进化 Main Agents 和各类型的 Exploiters
        6. 更新对手池胜率统计
        7. 检查是否注入对手池（每 N 代）
        8. 清理对手池
        9. 保存检查点

        Args:
            episode_callback: Episode 回调函数

        Returns:
            统计信息字典
        """
        assert self.pool_manager is not None
        assert self.arena_allocator is not None
        assert self.exploiter_manager is not None
        assert self.fitness_aggregator is not None

        # 1. 分配竞技场
        if self.pool_manager.has_any_historical_opponents():
            self._current_allocation = self.arena_allocator.allocate(self.pool_manager)
        else:
            # 对手池为空时，仅使用基准竞技场
            self._current_allocation = self.arena_allocator.allocate_baseline_only()

        self.logger.debug(f"竞技场分配完成: {len(self._current_allocation.assignments)} 个竞技场")

        # 2. 确保历史对手网络已缓存
        self._ensure_historical_networks_cached()

        # 3-4. 运行 episodes 并收集适应度（复用父类逻辑）
        # 注意：这里需要修改父类的 _run_episode_all_arenas 来支持多来源
        round_stats = super().run_round(episode_callback=episode_callback)

        # 5. Exploiter 进化（如果有适应度数据）
        self._evolve_exploiters()

        # 6. 更新胜率统计
        self._update_win_rate_statistics()

        # 7. 检查是否注入对手池
        if self.generation > 0 and self.generation % self.league_config.exploiter_inject_interval == 0:
            self._check_and_inject_to_pool()

        # 8. 检查里程碑保存
        if self.generation > 0 and self.generation % self.league_config.milestone_interval == 0:
            self._save_milestone()

        # 9. 清理对手池
        self.pool_manager.cleanup_all(self.generation)

        # 添加联盟训练统计
        round_stats['pool_sizes'] = self.pool_manager.get_pool_sizes()
        round_stats['exploiter_sizes'] = self.exploiter_manager.get_exploiter_sizes()

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

    def _evolve_exploiters(self) -> None:
        """进化 Exploiter

        使用 LeagueFitnessAggregator 收集的适应度进行进化。
        """
        if self.exploiter_manager is None or not self.exploiter_manager.is_initialized():
            return

        # 获取当前价格
        if self.arena_states:
            current_price = self.arena_states[0].matching_engine.orderbook.get_mid_price()
        else:
            current_price = self.config.market.initial_price

        # 进化 League Exploiter
        for agent_type in AgentType:
            pop = self.exploiter_manager.league_exploiter_populations.get(agent_type)
            if pop and pop.agents:
                # 使用累积的适应度进行进化
                self.exploiter_manager.evolve_league_exploiter(
                    agent_type,
                    np.zeros(len(pop.agents)),  # 临时：使用零适应度
                    current_price,
                )

        # 进化 Main Exploiter
        for agent_type in AgentType:
            pop = self.exploiter_manager.main_exploiter_populations.get(agent_type)
            if pop and pop.agents:
                self.exploiter_manager.evolve_main_exploiter(
                    agent_type,
                    np.zeros(len(pop.agents)),  # 临时：使用零适应度
                    current_price,
                )

    def _update_win_rate_statistics(self) -> None:
        """更新胜率统计"""
        # TODO: 根据实际对战结果更新胜率
        pass

    def _check_and_inject_to_pool(self) -> None:
        """检查是否将当前代或 Exploiter 注入对手池"""
        if self.pool_manager is None or self.exploiter_manager is None:
            return

        for agent_type in AgentType:
            # 检查 League Exploiter
            if self.exploiter_manager.should_inject_to_pool(agent_type, 'league_exploiter'):
                result = self.exploiter_manager.get_best_for_injection(agent_type, 'league_exploiter')
                if result:
                    genome_data, avg_fitness = result
                    self._inject_to_pool(
                        agent_type,
                        genome_data,
                        avg_fitness,
                        source='league_exploiter',
                        add_reason='exploiter_win_rate',
                    )
                    self.logger.info(
                        f"将 {agent_type.value} League Exploiter 注入对手池，"
                        f"平均适应度: {avg_fitness:.4f}"
                    )

            # 检查 Main Exploiter
            if self.exploiter_manager.should_inject_to_pool(agent_type, 'main_exploiter'):
                result = self.exploiter_manager.get_best_for_injection(agent_type, 'main_exploiter')
                if result:
                    genome_data, avg_fitness = result
                    self._inject_to_pool(
                        agent_type,
                        genome_data,
                        avg_fitness,
                        source='main_exploiter',
                        add_reason='exploiter_win_rate',
                    )
                    self.logger.info(
                        f"将 {agent_type.value} Main Exploiter 注入对手池，"
                        f"平均适应度: {avg_fitness:.4f}"
                    )

    def _inject_to_pool(
        self,
        agent_type: AgentType,
        genome_data: dict[int, tuple],
        avg_fitness: float,
        source: str,
        add_reason: str,
    ) -> None:
        """注入到对手池

        Args:
            agent_type: Agent 类型
            genome_data: 基因组数据
            avg_fitness: 平均适应度
            source: 来源
            add_reason: 添加原因
        """
        if self.pool_manager is None:
            return

        entry_id = f"gen_{self.generation:05d}_{source}"

        metadata = OpponentMetadata(
            entry_id=entry_id,
            agent_type=agent_type,
            source=source,
            source_generation=self.generation,
            add_reason=add_reason,
            avg_fitness=avg_fitness,
            agent_count=sum(len(g[0]) for g in genome_data.values()),
            sub_population_count=len(genome_data),
        )

        entry = OpponentEntry(
            metadata=metadata,
            genome_data=genome_data,
        )

        pool = self.pool_manager.get_pool(agent_type)
        pool.add_entry(entry)

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
        - Exploiter 种群（如果有）
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
            'exploiter_win_rates': {
                'league': dict(self.exploiter_manager.league_exploiter_win_rates) if self.exploiter_manager else {},
                'main': dict(self.exploiter_manager.main_exploiter_win_rates) if self.exploiter_manager else {},
            },
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

            # 恢复 Exploiter 胜率统计
            if self.exploiter_manager:
                win_rates = league_data.get('exploiter_win_rates', {})
                if 'league' in win_rates:
                    for agent_type, rates in win_rates['league'].items():
                        if isinstance(agent_type, str):
                            agent_type = AgentType(agent_type)
                        self.exploiter_manager.league_exploiter_win_rates[agent_type] = rates
                if 'main' in win_rates:
                    for agent_type, rate in win_rates['main'].items():
                        if isinstance(agent_type, str):
                            agent_type = AgentType(agent_type)
                        self.exploiter_manager.main_exploiter_win_rates[agent_type] = rate

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
