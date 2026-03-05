"""联盟训练器"""

from __future__ import annotations

import gc
import logging
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
from src.training.arena.arena_worker import AgentInfo
from src.training.arena.shared_network_memory import SharedNetworkMemory, SharedNetworkMetadata
from src.training.league.arena_allocator import HybridArenaAllocator, HybridSamplingResult
from src.training.league.config import LeagueTrainingConfig
from src.training.league.league_fitness import (
    HybridFitnessAggregator,
    GenerationalComparisonStats,
)
from src.training.league.opponent_pool_manager import OpponentPoolManager
from src.training.population import (
    SubPopulationManager,
    _serialize_genomes_numpy,
    _serialize_species_data,
    _pack_network_params_numpy,
    _unpack_network_params_numpy,
    _concat_network_params_numpy,
    _deserialize_genomes_numpy,
    _extract_and_pack_all_network_params,
)


# 缓存 libc 句柄，避免每次调用 CDLL
try:
    _libc: CDLL | None = CDLL("libc.so.6")
except OSError:
    _libc = None


@dataclass
class SpeciesFreezeState:
    """物种冻结状态"""

    is_frozen: bool = False
    freeze_generation: int = 0  # 冻结时的代数
    freeze_baseline_fitness: float = 0.0  # 冻结时的 baseline 平均适应度
    freeze_elite_fitness: float = 0.0  # 冻结时的精英 baseline 平均适应度
    thaw_count: int = 0  # 解冻次数


class LeagueTrainer(ParallelArenaTrainer):
    """联盟训练器

    管理按类型分离的联盟训练，包括：
    - 两种 Agent 类型的 Main Agents
    - 两种类型各自的对手池
    - 历史精英网络注入（混合竞技场）

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
        self.arena_allocator: HybridArenaAllocator | None = None
        self.fitness_aggregator: HybridFitnessAggregator | None = None

        # 当前竞技场分配
        self._current_sampling_result: HybridSamplingResult | None = None

        # 统计
        self._last_injection_generation: int = 0

        # 当前代网络参数缓存（用于与历史精英合并）
        self._current_gen_network_params: dict[AgentType, tuple[np.ndarray, ...]] = {}

        # 当前轮次的历史精英网络参数
        self._current_historical_params: dict[AgentType, tuple[np.ndarray, ...]] = {}

        # 当前轮次的历史Agent信息列表
        self._current_historical_agent_infos: list[AgentInfo] = []

        # 冻结状态
        self._freeze_states: dict[AgentType, SpeciesFreezeState] = {
            agent_type: SpeciesFreezeState() for agent_type in AgentType
        }

        # 预进化适应度缓存（供里程碑保存使用）
        self._pre_evolution_fitness: dict[tuple[AgentType, int], np.ndarray] = {}

        # 里程碑异步保存
        self._milestone_thread: threading.Thread | None = None
        self._checkpoint_thread: threading.Thread | None = None
        self._milestone_save_failed: bool = False

        # 追踪上一轮是否有历史Agent（用于检测有→无变化）
        self._had_historical_last_round: bool = False

        # 每轮进化后缓存 per-sub-pop 网络参数，供 _save_milestone 写入 networks.npz
        self._per_subpop_network_params: dict[AgentType, dict[int, tuple[np.ndarray, ...]]] = {}

    def setup(self) -> None:
        """初始化

        顺序（避免 COW 内存泄漏）：
        1. 调用父类 setup（创建 Worker 池、种群、竞技场状态等）
        2. 创建联盟训练组件
        3. 加载对手池
        """
        # 噪声交易者增强（在父类 setup 前设置，确保 Worker 使用增强参数）
        from dataclasses import replace
        self.config.noise_trader = replace(
            self.config.noise_trader,
            count=self.league_config.hybrid_noise_trader_count,
            quantity_mu=self.league_config.hybrid_noise_trader_quantity_mu,
        )

        # 1. 调用父类 setup
        super().setup()

        # 2. 创建联盟训练组件
        self.pool_manager = OpponentPoolManager(self.league_config)
        self.arena_allocator = HybridArenaAllocator(self.league_config)
        self.fitness_aggregator = HybridFitnessAggregator(self.league_config)

        # 3. 加载对手池
        self.pool_manager.load_all()

        self.logger.info("联盟训练器初始化完成")
        self.logger.info(f"对手池大小: {self.pool_manager.get_pool_sizes()}")

    def _build_agent_infos(self) -> list[AgentInfo]:
        """Override: 包含历史代Agent信息"""
        infos = super()._build_agent_infos()
        if self._current_historical_agent_infos:
            infos.extend(self._current_historical_agent_infos)
        return infos

    def _update_populations_from_evolution(
        self,
        evolution_results: dict[tuple[AgentType, int], tuple],
        deserialize_genomes: bool = False,
    ) -> None:
        """Override: 进化结果更新前保存 per-sub-pop 网络参数供里程碑写入 networks.npz

        冻结物种不参与进化（不在 evolution_results 中），但里程碑保存仍需其
        per-subpop 网络参数以写入 networks.npz，避免后续加载时走昂贵的
        _reconstruct_network_data 回退路径。
        """
        per_subpop: dict[AgentType, dict[int, tuple[np.ndarray, ...]]] = {}
        for (agent_type, sub_pop_id), (_, network_params_data, _) in evolution_results.items():
            per_subpop.setdefault(agent_type, {})[sub_pop_id] = network_params_data

        # 保留冻结物种的 per-subpop 网络参数（冻结物种不进化，网络不变）
        if self.league_config.freeze_on_convergence:
            for agent_type, state in self._freeze_states.items():
                if state.is_frozen and agent_type not in per_subpop:
                    if agent_type in self._per_subpop_network_params:
                        per_subpop[agent_type] = self._per_subpop_network_params[agent_type]

        self._per_subpop_network_params = per_subpop
        super()._update_populations_from_evolution(evolution_results, deserialize_genomes)

    def _sync_networks_to_workers(self) -> None:
        """Override: 合并历史精英网络参数后同步

        在父类逻辑基础上：
        1. 从 _cached_network_params_data 获取当前代网络参数
        2. 保存副本到 _current_gen_network_params（供下轮使用）
        3. 如果有历史精英，拼接到当前代参数之后
        4. 通过共享内存发送合并后的参数给所有 Worker
        """
        if self._arena_worker_pool is None:
            return

        # 获取当前代网络参数（与父类相同逻辑）
        network_params: dict[AgentType, tuple[np.ndarray, ...]] = {}
        for agent_type, pop in self.populations.items():
            cached = getattr(pop, '_cached_network_params_data', None)
            if cached is not None:
                network_params[agent_type] = cached
                pop._cached_network_params_data = None
                continue

            # 回退1：使用上一轮保存的参数（用于 run_round 开始时的手动同步）
            if agent_type in self._current_gen_network_params:
                network_params[agent_type] = self._current_gen_network_params[agent_type]
                continue

            # 回退2：首次同步（setup 阶段），从 Agent 的 brain.network 提取参数
            try:
                agents = pop.agents
                params_list: list[dict[str, np.ndarray | int]] = []
                for agent in agents:
                    params_list.append(agent.brain.network.get_params())
                packed = _pack_network_params_numpy(params_list)
                network_params[agent_type] = packed
                params_list.clear()
                del params_list
            except Exception as e:
                self.logger.warning(
                    f"无法提取 {agent_type.value} 的网络参数: {e}"
                )
                continue

        if not network_params:
            return

        # 保存当前代参数
        self._current_gen_network_params = dict(network_params)

        # 合并历史精英参数
        merged_params: dict[AgentType, tuple[np.ndarray, ...]] = {}
        for agent_type, params in network_params.items():
            if agent_type in self._current_historical_params:
                hist_params = self._current_historical_params[agent_type]
                merged_params[agent_type] = _concat_network_params_numpy([
                    params, hist_params
                ])
                self.logger.info(
                    f"网络合并 {agent_type.name}: "
                    f"当前代={len(params[0])} + 历史={len(hist_params[0])} "
                    f"= 合并后={len(merged_params[agent_type][0])}"
                )
            else:
                merged_params[agent_type] = params

        # 使用共享内存同步（复制父类的 SharedMemory 逻辑）
        new_shm_memories: dict[AgentType, SharedNetworkMemory] = {}
        try:
            metadata_map: dict[AgentType, SharedNetworkMetadata] = {}

            for agent_type, params in merged_params.items():
                shm_mem = SharedNetworkMemory()
                metadata = shm_mem.create_and_fill(
                    agent_type=agent_type,
                    network_params=params,
                    generation=self.generation,
                )
                metadata_map[agent_type] = metadata
                new_shm_memories[agent_type] = shm_mem

            # 发送元数据给所有 Worker，等待 ack
            self._arena_worker_pool.attach_shared_networks(metadata_map)

            # Worker 都已 ack，可以安全清理上一代的共享内存
            for shm_mem in self._prev_shared_network_memories.values():
                shm_mem.close_and_unlink()
            self._prev_shared_network_memories.clear()

            # 当前的变为上一代
            self._prev_shared_network_memories = self._shared_network_memories
            self._shared_network_memories = new_shm_memories

        except Exception as e:
            self.logger.warning(
                f"共享内存同步失败，回退到 Queue 模式: {e}"
            )
            # 清理已创建的共享内存
            for shm_mem in new_shm_memories.values():
                shm_mem.close_and_unlink()
            # 回退到原有模式
            self._arena_worker_pool.update_networks(merged_params)

    def _build_fitness_map(
        self,
        avg_fitness: dict[tuple[AgentType, int], np.ndarray],
    ) -> dict[tuple[AgentType, int], np.ndarray]:
        """构建进化所需的适应度映射（排除冻结物种）

        冻结物种不参与进化，从 fitness_map 中排除。
        Worker 不收到进化命令 → 基因组保持不变。

        在进化前缓存预进化适应度供里程碑保存使用。
        """
        # 缓存预进化适应度（在 NEAT 进化前调用，此时 avg_fitness 是当前代的真实适应度）
        self._pre_evolution_fitness = {
            key: arr.copy() for key, arr in avg_fitness.items()
            if key[1] < 1000  # 只保留当前代（排除历史Agent的sub_pop_id >= 1000）
        }

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

    def _sync_genomes_if_needed(self) -> None:
        """从 Worker 进程同步基因组数据到主进程

        在 lite 模式下，进化后 _pending_genome_data 为 None（基因组留在 Worker 中）。
        调用此方法后，所有 sub_pop 的 _pending_genome_data 被设置，
        后续的 _save_milestone() 和 save_checkpoint() 可直接引用。
        """
        if self.evolution_worker_pool is None:
            return

        needs_sync: bool = False
        for agent_type, population in self.populations.items():
            if isinstance(population, SubPopulationManager):
                for sub_pop in population.sub_populations:
                    if sub_pop._genomes_dirty and sub_pop._pending_genome_data is None:
                        needs_sync = True
                        break
            else:
                if (
                    population._genomes_dirty
                    and population._pending_genome_data is None
                ):
                    needs_sync = True
            if needs_sync:
                break

        if not needs_sync:
            return

        genomes_map = self.evolution_worker_pool.sync_genomes_from_workers()
        for (agent_type, sub_pop_id), genome_data in genomes_map.items():
            population = self.populations.get(agent_type)
            if population is None:
                continue
            if isinstance(population, SubPopulationManager):
                if sub_pop_id < len(population.sub_populations):
                    population.sub_populations[sub_pop_id]._pending_genome_data = (
                        genome_data
                    )
            else:
                if sub_pop_id == 0:
                    population._pending_genome_data = genome_data
        del genomes_map

    def _prepare_historical_agents(self) -> None:
        """从采样结果中提取精英网络，构建历史Agent信息

        对每种 AgentType，从采样到的历史 entries 中：
        1. 加载 entry 数据（pre_evolution_fitness + network_data）
        2. 调用 extract_elite_networks 提取 Top 精英
        3. 构建 AgentInfo 列表，使用特殊 ID 和 sub_pop_id

        历史Agent ID 方案：
        - 散户: 10,000,000 + entry_index * 1,000,000 + local_index
        - 做市商: 20,000,000 + entry_index * 1,000,000 + local_index

        历史Agent sub_pop_id 方案: 1000 + entry_index
        """
        self._current_historical_agent_infos = []
        self._current_historical_params = {}

        if self._current_sampling_result is None:
            return

        assert self.pool_manager is not None

        for agent_type in AgentType:
            entry_ids: list[str] = self._current_sampling_result.sampled_entries.get(
                agent_type, []
            )
            if not entry_ids:
                continue

            elite_params_list: list[tuple[np.ndarray, ...]] = []
            historical_infos: list[AgentInfo] = []

            # 当前代 agent 数量（用于 network_index 偏移）
            population = self.populations[agent_type]
            if isinstance(population, SubPopulationManager):
                current_gen_count: int = sum(
                    len(sub_pop.agents) for sub_pop in population.sub_populations
                )
                agent_config = population.sub_populations[0].agent_config
            else:
                current_gen_count = len(population.agents)
                agent_config = population.agent_config

            network_index_offset: int = current_gen_count

            # ID 基数
            id_base: int = (
                10_000_000 if agent_type == AgentType.RETAIL_PRO else 20_000_000
            )

            for entry_index, entry_id in enumerate(entry_ids):
                pool = self.pool_manager.get_pool(agent_type)
                entry = pool.get_entry(entry_id, load_networks=True)
                if entry is None:
                    continue

                # 如果没有 network_data 但有 genome_data，从基因组重建网络参数
                if entry.network_data is None and entry.genome_data is not None:
                    entry.network_data = _reconstruct_network_data(
                        entry.genome_data, agent_type, self.config,
                    )
                    if entry.network_data is not None:
                        self.logger.debug(
                            f"从基因组重建网络参数: {entry_id} "
                            f"({sum(len(v[0]) for v in entry.network_data.values())} 个网络)"
                        )

                if entry.network_data is None:
                    continue

                # 提取精英网络
                n_elite, packed = extract_elite_networks(
                    pre_evolution_fitness=entry.pre_evolution_fitness,
                    network_data=entry.network_data,
                    elite_ratio=self.league_config.historical_elite_ratio,
                    genome_data=entry.genome_data,
                )

                if n_elite == 0 or not packed:
                    # 清理 entry 数据
                    entry.genome_data = None
                    entry.network_data = None
                    entry.pre_evolution_fitness = None
                    continue

                elite_params_list.append(packed)

                # 构建 AgentInfo
                sub_pop_id: int = 1000 + entry_index
                agent_id_base: int = id_base + entry_index * 1_000_000

                for i in range(n_elite):
                    historical_infos.append(AgentInfo(
                        agent_id=agent_id_base + i,
                        agent_type=agent_type,
                        sub_pop_id=sub_pop_id,
                        network_index=network_index_offset + i,
                        initial_balance=agent_config.initial_balance,
                        leverage=agent_config.leverage,
                        maintenance_margin_rate=agent_config.maintenance_margin_rate,
                        maker_fee_rate=agent_config.maker_fee_rate,
                        taker_fee_rate=agent_config.taker_fee_rate,
                        is_historical=True,
                        historical_entry_id=entry_id,
                    ))

                network_index_offset += n_elite

                # 更新采样结果（用于日志）
                self._current_sampling_result.elite_networks.setdefault(
                    agent_type, {}
                )[entry_id] = (n_elite, packed)

                # 清理 entry 数据释放内存
                entry.genome_data = None
                entry.network_data = None
                entry.pre_evolution_fitness = None

            if elite_params_list:
                # 拼接所有精英参数
                self._current_historical_params[agent_type] = (
                    _concat_network_params_numpy(elite_params_list)
                )
                self._current_historical_agent_infos.extend(historical_infos)

                total_elites: int = sum(
                    p[0].shape[0] for p in elite_params_list
                )
                self._current_sampling_result.total_elite_counts[agent_type] = (
                    total_elites
                )

                self.logger.info(
                    f"历史精英提取 {agent_type.name}: "
                    f"{len(entry_ids)} entries → {total_elites} 精英网络"
                )

            del elite_params_list

    def run_round(
        self,
        episode_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """运行一轮训练

        混合竞技场流程：
        1. PFSP 采样历史对手（带新鲜度约束）
        2. 提取历史精英网络，构建混合 Agent 集合
        3. 更新 Workers 的 agent_infos 和合并后的网络参数
        4. 运行 episodes + NEAT 进化（复用父类逻辑）
        5. 计算代际适应度对比
        6. 保存里程碑
        7. 清理对手池

        Args:
            episode_callback: Episode 回调函数

        Returns:
            统计信息字典
        """
        assert self.pool_manager is not None
        assert self.arena_allocator is not None
        assert self.fitness_aggregator is not None

        # M3: 等待上一轮 checkpoint 线程完成（防止竞态）
        if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
            self._checkpoint_thread.join()

        # 检查上一轮里程碑保存是否失败
        if self._milestone_save_failed:
            self.logger.warning("上一轮里程碑保存失败，请检查磁盘空间和对手池目录权限")
            self._milestone_save_failed = False

        # 1. 清理旧采样结果和历史数据
        self._current_sampling_result = None
        self._current_historical_agent_infos = []
        self._current_historical_params = {}

        # 2. 采样历史对手
        has_historical: bool = self.pool_manager.has_any_historical_opponents()
        if has_historical:
            self._current_sampling_result = self.arena_allocator.sample_historical(
                self.pool_manager,
                current_generation=self.generation,
            )

        if self._current_sampling_result is not None:
            total_entries: int = sum(
                len(ids) for ids in self._current_sampling_result.sampled_entries.values()
            )
            self.logger.debug(f"历史对手采样完成: 共 {total_entries} 个 entries")
        else:
            self.logger.debug("无历史对手可采样")

        # 3. 提取精英网络并构建历史 Agent
        self._prepare_historical_agents()

        # 4. 如果有历史 Agent，更新 Workers 的 agent_infos 和网络
        if self._current_historical_agent_infos:
            combined_infos: list[AgentInfo] = self._build_agent_infos()
            self._arena_worker_pool.update_agent_infos(combined_infos)
            del combined_infos

            # 手动同步合并后的网络参数
            self._sync_networks_to_workers()
        elif self._had_historical_last_round:
            # S1: 上一轮有历史Agent但本轮没有 → 需要重新发送仅当前代的 agent_infos
            current_only_infos: list[AgentInfo] = self._build_agent_infos()
            self._arena_worker_pool.update_agent_infos(current_only_infos)
            del current_only_infos

            # update_agent_infos 会创建新的空 BatchNetworkCache 并清除共享内存，
            # 必须重新同步网络参数，否则 Workers 将使用空缓存导致零交易
            self._sync_networks_to_workers()

        # 更新追踪标志
        self._had_historical_last_round = bool(self._current_historical_agent_infos)

        # 缓存进化前的代数（super().run_round() 会递增 self.generation）
        pre_evolution_generation: int = self.generation

        # 5. 运行 episodes + NEAT 进化（复用父类逻辑）
        # 父类 run_round() 会：
        # - 运行 episodes（使用合并后的网络）
        # - 收集适应度（历史 sub_pop_id >= 1000 自动不参与进化）
        # - 执行 NEAT 进化
        # - 调用 _sync_networks_to_workers()（override 会合并历史参数）
        round_stats = super().run_round(episode_callback=episode_callback)

        # 6. 计算代际适应度对比（使用进化前代数）
        self._compute_generational_comparison(round_stats, pre_evolution_generation)

        # 7. 检查冻结/解冻（使用进化前代数）
        if self.league_config.freeze_on_convergence:
            self._check_freeze_thaw(round_stats, pre_evolution_generation)

        # 8. 同步基因组 + 保存里程碑
        self._sync_genomes_if_needed()
        if self.generation > 0 and self.generation % self.league_config.milestone_interval == 0:
            self._save_milestone()

        # 等待里程碑后台保存完成（防止与步骤9/10的对手池操作竞态）
        if self._milestone_thread is not None and self._milestone_thread.is_alive():
            self._milestone_thread.join()

        # H2: 里程碑保存失败时跳过步骤9-10，避免操作不一致的对手池
        if self._milestone_save_failed:
            self.logger.warning(
                "里程碑保存失败，跳过胜率更新和对手池清理（将在下轮开始时重置标志）"
            )
        else:
            # 9. 更新历史对手胜率 + 保存对手池索引
            if has_historical:
                self._update_historical_win_rates(round_stats)
                self.pool_manager.save_all()

            # 10. 清理对手池
            self.pool_manager.cleanup_all(self.generation)

        # 添加联盟训练统计
        round_stats["pool_sizes"] = self.pool_manager.get_pool_sizes()
        round_stats["has_historical_opponents"] = has_historical
        if self._current_sampling_result is not None:
            round_stats["historical_elite_counts"] = (
                self._current_sampling_result.total_elite_counts
            )

        return round_stats

    def _compute_generational_comparison(
        self,
        round_stats: dict[str, Any],
        generation: int | None = None,
    ) -> None:
        """计算代际适应度对比

        从 avg_fitness 中过滤出当前代适应度（sub_pop_id < 1000），
        按 AgentType 合并后传给 HybridFitnessAggregator 计算代际对比。
        """
        assert self.fitness_aggregator is not None

        avg_fitness: dict[tuple[AgentType, int], np.ndarray] = round_stats.get(
            "avg_fitnesses", {}
        )

        # 过滤并合并当前代适应度
        current_gen_fitness: dict[AgentType, np.ndarray] = {}
        for (agent_type, sub_pop_id), arr in avg_fitness.items():
            if sub_pop_id >= 1000:
                continue
            if agent_type not in current_gen_fitness:
                current_gen_fitness[agent_type] = arr.copy()
            else:
                current_gen_fitness[agent_type] = np.concatenate([
                    current_gen_fitness[agent_type], arr
                ])

        # 计算代际对比
        effective_generation: int = generation if generation is not None else self.generation
        stats: GenerationalComparisonStats | None = (
            self.fitness_aggregator.compute_generational_comparison(
                generation=effective_generation,
                fitness_arrays=current_gen_fitness,
            )
        )

        # 检查收敛
        is_converged, converged_by_type = self.fitness_aggregator.check_convergence()
        first_conv_gen: int | None = (
            self.fitness_aggregator.get_first_convergence_generation()
        )

        # 存储到 round_stats
        round_stats["is_converged"] = is_converged
        round_stats["converged_by_type"] = converged_by_type
        round_stats["first_convergence_generation"] = first_conv_gen

        if stats is not None:
            round_stats["current_avg_fitness"] = stats.current_avg_fitness
            round_stats["previous_avg_fitness"] = stats.previous_avg_fitness
            round_stats["fitness_improvement"] = stats.improvement
            round_stats["elite_current_avg"] = stats.elite_current_avg
            round_stats["elite_previous_avg"] = stats.elite_previous_avg
            round_stats["elite_improvement"] = stats.elite_improvement

            self._log_generational_comparison(
                stats, is_converged, converged_by_type
            )
        else:
            round_stats["current_avg_fitness"] = None
            round_stats["previous_avg_fitness"] = None
            round_stats["fitness_improvement"] = None
            round_stats["elite_current_avg"] = None
            round_stats["elite_previous_avg"] = None
            round_stats["elite_improvement"] = None

    def _log_generational_comparison(
        self,
        stats: GenerationalComparisonStats,
        is_converged: bool,
        converged_by_type: dict[AgentType, bool],
    ) -> None:
        """输出代际适应度对比日志"""
        assert self.fitness_aggregator is not None
        first_conv_gen = self.fitness_aggregator.get_first_convergence_generation()

        if first_conv_gen is not None:
            lines: list[str] = [
                f"第 {stats.generation} 代代际对比 "
                f"(首次收敛于第 {first_conv_gen} 代):"
            ]
        else:
            lines = [f"第 {stats.generation} 代代际对比:"]

        for agent_type in AgentType:
            current = stats.current_avg_fitness.get(agent_type, 0.0)
            previous = stats.previous_avg_fitness.get(agent_type, 0.0)
            improvement = stats.improvement.get(agent_type, 0.0)
            elite_current = stats.elite_current_avg.get(agent_type, 0.0)
            elite_previous = stats.elite_previous_avg.get(agent_type, 0.0)
            elite_improvement = stats.elite_improvement.get(agent_type, 0.0)
            type_converged = converged_by_type.get(agent_type, False)

            # 状态判断
            freeze_state = self._freeze_states.get(agent_type)
            if freeze_state and freeze_state.is_frozen:
                status = f"已冻结(第{freeze_state.freeze_generation}代起)"
            elif type_converged:
                status = "已收敛"
            elif improvement > 0.01:
                status = "提升中"
            elif improvement < -0.01:
                status = "下降中"
            else:
                status = "稳定"

            lines.append(
                f"  {agent_type.name}: "
                f"种群={current:.4f}(上代={previous:.4f},变化={improvement:+.4f}) | "
                f"精英={elite_current:.4f}(上代={elite_previous:.4f},"
                f"变化={elite_improvement:+.4f}) "
                f"[{status}]"
            )

        if is_converged and first_conv_gen == stats.generation:
            lines.append("  >>> 所有物种已收敛，可以考虑结束训练 <<<")

        self.logger.info("\n".join(lines))

    def _update_historical_win_rates(
        self, round_stats: dict[str, Any]
    ) -> None:
        """更新历史对手的胜率（EMA 平滑）

        根据当前代平均适应度判断胜负：
        - 当前代同类型 Agent 平均适应度 > 0 → outcome = 1.0（击败历史对手）
        - 否则 → outcome = 0.0
        """
        if self._current_sampling_result is None or self.pool_manager is None:
            return

        current_avg = round_stats.get("current_avg_fitness")
        if not isinstance(current_avg, dict):
            # M10: 回退——从 avg_fitnesses 计算当前代平均适应度
            avg_fitnesses = round_stats.get("avg_fitnesses")
            if not isinstance(avg_fitnesses, dict):
                return
            collected: dict[AgentType, list[np.ndarray]] = {}
            for (agent_type_key, sub_pop_id), arr in avg_fitnesses.items():
                if sub_pop_id < 1000 and isinstance(arr, np.ndarray) and len(arr) > 0:
                    collected.setdefault(agent_type_key, []).append(arr)
            current_avg = {
                at: float(np.mean(np.concatenate(arrs)))
                for at, arrs in collected.items()
            }
            if not current_avg:
                return

        ema_alpha: float = self.league_config.pfsp_win_rate_ema_alpha

        for agent_type, entry_ids in self._current_sampling_result.sampled_entries.items():
            avg_fitness: float = current_avg.get(agent_type, 0.0)
            outcome: float = 1.0 if avg_fitness > 0 else 0.0

            pool = self.pool_manager.get_pool(agent_type)
            for entry_id in entry_ids:
                pool.update_entry_win_rate(
                    entry_id=entry_id,
                    target_type=agent_type,
                    outcome=outcome,
                    ema_alpha=ema_alpha,
                )

    def _check_freeze_thaw(
        self, round_stats: dict[str, Any], generation: int | None = None
    ) -> None:
        """检查并执行冻结/解冻逻辑"""
        effective_generation: int = generation if generation is not None else self.generation
        converged_by_type: dict[AgentType, bool] | None = round_stats.get(
            "converged_by_type"
        )

        # 1. 检查未冻结物种是否需要冻结
        if converged_by_type is not None:
            for agent_type in AgentType:
                state = self._freeze_states[agent_type]
                if state.is_frozen:
                    continue

                is_converged: bool = converged_by_type.get(agent_type, False)
                if (
                    is_converged
                    and effective_generation >= self.league_config.min_freeze_generation
                ):
                    # 获取当前适应度作为基准
                    current_avg = round_stats.get("current_avg_fitness", {})
                    elite_avg = round_stats.get("elite_current_avg", {})

                    state.is_frozen = True
                    state.freeze_generation = effective_generation
                    state.freeze_baseline_fitness = (
                        current_avg.get(agent_type, 0.0)
                        if isinstance(current_avg, dict) else 0.0
                    )
                    state.freeze_elite_fitness = (
                        elite_avg.get(agent_type, 0.0)
                        if isinstance(elite_avg, dict) else 0.0
                    )

                    self.logger.info(
                        f"物种 {agent_type.name} 已冻结 (第 {effective_generation} 代, "
                        f"avg={state.freeze_baseline_fitness:.4f}, "
                        f"elite={state.freeze_elite_fitness:.4f})"
                    )

        # 2. 每代复评已冻结物种
        for agent_type in AgentType:
            state = self._freeze_states[agent_type]
            if not state.is_frozen:
                continue
            if effective_generation > state.freeze_generation:
                self._reevaluate_frozen_species(agent_type, round_stats, effective_generation)

        # 3. 检查是否所有物种都已冻结
        all_frozen: bool = all(s.is_frozen for s in self._freeze_states.values())
        round_stats["all_species_frozen"] = all_frozen
        if all_frozen:
            self.logger.info(">>> 所有物种已冻结，训练即将完成 <<<")

    def _reevaluate_frozen_species(
        self,
        agent_type: AgentType,
        round_stats: dict[str, Any],
        effective_generation: int,
    ) -> None:
        """复评冻结物种"""
        state = self._freeze_states[agent_type]
        threshold = self.league_config.freeze_thaw_threshold

        # 获取当前适应度
        current_avg = round_stats.get("current_avg_fitness", {})
        current_fitness: float = (
            current_avg.get(agent_type, 0.0) if isinstance(current_avg, dict) else 0.0
        )
        elite_avg = round_stats.get("elite_current_avg", {})
        current_elite: float = (
            elite_avg.get(agent_type, 0.0) if isinstance(elite_avg, dict) else 0.0
        )

        # 计算种群平均下降比例
        freeze_fitness = state.freeze_baseline_fitness
        if abs(freeze_fitness) > 0.01:
            pop_drop: float = (freeze_fitness - current_fitness) / abs(freeze_fitness)
        else:
            pop_drop = (freeze_fitness - current_fitness) / 0.01

        # 计算精英平均下降比例
        freeze_elite = state.freeze_elite_fitness
        if abs(freeze_elite) > 0.01:
            elite_drop: float = (freeze_elite - current_elite) / abs(freeze_elite)
        else:
            elite_drop = (freeze_elite - current_elite) / 0.01

        # 取两者中更严格的（下降更多的）
        drop_ratio: float = max(pop_drop, elite_drop)

        # 复评详情：每10代 info，其余 debug
        log_level: int = logging.INFO if effective_generation % 10 == 0 else logging.DEBUG
        self.logger.log(
            log_level,
            f"物种 {agent_type.name} 复评 (冻结于第 {state.freeze_generation} 代): "
            f"种群avg: 冻结={freeze_fitness:.4f} 当前={current_fitness:.4f} 下降={pop_drop:.4f} | "
            f"精英avg: 冻结={freeze_elite:.4f} 当前={current_elite:.4f} 下降={elite_drop:.4f} | "
            f"取max={drop_ratio:.4f}, 阈值={threshold:.4f}"
        )

        if drop_ratio > threshold:
            state.is_frozen = False
            state.thaw_count += 1
            # 解冻事件始终 info
            self.logger.info(
                f"物种 {agent_type.name} 已解冻 (下降 {drop_ratio:.2%} > {threshold:.2%}), "
                f"累计解冻 {state.thaw_count} 次"
            )
        else:
            if drop_ratio < 0:
                direction = f"上升 {-drop_ratio:.2%}"
            else:
                direction = f"下降 {drop_ratio:.2%}"
            self.logger.log(
                log_level,
                f"物种 {agent_type.name} 保持冻结 ({direction}, 未下降超过阈值 {threshold:.2%})"
            )

    def _collect_genome_data(self) -> tuple[
        dict[AgentType, dict[int, tuple]],
        dict[AgentType, int],
        dict[AgentType, int],
    ]:
        """收集所有种群的基因组数据

        Returns:
            (genome_data_map, sub_pop_counts, agent_counts)
        """
        genome_data_map: dict[AgentType, dict[int, tuple]] = {}
        agent_counts: dict[AgentType, int] = {}
        sub_pop_counts: dict[AgentType, int] = {}

        for agent_type, pop_or_manager in self.populations.items():
            genome_data: dict[int, tuple] = {}

            if isinstance(pop_or_manager, SubPopulationManager):
                for i, sub_pop in enumerate(pop_or_manager.sub_populations):
                    if (
                        getattr(sub_pop, "_genomes_dirty", False)
                        and getattr(sub_pop, "_pending_genome_data", None) is not None
                    ):
                        genome_data[i] = sub_pop._pending_genome_data
                    elif hasattr(sub_pop, "neat_pop"):
                        genomes = sub_pop.neat_pop.population
                        genome_data[i] = _serialize_genomes_numpy(genomes)
                sub_pop_counts[agent_type] = len(pop_or_manager.sub_populations)
                agent_counts[agent_type] = sum(
                    len(sp.agents) for sp in pop_or_manager.sub_populations
                )
            else:
                if (
                    getattr(pop_or_manager, "_genomes_dirty", False)
                    and getattr(pop_or_manager, "_pending_genome_data", None)
                    is not None
                ):
                    genome_data[0] = pop_or_manager._pending_genome_data
                elif hasattr(pop_or_manager, "neat_pop"):
                    genomes = pop_or_manager.neat_pop.population
                    genome_data[0] = _serialize_genomes_numpy(genomes)
                sub_pop_counts[agent_type] = 1
                agent_counts[agent_type] = len(pop_or_manager.agents)

            genome_data_map[agent_type] = genome_data

        return genome_data_map, sub_pop_counts, agent_counts

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
        genome_data_map, sub_pop_counts, agent_counts = self._collect_genome_data()
        fitness_map: dict[AgentType, float] = {}

        for agent_type in self.populations:
            # 计算平均适应度（优先使用预进化适应度缓存）
            cached_fitness_arrays: list[np.ndarray] = [
                arr for (at, sp_id), arr in self._pre_evolution_fitness.items()
                if at == agent_type
            ]
            if cached_fitness_arrays:
                fitness_map[agent_type] = float(np.mean(np.concatenate(cached_fitness_arrays)))
            else:
                # 回退：从 agents 遍历
                pop_or_manager = self.populations[agent_type]
                if isinstance(pop_or_manager, SubPopulationManager):
                    agents = [a for sp in pop_or_manager.sub_populations for a in sp.agents]
                else:
                    agents = pop_or_manager.agents
                fitnesses: list[float] = [
                    a.brain.get_genome().fitness
                    for a in agents
                    if a.brain.get_genome().fitness is not None
                ]
                fitness_map[agent_type] = (
                    sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
                )

        # 构建 pre_evolution_fitness（按 AgentType 分组）
        pre_evolution_fitness_map: dict[AgentType, dict[int, np.ndarray]] = {}
        if self._pre_evolution_fitness:
            for (agent_type, sub_pop_id), fitness_arr in self._pre_evolution_fitness.items():
                if agent_type not in pre_evolution_fitness_map:
                    pre_evolution_fitness_map[agent_type] = {}
                pre_evolution_fitness_map[agent_type][sub_pop_id] = fitness_arr

        # === 网络参数收集 ===
        network_data_map: dict[AgentType, dict[int, tuple[np.ndarray, ...]]] | None = (
            self._per_subpop_network_params if self._per_subpop_network_params else None
        )

        # === 异步 I/O（后台线程） ===
        generation = self.generation
        pool_manager = self.pool_manager

        def _do_save() -> None:
            try:
                pool_manager.add_snapshot(
                    generation=generation,
                    genome_data_map=genome_data_map,
                    network_data_map=network_data_map,
                    fitness_map=fitness_map,
                    source="main_agents",
                    add_reason="milestone",
                    sub_population_counts=sub_pop_counts,
                    agent_counts=agent_counts,
                    pre_evolution_fitness_map=pre_evolution_fitness_map if pre_evolution_fitness_map else None,
                )
            except Exception:
                import traceback
                self.logger.error(f"里程碑保存失败 (gen {generation}): {traceback.format_exc()}")
                self._milestone_save_failed = True

        self._milestone_thread = threading.Thread(
            target=_do_save,
            daemon=True,
            name=f"milestone_save_gen{generation}",
        )
        self._milestone_thread.start()

        # 闭包已捕获 network_data_map 引用，释放实例属性减少内存占用
        # 保留冻结物种的 per-subpop 参数（冻结物种不进化，需跨轮次复用）
        if self.league_config.freeze_on_convergence:
            frozen_types: set[AgentType] = {
                at for at, s in self._freeze_states.items() if s.is_frozen
            }
            self._per_subpop_network_params = {
                at: data for at, data in self._per_subpop_network_params.items()
                if at in frozen_types
            }
        else:
            self._per_subpop_network_params = {}

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

        # 确保基因组数据已同步（run_round 中通常已调用，此为安全兜底）
        self._sync_genomes_if_needed()

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
                    if (
                        sub_pop._genomes_dirty
                        and sub_pop._pending_genome_data is not None
                    ):
                        genome_data = sub_pop._pending_genome_data
                        species_data = sub_pop._pending_species_data
                        if species_data is None:
                            self.logger.warning(
                                f"save_checkpoint: {agent_type.name} sub_pop {sub_pop.sub_population_id} "
                                f"species_data 为 None，使用空数组（物种分配信息丢失）"
                            )
                            species_data = (
                                np.array([], dtype=np.int32),
                                np.array([], dtype=np.int32),
                            )
                    else:
                        sub_pop._cleanup_neat_history_light()
                        genome_data = _serialize_genomes_numpy(
                            sub_pop.neat_pop.population
                        )
                        species_data = _serialize_species_data(sub_pop.neat_pop.species)
                    sub_pop_data = {
                        "generation": sub_pop.generation,
                        "genome_data": genome_data,
                        "species_data": species_data,
                    }
                    pop_data["sub_populations"].append(sub_pop_data)
                checkpoint_data["populations"][agent_type] = pop_data
            else:
                if (
                    population._genomes_dirty
                    and population._pending_genome_data is not None
                ):
                    genome_data = population._pending_genome_data
                    species_data = population._pending_species_data
                    if species_data is None:
                        self.logger.warning(
                            f"save_checkpoint: {agent_type.name} "
                            f"species_data 为 None，使用空数组（物种分配信息丢失）"
                        )
                        species_data = (
                            np.array([], dtype=np.int32),
                            np.array([], dtype=np.int32),
                        )
                else:
                    population._cleanup_neat_history_light()
                    genome_data = _serialize_genomes_numpy(
                        population.neat_pop.population
                    )
                    species_data = _serialize_species_data(population.neat_pop.species)
                checkpoint_data["populations"][agent_type] = {
                    "generation": population.generation,
                    "genome_data": genome_data,
                    "species_data": species_data,
                }

        # 添加联盟训练数据
        checkpoint_data["league_training"] = {
            "generation": self.generation,
            "pool_sizes": (
                self.pool_manager.get_pool_sizes() if self.pool_manager else {}
            ),
            "last_injection_generation": self._last_injection_generation,
            "freeze_states": {
                agent_type.value: {
                    "is_frozen": state.is_frozen,
                    "freeze_generation": state.freeze_generation,
                    "freeze_baseline_fitness": state.freeze_baseline_fitness,
                    "freeze_elite_fitness": state.freeze_elite_fitness,
                    "thaw_count": state.thaw_count,
                }
                for agent_type, state in self._freeze_states.items()
            },
            "fitness_aggregator_state": (
                self.fitness_aggregator.get_state()
                if self.fitness_aggregator is not None
                else None
            ),
        }

        # 序列化到内存字节
        checkpoint_bytes: bytes = pickle.dumps(
            checkpoint_data, protocol=pickle.HIGHEST_PROTOCOL
        )

        # 释放原始数据
        del checkpoint_data
        gc.collect(0)

        # 捕获闭包变量
        logger = self.logger

        def _write_checkpoint_background(
            data: bytes,
            checkpoint_path: str,
        ) -> None:
            """后台线程：写入检查点文件"""
            p = Path(checkpoint_path)
            p.parent.mkdir(parents=True, exist_ok=True)

            with open(p, "wb") as f:
                f.write(data)

            logger.info(f"检查点已保存: {checkpoint_path}")

        # 启动后台线程写盘
        self._checkpoint_thread = threading.Thread(
            target=_write_checkpoint_background,
            args=(checkpoint_bytes, path),
            daemon=True,
        )
        self._checkpoint_thread.start()

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """加载检查点

        Args:
            path: 检查点文件路径

        Returns:
            检查点数据字典
        """
        # 加载父类检查点（父类已有 magic bytes 检测，支持 plain pickle 和 gzip）
        checkpoint_data: dict[str, Any] = super().load_checkpoint(path)

        if "league_training" in checkpoint_data:
            league_data = checkpoint_data["league_training"]
            self._last_injection_generation = league_data.get(
                "last_injection_generation", 0
            )

            # 恢复冻结状态
            freeze_states_data = league_data.get("freeze_states", {})
            for agent_type in AgentType:
                data = freeze_states_data.get(agent_type.value, {})
                if data:
                    state = self._freeze_states[agent_type]
                    state.is_frozen = data.get("is_frozen", False)
                    state.freeze_generation = data.get("freeze_generation", 0)
                    state.freeze_baseline_fitness = data.get(
                        "freeze_baseline_fitness", 0.0
                    )
                    state.freeze_elite_fitness = data.get("freeze_elite_fitness", 0.0)
                    state.thaw_count = data.get("thaw_count", 0)

            # 恢复 fitness_aggregator 状态
            fitness_agg_state = league_data.get("fitness_aggregator_state")
            if fitness_agg_state is not None and self.fitness_aggregator is not None:
                self.fitness_aggregator.set_state(fitness_agg_state)
                history_len: int = len(fitness_agg_state.get('fitness_history', []))
                self.logger.info(
                    f"已恢复适应度历史 ({history_len} 代)"
                )

            if any(s.is_frozen for s in self._freeze_states.values()):
                frozen_names = [
                    t.name for t, s in self._freeze_states.items() if s.is_frozen
                ]
                self.logger.info(f"已恢复冻结状态: {', '.join(frozen_names)}")

        # 加载对手池
        if self.pool_manager:
            self.pool_manager.load_all()

        # 恢复后标记为有历史（确保下一轮正确处理）
        if self.pool_manager and self.pool_manager.has_any_historical_opponents():
            self._had_historical_last_round = True

        self.logger.info(f"检查点已加载: {path}")

        return checkpoint_data

    def stop(self) -> None:
        """停止训练并清理资源"""
        # 等待异步里程碑保存完成
        if self._milestone_thread is not None and self._milestone_thread.is_alive():
            self.logger.info("等待里程碑保存完成...")
            self._milestone_thread.join(timeout=30.0)
            if self._milestone_thread.is_alive():
                self.logger.warning("里程碑保存线程在 30s 超时后仍在运行")

        # 等待异步检查点保存完成
        if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
            self.logger.info("等待检查点保存完成...")
            self._checkpoint_thread.join(timeout=30.0)
            if self._checkpoint_thread.is_alive():
                self.logger.warning("检查点保存线程在 30s 超时后仍在运行")

        # 保存对手池索引
        if self.pool_manager:
            self.pool_manager.save_all()

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
                if stats.get("all_species_frozen", False):
                    self.logger.info("所有物种已冻结，训练完成")
                    round_count += 1
                    if checkpoint_callback:
                        checkpoint_callback(self.generation)
                    if progress_callback:
                        progress_callback(stats)
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
                    # 清理对手池中 get_entry() 积累的大数据
                    if self.pool_manager is not None:
                        for pool in self.pool_manager.pools.values():
                            pool.clear_memory_cache()

                    gc.collect()
                    if _libc is not None:
                        _libc.malloc_trim(c_int(0))
        finally:
            self._is_running = False

        self.logger.info(f"联盟训练完成，共 {round_count} 轮")


def extract_elite_networks(
    pre_evolution_fitness: dict[int, np.ndarray] | None,
    network_data: dict[int, tuple[np.ndarray, ...]],
    elite_ratio: float = 0.05,
    genome_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> tuple[int, tuple[np.ndarray, ...]]:
    """从每个子种群中提取 Top 精英的网络参数

    流程：
    1. 对每个 sub_pop: 按 pre_evolution_fitness 排序取 Top elite_ratio indices
    2. 从 network_data 中用 _unpack_network_params_numpy 提取对应 indices 的网络参数字典
    3. 收集所有精英的 params_list
    4. 用 _pack_network_params_numpy 重新打包
    5. 清理中间变量

    如果 pre_evolution_fitness 为 None 或空（旧版 entry），回退到从 genome_data 提取 fitness。
    genome_data 的第二个元素是 fitnesses 数组，过滤掉 NaN（进化后新 offspring 无 fitness）。

    Args:
        pre_evolution_fitness: {sub_pop_id: fitness_array} - 预进化适应度，可为 None
        network_data: {sub_pop_id: packed_network_params_tuple} - 网络参数
        elite_ratio: 精英比例（默认 0.05 即 Top 5%）
        genome_data: {sub_pop_id: (keys, fitnesses, metadata, nodes, conns)} - 基因组数据，
                     用于 pre_evolution_fitness 不可用时的回退

    Returns:
        (精英总数, packed_network_params_tuple)
        如果没有精英，返回 (0, ())
    """
    elite_params_list: list[dict[str, np.ndarray | int]] = []

    # 确定要处理的 sub_pop_id 集合
    sub_pop_ids: set[int] = set(network_data.keys())

    for sub_pop_id in sorted(sub_pop_ids):
        if sub_pop_id not in network_data:
            continue

        # 获取适应度数组
        fitness_array: np.ndarray | None = None

        if pre_evolution_fitness and sub_pop_id in pre_evolution_fitness:
            fitness_array = pre_evolution_fitness[sub_pop_id]
        elif genome_data and sub_pop_id in genome_data:
            # 回退：从 genome_data 提取 fitness（第二个元素是 fitnesses 数组）
            raw_fitnesses: np.ndarray = genome_data[sub_pop_id][1]
            # 过滤掉 NaN（进化后新 offspring 无 fitness）
            valid_mask: np.ndarray = ~np.isnan(raw_fitnesses.astype(np.float64))
            if not np.any(valid_mask):
                continue
            fitness_array = raw_fitnesses
        else:
            continue

        if fitness_array is None or len(fitness_array) == 0:
            continue

        # 解包该 sub_pop 的 network_data
        packed_tuple: tuple[np.ndarray, ...] = network_data[sub_pop_id]
        all_params: list[dict[str, np.ndarray | int]] = _unpack_network_params_numpy(
            *packed_tuple
        )

        if len(all_params) == 0:
            del all_params
            continue

        # 计算精英数量
        if len(fitness_array) != len(all_params):
            logging.getLogger("league").warning(
                f"extract_elite_networks: sub_pop {sub_pop_id} "
                f"fitness({len(fitness_array)}) != params({len(all_params)})"
            )
        effective_len: int = min(len(fitness_array), len(all_params))
        n_elite: int = max(1, int(effective_len * elite_ratio))

        if pre_evolution_fitness and sub_pop_id in pre_evolution_fitness:
            # 直接按适应度排序取 Top
            elite_indices: np.ndarray = np.argsort(fitness_array)[::-1][:n_elite]
            elite_params: list[dict[str, np.ndarray | int]] = [
                all_params[i] for i in elite_indices if i < len(all_params)
            ]
        else:
            # 回退模式：仅取有效 fitness 的 agents
            valid_mask = ~np.isnan(fitness_array.astype(np.float64))
            valid_indices: np.ndarray = np.where(valid_mask)[0]
            valid_fitnesses: np.ndarray = fitness_array[valid_indices].astype(np.float64)
            # 基于有效样本数重新计算精英数量（总数含 NaN 会导致比例过高）
            n_elite_fallback: int = max(1, int(len(valid_indices) * elite_ratio))
            # 按适应度排序取 Top
            sorted_order: np.ndarray = np.argsort(valid_fitnesses)[::-1][:n_elite_fallback]
            actual_indices: np.ndarray = valid_indices[sorted_order]
            elite_params = [
                all_params[i] for i in actual_indices if i < len(all_params)
            ]

        elite_params_list.extend(elite_params)

        # 清理中间变量
        del all_params
        del elite_params

    if not elite_params_list:
        return (0, ())

    # 打包精英网络参数
    packed: tuple[np.ndarray, ...] = _pack_network_params_numpy(elite_params_list)
    total_elites: int = len(elite_params_list)

    # 清理
    del elite_params_list

    return (total_elites, packed)


def _reconstruct_network_data(
    genome_data: dict[int, tuple],
    agent_type: AgentType,
    config: Config,
) -> dict[int, tuple[np.ndarray, ...]] | None:
    """从基因组数据重建网络参数

    当历史 entry 没有 networks.npz 时，从 genomes.npz 重建网络参数。

    Args:
        genome_data: {sub_pop_id: (keys, fitnesses, metadata, nodes, conns)}
        agent_type: Agent 类型（用于确定 NEAT 配置）
        config: 全局配置

    Returns:
        {sub_pop_id: packed_network_params_tuple} 或 None（重建失败时）
    """
    import neat

    # 确定 NEAT 配置文件路径
    config_dir: str = config.training.neat_config_path
    if agent_type == AgentType.RETAIL_PRO:
        neat_config_path = f"{config_dir}/neat_retail_pro.cfg"
    elif agent_type == AgentType.MARKET_MAKER:
        neat_config_path = f"{config_dir}/neat_market_maker.cfg"
    else:
        return None

    try:
        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_config_path,
        )
    except Exception:
        logging.getLogger("league").warning(
            f"无法加载 NEAT 配置: {neat_config_path}"
        )
        return None

    network_data: dict[int, tuple[np.ndarray, ...]] = {}
    for sub_pop_id, gdata in genome_data.items():
        try:
            keys, fitnesses, metadata, all_nodes, all_conns = gdata
            population = _deserialize_genomes_numpy(
                keys, fitnesses, metadata, all_nodes, all_conns,
                neat_config.genome_config,
            )
            packed = _extract_and_pack_all_network_params(
                population, neat_config,
            )
            network_data[sub_pop_id] = packed
            del population
        except Exception as e:
            logging.getLogger("league").warning(
                f"重建 sub_pop {sub_pop_id} 网络参数失败: {e}"
            )
            continue

    return network_data if network_data else None
