"""竞技场管理器模块

负责创建、管理和协调多个竞技场的训练。
支持两种模式：
1. 同步模式（旧版兼容）：ArenaProcess + arena_worker
2. 异步监控模式（推荐）：ArenaProcessInfo + arena_worker_autonomous
"""

import gc
import os
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, Any

from src.core.log_engine.logger import get_logger
from src.training.population import malloc_trim


def _get_memory_mb() -> float:
    """获取当前进程的内存使用量（MB）

    使用 /proc/self/status 读取 VmRSS（常驻内存）
    """
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # 格式: VmRSS:    12345 kB
                    parts = line.split()
                    return float(parts[1]) / 1024.0  # kB -> MB
    except Exception:
        pass
    return 0.0


def _log_memory(logger: Any, tag: str, arena_id: int) -> float:
    """记录内存使用并返回当前内存值

    Args:
        logger: 日志器
        tag: 标签（如 "episode_start", "evolve_before" 等）
        arena_id: 竞技场 ID

    Returns:
        当前内存使用量（MB）
    """
    mem = _get_memory_mb()
    logger.info(f"[MEMORY] Arena-{arena_id} {tag}: {mem:.1f} MB")
    return mem

if TYPE_CHECKING:
    from src.bio.agents.base import AgentType
    from src.training.arena.arena import Arena
    from src.training.arena.config import ArenaConfig, MultiArenaConfig
    from src.training.arena.metrics import EpisodeMetrics, MetricsAggregator
    from src.training.arena.migration import MigrationPacket, MigrationSystem
    from src.training.arena.shared_checkpoint import SharedCheckpointManager


# =============================================================================
# 旧版同步模式（已弃用，保留用于兼容）
# =============================================================================


@dataclass
class ArenaProcess:
    """竞技场进程信息（已弃用，保留用于兼容）

    Deprecated: 请使用 ArenaProcessInfo 和 arena_worker_autonomous 替代。

    Attributes:
        arena_id: 竞技场 ID
        process: 进程对象
        cmd_queue: 命令队列（主进程 -> 子进程）
        result_queue: 结果队列（子进程 -> 主进程）
    """
    arena_id: int
    process: Process
    cmd_queue: Queue
    result_queue: Queue


def arena_worker(
    arena_config: "ArenaConfig",
    cmd_queue: Queue,
    result_queue: Queue,
) -> None:
    """竞技场工作进程入口（已弃用，保留用于兼容）

    Deprecated: 请使用 arena_worker_autonomous 替代。

    Args:
        arena_config: 竞技场配置
        cmd_queue: 命令队列
        result_queue: 结果队列
    """
    from src.training.arena.arena import Arena

    arena = Arena(arena_config)

    while True:
        try:
            cmd, args = cmd_queue.get()
        except Exception:
            break

        try:
            if cmd == "setup":
                arena.setup()
                result_queue.put(("setup_done", None))

            elif cmd == "run_episode":
                metrics = arena.run_episode()
                result_queue.put(("episode_done", metrics))

            elif cmd == "get_migration_candidates":
                best_count = args.get("best_count", 5)
                worst_count = args.get("worst_count", 5)

                candidates = []
                if best_count > 0:
                    candidates.extend(
                        arena.get_migration_candidates(best_count, select_best=True)
                    )
                if worst_count > 0:
                    candidates.extend(
                        arena.get_migration_candidates(worst_count, select_best=False)
                    )

                result_queue.put(("candidates", candidates))

            elif cmd == "inject_genomes":
                arena.inject_genomes(args)
                result_queue.put(("injected", None))

            elif cmd == "get_checkpoint":
                checkpoint = arena.get_checkpoint_data()
                result_queue.put(("checkpoint", checkpoint))

            elif cmd == "load_checkpoint":
                arena.load_checkpoint_data(args)
                result_queue.put(("checkpoint_loaded", None))

            elif cmd == "stop":
                arena.stop()
                break

            else:
                result_queue.put(("error", f"Unknown command: {cmd}"))

        except Exception as e:
            result_queue.put(("error", str(e)))


# =============================================================================
# 新版异步监控模式
# =============================================================================


@dataclass
class ArenaProcessInfo:
    """竞技场进程信息（异步监控模式）

    Attributes:
        arena_id: 竞技场 ID
        process: 进程对象
        status_queue: 状态报告队列（子进程 -> 主进程）
        control_queue: 控制命令队列（主进程 -> 子进程）
        episode: 当前 episode 编号
        is_finished: 是否已完成
    """
    arena_id: int
    process: Process
    status_queue: Queue  # 子进程 -> 主进程（状态报告）
    control_queue: Queue  # 主进程 -> 子进程（控制命令）
    episode: int = 0
    is_finished: bool = False


def _execute_migration_from_checkpoint(
    arena: "Arena",
    checkpoint_manager: "SharedCheckpointManager",
    arena_id: int,
) -> None:
    """从共享检查点执行迁移

    Args:
        arena: 竞技场实例
        checkpoint_manager: 共享检查点管理器
        arena_id: 竞技场 ID
    """
    import gc

    from src.bio.agents.base import AgentType
    from src.training.arena.migration import MigrationPacket

    # 从 checkpoint 获取其他竞技场的最佳个体
    candidates = checkpoint_manager.get_migration_candidates(
        requesting_arena_id=arena_id,
        count_per_arena=5,
    )

    if not candidates:
        return  # 没有可迁移的候选者

    # 构建 MigrationPacket 列表并注入
    packets: list[MigrationPacket] = []
    for agent_type_str, genomes in candidates.items():
        agent_type = AgentType(agent_type_str)
        for genome_data, fitness in genomes:
            packet = MigrationPacket(
                source_arena=-1,
                agent_type=agent_type,
                genome_data=genome_data,
                fitness=fitness,
                generation=0,
            )
            packets.append(packet)

    arena.inject_genomes(packets)

    # 迁移后强制多轮 GC，确保 NEAT 对象的循环引用被完全清理
    del packets
    del candidates
    gc.collect()
    gc.collect()
    gc.collect()
    malloc_trim()  # 将释放的内存归还给操作系统


def _save_arena_to_checkpoint(
    arena: "Arena",
    checkpoint_manager: "SharedCheckpointManager",
    arena_id: int,
    episode: int,
) -> None:
    """保存竞技场状态到共享检查点

    Args:
        arena: 竞技场实例
        checkpoint_manager: 共享检查点管理器
        arena_id: 竞技场 ID
        episode: 当前 episode 编号
    """
    # 先获取 best_genomes（相对轻量的操作）
    best_genomes = arena.get_best_genomes(top_n=10)

    # 然后获取完整种群数据
    checkpoint_data = arena.get_checkpoint_data()
    populations = checkpoint_data["trainer"]["populations"]

    # 分两步更新到共享检查点，减少同时存在的大对象
    # 先保存 best_genomes
    checkpoint_manager.update_arena(
        arena_id=arena_id,
        episode=episode,
        populations=populations,
        best_genomes=best_genomes,
    )

    # 立即释放 best_genomes
    del best_genomes
    gc.collect()

    # 释放 checkpoint_data 和 populations
    del checkpoint_data
    del populations
    gc.collect()
    gc.collect()
    malloc_trim()  # 将释放的内存归还给操作系统


def arena_worker_autonomous(
    arena_config: "ArenaConfig",
    status_queue: Queue,  # 状态报告
    control_queue: Queue,  # 控制命令
) -> None:
    """自治竞技场工作进程

    竞技场按自己的节奏独立运行，主动触发迁移和保存。

    Args:
        arena_config: 竞技场配置
        status_queue: 状态报告队列（子进程 -> 主进程）
        control_queue: 控制命令队列（主进程 -> 子进程）
    """
    import gc
    import os
    import pickle
    import queue as queue_module

    from src.training.arena.arena import Arena
    from src.training.arena.shared_checkpoint import SharedCheckpointManager

    arena_id = arena_config.arena_id
    logger = get_logger(f"arena_worker_{arena_id}")

    # 记录初始内存
    mem_initial = _log_memory(logger, "worker_start", arena_id)

    arena = Arena(arena_config)

    # 如果需要恢复，先读取检查点再 setup（避免先创建 Agent 再清理的开销）
    checkpoint_data: dict | None = None
    loaded_episode = 0

    if arena_config.should_resume:
        checkpoint_path = os.path.join(
            arena_config.checkpoint_dir,
            f"arena_{arena_id}",
            "checkpoint.pkl"
        )
        if os.path.exists(checkpoint_path):
            logger.info(f"Arena-{arena_id} 正在从文件读取检查点...")
            try:
                with open(checkpoint_path, "rb") as f:
                    arena_data = pickle.load(f)
                loaded_episode = arena_data.episode
                # 构建检查点数据格式
                checkpoint_data = {
                    "trainer": {
                        "populations": arena_data.populations,
                        "tick": 0,
                        "episode": arena_data.episode,
                    },
                }
                logger.info(f"Arena-{arena_id} 检查点读取完成，episode={loaded_episode}")
            except (pickle.PickleError, EOFError, OSError, AttributeError) as e:
                logger.warning(f"Arena-{arena_id} 检查点读取失败: {e}，将从头开始训练")
                checkpoint_data = None

    # 兼容旧接口：如果传入了 initial_checkpoint（已弃用）
    if checkpoint_data is None and arena_config.initial_checkpoint is not None:
        checkpoint_data = arena_config.initial_checkpoint

    # 初始化竞技场（如果有检查点，直接从检查点创建 Agent）
    arena.setup(checkpoint=checkpoint_data)

    # 清理临时对象
    if 'arena_data' in locals():
        del arena_data
    gc.collect()
    gc.collect()
    malloc_trim()

    checkpoint_manager = SharedCheckpointManager(
        checkpoint_path=arena_config.checkpoint_dir
    )

    migration_interval = arena_config.migration_interval
    checkpoint_interval = arena_config.checkpoint_interval
    max_episodes = arena_config.max_episodes

    # 记录 setup 后内存
    mem_after_setup = _log_memory(logger, "after_setup", arena_id)
    logger.info(
        f"[MEMORY] Arena-{arena_id} setup_delta: "
        f"+{mem_after_setup - mem_initial:.1f} MB"
    )

    # 通知主进程初始化完成
    status_queue.put(("setup_done", arena_id, None))

    episode = 0
    mem_prev_episode = mem_after_setup

    while episode < max_episodes:
        # 非阻塞检查停止命令
        try:
            cmd = control_queue.get_nowait()
            if cmd == "stop":
                break
        except queue_module.Empty:
            pass

        # 记录 episode 开始前内存
        mem_ep_start = _log_memory(logger, f"ep_{episode+1}_start", arena_id)

        # 运行一个 episode（包含进化）
        metrics = arena.run_episode()
        episode += 1

        # 记录 episode 结束（进化后）内存
        mem_ep_end = _log_memory(logger, f"ep_{episode}_end", arena_id)
        logger.info(
            f"[MEMORY] Arena-{arena_id} ep_{episode}_delta: "
            f"+{mem_ep_end - mem_ep_start:.1f} MB, "
            f"cumulative: +{mem_ep_end - mem_after_setup:.1f} MB"
        )

        # 检查迁移间隔
        if migration_interval > 0 and episode % migration_interval == 0:
            mem_before_migrate = _get_memory_mb()
            _execute_migration_from_checkpoint(arena, checkpoint_manager, arena_id)
            mem_after_migrate = _get_memory_mb()
            logger.info(
                f"[MEMORY] Arena-{arena_id} migration_delta: "
                f"+{mem_after_migrate - mem_before_migrate:.1f} MB"
            )
            # 迁移后立即强制多轮 GC，确保临时对象被完全清理
            gc.collect()
            gc.collect()
            gc.collect()
            malloc_trim()

        # 检查保存间隔
        if checkpoint_interval > 0 and episode % checkpoint_interval == 0:
            mem_before_save = _get_memory_mb()
            _save_arena_to_checkpoint(arena, checkpoint_manager, arena_id, episode)
            mem_after_save = _get_memory_mb()
            logger.info(
                f"[MEMORY] Arena-{arena_id} checkpoint_save_delta: "
                f"+{mem_after_save - mem_before_save:.1f} MB"
            )
            # 保存后立即强制多轮 GC，确保临时对象被完全清理
            gc.collect()
            gc.collect()
            gc.collect()
            malloc_trim()

        # 每个 episode 后都强制垃圾回收，防止进化阶段内存泄漏
        mem_before_gc = _get_memory_mb()
        gc.collect()
        gc.collect()
        malloc_trim()  # 将释放的内存归还给操作系统
        mem_after_gc = _get_memory_mb()

        # 检查 GC 是否有效
        gc_released = mem_before_gc - mem_after_gc
        if gc_released > 1.0:  # 超过 1MB 才记录
            logger.info(
                f"[MEMORY] Arena-{arena_id} ep_{episode}_gc_released: "
                f"-{gc_released:.1f} MB"
            )

        # 每 10 个 episode 输出内存增长统计
        if episode % 10 == 0:
            mem_current = _get_memory_mb()
            growth_per_ep = (mem_current - mem_after_setup) / episode
            logger.warning(
                f"[MEMORY_SUMMARY] Arena-{arena_id} ep_{episode}: "
                f"current={mem_current:.1f} MB, "
                f"since_setup=+{mem_current - mem_after_setup:.1f} MB, "
                f"avg_growth={growth_per_ep:.2f} MB/ep"
            )

        mem_prev_episode = mem_after_gc

        # 发送状态更新
        status_queue.put(("episode_done", arena_id, metrics))

    # 发送完成通知
    mem_final = _log_memory(logger, "worker_end", arena_id)
    logger.info(
        f"[MEMORY] Arena-{arena_id} total_growth: "
        f"+{mem_final - mem_initial:.1f} MB over {episode} episodes"
    )
    status_queue.put(("finished", arena_id, episode))
    arena.stop()


# =============================================================================
# ArenaManager 类（监控者模式）
# =============================================================================


class ArenaManager:
    """竞技场管理器（监控者模式）

    负责创建、管理和协调多个竞技场的训练。
    使用多进程实现真正的并行化。

    Attributes:
        config: 多竞技场配置
        arenas: 竞技场进程列表
        checkpoint_manager: 共享检查点管理器
        metrics_aggregator: 指标聚合器
        auto_resume: 是否自动从最新检查点恢复
    """

    config: "MultiArenaConfig"
    arenas: list[ArenaProcessInfo]
    checkpoint_manager: "SharedCheckpointManager"
    metrics_aggregator: "MetricsAggregator"
    auto_resume: bool
    _is_running: bool
    _logger: Any

    def __init__(self, config: "MultiArenaConfig", auto_resume: bool = True) -> None:
        """初始化竞技场管理器

        Args:
            config: 多竞技场配置
            auto_resume: 是否自动从最新检查点恢复（默认: True）
        """
        from src.training.arena.metrics import MetricsAggregator
        from src.training.arena.shared_checkpoint import SharedCheckpointManager

        self.config = config
        self.arenas = []
        self.checkpoint_manager = SharedCheckpointManager(
            checkpoint_path=config.checkpoint_dir
        )
        self.metrics_aggregator = MetricsAggregator()
        self.auto_resume = auto_resume
        self._is_running = False
        self._logger = get_logger("arena_manager")

    def setup(self) -> None:
        """初始化所有竞技场进程"""
        from src.training.arena.config import ArenaConfig

        self._logger.info(f"正在创建 {self.config.num_arenas} 个竞技场...")

        # 初始化共享检查点
        self.checkpoint_manager.initialize(
            config={"num_arenas": self.config.num_arenas},
            num_arenas=self.config.num_arenas,
        )

        # 检测是否有现有检查点（不加载数据，只检测存在性）
        # 子进程会自己从文件读取检查点，避免在主进程中占用大量内存
        has_checkpoint = False
        if self.auto_resume:
            import os
            for arena_id in range(self.config.num_arenas):
                checkpoint_path = self.checkpoint_manager._get_arena_checkpoint_path(
                    arena_id
                )
                if os.path.exists(checkpoint_path):
                    has_checkpoint = True
                    break

            if has_checkpoint:
                self._logger.info("检测到现有检查点，子进程将自动恢复")
            else:
                self._logger.info("未检测到现有检查点，从头开始训练")

        for i in range(self.config.num_arenas):
            arena_config = ArenaConfig(
                arena_id=i,
                config=self.config.base_config,
                seed=i + self.config.seed_offset,
                migration_interval=self.config.migration_interval,
                checkpoint_interval=self.config.checkpoint_interval,
                max_episodes=self.config.max_episodes,
                checkpoint_dir=self.config.checkpoint_dir,
                # 不传递检查点数据，让子进程自己从文件读取
                initial_checkpoint=None,
                # 新增：告诉子进程是否应该尝试从文件恢复
                should_resume=self.auto_resume and has_checkpoint,
            )

            status_queue: Queue = Queue()
            control_queue: Queue = Queue()

            process = Process(
                target=arena_worker_autonomous,
                args=(arena_config, status_queue, control_queue),
                name=f"Arena-{i}",
            )

            self.arenas.append(ArenaProcessInfo(
                arena_id=i,
                process=process,
                status_queue=status_queue,
                control_queue=control_queue,
            ))

        self._logger.info("竞技场进程已创建")

    def start(self) -> None:
        """启动所有竞技场进程"""
        self._logger.info("正在启动竞技场进程...")

        # 启动所有进程
        for arena in self.arenas:
            arena.process.start()

        # 等待所有竞技场完成初始化
        for arena in self.arenas:
            status, arena_id, _ = arena.status_queue.get()
            if status != "setup_done":
                raise RuntimeError(f"Arena {arena_id} 初始化失败")

        self._is_running = True
        self._logger.info("所有竞技场已启动")

    def monitor(
        self,
        progress_callback: Callable[[dict], None] | None = None,
        check_interval: float = 1.0,
    ) -> None:
        """监控所有竞技场的运行状态（非阻塞）

        Args:
            progress_callback: 进度回调函数
            check_interval: 检查间隔（秒）
        """
        while self._is_running:
            # 检查是否所有竞技场都已完成
            if all(arena.is_finished for arena in self.arenas):
                break

            # 收集状态更新
            for arena in self.arenas:
                if arena.is_finished:
                    continue

                while True:
                    try:
                        status, arena_id, data = arena.status_queue.get_nowait()

                        if status == "episode_done":
                            arena.episode += 1
                            self.metrics_aggregator.update(data)
                        elif status == "finished":
                            arena.is_finished = True
                            self._logger.info(f"Arena {arena_id} 已完成 {data} 个 episode")
                    except queue.Empty:
                        break

            # 进度回调
            if progress_callback:
                summary = self.metrics_aggregator.get_global_summary()
                summary["running_arenas"] = sum(1 for a in self.arenas if not a.is_finished)
                summary["arena_episodes"] = {a.arena_id: a.episode for a in self.arenas}
                progress_callback(summary)

            time.sleep(check_interval)

    def stop(self) -> None:
        """停止所有竞技场"""
        self._is_running = False
        self._logger.info("正在停止竞技场...")

        for arena in self.arenas:
            arena.control_queue.put("stop")
            arena.process.join(timeout=10)
            if arena.process.is_alive():
                self._logger.warning(f"Arena {arena.arena_id} 未响应，强制终止")
                arena.process.terminate()

        self._logger.info("所有竞技场已停止")

    def train(
        self,
        episodes: int,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        """训练（兼容旧接口，内部调用 monitor）

        Args:
            episodes: 训练的 episode 数量（忽略，使用 config.max_episodes）
            progress_callback: 进度回调函数
        """
        self.monitor(progress_callback=progress_callback)

    def get_summary(self) -> dict:
        """获取当前训练状态汇总

        Returns:
            状态汇总
        """
        summary = self.metrics_aggregator.get_global_summary()
        summary["arena_episodes"] = {
            arena.arena_id: arena.episode for arena in self.arenas
        }
        return summary
