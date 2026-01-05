"""竞技场管理器模块

负责创建、管理和协调多个竞技场的训练。
支持两种模式：
1. 同步模式（旧版兼容）：ArenaProcess + arena_worker
2. 异步监控模式（推荐）：ArenaProcessInfo + arena_worker_autonomous
"""

import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, Any

from src.core.log_engine.logger import get_logger

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
    # 获取完整种群数据
    populations = arena.get_checkpoint_data()["trainer"]["populations"]

    # 获取最佳个体
    best_genomes = arena.get_best_genomes(top_n=10)

    # 更新到共享检查点
    checkpoint_manager.update_arena(
        arena_id=arena_id,
        episode=episode,
        populations=populations,
        best_genomes=best_genomes,
    )


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
    import queue as queue_module

    from src.training.arena.arena import Arena
    from src.training.arena.shared_checkpoint import SharedCheckpointManager

    arena = Arena(arena_config)
    arena.setup()

    checkpoint_manager = SharedCheckpointManager(
        checkpoint_path=arena_config.checkpoint_dir
    )

    arena_id = arena_config.arena_id
    migration_interval = arena_config.migration_interval
    checkpoint_interval = arena_config.checkpoint_interval
    max_episodes = arena_config.max_episodes

    # 通知主进程初始化完成
    status_queue.put(("setup_done", arena_id, None))

    episode = 0
    while episode < max_episodes:
        # 非阻塞检查停止命令
        try:
            cmd = control_queue.get_nowait()
            if cmd == "stop":
                break
        except queue_module.Empty:
            pass

        # 运行一个 episode
        metrics = arena.run_episode()
        episode += 1

        # 检查迁移间隔
        if migration_interval > 0 and episode % migration_interval == 0:
            _execute_migration_from_checkpoint(arena, checkpoint_manager, arena_id)

        # 检查保存间隔
        if checkpoint_interval > 0 and episode % checkpoint_interval == 0:
            _save_arena_to_checkpoint(arena, checkpoint_manager, arena_id, episode)

        # 发送状态更新
        status_queue.put(("episode_done", arena_id, metrics))

    # 发送完成通知
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
    """

    config: "MultiArenaConfig"
    arenas: list[ArenaProcessInfo]
    checkpoint_manager: "SharedCheckpointManager"
    metrics_aggregator: "MetricsAggregator"
    _is_running: bool
    _logger: Any

    def __init__(self, config: "MultiArenaConfig") -> None:
        """初始化竞技场管理器

        Args:
            config: 多竞技场配置
        """
        from src.training.arena.metrics import MetricsAggregator
        from src.training.arena.shared_checkpoint import SharedCheckpointManager

        self.config = config
        self.arenas = []
        self.checkpoint_manager = SharedCheckpointManager(
            checkpoint_path=config.checkpoint_dir
        )
        self.metrics_aggregator = MetricsAggregator()
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

        for i in range(self.config.num_arenas):
            arena_config = ArenaConfig(
                arena_id=i,
                config=self.config.base_config,
                seed=i + self.config.seed_offset,
                migration_interval=self.config.migration_interval,
                checkpoint_interval=self.config.checkpoint_interval,
                max_episodes=self.config.max_episodes,
                checkpoint_dir=self.config.checkpoint_dir,
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
