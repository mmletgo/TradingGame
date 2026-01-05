"""竞技场管理器模块

负责创建、管理和协调多个竞技场的训练。
"""

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Process, Queue
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.log_engine.logger import get_logger

if TYPE_CHECKING:
    from src.training.arena.config import ArenaConfig, MultiArenaConfig
    from src.training.arena.metrics import EpisodeMetrics, MetricsAggregator
    from src.training.arena.migration import MigrationPacket, MigrationSystem


@dataclass
class ArenaProcess:
    """竞技场进程信息

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
    """竞技场工作进程入口

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


class ArenaManager:
    """竞技场管理器

    负责创建、管理和协调多个竞技场的训练。
    使用多进程实现真正的并行化。

    Attributes:
        config: 多竞技场配置
        arenas: 竞技场进程列表
        migration_system: 迁移系统
        metrics_aggregator: 指标聚合器
    """

    config: "MultiArenaConfig"
    arenas: list[ArenaProcess]
    migration_system: "MigrationSystem"
    metrics_aggregator: "MetricsAggregator"
    _is_running: bool
    _logger: Any

    def __init__(self, config: "MultiArenaConfig") -> None:
        """初始化竞技场管理器

        Args:
            config: 多竞技场配置
        """
        from src.training.arena.metrics import MetricsAggregator
        from src.training.arena.migration import MigrationSystem

        self.config = config
        self.arenas = []
        self.migration_system = MigrationSystem(
            num_arenas=config.num_arenas,
            strategy=config.migration_strategy,
        )
        self.metrics_aggregator = MetricsAggregator()
        self._is_running = False
        self._logger = get_logger("arena_manager")

    def setup(self) -> None:
        """初始化所有竞技场（每个竞技场一个进程）"""
        from src.training.arena.config import ArenaConfig

        self._logger.info(f"正在创建 {self.config.num_arenas} 个竞技场...")

        for i in range(self.config.num_arenas):
            arena_config = ArenaConfig(
                arena_id=i,
                config=self.config.base_config,
                seed=i + self.config.seed_offset,
            )

            # 创建进程间通信队列
            cmd_queue: Queue = Queue()
            result_queue: Queue = Queue()

            # 创建进程
            process = Process(
                target=arena_worker,
                args=(arena_config, cmd_queue, result_queue),
                name=f"Arena-{i}",
            )

            self.arenas.append(ArenaProcess(
                arena_id=i,
                process=process,
                cmd_queue=cmd_queue,
                result_queue=result_queue,
            ))

        self._logger.info("竞技场进程已创建")

    def start(self) -> None:
        """启动所有竞技场进程"""
        self._logger.info("正在启动竞技场进程...")

        # 启动所有进程
        for arena in self.arenas:
            arena.process.start()
            arena.cmd_queue.put(("setup", None))

        # 等待所有竞技场完成初始化
        for arena in self.arenas:
            result = arena.result_queue.get()
            if result[0] == "error":
                raise RuntimeError(f"Arena {arena.arena_id} setup failed: {result[1]}")

        self._is_running = True
        self._logger.info("所有竞技场已启动")

    def train(
        self,
        episodes: int,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        """训练主循环

        Args:
            episodes: 训练的 episode 数量
            progress_callback: 进度回调函数
        """
        self._logger.info(f"开始训练 {episodes} 个 episodes...")

        for ep in range(episodes):
            if not self._is_running:
                break

            # 1. 所有竞技场运行一个 episode
            for arena in self.arenas:
                arena.cmd_queue.put(("run_episode", None))

            # 2. 收集结果
            results: list["EpisodeMetrics"] = []
            for arena in self.arenas:
                result = arena.result_queue.get()
                if result[0] == "error":
                    self._logger.error(f"Arena {arena.arena_id} error: {result[1]}")
                    continue
                results.append(result[1])

            # 3. 聚合指标
            self.metrics_aggregator.update_batch(results)

            # 4. 迁移（每 N 个 episode）
            if (ep + 1) % self.config.migration_interval == 0:
                self._execute_migration()

            # 5. 检查点
            if (
                self.config.checkpoint_interval > 0
                and (ep + 1) % self.config.checkpoint_interval == 0
            ):
                self.save_checkpoint(f"checkpoints/multi_arena_ep_{ep + 1}.pkl")

            # 6. 进度回调
            if progress_callback:
                summary = self.metrics_aggregator.get_global_summary()
                summary["episode"] = ep + 1
                summary["total_episodes"] = episodes
                progress_callback(summary)

            # 日志
            if (ep + 1) % 10 == 0:
                summary = self.metrics_aggregator.get_global_summary()
                self._logger.info(
                    f"Episode {ep + 1}/{episodes} 完成, "
                    f"平均波动率: {summary.get('avg_volatility', 0):.4f}"
                )

        self._logger.info("训练完成")

    def _execute_migration(self) -> None:
        """执行一次迁移"""
        self._logger.info("开始迁移...")

        # 计算迁移数量
        migration_count = self.config.migration_count
        best_count = int(migration_count * self.config.migration_best_ratio)
        worst_count = migration_count - best_count

        # 1. 从所有竞技场收集候选者
        for arena in self.arenas:
            arena.cmd_queue.put(("get_migration_candidates", {
                "best_count": best_count,
                "worst_count": worst_count,
            }))

        # 收集候选者
        all_candidates: list["MigrationPacket"] = []
        for arena in self.arenas:
            result = arena.result_queue.get()
            if result[0] == "error":
                self._logger.error(f"获取候选者失败: {result[1]}")
                continue
            all_candidates.extend(result[1])

        self._logger.info(f"收集到 {len(all_candidates)} 个迁移候选者")

        # 2. 规划迁移方向
        migrations = self.migration_system.plan_migrations(all_candidates)

        # 3. 分发迁移
        for arena in self.arenas:
            packets = migrations.get(arena.arena_id, [])
            if packets:
                arena.cmd_queue.put(("inject_genomes", packets))
            else:
                arena.cmd_queue.put(("inject_genomes", []))

        # 等待完成
        for arena in self.arenas:
            result = arena.result_queue.get()
            if result[0] == "error":
                self._logger.error(f"注入失败: {result[1]}")

        self._logger.info("迁移完成")

    def stop(self) -> None:
        """停止所有竞技场"""
        self._is_running = False
        self._logger.info("正在停止竞技场...")

        for arena in self.arenas:
            arena.cmd_queue.put(("stop", None))
            arena.process.join(timeout=10)
            if arena.process.is_alive():
                self._logger.warning(f"Arena {arena.arena_id} 未响应，强制终止")
                arena.process.terminate()

        self._logger.info("所有竞技场已停止")

    def save_checkpoint(self, path: str) -> None:
        """保存所有竞技场的检查点

        Args:
            path: 检查点文件路径
        """
        self._logger.info(f"保存检查点到 {path}...")

        # 收集所有竞技场的检查点
        for arena in self.arenas:
            arena.cmd_queue.put(("get_checkpoint", None))

        checkpoints = {}
        for arena in self.arenas:
            result = arena.result_queue.get()
            if result[0] == "error":
                self._logger.error(f"获取检查点失败: {result[1]}")
                continue
            checkpoints[arena.arena_id] = result[1]

        # 保存
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "config": self.config,
                "arenas": checkpoints,
                "metrics": self.metrics_aggregator.get_history(),
            }, f)

        self._logger.info("检查点已保存")

    def load_checkpoint(self, path: str) -> None:
        """加载检查点

        Args:
            path: 检查点文件路径
        """
        self._logger.info(f"加载检查点 {path}...")

        with open(path, "rb") as f:
            data = pickle.load(f)

        # 加载到各竞技场
        for arena_id, checkpoint in data["arenas"].items():
            arena = self.arenas[arena_id]
            arena.cmd_queue.put(("load_checkpoint", checkpoint))

        # 等待完成
        for arena in self.arenas:
            result = arena.result_queue.get()
            if result[0] == "error":
                self._logger.error(f"加载失败: {result[1]}")

        self._logger.info("检查点已加载")

    def get_summary(self) -> dict:
        """获取当前训练状态汇总

        Returns:
            状态汇总
        """
        return self.metrics_aggregator.get_global_summary()
