"""竞技场进程池模块

管理多个 ArenaWorker 进程，协调并行训练。
"""

import logging
from multiprocessing import Process, Queue
from typing import Any

import numpy as np

from src.bio.agents.base import AgentType
from src.config.config import Config
from src.core.log_engine.logger import get_logger

from .arena_worker import arena_worker_process
from .fitness_aggregator import FitnessAggregator


class ArenaPool:
    """竞技场进程池

    管理多个 ArenaWorker 进程，协调并行训练。

    Attributes:
        num_arenas: 竞技场数量
        config: 全局配置
        workers: 工作进程列表
        cmd_queues: 命令队列（每个竞技场一个）
        result_queue: 结果队列（共享）
        logger: 日志器
        _started: 是否已启动
    """

    num_arenas: int
    config: Config
    workers: list[Process]
    cmd_queues: list["Queue[Any]"]
    result_queue: "Queue[tuple[str, int, Any]]"
    logger: logging.Logger
    _started: bool

    def __init__(self, num_arenas: int, config: Config) -> None:
        """初始化进程池

        Args:
            num_arenas: 竞技场数量
            config: 全局配置
        """
        self.num_arenas = num_arenas
        self.config = config
        self.logger = get_logger("arena_pool")

        # 创建共享结果队列
        self.result_queue = Queue()

        # 为每个竞技场创建独立的命令队列
        self.cmd_queues = [Queue() for _ in range(num_arenas)]

        # 工作进程列表（启动时填充）
        self.workers = []
        self._started = False

    def start(self) -> None:
        """启动所有工作进程"""
        if self._started:
            self.logger.warning("ArenaPool 已启动，跳过重复启动")
            return

        self.logger.info(f"正在启动 {self.num_arenas} 个竞技场工作进程...")

        for arena_id in range(self.num_arenas):
            p = Process(
                target=arena_worker_process,
                args=(
                    arena_id,
                    self.config,
                    self.cmd_queues[arena_id],
                    self.result_queue,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
            self.logger.debug(f"Arena {arena_id} 工作进程已启动，PID: {p.pid}")

        self._started = True
        self.logger.info(f"所有 {self.num_arenas} 个竞技场工作进程已启动")

    def broadcast_genomes(
        self,
        genome_data: dict[AgentType, bytes],
        network_params: dict[AgentType, tuple[np.ndarray, ...]],
    ) -> None:
        """广播基因组到所有竞技场

        一次序列化，广播到所有队列，等待所有竞技场 setup 完成。

        Args:
            genome_data: 各物种的序列化基因组数据
            network_params: 各物种的网络参数

        Raises:
            RuntimeError: 当任意竞技场 setup 失败时
            TimeoutError: 当等待 setup 完成超时时
        """
        if not self._started:
            raise RuntimeError("ArenaPool 尚未启动，请先调用 start()")

        self.logger.debug(f"广播基因组到 {self.num_arenas} 个竞技场...")

        # 向所有竞技场发送 setup 命令
        for arena_id in range(self.num_arenas):
            self.cmd_queues[arena_id].put(("setup", genome_data, network_params))

        # 等待所有竞技场 setup 完成
        setup_done_count = 0
        errors: list[tuple[int, str]] = []

        while setup_done_count < self.num_arenas:
            try:
                # 超时 60 秒，防止无限等待
                result = self.result_queue.get(timeout=60.0)
                result_type, arena_id, data = result

                if result_type == "setup_done":
                    setup_done_count += 1
                    self.logger.debug(
                        f"Arena {arena_id} setup 完成 "
                        f"({setup_done_count}/{self.num_arenas})"
                    )
                elif result_type == "error":
                    errors.append((arena_id, str(data)))
                    setup_done_count += 1  # 计入已处理
                    self.logger.error(f"Arena {arena_id} setup 失败: {data}")
                else:
                    self.logger.warning(
                        f"收到未预期的结果类型: {result_type}, "
                        f"arena_id={arena_id}, data={data}"
                    )

            except Exception as e:
                raise TimeoutError(
                    f"等待竞技场 setup 完成超时 "
                    f"(已完成: {setup_done_count}/{self.num_arenas})"
                ) from e

        # 检查是否有错误
        if errors:
            error_details = ", ".join(
                [f"Arena {aid}: {msg}" for aid, msg in errors]
            )
            raise RuntimeError(f"部分竞技场 setup 失败: {error_details}")

        self.logger.info(f"所有 {self.num_arenas} 个竞技场 setup 完成")

    def run_all(
        self,
        episodes_per_arena: int,
    ) -> dict[tuple[AgentType, int], np.ndarray]:
        """并行运行所有竞技场，收集并汇总适应度

        1. 向所有竞技场发送运行命令
        2. 收集所有结果
        3. 使用 FitnessAggregator 汇总为平均适应度

        Args:
            episodes_per_arena: 每个竞技场运行的 episode 数

        Returns:
            平均适应度字典
                - key: (agent_type, sub_pop_id)
                - value: 平均适应度数组

        Raises:
            RuntimeError: 当任意竞技场运行失败时
            TimeoutError: 当等待运行完成超时时
        """
        if not self._started:
            raise RuntimeError("ArenaPool 尚未启动，请先调用 start()")

        self.logger.debug(
            f"向 {self.num_arenas} 个竞技场发送运行命令 "
            f"(每个竞技场 {episodes_per_arena} episodes)..."
        )

        # 1. 向所有竞技场发送运行命令（非阻塞）
        for arena_id in range(self.num_arenas):
            self.cmd_queues[arena_id].put(("run", episodes_per_arena))

        # 2. 收集所有结果
        arena_fitnesses: list[dict[tuple[AgentType, int], np.ndarray]] = []
        episode_counts: list[int] = []
        run_done_count = 0
        errors: list[tuple[int, str]] = []

        # 计算超时时间：每个 episode 最多 episode_length 个 tick
        # 假设每个 tick 最多 1 秒，加上 60 秒额外缓冲
        timeout_seconds = (
            episodes_per_arena * self.config.training.episode_length * 1.0 + 60.0
        )
        # 限制最大超时为 10 分钟
        timeout_seconds = min(timeout_seconds, 600.0)

        while run_done_count < self.num_arenas:
            try:
                result = self.result_queue.get(timeout=timeout_seconds)
                result_type, arena_id, data = result

                if result_type == "run_done":
                    fitness_data, ep_count = data
                    arena_fitnesses.append(fitness_data)
                    episode_counts.append(ep_count)
                    run_done_count += 1
                    self.logger.debug(
                        f"Arena {arena_id} 运行完成 "
                        f"({run_done_count}/{self.num_arenas}), "
                        f"episodes={ep_count}"
                    )
                elif result_type == "error":
                    errors.append((arena_id, str(data)))
                    run_done_count += 1  # 计入已处理
                    self.logger.error(f"Arena {arena_id} 运行失败: {data}")
                else:
                    self.logger.warning(
                        f"收到未预期的结果类型: {result_type}, "
                        f"arena_id={arena_id}, data={data}"
                    )

            except Exception as e:
                raise TimeoutError(
                    f"等待竞技场运行完成超时 "
                    f"(已完成: {run_done_count}/{self.num_arenas})"
                ) from e

        # 检查是否有错误
        if errors:
            error_details = ", ".join(
                [f"Arena {aid}: {msg}" for aid, msg in errors]
            )
            raise RuntimeError(f"部分竞技场运行失败: {error_details}")

        # 3. 使用 FitnessAggregator 汇总为平均适应度
        total_episodes = sum(episode_counts)
        self.logger.info(
            f"所有 {self.num_arenas} 个竞技场运行完成，"
            f"总 episodes: {total_episodes}"
        )

        avg_fitness = FitnessAggregator.aggregate_simple_average(
            arena_fitnesses, episode_counts
        )

        return avg_fitness

    def shutdown(self) -> None:
        """关闭所有工作进程"""
        if not self._started:
            return

        self.logger.info("正在关闭所有竞技场工作进程...")

        # 向所有竞技场发送 shutdown 命令
        for arena_id in range(self.num_arenas):
            try:
                self.cmd_queues[arena_id].put(("shutdown",))
            except Exception as e:
                self.logger.warning(f"向 Arena {arena_id} 发送 shutdown 命令失败: {e}")

        # 等待所有进程退出（超时 5 秒）
        for i, p in enumerate(self.workers):
            try:
                p.join(timeout=5.0)
                if p.is_alive():
                    self.logger.warning(f"Arena {i} 工作进程未能正常退出，强制终止")
                    p.terminate()
                    p.join(timeout=1.0)
            except Exception as e:
                self.logger.error(f"关闭 Arena {i} 工作进程时出错: {e}")

        # 清理队列
        for q in self.cmd_queues:
            try:
                q.close()
            except Exception:
                pass

        try:
            self.result_queue.close()
        except Exception:
            pass

        self.workers.clear()
        self._started = False
        self.logger.info("所有竞技场工作进程已关闭")

    def __enter__(self) -> "ArenaPool":
        """上下文管理器入口，启动进程池"""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """上下文管理器出口，关闭进程池"""
        self.shutdown()
