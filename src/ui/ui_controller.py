"""UI控制器模块

管理训练线程与UI线程的交互。
使用队列传递数据，事件控制暂停/停止。
"""

import queue
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.trainer import Trainer
    from src.ui.data_collector import UIDataCollector, UIDataSnapshot


class UIController:
    """UI控制器

    管理训练线程与UI线程的交互。
    使用队列传递数据，事件控制暂停/停止。

    线程模型：
        主线程(UI渲染) <--Queue--> 训练线程(Trainer)

    Attributes:
        trainer: 训练器实例
        data_collector: 数据采集器
        sample_rate: 采样率，每N个tick采集一次数据
        data_queue: 数据队列（主线程消费）
    """

    trainer: "Trainer"
    data_collector: "UIDataCollector"
    sample_rate: int
    data_queue: "queue.Queue[UIDataSnapshot]"

    def __init__(
        self,
        trainer: "Trainer",
        data_collector: "UIDataCollector",
        sample_rate: int = 1,
    ) -> None:
        """创建UI控制器

        Args:
            trainer: 训练器实例
            data_collector: 数据采集器
            sample_rate: 采样率，每N个tick采集一次数据
        """
        self.trainer = trainer
        self.data_collector = data_collector
        self.sample_rate = sample_rate

        # 数据队列（主线程消费）
        self.data_queue: queue.Queue[UIDataSnapshot] = queue.Queue(maxsize=10)

        # 控制信号
        self._pause_event: threading.Event = threading.Event()
        self._stop_event: threading.Event = threading.Event()
        self._training_thread: threading.Thread | None = None

        # 状态
        self._tick_counter: int = 0
        self._speed_factor: float = 1.0
        self._is_demo_mode: bool = False

    def start_training(self, episodes: int) -> None:
        """启动训练模式（后台线程）

        训练模式会进行NEAT进化，每个episode结束后进化种群。

        Args:
            episodes: 训练的episode数量
        """
        self._is_demo_mode = False
        self._stop_event.clear()
        self._pause_event.clear()

        self._training_thread = threading.Thread(
            target=self._training_loop,
            args=(episodes,),
            daemon=True,
            name="TrainingThread"
        )
        self._training_thread.start()

    def start_demo(self) -> None:
        """启动演示模式（后台线程）

        演示模式不进化，只运行tick展示。
        适合用于展示已训练的模型效果。
        """
        self._is_demo_mode = True
        self._stop_event.clear()
        self._pause_event.clear()

        self._training_thread = threading.Thread(
            target=self._demo_loop,
            daemon=True,
            name="DemoThread"
        )
        self._training_thread.start()

    def _training_loop(self, episodes: int) -> None:
        """训练主循环（在后台线程运行）

        按episode循环：
        1. 重置Agent和市场
        2. 运行tick
        3. 采样数据发送到队列
        4. 进化种群

        Args:
            episodes: 训练的episode数量
        """
        try:
            for ep in range(episodes):
                if self._stop_event.is_set():
                    break

                # 运行一个episode
                self.trainer.episode = ep + 1

                # 重置所有种群的Agent账户
                for population in self.trainer.populations.values():
                    population.reset_agents()

                # 重置市场状态
                self.trainer._reset_market()
                self.trainer.tick = 0
                self.data_collector.reset()

                # 运行ticks
                episode_length = self.trainer.config.training.episode_length
                for _ in range(episode_length):
                    if self._stop_event.is_set():
                        break

                    # 检查暂停
                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        time.sleep(0.1)

                    # 执行tick
                    self.trainer.run_tick()

                    # 采样数据
                    self._tick_counter += 1
                    if self._tick_counter >= self.sample_rate:
                        self._tick_counter = 0
                        self._collect_and_send_data()

                # 进化（训练模式）
                if not self._is_demo_mode and not self._stop_event.is_set():
                    current_price = self.trainer.matching_engine._orderbook.last_price
                    for population in self.trainer.populations.values():
                        population.evolve(current_price)

                    # 重建映射
                    self.trainer._register_all_agents()
                    self.trainer._build_agent_map()
                    self.trainer._build_execution_order()
        except Exception as e:
            import traceback
            print(f"Training loop error: {e}")
            traceback.print_exc()

    def _demo_loop(self) -> None:
        """演示循环（不进化）

        无限循环运行episode，但不进行进化。
        适合用于展示已训练的模型效果。
        每tick都采集数据，并受速度控制。
        """
        while not self._stop_event.is_set():
            # 运行一个episode
            self.trainer.episode += 1

            # 重置所有种群的Agent账户
            for population in self.trainer.populations.values():
                population.reset_agents()

            # 重置市场状态
            self.trainer._reset_market()
            self.trainer.tick = 0
            self.data_collector.reset()

            # 运行ticks（受速度控制）
            episode_length = self.trainer.config.training.episode_length
            for _ in range(episode_length):
                if self._stop_event.is_set():
                    break

                # 检查暂停
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    time.sleep(0.1)

                # 速度控制（演示模式）
                if self._speed_factor < 100:
                    time.sleep(0.1 / self._speed_factor)

                # 执行tick
                self.trainer.run_tick()

                # 每tick都采集数据（演示模式）
                self._collect_and_send_data()

    def _collect_and_send_data(self) -> None:
        """采集数据并发送到队列

        采集当前tick的数据快照，非阻塞放入队列。
        队列满时丢弃最旧的数据，保证UI总能获取到最新数据。
        """
        try:
            snapshot = self.data_collector.collect_tick_data(self.trainer)
            # 非阻塞放入队列，队列满则丢弃最旧数据
            try:
                self.data_queue.put_nowait(snapshot)
            except queue.Full:
                # 队列满时丢弃最旧的数据，放入新数据
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put_nowait(snapshot)
                except queue.Empty:
                    pass
        except Exception as e:
            # 采集失败不影响训练，但打印错误
            import traceback
            print(f"Data collection error: {e}")
            traceback.print_exc()

    def pause(self) -> None:
        """暂停训练/演示

        设置暂停事件，训练/演示线程会在下一个tick开始前等待。
        """
        self._pause_event.set()

    def resume(self) -> None:
        """恢复训练/演示

        清除暂停事件，训练/演示线程继续执行。
        """
        self._pause_event.clear()

    def stop(self) -> None:
        """停止训练/演示

        设置停止事件，等待训练/演示线程结束。
        超时2秒后强制返回（daemon线程会自动清理）。
        """
        self._stop_event.set()
        if self._training_thread and self._training_thread.is_alive():
            self._training_thread.join(timeout=2.0)

    def set_speed(self, factor: float) -> None:
        """设置演示速度

        Args:
            factor: 速度倍率 (1.0=正常, 10.0=10倍速, 0.1=0.1倍速)
                    范围限制在 [0.1, 100.0]
        """
        self._speed_factor = max(0.1, min(100.0, factor))

    def get_latest_data(self) -> "UIDataSnapshot | None":
        """非阻塞获取最新数据

        从队列中获取最新的数据快照。

        Returns:
            数据快照，队列为空时返回None
        """
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def is_running(self) -> bool:
        """检查是否正在运行

        Returns:
            True表示训练/演示线程正在运行
        """
        return self._training_thread is not None and self._training_thread.is_alive()

    def is_paused(self) -> bool:
        """检查是否暂停

        Returns:
            True表示当前处于暂停状态
        """
        return self._pause_event.is_set()

    def get_speed(self) -> float:
        """获取当前速度倍率

        Returns:
            当前速度倍率
        """
        return self._speed_factor

    def is_demo_mode(self) -> bool:
        """检查是否为演示模式

        Returns:
            True表示当前为演示模式，False表示训练模式
        """
        return self._is_demo_mode
