# -*- coding: utf-8 -*-
"""日志配置模块。

提供统一的日志管理系统，控制台只显示警告及以上级别，
详细日志写入文件。
"""

import logging
import os


def setup_logging(log_dir: str = "logs", console_level: int = logging.WARNING) -> None:
    """初始化日志系统。

    配置根日志器，添加文件处理器和控制台处理器：
    - 文件处理器：记录 INFO 级别及以上的详细日志
    - 控制台处理器：只记录 console_level 及以上的日志（默认 WARNING）

    Args:
        log_dir: 日志文件目录，默认为 'logs'
        console_level: 控制台日志级别，默认为 logging.WARNING

    Raises:
        OSError: 如果日志目录创建失败
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # 根日志器设置为 INFO，以便文件处理器能记录所有日志

    # 防止重复添加处理器
    if root_logger.handlers:
        return

    # 定义日志格式
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件处理器：记录 INFO 及以上级别
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "trading_game.log"),
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    # 控制台处理器：只记录 WARNING 及以上级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """获取指定模块的日志器。

    返回命名日志器实例，同名多次调用返回相同的 Logger 对象。
    日志器会继承根日志器的配置（文件处理器和控制台处理器）。

    Args:
        name: 日志器名称，通常使用模块的 __name__ 变量

    Returns:
        命名日志器实例

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("这是一条日志")
    """
    return logging.getLogger(name)
