# -*- coding: utf-8 -*-
"""日志引擎测试。"""

import logging
import os
import shutil

import pytest

from src.core.log_engine.logger import get_logger, setup_logging


@pytest.fixture
def reset_logging():
    """每个测试前后重置根日志器。"""
    root_logger = logging.getLogger()
    # 保存原始处理器并清空
    original_handlers = root_logger.handlers[:]
    root_logger.handlers.clear()
    # 确保至少有 WARNING 级别
    root_logger.setLevel(logging.WARNING)
    yield
    # 测试后恢复
    root_logger.handlers.clear()
    for handler in original_handlers:
        root_logger.addHandler(handler)


@pytest.fixture
def clean_log_dir(reset_logging):
    """测试前后清理日志目录。"""
    log_dir = "test_logs"
    # 测试前清理
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    yield log_dir
    # 测试后清理
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)


def test_setup_logging_creates_directory(clean_log_dir):
    """测试 setup_logging 创建日志目录。"""
    setup_logging(log_dir=clean_log_dir)
    assert os.path.exists(clean_log_dir)
    assert os.path.isdir(clean_log_dir)


def test_setup_logging_creates_log_file(clean_log_dir):
    """测试 setup_logging 创建日志文件。"""
    setup_logging(log_dir=clean_log_dir)
    log_file = os.path.join(clean_log_dir, "trading_game.log")
    # 写入一条日志来触发文件创建
    logger = logging.getLogger(__name__)
    logger.info("Test message")
    # 关闭处理器以刷新缓冲区
    for handler in logging.getLogger().handlers:
        handler.flush()
        handler.close()
    assert os.path.exists(log_file)


def test_setup_logging_console_level(clean_log_dir):
    """测试 setup_logging 控制台级别正确。"""
    setup_logging(log_dir=clean_log_dir, console_level=logging.ERROR)
    root_logger = logging.getLogger()

    # 检查控制台处理器级别（排除 FileHandler，因为 FileHandler 继承自 StreamHandler）
    console_handlers = [
        h for h in root_logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert len(console_handlers) == 1
    assert console_handlers[0].level == logging.ERROR


def test_setup_logging_file_level(clean_log_dir):
    """测试 setup_logging 文件处理器级别为 INFO。"""
    setup_logging(log_dir=clean_log_dir)
    root_logger = logging.getLogger()

    # 检查文件处理器级别
    file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].level == logging.INFO


def test_setup_logging_prevents_duplicate_handlers(clean_log_dir):
    """测试 setup_logging 防止重复添加处理器。"""
    setup_logging(log_dir=clean_log_dir)
    root_logger = logging.getLogger()
    handler_count_before = len(root_logger.handlers)

    # 再次调用，不应添加新的处理器
    setup_logging(log_dir=clean_log_dir)
    handler_count_after = len(root_logger.handlers)

    assert handler_count_before == handler_count_after


def test_setup_logging_writes_to_file(clean_log_dir):
    """测试日志写入文件。"""
    setup_logging(log_dir=clean_log_dir)
    log_file = os.path.join(clean_log_dir, "trading_game.log")

    logger = logging.getLogger("test_module")
    logger.info("Test info message")
    logger.warning("Test warning message")

    # 关闭处理器以刷新缓冲区
    for handler in logging.getLogger().handlers:
        handler.flush()
        handler.close()

    # 读取日志文件验证内容
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Test info message" in content
    assert "Test warning message" in content


def test_setup_logging_default_console_level(clean_log_dir):
    """测试 setup_logging 默认控制台级别为 WARNING。"""
    setup_logging(log_dir=clean_log_dir)
    root_logger = logging.getLogger()

    # 排除 FileHandler
    console_handlers = [
        h for h in root_logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert console_handlers[0].level == logging.WARNING


def test_get_logger_returns_logger(clean_log_dir):
    """测试 get_logger 返回 Logger 实例。"""
    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"


def test_get_logger_same_instance(clean_log_dir):
    """测试 get_logger 同名返回相同实例。"""
    logger1 = get_logger("same.name")
    logger2 = get_logger("same.name")
    assert logger1 is logger2


def test_get_logger_different_names(clean_log_dir):
    """测试 get_logger 不同名称返回不同实例。"""
    logger1 = get_logger("module.a")
    logger2 = get_logger("module.b")
    assert logger1 is not logger2
    assert logger1.name == "module.a"
    assert logger2.name == "module.b"


def test_get_logger_writes_to_file(clean_log_dir):
    """测试 get_logger 获取的日志器能写入日志文件。"""
    setup_logging(log_dir=clean_log_dir)
    log_file = os.path.join(clean_log_dir, "trading_game.log")

    logger = get_logger("test.writer")
    logger.info("Test message from get_logger")

    # 关闭处理器以刷新缓冲区
    for handler in logging.getLogger().handlers:
        handler.flush()
        handler.close()

    # 读取日志文件验证内容
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Test message from get_logger" in content
    assert "test.writer" in content
