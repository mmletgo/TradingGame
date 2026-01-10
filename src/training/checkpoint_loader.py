# -*- coding: utf-8 -*-
"""Checkpoint 加载器模块

提供统一的 checkpoint 加载接口：
- 单训练场：checkpoints/ep_*.pkl 格式
"""

import os
import pickle
from enum import Enum
from typing import Any

from src.config.config import AgentType
from src.core.log_engine.logger import get_logger


class CheckpointType(Enum):
    """Checkpoint 类型枚举"""

    SINGLE_ARENA = "single"  # 单训练场


class CheckpointLoader:
    """Checkpoint 加载器

    提供统一的 checkpoint 加载接口，返回统一的数据结构。
    """

    _logger = get_logger(__name__)

    @staticmethod
    def detect_type(path: str) -> CheckpointType:
        """检测 checkpoint 类型

        检测逻辑:
        - 如果是 .pkl 文件 → SINGLE_ARENA

        Args:
            path: checkpoint 路径

        Returns:
            CheckpointType: 检测到的类型

        Raises:
            FileNotFoundError: 路径不存在
            ValueError: 无法识别的 checkpoint 格式
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint 路径不存在: {path}")

        # 如果是文件，检查是否是 .pkl 文件
        if os.path.isfile(path):
            if path.endswith(".pkl"):
                return CheckpointType.SINGLE_ARENA
            raise ValueError(f"无法识别的 checkpoint 文件格式: {path}")

        raise ValueError(f"无法识别的 checkpoint 路径类型: {path}")

    @staticmethod
    def load(path: str) -> dict[str, Any]:
        """加载 checkpoint，返回统一格式

        Args:
            path: checkpoint 路径

        Returns:
            统一格式的 checkpoint 数据:
            {
                "type": CheckpointType,
                "tick": int,
                "episode": int,
                "populations": {AgentType: {"generation": int, "neat_pop": ...}},
                "source_arena_id": int | None,
            }

        Raises:
            FileNotFoundError: 路径不存在
            ValueError: 无法识别的 checkpoint 格式或数据格式错误
        """
        CheckpointLoader.detect_type(path)
        return CheckpointLoader._load_single_arena(path)

    @staticmethod
    def _load_single_arena(path: str) -> dict[str, Any]:
        """加载单训练场 checkpoint

        Args:
            path: .pkl 文件路径

        Returns:
            统一格式的 checkpoint 数据
        """
        try:
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)
        except (pickle.PickleError, EOFError, OSError) as e:
            raise ValueError(f"加载 checkpoint 文件失败: {e}") from e

        # 验证数据格式
        if not isinstance(checkpoint, dict):
            raise ValueError(f"无效的 checkpoint 数据格式: 期望 dict，得到 {type(checkpoint)}")

        required_keys = ["tick", "episode", "populations"]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"checkpoint 缺少必要字段: {key}")

        # populations 的 key 已经是 AgentType，直接使用
        populations = checkpoint["populations"]

        # 验证 populations 格式
        if not isinstance(populations, dict):
            raise ValueError("populations 字段格式错误")

        return {
            "type": CheckpointType.SINGLE_ARENA,
            "tick": checkpoint["tick"],
            "episode": checkpoint["episode"],
            "populations": populations,
            "source_arena_id": None,
        }
