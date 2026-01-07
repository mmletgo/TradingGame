# -*- coding: utf-8 -*-
"""Checkpoint 加载器模块

提供统一的 checkpoint 加载接口，自动识别并加载两种格式：
- 单训练场：checkpoints/ep_*.pkl 格式
- 多训练场：checkpoints/multi_arena/arena_*/checkpoint.pkl 格式
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
    MULTI_ARENA = "multi"  # 多训练场


class CheckpointLoader:
    """Checkpoint 加载器

    提供统一的 checkpoint 加载接口，自动识别格式并返回统一的数据结构。
    """

    _logger = get_logger(__name__)

    @staticmethod
    def detect_type(path: str) -> CheckpointType:
        """检测 checkpoint 类型

        检测逻辑:
        - 如果是 .pkl 文件 → SINGLE_ARENA
        - 如果是目录且包含 arena_* 子目录 → MULTI_ARENA

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

        # 如果是目录，检查是否包含 arena_* 子目录
        if os.path.isdir(path):
            entries = os.listdir(path)
            arena_dirs = [e for e in entries if e.startswith("arena_")]
            if arena_dirs:
                return CheckpointType.MULTI_ARENA
            raise ValueError(f"目录中未找到 arena_* 子目录: {path}")

        raise ValueError(f"无法识别的 checkpoint 路径类型: {path}")

    @staticmethod
    def load(path: str, arena_id: int | None = None) -> dict[str, Any]:
        """加载 checkpoint，返回统一格式

        Args:
            path: checkpoint 路径
            arena_id: 多训练场模式下指定加载哪个 arena（可选，默认选择 episode 最高的）

        Returns:
            统一格式的 checkpoint 数据:
            {
                "type": CheckpointType,
                "tick": int,
                "episode": int,
                "populations": {AgentType: {"generation": int, "neat_pop": ...}},
                "source_arena_id": int | None,  # 多训练场时为 arena_id
            }

        Raises:
            FileNotFoundError: 路径不存在
            ValueError: 无法识别的 checkpoint 格式或数据格式错误
        """
        checkpoint_type = CheckpointLoader.detect_type(path)

        if checkpoint_type == CheckpointType.SINGLE_ARENA:
            return CheckpointLoader._load_single_arena(path)
        else:
            return CheckpointLoader._load_multi_arena(path, arena_id)

    @staticmethod
    def list_arenas(path: str) -> list[dict[str, Any]]:
        """列出多训练场的所有 arena 信息

        优先使用轻量级的 best_genomes.pkl 获取信息，避免加载完整 checkpoint。

        Args:
            path: 多训练场 checkpoint 目录路径

        Returns:
            [{"arena_id": int, "episode": int, "updated_at": float}, ...]

        Raises:
            ValueError: 不是多训练场格式
        """
        checkpoint_type = CheckpointLoader.detect_type(path)
        if checkpoint_type != CheckpointType.MULTI_ARENA:
            raise ValueError(f"不是多训练场格式的 checkpoint: {path}")

        arenas: list[dict[str, Any]] = []

        for entry in os.listdir(path):
            if not entry.startswith("arena_"):
                continue

            try:
                arena_id = int(entry.split("_")[1])
            except (IndexError, ValueError):
                continue

            # 优先使用轻量级的 best_genomes.pkl（~1-2MB）
            best_genomes_path = os.path.join(path, entry, "best_genomes.pkl")
            if os.path.exists(best_genomes_path):
                try:
                    with open(best_genomes_path, "rb") as f:
                        data = pickle.load(f)

                    arenas.append(
                        {
                            "arena_id": arena_id,
                            "episode": data.get("episode", 0),
                            "updated_at": data.get("updated_at", 0.0),
                        }
                    )
                    continue
                except (pickle.PickleError, EOFError, OSError, KeyError) as e:
                    CheckpointLoader._logger.debug(
                        f"读取 arena {arena_id} best_genomes 失败: {e}，尝试完整 checkpoint"
                    )

            # 回退到完整 checkpoint（~700MB，较慢）
            checkpoint_path = os.path.join(path, entry, "checkpoint.pkl")
            if not os.path.exists(checkpoint_path):
                continue

            try:
                with open(checkpoint_path, "rb") as f:
                    arena_data = pickle.load(f)

                arenas.append(
                    {
                        "arena_id": arena_id,
                        "episode": arena_data.episode,
                        "updated_at": arena_data.updated_at,
                    }
                )
            except (pickle.PickleError, EOFError, OSError, AttributeError) as e:
                CheckpointLoader._logger.warning(
                    f"读取 arena {arena_id} checkpoint 失败: {e}"
                )
                continue

        # 按 episode 降序排序
        arenas.sort(key=lambda x: x["episode"], reverse=True)

        return arenas

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

    @staticmethod
    def _load_multi_arena(
        path: str, arena_id: int | None = None
    ) -> dict[str, Any]:
        """加载多训练场 checkpoint

        Args:
            path: 多训练场目录路径
            arena_id: 指定加载的 arena ID（None 则选择 episode 最高的）

        Returns:
            统一格式的 checkpoint 数据
        """
        # 如果未指定 arena_id，选择 episode 最高的
        if arena_id is None:
            arenas = CheckpointLoader.list_arenas(path)
            if not arenas:
                raise ValueError(f"未找到有效的 arena checkpoint: {path}")
            arena_id = arenas[0]["arena_id"]
            CheckpointLoader._logger.info(
                f"未指定 arena_id，自动选择 episode 最高的 arena_{arena_id}"
            )

        # 加载指定 arena 的 checkpoint
        checkpoint_path = os.path.join(path, f"arena_{arena_id}", "checkpoint.pkl")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Arena {arena_id} checkpoint 不存在: {checkpoint_path}")

        try:
            with open(checkpoint_path, "rb") as f:
                arena_data = pickle.load(f)
        except (pickle.PickleError, EOFError, OSError) as e:
            raise ValueError(f"加载 arena checkpoint 失败: {e}") from e

        # ArenaCheckpointData 格式转换
        # populations 的 key 是 agent_type.value（字符串），需要转换为 AgentType
        raw_populations = arena_data.populations
        populations: dict[AgentType, dict[str, Any]] = {}

        for agent_type_value, pop_data in raw_populations.items():
            # 尝试将字符串转换为 AgentType
            if isinstance(agent_type_value, str):
                try:
                    agent_type = AgentType(agent_type_value)
                except ValueError:
                    CheckpointLoader._logger.warning(
                        f"跳过未知的 agent_type: {agent_type_value}"
                    )
                    continue
            elif isinstance(agent_type_value, AgentType):
                agent_type = agent_type_value
            else:
                CheckpointLoader._logger.warning(
                    f"跳过无效的 agent_type 类型: {type(agent_type_value)}"
                )
                continue

            populations[agent_type] = pop_data

        return {
            "type": CheckpointType.MULTI_ARENA,
            "tick": 0,  # 多训练场 checkpoint 不保存 tick，从 0 开始
            "episode": arena_data.episode,
            "populations": populations,
            "source_arena_id": arena_id,
        }
