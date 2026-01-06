#!/usr/bin/env python3
"""Checkpoint 迁移工具

将旧格式的 checkpoint 转换为新格式，提取 best_genomes 到单独文件。

旧格式：每个竞技场只有一个 checkpoint.pkl
新格式：每个竞技场有两个文件：
  - checkpoint.pkl：完整数据（恢复用）
  - best_genomes.pkl：仅包含 best_genomes（迁移用，轻量级）

使用方法:
    python scripts/migrate_checkpoints.py --checkpoint-dir checkpoints/multi_arena

示例:
    # 预览将要执行的操作（不实际执行）
    python scripts/migrate_checkpoints.py --dry-run

    # 执行迁移
    python scripts/migrate_checkpoints.py --checkpoint-dir checkpoints/multi_arena
"""

import argparse
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ArenaCheckpointData:
    """竞技场检查点数据结构（用于解析旧格式）"""

    arena_id: int
    episode: int = 0
    best_genomes: dict[str, list[tuple[bytes, float]]] = field(default_factory=dict)
    populations: dict[str, Any] = field(default_factory=dict)
    updated_at: float = 0.0


@dataclass
class MigrationStats:
    """迁移统计"""

    success: int = 0
    skipped: int = 0
    failed: int = 0


def atomic_write_pickle(file_path: Path, data: Any) -> None:
    """原子写入 pickle 文件

    先写入临时文件，再 rename 到目标路径，确保写入过程的原子性。

    Args:
        file_path: 目标文件路径
        data: 要写入的数据
    """
    # 在同一目录创建临时文件，确保 rename 是原子操作
    dir_path = file_path.parent
    fd, temp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_path)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(data, f)
        # 原子 rename
        os.rename(temp_path, file_path)
    except Exception:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def count_genomes(best_genomes: dict[str, list[tuple[bytes, float]]]) -> tuple[int, int]:
    """统计基因组数量

    Args:
        best_genomes: best_genomes 字典

    Returns:
        (种群数量, 基因组总数)
    """
    population_count = len(best_genomes)
    genome_count = sum(len(genomes) for genomes in best_genomes.values())
    return population_count, genome_count


def migrate_arena(
    arena_dir: Path, dry_run: bool = False
) -> tuple[bool, str, int, int]:
    """迁移单个竞技场的 checkpoint

    Args:
        arena_dir: 竞技场目录路径
        dry_run: 是否只预览不执行

    Returns:
        (成功/跳过/失败, 消息, 种群数量, 基因组数量)
        成功返回 (True, "成功", pop_count, genome_count)
        跳过返回 (True, "跳过 (原因)", 0, 0)
        失败返回 (False, "失败 (原因)", 0, 0)
    """
    checkpoint_path = arena_dir / "checkpoint.pkl"
    best_genomes_path = arena_dir / "best_genomes.pkl"

    # 检查 checkpoint.pkl 是否存在
    if not checkpoint_path.exists():
        return False, "失败 (checkpoint.pkl 不存在)", 0, 0

    # 检查 best_genomes.pkl 是否已存在
    if best_genomes_path.exists():
        return True, "跳过 (best_genomes.pkl 已存在)", 0, 0

    # 读取 checkpoint
    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
    except Exception as e:
        return False, f"失败 (checkpoint.pkl 损坏: {e})", 0, 0

    # 提取 best_genomes
    best_genomes: dict[str, list[tuple[bytes, float]]]
    if isinstance(checkpoint_data, ArenaCheckpointData):
        best_genomes = checkpoint_data.best_genomes
    elif isinstance(checkpoint_data, dict):
        best_genomes = checkpoint_data.get("best_genomes", {})
    elif hasattr(checkpoint_data, "best_genomes"):
        best_genomes = checkpoint_data.best_genomes
    else:
        return False, "失败 (无法识别的 checkpoint 格式)", 0, 0

    # 统计
    pop_count, genome_count = count_genomes(best_genomes)

    if dry_run:
        return True, f"将迁移 ({pop_count} 种群, {genome_count} 个基因组)", pop_count, genome_count

    # 写入 best_genomes.pkl（格式需要与 _read_arena_best_genomes 期望的一致）
    # 从 checkpoint 提取 arena_id 和 episode
    if isinstance(checkpoint_data, ArenaCheckpointData):
        arena_id = checkpoint_data.arena_id
        episode = checkpoint_data.episode
        updated_at = checkpoint_data.updated_at
    elif isinstance(checkpoint_data, dict):
        arena_id = checkpoint_data.get("arena_id", 0)
        episode = checkpoint_data.get("episode", 0)
        updated_at = checkpoint_data.get("updated_at", 0.0)
    else:
        arena_id = getattr(checkpoint_data, "arena_id", 0)
        episode = getattr(checkpoint_data, "episode", 0)
        updated_at = getattr(checkpoint_data, "updated_at", 0.0)

    best_genomes_data = {
        "arena_id": arena_id,
        "episode": episode,
        "best_genomes": best_genomes,
        "updated_at": updated_at,
    }
    try:
        atomic_write_pickle(best_genomes_path, best_genomes_data)
    except Exception as e:
        return False, f"失败 (写入失败: {e})", 0, 0

    return True, f"成功 ({pop_count} 种群, {genome_count} 个基因组)", pop_count, genome_count


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Checkpoint 迁移工具 - 将旧格式转换为新格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/multi_arena",
        help="checkpoint 目录路径（默认: checkpoints/multi_arena）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将要执行的操作，不实际执行",
    )

    args = parser.parse_args()

    # 解析路径
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        # 相对于项目根目录
        project_root = Path(__file__).parent.parent
        checkpoint_dir = project_root / checkpoint_dir

    print("Checkpoint 迁移工具")
    print("=" * 50)
    print(f"目标目录: {checkpoint_dir}")
    if args.dry_run:
        print("模式: 预览 (--dry-run)")
    print()

    # 检查目录是否存在
    if not checkpoint_dir.exists():
        print(f"错误: 目录不存在: {checkpoint_dir}")
        sys.exit(1)

    if not checkpoint_dir.is_dir():
        print(f"错误: 路径不是目录: {checkpoint_dir}")
        sys.exit(1)

    # 扫描竞技场目录
    print("正在扫描竞技场目录...")
    arena_dirs: list[Path] = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("arena_")],
        key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0,
    )

    if not arena_dirs:
        print("未找到任何竞技场目录 (arena_*)")
        sys.exit(0)

    print(f"找到 {len(arena_dirs)} 个竞技场")
    print()

    # 迁移统计
    stats = MigrationStats()

    # 处理每个竞技场
    for arena_dir in arena_dirs:
        arena_name = arena_dir.name
        success, message, pop_count, genome_count = migrate_arena(arena_dir, args.dry_run)

        if success:
            if "跳过" in message:
                stats.skipped += 1
            else:
                stats.success += 1
        else:
            stats.failed += 1

        print(f"处理 {arena_name}... {message}")

    # 打印统计
    print()
    if args.dry_run:
        print("预览完成！")
        print(f"- 将迁移: {stats.success}")
    else:
        print("迁移完成！")
        print(f"- 成功: {stats.success}")
    print(f"- 跳过: {stats.skipped}")
    print(f"- 失败: {stats.failed}")


if __name__ == "__main__":
    main()
