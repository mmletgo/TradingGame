#!/usr/bin/env python3
"""Checkpoint 格式转换脚本

将旧格式（包含完整 neat_pop 对象）转换为新的精简格式（只包含核心数据）。

使用方法:
    # 转换单个文件
    python scripts/convert_checkpoint.py checkpoints/parallel_arena_gen_100.pkl

    # 转换目录下所有 checkpoint
    python scripts/convert_checkpoint.py checkpoints/

    # 转换并覆盖原文件（默认会备份）
    python scripts/convert_checkpoint.py checkpoints/ --overwrite

    # 不备份原文件
    python scripts/convert_checkpoint.py checkpoints/ --overwrite --no-backup
"""

import argparse
import gzip
import pickle
import shutil
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.population import (
    _serialize_genomes_numpy,
    _serialize_species_data,
)


def get_checkpoint_version(checkpoint_data: dict) -> int:
    """获取 checkpoint 版本"""
    return checkpoint_data.get("checkpoint_version", 1)


def convert_checkpoint(input_path: Path, output_path: Path) -> dict:
    """转换单个 checkpoint 文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径

    Returns:
        转换统计信息
    """
    # 读取 checkpoint
    with open(input_path, "rb") as f:
        magic = f.read(2)

    if magic == b"\x1f\x8b":
        with gzip.open(input_path, "rb") as f:
            checkpoint_data = pickle.load(f)
    else:
        with open(input_path, "rb") as f:
            checkpoint_data = pickle.load(f)

    # 检查版本
    old_version = get_checkpoint_version(checkpoint_data)
    if old_version >= 2:
        return {"status": "skipped", "reason": f"已是 version {old_version}"}

    # 转换
    new_checkpoint: dict = {
        "checkpoint_version": 2,
        "generation": checkpoint_data.get("generation", checkpoint_data.get("episode", 0)),
        "populations": {},
    }

    populations_data = checkpoint_data.get("populations", {})
    stats = {
        "status": "converted",
        "populations": 0,
        "sub_populations": 0,
    }

    for agent_type, pop_data in populations_data.items():
        if pop_data.get("is_sub_population_manager"):
            # SubPopulationManager 格式
            new_pop_data: dict = {
                "is_sub_population_manager": True,
                "sub_population_count": pop_data.get("sub_population_count", 0),
                "sub_populations": [],
            }

            sub_pops_data = pop_data.get("sub_populations", [])
            for sub_pop_data in sub_pops_data:
                neat_pop = sub_pop_data.get("neat_pop")
                if neat_pop is None:
                    continue

                # 序列化核心数据
                genome_data = _serialize_genomes_numpy(neat_pop.population)
                species_data = _serialize_species_data(neat_pop.species)

                new_sub_pop_data = {
                    "generation": sub_pop_data.get("generation", 0),
                    "genome_data": genome_data,
                    "species_data": species_data,
                }
                new_pop_data["sub_populations"].append(new_sub_pop_data)
                stats["sub_populations"] += 1

            new_checkpoint["populations"][agent_type] = new_pop_data
            stats["populations"] += 1
        else:
            # 单个 Population 格式
            neat_pop = pop_data.get("neat_pop")
            if neat_pop is None:
                continue

            # 序列化核心数据
            genome_data = _serialize_genomes_numpy(neat_pop.population)
            species_data = _serialize_species_data(neat_pop.species)

            new_checkpoint["populations"][agent_type] = {
                "generation": pop_data.get("generation", 0),
                "genome_data": genome_data,
                "species_data": species_data,
            }
            stats["populations"] += 1

    # 保存新格式
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wb") as f:
        pickle.dump(new_checkpoint, f)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将旧格式 checkpoint 转换为精简格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=str,
        help="Checkpoint 文件或目录路径",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖原文件（默认输出到 .v2.pkl）",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="覆盖时不备份原文件",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示要转换的文件，不实际执行",
    )
    args = parser.parse_args()

    input_path = Path(args.path)

    # 收集要转换的文件
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.pkl"))
        if not files:
            print(f"目录 {input_path} 中没有找到 .pkl 文件")
            return
    else:
        print(f"路径不存在: {input_path}")
        return

    print(f"找到 {len(files)} 个文件")
    print("=" * 60)

    converted = 0
    skipped = 0
    errors = 0

    for file_path in sorted(files):
        if args.overwrite:
            output_path = file_path
            backup_path = file_path.with_suffix(".pkl.bak")
        else:
            # 输出到 .v2.pkl
            output_path = file_path.with_suffix(".v2.pkl")
            backup_path = None

        print(f"\n处理: {file_path.name}")

        if args.dry_run:
            print(f"  -> 将输出到: {output_path.name}")
            continue

        try:
            # 如果覆盖且需要备份
            if args.overwrite and not args.no_backup and file_path.exists():
                shutil.copy2(file_path, backup_path)
                print(f"  备份到: {backup_path.name}")

            # 转换
            stats = convert_checkpoint(file_path, output_path)

            if stats["status"] == "skipped":
                print(f"  跳过: {stats['reason']}")
                skipped += 1
            else:
                print(f"  转换成功: {stats['populations']} 个种群, {stats['sub_populations']} 个子种群")
                converted += 1

                # 显示文件大小变化
                old_size = file_path.stat().st_size / 1024 / 1024
                new_size = output_path.stat().st_size / 1024 / 1024
                reduction = (1 - new_size / old_size) * 100 if old_size > 0 else 0
                print(f"  文件大小: {old_size:.1f}MB -> {new_size:.1f}MB ({reduction:.1f}% 减少)")

        except Exception as e:
            print(f"  错误: {e}")
            errors += 1

    print("\n" + "=" * 60)
    print(f"完成: 转换 {converted}, 跳过 {skipped}, 错误 {errors}")


if __name__ == "__main__":
    main()
