#!/usr/bin/env python3
"""
多进程搜索合适的 NEAT compatibility_threshold 参数

目标 species 数量：
- 散户（pop_size=150）：> 10
- 做市商（pop_size=150）：> 4
- 高级散户（pop_size=100）：> 1
- 庄家（pop_size=200）：> 1

使用方法：
    python scripts/tune_compatibility_threshold.py
"""

import os
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import neat
import numpy as np


@dataclass
class SpeciesConfig:
    """物种配置"""
    name: str
    config_file: str
    pop_size: int
    min_species: int
    max_species: int


# 物种配置列表
SPECIES_CONFIGS: list[SpeciesConfig] = [
    SpeciesConfig("散户", "neat_retail.cfg", 150, 10, 100),
    SpeciesConfig("高级散户", "neat_retail_pro.cfg", 100, 2, 50),
    SpeciesConfig("庄家", "neat_whale.cfg", 200, 2, 100),
    SpeciesConfig("做市商", "neat_market_maker.cfg", 150, 4, 50),
]

# 要测试的 threshold 范围
THRESHOLD_RANGE = np.arange(0.5, 5.1, 0.1)


def create_temp_config(
    original_config_path: str,
    threshold: float,
    pop_size: int,
) -> str:
    """创建临时配置文件，修改 threshold 值"""
    temp_dir = tempfile.mkdtemp()
    temp_config_path = os.path.join(temp_dir, "temp_config.cfg")

    with open(original_config_path, 'r') as f:
        content = f.read()

    # 修改 compatibility_threshold
    lines = content.split('\n')
    new_lines: list[str] = []
    for line in lines:
        if line.strip().startswith('compatibility_threshold'):
            new_lines.append(f'compatibility_threshold = {threshold}')
        elif line.strip().startswith('pop_size'):
            new_lines.append(f'pop_size                 = {pop_size}')
        else:
            new_lines.append(line)

    with open(temp_config_path, 'w') as f:
        f.write('\n'.join(new_lines))

    return temp_config_path


def test_single_threshold(
    config_path: str,
    threshold: float,
    pop_size: int,
    num_tests: int = 5,
) -> tuple[float, float, int, int]:
    """
    测试单个 threshold 值

    Args:
        config_path: NEAT 配置文件路径
        threshold: 要测试的 compatibility_threshold 值
        pop_size: 种群大小
        num_tests: 测试次数（取平均）

    Returns:
        (threshold, avg_species_count, min_species, max_species)
    """
    temp_config_path = create_temp_config(config_path, threshold, pop_size)

    try:
        species_counts: list[int] = []

        for _ in range(num_tests):
            try:
                config = neat.Config(
                    neat.DefaultGenome,
                    neat.DefaultReproduction,
                    neat.DefaultSpeciesSet,
                    neat.DefaultStagnation,
                    temp_config_path,
                )

                # 创建种群
                pop = neat.Population(config)

                # 统计 species 数量
                num_species = len(pop.species.species)
                species_counts.append(num_species)

            except Exception as e:
                print(f"测试失败 (threshold={threshold}): {e}")
                species_counts.append(0)

        avg_count = float(np.mean(species_counts))
        min_count = min(species_counts)
        max_count = max(species_counts)

        return (threshold, avg_count, min_count, max_count)

    finally:
        # 清理临时文件
        temp_dir = os.path.dirname(temp_config_path)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_species_config(
    species_config: SpeciesConfig,
    config_dir: str,
    num_tests: int = 5,
) -> dict[float, tuple[float, int, int]]:
    """
    测试单个物种的所有 threshold 值

    Returns:
        {threshold: (avg_species, min_species, max_species)}
    """
    config_path = os.path.join(config_dir, species_config.config_file)
    results: dict[float, tuple[float, int, int]] = {}

    print(f"\n{'='*60}")
    print(f"测试 {species_config.name} (pop_size={species_config.pop_size})")
    print(f"目标 species: {species_config.min_species} < n < {species_config.max_species}")
    print(f"{'='*60}")

    # 使用多进程并行测试
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {}

        for t in THRESHOLD_RANGE:
            threshold_val = float(t)
            future = executor.submit(
                test_single_threshold,
                config_path,
                threshold_val,
                species_config.pop_size,
                num_tests,
            )
            futures[future] = threshold_val

        for future in as_completed(futures):
            threshold, avg, min_s, max_s = future.result()
            results[float(threshold)] = (avg, min_s, max_s)

            # 判断是否满足条件
            meets_criteria = species_config.min_species < avg < species_config.max_species
            marker = "✓" if meets_criteria else " "
            print(f"  threshold={threshold:.1f}: avg={avg:6.1f}, range=[{min_s:3d}, {max_s:3d}] {marker}")

    return results


def find_optimal_threshold(
    results: dict[float, tuple[float, int, int]],
    min_species: int,
    max_species: int,
) -> Optional[float]:
    """
    找到满足条件的最优 threshold

    优先选择 avg_species 最接近目标范围中点的 threshold
    """
    target_mid = (min_species + max_species) / 2

    valid_thresholds: list[tuple[float, float]] = []
    for threshold, (avg, min_s, max_s) in results.items():
        if min_species < avg < max_species:
            distance = abs(avg - target_mid)
            valid_thresholds.append((threshold, distance))

    if not valid_thresholds:
        return None

    # 按距离排序，取最接近中点的
    valid_thresholds.sort(key=lambda x: x[1])
    return valid_thresholds[0][0]


def update_config_file(config_path: str, threshold: float) -> None:
    """更新配置文件中的 threshold 值"""
    with open(config_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    new_lines: list[str] = []
    for line in lines:
        if line.strip().startswith('compatibility_threshold'):
            new_lines.append(f'compatibility_threshold = {threshold}')
        else:
            new_lines.append(line)

    with open(config_path, 'w') as f:
        f.write('\n'.join(new_lines))


def main() -> None:
    """主函数"""
    # 确定配置目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_dir = project_root / "config"

    print("=" * 60)
    print("NEAT compatibility_threshold 参数调优")
    print("=" * 60)
    print(f"配置目录: {config_dir}")
    print(f"测试范围: threshold = {THRESHOLD_RANGE[0]:.1f} ~ {THRESHOLD_RANGE[-1]:.1f}")

    # 存储所有结果
    all_results: dict[str, dict[float, tuple[float, int, int]]] = {}
    optimal_thresholds: dict[str, Optional[float]] = {}

    # 测试每个物种
    for species_config in SPECIES_CONFIGS:
        results = test_species_config(
            species_config,
            str(config_dir),
            num_tests=5,
        )
        all_results[species_config.name] = results

        # 找到最优 threshold
        optimal = find_optimal_threshold(
            results,
            species_config.min_species,
            species_config.max_species,
        )
        optimal_thresholds[species_config.name] = optimal

    # 打印总结
    print("\n" + "=" * 60)
    print("调优结果总结")
    print("=" * 60)

    for species_config in SPECIES_CONFIGS:
        optimal = optimal_thresholds[species_config.name]
        if optimal is not None:
            avg, min_s, max_s = all_results[species_config.name][optimal]
            print(f"{species_config.name}:")
            print(f"  推荐 threshold = {optimal:.1f}")
            print(f"  预期 species 数量: {avg:.1f} (范围: [{min_s}, {max_s}])")
            print(f"  目标范围: {species_config.min_species} < n < {species_config.max_species}")
        else:
            print(f"{species_config.name}:")
            print(f"  未找到满足条件的 threshold!")
            print(f"  请调整 compatibility_disjoint_coefficient 或 compatibility_weight_coefficient")

    # 询问是否应用
    print("\n" + "=" * 60)
    apply = input("是否将推荐的 threshold 应用到配置文件? (y/n): ").strip().lower()

    if apply == 'y':
        for species_config in SPECIES_CONFIGS:
            optimal = optimal_thresholds[species_config.name]
            if optimal is not None:
                config_path = config_dir / species_config.config_file
                update_config_file(str(config_path), optimal)
                print(f"已更新 {species_config.config_file}: threshold = {optimal}")
        print("\n配置文件已更新完成!")
    else:
        print("\n未应用更改。")


if __name__ == "__main__":
    main()
