#!/usr/bin/env python3
"""进化曲线可视化脚本

读取所有已保存的测试结果，绘制进化曲线图。
以第一代的数值为基准（1.0），使用 relative_improvement_pct 累积计算各代的进化值。

使用方法:
    # 默认绘制并保存到 outputs/evolution_curve.png
    python scripts/plot_evolution_curve.py

    # 指定输出路径
    python scripts/plot_evolution_curve.py --output my_curve.png

    # 指定测试结果目录
    python scripts/plot_evolution_curve.py --results-dir checkpoints/test_results

    # 显示详细信息
    python scripts/plot_evolution_curve.py --verbose
"""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import AgentType

# 物种颜色方案
SPECIES_COLORS: dict[AgentType, str] = {
    AgentType.RETAIL_PRO: "#2ca02c",   # 绿色
    AgentType.MARKET_MAKER: "#9467bd", # 紫色
}

# 物种中文名称
SPECIES_NAMES: dict[AgentType, str] = {
    AgentType.RETAIL_PRO: "高级散户",
    AgentType.MARKET_MAKER: "做市商",
}


def load_baseline_results(
    results_dir: str,
) -> tuple[dict[int, dict[AgentType, float]], int | None]:
    """加载所有基准测试结果

    Args:
        results_dir: 测试结果目录

    Returns:
        ({generation: {AgentType: avg_fitness}}, episodes_per_run)
        episodes_per_run 为 None 表示数据中不存在该字段
    """
    baseline_dir = Path(results_dir) / "baseline"
    results: dict[int, dict[AgentType, float]] = {}
    episodes_per_run: int | None = None

    if not baseline_dir.exists():
        return results, episodes_per_run

    for pkl_file in baseline_dir.glob("gen_*.pkl"):
        # 解析文件名：gen_{N}.pkl
        match = re.match(r"gen_(\d+)\.pkl", pkl_file.name)
        if not match:
            continue

        generation = int(match.group(1))

        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            # 尝试提取 episodes_per_run（只需读取一次）
            if episodes_per_run is None:
                episodes_per_run = data.get("episodes_per_run")

            species_summary = data.get("species_summary", {})
            gen_results: dict[AgentType, float] = {}

            for agent_type in AgentType:
                species_data = species_summary.get(agent_type, {})
                avg_fitness = species_data.get("avg_fitness", 0.0)
                gen_results[agent_type] = avg_fitness

            results[generation] = gen_results

        except Exception as e:
            print(f"加载 {pkl_file} 失败: {e}")
            continue

    return results, episodes_per_run


def load_comparison_results(
    results_dir: str,
) -> dict[int, dict[AgentType, dict[str, float]]]:
    """加载所有比较测试结果

    Args:
        results_dir: 测试结果目录

    Returns:
        {generation: {AgentType: {"baseline_return": float, "comparison_return": float}}}
    """
    comparison_dir = Path(results_dir) / "comparison"
    baseline_dir = Path(results_dir) / "baseline"
    results: dict[int, dict[AgentType, dict[str, float]]] = {}

    if not comparison_dir.exists():
        return results

    for pkl_file in comparison_dir.glob("gen_*_vs_gen_*_*.pkl"):
        # 解析文件名：gen_{N}_vs_gen_{M}_{species}.pkl
        match = re.match(r"gen_(\d+)_vs_gen_(\d+)_(.+)\.pkl", pkl_file.name)
        if not match:
            continue

        generation = int(match.group(1))
        _base_generation = int(match.group(2))  # 提取但不使用，仅用于文件名解析
        species_name = match.group(3)

        # 找到对应的 AgentType
        agent_type = None
        for at in AgentType:
            if at.value == species_name:
                agent_type = at
                break

        if agent_type is None:
            continue

        try:
            # 加载比较测试结果
            with open(pkl_file, "rb") as f:
                comparison_data = pickle.load(f)

            # 加载对应的基准测试结果
            baseline_file = baseline_dir / f"gen_{generation}.pkl"
            if not baseline_file.exists():
                continue

            with open(baseline_file, "rb") as f:
                baseline_data = pickle.load(f)

            # 获取该物种的适应度
            baseline_species = baseline_data.get("species_summary", {}).get(agent_type, {})
            comparison_species = comparison_data.get("species_summary", {}).get(agent_type, {})

            baseline_fitness = baseline_species.get("avg_fitness", 0.0)
            comparison_fitness = comparison_species.get("avg_fitness", 0.0)

            if generation not in results:
                results[generation] = {}

            results[generation][agent_type] = {
                "baseline_fitness": baseline_fitness,
                "comparison_fitness": comparison_fitness,
            }

        except Exception as e:
            print(f"加载 {pkl_file} 失败: {e}")
            continue

    return results


def calculate_relative_improvement(
    comparison_results: dict[int, dict[AgentType, dict[str, float]]],
) -> dict[int, dict[AgentType, float]]:
    """计算相对改善百分比

    Args:
        comparison_results: 比较测试结果

    Returns:
        {generation: {AgentType: relative_improvement_pct}}
    """
    improvements: dict[int, dict[AgentType, float]] = {}

    for generation, species_data in comparison_results.items():
        improvements[generation] = {}

        for agent_type, fitness_data in species_data.items():
            baseline_fitness = fitness_data["baseline_fitness"]
            comparison_fitness = fitness_data["comparison_fitness"]

            # 计算相对改善百分比
            improvement = comparison_fitness - baseline_fitness
            if abs(baseline_fitness) > 0.001:
                relative_improvement = improvement / abs(baseline_fitness) * 100
            else:
                relative_improvement = 100.0 if improvement > 0 else 0.0

            improvements[generation][agent_type] = relative_improvement

    return improvements


def calculate_cumulative_evolution(
    improvements: dict[int, dict[AgentType, float]],
) -> dict[AgentType, dict[int, float]]:
    """计算累积进化值

    以第一代（或最小代）为基准值 1.0，使用乘法累积计算各代的进化值。

    Args:
        improvements: {generation: {AgentType: relative_improvement_pct}}

    Returns:
        {AgentType: {generation: cumulative_value}}
    """
    if not improvements:
        return {}

    generations = sorted(improvements.keys())
    min_gen = generations[0]

    # 初始化结果
    cumulative: dict[AgentType, dict[int, float]] = {
        agent_type: {} for agent_type in AgentType
    }

    # 第一代为基准值 1.0
    for agent_type in AgentType:
        cumulative[agent_type][min_gen] = 1.0

    # 从第二代开始累积
    for i, gen in enumerate(generations[1:], 1):
        prev_gen = generations[i - 1]

        for agent_type in AgentType:
            # 获取该代该物种的改善百分比
            improvement_pct = improvements.get(gen, {}).get(agent_type, 0.0)

            # 处理无穷大的情况
            if improvement_pct == float("inf"):
                improvement_pct = 100.0  # 限制最大改善为 100%
            elif improvement_pct == float("-inf"):
                improvement_pct = -50.0  # 限制最大退化为 -50%

            # 累积计算
            prev_value = cumulative[agent_type].get(prev_gen, 1.0)
            new_value = prev_value * (1 + improvement_pct / 100)

            # 限制范围，防止极端值
            new_value = max(0.01, min(100.0, new_value))

            cumulative[agent_type][gen] = new_value

    return cumulative


def calculate_combined_evolution(
    cumulative: dict[AgentType, dict[int, float]],
) -> dict[int, float]:
    """计算综合进化值（几何平均）

    Args:
        cumulative: {AgentType: {generation: cumulative_value}}

    Returns:
        {generation: combined_value}
    """
    if not cumulative:
        return {}

    # 获取所有代数
    all_generations: set[int] = set()
    for species_data in cumulative.values():
        all_generations.update(species_data.keys())

    combined: dict[int, float] = {}
    for gen in sorted(all_generations):
        values = []
        for agent_type in AgentType:
            if gen in cumulative.get(agent_type, {}):
                values.append(cumulative[agent_type][gen])

        if values:
            # 使用几何平均
            combined[gen] = float(np.power(np.prod(values), 1 / len(values)))

    return combined


def find_chinese_font() -> str | None:
    """查找可用的中文字体"""
    import matplotlib.font_manager as fm

    # 候选字体列表
    candidates = [
        # Linux
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "AR PL UMing CN",
        "Droid Sans Fallback",
        # Windows
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        # macOS
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
    ]

    # 获取系统可用字体
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font in candidates:
        if font in available_fonts:
            return font

    return None


def plot_evolution_curves(
    cumulative: dict[AgentType, dict[int, float]],
    combined: dict[int, float],
    output_path: str,
    verbose: bool = False,
    episodes_per_run: int | None = None,
) -> str:
    """绘制进化曲线图

    Args:
        cumulative: 各物种的累积进化值
        combined: 综合进化值
        output_path: 输出路径
        verbose: 是否显示详细信息
        episodes_per_run: 每代测试的 episode 数量

    Returns:
        保存的文件路径
    """
    # 设置中文字体
    chinese_font = find_chinese_font()
    if chinese_font:
        plt.rcParams["font.sans-serif"] = [chinese_font]
        plt.rcParams["axes.unicode_minus"] = False
        if verbose:
            print(f"使用中文字体: {chinese_font}")
    else:
        if verbose:
            print("警告: 未找到中文字体，图表可能显示乱码")

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))

    # 获取所有代数
    all_generations: set[int] = set()
    for species_data in cumulative.values():
        all_generations.update(species_data.keys())
    generations = sorted(all_generations)

    if not generations:
        print("错误: 没有可用的数据")
        return ""

    # 绘制各物种曲线
    for agent_type in AgentType:
        if agent_type not in cumulative:
            continue

        species_data = cumulative[agent_type]
        x = sorted(species_data.keys())
        y = [species_data[gen] for gen in x]

        ax.plot(
            x,
            y,
            color=SPECIES_COLORS[agent_type],
            linewidth=1.5,
            marker="o",
            markersize=4,
            label=SPECIES_NAMES[agent_type],
            alpha=0.8,
        )

    # 绘制综合曲线
    if combined:
        x = sorted(combined.keys())
        y = [combined[gen] for gen in x]

        ax.plot(
            x,
            y,
            color="black",
            linewidth=2.5,
            marker="s",
            markersize=5,
            label="综合平均",
            linestyle="-",
        )

    # 绘制基准线 (y=1.0)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="基准 (1.0)")

    # 设置图表属性
    ax.set_xlabel("代数 (Generation)", fontsize=12)
    ax.set_ylabel("累积进化值 (以第 1 代为 1.0)", fontsize=12)

    # 构建标题，如果 episodes_per_run > 1 则添加说明
    title = "进化曲线图 - 各物种相对于第 1 代的累积进化效果"
    if episodes_per_run is not None and episodes_per_run > 1:
        title += f"（每代测试 {episodes_per_run} 个 episode）"
    ax.set_title(title, fontsize=14)

    # 设置网格
    ax.grid(True, linestyle="-", alpha=0.3)

    # 设置图例
    ax.legend(loc="upper left", fontsize=10)

    # 设置 x 轴刻度
    if len(generations) > 20:
        step = len(generations) // 10
        ax.set_xticks(generations[::step])
    else:
        ax.set_xticks(generations)

    # 调整布局
    plt.tight_layout()

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def print_summary(
    cumulative: dict[AgentType, dict[int, float]],
    combined: dict[int, float],
    _improvements: dict[int, dict[AgentType, float]],
    episodes_per_run: int | None = None,
) -> None:
    """打印进化摘要

    Args:
        cumulative: 各物种的累积进化值
        combined: 综合进化值
        _improvements: 各代各物种的改善百分比（保留参数，未来可能用于详细输出）
        episodes_per_run: 每代测试的 episode 数量
    """
    if not cumulative or not combined:
        print("没有可用的数据")
        return

    generations = sorted(combined.keys())
    min_gen = generations[0]
    max_gen = generations[-1]

    print("=" * 64)
    print("进化曲线摘要")
    print("=" * 64)

    print(f"\n代数范围: 第 {min_gen} 代 ~ 第 {max_gen} 代 (共 {len(generations)} 代)")

    # 显示 episodes_per_run 信息
    if episodes_per_run is not None:
        print(f"每代测试 episode 数: {episodes_per_run}")

    print(f"\n最终累积进化值 (第 {max_gen} 代):")
    for agent_type in AgentType:
        if agent_type in cumulative and max_gen in cumulative[agent_type]:
            value = cumulative[agent_type][max_gen]
            change = (value - 1.0) * 100
            symbol = "↑" if change > 0 else "↓" if change < 0 else "→"
            print(f"  {SPECIES_NAMES[agent_type]:<8} {value:.3f}  ({change:+.1f}%)  {symbol}")

    print(f"  {'综合平均':<8} {combined[max_gen]:.3f}  ({(combined[max_gen] - 1.0) * 100:+.1f}%)")

    # 统计进化趋势
    effective_count = 0
    for agent_type in AgentType:
        if agent_type in cumulative and max_gen in cumulative[agent_type]:
            if cumulative[agent_type][max_gen] > 1.0:
                effective_count += 1

    print(f"\n进化趋势: {effective_count}/4 个物种呈上升趋势")
    print("=" * 64)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="进化曲线可视化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="checkpoints/test_results",
        help="测试结果目录（默认: checkpoints/test_results）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evolution_curve.png",
        help="输出文件路径（默认: outputs/evolution_curve.png）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细信息",
    )

    args = parser.parse_args()

    # 检查目录
    if not Path(args.results_dir).exists():
        print(f"错误: 测试结果目录不存在: {args.results_dir}")
        print("请先运行 python scripts/test_evolution.py 生成测试结果")
        sys.exit(1)

    # 加载数据
    if args.verbose:
        print("加载基准测试结果...")
    baseline_results, episodes_per_run = load_baseline_results(args.results_dir)

    if args.verbose:
        print("加载比较测试结果...")
    comparison_results = load_comparison_results(args.results_dir)

    if not comparison_results:
        print("错误: 未找到比较测试结果")
        print("请先运行 python scripts/test_evolution.py 生成测试结果")
        sys.exit(1)

    if args.verbose:
        print(f"找到 {len(baseline_results)} 代基准测试结果")
        print(f"找到 {len(comparison_results)} 代比较测试结果")

    # 计算相对改善
    if args.verbose:
        print("计算相对改善百分比...")
    improvements = calculate_relative_improvement(comparison_results)

    # 计算累积进化值
    if args.verbose:
        print("计算累积进化值...")
    cumulative = calculate_cumulative_evolution(improvements)

    # 计算综合进化值
    combined = calculate_combined_evolution(cumulative)

    # 打印摘要
    print_summary(cumulative, combined, improvements, episodes_per_run)

    # 绘制图表
    if args.verbose:
        print(f"\n绘制进化曲线图...")
    output_path = plot_evolution_curves(
        cumulative,
        combined,
        args.output,
        args.verbose,
        episodes_per_run,
    )

    if output_path:
        print(f"\n进化曲线图已保存到: {output_path}")


if __name__ == "__main__":
    main()
