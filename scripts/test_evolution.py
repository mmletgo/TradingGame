#!/usr/bin/env python3
"""进化效果测试脚本

自动检测并测试所有未完成的代，评估进化效果。

使用方法:
    # 自动测试所有未完成的代（默认行为）
    python scripts/test_evolution.py

    # 测试指定代
    python scripts/test_evolution.py --generation 100

    # 列出所有代及其测试状态
    python scripts/test_evolution.py --list

    # 强制重新测试所有代
    python scripts/test_evolution.py --force

    # 指定测试参数
    python scripts/test_evolution.py --num-runs 5 --episode-length 2000
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.evolution_tester import EvolutionTester
from src.config.config import AgentType
from src.core.log_engine.logger import setup_logging
from src.training.generation_saver import GenerationSaver

from create_config import create_default_config

# 物种中文名称映射
AGENT_TYPE_NAMES: dict[str, str] = {
    "retail": "散户",
    "retail_pro": "高级散户",
    "whale": "庄家",
    "market_maker": "做市商",
}


def get_chinese_name(agent_type: AgentType) -> str:
    """获取物种中文名称"""
    return AGENT_TYPE_NAMES.get(agent_type.value.lower(), agent_type.value)


def check_test_status(
    generation: int,
    results_dir: str,
) -> dict[str, bool]:
    """检查某代的测试完成状态

    Args:
        generation: 代数
        results_dir: 测试结果保存目录

    Returns:
        测试状态字典：
        - baseline: 基准测试是否完成
        - comparisons: 各物种比较测试是否完成
        - complete: 是否全部完成
    """
    results_path = Path(results_dir)

    # 检查基准测试
    baseline_path = results_path / "baseline" / f"gen_{generation}.pkl"
    baseline_done = baseline_path.exists()

    # 第 1 代不需要比较测试
    if generation <= 1:
        return {
            "baseline": baseline_done,
            "comparisons": {},
            "complete": baseline_done,
        }

    # 检查比较测试
    base_generation = generation - 1
    comparisons: dict[str, bool] = {}
    all_comparisons_done = True

    for agent_type in AgentType:
        comparison_path = (
            results_path / "comparison"
            / f"gen_{generation}_vs_gen_{base_generation}_{agent_type.value}.pkl"
        )
        done = comparison_path.exists()
        comparisons[agent_type.value] = done
        if not done:
            all_comparisons_done = False

    return {
        "baseline": baseline_done,
        "comparisons": comparisons,
        "complete": baseline_done and all_comparisons_done,
    }


def get_testable_generations(
    generations_dir: str,
    results_dir: str,
    force: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """获取可测试的代列表

    Args:
        generations_dir: 代数据保存目录
        results_dir: 测试结果保存目录
        force: 是否强制重新测试

    Returns:
        (所有代列表, 未完成测试的代列表, 已完成测试的代列表)
    """
    saver = GenerationSaver(generations_dir)
    all_generations = saver.list_generations()

    if force:
        return all_generations, all_generations, []

    pending: list[int] = []
    completed: list[int] = []

    for gen in all_generations:
        status = check_test_status(gen, results_dir)
        if status["complete"]:
            completed.append(gen)
        else:
            pending.append(gen)

    return all_generations, pending, completed


def list_generations_with_status(
    generations_dir: str,
    results_dir: str,
) -> None:
    """列出所有代及其测试状态"""
    saver = GenerationSaver(generations_dir)
    generations = saver.list_generations()

    if not generations:
        print(f"在 {generations_dir} 中未找到任何已保存的代")
        return

    print("=" * 80)
    print("代列表及测试状态")
    print("=" * 80)
    print(f"{'代数':>6} | {'保存时间':^20} | {'物种数':>6} | {'测试状态':^20}")
    print("-" * 80)

    completed_count = 0
    pending_count = 0

    for gen_num in generations:
        gen_data = saver.load_generation(gen_num)
        if gen_data:
            timestamp = gen_data.get("timestamp", 0)
            save_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            species_count = len(gen_data.get("best_genomes", {}))

            status = check_test_status(gen_num, results_dir)
            if status["complete"]:
                status_str = "✓ 已完成"
                completed_count += 1
            elif status["baseline"]:
                # 部分完成
                done_comparisons = sum(1 for v in status["comparisons"].values() if v)
                total_comparisons = len(status["comparisons"])
                status_str = f"◐ 部分完成 ({done_comparisons}/{total_comparisons})"
                pending_count += 1
            else:
                status_str = "○ 未测试"
                pending_count += 1

            print(f"{gen_num:>6} | {save_time:^20} | {species_count:>6} | {status_str:<20}")

    print("=" * 80)
    print(f"共 {len(generations)} 代: {completed_count} 已完成, {pending_count} 待测试")


def print_baseline_result(result: dict[str, Any], verbose: bool = True) -> None:
    """打印基准测试结果"""
    if "error" in result:
        print(f"\n错误: {result['error']}")
        return

    num_runs = result.get("num_runs", 0)
    species_summary = result.get("species_summary", {})

    if verbose:
        print(f"\n基准测试（{num_runs} 次运行）:")

    for agent_type in AgentType:
        data = species_summary.get(agent_type, {})
        if not data:
            continue

        chinese_name = get_chinese_name(agent_type)
        avg_return = data.get("avg_return_rate", 0.0) * 100
        std_return = data.get("std_return_rate", 0.0) * 100
        survival_rate = data.get("avg_survival_rate", 0.0) * 100

        if verbose:
            print(
                f"  {chinese_name:<8} 平均收益率: {avg_return:+6.1f}% (±{std_return:.1f}%)  "
                f"存活率: {survival_rate:.1f}%"
            )


def print_evaluation_result(result: dict[str, Any], verbose: bool = True) -> None:
    """打印评估结果"""
    if "error" in result:
        print(f"\n错误: {result['error']}")
        return

    generation = result.get("generation", 0)
    base_generation = result.get("base_generation", 0)
    effectiveness = result.get("effectiveness", {})
    summary = result.get("summary", {})

    # 先打印基准测试结果
    baseline = result.get("baseline", {})
    if baseline and verbose:
        print_baseline_result(baseline)

    # 打印比较测试结果
    if verbose:
        print(f"\n比较测试（新物种 vs 第 {base_generation} 代对手）:")

    for agent_type in AgentType:
        eff = effectiveness.get(agent_type, {})
        if not eff:
            continue

        chinese_name = get_chinese_name(agent_type)
        baseline_return = eff.get("baseline_return_rate", 0.0) * 100
        comparison_return = eff.get("comparison_return_rate", 0.0) * 100
        relative_improvement = eff.get("relative_improvement_pct", 0.0)
        is_effective = eff.get("is_effective", False)

        symbol = "↑" if is_effective else "↓"
        status = "有效" if is_effective else "无效"

        if verbose:
            print(
                f"  {chinese_name:<8} 收益率: {comparison_return:+6.1f}% "
                f"(基准: {baseline_return:+6.1f}%)  "
                f"变化: {relative_improvement:+6.1f}%  {symbol} {status}"
            )

    # 打印总结
    effective_species = summary.get("effective_species", [])
    ineffective_species = summary.get("ineffective_species", [])
    print(
        f"\n总结: {len(effective_species)}/{len(effective_species) + len(ineffective_species)} "
        f"个物种进化有效"
    )


def test_single_generation(
    tester: EvolutionTester,
    generation: int,
    num_runs: int,
    episode_length: int,
    force: bool,
    verbose: bool = True,
) -> dict[str, Any] | None:
    """测试单个代

    Args:
        tester: EvolutionTester 实例
        generation: 代数
        num_runs: 测试运行次数
        episode_length: 每次测试的 tick 数
        force: 是否强制重新运行
        verbose: 是否详细输出

    Returns:
        测试结果，如果出错返回 None
    """
    if verbose:
        print("=" * 64)
        print(f"进化效果测试 - 第 {generation} 代")
        print("=" * 64)

    if generation <= 1:
        # 第 1 代没有上一代可以比较，只运行基准测试
        result = tester.run_baseline_test(
            generation=generation,
            num_runs=num_runs,
            episode_length=episode_length,
            force=force,
        )
        if verbose:
            print_baseline_result(result)
            print("\n(跳过比较测试，因为是第 1 代，没有上一代可以比较)")
    else:
        # 运行完整评估
        result = tester.evaluate_evolution_effectiveness(
            generation=generation,
            num_runs=num_runs,
            episode_length=episode_length,
            force=force,
        )
        if verbose:
            print_evaluation_result(result)

    if verbose:
        print("=" * 64)

    return result


def test_all_pending_generations(
    tester: EvolutionTester,
    pending_generations: list[int],
    num_runs: int,
    episode_length: int,
    force: bool,
) -> None:
    """测试所有待测试的代

    Args:
        tester: EvolutionTester 实例
        pending_generations: 待测试的代列表
        num_runs: 测试运行次数
        episode_length: 每次测试的 tick 数
        force: 是否强制重新运行
    """
    total = len(pending_generations)

    print("=" * 64)
    print(f"批量进化效果测试 - 共 {total} 代待测试")
    print("=" * 64)

    successful = 0
    failed = 0

    for i, generation in enumerate(pending_generations, 1):
        print(f"\n[{i}/{total}] 测试第 {generation} 代...")
        print("-" * 64)

        try:
            result = test_single_generation(
                tester=tester,
                generation=generation,
                num_runs=num_runs,
                episode_length=episode_length,
                force=force,
                verbose=True,
            )

            if result and "error" not in result:
                successful += 1
            else:
                failed += 1

        except Exception as e:
            print(f"测试第 {generation} 代时出错: {e}")
            failed += 1

    # 打印汇总
    print("\n" + "=" * 64)
    print("批量测试完成")
    print("=" * 64)
    print(f"成功: {successful} 代")
    print(f"失败: {failed} 代")
    print(f"总计: {total} 代")


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="进化效果测试脚本 - 自动测试所有未完成的代",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--generation",
        type=int,
        help="只测试指定代（不指定则测试所有未完成的代）",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="测试运行次数（默认: 3）",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=1000,
        help="每次测试的 tick 数（默认: 1000）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_generations",
        help="列出所有代及其测试状态",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新运行（忽略已完成的测试结果）",
    )
    parser.add_argument(
        "--generations-dir",
        type=str,
        default="checkpoints/generations",
        help="代数据保存目录（默认: checkpoints/generations）",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="checkpoints/test_results",
        help="测试结果保存目录（默认: checkpoints/test_results）",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="配置文件目录（默认: config）",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="日志目录（默认: logs）",
    )

    args = parser.parse_args()

    # 处理 --list 参数
    if args.list_generations:
        list_generations_with_status(args.generations_dir, args.results_dir)
        return

    # 设置日志
    setup_logging(args.log_dir)

    # 检查 generations 目录
    saver = GenerationSaver(args.generations_dir)
    all_generations = saver.list_generations()

    if not all_generations:
        print(f"在 {args.generations_dir} 中未找到任何已保存的代")
        print("请先运行训练以生成代数据")
        sys.exit(1)

    # 创建测试专用配置（禁用鲶鱼，不保存 checkpoint）
    config = create_default_config(
        episode_length=args.episode_length,
        checkpoint_interval=0,
        config_dir=args.config_dir,
        catfish_enabled=False,
    )

    # 创建测试器
    tester = EvolutionTester(
        config=config,
        generations_dir=args.generations_dir,
        results_dir=args.results_dir,
    )

    if args.generation is not None:
        # 测试指定代
        if not saver.generation_exists(args.generation):
            print(f"错误: 第 {args.generation} 代不存在")
            print(f"请使用 --list 查看已保存的代列表")
            sys.exit(1)

        test_single_generation(
            tester=tester,
            generation=args.generation,
            num_runs=args.num_runs,
            episode_length=args.episode_length,
            force=args.force,
            verbose=True,
        )
    else:
        # 自动测试所有未完成的代
        _, pending, completed = get_testable_generations(
            args.generations_dir,
            args.results_dir,
            force=args.force,
        )

        if not pending:
            print("所有代都已完成测试！")
            print(f"已完成: {len(completed)} 代")
            print("使用 --force 强制重新测试，或使用 --list 查看详细状态")
            return

        print(f"发现 {len(pending)} 代待测试（已跳过 {len(completed)} 个已完成的代）")
        print(f"待测试的代: {pending}")
        print()

        test_all_pending_generations(
            tester=tester,
            pending_generations=pending,
            num_runs=args.num_runs,
            episode_length=args.episode_length,
            force=args.force,
        )


if __name__ == "__main__":
    main()
