#!/usr/bin/env python3
"""进化效果测试脚本

评估指定代数的进化效果，支持基准测试和比较测试。

使用方法:
    # 评估第 N 代进化效果
    python scripts/test_evolution.py --generation 100

    # 列出所有已保存的代
    python scripts/test_evolution.py --list

    # 只运行基准测试
    python scripts/test_evolution.py --generation 100 --baseline-only

    # 强制重新测试（忽略缓存）
    python scripts/test_evolution.py --generation 100 --force

    # 指定测试参数
    python scripts/test_evolution.py --generation 100 --num-runs 5 --episode-length 2000
"""

import argparse
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


def list_generations(generations_dir: str) -> None:
    """列出所有已保存的代"""
    saver = GenerationSaver(generations_dir)
    generations = saver.list_generations()

    if not generations:
        print(f"在 {generations_dir} 中未找到任何已保存的代")
        return

    print("=" * 60)
    print("已保存的代列表")
    print("=" * 60)

    for gen_num in generations:
        gen_data = saver.load_generation(gen_num)
        if gen_data:
            timestamp = gen_data.get("timestamp", 0)
            save_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            species_count = len(gen_data.get("best_genomes", {}))
            print(f"  第 {gen_num:4d} 代  |  {save_time}  |  {species_count} 个物种")

    print("=" * 60)
    print(f"共 {len(generations)} 代")


def print_baseline_result(result: dict[str, Any]) -> None:
    """打印基准测试结果"""
    if "error" in result:
        print(f"\n错误: {result['error']}")
        return

    num_runs = result.get("num_runs", 0)
    species_summary = result.get("species_summary", {})

    print(f"\n基准测试（{num_runs} 次运行）:")

    for agent_type in AgentType:
        data = species_summary.get(agent_type, {})
        if not data:
            continue

        chinese_name = get_chinese_name(agent_type)
        avg_return = data.get("avg_return_rate", 0.0) * 100
        std_return = data.get("std_return_rate", 0.0) * 100
        survival_rate = data.get("avg_survival_rate", 0.0) * 100

        print(
            f"  {chinese_name:<8} 平均收益率: {avg_return:+6.1f}% (±{std_return:.1f}%)  "
            f"存活率: {survival_rate:.1f}%"
        )


def print_evaluation_result(result: dict[str, Any]) -> None:
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
    if baseline:
        print_baseline_result(baseline)

    # 打印比较测试结果
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


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="进化效果测试脚本 - 评估指定代数的进化效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--generation",
        type=int,
        help="要测试的代数",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="只运行基准测试（跳过比较测试）",
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
        help="列出所有已保存的代",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新运行（忽略缓存）",
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
        list_generations(args.generations_dir)
        return

    # 验证必要参数
    if args.generation is None:
        parser.error("必须指定 --generation 或 --list")

    # 设置日志
    setup_logging(args.log_dir)

    # 检查代数是否存在
    saver = GenerationSaver(args.generations_dir)
    if not saver.generation_exists(args.generation):
        print(f"错误: 第 {args.generation} 代不存在")
        print(f"请使用 --list 查看已保存的代列表")
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

    # 打印测试信息
    print("=" * 64)
    print(f"进化效果测试 - 第 {args.generation} 代")
    print("=" * 64)

    if args.baseline_only:
        # 只运行基准测试
        result = tester.run_baseline_test(
            generation=args.generation,
            num_runs=args.num_runs,
            episode_length=args.episode_length,
            force=args.force,
        )
        print_baseline_result(result)
    elif args.generation <= 1:
        # 第 1 代没有上一代可以比较，只运行基准测试
        result = tester.run_baseline_test(
            generation=args.generation,
            num_runs=args.num_runs,
            episode_length=args.episode_length,
            force=args.force,
        )
        print_baseline_result(result)
        print("\n(跳过比较测试，因为是第 1 代，没有上一代可以比较)")
    else:
        # 运行完整评估
        result = tester.evaluate_evolution_effectiveness(
            generation=args.generation,
            num_runs=args.num_runs,
            episode_length=args.episode_length,
            force=args.force,
        )
        print_evaluation_result(result)

    print("=" * 64)


if __name__ == "__main__":
    main()
