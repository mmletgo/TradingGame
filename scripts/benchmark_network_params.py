#!/usr/bin/env python3
"""RETAIL子种群进化方案完整性能测试

测试比较所有可用的进化方法：
1. evolve() - 串行进化（最基本的 baseline）
2. evolve_parallel_simple() - 推荐的简化并行进化（ThreadPoolExecutor + update_brain）
3. evolve_parallel() - 进程池并行进化（ProcessPoolExecutor）
4. evolve_parallel_with_network_params() - 持久Worker池 + 网络参数传输
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from scripts.create_config import create_default_config
from src.config.config import AgentType
from src.training.population import (
    PersistentWorkerPool,
    SubPopulationManager,
)


def set_random_fitness(manager: SubPopulationManager) -> None:
    """为所有Agent设置随机适应度"""
    for pop in manager.sub_populations:
        for agent in pop.agents:
            agent.brain.get_genome().fitness = np.random.random()


def benchmark_serial(manager: SubPopulationManager, iterations: int = 3) -> float:
    """测试串行进化方案（最基本的baseline）"""
    times: list[float] = []

    for i in range(iterations):
        set_random_fitness(manager)

        start = time.perf_counter()
        manager.evolve(current_price=10000.0)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  串行进化 iteration {i+1}: {elapsed:.2f}s")

    avg = sum(times) / len(times)
    return avg


def benchmark_parallel_simple(manager: SubPopulationManager, iterations: int = 3) -> float:
    """测试简化并行进化方案（推荐方案）"""
    times: list[float] = []

    for i in range(iterations):
        set_random_fitness(manager)

        start = time.perf_counter()
        manager.evolve_parallel_simple(current_price=10000.0, max_workers=10)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  简化并行 iteration {i+1}: {elapsed:.2f}s")

    avg = sum(times) / len(times)
    return avg


def benchmark_original_parallel(manager: SubPopulationManager, iterations: int = 3) -> float:
    """测试原始进程池并行进化方案"""
    times: list[float] = []

    for i in range(iterations):
        set_random_fitness(manager)

        start = time.perf_counter()
        manager.evolve_parallel(current_price=10000.0, max_workers=10)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  进程池并行 iteration {i+1}: {elapsed:.2f}s")

    avg = sum(times) / len(times)
    return avg


def benchmark_network_params_full(
    manager: SubPopulationManager,
    worker_pool: PersistentWorkerPool,
    iterations: int = 3,
) -> float:
    """测试网络参数传输方案（完整反序列化）"""
    times: list[float] = []

    for i in range(iterations):
        set_random_fitness(manager)

        start = time.perf_counter()
        # 首次调用需要同步基因组
        sync_needed = (i == 0)
        manager.evolve_parallel_with_network_params(
            current_price=10000.0,
            worker_pool=worker_pool,
            sync_genomes=sync_needed,
            deserialize_genomes=True,  # 完整反序列化
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Worker池+完整反序列化 iteration {i+1}: {elapsed:.2f}s")

    avg = sum(times) / len(times)
    return avg


def benchmark_network_params_lazy(
    manager: SubPopulationManager,
    worker_pool: PersistentWorkerPool,
    iterations: int = 3,
) -> float:
    """测试网络参数传输方案（延迟反序列化）"""
    times: list[float] = []

    for i in range(iterations):
        set_random_fitness(manager)

        start = time.perf_counter()
        # 首次调用需要同步基因组
        sync_needed = (i == 0)
        manager.evolve_parallel_with_network_params(
            current_price=10000.0,
            worker_pool=worker_pool,
            sync_genomes=sync_needed,
            deserialize_genomes=False,  # 延迟反序列化
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Worker池+延迟反序列化 iteration {i+1}: {elapsed:.2f}s")

    avg = sum(times) / len(times)
    return avg


def main() -> None:
    print("=" * 70)
    print("RETAIL子种群进化方案完整性能测试")
    print("=" * 70)

    # 创建配置
    config = create_default_config()

    # 使用完整规模：10个子种群，每个1000个Agent，共10000个
    config.agents[AgentType.RETAIL].count = 10000  # 总共10000个
    sub_count = 10  # 10个子种群，每个1000个

    agents_per_sub = config.agents[AgentType.RETAIL].count // sub_count

    print(f"\n配置：")
    print(f"  - 子种群数量: {sub_count}")
    print(f"  - 每个子种群 Agent 数: {agents_per_sub}")
    print(f"  - 总 Agent 数: {config.agents[AgentType.RETAIL].count}")

    # 创建子种群管理器
    print("\n创建 SubPopulationManager...")
    create_start = time.perf_counter()
    manager = SubPopulationManager(config, AgentType.RETAIL, sub_count=sub_count)
    create_time = time.perf_counter() - create_start
    print(f"  创建耗时: {create_time:.2f}s")

    # 获取 NEAT 配置路径
    neat_config_path = manager.sub_populations[0].neat_config_path
    pop_size = len(manager.sub_populations[0].agents)

    print(f"  - NEAT 配置路径: {neat_config_path}")
    print(f"  - 每个 Worker 的种群大小: {pop_size}")

    results: dict[str, float] = {}

    # 测试1：串行进化（baseline）
    print("\n" + "-" * 50)
    print("测试1: 串行进化 (evolve) - Baseline")
    print("-" * 50)
    avg_serial = benchmark_serial(manager, iterations=3)
    results["串行进化"] = avg_serial
    print(f"\n串行进化平均耗时: {avg_serial:.2f}s")

    # 测试2：简化并行进化（推荐方案）
    print("\n" + "-" * 50)
    print("测试2: 简化并行进化 (evolve_parallel_simple) - 推荐")
    print("-" * 50)
    avg_simple = benchmark_parallel_simple(manager, iterations=3)
    results["简化并行"] = avg_simple
    print(f"\n简化并行平均耗时: {avg_simple:.2f}s")

    # 测试3：进程池并行进化
    print("\n" + "-" * 50)
    print("测试3: 进程池并行进化 (evolve_parallel)")
    print("-" * 50)
    avg_process = benchmark_original_parallel(manager, iterations=3)
    results["进程池并行"] = avg_process
    print(f"\n进程池并行平均耗时: {avg_process:.2f}s")

    # 创建 PersistentWorkerPool
    print("\n创建 PersistentWorkerPool...")
    pool_start = time.perf_counter()
    worker_pool = PersistentWorkerPool(
        num_workers=sub_count,
        neat_config_path=neat_config_path,
        pop_size=pop_size,
    )
    pool_time = time.perf_counter() - pool_start
    print(f"  Worker池创建耗时: {pool_time:.2f}s")

    # 测试4：Worker池 + 延迟反序列化
    print("\n" + "-" * 50)
    print("测试4: Worker池 + 延迟反序列化")
    print("-" * 50)
    avg_lazy = benchmark_network_params_lazy(manager, worker_pool, iterations=3)
    results["Worker池+延迟"] = avg_lazy
    print(f"\nWorker池+延迟反序列化平均耗时: {avg_lazy:.2f}s")

    # 测试5：Worker池 + 完整反序列化
    print("\n" + "-" * 50)
    print("测试5: Worker池 + 完整反序列化")
    print("-" * 50)
    avg_full = benchmark_network_params_full(manager, worker_pool, iterations=3)
    results["Worker池+完整"] = avg_full
    print(f"\nWorker池+完整反序列化平均耗时: {avg_full:.2f}s")

    # 关闭 Worker 池
    worker_pool.shutdown()

    # 输出对比结果
    print("\n" + "=" * 70)
    print("对比结果 (以串行进化为基准)")
    print("=" * 70)

    baseline = results["串行进化"]

    # 按耗时排序
    sorted_results = sorted(results.items(), key=lambda x: x[1])

    print(f"\n{'方案':<25} {'耗时(s)':<12} {'加速比':<12} {'节省时间':<12}")
    print("-" * 65)

    for name, time_val in sorted_results:
        speedup = baseline / time_val if time_val > 0 else float('inf')
        saved = baseline - time_val
        saved_pct = (saved / baseline) * 100 if baseline > 0 else 0

        if name == "串行进化":
            print(f"{name:<25} {time_val:>8.2f}s    {'(基准)':<12} {'-':<12}")
        else:
            speedup_str = f"{speedup:.2f}x"
            saved_str = f"{saved:+.2f}s ({saved_pct:+.1f}%)"
            print(f"{name:<25} {time_val:>8.2f}s    {speedup_str:<12} {saved_str:<12}")

    # 找出最优方案
    best_name, best_time = sorted_results[0]
    print(f"\n最优方案: {best_name} ({best_time:.2f}s)")

    # 瓶颈分析建议
    print("\n" + "=" * 70)
    print("瓶颈分析")
    print("=" * 70)

    print("""
主要耗时来源分析:
1. NEAT进化算法本身 - 物种划分、选择、交叉、变异
2. Brain重建/更新 - 从基因组创建神经网络
3. 序列化/反序列化 - 跨进程传输基因组数据
4. 线程/进程调度开销

优化建议:
- 如果"简化并行"最快 → GIL不是主要瓶颈，线程池开销最小
- 如果"进程池并行"最快 → 序列化开销可接受，真正的多核并行
- 如果"Worker池+延迟"最快 → 持久进程避免了启动开销
- 如果"串行进化"最快 → 并行化开销超过了收益，建议优化单线程性能
""")


if __name__ == "__main__":
    main()
