#!/usr/bin/env python3
"""网络参数传输方案性能测试

测试比较：
1. 原始并行进化（evolve_parallel）：序列化基因组，反序列化后重建网络
2. 网络参数传输（evolve_parallel_with_network_params）：返回网络参数，直接创建网络
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
    RetailSubPopulationManager,
    _serialize_genomes_numpy,
)


def benchmark_original_parallel(manager: RetailSubPopulationManager, iterations: int = 3) -> float:
    """测试原始并行进化方案"""
    times: list[float] = []

    for i in range(iterations):
        # 模拟一些随机适应度
        for pop in manager.sub_populations:
            for agent in pop.agents:
                agent.brain.get_genome().fitness = np.random.random()

        start = time.perf_counter()
        manager.evolve_parallel(current_price=10000.0, max_workers=10)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  原始并行 iteration {i+1}: {elapsed:.2f}s")

    avg = sum(times) / len(times)
    return avg


def benchmark_network_params_full(
    manager: RetailSubPopulationManager,
    worker_pool: PersistentWorkerPool,
    iterations: int = 3,
) -> float:
    """测试网络参数传输方案（完整反序列化）"""
    times: list[float] = []

    for i in range(iterations):
        # 模拟一些随机适应度
        for pop in manager.sub_populations:
            for agent in pop.agents:
                agent.brain.get_genome().fitness = np.random.random()

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
        print(f"  完整反序列化 iteration {i+1}: {elapsed:.2f}s")

    avg = sum(times) / len(times)
    return avg


def benchmark_network_params_lazy(
    manager: RetailSubPopulationManager,
    worker_pool: PersistentWorkerPool,
    iterations: int = 3,
) -> float:
    """测试网络参数传输方案（延迟反序列化）"""
    times: list[float] = []

    for i in range(iterations):
        # 模拟一些随机适应度
        for pop in manager.sub_populations:
            for agent in pop.agents:
                agent.brain.get_genome().fitness = np.random.random()

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
        print(f"  延迟反序列化 iteration {i+1}: {elapsed:.2f}s")

    avg = sum(times) / len(times)
    return avg


def main() -> None:
    print("=" * 60)
    print("网络参数传输方案性能测试")
    print("=" * 60)

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
    print("\n创建 RetailSubPopulationManager...")
    manager = RetailSubPopulationManager(config, sub_count=sub_count)

    # 获取 NEAT 配置路径
    neat_config_path = manager.sub_populations[0].neat_config_path
    pop_size = len(manager.sub_populations[0].agents)

    print(f"  - NEAT 配置路径: {neat_config_path}")
    print(f"  - 每个 Worker 的种群大小: {pop_size}")

    # 创建 PersistentWorkerPool（放在测试前创建，避免重复创建）
    print("\n创建 PersistentWorkerPool...")
    worker_pool = PersistentWorkerPool(
        num_workers=sub_count,
        neat_config_path=neat_config_path,
        pop_size=pop_size,
    )

    # 测试1：延迟反序列化（最快方案）
    print("\n" + "-" * 40)
    print("测试1: 延迟反序列化 (deserialize_genomes=False)")
    print("-" * 40)
    avg_lazy = benchmark_network_params_lazy(manager, worker_pool, iterations=3)
    print(f"\n延迟反序列化平均耗时: {avg_lazy:.2f}s")

    # 测试2：完整反序列化（对比方案）
    print("\n" + "-" * 40)
    print("测试2: 完整反序列化 (deserialize_genomes=True)")
    print("-" * 40)
    avg_full = benchmark_network_params_full(manager, worker_pool, iterations=3)
    print(f"\n完整反序列化平均耗时: {avg_full:.2f}s")

    # 关闭 Worker 池
    worker_pool.shutdown()

    # 测试3：原始并行进化（基准）
    print("\n" + "-" * 40)
    print("测试3: 原始并行进化 (evolve_parallel)")
    print("-" * 40)
    avg_original = benchmark_original_parallel(manager, iterations=3)
    print(f"\n原始并行平均耗时: {avg_original:.2f}s")

    # 输出对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"原始并行:       {avg_original:.2f}s (基准)")
    print(f"完整反序列化:   {avg_full:.2f}s ({avg_original / avg_full:.2f}x 加速)")
    print(f"延迟反序列化:   {avg_lazy:.2f}s ({avg_original / avg_lazy:.2f}x 加速)")
    print()
    print(f"延迟 vs 完整:   {avg_full - avg_lazy:.2f}s 节省 ({(1 - avg_lazy / avg_full) * 100:.1f}%)")
    print(f"延迟 vs 原始:   {avg_original - avg_lazy:.2f}s 节省 ({(1 - avg_lazy / avg_original) * 100:.1f}%)")


if __name__ == "__main__":
    main()
