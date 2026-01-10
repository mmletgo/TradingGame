#!/usr/bin/env python3
"""详细性能剖析脚本 - 定位具体瓶颈"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from scripts.create_config import create_default_config
from src.config.config import AgentType
from src.training.population import (
    PersistentWorkerPool,
    SubPopulationManager,
    _serialize_genomes_numpy,
    _unpack_network_params_numpy,
)


def profile_serial_evolve(manager: SubPopulationManager) -> None:
    """剖析串行进化的各阶段耗时"""
    print("\n" + "=" * 60)
    print("串行进化详细剖析 (单个子种群)")
    print("=" * 60)

    pop = manager.sub_populations[0]

    # 设置随机适应度
    for agent in pop.agents:
        agent.brain.get_genome().fitness = np.random.random()

    # 阶段1: evaluate
    start = time.perf_counter()
    agent_fitnesses = pop.evaluate(10000.0)
    eval_time = time.perf_counter() - start
    print(f"  evaluate(): {eval_time:.3f}s")

    # 设置适应度到基因组
    start = time.perf_counter()
    for agent, fitness in agent_fitnesses:
        genome = agent.brain.get_genome()
        genome.fitness = fitness
    set_fitness_time = time.perf_counter() - start
    print(f"  设置适应度: {set_fitness_time:.3f}s")

    # 阶段2: NEAT进化
    import neat
    from neat.reporting import BaseReporter

    def dummy_eval(genomes, config):
        pass

    start = time.perf_counter()
    pop.neat_pop.run(dummy_eval, n=1)
    neat_time = time.perf_counter() - start
    print(f"  NEAT进化: {neat_time:.3f}s")

    # 阶段3: 清理历史
    start = time.perf_counter()
    pop._cleanup_neat_history()
    cleanup_time = time.perf_counter() - start
    print(f"  清理历史: {cleanup_time:.3f}s")

    # 阶段4: 创建新Agent
    start = time.perf_counter()
    new_genomes = list(pop.neat_pop.population.items())
    pop.agents = pop.create_agents(new_genomes)
    create_time = time.perf_counter() - start
    print(f"  创建Agent: {create_time:.3f}s")

    total = eval_time + set_fitness_time + neat_time + cleanup_time + create_time
    print(f"\n  总计: {total:.3f}s")


def profile_worker_pool_evolve(manager: SubPopulationManager) -> None:
    """剖析Worker池进化的各阶段耗时"""
    print("\n" + "=" * 60)
    print("Worker池+延迟反序列化 详细剖析")
    print("=" * 60)

    neat_config_path = manager.sub_populations[0].neat_config_path
    pop_size = len(manager.sub_populations[0].agents)

    # 创建Worker池
    print("\n1. 创建Worker池...")
    start = time.perf_counter()
    worker_pool = PersistentWorkerPool(
        num_workers=len(manager.sub_populations),
        neat_config_path=neat_config_path,
        pop_size=pop_size,
    )
    pool_create_time = time.perf_counter() - start
    print(f"   耗时: {pool_create_time:.3f}s")

    # 同步基因组
    print("\n2. 同步基因组到Worker...")
    start = time.perf_counter()
    genomes_list = []
    for pop in manager.sub_populations:
        genome_data = _serialize_genomes_numpy(pop.neat_pop.population)
        genomes_list.append(genome_data)
    serialize_time = time.perf_counter() - start
    print(f"   序列化耗时: {serialize_time:.3f}s")

    # 计算序列化数据大小
    total_size = 0
    for gd in genomes_list:
        for arr in gd:
            total_size += arr.nbytes
    print(f"   序列化数据大小: {total_size / 1024 / 1024:.2f} MB")

    start = time.perf_counter()
    worker_pool.set_all_genomes(genomes_list)
    transfer_time = time.perf_counter() - start
    print(f"   传输到Worker耗时: {transfer_time:.3f}s")

    # 设置随机适应度
    for pop in manager.sub_populations:
        for agent in pop.agents:
            agent.brain.get_genome().fitness = np.random.random()

    # 评估并构建适应度数组
    print("\n3. 评估并构建适应度数组...")
    start = time.perf_counter()
    fitnesses_list = []
    for pop in manager.sub_populations:
        agent_fitnesses = pop.evaluate(10000.0)
        fitness_dict = {agent.agent_id: fitness for agent, fitness in agent_fitnesses}
        fitness_arr = np.empty(len(pop.agents), dtype=np.float32)
        for idx, agent in enumerate(pop.agents):
            fitness = fitness_dict.get(agent.agent_id, 0.0)
            fitness_arr[idx] = fitness
        fitnesses_list.append(fitness_arr)
    eval_time = time.perf_counter() - start
    print(f"   耗时: {eval_time:.3f}s")

    # Worker进化
    print("\n4. Worker并行进化...")
    start = time.perf_counter()
    results = worker_pool.evolve_all_return_params(fitnesses_list)
    evolve_time = time.perf_counter() - start
    print(f"   耗时: {evolve_time:.3f}s")

    # 解包网络参数
    print("\n5. 解包网络参数并更新Agent...")
    start = time.perf_counter()
    for i, (genome_data, network_params_data) in enumerate(results):
        pop = manager.sub_populations[i]
        params_list = _unpack_network_params_numpy(*network_params_data)
        for idx, params in enumerate(params_list):
            if idx < len(pop.agents):
                pop.agents[idx].brain.update_network_only(params)
    update_time = time.perf_counter() - start
    print(f"   耗时: {update_time:.3f}s")

    # 第二次迭代（不需要同步）
    print("\n" + "-" * 40)
    print("第二次迭代（无需同步）")
    print("-" * 40)

    for pop in manager.sub_populations:
        for agent in pop.agents:
            agent.brain.get_genome().fitness = np.random.random()

    start = time.perf_counter()
    fitnesses_list = []
    for pop in manager.sub_populations:
        agent_fitnesses = pop.evaluate(10000.0)
        fitness_dict = {agent.agent_id: fitness for agent, fitness in agent_fitnesses}
        fitness_arr = np.empty(len(pop.agents), dtype=np.float32)
        for idx, agent in enumerate(pop.agents):
            fitness = fitness_dict.get(agent.agent_id, 0.0)
            fitness_arr[idx] = fitness
        fitnesses_list.append(fitness_arr)
    eval_time2 = time.perf_counter() - start
    print(f"   评估耗时: {eval_time2:.3f}s")

    start = time.perf_counter()
    results = worker_pool.evolve_all_return_params(fitnesses_list)
    evolve_time2 = time.perf_counter() - start
    print(f"   Worker进化耗时: {evolve_time2:.3f}s")

    start = time.perf_counter()
    for i, (genome_data, network_params_data) in enumerate(results):
        pop = manager.sub_populations[i]
        params_list = _unpack_network_params_numpy(*network_params_data)
        for idx, params in enumerate(params_list):
            if idx < len(pop.agents):
                pop.agents[idx].brain.update_network_only(params)
    update_time2 = time.perf_counter() - start
    print(f"   更新Agent耗时: {update_time2:.3f}s")

    total2 = eval_time2 + evolve_time2 + update_time2
    print(f"\n   第二次迭代总计: {total2:.3f}s")

    worker_pool.shutdown()

    # 总结
    print("\n" + "=" * 60)
    print("瓶颈分析总结")
    print("=" * 60)

    print(f"""
首次调用耗时分解:
  - Worker池创建:      {pool_create_time:.3f}s
  - 基因组序列化:       {serialize_time:.3f}s
  - 基因组传输:         {transfer_time:.3f}s
  - 评估适应度:         {eval_time:.3f}s
  - Worker进化:        {evolve_time:.3f}s
  - 更新Agent:         {update_time:.3f}s

后续调用耗时分解:
  - 评估适应度:         {eval_time2:.3f}s
  - Worker进化:        {evolve_time2:.3f}s
  - 更新Agent:         {update_time2:.3f}s

主要瓶颈:
  1. 基因组序列化+传输: {serialize_time + transfer_time:.3f}s (首次调用)
  2. Worker进化:       {evolve_time:.3f}s / {evolve_time2:.3f}s
  3. 更新Agent:        {update_time:.3f}s / {update_time2:.3f}s
""")


def main() -> None:
    print("=" * 70)
    print("RETAIL子种群进化 - 详细性能剖析")
    print("=" * 70)

    # 创建配置
    config = create_default_config()
    config.agents[AgentType.RETAIL].count = 10000
    sub_count = 10

    print(f"\n配置: {sub_count} 个子种群 × {config.agents[AgentType.RETAIL].count // sub_count} Agent")

    # 创建管理器
    print("\n创建 SubPopulationManager...")
    start = time.perf_counter()
    manager = SubPopulationManager(config, AgentType.RETAIL, sub_count=sub_count)
    create_time = time.perf_counter() - start
    print(f"创建耗时: {create_time:.2f}s")

    # 剖析串行进化
    profile_serial_evolve(manager)

    # 剖析Worker池进化
    profile_worker_pool_evolve(manager)


if __name__ == "__main__":
    main()
