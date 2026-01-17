#!/usr/bin/env python3
"""
内存泄漏诊断脚本 - 检查多竞技场训练模式的内存增长

运行方式: python scripts/diagnose_memory.py
"""
import gc
import sys
import tracemalloc
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psutil
import neat


def get_memory_mb() -> float:
    """获取当前进程的内存使用量 (MB)"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def analyze_neat_population(neat_pop: neat.Population, label: str = "") -> dict:
    """分析 NEAT 种群的内存占用来源"""
    stats = {}

    # 1. 检查 population 字典
    stats['population_size'] = len(neat_pop.population)

    # 2. 检查 species
    if hasattr(neat_pop, 'species') and neat_pop.species is not None:
        species_set = neat_pop.species

        # genome_to_species 字典
        if hasattr(species_set, 'genome_to_species'):
            stats['genome_to_species_size'] = len(species_set.genome_to_species)

        # species 字典
        if hasattr(species_set, 'species'):
            stats['species_count'] = len(species_set.species)
            total_members = 0
            total_fitness_history = 0
            for sid, s in species_set.species.items():
                if hasattr(s, 'members') and s.members:
                    total_members += len(s.members)
                if hasattr(s, 'fitness_history'):
                    total_fitness_history += len(s.fitness_history)
            stats['species_total_members'] = total_members
            stats['species_total_fitness_history'] = total_fitness_history

    # 3. 检查 reproduction.ancestors
    if hasattr(neat_pop, 'reproduction') and neat_pop.reproduction is not None:
        reproduction = neat_pop.reproduction
        if hasattr(reproduction, 'ancestors'):
            stats['ancestors_size'] = len(reproduction.ancestors)
        if hasattr(reproduction, 'innovation_tracker'):
            tracker = reproduction.innovation_tracker
            if hasattr(tracker, 'generation_innovations'):
                stats['generation_innovations_size'] = len(tracker.generation_innovations)
            if hasattr(tracker, 'global_counter'):
                stats['innovation_counter'] = tracker.global_counter

    # 4. 检查 stagnation
    if hasattr(neat_pop, 'stagnation') and neat_pop.stagnation is not None:
        stagnation = neat_pop.stagnation
        if hasattr(stagnation, 'species_fitness'):
            stats['stagnation_species_fitness_size'] = len(stagnation.species_fitness)

    # 5. 检查 best_genome
    if hasattr(neat_pop, 'best_genome') and neat_pop.best_genome is not None:
        stats['has_best_genome'] = True
        if neat_pop.best_genome.key in neat_pop.population:
            stats['best_genome_in_population'] = True
        else:
            stats['best_genome_in_population'] = False
    else:
        stats['has_best_genome'] = False

    # 6. 检查 reporters
    if hasattr(neat_pop, 'reporters') and neat_pop.reporters is not None:
        reporters = neat_pop.reporters
        if hasattr(reporters, 'reporters'):
            stats['reporters_count'] = len(reporters.reporters)
            for i, reporter in enumerate(reporters.reporters):
                if hasattr(reporter, 'generation_statistics'):
                    stats[f'reporter_{i}_gen_stats'] = len(reporter.generation_statistics)
                if hasattr(reporter, 'species_statistics'):
                    stats[f'reporter_{i}_species_stats'] = len(reporter.species_statistics)
                if hasattr(reporter, 'most_fit_genomes'):
                    stats[f'reporter_{i}_most_fit_genomes'] = len(reporter.most_fit_genomes)

    if label:
        print(f"\n=== NEAT 种群分析 ({label}) ===")
    else:
        print(f"\n=== NEAT 种群分析 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats


def check_genome_connections_count(neat_pop: neat.Population) -> dict:
    """检查基因组的连接数量"""
    conn_counts = []
    node_counts = []
    for genome in neat_pop.population.values():
        conn_counts.append(len(genome.connections))
        node_counts.append(len(genome.nodes))

    if conn_counts:
        avg_conns = sum(conn_counts) / len(conn_counts)
        max_conns = max(conn_counts)
        min_conns = min(conn_counts)
    else:
        avg_conns = max_conns = min_conns = 0

    if node_counts:
        avg_nodes = sum(node_counts) / len(node_counts)
        max_nodes = max(node_counts)
    else:
        avg_nodes = max_nodes = 0

    print(f"\n=== 基因组结构统计 ===")
    print(f"  平均连接数: {avg_conns:.1f}")
    print(f"  最大连接数: {max_conns}")
    print(f"  最小连接数: {min_conns}")
    print(f"  平均节点数: {avg_nodes:.1f}")
    print(f"  最大节点数: {max_nodes}")

    return {
        'avg_connections': avg_conns,
        'max_connections': max_conns,
        'avg_nodes': avg_nodes,
        'max_nodes': max_nodes,
    }


def run_memory_test():
    """运行内存泄漏测试"""
    from src.config.config import Config, AgentType
    from src.training.arena import ParallelArenaTrainer, MultiArenaConfig

    print("=" * 60)
    print("内存泄漏诊断脚本")
    print("=" * 60)

    # 启用 tracemalloc
    tracemalloc.start()

    initial_mem = get_memory_mb()
    print(f"\n初始内存: {initial_mem:.1f} MB")

    # 创建配置
    from scripts.create_config import create_default_config
    config = create_default_config(
        episode_length=10,  # 短 episode
        checkpoint_interval=10,
        catfish_enabled=False
    )

    # 减少 agent 数量以加快测试
    config.agents[AgentType.RETAIL].count = 100
    config.agents[AgentType.RETAIL_PRO].count = 10
    config.agents[AgentType.WHALE].count = 10
    config.agents[AgentType.MARKET_MAKER].count = 10
    config.training.retail_sub_population_count = 1  # 减少子种群

    multi_config = MultiArenaConfig(
        num_arenas=2,
        episodes_per_arena=2,
    )

    mem_after_config = get_memory_mb()
    print(f"配置加载后内存: {mem_after_config:.1f} MB (+{mem_after_config - initial_mem:.1f} MB)")

    try:
        trainer = ParallelArenaTrainer(config, multi_config)
        trainer.setup()

        mem_after_setup = get_memory_mb()
        print(f"Setup 后内存: {mem_after_setup:.1f} MB (+{mem_after_setup - mem_after_config:.1f} MB)")

        # 分析各种群的初始状态
        for agent_type, population in trainer.populations.items():
            if hasattr(population, 'sub_populations'):
                for i, sub_pop in enumerate(population.sub_populations):
                    analyze_neat_population(sub_pop.neat_pop, f"{agent_type.value}_sub_{i}")
            else:
                analyze_neat_population(population.neat_pop, agent_type.value)

        # 运行几轮训练
        print("\n" + "=" * 60)
        print("开始训练轮次...")
        print("=" * 60)

        for round_idx in range(5):
            gc.collect()
            mem_before = get_memory_mb()

            stats = trainer.run_round()

            gc.collect()
            gc.collect()
            gc.collect()
            mem_after = get_memory_mb()

            print(f"\n--- Round {round_idx + 1} ---")
            print(f"  内存: {mem_before:.1f} MB -> {mem_after:.1f} MB (增长: {mem_after - mem_before:+.1f} MB)")
            print(f"  Generation: {stats.get('generation', 'N/A')}")

            # 分析种群状态
            for agent_type, population in trainer.populations.items():
                if hasattr(population, 'sub_populations'):
                    sub_pop = population.sub_populations[0]
                    neat_stats = analyze_neat_population(sub_pop.neat_pop, f"{agent_type.value}_sub_0")
                    check_genome_connections_count(sub_pop.neat_pop)
                else:
                    neat_stats = analyze_neat_population(population.neat_pop, agent_type.value)
                    check_genome_connections_count(population.neat_pop)

        # 最终内存统计
        final_mem = get_memory_mb()
        print("\n" + "=" * 60)
        print("最终内存统计")
        print("=" * 60)
        print(f"初始内存: {initial_mem:.1f} MB")
        print(f"最终内存: {final_mem:.1f} MB")
        print(f"总增长: {final_mem - initial_mem:.1f} MB")

        # tracemalloc 统计
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        print("\n=== Top 20 内存分配位置 ===")
        for stat in top_stats[:20]:
            print(stat)

        trainer.stop()

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        tracemalloc.stop()


def run_simple_neat_test():
    """简单的 NEAT 内存测试"""
    print("=" * 60)
    print("简单 NEAT 进化内存测试")
    print("=" * 60)

    config_path = project_root / "config" / "neat_retail.cfg"
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    # 减少种群大小
    neat_config.pop_size = 100

    initial_mem = get_memory_mb()
    print(f"初始内存: {initial_mem:.1f} MB")

    pop = neat.Population(neat_config)

    mem_after_create = get_memory_mb()
    print(f"创建种群后: {mem_after_create:.1f} MB (+{mem_after_create - initial_mem:.1f} MB)")

    def dummy_fitness(genomes, config):
        for gid, genome in genomes:
            genome.fitness = 1.0

    # 运行多代进化
    for gen in range(20):
        mem_before = get_memory_mb()

        # 评估
        dummy_fitness(list(pop.population.items()), neat_config)

        # 找最优
        best = max(pop.population.values(), key=lambda g: g.fitness)
        if pop.best_genome is None or best.fitness > pop.best_genome.fitness:
            pop.best_genome = best

        # 繁殖
        pop.population = pop.reproduction.reproduce(
            neat_config, pop.species, neat_config.pop_size, pop.generation
        )

        # 物种划分
        pop.species.speciate(neat_config, pop.population, pop.generation)

        pop.generation += 1

        gc.collect()
        mem_after = get_memory_mb()

        # 分析状态
        stats = analyze_neat_population(pop, f"Gen {gen}")

        print(f"Gen {gen}: 内存 {mem_before:.1f} -> {mem_after:.1f} MB ({mem_after - mem_before:+.1f} MB)")
        print(f"  ancestors: {stats.get('ancestors_size', 0)}, "
              f"genome_to_species: {stats.get('genome_to_species_size', 0)}, "
              f"fitness_history: {stats.get('species_total_fitness_history', 0)}")

    final_mem = get_memory_mb()
    print(f"\n最终内存: {final_mem:.1f} MB (总增长: {final_mem - initial_mem:.1f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="内存泄漏诊断脚本")
    parser.add_argument("--simple", action="store_true", help="只运行简单的 NEAT 测试")
    args = parser.parse_args()

    if args.simple:
        run_simple_neat_test()
    else:
        run_memory_test()
