"""进化效果测试器模块

通过对比测试评估进化是否有效。使用 ParallelArenaTrainer 多竞技场并行测试。
"""

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.checkpoint_loader import CheckpointLoader
from src.config.config import AgentType, Config
from src.core.log_engine.logger import get_logger


class EvolutionTester:
    """进化效果测试器

    通过对比测试评估进化是否有效。

    测试方法：
    1. 基准测试：使用第 N 代 4 个物种的全部基因组竞技
    2. 比较测试：第 N 代某物种 + 第 N-1 代其他物种

    如果进化有效，第 N 代物种在比较测试中应表现更好。
    """

    config: Config
    checkpoint_dir: str
    results_dir: str

    def __init__(
        self,
        config: Config,
        checkpoint_dir: str = "checkpoints",
        results_dir: str = "checkpoints/test_results",
    ) -> None:
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.logger = get_logger("evolution_tester")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/baseline", exist_ok=True)
        os.makedirs(f"{self.results_dir}/comparison", exist_ok=True)

    def _load_generation_data(
        self, generation: int
    ) -> dict[AgentType, list[bytes]] | None:
        """加载某一代的全部基因组数据"""
        loader = CheckpointLoader(self.checkpoint_dir)
        return loader.load_genomes(generation)

    def _run_test_with_multi_arena(
        self,
        populations_data: dict[AgentType, list[bytes]],
        num_arenas: int,
        episodes_per_run: int,
        episode_length: int,
        test_type: str,
        run_idx: int,
    ) -> dict[str, Any]:
        """使用多竞技场运行单次测试

        创建 ParallelArenaTrainer，从 genome 数据初始化，
        运行 episodes 并收集适应度。

        Args:
            populations_data: 各物种的序列化基因组列表字典
            num_arenas: 竞技场数量
            episodes_per_run: 每次运行的 episode 数量
            episode_length: 每个 episode 的 tick 数量
            test_type: 测试类型
            run_idx: 运行索引
        """
        from src.training.arena import MultiArenaConfig, ParallelArenaTrainer
        from src.training.population import SubPopulationManager

        multi_config = MultiArenaConfig(
            num_arenas=num_arenas,
            episodes_per_arena=episodes_per_run,
        )
        trainer = ParallelArenaTrainer(self.config, multi_config)
        trainer.setup_for_testing(populations_data)

        try:
            all_episode_results: list[dict[str, Any]] = []
            total_ticks: int = 0

            for ep_idx in range(episodes_per_run):
                trainer._reset_all_arenas()
                trainer._init_market_all_arenas()

                trainer._is_running = True
                for _ in range(episode_length):
                    if not trainer._is_running:
                        break
                    should_continue = trainer.run_tick_all_arenas()
                    if not should_continue:
                        break
                trainer._is_running = False

                ep_ticks = max(arena.tick for arena in trainer.arena_states)
                total_ticks += ep_ticks

                episode_fitness = trainer._collect_episode_fitness()

                episode_result: dict[str, Any] = {
                    "episode_idx": ep_idx,
                    "species_results": {},
                }

                for agent_type, population in trainer.populations.items():
                    fitnesses_list: list[float] = []

                    if isinstance(population, SubPopulationManager):
                        for sub_pop in population.sub_populations:
                            key = (agent_type, sub_pop.sub_population_id or 0)
                            if key in episode_fitness:
                                avg_fitness = episode_fitness[key] / num_arenas
                                fitnesses_list.extend(avg_fitness.tolist())
                    else:
                        key = (agent_type, 0)
                        if key in episode_fitness:
                            avg_fitness = episode_fitness[key] / num_arenas
                            fitnesses_list.extend(avg_fitness.tolist())

                    if fitnesses_list:
                        alive_count = sum(
                            1 for f in fitnesses_list if f > -0.99
                        )
                        total_count = len(fitnesses_list)
                        survival_rate = (
                            alive_count / total_count if total_count > 0 else 0.0
                        )
                        avg_fitness_val = float(np.mean(fitnesses_list))

                        episode_result["species_results"][agent_type] = {
                            "total_count": total_count,
                            "alive_count": alive_count,
                            "survival_rate": survival_rate,
                            "avg_fitness": avg_fitness_val,
                        }

                all_episode_results.append(episode_result)

            results: dict[str, Any] = {
                "test_type": test_type,
                "run_idx": run_idx,
                "episodes_per_run": episodes_per_run,
                "total_ticks": total_ticks,
                "avg_ticks_per_episode": (
                    total_ticks / episodes_per_run if episodes_per_run > 0 else 0
                ),
                "species_results": {},
            }

            for agent_type in AgentType:
                fitnesses_across_eps: list[float] = []
                survival_rates: list[float] = []
                total_count: int = 0

                for ep_result in all_episode_results:
                    species_result = ep_result.get("species_results", {}).get(
                        agent_type
                    )
                    if species_result:
                        fitnesses_across_eps.append(species_result["avg_fitness"])
                        survival_rates.append(species_result["survival_rate"])
                        total_count = species_result["total_count"]

                if fitnesses_across_eps:
                    results["species_results"][agent_type] = {
                        "total_count": total_count,
                        "avg_fitness": float(np.mean(fitnesses_across_eps)),
                        "std_fitness": float(np.std(fitnesses_across_eps)),
                        "avg_survival_rate": float(np.mean(survival_rates)),
                        "std_survival_rate": float(np.std(survival_rates)),
                    }

            return results
        finally:
            trainer.stop()

    def run_baseline_test(
        self,
        generation: int,
        num_runs: int = 3,
        episode_length: int = 1000,
        episodes_per_run: int = 10,
        num_arenas: int = 2,
        force: bool = False,
    ) -> dict[str, Any]:
        """基准测试：使用第 N 代 4 个物种的全部基因组竞技

        Args:
            generation: 代数
            num_runs: 运行次数
            episode_length: 每个 episode 的 tick 数量
            episodes_per_run: 每次运行的 episode 数量
            num_arenas: 竞技场数量
            force: 是否强制重新运行（忽略缓存）
        """
        cache_path = Path(self.results_dir) / "baseline" / f"gen_{generation}.pkl"
        if cache_path.exists() and not force:
            self.logger.info(f"加载基准测试缓存: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        gen_data = self._load_generation_data(generation)
        if gen_data is None:
            return {"error": f"无法加载代 {generation} 的数据"}

        self.logger.info(
            f"开始基准测试: 代 {generation}, {num_runs} 次运行, "
            f"每次 {episodes_per_run} 个 episode, {num_arenas} 个竞技场"
        )

        results: list[dict[str, Any]] = []
        for run_idx in range(num_runs):
            try:
                result = self._run_test_with_multi_arena(
                    populations_data=gen_data,
                    num_arenas=num_arenas,
                    episodes_per_run=episodes_per_run,
                    episode_length=episode_length,
                    test_type="baseline",
                    run_idx=run_idx,
                )
                results.append(result)
                self.logger.info(
                    f"基准测试完成: run_{run_idx}, "
                    f"avg_ticks={result['avg_ticks_per_episode']:.0f}"
                )
            except Exception as e:
                self.logger.error(f"基准测试运行 {run_idx} 失败: {e}")

        summary = self._summarize_results(results, "baseline", generation)

        with open(cache_path, "wb") as f:
            pickle.dump(summary, f)
        self.logger.info(f"基准测试结果已保存: {cache_path}")

        return summary

    def run_comparison_test(
        self,
        target_generation: int,
        base_generation: int,
        target_species: AgentType,
        num_runs: int = 3,
        episode_length: int = 1000,
        episodes_per_run: int = 10,
        num_arenas: int = 2,
        force: bool = False,
    ) -> dict[str, Any]:
        """比较测试：第 N 代某物种 + 第 N-1 代其他物种

        Args:
            target_generation: 目标代数（新进化的代）
            base_generation: 基准代数（旧的代）
            target_species: 目标物种（使用新代）
            num_runs: 运行次数
            episode_length: 每个 episode 的 tick 数量
            episodes_per_run: 每次运行的 episode 数量
            num_arenas: 竞技场数量
            force: 是否强制重新运行（忽略缓存）
        """
        cache_path = (
            Path(self.results_dir)
            / "comparison"
            / f"gen_{target_generation}_vs_gen_{base_generation}_{target_species.value}.pkl"
        )
        if cache_path.exists() and not force:
            self.logger.info(f"加载比较测试缓存: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        target_data = self._load_generation_data(target_generation)
        base_data = self._load_generation_data(base_generation)

        if target_data is None:
            return {"error": f"无法加载代 {target_generation} 的数据"}
        if base_data is None:
            return {"error": f"无法加载代 {base_generation} 的数据"}

        mixed_data: dict[AgentType, list[bytes]] = {}
        for agent_type in AgentType:
            if agent_type == target_species:
                mixed_data[agent_type] = target_data[agent_type]
            else:
                mixed_data[agent_type] = base_data[agent_type]

        self.logger.info(
            f"开始比较测试: 代 {target_generation} 的 {target_species.value} "
            f"vs 代 {base_generation} 的其他物种, {num_runs} 次运行, "
            f"每次 {episodes_per_run} 个 episode, {num_arenas} 个竞技场"
        )

        results: list[dict[str, Any]] = []
        for run_idx in range(num_runs):
            try:
                result = self._run_test_with_multi_arena(
                    populations_data=mixed_data,
                    num_arenas=num_arenas,
                    episodes_per_run=episodes_per_run,
                    episode_length=episode_length,
                    test_type="comparison",
                    run_idx=run_idx,
                )
                results.append(result)
                self.logger.info(
                    f"比较测试完成: run_{run_idx}, "
                    f"avg_ticks={result['avg_ticks_per_episode']:.0f}"
                )
            except Exception as e:
                self.logger.error(f"比较测试运行 {run_idx} 失败: {e}")

        summary = self._summarize_results(
            results,
            "comparison",
            target_generation,
            base_generation=base_generation,
            target_species=target_species,
        )

        with open(cache_path, "wb") as f:
            pickle.dump(summary, f)
        self.logger.info(f"比较测试结果已保存: {cache_path}")

        return summary

    def _summarize_results(
        self,
        results: list[dict[str, Any]],
        test_type: str,
        generation: int,
        base_generation: int | None = None,
        target_species: AgentType | None = None,
    ) -> dict[str, Any]:
        """汇总测试结果"""
        if not results:
            return {"error": "没有有效的测试结果"}

        episodes_per_run: int = results[0].get("episodes_per_run", 1)

        species_summary: dict[AgentType, dict[str, Any]] = {}

        for agent_type in AgentType:
            fitnesses: list[float] = []
            survival_rates: list[float] = []

            for run_result in results:
                species_result = run_result.get("species_results", {}).get(agent_type)
                if species_result:
                    fitnesses.append(species_result["avg_fitness"])
                    survival_rates.append(species_result["avg_survival_rate"])

            if fitnesses:
                species_summary[agent_type] = {
                    "avg_fitness": float(np.mean(fitnesses)),
                    "std_fitness": float(np.std(fitnesses)),
                    "avg_survival_rate": float(np.mean(survival_rates)),
                    "std_survival_rate": float(np.std(survival_rates)),
                    "runs": len(fitnesses),
                }

        summary: dict[str, Any] = {
            "test_type": test_type,
            "generation": generation,
            "num_runs": len(results),
            "episodes_per_run": episodes_per_run,
            "species_summary": species_summary,
        }

        if base_generation is not None:
            summary["base_generation"] = base_generation
        if target_species is not None:
            summary["target_species"] = target_species

        return summary

    def evaluate_evolution_effectiveness(
        self,
        generation: int,
        num_runs: int = 3,
        episode_length: int = 1000,
        episodes_per_run: int = 10,
        num_arenas: int = 2,
        force: bool = False,
    ) -> dict[str, Any]:
        """评估进化有效性

        串行运行基准测试和所有比较测试，然后分析结果。
        PAT 自身通过 OpenMP 实现并行推理，不需要多进程。

        Args:
            generation: 要评估的代数（N）
            num_runs: 每个测试的运行次数
            episode_length: 每个 episode 的 tick 数量
            episodes_per_run: 每次运行的 episode 数量
            num_arenas: 竞技场数量
            force: 是否强制重新运行（忽略缓存）
        """
        if generation < 1:
            return {"error": "代数必须大于 0"}

        base_generation = generation - 1

        self.logger.info(
            f"开始评估进化有效性: 代 {generation}, "
            f"基准代 {base_generation}, "
            f"每测试 {num_runs} 次运行, "
            f"每次 {episodes_per_run} 个 episode, {num_arenas} 个竞技场"
        )

        # 加载代数据
        target_data = self._load_generation_data(generation)
        base_data = self._load_generation_data(base_generation)

        if target_data is None:
            return {"error": f"无法加载代 {generation} 的数据"}
        if base_data is None:
            return {"error": f"无法加载代 {base_generation} 的数据"}

        # 检查缓存
        baseline_cache = Path(self.results_dir) / "baseline" / f"gen_{generation}.pkl"
        comparison_caches = {
            agent_type: (
                Path(self.results_dir)
                / "comparison"
                / f"gen_{generation}_vs_gen_{base_generation}_{agent_type.value}.pkl"
            )
            for agent_type in AgentType
        }

        if not force and all(
            [baseline_cache.exists()]
            + [p.exists() for p in comparison_caches.values()]
        ):
            self.logger.info("所有测试已有缓存，直接汇总结果")
            return self._compile_effectiveness_report(
                generation, base_generation
            )

        # 串行运行基准测试
        self.logger.info(f"运行基准测试: {num_runs} 次")
        baseline_results: list[dict[str, Any]] = []
        for run_idx in range(num_runs):
            try:
                result = self._run_test_with_multi_arena(
                    populations_data=target_data,
                    num_arenas=num_arenas,
                    episodes_per_run=episodes_per_run,
                    episode_length=episode_length,
                    test_type="baseline",
                    run_idx=run_idx,
                )
                baseline_results.append(result)
                self.logger.info(
                    f"基准测试 run_{run_idx} 完成, "
                    f"avg_ticks={result['avg_ticks_per_episode']:.0f}"
                )
            except Exception as e:
                self.logger.error(f"基准测试 run_{run_idx} 失败: {e}")

        baseline_summary = self._summarize_results(
            baseline_results, "baseline", generation
        )
        with open(baseline_cache, "wb") as f:
            pickle.dump(baseline_summary, f)

        # 串行运行比较测试（每个物种）
        comparison_summaries: dict[AgentType, dict[str, Any]] = {}
        for target_species in AgentType:
            mixed_data: dict[AgentType, list[bytes]] = {}
            for agent_type in AgentType:
                if agent_type == target_species:
                    mixed_data[agent_type] = target_data[agent_type]
                else:
                    mixed_data[agent_type] = base_data[agent_type]

            self.logger.info(
                f"运行比较测试: {target_species.value}, {num_runs} 次"
            )
            comparison_results: list[dict[str, Any]] = []
            for run_idx in range(num_runs):
                try:
                    result = self._run_test_with_multi_arena(
                        populations_data=mixed_data,
                        num_arenas=num_arenas,
                        episodes_per_run=episodes_per_run,
                        episode_length=episode_length,
                        test_type="comparison",
                        run_idx=run_idx,
                    )
                    comparison_results.append(result)
                    self.logger.info(
                        f"比较测试 {target_species.value} run_{run_idx} 完成, "
                        f"avg_ticks={result['avg_ticks_per_episode']:.0f}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"比较测试 {target_species.value} run_{run_idx} 失败: {e}"
                    )

            summary = self._summarize_results(
                comparison_results,
                "comparison",
                generation,
                base_generation=base_generation,
                target_species=target_species,
            )
            comparison_summaries[target_species] = summary
            with open(comparison_caches[target_species], "wb") as f:
                pickle.dump(summary, f)

        self.logger.info("所有测试完成")

        return self._compile_effectiveness_report(generation, base_generation)

    def _compile_effectiveness_report(
        self,
        generation: int,
        base_generation: int,
    ) -> dict[str, Any]:
        """编译进化有效性报告"""
        baseline_cache = Path(self.results_dir) / "baseline" / f"gen_{generation}.pkl"
        with open(baseline_cache, "rb") as f:
            baseline = pickle.load(f)

        comparisons: dict[AgentType, dict[str, Any]] = {}
        for agent_type in AgentType:
            cache_path = (
                Path(self.results_dir)
                / "comparison"
                / f"gen_{generation}_vs_gen_{base_generation}_{agent_type.value}.pkl"
            )
            with open(cache_path, "rb") as f:
                comparisons[agent_type] = pickle.load(f)

        effectiveness: dict[AgentType, dict[str, Any]] = {}

        for agent_type in AgentType:
            baseline_species = baseline.get("species_summary", {}).get(agent_type, {})
            comparison_species = (
                comparisons[agent_type].get("species_summary", {}).get(agent_type, {})
            )

            baseline_fitness = baseline_species.get("avg_fitness", 0.0)
            comparison_fitness = comparison_species.get("avg_fitness", 0.0)

            improvement = comparison_fitness - baseline_fitness

            if abs(baseline_fitness) > 0.001:
                relative_improvement = improvement / abs(baseline_fitness) * 100
            else:
                relative_improvement = float("inf") if improvement > 0 else 0.0

            effectiveness[agent_type] = {
                "baseline_fitness": baseline_fitness,
                "comparison_fitness": comparison_fitness,
                "absolute_improvement": improvement,
                "relative_improvement_pct": relative_improvement,
                "is_effective": improvement > 0,
            }

        report: dict[str, Any] = {
            "generation": generation,
            "base_generation": base_generation,
            "baseline": baseline,
            "comparisons": comparisons,
            "effectiveness": effectiveness,
            "summary": {
                "effective_species": [
                    agent_type.value
                    for agent_type, eff in effectiveness.items()
                    if eff["is_effective"]
                ],
                "ineffective_species": [
                    agent_type.value
                    for agent_type, eff in effectiveness.items()
                    if not eff["is_effective"]
                ],
            },
        }

        self.logger.info(
            f"进化有效性评估完成: "
            f"有效物种={report['summary']['effective_species']}, "
            f"无效物种={report['summary']['ineffective_species']}"
        )

        return report
