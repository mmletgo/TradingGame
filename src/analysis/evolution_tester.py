"""进化效果测试器模块

通过对比测试评估进化是否有效。使用多进程并行运行所有测试。
"""

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import neat
import numpy as np

from src.analysis.checkpoint_loader import CheckpointLoader
from src.bio.agents.base import Agent, AgentType
from src.bio.agents.market_maker import MarketMakerAgent
from src.bio.agents.retail import RetailAgent
from src.bio.agents.retail_pro import RetailProAgent
from src.bio.agents.whale import WhaleAgent
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, Config
from src.core.log_engine.logger import get_logger
from src.market.adl.adl_manager import ADLManager
from src.market.matching.matching_engine import MatchingEngine


# Agent ID 偏移量，与 Population 保持一致
_AGENT_ID_OFFSET: dict[AgentType, int] = {
    AgentType.RETAIL: 0,
    AgentType.RETAIL_PRO: 1_000_000,
    AgentType.WHALE: 2_000_000,
    AgentType.MARKET_MAKER: 3_000_000,
}

# Agent 类构造函数类型
AgentConstructor = Callable[[int, Brain, AgentConfig], Agent]

# Agent 类映射
_AGENT_CLASSES: dict[AgentType, AgentConstructor] = {
    AgentType.RETAIL: RetailAgent,
    AgentType.RETAIL_PRO: RetailProAgent,
    AgentType.WHALE: WhaleAgent,
    AgentType.MARKET_MAKER: MarketMakerAgent,
}


def _run_single_test_worker(params: dict[str, Any]) -> dict[str, Any]:
    """进程池 worker 函数，执行单次测试（支持多 episode）

    Args:
        params: 测试参数字典，包含：
            - config: Config 对象
            - populations_data: 各物种的 genome_data 列表字典
            - episode_length: 每个 episode 的 tick 数量
            - episodes_per_run: 每次运行的 episode 数量（默认 10）
            - test_type: 测试类型 ("baseline" 或 "comparison")
            - run_idx: 运行索引

    Returns:
        测试结果字典，包含多个 episode 的平均表现
    """
    # 延迟导入，避免循环依赖
    from src.training.trainer import Trainer
    from src.market.catfish import create_all_catfish, create_catfish

    config: Config = params["config"]
    populations_data: dict[AgentType, list[bytes]] = params["populations_data"]
    episode_length: int = params["episode_length"]
    episodes_per_run: int = params.get("episodes_per_run", 10)
    test_type: str = params["test_type"]
    run_idx: int = params["run_idx"]

    # 创建 Trainer（保留鲶鱼配置）
    trainer = Trainer(config)

    # 加载 NEAT 配置
    neat_configs: dict[AgentType, neat.Config] = {}
    config_dir = Path(config.training.neat_config_path)
    for agent_type in AgentType:
        if agent_type == AgentType.MARKET_MAKER:
            neat_config_path = config_dir / "neat_market_maker.cfg"
        elif agent_type == AgentType.WHALE:
            neat_config_path = config_dir / "neat_whale.cfg"
        elif agent_type == AgentType.RETAIL_PRO:
            neat_config_path = config_dir / "neat_retail_pro.cfg"
        else:
            neat_config_path = config_dir / "neat_retail.cfg"

        neat_configs[agent_type] = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(neat_config_path),
        )

    # 创建撮合引擎和 ADL 管理器
    trainer.matching_engine = MatchingEngine(config.market)
    trainer.adl_manager = ADLManager()

    # 从 genome 创建 Agent
    for agent_type in AgentType:
        genome_data_list = populations_data.get(agent_type, [])
        if not genome_data_list:
            continue

        # 反序列化所有 genome
        genome_list: list[neat.DefaultGenome] = []
        for gd in genome_data_list:
            genome = pickle.loads(gd)
            if isinstance(genome, neat.DefaultGenome):
                genome_list.append(genome)

        if not genome_list:
            continue

        # 获取该物种的配置
        agent_config = config.agents[agent_type]
        neat_config = neat_configs[agent_type]
        agent_class = _AGENT_CLASSES[agent_type]
        agent_id_offset = _AGENT_ID_OFFSET[agent_type]

        # 创建 count 个 Agent（循环使用 genome 列表中的基因组）
        agents: list[Agent] = []
        for i in range(agent_config.count):
            # 循环使用 genome 列表中的基因组
            genome = genome_list[i % len(genome_list)]
            brain = Brain.from_genome(genome, neat_config)
            agent = agent_class(agent_id_offset + i, brain, agent_config)
            agents.append(agent)

        # 创建简化的 Population 对象
        from src.training.population import Population

        pop = Population.__new__(Population)
        pop.agent_type = agent_type
        pop.agent_config = agent_config
        pop.generation = 0
        pop.logger = get_logger("population")
        pop._executor = None
        pop._num_workers = 8
        pop.neat_config = neat_config
        pop.neat_pop = None  # 测试模式不需要 NEAT 种群
        pop.agents = agents

        trainer.populations[agent_type] = pop

    # 注册 Agent 费率
    trainer._register_all_agents()
    trainer._build_agent_map()
    trainer._build_execution_order()
    trainer._update_pop_total_counts()

    # 初始化鲶鱼（如果配置中启用）
    if config.catfish and config.catfish.enabled:
        catfish_initial_balance = trainer._calculate_catfish_initial_balance()
        whale_config = config.agents[AgentType.WHALE]
        catfish_leverage = whale_config.leverage
        catfish_mmr = whale_config.maintenance_margin_rate

        if config.catfish.multi_mode:
            trainer.catfish_list = create_all_catfish(
                config.catfish,
                initial_balance=catfish_initial_balance,
                leverage=catfish_leverage,
                maintenance_margin_rate=catfish_mmr,
            )
            for catfish in trainer.catfish_list:
                trainer.matching_engine.register_agent(catfish.catfish_id, 0.0, 0.0)
        else:
            catfish = create_catfish(
                -1,
                config.catfish,
                initial_balance=catfish_initial_balance,
                leverage=catfish_leverage,
                maintenance_margin_rate=catfish_mmr,
            )
            trainer.catfish_list = [catfish]
            trainer.matching_engine.register_agent(catfish.catfish_id, 0.0, 0.0)

    # 初始化市场
    trainer._ema_alpha = config.market.ema_alpha
    trainer._init_ema_price(config.market.initial_price)
    trainer._init_market()

    # 记录初始余额（各物种共用）
    initial_balances: dict[AgentType, float] = {}
    for agent_type, population in trainer.populations.items():
        if population.agents:
            initial_balances[agent_type] = population.agents[0].account.initial_balance

    # 运行多个 episode，收集每个 episode 的结果
    all_episode_results: list[dict[str, Any]] = []
    total_ticks: int = 0

    for ep_idx in range(episodes_per_run):
        # 重置所有 Agent
        for population in trainer.populations.values():
            population.reset_agents()

        # 重置鲶鱼
        for catfish in trainer.catfish_list:
            catfish.reset()

        # 重置市场状态
        trainer._reset_market()
        trainer.tick = 0
        trainer._pop_liquidated_counts.clear()
        trainer._catfish_liquidated = False

        # 初始化 episode 价格统计
        initial_price = config.market.initial_price
        trainer._episode_high_price = initial_price
        trainer._episode_low_price = initial_price

        # 运行单个 episode
        trainer.is_running = True
        trainer.episode = ep_idx + 1

        # 测试模式下，鲶鱼爆仓不结束 episode，只有以下情况才结束：
        # 1. tick 达到 episode_length
        # 2. 任一物种被淘汰到只剩 1/4
        for _ in range(episode_length):
            if not trainer.is_running:
                break
            trainer.run_tick()

            # 检查提前结束条件（物种淘汰到 1/4 或订单簿单边）
            early_end_result = trainer._should_end_episode_early()
            if early_end_result is not None:
                break

        trainer.is_running = False
        total_ticks += trainer.tick

        # 收集本 episode 的结果
        current_price = trainer.matching_engine._orderbook.last_price
        episode_result: dict[str, Any] = {
            "episode_idx": ep_idx,
            "tick": trainer.tick,
            "final_price": current_price,
            "species_results": {},
        }

        for agent_type, population in trainer.populations.items():
            # 调用 population.evaluate 获取每个 Agent 的适应度
            # 适应度计算：
            # - 散户/高级散户：适应度 = 收益率
            # - 做市商：适应度 = 0.5 × 收益率 + 0.5 × maker_volume 排名归一化
            # - 庄家：适应度 = 0.5 × 收益率 + 0.5 × volatility_contribution 排名归一化
            agent_fitnesses = population.evaluate(current_price)

            # 统计存活和适应度
            alive_count = 0
            fitnesses: list[float] = []
            position_distribution = {"long": 0, "short": 0, "flat": 0}

            for agent, fitness in agent_fitnesses:
                if not agent.is_liquidated:
                    alive_count += 1
                    fitnesses.append(fitness)

                    pos_qty = agent.account.position.quantity
                    if pos_qty > 0:
                        position_distribution["long"] += 1
                    elif pos_qty < 0:
                        position_distribution["short"] += 1
                    else:
                        position_distribution["flat"] += 1

            total_count = len(population.agents)
            survival_rate = alive_count / total_count if total_count > 0 else 0.0

            if fitnesses:
                avg_fitness = float(np.mean(fitnesses))
            else:
                avg_fitness = -1.0  # 全部爆仓

            episode_result["species_results"][agent_type] = {
                "total_count": total_count,
                "alive_count": alive_count,
                "survival_rate": survival_rate,
                "avg_fitness": avg_fitness,
                "position_distribution": position_distribution,
            }

        all_episode_results.append(episode_result)

    # 汇总多个 episode 的平均结果
    results: dict[str, Any] = {
        "test_type": test_type,
        "run_idx": run_idx,
        "episodes_per_run": episodes_per_run,
        "total_ticks": total_ticks,
        "avg_ticks_per_episode": total_ticks / episodes_per_run if episodes_per_run > 0 else 0,
        "species_results": {},
    }

    for agent_type in AgentType:
        # 收集该物种在所有 episode 中的数据
        fitnesses_list: list[float] = []
        survival_rates: list[float] = []
        total_count: int = 0

        for ep_result in all_episode_results:
            species_result = ep_result.get("species_results", {}).get(agent_type)
            if species_result:
                fitnesses_list.append(species_result["avg_fitness"])
                survival_rates.append(species_result["survival_rate"])
                total_count = species_result["total_count"]

        if fitnesses_list:
            results["species_results"][agent_type] = {
                "total_count": total_count,
                "avg_fitness": float(np.mean(fitnesses_list)),
                "std_fitness": float(np.std(fitnesses_list)),
                "avg_survival_rate": float(np.mean(survival_rates)),
                "std_survival_rate": float(np.std(survival_rates)),
            }

    return results


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
        """初始化测试器

        Args:
            config: 全局配置对象
            checkpoint_dir: checkpoint 文件目录
            results_dir: 测试结果保存目录
        """
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.logger = get_logger("evolution_tester")

        # 确保目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/baseline", exist_ok=True)
        os.makedirs(f"{self.results_dir}/comparison", exist_ok=True)

    def _load_generation_data(
        self, generation: int
    ) -> dict[AgentType, list[bytes]] | None:
        """加载某一代的全部基因组数据

        Args:
            generation: 代数

        Returns:
            各物种的 genome_data 列表字典，如果文件不存在返回 None
        """
        loader = CheckpointLoader(self.checkpoint_dir)
        return loader.load_genomes(generation)

    def create_agents_from_genome(
        self,
        agent_type: AgentType,
        genome_data: bytes,
        count: int,
    ) -> list[Agent]:
        """从单个 genome 复制创建完整种群

        Args:
            agent_type: Agent 类型
            genome_data: 序列化的 genome 数据
            count: 要创建的 Agent 数量

        Returns:
            Agent 列表，每个 Agent 有独立账户
        """
        # 反序列化 genome
        genome = pickle.loads(genome_data)
        if not isinstance(genome, neat.DefaultGenome):
            raise ValueError("无效的 genome 数据")

        # 加载 NEAT 配置
        config_dir = Path(self.config.training.neat_config_path)
        if agent_type == AgentType.MARKET_MAKER:
            neat_config_path = config_dir / "neat_market_maker.cfg"
        elif agent_type == AgentType.WHALE:
            neat_config_path = config_dir / "neat_whale.cfg"
        elif agent_type == AgentType.RETAIL_PRO:
            neat_config_path = config_dir / "neat_retail_pro.cfg"
        else:
            neat_config_path = config_dir / "neat_retail.cfg"

        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(neat_config_path),
        )

        # 获取 Agent 配置和类
        agent_config = self.config.agents[agent_type]
        agent_class = _AGENT_CLASSES[agent_type]
        agent_id_offset = _AGENT_ID_OFFSET[agent_type]

        # 创建 Agent 列表
        agents: list[Agent] = []
        for i in range(count):
            brain = Brain.from_genome(genome, neat_config)
            agent = agent_class(agent_id_offset + i, brain, agent_config)
            agents.append(agent)

        return agents

    def run_baseline_test(
        self,
        generation: int,
        num_runs: int = 3,
        episode_length: int = 1000,
        episodes_per_run: int = 10,
        force: bool = False,
    ) -> dict[str, Any]:
        """基准测试：使用第 N 代 4 个物种的 best_genome 竞技

        Args:
            generation: 代数
            num_runs: 运行次数
            episode_length: 每个 episode 的 tick 数量
            episodes_per_run: 每次运行的 episode 数量（默认 10）
            force: 是否强制重新运行（忽略缓存）

        Returns:
            测试结果字典
        """
        # 检查缓存
        cache_path = Path(self.results_dir) / "baseline" / f"gen_{generation}.pkl"
        if cache_path.exists() and not force:
            self.logger.info(f"加载基准测试缓存: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # 加载代数据
        gen_data = self._load_generation_data(generation)
        if gen_data is None:
            return {"error": f"无法加载代 {generation} 的数据"}

        self.logger.info(
            f"开始基准测试: 代 {generation}, {num_runs} 次运行, "
            f"每次 {episodes_per_run} 个 episode"
        )

        # 准备测试参数
        test_params: list[dict[str, Any]] = []
        for run_idx in range(num_runs):
            test_params.append(
                {
                    "config": self.config,
                    "populations_data": gen_data,
                    "episode_length": episode_length,
                    "episodes_per_run": episodes_per_run,
                    "test_type": "baseline",
                    "run_idx": run_idx,
                }
            )

        # 并行运行测试
        results: list[dict[str, Any]] = []
        num_workers = min(os.cpu_count() or 4, num_runs)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_run_single_test_worker, params)
                for params in test_params
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(
                        f"基准测试完成: run_{result['run_idx']}, "
                        f"tick={result['tick']}"
                    )
                except Exception as e:
                    self.logger.error(f"基准测试运行失败: {e}")

        # 汇总结果
        summary = self._summarize_results(results, "baseline", generation)

        # 保存缓存
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
        force: bool = False,
    ) -> dict[str, Any]:
        """比较测试：第 N 代某物种 + 第 N-1 代其他物种

        Args:
            target_generation: 目标代数（新进化的代）
            base_generation: 基准代数（旧的代）
            target_species: 目标物种（使用新代）
            num_runs: 运行次数
            episode_length: 每个 episode 的 tick 数量
            episodes_per_run: 每次运行的 episode 数量（默认 10）
            force: 是否强制重新运行（忽略缓存）

        Returns:
            测试结果字典
        """
        # 检查缓存
        cache_path = (
            Path(self.results_dir)
            / "comparison"
            / f"gen_{target_generation}_vs_gen_{base_generation}_{target_species.value}.pkl"
        )
        if cache_path.exists() and not force:
            self.logger.info(f"加载比较测试缓存: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # 加载代数据
        target_data = self._load_generation_data(target_generation)
        base_data = self._load_generation_data(base_generation)

        if target_data is None:
            return {"error": f"无法加载代 {target_generation} 的数据"}
        if base_data is None:
            return {"error": f"无法加载代 {base_generation} 的数据"}

        # 构建混合种群数据
        mixed_data: dict[AgentType, list[bytes]] = {}
        for agent_type in AgentType:
            if agent_type == target_species:
                # 目标物种使用新代
                mixed_data[agent_type] = target_data[agent_type]
            else:
                # 其他物种使用旧代
                mixed_data[agent_type] = base_data[agent_type]

        self.logger.info(
            f"开始比较测试: 代 {target_generation} 的 {target_species.value} "
            f"vs 代 {base_generation} 的其他物种, {num_runs} 次运行, "
            f"每次 {episodes_per_run} 个 episode"
        )

        # 准备测试参数
        test_params: list[dict[str, Any]] = []
        for run_idx in range(num_runs):
            test_params.append(
                {
                    "config": self.config,
                    "populations_data": mixed_data,
                    "episode_length": episode_length,
                    "episodes_per_run": episodes_per_run,
                    "test_type": "comparison",
                    "run_idx": run_idx,
                }
            )

        # 并行运行测试
        results: list[dict[str, Any]] = []
        num_workers = min(os.cpu_count() or 4, num_runs)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_run_single_test_worker, params)
                for params in test_params
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(
                        f"比较测试完成: run_{result['run_idx']}, "
                        f"tick={result['tick']}"
                    )
                except Exception as e:
                    self.logger.error(f"比较测试运行失败: {e}")

        # 汇总结果
        summary = self._summarize_results(
            results,
            "comparison",
            target_generation,
            base_generation=base_generation,
            target_species=target_species,
        )

        # 保存缓存
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
        """汇总测试结果

        Args:
            results: 单次运行结果列表（每次运行包含多个 episode 的平均结果）
            test_type: 测试类型
            generation: 代数
            base_generation: 基准代数（比较测试时使用）
            target_species: 目标物种（比较测试时使用）

        Returns:
            汇总结果字典
        """
        if not results:
            return {"error": "没有有效的测试结果"}

        # 获取 episodes_per_run（从第一个结果中获取）
        episodes_per_run: int = results[0].get("episodes_per_run", 1)

        # 按物种汇总
        species_summary: dict[AgentType, dict[str, Any]] = {}

        for agent_type in AgentType:
            fitnesses: list[float] = []
            survival_rates: list[float] = []

            for run_result in results:
                species_result = run_result.get("species_results", {}).get(agent_type)
                if species_result:
                    # worker 返回的结果已经是多 episode 的平均值
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

        # 构建汇总
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
        force: bool = False,
    ) -> dict[str, Any]:
        """评估进化有效性

        并行运行基准测试和所有比较测试，然后分析结果。

        Args:
            generation: 要评估的代数（N）
            num_runs: 每个测试的运行次数
            episode_length: 每个 episode 的 tick 数量
            episodes_per_run: 每次运行的 episode 数量（默认 10）
            force: 是否强制重新运行（忽略缓存）

        Returns:
            评估结果字典，包含：
            - baseline: 基准测试结果
            - comparisons: 各物种的比较测试结果
            - effectiveness: 各物种的进化有效性评估
        """
        if generation < 1:
            return {"error": "代数必须大于 0"}

        base_generation = generation - 1

        self.logger.info(
            f"开始评估进化有效性: 代 {generation}, "
            f"基准代 {base_generation}, "
            f"每测试 {num_runs} 次运行, "
            f"每次 {episodes_per_run} 个 episode"
        )

        # 准备所有测试任务
        # 1 个基准测试 + 4 个比较测试 = 5 个场景
        # 每个场景 num_runs 次 = 共 5 * num_runs 个任务
        all_params: list[dict[str, Any]] = []

        # 加载代数据
        target_data = self._load_generation_data(generation)
        base_data = self._load_generation_data(base_generation)

        if target_data is None:
            return {"error": f"无法加载代 {generation} 的数据"}
        if base_data is None:
            return {"error": f"无法加载代 {base_generation} 的数据"}

        # 检查缓存（如果所有缓存都存在且不强制，直接汇总）
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
                generation, base_generation, force=False
            )

        # 基准测试参数
        for run_idx in range(num_runs):
            all_params.append(
                {
                    "config": self.config,
                    "populations_data": target_data,
                    "episode_length": episode_length,
                    "episodes_per_run": episodes_per_run,
                    "test_type": "baseline",
                    "run_idx": run_idx,
                    "scenario": f"baseline_gen_{generation}",
                }
            )

        # 比较测试参数（每个物种）
        for target_species in AgentType:
            # 构建混合种群数据
            mixed_data: dict[AgentType, list[bytes]] = {}
            for agent_type in AgentType:
                if agent_type == target_species:
                    mixed_data[agent_type] = target_data[agent_type]
                else:
                    mixed_data[agent_type] = base_data[agent_type]

            for run_idx in range(num_runs):
                all_params.append(
                    {
                        "config": self.config,
                        "populations_data": mixed_data,
                        "episode_length": episode_length,
                        "episodes_per_run": episodes_per_run,
                        "test_type": "comparison",
                        "run_idx": run_idx,
                        "target_species": target_species,
                        "scenario": f"comparison_{target_species.value}",
                    }
                )

        self.logger.info(f"共 {len(all_params)} 个测试任务，开始并行执行")

        # 并行执行所有测试
        num_workers = os.cpu_count() or 4
        all_results: list[dict[str, Any]] = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_run_single_test_worker, params)
                for params in all_params
            ]
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    # 添加场景信息
                    result["scenario"] = all_params[i].get(
                        "scenario", result["test_type"]
                    )
                    if "target_species" in all_params[i]:
                        result["target_species"] = all_params[i]["target_species"]
                    all_results.append(result)

                    completed = len(all_results)
                    total = len(all_params)
                    self.logger.info(f"测试进度: {completed}/{total}")
                except Exception as e:
                    self.logger.error(f"测试运行失败: {e}")

        # 分类结果
        baseline_results: list[dict[str, Any]] = []
        comparison_results: dict[AgentType, list[dict[str, Any]]] = {
            agent_type: [] for agent_type in AgentType
        }

        for result in all_results:
            if result["test_type"] == "baseline":
                baseline_results.append(result)
            else:
                target_species = result.get("target_species")
                if target_species:
                    comparison_results[target_species].append(result)

        # 汇总并保存结果
        baseline_summary = self._summarize_results(
            baseline_results, "baseline", generation
        )
        with open(baseline_cache, "wb") as f:
            pickle.dump(baseline_summary, f)

        comparison_summaries: dict[AgentType, dict[str, Any]] = {}
        for agent_type, results in comparison_results.items():
            summary = self._summarize_results(
                results,
                "comparison",
                generation,
                base_generation=base_generation,
                target_species=agent_type,
            )
            comparison_summaries[agent_type] = summary
            with open(comparison_caches[agent_type], "wb") as f:
                pickle.dump(summary, f)

        # 编译最终报告
        return self._compile_effectiveness_report(generation, base_generation)

    def _compile_effectiveness_report(
        self,
        generation: int,
        base_generation: int,
        force: bool = False,
    ) -> dict[str, Any]:
        """编译进化有效性报告

        Args:
            generation: 目标代数
            base_generation: 基准代数
            force: 是否强制重新计算

        Returns:
            完整的有效性评估报告
        """
        # 加载基准测试结果
        baseline_cache = Path(self.results_dir) / "baseline" / f"gen_{generation}.pkl"
        with open(baseline_cache, "rb") as f:
            baseline = pickle.load(f)

        # 加载比较测试结果
        comparisons: dict[AgentType, dict[str, Any]] = {}
        for agent_type in AgentType:
            cache_path = (
                Path(self.results_dir)
                / "comparison"
                / f"gen_{generation}_vs_gen_{base_generation}_{agent_type.value}.pkl"
            )
            with open(cache_path, "rb") as f:
                comparisons[agent_type] = pickle.load(f)

        # 计算进化有效性
        effectiveness: dict[AgentType, dict[str, Any]] = {}

        for agent_type in AgentType:
            baseline_species = baseline.get("species_summary", {}).get(agent_type, {})
            comparison_species = (
                comparisons[agent_type].get("species_summary", {}).get(agent_type, {})
            )

            baseline_fitness = baseline_species.get("avg_fitness", 0.0)
            comparison_fitness = comparison_species.get("avg_fitness", 0.0)

            # 进化有效性 = 比较测试适应度 - 基准测试适应度
            # 正值表示新代在与旧代竞争时表现更好（进化有效）
            improvement = comparison_fitness - baseline_fitness

            # 计算相对改善百分比
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

        # 构建最终报告
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
