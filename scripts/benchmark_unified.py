#!/usr/bin/env python3
"""统一性能基准测试脚本

整合了4个脚本的功能，通过 --mode 参数选择测试模式：

1. profile - 使用 cProfile 进行整体性能分析
2. arena - 单训练场完整基准测试（默认）
3. evolution - 进化方法对比测试
4. detailed - 详细瓶颈剖析

使用方法:
    # 默认模式：arena 基准测试
    python scripts/benchmark_unified.py --episodes 5

    # cProfile 性能分析
    python scripts/benchmark_unified.py --mode profile

    # 进化方法对比
    python scripts/benchmark_unified.py --mode evolution --iterations 3

    # 详细瓶颈剖析
    python scripts/benchmark_unified.py --mode detailed

    # 输出到 JSON
    python scripts/benchmark_unified.py --mode arena --output results.json
"""

import argparse
import gc
import importlib
import json
import random
import sys
import time
import cProfile
import pstats
from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

# 关键：在导入任何项目模块之前，先清除 importlib 缓存
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bio.agents.base import AgentType
from src.core.log_engine.logger import setup_logging
from src.training.trainer import Trainer
from src.training.population import (
    PersistentWorkerPool,
    RetailSubPopulationManager,
    SubPopulationManager,
    _serialize_genomes_numpy,
    _unpack_network_params_numpy,
)

from scripts.create_config import create_default_config


# ============================================================================
# 共享数据类
# ============================================================================

@dataclass
class TimingStats:
    """统计数据类"""
    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        """添加一个值"""
        self.values.append(value)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        mean = self.mean
        variance = sum((x - mean) ** 2 for x in self.values) / len(self.values)
        return variance ** 0.5

    @property
    def total(self) -> float:
        return sum(self.values)

    def percentile(self, p: float) -> float:
        """计算百分位数"""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * p / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]


@dataclass
class EpisodeTimings:
    """单个 Episode 的计时数据"""
    agent_reset: float = 0.0
    market_reset: float = 0.0
    tick_loop_total: float = 0.0
    evolution_total: float = 0.0

    # Tick 级别明细
    tick_liquidation_check: float = 0.0
    tick_market_state: float = 0.0
    tick_parallel_decide: float = 0.0
    tick_serial_execute: float = 0.0
    tick_other: float = 0.0


@dataclass
class EvolutionTimings:
    """进化阶段的计时数据（按种群）"""
    evaluate: dict[AgentType, float] = field(default_factory=dict)
    cleanup_old_agents: dict[AgentType, float] = field(default_factory=dict)
    neat_run: dict[AgentType, float] = field(default_factory=dict)
    create_agents: dict[AgentType, float] = field(default_factory=dict)
    gc_time: float = 0.0
    cache_update_time: float = 0.0


# ============================================================================
# 模式 1: Profile 模式 - cProfile 性能分析
# ============================================================================

def run_profile_mode(args: argparse.Namespace) -> dict[str, Any]:
    """使用 cProfile 进行性能分析"""
    result: dict[str, Any] = {}

    def run_training() -> tuple[float, float]:
        """运行训练（被 profile 的函数）

        Returns:
            tuple[float, float]: (初始化耗时, tick+进化耗时)
        """
        config = create_default_config(
            episode_length=args.episode_length,
            checkpoint_interval=0,
        )
        trainer = Trainer(config)

        init_start = time.perf_counter()
        trainer.setup()
        init_time = time.perf_counter() - init_start

        train_start = time.perf_counter()
        trainer.train(episodes=1)
        train_time = time.perf_counter() - train_start

        return init_time, train_time

    print("=" * 60)
    print("性能分析 - NEAT AI 交易模拟")
    print("=" * 60)
    print(f"配置: 1000 散户, 100 高级散户, 100 庄家, 100 做市商")
    print(f"运行: 1 episode x {args.episode_length} ticks")
    print("=" * 60)
    print()

    profiler = cProfile.Profile()
    profiler.enable()

    init_time, train_time = run_training()

    profiler.disable()

    print("\n" + "=" * 60)
    print("耗时统计")
    print("=" * 60)
    print(f"初始化耗时 (setup):     {init_time:.3f} 秒")
    print(f"tick+进化耗时 (train):  {train_time:.3f} 秒")
    print(f"总耗时:                 {init_time + train_time:.3f} 秒")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("性能分析结果 (按累计时间排序 - 前 50)")
    print("=" * 60)

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(50)

    print("\n" + "=" * 60)
    print("性能分析结果 (按自身时间排序 - 前 50)")
    print("=" * 60)

    stats.sort_stats("tottime")
    stats.print_stats(50)

    profile_file = args.output or "profile_results.prof"
    stats.dump_stats(profile_file)
    print(f"\n完整结果已保存到: {profile_file}")
    print("使用 snakeviz profile_results.prof 可视化查看")

    result["init_time_s"] = init_time
    result["train_time_s"] = train_time
    result["total_time_s"] = init_time + train_time
    result["profile_file"] = profile_file

    return result


# ============================================================================
# 模式 2: Arena 模式 - 单训练场完整基准测试
# ============================================================================

class BenchmarkTrainer(Trainer):
    """带计时功能的 Trainer"""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.episode_timings: list[EpisodeTimings] = []
        self.evolution_timings: list[EvolutionTimings] = []
        self.inference_samples: list[float] = []

    def run_tick_with_timing(self) -> dict[str, float]:
        """带详细计时的 tick 执行"""
        if not self.matching_engine:
            return {}

        tick_timings: dict[str, float] = {}
        tick_start = time.perf_counter()

        self.tick += 1
        orderbook = self.matching_engine._orderbook

        current_price = (
            self._smooth_mid_price
            if self._smooth_mid_price > 0
            else orderbook.last_price
        )
        self.tick_start_price = current_price

        if self.tick == 1:
            self._price_history.append(current_price)
            self._tick_history_prices.append(current_price)
            self._tick_history_volumes.append(0.0)
            self._tick_history_amounts.append(0.0)
            if current_price > self._episode_high_price:
                self._episode_high_price = current_price
            if current_price < self._episode_low_price:
                self._episode_low_price = current_price
            return {"tick_1_setup": time.perf_counter() - tick_start}

        # 强平检查
        t0 = time.perf_counter()
        agents_to_liquidate = self._check_liquidations_vectorized(current_price)
        agents_to_liquidate_ids: set[int] = set()
        for agent in agents_to_liquidate:
            agents_to_liquidate_ids.add(agent.agent_id)
            self._cancel_agent_orders(agent)
        tick_timings["liquidation_check"] = time.perf_counter() - t0

        # 强平处理
        t0 = time.perf_counter()
        agents_need_adl: list[tuple[Any, int, bool]] = []
        if len(agents_to_liquidate) > 0:
            for agent in agents_to_liquidate:
                remaining_qty, is_long = self._execute_liquidation_market_order(agent)
                if remaining_qty > 0:
                    agents_need_adl.append((agent, remaining_qty, is_long))
                agent.is_liquidated = True
                self._pop_liquidated_counts[agent.agent_type] = (
                    self._pop_liquidated_counts.get(agent.agent_type, 0) + 1
                )
                if agent.account.balance < 0:
                    agent.account.balance = 0.0

            if agents_need_adl:
                latest_price = orderbook.last_price
                self._prepare_adl_candidates(latest_price, agents_to_liquidate_ids)
                for agent, remaining_qty, is_long in agents_need_adl:
                    self._execute_adl(agent, remaining_qty, latest_price, is_long)
                    if agent.account.position.quantity != 0:
                        agent.account.position.quantity = 0
                        agent.account.position.avg_price = 0.0
        tick_timings["liquidation_execute"] = time.perf_counter() - t0

        # 鲶鱼行动
        t0 = time.perf_counter()
        tick_trades: list[Any] = []
        for catfish in self.catfish_list:
            if catfish.is_liquidated:
                continue
            should_act, direction = catfish.decide(
                orderbook, self.tick, self._price_history
            )
            if should_act and direction != 0:
                catfish_trades = catfish.execute(direction, self.matching_engine)
                catfish.record_action(self.tick)
                for trade in catfish_trades:
                    self.recent_trades.append(trade)
                    tick_trades.append(trade)
                    is_buyer = trade.is_buyer_taker
                    catfish.account.on_trade(trade, is_buyer)
                    maker_id = (
                        trade.seller_id if trade.is_buyer_taker else trade.buyer_id
                    )
                    if maker_id > 0:
                        maker_agent = self.agent_map.get(maker_id)
                        if maker_agent is not None:
                            maker_is_buyer = not trade.is_buyer_taker
                            maker_agent.account.on_trade(trade, maker_is_buyer)
        tick_timings["catfish"] = time.perf_counter() - t0

        # 市场状态计算
        t0 = time.perf_counter()
        market_state = self._compute_normalized_market_state()
        tick_timings["market_state"] = time.perf_counter() - t0

        random.shuffle(self.agent_execution_order)

        # 并行决策
        t0 = time.perf_counter()
        decisions = self._batch_decide_parallel(
            self.agent_execution_order, market_state, orderbook
        )
        tick_timings["parallel_decide"] = time.perf_counter() - t0

        # 串行执行
        t0 = time.perf_counter()
        for agent, action, params in decisions:
            pre_trade_price = (
                orderbook.last_price if agent.agent_type == AgentType.WHALE else 0.0
            )
            trades = agent.execute_action(action, params, self.matching_engine)
            if agent.agent_type == AgentType.WHALE and trades:
                post_trade_price = orderbook.last_price
                if pre_trade_price > 0 and post_trade_price > 0:
                    price_impact = (
                        abs(post_trade_price - pre_trade_price) / pre_trade_price
                    )
                    agent.account.volatility_contribution += price_impact
            for trade in trades:
                self.recent_trades.append(trade)
                tick_trades.append(trade)
                maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
                maker_agent = self.agent_map.get(maker_id)
                if maker_agent is not None:
                    is_buyer = not trade.is_buyer_taker
                    maker_agent.account.on_trade(trade, is_buyer)
        tick_timings["serial_execute"] = time.perf_counter() - t0

        # 记录价格历史
        current_price = orderbook.last_price
        self._price_history.append(current_price)
        if current_price > self._episode_high_price:
            self._episode_high_price = current_price
        if current_price < self._episode_low_price:
            self._episode_low_price = current_price
        if len(self._price_history) > 1000:
            self._price_history = self._price_history[-1000:]

        self._tick_history_prices.append(current_price)
        volume, amount = self._aggregate_tick_trades(tick_trades)
        self._tick_history_volumes.append(volume)
        self._tick_history_amounts.append(amount)
        if len(self._tick_history_prices) > 100:
            self._tick_history_prices = self._tick_history_prices[-100:]
            self._tick_history_volumes = self._tick_history_volumes[-100:]
            self._tick_history_amounts = self._tick_history_amounts[-100:]

        self._check_catfish_liquidation(current_price)

        tick_timings["total"] = time.perf_counter() - tick_start
        return tick_timings

    def _prepare_adl_candidates(
        self, latest_price: float, agents_to_liquidate_ids: set[int]
    ) -> None:
        """准备 ADL 候选清单"""
        from src.market.adl.adl_manager import ADLCandidate

        self._adl_long_candidates = []
        self._adl_short_candidates = []

        for agent in self.agent_execution_order:
            if agent.is_liquidated:
                continue
            if agent.agent_id in agents_to_liquidate_ids:
                continue

            if self.adl_manager is None:
                continue
            candidate = self.adl_manager.calculate_adl_score(agent, latest_price)
            if candidate is None:
                continue

            if candidate.pnl_percent <= 0:
                continue

            if candidate.position_qty > 0:
                self._adl_long_candidates.append(candidate)
            else:
                self._adl_short_candidates.append(candidate)

        for catfish in self.catfish_list:
            if catfish.is_liquidated:
                continue

            position_qty = catfish.account.position.quantity
            if position_qty == 0:
                continue

            equity = catfish.account.get_equity(latest_price)
            pnl_percent = (
                equity - catfish.account.initial_balance
            ) / catfish.account.initial_balance

            if pnl_percent <= 0:
                continue

            position_value = abs(position_qty) * latest_price
            effective_leverage = position_value / equity if equity > 0 else 0.0
            adl_score = pnl_percent * effective_leverage

            candidate = ADLCandidate(
                participant=catfish,
                position_qty=position_qty,
                pnl_percent=pnl_percent,
                effective_leverage=effective_leverage,
                adl_score=adl_score,
            )

            if position_qty > 0:
                self._adl_long_candidates.append(candidate)
            else:
                self._adl_short_candidates.append(candidate)

        self._adl_long_candidates.sort(key=lambda c: c.adl_score, reverse=True)
        self._adl_short_candidates.sort(key=lambda c: c.adl_score, reverse=True)

    def evolve_with_timing(self) -> EvolutionTimings:
        """带详细计时的进化"""
        evolution_timings = EvolutionTimings()
        if self.matching_engine is None:
            return evolution_timings
        current_price = self.matching_engine._orderbook.last_price

        for agent_type, population in self.populations.items():
            if isinstance(population, RetailSubPopulationManager):
                eval_time = 0.0
                cleanup_time = 0.0
                neat_time = 0.0
                create_time = 0.0

                t0 = time.perf_counter()
                for sub_pop in population.sub_populations:
                    agent_fitnesses = sub_pop.evaluate(current_price)
                    for agent, fitness in agent_fitnesses:
                        genome = agent.brain.get_genome()
                        genome.fitness = fitness
                eval_time = time.perf_counter() - t0

                from concurrent.futures import ThreadPoolExecutor, as_completed
                from neat.population import CompleteExtinctionException
                import neat

                t_neat_start = time.perf_counter()

                def evolve_sub_pop(sub_pop: Any) -> tuple[int, float, float]:
                    sub_id = sub_pop.sub_population_id or 0
                    t0 = time.perf_counter()

                    old_genomes = list(sub_pop.neat_pop.population.values())

                    def eval_genomes(genomes: list, config: Any) -> None:
                        pass

                    try:
                        sub_pop.neat_pop.run(eval_genomes, n=1)
                    except (RuntimeError, CompleteExtinctionException):
                        sub_pop._cleanup_genome_internals(old_genomes)
                        sub_pop._reset_neat_population()
                        return (sub_id, time.perf_counter() - t0, 0.0)

                    t_neat = time.perf_counter() - t0

                    new_genome_ids = set(sub_pop.neat_pop.population.keys())
                    old_to_clean = [g for g in old_genomes if g.key not in new_genome_ids]
                    sub_pop._cleanup_genome_internals(old_to_clean)
                    del old_genomes, old_to_clean

                    sub_pop.generation += 1
                    sub_pop._cleanup_neat_history()

                    t_update_start = time.perf_counter()
                    new_genomes = list(sub_pop.neat_pop.population.items())
                    for idx, (gid, genome) in enumerate(new_genomes):
                        if idx < len(sub_pop.agents):
                            sub_pop.agents[idx].update_brain(genome, sub_pop.neat_config)
                    t_update = time.perf_counter() - t_update_start

                    return (sub_id, t_neat, t_update)

                parallel_start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(evolve_sub_pop, sp) for sp in population.sub_populations]
                    for future in as_completed(futures):
                        _, _, _ = future.result()
                parallel_time = time.perf_counter() - parallel_start

                neat_time = parallel_time * 0.9
                create_time = parallel_time * 0.1
                cleanup_time = 0.0

                evolution_timings.evaluate[agent_type] = eval_time
                evolution_timings.cleanup_old_agents[agent_type] = cleanup_time
                evolution_timings.neat_run[agent_type] = neat_time
                evolution_timings.create_agents[agent_type] = create_time
            else:
                t0 = time.perf_counter()
                agent_fitnesses = population.evaluate(current_price)
                for agent, fitness in agent_fitnesses:
                    genome = agent.brain.get_genome()
                    genome.fitness = fitness
                evolution_timings.evaluate[agent_type] = time.perf_counter() - t0

                t0 = time.perf_counter()
                old_genomes = list(population.neat_pop.population.values())
                evolution_timings.cleanup_old_agents[agent_type] = time.perf_counter() - t0

                t0 = time.perf_counter()

                def eval_genomes(genomes: list, config: Any) -> None:
                    pass

                try:
                    population.neat_pop.run(eval_genomes, n=1)
                except Exception:
                    population._cleanup_genome_internals(old_genomes)
                    del old_genomes
                    population._reset_neat_population()
                    evolution_timings.neat_run[agent_type] = time.perf_counter() - t0
                    evolution_timings.create_agents[agent_type] = 0.0
                    continue

                evolution_timings.neat_run[agent_type] = time.perf_counter() - t0

                new_genome_ids = set(population.neat_pop.population.keys())
                old_genomes_to_clean = [
                    g for g in old_genomes if g.key not in new_genome_ids
                ]
                population._cleanup_genome_internals(old_genomes_to_clean)
                del old_genomes
                del old_genomes_to_clean

                population.generation += 1
                population._cleanup_neat_history()

                t0 = time.perf_counter()
                new_genomes = list(population.neat_pop.population.items())
                for idx, (gid, genome) in enumerate(new_genomes):
                    if idx < len(population.agents):
                        population.agents[idx].update_brain(genome, population.neat_config)
                evolution_timings.create_agents[agent_type] = time.perf_counter() - t0

        t0 = time.perf_counter()
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        evolution_timings.gc_time = time.perf_counter() - t0

        self._register_all_agents()
        self._build_agent_map()
        self._build_execution_order()
        self._update_pop_total_counts()

        t0 = time.perf_counter()
        self._update_network_caches()
        evolution_timings.cache_update_time = time.perf_counter() - t0

        return evolution_timings

    def sample_inference_times(self, n_samples: int = 1000) -> list[float]:
        """随机采样 agent 的 brain.forward() 耗时"""
        if not self.matching_engine:
            return []

        orderbook = self.matching_engine._orderbook
        market_state = self._compute_normalized_market_state()

        active_agents = [a for a in self.agent_execution_order if not a.is_liquidated]
        if not active_agents:
            return []

        samples: list[float] = []
        for _ in range(n_samples):
            agent = random.choice(active_agents)
            inputs = agent.observe(market_state, orderbook)

            t0 = time.perf_counter()
            _ = agent.brain.forward(inputs)
            elapsed = time.perf_counter() - t0
            samples.append(elapsed)

        return samples

    def run_benchmark_episode(self, episode_length: int) -> tuple[EpisodeTimings, EvolutionTimings]:
        """运行一个带基准测试的 episode"""
        if not self.matching_engine:
            return EpisodeTimings(), EvolutionTimings()

        episode_timing = EpisodeTimings()
        self.episode += 1

        t0 = time.perf_counter()
        for population in self.populations.values():
            population.reset_agents()
        episode_timing.agent_reset = time.perf_counter() - t0

        for catfish in self.catfish_list:
            catfish.reset()
        self._catfish_liquidated = False

        t0 = time.perf_counter()
        self._reset_market()
        episode_timing.market_reset = time.perf_counter() - t0

        self.tick = 0
        self._pop_liquidated_counts.clear()
        self._eliminating_agents.clear()

        initial_price = self.config.market.initial_price
        self._episode_high_price = initial_price
        self._episode_low_price = initial_price

        tick_loop_start = time.perf_counter()
        tick_stats: dict[str, TimingStats] = defaultdict(TimingStats)

        for _ in range(episode_length):
            tick_timings = self.run_tick_with_timing()

            for key, value in tick_timings.items():
                tick_stats[key].add(value)

            if self._catfish_liquidated:
                break

            early_end = self._should_end_episode_early()
            if early_end is not None:
                break

        episode_timing.tick_loop_total = time.perf_counter() - tick_loop_start

        episode_timing.tick_liquidation_check = tick_stats["liquidation_check"].total
        episode_timing.tick_market_state = tick_stats["market_state"].total
        episode_timing.tick_parallel_decide = tick_stats["parallel_decide"].total
        episode_timing.tick_serial_execute = tick_stats["serial_execute"].total

        accounted = (
            episode_timing.tick_liquidation_check
            + tick_stats["liquidation_execute"].total
            + tick_stats["catfish"].total
            + episode_timing.tick_market_state
            + episode_timing.tick_parallel_decide
            + episode_timing.tick_serial_execute
        )
        episode_timing.tick_other = episode_timing.tick_loop_total - accounted

        evolution_start = time.perf_counter()
        if self.tick >= 10:
            evolution_timing = self.evolve_with_timing()
        else:
            evolution_timing = EvolutionTimings()
        episode_timing.evolution_total = time.perf_counter() - evolution_start

        return episode_timing, evolution_timing


def run_arena_mode(args: argparse.Namespace) -> dict[str, Any]:
    """运行单训练场基准测试"""
    config = create_default_config(
        episode_length=args.episode_length,
        checkpoint_interval=0,
        catfish_enabled=False,
    )

    total_agents = sum(cfg.count for cfg in config.agents.values())

    print("=" * 60)
    print("单训练场性能基准测试")
    print("=" * 60)
    print(f"配置: {total_agents:,} agents, {args.episode_length} ticks/episode, {args.episodes} episodes")
    print()

    print("初始化中...")
    init_start = time.perf_counter()
    trainer = BenchmarkTrainer(config)
    trainer.setup()
    trainer.is_running = True
    init_time = time.perf_counter() - init_start
    print(f"初始化完成: {init_time:.2f}s")
    print()

    episode_timings: list[EpisodeTimings] = []
    evolution_timings: list[EvolutionTimings] = []
    inference_times: list[float] = []

    for ep in range(args.episodes):
        print(f"Episode {ep + 1}/{args.episodes}")

        episode_timing, evolution_timing = trainer.run_benchmark_episode(args.episode_length)
        episode_timings.append(episode_timing)
        evolution_timings.append(evolution_timing)

        tick_total = episode_timing.tick_loop_total
        if tick_total > 0:
            tick_per_tick_ms = tick_total / args.episode_length * 1000
        else:
            tick_per_tick_ms = 0

        print(f"  Tick 循环: {tick_total:.2f}s (平均 {tick_per_tick_ms:.1f}ms/tick)")

        if tick_total > 0:
            print(f"    - 并行决策: {episode_timing.tick_parallel_decide:.2f}s "
                  f"({episode_timing.tick_parallel_decide / tick_total * 100:.1f}%)")
            print(f"    - 串行执行: {episode_timing.tick_serial_execute:.2f}s "
                  f"({episode_timing.tick_serial_execute / tick_total * 100:.1f}%)")
            print(f"    - 市场状态: {episode_timing.tick_market_state:.2f}s "
                  f"({episode_timing.tick_market_state / tick_total * 100:.1f}%)")
            print(f"    - 其他: {episode_timing.tick_other:.2f}s "
                  f"({episode_timing.tick_other / tick_total * 100:.1f}%)")

        evo_total = episode_timing.evolution_total
        print(f"  进化: {evo_total:.2f}s")

        if ep == args.episodes - 1:
            print(f"  采样推理时间 (N={args.inference_samples})...")
            inference_times = trainer.sample_inference_times(args.inference_samples)

    # 汇总统计
    tick_stats: dict[str, TimingStats] = {
        "parallel_decide": TimingStats(),
        "serial_execute": TimingStats(),
        "market_state": TimingStats(),
        "liquidation_check": TimingStats(),
        "other": TimingStats(),
        "total": TimingStats(),
    }

    for et in episode_timings:
        n_ticks = args.episode_length
        if et.tick_loop_total > 0 and n_ticks > 0:
            tick_stats["parallel_decide"].add(et.tick_parallel_decide / n_ticks * 1000)
            tick_stats["serial_execute"].add(et.tick_serial_execute / n_ticks * 1000)
            tick_stats["market_state"].add(et.tick_market_state / n_ticks * 1000)
            tick_stats["liquidation_check"].add(et.tick_liquidation_check / n_ticks * 1000)
            tick_stats["other"].add(et.tick_other / n_ticks * 1000)
            tick_stats["total"].add(et.tick_loop_total / n_ticks * 1000)

    print()
    print("=" * 60)
    print("汇总统计")
    print("=" * 60)
    print()

    print("Tick 阶段平均耗时 (每 tick):")
    total_mean = tick_stats["total"].mean
    for key, label in [
        ("parallel_decide", "并行决策"),
        ("serial_execute", "串行执行"),
        ("market_state", "市场状态"),
        ("liquidation_check", "强平检查"),
        ("other", "其他"),
    ]:
        stats = tick_stats[key]
        if stats.count > 0:
            pct = stats.mean / total_mean * 100 if total_mean > 0 else 0
            print(f"  {label}: {stats.mean:.2f}ms +/- {stats.std:.2f}ms ({pct:.1f}%)")
    print(f"  总计: {total_mean:.2f}ms +/- {tick_stats['total'].std:.2f}ms")
    print()

    if inference_times:
        inference_arr = np.array(inference_times) * 1_000_000
        print(f"单次推理采样 (N={len(inference_times)}):")
        print(f"  平均: {np.mean(inference_arr):.2f}us +/- {np.std(inference_arr):.2f}us")
        print(f"  P50: {np.percentile(inference_arr, 50):.2f}us")
        print(f"  P95: {np.percentile(inference_arr, 95):.2f}us")
        print(f"  P99: {np.percentile(inference_arr, 99):.2f}us")
        print()

    trainer.stop()

    # 构建结果字典
    result: dict[str, Any] = {
        "config": {
            "episodes": args.episodes,
            "episode_length": args.episode_length,
            "inference_samples": args.inference_samples,
        },
        "init_time_s": init_time,
        "tick_stats_ms": {},
        "inference_stats_us": {},
    }

    for key, stats in tick_stats.items():
        if stats.count > 0:
            result["tick_stats_ms"][key] = {
                "mean": stats.mean,
                "std": stats.std,
                "count": stats.count,
            }

    if inference_times:
        inference_arr = np.array(inference_times) * 1_000_000
        result["inference_stats_us"] = {
            "mean": float(np.mean(inference_arr)),
            "std": float(np.std(inference_arr)),
            "p50": float(np.percentile(inference_arr, 50)),
            "p95": float(np.percentile(inference_arr, 95)),
            "p99": float(np.percentile(inference_arr, 99)),
            "count": len(inference_times),
        }

    return result


# ============================================================================
# 模式 3: Evolution 模式 - 进化方法对比测试
# ============================================================================

def set_random_fitness(manager: SubPopulationManager) -> None:
    """为所有Agent设置随机适应度"""
    for pop in manager.sub_populations:
        for agent in pop.agents:
            agent.brain.get_genome().fitness = np.random.random()


def benchmark_serial(manager: SubPopulationManager, iterations: int) -> float:
    """测试串行进化方案"""
    times: list[float] = []

    for i in range(iterations):
        set_random_fitness(manager)

        start = time.perf_counter()
        manager.evolve(current_price=10000.0)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  串行进化 iteration {i+1}: {elapsed:.2f}s")

    return sum(times) / len(times)


def benchmark_parallel_simple(manager: SubPopulationManager, iterations: int) -> float:
    """测试简化并行进化方案"""
    times: list[float] = []

    for i in range(iterations):
        set_random_fitness(manager)

        start = time.perf_counter()
        manager.evolve_parallel_simple(current_price=10000.0, max_workers=10)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  简化并行 iteration {i+1}: {elapsed:.2f}s")

    return sum(times) / len(times)


def benchmark_original_parallel(manager: SubPopulationManager, iterations: int) -> float:
    """测试原始进程池并行进化方案"""
    times: list[float] = []

    for i in range(iterations):
        set_random_fitness(manager)

        start = time.perf_counter()
        manager.evolve_parallel(current_price=10000.0, max_workers=10)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  进程池并行 iteration {i+1}: {elapsed:.2f}s")

    return sum(times) / len(times)


def benchmark_network_params_lazy(
    manager: SubPopulationManager,
    worker_pool: PersistentWorkerPool,
    iterations: int,
) -> float:
    """测试网络参数传输方案（延迟反序列化）"""
    times: list[float] = []

    for i in range(iterations):
        set_random_fitness(manager)

        start = time.perf_counter()
        sync_needed = (i == 0)
        manager.evolve_parallel_with_network_params(
            current_price=10000.0,
            worker_pool=worker_pool,
            sync_genomes=sync_needed,
            deserialize_genomes=False,
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Worker池+延迟反序列化 iteration {i+1}: {elapsed:.2f}s")

    return sum(times) / len(times)


def run_evolution_mode(args: argparse.Namespace) -> dict[str, Any]:
    """运行进化方法对比测试"""
    print("=" * 70)
    print("RETAIL子种群进化方案完整性能测试")
    print("=" * 70)

    config = create_default_config()
    config.agents[AgentType.RETAIL].count = 10000
    sub_count = 10

    agents_per_sub = config.agents[AgentType.RETAIL].count // sub_count

    print(f"\n配置：")
    print(f"  - 子种群数量: {sub_count}")
    print(f"  - 每个子种群 Agent 数: {agents_per_sub}")
    print(f"  - 总 Agent 数: {config.agents[AgentType.RETAIL].count}")

    print("\n创建 SubPopulationManager...")
    create_start = time.perf_counter()
    manager = SubPopulationManager(config, AgentType.RETAIL, sub_count=sub_count)
    create_time = time.perf_counter() - create_start
    print(f"  创建耗时: {create_time:.2f}s")

    neat_config_path = manager.sub_populations[0].neat_config_path
    pop_size = len(manager.sub_populations[0].agents)

    print(f"  - NEAT 配置路径: {neat_config_path}")
    print(f"  - 每个 Worker 的种群大小: {pop_size}")

    results: dict[str, float] = {}

    # 测试1：串行进化
    print("\n" + "-" * 50)
    print("测试1: 串行进化 (evolve) - Baseline")
    print("-" * 50)
    avg_serial = benchmark_serial(manager, args.iterations)
    results["串行进化"] = avg_serial
    print(f"\n串行进化平均耗时: {avg_serial:.2f}s")

    # 测试2：简化并行进化
    print("\n" + "-" * 50)
    print("测试2: 简化并行进化 (evolve_parallel_simple) - 推荐")
    print("-" * 50)
    avg_simple = benchmark_parallel_simple(manager, args.iterations)
    results["简化并行"] = avg_simple
    print(f"\n简化并行平均耗时: {avg_simple:.2f}s")

    # 测试3：进程池并行进化
    print("\n" + "-" * 50)
    print("测试3: 进程池并行进化 (evolve_parallel)")
    print("-" * 50)
    avg_process = benchmark_original_parallel(manager, args.iterations)
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
    avg_lazy = benchmark_network_params_lazy(manager, worker_pool, args.iterations)
    results["Worker池+延迟"] = avg_lazy
    print(f"\nWorker池+延迟反序列化平均耗时: {avg_lazy:.2f}s")

    worker_pool.shutdown()

    # 输出对比结果
    print("\n" + "=" * 70)
    print("对比结果 (以串行进化为基准)")
    print("=" * 70)

    baseline = results["串行进化"]
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

    best_name, best_time = sorted_results[0]
    print(f"\n最优方案: {best_name} ({best_time:.2f}s)")

    return {
        "results": results,
        "baseline": baseline,
        "best": {"name": best_name, "time": best_time},
    }


# ============================================================================
# 模式 4: Detailed 模式 - 详细瓶颈剖析
# ============================================================================

def profile_serial_evolve(manager: SubPopulationManager) -> dict[str, float]:
    """剖析串行进化的各阶段耗时"""
    print("\n" + "=" * 60)
    print("串行进化详细剖析 (单个子种群)")
    print("=" * 60)

    pop = manager.sub_populations[0]

    for agent in pop.agents:
        agent.brain.get_genome().fitness = np.random.random()

    # evaluate
    start = time.perf_counter()
    agent_fitnesses = pop.evaluate(10000.0)
    eval_time = time.perf_counter() - start
    print(f"  evaluate(): {eval_time:.3f}s")

    # 设置适应度
    start = time.perf_counter()
    for agent, fitness in agent_fitnesses:
        genome = agent.brain.get_genome()
        genome.fitness = fitness
    set_fitness_time = time.perf_counter() - start
    print(f"  设置适应度: {set_fitness_time:.3f}s")

    # NEAT进化
    import neat

    def dummy_eval(genomes, config):
        pass

    start = time.perf_counter()
    pop.neat_pop.run(dummy_eval, n=1)
    neat_time = time.perf_counter() - start
    print(f"  NEAT进化: {neat_time:.3f}s")

    # 清理历史
    start = time.perf_counter()
    pop._cleanup_neat_history()
    cleanup_time = time.perf_counter() - start
    print(f"  清理历史: {cleanup_time:.3f}s")

    # 创建新Agent
    start = time.perf_counter()
    new_genomes = list(pop.neat_pop.population.items())
    pop.agents = pop.create_agents(new_genomes)
    create_time = time.perf_counter() - start
    print(f"  创建Agent: {create_time:.3f}s")

    total = eval_time + set_fitness_time + neat_time + cleanup_time + create_time
    print(f"\n  总计: {total:.3f}s")

    return {
        "evaluate": eval_time,
        "set_fitness": set_fitness_time,
        "neat": neat_time,
        "cleanup": cleanup_time,
        "create": create_time,
        "total": total,
    }


def profile_worker_pool_evolve(manager: SubPopulationManager) -> dict[str, Any]:
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

    total_size = 0
    for gd in genomes_list:
        for arr in gd:
            total_size += arr.nbytes
    print(f"   序列化数据大小: {total_size / 1024 / 1024:.2f} MB")

    start = time.perf_counter()
    worker_pool.set_all_genomes(genomes_list)
    transfer_time = time.perf_counter() - start
    print(f"   传输到Worker耗时: {transfer_time:.3f}s")

    for pop in manager.sub_populations:
        for agent in pop.agents:
            agent.brain.get_genome().fitness = np.random.random()

    # 评估
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

    # 第二次迭代
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

    print("\n" + "=" * 60)
    print("瓶颈分析总结")
    print("=" * 60)

    return {
        "first_call": {
            "pool_create": pool_create_time,
            "serialize": serialize_time,
            "transfer": transfer_time,
            "evaluate": eval_time,
            "evolve": evolve_time,
            "update": update_time,
        },
        "second_call": {
            "evaluate": eval_time2,
            "evolve": evolve_time2,
            "update": update_time2,
            "total": total2,
        },
    }


def run_detailed_mode(args: argparse.Namespace) -> dict[str, Any]:
    """运行详细瓶颈剖析"""
    print("=" * 70)
    print("RETAIL子种群进化 - 详细性能剖析")
    print("=" * 70)

    config = create_default_config()
    config.agents[AgentType.RETAIL].count = 10000
    sub_count = 10

    print(f"\n配置: {sub_count} 个子种群 × {config.agents[AgentType.RETAIL].count // sub_count} Agent")

    print("\n创建 SubPopulationManager...")
    start = time.perf_counter()
    manager = SubPopulationManager(config, AgentType.RETAIL, sub_count=sub_count)
    create_time = time.perf_counter() - start
    print(f"创建耗时: {create_time:.2f}s")

    serial_result = profile_serial_evolve(manager)
    worker_result = profile_worker_pool_evolve(manager)

    return {
        "serial": serial_result,
        "worker_pool": worker_result,
        "create_time": create_time,
    }


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="统一性能基准测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["profile", "arena", "evolution", "detailed"],
        default="arena",
        help="测试模式: profile(cProfile分析), arena(基准测试), evolution(进化对比), detailed(详细剖析)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="运行的 episode 数量（默认: 5）",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=100,
        help="每个 episode 的 tick 数（默认: 100）",
    )
    parser.add_argument(
        "--inference-samples",
        type=int,
        default=1000,
        help="推理采样数量（默认: 1000）",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="进化测试迭代次数（默认: 3）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径（可选）",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="日志目录（默认: logs）",
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_dir)

    # 根据模式运行相应测试
    result: dict[str, Any] = {}

    if args.mode == "profile":
        result = run_profile_mode(args)
    elif args.mode == "arena":
        result = run_arena_mode(args)
    elif args.mode == "evolution":
        result = run_evolution_mode(args)
    elif args.mode == "detailed":
        result = run_detailed_mode(args)

    # 输出到 JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
