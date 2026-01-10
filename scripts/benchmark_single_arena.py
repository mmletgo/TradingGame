#!/usr/bin/env python3
"""单训练场性能基准测试脚本

用于分析单训练场模式的性能瓶颈，测量各阶段耗时。

使用方法:
    python scripts/benchmark_single_arena.py [选项]

示例:
    # 默认参数运行
    python scripts/benchmark_single_arena.py

    # 自定义参数
    python scripts/benchmark_single_arena.py --episodes 10 --episode-length 50

    # 输出到 JSON 文件
    python scripts/benchmark_single_arena.py --output benchmark_results.json
"""

import argparse
import gc
import importlib
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
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
from src.training.population import RetailSubPopulationManager

from create_config import create_default_config


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


class BenchmarkTrainer(Trainer):
    """带计时功能的 Trainer"""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # 计时数据存储
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.episode_timings: list[EpisodeTimings] = []
        self.evolution_timings: list[EvolutionTimings] = []
        self.inference_samples: list[float] = []

    def run_tick_with_timing(self) -> dict[str, float]:
        """带详细计时的 tick 执行

        Returns:
            各阶段耗时字典
        """
        if not self.matching_engine:
            return {}

        tick_timings: dict[str, float] = {}
        tick_start = time.perf_counter()

        self.tick += 1
        orderbook = self.matching_engine._orderbook

        # 获取当前价格
        current_price = (
            self._smooth_mid_price
            if self._smooth_mid_price > 0
            else orderbook.last_price
        )
        self.tick_start_price = current_price

        # Tick 1: 只记录状态
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

        # === 强平检查 ===
        t0 = time.perf_counter()
        agents_to_liquidate = self._check_liquidations_vectorized(current_price)
        agents_to_liquidate_ids: set[int] = set()
        for agent in agents_to_liquidate:
            agents_to_liquidate_ids.add(agent.agent_id)
            self._cancel_agent_orders(agent)
        tick_timings["liquidation_check"] = time.perf_counter() - t0

        # 强平处理（阶段2和3）
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

            # ADL 处理
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

        # === 市场状态计算 ===
        t0 = time.perf_counter()
        market_state = self._compute_normalized_market_state()
        tick_timings["market_state"] = time.perf_counter() - t0

        # 随机打乱执行顺序
        random.shuffle(self.agent_execution_order)

        # === 并行决策 ===
        t0 = time.perf_counter()
        decisions = self._batch_decide_parallel(
            self.agent_execution_order, market_state, orderbook
        )
        tick_timings["parallel_decide"] = time.perf_counter() - t0

        # === 串行执行 ===
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

        # 记录 tick 历史
        self._tick_history_prices.append(current_price)
        volume, amount = self._aggregate_tick_trades(tick_trades)
        self._tick_history_volumes.append(volume)
        self._tick_history_amounts.append(amount)
        if len(self._tick_history_prices) > 100:
            self._tick_history_prices = self._tick_history_prices[-100:]
            self._tick_history_volumes = self._tick_history_volumes[-100:]
            self._tick_history_amounts = self._tick_history_amounts[-100:]

        # 鲶鱼强平检查
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

        # 将鲶鱼加入候选
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
        """带详细计时的进化

        Returns:
            各阶段耗时数据
        """
        evolution_timings = EvolutionTimings()
        if self.matching_engine is None:
            return evolution_timings
        current_price = self.matching_engine._orderbook.last_price

        for agent_type, population in self.populations.items():
            # 处理 RetailSubPopulationManager
            if isinstance(population, RetailSubPopulationManager):
                # 对子种群管理器，遍历所有子种群
                eval_time = 0.0
                cleanup_time = 0.0
                neat_time = 0.0
                create_time = 0.0

                for sub_pop in population.sub_populations:
                    # 1. 评估适应度
                    t0 = time.perf_counter()
                    agent_fitnesses = sub_pop.evaluate(current_price)
                    for agent, fitness in agent_fitnesses:
                        genome = agent.brain.get_genome()
                        genome.fitness = fitness
                    eval_time += time.perf_counter() - t0

                    # 2. 清理
                    t0 = time.perf_counter()
                    old_genomes = list(sub_pop.neat_pop.population.values())
                    sub_pop._cleanup_old_agents()
                    cleanup_time += time.perf_counter() - t0

                    # 3. NEAT run
                    t0 = time.perf_counter()
                    def eval_genomes(genomes: list, config: Any) -> None:
                        pass
                    try:
                        sub_pop.neat_pop.run(eval_genomes, n=1)
                    except Exception as e:
                        self.logger.warning(f"子种群进化失败: {e}")
                        sub_pop._cleanup_genome_internals(old_genomes)
                        continue
                    neat_time += time.perf_counter() - t0

                    # 清理旧基因组
                    new_genome_ids = set(sub_pop.neat_pop.population.keys())
                    old_to_clean = [g for g in old_genomes if g.key not in new_genome_ids]
                    sub_pop._cleanup_genome_internals(old_to_clean)
                    del old_genomes, old_to_clean

                    sub_pop.generation += 1
                    sub_pop._cleanup_neat_history()

                    # 4. 创建新 Agent
                    t0 = time.perf_counter()
                    new_genomes = list(sub_pop.neat_pop.population.items())
                    sub_pop.agents = sub_pop.create_agents(new_genomes)
                    create_time += time.perf_counter() - t0

                evolution_timings.evaluate[agent_type] = eval_time
                evolution_timings.cleanup_old_agents[agent_type] = cleanup_time
                evolution_timings.neat_run[agent_type] = neat_time
                evolution_timings.create_agents[agent_type] = create_time
            else:
                # 普通 Population
                # 1. 评估适应度
                t0 = time.perf_counter()
                agent_fitnesses = population.evaluate(current_price)
                for agent, fitness in agent_fitnesses:
                    genome = agent.brain.get_genome()
                    genome.fitness = fitness
                evolution_timings.evaluate[agent_type] = time.perf_counter() - t0

                # 2. 保存旧基因组，清理旧 Agent
                t0 = time.perf_counter()
                old_genomes = list(population.neat_pop.population.values())
                population._cleanup_old_agents()
                gc.collect()
                gc.collect()
                evolution_timings.cleanup_old_agents[agent_type] = time.perf_counter() - t0

                # 3. NEAT run
                t0 = time.perf_counter()

                def eval_genomes(
                    genomes: list[tuple[int, Any]], config: Any
                ) -> None:
                    _ = genomes, config

                try:
                    population.neat_pop.run(eval_genomes, n=1)
                except Exception as e:
                    self.logger.warning(f"{agent_type.value} 进化失败: {e}")
                    population._cleanup_genome_internals(old_genomes)
                    del old_genomes
                    gc.collect()
                    population._reset_neat_population()
                    evolution_timings.neat_run[agent_type] = time.perf_counter() - t0
                    evolution_timings.create_agents[agent_type] = 0.0
                    continue

                evolution_timings.neat_run[agent_type] = time.perf_counter() - t0

                # 清理旧基因组
                new_genome_ids = set(population.neat_pop.population.keys())
                old_genomes_to_clean = [
                    g for g in old_genomes if g.key not in new_genome_ids
                ]
                population._cleanup_genome_internals(old_genomes_to_clean)
                del old_genomes
                del old_genomes_to_clean
                gc.collect()

                population.generation += 1
                population._cleanup_neat_history()
                gc.collect()

                # 4. 创建新 Agent
                t0 = time.perf_counter()
                new_genomes = list(population.neat_pop.population.items())
                population.agents = population.create_agents(new_genomes)
                evolution_timings.create_agents[agent_type] = time.perf_counter() - t0

        # GC 时间
        t0 = time.perf_counter()
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        evolution_timings.gc_time = time.perf_counter() - t0

        # 重建映射
        self._register_all_agents()
        self._build_agent_map()
        self._build_execution_order()
        self._update_pop_total_counts()

        return evolution_timings

    def sample_inference_times(self, n_samples: int = 1000) -> list[float]:
        """随机采样 agent 的 brain.forward() 耗时

        Args:
            n_samples: 采样数量

        Returns:
            耗时列表（秒）
        """
        if not self.matching_engine:
            return []

        orderbook = self.matching_engine._orderbook
        market_state = self._compute_normalized_market_state()

        # 收集所有未被淘汰的 agent
        active_agents = [a for a in self.agent_execution_order if not a.is_liquidated]
        if not active_agents:
            return []

        # 随机采样
        samples: list[float] = []
        for _ in range(n_samples):
            agent = random.choice(active_agents)
            # 使用 observe 方法准备输入（所有 Agent 都有这个方法）
            inputs = agent.observe(market_state, orderbook)

            # 计时
            t0 = time.perf_counter()
            _ = agent.brain.forward(inputs)
            elapsed = time.perf_counter() - t0
            samples.append(elapsed)

        return samples

    def run_benchmark_episode(self) -> tuple[EpisodeTimings, EvolutionTimings]:
        """运行一个带基准测试的 episode

        Returns:
            (episode 计时数据, 进化计时数据)
        """
        if not self.matching_engine:
            return EpisodeTimings(), EvolutionTimings()

        episode_timing = EpisodeTimings()
        self.episode += 1
        episode_length = self.config.training.episode_length

        # 1. 重置 Agent
        t0 = time.perf_counter()
        for population in self.populations.values():
            population.reset_agents()
        episode_timing.agent_reset = time.perf_counter() - t0

        # 重置鲶鱼
        for catfish in self.catfish_list:
            catfish.reset()
        self._catfish_liquidated = False

        # 2. 重置市场
        t0 = time.perf_counter()
        self._reset_market()
        episode_timing.market_reset = time.perf_counter() - t0

        # 重置 tick 和计数
        self.tick = 0
        self._pop_liquidated_counts.clear()
        self._eliminating_agents.clear()

        initial_price = self.config.market.initial_price
        self._episode_high_price = initial_price
        self._episode_low_price = initial_price

        # 3. 运行 tick 循环
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

        # 汇总 tick 统计
        episode_timing.tick_liquidation_check = tick_stats["liquidation_check"].total
        episode_timing.tick_market_state = tick_stats["market_state"].total
        episode_timing.tick_parallel_decide = tick_stats["parallel_decide"].total
        episode_timing.tick_serial_execute = tick_stats["serial_execute"].total

        # 计算其他时间
        accounted = (
            episode_timing.tick_liquidation_check
            + tick_stats["liquidation_execute"].total
            + tick_stats["catfish"].total
            + episode_timing.tick_market_state
            + episode_timing.tick_parallel_decide
            + episode_timing.tick_serial_execute
        )
        episode_timing.tick_other = episode_timing.tick_loop_total - accounted

        # 4. 进化
        evolution_start = time.perf_counter()
        min_ticks_for_evolution = 10
        if self.tick >= min_ticks_for_evolution:
            evolution_timing = self.evolve_with_timing()
        else:
            evolution_timing = EvolutionTimings()
        episode_timing.evolution_total = time.perf_counter() - evolution_start

        return episode_timing, evolution_timing


class SingleArenaBenchmarker:
    """单训练场性能基准测试器"""

    def __init__(
        self,
        episodes: int = 5,
        episode_length: int = 100,
        inference_samples: int = 1000,
    ) -> None:
        """初始化基准测试器

        Args:
            episodes: 运行的 episode 数量
            episode_length: 每个 episode 的 tick 数
            inference_samples: 推理采样数量
        """
        self.episodes = episodes
        self.episode_length = episode_length
        self.inference_samples = inference_samples

        # 结果存储
        self.episode_timings: list[EpisodeTimings] = []
        self.evolution_timings: list[EvolutionTimings] = []
        self.inference_times: list[float] = []
        self.init_time: float = 0.0

        # 汇总统计
        self.tick_stats: dict[str, TimingStats] = {}
        self.evolution_stats: dict[str, dict[AgentType, TimingStats]] = {}

    def run(self) -> dict[str, Any]:
        """运行基准测试

        Returns:
            基准测试结果字典
        """
        # 创建配置
        config = create_default_config(
            episode_length=self.episode_length,
            checkpoint_interval=0,  # 不保存检查点
            catfish_enabled=False,
        )

        # 计算 agent 总数
        total_agents = sum(cfg.count for cfg in config.agents.values())

        print("=" * 60)
        print("单训练场性能基准测试")
        print("=" * 60)
        print(f"配置: {total_agents:,} agents, {self.episode_length} ticks/episode, {self.episodes} episodes")
        print()

        # 初始化
        print("初始化中...")
        init_start = time.perf_counter()
        trainer = BenchmarkTrainer(config)
        trainer.setup()
        trainer.is_running = True
        self.init_time = time.perf_counter() - init_start
        print(f"初始化完成: {self.init_time:.2f}s")
        print()

        # 运行 episodes
        for ep in range(self.episodes):
            print(f"Episode {ep + 1}/{self.episodes}")

            episode_timing, evolution_timing = trainer.run_benchmark_episode()
            self.episode_timings.append(episode_timing)
            self.evolution_timings.append(evolution_timing)

            # 打印 episode 结果
            self._print_episode_result(ep + 1, episode_timing, evolution_timing)

            # 在最后一个 episode 采样推理时间
            if ep == self.episodes - 1:
                print(f"  采样推理时间 (N={self.inference_samples})...")
                self.inference_times = trainer.sample_inference_times(
                    self.inference_samples
                )

        # 汇总统计
        print()
        self._compute_summary_stats()
        self._print_summary()

        # 清理
        trainer.stop()

        return self._build_result_dict()

    def _print_episode_result(
        self,
        _ep_num: int,
        episode_timing: EpisodeTimings,
        evolution_timing: EvolutionTimings,
    ) -> None:
        """打印单个 episode 的结果"""
        tick_total = episode_timing.tick_loop_total
        if tick_total > 0:
            tick_per_tick_ms = tick_total / self.episode_length * 1000
        else:
            tick_per_tick_ms = 0

        print(f"  Tick 循环: {tick_total:.2f}s (平均 {tick_per_tick_ms:.1f}ms/tick)")

        # Tick 阶段明细
        if tick_total > 0:
            print(
                f"    - 并行决策: {episode_timing.tick_parallel_decide:.2f}s "
                f"({episode_timing.tick_parallel_decide / tick_total * 100:.1f}%)"
            )
            print(
                f"    - 串行执行: {episode_timing.tick_serial_execute:.2f}s "
                f"({episode_timing.tick_serial_execute / tick_total * 100:.1f}%)"
            )
            print(
                f"    - 市场状态: {episode_timing.tick_market_state:.2f}s "
                f"({episode_timing.tick_market_state / tick_total * 100:.1f}%)"
            )
            print(
                f"    - 强平检查: {episode_timing.tick_liquidation_check:.2f}s "
                f"({episode_timing.tick_liquidation_check / tick_total * 100:.1f}%)"
            )
            print(
                f"    - 其他: {episode_timing.tick_other:.2f}s "
                f"({episode_timing.tick_other / tick_total * 100:.1f}%)"
            )

        # 进化阶段
        evo_total = episode_timing.evolution_total
        print(f"  进化: {evo_total:.2f}s")
        if evo_total > 0:
            neat_run_total = sum(evolution_timing.neat_run.values())
            create_agents_total = sum(evolution_timing.create_agents.values())
            evaluate_total = sum(evolution_timing.evaluate.values())
            cleanup_total = sum(evolution_timing.cleanup_old_agents.values())

            print(
                f"    - NEAT run: {neat_run_total:.2f}s "
                f"({neat_run_total / evo_total * 100:.1f}%)"
            )
            print(
                f"    - Agent创建: {create_agents_total:.2f}s "
                f"({create_agents_total / evo_total * 100:.1f}%)"
            )
            print(
                f"    - 评估: {evaluate_total:.2f}s "
                f"({evaluate_total / evo_total * 100:.1f}%)"
            )
            print(
                f"    - 清理: {cleanup_total:.2f}s "
                f"({cleanup_total / evo_total * 100:.1f}%)"
            )
            print(
                f"    - GC: {evolution_timing.gc_time:.2f}s "
                f"({evolution_timing.gc_time / evo_total * 100:.1f}%)"
            )

        total = tick_total + evo_total
        print(f"  Episode 总耗时: {total:.2f}s")
        print()

    def _compute_summary_stats(self) -> None:
        """计算汇总统计"""
        # Tick 阶段统计
        self.tick_stats = {
            "parallel_decide": TimingStats(),
            "serial_execute": TimingStats(),
            "market_state": TimingStats(),
            "liquidation_check": TimingStats(),
            "other": TimingStats(),
            "total": TimingStats(),
        }

        for et in self.episode_timings:
            n_ticks = self.episode_length
            if et.tick_loop_total > 0 and n_ticks > 0:
                self.tick_stats["parallel_decide"].add(
                    et.tick_parallel_decide / n_ticks * 1000
                )
                self.tick_stats["serial_execute"].add(
                    et.tick_serial_execute / n_ticks * 1000
                )
                self.tick_stats["market_state"].add(
                    et.tick_market_state / n_ticks * 1000
                )
                self.tick_stats["liquidation_check"].add(
                    et.tick_liquidation_check / n_ticks * 1000
                )
                self.tick_stats["other"].add(et.tick_other / n_ticks * 1000)
                self.tick_stats["total"].add(et.tick_loop_total / n_ticks * 1000)

        # 进化阶段统计
        self.evolution_stats = {
            "neat_run": {at: TimingStats() for at in AgentType},
            "create_agents": {at: TimingStats() for at in AgentType},
            "evaluate": {at: TimingStats() for at in AgentType},
            "cleanup": {at: TimingStats() for at in AgentType},
        }

        for evt in self.evolution_timings:
            for at in AgentType:
                if at in evt.neat_run:
                    self.evolution_stats["neat_run"][at].add(evt.neat_run[at])
                if at in evt.create_agents:
                    self.evolution_stats["create_agents"][at].add(evt.create_agents[at])
                if at in evt.evaluate:
                    self.evolution_stats["evaluate"][at].add(evt.evaluate[at])
                if at in evt.cleanup_old_agents:
                    self.evolution_stats["cleanup"][at].add(
                        evt.cleanup_old_agents[at]
                    )

    def _print_summary(self) -> None:
        """打印汇总统计"""
        print("=" * 60)
        print("汇总统计")
        print("=" * 60)
        print()

        # Tick 阶段平均耗时
        print("Tick 阶段平均耗时 (每 tick):")
        total_mean = self.tick_stats["total"].mean
        for key in ["parallel_decide", "serial_execute", "market_state", "liquidation_check", "other"]:
            stats = self.tick_stats[key]
            if stats.count > 0:
                pct = stats.mean / total_mean * 100 if total_mean > 0 else 0
                print(f"  {key}: {stats.mean:.2f}ms +/- {stats.std:.2f}ms ({pct:.1f}%)")
        print(f"  总计: {total_mean:.2f}ms +/- {self.tick_stats['total'].std:.2f}ms")
        print()

        # 进化阶段平均耗时
        print("进化阶段平均耗时:")
        for phase_name in ["neat_run", "create_agents", "evaluate", "cleanup"]:
            phase_stats = self.evolution_stats[phase_name]
            total_time = sum(s.mean for s in phase_stats.values())
            print(f"  {phase_name}:")
            for at in AgentType:
                s = phase_stats[at]
                if s.count > 0:
                    print(f"    {at.value}: {s.mean:.3f}s +/- {s.std:.3f}s")
            print(f"    合计: {total_time:.3f}s")
        print()

        # 推理采样统计
        if self.inference_times:
            inference_arr = np.array(self.inference_times) * 1_000_000  # 转换为微秒
            print(f"单次推理采样 (N={len(self.inference_times)}):")
            print(f"  平均: {np.mean(inference_arr):.2f}us +/- {np.std(inference_arr):.2f}us")
            print(f"  P50: {np.percentile(inference_arr, 50):.2f}us")
            print(f"  P95: {np.percentile(inference_arr, 95):.2f}us")
            print(f"  P99: {np.percentile(inference_arr, 99):.2f}us")
            print()

    def _build_result_dict(self) -> dict[str, Any]:
        """构建结果字典"""
        result: dict[str, Any] = {
            "config": {
                "episodes": self.episodes,
                "episode_length": self.episode_length,
                "inference_samples": self.inference_samples,
            },
            "init_time_s": self.init_time,
            "tick_stats_ms": {},
            "evolution_stats_s": {},
            "inference_stats_us": {},
        }

        # Tick 统计
        for key, stats in self.tick_stats.items():
            if stats.count > 0:
                result["tick_stats_ms"][key] = {
                    "mean": stats.mean,
                    "std": stats.std,
                    "count": stats.count,
                }

        # 进化统计
        for phase_name, phase_stats in self.evolution_stats.items():
            result["evolution_stats_s"][phase_name] = {}
            for at, s in phase_stats.items():
                if s.count > 0:
                    result["evolution_stats_s"][phase_name][at.value] = {
                        "mean": s.mean,
                        "std": s.std,
                        "count": s.count,
                    }

        # 推理统计
        if self.inference_times:
            inference_arr = np.array(self.inference_times) * 1_000_000
            result["inference_stats_us"] = {
                "mean": float(np.mean(inference_arr)),
                "std": float(np.std(inference_arr)),
                "p50": float(np.percentile(inference_arr, 50)),
                "p95": float(np.percentile(inference_arr, 95)),
                "p99": float(np.percentile(inference_arr, 99)),
                "count": len(self.inference_times),
            }

        return result


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="单训练场性能基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    # 运行基准测试
    benchmarker = SingleArenaBenchmarker(
        episodes=args.episodes,
        episode_length=args.episode_length,
        inference_samples=args.inference_samples,
    )

    result = benchmarker.run()

    # 输出到 JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
