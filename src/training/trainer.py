"""训练器模块

管理训练流程，协调种群和撮合引擎。
"""

import gc
import pickle
import random
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.bio.agents.base import AgentType


def _get_memory_mb() -> float:
    """获取当前进程的内存使用量（MB）"""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:
        pass
    return 0.0


if TYPE_CHECKING:
    from src.bio.agents.base import ActionType, Agent
    from src.market.catfish.catfish_base import CatfishBase
    from src.market.market_state import (
        NormalizedMarketState as NormalizedMarketStateType,
    )
    from src.market.orderbook.orderbook import OrderBook
    from src.training.generation_saver import GenerationSaver
from src.config.config import Config
from src.core.log_engine.logger import get_logger
from src.market.adl.adl_manager import ADLCandidate, ADLManager
from src.market.catfish import CatfishBase, create_all_catfish, create_catfish
from src.market.market_state import NormalizedMarketState
from src.market.matching.matching_engine import MatchingEngine
from src.market.matching.trade import Trade
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.training.population import Population, malloc_trim


class Trainer:
    """训练器

    管理 NEAT 进化训练流程，协调四种群在模拟市场中竞争。

    Attributes:
        config: 全局配置
        populations: 四种群（散户/高级散户/庄家/做市商）
        matching_engine: 撮合引擎
        tick: 当前 tick
        episode: 当前 episode
        is_running: 是否运行中
        is_paused: 是否暂停
        recent_trades: 最近成交记录
    """

    config: Config
    populations: dict[AgentType, Population]
    matching_engine: MatchingEngine | None
    adl_manager: ADLManager | None
    tick: int
    episode: int
    is_running: bool
    is_paused: bool
    recent_trades: deque[Trade]
    agent_map: dict[int, "Agent"]
    agent_execution_order: list["Agent"]
    _pop_total_counts: dict[AgentType, int]  # 各种群总数
    _pop_liquidated_counts: dict[
        AgentType, int
    ]  # 各种群当前 episode 已淘汰数量（爆仓即淘汰）
    tick_start_price: float  # tick 开始时的价格，供数据采集使用
    _eliminating_agents: set[int]  # 正在强平/淘汰的 Agent ID 集合（防止重入）
    _adl_long_candidates: list[
        ADLCandidate
    ]  # 多头 ADL 候选（已排序，持仓数量会动态更新）
    _adl_short_candidates: list[
        ADLCandidate
    ]  # 空头 ADL 候选（已排序，持仓数量会动态更新）
    _executor: ThreadPoolExecutor | None
    _num_workers: int
    catfish_list: list["CatfishBase"]
    _price_history: list[float]
    _tick_history_prices: list[float]
    _tick_history_volumes: list[float]
    _tick_history_amounts: list[float]
    _episode_high_price: float  # 当前 episode 最高价
    _episode_low_price: float  # 当前 episode 最低价
    arena_id: int | None  # 竞技场 ID（多竞技场场景）
    _generation_saver: "GenerationSaver | None"

    def __init__(self, config: Config, arena_id: int | None = None) -> None:
        """创建训练器

        Args:
            config: 全局配置对象
            arena_id: 竞技场 ID（多竞技场场景，默认 None）
        """
        self.config = config
        self.arena_id = arena_id
        self.logger = get_logger("trainer")

        self.tick = 0
        self.episode = 0
        self.is_running = False
        self.is_paused = False

        self.populations = {}
        self.matching_engine = None
        self.adl_manager = None
        self.recent_trades = deque(maxlen=100)
        self.agent_map = {}
        self.agent_execution_order = []
        self._pop_total_counts = {}
        self._pop_liquidated_counts = {}
        self.tick_start_price = 0.0
        self._eliminating_agents = set()
        self._adl_long_candidates = []
        self._adl_short_candidates = []
        self._executor = None
        self._num_workers = 16

        # EMA 平滑价格相关
        self._smooth_mid_price: float = 0.0
        self._ema_alpha: float = 0.1

        # 鲶鱼相关
        self.catfish_list: list["CatfishBase"] = []
        self._price_history: list[float] = []
        self._catfish_liquidated: bool = False  # 鲶鱼是否被强平（触发 episode 结束）

        # Tick 历史数据（用于 Agent 输入特征）
        self._tick_history_prices: list[float] = []  # 每 tick 价格
        self._tick_history_volumes: list[float] = []  # 每 tick 成交量（带方向）
        self._tick_history_amounts: list[float] = []  # 每 tick 成交额（带方向）

        # Episode 价格统计
        self._episode_high_price: float = 0.0
        self._episode_low_price: float = 0.0

        # 每代保存器
        self._generation_saver: "GenerationSaver | None" = None

    def set_generation_saver(self, saver: "GenerationSaver") -> None:
        """设置每代保存器

        设置后，每次进化完成会自动保存该代的最佳基因组。

        Args:
            saver: GenerationSaver 实例
        """
        self._generation_saver = saver

    def _init_ema_price(self, initial_price: float) -> None:
        """初始化 EMA 平滑价格

        在 episode 开始时调用，使用初始价格作为 EMA 的初始值。

        Args:
            initial_price: 初始价格
        """
        self._smooth_mid_price = initial_price

    def _update_ema_price(self, current_mid_price: float) -> float:
        """更新 EMA 平滑价格

        公式: smooth = alpha * current + (1 - alpha) * prev_smooth

        Args:
            current_mid_price: 当前实时 mid_price

        Returns:
            更新后的 smooth_mid_price
        """
        self._smooth_mid_price = (
            self._ema_alpha * current_mid_price
            + (1 - self._ema_alpha) * self._smooth_mid_price
        )
        return self._smooth_mid_price

    def _aggregate_tick_trades(self, tick_trades: list[Trade]) -> tuple[float, float]:
        """聚合本 tick 的成交量和成交额（符号+总量方式）

        统计本 tick 所有成交的 taker 买入和 taker 卖出。
        如果买入成交额 > 卖出成交额，结果为 +总量
        如果卖出成交额 > 买入成交额，结果为 -总量
        如果相等，结果为 0

        Returns:
            (带方向的成交量, 带方向的成交额)
        """
        if not tick_trades:
            return 0.0, 0.0

        buy_volume = sum(t.quantity for t in tick_trades if t.is_buyer_taker)
        sell_volume = sum(t.quantity for t in tick_trades if not t.is_buyer_taker)
        buy_amount = sum(t.price * t.quantity for t in tick_trades if t.is_buyer_taker)
        sell_amount = sum(
            t.price * t.quantity for t in tick_trades if not t.is_buyer_taker
        )

        total_volume = buy_volume + sell_volume
        total_amount = buy_amount + sell_amount

        if buy_amount > sell_amount:
            return float(total_volume), total_amount
        elif sell_amount > buy_amount:
            return float(-total_volume), -total_amount
        return 0.0, 0.0

    def _setup_populations_from_checkpoint(
        self, populations_data: dict[AgentType, dict]
    ) -> None:
        """从检查点数据直接创建种群

        直接从检查点的 NEAT 种群数据创建 Agent，避免先创建全新 Agent 再清理的开销。

        Args:
            populations_data: 检查点中的种群数据
                格式: {agent_type: {"generation": int, "neat_pop": neat.Population}}
        """
        for agent_type in AgentType:
            if agent_type not in populations_data:
                # 如果检查点中没有此种群，创建全新种群
                self.populations[agent_type] = Population(agent_type, self.config)
                continue

            pop_data = populations_data[agent_type]

            # 创建 Population 对象但不初始化 NEAT 种群（使用检查点数据）
            pop = Population.__new__(Population)
            pop.agent_type = agent_type
            pop.agent_config = self.config.agents[agent_type]
            pop.generation = pop_data["generation"]
            pop.logger = get_logger("population")
            pop._executor = None
            pop._num_workers = 8

            # 加载 NEAT 配置
            from pathlib import Path
            import neat

            config_dir = Path(self.config.training.neat_config_path)
            if agent_type == AgentType.MARKET_MAKER:
                neat_config_path = config_dir / "neat_market_maker.cfg"
            elif agent_type == AgentType.WHALE:
                neat_config_path = config_dir / "neat_whale.cfg"
            elif agent_type == AgentType.RETAIL_PRO:
                neat_config_path = config_dir / "neat_retail_pro.cfg"
            else:
                neat_config_path = config_dir / "neat_retail.cfg"

            pop.neat_config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                str(neat_config_path),
            )
            pop.neat_config.pop_size = pop.agent_config.count

            # 直接使用检查点中的 NEAT 种群
            pop.neat_pop = pop_data["neat_pop"]

            # 从 NEAT 种群创建 Agent
            genomes = list(pop.neat_pop.population.items())
            pop.agents = pop.create_agents(genomes)

            self.populations[agent_type] = pop

            self.logger.info(
                f"从检查点恢复 {agent_type.value} 种群，"
                f"代数: {pop.generation}，Agent 数量: {len(pop.agents)}"
            )

    def _calculate_catfish_initial_balance(self) -> float:
        """计算每条鲶鱼的初始资金

        公式：鲶鱼总资金 = 做市商杠杆后资金 - 其他物种杠杆后资金
        每条鲶鱼分配 1/3

        Returns:
            每条鲶鱼的初始资金

        Raises:
            ValueError: 如果其他物种资金 >= 做市商资金
        """
        agents_config = self.config.agents

        # 做市商杠杆后资金
        mm_config = agents_config[AgentType.MARKET_MAKER]
        mm_fund = mm_config.count * mm_config.initial_balance * mm_config.leverage

        # 其他物种杠杆后资金
        other_fund = 0.0
        for agent_type in [AgentType.RETAIL, AgentType.RETAIL_PRO, AgentType.WHALE]:
            cfg = agents_config[agent_type]
            other_fund += cfg.count * cfg.initial_balance * cfg.leverage

        # 校验资金配置
        if other_fund >= mm_fund:
            raise ValueError(
                f"鲶鱼资金配置错误：做市商杠杆后资金({mm_fund})必须大于"
                f"其他物种杠杆后资金({other_fund})"
            )

        # 每条鲶鱼资金（3条鲶鱼）
        catfish_count = 3
        per_catfish = (mm_fund - other_fund) / catfish_count
        return per_catfish

    def _get_executor(self) -> ThreadPoolExecutor:
        """获取或创建线程池（惰性初始化）"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._num_workers, thread_name_prefix="neat_worker"
            )
        return self._executor

    def _shutdown_executor(self) -> None:
        """关闭线程池"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def setup(self, checkpoint: dict | None = None) -> None:
        """初始化训练环境

        创建四种群、撮合引擎，初始化市场。
        训练模式使用直接调用。

        Args:
            checkpoint: 可选的检查点数据，如果提供则直接从检查点恢复种群
                       而不是创建全新种群（避免不必要的内存分配）
        """
        # 串行创建四种群
        if checkpoint is not None and "populations" in checkpoint:
            # 直接从检查点恢复，不创建全新 Agent
            self._setup_populations_from_checkpoint(checkpoint["populations"])
            self.tick = checkpoint.get("tick", 0)
            self.episode = checkpoint.get("episode", 0)
        else:
            # 创建全新种群（每个种群内部的Agent创建是并行的）
            for agent_type in AgentType:
                self.populations[agent_type] = Population(agent_type, self.config)

        # 创建撮合引擎
        self.matching_engine = MatchingEngine(self.config.market)

        # 创建 ADL 管理器
        self.adl_manager = ADLManager()

        # 初始化鲶鱼（如果配置中启用）
        if self.config.catfish and self.config.catfish.enabled:
            # 计算鲶鱼初始资金
            catfish_initial_balance = self._calculate_catfish_initial_balance()

            # 鲶鱼使用庄家的杠杆和维持保证金率
            whale_config = self.config.agents[AgentType.WHALE]
            catfish_leverage = whale_config.leverage
            catfish_mmr = whale_config.maintenance_margin_rate

            if self.config.catfish.multi_mode:
                # 多模式：同时创建三种鲶鱼
                self.catfish_list = create_all_catfish(
                    self.config.catfish,
                    initial_balance=catfish_initial_balance,
                    leverage=catfish_leverage,
                    maintenance_margin_rate=catfish_mmr,
                )
                for catfish in self.catfish_list:
                    self.matching_engine.register_agent(
                        catfish.catfish_id,
                        0.0,  # maker fee
                        0.0,  # taker fee
                    )
                self.logger.info(
                    f"鲶鱼已启用: 多模式（三种鲶鱼同时运行，相位错开）, "
                    f"每条初始资金={catfish_initial_balance/1e8:.2f}亿"
                )
            else:
                # 单模式：只创建一种鲶鱼
                catfish = create_catfish(
                    -1,
                    self.config.catfish,
                    initial_balance=catfish_initial_balance,
                    leverage=catfish_leverage,
                    maintenance_margin_rate=catfish_mmr,
                )
                self.catfish_list = [catfish]
                self.matching_engine.register_agent(
                    catfish.catfish_id,
                    0.0,  # maker fee
                    0.0,  # taker fee
                )
                self.logger.info(
                    f"鲶鱼已启用: 模式={self.config.catfish.mode.value}, "
                    f"初始资金={catfish_initial_balance/1e8:.2f}亿"
                )

        # 注册所有 Agent 的费率
        self._register_all_agents()

        # 构建 Agent 映射表和执行顺序
        self._build_agent_map()
        self._build_execution_order()

        # 记录各种群总数
        self._update_pop_total_counts()

        # 读取 EMA 配置并初始化
        self._ema_alpha = self.config.market.ema_alpha
        self._init_ema_price(self.config.market.initial_price)

        # 初始化市场（做市商先行动）
        self._init_market()

        self.logger.info("训练环境初始化完成")

    def _register_all_agents(self) -> None:
        """注册所有 Agent 的费率到撮合引擎

        遍历所有种群的 Agent，将其费率信息注册到撮合引擎。
        应在 setup() 后和 evolve() 后调用。
        """
        if not self.matching_engine:
            return

        for agent_type, population in self.populations.items():
            agent_config = population.agent_config
            for agent in population.agents:
                self.matching_engine.register_agent(
                    agent.agent_id,
                    agent_config.maker_fee_rate,
                    agent_config.taker_fee_rate,
                )

    def _build_agent_map(self) -> None:
        """构建 Agent ID 到 Agent 对象的映射表"""
        self.agent_map.clear()
        for population in self.populations.values():
            for agent in population.agents:
                self.agent_map[agent.agent_id] = agent

    def _build_execution_order(self) -> None:
        """构建 Agent 执行顺序列表（做市商 -> 庄家 -> 高级散户 -> 散户）"""
        self.agent_execution_order.clear()
        for agent_type in [
            AgentType.MARKET_MAKER,
            AgentType.WHALE,
            AgentType.RETAIL_PRO,
            AgentType.RETAIL,
        ]:
            population = self.populations.get(agent_type)
            if population:
                self.agent_execution_order.extend(population.agents)

    def _update_pop_total_counts(self) -> None:
        """更新各种群总数"""
        for agent_type, population in self.populations.items():
            self._pop_total_counts[agent_type] = len(population.agents)

    def _update_agents_after_migration(
        self,
        replaced_info: dict[AgentType, list[tuple[int, "Agent", "Agent"]]],
    ) -> None:
        """迁移后增量更新内部状态

        仅更新被替换的 Agent 相关状态，避免重建整个 agent_map 和 agent_execution_order。

        Args:
            replaced_info: {agent_type: [(idx, old_agent, new_agent), ...]}
        """
        if not self.matching_engine:
            return

        # 1. 遍历所有被替换的 agent
        for agent_type, replacements in replaced_info.items():
            for idx, old_agent, new_agent in replacements:
                # 2. 更新 agent_map：移除旧 agent 的映射，添加新 agent 的映射
                if old_agent.agent_id in self.agent_map:
                    del self.agent_map[old_agent.agent_id]
                self.agent_map[new_agent.agent_id] = new_agent

                # 3. 更新 agent_execution_order：找到旧 agent 的位置，替换为新 agent
                try:
                    exec_idx = self.agent_execution_order.index(old_agent)
                    self.agent_execution_order[exec_idx] = new_agent
                except ValueError:
                    # 旧 agent 不在执行顺序中（可能已被移除），添加新 agent
                    self.agent_execution_order.append(new_agent)

                # 4. 注册新 agent 的费率到撮合引擎
                agent_config = self.populations[agent_type].agent_config
                self.matching_engine.register_agent(
                    new_agent.agent_id,
                    agent_config.maker_fee_rate,
                    agent_config.taker_fee_rate,
                )

    def _mark_agent_liquidated(self, agent_id: int) -> None:
        """标记 Agent 已被强平

        Args:
            agent_id: 被强平的 Agent ID
        """
        agent = self.agent_map.get(agent_id)
        if agent and not agent.is_liquidated:
            agent.is_liquidated = True
            # 根据类型输出不同的日志
            type_name = {
                AgentType.RETAIL: "散户",
                AgentType.RETAIL_PRO: "高级散户",
                AgentType.WHALE: "庄家",
                AgentType.MARKET_MAKER: "做市商",
            }.get(agent.agent_type, str(agent.agent_type.value))

            self.logger.info(
                f"{type_name} Agent {agent_id} 已被强平，本轮 episode 禁用"
            )

    def _cancel_agent_orders(self, agent: "Agent") -> None:
        """撤销 agent 的所有挂单

        Args:
            agent: 要撤销挂单的 Agent
        """
        if not self.matching_engine:
            return

        # 做市商有多个挂单（bid_order_ids 和 ask_order_ids），需要全部撤销
        if agent.agent_type == AgentType.MARKET_MAKER:
            from src.bio.agents.market_maker import MarketMakerAgent

            if isinstance(agent, MarketMakerAgent):
                agent._cancel_all_orders(self.matching_engine)
        elif agent.account.pending_order_id is not None:
            self.matching_engine.cancel_order(agent.account.pending_order_id)
            agent.account.pending_order_id = None

    def _execute_liquidation_market_order(self, agent: "Agent") -> tuple[int, bool]:
        """执行强平市价单（不含 ADL）

        只执行市价单平仓，不执行 ADL。用于三阶段强平流程中的阶段2。

        Args:
            agent: 被强平的 Agent

        Returns:
            (剩余未平仓数量, 是否为多头)
        """
        position_qty = agent.account.position.quantity
        if position_qty == 0:
            return 0, True

        # 记录原始持仓方向
        is_long = position_qty > 0
        target_qty = abs(position_qty)

        # 创建市价平仓单
        side = OrderSide.SELL if is_long else OrderSide.BUY
        order = Order(
            order_id=agent._generate_order_id(),
            agent_id=agent.agent_id,
            side=side,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=target_qty,
        )

        # 直接撮合
        trades = self.matching_engine.process_order(order)

        # 更新账户（taker 和 maker）
        for trade in trades:
            is_buyer = trade.is_buyer_taker
            agent.account.on_trade(trade, is_buyer)
            self.recent_trades.append(trade)
            # 更新 maker 的账户
            maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
            maker_agent = self.agent_map.get(maker_id)
            if maker_agent is not None:
                maker_is_buyer = not trade.is_buyer_taker
                maker_agent.account.on_trade(trade, maker_is_buyer)

        # 返回剩余未平仓数量和方向
        remaining_qty = abs(agent.account.position.quantity)
        return remaining_qty, is_long

    def _execute_adl(
        self,
        liquidated_agent: "Agent",
        remaining_qty: int,
        current_price: float,
        is_long: bool,
    ) -> None:
        """执行 ADL 自动减仓

        使用 tick 开始时预计算的候选清单，避免重复计算 ADL 分数。
        ADL 成交后更新候选清单中的 position_qty，确保后续 ADL 不会重复使用已减掉的仓位。

        Args:
            liquidated_agent: 被强平的 Agent
            remaining_qty: 剩余需要平仓的数量
            current_price: 当前市场价格
            is_long: 被强平方是否为多头
        """
        if not self.adl_manager:
            return

        # ADL 成交价格：直接使用当前市场价格
        adl_price = self.adl_manager.get_adl_price(current_price)

        # 选择对应方向的预计算候选清单
        # 被强平方是多头（需要卖出平仓），则需要空头对手
        # 被强平方是空头（需要买入平仓），则需要多头对手
        candidates = (
            self._adl_short_candidates if is_long else self._adl_long_candidates
        )

        # self.logger.info(
        #     f"ADL 触发: Agent {liquidated_agent.agent_id} "
        #     f"剩余平仓量 {remaining_qty}, "
        #     f"成交价 {adl_price:.2f}, "
        #     f"候选人数 {len(candidates)}"
        # )

        for candidate in candidates:
            if remaining_qty <= 0:
                break
            # 使用候选清单中的 position_qty（已被之前的 ADL 更新过）
            # 同时也要检查实际仓位，取两者最小值
            candidate_available_qty = abs(candidate.position_qty)
            actual_position = abs(candidate.participant.account.position.quantity)
            available_qty = min(candidate_available_qty, actual_position)

            liquidated_actual_position = abs(liquidated_agent.account.position.quantity)
            trade_qty = min(available_qty, remaining_qty, liquidated_actual_position)

            if trade_qty <= 0:
                continue

            # 更新账户
            liquidated_agent.account.on_adl_trade(trade_qty, adl_price, is_taker=True)
            candidate.participant.account.on_adl_trade(
                trade_qty, adl_price, is_taker=False
            )

            # 更新候选清单中的 position_qty，确保后续 ADL 不会重复使用
            if candidate.position_qty > 0:
                candidate.position_qty -= trade_qty
            else:
                candidate.position_qty += trade_qty

            # self.logger.info(
            #     f"ADL 成交: Agent {liquidated_agent.agent_id} 与候选者 "
            #     f"成交 {trade_qty} @ {adl_price:.2f}, "
            #     f"候选剩余持仓 {candidate.position_qty}"
            # )

            remaining_qty -= trade_qty

        # if remaining_qty > 0:
        #     self.logger.warning(
        #         f"ADL 未能完全平仓: Agent {liquidated_agent.agent_id} "
        #         f"剩余 {remaining_qty} 无法匹配 "
        #         f"(盈利对手候选数={len(candidates)}, 将由系统兜底清零)"
        #     )

        # 兜底处理：确保 liquidated_agent 的仓位清零
        actual_remaining = abs(liquidated_agent.account.position.quantity)
        if actual_remaining > 0:
            # self.logger.warning(
            #     f"ADL 兜底清零: Agent {liquidated_agent.agent_id} "
            #     f"实际剩余仓位 {liquidated_agent.account.position.quantity}, "
            #     f"强制清零"
            # )
            liquidated_agent.account.position.quantity = 0
            liquidated_agent.account.position.avg_price = 0.0
        if liquidated_agent.account.balance < 0:
            liquidated_agent.account.balance = 0.0

    def _should_end_episode_early(self) -> tuple[str, AgentType | None] | None:
        """检查是否满足提前结束 episode 的条件（O(1) 复杂度）

        触发条件：
        - 任意种群的存活数量少于初始值的 1/4
        - 订单簿只有单边挂单（只有 bid 或只有 ask）

        这确保每个种群都有足够的个体用于 NEAT 进化，以及市场流动性正常。

        Returns:
            tuple[str, AgentType | None] | None: (原因, 种群类型)，如果没有则返回 None
        """
        # 检查种群存活数量
        for agent_type, total in self._pop_total_counts.items():
            if total > 0:
                liquidated = self._pop_liquidated_counts.get(agent_type, 0)
                alive = total - liquidated
                # 任意种群存活少于初始值的 1/4 时触发早停
                if alive < total / 4:
                    return ("population_depleted", agent_type)

        # 检查订单簿单边挂单
        orderbook = self.matching_engine._orderbook
        has_bids = orderbook.get_best_bid() is not None
        has_asks = orderbook.get_best_ask() is not None
        if has_bids != has_asks:  # 只有一边有挂单
            return ("one_sided_orderbook", None)

        return None

    def _log_market_maker_status(self) -> None:
        """调试日志：输出做市商状态，帮助排查单边挂单问题"""
        mm_population = self.populations.get(AgentType.MARKET_MAKER)
        if not mm_population:
            self.logger.warning("没有做市商种群")
            return

        orderbook = self.matching_engine._orderbook
        current_price = orderbook.last_price

        self.logger.warning("=== 做市商状态调试 ===")
        alive_count = 0
        total_long_pos = 0
        total_short_pos = 0
        leverage_full_count = 0

        for agent in mm_population.agents:
            if agent.is_liquidated:
                continue

            alive_count += 1
            equity = agent.account.get_equity(current_price)
            position_qty = agent.account.position.quantity
            position_value = abs(position_qty) * current_price
            max_pos_value = equity * agent.account.leverage

            if position_qty > 0:
                total_long_pos += position_qty
            elif position_qty < 0:
                total_short_pos += abs(position_qty)

            # 检查杠杆上限
            leverage_full = (
                position_value >= max_pos_value * 0.99 if max_pos_value > 0 else False
            )
            if leverage_full:
                leverage_full_count += 1

            # 检查订单状态
            bid_count = len(agent.bid_order_ids)
            ask_count = len(agent.ask_order_ids)

            # 只输出问题做市商（杠杆满或无双边订单）
            if leverage_full or bid_count == 0 or ask_count == 0:
                self.logger.warning(
                    f"  MM {agent.agent_id}: "
                    f"pos={position_qty}, equity={equity:.0f}, "
                    f"pos_value={position_value:.0f}, max_pos={max_pos_value:.0f}, "
                    f"杠杆满={leverage_full}, "
                    f"bid_orders={bid_count}, ask_orders={ask_count}"
                )

        # 统计被强平的做市商
        liquidated_count = sum(1 for a in mm_population.agents if a.is_liquidated)

        self.logger.warning(
            f"存活做市商: {alive_count}/{len(mm_population.agents)}, "
            f"被强平: {liquidated_count}, 杠杆满: {leverage_full_count}"
        )
        self.logger.warning(
            f"多头总仓位: {total_long_pos}, 空头总仓位: {total_short_pos}, "
            f"净仓位: {total_long_pos - total_short_pos}"
        )

        # 输出订单簿状态
        best_bid = orderbook.get_best_bid()
        best_ask = orderbook.get_best_ask()
        bid_volume = (
            sum(pl.total_quantity for pl in orderbook.bids.values())
            if orderbook.bids
            else 0
        )
        ask_volume = (
            sum(pl.total_quantity for pl in orderbook.asks.values())
            if orderbook.asks
            else 0
        )
        self.logger.warning(
            f"订单簿: best_bid={best_bid}, best_ask={best_ask}, "
            f"bid_volume={bid_volume}, ask_volume={ask_volume}, "
            f"last_price={current_price}"
        )

        # 输出其他物种的净仓位
        other_long = 0
        other_short = 0
        for agent_type in [AgentType.RETAIL, AgentType.RETAIL_PRO, AgentType.WHALE]:
            pop = self.populations.get(agent_type)
            if pop:
                for agent in pop.agents:
                    if not agent.is_liquidated:
                        qty = agent.account.position.quantity
                        if qty > 0:
                            other_long += qty
                        elif qty < 0:
                            other_short += abs(qty)
        self.logger.warning(
            f"其他物种仓位: 多头={other_long}, 空头={other_short}, "
            f"净仓位={other_long - other_short}"
        )

    def _compute_normalized_market_state(self) -> NormalizedMarketState:
        """预计算归一化的公共市场数据

        在每个 tick 开始时调用，避免每个 Agent 重复计算相同的归一化数据。
        使用 EMA 平滑后的 mid_price 作为参考价格，减缓价格变化传导速度。

        Returns:
            NormalizedMarketState: 归一化后的市场状态
        """
        orderbook = self.matching_engine._orderbook

        # 获取实时参考价格
        current_mid_price = orderbook.get_mid_price()
        if current_mid_price is None:
            current_mid_price = orderbook.last_price
        if current_mid_price == 0:
            current_mid_price = 100.0

        # 更新 EMA 平滑价格，用于 Agent 报价和归一化计算
        smooth_mid_price = self._update_ema_price(current_mid_price)

        tick_size = orderbook.tick_size
        depth = orderbook.get_depth(levels=100)

        # 向量化买盘：100档 × 2 = 200（使用平滑价格归一化）
        bid_data = np.zeros(200, dtype=np.float32)
        bids = depth["bids"]
        if bids:
            bid_prices = np.array([p for p, _ in bids], dtype=np.float32)
            bid_qtys = np.array([q for _, q in bids], dtype=np.float32)
            n = len(bids)
            if smooth_mid_price > 0:
                bid_data[0 : n * 2 : 2] = (
                    bid_prices - smooth_mid_price
                ) / smooth_mid_price
            # 数量使用对数归一化：log10(qty + 1) / 10，将 1e10 压缩到 ~1.0
            bid_data[1 : n * 2 : 2] = np.log10(bid_qtys + 1) / 10.0

        # 向量化卖盘：100档 × 2 = 200（使用平滑价格归一化）
        ask_data = np.zeros(200, dtype=np.float32)
        asks = depth["asks"]
        if asks:
            ask_prices = np.array([p for p, _ in asks], dtype=np.float32)
            ask_qtys = np.array([q for _, q in asks], dtype=np.float32)
            n = len(asks)
            if smooth_mid_price > 0:
                ask_data[0 : n * 2 : 2] = (
                    ask_prices - smooth_mid_price
                ) / smooth_mid_price
            # 数量使用对数归一化
            ask_data[1 : n * 2 : 2] = np.log10(ask_qtys + 1) / 10.0

        # 向量化成交：100笔（数量带方向：正=taker买入，负=taker卖出）
        trade_prices = np.zeros(100, dtype=np.float32)
        trade_quantities = np.zeros(100, dtype=np.float32)
        trades = list(self.recent_trades)  # deque 转换为 list
        if trades:
            prices = np.array([t.price for t in trades], dtype=np.float32)
            qtys = np.array(
                [t.quantity if t.is_buyer_taker else -t.quantity for t in trades],
                dtype=np.float32,
            )
            n = len(trades)
            if smooth_mid_price > 0:
                trade_prices[:n] = (prices - smooth_mid_price) / smooth_mid_price
            # 成交数量带方向的对数归一化：sign(qty) * log10(|qty| + 1) / 10
            trade_quantities[:n] = np.sign(qtys) * np.log10(np.abs(qtys) + 1) / 10.0

        # Tick 历史价格归一化（以第一个 tick 价格为基准）
        tick_prices_normalized = np.zeros(100, dtype=np.float32)
        tick_volumes_normalized = np.zeros(100, dtype=np.float32)
        tick_amounts_normalized = np.zeros(100, dtype=np.float32)

        if self._tick_history_prices:
            hist_prices = np.array(self._tick_history_prices[-100:], dtype=np.float32)
            volumes = np.array(self._tick_history_volumes[-100:], dtype=np.float32)
            amounts = np.array(self._tick_history_amounts[-100:], dtype=np.float32)
            n = len(hist_prices)

            # 价格归一化：以第一个 tick 价格为基准
            base_price = hist_prices[0]
            if base_price > 0:
                tick_prices_normalized[-n:] = (hist_prices - base_price) / base_price

            # 成交量归一化：sign(vol) * log10(|vol| + 1) / 10
            tick_volumes_normalized[-n:] = (
                np.sign(volumes) * np.log10(np.abs(volumes) + 1) / 10.0
            )

            # 成交额归一化：sign(amt) * log10(|amt| + 1) / 12
            tick_amounts_normalized[-n:] = (
                np.sign(amounts) * np.log10(np.abs(amounts) + 1) / 12.0
            )

        return NormalizedMarketState(
            mid_price=smooth_mid_price,  # 使用 EMA 平滑后的价格
            tick_size=tick_size,
            bid_data=bid_data,
            ask_data=ask_data,
            trade_prices=trade_prices,
            trade_quantities=trade_quantities,
            tick_history_prices=tick_prices_normalized,
            tick_history_volumes=tick_volumes_normalized,
            tick_history_amounts=tick_amounts_normalized,
        )

    def _evolve_populations_parallel(self, current_price: float) -> None:
        """并行进化所有种群

        关键优化：
        1. 串行进化每个种群，避免并行导致的内存峰值
        2. 每个种群进化后立即 GC，释放旧对象
        3. 清理 futures 引用，避免持有已完成任务的引用
        """
        # [MEMORY] 记录进化开始前的内存
        mem_before_evolve = _get_memory_mb()

        # 改为串行进化，每个种群进化后立即 GC，避免内存峰值
        # 并行进化会导致所有旧 genome 同时存在于内存中
        for pop in self.populations.values():
            try:
                pop.evolve(current_price)
            except Exception as e:
                self.logger.error(f"种群 {pop.agent_type.value} 进化失败: {e}")
                raise

            # 每个种群进化后立即 GC
            gc.collect()

        # [MEMORY] 记录进化后、最终 GC 前的内存
        mem_after_evolve = _get_memory_mb()

        # 最终全面 GC，确保所有临时对象被回收
        # 包括 Python 2 代和 3 代垃圾
        gc.collect(0)  # 年轻代
        gc.collect(1)  # 中年代
        gc.collect(2)  # 老年代

        # 【关键】调用 malloc_trim 将释放的内存归还给操作系统
        # Python 的内存分配器通常不会主动归还内存，导致 VmRSS 持续增长
        malloc_trim()

        # [MEMORY] 记录 GC 后的内存
        mem_after_gc = _get_memory_mb()

        # 输出内存变化统计
        arena_tag = f"Arena-{self.arena_id}" if self.arena_id is not None else "Trainer"
        self.logger.info(
            f"[MEMORY_EVOLVE_PARALLEL] {arena_tag} ep_{self.episode}: "
            f"evolve={mem_after_evolve - mem_before_evolve:+.1f}MB, "
            f"gc_released={mem_after_evolve - mem_after_gc:.1f}MB, "
            f"net={mem_after_gc - mem_before_evolve:+.1f}MB"
        )

    def _evolve_populations_with_cached_fitness(self) -> None:
        """使用缓存适应度进化所有种群

        当 tick 数不足时调用，跳过适应度计算，使用之前 episode 缓存的适应度进行进化。
        这样可以打破"tick 不足 → 不进化 → 行为不变 → tick 不足"的死循环。
        """
        for agent_type, population in self.populations.items():
            try:
                success = population.evolve_with_cached_fitness()
                if success:
                    self.logger.debug(
                        f"{agent_type.value} 种群使用缓存适应度进化成功"
                    )
                else:
                    self.logger.warning(
                        f"{agent_type.value} 种群没有缓存适应度，跳过本次进化"
                    )
            except Exception as e:
                self.logger.error(
                    f"种群 {agent_type.value} 缓存进化失败: {e}"
                )

            # 每个种群进化后立即 GC
            gc.collect()

        # 最终全面 GC
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        malloc_trim()

    def _batch_decide_parallel(
        self,
        agents: list["Agent"],
        market_state: "NormalizedMarketStateType",
        orderbook: "OrderBook",
        timeout: float = 60.0,  # 单个 tick 决策超时时间（秒）
    ) -> list[tuple["Agent", "ActionType", dict[str, Any]]]:
        """并行执行所有 Agent 的决策

        Args:
            agents: Agent 列表
            market_state: 归一化市场状态
            orderbook: 订单簿
            timeout: 超时时间（秒），防止死锁

        Returns:
            决策结果列表
        """
        executor = self._get_executor()

        future_to_idx: dict[
            Future[tuple["ActionType", dict[str, Any]]], tuple[int, "Agent"]
        ] = {}
        for idx, agent in enumerate(agents):
            if not agent.is_liquidated:
                future = executor.submit(agent.decide, market_state, orderbook)
                future_to_idx[future] = (idx, agent)

        results: list[tuple["Agent", "ActionType", dict[str, Any]] | None] = [
            None
        ] * len(agents)

        # 使用超时机制防止死锁
        try:
            for future in as_completed(future_to_idx, timeout=timeout):
                idx, agent = future_to_idx[future]
                try:
                    action, params = future.result(timeout=5.0)  # 单个结果也有超时
                    results[idx] = (agent, action, params)
                except TimeoutError:
                    self.logger.warning(f"Agent {agent.agent_id} 决策获取结果超时")
                except Exception as e:
                    self.logger.warning(f"Agent {agent.agent_id} 决策异常: {e}")
        except TimeoutError:
            # 整体超时，记录警告并取消未完成的任务
            arena_tag = f"Arena-{self.arena_id}" if self.arena_id is not None else ""
            self.logger.error(
                f"{arena_tag} tick {self.tick} 决策并行执行超时 ({timeout}s)，"
                f"可能存在死锁"
            )
            # 取消未完成的 future
            for future in future_to_idx:
                if not future.done():
                    future.cancel()

        return [r for r in results if r is not None]

    def _check_liquidations_vectorized(self, current_price: float) -> list["Agent"]:
        """向量化检查所有 Agent 的强平条件"""
        active_agents = [a for a in self.agent_execution_order if not a.is_liquidated]
        n = len(active_agents)

        if n == 0:
            return []

        balances = np.array(
            [a.account.balance for a in active_agents], dtype=np.float64
        )
        quantities = np.array(
            [a.account.position.quantity for a in active_agents], dtype=np.float64
        )
        avg_prices = np.array(
            [a.account.position.avg_price for a in active_agents], dtype=np.float64
        )
        maintenance_rates = np.array(
            [a.account.maintenance_margin_rate for a in active_agents], dtype=np.float64
        )

        unrealized_pnl = (current_price - avg_prices) * quantities
        equities = balances + unrealized_pnl
        position_values = np.abs(quantities) * current_price

        # 使用 np.divide 的 where 参数避免除零警告
        margin_ratios = np.divide(
            equities,
            position_values,
            out=np.full_like(equities, np.inf),
            where=position_values > 0,
        )

        need_liquidation = margin_ratios < maintenance_rates
        return [active_agents[i] for i in range(n) if need_liquidation[i]]

    def _init_market(self) -> None:
        """初始化市场（直接调用模式）

        只有做市商先行动，建立初始流动性。

        每个做市商按顺序行动，后一个做市商看到的是前一个做市商
        下单后的市场状态。
        """
        if not self.matching_engine:
            return

        orderbook = self.matching_engine._orderbook

        mm_population = self.populations.get(AgentType.MARKET_MAKER)
        if mm_population:
            for agent in mm_population.agents:
                # 每个做市商决策前重新计算市场状态，确保看到最新的订单簿
                market_state = self._compute_normalized_market_state()
                action, params = agent.decide(market_state, orderbook)
                # 直接执行（绕过事件系统）
                trades = agent.execute_action(action, params, self.matching_engine)
                for trade in trades:
                    self.recent_trades.append(trade)
                    # 更新 maker 的账户（即使自成交也需要更新）
                    maker_id = (
                        trade.seller_id if trade.is_buyer_taker else trade.buyer_id
                    )
                    maker_agent = self.agent_map.get(maker_id)
                    if maker_agent is not None:
                        # maker 的方向与 taker 相反
                        is_buyer = not trade.is_buyer_taker
                        maker_agent.account.on_trade(trade, is_buyer)

                # 市场初始化阶段不检查淘汰（此时价格还未稳定）

    def _reset_market(self) -> None:
        """重置市场状态

        清空订单簿，做市商重新建立流动性。
        """
        if not self.matching_engine:
            return

        # 清空订单簿并重置最新价
        self.matching_engine._orderbook.clear(
            reset_price=self.config.market.initial_price
        )

        # 清空最近成交
        self.recent_trades.clear()

        # 重置价格历史
        self._price_history.clear()
        self._price_history.append(self.config.market.initial_price)

        # 重置 tick 历史数据
        self._tick_history_prices.clear()
        self._tick_history_volumes.clear()
        self._tick_history_amounts.clear()
        self._tick_history_prices.append(self.config.market.initial_price)
        self._tick_history_volumes.append(0.0)
        self._tick_history_amounts.append(0.0)

        # 重置 EMA 平滑价格
        self._init_ema_price(self.config.market.initial_price)

        # 做市商重新初始化
        self._init_market()

        # 调试：检查初始化后的仓位平衡
        total_position = 0
        long_qty = 0
        short_qty = 0
        for population in self.populations.values():
            for agent in population.agents:
                qty = agent.account.position.quantity
                total_position += qty
                if qty > 0:
                    long_qty += qty
                elif qty < 0:
                    short_qty += qty
        if total_position != 0:
            self.logger.error(
                f"初始化后仓位不对等！总偏差={total_position}, 多头={long_qty}, 空头={short_qty}"
            )

        # 初始化 tick_start_price，供数据采集使用
        self.tick_start_price = self.matching_engine._orderbook.last_price

    def run_tick(self) -> None:
        """执行一个 tick（直接调用模式）

        时序设计：
        1. Tick 1：只展示做市商初始挂单后的市场状态，其他 agent 不行动
        2. Tick 2+：
           - Tick 开始：用 smooth_mid_price 检查所有 agent 的强平条件（爆仓即淘汰）
           - Tick 过程：Agent 按顺序决策和下单
           - Tick 结束：下单效果（价格变动）在下个 tick 被感知

        这样设计确保：
        - UI 模式下第一个可见 tick 是做市商初始挂单后的状态
        - 强平检查和 Agent 报价使用同一价格基准（smooth_mid_price）
        - 所有 agent 在同一价格基础上被检查，公平
        """
        if not self.matching_engine:
            return

        self.tick += 1
        tick_trades: list[Trade] = []  # 收集本 tick 所有成交
        orderbook = self.matching_engine._orderbook

        # 使用 smooth_mid_price 作为强平检查的价格依据，与 Agent 报价逻辑一致
        # 注意：smooth_mid_price 在 _compute_normalized_market_state() 中更新
        # 这里先获取当前的平滑价格用于强平检查
        current_price = (
            self._smooth_mid_price
            if self._smooth_mid_price > 0
            else orderbook.last_price
        )
        self.tick_start_price = current_price  # 保存 tick 开始时的价格

        # === Tick 1：只记录做市商初始挂单后的状态，不执行 agent 行动 ===
        if self.tick == 1:
            # 只记录价格历史和 tick 数据
            self._price_history.append(current_price)
            self._tick_history_prices.append(current_price)
            self._tick_history_volumes.append(0.0)
            self._tick_history_amounts.append(0.0)
            # 更新 episode 价格统计
            if current_price > self._episode_high_price:
                self._episode_high_price = current_price
            if current_price < self._episode_low_price:
                self._episode_low_price = current_price
            return

        # === Tick 2+：正常执行所有 agent 行动 ===
        # === Tick 开始：检查所有 agent 的强平条件（爆仓即淘汰）===
        # 阶段1：向量化收集需要淘汰的 agent，统一撤销挂单
        agents_to_liquidate = self._check_liquidations_vectorized(current_price)
        agents_to_liquidate_ids: set[int] = set()
        for agent in agents_to_liquidate:
            agents_to_liquidate_ids.add(agent.agent_id)
            # 立即撤销挂单，防止在后续平仓过程中作为 maker 被成交
            self._cancel_agent_orders(agent)

        # === 三阶段强平处理 ===
        # 阶段2：统一执行市价单平仓，收集需要 ADL 的 Agent
        agents_need_adl: list[tuple["Agent", int, bool]] = (
            []
        )  # (agent, remaining_qty, is_long)
        if len(agents_to_liquidate) > 0:
            for agent in agents_to_liquidate:
                remaining_qty, is_long = self._execute_liquidation_market_order(agent)
                if remaining_qty > 0:
                    agents_need_adl.append((agent, remaining_qty, is_long))
                # 标记淘汰
                agent.is_liquidated = True
                self._pop_liquidated_counts[agent.agent_type] = (
                    self._pop_liquidated_counts.get(agent.agent_type, 0) + 1
                )
                # 穿仓兜底
                if agent.account.balance < 0:
                    agent.account.balance = 0.0

            # 阶段3：用最新价格计算 ADL 候选并执行
            if agents_need_adl:
                # 获取订单簿最新价格（强平市价单执行后的价格）
                latest_price = orderbook.last_price

                # 按方向计算 ADL 候选清单
                self._adl_long_candidates = []
                self._adl_short_candidates = []
                for agent in self.agent_execution_order:
                    if agent.is_liquidated:
                        continue
                    if agent.agent_id in agents_to_liquidate_ids:
                        continue

                    # 用最新价格计算 ADL 候选信息
                    candidate = self.adl_manager.calculate_adl_score(
                        agent, latest_price
                    )
                    if candidate is None:
                        continue

                    # 只有盈利的 Agent 才能作为 ADL 对手方
                    if candidate.pnl_percent <= 0:
                        continue

                    # 按持仓方向分类
                    if candidate.position_qty > 0:
                        self._adl_long_candidates.append(candidate)
                    else:
                        self._adl_short_candidates.append(candidate)

                # 将鲶鱼加入 ADL 候选列表
                for catfish in self.catfish_list:
                    if catfish.is_liquidated:
                        continue

                    position_qty = catfish.account.position.quantity
                    if position_qty == 0:
                        continue

                    # 计算 ADL 分数
                    equity = catfish.account.get_equity(latest_price)
                    pnl_percent = (
                        equity - catfish.account.initial_balance
                    ) / catfish.account.initial_balance

                    # 只有盈利的才能作为 ADL 候选
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

                # 按 ADL 分数从高到低排序
                self._adl_long_candidates.sort(key=lambda c: c.adl_score, reverse=True)
                self._adl_short_candidates.sort(key=lambda c: c.adl_score, reverse=True)

                # 执行 ADL
                for agent, remaining_qty, is_long in agents_need_adl:
                    self._execute_adl(agent, remaining_qty, latest_price, is_long)
                    # 兜底处理：确保仓位清零
                    if agent.account.position.quantity != 0:
                        self.logger.warning(
                            f"ADL 后仓位未清零: Agent {agent.agent_id}, "
                            f"pos={agent.account.position.quantity}, 强制清零"
                        )
                        agent.account.position.quantity = 0
                        agent.account.position.avg_price = 0.0

        # === 鲶鱼行动（在 Agent 之前）===
        for catfish in self.catfish_list:
            # 检查鲶鱼是否已被强平
            if catfish.is_liquidated:
                continue

            should_act, direction = catfish.decide(
                orderbook,
                self.tick,
                self._price_history,
            )
            if should_act and direction != 0:
                catfish_trades = catfish.execute(direction, self.matching_engine)
                catfish.record_action(self.tick)

                for trade in catfish_trades:
                    self.recent_trades.append(trade)
                    tick_trades.append(trade)  # 收集到本 tick 成交

                    # 更新鲶鱼账户（taker）
                    is_buyer = trade.is_buyer_taker
                    catfish.account.on_trade(trade, is_buyer)

                    # 更新 maker 账户（与鲶鱼成交的对手方）
                    maker_id = (
                        trade.seller_id if trade.is_buyer_taker else trade.buyer_id
                    )
                    if maker_id > 0:
                        maker_agent = self.agent_map.get(maker_id)
                        if maker_agent is not None:
                            maker_is_buyer = not trade.is_buyer_taker
                            maker_agent.account.on_trade(trade, maker_is_buyer)

        # 预计算归一化市场数据
        market_state = self._compute_normalized_market_state()

        # === 随机打乱执行顺序（每个 tick 都不同，模拟真实环境）===
        random.shuffle(self.agent_execution_order)

        # === 并行决策阶段 ===
        decisions = self._batch_decide_parallel(
            self.agent_execution_order, market_state, orderbook
        )

        # === 串行执行阶段 ===
        for agent, action, params in decisions:
            trades = agent.execute_action(action, params, self.matching_engine)

            for trade in trades:
                self.recent_trades.append(trade)
                tick_trades.append(trade)  # 收集到本 tick 成交
                maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
                maker_agent = self.agent_map.get(maker_id)
                if maker_agent is not None:
                    is_buyer = not trade.is_buyer_taker
                    maker_agent.account.on_trade(trade, is_buyer)

        # 记录价格历史（tick 结束时）
        # 使用 last_price 而非 mid_price，确保价格符合 tick_size
        current_price = orderbook.last_price
        self._price_history.append(current_price)

        # 更新 episode 价格统计
        if current_price > self._episode_high_price:
            self._episode_high_price = current_price
        if current_price < self._episode_low_price:
            self._episode_low_price = current_price
        # 限制历史长度（避免内存无限增长）
        if len(self._price_history) > 1000:
            self._price_history = self._price_history[-1000:]

        # 记录 tick 历史数据
        self._tick_history_prices.append(current_price)
        volume, amount = self._aggregate_tick_trades(tick_trades)
        self._tick_history_volumes.append(volume)
        self._tick_history_amounts.append(amount)
        # 限制历史长度（最多 100 条）
        if len(self._tick_history_prices) > 100:
            self._tick_history_prices = self._tick_history_prices[-100:]
            self._tick_history_volumes = self._tick_history_volumes[-100:]
            self._tick_history_amounts = self._tick_history_amounts[-100:]

        # === 鲶鱼强平检查（放在所有 Agent 之后）===
        self._check_catfish_liquidation(current_price)

    def _check_catfish_liquidation(self, current_price: float) -> None:
        """检查鲶鱼强平

        鲶鱼强平后本轮 episode 立即结束，进入进化阶段。

        Args:
            current_price: 当前价格
        """
        for catfish in self.catfish_list:
            if catfish.is_liquidated:
                continue

            if catfish.account.check_liquidation(current_price):
                # 执行鲶鱼强平
                self._execute_catfish_liquidation(catfish, current_price)
                catfish.is_liquidated = True

                # 穿仓兜底
                if catfish.account.balance < 0:
                    catfish.account.balance = 0.0

                # 标记鲶鱼强平（调用方决定是否结束 episode）
                self._catfish_liquidated = True
                self.logger.info(
                    f"鲶鱼 {catfish.__class__.__name__} 被强平 "
                    f"(episode={self.episode}, tick={self.tick})"
                )
                break  # 一条鲶鱼强平就停止检查

    def _execute_catfish_liquidation(
        self, catfish: "CatfishBase", current_price: float
    ) -> None:
        """执行鲶鱼强平

        Args:
            catfish: 被强平的鲶鱼
            current_price: 当前价格
        """
        position_qty = catfish.account.position.quantity
        if position_qty == 0:
            return

        is_long = position_qty > 0
        target_qty = abs(position_qty)

        # 创建市价平仓单
        side = OrderSide.SELL if is_long else OrderSide.BUY
        order = Order(
            order_id=catfish._generate_order_id(),
            agent_id=catfish.catfish_id,
            side=side,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=target_qty,
        )

        # 撮合
        trades = self.matching_engine.match_market_order(order)

        # 更新账户
        for trade in trades:
            is_buyer = trade.is_buyer_taker
            catfish.account.on_trade(trade, is_buyer)
            self.recent_trades.append(trade)

            # 更新 maker 账户
            maker_id = trade.seller_id if trade.is_buyer_taker else trade.buyer_id
            if maker_id > 0:
                maker_agent = self.agent_map.get(maker_id)
                if maker_agent is not None:
                    maker_is_buyer = not trade.is_buyer_taker
                    maker_agent.account.on_trade(trade, maker_is_buyer)

    def run_episode(self) -> None:
        """运行一个 episode

        1. 重置所有 Agent 账户
        2. 运行 episode_length 个 tick
        3. 评估适应度并进化
        """
        if not self.matching_engine:
            return

        self.episode += 1
        episode_length = self.config.training.episode_length

        # 1. 重置所有种群的 Agent 账户
        for population in self.populations.values():
            population.reset_agents()

        # 重置鲶鱼
        for catfish in self.catfish_list:
            catfish.reset()
        self._catfish_liquidated = False  # 重置鲶鱼强平标志

        # 重置市场状态
        self._reset_market()

        # 重置 tick 计数和各种群淘汰计数（每个 episode 从 0 开始）
        self.tick = 0
        self._pop_liquidated_counts.clear()
        self._eliminating_agents.clear()  # 清空重入保护集合

        # 初始化 episode 价格统计
        initial_price = self.config.market.initial_price
        self._episode_high_price = initial_price
        self._episode_low_price = initial_price

        # 2. 运行 episode_length 个 tick
        for _ in range(episode_length):
            if not self.is_running or self.is_paused:
                break
            self.run_tick()

            # 检查鲶鱼是否被强平
            if self._catfish_liquidated:
                self.logger.info(
                    f"Episode {self.episode} 因鲶鱼强平提前结束 (tick={self.tick})"
                )
                break

            # 检查是否满足提前结束条件
            early_end_result = self._should_end_episode_early()
            if early_end_result is not None:
                reason, agent_type = early_end_result
                if reason == "population_depleted" and agent_type is not None:
                    total = self._pop_total_counts[agent_type]
                    liquidated = self._pop_liquidated_counts.get(agent_type, 0)
                    alive = total - liquidated
                    self.logger.warning(
                        f"Episode {self.episode} 提前结束：{agent_type.value} "
                        f"存活不足 1/4 ({alive}/{total}) (tick={self.tick})"
                    )
                elif reason == "one_sided_orderbook":
                    orderbook = self.matching_engine._orderbook
                    side = "只有买盘" if orderbook.get_best_bid() else "只有卖盘"
                    self.logger.warning(
                        f"Episode {self.episode} 提前结束：订单簿{side} (tick={self.tick})"
                    )
                    # 调试日志：输出做市商状态
                    # self._log_market_maker_status()
                break

        # 3. 进化
        # - tick 数 >= 10 时：正常进化（重新计算适应度）
        # - tick 数 < 10 时：使用缓存适应度进化（打破死循环）
        min_ticks_for_evolution = 10
        if self.is_running and not self.is_paused:
            if self.tick >= min_ticks_for_evolution:
                # 正常进化：重新计算适应度 + 选择 + 繁殖
                current_price = self.matching_engine._orderbook.last_price
                self._evolve_populations_parallel(current_price)

                # 保存每代的 best_genome（如果设置了保存器）
                if self._generation_saver is not None:
                    generation = next(iter(self.populations.values())).generation
                    self._generation_saver.save_generation(
                        generation=generation,
                        populations=self.populations,
                        current_price=current_price,
                    )

                # 进化后重新注册新 Agent 的费率，重建映射表和执行顺序
                self._register_all_agents()
                self._build_agent_map()
                self._build_execution_order()
                self._update_pop_total_counts()

                self.logger.info(f"Episode {self.episode} 完成，tick={self.tick}")
            else:
                # tick 不足：使用缓存适应度进化，打破死循环
                self.logger.warning(
                    f"Episode {self.episode} tick 数不足（{self.tick} < {min_ticks_for_evolution}），"
                    f"使用缓存适应度进化"
                )
                self._evolve_populations_with_cached_fitness()

                # 进化后重新注册新 Agent 的费率，重建映射表和执行顺序
                self._register_all_agents()
                self._build_agent_map()
                self._build_execution_order()
                self._update_pop_total_counts()

    def train(
        self,
        episodes: int,
        state_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """主训练循环

        Args:
            episodes: 训练的 episode 数量
            state_callback: 可选的状态回调函数（用于 UI 更新）
        """
        self.is_running = True
        checkpoint_interval = self.config.training.checkpoint_interval

        for ep in range(episodes):
            if not self.is_running:
                break

            self.run_episode()

            # 状态回调
            if state_callback:
                state_callback(self._get_state())

            # 定期保存检查点
            if checkpoint_interval > 0 and (ep + 1) % checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoints/ep_{ep + 1}.pkl")

        self.is_running = False

    def _get_state(self) -> dict[str, Any]:
        """获取当前状态快照"""
        return {
            "tick": self.tick,
            "episode": self.episode,
            "populations": {
                agent_type.value: {
                    "count": len(pop.agents),
                    "generation": pop.generation,
                }
                for agent_type, pop in self.populations.items()
            },
            "high_price": self._episode_high_price,
            "low_price": self._episode_low_price,
        }

    def get_price_stats(self) -> dict:
        """获取当前 episode 的价格统计

        Returns:
            价格统计字典
        """
        total_volume = 0.0
        if self._tick_history_volumes:
            total_volume = sum(abs(v) for v in self._tick_history_volumes)

        return {
            "tick_count": self.tick,
            "high_price": self._episode_high_price,
            "low_price": self._episode_low_price,
            "final_price": (
                self.matching_engine._orderbook.last_price
                if self.matching_engine
                else 0.0
            ),
            "total_volume": total_volume,
        }

    def get_population_stats(self) -> dict:
        """获取当前种群统计

        Returns:
            种群统计字典，包含淘汰数、平均适应度和最精英 species 的平均适应度
        """
        stats: dict = {
            "liquidations": dict(self._pop_liquidated_counts),
            "avg_fitness": {},
            "elite_species_fitness": {},
        }

        if not self.matching_engine:
            return stats

        current_price = self.matching_engine._orderbook.last_price

        for agent_type, pop in self.populations.items():
            agent_fitnesses = pop.evaluate(current_price)
            if agent_fitnesses:
                stats["avg_fitness"][agent_type] = sum(
                    f for _, f in agent_fitnesses
                ) / len(agent_fitnesses)

            # 获取最精英 species 的平均适应度
            elite_fitness = pop.get_elite_species_avg_fitness()
            if elite_fitness is not None:
                stats["elite_species_fitness"][agent_type] = elite_fitness

        return stats

    def save_checkpoint_data(self) -> dict:
        """返回检查点数据（不写入文件）

        用于多竞技场场景下由 ArenaManager 统一保存。

        Returns:
            检查点数据字典
        """
        return {
            "tick": self.tick,
            "episode": self.episode,
            "populations": {
                agent_type: {
                    "generation": pop.generation,
                    "neat_pop": pop.neat_pop,
                }
                for agent_type, pop in self.populations.items()
            },
        }

    def save_checkpoint(self, path: str) -> None:
        """保存检查点

        Args:
            path: 检查点文件路径
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "tick": self.tick,
            "episode": self.episode,
            "populations": {
                agent_type: {
                    "generation": pop.generation,
                    "neat_pop": pop.neat_pop,
                }
                for agent_type, pop in self.populations.items()
            },
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        self.logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str) -> None:
        """加载检查点

        Args:
            path: 检查点文件路径
        """
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.tick = checkpoint["tick"]
        self.episode = checkpoint["episode"]

        for agent_type, pop_data in checkpoint["populations"].items():
            if agent_type in self.populations:
                pop = self.populations[agent_type]
                pop.generation = pop_data["generation"]
                pop.neat_pop = pop_data["neat_pop"]
                # 重建 agents
                genomes = list(pop.neat_pop.population.items())
                pop.agents = pop.create_agents(genomes)

        # 注册恢复的 Agent 费率，重建映射表和执行顺序
        self._register_all_agents()
        self._build_agent_map()
        self._build_execution_order()
        self._update_pop_total_counts()

        self.logger.info(f"检查点已加载: {path}")

    def load_checkpoint_data(self, checkpoint: dict) -> None:
        """从检查点数据恢复（不读取文件）

        用于多竞技场场景下由 ArenaManager 统一加载。

        Args:
            checkpoint: 检查点数据字典
        """
        self.tick = checkpoint["tick"]
        self.episode = checkpoint["episode"]

        for agent_type, pop_data in checkpoint["populations"].items():
            if agent_type in self.populations:
                pop = self.populations[agent_type]
                # 【关键修复】先清理旧 Agent，防止内存泄漏
                # setup() 时已创建了 Agent，加载检查点时需要先清理
                pop._cleanup_old_agents()
                gc.collect()
                gc.collect()

                pop.generation = pop_data["generation"]
                pop.neat_pop = pop_data["neat_pop"]
                genomes = list(pop.neat_pop.population.items())
                pop.agents = pop.create_agents(genomes)

        self._register_all_agents()
        self._build_agent_map()
        self._build_execution_order()
        self._update_pop_total_counts()

    def pause(self) -> None:
        """暂停训练"""
        self.is_paused = True
        self.logger.info("训练已暂停")

    def resume(self) -> None:
        """恢复训练"""
        self.is_paused = False
        self.logger.info("训练已恢复")

    def stop(self) -> None:
        """停止训练"""
        self.is_running = False
        self.logger.info("训练已停止")
        self._shutdown_executor()
