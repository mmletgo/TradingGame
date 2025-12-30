"""训练器模块

管理训练流程，协调种群、撮合引擎和事件系统。
"""

import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.bio.agents.base import AgentType
from src.config.config import Config
from src.core.event_engine.event_bus import EventBus
from src.core.event_engine.events import Event, EventType
from src.core.log_engine.logger import get_logger
from src.market.market_state import NormalizedMarketState
from src.market.matching.matching_engine import MatchingEngine
from src.market.matching.trade import Trade
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.training.population import Population


class Trainer:
    """训练器

    管理 NEAT 进化训练流程，协调三种群在模拟市场中竞争。

    Attributes:
        config: 全局配置
        event_bus: 事件总线
        populations: 三种群（散户/庄家/做市商）
        matching_engine: 撮合引擎
        tick: 当前 tick
        episode: 当前 episode
        is_running: 是否运行中
        is_paused: 是否暂停
        recent_trades: 最近成交记录
    """

    config: Config
    event_bus: EventBus
    populations: dict[AgentType, Population]
    matching_engine: MatchingEngine | None
    tick: int
    episode: int
    is_running: bool
    is_paused: bool
    recent_trades: list[Trade]

    def __init__(self, config: Config) -> None:
        """创建训练器

        Args:
            config: 全局配置对象
        """
        self.config = config
        self.event_bus = EventBus()
        self.logger = get_logger("trainer")

        self.tick = 0
        self.episode = 0
        self.is_running = False
        self.is_paused = False

        self.populations = {}
        self.matching_engine = None
        self.recent_trades = []

    def setup(self) -> None:
        """初始化训练环境

        创建三种群、撮合引擎，订阅事件，初始化市场。
        """
        # 创建三种群
        for agent_type in AgentType:
            self.populations[agent_type] = Population(
                agent_type, self.config, self.event_bus
            )

        # 创建撮合引擎
        self.matching_engine = MatchingEngine(self.event_bus, self.config.market)

        # 订阅成交事件
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self._on_trade)

        # 订阅强平事件
        self.event_bus.subscribe(EventType.LIQUIDATION, self._on_liquidation)

        # 注册所有 Agent 的费率
        self._register_all_agents()

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

    def _on_trade(self, event: Event) -> None:
        """处理成交事件，记录最近成交

        从事件数据中重建 Trade 对象并记录。
        """
        data = event.data
        # 从事件数据重建 Trade 对象
        trade = Trade(
            trade_id=data.get("trade_id", 0),
            price=data.get("price", 0.0),
            quantity=data.get("quantity", 0.0),
            buyer_id=data.get("buyer_id", 0),
            seller_id=data.get("seller_id", 0),
            buyer_fee=data.get("buyer_fee", 0.0),
            seller_fee=data.get("seller_fee", 0.0),
            is_buyer_taker=data.get("is_buyer_taker", True),
        )
        self.recent_trades.append(trade)
        # 保留最近 100 笔
        if len(self.recent_trades) > 100:
            self.recent_trades = self.recent_trades[-100:]

    def _on_liquidation(self, event: Event) -> None:
        """处理强平事件，提交市价单平仓"""
        agent_id = event.data.get("agent_id")
        quantity = event.data.get("position_quantity")
        if agent_id is not None and quantity is not None and self.matching_engine:
            # 平仓：如果持仓为正（多头），则卖出；如果为负（空头），则买入
            side = OrderSide.SELL if quantity > 0 else OrderSide.BUY
            order = Order(
                order_id=0,  # 由撮合引擎分配
                agent_id=agent_id,
                side=side,
                order_type=OrderType.MARKET,
                price=0.0,  # 市价单不需要价格
                quantity=abs(quantity),
            )
            self.matching_engine.process_order(order)

        # 标记 Agent 已被强平
        self._mark_agent_liquidated(agent_id)

    def _mark_agent_liquidated(self, agent_id: int) -> None:
        """标记 Agent 已被强平

        Args:
            agent_id: 被强平的 Agent ID
        """
        for population in self.populations.values():
            for agent in population.agents:
                if agent.agent_id == agent_id:
                    if not agent.is_liquidated:
                        agent.is_liquidated = True
                        self.logger.info(f"Agent {agent_id} 已被强平，本轮 episode 禁用")
                    return

    def _compute_normalized_market_state(self) -> NormalizedMarketState:
        """预计算归一化的公共市场数据

        在每个 tick 开始时调用，避免每个 Agent 重复计算相同的归一化数据。

        Returns:
            NormalizedMarketState: 归一化后的市场状态
        """
        orderbook = self.matching_engine._orderbook

        # 获取参考价格
        mid_price = orderbook.get_mid_price()
        if mid_price is None:
            mid_price = orderbook.last_price
        if mid_price == 0:
            mid_price = 100.0

        tick_size = orderbook.tick_size
        depth = orderbook.get_depth(levels=100)

        # 归一化买盘：100档 × 2 = 200
        bid_data = np.zeros(200, dtype=np.float32)
        for i, (price, qty) in enumerate(depth["bids"]):
            bid_data[i * 2] = (price - mid_price) / mid_price if mid_price > 0 else 0
            bid_data[i * 2 + 1] = qty

        # 归一化卖盘：100档 × 2 = 200
        ask_data = np.zeros(200, dtype=np.float32)
        for i, (price, qty) in enumerate(depth["asks"]):
            ask_data[i * 2] = (price - mid_price) / mid_price if mid_price > 0 else 0
            ask_data[i * 2 + 1] = qty

        # 归一化成交：100笔（数量带方向：正=taker买入，负=taker卖出）
        trade_prices = np.zeros(100, dtype=np.float32)
        trade_quantities = np.zeros(100, dtype=np.float32)

        for i, trade in enumerate(self.recent_trades[:100]):
            trade_prices[i] = (trade.price - mid_price) / mid_price if mid_price > 0 else 0
            # 用正负表示方向：正数=taker买入，负数=taker卖出
            trade_quantities[i] = trade.quantity if trade.is_buyer_taker else -trade.quantity

        return NormalizedMarketState(
            mid_price=mid_price,
            tick_size=tick_size,
            bid_data=bid_data,
            ask_data=ask_data,
            trade_prices=trade_prices,
            trade_quantities=trade_quantities,
        )

    def _init_market(self) -> None:
        """初始化市场

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
                agent.execute_action(action, params, self.event_bus)

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

        # 做市商重新初始化
        self._init_market()

    def run_tick(self) -> None:
        """执行一个 tick

        按顺序执行：做市商->庄家->散户 的决策和交易。
        检查强平条件。
        """
        if not self.matching_engine:
            return

        self.tick += 1
        orderbook = self.matching_engine._orderbook

        # 发布 TICK_START 事件
        self.event_bus.publish(
            Event(EventType.TICK_START, time.time(), {"tick": self.tick})
        )

        # 预计算归一化市场数据
        market_state = self._compute_normalized_market_state()

        # 按顺序执行：做市商->庄家->散户
        for agent_type in [AgentType.MARKET_MAKER, AgentType.WHALE, AgentType.RETAIL]:
            population = self.populations.get(agent_type)
            if population:
                for agent in population.agents:
                    # 决策（传入预计算的市场状态和订单簿）
                    action, params = agent.decide(market_state, orderbook)
                    # 执行
                    agent.execute_action(action, params, self.event_bus)

        # 检查强平
        current_price = orderbook.last_price
        for population in self.populations.values():
            for agent in population.agents:
                if agent.account.check_liquidation(current_price):
                    agent.account.liquidate(current_price, self.event_bus)

        # 发布 TICK_END 事件
        self.event_bus.publish(
            Event(EventType.TICK_END, time.time(), {"tick": self.tick})
        )

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

        # 重置市场状态
        self._reset_market()

        # 重置 tick 计数（每个 episode 从 0 开始）
        self.tick = 0

        # 2. 运行 episode_length 个 tick
        for _ in range(episode_length):
            if not self.is_running or self.is_paused:
                break
            self.run_tick()

        # 3. 进化（仅在正常完成 episode 时）
        if self.is_running and not self.is_paused:
            current_price = self.matching_engine._orderbook.last_price
            for population in self.populations.values():
                population.evolve(current_price)

            # 进化后重新注册新 Agent 的费率
            self._register_all_agents()

            self.logger.info(f"Episode {self.episode} 完成，tick={self.tick}")

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
                pop.agents = pop.create_agents(genomes, self.event_bus)

        # 注册恢复的 Agent 费率
        self._register_all_agents()

        self.logger.info(f"检查点已加载: {path}")

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
