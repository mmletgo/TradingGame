#!/usr/bin/env python3
"""调试 fitness 为 0 的问题"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.arena import ParallelArenaTrainer, MultiArenaConfig
from create_config import create_default_config


def main() -> None:
    # 创建配置
    config = create_default_config(
        episode_length=100,
        config_dir="config",
        catfish_enabled=False,
    )
    config.training.num_arenas = 2
    config.training.episodes_per_arena = 1

    # 创建多竞技场配置
    multi_config = MultiArenaConfig(
        num_arenas=2,
        episodes_per_arena=1,
    )

    # 创建训练器
    trainer = ParallelArenaTrainer(config, multi_config)

    # 禁用 Worker 池模式进行测试
    trainer._use_execute_workers = False

    trainer.setup()

    # 重置并初始化
    trainer._reset_all_arenas()

    # 调试做市商初始化
    print("\n=== 调试做市商初始化 ===")
    from src.bio.agents.base import AgentType
    from src.training.arena.arena_state import calculate_order_quantity_from_state

    mm_pop = trainer.populations.get(AgentType.MARKET_MAKER)
    if mm_pop:
        first_arena = trainer.arena_states[0]
        first_mm = list(mm_pop.agents)[0]
        mm_state = first_arena.agent_states.get(first_mm.agent_id)

        print(f"做市商 {first_mm.agent_id}:")
        print(f"  初始余额: {mm_state.initial_balance}")
        print(f"  当前余额: {mm_state.balance}")
        print(f"  杠杆: {mm_state.leverage}")

        # 计算市场状态
        market_state = trainer._compute_market_state_for_arena(first_arena)
        print(f"\n市场状态:")
        print(f"  mid_price: {market_state.mid_price}")
        print(f"  tick_size: {market_state.tick_size}")

        # 进行一次推理
        orderbook = first_arena.matching_engine._orderbook
        inputs = first_mm.observe(market_state, orderbook)
        outputs = first_mm.brain.forward(inputs)

        print(f"\n神经网络输出 (前 45 个值):")
        print(f"  买单价格偏移 [0-9]: {outputs[0:10]}")
        print(f"  买单数量权重 [10-19]: {outputs[10:20]}")
        print(f"  卖单价格偏移 [20-29]: {outputs[20:30]}")
        print(f"  卖单数量权重 [30-39]: {outputs[30:40]}")
        print(f"  总下单比例 [40]: {outputs[40] if len(outputs) > 40 else 'N/A'}")

        # 解析输出
        import numpy as np
        _, params = trainer._parse_market_maker_output(
            mm_state, np.array(outputs), market_state.mid_price, market_state.tick_size
        )
        print(f"\n解析后的订单:")
        print(f"  买单数量: {len(params.get('bid_orders', []))}")
        print(f"  卖单数量: {len(params.get('ask_orders', []))}")
        if params.get('bid_orders'):
            print(f"  第一个买单: {params['bid_orders'][0]}")
        if params.get('ask_orders'):
            print(f"  第一个卖单: {params['ask_orders'][0]}")

        # 检查数量计算
        test_qty = calculate_order_quantity_from_state(
            mm_state, 99.0, 0.1, is_buy=True, ref_price=100.0
        )
        print(f"\n数量计算测试 (ratio=0.1, price=99): {test_qty}")

    # 检查 Worker 池状态
    print(f"\n=== Worker 池状态 ===")
    print(f"  _execute_worker_pool: {trainer._execute_worker_pool}")
    print(f"  _use_execute_workers: {trainer._use_execute_workers}")

    # 检查网络索引映射
    print(f"\n=== 网络索引映射检查 ===")
    for agent_type, type_map in trainer._network_index_map.items():
        sample_ids = list(type_map.keys())[:5]
        print(f"  {agent_type.value}: {len(type_map)} 条映射, 样本ID: {sample_ids}")

    # 检查 ArenaState 中的 agent_states
    first_arena = trainer.arena_states[0]
    retail_states = [s for s in first_arena.agent_states.values() if s.agent_type == AgentType.RETAIL]
    print(f"\n  ArenaState 中的 RETAIL agent_states: {len(retail_states)} 个")
    if retail_states:
        sample_retail_ids = [s.agent_id for s in retail_states[:5]]
        print(f"    样本ID: {sample_retail_ids}")

        # 检查这些ID是否在映射中
        retail_map = trainer._network_index_map.get(AgentType.RETAIL, {})
        for agent_id in sample_retail_ids[:3]:
            idx = retail_map.get(agent_id, -1)
            print(f"    Agent {agent_id} 在映射中的索引: {idx}")

    trainer._init_market_all_arenas()

    print("\n=== 做市商初始化后的订单簿状态 ===")
    for arena in trainer.arena_states:
        orderbook = arena.matching_engine._orderbook
        print(f"\n竞技场 {arena.arena_id}:")
        depth = orderbook.get_depth(100)
        bid_depth = depth["bids"]
        ask_depth = depth["asks"]
        print(f"  买盘档位数: {len([d for d in bid_depth if d[1] > 0])}")
        print(f"  卖盘档位数: {len([d for d in ask_depth if d[1] > 0])}")
        print(f"  最新价: {orderbook.last_price}")
        mid = orderbook.get_mid_price()
        print(f"  中间价: {mid}")

        # 统计总挂单量
        total_bid_qty = sum(d[1] for d in bid_depth)
        total_ask_qty = sum(d[1] for d in ask_depth)
        print(f"  买盘总量: {total_bid_qty}")
        print(f"  卖盘总量: {total_ask_qty}")

    # 分析 RETAIL Agent 的动作选择
    print("\n=== 分析 RETAIL Agent 动作选择 ===")
    from src.bio.agents.base import ActionType
    from collections import Counter

    retail_pop = trainer.populations.get(AgentType.RETAIL)
    if retail_pop:
        first_arena = trainer.arena_states[0]
        market_state = trainer._compute_market_state_for_arena(first_arena)
        orderbook = first_arena.matching_engine._orderbook

        action_counter: Counter[str] = Counter()
        zero_qty_count = 0
        sample_outputs: list[tuple[int, list[float], str, dict]] = []

        # 分析前 100 个 RETAIL Agent
        agents_to_check = list(retail_pop.agents)[:100]
        for agent in agents_to_check:
            # 获取神经网络输入和输出
            inputs = agent.observe(market_state, orderbook)
            outputs = agent.brain.forward(inputs)

            # 解析动作
            from src.bio.agents.base import fast_argmax, fast_clip
            action_idx = fast_argmax(outputs, 0, 6)
            action = ActionType(action_idx)
            action_counter[action.name] += 1

            # 解析参数
            price_offset_norm = fast_clip(outputs[6], -1.0, 1.0)
            quantity_ratio_norm = fast_clip(outputs[7], -1.0, 1.0)
            quantity_ratio = (quantity_ratio_norm + 1) * 0.5

            # 计算实际数量
            mid_price = market_state.mid_price
            tick_size = market_state.tick_size
            qty = 0
            if action in [ActionType.PLACE_BID, ActionType.MARKET_BUY]:
                qty = agent._calculate_order_quantity(mid_price, quantity_ratio, is_buy=True)
            elif action in [ActionType.PLACE_ASK, ActionType.MARKET_SELL]:
                position_qty = agent.account.position.quantity
                if action == ActionType.MARKET_SELL and position_qty > 0:
                    qty = max(1, int(position_qty * quantity_ratio))
                else:
                    qty = agent._calculate_order_quantity(mid_price, quantity_ratio, is_buy=False)

            if action not in [ActionType.HOLD, ActionType.CANCEL] and qty == 0:
                zero_qty_count += 1

            # 记录样本输出（前 5 个）
            if len(sample_outputs) < 5:
                sample_outputs.append((
                    agent.agent_id,
                    [float(x) for x in outputs[:8]],
                    action.name,
                    {"price_offset": price_offset_norm, "qty_ratio": quantity_ratio, "calc_qty": qty}
                ))

        print(f"动作分布（前 100 个 RETAIL Agent）:")
        for action_name, count in action_counter.most_common():
            print(f"  {action_name}: {count}")
        print(f"非 HOLD/CANCEL 但数量为 0: {zero_qty_count}")

        print(f"\n样本 Agent 输出:")
        for agent_id, outputs_list, action_name, params in sample_outputs:
            print(f"  Agent {agent_id}:")
            print(f"    动作得分 [0-5]: {[f'{x:.4f}' for x in outputs_list[:6]]}")
            print(f"    价格偏移/数量比例: {outputs_list[6]:.4f}, {outputs_list[7]:.4f}")
            print(f"    解析动作: {action_name}, 参数: {params}")

        # 检查账户状态
        print(f"\n样本 RETAIL Agent 账户状态:")
        for agent in agents_to_check[:5]:
            agent_state = first_arena.agent_states.get(agent.agent_id)
            if agent_state:
                equity = agent_state.get_equity(market_state.mid_price)
                print(f"  Agent {agent.agent_id}:")
                print(f"    初始余额: {agent_state.initial_balance}")
                print(f"    当前余额: {agent_state.balance}")
                print(f"    杠杆: {agent_state.leverage}")
                print(f"    净值: {equity}")

    # 手动调用推理检查 decisions 是否包含 RETAIL
    print("\n=== 检查批量推理结果 ===")
    first_arena = trainer.arena_states[0]
    first_arena.tick = 1  # 重置 tick（让它变成 tick 2）
    market_state = trainer._compute_market_state_for_arena(first_arena)

    # 收集活跃状态
    active_states = [
        state for state in first_arena.agent_states.values()
        if not state.is_liquidated
    ]
    print(f"活跃状态数量: {len(active_states)}")

    # 调用批量推理
    all_decisions = trainer._batch_inference_all_arenas_direct(
        [market_state], [active_states]
    )

    decisions_for_arena_0 = all_decisions.get(0, [])
    print(f"Arena 0 的决策数量: {len(decisions_for_arena_0)}")

    # 按类型统计
    type_counts: Counter[str] = Counter()
    action_dist: Counter[str] = Counter()
    for state, action, params in decisions_for_arena_0:
        type_counts[state.agent_type.value] += 1
        action_dist[f"{state.agent_type.value}:{action.name}"] += 1

    print(f"\n按类型统计:")
    for agent_type, count in type_counts.most_common():
        print(f"  {agent_type}: {count}")

    print(f"\n动作分布（样本）:")
    for key, count in action_dist.most_common()[:20]:
        print(f"  {key}: {count}")

    # 检查 RETAIL 的决策
    retail_decisions = [
        (state, action, params) for state, action, params in decisions_for_arena_0
        if state.agent_type == AgentType.RETAIL
    ]
    print(f"\nRETAIL 决策数量: {len(retail_decisions)}")
    if retail_decisions:
        for state, action, params in retail_decisions[:5]:
            print(f"  Agent {state.agent_id}: {action.name}, params={params}")

    # 运行几个 tick，统计动作分布
    print("\n=== 运行 10 个 tick 的动作统计 ===")

    action_counts: Counter[str] = Counter()
    trade_count = 0
    trade_details: list[tuple[int, int, int, float]] = []  # (buyer_id, seller_id, qty, price)

    for tick in range(10):
        # 运行一个 tick
        trainer.run_tick_all_arenas()

        # 统计每个竞技场的成交
        for arena in trainer.arena_states:
            for trade in arena.recent_trades:
                trade_count += 1
                if len(trade_details) < 20:  # 只记录前 20 笔
                    trade_details.append((
                        trade.buyer_id, trade.seller_id, trade.quantity, trade.price
                    ))

    print(f"总成交笔数: {trade_count}")
    print(f"\n前 {len(trade_details)} 笔成交:")
    for i, (buyer_id, seller_id, qty, price) in enumerate(trade_details):
        print(f"  {i+1}. buyer={buyer_id}, seller={seller_id}, qty={qty}, price={price}")

    # 检查成交参与者的账户状态
    print("\n=== 成交参与者账户状态 ===")
    arena = trainer.arena_states[0]
    checked_agents: set[int] = set()
    for buyer_id, seller_id, _, _ in trade_details[:10]:
        for agent_id in [buyer_id, seller_id]:
            if agent_id in checked_agents:
                continue
            checked_agents.add(agent_id)
            state = arena.agent_states.get(agent_id)
            if state:
                equity = state.get_equity(arena.matching_engine._orderbook.last_price)
                fitness = (equity - state.initial_balance) / state.initial_balance if state.initial_balance > 0 else 0
                print(f"\nAgent {agent_id} ({state.agent_type.value}):")
                print(f"  初始余额: {state.initial_balance}")
                print(f"  当前余额: {state.balance}")
                print(f"  持仓: {state.position_quantity}")
                print(f"  已实现盈亏: {state.realized_pnl}")
                print(f"  净值: {equity}")
                print(f"  收益率: {fitness:.6f}")

    # 收集适应度
    print("\n=== 适应度统计 ===")
    episode_fitness = trainer._collect_episode_fitness()

    for (agent_type, sub_pop_id), fitness_arr in episode_fitness.items():
        non_zero = np.count_nonzero(fitness_arr)
        mean_val = float(fitness_arr.mean())
        std_val = float(fitness_arr.std())
        max_val = float(fitness_arr.max())
        min_val = float(fitness_arr.min())

        print(f"\n{agent_type.value} (子种群 {sub_pop_id}):")
        print(f"  总数: {len(fitness_arr)}")
        print(f"  非零数: {non_zero} ({100*non_zero/len(fitness_arr):.1f}%)")
        print(f"  均值: {mean_val:.6f}")
        print(f"  标准差: {std_val:.6f}")
        print(f"  最大值: {max_val:.6f}")
        print(f"  最小值: {min_val:.6f}")

    # 检查一些 Agent 的具体状态
    print("\n=== 部分 Agent 账户状态 ===")
    arena = trainer.arena_states[0]
    checked = 0
    for agent_state in arena.agent_states.values():
        if checked >= 10:
            break
        if agent_state.agent_type.value == "RETAIL":
            print(f"\nAgent {agent_state.agent_id}:")
            print(f"  初始余额: {agent_state.initial_balance}")
            print(f"  当前余额: {agent_state.balance}")
            print(f"  持仓: {agent_state.position_quantity}")
            print(f"  已实现盈亏: {agent_state.realized_pnl}")
            equity = agent_state.get_equity(arena.matching_engine._orderbook.last_price)
            print(f"  净值: {equity}")
            fitness = (equity - agent_state.initial_balance) / agent_state.initial_balance
            print(f"  收益率: {fitness:.6f}")
            checked += 1

    trainer.stop()


if __name__ == "__main__":
    main()
