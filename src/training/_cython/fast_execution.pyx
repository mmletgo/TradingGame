# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
快速订单执行模块（Cython 加速）

本模块实现非做市商（高级散户）的批量订单执行逻辑，
使用 Cython 优化以减少 Python 对象开销。

注意：不使用 OpenMP 并行，因为订单执行有顺序依赖。
"""

from src.market.orderbook.order import Order, OrderSide, OrderType

# Action 类型常量
DEF ACTION_HOLD = 0
DEF ACTION_PLACE_BID = 1
DEF ACTION_PLACE_ASK = 2
DEF ACTION_CANCEL = 3
DEF ACTION_MARKET_BUY = 4
DEF ACTION_MARKET_SELL = 5


cpdef list execute_non_mm_batch(
    list decisions,
    object matching_engine,
    object orderbook,
    object recent_trades,
):
    """
    批量执行非做市商的订单决策

    使用 Cython 优化的循环处理高级散户的订单执行。
    做市商不在此函数中处理。

    Args:
        decisions: 决策列表，每个元素为 (agent, action, params)
            - agent: Agent 实例（非做市商）
            - action: ActionType 枚举值
            - params: 参数字典 {"price": float, "quantity": int}
        matching_engine: MatchingEngine 实例
        orderbook: OrderBook 实例
        recent_trades: deque，用于记录成交

    Returns:
        所有成交记录列表
    """
    # 返回的所有成交记录
    cdef list all_trades = []

    # 决策数量
    cdef int n_decisions = len(decisions)
    if n_decisions == 0:
        return all_trades

    # 缓存常用对象和方法引用，减少属性查找开销
    cdef object process_order = matching_engine.process_order
    cdef object cancel_order = matching_engine.cancel_order
    cdef object order_map = orderbook.order_map
    cdef object order_map_get = order_map.get
    cdef object recent_trades_append = recent_trades.append

    # 循环变量
    cdef int i
    cdef object agent
    cdef object action
    cdef object params
    cdef object account
    cdef bint is_liquidated
    cdef object old_order_id
    cdef object order_id
    cdef int quantity
    cdef double price
    cdef int side_value
    cdef object order
    cdef list trades
    cdef object trade
    cdef long long agent_id  # 使用 64 位整数以支持 << 32 操作
    cdef int order_counter
    cdef int action_int_1  # ActionType 枚举转换后的整数值

    # 主循环：处理每个决策
    for i in range(n_decisions):
        agent, action, params = decisions[i]

        # 跳过已被强平的 agent
        is_liquidated = agent.is_liquidated
        if is_liquidated:
            continue

        account = agent.account
        agent_id = <long long>agent.agent_id  # 强制转换为 64 位整数
        trades = []

        # 将 ActionType 枚举转换为整数值（兼容枚举对象和整数）
        if hasattr(action, 'value'):
            action_int_1 = action.value
        else:
            action_int_1 = <int>action

        # 根据动作类型执行
        if action_int_1 == ACTION_CANCEL:
            # === 撤单 ===
            order_id = account.pending_order_id
            if order_id is not None:
                cancel_order(order_id)
                account.pending_order_id = None

        elif action_int_1 == ACTION_PLACE_BID or action_int_1 == ACTION_PLACE_ASK:
            # === 限价单：先撤旧单再挂新单 ===
            old_order_id = account.pending_order_id
            if old_order_id is not None:
                cancel_order(old_order_id)
                account.pending_order_id = None

            quantity = params.get("quantity", 0)
            if quantity > 0:
                price = params["price"]
                side_value = 1 if action_int_1 == ACTION_PLACE_BID else -1  # OrderSide.BUY=1, SELL=-1

                # 内联订单 ID 生成
                order_counter = agent._order_counter + 1
                agent._order_counter = order_counter
                order_id = (agent_id << 32) | order_counter

                # 创建订单
                order = Order(
                    order_id=order_id,
                    agent_id=agent_id,
                    side=OrderSide(side_value),
                    order_type=OrderType.LIMIT,
                    price=price,
                    quantity=quantity,
                )

                # 处理订单
                trades = process_order(order)

                # 更新 taker 账户
                for trade in trades:
                    account.on_trade(trade, trade.is_buyer_taker)

                # 更新挂单 ID
                if order_map_get(order_id) is not None:
                    account.pending_order_id = order_id
                else:
                    account.pending_order_id = None

        elif action_int_1 == ACTION_MARKET_BUY or action_int_1 == ACTION_MARKET_SELL:
            # === 市价单 ===
            quantity = params.get("quantity", 0)
            if quantity > 0:
                side_value = 1 if action_int_1 == ACTION_MARKET_BUY else -1  # OrderSide.BUY=1, SELL=-1

                # 内联订单 ID 生成
                order_counter = agent._order_counter + 1
                agent._order_counter = order_counter
                order_id = (agent_id << 32) | order_counter

                # 创建订单
                order = Order(
                    order_id=order_id,
                    agent_id=agent_id,
                    side=OrderSide(side_value),
                    order_type=OrderType.MARKET,
                    price=0.0,
                    quantity=quantity,
                )

                # 处理订单
                trades = process_order(order)

                # 更新 taker 账户
                for trade in trades:
                    account.on_trade(trade, trade.is_buyer_taker)

        # HOLD 动作：不执行任何操作（跳过）

        # 记录成交到 recent_trades
        for trade in trades:
            recent_trades_append(trade)
            all_trades.append(trade)

    return all_trades


cpdef list execute_non_mm_batch_with_maker_update(
    list decisions,
    object matching_engine,
    object orderbook,
    object recent_trades,
    dict agent_map,
    list tick_trades,
):
    """
    批量执行非做市商的订单决策（包含 maker 账户更新）

    与 execute_non_mm_batch 类似，但额外处理：
    - 更新 maker 账户
    - 记录到 tick_trades 列表

    Args:
        decisions: 决策列表，每个元素为 (agent, action, params)
        matching_engine: MatchingEngine 实例
        orderbook: OrderBook 实例
        recent_trades: deque，用于记录成交
        agent_map: dict[int, Agent]，Agent ID 到 Agent 对象的映射
        tick_trades: list，用于记录本 tick 的成交

    Returns:
        所有成交记录列表
    """
    # 返回的所有成交记录
    cdef list all_trades = []

    # 决策数量
    cdef int n_decisions = len(decisions)
    if n_decisions == 0:
        return all_trades

    # 缓存常用对象和方法引用
    cdef object process_order = matching_engine.process_order
    cdef object cancel_order = matching_engine.cancel_order
    cdef object order_map = orderbook.order_map
    cdef object order_map_get = order_map.get
    cdef object recent_trades_append = recent_trades.append
    cdef object tick_trades_append = tick_trades.append
    cdef object agent_map_get = agent_map.get

    # 循环变量
    cdef int i
    cdef object agent
    cdef object action
    cdef object params
    cdef object account
    cdef bint is_liquidated
    cdef object old_order_id
    cdef object order_id
    cdef int quantity
    cdef double price
    cdef int side_value
    cdef object order
    cdef list trades
    cdef object trade
    cdef long long agent_id  # 使用 64 位整数以支持 << 32 操作
    cdef int order_counter
    cdef int maker_id
    cdef object maker_agent
    cdef int action_int  # ActionType 枚举转换后的整数值

    # 主循环：处理每个决策
    for i in range(n_decisions):
        agent, action, params = decisions[i]

        # 跳过已被强平的 agent
        is_liquidated = agent.is_liquidated
        if is_liquidated:
            continue

        account = agent.account
        agent_id = <long long>agent.agent_id  # 强制转换为 64 位整数
        trades = []

        # 将 ActionType 枚举转换为整数值（兼容枚举对象和整数）
        if hasattr(action, 'value'):
            action_int = action.value
        else:
            action_int = <int>action

        # 根据动作类型执行
        if action_int == ACTION_CANCEL:
            # === 撤单 ===
            order_id = account.pending_order_id
            if order_id is not None:
                cancel_order(order_id)
                account.pending_order_id = None

        elif action_int == ACTION_PLACE_BID or action_int == ACTION_PLACE_ASK:
            # === 限价单：先撤旧单再挂新单 ===
            old_order_id = account.pending_order_id
            if old_order_id is not None:
                cancel_order(old_order_id)
                account.pending_order_id = None

            quantity = params.get("quantity", 0)
            if quantity > 0:
                price = params["price"]
                side_value = 1 if action_int == ACTION_PLACE_BID else -1

                # 内联订单 ID 生成
                order_counter = agent._order_counter + 1
                agent._order_counter = order_counter
                order_id = (agent_id << 32) | order_counter

                order = Order(
                    order_id=order_id,
                    agent_id=agent_id,
                    side=OrderSide(side_value),
                    order_type=OrderType.LIMIT,
                    price=price,
                    quantity=quantity,
                )

                trades = process_order(order)

                for trade in trades:
                    account.on_trade(trade, trade.is_buyer_taker)

                if order_map_get(order_id) is not None:
                    account.pending_order_id = order_id
                else:
                    account.pending_order_id = None

        elif action_int == ACTION_MARKET_BUY or action_int == ACTION_MARKET_SELL:
            # === 市价单 ===
            quantity = params.get("quantity", 0)
            if quantity > 0:
                side_value = 1 if action_int == ACTION_MARKET_BUY else -1

                order_counter = agent._order_counter + 1
                agent._order_counter = order_counter
                order_id = (agent_id << 32) | order_counter

                order = Order(
                    order_id=order_id,
                    agent_id=agent_id,
                    side=OrderSide(side_value),
                    order_type=OrderType.MARKET,
                    price=0.0,
                    quantity=quantity,
                )

                trades = process_order(order)

                for trade in trades:
                    account.on_trade(trade, trade.is_buyer_taker)

        # 记录成交、更新 maker 账户
        for trade in trades:
            recent_trades_append(trade)
            tick_trades_append(trade)
            all_trades.append(trade)

            # 查找并更新 maker 账户
            if trade.is_buyer_taker:
                maker_id = trade.seller_id
            else:
                maker_id = trade.buyer_id

            maker_agent = agent_map_get(maker_id)
            if maker_agent is not None:
                # maker 的 is_buyer 与 taker 相反
                maker_agent.account.on_trade(trade, not trade.is_buyer_taker)

    return all_trades


# 订单数量上限，防止 int 溢出（与 Agent.MAX_ORDER_QUANTITY 一致）
DEF MAX_ORDER_QUANTITY = 100_000_000


cpdef list execute_non_mm_batch_raw(
    list raw_decisions,
    object matching_engine,
    object orderbook,
    object recent_trades,
    dict agent_map,
    list tick_trades,
    double mid_price,
):
    """
    批量执行非做市商的原始决策数据（内联数量计算逻辑）

    此函数直接接受原始决策数据，避免创建 Python dict 和调用 Python 方法的开销。
    数量计算逻辑内联到此函数中，避免调用 agent._calculate_order_quantity()。

    Args:
        raw_decisions: 原始决策列表，每个元素为 (agent, action_type_int, side_int, price, quantity_ratio)
            - agent: Agent 实例（非做市商）
            - action_type_int: 动作类型整数（0=HOLD, 1=PLACE_BID, 2=PLACE_ASK, 3=CANCEL, 4=MARKET_BUY, 5=MARKET_SELL）
            - side_int: 方向整数（1=买, 2=卖），用于限价单
            - price: 订单价格
            - quantity_ratio: 数量比例（0.0-1.0）
        matching_engine: MatchingEngine 实例
        orderbook: OrderBook 实例
        recent_trades: deque，用于记录成交
        agent_map: dict[int, Agent]，Agent ID 到 Agent 对象的映射
        tick_trades: list，用于记录本 tick 的成交
        mid_price: 中间价（用于市价单的数量计算）

    Returns:
        所有成交记录列表
    """
    # 返回的所有成交记录
    cdef list all_trades = []

    # 决策数量
    cdef int n_decisions = len(raw_decisions)
    if n_decisions == 0:
        return all_trades

    # 缓存常用对象和方法引用
    cdef object process_order = matching_engine.process_order
    cdef object cancel_order = matching_engine.cancel_order
    cdef object order_map = orderbook.order_map
    cdef object order_map_get = order_map.get
    cdef object recent_trades_append = recent_trades.append
    cdef object tick_trades_append = tick_trades.append
    cdef object agent_map_get = agent_map.get

    # 循环变量
    cdef int i
    cdef object agent
    cdef int action_type_int
    cdef int side_int
    cdef double price
    cdef double quantity_ratio
    cdef object account
    cdef object position
    cdef bint is_liquidated
    cdef object old_order_id
    cdef object order_id
    cdef int quantity
    cdef int side_value
    cdef object order
    cdef list trades
    cdef object trade
    cdef long long agent_id  # 使用 64 位整数以支持 << 32 操作
    cdef int order_counter
    cdef int maker_id
    cdef object maker_agent

    # 内联数量计算所需变量
    cdef double calc_price
    cdef double equity
    cdef double leverage
    cdef double max_pos_value
    cdef double current_pos
    cdef double current_pos_value
    cdef double available_pos_value
    cdef double raw_quantity
    cdef int position_qty_int

    # 主循环：处理每个决策
    for i in range(n_decisions):
        agent, action_type_int, side_int, price, quantity_ratio = raw_decisions[i]

        # 跳过已被强平的 agent
        is_liquidated = agent.is_liquidated
        if is_liquidated:
            continue

        account = agent.account
        position = account.position
        agent_id = <long long>agent.agent_id  # 强制转换为 64 位整数
        trades = []

        # 根据动作类型执行
        if action_type_int == ACTION_CANCEL:
            # === 撤单 ===
            order_id = account.pending_order_id
            if order_id is not None:
                cancel_order(order_id)
                account.pending_order_id = None

        elif action_type_int == ACTION_PLACE_BID or action_type_int == ACTION_PLACE_ASK:
            # === 限价单：先撤旧单再挂新单 ===
            old_order_id = account.pending_order_id
            if old_order_id is not None:
                cancel_order(old_order_id)
                account.pending_order_id = None

            # 内联数量计算逻辑
            # 使用订单价格作为计算价格
            calc_price = price if price > 0 else mid_price
            equity = account.get_equity(calc_price)

            if equity > 0 and calc_price > 0:
                leverage = account.leverage
                max_pos_value = equity * leverage
                current_pos = position.quantity
                current_pos_value = abs(current_pos) * calc_price

                # 根据买卖方向计算可用持仓空间
                if action_type_int == ACTION_PLACE_BID:
                    # 买入
                    if current_pos >= 0:
                        # 多头或空仓，买入是同向加仓
                        available_pos_value = max_pos_value - current_pos_value
                        if available_pos_value < 0:
                            available_pos_value = 0
                    else:
                        # 空头，买入是反向平仓+可能开多仓
                        available_pos_value = current_pos_value + max_pos_value
                else:
                    # 卖出
                    if current_pos <= 0:
                        # 空头或空仓，卖出是同向加仓
                        available_pos_value = max_pos_value - current_pos_value
                        if available_pos_value < 0:
                            available_pos_value = 0
                    else:
                        # 多头，卖出是反向平仓+可能开空仓
                        available_pos_value = current_pos_value + max_pos_value

                # 计算数量
                if quantity_ratio > 1.0:
                    quantity_ratio = 1.0
                raw_quantity = (available_pos_value * quantity_ratio) / price
                if raw_quantity >= 1:
                    quantity = <int>raw_quantity
                    if quantity > MAX_ORDER_QUANTITY:
                        quantity = MAX_ORDER_QUANTITY
                else:
                    quantity = 0
            else:
                quantity = 0

            if quantity > 0:
                side_value = 1 if action_type_int == ACTION_PLACE_BID else -1

                # 内联订单 ID 生成
                order_counter = agent._order_counter + 1
                agent._order_counter = order_counter
                order_id = (agent_id << 32) | order_counter

                order = Order(
                    order_id=order_id,
                    agent_id=agent_id,
                    side=OrderSide(side_value),
                    order_type=OrderType.LIMIT,
                    price=price,
                    quantity=quantity,
                )

                trades = process_order(order)

                for trade in trades:
                    account.on_trade(trade, trade.is_buyer_taker)

                if order_map_get(order_id) is not None:
                    account.pending_order_id = order_id
                else:
                    account.pending_order_id = None

        elif action_type_int == ACTION_MARKET_BUY:
            # === 市价买入 ===
            # 内联数量计算逻辑（使用 mid_price）
            calc_price = mid_price if mid_price > 0 else 100.0
            equity = account.get_equity(calc_price)

            if equity > 0:
                leverage = account.leverage
                max_pos_value = equity * leverage
                current_pos = position.quantity
                current_pos_value = abs(current_pos) * calc_price

                # 买入方向
                if current_pos >= 0:
                    available_pos_value = max_pos_value - current_pos_value
                    if available_pos_value < 0:
                        available_pos_value = 0
                else:
                    available_pos_value = current_pos_value + max_pos_value

                if quantity_ratio > 1.0:
                    quantity_ratio = 1.0
                raw_quantity = (available_pos_value * quantity_ratio) / calc_price
                if raw_quantity >= 1:
                    quantity = <int>raw_quantity
                    if quantity > MAX_ORDER_QUANTITY:
                        quantity = MAX_ORDER_QUANTITY
                else:
                    quantity = 0
            else:
                quantity = 0

            if quantity > 0:
                order_counter = agent._order_counter + 1
                agent._order_counter = order_counter
                order_id = (agent_id << 32) | order_counter

                order = Order(
                    order_id=order_id,
                    agent_id=agent_id,
                    side=OrderSide(1),  # BUY
                    order_type=OrderType.MARKET,
                    price=0.0,
                    quantity=quantity,
                )

                trades = process_order(order)

                for trade in trades:
                    account.on_trade(trade, trade.is_buyer_taker)

        elif action_type_int == ACTION_MARKET_SELL:
            # === 市价卖出 ===
            # 特殊逻辑：如果持有多仓，卖出持仓的比例；否则开空仓
            position_qty_int = <int>position.quantity

            if position_qty_int > 0:
                # 有多仓时卖出（卖出比例由神经网络决定）
                if quantity_ratio > 1.0:
                    quantity_ratio = 1.0
                quantity = <int>(position_qty_int * quantity_ratio)
                if quantity < 1:
                    quantity = 1
                if quantity > position_qty_int:
                    quantity = position_qty_int
            else:
                # 空仓或空头，开空仓（使用 mid_price）
                calc_price = mid_price if mid_price > 0 else 100.0
                equity = account.get_equity(calc_price)

                if equity > 0:
                    leverage = account.leverage
                    max_pos_value = equity * leverage
                    current_pos = position.quantity
                    current_pos_value = abs(current_pos) * calc_price

                    # 卖出方向
                    if current_pos <= 0:
                        available_pos_value = max_pos_value - current_pos_value
                        if available_pos_value < 0:
                            available_pos_value = 0
                    else:
                        available_pos_value = current_pos_value + max_pos_value

                    if quantity_ratio > 1.0:
                        quantity_ratio = 1.0
                    raw_quantity = (available_pos_value * quantity_ratio) / calc_price
                    if raw_quantity >= 1:
                        quantity = <int>raw_quantity
                        if quantity > MAX_ORDER_QUANTITY:
                            quantity = MAX_ORDER_QUANTITY
                    else:
                        quantity = 0
                else:
                    quantity = 0

            if quantity > 0:
                order_counter = agent._order_counter + 1
                agent._order_counter = order_counter
                order_id = (agent_id << 32) | order_counter

                order = Order(
                    order_id=order_id,
                    agent_id=agent_id,
                    side=OrderSide(-1),  # SELL
                    order_type=OrderType.MARKET,
                    price=0.0,
                    quantity=quantity,
                )

                trades = process_order(order)

                for trade in trades:
                    account.on_trade(trade, trade.is_buyer_taker)

        # HOLD 动作（action_type_int == 0）：不执行任何操作（跳过）

        # 记录成交、更新 maker 账户
        for trade in trades:
            recent_trades_append(trade)
            tick_trades_append(trade)
            all_trades.append(trade)

            # 查找并更新 maker 账户
            if trade.is_buyer_taker:
                maker_id = trade.seller_id
            else:
                maker_id = trade.buyer_id

            maker_agent = agent_map_get(maker_id)
            if maker_agent is not None:
                # maker 的 is_buyer 与 taker 相反
                maker_agent.account.on_trade(trade, not trade.is_buyer_taker)

    return all_trades


cpdef list execute_mm_batch(
    list mm_decisions,
    object matching_engine,
    object orderbook,
    object recent_trades,
    dict agent_map,
    list tick_trades,
):
    """
    批量执行做市商的订单决策

    使用 Cython 优化的循环处理做市商的订单执行。
    做市商每 tick 先撤所有旧单，然后双边各挂多单。

    Args:
        mm_decisions: 决策列表，每个元素为 (agent, action, params)
            - agent: MarketMaker 实例
            - action: ActionType 枚举值（忽略，做市商始终执行双边挂单）
            - params: 参数字典 {"bid_orders": [...], "ask_orders": [...]}
        matching_engine: MatchingEngine 或 FastMatchingEngine 实例
        orderbook: OrderBook 实例
        recent_trades: deque，用于记录成交
        agent_map: dict[int, Agent]，Agent ID 到 Agent 对象的映射
        tick_trades: list，用于记录本 tick 的成交

    Returns:
        所有成交记录列表
    """
    # 返回的所有成交记录
    cdef list all_trades = []

    # 决策数量
    cdef int n_decisions = len(mm_decisions)
    if n_decisions == 0:
        return all_trades

    # 缓存常用对象和方法引用
    cdef object process_order = matching_engine.process_order
    cdef object cancel_order = matching_engine.cancel_order
    cdef object recent_trades_append = recent_trades.append
    cdef object tick_trades_append = tick_trades.append
    cdef object agent_map_get = agent_map.get

    # 循环变量
    cdef int i, j
    cdef object agent
    cdef object action
    cdef object params
    cdef object account
    cdef bint is_liquidated
    cdef list bid_order_ids
    cdef list ask_order_ids
    cdef list bid_orders
    cdef list ask_orders
    cdef object order_spec
    cdef object order_id
    cdef int order_counter
    cdef long long agent_id
    cdef double price
    cdef int quantity
    cdef object order
    cdef list trades
    cdef object trade
    cdef int maker_id
    cdef object maker_agent

    # 主循环：处理每个做市商决策
    for i in range(n_decisions):
        agent, action, params = mm_decisions[i]

        # 跳过已被强平的做市商
        is_liquidated = agent.is_liquidated
        if is_liquidated:
            continue

        account = agent.account
        agent_id = <long long>agent.agent_id
        bid_order_ids = agent.bid_order_ids
        ask_order_ids = agent.ask_order_ids

        # === 1. 撤销所有旧订单 ===
        for order_id in bid_order_ids:
            cancel_order(order_id)
        for order_id in ask_order_ids:
            cancel_order(order_id)
        bid_order_ids.clear()
        ask_order_ids.clear()

        # === 2. 挂买单 ===
        bid_orders = params.get("bid_orders", [])
        for order_spec in bid_orders:
            price = order_spec["price"]
            quantity = <int>order_spec["quantity"]
            if quantity <= 0:
                continue

            # 生成订单 ID
            order_counter = agent._order_counter + 1
            agent._order_counter = order_counter
            order_id = (agent_id << 32) | order_counter

            # 创建买单
            order = Order(
                order_id=order_id,
                agent_id=agent_id,
                side=OrderSide(1),  # BUY
                order_type=OrderType.LIMIT,
                price=price,
                quantity=quantity,
            )

            # 处理订单
            trades = process_order(order)

            # 更新 taker (做市商) 账户
            for trade in trades:
                account.on_trade(trade, trade.is_buyer_taker)

                # 记录成交
                recent_trades_append(trade)
                tick_trades_append(trade)
                all_trades.append(trade)

                # 更新 maker 账户
                if trade.is_buyer_taker:
                    maker_id = trade.seller_id
                else:
                    maker_id = trade.buyer_id
                maker_agent = agent_map_get(maker_id)
                if maker_agent is not None:
                    maker_agent.account.on_trade(trade, not trade.is_buyer_taker)

            # 记录订单 ID
            bid_order_ids.append(order_id)

        # === 3. 挂卖单 ===
        ask_orders = params.get("ask_orders", [])
        for order_spec in ask_orders:
            price = order_spec["price"]
            quantity = <int>order_spec["quantity"]
            if quantity <= 0:
                continue

            # 生成订单 ID
            order_counter = agent._order_counter + 1
            agent._order_counter = order_counter
            order_id = (agent_id << 32) | order_counter

            # 创建卖单
            order = Order(
                order_id=order_id,
                agent_id=agent_id,
                side=OrderSide(-1),  # SELL
                order_type=OrderType.LIMIT,
                price=price,
                quantity=quantity,
            )

            # 处理订单
            trades = process_order(order)

            # 更新 taker (做市商) 账户
            for trade in trades:
                account.on_trade(trade, trade.is_buyer_taker)

                # 记录成交
                recent_trades_append(trade)
                tick_trades_append(trade)
                all_trades.append(trade)

                # 更新 maker 账户
                if trade.is_buyer_taker:
                    maker_id = trade.seller_id
                else:
                    maker_id = trade.buyer_id
                maker_agent = agent_map_get(maker_id)
                if maker_agent is not None:
                    maker_agent.account.on_trade(trade, not trade.is_buyer_taker)

            # 记录订单 ID
            ask_order_ids.append(order_id)

    return all_trades
