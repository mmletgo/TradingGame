# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language = c++
"""
快速 tick 执行模块（Cython + C++ 加速）

本模块实现整个 tick 的执行逻辑，包括：
- 收集所有原子动作（噪声交易者、做市商、散户）到 C 数组
- Fisher-Yates 随机打乱
- 逐个执行原子动作（撤单、限价单、市价单）
- 内联 on_trade 和 sync_state_to_array 逻辑

替代 arena_worker.py 中的 execute_tick_local() + _execute_atomic_action_local()
+ _update_trade_participants() 函数链。
"""

import numpy as np
cimport numpy as np

from libc.stdlib cimport rand, srand, malloc, free
from libc.time cimport time as c_time

from src.market.matching.fast_matching cimport FastMatchingEngine
from src.market.orderbook.orderbook cimport OrderBook

# 原子动作类型常量
DEF AT_CANCEL = 1
DEF AT_LIMIT_BUY = 2
DEF AT_LIMIT_SELL = 3
DEF AT_MARKET_BUY = 4
DEF AT_MARKET_SELL = 5

# 标志位常量
DEF FLAG_IS_MARKET_MAKER = 1   # bit 0
DEF FLAG_IS_NOISE_TRADER = 2   # bit 1

# 最大原子动作数量
# 需要足够大以容纳所有 agent 的动作：
# 噪声(~100) + 做市商(600×40=24000) + 散户(2400×2=4800) ≈ 29000
DEF MAX_ACTIONS = 32768


# ============================================================================
# 内联辅助函数
# ============================================================================

cdef inline long long _generate_order_id_agent(object state, int arena_id):
    """为 AgentAccountState 生成唯一订单 ID"""
    cdef int counter = state.order_counter + 1
    state.order_counter = counter
    return (<long long>arena_id << 48) | (<long long>state.agent_id << 16) | (counter & 0xFFFF)


cdef inline long long _generate_order_id_noise(object nt_state, int arena_id):
    """为 NoiseTraderAccountState 生成唯一订单 ID

    噪声交易者使用负数 order_id，格式：-(arena_id * 1e9 + |trader_id| * 1e6 + counter)
    """
    cdef int counter = nt_state.order_counter + 1
    nt_state.order_counter = counter
    cdef long long trader_id = nt_state.trader_id
    if trader_id < 0:
        trader_id = -trader_id
    return -(arena_id * 1000000000 + trader_id * 1000000 + counter)


cdef inline double _update_position_cython(
    object state,
    int side,
    int quantity,
    double price,
):
    """内联持仓更新（AgentAccountState），返回已实现盈亏

    Args:
        state: AgentAccountState 对象
        side: 1=BUY, -1=SELL
        quantity: 成交数量
        price: 成交价格

    Returns:
        已实现盈亏
    """
    cdef int pos_qty = state.position_quantity
    cdef double avg_price = state.position_avg_price
    cdef double realized = 0.0
    cdef double total_cost
    cdef int remaining
    cdef int abs_pos

    # Case 1: 空仓
    if pos_qty == 0:
        state.position_quantity = quantity * side
        state.position_avg_price = price
        return 0.0

    # Case 2: 持多头
    if pos_qty > 0:
        if side == 1:  # BUY - 加多仓
            total_cost = pos_qty * avg_price + quantity * price
            pos_qty += quantity
            state.position_quantity = pos_qty
            state.position_avg_price = total_cost / pos_qty if pos_qty > 0 else 0.0
            return 0.0
        else:  # SELL
            if quantity < pos_qty:
                realized = (price - avg_price) * quantity
                state.position_quantity = pos_qty - quantity
            elif quantity == pos_qty:
                realized = (price - avg_price) * pos_qty
                state.position_quantity = 0
                state.position_avg_price = 0.0
            else:
                realized = (price - avg_price) * pos_qty
                remaining = quantity - pos_qty
                state.position_quantity = -remaining
                state.position_avg_price = price
            state.realized_pnl += realized
            return realized

    # Case 3: 持空头(pos_qty < 0)
    abs_pos = -pos_qty
    if side == -1:  # SELL - 加空仓
        total_cost = abs_pos * avg_price + quantity * price
        abs_pos += quantity
        state.position_quantity = -abs_pos
        state.position_avg_price = total_cost / abs_pos if abs_pos > 0 else 0.0
        return 0.0
    else:  # BUY - 减空仓
        if quantity < abs_pos:
            realized = (avg_price - price) * quantity
            state.position_quantity = -(abs_pos - quantity)
        elif quantity == abs_pos:
            realized = (avg_price - price) * abs_pos
            state.position_quantity = 0
            state.position_avg_price = 0.0
        else:
            realized = (avg_price - price) * abs_pos
            remaining = quantity - abs_pos
            state.position_quantity = remaining
            state.position_avg_price = price
        state.realized_pnl += realized
        return realized


cdef inline void _update_noise_trader_position(
    object nt_state,
    double price,
    int quantity,
    bint is_buyer,
):
    """内联噪声交易者持仓更新（无手续费）

    Args:
        nt_state: NoiseTraderAccountState 对象
        price: 成交价格
        quantity: 成交数量
        is_buyer: 是否为买方
    """
    cdef int direction = 1 if is_buyer else -1
    cdef int signed_qty = quantity * direction
    cdef int pos_qty = nt_state.position_quantity
    cdef double avg_price = nt_state.position_avg_price
    cdef double total_cost
    cdef int close_qty
    cdef double pnl
    cdef int remaining_qty
    cdef int abs_pos_nt
    cdef int new_abs
    cdef int cur_abs

    if pos_qty == 0:
        # 空仓：直接开仓
        nt_state.position_quantity = signed_qty
        nt_state.position_avg_price = price
    elif (pos_qty > 0 and direction > 0) or (pos_qty < 0 and direction < 0):
        # 同向加仓
        abs_pos_nt = pos_qty if pos_qty > 0 else -pos_qty
        total_cost = abs_pos_nt * avg_price + quantity * price
        nt_state.position_quantity = pos_qty + signed_qty
        new_abs = nt_state.position_quantity if nt_state.position_quantity > 0 else -nt_state.position_quantity
        if new_abs != 0:
            nt_state.position_avg_price = total_cost / new_abs
    else:
        # 反向：平仓（+可能反向开仓）
        cur_abs = pos_qty if pos_qty > 0 else -pos_qty
        close_qty = quantity if quantity < cur_abs else cur_abs
        if pos_qty > 0:
            pnl = close_qty * (price - avg_price)
        else:
            pnl = close_qty * (avg_price - price)
        nt_state.realized_pnl += pnl
        nt_state.balance += pnl

        remaining_qty = quantity - close_qty
        if pos_qty > 0:
            nt_state.position_quantity = pos_qty - close_qty
        else:
            nt_state.position_quantity = pos_qty + close_qty

        if remaining_qty > 0 and nt_state.position_quantity == 0:
            nt_state.position_quantity = remaining_qty * direction
            nt_state.position_avg_price = price


cdef inline void _inline_sync_state_to_array(
    object state,
    int idx,
    object balances_arr,
    object pos_quantities_arr,
    object pos_avg_prices_arr,
):
    """内联同步 agent 状态到扁平化数组

    Args:
        state: AgentAccountState 对象
        idx: 在数组中的索引
        balances_arr: 余额数组
        pos_quantities_arr: 持仓数量数组
        pos_avg_prices_arr: 持仓均价数组
    """
    balances_arr[idx] = state.balance
    pos_quantities_arr[idx] = state.position_quantity
    pos_avg_prices_arr[idx] = state.position_avg_price


# ============================================================================
# 主函数
# ============================================================================

cpdef tuple execute_tick_cython(
    object arena,
    object retail_decisions,
    object retail_agent_ids,
    object mm_decisions,
    object mm_agent_ids,
    object noise_trader_ids,
    object noise_directions,
    object noise_quantities,
):
    """在 Cython 中执行一个 tick 的所有订单

    使用 C 数组收集原子动作，Fisher-Yates 打乱后执行。
    内联 on_trade 和 sync_state_to_array 逻辑以避免 Python 调用开销。

    Args:
        arena: ArenaState 对象
        retail_decisions: NDArray[float64] shape (N, 4) or None
            列顺序 [action_type, side, price, quantity]
        retail_agent_ids: NDArray[int64] shape (N,) or None
        mm_decisions: NDArray[float64] shape (M, 42) or None
            列顺序 [num_bid, num_ask, bid_prices[10], bid_qtys[10],
                    ask_prices[10], ask_qtys[10]]
        mm_agent_ids: NDArray[int64] shape (M,) or None
        noise_trader_ids: NDArray[int64] shape (K,) — active noise traders only
        noise_directions: NDArray[int32] shape (K,) — +1 or -1
        noise_quantities: NDArray[int32] shape (K,)

    Returns:
        tuple: (all_trades: list[Trade], volume: float, amount: float)
    """
    # ====== C 数组：堆上动态分配 ======
    cdef int* action_types = <int*>malloc(MAX_ACTIONS * sizeof(int))
    cdef long long* agent_ids_arr = <long long*>malloc(MAX_ACTIONS * sizeof(long long))
    cdef long long* order_ids_arr = <long long*>malloc(MAX_ACTIONS * sizeof(long long))
    cdef double* prices_arr = <double*>malloc(MAX_ACTIONS * sizeof(double))
    cdef int* quantities_arr = <int*>malloc(MAX_ACTIONS * sizeof(int))
    cdef char* flags_arr = <char*>malloc(MAX_ACTIONS * sizeof(char))
    cdef int n_actions = 0

    if (action_types is NULL or agent_ids_arr is NULL or order_ids_arr is NULL or
            prices_arr is NULL or quantities_arr is NULL or flags_arr is NULL):
        # 内存分配失败，释放已分配的内存
        if action_types is not NULL: free(action_types)
        if agent_ids_arr is not NULL: free(agent_ids_arr)
        if order_ids_arr is not NULL: free(order_ids_arr)
        if prices_arr is not NULL: free(prices_arr)
        if quantities_arr is not NULL: free(quantities_arr)
        if flags_arr is not NULL: free(flags_arr)
        return ([], 0.0, 0.0)

    # ====== 缓存引用 ======
    # 支持 MatchingEngine(Python 包装层) 和 FastMatchingEngine(Cython) 两种类型
    cdef object _me_obj = arena.matching_engine
    cdef FastMatchingEngine me
    if isinstance(_me_obj, FastMatchingEngine):
        me = <FastMatchingEngine>_me_obj
    else:
        # MatchingEngine 包装层，取内部的 _fast 属性
        me = <FastMatchingEngine>_me_obj._fast
    cdef OrderBook orderbook = <OrderBook>me._orderbook
    cdef dict agent_states = arena.agent_states
    cdef dict noise_trader_states = arena.noise_trader_states
    cdef object recent_trades_deque = arena.recent_trades
    cdef object recent_trades_append = recent_trades_deque.append
    cdef dict agent_id_to_idx = arena._agent_id_to_idx if arena._agent_id_to_idx is not None else {}
    cdef object balances_arr_ref = arena._balances
    cdef object pos_quantities_arr_ref = arena._position_quantities
    cdef object pos_avg_prices_arr_ref = arena._position_avg_prices
    cdef int arena_id = arena.arena_id

    cdef object agent_states_get = agent_states.get
    cdef object noise_states_get = noise_trader_states.get
    cdef object agent_idx_get = agent_id_to_idx.get

    # 循环变量
    cdef int i, j, k, row_idx
    cdef long long agent_id
    cdef int action_int, qty
    cdef double price
    cdef object mm_state, r_state, agent_state, nt_state
    cdef long long oid
    cdef int num_bid, num_ask

    # ====== 第一部分：收集噪声交易者原子动作 ======
    cdef int n_noise = 0
    if noise_trader_ids is not None:
        n_noise = len(noise_trader_ids)

    for i in range(n_noise):
        if n_actions >= MAX_ACTIONS:
            break
        qty = noise_quantities[i]
        if qty <= 0:
            continue
        agent_id = noise_trader_ids[i]
        if noise_directions[i] > 0:
            action_types[n_actions] = AT_MARKET_BUY
        else:
            action_types[n_actions] = AT_MARKET_SELL
        agent_ids_arr[n_actions] = agent_id
        order_ids_arr[n_actions] = 0
        prices_arr[n_actions] = 0.0
        quantities_arr[n_actions] = qty
        flags_arr[n_actions] = FLAG_IS_NOISE_TRADER
        n_actions += 1

    # ====== 第二部分：收集做市商原子动作 ======
    cdef int n_mm = 0
    if mm_decisions is not None and mm_agent_ids is not None:
        n_mm = len(mm_decisions)

    for row_idx in range(n_mm):
        agent_id = mm_agent_ids[row_idx]
        num_bid = int(mm_decisions[row_idx, 0])
        num_ask = int(mm_decisions[row_idx, 1])

        # 撤旧单
        mm_state = agent_states_get(agent_id)
        if mm_state is not None:
            for oid in mm_state.bid_order_ids:
                if n_actions >= MAX_ACTIONS:
                    break
                action_types[n_actions] = AT_CANCEL
                agent_ids_arr[n_actions] = agent_id
                order_ids_arr[n_actions] = oid
                prices_arr[n_actions] = 0.0
                quantities_arr[n_actions] = 0
                flags_arr[n_actions] = FLAG_IS_MARKET_MAKER
                n_actions += 1
            for oid in mm_state.ask_order_ids:
                if n_actions >= MAX_ACTIONS:
                    break
                action_types[n_actions] = AT_CANCEL
                agent_ids_arr[n_actions] = agent_id
                order_ids_arr[n_actions] = oid
                prices_arr[n_actions] = 0.0
                quantities_arr[n_actions] = 0
                flags_arr[n_actions] = FLAG_IS_MARKET_MAKER
                n_actions += 1
            # 清空旧挂单列表
            mm_state.bid_order_ids = []
            mm_state.ask_order_ids = []

        # 挂新买单（价格在列 2-11，数量在列 12-21）
        for k in range(num_bid):
            if n_actions >= MAX_ACTIONS:
                break
            price = mm_decisions[row_idx, 2 + k]
            qty = int(mm_decisions[row_idx, 12 + k])
            if qty > 0:
                action_types[n_actions] = AT_LIMIT_BUY
                agent_ids_arr[n_actions] = agent_id
                order_ids_arr[n_actions] = 0
                prices_arr[n_actions] = price
                quantities_arr[n_actions] = qty
                flags_arr[n_actions] = FLAG_IS_MARKET_MAKER
                n_actions += 1

        # 挂新卖单（价格在列 22-31，数量在列 32-41）
        for k in range(num_ask):
            if n_actions >= MAX_ACTIONS:
                break
            price = mm_decisions[row_idx, 22 + k]
            qty = int(mm_decisions[row_idx, 32 + k])
            if qty > 0:
                action_types[n_actions] = AT_LIMIT_SELL
                agent_ids_arr[n_actions] = agent_id
                order_ids_arr[n_actions] = 0
                prices_arr[n_actions] = price
                quantities_arr[n_actions] = qty
                flags_arr[n_actions] = FLAG_IS_MARKET_MAKER
                n_actions += 1

    # ====== 第三部分：收集散户原子动作 ======
    cdef int n_retail = 0
    if retail_decisions is not None and retail_agent_ids is not None:
        n_retail = len(retail_decisions)

    for i in range(n_retail):
        if n_actions >= MAX_ACTIONS:
            break
        agent_id = retail_agent_ids[i]
        action_int = int(retail_decisions[i, 0])
        price = retail_decisions[i, 2]
        qty = int(retail_decisions[i, 3])

        if action_int == 0:  # HOLD
            continue
        elif action_int == 1 or action_int == 2:  # PLACE_BID / PLACE_ASK
            # 先撤旧单
            r_state = agent_states_get(agent_id)
            if r_state is not None and r_state.pending_order_id is not None:
                if n_actions < MAX_ACTIONS:
                    action_types[n_actions] = AT_CANCEL
                    agent_ids_arr[n_actions] = agent_id
                    order_ids_arr[n_actions] = r_state.pending_order_id
                    prices_arr[n_actions] = 0.0
                    quantities_arr[n_actions] = 0
                    flags_arr[n_actions] = 0
                    n_actions += 1
                r_state.pending_order_id = None
            # 挂新单
            if n_actions < MAX_ACTIONS:
                if action_int == 1:
                    action_types[n_actions] = AT_LIMIT_BUY
                else:
                    action_types[n_actions] = AT_LIMIT_SELL
                agent_ids_arr[n_actions] = agent_id
                order_ids_arr[n_actions] = 0
                prices_arr[n_actions] = price
                quantities_arr[n_actions] = qty
                flags_arr[n_actions] = 0
                n_actions += 1
        elif action_int == 3:  # CANCEL
            r_state = agent_states_get(agent_id)
            if r_state is not None and r_state.pending_order_id is not None:
                if n_actions < MAX_ACTIONS:
                    action_types[n_actions] = AT_CANCEL
                    agent_ids_arr[n_actions] = agent_id
                    order_ids_arr[n_actions] = r_state.pending_order_id
                    prices_arr[n_actions] = 0.0
                    quantities_arr[n_actions] = 0
                    flags_arr[n_actions] = 0
                    n_actions += 1
                r_state.pending_order_id = None
        elif action_int == 4 or action_int == 5:  # MARKET_BUY / MARKET_SELL
            if action_int == 4:
                action_types[n_actions] = AT_MARKET_BUY
            else:
                action_types[n_actions] = AT_MARKET_SELL
            agent_ids_arr[n_actions] = agent_id
            order_ids_arr[n_actions] = 0
            prices_arr[n_actions] = 0.0
            quantities_arr[n_actions] = qty
            flags_arr[n_actions] = 0
            n_actions += 1

    # ====== 第四部分：Fisher-Yates shuffle ======
    cdef int* shuffle_indices = <int*>malloc(MAX_ACTIONS * sizeof(int))
    cdef int tmp_idx

    if shuffle_indices is NULL:
        free(action_types)
        free(agent_ids_arr)
        free(order_ids_arr)
        free(prices_arr)
        free(quantities_arr)
        free(flags_arr)
        return ([], 0.0, 0.0)

    srand(<unsigned int>c_time(NULL))

    for i in range(n_actions):
        shuffle_indices[i] = i
    for i in range(n_actions - 1, 0, -1):
        j = rand() % (i + 1)
        tmp_idx = shuffle_indices[i]
        shuffle_indices[i] = shuffle_indices[j]
        shuffle_indices[j] = tmp_idx

    # ====== 第五部分：按打乱顺序执行原子动作 ======
    cdef list all_trades = []
    cdef double buy_volume = 0.0, sell_volume = 0.0
    cdef double buy_amount = 0.0, sell_amount = 0.0

    cdef int act_idx
    cdef int act_type
    cdef long long act_agent_id
    cdef long long act_order_id
    cdef double act_price
    cdef int act_qty
    cdef char act_flags
    cdef bint is_mm, is_nt

    cdef long long order_id
    cdef list trades
    cdef object trade

    # 成交处理变量
    cdef long long taker_id, maker_id
    cdef double taker_fee, maker_fee
    cdef bint taker_is_buyer
    cdef object taker_state, maker_state_obj
    cdef object nt_taker, nt_maker
    cdef double realized_pnl
    cdef int side_int
    cdef object idx_obj

    for i in range(n_actions):
        act_idx = shuffle_indices[i]
        act_type = action_types[act_idx]
        act_agent_id = agent_ids_arr[act_idx]
        act_order_id = order_ids_arr[act_idx]
        act_price = prices_arr[act_idx]
        act_qty = quantities_arr[act_idx]
        act_flags = flags_arr[act_idx]
        is_mm = (act_flags & FLAG_IS_MARKET_MAKER) != 0
        is_nt = (act_flags & FLAG_IS_NOISE_TRADER) != 0

        # ---- CANCEL ----
        if act_type == AT_CANCEL:
            if act_order_id != 0:
                orderbook.cancel_order_fast(act_order_id)
                # 更新状态：做市商的 bid/ask_order_ids 已在收集阶段清空
                # 散户的 pending_order_id 已在收集阶段设为 None
                # 做市商取消时可能需要从列表中移除（如果还在列表中）
                if is_mm:
                    mm_state = agent_states_get(act_agent_id)
                    if mm_state is not None:
                        try:
                            mm_state.bid_order_ids.remove(act_order_id)
                        except ValueError:
                            pass
                        try:
                            mm_state.ask_order_ids.remove(act_order_id)
                        except ValueError:
                            pass
                elif not is_nt:
                    r_state = agent_states_get(act_agent_id)
                    if r_state is not None and r_state.pending_order_id == act_order_id:
                        r_state.pending_order_id = None

        # ---- LIMIT_BUY ----
        elif act_type == AT_LIMIT_BUY:
            # 生成 order_id
            if is_nt:
                nt_state = noise_states_get(act_agent_id)
                if nt_state is None:
                    continue
                order_id = _generate_order_id_noise(nt_state, arena_id)
            else:
                agent_state = agent_states_get(act_agent_id)
                if agent_state is None:
                    continue
                order_id = _generate_order_id_agent(agent_state, arena_id)

            trades = me.process_order_raw(order_id, act_agent_id, 1, 1, act_price, act_qty)

            for trade in trades:
                all_trades.append(trade)
                recent_trades_append(trade)

                # 内联 _update_trade_participants
                if trade.is_buyer_taker:
                    taker_id = trade.buyer_id
                    maker_id = trade.seller_id
                    taker_fee = trade.buyer_fee
                    maker_fee = trade.seller_fee
                    taker_is_buyer = True
                    buy_volume += trade.quantity
                    buy_amount += trade.price * trade.quantity
                else:
                    taker_id = trade.seller_id
                    maker_id = trade.buyer_id
                    taker_fee = trade.seller_fee
                    maker_fee = trade.buyer_fee
                    taker_is_buyer = False
                    sell_volume += trade.quantity
                    sell_amount += trade.price * trade.quantity

                # 更新 taker
                taker_state = agent_states_get(taker_id)
                if taker_state is not None:
                    taker_state.total_volume += trade.quantity
                    taker_state.trade_count += 1
                    side_int = 1 if taker_is_buyer else -1
                    realized_pnl = _update_position_cython(taker_state, side_int, trade.quantity, trade.price)
                    taker_state.balance += realized_pnl - taker_fee
                    # sync to array
                    idx_obj = agent_idx_get(taker_id)
                    if idx_obj is not None and balances_arr_ref is not None:
                        _inline_sync_state_to_array(taker_state, idx_obj, balances_arr_ref, pos_quantities_arr_ref, pos_avg_prices_arr_ref)
                else:
                    nt_taker = noise_states_get(taker_id)
                    if nt_taker is not None:
                        _update_noise_trader_position(nt_taker, trade.price, trade.quantity, taker_is_buyer)

                # 更新 maker
                maker_state_obj = agent_states_get(maker_id)
                if maker_state_obj is not None:
                    side_int = -1 if taker_is_buyer else 1  # maker 方向与 taker 相反
                    maker_state_obj.maker_volume += trade.quantity
                    maker_state_obj.total_volume += trade.quantity
                    maker_state_obj.trade_count += 1
                    realized_pnl = _update_position_cython(maker_state_obj, side_int, trade.quantity, trade.price)
                    maker_state_obj.balance += realized_pnl - maker_fee
                    # sync to array
                    idx_obj = agent_idx_get(maker_id)
                    if idx_obj is not None and balances_arr_ref is not None:
                        _inline_sync_state_to_array(maker_state_obj, idx_obj, balances_arr_ref, pos_quantities_arr_ref, pos_avg_prices_arr_ref)
                else:
                    nt_maker = noise_states_get(maker_id)
                    if nt_maker is not None:
                        _update_noise_trader_position(nt_maker, trade.price, trade.quantity, not taker_is_buyer)

            # 更新挂单状态
            if orderbook.has_order(order_id):
                if is_mm:
                    mm_state = agent_states_get(act_agent_id)
                    if mm_state is not None:
                        mm_state.bid_order_ids.append(order_id)
                elif not is_nt:
                    r_state = agent_states_get(act_agent_id)
                    if r_state is not None:
                        r_state.pending_order_id = order_id
            elif not is_mm and not is_nt:
                r_state = agent_states_get(act_agent_id)
                if r_state is not None:
                    r_state.pending_order_id = None

        # ---- LIMIT_SELL ----
        elif act_type == AT_LIMIT_SELL:
            # 生成 order_id
            if is_nt:
                nt_state = noise_states_get(act_agent_id)
                if nt_state is None:
                    continue
                order_id = _generate_order_id_noise(nt_state, arena_id)
            else:
                agent_state = agent_states_get(act_agent_id)
                if agent_state is None:
                    continue
                order_id = _generate_order_id_agent(agent_state, arena_id)

            trades = me.process_order_raw(order_id, act_agent_id, -1, 1, act_price, act_qty)

            for trade in trades:
                all_trades.append(trade)
                recent_trades_append(trade)

                # 内联 _update_trade_participants
                if trade.is_buyer_taker:
                    taker_id = trade.buyer_id
                    maker_id = trade.seller_id
                    taker_fee = trade.buyer_fee
                    maker_fee = trade.seller_fee
                    taker_is_buyer = True
                    buy_volume += trade.quantity
                    buy_amount += trade.price * trade.quantity
                else:
                    taker_id = trade.seller_id
                    maker_id = trade.buyer_id
                    taker_fee = trade.seller_fee
                    maker_fee = trade.buyer_fee
                    taker_is_buyer = False
                    sell_volume += trade.quantity
                    sell_amount += trade.price * trade.quantity

                # 更新 taker
                taker_state = agent_states_get(taker_id)
                if taker_state is not None:
                    taker_state.total_volume += trade.quantity
                    taker_state.trade_count += 1
                    side_int = 1 if taker_is_buyer else -1
                    realized_pnl = _update_position_cython(taker_state, side_int, trade.quantity, trade.price)
                    taker_state.balance += realized_pnl - taker_fee
                    idx_obj = agent_idx_get(taker_id)
                    if idx_obj is not None and balances_arr_ref is not None:
                        _inline_sync_state_to_array(taker_state, idx_obj, balances_arr_ref, pos_quantities_arr_ref, pos_avg_prices_arr_ref)
                else:
                    nt_taker = noise_states_get(taker_id)
                    if nt_taker is not None:
                        _update_noise_trader_position(nt_taker, trade.price, trade.quantity, taker_is_buyer)

                # 更新 maker
                maker_state_obj = agent_states_get(maker_id)
                if maker_state_obj is not None:
                    side_int = -1 if taker_is_buyer else 1
                    maker_state_obj.maker_volume += trade.quantity
                    maker_state_obj.total_volume += trade.quantity
                    maker_state_obj.trade_count += 1
                    realized_pnl = _update_position_cython(maker_state_obj, side_int, trade.quantity, trade.price)
                    maker_state_obj.balance += realized_pnl - maker_fee
                    idx_obj = agent_idx_get(maker_id)
                    if idx_obj is not None and balances_arr_ref is not None:
                        _inline_sync_state_to_array(maker_state_obj, idx_obj, balances_arr_ref, pos_quantities_arr_ref, pos_avg_prices_arr_ref)
                else:
                    nt_maker = noise_states_get(maker_id)
                    if nt_maker is not None:
                        _update_noise_trader_position(nt_maker, trade.price, trade.quantity, not taker_is_buyer)

            # 更新挂单状态
            if orderbook.has_order(order_id):
                if is_mm:
                    mm_state = agent_states_get(act_agent_id)
                    if mm_state is not None:
                        mm_state.ask_order_ids.append(order_id)
                elif not is_nt:
                    r_state = agent_states_get(act_agent_id)
                    if r_state is not None:
                        r_state.pending_order_id = order_id
            elif not is_mm and not is_nt:
                r_state = agent_states_get(act_agent_id)
                if r_state is not None:
                    r_state.pending_order_id = None

        # ---- MARKET_BUY ----
        elif act_type == AT_MARKET_BUY:
            # 生成 order_id
            if is_nt:
                nt_state = noise_states_get(act_agent_id)
                if nt_state is None:
                    continue
                order_id = _generate_order_id_noise(nt_state, arena_id)
            else:
                agent_state = agent_states_get(act_agent_id)
                if agent_state is None:
                    continue
                order_id = _generate_order_id_agent(agent_state, arena_id)

            trades = me.process_order_raw(order_id, act_agent_id, 1, 2, 0.0, act_qty)

            for trade in trades:
                all_trades.append(trade)
                recent_trades_append(trade)

                # 内联 _update_trade_participants
                if trade.is_buyer_taker:
                    taker_id = trade.buyer_id
                    maker_id = trade.seller_id
                    taker_fee = trade.buyer_fee
                    maker_fee = trade.seller_fee
                    taker_is_buyer = True
                    buy_volume += trade.quantity
                    buy_amount += trade.price * trade.quantity
                else:
                    taker_id = trade.seller_id
                    maker_id = trade.buyer_id
                    taker_fee = trade.seller_fee
                    maker_fee = trade.buyer_fee
                    taker_is_buyer = False
                    sell_volume += trade.quantity
                    sell_amount += trade.price * trade.quantity

                # 更新 taker
                taker_state = agent_states_get(taker_id)
                if taker_state is not None:
                    taker_state.total_volume += trade.quantity
                    taker_state.trade_count += 1
                    side_int = 1 if taker_is_buyer else -1
                    realized_pnl = _update_position_cython(taker_state, side_int, trade.quantity, trade.price)
                    taker_state.balance += realized_pnl - taker_fee
                    idx_obj = agent_idx_get(taker_id)
                    if idx_obj is not None and balances_arr_ref is not None:
                        _inline_sync_state_to_array(taker_state, idx_obj, balances_arr_ref, pos_quantities_arr_ref, pos_avg_prices_arr_ref)
                else:
                    nt_taker = noise_states_get(taker_id)
                    if nt_taker is not None:
                        _update_noise_trader_position(nt_taker, trade.price, trade.quantity, taker_is_buyer)

                # 更新 maker
                maker_state_obj = agent_states_get(maker_id)
                if maker_state_obj is not None:
                    side_int = -1 if taker_is_buyer else 1
                    maker_state_obj.maker_volume += trade.quantity
                    maker_state_obj.total_volume += trade.quantity
                    maker_state_obj.trade_count += 1
                    realized_pnl = _update_position_cython(maker_state_obj, side_int, trade.quantity, trade.price)
                    maker_state_obj.balance += realized_pnl - maker_fee
                    idx_obj = agent_idx_get(maker_id)
                    if idx_obj is not None and balances_arr_ref is not None:
                        _inline_sync_state_to_array(maker_state_obj, idx_obj, balances_arr_ref, pos_quantities_arr_ref, pos_avg_prices_arr_ref)
                else:
                    nt_maker = noise_states_get(maker_id)
                    if nt_maker is not None:
                        _update_noise_trader_position(nt_maker, trade.price, trade.quantity, not taker_is_buyer)

        # ---- MARKET_SELL ----
        elif act_type == AT_MARKET_SELL:
            # 生成 order_id
            if is_nt:
                nt_state = noise_states_get(act_agent_id)
                if nt_state is None:
                    continue
                order_id = _generate_order_id_noise(nt_state, arena_id)
            else:
                agent_state = agent_states_get(act_agent_id)
                if agent_state is None:
                    continue
                order_id = _generate_order_id_agent(agent_state, arena_id)

            trades = me.process_order_raw(order_id, act_agent_id, -1, 2, 0.0, act_qty)

            for trade in trades:
                all_trades.append(trade)
                recent_trades_append(trade)

                # 内联 _update_trade_participants
                if trade.is_buyer_taker:
                    taker_id = trade.buyer_id
                    maker_id = trade.seller_id
                    taker_fee = trade.buyer_fee
                    maker_fee = trade.seller_fee
                    taker_is_buyer = True
                    buy_volume += trade.quantity
                    buy_amount += trade.price * trade.quantity
                else:
                    taker_id = trade.seller_id
                    maker_id = trade.buyer_id
                    taker_fee = trade.seller_fee
                    maker_fee = trade.buyer_fee
                    taker_is_buyer = False
                    sell_volume += trade.quantity
                    sell_amount += trade.price * trade.quantity

                # 更新 taker
                taker_state = agent_states_get(taker_id)
                if taker_state is not None:
                    taker_state.total_volume += trade.quantity
                    taker_state.trade_count += 1
                    side_int = 1 if taker_is_buyer else -1
                    realized_pnl = _update_position_cython(taker_state, side_int, trade.quantity, trade.price)
                    taker_state.balance += realized_pnl - taker_fee
                    idx_obj = agent_idx_get(taker_id)
                    if idx_obj is not None and balances_arr_ref is not None:
                        _inline_sync_state_to_array(taker_state, idx_obj, balances_arr_ref, pos_quantities_arr_ref, pos_avg_prices_arr_ref)
                else:
                    nt_taker = noise_states_get(taker_id)
                    if nt_taker is not None:
                        _update_noise_trader_position(nt_taker, trade.price, trade.quantity, taker_is_buyer)

                # 更新 maker
                maker_state_obj = agent_states_get(maker_id)
                if maker_state_obj is not None:
                    side_int = -1 if taker_is_buyer else 1
                    maker_state_obj.maker_volume += trade.quantity
                    maker_state_obj.total_volume += trade.quantity
                    maker_state_obj.trade_count += 1
                    realized_pnl = _update_position_cython(maker_state_obj, side_int, trade.quantity, trade.price)
                    maker_state_obj.balance += realized_pnl - maker_fee
                    idx_obj = agent_idx_get(maker_id)
                    if idx_obj is not None and balances_arr_ref is not None:
                        _inline_sync_state_to_array(maker_state_obj, idx_obj, balances_arr_ref, pos_quantities_arr_ref, pos_avg_prices_arr_ref)
                else:
                    nt_maker = noise_states_get(maker_id)
                    if nt_maker is not None:
                        _update_noise_trader_position(nt_maker, trade.price, trade.quantity, not taker_is_buyer)

    # ====== 释放内存 ======
    free(action_types)
    free(agent_ids_arr)
    free(order_ids_arr)
    free(prices_arr)
    free(quantities_arr)
    free(flags_arr)
    free(shuffle_indices)

    # ====== 返回结果（与原始 aggregate_tick_trades 保持一致的符号语义）======
    cdef double total_volume = buy_volume + sell_volume
    cdef double total_amount = buy_amount + sell_amount

    if buy_amount > sell_amount:
        return (all_trades, total_volume, total_amount)
    elif sell_amount > buy_amount:
        return (all_trades, -total_volume, -total_amount)
    return (all_trades, 0.0, 0.0)
