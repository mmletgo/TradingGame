# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""
批量决策 OpenMP 并行模块

实现完全 nogil 的 OpenMP 并行批量决策，包括：
- 批量观察（observe）
- 批量前向传播（forward）
- 批量解析（parse）

所有核心计算在 nogil 块中执行，使用 OpenMP prange 实现真正的多核并行。
"""

from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset, memcpy
from libc.math cimport tanh, exp, fabs, fmax, fmin, log10, sin
cimport openmp
from cython.parallel cimport prange, parallel
cimport numpy as np
import numpy as np

# 类型定义
from numpy cimport float64_t as DTYPE_t, float32_t as DTYPE32_t

# 从 .pxd 文件导入结构体定义
from src.training._cython.batch_decide_openmp cimport (
    BatchNetworkData,
    BatchAgentState,
    ThreadLocalBuffer,
    MarketStateData,
    DecisionResult,
    MarketMakerOrdersResult,
)

# ============================================================================
# 常量定义
# ============================================================================

# 激活函数类型
DEF ACT_SIGMOID = 0
DEF ACT_TANH = 1
DEF ACT_RELU = 2
DEF ACT_IDENTITY = 3
DEF ACT_SIN = 4
DEF ACT_GAUSS = 5

# Agent 类型
DEF AGENT_RETAIL = 0
DEF AGENT_RETAIL_PRO = 1
DEF AGENT_WHALE = 2
DEF AGENT_MARKET_MAKER = 3

# 输入/输出维度
DEF INPUT_DIM_RETAIL = 127
DEF INPUT_DIM_FULL = 907
DEF INPUT_DIM_MARKET_MAKER = 964
DEF OUTPUT_DIM_RETAIL = 8
DEF OUTPUT_DIM_MARKET_MAKER = 41

# 动作类型（散户/高级散户/庄家共用6种动作）
DEF ACTION_HOLD = 0
DEF ACTION_PLACE_BID = 1
DEF ACTION_PLACE_ASK = 2
DEF ACTION_CANCEL = 3
DEF ACTION_MARKET_BUY = 4
DEF ACTION_MARKET_SELL = 5
DEF ACTION_QUOTE = 6  # 做市商专用


# ============================================================================
# 内联激活函数
# ============================================================================

cdef inline double activate(double x, int act_type) noexcept nogil:
    """内联激活函数"""
    if act_type == ACT_SIGMOID:
        return 1.0 / (1.0 + exp(-x))
    elif act_type == ACT_TANH:
        return tanh(x)
    elif act_type == ACT_RELU:
        return x if x > 0 else 0.0
    elif act_type == ACT_IDENTITY:
        return x
    elif act_type == ACT_SIN:
        return sin(x)
    elif act_type == ACT_GAUSS:
        return exp(-x * x)
    else:
        return tanh(x)  # 默认 tanh


cdef inline double clip(double x, double min_val, double max_val) noexcept nogil:
    """裁剪值到指定范围"""
    if x < min_val:
        return min_val
    elif x > max_val:
        return max_val
    return x


cdef inline int argmax(double* arr, int start, int end) noexcept nogil:
    """查找最大值索引"""
    cdef int max_idx = start
    cdef double max_val = arr[start]
    cdef int i
    for i in range(start + 1, end):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_idx


cdef inline double round_price(double price, double tick_size) noexcept nogil:
    """将价格取整到 tick_size 的整数倍"""
    cdef double rounded = (<int>(price / tick_size + 0.5)) * tick_size
    if rounded < tick_size:
        rounded = tick_size
    return rounded


# ============================================================================
# 内存管理函数
# ============================================================================

cdef BatchNetworkData* alloc_batch_network_data(
    int num_networks,
    int max_nodes,
    int max_connections,
    int max_inputs,
    int max_outputs
) noexcept:
    """分配批量网络数据结构"""
    cdef BatchNetworkData* data = <BatchNetworkData*>malloc(sizeof(BatchNetworkData))
    if data == NULL:
        return NULL

    data.num_networks = num_networks
    data.max_nodes = max_nodes
    data.max_connections = max_connections
    data.max_inputs = max_inputs
    data.max_outputs = max_outputs

    # 分配元信息数组
    data.num_inputs_arr = <int*>calloc(num_networks, sizeof(int))
    data.num_outputs_arr = <int*>calloc(num_networks, sizeof(int))
    data.num_nodes_arr = <int*>calloc(num_networks, sizeof(int))
    data.node_offsets = <int*>calloc(num_networks, sizeof(int))
    data.conn_offsets = <int*>calloc(num_networks, sizeof(int))
    data.output_idx_offsets = <int*>calloc(num_networks, sizeof(int))

    # 分配扁平化数组（使用最大可能大小）
    cdef int total_nodes = num_networks * max_nodes
    cdef int total_conns = num_networks * max_connections
    cdef int total_outputs = num_networks * max_outputs

    data.biases = <double*>calloc(total_nodes, sizeof(double))
    data.responses = <double*>calloc(total_nodes, sizeof(double))
    data.act_types = <int*>calloc(total_nodes, sizeof(int))

    data.conn_indptr = <int*>calloc(total_nodes + num_networks, sizeof(int))
    data.conn_sources = <int*>calloc(total_conns, sizeof(int))
    data.conn_weights = <double*>calloc(total_conns, sizeof(double))

    data.output_indices = <int*>calloc(total_outputs, sizeof(int))

    # 检查分配是否成功
    if (data.num_inputs_arr == NULL or data.num_outputs_arr == NULL or
        data.num_nodes_arr == NULL or data.node_offsets == NULL or
        data.conn_offsets == NULL or data.output_idx_offsets == NULL or
        data.biases == NULL or data.responses == NULL or data.act_types == NULL or
        data.conn_indptr == NULL or data.conn_sources == NULL or
        data.conn_weights == NULL or data.output_indices == NULL):
        free_batch_network_data(data)
        return NULL

    return data


cdef void free_batch_network_data(BatchNetworkData* data) noexcept:
    """释放批量网络数据结构"""
    if data == NULL:
        return

    if data.num_inputs_arr != NULL:
        free(data.num_inputs_arr)
    if data.num_outputs_arr != NULL:
        free(data.num_outputs_arr)
    if data.num_nodes_arr != NULL:
        free(data.num_nodes_arr)
    if data.node_offsets != NULL:
        free(data.node_offsets)
    if data.conn_offsets != NULL:
        free(data.conn_offsets)
    if data.output_idx_offsets != NULL:
        free(data.output_idx_offsets)
    if data.biases != NULL:
        free(data.biases)
    if data.responses != NULL:
        free(data.responses)
    if data.act_types != NULL:
        free(data.act_types)
    if data.conn_indptr != NULL:
        free(data.conn_indptr)
    if data.conn_sources != NULL:
        free(data.conn_sources)
    if data.conn_weights != NULL:
        free(data.conn_weights)
    if data.output_indices != NULL:
        free(data.output_indices)

    free(data)


cdef BatchAgentState* alloc_batch_agent_state(int num_agents) noexcept:
    """分配批量 Agent 状态结构"""
    cdef BatchAgentState* data = <BatchAgentState*>malloc(sizeof(BatchAgentState))
    if data == NULL:
        return NULL

    data.num_agents = num_agents

    # 账户信息
    data.balance = <double*>calloc(num_agents, sizeof(double))
    data.position_quantity = <double*>calloc(num_agents, sizeof(double))
    data.position_avg_price = <double*>calloc(num_agents, sizeof(double))
    data.unrealized_pnl = <double*>calloc(num_agents, sizeof(double))
    data.margin_ratio = <double*>calloc(num_agents, sizeof(double))
    data.available_margin = <double*>calloc(num_agents, sizeof(double))
    data.leverage = <double*>calloc(num_agents, sizeof(double))

    # 挂单信息
    data.has_pending_order = <int*>calloc(num_agents, sizeof(int))
    data.pending_side = <int*>calloc(num_agents, sizeof(int))
    data.pending_price = <double*>calloc(num_agents, sizeof(double))
    data.pending_quantity = <double*>calloc(num_agents, sizeof(double))

    # 检查分配
    if (data.balance == NULL or data.position_quantity == NULL or
        data.position_avg_price == NULL or data.unrealized_pnl == NULL or
        data.margin_ratio == NULL or data.available_margin == NULL or
        data.leverage == NULL or
        data.has_pending_order == NULL or data.pending_side == NULL or
        data.pending_price == NULL or data.pending_quantity == NULL):
        free_batch_agent_state(data)
        return NULL

    return data


cdef void free_batch_agent_state(BatchAgentState* data) noexcept:
    """释放批量 Agent 状态结构"""
    if data == NULL:
        return

    if data.balance != NULL:
        free(data.balance)
    if data.position_quantity != NULL:
        free(data.position_quantity)
    if data.position_avg_price != NULL:
        free(data.position_avg_price)
    if data.unrealized_pnl != NULL:
        free(data.unrealized_pnl)
    if data.margin_ratio != NULL:
        free(data.margin_ratio)
    if data.available_margin != NULL:
        free(data.available_margin)
    if data.leverage != NULL:
        free(data.leverage)
    if data.has_pending_order != NULL:
        free(data.has_pending_order)
    if data.pending_side != NULL:
        free(data.pending_side)
    if data.pending_price != NULL:
        free(data.pending_price)
    if data.pending_quantity != NULL:
        free(data.pending_quantity)

    free(data)


cdef ThreadLocalBuffer* alloc_thread_buffers(
    int num_threads,
    int max_nodes,
    int max_inputs,
    int max_outputs
) noexcept:
    """分配线程本地缓冲区数组"""
    cdef ThreadLocalBuffer* buffers = <ThreadLocalBuffer*>calloc(
        num_threads, sizeof(ThreadLocalBuffer)
    )
    if buffers == NULL:
        return NULL

    cdef int i
    cdef int values_size = max_nodes + max_inputs

    for i in range(num_threads):
        buffers[i].values = <double*>calloc(values_size, sizeof(double))
        buffers[i].inputs = <double*>calloc(max_inputs, sizeof(double))
        buffers[i].outputs = <double*>calloc(max_outputs, sizeof(double))

        if (buffers[i].values == NULL or buffers[i].inputs == NULL or
            buffers[i].outputs == NULL):
            free_thread_buffers(buffers, num_threads)
            return NULL

    return buffers


cdef void free_thread_buffers(ThreadLocalBuffer* buffers, int num_threads) noexcept:
    """释放线程本地缓冲区数组"""
    if buffers == NULL:
        return

    cdef int i
    for i in range(num_threads):
        if buffers[i].values != NULL:
            free(buffers[i].values)
        if buffers[i].inputs != NULL:
            free(buffers[i].inputs)
        if buffers[i].outputs != NULL:
            free(buffers[i].outputs)

    free(buffers)


cdef MarketStateData* alloc_market_state_data() noexcept:
    """分配市场状态数据结构"""
    cdef MarketStateData* data = <MarketStateData*>malloc(sizeof(MarketStateData))
    if data == NULL:
        return NULL

    data.mid_price = 0.0
    data.tick_size = 0.0

    data.bid_data = <double*>calloc(200, sizeof(double))
    data.ask_data = <double*>calloc(200, sizeof(double))
    data.trade_prices = <double*>calloc(100, sizeof(double))
    data.trade_quantities = <double*>calloc(100, sizeof(double))
    data.tick_history_prices = <double*>calloc(100, sizeof(double))
    data.tick_history_volumes = <double*>calloc(100, sizeof(double))
    data.tick_history_amounts = <double*>calloc(100, sizeof(double))

    if (data.bid_data == NULL or data.ask_data == NULL or
        data.trade_prices == NULL or data.trade_quantities == NULL or
        data.tick_history_prices == NULL or data.tick_history_volumes == NULL or
        data.tick_history_amounts == NULL):
        free_market_state_data(data)
        return NULL

    return data


cdef void free_market_state_data(MarketStateData* data) noexcept:
    """释放市场状态数据结构"""
    if data == NULL:
        return

    if data.bid_data != NULL:
        free(data.bid_data)
    if data.ask_data != NULL:
        free(data.ask_data)
    if data.trade_prices != NULL:
        free(data.trade_prices)
    if data.trade_quantities != NULL:
        free(data.trade_quantities)
    if data.tick_history_prices != NULL:
        free(data.tick_history_prices)
    if data.tick_history_volumes != NULL:
        free(data.tick_history_volumes)
    if data.tick_history_amounts != NULL:
        free(data.tick_history_amounts)

    free(data)


cdef MarketStateData** alloc_multi_market_state_data(int num_arenas) noexcept:
    """分配多个市场状态数据结构"""
    cdef MarketStateData** data = <MarketStateData**>malloc(num_arenas * sizeof(MarketStateData*))
    if data == NULL:
        return NULL

    cdef int i, j
    for i in range(num_arenas):
        data[i] = alloc_market_state_data()
        if data[i] == NULL:
            # 清理已分配的
            for j in range(i):
                free_market_state_data(data[j])
            free(data)
            return NULL

    return data


cdef void free_multi_market_state_data(MarketStateData** data, int num_arenas) noexcept:
    """释放多个市场状态数据结构"""
    if data == NULL:
        return

    cdef int i
    for i in range(num_arenas):
        if data[i] != NULL:
            free_market_state_data(data[i])

    free(data)


cdef MarketMakerOrdersResult* alloc_mm_orders_results(int num_agents) noexcept:
    """分配做市商订单结果数组"""
    return <MarketMakerOrdersResult*>calloc(num_agents, sizeof(MarketMakerOrdersResult))


cdef void free_mm_orders_results(MarketMakerOrdersResult* data) noexcept:
    """释放做市商订单结果数组"""
    if data != NULL:
        free(data)


# ============================================================================
# 数据提取函数
# ============================================================================

cdef void _extract_networks_to_batch(list networks, BatchNetworkData* batch_data):
    """从 FastFeedForwardNetwork 列表提取数据到批量结构

    Args:
        networks: FastFeedForwardNetwork 实例列表
        batch_data: 预分配的 BatchNetworkData 结构
    """
    cdef int num_networks = len(networks)
    cdef int i, j, k
    cdef int node_offset = 0
    cdef int conn_offset = 0
    cdef int output_offset = 0
    cdef int num_conns
    cdef int num_nodes_i, num_outputs_i

    for i in range(num_networks):
        net = networks[i]

        # 基本信息
        batch_data.num_inputs_arr[i] = net.num_inputs
        batch_data.num_outputs_arr[i] = net.num_outputs
        batch_data.num_nodes_arr[i] = net.num_nodes
        num_nodes_i = net.num_nodes
        num_outputs_i = net.num_outputs

        # 偏移量
        batch_data.node_offsets[i] = node_offset
        batch_data.conn_offsets[i] = conn_offset
        batch_data.output_idx_offsets[i] = output_offset

        # 复制节点数据（使用普通 Python 变量）
        biases = net.biases
        responses = net.responses
        act_types = net.act_types

        for j in range(num_nodes_i):
            batch_data.biases[node_offset + j] = biases[j]
            batch_data.responses[node_offset + j] = responses[j]
            batch_data.act_types[node_offset + j] = act_types[j]

        # 复制连接数据（CSR 格式）
        conn_indptr = net.conn_indptr
        conn_sources = net.conn_sources
        conn_weights = net.conn_weights

        num_conns = len(conn_sources)

        # 复制 indptr（需要调整偏移）
        for j in range(num_nodes_i + 1):
            batch_data.conn_indptr[node_offset + j] = conn_indptr[j] + conn_offset

        # 复制源节点和权重
        for j in range(num_conns):
            batch_data.conn_sources[conn_offset + j] = conn_sources[j]
            batch_data.conn_weights[conn_offset + j] = conn_weights[j]

        # 复制输出索引
        output_indices = net.output_indices
        for j in range(num_outputs_i):
            batch_data.output_indices[output_offset + j] = output_indices[j]

        # 更新偏移量
        node_offset += num_nodes_i
        conn_offset += num_conns
        output_offset += num_outputs_i



cdef void _fill_batch_data_from_numpy(
    BatchNetworkData* batch_data,
    # header 信息
    int[:] h_num_inputs,
    int[:] h_num_outputs,
    int[:] h_num_nodes,
    int[:] h_n_connections,
    int num_networks,
    # 节点数据（packed，float32/int32）
    float[:] all_biases,
    float[:] all_responses,
    int[:] all_act_types,
    # 连接数据（packed）
    int[:] all_conn_indptr,
    int[:] all_conn_sources,
    float[:] all_conn_weights,
    # 输出索引
    int[:] all_output_indices,
    int[:] h_n_output_keys,
) noexcept:
    """从 packed numpy 数组直接填充 BatchNetworkData C 结构
    
    跳过 Python 对象创建，直接使用 typed memoryview 读取数据。
    """
    cdef int i, j
    cdef int node_offset = 0
    cdef int conn_offset = 0
    cdef int output_offset = 0
    cdef int src_node_offset = 0
    cdef int src_conn_offset = 0
    cdef int src_indptr_offset = 0
    cdef int src_output_offset = 0
    cdef int num_nodes_i, num_conns_i, num_outputs_i, num_inputs_i
    
    for i in range(num_networks):
        num_inputs_i = h_num_inputs[i]
        num_outputs_i = h_num_outputs[i]
        num_nodes_i = h_num_nodes[i]
        num_conns_i = h_n_connections[i]
        
        # 元信息
        batch_data.num_inputs_arr[i] = num_inputs_i
        batch_data.num_outputs_arr[i] = num_outputs_i
        batch_data.num_nodes_arr[i] = num_nodes_i
        batch_data.node_offsets[i] = node_offset
        batch_data.conn_offsets[i] = conn_offset
        batch_data.output_idx_offsets[i] = output_offset
        
        # 节点数据 (float32 -> double, int32 -> int)
        for j in range(num_nodes_i):
            batch_data.biases[node_offset + j] = <double>all_biases[src_node_offset + j]
            batch_data.responses[node_offset + j] = <double>all_responses[src_node_offset + j]
            batch_data.act_types[node_offset + j] = all_act_types[src_node_offset + j]
        
        # 连接 indptr (需要加 conn_offset)
        for j in range(num_nodes_i + 1):
            batch_data.conn_indptr[node_offset + j] = all_conn_indptr[src_indptr_offset + j] + conn_offset
        
        # 连接 sources 和 weights
        for j in range(num_conns_i):
            batch_data.conn_sources[conn_offset + j] = all_conn_sources[src_conn_offset + j]
            batch_data.conn_weights[conn_offset + j] = <double>all_conn_weights[src_conn_offset + j]
        
        # 输出索引
        for j in range(h_n_output_keys[i]):
            batch_data.output_indices[output_offset + j] = all_output_indices[src_output_offset + j]
        
        # 更新 batch 偏移量
        node_offset += num_nodes_i
        conn_offset += num_conns_i
        output_offset += h_n_output_keys[i]
        
        # 更新源数据偏移量
        src_node_offset += num_nodes_i
        src_conn_offset += num_conns_i
        src_indptr_offset += num_nodes_i + 1
        src_output_offset += h_n_output_keys[i]


cdef void _extract_agents_to_batch(list agents, BatchAgentState* batch_data, double mid_price):
    """从 Agent 列表提取状态到批量结构

    Args:
        agents: Agent 实例列表
        batch_data: 预分配的 BatchAgentState 结构
        mid_price: 当前中间价（用于计算未实现盈亏）
    """
    cdef int num_agents = len(agents)
    cdef int i
    cdef int pos_qty

    for i in range(num_agents):
        agent = agents[i]
        account = agent.account
        position = account.position
        pos_qty = position.quantity

        # 账户信息
        batch_data.balance[i] = account.balance
        batch_data.position_quantity[i] = <double>pos_qty
        batch_data.position_avg_price[i] = position.avg_price

        # 计算未实现盈亏
        if pos_qty != 0:
            if pos_qty > 0:
                batch_data.unrealized_pnl[i] = (mid_price - position.avg_price) * pos_qty
            else:
                batch_data.unrealized_pnl[i] = (position.avg_price - mid_price) * (-pos_qty)
        else:
            batch_data.unrealized_pnl[i] = 0.0

        # 保证金信息
        batch_data.margin_ratio[i] = account.get_margin_ratio(mid_price)
        # available_margin 不存在于 Account 类，使用 get_equity 代替
        batch_data.available_margin[i] = account.get_equity(mid_price)
        # 杠杆倍数（用于计算 quantity）
        batch_data.leverage[i] = account.leverage

        # 挂单信息（简化处理）
        batch_data.has_pending_order[i] = 0
        batch_data.pending_side[i] = 0
        batch_data.pending_price[i] = 0.0
        batch_data.pending_quantity[i] = 0.0


cdef void _extract_market_state(object market_state, MarketStateData* data):
    """从 NormalizedMarketState 提取数据

    Args:
        market_state: NormalizedMarketState 实例
        data: 预分配的 MarketStateData 结构
    """
    cdef int i

    data.mid_price = market_state.mid_price
    data.tick_size = market_state.tick_size

    # 复制订单簿数据
    bid_data = market_state.bid_data
    ask_data = market_state.ask_data

    for i in range(200):
        data.bid_data[i] = <double>bid_data[i]
        data.ask_data[i] = <double>ask_data[i]

    # 复制成交数据
    trade_prices = market_state.trade_prices
    trade_quantities = market_state.trade_quantities

    for i in range(100):
        data.trade_prices[i] = <double>trade_prices[i]
        data.trade_quantities[i] = <double>trade_quantities[i]

    # 复制 tick 历史数据
    tick_history_prices = market_state.tick_history_prices
    tick_history_volumes = market_state.tick_history_volumes
    tick_history_amounts = market_state.tick_history_amounts

    for i in range(100):
        data.tick_history_prices[i] = <double>tick_history_prices[i]
        data.tick_history_volumes[i] = <double>tick_history_volumes[i]
        data.tick_history_amounts[i] = <double>tick_history_amounts[i]


# ============================================================================
# 批量观察函数 (nogil)
# ============================================================================

cdef void _observe_retail_single(
    int agent_idx,
    BatchAgentState* agents,
    MarketStateData* market,
    double* output,
    double initial_balance,
    double leverage,
) noexcept nogil:
    """构建散户的神经网络输入向量（127 维）

    输入布局：
    - 0-19: 买盘前10档（每档2个值：价格归一化 + 数量）
    - 20-39: 卖盘前10档（每档2个值：价格归一化 + 数量）
    - 40-49: 最近10笔成交价格
    - 50-59: 最近10笔成交数量
    - 60-63: 持仓信息（4个值）
    - 64-66: 挂单信息（3个值）
    - 67-86: tick历史价格（最近20个）
    - 87-106: tick历史成交量（最近20个）
    - 107-126: tick历史成交额（最近20个）
    """
    cdef int i
    cdef double mid_price = market.mid_price
    cdef double balance = agents.balance[agent_idx]
    cdef double pos_qty = agents.position_quantity[agent_idx]
    cdef double pos_avg_price = agents.position_avg_price[agent_idx]
    cdef double unrealized_pnl = agents.unrealized_pnl[agent_idx]
    cdef double equity = balance + unrealized_pnl

    # 买盘前10档（20个值）
    for i in range(20):
        output[i] = market.bid_data[i]

    # 卖盘前10档（20个值）
    for i in range(20):
        output[20 + i] = market.ask_data[i]

    # 最近10笔成交价格
    for i in range(10):
        output[40 + i] = market.trade_prices[i]

    # 最近10笔成交数量
    for i in range(10):
        output[50 + i] = market.trade_quantities[i]

    # 持仓信息（4个值）
    cdef double position_value = fabs(pos_qty) * mid_price
    if equity > 0 and leverage > 0:
        output[60] = position_value / (equity * leverage)
    else:
        output[60] = 0.0

    if pos_qty == 0 or mid_price == 0:
        output[61] = 0.0
    else:
        output[61] = (pos_avg_price - mid_price) / mid_price

    if initial_balance > 0:
        output[62] = balance / initial_balance
        output[63] = equity / initial_balance
    else:
        output[62] = 0.0
        output[63] = 0.0

    # 挂单信息（3个值）- 简化处理
    output[64] = 0.0  # pending_price_normalized
    output[65] = 0.0  # pending_qty_normalized
    output[66] = 0.0  # pending_side

    # tick历史价格（最近20个，从100维数组的后20个取）
    for i in range(20):
        output[67 + i] = market.tick_history_prices[80 + i]

    # tick历史成交量（最近20个）
    for i in range(20):
        output[87 + i] = market.tick_history_volumes[80 + i]

    # tick历史成交额（最近20个）
    for i in range(20):
        output[107 + i] = market.tick_history_amounts[80 + i]


cdef void batch_observe_retail_nogil(
    BatchAgentState* agents,
    MarketStateData* market,
    double[:, :] outputs,
    int num_threads
) noexcept nogil:
    """批量构建散户的神经网络输入向量（127 维）"""
    cdef int num_agents = agents.num_agents
    cdef int i
    cdef double initial_balance = 100000.0  # 散户初始资金
    cdef double leverage = 100.0  # 散户杠杆

    for i in prange(num_agents, nogil=True, num_threads=num_threads, schedule='static'):
        _observe_retail_single(i, agents, market, &outputs[i, 0], initial_balance, leverage)


cdef void _observe_full_single(
    int agent_idx,
    BatchAgentState* agents,
    MarketStateData* market,
    double* output,
    double initial_balance,
    double leverage,
) noexcept nogil:
    """构建高级散户/庄家的神经网络输入向量（907 维）

    输入布局：
    - 0-199: 买盘100档（每档2个值：价格归一化 + 数量）
    - 200-399: 卖盘100档（每档2个值：价格归一化 + 数量）
    - 400-499: 最近100笔成交价格
    - 500-599: 最近100笔成交数量
    - 600-603: 持仓信息（4个值）
    - 604-606: 挂单信息（3个值）
    - 607-706: tick历史价格（100个）
    - 707-806: tick历史成交量（100个）
    - 807-906: tick历史成交额（100个）
    """
    cdef int i
    cdef double mid_price = market.mid_price
    cdef double balance = agents.balance[agent_idx]
    cdef double pos_qty = agents.position_quantity[agent_idx]
    cdef double pos_avg_price = agents.position_avg_price[agent_idx]
    cdef double unrealized_pnl = agents.unrealized_pnl[agent_idx]
    cdef double equity = balance + unrealized_pnl

    # 买盘100档（200个值）
    for i in range(200):
        output[i] = market.bid_data[i]

    # 卖盘100档（200个值）
    for i in range(200):
        output[200 + i] = market.ask_data[i]

    # 最近100笔成交价格
    for i in range(100):
        output[400 + i] = market.trade_prices[i]

    # 最近100笔成交数量
    for i in range(100):
        output[500 + i] = market.trade_quantities[i]

    # 持仓信息（4个值）
    cdef double position_value = fabs(pos_qty) * mid_price
    if equity > 0 and leverage > 0:
        output[600] = position_value / (equity * leverage)
    else:
        output[600] = 0.0

    if pos_qty == 0 or mid_price == 0:
        output[601] = 0.0
    else:
        output[601] = (pos_avg_price - mid_price) / mid_price

    if initial_balance > 0:
        output[602] = balance / initial_balance
        output[603] = equity / initial_balance
    else:
        output[602] = 0.0
        output[603] = 0.0

    # 挂单信息（3个值）- 简化处理
    output[604] = 0.0
    output[605] = 0.0
    output[606] = 0.0

    # tick历史价格（100个）
    for i in range(100):
        output[607 + i] = market.tick_history_prices[i]

    # tick历史成交量（100个）
    for i in range(100):
        output[707 + i] = market.tick_history_volumes[i]

    # tick历史成交额（100个）
    for i in range(100):
        output[807 + i] = market.tick_history_amounts[i]


cdef void batch_observe_full_nogil(
    BatchAgentState* agents,
    MarketStateData* market,
    double[:, :] outputs,
    int num_threads
) noexcept nogil:
    """批量构建高级散户/庄家的神经网络输入向量（907 维）"""
    cdef int num_agents = agents.num_agents
    cdef int i
    cdef double initial_balance = 100000.0  # 默认值，高级散户
    cdef double leverage = 100.0

    for i in prange(num_agents, nogil=True, num_threads=num_threads, schedule='static'):
        _observe_full_single(i, agents, market, &outputs[i, 0], initial_balance, leverage)


cdef void _observe_market_maker_single(
    int agent_idx,
    BatchAgentState* agents,
    MarketStateData* market,
    double* output,
    double initial_balance,
    double leverage,
) noexcept nogil:
    """构建做市商的神经网络输入向量（964 维）

    输入布局：
    - 0-199: 买盘100档（每档2个值：价格归一化 + 数量）
    - 200-399: 卖盘100档（每档2个值：价格归一化 + 数量）
    - 400-499: 最近100笔成交价格
    - 500-599: 最近100笔成交数量
    - 600-603: 持仓信息（4个值）
    - 604-663: 挂单信息（60个值：10买单+10卖单，每单3个值）
    - 664-763: tick历史价格（100个）
    - 764-863: tick历史成交量（100个）
    - 864-963: tick历史成交额（100个）
    """
    cdef int i
    cdef double mid_price = market.mid_price
    cdef double balance = agents.balance[agent_idx]
    cdef double pos_qty = agents.position_quantity[agent_idx]
    cdef double pos_avg_price = agents.position_avg_price[agent_idx]
    cdef double unrealized_pnl = agents.unrealized_pnl[agent_idx]
    cdef double equity = balance + unrealized_pnl

    # 买盘100档（200个值）
    for i in range(200):
        output[i] = market.bid_data[i]

    # 卖盘100档（200个值）
    for i in range(200):
        output[200 + i] = market.ask_data[i]

    # 最近100笔成交价格
    for i in range(100):
        output[400 + i] = market.trade_prices[i]

    # 最近100笔成交数量
    for i in range(100):
        output[500 + i] = market.trade_quantities[i]

    # 持仓信息（4个值）
    cdef double position_value = fabs(pos_qty) * mid_price
    if equity > 0 and leverage > 0:
        output[600] = position_value / (equity * leverage)
    else:
        output[600] = 0.0

    if pos_qty == 0 or mid_price == 0:
        output[601] = 0.0
    else:
        output[601] = (pos_avg_price - mid_price) / mid_price

    if initial_balance > 0:
        output[602] = balance / initial_balance
        output[603] = equity / initial_balance
    else:
        output[602] = 0.0
        output[603] = 0.0

    # 挂单信息（60个值）- 简化处理，全部置零
    for i in range(60):
        output[604 + i] = 0.0

    # tick历史价格（100个）
    for i in range(100):
        output[664 + i] = market.tick_history_prices[i]

    # tick历史成交量（100个）
    for i in range(100):
        output[764 + i] = market.tick_history_volumes[i]

    # tick历史成交额（100个）
    for i in range(100):
        output[864 + i] = market.tick_history_amounts[i]


cdef void batch_observe_market_maker_nogil(
    BatchAgentState* agents,
    MarketStateData* market,
    double[:, :] outputs,
    int num_threads
) noexcept nogil:
    """批量构建做市商的神经网络输入向量（964 维）"""
    cdef int num_agents = agents.num_agents
    cdef int i
    cdef double initial_balance = 10000000.0  # 做市商初始资金
    cdef double leverage = 10.0  # 做市商杠杆

    for i in prange(num_agents, nogil=True, num_threads=num_threads, schedule='static'):
        _observe_market_maker_single(i, agents, market, &outputs[i, 0], initial_balance, leverage)


# ============================================================================
# 多市场状态版本的批量观察函数 (nogil)
# ============================================================================

cdef void batch_observe_retail_multi_market_nogil(
    BatchAgentState* agents,
    MarketStateData** markets,
    int* market_indices,
    double[:, :] outputs,
    int num_threads
) noexcept nogil:
    """批量构建散户的神经网络输入向量（支持多市场状态）"""
    cdef int num_agents = agents.num_agents
    cdef int i
    cdef double initial_balance = 100000.0
    cdef double leverage = 100.0
    cdef int market_idx

    for i in prange(num_agents, nogil=True, num_threads=num_threads, schedule='static'):
        market_idx = market_indices[i]
        _observe_retail_single(i, agents, markets[market_idx], &outputs[i, 0], initial_balance, leverage)


cdef void batch_observe_full_multi_market_nogil(
    BatchAgentState* agents,
    MarketStateData** markets,
    int* market_indices,
    double[:, :] outputs,
    int num_threads
) noexcept nogil:
    """批量构建高级散户/庄家的神经网络输入向量（支持多市场状态）"""
    cdef int num_agents = agents.num_agents
    cdef int i
    cdef double initial_balance = 100000.0
    cdef double leverage = 100.0
    cdef int market_idx

    for i in prange(num_agents, nogil=True, num_threads=num_threads, schedule='static'):
        market_idx = market_indices[i]
        _observe_full_single(i, agents, markets[market_idx], &outputs[i, 0], initial_balance, leverage)


cdef void batch_observe_market_maker_multi_market_nogil(
    BatchAgentState* agents,
    MarketStateData** markets,
    int* market_indices,
    double[:, :] outputs,
    int num_threads
) noexcept nogil:
    """批量构建做市商的神经网络输入向量（支持多市场状态）"""
    cdef int num_agents = agents.num_agents
    cdef int i
    cdef double initial_balance = 10000000.0
    cdef double leverage = 10.0
    cdef int market_idx

    for i in prange(num_agents, nogil=True, num_threads=num_threads, schedule='static'):
        market_idx = market_indices[i]
        _observe_market_maker_single(i, agents, markets[market_idx], &outputs[i, 0], initial_balance, leverage)


# ============================================================================
# 批量前向传播函数 (nogil)
# ============================================================================

cdef void _forward_single(
    int net_idx,
    BatchNetworkData* networks,
    double* input_data,
    double* output_data,
    double* values_buffer,
) noexcept nogil:
    """单个网络的前向传播

    Args:
        net_idx: 网络索引
        networks: 批量网络数据
        input_data: 输入数据指针
        output_data: 输出数据指针
        values_buffer: 节点值缓冲区
    """
    cdef int num_inputs = networks.num_inputs_arr[net_idx]
    cdef int num_outputs = networks.num_outputs_arr[net_idx]
    cdef int num_nodes = networks.num_nodes_arr[net_idx]
    cdef int node_offset = networks.node_offsets[net_idx]
    cdef int output_offset = networks.output_idx_offsets[net_idx]

    cdef int i, j, start, end, src_idx
    cdef double node_sum, weighted_input

    # 1. 复制输入到 values 缓冲区
    for i in range(num_inputs):
        values_buffer[i] = input_data[i]

    # 2. 前向传播
    for i in range(num_nodes):
        # 获取该节点的连接范围
        start = networks.conn_indptr[node_offset + i]
        end = networks.conn_indptr[node_offset + i + 1]

        # 计算加权输入和
        node_sum = 0.0
        for j in range(start, end):
            src_idx = networks.conn_sources[j]
            node_sum += values_buffer[src_idx] * networks.conn_weights[j]

        # 应用偏置、响应和激活函数
        weighted_input = networks.biases[node_offset + i] + networks.responses[node_offset + i] * node_sum
        values_buffer[num_inputs + i] = activate(weighted_input, networks.act_types[node_offset + i])

    # 3. 提取输出
    for i in range(num_outputs):
        output_data[i] = values_buffer[networks.output_indices[output_offset + i]]


cdef void batch_forward_nogil(
    BatchNetworkData* networks,
    double[:, :] inputs,
    double[:, :] outputs,
    ThreadLocalBuffer* buffers,
    int num_threads
) noexcept nogil:
    """批量前向传播

    使用 OpenMP 并行执行多个网络的前向传播。
    每个线程使用独立的 values 缓冲区避免竞争。

    Args:
        networks: 批量网络数据
        inputs: 输入数组 [num_networks, num_inputs]
        outputs: 输出数组 [num_networks, num_outputs]
        buffers: 线程本地缓冲区数组
        num_threads: 线程数
    """
    cdef int num_networks = networks.num_networks
    cdef int i
    cdef int tid

    for i in prange(num_networks, nogil=True, num_threads=num_threads, schedule='dynamic'):
        tid = openmp.omp_get_thread_num()
        _forward_single(
            i, networks,
            &inputs[i, 0],
            &outputs[i, 0],
            buffers[tid].values
        )


cdef void batch_forward_with_indices_nogil(
    BatchNetworkData* networks,
    int* network_indices,
    double[:, :] inputs,
    double[:, :] outputs,
    ThreadLocalBuffer* buffers,
    int num_tasks,
    int num_threads
) noexcept nogil:
    """批量前向传播（支持网络索引）

    每个任务可以使用不同的网络，通过 network_indices 指定。

    Args:
        networks: 批量网络数据
        network_indices: 每个任务对应的网络索引数组
        inputs: 输入数组 [num_tasks, num_inputs]
        outputs: 输出数组 [num_tasks, num_outputs]
        buffers: 线程本地缓冲区数组
        num_tasks: 总任务数
        num_threads: 线程数
    """
    cdef int i
    cdef int tid
    cdef int net_idx

    for i in prange(num_tasks, nogil=True, num_threads=num_threads, schedule='dynamic'):
        tid = openmp.omp_get_thread_num()
        net_idx = network_indices[i]
        _forward_single(
            net_idx, networks,
            &inputs[i, 0],
            &outputs[i, 0],
            buffers[tid].values
        )


# ============================================================================
# 批量解析函数 (nogil)
# ============================================================================

cdef void _parse_retail_single(
    int agent_idx,
    double* nn_output,
    DecisionResult* result,
    BatchAgentState* agents,
    double mid_price,
    double tick_size,
) noexcept nogil:
    """解析散户/高级散户/庄家的神经网络输出，并计算实际订单数量

    输出结构（8个值）：
    - [0-5]: 动作类型得分（6种动作）
    - [6]: 价格偏移（-1 到 1）
    - [7]: 数量比例（-1 到 1）

    动作类型（统一6种）：
    - 0: HOLD
    - 1: PLACE_BID
    - 2: PLACE_ASK
    - 3: CANCEL
    - 4: MARKET_BUY
    - 5: MARKET_SELL
    """
    # 订单数量上限
    cdef double MAX_ORDER_QUANTITY = 100000000.0

    # 所有变量声明必须在函数开头
    cdef int action_idx
    cdef double price_offset_norm
    cdef double quantity_ratio_norm
    cdef double quantity_ratio
    cdef double price_offset_ticks
    cdef double order_price

    # 内联 quantity 计算所需变量
    cdef double equity, leverage, max_pos_value
    cdef double current_pos, current_pos_value, available_pos_value
    cdef double calc_price, raw_quantity
    cdef int quantity

    # 统一使用6种动作
    action_idx = argmax(nn_output, 0, 6)

    # 解析参数（索引已调整）
    price_offset_norm = clip(nn_output[6], -1.0, 1.0)
    quantity_ratio_norm = clip(nn_output[7], -1.0, 1.0)
    quantity_ratio = (quantity_ratio_norm + 1.0) * 0.5

    # 设置结果默认值
    result.action_type = action_idx
    result.side = 0
    result.price = 0.0
    result.quantity = 0.0

    # 提取 agent 账户数据
    equity = agents.available_margin[agent_idx]  # 实际是 equity
    leverage = agents.leverage[agent_idx]
    current_pos = agents.position_quantity[agent_idx]

    if action_idx == ACTION_PLACE_BID:
        # 买单
        result.side = 1  # BUY
        price_offset_ticks = price_offset_norm * 100.0
        order_price = round_price(mid_price + price_offset_ticks * tick_size, tick_size)
        result.price = order_price

        # 内联 quantity 计算
        calc_price = order_price if order_price > 0 else mid_price
        if equity > 0 and calc_price > 0 and leverage > 0:
            max_pos_value = equity * leverage
            current_pos_value = fabs(current_pos) * calc_price
            # 买入
            if current_pos >= 0:
                # 多头或空仓，买入是同向加仓
                available_pos_value = max_pos_value - current_pos_value
                if available_pos_value < 0:
                    available_pos_value = 0
            else:
                # 空头，买入是反向平仓+可能开多仓
                available_pos_value = current_pos_value + max_pos_value
            raw_quantity = (available_pos_value * quantity_ratio) / calc_price
            if raw_quantity >= 1.0:
                quantity = <int>raw_quantity
                if quantity > <int>MAX_ORDER_QUANTITY:
                    quantity = <int>MAX_ORDER_QUANTITY
                result.quantity = <double>quantity

    elif action_idx == ACTION_PLACE_ASK:
        # 卖单
        result.side = 2  # SELL
        price_offset_ticks = price_offset_norm * 100.0
        order_price = round_price(mid_price + price_offset_ticks * tick_size, tick_size)
        result.price = order_price

        # 内联 quantity 计算
        calc_price = order_price if order_price > 0 else mid_price
        if equity > 0 and calc_price > 0 and leverage > 0:
            max_pos_value = equity * leverage
            current_pos_value = fabs(current_pos) * calc_price
            # 卖出
            if current_pos <= 0:
                # 空头或空仓，卖出是同向加仓
                available_pos_value = max_pos_value - current_pos_value
                if available_pos_value < 0:
                    available_pos_value = 0
            else:
                # 多头，卖出是反向平仓+可能开空仓
                available_pos_value = current_pos_value + max_pos_value
            raw_quantity = (available_pos_value * quantity_ratio) / calc_price
            if raw_quantity >= 1.0:
                quantity = <int>raw_quantity
                if quantity > <int>MAX_ORDER_QUANTITY:
                    quantity = <int>MAX_ORDER_QUANTITY
                result.quantity = <double>quantity

    elif action_idx == ACTION_MARKET_BUY:
        result.side = 1  # BUY

        # 内联 quantity 计算（使用 mid_price）
        calc_price = mid_price if mid_price > 0 else 100.0
        if equity > 0 and leverage > 0:
            max_pos_value = equity * leverage
            current_pos_value = fabs(current_pos) * calc_price
            # 买入
            if current_pos >= 0:
                available_pos_value = max_pos_value - current_pos_value
                if available_pos_value < 0:
                    available_pos_value = 0
            else:
                available_pos_value = current_pos_value + max_pos_value
            raw_quantity = (available_pos_value * quantity_ratio) / calc_price
            if raw_quantity >= 1.0:
                quantity = <int>raw_quantity
                if quantity > <int>MAX_ORDER_QUANTITY:
                    quantity = <int>MAX_ORDER_QUANTITY
                result.quantity = <double>quantity

    elif action_idx == ACTION_MARKET_SELL:
        result.side = 2  # SELL

        # 特殊逻辑：如果持有多仓，卖出持仓的比例
        if current_pos > 0:
            quantity = <int>(current_pos * quantity_ratio)
            if quantity < 1:
                quantity = 1
            if quantity > <int>current_pos:
                quantity = <int>current_pos
            result.quantity = <double>quantity
        else:
            # 空仓或空头，开空仓（使用 mid_price）
            calc_price = mid_price if mid_price > 0 else 100.0
            if equity > 0 and leverage > 0:
                max_pos_value = equity * leverage
                current_pos_value = fabs(current_pos) * calc_price
                # 卖出
                if current_pos <= 0:
                    available_pos_value = max_pos_value - current_pos_value
                    if available_pos_value < 0:
                        available_pos_value = 0
                else:
                    available_pos_value = current_pos_value + max_pos_value
                raw_quantity = (available_pos_value * quantity_ratio) / calc_price
                if raw_quantity >= 1.0:
                    quantity = <int>raw_quantity
                    if quantity > <int>MAX_ORDER_QUANTITY:
                        quantity = <int>MAX_ORDER_QUANTITY
                    result.quantity = <double>quantity


cdef void batch_parse_retail_nogil(
    double[:, :] nn_outputs,
    DecisionResult* results,
    BatchAgentState* agents,
    double mid_price,
    double tick_size,
    int num_agents,
    int num_threads
) noexcept nogil:
    """批量解析散户/高级散户/庄家的神经网络输出"""
    cdef int i

    for i in prange(num_agents, nogil=True, num_threads=num_threads, schedule='static'):
        _parse_retail_single(i, &nn_outputs[i, 0], &results[i], agents, mid_price, tick_size)


cdef void _parse_market_maker_single(
    int agent_idx,
    double* nn_output,
    DecisionResult* result,
    double mid_price,
    double tick_size,
) noexcept nogil:
    """解析做市商的神经网络输出

    输出结构（41个值）：
    - [0-9]: 买单价格偏移（10个买单）
    - [10-19]: 买单数量权重（10个买单）
    - [20-29]: 卖单价格偏移（10个卖单）
    - [30-39]: 卖单数量权重（10个卖单）
    - [40]: 总下单比例基准

    做市商始终执行 QUOTE 动作，价格和数量信息存储在 result 中供后续处理。
    """
    # 做市商始终返回 QUOTE 动作
    result.action_type = ACTION_QUOTE
    result.side = 0  # 双边
    result.price = mid_price  # 参考价格
    result.quantity = (clip(nn_output[40], -1.0, 1.0) + 1.0) * 0.5  # 总下单比例


cdef void batch_parse_market_maker_nogil(
    double[:, :] nn_outputs,
    DecisionResult* results,
    double mid_price,
    double tick_size,
    int num_agents,
    int num_threads
) noexcept nogil:
    """批量解析做市商的神经网络输出"""
    cdef int i

    for i in prange(num_agents, nogil=True, num_threads=num_threads, schedule='static'):
        _parse_market_maker_single(i, &nn_outputs[i, 0], &results[i], mid_price, tick_size)


# ============================================================================
# 多市场状态版本的批量解析函数 (nogil)
# ============================================================================

cdef void batch_parse_retail_multi_market_nogil(
    double[:, :] nn_outputs,
    DecisionResult* results,
    BatchAgentState* agents,
    MarketStateData** markets,
    int* market_indices,
    int num_agents,
    int num_threads
) noexcept nogil:
    """批量解析散户/高级散户/庄家的神经网络输出（多市场版本）"""
    cdef int i
    cdef int market_idx
    cdef double mid_price, tick_size

    for i in prange(num_agents, nogil=True, num_threads=num_threads, schedule='static'):
        market_idx = market_indices[i]
        mid_price = markets[market_idx].mid_price
        tick_size = markets[market_idx].tick_size
        _parse_retail_single(i, &nn_outputs[i, 0], &results[i], agents, mid_price, tick_size)


cdef void batch_parse_market_maker_multi_market_nogil(
    double[:, :] nn_outputs,
    DecisionResult* results,
    MarketStateData** markets,
    int* market_indices,
    int num_agents,
    int num_threads
) noexcept nogil:
    """批量解析做市商的神经网络输出（多市场版本）"""
    cdef int i
    cdef int market_idx
    cdef double mid_price, tick_size

    for i in prange(num_agents, nogil=True, num_threads=num_threads, schedule='static'):
        market_idx = market_indices[i]
        mid_price = markets[market_idx].mid_price
        tick_size = markets[market_idx].tick_size
        _parse_market_maker_single(i, &nn_outputs[i, 0], &results[i], mid_price, tick_size)


# ============================================================================
# 做市商完整解析函数（直接返回订单数据）
# ============================================================================

cdef inline double _calculate_skew_factor(
    double equity,
    double leverage,
    double position_qty,
    double mid_price
) noexcept nogil:
    """计算做市商仓位倾斜因子

    与 Python 端 calculate_skew_factor_from_state 逻辑完全一致。

    Returns:
        倾斜因子，范围 [-1, 1]
    """
    if equity <= 0:
        return 0.0

    if position_qty == 0:
        return 0.0

    cdef double position_value = fabs(position_qty) * mid_price
    cdef double max_position_value = equity * leverage
    cdef double pos_ratio

    if max_position_value > 0:
        pos_ratio = position_value / max_position_value
        if pos_ratio > 1.0:
            pos_ratio = 1.0
    else:
        pos_ratio = 0.0

    if position_qty > 0:
        return -pos_ratio
    else:
        return pos_ratio


cdef inline void _apply_position_skew_nogil(
    double* bid_ratios,
    double* ask_ratios,
    double skew_factor,
    double min_side_weight
) noexcept nogil:
    """应用仓位倾斜到买卖权重（与 Python 端逻辑完全一致）

    直接修改传入的数组，无需返回值。
    """
    cdef double bid_multiplier = 1.0 + skew_factor
    cdef double ask_multiplier = 1.0 - skew_factor
    cdef int i
    cdef double total_bid = 0.0
    cdef double total_ask = 0.0
    cdef double total
    cdef double bid_side_ratio, ask_side_ratio
    cdef double target_bid_total, target_ask_total
    cdef double bid_adjusted[10]
    cdef double ask_adjusted[10]

    # 计算调整后的权重
    for i in range(10):
        bid_adjusted[i] = bid_ratios[i] * bid_multiplier
        ask_adjusted[i] = ask_ratios[i] * ask_multiplier
        total_bid = total_bid + bid_adjusted[i]
        total_ask = total_ask + ask_adjusted[i]

    total = total_bid + total_ask

    if total <= 0:
        # 平均分配
        for i in range(10):
            bid_ratios[i] = 0.1
            ask_ratios[i] = 0.1
        return

    bid_side_ratio = total_bid / total
    ask_side_ratio = total_ask / total

    # 确保每边至少有 min_side_weight
    if bid_side_ratio < min_side_weight:
        target_bid_total = min_side_weight
        target_ask_total = 1.0 - min_side_weight
    elif ask_side_ratio < min_side_weight:
        target_ask_total = min_side_weight
        target_bid_total = 1.0 - min_side_weight
    else:
        target_bid_total = bid_side_ratio
        target_ask_total = ask_side_ratio

    # 归一化并应用目标比例
    if total_bid > 0:
        for i in range(10):
            bid_ratios[i] = bid_adjusted[i] / total_bid * target_bid_total
    else:
        for i in range(10):
            bid_ratios[i] = target_bid_total / 10

    if total_ask > 0:
        for i in range(10):
            ask_ratios[i] = ask_adjusted[i] / total_ask * target_ask_total
    else:
        for i in range(10):
            ask_ratios[i] = target_ask_total / 10


cdef void _parse_market_maker_full_single(
    int agent_idx,
    double* nn_output,
    MarketMakerOrdersResult* result,
    BatchAgentState* agents,
    double mid_price,
    double tick_size,
) noexcept nogil:
    """解析做市商的神经网络输出并直接生成订单数据

    神经网络输出结构（共 41 个值）：
    - 输出[0-9]: 买单1-10的价格偏移（-1到1，映射到1-100 ticks）
    - 输出[10-19]: 买单1-10的数量权重（-1到1，映射到0-1）
    - 输出[20-29]: 卖单1-10的价格偏移（-1到1，映射到1-100 ticks）
    - 输出[30-39]: 卖单1-10的数量权重（-1到1，映射到0-1）
    - 输出[40]: 总下单比例基准（-1到1，映射到0.01-1）
    """
    cdef double MAX_ORDER_QUANTITY = 100000000.0
    cdef double MIN_SIDE_WEIGHT = 0.1
    cdef int MM_MIN_ORDER_QUANTITY = 1  # 最小订单数量
    cdef double MM_MIN_RATIO_THRESHOLD = 0.001  # 最小权重阈值 (0.1%)

    cdef int i, num_bid, num_ask
    cdef double bid_ratios[10]
    cdef double ask_ratios[10]
    cdef double total_raw_ratio, total_ratio_raw, total_ratio_clipped, total_ratio
    cdef double skew_factor
    cdef double equity, leverage, max_pos_value, current_pos, current_pos_value
    cdef double avail_buy, avail_sell
    cdef double offset, ticks, price, ratio, qty_raw
    cdef int qty

    # 初始化结果
    result.num_bid_orders = 0
    result.num_ask_orders = 0

    # 提取 agent 状态
    equity = agents.available_margin[agent_idx]  # 实际是 equity
    leverage = agents.leverage[agent_idx]
    current_pos = agents.position_quantity[agent_idx]

    if equity <= 0:
        return

    # 计算权重比例（从神经网络输出）
    total_raw_ratio = 0.0
    for i in range(10):
        # bid_qty_weights[i] = nn_output[10+i]
        bid_ratios[i] = (clip(nn_output[10 + i], -1.0, 1.0) + 1.0) * 0.5
        if bid_ratios[i] < 0:
            bid_ratios[i] = 0
        total_raw_ratio = total_raw_ratio + bid_ratios[i]

        # ask_qty_weights[i] = nn_output[30+i]
        ask_ratios[i] = (clip(nn_output[30 + i], -1.0, 1.0) + 1.0) * 0.5
        if ask_ratios[i] < 0:
            ask_ratios[i] = 0
        total_raw_ratio = total_raw_ratio + ask_ratios[i]

    # 归一化
    if total_raw_ratio > 0:
        for i in range(10):
            bid_ratios[i] = bid_ratios[i] / total_raw_ratio
            ask_ratios[i] = ask_ratios[i] / total_raw_ratio
    else:
        # 全部为零，平均分配
        for i in range(10):
            bid_ratios[i] = 0.05
            ask_ratios[i] = 0.05

    # 应用仓位倾斜
    skew_factor = _calculate_skew_factor(equity, leverage, current_pos, mid_price)
    _apply_position_skew_nogil(bid_ratios, ask_ratios, skew_factor, MIN_SIDE_WEIGHT)

    # 总下单比例
    total_ratio_raw = nn_output[40]
    total_ratio_clipped = clip(total_ratio_raw, -1.0, 1.0)
    total_ratio = 0.01 + (total_ratio_clipped + 1) * 0.5 * 0.99

    for i in range(10):
        bid_ratios[i] = bid_ratios[i] * total_ratio
        ask_ratios[i] = ask_ratios[i] * total_ratio

    # 计算可用仓位价值
    max_pos_value = equity * leverage
    current_pos_value = fabs(current_pos) * mid_price

    if current_pos >= 0:
        avail_buy = max_pos_value - current_pos_value
        if avail_buy < 0:
            avail_buy = 0
    else:
        avail_buy = current_pos_value + max_pos_value

    if current_pos <= 0:
        avail_sell = max_pos_value - current_pos_value
        if avail_sell < 0:
            avail_sell = 0
    else:
        avail_sell = current_pos_value + max_pos_value

    # 生成订单
    num_bid = 0
    num_ask = 0

    for i in range(10):
        # 买单处理
        ratio = bid_ratios[i]
        if ratio > 0:
            offset = clip(nn_output[i], -1.0, 1.0)
            ticks = 1 + (offset + 1) * 49.5
            price = mid_price - ticks * tick_size
            price = round_price(price, tick_size)
            if price > 0:
                qty_raw = avail_buy * ratio / price
                qty = <int>qty_raw
                # 最小数量保证：当权重大于阈值但数量为0时，强制下1单位
                if qty == 0 and bid_ratios[i] > MM_MIN_RATIO_THRESHOLD:
                    qty = MM_MIN_ORDER_QUANTITY
                if qty >= 1:
                    if qty > <int>MAX_ORDER_QUANTITY:
                        qty = <int>MAX_ORDER_QUANTITY
                    result.bid_prices[num_bid] = price
                    result.bid_quantities[num_bid] = qty
                    num_bid = num_bid + 1

        # 卖单处理
        ratio = ask_ratios[i]
        if ratio > 0:
            offset = clip(nn_output[20 + i], -1.0, 1.0)
            ticks = 1 + (offset + 1) * 49.5
            price = mid_price + ticks * tick_size
            price = round_price(price, tick_size)
            if price > 0:
                qty_raw = avail_sell * ratio / price
                qty = <int>qty_raw
                # 最小数量保证：当权重大于阈值但数量为0时，强制下1单位
                if qty == 0 and ask_ratios[i] > MM_MIN_RATIO_THRESHOLD:
                    qty = MM_MIN_ORDER_QUANTITY
                if qty >= 1:
                    if qty > <int>MAX_ORDER_QUANTITY:
                        qty = <int>MAX_ORDER_QUANTITY
                    result.ask_prices[num_ask] = price
                    result.ask_quantities[num_ask] = qty
                    num_ask = num_ask + 1

    result.num_bid_orders = num_bid
    result.num_ask_orders = num_ask


cdef void batch_parse_market_maker_full_multi_market_nogil(
    double[:, :] nn_outputs,
    MarketMakerOrdersResult* mm_orders,
    BatchAgentState* agents,
    MarketStateData** markets,
    int* market_indices,
    int num_agents,
    int num_threads
) noexcept nogil:
    """批量解析做市商的神经网络输出（完整版本，直接返回订单数据）"""
    cdef int i
    cdef int market_idx
    cdef double mid_price, tick_size

    for i in prange(num_agents, nogil=True, num_threads=num_threads, schedule='static'):
        market_idx = market_indices[i]
        mid_price = markets[market_idx].mid_price
        tick_size = markets[market_idx].tick_size
        _parse_market_maker_full_single(i, &nn_outputs[i, 0], &mm_orders[i], agents, mid_price, tick_size)


# ============================================================================
# Python 入口函数
# ============================================================================

def batch_decide_retail(
    list networks,
    list agents,
    market_state,
    int num_threads=0,
) -> list:
    """批量决策入口 - 散户版本 (127维输入, 9维输出)

    Args:
        networks: FastFeedForwardNetwork 列表
        agents: Agent 列表
        market_state: NormalizedMarketState
        num_threads: 线程数（0 表示自动检测）

    Returns:
        决策结果列表：[(action_type, side, price, quantity), ...]
    """
    cdef int num_agents = len(agents)
    if num_agents == 0:
        return []

    # 自动检测线程数
    if num_threads <= 0:
        num_threads = openmp.omp_get_max_threads()

    # 获取网络参数
    cdef int max_nodes = 0
    cdef int max_connections = 0
    for net in networks:
        if net.num_nodes > max_nodes:
            max_nodes = net.num_nodes
        num_conns = len(net.conn_sources)
        if num_conns > max_connections:
            max_connections = num_conns

    # 分配数据结构
    cdef BatchNetworkData* batch_networks = alloc_batch_network_data(
        num_agents, max_nodes, max_connections, INPUT_DIM_RETAIL, OUTPUT_DIM_RETAIL
    )
    cdef BatchAgentState* batch_agents = alloc_batch_agent_state(num_agents)
    cdef MarketStateData* market_data = alloc_market_state_data()
    cdef ThreadLocalBuffer* buffers = alloc_thread_buffers(
        num_threads, max_nodes + INPUT_DIM_RETAIL, INPUT_DIM_RETAIL, OUTPUT_DIM_RETAIL
    )

    if (batch_networks == NULL or batch_agents == NULL or
        market_data == NULL or buffers == NULL):
        # 清理并返回空结果
        if batch_networks != NULL:
            free_batch_network_data(batch_networks)
        if batch_agents != NULL:
            free_batch_agent_state(batch_agents)
        if market_data != NULL:
            free_market_state_data(market_data)
        if buffers != NULL:
            free_thread_buffers(buffers, num_threads)
        return []

    # 提取数据
    _extract_networks_to_batch(networks, batch_networks)
    _extract_agents_to_batch(agents, batch_agents, market_state.mid_price)
    _extract_market_state(market_state, market_data)

    # 分配输入输出数组
    cdef np.ndarray[DTYPE_t, ndim=2] inputs = np.zeros(
        (num_agents, INPUT_DIM_RETAIL), dtype=np.float64
    )
    cdef np.ndarray[DTYPE_t, ndim=2] outputs = np.zeros(
        (num_agents, OUTPUT_DIM_RETAIL), dtype=np.float64
    )

    # 分配决策结果数组
    cdef DecisionResult* results = <DecisionResult*>calloc(num_agents, sizeof(DecisionResult))

    cdef double[:, :] inputs_view = inputs
    cdef double[:, :] outputs_view = outputs
    cdef double mid_price = market_state.mid_price
    cdef double tick_size = market_state.tick_size

    # 执行批量计算
    with nogil:
        # 1. 批量观察
        batch_observe_retail_nogil(batch_agents, market_data, inputs_view, num_threads)

        # 2. 批量前向传播
        batch_forward_nogil(batch_networks, inputs_view, outputs_view, buffers, num_threads)

        # 3. 批量解析（传入 agents 以计算 quantity）
        batch_parse_retail_nogil(outputs_view, results, batch_agents, mid_price, tick_size, num_agents, num_threads)

    # 转换结果为 Python 列表
    cdef list result_list = []
    cdef int i
    for i in range(num_agents):
        result_list.append((
            results[i].action_type,
            results[i].side,
            results[i].price,
            results[i].quantity,
        ))

    # 清理
    free_batch_network_data(batch_networks)
    free_batch_agent_state(batch_agents)
    free_market_state_data(market_data)
    free_thread_buffers(buffers, num_threads)
    free(results)

    return result_list


def batch_decide_full(
    list networks,
    list agents,
    market_state,
    int num_threads=0,
) -> list:
    """批量决策入口 - 完整版本 (907维输入, 9维输出)

    用于高级散户和庄家。

    Args:
        networks: FastFeedForwardNetwork 列表
        agents: Agent 列表
        market_state: NormalizedMarketState
        num_threads: 线程数（0 表示自动检测）

    Returns:
        决策结果列表：[(action_type, side, price, quantity), ...]
    """
    cdef int num_agents = len(agents)
    if num_agents == 0:
        return []

    if num_threads <= 0:
        num_threads = openmp.omp_get_max_threads()

    # 获取网络参数
    cdef int max_nodes = 0
    cdef int max_connections = 0
    for net in networks:
        if net.num_nodes > max_nodes:
            max_nodes = net.num_nodes
        num_conns = len(net.conn_sources)
        if num_conns > max_connections:
            max_connections = num_conns

    # 分配数据结构
    cdef BatchNetworkData* batch_networks = alloc_batch_network_data(
        num_agents, max_nodes, max_connections, INPUT_DIM_FULL, OUTPUT_DIM_RETAIL
    )
    cdef BatchAgentState* batch_agents = alloc_batch_agent_state(num_agents)
    cdef MarketStateData* market_data = alloc_market_state_data()
    cdef ThreadLocalBuffer* buffers = alloc_thread_buffers(
        num_threads, max_nodes + INPUT_DIM_FULL, INPUT_DIM_FULL, OUTPUT_DIM_RETAIL
    )

    if (batch_networks == NULL or batch_agents == NULL or
        market_data == NULL or buffers == NULL):
        if batch_networks != NULL:
            free_batch_network_data(batch_networks)
        if batch_agents != NULL:
            free_batch_agent_state(batch_agents)
        if market_data != NULL:
            free_market_state_data(market_data)
        if buffers != NULL:
            free_thread_buffers(buffers, num_threads)
        return []

    # 提取数据
    _extract_networks_to_batch(networks, batch_networks)
    _extract_agents_to_batch(agents, batch_agents, market_state.mid_price)
    _extract_market_state(market_state, market_data)

    # 分配输入输出数组
    cdef np.ndarray[DTYPE_t, ndim=2] inputs = np.zeros(
        (num_agents, INPUT_DIM_FULL), dtype=np.float64
    )
    cdef np.ndarray[DTYPE_t, ndim=2] outputs = np.zeros(
        (num_agents, OUTPUT_DIM_RETAIL), dtype=np.float64
    )

    cdef DecisionResult* results = <DecisionResult*>calloc(num_agents, sizeof(DecisionResult))

    cdef double[:, :] inputs_view = inputs
    cdef double[:, :] outputs_view = outputs
    cdef double mid_price = market_state.mid_price
    cdef double tick_size = market_state.tick_size

    with nogil:
        batch_observe_full_nogil(batch_agents, market_data, inputs_view, num_threads)
        batch_forward_nogil(batch_networks, inputs_view, outputs_view, buffers, num_threads)
        batch_parse_retail_nogil(outputs_view, results, batch_agents, mid_price, tick_size, num_agents, num_threads)

    cdef list result_list = []
    cdef int i
    for i in range(num_agents):
        result_list.append((
            results[i].action_type,
            results[i].side,
            results[i].price,
            results[i].quantity,
        ))

    free_batch_network_data(batch_networks)
    free_batch_agent_state(batch_agents)
    free_market_state_data(market_data)
    free_thread_buffers(buffers, num_threads)
    free(results)

    return result_list


def batch_decide_market_maker(
    list networks,
    list agents,
    market_state,
    int num_threads=0,
) -> list:
    """批量决策入口 - 做市商版本 (964维输入, 41维输出)

    Args:
        networks: FastFeedForwardNetwork 列表
        agents: Agent 列表
        market_state: NormalizedMarketState
        num_threads: 线程数（0 表示自动检测）

    Returns:
        决策结果列表，每个元素为 (nn_output, mid_price, tick_size)
        做市商的完整解析需要访问 Agent 对象，因此返回原始输出供后续处理
    """
    cdef int num_agents = len(agents)
    if num_agents == 0:
        return []

    if num_threads <= 0:
        num_threads = openmp.omp_get_max_threads()

    # 获取网络参数
    cdef int max_nodes = 0
    cdef int max_connections = 0
    for net in networks:
        if net.num_nodes > max_nodes:
            max_nodes = net.num_nodes
        num_conns = len(net.conn_sources)
        if num_conns > max_connections:
            max_connections = num_conns

    # 分配数据结构
    cdef BatchNetworkData* batch_networks = alloc_batch_network_data(
        num_agents, max_nodes, max_connections, INPUT_DIM_MARKET_MAKER, OUTPUT_DIM_MARKET_MAKER
    )
    cdef BatchAgentState* batch_agents = alloc_batch_agent_state(num_agents)
    cdef MarketStateData* market_data = alloc_market_state_data()
    cdef ThreadLocalBuffer* buffers = alloc_thread_buffers(
        num_threads, max_nodes + INPUT_DIM_MARKET_MAKER, INPUT_DIM_MARKET_MAKER, OUTPUT_DIM_MARKET_MAKER
    )

    if (batch_networks == NULL or batch_agents == NULL or
        market_data == NULL or buffers == NULL):
        if batch_networks != NULL:
            free_batch_network_data(batch_networks)
        if batch_agents != NULL:
            free_batch_agent_state(batch_agents)
        if market_data != NULL:
            free_market_state_data(market_data)
        if buffers != NULL:
            free_thread_buffers(buffers, num_threads)
        return []

    # 提取数据
    _extract_networks_to_batch(networks, batch_networks)
    _extract_agents_to_batch(agents, batch_agents, market_state.mid_price)
    _extract_market_state(market_state, market_data)

    # 分配输入输出数组
    cdef np.ndarray[DTYPE_t, ndim=2] inputs = np.zeros(
        (num_agents, INPUT_DIM_MARKET_MAKER), dtype=np.float64
    )
    cdef np.ndarray[DTYPE_t, ndim=2] outputs = np.zeros(
        (num_agents, OUTPUT_DIM_MARKET_MAKER), dtype=np.float64
    )

    cdef double[:, :] inputs_view = inputs
    cdef double[:, :] outputs_view = outputs

    with nogil:
        batch_observe_market_maker_nogil(batch_agents, market_data, inputs_view, num_threads)
        batch_forward_nogil(batch_networks, inputs_view, outputs_view, buffers, num_threads)

    # 做市商返回原始神经网络输出，由调用方进行完整解析
    # 因为做市商解析需要调用 Agent 方法（_calculate_skew_factor, _calculate_order_quantity）
    cdef list result_list = []
    cdef int i
    cdef double mid_price = market_state.mid_price
    cdef double tick_size = market_state.tick_size

    for i in range(num_agents):
        # 返回神经网络输出数组（复制一份）
        result_list.append((
            np.asarray(outputs[i]).copy(),
            mid_price,
            tick_size,
        ))

    free_batch_network_data(batch_networks)
    free_batch_agent_state(batch_agents)
    free_market_state_data(market_data)
    free_thread_buffers(buffers, num_threads)

    return result_list


# ============================================================================
# 辅助函数
# ============================================================================

def get_max_threads() -> int:
    """获取 OpenMP 最大线程数"""
    return openmp.omp_get_max_threads()


def set_num_threads(int num_threads):
    """设置 OpenMP 线程数"""
    openmp.omp_set_num_threads(num_threads)


# ============================================================================
# 缓存类型常量（模块级）
# ============================================================================

CACHE_TYPE_RETAIL = 0
CACHE_TYPE_FULL = 1
CACHE_TYPE_MARKET_MAKER = 2


# ============================================================================
# 网络数据缓存类
# ============================================================================

cdef class BatchNetworkCache:
    """网络数据缓存类，避免每次调用都重新提取网络数据

    使用方法：
    1. 创建缓存: cache = BatchNetworkCache(num_networks, cache_type, num_threads)
    2. 更新网络数据: cache.update_networks(networks)
    3. 执行决策: results = cache.decide(agents, market_state)

    性能优化原理：
    - 网络结构在进化前不变，只需提取一次
    - Agent 状态和市场状态每 tick 都变，每次都需更新
    - 预分配所有内存，避免 malloc/free 开销
    """

    def __cinit__(self, int num_networks, int cache_type, int num_threads=0,
                  int max_arenas=8, int max_tasks_per_arena=0):
        """初始化缓存

        Args:
            num_networks: 网络数量
            cache_type: 类型 (0=retail 127维, 1=full 907维, 2=market_maker 934维)
            num_threads: OpenMP 线程数，0 表示自动检测
            max_arenas: 最大竞技场数量（用于预分配多竞技场缓冲区）
            max_tasks_per_arena: 每竞技场最大任务数，0 表示使用 num_networks
        """
        # 自动检测线程数
        if num_threads <= 0:
            num_threads = openmp.omp_get_max_threads()

        self.num_networks = num_networks
        self.num_threads = num_threads
        self.cache_type = cache_type
        self.network_ids = []

        # 根据类型设置输入输出维度
        if cache_type == 0:  # retail
            self.input_dim = INPUT_DIM_RETAIL
            self.output_dim = OUTPUT_DIM_RETAIL
        elif cache_type == 1:  # full
            self.input_dim = INPUT_DIM_FULL
            self.output_dim = OUTPUT_DIM_RETAIL
        else:  # market_maker
            self.input_dim = INPUT_DIM_MARKET_MAKER
            self.output_dim = OUTPUT_DIM_MARKET_MAKER

        # 初始估算网络大小（update_networks 时会调整）
        self.max_nodes = 100
        self.max_connections = 1000

        # 初始化所有指针为 NULL（网络数据在 update_networks 时分配）
        self.network_data = NULL
        self.thread_buffers = NULL

        # 分配 Agent 状态和市场数据结构（每次 decide 都需要更新，但结构不变）
        self.agent_state = alloc_batch_agent_state(num_networks)
        self.market_data = alloc_market_state_data()
        self.results = <DecisionResult*>calloc(num_networks, sizeof(DecisionResult))

        # 预分配 NumPy 数组（用于存储输入和输出）
        self.inputs_array = np.zeros((num_networks, self.input_dim), dtype=np.float64)
        self.outputs_array = np.zeros((num_networks, self.output_dim), dtype=np.float64)

        # ========== 多竞技场缓冲区预分配 ==========
        if max_tasks_per_arena <= 0:
            max_tasks_per_arena = num_networks

        cdef int max_total_tasks = max_arenas * max_tasks_per_arena
        self.multi_arena_max_tasks = max_total_tasks
        self.multi_arena_max_arenas = max_arenas

        # 预分配 C 结构体
        self.multi_arena_agents = alloc_batch_agent_state(max_total_tasks)
        self.multi_arena_markets = alloc_multi_market_state_data(max_arenas)
        self.multi_arena_network_indices = <int*>calloc(max_total_tasks, sizeof(int))
        self.multi_arena_market_indices = <int*>calloc(max_total_tasks, sizeof(int))
        self.multi_arena_results = <DecisionResult*>calloc(max_total_tasks, sizeof(DecisionResult))

        # 预分配 NumPy 数组
        self.multi_arena_inputs = np.zeros((max_total_tasks, self.input_dim), dtype=np.float64)
        self.multi_arena_outputs = np.zeros((max_total_tasks, self.output_dim), dtype=np.float64)

        # 做市商订单结果预分配
        self.multi_arena_mm_orders = alloc_mm_orders_results(max_total_tasks)

    def __dealloc__(self):
        """释放所有分配的内存"""
        if self.network_data != NULL:
            free_batch_network_data(self.network_data)
        if self.agent_state != NULL:
            free_batch_agent_state(self.agent_state)
        if self.market_data != NULL:
            free_market_state_data(self.market_data)
        if self.results != NULL:
            free(self.results)
        if self.thread_buffers != NULL:
            free_thread_buffers(self.thread_buffers, self.num_threads)

        # 释放多竞技场缓冲区
        if self.multi_arena_agents != NULL:
            free_batch_agent_state(self.multi_arena_agents)
        if self.multi_arena_markets != NULL:
            free_multi_market_state_data(self.multi_arena_markets, self.multi_arena_max_arenas)
        if self.multi_arena_network_indices != NULL:
            free(self.multi_arena_network_indices)
        if self.multi_arena_market_indices != NULL:
            free(self.multi_arena_market_indices)
        if self.multi_arena_results != NULL:
            free(self.multi_arena_results)
        if self.multi_arena_mm_orders != NULL:
            free_mm_orders_results(self.multi_arena_mm_orders)

    def update_networks(self, list networks):
        """更新缓存的网络数据

        在网络结构变化时（NEAT 进化后）调用此方法。
        如果网络没有变化（通过 id() 检测），则跳过更新。

        Args:
            networks: FastFeedForwardNetwork 列表
        """
        cdef int num_networks = len(networks)
        if num_networks == 0:
            return

        # 检查是否需要更新（通过对象 id 检测）
        new_ids = [id(net) for net in networks]
        if new_ids == self.network_ids:
            return  # 网络没有变化，跳过更新

        self.network_ids = new_ids
        self.num_networks = num_networks

        # 计算所有网络的最大节点数和连接数
        cdef int max_nodes = 0
        cdef int max_connections = 0
        cdef int num_conns

        for net in networks:
            if net.num_nodes > max_nodes:
                max_nodes = net.num_nodes
            num_conns = len(net.conn_sources)
            if num_conns > max_connections:
                max_connections = num_conns

        self.max_nodes = max_nodes
        self.max_connections = max_connections

        # 释放旧的网络数据和线程缓冲区
        if self.network_data != NULL:
            free_batch_network_data(self.network_data)
        if self.thread_buffers != NULL:
            free_thread_buffers(self.thread_buffers, self.num_threads)

        # 分配新的网络数据结构
        self.network_data = alloc_batch_network_data(
            num_networks, max_nodes, max_connections,
            self.input_dim, self.output_dim
        )

        # 分配线程本地缓冲区
        self.thread_buffers = alloc_thread_buffers(
            self.num_threads, max_nodes + self.input_dim,
            self.input_dim, self.output_dim
        )

        # 提取网络数据到缓存（这是主要的耗时操作，只在进化后执行一次）
        _extract_networks_to_batch(networks, self.network_data)

        # 如果 Agent 数量变化，重新分配相关结构
        if self.agent_state == NULL or self.agent_state.num_agents != num_networks:
            if self.agent_state != NULL:
                free_batch_agent_state(self.agent_state)
            self.agent_state = alloc_batch_agent_state(num_networks)

        # 重新分配结果数组
        if self.results != NULL:
            free(self.results)
        self.results = <DecisionResult*>calloc(num_networks, sizeof(DecisionResult))

        # 重新分配 NumPy 数组（如果大小变化）
        if self.inputs_array.shape[0] != num_networks:
            self.inputs_array = np.zeros((num_networks, self.input_dim), dtype=np.float64)
            self.outputs_array = np.zeros((num_networks, self.output_dim), dtype=np.float64)

    def update_networks_from_numpy(
        self,
        headers_arr,
        all_input_keys,
        all_output_keys,
        all_node_ids,
        all_biases,
        all_responses,
        all_act_types,
        all_conn_indptr,
        all_conn_sources,
        all_conn_weights,
        all_output_indices,
    ):
        """从 packed numpy 数组直接更新网络缓存
        
        跳过 Python 对象创建（FastFeedForwardNetwork），直接从 packed numpy 
        数组填充 BatchNetworkData C 结构。
        
        Args:
            headers_arr: 结构化数组，包含每个网络的元信息
            all_input_keys .. all_output_indices: 与 _pack_network_params_numpy 返回格式一致
        """
        cdef int num_networks = len(headers_arr)
        if num_networks == 0:
            return
            
        # 设置 network_ids 为有效的索引列表，确保 is_valid() 返回 True
        self.network_ids = list(range(num_networks))
        self.num_networks = num_networks
        
        # 从 headers 提取元信息
        cdef np.ndarray[int, ndim=1] h_num_inputs = np.ascontiguousarray(
            headers_arr['num_inputs'], dtype=np.intc)
        cdef np.ndarray[int, ndim=1] h_num_outputs = np.ascontiguousarray(
            headers_arr['num_outputs'], dtype=np.intc)
        cdef np.ndarray[int, ndim=1] h_num_nodes = np.ascontiguousarray(
            headers_arr['num_nodes'], dtype=np.intc)
        cdef np.ndarray[int, ndim=1] h_n_connections = np.ascontiguousarray(
            headers_arr['n_connections'], dtype=np.intc)
        cdef np.ndarray[int, ndim=1] h_n_output_keys = np.ascontiguousarray(
            headers_arr['n_output_keys'], dtype=np.intc)
        
        # 计算 max_nodes, max_connections
        cdef int max_nodes = 0
        cdef int max_connections = 0
        cdef int i
        for i in range(num_networks):
            if h_num_nodes[i] > max_nodes:
                max_nodes = h_num_nodes[i]
            if h_n_connections[i] > max_connections:
                max_connections = h_n_connections[i]
        
        self.max_nodes = max_nodes
        self.max_connections = max_connections
        
        # 释放旧的网络数据和线程缓冲区
        if self.network_data != NULL:
            free_batch_network_data(self.network_data)
        if self.thread_buffers != NULL:
            free_thread_buffers(self.thread_buffers, self.num_threads)
        
        # 分配新的网络数据结构
        self.network_data = alloc_batch_network_data(
            num_networks, max_nodes, max_connections,
            self.input_dim, self.output_dim
        )
        
        # 分配线程本地缓冲区
        self.thread_buffers = alloc_thread_buffers(
            self.num_threads, max_nodes + self.input_dim,
            self.input_dim, self.output_dim
        )
        
        # 准备 typed memoryview
        cdef float[:] biases_view = np.ascontiguousarray(all_biases, dtype=np.float32)
        cdef float[:] responses_view = np.ascontiguousarray(all_responses, dtype=np.float32)
        cdef int[:] act_types_view = np.ascontiguousarray(all_act_types, dtype=np.intc)
        cdef int[:] conn_indptr_view = np.ascontiguousarray(all_conn_indptr, dtype=np.intc)
        cdef int[:] conn_sources_view = np.ascontiguousarray(all_conn_sources, dtype=np.intc)
        cdef float[:] conn_weights_view = np.ascontiguousarray(all_conn_weights, dtype=np.float32)
        cdef int[:] output_indices_view = np.ascontiguousarray(all_output_indices, dtype=np.intc)
        
        # 直接填充 C 结构
        _fill_batch_data_from_numpy(
            self.network_data,
            h_num_inputs, h_num_outputs, h_num_nodes, h_n_connections,
            num_networks,
            biases_view, responses_view, act_types_view,
            conn_indptr_view, conn_sources_view, conn_weights_view,
            output_indices_view, h_n_output_keys,
        )
        
        # 如果 Agent 数量变化，重新分配相关结构
        if self.agent_state == NULL or self.agent_state.num_agents != num_networks:
            if self.agent_state != NULL:
                free_batch_agent_state(self.agent_state)
            self.agent_state = alloc_batch_agent_state(num_networks)
        
        # 重新分配结果数组
        if self.results != NULL:
            free(self.results)
        self.results = <DecisionResult*>calloc(num_networks, sizeof(DecisionResult))
        
        # 重新分配 NumPy 数组（如果大小变化）
        if self.inputs_array.shape[0] != num_networks:
            self.inputs_array = np.zeros((num_networks, self.input_dim), dtype=np.float64)
            self.outputs_array = np.zeros((num_networks, self.output_dim), dtype=np.float64)


    def decide(self, list agents, market_state) -> list:
        """执行批量决策（使用缓存的网络数据）

        Args:
            agents: Agent 列表（顺序必须与 update_networks 时的 networks 一致）
            market_state: NormalizedMarketState

        Returns:
            - retail/full: [(action_type, side, price, quantity), ...]
            - market_maker: [(nn_output_array, mid_price, tick_size), ...]
        """
        cdef int num_agents = len(agents)
        if num_agents == 0 or self.network_data == NULL:
            return []

        # 提取 Agent 状态（每次都需要更新，因为持仓、余额等会变化）
        _extract_agents_to_batch(agents, self.agent_state, market_state.mid_price)

        # 提取市场数据（每次都需要更新，因为订单簿、成交等会变化）
        _extract_market_state(market_state, self.market_data)

        cdef double[:, :] inputs_view = self.inputs_array
        cdef double[:, :] outputs_view = self.outputs_array
        cdef double mid_price = market_state.mid_price
        cdef double tick_size = market_state.tick_size

        # 执行批量计算（nogil 并行）
        with nogil:
            # 1. 批量观察（构建神经网络输入）
            if self.cache_type == 0:  # retail
                batch_observe_retail_nogil(
                    self.agent_state, self.market_data,
                    inputs_view, self.num_threads
                )
            elif self.cache_type == 1:  # full
                batch_observe_full_nogil(
                    self.agent_state, self.market_data,
                    inputs_view, self.num_threads
                )
            else:  # market_maker
                batch_observe_market_maker_nogil(
                    self.agent_state, self.market_data,
                    inputs_view, self.num_threads
                )

            # 2. 批量前向传播
            batch_forward_nogil(
                self.network_data, inputs_view, outputs_view,
                self.thread_buffers, self.num_threads
            )

            # 3. 批量解析（非 market_maker）
            if self.cache_type != 2:  # retail or full
                batch_parse_retail_nogil(
                    outputs_view, self.results, self.agent_state, mid_price,
                    tick_size, num_agents, self.num_threads
                )
            else:  # market_maker
                batch_parse_market_maker_nogil(
                    outputs_view, self.results, mid_price,
                    tick_size, num_agents, self.num_threads
                )

        # 转换结果为 Python 列表
        cdef list result_list = []
        cdef int i

        if self.cache_type == 2:  # market_maker 返回原始输出
            for i in range(num_agents):
                result_list.append((
                    np.asarray(self.outputs_array[i]).copy(),
                    mid_price,
                    tick_size,
                ))
        else:  # retail/full 返回解析后的决策
            for i in range(num_agents):
                result_list.append((
                    self.results[i].action_type,
                    self.results[i].side,
                    self.results[i].price,
                    self.results[i].quantity,
                ))

        return result_list

    def decide_multi_arena(
        self,
        list agents_per_arena,
        list market_states,
        list network_indices_per_arena,
    ) -> dict:
        """跨竞技场批量推理

        将多个竞技场的推理任务合并成一个批量，使用 OpenMP 并行执行。

        Args:
            agents_per_arena: 每个竞技场的 Agent 列表，list[list[Agent]]
            market_states: 每个竞技场的市场状态，list[MarketState]
            network_indices_per_arena: 每个竞技场每个 agent 对应的网络索引，list[list[int]]

        Returns:
            dict[arena_idx, list[(action_type, side, price, quantity)]]
            对于做市商类型，返回 dict[arena_idx, list[(nn_output, mid_price, tick_size)]]
        """
        cdef int num_arenas = len(agents_per_arena)
        if num_arenas == 0 or self.network_data == NULL:
            return {}

        # 计算总任务数和各竞技场偏移量
        cdef int total_tasks = 0
        cdef int arena_idx, i, j
        cdef list arena_offsets = [0]

        for arena_idx in range(num_arenas):
            total_tasks += len(agents_per_arena[arena_idx])
            arena_offsets.append(total_tasks)

        if total_tasks == 0:
            return {}

        # 分配扩展的 Agent 状态数组
        cdef BatchAgentState* all_agents = alloc_batch_agent_state(total_tasks)

        # 分配多个市场状态
        cdef MarketStateData** multi_markets = alloc_multi_market_state_data(num_arenas)

        # 分配网络索引数组
        cdef int* network_indices = <int*>calloc(total_tasks, sizeof(int))

        # 分配市场索引数组
        cdef int* market_indices = <int*>calloc(total_tasks, sizeof(int))

        # 检查分配
        if all_agents == NULL or multi_markets == NULL or network_indices == NULL or market_indices == NULL:
            if all_agents != NULL:
                free_batch_agent_state(all_agents)
            if multi_markets != NULL:
                free_multi_market_state_data(multi_markets, num_arenas)
            if network_indices != NULL:
                free(network_indices)
            if market_indices != NULL:
                free(market_indices)
            return {}

        # 提取所有市场状态
        for arena_idx in range(num_arenas):
            _extract_market_state(market_states[arena_idx], multi_markets[arena_idx])

        # 提取所有 agent 状态并设置索引
        cdef int task_idx = 0
        cdef int pos_qty
        cdef double mid_price
        cdef object agents, indices, agent, account, position

        for arena_idx in range(num_arenas):
            agents = agents_per_arena[arena_idx]
            indices = network_indices_per_arena[arena_idx]
            mid_price = market_states[arena_idx].mid_price

            for i in range(len(agents)):
                agent = agents[i]
                account = agent.account
                position = account.position
                pos_qty = position.quantity

                # 填充 agent 状态
                all_agents.balance[task_idx] = account.balance
                all_agents.position_quantity[task_idx] = <double>pos_qty
                all_agents.position_avg_price[task_idx] = position.avg_price

                if pos_qty != 0:
                    if pos_qty > 0:
                        all_agents.unrealized_pnl[task_idx] = (mid_price - position.avg_price) * pos_qty
                    else:
                        all_agents.unrealized_pnl[task_idx] = (position.avg_price - mid_price) * (-pos_qty)
                else:
                    all_agents.unrealized_pnl[task_idx] = 0.0

                all_agents.margin_ratio[task_idx] = account.get_margin_ratio(mid_price)
                all_agents.available_margin[task_idx] = account.get_equity(mid_price)
                all_agents.has_pending_order[task_idx] = 0
                all_agents.pending_side[task_idx] = 0
                all_agents.pending_price[task_idx] = 0.0
                all_agents.pending_quantity[task_idx] = 0.0

                # 设置索引
                network_indices[task_idx] = indices[i]
                market_indices[task_idx] = arena_idx

                task_idx += 1

        all_agents.num_agents = total_tasks

        # 分配输入输出数组
        cdef np.ndarray[DTYPE_t, ndim=2] inputs = np.zeros((total_tasks, self.input_dim), dtype=np.float64)
        cdef np.ndarray[DTYPE_t, ndim=2] outputs = np.zeros((total_tasks, self.output_dim), dtype=np.float64)

        # 分配结果数组
        cdef DecisionResult* results = <DecisionResult*>calloc(total_tasks, sizeof(DecisionResult))

        if results == NULL:
            free_batch_agent_state(all_agents)
            free_multi_market_state_data(multi_markets, num_arenas)
            free(network_indices)
            free(market_indices)
            return {}

        cdef double[:, :] inputs_view = inputs
        cdef double[:, :] outputs_view = outputs

        # 执行批量计算
        with nogil:
            # 1. 批量观察（使用多市场状态版本）
            if self.cache_type == 0:  # retail
                batch_observe_retail_multi_market_nogil(
                    all_agents, multi_markets, market_indices,
                    inputs_view, self.num_threads
                )
            elif self.cache_type == 1:  # full
                batch_observe_full_multi_market_nogil(
                    all_agents, multi_markets, market_indices,
                    inputs_view, self.num_threads
                )
            else:  # market_maker
                batch_observe_market_maker_multi_market_nogil(
                    all_agents, multi_markets, market_indices,
                    inputs_view, self.num_threads
                )

            # 2. 批量前向传播（使用网络索引版本）
            batch_forward_with_indices_nogil(
                self.network_data, network_indices,
                inputs_view, outputs_view,
                self.thread_buffers, total_tasks, self.num_threads
            )

            # 3. 批量解析（使用多市场状态版本）
            if self.cache_type != 2:  # retail or full
                batch_parse_retail_multi_market_nogil(
                    outputs_view, results, all_agents, multi_markets, market_indices,
                    total_tasks, self.num_threads
                )
            else:  # market_maker
                batch_parse_market_maker_multi_market_nogil(
                    outputs_view, results, multi_markets, market_indices,
                    total_tasks, self.num_threads
                )

        # 转换结果为 Python 字典
        cdef dict result_dict = {}
        cdef int offset, num_agents_in_arena
        cdef list result_list
        cdef double tick_size

        for arena_idx in range(num_arenas):
            result_list = []
            offset = arena_offsets[arena_idx]
            num_agents_in_arena = arena_offsets[arena_idx + 1] - offset

            if self.cache_type == 2:  # market_maker
                mid_price = market_states[arena_idx].mid_price
                tick_size = market_states[arena_idx].tick_size
                for i in range(num_agents_in_arena):
                    result_list.append((
                        np.asarray(outputs[offset + i]).copy(),
                        mid_price,
                        tick_size,
                    ))
            else:  # retail/full
                for i in range(num_agents_in_arena):
                    result_list.append((
                        results[offset + i].action_type,
                        results[offset + i].side,
                        results[offset + i].price,
                        results[offset + i].quantity,
                    ))

            result_dict[arena_idx] = result_list

        # 清理
        free_batch_agent_state(all_agents)
        free_multi_market_state_data(multi_markets, num_arenas)
        free(network_indices)
        free(market_indices)
        free(results)

        return result_dict

    def decide_multi_arena_direct(
        self,
        list agent_states_per_arena,
        list market_states,
        list network_indices_per_arena,
        bint return_array=False,
    ) -> dict:
        """跨竞技场批量推理（直接使用 AgentAccountState，无需 adapter）

        将多个竞技场的推理任务合并成一个批量，使用 OpenMP 并行执行。
        直接从 AgentAccountState 对象提取数据，避免创建 adapter 对象的开销。

        使用预分配的缓冲区，避免每次调用都分配内存。

        Args:
            agent_states_per_arena: 每个竞技场的 AgentAccountState 列表，list[list[AgentAccountState]]
            market_states: 每个竞技场的市场状态，list[MarketState]
            network_indices_per_arena: 每个竞技场每个 agent 对应的网络索引，list[list[int]]
            return_array: 是否返回 NumPy 数组格式（用于共享内存优化）

        Returns:
            如果 return_array=False:
                dict[arena_idx, list[(action_type, side, price, quantity)]]
                对于做市商类型，返回 dict[arena_idx, list[dict]] 其中 dict 包含:
                    - "bid_orders": list[dict] 包含 {"price": float, "quantity": float}
                    - "ask_orders": list[dict] 包含 {"price": float, "quantity": float}
            如果 return_array=True:
                dict[arena_idx, np.ndarray] 其中数组 shape=(num_agents, 4)
                列顺序: [action_type, side, price, quantity]
                做市商类型返回 dict[arena_idx, np.ndarray] shape=(num_agents, 42)
                列顺序: [num_bid, num_ask, bid_prices[10], bid_qtys[10], ask_prices[10], ask_qtys[10]]
        """
        cdef int num_arenas = len(agent_states_per_arena)
        if num_arenas == 0 or self.network_data == NULL:
            return {}

        # 计算总任务数和各竞技场偏移量
        cdef int total_tasks = 0
        cdef int arena_idx, i, j
        cdef list arena_offsets = [0]

        for arena_idx in range(num_arenas):
            total_tasks += len(agent_states_per_arena[arena_idx])
            arena_offsets.append(total_tasks)

        if total_tasks == 0:
            return {}

        # ========== 使用预分配缓冲区 ==========
        cdef BatchAgentState* all_agents
        cdef MarketStateData** multi_markets
        cdef int* network_indices
        cdef int* market_indices
        cdef DecisionResult* results
        cdef MarketMakerOrdersResult* mm_orders
        cdef np.ndarray inputs
        cdef np.ndarray outputs
        cdef bint use_preallocated = (total_tasks <= self.multi_arena_max_tasks and
                                       num_arenas <= self.multi_arena_max_arenas)

        if use_preallocated:
            # 使用预分配的缓冲区
            all_agents = self.multi_arena_agents
            multi_markets = self.multi_arena_markets
            network_indices = self.multi_arena_network_indices
            market_indices = self.multi_arena_market_indices
            results = self.multi_arena_results
            mm_orders = self.multi_arena_mm_orders
            inputs = self.multi_arena_inputs
            outputs = self.multi_arena_outputs
        else:
            # 超出预分配容量，动态分配
            all_agents = alloc_batch_agent_state(total_tasks)
            multi_markets = alloc_multi_market_state_data(num_arenas)
            network_indices = <int*>calloc(total_tasks, sizeof(int))
            market_indices = <int*>calloc(total_tasks, sizeof(int))
            results = <DecisionResult*>calloc(total_tasks, sizeof(DecisionResult))
            mm_orders = alloc_mm_orders_results(total_tasks)
            inputs = np.zeros((total_tasks, self.input_dim), dtype=np.float64)
            outputs = np.zeros((total_tasks, self.output_dim), dtype=np.float64)

            # 检查分配
            if all_agents == NULL or multi_markets == NULL or network_indices == NULL or market_indices == NULL or results == NULL or mm_orders == NULL:
                if all_agents != NULL:
                    free_batch_agent_state(all_agents)
                if multi_markets != NULL:
                    free_multi_market_state_data(multi_markets, num_arenas)
                if network_indices != NULL:
                    free(network_indices)
                if market_indices != NULL:
                    free(market_indices)
                if results != NULL:
                    free(results)
                if mm_orders != NULL:
                    free_mm_orders_results(mm_orders)
                return {}

        # 提取所有市场状态
        for arena_idx in range(num_arenas):
            _extract_market_state(market_states[arena_idx], multi_markets[arena_idx])

        # 直接从 AgentAccountState 提取数据（无需 adapter）
        cdef int task_idx = 0
        cdef int pos_qty
        cdef double mid_price
        cdef object states, indices, state

        for arena_idx in range(num_arenas):
            states = agent_states_per_arena[arena_idx]
            indices = network_indices_per_arena[arena_idx]
            mid_price = market_states[arena_idx].mid_price

            for i in range(len(states)):
                state = states[i]
                pos_qty = state.position_quantity

                # 直接从 AgentAccountState 提取扁平数据
                all_agents.balance[task_idx] = state.balance
                all_agents.position_quantity[task_idx] = <double>pos_qty
                all_agents.position_avg_price[task_idx] = state.position_avg_price

                if pos_qty != 0:
                    if pos_qty > 0:
                        all_agents.unrealized_pnl[task_idx] = (mid_price - state.position_avg_price) * pos_qty
                    else:
                        all_agents.unrealized_pnl[task_idx] = (state.position_avg_price - mid_price) * (-pos_qty)
                else:
                    all_agents.unrealized_pnl[task_idx] = 0.0

                all_agents.margin_ratio[task_idx] = state.get_margin_ratio(mid_price)
                all_agents.available_margin[task_idx] = state.get_equity(mid_price)
                all_agents.leverage[task_idx] = state.leverage
                all_agents.has_pending_order[task_idx] = 0
                all_agents.pending_side[task_idx] = 0
                all_agents.pending_price[task_idx] = 0.0
                all_agents.pending_quantity[task_idx] = 0.0

                # 设置索引
                network_indices[task_idx] = indices[i]
                market_indices[task_idx] = arena_idx

                task_idx += 1

        all_agents.num_agents = total_tasks

        cdef double[:, :] inputs_view = inputs[:total_tasks]
        cdef double[:, :] outputs_view = outputs[:total_tasks]

        # 执行批量计算
        with nogil:
            # 1. 批量观察（使用多市场状态版本）
            if self.cache_type == 0:  # retail
                batch_observe_retail_multi_market_nogil(
                    all_agents, multi_markets, market_indices,
                    inputs_view, self.num_threads
                )
            elif self.cache_type == 1:  # full
                batch_observe_full_multi_market_nogil(
                    all_agents, multi_markets, market_indices,
                    inputs_view, self.num_threads
                )
            else:  # market_maker
                batch_observe_market_maker_multi_market_nogil(
                    all_agents, multi_markets, market_indices,
                    inputs_view, self.num_threads
                )

            # 2. 批量前向传播（使用网络索引版本）
            batch_forward_with_indices_nogil(
                self.network_data, network_indices,
                inputs_view, outputs_view,
                self.thread_buffers, total_tasks, self.num_threads
            )

            # 3. 批量解析（使用多市场状态版本）
            if self.cache_type != 2:  # retail or full
                batch_parse_retail_multi_market_nogil(
                    outputs_view, results, all_agents, multi_markets, market_indices,
                    total_tasks, self.num_threads
                )
            else:  # market_maker - 使用完整解析，直接生成订单数据
                batch_parse_market_maker_full_multi_market_nogil(
                    outputs_view, mm_orders, all_agents, multi_markets, market_indices,
                    total_tasks, self.num_threads
                )

        # 转换结果为 Python 字典
        cdef dict result_dict = {}
        cdef int offset, num_agents_in_arena
        cdef list result_list
        cdef double tick_size
        cdef np.ndarray result_array
        cdef double[:, :] result_array_view
        cdef MarketMakerOrdersResult* mm_result
        cdef int k
        cdef dict mm_params
        cdef list bid_orders, ask_orders

        for arena_idx in range(num_arenas):
            offset = arena_offsets[arena_idx]
            num_agents_in_arena = arena_offsets[arena_idx + 1] - offset

            if self.cache_type == 2:  # market_maker
                if return_array:
                    # 返回 NumPy 数组格式：shape=(num_agents, 42)
                    # [num_bid, num_ask, bid_prices[10], bid_qtys[10], ask_prices[10], ask_qtys[10]]
                    result_array = np.zeros((num_agents_in_arena, 42), dtype=np.float64)
                    result_array_view = result_array
                    for i in range(num_agents_in_arena):
                        mm_result = &mm_orders[offset + i]
                        result_array_view[i, 0] = <double>mm_result.num_bid_orders
                        result_array_view[i, 1] = <double>mm_result.num_ask_orders
                        for k in range(10):
                            result_array_view[i, 2 + k] = mm_result.bid_prices[k]
                            result_array_view[i, 12 + k] = <double>mm_result.bid_quantities[k]
                            result_array_view[i, 22 + k] = mm_result.ask_prices[k]
                            result_array_view[i, 32 + k] = <double>mm_result.ask_quantities[k]
                    result_dict[arena_idx] = result_array
                else:
                    # 返回 list[dict] 格式
                    result_list = []
                    for i in range(num_agents_in_arena):
                        mm_result = &mm_orders[offset + i]
                        bid_orders = []
                        ask_orders = []
                        for k in range(mm_result.num_bid_orders):
                            bid_orders.append({
                                "price": mm_result.bid_prices[k],
                                "quantity": <double>mm_result.bid_quantities[k]
                            })
                        for k in range(mm_result.num_ask_orders):
                            ask_orders.append({
                                "price": mm_result.ask_prices[k],
                                "quantity": <double>mm_result.ask_quantities[k]
                            })
                        mm_params = {"bid_orders": bid_orders, "ask_orders": ask_orders}
                        result_list.append(mm_params)
                    result_dict[arena_idx] = result_list
            elif return_array:  # retail/full - 返回 NumPy 数组
                result_array = np.zeros((num_agents_in_arena, 4), dtype=np.float64)
                result_array_view = result_array
                for i in range(num_agents_in_arena):
                    result_array_view[i, 0] = <double>results[offset + i].action_type
                    result_array_view[i, 1] = <double>results[offset + i].side
                    result_array_view[i, 2] = results[offset + i].price
                    result_array_view[i, 3] = results[offset + i].quantity
                result_dict[arena_idx] = result_array
            else:  # retail/full - 返回 list 格式
                result_list = []
                for i in range(num_agents_in_arena):
                    result_list.append((
                        results[offset + i].action_type,
                        results[offset + i].side,
                        results[offset + i].price,
                        results[offset + i].quantity,
                    ))
                result_dict[arena_idx] = result_list

        # 只清理动态分配的内存
        if not use_preallocated:
            free_batch_agent_state(all_agents)
            free_multi_market_state_data(multi_markets, num_arenas)
            free(network_indices)
            free(market_indices)
            free(results)
            free_mm_orders_results(mm_orders)

        return result_dict

    def is_valid(self) -> bool:
        """检查缓存是否有效（已初始化网络数据）"""
        return self.network_data != NULL and len(self.network_ids) > 0

    def clear(self):
        """清除缓存的网络数据，强制下次 update_networks 时重新提取"""
        self.network_ids = []

    @property
    def size(self) -> int:
        """返回缓存的网络数量"""
        return self.num_networks

    @property
    def type_name(self) -> str:
        """返回缓存类型名称"""
        if self.cache_type == 0:
            return "retail"
        elif self.cache_type == 1:
            return "full"
        else:
            return "market_maker"
