# cython: language_level=3
# 批量决策 OpenMP 并行模块的声明文件
# 定义 C 结构体和 cdef 函数签名，供其他 Cython 模块 cimport

cimport numpy as np
from numpy cimport float64_t as DTYPE_t

# ============================================================================
# C 结构体定义
# ============================================================================

# 批量网络数据的扁平化存储
cdef struct BatchNetworkData:
    int num_networks
    int max_nodes
    int max_connections
    int max_inputs
    int max_outputs

    # 每个网络的元信息 (num_networks,)
    int* num_inputs_arr
    int* num_outputs_arr
    int* num_nodes_arr
    int* node_offsets
    int* conn_offsets
    int* output_idx_offsets

    # 节点数据 (扁平化存储)
    double* biases
    double* responses
    int* act_types

    # 连接数据 (CSR 格式，扁平化)
    int* conn_indptr
    int* conn_sources
    double* conn_weights

    # 输出索引
    int* output_indices


# Agent 状态批量数据
cdef struct BatchAgentState:
    int num_agents

    # 账户信息 (num_agents,)
    double* balance
    double* position_quantity
    double* position_avg_price
    double* unrealized_pnl
    double* margin_ratio
    double* available_margin

    # 挂单信息 (num_agents,)
    int* has_pending_order
    int* pending_side           # 0=无, 1=买, 2=卖
    double* pending_price
    double* pending_quantity


# 线程本地计算缓冲区
cdef struct ThreadLocalBuffer:
    double* values              # 节点值缓冲区 (max_nodes,)
    double* inputs              # 输入缓冲区 (max_inputs,)
    double* outputs             # 输出缓冲区 (max_outputs,)


# 市场状态数据
cdef struct MarketStateData:
    double mid_price
    double tick_size
    double* bid_data            # (200,)
    double* ask_data            # (200,)
    double* trade_prices        # (100,)
    double* trade_quantities    # (100,)
    double* tick_history_prices     # (100,)
    double* tick_history_volumes    # (100,)
    double* tick_history_amounts    # (100,)


# 单个 Agent 的决策结果
cdef struct DecisionResult:
    int action_type             # ActionType 枚举值
    int side                    # 0=无, 1=买, 2=卖
    double price
    double quantity


# ============================================================================
# 内存管理函数
# ============================================================================

cdef BatchNetworkData* alloc_batch_network_data(int num_networks, int max_nodes,
                                                 int max_connections, int max_inputs,
                                                 int max_outputs) noexcept
cdef void free_batch_network_data(BatchNetworkData* data) noexcept

cdef BatchAgentState* alloc_batch_agent_state(int num_agents) noexcept
cdef void free_batch_agent_state(BatchAgentState* data) noexcept

cdef ThreadLocalBuffer* alloc_thread_buffers(int num_threads, int max_nodes,
                                              int max_inputs, int max_outputs) noexcept
cdef void free_thread_buffers(ThreadLocalBuffer* buffers, int num_threads) noexcept

cdef MarketStateData* alloc_market_state_data() noexcept
cdef void free_market_state_data(MarketStateData* data) noexcept


# ============================================================================
# 核心计算函数 (nogil)
# ============================================================================

# 批量 observe
cdef void batch_observe_retail_nogil(
    BatchAgentState* agents,
    MarketStateData* market,
    double[:, :] outputs,
    int num_threads
) noexcept nogil

cdef void batch_observe_full_nogil(
    BatchAgentState* agents,
    MarketStateData* market,
    double[:, :] outputs,
    int num_threads
) noexcept nogil

cdef void batch_observe_market_maker_nogil(
    BatchAgentState* agents,
    MarketStateData* market,
    double[:, :] outputs,
    int num_threads
) noexcept nogil


# 批量 forward
cdef void batch_forward_nogil(
    BatchNetworkData* networks,
    double[:, :] inputs,
    double[:, :] outputs,
    ThreadLocalBuffer* buffers,
    int num_threads
) noexcept nogil


# 批量 parse
cdef void batch_parse_retail_nogil(
    double[:, :] nn_outputs,
    DecisionResult* results,
    double mid_price,
    double tick_size,
    int num_agents,
    int num_threads
) noexcept nogil

cdef void batch_parse_market_maker_nogil(
    double[:, :] nn_outputs,
    DecisionResult* results,
    double mid_price,
    double tick_size,
    int num_agents,
    int num_threads
) noexcept nogil
