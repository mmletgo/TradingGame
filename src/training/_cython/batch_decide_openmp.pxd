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

# 多市场状态内存管理
cdef MarketStateData** alloc_multi_market_state_data(int num_arenas) noexcept
cdef void free_multi_market_state_data(MarketStateData** data, int num_arenas) noexcept


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

# 多市场状态版本的批量 observe
cdef void batch_observe_retail_multi_market_nogil(
    BatchAgentState* agents,
    MarketStateData** markets,
    int* market_indices,
    double[:, :] outputs,
    int num_threads
) noexcept nogil

cdef void batch_observe_full_multi_market_nogil(
    BatchAgentState* agents,
    MarketStateData** markets,
    int* market_indices,
    double[:, :] outputs,
    int num_threads
) noexcept nogil

cdef void batch_observe_market_maker_multi_market_nogil(
    BatchAgentState* agents,
    MarketStateData** markets,
    int* market_indices,
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

# 带网络索引的批量 forward
cdef void batch_forward_with_indices_nogil(
    BatchNetworkData* networks,
    int* network_indices,
    double[:, :] inputs,
    double[:, :] outputs,
    ThreadLocalBuffer* buffers,
    int num_tasks,
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

# 多市场状态版本的批量 parse
cdef void batch_parse_retail_multi_market_nogil(
    double[:, :] nn_outputs,
    DecisionResult* results,
    MarketStateData** markets,
    int* market_indices,
    int num_agents,
    int num_threads
) noexcept nogil

cdef void batch_parse_market_maker_multi_market_nogil(
    double[:, :] nn_outputs,
    DecisionResult* results,
    MarketStateData** markets,
    int* market_indices,
    int num_agents,
    int num_threads
) noexcept nogil


# ============================================================================
# 缓存类型常量
# ============================================================================

cdef int CACHE_TYPE_RETAIL
cdef int CACHE_TYPE_FULL
cdef int CACHE_TYPE_MARKET_MAKER


# ============================================================================
# 网络数据缓存类
# ============================================================================

cdef class BatchNetworkCache:
    """网络数据缓存类，避免每次调用都重新提取网络数据"""

    # C 结构体指针
    cdef BatchNetworkData* network_data
    cdef ThreadLocalBuffer* thread_buffers
    cdef BatchAgentState* agent_state
    cdef MarketStateData* market_data
    cdef DecisionResult* results

    # 缓存参数
    cdef int num_networks
    cdef int num_threads
    cdef int input_dim
    cdef int output_dim
    cdef int max_nodes
    cdef int max_connections

    # 预分配的 NumPy 数组
    cdef object inputs_array   # np.ndarray
    cdef object outputs_array  # np.ndarray

    # 网络 ID 列表（用于检测是否需要更新）
    cdef list network_ids

    # 类型标识 (0=retail, 1=full, 2=market_maker)
    cdef int cache_type

    # ========== 多竞技场预分配缓冲区 ==========
    # 预分配容量
    cdef int multi_arena_max_tasks
    cdef int multi_arena_max_arenas

    # 预分配的 C 结构体
    cdef BatchAgentState* multi_arena_agents
    cdef MarketStateData** multi_arena_markets
    cdef int* multi_arena_network_indices
    cdef int* multi_arena_market_indices
    cdef DecisionResult* multi_arena_results

    # 预分配的 NumPy 数组（多竞技场用）
    cdef object multi_arena_inputs   # np.ndarray
    cdef object multi_arena_outputs  # np.ndarray
