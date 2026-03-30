"""种群管理模块

管理特定类型 Agent 的种群，支持从 NEAT 基因组创建 Agent。
"""

import ctypes
import gc
import time
import logging
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

import neat
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.population import CompleteExtinctionException
import numpy as np

from src.bio.agents.base import Agent


# =============================================
# 轻量级基因组序列化/反序列化函数（模块级）
# =============================================


# =============================================
# 激活函数和聚合函数的索引映射
# =============================================
_ACTIVATION_TO_IDX: dict[str, int] = {
    'sigmoid': 0, 'tanh': 1, 'relu': 2, 'identity': 3,
    'sin': 4, 'gauss': 5, 'softplus': 6, 'clamped': 7,
    'exp': 8, 'abs': 9, 'inv': 10, 'log': 11, 'hat': 12,
    'square': 13, 'cube': 14,
}
_IDX_TO_ACTIVATION: dict[int, str] = {v: k for k, v in _ACTIVATION_TO_IDX.items()}

_AGGREGATION_TO_IDX: dict[str, int] = {
    'sum': 0, 'product': 1, 'min': 2, 'max': 3, 'mean': 4,
    'median': 5, 'maxabs': 6,
}
_IDX_TO_AGGREGATION: dict[int, str] = {v: k for k, v in _AGGREGATION_TO_IDX.items()}

# NumPy 结构化数组 dtype
_NODE_DTYPE = np.dtype([
    ('key', 'i4'),
    ('bias', 'f4'),
    ('response', 'f4'),
    ('activation', 'u1'),
    ('aggregation', 'u1'),
])

_CONN_DTYPE = np.dtype([
    ('in_node', 'i4'),
    ('out_node', 'i4'),
    ('innovation', 'i4'),
    ('weight', 'f4'),
    ('enabled', '?'),
])


def _serialize_genomes_numpy(
    genomes: dict[int, neat.DefaultGenome],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将基因组字典序列化为 NumPy 紧凑格式

    使用 NumPy 结构化数组，显著减少 pickle 序列化开销。

    Args:
        genomes: NEAT 基因组字典 {genome_key: genome}

    Returns:
        (keys, fitnesses, all_nodes, all_connections) 元组
        - keys: shape=(N,), 基因组 key
        - fitnesses: shape=(N,), 适应度
        - all_nodes: 所有基因组的节点数据（连续存储）
        - all_connections: 所有基因组的连接数据（连续存储）
    """
    genome_list = list(genomes.values())
    n_genomes = len(genome_list)

    # 1. 基因组 key 和适应度
    keys = np.array([g.key for g in genome_list], dtype=np.int32)
    fitnesses = np.array(
        [g.fitness if g.fitness is not None else np.nan for g in genome_list],
        dtype=np.float32,
    )

    # 2. 统计总节点数和连接数
    total_nodes = sum(len(g.nodes) for g in genome_list)
    total_conns = sum(len(g.connections) for g in genome_list)

    # 3. 分配数组
    all_nodes = np.empty(total_nodes, dtype=_NODE_DTYPE)
    all_conns = np.empty(total_conns, dtype=_CONN_DTYPE)

    # 4. 记录每个基因组的节点/连接起始位置和数量
    node_offsets = np.empty(n_genomes + 1, dtype=np.int32)
    conn_offsets = np.empty(n_genomes + 1, dtype=np.int32)
    node_offsets[0] = 0
    conn_offsets[0] = 0

    node_idx = 0
    conn_idx = 0

    for i, genome in enumerate(genome_list):
        # 序列化节点
        for nk, node in genome.nodes.items():
            all_nodes[node_idx]['key'] = nk
            all_nodes[node_idx]['bias'] = node.bias
            all_nodes[node_idx]['response'] = node.response
            all_nodes[node_idx]['activation'] = _ACTIVATION_TO_IDX.get(node.activation, 0)
            all_nodes[node_idx]['aggregation'] = _AGGREGATION_TO_IDX.get(node.aggregation, 0)
            node_idx += 1

        # 序列化连接
        for ck, conn in genome.connections.items():
            all_conns[conn_idx]['in_node'] = ck[0]
            all_conns[conn_idx]['out_node'] = ck[1]
            all_conns[conn_idx]['innovation'] = conn.innovation
            all_conns[conn_idx]['weight'] = conn.weight
            all_conns[conn_idx]['enabled'] = conn.enabled
            conn_idx += 1

        node_offsets[i + 1] = node_idx
        conn_offsets[i + 1] = conn_idx

    # 5. 打包偏移量到元数据数组
    # 使用 node_offsets 和 conn_offsets 重建时恢复每个基因组的边界
    metadata = np.stack([node_offsets, conn_offsets], axis=1)  # shape: (N+1, 2)

    return keys, fitnesses, metadata, all_nodes, all_conns


def _deserialize_genomes_numpy(
    keys: np.ndarray,
    fitnesses: np.ndarray,
    metadata: np.ndarray,
    all_nodes: np.ndarray,
    all_conns: np.ndarray,
    genome_config: Any,
) -> dict[int, neat.DefaultGenome]:
    """从 NumPy 紧凑格式重建基因组字典

    Args:
        keys: 基因组 key 数组
        fitnesses: 适应度数组
        metadata: 偏移量元数据 (N+1, 2)
        all_nodes: 所有节点数据
        all_conns: 所有连接数据
        genome_config: NEAT genome_config（未使用）

    Returns:
        NEAT 基因组字典 {genome_key: genome}
    """
    population: dict[int, neat.DefaultGenome] = {}
    n_genomes = len(keys)
    node_offsets = metadata[:, 0]
    conn_offsets = metadata[:, 1]

    for i in range(n_genomes):
        key = int(keys[i])
        fitness = float(fitnesses[i]) if not np.isnan(fitnesses[i]) else None

        genome = neat.DefaultGenome(key)
        genome.fitness = fitness
        genome.nodes = {}
        genome.connections = {}

        # 恢复节点
        node_start = node_offsets[i]
        node_end = node_offsets[i + 1]
        for j in range(node_start, node_end):
            node_data = all_nodes[j]
            node_key = int(node_data['key'])
            node = DefaultNodeGene(node_key)
            node.bias = float(node_data['bias'])
            node.response = float(node_data['response'])
            node.activation = _IDX_TO_ACTIVATION.get(int(node_data['activation']), 'sigmoid')
            node.aggregation = _IDX_TO_AGGREGATION.get(int(node_data['aggregation']), 'sum')
            genome.nodes[node_key] = node

        # 恢复连接
        conn_start = conn_offsets[i]
        conn_end = conn_offsets[i + 1]
        for j in range(conn_start, conn_end):
            conn_data = all_conns[j]
            ck = (int(conn_data['in_node']), int(conn_data['out_node']))
            conn = DefaultConnectionGene(ck, innovation=int(conn_data['innovation']))
            conn.weight = float(conn_data['weight'])
            conn.enabled = bool(conn_data['enabled'])
            genome.connections[ck] = conn

        population[key] = genome

    return population


def _serialize_species_data(species_set: Any) -> tuple[np.ndarray, np.ndarray]:
    """序列化 species 数据为 NumPy 格式

    Args:
        species_set: NEAT DefaultSpeciesSet 对象

    Returns:
        (genome_ids, species_ids) 元组
        - genome_ids: shape=(N,), 所有 genome 的 ID
        - species_ids: shape=(N,), 对应的 species ID
    """
    if species_set is None or not hasattr(species_set, 'genome_to_species'):
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    genome_to_species = species_set.genome_to_species
    if not genome_to_species:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    genome_ids = np.array(list(genome_to_species.keys()), dtype=np.int32)
    species_ids = np.array(list(genome_to_species.values()), dtype=np.int32)

    return genome_ids, species_ids


def _apply_species_data_to_population(
    neat_pop: Any,
    species_genome_ids: np.ndarray,
    species_species_ids: np.ndarray,
    generation: int,
) -> None:
    """将 species 数据应用到 NEAT 种群

    根据从 Worker 返回的 genome_id -> species_id 映射，
    重建主进程中 neat_pop.species 的结构。

    Args:
        neat_pop: NEAT Population 对象
        species_genome_ids: genome ID 数组
        species_species_ids: 对应的 species ID 数组
        generation: 当前代数
    """
    if len(species_genome_ids) == 0:
        return

    species_set = neat_pop.species
    if species_set is None:
        return

    # 1. 重建 genome_to_species 映射
    genome_to_species: dict[int, int] = {}
    for gid, sid in zip(species_genome_ids, species_species_ids):
        genome_to_species[int(gid)] = int(sid)
    species_set.genome_to_species = genome_to_species

    # 2. 重建 species.species 字典
    # 首先收集每个 species 的成员
    species_members: dict[int, dict[int, Any]] = {}
    for gid, sid in genome_to_species.items():
        if sid not in species_members:
            species_members[sid] = {}
        genome = neat_pop.population.get(gid)
        if genome is not None:
            species_members[sid][gid] = genome

    # 3. 更新或创建 species
    current_species_ids = set(species_members.keys())
    existing_species_ids = set(species_set.species.keys())

    # 删除不再存在的 species（先清理内部数据，防止内存泄漏）
    for sid in existing_species_ids - current_species_ids:
        old_species = species_set.species[sid]
        # 清理 members 字典引用
        if hasattr(old_species, 'members') and old_species.members:
            old_species.members.clear()
        # 清理 representative 引用
        if hasattr(old_species, 'representative'):
            old_species.representative = None
        # 清理 fitness_history
        if hasattr(old_species, 'fitness_history'):
            old_species.fitness_history = []
        del species_set.species[sid]

    # 更新或创建 species
    for sid, members in species_members.items():
        if sid in species_set.species:
            # 更新现有 species 的 members
            species = species_set.species[sid]
            if species.members:
                species.members.clear()  # 显式清空旧 dict
            species.members = members
            # 更新 representative（使用第一个成员）
            if members:
                species.representative = next(iter(members.values()))
        else:
            # 创建新的 species
            from neat.species import Species
            species = Species(sid, generation)
            species.members = members
            if members:
                species.representative = next(iter(members.values()))
            species_set.species[sid] = species


# 保留旧接口用于兼容
def _serialize_genomes(genomes: dict[int, neat.DefaultGenome]) -> list[tuple[Any, ...]]:
    """将基因组字典序列化为轻量级格式（紧凑版）

    使用元组列表而非字典，减少序列化开销。
    格式：(key, fitness, nodes_list, connections_list)
    其中：
    - nodes_list: [(node_key, bias, response, activation, aggregation), ...]
    - connections_list: [(in_node, out_node, innovation, weight, enabled), ...]

    Args:
        genomes: NEAT 基因组字典 {genome_key: genome}

    Returns:
        轻量级基因组数据列表
    """
    result: list[tuple[Any, ...]] = []
    for gid, genome in genomes.items():
        # 序列化节点为元组列表
        nodes_list: list[tuple[int, float, float, str, str]] = [
            (int(nk), float(node.bias), float(node.response), node.activation, node.aggregation)
            for nk, node in genome.nodes.items()
        ]
        # 序列化连接为元组列表
        connections_list: list[tuple[int, int, int, float, bool]] = [
            (int(ck[0]), int(ck[1]), int(conn.innovation), float(conn.weight), bool(conn.enabled))
            for ck, conn in genome.connections.items()
        ]
        result.append((
            int(genome.key),
            float(genome.fitness) if genome.fitness is not None else None,
            nodes_list,
            connections_list,
        ))
    return result


def _deserialize_genomes(
    genome_data_list: list[tuple[Any, ...]],
    genome_config: Any,
) -> dict[int, neat.DefaultGenome]:
    """从轻量级格式重建基因组字典（紧凑版）

    Args:
        genome_data_list: 轻量级基因组数据列表
        genome_config: NEAT 的 genome_config（未使用，保留接口一致性）

    Returns:
        NEAT 基因组字典 {genome_key: genome}
    """
    population: dict[int, neat.DefaultGenome] = {}

    for gdata in genome_data_list:
        key, fitness, nodes_list, connections_list = gdata
        genome = neat.DefaultGenome(key)
        genome.fitness = fitness
        genome.nodes = {}
        genome.connections = {}

        # 重建节点
        for node_key, bias, response, activation, aggregation in nodes_list:
            node = DefaultNodeGene(node_key)
            node.bias = bias
            node.response = response
            node.activation = activation
            node.aggregation = aggregation
            genome.nodes[node_key] = node

        # 重建连接
        for in_node, out_node, innovation, weight, enabled in connections_list:
            ck = (in_node, out_node)
            conn = DefaultConnectionGene(ck, innovation=innovation)
            conn.weight = weight
            conn.enabled = enabled
            genome.connections[ck] = conn

        population[genome.key] = genome

    return population


def _evolve_subpop_lightweight(
    args: tuple[str, list[tuple[Any, ...]], int, int],
) -> list[tuple[Any, ...]]:
    """在子进程中执行轻量级进化（元组格式，已废弃）

    这是一个模块级函数，可被 ProcessPoolExecutor 调用。

    Args:
        args: (neat_config_path, genome_data_list, pop_size, generation)

    Returns:
        新基因组的轻量级数据列表
    """
    neat_config_path, genome_data_list, pop_size, generation = args

    # 1. 加载 NEAT 配置
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )
    neat_config.pop_size = pop_size

    # 2. 创建新的 NEAT Population（空的）
    neat_pop = neat.Population(neat_config)

    # 3. 用传入的数据重建种群
    neat_pop.population = _deserialize_genomes(genome_data_list, neat_config.genome_config)
    neat_pop.generation = generation

    # 4. 重新进行物种划分
    neat_pop.species.speciate(neat_config, neat_pop.population, generation)

    # 5. 执行繁殖
    def eval_genomes(
        genomes: list[tuple[int, neat.DefaultGenome]],
        config: neat.Config,
    ) -> None:
        pass  # 适应度已设置

    try:
        neat_pop.run(eval_genomes, n=1)
    except Exception:
        # 进化失败，返回原始基因组
        return genome_data_list

    # 6. 序列化新种群返回
    return _serialize_genomes(neat_pop.population)


def _evolve_subpop_numpy(
    args: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在子进程中执行进化（NumPy 格式）

    使用 NumPy 结构化数组，显著减少序列化开销。

    Args:
        args: (neat_config_path, keys, fitnesses, metadata, nodes, conns, pop_size, generation)

    Returns:
        (new_keys, new_fitnesses, new_metadata, new_nodes, new_conns)
    """
    neat_config_path, keys, fitnesses, metadata, nodes, conns, pop_size, generation = args

    # 1. 加载 NEAT 配置
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )
    neat_config.pop_size = pop_size

    # 2. 创建新的 NEAT Population
    neat_pop = neat.Population(neat_config)

    # 3. 用传入的 NumPy 数据重建种群
    neat_pop.population = _deserialize_genomes_numpy(
        keys, fitnesses, metadata, nodes, conns, neat_config.genome_config
    )
    neat_pop.generation = generation

    # 4. 重新进行物种划分
    neat_pop.species.speciate(neat_config, neat_pop.population, generation)

    # 5. 执行繁殖
    def eval_genomes(
        genomes: list[tuple[int, neat.DefaultGenome]],
        config: neat.Config,
    ) -> None:
        pass  # 适应度已设置

    try:
        neat_pop.run(eval_genomes, n=1)
    except Exception:
        # 进化失败，返回原始数据
        return keys, fitnesses, metadata, nodes, conns

    # 6. 序列化新种群返回（NumPy 格式）
    return _serialize_genomes_numpy(neat_pop.population)


# =============================================
# 网络参数序列化/反序列化函数（用于快速传输）
# =============================================

# 网络参数 NumPy dtype
_NETWORK_PARAM_HEADER_DTYPE = np.dtype([
    ('num_inputs', 'i4'),
    ('num_outputs', 'i4'),
    ('num_nodes', 'i4'),
    ('n_input_keys', 'i4'),
    ('n_output_keys', 'i4'),
    ('n_node_ids', 'i4'),
    ('n_connections', 'i4'),  # conn_weights 的长度
])


def _extract_network_params_from_genome(
    genome: neat.DefaultGenome,
    config: neat.Config,
) -> dict[str, np.ndarray | int]:
    """从基因组提取网络参数（在 Worker 进程中使用）

    创建临时网络并提取参数，然后释放网络。
    这样 Worker 不需要维护网络对象。

    Args:
        genome: NEAT 基因组
        config: NEAT 配置

    Returns:
        网络参数字典
    """
    from neat.nn import FastFeedForwardNetwork
    network = FastFeedForwardNetwork.create(genome, config)
    params = network.get_params()
    # 【内存泄漏修复】显式清理临时网络对象的内部状态
    # FastFeedForwardNetwork 内部可能持有对 genome 和 config 的引用
    if hasattr(network, 'node_evals'):
        network.node_evals = None  # type: ignore[assignment]
    if hasattr(network, 'values'):
        network.values = None  # type: ignore[assignment]
    if hasattr(network, 'input_nodes'):
        network.input_nodes = None  # type: ignore[assignment]
    if hasattr(network, 'output_nodes'):
        network.output_nodes = None  # type: ignore[assignment]
    # 显式删除网络对象引用
    del network
    return params


def _pack_network_params_numpy(
    params_list: list[dict[str, np.ndarray | int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将多个网络参数打包成 NumPy 数组

    将多个网络的参数打包成紧凑的 NumPy 数组格式，减少 pickle 开销。

    Args:
        params_list: 网络参数字典列表

    Returns:
        (headers, all_input_keys, all_output_keys, all_node_ids,
         all_biases, all_responses, all_act_types,
         all_conn_indptr, all_conn_sources, all_conn_weights,
         all_output_indices) 元组
    """
    n_networks = len(params_list)

    # 1. 构建 header 数组
    headers = np.empty(n_networks, dtype=_NETWORK_PARAM_HEADER_DTYPE)

    # 2. 统计总大小
    total_input_keys = 0
    total_output_keys = 0
    total_node_ids = 0
    total_connections = 0

    for i, params in enumerate(params_list):
        n_input_keys = len(params['input_keys'])
        n_output_keys = len(params['output_keys'])
        n_node_ids = len(params['node_ids'])
        n_connections = len(params['conn_weights'])

        headers[i]['num_inputs'] = params['num_inputs']
        headers[i]['num_outputs'] = params['num_outputs']
        headers[i]['num_nodes'] = params['num_nodes']
        headers[i]['n_input_keys'] = n_input_keys
        headers[i]['n_output_keys'] = n_output_keys
        headers[i]['n_node_ids'] = n_node_ids
        headers[i]['n_connections'] = n_connections

        total_input_keys += n_input_keys
        total_output_keys += n_output_keys
        total_node_ids += n_node_ids
        total_connections += n_connections

    # 3. 分配连续数组
    all_input_keys = np.empty(total_input_keys, dtype=np.int32)
    all_output_keys = np.empty(total_output_keys, dtype=np.int32)
    all_node_ids = np.empty(total_node_ids, dtype=np.int32)
    all_biases = np.empty(total_node_ids, dtype=np.float32)
    all_responses = np.empty(total_node_ids, dtype=np.float32)
    all_act_types = np.empty(total_node_ids, dtype=np.int32)
    # conn_indptr 的长度是 num_nodes + 1
    total_conn_indptr = sum(params['num_nodes'] + 1 for params in params_list)
    all_conn_indptr = np.empty(total_conn_indptr, dtype=np.int32)
    all_conn_sources = np.empty(total_connections, dtype=np.int32)
    all_conn_weights = np.empty(total_connections, dtype=np.float32)
    all_output_indices = np.empty(total_output_keys, dtype=np.int32)

    # 4. 填充数据
    input_keys_offset = 0
    output_keys_offset = 0
    node_ids_offset = 0
    conn_indptr_offset = 0
    connections_offset = 0

    for params in params_list:
        n_input_keys = len(params['input_keys'])
        n_output_keys = len(params['output_keys'])
        n_node_ids = len(params['node_ids'])
        n_conn_indptr = params['num_nodes'] + 1
        n_connections = len(params['conn_weights'])

        # 复制数据
        all_input_keys[input_keys_offset:input_keys_offset + n_input_keys] = params['input_keys']
        all_output_keys[output_keys_offset:output_keys_offset + n_output_keys] = params['output_keys']
        all_node_ids[node_ids_offset:node_ids_offset + n_node_ids] = params['node_ids']
        all_biases[node_ids_offset:node_ids_offset + n_node_ids] = params['biases']
        all_responses[node_ids_offset:node_ids_offset + n_node_ids] = params['responses']
        all_act_types[node_ids_offset:node_ids_offset + n_node_ids] = params['act_types']
        all_conn_indptr[conn_indptr_offset:conn_indptr_offset + n_conn_indptr] = params['conn_indptr']
        all_conn_sources[connections_offset:connections_offset + n_connections] = params['conn_sources']
        all_conn_weights[connections_offset:connections_offset + n_connections] = params['conn_weights']
        all_output_indices[output_keys_offset:output_keys_offset + n_output_keys] = params['output_indices']

        input_keys_offset += n_input_keys
        output_keys_offset += n_output_keys
        node_ids_offset += n_node_ids
        conn_indptr_offset += n_conn_indptr
        connections_offset += n_connections

    return (
        headers, all_input_keys, all_output_keys, all_node_ids,
        all_biases, all_responses, all_act_types,
        all_conn_indptr, all_conn_sources, all_conn_weights,
        all_output_indices,
    )


def _unpack_network_params_numpy(
    headers: np.ndarray,
    all_input_keys: np.ndarray,
    all_output_keys: np.ndarray,
    all_node_ids: np.ndarray,
    all_biases: np.ndarray,
    all_responses: np.ndarray,
    all_act_types: np.ndarray,
    all_conn_indptr: np.ndarray,
    all_conn_sources: np.ndarray,
    all_conn_weights: np.ndarray,
    all_output_indices: np.ndarray,
) -> list[dict[str, np.ndarray | int]]:
    """从 NumPy 数组解包网络参数

    Args:
        headers: 网络头信息数组
        all_*: 各类数据的连续数组

    Returns:
        网络参数字典列表
    """
    n_networks = len(headers)
    params_list: list[dict[str, np.ndarray | int]] = []

    input_keys_offset = 0
    output_keys_offset = 0
    node_ids_offset = 0
    conn_indptr_offset = 0
    connections_offset = 0

    for i in range(n_networks):
        h = headers[i]
        num_inputs = int(h['num_inputs'])
        num_outputs = int(h['num_outputs'])
        num_nodes = int(h['num_nodes'])
        n_input_keys = int(h['n_input_keys'])
        n_output_keys = int(h['n_output_keys'])
        n_node_ids = int(h['n_node_ids'])
        n_connections = int(h['n_connections'])
        n_conn_indptr = num_nodes + 1

        params = {
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'num_nodes': num_nodes,
            'input_keys': all_input_keys[input_keys_offset:input_keys_offset + n_input_keys],
            'output_keys': all_output_keys[output_keys_offset:output_keys_offset + n_output_keys],
            'node_ids': all_node_ids[node_ids_offset:node_ids_offset + n_node_ids],
            'biases': all_biases[node_ids_offset:node_ids_offset + n_node_ids],
            'responses': all_responses[node_ids_offset:node_ids_offset + n_node_ids],
            'act_types': all_act_types[node_ids_offset:node_ids_offset + n_node_ids],
            'conn_indptr': all_conn_indptr[conn_indptr_offset:conn_indptr_offset + n_conn_indptr],
            'conn_sources': all_conn_sources[connections_offset:connections_offset + n_connections],
            'conn_weights': all_conn_weights[connections_offset:connections_offset + n_connections],
            'output_indices': all_output_indices[output_keys_offset:output_keys_offset + n_output_keys],
        }
        params_list.append(params)

        input_keys_offset += n_input_keys
        output_keys_offset += n_output_keys
        node_ids_offset += n_node_ids
        conn_indptr_offset += n_conn_indptr
        connections_offset += n_connections

    return params_list



def _extract_and_pack_all_network_params(
    population: dict[int, "neat.DefaultGenome"],
    config: "neat.Config",
) -> tuple[np.ndarray, ...]:
    """从种群中提取所有网络参数并直接打包为 NumPy 数组

    合并 _extract_network_params_from_genome + _pack_network_params_numpy 的逻辑，
    避免创建 N 个中间 dict 对象。

    Two-pass approach:
    - Pass 1: Create networks, get params, count sizes
    - Pass 2: Fill pre-allocated arrays and clean up immediately

    Args:
        population: NEAT 种群（genome_id -> genome 映射）
        config: NEAT 配置

    Returns:
        与 _pack_network_params_numpy 相同格式的元组:
        (headers, all_input_keys, all_output_keys, all_node_ids,
         all_biases, all_responses, all_act_types,
         all_conn_indptr, all_conn_sources, all_conn_weights,
         all_output_indices)
    """
    from neat.nn import FastFeedForwardNetwork

    genome_list = list(population.values())
    n_networks = len(genome_list)

    # Pass 1: Extract params and count sizes
    headers = np.empty(n_networks, dtype=_NETWORK_PARAM_HEADER_DTYPE)

    total_input_keys: int = 0
    total_output_keys: int = 0
    total_node_ids: int = 0
    total_connections: int = 0
    total_conn_indptr: int = 0

    # Store extracted params temporarily
    all_params: list[dict[str, np.ndarray | int]] = []

    for i, genome in enumerate(genome_list):
        network = FastFeedForwardNetwork.create(genome, config)
        params = network.get_params()

        # Clean up network immediately
        if hasattr(network, 'node_evals'):
            network.node_evals = None  # type: ignore[assignment]
        if hasattr(network, 'values'):
            network.values = None  # type: ignore[assignment]
        if hasattr(network, 'input_nodes'):
            network.input_nodes = None  # type: ignore[assignment]
        if hasattr(network, 'output_nodes'):
            network.output_nodes = None  # type: ignore[assignment]
        del network

        n_input_keys = len(params['input_keys'])
        n_output_keys = len(params['output_keys'])
        n_node_ids = len(params['node_ids'])
        n_connections = len(params['conn_weights'])

        headers[i]['num_inputs'] = params['num_inputs']
        headers[i]['num_outputs'] = params['num_outputs']
        headers[i]['num_nodes'] = params['num_nodes']
        headers[i]['n_input_keys'] = n_input_keys
        headers[i]['n_output_keys'] = n_output_keys
        headers[i]['n_node_ids'] = n_node_ids
        headers[i]['n_connections'] = n_connections

        total_input_keys += n_input_keys
        total_output_keys += n_output_keys
        total_node_ids += n_node_ids
        total_connections += n_connections
        total_conn_indptr += params['num_nodes'] + 1

        all_params.append(params)

    # Allocate arrays
    all_input_keys_arr = np.empty(total_input_keys, dtype=np.int32)
    all_output_keys_arr = np.empty(total_output_keys, dtype=np.int32)
    all_node_ids_arr = np.empty(total_node_ids, dtype=np.int32)
    all_biases_arr = np.empty(total_node_ids, dtype=np.float32)
    all_responses_arr = np.empty(total_node_ids, dtype=np.float32)
    all_act_types_arr = np.empty(total_node_ids, dtype=np.int32)
    all_conn_indptr_arr = np.empty(total_conn_indptr, dtype=np.int32)
    all_conn_sources_arr = np.empty(total_connections, dtype=np.int32)
    all_conn_weights_arr = np.empty(total_connections, dtype=np.float32)
    all_output_indices_arr = np.empty(total_output_keys, dtype=np.int32)

    # Pass 2: Fill arrays and clean up params immediately
    ik_off: int = 0
    ok_off: int = 0
    nid_off: int = 0
    ci_off: int = 0
    c_off: int = 0

    for params in all_params:
        nik = len(params['input_keys'])
        nok = len(params['output_keys'])
        nnid = len(params['node_ids'])
        nc = len(params['conn_weights'])
        nci = params['num_nodes'] + 1

        all_input_keys_arr[ik_off:ik_off + nik] = params['input_keys']
        all_output_keys_arr[ok_off:ok_off + nok] = params['output_keys']
        all_node_ids_arr[nid_off:nid_off + nnid] = params['node_ids']
        all_biases_arr[nid_off:nid_off + nnid] = params['biases']
        all_responses_arr[nid_off:nid_off + nnid] = params['responses']
        all_act_types_arr[nid_off:nid_off + nnid] = params['act_types']
        all_conn_indptr_arr[ci_off:ci_off + nci] = params['conn_indptr']
        all_conn_sources_arr[c_off:c_off + nc] = params['conn_sources']
        all_conn_weights_arr[c_off:c_off + nc] = params['conn_weights']
        all_output_indices_arr[ok_off:ok_off + nok] = params['output_indices']

        ik_off += nik
        ok_off += nok
        nid_off += nnid
        ci_off += nci
        c_off += nc

        # Clean up params dict immediately
        params.clear()

    all_params.clear()
    del all_params

    return (
        headers, all_input_keys_arr, all_output_keys_arr, all_node_ids_arr,
        all_biases_arr, all_responses_arr, all_act_types_arr,
        all_conn_indptr_arr, all_conn_sources_arr, all_conn_weights_arr,
        all_output_indices_arr,
    )


def _concat_network_params_numpy(
    params_list: list[tuple[np.ndarray, ...]],
) -> tuple[np.ndarray, ...]:
    """将多个子种群的 network_params_data 拼接为一个
    
    用于将 SubPopulationManager 下多个子种群的 packed numpy 数组
    合并成一个完整的 packed numpy 数组，供 BatchNetworkCache.update_networks_from_numpy 使用。
    
    Args:
        params_list: 多个子种群的 network_params_data 元组列表，
                     每个元组包含 11 个 numpy 数组
    
    Returns:
        拼接后的 11 个 numpy 数组的元组
    """
    if len(params_list) == 1:
        return params_list[0]
    return tuple(np.concatenate([p[i] for p in params_list]) for i in range(len(params_list[0])))


# =============================================
# 持久 Worker 进程相关函数
# =============================================


def _worker_process_main(
    worker_id: int,
    neat_config_path: str,
    pop_size: int,
    cmd_queue: "multiprocessing.Queue[tuple[str, Any]]",
    result_queue: "multiprocessing.Queue[tuple[int, Any]]",
) -> None:
    """Worker 进程主函数

    在子进程中运行，维护一个 NEAT 种群，接收命令并执行。

    Args:
        worker_id: Worker ID
        neat_config_path: NEAT 配置文件路径
        pop_size: 种群大小
        cmd_queue: 命令队列 (command, args)
        result_queue: 结果队列 (worker_id, result)
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # 忽略 Ctrl+C

    # 1. 初始化 NEAT 配置和种群
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )
    neat_config.pop_size = pop_size
    neat_pop = neat.Population(neat_config)
    generation = 0

    # 2. 主循环：等待命令
    while True:
        try:
            cmd, args = cmd_queue.get()
        except Exception:
            break

        if cmd == "shutdown":
            break

        elif cmd == "evolve":
            # args: np.ndarray of fitnesses (shape: pop_size,)
            fitnesses = args

            # 更新适应度
            for i, (gid, genome) in enumerate(neat_pop.population.items()):
                if i < len(fitnesses):
                    genome.fitness = float(fitnesses[i])

            # 执行进化
            def eval_genomes(genomes: list, config: Any) -> None:
                pass  # 适应度已设置

            try:
                neat_pop.run(eval_genomes, n=1)
                generation += 1
            except Exception as e:
                result_queue.put((worker_id, ("error", str(e))))
                continue

            # 【关键修复】清理 NEAT 历史数据，防止内存泄漏
            _cleanup_worker_neat_history(neat_pop)

            # 返回新基因组数据（NumPy 格式）
            new_data = _serialize_genomes_numpy(neat_pop.population)
            result_queue.put((worker_id, ("success", new_data)))
            # 【内存泄漏修复】发送后删除大型数据
            del new_data
            del fitnesses  # 也删除输入的适应度数组
            # 显式触发年轻代 GC
            gc.collect(0)

        elif cmd == "get_genomes":
            # 返回当前基因组数据
            data = _serialize_genomes_numpy(neat_pop.population)
            result_queue.put((worker_id, data))
            # 【内存泄漏修复】发送后删除大型数据
            del data

        elif cmd == "set_genomes":
            # 设置基因组数据
            keys, fitnesses_in, metadata, nodes, conns = args

            # 【内存优化】在替换种群之前，先清理旧数据
            # 帮助 GC 更快回收旧的 genome 对象
            if neat_pop.population:
                for genome in neat_pop.population.values():
                    if hasattr(genome, 'nodes'):
                        genome.nodes.clear()
                    if hasattr(genome, 'connections'):
                        genome.connections.clear()
                neat_pop.population.clear()

            neat_pop.population = _deserialize_genomes_numpy(
                keys, fitnesses_in, metadata, nodes, conns, neat_config.genome_config
            )

            # 【关键修复】同步 pop_size 为实际种群大小
            # 防止 checkpoint 中的种群大小与配置不一致导致进化失败
            actual_pop_size = len(neat_pop.population)
            if neat_config.pop_size != actual_pop_size:
                neat_config.pop_size = actual_pop_size

            neat_pop.species.speciate(neat_config, neat_pop.population, generation)

            # 【关键修复】同步基因组后立即清理历史数据
            # checkpoint 中可能包含大量历史数据（ancestors、fitness_history 等）
            # 如果不清理，会导致 Worker 进程内存暴涨（20GB+）
            _cleanup_worker_neat_history(neat_pop)

            # 【内存优化】清理命令参数
            del keys, fitnesses_in, metadata, nodes, conns
            gc.collect(0)

            result_queue.put((worker_id, "ok"))

        elif cmd == "evolve_return_params":
            # args: np.ndarray of fitnesses (shape: pop_size,)
            # 返回 genome 数据 + 网络参数 + species 数据，减少主进程重建网络的开销
            fitnesses = args

            # 更新适应度
            for i, (gid, genome) in enumerate(neat_pop.population.items()):
                if i < len(fitnesses):
                    genome.fitness = float(fitnesses[i])

            # 执行进化
            def eval_genomes_params(genomes: list, config: Any) -> None:
                pass  # 适应度已设置

            try:
                neat_pop.run(eval_genomes_params, n=1)
                generation += 1
            except Exception as e:
                result_queue.put((worker_id, ("error", str(e))))
                continue

            # 【关键修复】清理 NEAT 历史数据，防止内存泄漏
            _cleanup_worker_neat_history(neat_pop)

            # 1. 序列化基因组数据（NumPy 格式）
            genome_data = _serialize_genomes_numpy(neat_pop.population)

            # 2. 提取所有网络参数
            params_list = []
            for gid, genome in neat_pop.population.items():
                params = _extract_network_params_from_genome(genome, neat_config)
                params_list.append(params)

            # 3. 打包网络参数（NumPy 格式）
            network_params_data = _pack_network_params_numpy(params_list)

            # 4. 序列化 species 数据（genome_id -> species_id 映射）
            species_data = _serialize_species_data(neat_pop.species)

            # 5. 返回三者
            result_queue.put((worker_id, ("success_params", genome_data, network_params_data, species_data)))

            # 【内存泄漏修复】发送后删除大型数据
            # 与 evolve 命令保持一致的清理方式
            del genome_data
            del params_list
            del network_params_data
            del species_data
            del fitnesses  # 也删除输入的适应度数组
            # 显式触发年轻代 GC
            gc.collect(0)


import multiprocessing


class PersistentWorkerPool:
    """持久 Worker 进程池

    维护多个持久的子进程，每个子进程维护自己的 NEAT 种群。
    避免每次进化都需要序列化整个种群数据。
    """

    def __init__(
        self,
        num_workers: int,
        neat_config_path: str,
        pop_size: int,
    ):
        """初始化 Worker 池

        Args:
            num_workers: Worker 数量
            neat_config_path: NEAT 配置文件路径
            pop_size: 每个 Worker 的种群大小
        """
        self.num_workers = num_workers
        self.neat_config_path = neat_config_path
        self.pop_size = pop_size

        # 创建队列
        self.cmd_queues: list[multiprocessing.Queue[tuple[str, Any]]] = [
            multiprocessing.Queue() for _ in range(num_workers)
        ]
        self.result_queue: multiprocessing.Queue[tuple[int, Any]] = multiprocessing.Queue()

        # 启动 Worker 进程
        self.workers: list[multiprocessing.Process] = []
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=_worker_process_main,
                args=(i, neat_config_path, pop_size, self.cmd_queues[i], self.result_queue),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def evolve_all(
        self,
        fitnesses_list: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """并行进化所有 Worker 的种群

        Args:
            fitnesses_list: 每个 Worker 的适应度数组列表

        Returns:
            每个 Worker 的新基因组数据列表
        """
        # 发送进化命令
        for i, fitnesses in enumerate(fitnesses_list):
            self.cmd_queues[i].put(("evolve", fitnesses))

        # 收集结果
        results: dict[int, tuple[np.ndarray, ...]] = {}
        for _ in range(self.num_workers):
            worker_id, result = self.result_queue.get()
            if result[0] == "success":
                results[worker_id] = result[1]
            else:
                raise RuntimeError(f"Worker {worker_id} evolve failed: {result[1]}")

        # 按 Worker ID 排序返回
        return [results[i] for i in range(self.num_workers)]

    def get_all_genomes(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """获取所有 Worker 的基因组数据

        Returns:
            每个 Worker 的基因组数据列表
        """
        for i in range(self.num_workers):
            self.cmd_queues[i].put(("get_genomes", None))

        results: dict[int, tuple[np.ndarray, ...]] = {}
        for _ in range(self.num_workers):
            worker_id, data = self.result_queue.get()
            results[worker_id] = data

        return [results[i] for i in range(self.num_workers)]

    def set_all_genomes(
        self,
        genomes_list: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        """同步所有 Worker 的基因组数据

        用于初始化 Worker 进程的种群，确保与主进程同步。

        Args:
            genomes_list: 每个 Worker 的基因组数据列表（NumPy 格式）
        """
        # 发送 set_genomes 命令
        for i, genome_data in enumerate(genomes_list):
            self.cmd_queues[i].put(("set_genomes", genome_data))

        # 等待所有 Worker 完成
        for _ in range(self.num_workers):
            worker_id, result = self.result_queue.get()
            if result != "ok":
                raise RuntimeError(f"Worker {worker_id} set_genomes failed: {result}")

    def evolve_all_return_params(
        self,
        fitnesses_list: list[np.ndarray],
    ) -> list[tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],  # genome_data
        tuple[np.ndarray, ...],  # network_params_data (11 arrays)
    ]]:
        """并行进化所有 Worker 的种群，返回基因组数据和网络参数

        相比 evolve_all()，此方法在 Worker 进程中直接提取网络参数，
        主进程可以直接使用网络参数重建网络，避免从基因组重建的开销。

        Args:
            fitnesses_list: 每个 Worker 的适应度数组列表

        Returns:
            每个 Worker 的 (genome_data, network_params_data) 列表
            - genome_data: (keys, fitnesses, metadata, nodes, conns)
            - network_params_data: 11个数组的元组
        """
        # 发送进化命令
        for i, fitnesses in enumerate(fitnesses_list):
            self.cmd_queues[i].put(("evolve_return_params", fitnesses))

        # 收集结果
        results: dict[int, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]] = {}
        for _ in range(self.num_workers):
            worker_id, result = self.result_queue.get()
            if result[0] == "success_params":
                # result = ("success_params", genome_data, network_params_data)
                results[worker_id] = (result[1], result[2])
            elif result[0] == "error":
                raise RuntimeError(f"Worker {worker_id} evolve failed: {result[1]}")
            else:
                raise RuntimeError(f"Worker {worker_id} unexpected result: {result}")

        # 按 Worker ID 排序返回
        return [results[i] for i in range(self.num_workers)]

    def shutdown(self) -> None:
        """关闭所有 Worker"""
        for i in range(self.num_workers):
            try:
                self.cmd_queues[i].put(("shutdown", None))
            except Exception:
                pass

        for p in self.workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()


@dataclass
class WorkerConfig:
    """Worker 配置

    Attributes:
        agent_type: Agent 类型
        sub_pop_id: 子种群 ID
        neat_config_path: NEAT 配置文件路径
        pop_size: 种群大小
    """
    agent_type: "AgentType"
    sub_pop_id: int
    neat_config_path: str
    pop_size: int


class MultiPopulationWorkerPool:
    """多种群统一 Worker 池

    管理多个不同配置的 Worker，每个 Worker 可以有不同的 NEAT 配置和种群大小。
    使用统一的 result_queue 收集所有 Worker 的结果，支持一次性发送/收集所有进化结果（真正并行）。

    Attributes:
        config_dir: NEAT 配置文件目录
        worker_configs: Worker 配置列表
        workers: Worker 进程列表
        cmd_queues: 每个 Worker 的命令队列
        result_queue: 统一结果队列
        worker_id_map: Worker ID 到配置的映射
    """

    config_dir: str
    worker_configs: list[WorkerConfig]
    workers: list[multiprocessing.Process]
    cmd_queues: dict[tuple["AgentType", int], "multiprocessing.Queue[tuple[str, Any]]"]
    result_queue: "multiprocessing.Queue[tuple[tuple[AgentType, int], Any]]"
    worker_id_map: dict[tuple["AgentType", int], WorkerConfig]

    def __init__(self, config_dir: str, worker_configs: list[WorkerConfig]) -> None:
        """初始化多种群 Worker 池

        Args:
            config_dir: NEAT 配置文件目录
            worker_configs: Worker 配置列表，每个配置指定一个 Worker 的参数
        """
        self.config_dir = config_dir
        self.worker_configs = worker_configs

        # 创建统一结果队列
        self.result_queue = multiprocessing.Queue()

        # 创建命令队列和 Worker 进程
        self.cmd_queues = {}
        self.workers = []
        self.worker_id_map = {}

        for wc in worker_configs:
            worker_id = (wc.agent_type, wc.sub_pop_id)
            self.worker_id_map[worker_id] = wc

            # 为每个 Worker 创建独立的命令队列
            cmd_queue: multiprocessing.Queue[tuple[str, Any]] = multiprocessing.Queue()
            self.cmd_queues[worker_id] = cmd_queue

            # 启动 Worker 进程
            p = multiprocessing.Process(
                target=_multi_worker_process_main,
                args=(
                    worker_id,
                    wc.neat_config_path,
                    wc.pop_size,
                    cmd_queue,
                    self.result_queue,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def evolve_all_parallel(
        self,
        fitness_map: dict[tuple["AgentType", int], np.ndarray],
        lite: bool = True,
    ) -> dict[
        tuple["AgentType", int],
        tuple[tuple[np.ndarray, ...] | None, tuple[np.ndarray, ...], tuple[np.ndarray, np.ndarray]]
    ]:
        """同时进化所有 Worker 的种群

        1. 同时向所有 Worker 发送进化命令（非阻塞）
        2. 一次性收集所有结果（真正并行）

        Args:
            fitness_map: 字典，key 为 (agent_type, sub_pop_id)，value 为适应度数组
            lite: 是否使用轻量模式（跳过基因组序列化，默认 True）

        Returns:
            字典，key 为 (agent_type, sub_pop_id)，
            value 为 (genome_data, network_params_data, species_data) 元组
            其中 genome_data 在 lite 模式下为 None
            species_data = (genome_ids, species_ids)
        """
        logger = get_logger("training")

        # 1. 同时向所有 Worker 发送进化命令（非阻塞）
        cmd = "evolve_return_params_lite" if lite else "evolve_return_params"
        for worker_id, fitnesses in fitness_map.items():
            if worker_id in self.cmd_queues:
                self.cmd_queues[worker_id].put((cmd, fitnesses))

        # 2. 收集所有结果
        results: dict[
            tuple["AgentType", int],
            tuple[tuple[np.ndarray, ...] | None, tuple[np.ndarray, ...], tuple[np.ndarray, np.ndarray]]
        ] = {}
        expected_count = len(fitness_map)

        t_collect_start = time.perf_counter()
        t_first_result: float | None = None
        t_last_result: float = 0.0

        for _ in range(expected_count):
            worker_id, result = self.result_queue.get()
            t_now = time.perf_counter()
            if t_first_result is None:
                t_first_result = t_now
            t_last_result = t_now

            if result[0] in ("success_params", "success_params_lite"):
                # result = ("success_params[_lite]", genome_data|None, network_params_data, species_data, timing)
                genome_data = result[1]  # None for lite
                network_params_data = result[2]
                species_data = result[3] if len(result) > 3 else (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
                results[worker_id] = (genome_data, network_params_data, species_data)

                # Log worker timing data
                if len(result) > 4 and result[4] is not None:
                    worker_timing: dict[str, float] = result[4]
                    timing_parts = [f"{k}={v:.3f}s" for k, v in worker_timing.items()]
                    logger.debug(
                        f"Worker {worker_id} timing: {', '.join(timing_parts)}"
                    )
            elif result[0] == "error":
                raise RuntimeError(f"Worker {worker_id} evolve failed: {result[1]}")
            else:
                raise RuntimeError(f"Worker {worker_id} unexpected result: {result}")

        # Log collection timing
        t_total = time.perf_counter() - t_collect_start
        t_spread = t_last_result - (t_first_result or t_last_result)
        logger.debug(
            f"evolve_all_parallel collect: total={t_total:.3f}s, "
            f"first_to_last_spread={t_spread:.3f}s, workers={expected_count}"
        )

        return results

    def sync_genomes_from_workers(
        self,
    ) -> dict[tuple["AgentType", int], tuple[np.ndarray, ...]]:
        """从所有 Worker 同步基因组数据（用于 checkpoint 保存）

        向每个 Worker 发送 sync_genomes 命令并收集序列化后的基因组数据。

        Returns:
            字典，key 为 (agent_type, sub_pop_id)，value 为序列化的基因组数据元组
        """
        for worker_id, cmd_queue in self.cmd_queues.items():
            cmd_queue.put(("sync_genomes", None))

        results: dict[tuple["AgentType", int], tuple[np.ndarray, ...]] = {}
        for _ in range(len(self.cmd_queues)):
            worker_id, result = self.result_queue.get()
            if result[0] == "genomes":
                results[worker_id] = result[1]
            else:
                raise RuntimeError(f"Worker {worker_id} sync_genomes failed: {result}")

        return results

    def set_genomes(
        self,
        genomes_map: dict[tuple["AgentType", int], tuple[np.ndarray, ...]]
    ) -> None:
        """同步基因组到所有 Worker

        Args:
            genomes_map: 字典，key 为 (agent_type, sub_pop_id)，
                        value 为基因组数据元组
        """
        # 发送 set_genomes 命令
        for worker_id, genome_data in genomes_map.items():
            if worker_id in self.cmd_queues:
                self.cmd_queues[worker_id].put(("set_genomes", genome_data))

        # 等待所有 Worker 完成
        for _ in range(len(genomes_map)):
            worker_id, result = self.result_queue.get()
            if result != "ok":
                raise RuntimeError(f"Worker {worker_id} set_genomes failed: {result}")

    def shutdown(self) -> None:
        """关闭所有 Worker"""
        for worker_id, cmd_queue in self.cmd_queues.items():
            try:
                cmd_queue.put(("shutdown", None))
            except Exception:
                pass

        for p in self.workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()


def _cleanup_worker_neat_history(neat_pop: neat.Population) -> None:
    """清理 Worker 进程中 NEAT 种群的历史数据

    这是内存泄漏的关键修复点！Worker 进程持久运行，每次进化后如果不清理：
    - ancestors 字典会无限增长（每代 +75 条目）
    - fitness_history 列表会无限增长（每代 +27 条目）
    - best_genome 会持有对旧基因组的引用

    Args:
        neat_pop: NEAT 种群对象
    """
    current_genome_ids = set(neat_pop.population.keys())

    # 1. 清理 best_genome 引用
    if hasattr(neat_pop, 'best_genome') and neat_pop.best_genome is not None:
        if neat_pop.best_genome.key not in current_genome_ids:
            # 用当前种群中最优的替代
            best_in_current = max(
                neat_pop.population.values(),
                key=lambda g: g.fitness if g.fitness is not None else float('-inf')
            )
            neat_pop.best_genome = best_in_current

    # 2. 清理 species_set 中的历史数据
    if hasattr(neat_pop, 'species') and neat_pop.species is not None:
        species_set = neat_pop.species

        # 清理 genome_to_species 映射
        if hasattr(species_set, 'genome_to_species'):
            species_set.genome_to_species = {
                gid: sid
                for gid, sid in species_set.genome_to_species.items()
                if gid in current_genome_ids
            }

        # 清理每个物种的 fitness_history（限制长度为 5）
        # 【关键修复】同时删除空物种，防止物种数量无限增长
        if hasattr(species_set, 'species'):
            empty_species_ids: list[int] = []
            for sid, species in list(species_set.species.items()):
                if hasattr(species, 'fitness_history') and len(species.fitness_history) > 5:
                    species.fitness_history = species.fitness_history[-5:]
                # 【内存泄漏修复】清理 members 字典
                # 先保存需要保留的 genome，然后清空旧字典，再赋值新字典
                if hasattr(species, 'members') and species.members:
                    new_members = {
                        gid: genome
                        for gid, genome in species.members.items()
                        if gid in current_genome_ids
                    }
                    species.members.clear()  # 显式清空旧字典
                    species.members = new_members  # 赋值新字典
                # 【关键修复】清理 representative（如果不在当前种群中）
                # 这是 Worker 版本遗漏的重要清理点！
                if hasattr(species, 'representative') and species.representative is not None:
                    if species.representative.key not in current_genome_ids:
                        # 用当前成员中的第一个作为代表
                        if species.members:
                            species.representative = next(iter(species.members.values()))
                        else:
                            species.representative = None
                # 【关键修复】标记空物种以便删除
                if not species.members:
                    empty_species_ids.append(sid)

            # 【关键修复】删除空物种，防止物种数量无限增长导致内存泄漏
            for sid in empty_species_ids:
                old_species = species_set.species[sid]
                # 清理内部引用
                if hasattr(old_species, 'fitness_history'):
                    old_species.fitness_history = []
                old_species.representative = None
                del species_set.species[sid]

    # 3. 清理 reproduction.ancestors（关键！这是最大的泄漏来源）
    if hasattr(neat_pop, 'reproduction') and neat_pop.reproduction is not None:
        reproduction = neat_pop.reproduction
        if hasattr(reproduction, 'ancestors'):
            reproduction.ancestors = {}  # 完全清空

    # 4. 清理 stagnation 中的历史数据（关键！每代都会累积）
    if hasattr(neat_pop, 'stagnation') and neat_pop.stagnation is not None:
        stagnation = neat_pop.stagnation
        # 清理 species_fitness 历史
        if hasattr(stagnation, 'species_fitness'):
            stagnation.species_fitness = {}

    # 5. 清理 reporters 中可能积累的数据
    if hasattr(neat_pop, 'reporters') and neat_pop.reporters is not None:
        reporters = neat_pop.reporters
        if hasattr(reporters, 'reporters'):
            for reporter in reporters.reporters:
                # StdOutReporter 等可能有 generation_statistics
                if hasattr(reporter, 'generation_statistics'):
                    if len(reporter.generation_statistics) > 5:
                        reporter.generation_statistics = reporter.generation_statistics[-5:]
                # 清理 species_statistics
                if hasattr(reporter, 'species_statistics'):
                    if len(reporter.species_statistics) > 5:
                        reporter.species_statistics = reporter.species_statistics[-5:]
                # 清理 most_fit_genomes
                if hasattr(reporter, 'most_fit_genomes'):
                    if len(reporter.most_fit_genomes) > 5:
                        reporter.most_fit_genomes = reporter.most_fit_genomes[-5:]

    # 6. 强制 GC 回收并释放内存给操作系统
    import gc
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    malloc_trim()


def _multi_worker_process_main(
    worker_id: tuple["AgentType", int],
    neat_config_path: str,
    pop_size: int,
    cmd_queue: "multiprocessing.Queue[tuple[str, Any]]",
    result_queue: "multiprocessing.Queue[tuple[tuple[AgentType, int], Any]]",
) -> None:
    """多种群 Worker 进程主函数

    与 _worker_process_main 类似，但使用 (agent_type, sub_pop_id) 作为 worker_id。

    Args:
        worker_id: Worker ID，(agent_type, sub_pop_id) 元组
        neat_config_path: NEAT 配置文件路径
        pop_size: 种群大小
        cmd_queue: 命令队列 (command, args)
        result_queue: 结果队列 (worker_id, result)
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # 忽略 Ctrl+C

    # 1. 初始化 NEAT 配置和种群
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )
    neat_config.pop_size = pop_size
    neat_pop = neat.Population(neat_config)
    generation = 0

    # 2. 主循环：等待命令
    while True:
        try:
            cmd, args = cmd_queue.get()
        except Exception:
            break

        if cmd == "shutdown":
            break

        elif cmd == "evolve":
            # args: np.ndarray of fitnesses (shape: pop_size,)
            fitnesses = args

            # 更新适应度
            for i, (gid, genome) in enumerate(neat_pop.population.items()):
                if i < len(fitnesses):
                    genome.fitness = float(fitnesses[i])

            # 执行进化
            def eval_genomes(genomes: list, config: Any) -> None:
                pass  # 适应度已设置

            try:
                neat_pop.run(eval_genomes, n=1)
                generation += 1
            except Exception as e:
                result_queue.put((worker_id, ("error", str(e))))
                continue

            # 【关键修复】清理 NEAT 历史数据，防止内存泄漏
            _cleanup_worker_neat_history(neat_pop)

            # 返回新基因组数据（NumPy 格式）
            new_data = _serialize_genomes_numpy(neat_pop.population)
            result_queue.put((worker_id, ("success", new_data)))
            # 【内存泄漏修复】发送后删除大型数据
            del new_data
            del fitnesses  # 也删除输入的适应度数组
            # 显式触发年轻代 GC
            gc.collect(0)

        elif cmd == "get_genomes":
            # 返回当前基因组数据
            data = _serialize_genomes_numpy(neat_pop.population)
            result_queue.put((worker_id, data))
            # 【内存泄漏修复】发送后删除大型数据
            del data

        elif cmd == "set_genomes":
            # 设置基因组数据
            keys, fitnesses_in, metadata, nodes, conns = args

            # 【内存优化】在替换种群之前，先清理旧数据
            # 帮助 GC 更快回收旧的 genome 对象
            if neat_pop.population:
                for genome in neat_pop.population.values():
                    if hasattr(genome, 'nodes'):
                        genome.nodes.clear()
                    if hasattr(genome, 'connections'):
                        genome.connections.clear()
                neat_pop.population.clear()

            neat_pop.population = _deserialize_genomes_numpy(
                keys, fitnesses_in, metadata, nodes, conns, neat_config.genome_config
            )

            # 【关键修复】同步 pop_size 为实际种群大小
            # 防止 checkpoint 中的种群大小与配置不一致导致进化失败
            actual_pop_size = len(neat_pop.population)
            if neat_config.pop_size != actual_pop_size:
                neat_config.pop_size = actual_pop_size

            neat_pop.species.speciate(neat_config, neat_pop.population, generation)

            # 【关键修复】同步基因组后立即清理历史数据
            # checkpoint 中可能包含大量历史数据（ancestors、fitness_history 等）
            # 如果不清理，会导致 Worker 进程内存暴涨（20GB+）
            _cleanup_worker_neat_history(neat_pop)

            # 【内存优化】清理命令参数
            del keys, fitnesses_in, metadata, nodes, conns
            gc.collect(0)

            result_queue.put((worker_id, "ok"))

        elif cmd == "evolve_return_params":
            # args: np.ndarray of fitnesses (shape: pop_size,)
            # 返回 genome 数据 + 网络参数 + species 数据，减少主进程重建网络的开销
            fitnesses = args
            timing: dict[str, float] = {}

            # 更新适应度
            for i, (gid, genome) in enumerate(neat_pop.population.items()):
                if i < len(fitnesses):
                    genome.fitness = float(fitnesses[i])

            # 执行进化
            def eval_genomes_params(genomes: list, config: Any) -> None:
                pass  # 适应度已设置

            t0 = time.perf_counter()
            try:
                neat_pop.run(eval_genomes_params, n=1)
                generation += 1
            except Exception as e:
                result_queue.put((worker_id, ("error", str(e))))
                continue
            timing['neat_run'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            # 【关键修复】清理 NEAT 历史数据，防止内存泄漏
            _cleanup_worker_neat_history(neat_pop)
            timing['cleanup'] = time.perf_counter() - t0

            # 1. 序列化基因组数据（NumPy 格式）
            t0 = time.perf_counter()
            genome_data = _serialize_genomes_numpy(neat_pop.population)
            timing['serialize_genomes'] = time.perf_counter() - t0

            # 2. 提取所有网络参数并直接打包（合并 extract + pack）
            t0 = time.perf_counter()
            network_params_data = _extract_and_pack_all_network_params(
                neat_pop.population, neat_config
            )
            timing['extract_pack_params'] = time.perf_counter() - t0

            # 3. 序列化 species 数据（genome_id -> species_id 映射）
            t0 = time.perf_counter()
            species_data = _serialize_species_data(neat_pop.species)
            timing['serialize_species'] = time.perf_counter() - t0

            # 4. 返回结果（含 timing）
            t0 = time.perf_counter()
            result_queue.put((worker_id, ("success_params", genome_data, network_params_data, species_data, timing)))
            timing['queue_put'] = time.perf_counter() - t0

            # 【内存泄漏修复】发送后立即删除大型数据，避免占用内存直到下一次循环
            del genome_data
            del network_params_data
            del species_data
            del fitnesses
            # 显式触发年轻代 GC
            gc.collect(0)

        elif cmd == "evolve_return_params_lite":
            # 轻量版：跳过基因组序列化，仅返回网络参数 + species 数据
            fitnesses = args
            timing_lite: dict[str, float] = {}

            # 更新适应度
            for i, (gid, genome) in enumerate(neat_pop.population.items()):
                if i < len(fitnesses):
                    genome.fitness = float(fitnesses[i])

            # 执行进化
            def eval_genomes_lite(genomes: list, config: Any) -> None:
                pass  # 适应度已设置

            t0 = time.perf_counter()
            try:
                neat_pop.run(eval_genomes_lite, n=1)
                generation += 1
            except Exception as e:
                result_queue.put((worker_id, ("error", str(e))))
                continue
            timing_lite['neat_run'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            _cleanup_worker_neat_history(neat_pop)
            timing_lite['cleanup'] = time.perf_counter() - t0

            # Skip genome serialization!

            # 提取所有网络参数并直接打包
            t0 = time.perf_counter()
            network_params_data = _extract_and_pack_all_network_params(
                neat_pop.population, neat_config
            )
            timing_lite['extract_pack_params'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            species_data = _serialize_species_data(neat_pop.species)
            timing_lite['serialize_species'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            result_queue.put((worker_id, ("success_params_lite", None, network_params_data, species_data, timing_lite)))
            timing_lite['queue_put'] = time.perf_counter() - t0

            # 【内存泄漏修复】发送后立即删除大型数据
            del network_params_data
            del species_data
            del fitnesses
            gc.collect(0)

        elif cmd == "sync_genomes":
            # 主进程请求基因组数据（用于 checkpoint 保存）
            genome_data = _serialize_genomes_numpy(neat_pop.population)
            result_queue.put((worker_id, ("genomes", genome_data)))
            del genome_data
            gc.collect(0)


def malloc_trim() -> None:
    """调用 glibc 的 malloc_trim 将释放的内存归还给操作系统

    Python 的内存分配器通常不会将释放的内存归还给 OS，而是保留在内部
    供将来使用。这会导致 VmRSS 持续增长。通过调用 malloc_trim(0)，
    可以强制将未使用的内存页归还给操作系统。

    仅在 Linux 系统上有效。
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        # 非 Linux 系统或无法加载 libc，静默跳过
        pass


def _evolve_single_population(pop: "Population", current_price: float) -> None:
    """进化单个种群

    Args:
        pop: 种群对象
        current_price: 当前价格
    """
    pop.evolve(current_price)


from src.bio.agents.market_maker import MarketMakerAgent
from src.bio.agents.retail_pro import RetailProAgent
from src.bio.brain.brain import Brain
from src.config.config import ASConfig, AgentConfig, AgentType, Config, TrainingConfig
from src.core.log_engine.logger import get_logger


def _get_memory_mb() -> float:
    """获取当前进程的内存使用量（MB）"""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:
        pass
    return 0.0


class Population:
    """种群管理类

    管理特定类型的 Agent 种群，包括创建、评估、淘汰和繁殖。

    Attributes:
        agent_type: Agent 类型（高级散户/做市商）
        agents: Agent 列表
        neat_pop: NEAT 种群对象
        neat_config: NEAT 配置
        neat_config_path: NEAT 配置文件路径（用于并行进化时在子进程中重新加载）
        agent_config: Agent 配置
        generation: 当前代数
        logger: 日志器
    """

    agent_type: AgentType
    agents: list[Agent]
    neat_pop: neat.Population
    neat_config: neat.Config
    neat_config_path: str  # NEAT 配置文件路径
    agent_config: AgentConfig
    _training_config: TrainingConfig  # 训练配置（用于做市商复合适应度权重）
    generation: int
    logger: logging.Logger
    _executor: ThreadPoolExecutor | None
    _num_workers: int
    sub_population_id: int | None  # 子种群ID（子种群管理器使用）
    _pending_genome_data: tuple[np.ndarray, ...] | None  # 待反序列化的基因组数据
    _genomes_dirty: bool  # 标记基因组是否需要同步
    _pending_species_data: tuple[np.ndarray, np.ndarray] | None  # 待恢复的species数据
    _cached_network_params_data: tuple[np.ndarray, ...] | None  # 缓存的网络参数（供 update_networks_from_numpy 使用）

    def _get_executor(self) -> ThreadPoolExecutor:
        """获取实例级别线程池"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._num_workers,
                thread_name_prefix=f"pop_{self.agent_type.value}"
            )
        return self._executor

    def shutdown_executor(self) -> None:
        """关闭实例级别线程池"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __init__(self, agent_type: AgentType, config: Config, defer_agent_creation: bool = False) -> None:
        """创建种群

        初始化 NEAT 种群，创建初始 Agent 列表。

        Args:
            agent_type: Agent 类型（高级散户/做市商）
            config: 全局配置对象
            defer_agent_creation: 是否延迟创建 Agent（用于子种群管理器避免重复创建）
        """
        self.agent_type = agent_type
        self.agent_config = config.agents[agent_type]
        self._training_config: TrainingConfig = config.training  # 用于做市商复合适应度权重
        self._as_config: ASConfig = config.as_model  # AS 模型配置（做市商构造时传递）
        self.generation = 0
        self.logger = get_logger("population")
        self._executor = None
        self._num_workers = 8
        self.sub_population_id = None  # 默认不是子种群

        # 延迟反序列化支持
        self._pending_genome_data: tuple[np.ndarray, ...] | None = None
        self._genomes_dirty: bool = False
        self._pending_species_data: tuple[np.ndarray, np.ndarray] | None = None
        self._cached_network_params_data: tuple[np.ndarray, ...] | None = None

        # 根据 Agent 类型选择 NEAT 配置文件
        from pathlib import Path

        config_dir = Path(config.training.neat_config_path)
        if agent_type == AgentType.MARKET_MAKER:
            neat_config_path = config_dir / "neat_market_maker.cfg"
        elif agent_type == AgentType.RETAIL_PRO:
            neat_config_path = config_dir / "neat_retail_pro.cfg"
        else:
            raise ValueError(f"未知的 Agent 类型: {agent_type}")

        # 保存 NEAT 配置文件路径（用于并行进化时在子进程中重新加载）
        self.neat_config_path = str(neat_config_path)

        # 加载 NEAT 配置
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.neat_config_path,
        )

        # 动态设置 pop_size 为 AgentConfig.count
        self.neat_config.pop_size = self.agent_config.count

        # 创建 NEAT 种群
        self.neat_pop = neat.Population(self.neat_config)

        # 获取初始基因组并创建 Agent
        if defer_agent_creation:
            self.agents = []
        else:
            genomes = list(self.neat_pop.population.items())
            self.agents = self.create_agents(genomes)
            self.logger.info(
                f"创建 {agent_type.value} 种群，初始 Agent 数量: {len(self.agents)}"
            )

        # 适应度累积存储
        self._accumulated_fitness: dict[int, float] = {}  # genome_id -> 累积适应度
        self._accumulation_count: int = 0  # 累积的 episode 数量

    # Agent ID 偏移量，确保不同种群的 agent_id 不冲突
    # ID分配方案：
    # RETAIL_PRO_SUB_0:  0 ~ 99,999 (每个子种群预留100K空间)
    # RETAIL_PRO_SUB_1: 100,000 ~ 199,999
    # ...
    # RETAIL_PRO_SUB_11: 1,100,000 ~ 1,199,999
    # MARKET_MAKER_SUB_0: 2,000,000 ~ 2,099,999
    # MARKET_MAKER_SUB_1: 2,100,000 ~ 2,199,999
    # ...
    _AGENT_ID_OFFSET = {
        AgentType.RETAIL_PRO: 0,  # 基础偏移，子种群会在此基础上再加偏移
        AgentType.MARKET_MAKER: 2_000_000,  # 基础偏移，子种群会在此基础上再加偏移
    }

    # 子种群偏移量（每个子种群的ID空间，适用于所有支持子种群的类型）
    _SUB_POPULATION_OFFSET = 100_000

    def _create_single_agent(
        self,
        idx: int,
        genome_id: int,
        genome: neat.DefaultGenome,
        agent_class: type[Agent],
        agent_id_offset: int,
    ) -> tuple[int, Agent]:
        """创建单个 Agent（线程安全）

        Args:
            idx: Agent 在列表中的索引
            genome_id: 基因组 ID
            genome: NEAT 基因组对象
            agent_class: Agent 类（RetailProAgent/MarketMakerAgent）
            agent_id_offset: Agent ID 偏移量

        Returns:
            (索引, Agent) 元组
        """
        brain = Brain.from_genome(genome, self.neat_config)
        unique_agent_id = agent_id_offset + idx
        if self.agent_type == AgentType.MARKET_MAKER:
            agent = agent_class(unique_agent_id, brain, self.agent_config, as_config=self._as_config)
        else:
            agent = agent_class(unique_agent_id, brain, self.agent_config)
        return (idx, agent)

    def create_agents(
        self,
        genomes: list[tuple[int, neat.DefaultGenome]],
    ) -> list[Agent]:
        """从基因组创建 Agent 列表（并行化版本）

        遍历基因组列表，为每个基因组创建对应的 Brain 和 Agent。
        根据种群的 agent_type 创建对应类型的 Agent（高级散户/做市商）。
        小批量（<50）串行处理，大批量并行处理以提升性能。

        Args:
            genomes: NEAT 基因组列表，每项为 (genome_id, genome) 元组

        Returns:
            创建的 Agent 列表
        """
        # 确定 Agent 类
        if self.agent_type == AgentType.RETAIL_PRO:
            agent_class: type[Agent] = RetailProAgent
        elif self.agent_type == AgentType.MARKET_MAKER:
            agent_class = MarketMakerAgent
        else:
            raise ValueError(f"未知的 Agent 类型: {self.agent_type}")

        agent_id_offset = self._AGENT_ID_OFFSET.get(self.agent_type, 0)

        # 如果是子种群，添加子种群偏移
        if self.sub_population_id is not None:
            agent_id_offset += self.sub_population_id * self._SUB_POPULATION_OFFSET

        # 小批量直接串行处理，避免线程池开销
        if len(genomes) < 50:
            agents: list[Agent] = []
            for idx, (genome_id, genome) in enumerate(genomes):
                brain = Brain.from_genome(genome, self.neat_config)
                unique_agent_id = agent_id_offset + idx
                if self.agent_type == AgentType.MARKET_MAKER:
                    agent = agent_class(unique_agent_id, brain, self.agent_config, as_config=self._as_config)
                else:
                    agent = agent_class(unique_agent_id, brain, self.agent_config)
                agents.append(agent)
            return agents

        # 大批量并行创建
        executor = self._get_executor()
        futures: dict[Future[tuple[int, Agent]], int] = {}

        for idx, (genome_id, genome) in enumerate(genomes):
            future = executor.submit(
                self._create_single_agent,
                idx,
                genome_id,
                genome,
                agent_class,
                agent_id_offset,
            )
            futures[future] = idx

        # 收集结果，按索引排序（添加超时防止死锁）
        results: list[tuple[int, Agent]] = []
        timeout_seconds = 120.0  # 2分钟超时
        try:
            for future in as_completed(futures, timeout=timeout_seconds):
                try:
                    result = future.result(timeout=10.0)  # 单个结果也有超时
                    results.append(result)
                except TimeoutError:
                    self.logger.error(f"创建 Agent 获取结果超时")
                except Exception as e:
                    raise RuntimeError(f"创建 Agent 失败: {e}") from e
        except TimeoutError:
            # 整体超时
            self.logger.error(
                f"创建 {self.agent_type.value} Agent 整体超时 ({timeout_seconds}s)，"
                f"已完成 {len(results)}/{len(genomes)}，可能存在死锁"
            )
            # 取消未完成的任务
            for future in futures:
                if not future.done():
                    future.cancel()
            raise RuntimeError(f"创建 Agent 超时，可能存在死锁")

        results.sort(key=lambda x: x[0])
        return [agent for _, agent in results]

    def evaluate(
        self,
        current_price: float,
    ) -> list[tuple[Agent, float]]:
        """评估种群适应度

        使用向量化运算计算所有 Agent 的适应度，并按适应度从高到低排序。
        非做市商：统一使用实际收益率 (equity - initial) / initial
        做市商：使用四组件复合适应度（PnL + 盘口价差 + 成交量 + 存活）

        Args:
            current_price: 当前市场价格，用于计算未实现盈亏

        Returns:
            按适应度从高到低排序的 (Agent, 适应度) 元组列表
        """
        n = len(self.agents)
        if n == 0:
            return []

        if self.agent_type == AgentType.MARKET_MAKER:
            return self._evaluate_market_maker(current_price, n)

        # 非做市商：已实现 PnL + 对称持仓成本 + 活跃度激励
        # 1. 收集所有 Agent 的账户数据到 numpy 数组
        balances = np.array([a.account.balance for a in self.agents])
        quantities = np.array([a.account.position.quantity for a in self.agents])
        initial_balances = np.array([a.account.initial_balance for a in self.agents])

        # 2. 纯已实现 PnL 收益率: (balance - initial) / initial
        fitnesses = (balances - initial_balances) / initial_balances

        # 3. 对称持仓成本: λ × |position_qty × current_price| / initial
        if self._training_config.position_cost_weight > 0:
            position_values = np.abs(quantities * current_price)
            fitnesses -= self._training_config.position_cost_weight * position_values / initial_balances

        # 4. 活跃度激励: β × activity_score
        beta = self._training_config.retail_fitness_activity_weight
        if beta > 0:
            trade_counts = np.array(
                [a.account.trade_count for a in self.agents], dtype=np.float64
            )
            max_tc = np.max(trade_counts) if n > 0 else 0.0
            activity_scores = trade_counts / (max_tc + 1.0)
            fitnesses = (1.0 - beta) * fitnesses + beta * activity_scores

        # 5. 获取从高到低的排序索引
        sorted_indices = np.argsort(fitnesses)[::-1]

        # 6. 按排序索引构建结果
        return [(self.agents[i], float(fitnesses[i])) for i in sorted_indices]

    def _evaluate_market_maker(
        self,
        current_price: float,
        n: int,
    ) -> list[tuple[Agent, float]]:
        """做市商复合适应度评估

        双组件加权适应度：
        - PnL 收益率：(equity - initial) / initial
        - Maker 成交量：归一化后的 maker_volume

        mm_fitness = α × pnl + γ × volume_score

        Args:
            current_price: 当前市场价格
            n: Agent 数量

        Returns:
            按适应度从高到低排序的 (Agent, 适应度) 元组列表
        """
        pnl_arr = np.zeros(n, dtype=np.float64)
        volume_arr = np.zeros(n, dtype=np.float64)

        for idx, agent in enumerate(self.agents):
            # PnL 收益率（纯已实现 PnL + 对称持仓成本）
            initial = agent.account.initial_balance
            if initial > 0:
                pnl_arr[idx] = (agent.account.balance - initial) / initial
                if self._training_config.mm_position_cost_weight > 0:
                    pos_value = abs(agent.account.position.quantity * current_price)
                    pnl_arr[idx] -= self._training_config.mm_position_cost_weight * pos_value / initial

            # Maker 成交量（原始值，后续归一化）
            volume_arr[idx] = float(agent.account.maker_volume)

        # 归一化 maker_volume：除以 (max_volume + 1.0) 避免除零
        max_volume = np.max(volume_arr) if n > 0 else 0.0
        norm_volume = volume_arr / (max_volume + 1.0)

        # 加权求和: α × pnl + γ × volume
        w = self._training_config
        fitnesses = (
            w.mm_fitness_pnl_weight * pnl_arr
            + w.mm_fitness_volume_weight * norm_volume
        )

        # 获取从高到低的排序索引
        sorted_indices = np.argsort(fitnesses)[::-1]

        # 按排序索引构建结果
        return [(self.agents[i], float(fitnesses[i])) for i in sorted_indices]

    def evolve(self, current_price: float) -> None:
        """进化种群

        使用 NEAT 算法进行一代进化：
        1. 使用向量化 evaluate 计算所有 Agent 适应度
        2. 通过 agent -> genome 映射为基因组设置适应度
        3. 调用 NEAT 种群的 run 方法进行一代进化
        4. 清理 NEAT 历史数据
        5. 复用 Agent 对象，只更新 Brain（对象池化优化）

        当 NEAT 进化失败（如种群灭绝）时，自动重置种群并重新开始。

        对象池化优化：不再销毁旧 Agent 对象，而是复用它们，只更新 Brain。
        这样可以避免大量对象的创建和销毁开销，显著提升性能。

        Args:
            current_price: 当前市场价格，用于计算适应度
        """
        # [MEMORY] 记录进化开始时内存
        mem_start = _get_memory_mb()

        # 1. 使用向量化 evaluate 获取所有 Agent 的适应度
        agent_fitnesses = self.evaluate(current_price)

        # 2. 通过 agent -> genome 映射为基因组设置适应度
        for agent, fitness in agent_fitnesses:
            genome = agent.brain.get_genome()
            genome.fitness = fitness  # type: ignore[assignment]

        # 3. 保存旧基因组引用（用于后续清理其内部数据）
        mem_before_cleanup = _get_memory_mb()
        old_genomes = list(self.neat_pop.population.values())

        # 4. 调用 NEAT 进化（eval_genomes 留空，适应度已设置）
        def eval_genomes(
            _genomes: list[tuple[int, neat.DefaultGenome]], _config: neat.Config
        ) -> None:
            # 适应度已经在上面设置好了，这里不需要再计算
            pass

        try:
            self.neat_pop.run(eval_genomes, n=1)
        except (RuntimeError, CompleteExtinctionException) as e:
            # NEAT 进化失败（通常是因为种群灭绝或无法繁殖足够后代）
            # 重置种群并重新开始
            error_msg = str(e) if str(e) else type(e).__name__
            self.logger.warning(
                f"{self.agent_type.value} 种群进化失败: {error_msg}，正在重置种群..."
            )
            self.logger.debug(f"完整异常堆栈:\n{traceback.format_exc()}")
            # 清理旧基因组数据
            self._cleanup_genome_internals(old_genomes)
            del old_genomes
            gc.collect()
            gc.collect()
            malloc_trim()
            self._reset_neat_population()
            return
        except Exception as e:
            # 捕获其他未预期的异常，记录完整堆栈
            error_msg = str(e) if str(e) else type(e).__name__
            self.logger.error(
                f"{self.agent_type.value} 种群进化遇到未预期异常: {error_msg}"
            )
            self.logger.error(f"完整异常堆栈:\n{traceback.format_exc()}")
            # 清理旧基因组数据
            self._cleanup_genome_internals(old_genomes)
            del old_genomes
            gc.collect()
            gc.collect()
            malloc_trim()
            self._reset_neat_population()
            return

        # 5. 【关键】清理旧基因组的内部数据
        # NEAT 进化后，新基因组已经创建，旧基因组不再需要
        # 但旧基因组的 connections 和 nodes 字典仍占用大量内存
        # 注意：由于精英保留（elitism），某些旧基因组可能仍在新种群中
        # 只清理不在新种群中的旧基因组
        new_genome_ids = set(self.neat_pop.population.keys())
        old_genomes_to_clean = [
            g for g in old_genomes if g.key not in new_genome_ids
        ]
        self._cleanup_genome_internals(old_genomes_to_clean)
        del old_genomes
        del old_genomes_to_clean

        # [MEMORY] 记录 NEAT run 后内存
        mem_after_neat = _get_memory_mb()

        # 6. 增加代数计数
        self.generation += 1

        # 7. 【对象池化优化】复用 Agent 对象，只更新 Brain
        # 注：NEAT 历史数据清理已移至 save_checkpoint 时统一执行
        mem_before_update = _get_memory_mb()
        new_genomes = list(self.neat_pop.population.items())
        for idx, (genome_id, genome) in enumerate(new_genomes):
            self.agents[idx].update_brain(genome, self.neat_config)
        mem_after_update = _get_memory_mb()

        # [MEMORY] 输出详细的内存变化日志
        mem_end = _get_memory_mb()
        self.logger.info(
            f"[MEMORY_EVOLVE] {self.agent_type.value} gen_{self.generation}: "
            f"neat_run=+{mem_after_neat - mem_before_cleanup:.1f}MB, "
            f"brain_update=+{mem_after_update - mem_before_update:.1f}MB, "
            f"total={mem_end - mem_start:+.1f}MB"
        )

        self.logger.info(
            f"{self.agent_type.value} 种群完成第 {self.generation} 代进化，"
            f"Agent 数量: {len(self.agents)}"
        )

    def evolve_with_cached_fitness(self) -> bool:
        """使用缓存的适应度进行进化（不重新计算适应度）

        用于 tick 数不足时，基于历史适应度继续进化，打破死循环。

        与 evolve() 的区别：
        1. 不调用 evaluate() 重新计算适应度
        2. 使用基因组中已缓存的 fitness 值
        3. 如果所有 fitness 都是 None，返回 False 跳过进化

        对象池化优化：不再销毁旧 Agent 对象，而是复用它们，只更新 Brain。

        Returns:
            bool: 是否成功进化。如果所有基因组的 fitness 都是 None，返回 False。
        """
        # 1. 检查是否有可用的缓存适应度
        genomes_with_fitness = [
            g for g in self.neat_pop.population.values()
            if g.fitness is not None
        ]

        if not genomes_with_fitness:
            self.logger.warning(
                f"{self.agent_type.value} 没有缓存的适应度，无法进行缓存进化"
            )
            return False

        # 2. 对于 fitness 为 None 的基因组，设置最低适应度（确保被淘汰）
        min_fitness = min(g.fitness for g in genomes_with_fitness)
        for genome in self.neat_pop.population.values():
            if genome.fitness is None:
                genome.fitness = min_fitness - 1.0  # type: ignore[assignment]

        # 3. 保存旧基因组引用用于清理
        old_genomes = list(self.neat_pop.population.values())

        # 4. 调用 NEAT 进化（使用缓存的 fitness，不重新评估）
        def eval_genomes(
            _genomes: list[tuple[int, neat.DefaultGenome]], _config: neat.Config
        ) -> None:
            # 适应度已经在缓存中，这里不需要再计算
            pass

        try:
            self.neat_pop.run(eval_genomes, n=1)
        except (RuntimeError, CompleteExtinctionException) as e:
            error_msg = str(e) if str(e) else type(e).__name__
            self.logger.warning(
                f"{self.agent_type.value} 缓存进化失败: {error_msg}，正在重置种群..."
            )
            self._cleanup_genome_internals(old_genomes)
            del old_genomes
            gc.collect()
            gc.collect()
            malloc_trim()
            self._reset_neat_population()
            return False
        except Exception as e:
            error_msg = str(e) if str(e) else type(e).__name__
            self.logger.error(
                f"{self.agent_type.value} 缓存进化遇到未预期异常: {error_msg}"
            )
            self.logger.error(f"完整异常堆栈:\n{traceback.format_exc()}")
            self._cleanup_genome_internals(old_genomes)
            del old_genomes
            gc.collect()
            gc.collect()
            malloc_trim()
            self._reset_neat_population()
            return False

        # 5. 清理旧基因组的内部数据
        new_genome_ids = set(self.neat_pop.population.keys())
        old_genomes_to_clean = [
            g for g in old_genomes if g.key not in new_genome_ids
        ]
        self._cleanup_genome_internals(old_genomes_to_clean)
        del old_genomes
        del old_genomes_to_clean

        # 6. 增加代数计数
        self.generation += 1

        # 7. 【对象池化优化】复用 Agent 对象，只更新 Brain
        # 注：NEAT 历史数据清理已移至 save_checkpoint 时统一执行
        new_genomes = list(self.neat_pop.population.items())
        for idx, (genome_id, genome) in enumerate(new_genomes):
            self.agents[idx].update_brain(genome, self.neat_config)

        self.logger.info(
            f"{self.agent_type.value} 种群使用缓存适应度完成第 {self.generation} 代进化，"
            f"Agent 数量: {len(self.agents)}"
        )
        return True

    def _cleanup_genome_internals(self, genomes: list[neat.DefaultGenome]) -> None:
        """清理基因组内部数据结构

        NEAT 基因组包含 connections 和 nodes 字典，每个都存储大量的 Gene 对象。
        对于大种群（如 10000 个散户），这些数据结构占用大量内存。
        当基因组不再需要时，显式清理这些数据结构可以让 GC 更快回收内存。

        Args:
            genomes: 需要清理的基因组列表
        """
        for genome in genomes:
            # 【内存泄漏修复】清理 connections 字典并置为空字典
            # 不能设为 None，因为 NEAT 库其他地方可能会访问
            if hasattr(genome, 'connections') and genome.connections is not None:
                genome.connections.clear()
                genome.connections = {}  # 置为新的空字典，释放旧字典对象
            # 【内存泄漏修复】清理 nodes 字典并置为空字典
            if hasattr(genome, 'nodes') and genome.nodes is not None:
                genome.nodes.clear()
                genome.nodes = {}  # 置为新的空字典，释放旧字典对象
            # 清理 fitness
            genome.fitness = None  # type: ignore[assignment]

    def _cleanup_old_agents(self) -> None:
        """清理旧 Agent 对象

        显式打破循环引用，帮助垃圾回收器及时回收内存。
        必须在 neat_pop.run() 之前调用，以确保旧 genome 对象可以被立即回收。

        优化版本：将原来的多次循环合并为单次循环，减少遍历开销。
        对于大种群（如10000个散户），可以显著减少清理时间。

        清理策略：
        1. 单次遍历所有 Agent，同时清理 Network/Brain/Account
        2. 清空 agents 列表
        """
        for agent in self.agents:
            # 清理 Brain 及其内部 Network
            if hasattr(agent, 'brain') and agent.brain is not None:
                brain = agent.brain
                # 清理 Network 内部状态（先清理最内层）
                if hasattr(brain, 'network') and brain.network is not None:
                    network = brain.network
                    if hasattr(network, 'node_evals'):
                        network.node_evals = None  # type: ignore[assignment]
                    if hasattr(network, 'values'):
                        network.values = None  # type: ignore[assignment]
                    if hasattr(network, 'input_nodes'):
                        network.input_nodes = None  # type: ignore[assignment]
                    if hasattr(network, 'output_nodes'):
                        network.output_nodes = None  # type: ignore[assignment]
                    # Cython FastFeedForwardNetwork 可能有额外属性
                    if hasattr(network, 'active'):
                        network.active = None  # type: ignore[assignment]
                # 清理 Brain 引用
                brain.genome = None  # type: ignore[assignment]
                brain.network = None  # type: ignore[assignment]
                brain.config = None  # type: ignore[assignment]

            # 清理 Account 引用
            if hasattr(agent, 'account') and agent.account is not None:
                if hasattr(agent.account, 'position'):
                    agent.account.position = None  # type: ignore[assignment]

            # 断开 Agent 的引用
            agent.brain = None  # type: ignore[assignment]
            agent.account = None  # type: ignore[assignment]
            # 清理 Agent 可能持有的其他引用
            if hasattr(agent, '_input_buffer'):
                agent._input_buffer = None  # type: ignore[assignment]

        # 清空列表
        self.agents.clear()

        # 注意：这里不调用 gc.collect()，让调用方决定何时 GC
        # 这样可以批量处理多个种群后再统一 GC，提高效率

    def _cleanup_neat_history_light(self) -> None:
        """轻量级 NEAT 历史清理（每代调用）

        只清理 3 个关键数据结构，避免每代都执行完整清理的开销。
        """
        current_genome_ids = set(self.neat_pop.population.keys())

        if hasattr(self.neat_pop, 'species') and self.neat_pop.species is not None:
            species_set = self.neat_pop.species
            if hasattr(species_set, 'genome_to_species'):
                species_set.genome_to_species = {
                    gid: sid for gid, sid in species_set.genome_to_species.items()
                    if gid in current_genome_ids
                }

        if hasattr(self.neat_pop, 'stagnation') and self.neat_pop.stagnation is not None:
            if hasattr(self.neat_pop.stagnation, 'species_fitness'):
                self.neat_pop.stagnation.species_fitness = {}

        if hasattr(self.neat_pop, 'reproduction') and self.neat_pop.reproduction is not None:
            if hasattr(self.neat_pop.reproduction, 'ancestors'):
                self.neat_pop.reproduction.ancestors = {}

    def _reset_neat_population(self) -> None:
        """重置 NEAT 种群

        当进化失败时调用，创建一个全新的随机种群。
        """
        # 先清理旧的 Agent 对象
        self._cleanup_old_agents()

        self.neat_pop = neat.Population(self.neat_config)
        genomes = list(self.neat_pop.population.items())
        self.agents = self.create_agents(genomes)
        # 重置后代数不变，表示这是一次意外重置
        self.logger.info(
            f"{self.agent_type.value} 种群已重置，新 Agent 数量: {len(self.agents)}"
        )

    def reset_agents(self) -> None:
        """重置所有 Agent 的账户状态

        在 episode 开始时调用，将所有 Agent 的账户恢复到初始状态。
        """
        for agent in self.agents:
            agent.reset(self.agent_config)

    def get_all_genomes(self) -> list[neat.DefaultGenome]:
        """获取所有基因组（用于GenerationSaver等）

        Returns:
            所有基因组列表
        """
        return list(self.neat_pop.population.values())

    def replace_worst_agents(
        self,
        new_genomes: list[neat.DefaultGenome],
    ) -> list[tuple[int, Agent, Agent]]:
        """增量替换最差的 Agent（不重建整个种群）

        优化点：只创建/替换需要的 Agent，避免重建整个种群。

        Args:
            new_genomes: 要注入的新 genome 列表（已设置 fitness 和 key）

        Returns:
            被替换的 (索引, 旧Agent, 新Agent) 列表
        """
        if not new_genomes:
            return []

        n_to_replace = len(new_genomes)
        neat_pop = self.neat_pop

        # 1. 按 genome.fitness 找到当前种群中最差的 N 个 agent
        # 构建 (agent_idx, genome_id, fitness) 列表
        agent_genome_fitness: list[tuple[int, int, float]] = []
        for idx, agent in enumerate(self.agents):
            genome = agent.brain.get_genome()
            fitness = genome.fitness if genome.fitness is not None else float('-inf')
            agent_genome_fitness.append((idx, genome.key, fitness))

        # 按 fitness 升序排序（最差的在前面）
        agent_genome_fitness.sort(key=lambda x: x[2])

        # 取最差的 N 个
        worst_entries = agent_genome_fitness[:n_to_replace]

        # 2. 收集这些 agent 的索引和对应的 genome_id
        worst_indices: list[int] = [entry[0] for entry in worst_entries]
        worst_genome_ids: list[int] = [entry[1] for entry in worst_entries]

        # 3. 从 neat_pop.population 中删除这些旧 genome，添加新 genome
        # 先删除旧的
        for old_genome_id in worst_genome_ids:
            if old_genome_id in neat_pop.population:
                del neat_pop.population[old_genome_id]

        # 为新 genome 分配唯一 key 并添加到种群
        max_key = max(neat_pop.population.keys()) if neat_pop.population else 0
        for i, genome in enumerate(new_genomes):
            new_key = max_key + 1 + i
            genome.key = new_key
            neat_pop.population[new_key] = genome

        # 3.5. 【关键修复】更新 node_indexer，确保新生成的节点 ID 不会与迁入的 genome 冲突
        # 迁入的 genome 可能来自其他竞技场，包含比当前竞技场 node_indexer 更大的节点 ID
        # 如果不更新，后续的 mutate_add_node 可能会生成重复的节点 ID
        genome_config = neat_pop.config.genome_config
        max_node_id_in_new_genomes = 0
        for genome in new_genomes:
            if genome.nodes:
                local_max = max(genome.nodes.keys())
                if local_max > max_node_id_in_new_genomes:
                    max_node_id_in_new_genomes = local_max
        # 如果新 genome 的最大节点 ID 超过了当前 node_indexer 的下一个值，需要更新
        if max_node_id_in_new_genomes > 0:
            if genome_config.node_indexer is None:
                # node_indexer 还未初始化，直接设置为 max_node_id + 1
                genome_config.node_indexer = count(max_node_id_in_new_genomes + 1)
            else:
                # 需要检查当前 node_indexer 的下一个值
                # 由于 count 对象无法直接获取当前值，我们通过 peek 方式获取
                # 但这会消耗一个值，所以需要用新的 count 替换
                current_next = next(genome_config.node_indexer)
                # 取两者的最大值 + 1 作为新的起点
                new_start = max(current_next, max_node_id_in_new_genomes + 1)
                genome_config.node_indexer = count(new_start)

        # 4. 只为新 genome 创建新的 Agent 对象
        # 确定 Agent 类
        if self.agent_type == AgentType.RETAIL_PRO:
            agent_class: type[Agent] = RetailProAgent
        elif self.agent_type == AgentType.MARKET_MAKER:
            agent_class = MarketMakerAgent
        else:
            raise ValueError(f"未知的 Agent 类型: {self.agent_type}")

        agent_id_offset = self._AGENT_ID_OFFSET.get(self.agent_type, 0)

        # 构建替换信息列表
        replaced_info: list[tuple[int, Agent, Agent]] = []

        for i, genome in enumerate(new_genomes):
            idx = worst_indices[i]
            old_agent = self.agents[idx]

            # 5. 创建新的 Agent 对象
            _, new_agent = self._create_single_agent(
                idx,
                genome.key,
                genome,
                agent_class,
                agent_id_offset,
            )

            # 6. 清理被替换的旧 Agent 对象（参考 _cleanup_old_agents 的实现）
            if hasattr(old_agent, 'brain') and old_agent.brain is not None:
                brain = old_agent.brain
                # 清理 Network 内部状态
                if hasattr(brain, 'network') and brain.network is not None:
                    network = brain.network
                    if hasattr(network, 'node_evals'):
                        network.node_evals = None  # type: ignore[assignment]
                    if hasattr(network, 'values'):
                        network.values = None  # type: ignore[assignment]
                    if hasattr(network, 'input_nodes'):
                        network.input_nodes = None  # type: ignore[assignment]
                    if hasattr(network, 'output_nodes'):
                        network.output_nodes = None  # type: ignore[assignment]
                # 清理 Brain 引用
                brain.genome = None  # type: ignore[assignment]
                brain.network = None  # type: ignore[assignment]
                brain.config = None  # type: ignore[assignment]
                old_agent.brain = None  # type: ignore[assignment]
            if hasattr(old_agent, 'account') and old_agent.account is not None:
                if hasattr(old_agent.account, 'position'):
                    old_agent.account.position = None  # type: ignore[assignment]
                old_agent.account = None  # type: ignore[assignment]
            if hasattr(old_agent, '_input_buffer'):
                old_agent._input_buffer = None  # type: ignore[assignment]

            # 更新 self.agents[idx] 为新创建的 Agent
            self.agents[idx] = new_agent

            replaced_info.append((idx, old_agent, new_agent))

        # 7. 调用 speciate() 重新划分物种
        neat_pop.species.speciate(
            neat_pop.config,
            neat_pop.population,
            neat_pop.generation,
        )

        self.logger.info(
            f"{self.agent_type.value} 种群增量替换了 {n_to_replace} 个最差 Agent"
        )

        # 8. 返回被替换的信息列表
        return replaced_info

    def get_elite_species_avg_fitness(self) -> float | None:
        """获取最精英 species 的平均适应度

        遍历 NEAT 种群中的所有 species，计算每个 species 的平均适应度，
        返回平均适应度最高的那个 species 的值。

        Returns:
            最精英 species 的平均适应度，如果没有有效数据则返回 None
        """
        neat_pop = self.neat_pop
        if not hasattr(neat_pop, 'species') or neat_pop.species is None:
            return None

        species_set = neat_pop.species
        if not hasattr(species_set, 'species') or not species_set.species:
            return None

        best_avg: float | None = None
        for species in species_set.species.values():
            # 获取该 species 所有成员的适应度
            member_fitnesses = species.get_fitnesses()
            # 过滤掉 None 值
            valid_fitnesses = [f for f in member_fitnesses if f is not None]
            if valid_fitnesses:
                avg = sum(valid_fitnesses) / len(valid_fitnesses)
                if best_avg is None or avg > best_avg:
                    best_avg = avg

        return best_avg

    def sync_genomes_from_pending(self) -> None:
        """从待处理数据同步基因组（延迟反序列化）

        当需要保存检查点时调用此方法，将待处理的基因组数据
        反序列化到主进程的 NEAT 种群中。
        """
        if not self._genomes_dirty or self._pending_genome_data is None:
            return

        # 保存旧基因组用于清理
        old_genomes = list(self.neat_pop.population.values())

        # 反序列化基因组
        new_keys, new_fitnesses, new_metadata, new_nodes, new_conns = self._pending_genome_data
        self.neat_pop.population = _deserialize_genomes_numpy(
            new_keys, new_fitnesses, new_metadata, new_nodes, new_conns,
            self.neat_config.genome_config
        )

        # 清理旧基因组
        new_genome_ids = set(self.neat_pop.population.keys())
        old_to_clean = [g for g in old_genomes if g.key not in new_genome_ids]
        self._cleanup_genome_internals(old_to_clean)

        # 注：NEAT 历史数据清理已移至 save_checkpoint 时统一执行
        # 恢复 species 数据
        if self._pending_species_data is not None:
            species_genome_ids, species_species_ids = self._pending_species_data
            _apply_species_data_to_population(
                self.neat_pop, species_genome_ids, species_species_ids, self.generation
            )
            self._pending_species_data = None


        # 更新 Agent Brain 的 genome 引用
        new_genomes = list(self.neat_pop.population.items())
        for idx, (gid, genome) in enumerate(new_genomes):
            if idx < len(self.agents):
                self.agents[idx].brain.genome = genome

        self._pending_genome_data = None
        self._pending_species_data = None
        self._genomes_dirty = False

    def accumulate_fitness(
        self, current_price: float,
    ) -> None:
        """累积当前 episode 的适应度

        计算当前 episode 的适应度并累加到内部存储。

        Args:
            current_price: 当前市场价格，用于计算未实现盈亏
        """
        agent_fitnesses = self.evaluate(current_price)
        for agent, fitness in agent_fitnesses:
            genome = agent.brain.get_genome()
            gid = genome.key
            if gid in self._accumulated_fitness:
                self._accumulated_fitness[gid] += fitness
            else:
                self._accumulated_fitness[gid] = fitness
        self._accumulation_count += 1

    def apply_accumulated_fitness(self) -> None:
        """将累积的平均适应度应用到基因组

        在进化前调用，将累积的平均适应度设置到每个基因组。
        """
        if self._accumulation_count == 0:
            return

        for agent in self.agents:
            genome = agent.brain.get_genome()
            gid = genome.key
            if gid in self._accumulated_fitness:
                genome.fitness = self._accumulated_fitness[gid] / self._accumulation_count

    def clear_accumulated_fitness(self) -> None:
        """清空累积的适应度数据"""
        self._accumulated_fitness.clear()
        self._accumulation_count = 0


class SubPopulationManager:
    """通用子种群管理器

    将指定类型的种群拆分成多个子种群，支持并行进化。
    每个子种群独立进行NEAT进化，但在训练过程中共享市场环境。

    Attributes:
        sub_populations: 子种群列表
        sub_population_count: 子种群数量
        agents_per_sub: 每个子种群的Agent数量
        agent_type: Agent类型
        logger: 日志器
        _pending_genome_data: 待反序列化的基因组数据（延迟反序列化用）
        _genomes_dirty: 标记基因组是否需要同步
    """

    sub_populations: list[Population]
    sub_population_count: int
    agents_per_sub: int
    agent_type: AgentType
    logger: logging.Logger
    _pending_genome_data: list[tuple[np.ndarray, ...]] | None
    _genomes_dirty: bool
    _cached_network_params_data: tuple[np.ndarray, ...] | None

    def __init__(
        self,
        config: Config,
        agent_type: AgentType,
        sub_count: int = 10,
    ) -> None:
        """创建子种群管理器

        Args:
            config: 全局配置对象
            agent_type: Agent类型（RETAIL_PRO 或 MARKET_MAKER）
            sub_count: 子种群数量（默认10）
        """
        self.agent_type = agent_type
        self.sub_population_count = sub_count
        self.logger = get_logger(f"{agent_type.value}_sub_pop_manager")

        # 计算每个子种群的Agent数量
        total_count = config.agents[agent_type].count
        self.agents_per_sub = total_count // sub_count

        # 检查是否能整除
        if total_count % sub_count != 0:
            self.logger.warning(
                f"{agent_type.value} 总数 {total_count} 不能被子种群数 {sub_count} 整除，"
                f"每个子种群将有 {self.agents_per_sub} 个Agent"
            )

        # 创建子种群
        # Agent ID 偏移由 Population.create_agents 自动处理
        # 基于 agent_type 和 sub_population_id 计算
        self.sub_populations = []
        for i in range(sub_count):
            # 创建子种群专用配置
            sub_config = self._create_sub_config(config, i)

            # 创建Population，设置子种群ID（延迟创建Agent避免重复创建）
            pop = Population(agent_type, sub_config, defer_agent_creation=True)
            # 子种群 ID 用于 Agent ID 计算
            # 实际 ID 偏移 = base_id_offset + sub_population_id * _SUB_POPULATION_OFFSET
            pop.sub_population_id = i

            # 创建agents（使用正确的ID偏移）
            genomes = list(pop.neat_pop.population.items())
            pop.agents = pop.create_agents(genomes)

            self.sub_populations.append(pop)

        # 延迟反序列化相关
        self._pending_genome_data = None
        self._genomes_dirty = False
        self._cached_network_params_data: tuple[np.ndarray, ...] | None = None

        self.logger.info(
            f"创建 {agent_type.value} 子种群管理器: {sub_count} 个子种群，每个 {self.agents_per_sub} 个Agent"
        )

    def _create_sub_config(self, config: Config, sub_id: int) -> Config:
        """创建子种群专用配置

        Args:
            config: 原始配置
            sub_id: 子种群ID

        Returns:
            子种群专用配置（修改了对应类型的count）
        """
        from dataclasses import replace
        sub_agent_config = replace(config.agents[self.agent_type], count=self.agents_per_sub)
        new_agents = dict(config.agents)
        new_agents[self.agent_type] = sub_agent_config
        return replace(config, agents=new_agents)

    @property
    def agents(self) -> list[Agent]:
        """返回所有子种群的Agent（合并视图）"""
        all_agents: list[Agent] = []
        for pop in self.sub_populations:
            all_agents.extend(pop.agents)
        return all_agents

    @property
    def generation(self) -> int:
        """返回第一个子种群的代数（所有子种群应该同步）"""
        if self.sub_populations:
            return self.sub_populations[0].generation
        return 0

    @property
    def agent_config(self) -> AgentConfig:
        """返回Agent配置（从第一个子种群获取）"""
        if self.sub_populations:
            return self.sub_populations[0].agent_config
        raise RuntimeError("No sub-populations available")

    def reset_agents(self) -> None:
        """重置所有子种群的Agent"""
        for pop in self.sub_populations:
            pop.reset_agents()

    def evaluate(
        self, current_price: float,
    ) -> list[tuple[Agent, float]]:
        """评估所有Agent适应度

        委托给各子种群的 Population.evaluate()。
        非做市商使用纯收益率，做市商使用四组件复合适应度。

        Args:
            current_price: 当前价格

        Returns:
            (Agent, fitness) 元组列表
        """
        all_results: list[tuple[Agent, float]] = []
        for pop in self.sub_populations:
            results = pop.evaluate(current_price)
            all_results.extend(results)
        return all_results

    def evolve(self, current_price: float) -> None:
        """进化所有子种群（串行版本）

        Args:
            current_price: 当前价格
        """
        for pop in self.sub_populations:
            pop.evolve(current_price)

    def _evolve_single_sub_pop(
        self, pop: Population, current_price: float
    ) -> tuple[int, bool]:
        """进化单个子种群（在线程中调用）

        Args:
            pop: 子种群对象
            current_price: 当前价格

        Returns:
            (子种群ID, 是否成功) 元组
        """
        try:
            pop.evolve(current_price)
            return (pop.sub_population_id or 0, True)
        except Exception as e:
            self.logger.error(
                f"子种群 {pop.sub_population_id} 进化失败: {e}"
            )
            return (pop.sub_population_id or 0, False)

    def evolve_parallel_simple(
        self, current_price: float, max_workers: int = 10
    ) -> None:
        """简化的并行进化方法 - 使用线程池

        使用 ThreadPoolExecutor 并行进化所有子种群。
        相比 ProcessPoolExecutor，线程池避免了序列化开销，
        虽然受 GIL 限制，但 NEAT 内部有很多 C 扩展操作可以释放 GIL。

        实现流程:
        1. 先串行评估所有子种群的适应度（共用同一个价格）
        2. 为每个子种群的基因组设置适应度
        3. 使用线程池并行执行各子种群的 NEAT 进化
        4. 等待所有子种群完成

        Args:
            current_price: 当前价格（用于计算适应度）
            max_workers: 最大并行线程数（默认10，对应10个子种群）
        """
        import time
        start_time = time.perf_counter()

        # 1. 先评估所有子种群的适应度并设置到基因组
        for pop in self.sub_populations:
            agent_fitnesses = pop.evaluate(current_price)
            for agent, fitness in agent_fitnesses:
                genome = agent.brain.get_genome()
                genome.fitness = fitness

        eval_time = time.perf_counter() - start_time

        # 2. 使用线程池并行进化
        # 注意：虽然 Python 有 GIL，但 NEAT 的 C 扩展操作可以释放 GIL
        # 同时，IO 等待和对象创建开销也可以并行化
        evolve_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有进化任务
            futures: list[Future[tuple[int, bool]]] = []
            for pop in self.sub_populations:
                # 创建一个专门执行进化（不重新评估）的函数
                future = executor.submit(
                    self._evolve_sub_pop_without_eval, pop
                )
                futures.append(future)

            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    sub_id, success = future.result()
                    if not success:
                        self.logger.warning(f"子种群 {sub_id} 进化失败")
                except Exception as e:
                    self.logger.error(f"子种群进化异常: {e}")

        evolve_time = time.perf_counter() - evolve_start
        total_time = time.perf_counter() - start_time

        self.logger.info(
            f"{self.agent_type.value} 子种群简化并行进化完成，共 {len(self.sub_populations)} 个子种群，"
            f"耗时: eval={eval_time:.2f}s, evolve={evolve_time:.2f}s, "
            f"total={total_time:.2f}s"
        )

    def _evolve_sub_pop_without_eval(
        self, pop: Population
    ) -> tuple[int, bool]:
        """进化单个子种群（不重新评估适应度，在线程中调用）

        Args:
            pop: 子种群对象

        Returns:
            (子种群ID, 是否成功) 元组
        """
        try:
            # 保存旧基因组引用
            old_genomes = list(pop.neat_pop.population.values())

            # 调用 NEAT 进化（适应度已经设置好了）
            def eval_genomes(
                _genomes: list[tuple[int, neat.DefaultGenome]],
                _config: neat.Config
            ) -> None:
                pass  # 适应度已设置

            try:
                pop.neat_pop.run(eval_genomes, n=1)
            except (RuntimeError, CompleteExtinctionException) as e:
                self.logger.warning(
                    f"子种群 {pop.sub_population_id} NEAT 进化失败: {e}"
                )
                pop._cleanup_genome_internals(old_genomes)
                del old_genomes
                pop._reset_neat_population()
                return (pop.sub_population_id or 0, False)
            except Exception as e:
                self.logger.error(
                    f"子种群 {pop.sub_population_id} NEAT 进化异常: {e}"
                )
                pop._cleanup_genome_internals(old_genomes)
                del old_genomes
                pop._reset_neat_population()
                return (pop.sub_population_id or 0, False)

            # 清理旧基因组
            new_genome_ids = set(pop.neat_pop.population.keys())
            old_to_clean = [g for g in old_genomes if g.key not in new_genome_ids]
            pop._cleanup_genome_internals(old_to_clean)
            del old_genomes
            del old_to_clean

            # 增加代数
            pop.generation += 1

            # 注：NEAT 历史数据清理已移至 save_checkpoint 时统一执行

            # 更新 Agent Brain
            new_genomes = list(pop.neat_pop.population.items())
            for idx, (gid, genome) in enumerate(new_genomes):
                if idx < len(pop.agents):
                    pop.agents[idx].update_brain(genome, pop.neat_config)

            return (pop.sub_population_id or 0, True)
        except Exception as e:
            self.logger.error(
                f"子种群 {pop.sub_population_id} 进化失败: {e}"
            )
            return (pop.sub_population_id or 0, False)

    def evolve_parallel(self, current_price: float, max_workers: int = 10) -> None:
        """并行进化所有子种群（NumPy 格式）

        使用 NumPy 结构化数组序列化，显著减少序列化开销。

        实现原理：
        1. 在主进程中评估所有子种群的适应度
        2. 将基因组数据序列化为 NumPy 紧凑格式
        3. 使用 ProcessPoolExecutor 并行进化各子种群
        4. 子进程中重建 NEAT Population 并执行一代进化
        5. 将新基因组以 NumPy 格式返回主进程
        6. 主进程重建种群并更新 Agent Brain

        Args:
            current_price: 当前价格（用于计算适应度）
            max_workers: 最大并行进程数
        """
        import pickle
        import time
        start_time = time.perf_counter()

        # 1. 评估所有子种群的适应度
        for pop in self.sub_populations:
            agent_fitnesses = pop.evaluate(current_price)
            for agent, fitness in agent_fitnesses:
                genome = agent.brain.get_genome()
                genome.fitness = fitness

        eval_time = time.perf_counter() - start_time

        # 2. 准备 NumPy 格式进化参数
        evolve_args: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]] = []
        total_serialize_size = 0

        for pop in self.sub_populations:
            # 序列化基因组数据（NumPy 格式）
            keys, fitnesses, metadata, nodes, conns = _serialize_genomes_numpy(pop.neat_pop.population)

            # 统计序列化大小
            arg_tuple = (
                pop.neat_config_path,
                keys, fitnesses, metadata, nodes, conns,
                len(pop.agents),
                pop.generation,
            )
            total_serialize_size += len(pickle.dumps(arg_tuple))
            evolve_args.append(arg_tuple)

        serialize_time = time.perf_counter() - start_time - eval_time

        # 3. 并行进化
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_evolve_subpop_numpy, evolve_args))

        parallel_time = time.perf_counter() - start_time - eval_time - serialize_time

        # 4. 更新子种群
        for i, result in enumerate(results):
            pop = self.sub_populations[i]
            new_keys, new_fitnesses, new_metadata, new_nodes, new_conns = result

            # 保存旧基因组用于清理
            old_genomes = list(pop.neat_pop.population.values())

            # 重建种群（NumPy 格式）
            pop.neat_pop.population = _deserialize_genomes_numpy(
                new_keys, new_fitnesses, new_metadata, new_nodes, new_conns,
                pop.neat_config.genome_config
            )

            # 更新 generation
            pop.generation += 1

            # 清理旧基因组
            new_genome_ids = set(pop.neat_pop.population.keys())
            old_to_clean = [g for g in old_genomes if g.key not in new_genome_ids]
            pop._cleanup_genome_internals(old_to_clean)

            # 注：NEAT 历史数据清理已移至 save_checkpoint 时统一执行

            # 更新 Agent Brain
            new_genomes = list(pop.neat_pop.population.items())
            for idx, (gid, genome) in enumerate(new_genomes):
                if idx < len(pop.agents):
                    pop.agents[idx].update_brain(genome, pop.neat_config)

        update_time = time.perf_counter() - start_time - eval_time - serialize_time - parallel_time
        total_time = time.perf_counter() - start_time

        self.logger.info(
            f"{self.agent_type.value} 子种群并行进化完成，共 {len(self.sub_populations)} 个子种群，"
            f"序列化大小: {total_serialize_size / 1024 / 1024:.2f} MB，"
            f"耗时: eval={eval_time:.2f}s, serialize={serialize_time:.2f}s, "
            f"parallel={parallel_time:.2f}s, update={update_time:.2f}s, total={total_time:.2f}s"
        )

    def evolve_parallel_with_network_params(
        self,
        current_price: float,
        worker_pool: PersistentWorkerPool,
        sync_genomes: bool = False,
        deserialize_genomes: bool = False,
    ) -> None:
        """使用 PersistentWorkerPool 并行进化，直接传输网络参数

        相比 evolve_parallel()，此方法有以下优势：
        1. Worker 进程持久化，避免每次创建新进程的开销
        2. 只传输适应度数组（~40KB），不传输基因组数据（~25MB）
        3. 返回网络参数，主进程直接使用参数重建网络，跳过基因组解析
        4. 延迟反序列化：默认不反序列化基因组，只在需要时（检查点/迁移）反序列化

        Args:
            current_price: 当前价格（用于计算适应度）
            worker_pool: 持久 Worker 进程池
            sync_genomes: 是否在进化前同步基因组（首次调用时需设为 True）
            deserialize_genomes: 是否反序列化基因组（检查点保存时设为 True）
        """
        import time
        start_time = time.perf_counter()
        sync_time = 0.0

        # 0. 如果需要，先同步基因组到 Worker
        if sync_genomes:
            genomes_list: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            for pop in self.sub_populations:
                genome_data = _serialize_genomes_numpy(pop.neat_pop.population)
                genomes_list.append(genome_data)
            worker_pool.set_all_genomes(genomes_list)
            sync_time = time.perf_counter() - start_time
            self.logger.info(f"同步基因组到 Worker 完成，耗时: {sync_time:.2f}s")

        # 1. 评估所有子种群的适应度，构建适应度数组列表
        fitnesses_list: list[np.ndarray] = []

        for pop in self.sub_populations:
            # 先评估获取每个 agent 的适应度
            agent_fitnesses = pop.evaluate(current_price)
            fitness_dict = {agent.agent_id: fitness for agent, fitness in agent_fitnesses}

            # 构建适应度数组（按 agent 顺序）
            fitness_arr = np.empty(len(pop.agents), dtype=np.float32)
            for idx, agent in enumerate(pop.agents):
                fitness = fitness_dict.get(agent.agent_id, 0.0)
                fitness_arr[idx] = fitness
            fitnesses_list.append(fitness_arr)

        eval_time = time.perf_counter() - start_time - sync_time

        # 2. 并行进化，返回基因组数据和网络参数
        results = worker_pool.evolve_all_return_params(fitnesses_list)
        evolve_time = time.perf_counter() - start_time - sync_time - eval_time

        # 3. 更新子种群
        update_start = time.perf_counter()

        # 存储待反序列化的基因组数据
        pending_genome_data: list[tuple[np.ndarray, ...]] = []

        for i, (genome_data, network_params_data) in enumerate(results):
            pop = self.sub_populations[i]

            # 更新 generation
            pop.generation += 1

            if deserialize_genomes:
                # 完整反序列化：重建 NEAT 种群
                old_genomes = list(pop.neat_pop.population.values())

                new_keys, new_fitnesses, new_metadata, new_nodes, new_conns = genome_data
                pop.neat_pop.population = _deserialize_genomes_numpy(
                    new_keys, new_fitnesses, new_metadata, new_nodes, new_conns,
                    pop.neat_config.genome_config
                )

                # 清理旧基因组
                new_genome_ids = set(pop.neat_pop.population.keys())
                old_to_clean = [g for g in old_genomes if g.key not in new_genome_ids]
                pop._cleanup_genome_internals(old_to_clean)

                # 注：NEAT 历史数据清理已移至 save_checkpoint 时统一执行

                # 解包网络参数并更新 Brain（包含 genome 引用）
                params_list = _unpack_network_params_numpy(*network_params_data)
                new_genomes = list(pop.neat_pop.population.items())
                for idx, (gid, genome) in enumerate(new_genomes):
                    if idx < len(pop.agents) and idx < len(params_list):
                        pop.agents[idx].brain.update_from_network_params(
                            genome, params_list[idx]
                        )
            else:
                # 延迟反序列化：只更新网络，不更新基因组
                pending_genome_data.append(genome_data)

                # 解包网络参数并只更新网络（不更新 genome 引用）
                params_list = _unpack_network_params_numpy(*network_params_data)
                for idx, params in enumerate(params_list):
                    if idx < len(pop.agents):
                        pop.agents[idx].brain.update_network_only(params)

        # 存储待反序列化数据
        if not deserialize_genomes:
            self._pending_genome_data = pending_genome_data
            self._genomes_dirty = True
        else:
            self._pending_genome_data = None
            self._genomes_dirty = False

        update_time = time.perf_counter() - update_start
        total_time = time.perf_counter() - start_time

        mode = "完整" if deserialize_genomes else "延迟"
        self.logger.info(
            f"{self.agent_type.value} 子种群网络参数并行进化完成（{mode}反序列化），"
            f"共 {len(self.sub_populations)} 个子种群，"
            f"耗时: sync={sync_time:.2f}s, eval={eval_time:.2f}s, evolve={evolve_time:.2f}s, "
            f"update={update_time:.2f}s, total={total_time:.2f}s"
        )

    def sync_genomes_from_pending(self) -> None:
        """从待处理数据同步基因组（延迟反序列化）

        当需要保存检查点或迁移时调用此方法，将待处理的基因组数据
        反序列化到主进程的 NEAT 种群中。
        """
        if not self._genomes_dirty or self._pending_genome_data is None:
            return

        import time
        start_time = time.perf_counter()

        for i, genome_data in enumerate(self._pending_genome_data):
            pop = self.sub_populations[i]

            # 保存旧基因组用于清理
            old_genomes = list(pop.neat_pop.population.values())

            # 反序列化基因组
            new_keys, new_fitnesses, new_metadata, new_nodes, new_conns = genome_data
            pop.neat_pop.population = _deserialize_genomes_numpy(
                new_keys, new_fitnesses, new_metadata, new_nodes, new_conns,
                pop.neat_config.genome_config
            )

            # 清理旧基因组
            new_genome_ids = set(pop.neat_pop.population.keys())
            old_to_clean = [g for g in old_genomes if g.key not in new_genome_ids]
            pop._cleanup_genome_internals(old_to_clean)

            # 注：NEAT 历史数据清理已移至 save_checkpoint 时统一执行

            # 更新 Agent Brain 的 genome 引用
            new_genomes = list(pop.neat_pop.population.items())
            for idx, (gid, genome) in enumerate(new_genomes):
                if idx < len(pop.agents):
                    pop.agents[idx].brain.genome = genome

        self._pending_genome_data = None
        self._genomes_dirty = False

        elapsed = time.perf_counter() - start_time
        self.logger.info(f"延迟反序列化基因组完成，耗时: {elapsed:.2f}s")

    def evolve_with_cached_fitness(self) -> bool:
        """使用缓存的适应度进化所有子种群

        Returns:
            是否成功进化
        """
        all_success = True
        for pop in self.sub_populations:
            success = pop.evolve_with_cached_fitness()
            all_success = all_success and success
        return all_success

    def shutdown_executor(self) -> None:
        """关闭所有子种群的线程池"""
        for pop in self.sub_populations:
            pop.shutdown_executor()

    def get_all_genomes(self) -> list[neat.DefaultGenome]:
        """获取所有子种群的基因组（用于GenerationSaver等）

        Returns:
            所有子种群的基因组列表
        """
        all_genomes: list[neat.DefaultGenome] = []
        for pop in self.sub_populations:
            all_genomes.extend(pop.neat_pop.population.values())
        return all_genomes

    def accumulate_fitness(
        self, current_price: float,
    ) -> None:
        """累积所有子种群的适应度

        Args:
            current_price: 当前市场价格
        """
        for sub_pop in self.sub_populations:
            sub_pop.accumulate_fitness(current_price)

    def apply_accumulated_fitness(self) -> None:
        """应用所有子种群的累积适应度"""
        for sub_pop in self.sub_populations:
            sub_pop.apply_accumulated_fitness()

    def clear_accumulated_fitness(self) -> None:
        """清空所有子种群的累积适应度"""
        for sub_pop in self.sub_populations:
            sub_pop.clear_accumulated_fitness()

