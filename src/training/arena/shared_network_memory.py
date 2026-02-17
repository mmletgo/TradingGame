"""共享内存网络数据管理模块

管理 BatchNetworkData 的共享内存生命周期。
主进程创建共享内存并填充网络数据，Worker 进程通过共享内存名称
附着到同一块内存实现零拷贝访问。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from src.bio.agents.base import AgentType

logger = logging.getLogger(__name__)


@dataclass
class SharedNetworkMetadata:
    """轻量级元数据，通过 Queue 发送给 Worker（~200 字节）"""

    shm_name: str
    agent_type: AgentType
    num_networks: int
    max_nodes: int
    max_connections: int
    max_inputs: int
    max_outputs: int
    total_nodes: int
    total_connections: int
    total_outputs: int
    generation: int


class SharedNetworkMemory:
    """管理 BatchNetworkData 的共享内存生命周期

    主进程调用 create_and_fill() 创建并填充共享内存。
    Worker 进程调用 attach() 附着到已有共享内存。

    Attributes:
        _shm: SharedMemory 实例（创建者或附着者）
        _is_creator: 是否是创建者（决定是否需要 unlink）
        _agent_type: Agent 类型
    """

    def __init__(self) -> None:
        self._shm: SharedMemory | None = None
        self._is_creator: bool = False
        self._agent_type: AgentType | None = None

    @staticmethod
    def compute_buffer_size(
        num_networks: int,
        total_nodes: int,
        total_connections: int,
        total_outputs: int,
    ) -> int:
        """根据网络维度计算所需共享内存大小

        Buffer 布局：
        - Header (128 bytes)
        - Section A: double 数组 (biases + responses + conn_weights)
        - Section B: int32 数组 (meta arrays + act_types + conn_indptr + conn_sources + output_indices)
        """
        from src.training._cython.batch_decide_openmp import compute_shared_memory_size

        return compute_shared_memory_size(
            num_networks, total_nodes, total_connections, total_outputs
        )

    def create_and_fill(
        self,
        agent_type: AgentType,
        network_params: tuple[np.ndarray, ...],
        generation: int,
    ) -> SharedNetworkMetadata:
        """主进程调用：创建共享内存，填充转换后的网络数据

        Args:
            agent_type: Agent 类型
            network_params: packed numpy 参数元组 (headers, all_input_keys, all_output_keys,
                           all_node_ids, all_biases, all_responses, all_act_types,
                           all_conn_indptr, all_conn_sources, all_conn_weights, all_output_indices)
            generation: 当前进化代数

        Returns:
            SharedNetworkMetadata 元数据，供 Worker 进程使用
        """
        # 1. 解包 network_params
        (
            headers,
            _all_input_keys,
            _all_output_keys,
            _all_node_ids,
            all_biases,
            all_responses,
            all_act_types,
            all_conn_indptr,
            all_conn_sources,
            all_conn_weights,
            all_output_indices,
        ) = network_params

        # 2. 从 headers 计算维度
        num_networks: int = len(headers)
        h_num_inputs = headers["num_inputs"]
        h_num_outputs = headers["num_outputs"]
        h_num_nodes = headers["num_nodes"]
        h_n_connections = headers["n_connections"]
        h_n_output_keys = headers["n_output_keys"]

        max_nodes: int = int(np.max(h_num_nodes))
        max_connections: int = int(np.max(h_n_connections))
        max_inputs: int = int(np.max(h_num_inputs))
        max_outputs: int = int(np.max(h_num_outputs))
        total_nodes: int = int(np.sum(h_num_nodes))
        total_connections: int = int(np.sum(h_n_connections))
        total_outputs: int = int(np.sum(h_n_output_keys))

        # 3. 计算所需 buffer 大小
        buf_size: int = self.compute_buffer_size(
            num_networks, total_nodes, total_connections, total_outputs
        )

        # 4. 创建 SharedMemory
        self._shm = SharedMemory(create=True, size=buf_size)
        self._is_creator = True
        self._agent_type = agent_type

        # 5. 调用 Cython fill_shared_memory_buffer 填充数据
        from src.training._cython.batch_decide_openmp import fill_shared_memory_buffer

        shm_buf: np.ndarray = np.frombuffer(self._shm.buf, dtype=np.uint8)
        fill_shared_memory_buffer(
            shm_buf,
            headers,
            all_biases,
            all_responses,
            all_act_types,
            all_conn_indptr,
            all_conn_sources,
            all_conn_weights,
            all_output_indices,
        )

        # 6. 返回元数据
        return SharedNetworkMetadata(
            shm_name=self._shm.name,
            agent_type=agent_type,
            num_networks=num_networks,
            max_nodes=max_nodes,
            max_connections=max_connections,
            max_inputs=max_inputs,
            max_outputs=max_outputs,
            total_nodes=total_nodes,
            total_connections=total_connections,
            total_outputs=total_outputs,
            generation=generation,
        )

    def attach(self, metadata: SharedNetworkMetadata) -> memoryview:
        """Worker 调用：附着到已有共享内存

        Args:
            metadata: 从主进程接收的元数据

        Returns:
            共享内存的 buffer (memoryview)
        """
        self._shm = SharedMemory(name=metadata.shm_name, create=False)
        self._is_creator = False
        self._agent_type = metadata.agent_type
        return self._shm.buf

    def close(self) -> None:
        """关闭共享内存映射（不删除底层共享内存段）"""
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None

    def unlink(self) -> None:
        """删除底层共享内存段（仅创建者调用）"""
        if self._shm is not None and self._is_creator:
            try:
                self._shm.unlink()
            except Exception:
                pass

    def close_and_unlink(self) -> None:
        """关闭并删除共享内存"""
        if self._shm is not None:
            shm_name: str = self._shm.name
            try:
                if self._is_creator:
                    self._shm.unlink()
                    logger.debug("已删除共享内存段: %s", shm_name)
            except Exception:
                logger.warning("删除共享内存段失败: %s", shm_name, exc_info=True)
            try:
                self._shm.close()
                logger.debug("已关闭共享内存映射: %s", shm_name)
            except Exception:
                logger.warning("关闭共享内存映射失败: %s", shm_name, exc_info=True)
            self._shm = None
