"""对手条目数据结构"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config.config import AgentType


@dataclass
class OpponentMetadata:
    """单个类型的对手元数据"""
    entry_id: str
    agent_type: AgentType
    source: str  # 'main_agents', 'league_exploiter', 'main_exploiter'
    source_generation: int
    add_reason: str  # 'milestone', 'elite', 'diverse', 'exploiter_win_rate'
    avg_fitness: float
    # 与其他三种类型的胜率
    win_rates: dict[str, float] = field(default_factory=dict)  # {"vs_RETAIL_PRO": 0.45, ...}
    match_counts: dict[str, int] = field(default_factory=dict)  # {"vs_RETAIL_PRO": 100, ...}
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    agent_count: int = 0
    sub_population_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        d = asdict(self)
        d['agent_type'] = self.agent_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpponentMetadata:
        """从字典创建"""
        data = data.copy()
        data['agent_type'] = AgentType(data['agent_type'])
        return cls(**data)


@dataclass
class OpponentEntry:
    """单个类型的对手池条目

    存储对手的完整数据，包括元数据、基因组数据和网络参数。

    注意：genome_data 和 network_data 可以为 None，用于内存优化场景
    （保存到磁盘后清理内存中的大数据，只保留元数据）。
    """
    metadata: OpponentMetadata
    # {sub_pop_id: (keys, fitnesses, metadata, nodes, conns)}
    # 可以为 None（内存优化：保存后清理，需要时重新加载）
    genome_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None
    # {sub_pop_id: network_params_tuple} - 可选，延迟加载
    network_data: dict[int, tuple] | None = None

    @property
    def entry_id(self) -> str:
        """获取条目 ID"""
        return self.metadata.entry_id

    @property
    def agent_type(self) -> AgentType:
        """获取 Agent 类型"""
        return self.metadata.agent_type

    def save(self, entry_dir: Path) -> None:
        """保存条目到目录

        Args:
            entry_dir: 条目目录路径
        """
        entry_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存元数据
        metadata_path = entry_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata.to_dict(), f, indent=2, ensure_ascii=False)

        # 2. 保存基因组数据（如果有）
        if self.genome_data is not None:
            genomes_path = entry_dir / "genomes.npz"
            genome_arrays: dict[str, np.ndarray] = {}
            for sub_pop_id, (keys, fitnesses, meta, nodes, conns) in self.genome_data.items():
                genome_arrays[f"sub_{sub_pop_id}_keys"] = keys
                genome_arrays[f"sub_{sub_pop_id}_fitnesses"] = fitnesses
                genome_arrays[f"sub_{sub_pop_id}_metadata"] = meta
                genome_arrays[f"sub_{sub_pop_id}_nodes"] = nodes
                genome_arrays[f"sub_{sub_pop_id}_conns"] = conns
            np.savez(genomes_path, **genome_arrays)

        # 3. 保存网络参数（如果有）
        if self.network_data is not None:
            networks_path = entry_dir / "networks.npz"
            network_arrays: dict[str, np.ndarray] = {}
            for sub_pop_id, params_tuple in self.network_data.items():
                # params_tuple = (headers, input_keys, output_keys, node_ids,
                #                 biases, responses, act_types,
                #                 conn_indptr, conn_sources, conn_weights, output_indices)
                network_arrays[f"sub_{sub_pop_id}_headers"] = params_tuple[0]
                network_arrays[f"sub_{sub_pop_id}_input_keys"] = params_tuple[1]
                network_arrays[f"sub_{sub_pop_id}_output_keys"] = params_tuple[2]
                network_arrays[f"sub_{sub_pop_id}_node_ids"] = params_tuple[3]
                network_arrays[f"sub_{sub_pop_id}_biases"] = params_tuple[4]
                network_arrays[f"sub_{sub_pop_id}_responses"] = params_tuple[5]
                network_arrays[f"sub_{sub_pop_id}_act_types"] = params_tuple[6]
                network_arrays[f"sub_{sub_pop_id}_conn_indptr"] = params_tuple[7]
                network_arrays[f"sub_{sub_pop_id}_conn_sources"] = params_tuple[8]
                network_arrays[f"sub_{sub_pop_id}_conn_weights"] = params_tuple[9]
                network_arrays[f"sub_{sub_pop_id}_output_indices"] = params_tuple[10]
            np.savez(networks_path, **network_arrays)

    @classmethod
    def load(cls, entry_dir: Path, load_networks: bool = False) -> OpponentEntry:
        """从目录加载条目

        Args:
            entry_dir: 条目目录路径
            load_networks: 是否加载网络参数

        Returns:
            加载的条目对象
        """
        # 1. 加载元数据
        metadata_path = entry_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = OpponentMetadata.from_dict(json.load(f))

        # 2. 加载基因组数据
        # 【内存泄漏修复】使用 with 上下文管理器确保 NpzFile 关闭，
        # 并用 np.array() 拷贝独立数组（脱离 mmap 引用）
        genomes_path = entry_dir / "genomes.npz"
        with np.load(genomes_path) as genome_arrays:
            # 解析子种群 ID
            sub_pop_ids: set[int] = set()
            for key in genome_arrays.files:
                if key.startswith("sub_") and "_keys" in key:
                    sub_pop_id = int(key.split("_")[1])
                    sub_pop_ids.add(sub_pop_id)

            genome_data: dict[int, tuple] = {}
            for sub_pop_id in sorted(sub_pop_ids):
                keys = np.array(genome_arrays[f"sub_{sub_pop_id}_keys"])
                fitnesses = np.array(genome_arrays[f"sub_{sub_pop_id}_fitnesses"])
                meta = np.array(genome_arrays[f"sub_{sub_pop_id}_metadata"])
                nodes = np.array(genome_arrays[f"sub_{sub_pop_id}_nodes"])
                conns = np.array(genome_arrays[f"sub_{sub_pop_id}_conns"])
                genome_data[sub_pop_id] = (keys, fitnesses, meta, nodes, conns)

        # 3. 加载网络参数（可选）
        network_data: dict[int, tuple] | None = None
        if load_networks:
            networks_path = entry_dir / "networks.npz"
            if networks_path.exists():
                with np.load(networks_path) as network_arrays:
                    network_data = {}
                    for sub_pop_id in sorted(sub_pop_ids):
                        prefix = f"sub_{sub_pop_id}_"
                        network_data[sub_pop_id] = (
                            np.array(network_arrays[f"{prefix}headers"]),
                            np.array(network_arrays[f"{prefix}input_keys"]),
                            np.array(network_arrays[f"{prefix}output_keys"]),
                            np.array(network_arrays[f"{prefix}node_ids"]),
                            np.array(network_arrays[f"{prefix}biases"]),
                            np.array(network_arrays[f"{prefix}responses"]),
                            np.array(network_arrays[f"{prefix}act_types"]),
                            np.array(network_arrays[f"{prefix}conn_indptr"]),
                            np.array(network_arrays[f"{prefix}conn_sources"]),
                            np.array(network_arrays[f"{prefix}conn_weights"]),
                            np.array(network_arrays[f"{prefix}output_indices"]),
                        )

        return cls(
            metadata=metadata,
            genome_data=genome_data,
            network_data=network_data,
        )

    def get_total_agent_count(self) -> int:
        """获取总 Agent 数量"""
        if self.genome_data is None:
            return self.metadata.agent_count
        total = 0
        for keys, _, _, _, _ in self.genome_data.values():
            total += len(keys)
        return total
