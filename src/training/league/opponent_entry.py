"""对手条目数据结构"""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field, fields, asdict
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
    has_pre_evolution_fitness: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        d = asdict(self)
        d['agent_type'] = self.agent_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpponentMetadata:
        """从字典创建（容忍未知字段）"""
        data = data.copy()
        data['agent_type'] = AgentType(data['agent_type'])
        # 过滤未知字段，确保向前兼容
        known_fields: set[str] = {f.name for f in fields(cls)}
        filtered: dict[str, Any] = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


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
    network_data: dict[int, tuple[np.ndarray, ...]] | None = None
    # 预进化适应度数据: {sub_pop_id: fitness_array}
    pre_evolution_fitness: dict[int, np.ndarray] | None = None

    @property
    def entry_id(self) -> str:
        """获取条目 ID"""
        return self.metadata.entry_id

    @property
    def agent_type(self) -> AgentType:
        """获取 Agent 类型"""
        return self.metadata.agent_type

    def save(self, entry_dir: Path) -> None:
        """保存条目到目录（原子写入）

        使用临时目录写入后替换，确保保存过程的原子性。

        Args:
            entry_dir: 条目目录路径
        """
        tmp_dir = entry_dir.parent / f"{entry_dir.name}.tmp"
        try:
            # 清理可能残留的临时目录
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # 预先设置 has_pre_evolution_fitness 标记（避免双写 metadata.json）
            if self.pre_evolution_fitness is not None:
                self.metadata.has_pre_evolution_fitness = True

            # 自动更新 agent_count
            self.metadata.agent_count = self.get_total_agent_count()

            # 1. 保存元数据（只写一次）
            metadata_path = tmp_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata.to_dict(), f, indent=2, ensure_ascii=False)

            # 2. 保存基因组数据（如果有）
            if self.genome_data is not None:
                genomes_path = tmp_dir / "genomes.npz"
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
                networks_path = tmp_dir / "networks.npz"
                network_arrays: dict[str, np.ndarray] = {}
                for sub_pop_id, params_tuple in self.network_data.items():
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

            # 4. 保存预进化适应度（如果有）
            if self.pre_evolution_fitness is not None:
                fitness_path = tmp_dir / "pre_evolution_fitness.npz"
                fitness_arrays: dict[str, np.ndarray] = {}
                for sub_pop_id, fitness_arr in self.pre_evolution_fitness.items():
                    fitness_arrays[f"sub_{sub_pop_id}_fitness"] = fitness_arr
                np.savez(fitness_path, **fitness_arrays)

            # 原子替换：删除旧目录，移动临时目录
            if entry_dir.exists():
                shutil.rmtree(entry_dir)
            shutil.move(str(tmp_dir), str(entry_dir))

        except Exception:
            # 清理临时目录
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise

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
        genomes_path = entry_dir / "genomes.npz"
        genome_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None
        if genomes_path.exists():
            # 【内存泄漏修复】使用 with 上下文管理器确保 NpzFile 关闭，
            # 并用 np.array() 拷贝独立数组（脱离 mmap 引用）
            try:
                with np.load(genomes_path) as genome_arrays:
                    # 解析子种群 ID
                    sub_pop_ids: set[int] = set()
                    for key in genome_arrays.files:
                        match = re.match(r"^sub_(\d+)_keys$", key)
                        if match:
                            sub_pop_id = int(match.group(1))
                            sub_pop_ids.add(sub_pop_id)

                    genome_data = {}
                    for sub_pop_id in sorted(sub_pop_ids):
                        keys = np.array(genome_arrays[f"sub_{sub_pop_id}_keys"])
                        fitnesses = np.array(genome_arrays[f"sub_{sub_pop_id}_fitnesses"])
                        meta = np.array(genome_arrays[f"sub_{sub_pop_id}_metadata"])
                        nodes = np.array(genome_arrays[f"sub_{sub_pop_id}_nodes"])
                        conns = np.array(genome_arrays[f"sub_{sub_pop_id}_conns"])
                        genome_data[sub_pop_id] = (keys, fitnesses, meta, nodes, conns)
            except (EOFError, OSError, ValueError) as e:
                logging.getLogger("league").warning(
                    f"genomes.npz 损坏或为空: {genomes_path}, 错误: {e}"
                )
                genome_data = None
        else:
            logging.getLogger("league").warning(
                f"genomes.npz 不存在: {genomes_path}，genome_data 将为 None"
            )

        # 3. 加载网络参数（可选）
        network_data: dict[int, tuple[np.ndarray, ...]] | None = None
        if load_networks:
            networks_path = entry_dir / "networks.npz"
            if networks_path.exists():
                try:
                    with np.load(networks_path) as network_arrays:
                        network_data = {}
                        # 从 network_arrays 解析 sub_pop_ids
                        net_sub_pop_ids: set[int] = set()
                        for key in network_arrays.files:
                            if key.startswith("sub_") and key.endswith("_headers"):
                                net_sub_pop_id = int(key.split("_")[1])
                                net_sub_pop_ids.add(net_sub_pop_id)
                        for sub_pop_id in sorted(net_sub_pop_ids):
                            prefix = f"sub_{sub_pop_id}_"
                            header_key = f"{prefix}headers"
                            if header_key not in network_arrays:
                                logging.getLogger("league").warning(
                                    f"networks.npz 中缺少 sub_pop_id={sub_pop_id} 的数据，跳过"
                                )
                                continue
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
                except (EOFError, OSError, ValueError) as e:
                    logging.getLogger("league").warning(
                        f"networks.npz 损坏或为空: {networks_path}, 错误: {e}，将尝试从基因组重建"
                    )
                    network_data = None

        # 4. 加载预进化适应度（如果有）
        pre_evolution_fitness: dict[int, np.ndarray] | None = None
        if metadata.has_pre_evolution_fitness:
            fitness_path = entry_dir / "pre_evolution_fitness.npz"
            if fitness_path.exists():
                try:
                    with np.load(fitness_path) as fitness_arrays:
                        pre_evolution_fitness = {}
                        for key in fitness_arrays.files:
                            match = re.match(r"^sub_(\d+)_fitness$", key)
                            if match:
                                sub_pop_id = int(match.group(1))
                                pre_evolution_fitness[sub_pop_id] = np.array(fitness_arrays[key])
                except (EOFError, OSError, ValueError) as e:
                    logging.getLogger("league").warning(
                        f"pre_evolution_fitness.npz 损坏或为空: {fitness_path}, 错误: {e}"
                    )
                    pre_evolution_fitness = None

        return cls(
            metadata=metadata,
            genome_data=genome_data,
            network_data=network_data,
            pre_evolution_fitness=pre_evolution_fitness,
        )

    def get_total_agent_count(self) -> int:
        """获取总 Agent 数量"""
        if self.genome_data is None:
            return self.metadata.agent_count
        total = 0
        for keys, _, _, _, _ in self.genome_data.values():
            total += len(keys)
        return total
