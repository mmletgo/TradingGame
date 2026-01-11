"""Checkpoint 加载器模块

从主 checkpoint 文件加载所有基因组数据。
"""

import gzip
import pickle
import re
from pathlib import Path
from typing import Any

from src.config.config import AgentType
from src.core.log_engine.logger import get_logger


class CheckpointLoader:
    """从主 checkpoint 加载基因组数据

    支持两种 checkpoint 格式：
    - `multi_arena_gen_{N}.pkl`: 文件名直接对应代数
    - `ep_{episode}.pkl`: 代数 = episode / evolution_interval

    自动检测 gzip 压缩（magic bytes: 0x1f 0x8b）。

    Attributes:
        checkpoint_dir: checkpoint 文件目录
        evolution_interval: 进化间隔（用于计算 ep_*.pkl 的代数）
        logger: 日志器
    """

    DEFAULT_EVOLUTION_INTERVAL: int = 10

    checkpoint_dir: Path
    evolution_interval: int

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        evolution_interval: int | None = None,
    ) -> None:
        """初始化加载器

        Args:
            checkpoint_dir: checkpoint 文件目录
            evolution_interval: 进化间隔，默认为 DEFAULT_EVOLUTION_INTERVAL
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.evolution_interval = (
            evolution_interval
            if evolution_interval is not None
            else self.DEFAULT_EVOLUTION_INTERVAL
        )
        self.logger = get_logger("checkpoint_loader")

    def list_generations(self) -> list[int]:
        """列出所有可用代数

        扫描两种格式的 checkpoint 文件，返回排序后的代数列表。

        Returns:
            排序后的代数列表
        """
        generations: set[int] = set()

        if not self.checkpoint_dir.exists():
            self.logger.warning(f"checkpoint 目录不存在: {self.checkpoint_dir}")
            return []

        # 扫描 multi_arena_gen_{N}.pkl 格式
        multi_arena_pattern = re.compile(r"multi_arena_gen_(\d+)\.pkl$")
        for f in self.checkpoint_dir.glob("multi_arena_gen_*.pkl"):
            match = multi_arena_pattern.match(f.name)
            if match:
                gen = int(match.group(1))
                generations.add(gen)

        # 扫描 ep_{episode}.pkl 格式
        ep_pattern = re.compile(r"ep_(\d+)\.pkl$")
        for f in self.checkpoint_dir.glob("ep_*.pkl"):
            match = ep_pattern.match(f.name)
            if match:
                episode = int(match.group(1))
                gen = episode // self.evolution_interval
                generations.add(gen)

        result = sorted(generations)
        self.logger.debug(f"找到 {len(result)} 个可用代数")
        return result

    def find_checkpoint_for_generation(self, generation: int) -> Path | None:
        """查找对应代数的 checkpoint

        优先使用 multi_arena 格式。
        ep_{N}.pkl 的代数 = N / evolution_interval

        Args:
            generation: 代数

        Returns:
            checkpoint 文件路径，不存在返回 None
        """
        if not self.checkpoint_dir.exists():
            return None

        # 优先查找 multi_arena_gen_{N}.pkl
        multi_arena_path = self.checkpoint_dir / f"multi_arena_gen_{generation}.pkl"
        if multi_arena_path.exists():
            return multi_arena_path

        # 查找 ep_{episode}.pkl
        # 代数 N 对应的 episode = N * evolution_interval
        episode = generation * self.evolution_interval
        ep_path = self.checkpoint_dir / f"ep_{episode}.pkl"
        if ep_path.exists():
            return ep_path

        # 如果精确匹配不到，尝试找到最接近的 ep_*.pkl
        # 因为 episode 可能不是精确的倍数
        ep_pattern = re.compile(r"ep_(\d+)\.pkl$")
        best_match: Path | None = None
        best_gen_diff = float("inf")

        for f in self.checkpoint_dir.glob("ep_*.pkl"):
            match = ep_pattern.match(f.name)
            if match:
                ep = int(match.group(1))
                gen = ep // self.evolution_interval
                if gen == generation and abs(ep - episode) < best_gen_diff:
                    best_gen_diff = abs(ep - episode)
                    best_match = f

        return best_match

    def load_genomes(self, generation: int) -> dict[AgentType, list[bytes]] | None:
        """加载某代的全部基因组

        Args:
            generation: 代数

        Returns:
            dict[AgentType, list[bytes]]: 每物种的序列化基因组列表
            None: 文件不存在或加载失败
        """
        checkpoint_path = self.find_checkpoint_for_generation(generation)
        if checkpoint_path is None:
            self.logger.warning(f"未找到第 {generation} 代的 checkpoint")
            return None

        checkpoint_data = self._load_checkpoint(checkpoint_path)
        if checkpoint_data is None:
            return None

        result: dict[AgentType, list[bytes]] = {}

        populations_data = checkpoint_data.get("populations", {})
        for agent_type, pop_data in populations_data.items():
            # 确保 agent_type 是 AgentType 枚举
            if isinstance(agent_type, str):
                try:
                    agent_type = AgentType(agent_type)
                except ValueError:
                    self.logger.warning(f"未知的 Agent 类型: {agent_type}")
                    continue

            genomes = self._extract_genomes_from_pop_data(pop_data)
            if genomes:
                result[agent_type] = genomes
                self.logger.debug(
                    f"提取 {agent_type.value} 的 {len(genomes)} 个基因组"
                )

        self.logger.info(
            f"成功加载第 {generation} 代的基因组: "
            + ", ".join(f"{k.value}={len(v)}" for k, v in result.items())
        )

        return result

    def _load_checkpoint(self, path: Path) -> dict[str, Any] | None:
        """加载 checkpoint（自动检测压缩格式）

        Args:
            path: checkpoint 文件路径

        Returns:
            checkpoint 数据字典，加载失败返回 None
        """
        try:
            # 检测是否为 gzip 格式
            with open(path, "rb") as f:
                magic = f.read(2)

            # gzip 文件的魔数是 0x1f 0x8b
            if magic == b"\x1f\x8b":
                with gzip.open(path, "rb") as f:
                    data: dict[str, Any] = pickle.load(f)
            else:
                with open(path, "rb") as f:
                    data = pickle.load(f)

            self.logger.debug(f"成功加载 checkpoint: {path}")
            return data

        except Exception as e:
            self.logger.error(f"加载 checkpoint 失败: {path}, 错误: {e}")
            return None

    def _extract_genomes_from_pop_data(
        self, pop_data: dict[str, Any]
    ) -> list[bytes]:
        """从种群数据提取全部基因组

        支持两种格式：
        1. SubPopulationManager: is_sub_population_manager=True
        2. 普通 Population: 有 neat_pop 字段

        Args:
            pop_data: 种群数据字典

        Returns:
            序列化的基因组列表
        """
        genomes: list[bytes] = []

        if pop_data.get("is_sub_population_manager"):
            # SubPopulationManager 格式
            sub_populations = pop_data.get("sub_populations", [])
            for sub_pop_data in sub_populations:
                neat_pop = sub_pop_data.get("neat_pop")
                if neat_pop is not None:
                    genomes.extend(self._extract_genomes_from_neat_pop(neat_pop))
        else:
            # 普通 Population 格式
            neat_pop = pop_data.get("neat_pop")
            if neat_pop is not None:
                genomes.extend(self._extract_genomes_from_neat_pop(neat_pop))

        return genomes

    def _extract_genomes_from_neat_pop(self, neat_pop: Any) -> list[bytes]:
        """从 neat.Population 对象提取基因组

        Args:
            neat_pop: NEAT Population 对象

        Returns:
            序列化的基因组列表
        """
        genomes: list[bytes] = []

        try:
            # neat_pop.population 是 dict[int, DefaultGenome]
            if hasattr(neat_pop, "population"):
                for genome in neat_pop.population.values():
                    genome_data = pickle.dumps(genome)
                    genomes.append(genome_data)
        except Exception as e:
            self.logger.error(f"从 neat_pop 提取基因组失败: {e}")

        return genomes

    def get_generation_from_checkpoint_path(self, path: str | Path) -> int | None:
        """从 checkpoint 路径解析代数

        Args:
            path: checkpoint 文件路径

        Returns:
            代数，无法解析返回 None
        """
        path = Path(path)
        filename = path.name

        # 尝试 multi_arena_gen_{N}.pkl 格式
        multi_arena_pattern = re.compile(r"multi_arena_gen_(\d+)\.pkl$")
        match = multi_arena_pattern.match(filename)
        if match:
            return int(match.group(1))

        # 尝试 ep_{episode}.pkl 格式
        ep_pattern = re.compile(r"ep_(\d+)\.pkl$")
        match = ep_pattern.match(filename)
        if match:
            episode = int(match.group(1))
            return episode // self.evolution_interval

        return None

    def load_checkpoint_data(self, path: str | Path) -> dict[str, Any] | None:
        """直接加载 checkpoint 数据

        Args:
            path: checkpoint 文件路径

        Returns:
            checkpoint 数据字典，加载失败返回 None
        """
        return self._load_checkpoint(Path(path))
