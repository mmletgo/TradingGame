"""每代最佳基因组保存器模块

在进化后保存各物种的最佳基因组，用于后续进化效果评估。
"""

import logging
import os
import pickle
import time
from typing import TYPE_CHECKING, Any

from src.core.log_engine.logger import get_logger

if TYPE_CHECKING:
    from src.config.config import AgentType
    from src.training.population import Population, RetailSubPopulationManager


class GenerationSaver:
    """每代最佳基因组保存器

    在每代进化后保存各物种 fitness 最高的基因组，
    用于后续进化效果评估和分析。

    Attributes:
        output_dir: 输出目录路径
        logger: 日志器
    """

    output_dir: str
    logger: logging.Logger

    def __init__(self, output_dir: str = "checkpoints/generations") -> None:
        """初始化保存器，创建输出目录

        Args:
            output_dir: 输出目录路径，默认为 "checkpoints/generations"
        """
        self.output_dir = output_dir
        self.logger = get_logger("generation_saver")

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"GenerationSaver 初始化完成，输出目录: {self.output_dir}")

    def save_generation(
        self,
        generation: int,
        populations: dict["AgentType", "Population | RetailSubPopulationManager"],
        current_price: float,
    ) -> str:
        """保存一代的最佳基因组

        对每个种群获取 fitness 最高的 genome，序列化后保存。

        Args:
            generation: 当前代数
            populations: 各类型种群的字典，键为 AgentType，值为 Population
            current_price: 当前市场价格（暂未使用，预留扩展）

        Returns:
            保存的文件路径
        """
        # 延迟导入避免循环依赖
        from src.training.arena.migration import MigrationSystem

        best_genomes: dict[str, tuple[bytes, float]] = {}

        for agent_type, population in populations.items():
            # 获取种群中的所有基因组（兼容 Population 和 RetailSubPopulationManager）
            genomes = population.get_all_genomes()

            if not genomes:
                self.logger.warning(
                    f"种群 {agent_type.value} 为空，跳过"
                )
                continue

            # 按 fitness 降序排序，取第一个作为 best_genome
            # 过滤掉 fitness 为 None 的基因组
            valid_genomes = [g for g in genomes if g.fitness is not None]

            if not valid_genomes:
                self.logger.warning(
                    f"种群 {agent_type.value} 没有有效的 fitness，跳过"
                )
                continue

            # 按 fitness 降序排序
            sorted_genomes = sorted(
                valid_genomes,
                key=lambda g: g.fitness,  # type: ignore[arg-type, return-value]
                reverse=True,
            )
            best_genome = sorted_genomes[0]

            # 序列化 genome
            genome_data = MigrationSystem.serialize_genome(best_genome)
            fitness = float(best_genome.fitness)  # type: ignore[arg-type]

            # 使用 agent_type 的小写值作为键
            key = agent_type.value.lower()
            best_genomes[key] = (genome_data, fitness)

            self.logger.info(
                f"种群 {agent_type.value} 最佳基因组: fitness={fitness:.6f}"
            )

        # 构建保存数据
        save_data: dict[str, Any] = {
            "generation": generation,
            "timestamp": time.time(),
            "best_genomes": best_genomes,
        }

        # 保存到文件
        file_path = os.path.join(self.output_dir, f"gen_{generation}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(save_data, f)

        self.logger.info(
            f"第 {generation} 代最佳基因组已保存到: {file_path}"
        )

        return file_path

    def load_generation(self, generation: int) -> dict[str, Any] | None:
        """加载指定代的最佳基因组数据

        Args:
            generation: 要加载的代数

        Returns:
            保存的数据字典，不存在返回 None
        """
        file_path = os.path.join(self.output_dir, f"gen_{generation}.pkl")

        if not os.path.exists(file_path):
            self.logger.warning(f"第 {generation} 代的保存文件不存在: {file_path}")
            return None

        try:
            with open(file_path, "rb") as f:
                data: dict[str, Any] = pickle.load(f)
            self.logger.info(f"成功加载第 {generation} 代的最佳基因组数据")
            return data
        except Exception as e:
            self.logger.error(f"加载第 {generation} 代数据失败: {e}")
            return None

    def list_generations(self) -> list[int]:
        """列出所有已保存的代数

        Returns:
            升序排列的代数列表
        """
        generations: list[int] = []

        if not os.path.exists(self.output_dir):
            return generations

        for filename in os.listdir(self.output_dir):
            # 匹配 gen_{N}.pkl 格式
            if filename.startswith("gen_") and filename.endswith(".pkl"):
                try:
                    # 提取代数
                    gen_str = filename[4:-4]  # 去掉 "gen_" 和 ".pkl"
                    gen_num = int(gen_str)
                    generations.append(gen_num)
                except ValueError:
                    # 文件名格式不正确，跳过
                    self.logger.warning(f"无法解析文件名: {filename}")
                    continue

        # 升序排序
        generations.sort()

        return generations

    def generation_exists(self, generation: int) -> bool:
        """检查指定代是否已保存

        Args:
            generation: 代数

        Returns:
            如果该代已保存返回 True，否则返回 False
        """
        file_path = os.path.join(self.output_dir, f"gen_{generation}.pkl")
        return os.path.exists(file_path)
