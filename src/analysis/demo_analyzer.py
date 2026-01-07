"""演示模式分析器模块

分析演示模式结束时各物种存活个体的分布，生成分析图和终端摘要。
"""

from typing import TYPE_CHECKING, Any
from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

if TYPE_CHECKING:
    from src.training.trainer import Trainer
    from src.config.config import AgentType


# 中文名称映射
AGENT_TYPE_NAMES: dict[str, str] = {
    "RETAIL": "散户",
    "RETAIL_PRO": "高级散户",
    "WHALE": "庄家",
    "MARKET_MAKER": "做市商",
}


class DemoAnalyzer:
    """演示模式分析器

    分析演示结束时各物种存活个体的分布，生成分析图和终端摘要。
    """

    _output_dir: str
    _chinese_font: str | None

    def __init__(self, output_dir: str = "analysis_output") -> None:
        """初始化分析器

        Args:
            output_dir: 分析结果输出目录
        """
        self._output_dir = output_dir
        self._chinese_font = self._find_chinese_font()

    def _find_chinese_font(self) -> str | None:
        """查找可用的中文字体

        Returns:
            字体路径，如果未找到返回 None
        """
        # 常见中文字体路径
        font_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            # Windows
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            # macOS
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
        ]

        for path in font_paths:
            if os.path.exists(path):
                return path
        return None

    def analyze(
        self,
        trainer: "Trainer",
        end_reason: str,
        end_agent_type: "AgentType | None",
    ) -> None:
        """执行分析并输出结果

        Args:
            trainer: 训练器实例
            end_reason: 结束原因 ("population_depleted")
            end_agent_type: 触发结束的 Agent 类型
        """
        # 1. 收集数据
        data = self._collect_data(trainer)

        # 2. 打印终端摘要
        self._print_summary(data, end_reason, end_agent_type)

        # 3. 生成分析图
        plot_path = self._generate_plots(data)

        print(f"\n分析图已保存到: {plot_path}")
        print("=" * 50)

    def _collect_data(self, trainer: "Trainer") -> dict[str, Any]:
        """收集各物种存活个体的数据

        收集内容（每个存活 Agent）:
        - equity: 净值
        - balance: 余额
        - unrealized_pnl: 浮动盈亏
        - position_qty: 持仓量
        - leverage: 杠杆率 (position_value / equity)

        Returns:
            {
                "episode": int,
                "tick": int,
                "final_price": float,
                "high_price": float,
                "low_price": float,
                "populations": {
                    AgentType: {
                        "total_count": int,
                        "alive_count": int,
                        "agents": [
                            {"equity": float, "balance": float, "unrealized_pnl": float,
                             "position_qty": int, "leverage": float},
                            ...
                        ]
                    },
                    ...
                }
            }
        """
        from src.config.config import AgentType

        # 获取当前价格
        current_price = trainer.matching_engine._orderbook.last_price

        data: dict[str, Any] = {
            "episode": trainer.episode,
            "tick": trainer.tick,
            "final_price": current_price,
            "high_price": trainer._episode_high_price,
            "low_price": trainer._episode_low_price,
            "populations": {},
        }

        for agent_type in AgentType:
            population = trainer.populations.get(agent_type)
            if population is None:
                continue

            total_count = len(population.agents)
            agents_data: list[dict[str, Any]] = []

            for agent in population.agents:
                if agent.is_liquidated:
                    continue

                # 收集存活 Agent 的数据
                equity = agent.account.get_equity(current_price)
                balance = agent.account.balance
                position_qty = agent.account.position.quantity
                unrealized_pnl = agent.account.position.get_unrealized_pnl(
                    current_price
                )

                # 计算杠杆率
                position_value = abs(position_qty) * current_price
                leverage = position_value / equity if equity > 0 else 0.0

                agents_data.append(
                    {
                        "equity": equity,
                        "balance": balance,
                        "unrealized_pnl": unrealized_pnl,
                        "position_qty": position_qty,
                        "leverage": leverage,
                    }
                )

            data["populations"][agent_type] = {
                "total_count": total_count,
                "alive_count": len(agents_data),
                "agents": agents_data,
            }

        return data

    def _format_money(self, amount: float) -> str:
        """格式化金额显示

        Args:
            amount: 金额

        Returns:
            格式化后的字符串
        """
        abs_amount = abs(amount)
        if abs_amount >= 1e8:
            return f"{amount / 1e8:.2f}亿"
        elif abs_amount >= 1e4:
            return f"{amount / 1e4:.2f}万"
        else:
            return f"{amount:.2f}"

    def _print_summary(
        self,
        data: dict[str, Any],
        end_reason: str,
        end_agent_type: "AgentType | None",
    ) -> None:
        """终端打印摘要

        格式:
        ==================================================
        演示模式分析结果
        ==================================================

        基本信息:
          Episode: 1 | Tick: 847
          结束原因: 散户种群存活不足 1/4

        价格统计:
          最终价格: 105.20 | 最高: 112.50 | 最低: 95.30

        种群统计:
          散户      存活 2450/10000 (24.5%)  平均净值 485万  盈利/亏损 850/1600
          高级散户  存活 85/100 (85.0%)      平均净值 523万  盈利/亏损 60/25
          庄家      存活 180/200 (90.0%)     平均净值 1.2亿  盈利/亏损 150/30
          做市商    存活 145/150 (96.7%)     平均净值 2.1亿  盈利/亏损 130/15

        分析图已保存到: analysis_output/demo_analysis_20260107_123456.png
        ==================================================
        """
        from src.config.config import AgentType

        print()
        print("=" * 50)
        print("演示模式分析结果")
        print("=" * 50)
        print()

        # 基本信息
        print("基本信息:")
        print(f"  Episode: {data['episode']} | Tick: {data['tick']}")

        # 结束原因
        if end_reason == "population_depleted" and end_agent_type is not None:
            agent_name = AGENT_TYPE_NAMES.get(end_agent_type.value, end_agent_type.value)
            pop_data = data["populations"].get(end_agent_type, {})
            alive = pop_data.get("alive_count", 0)
            total = pop_data.get("total_count", 0)
            print(f"  结束原因: {agent_name}种群存活不足 1/4 ({alive}/{total})")
        elif end_reason == "one_sided_orderbook":
            print("  结束原因: 订单簿只有单边挂单")
        else:
            print(f"  结束原因: {end_reason}")
        print()

        # 价格统计
        print("价格统计:")
        print(
            f"  最终价格: {data['final_price']:.2f} | "
            f"最高: {data['high_price']:.2f} | "
            f"最低: {data['low_price']:.2f}"
        )
        print()

        # 种群统计
        print("种群统计:")

        # 按照固定顺序输出
        agent_type_order = [
            AgentType.RETAIL,
            AgentType.RETAIL_PRO,
            AgentType.WHALE,
            AgentType.MARKET_MAKER,
        ]

        for agent_type in agent_type_order:
            pop_data = data["populations"].get(agent_type)
            if pop_data is None:
                continue

            agent_name = AGENT_TYPE_NAMES.get(agent_type.value, agent_type.value)
            alive_count = pop_data["alive_count"]
            total_count = pop_data["total_count"]
            alive_ratio = (alive_count / total_count * 100) if total_count > 0 else 0

            agents = pop_data["agents"]

            # 计算平均净值
            if agents:
                equities = [a["equity"] for a in agents]
                avg_equity = sum(equities) / len(equities)
                avg_equity_str = self._format_money(avg_equity)
            else:
                avg_equity_str = "N/A"

            # 计算盈利/亏损数量
            profit_count = sum(1 for a in agents if a["equity"] > a["balance"])
            loss_count = sum(1 for a in agents if a["equity"] < a["balance"])

            print(
                f"  {agent_name:<6}  "
                f"存活 {alive_count}/{total_count} ({alive_ratio:.1f}%)  "
                f"平均净值 {avg_equity_str}  "
                f"盈利/亏损 {profit_count}/{loss_count}"
            )

        print()

    def _generate_plots(self, data: dict[str, Any]) -> str:
        """生成分析图

        使用 matplotlib 生成:
        1. 资产分布图 - 4 个子图（每种群一个），箱线图显示净值分布
        2. 交易行为图 - 4 个子图，显示持仓量分布

        Returns:
            保存的图片路径
        """
        from src.config.config import AgentType

        # 确保输出目录存在
        os.makedirs(self._output_dir, exist_ok=True)

        # 设置中文字体
        if self._chinese_font:
            font_prop = fm.FontProperties(fname=self._chinese_font)
            plt.rcParams["font.family"] = font_prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
        else:
            # 尝试使用系统中文字体
            plt.rcParams["font.sans-serif"] = [
                "WenQuanYi Micro Hei",
                "SimHei",
                "Microsoft YaHei",
                "Arial Unicode MS",
            ]
            plt.rcParams["axes.unicode_minus"] = False

        # 创建 2x4 子图布局
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(
            f"演示模式分析 - Episode {data['episode']}, Tick {data['tick']}",
            fontsize=14,
            fontweight="bold",
        )

        # 种群顺序和颜色
        agent_type_order = [
            AgentType.RETAIL,
            AgentType.RETAIL_PRO,
            AgentType.WHALE,
            AgentType.MARKET_MAKER,
        ]
        colors = ["#64C864", "#6496FF", "#FF6496", "#C864FF"]

        # 第一行：资产分布（箱线图）
        for idx, agent_type in enumerate(agent_type_order):
            ax = axes[0, idx]
            pop_data = data["populations"].get(agent_type)
            agent_name = AGENT_TYPE_NAMES.get(agent_type.value, agent_type.value)

            if pop_data is None or not pop_data["agents"]:
                ax.text(
                    0.5,
                    0.5,
                    "无数据",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{agent_name}\n净值分布")
                ax.set_xticks([])
                continue

            equities = [a["equity"] for a in pop_data["agents"]]
            equities_array = np.array(equities)

            # 转换为万元
            equities_wan = equities_array / 1e4

            bp = ax.boxplot(
                equities_wan,
                vert=True,
                patch_artist=True,
                widths=0.6,
            )

            # 设置颜色
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[idx])
                patch.set_alpha(0.7)

            alive_count = pop_data["alive_count"]
            total_count = pop_data["total_count"]
            ax.set_title(f"{agent_name}\n存活 {alive_count}/{total_count}")
            ax.set_ylabel("净值 (万元)")
            ax.set_xticks([])

            # 添加统计信息
            median_val = np.median(equities_wan)
            mean_val = np.mean(equities_wan)
            ax.axhline(
                y=mean_val,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"均值: {mean_val:.1f}万",
            )
            ax.legend(loc="upper right", fontsize=8)

        # 第二行：持仓量分布
        for idx, agent_type in enumerate(agent_type_order):
            ax = axes[1, idx]
            pop_data = data["populations"].get(agent_type)
            agent_name = AGENT_TYPE_NAMES.get(agent_type.value, agent_type.value)

            if pop_data is None or not pop_data["agents"]:
                ax.text(
                    0.5,
                    0.5,
                    "无数据",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{agent_name}\n持仓分布")
                ax.set_xticks([])
                continue

            position_qtys = [a["position_qty"] for a in pop_data["agents"]]
            position_array = np.array(position_qtys)

            # 分离多头、空头、空仓
            long_count = np.sum(position_array > 0)
            short_count = np.sum(position_array < 0)
            flat_count = np.sum(position_array == 0)

            # 绘制条形图
            categories = ["多头", "空仓", "空头"]
            counts = [long_count, flat_count, short_count]
            bar_colors = ["#FF6B6B", "#AAAAAA", "#4ECDC4"]

            bars = ax.bar(categories, counts, color=bar_colors)

            # 添加数量标签
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        str(int(count)),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

            ax.set_title(f"{agent_name}\n持仓分布")
            ax.set_ylabel("数量")

        plt.tight_layout()

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_analysis_{timestamp}.png"
        filepath = os.path.join(self._output_dir, filename)

        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return filepath
