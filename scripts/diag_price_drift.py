#!/usr/bin/env python3
"""诊断价格漂移的快速脚本

使用单竞技场 Trainer 模式运行多个 episode，记录每 episode 的关键指标。
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.create_config import create_default_config
from src.core.log_engine import setup_logging
from src.training.trainer import Trainer
from src.bio.agents.base import AgentType
import logging


def main() -> None:
    setup_logging(console_level=logging.WARNING)

    config = create_default_config(episode_length=100, checkpoint_interval=999)

    trainer = Trainer(config)
    trainer.setup()

    num_episodes = 50  # 5代进化
    diag_file = open("/tmp/price_drift_diag.log", "w")

    trainer.is_running = True

    for ep in range(1, num_episodes + 1):
        trainer.run_episode()

        # Episode 结束后收集统计
        assert trainer.matching_engine is not None
        ob = trainer.matching_engine._orderbook
        mid = ob.get_mid_price() or 100.0
        last = ob.last_price
        high = trainer._episode_high_price
        low = trainer._episode_low_price
        smooth = trainer._smooth_mid_price
        buy_prob = trainer._episode_buy_probability

        # MM 聚合仓位
        mm_long_total = 0
        mm_short_total = 0
        mm_count = 0
        for agent in trainer.populations[AgentType.MARKET_MAKER].agents:
            if not agent.is_liquidated:
                mm_count += 1
                pos = agent.account.position.quantity
                if pos > 0:
                    mm_long_total += pos
                elif pos < 0:
                    mm_short_total += pos

        # 散户仓位统计
        retail_long = 0
        retail_short = 0
        retail_flat = 0
        retail_alive = 0
        for agent in trainer.populations[AgentType.RETAIL_PRO].agents:
            if not agent.is_liquidated:
                retail_alive += 1
                pos = agent.account.position.quantity
                if pos > 0:
                    retail_long += 1
                elif pos < 0:
                    retail_short += 1
                else:
                    retail_flat += 1

        # 噪声交易者仓位
        nt_long = 0
        nt_short = 0
        for nt in trainer.noise_traders:
            pos = nt.account.position_qty
            if pos > 0:
                nt_long += pos
            elif pos < 0:
                nt_short += pos

        line = (
            f"Ep {ep:3d} | high={high:.4f} low={low:.4f} last={last:.4f}"
            f" | buy_prob={buy_prob:.3f}"
            f" | MM[{mm_count}]: L={mm_long_total:>10,} S={mm_short_total:>10,}"
            f" | Ret[{retail_alive}]: L={retail_long} S={retail_short} F={retail_flat}"
            f" | NT: L={nt_long:>10,} S={nt_short:>10,}"
        )
        print(line, flush=True)
        diag_file.write(line + "\n")
        diag_file.flush()

    diag_file.close()
    print("\nDiagnostics written to /tmp/price_drift_diag.log")


if __name__ == "__main__":
    main()
