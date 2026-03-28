"""HFtrade Monitor 数据加载器

加载 HFtrade Monitor 录制的 5 档订单簿快照 + 逐笔成交 parquet 数据。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.replay.config import ReplayConfig

logger = logging.getLogger(__name__)


@dataclass
class OrderbookSnapshot:
    """订单簿快照"""

    timestamp_ms: int
    mid_price: float
    bid_prices: np.ndarray  # shape (depth,), dtype float64, 降序（最优在前）
    bid_amounts: np.ndarray  # shape (depth,), dtype float64
    ask_prices: np.ndarray  # shape (depth,), dtype float64, 升序（最优在前）
    ask_amounts: np.ndarray  # shape (depth,), dtype float64


@dataclass
class MarketTrade:
    """市场成交"""

    timestamp_ms: int
    price: float
    amount: float  # 数量（已转换为基础货币单位）
    side: str  # 'buy' 或 'sell'（taker 方向）


class DataLoader:
    """HFtrade Monitor 数据加载器"""

    def __init__(self, config: ReplayConfig) -> None:
        self._config = config

    def load(self) -> tuple[list[OrderbookSnapshot], list[MarketTrade]]:
        """加载日期范围内的订单簿和成交数据

        Returns:
            (ob_snapshots, trades) 各自按 timestamp_ms 升序排列
        """
        dates = self._date_range()
        if not dates:
            logger.warning("日期范围为空，未加载任何数据")
            return [], []

        logger.info("加载数据: %s ~ %s (%d 天)", dates[0], dates[-1], len(dates))

        ob_df = self._load_orderbook_parquets(dates)
        trade_df = self._load_trade_parquets(dates)

        snapshots = self._df_to_snapshots(ob_df) if not ob_df.empty else []
        trades = self._df_to_trades(trade_df) if not trade_df.empty else []

        logger.info("加载完成: %d 订单簿快照, %d 逐笔成交", len(snapshots), len(trades))
        return snapshots, trades

    def _date_range(self) -> list[str]:
        """生成日期范围列表 ["2026-03-01", "2026-03-02", ...]"""
        start = datetime.strptime(self._config.date_start, "%Y-%m-%d")
        end = datetime.strptime(self._config.date_end, "%Y-%m-%d")

        dates: list[str] = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates

    def _load_orderbook_parquets(self, dates: list[str]) -> pd.DataFrame:
        """加载所有订单簿 parquet 文件并合并

        路径: {data_dir}/orderbooks/{exchange}/{pair}/{date}/*.parquet
        """
        base_dir = Path(self._config.hftrade_data_dir) / "orderbooks" / self._config.exchange / self._config.pair
        return self._load_parquets_from_dates(base_dir, dates, "订单簿")

    def _load_trade_parquets(self, dates: list[str]) -> pd.DataFrame:
        """加载所有成交 parquet 文件并合并

        路径: {data_dir}/trades/{exchange}/{pair}/{date}/*.parquet
        """
        base_dir = Path(self._config.hftrade_data_dir) / "trades" / self._config.exchange / self._config.pair
        return self._load_parquets_from_dates(base_dir, dates, "成交")

    def _load_parquets_from_dates(
        self,
        base_dir: Path,
        dates: list[str],
        label: str,
    ) -> pd.DataFrame:
        """从多个日期加载 parquet 文件并合并

        支持两种路径格式：
        - 扁平格式: {base_dir}/{date}.parquet（百度网盘下载）
        - 嵌套格式: {base_dir}/{date}/*.parquet（Monitor 直接录制）

        Args:
            base_dir: 数据类型基础目录 (orderbooks/trades 下的 exchange/pair)
            dates: 日期列表
            label: 日志标签

        Returns:
            合并后的 DataFrame，按 timestamp 升序排列
        """
        frames: list[pd.DataFrame] = []

        for date_str in dates:
            # 优先尝试扁平格式: {date}.parquet
            flat_file = base_dir / f"{date_str}.parquet"
            if flat_file.exists():
                try:
                    df = pd.read_parquet(flat_file, use_threads=False)
                    frames.append(df)
                except Exception as e:
                    logger.warning("读取 %s 失败: %s", flat_file, e)
                continue

            # 回退到嵌套格式: {date}/*.parquet
            date_dir = base_dir / date_str
            if not date_dir.exists():
                logger.debug("%s 数据不存在: %s", label, date_str)
                continue

            parquet_files = sorted(date_dir.glob("*.parquet"))
            if not parquet_files:
                logger.debug("%s 目录为空: %s", label, date_dir)
                continue

            for pf in parquet_files:
                try:
                    df = pd.read_parquet(pf, use_threads=False)
                    frames.append(df)
                except Exception as e:
                    logger.warning("读取 %s 失败: %s", pf, e)

        if not frames:
            logger.warning("未找到任何 %s parquet 文件", label)
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        merged.sort_values("timestamp", inplace=True)
        merged.reset_index(drop=True, inplace=True)
        logger.info("已加载 %s: %d 行", label, len(merged))
        return merged

    def _df_to_snapshots(self, df: pd.DataFrame) -> list[OrderbookSnapshot]:
        """将 DataFrame 转为 OrderbookSnapshot 列表（向量化提取）"""
        depth = self._config.ob_depth

        # 构建列名列表
        bid_price_cols: list[str] = [f"bid{i}_price" for i in range(1, depth + 1)]
        bid_amount_cols: list[str] = [f"bid{i}_amount" for i in range(1, depth + 1)]
        ask_price_cols: list[str] = [f"ask{i}_price" for i in range(1, depth + 1)]
        ask_amount_cols: list[str] = [f"ask{i}_amount" for i in range(1, depth + 1)]

        # 向量化提取为 numpy 数组
        timestamps: np.ndarray = df["timestamp"].values.astype(np.int64)
        mid_prices: np.ndarray = df["mid_price"].values.astype(np.float64)
        bid_prices_arr: np.ndarray = df[bid_price_cols].values.astype(np.float64)  # (N, depth)
        bid_amounts_arr: np.ndarray = df[bid_amount_cols].values.astype(np.float64)
        ask_prices_arr: np.ndarray = df[ask_price_cols].values.astype(np.float64)
        ask_amounts_arr: np.ndarray = df[ask_amount_cols].values.astype(np.float64)

        snapshots: list[OrderbookSnapshot] = []
        n = len(df)
        for i in range(n):
            snapshots.append(
                OrderbookSnapshot(
                    timestamp_ms=int(timestamps[i]),
                    mid_price=float(mid_prices[i]),
                    bid_prices=bid_prices_arr[i].copy(),
                    bid_amounts=bid_amounts_arr[i].copy(),
                    ask_prices=ask_prices_arr[i].copy(),
                    ask_amounts=ask_amounts_arr[i].copy(),
                )
            )

        return snapshots

    def _df_to_trades(self, df: pd.DataFrame) -> list[MarketTrade]:
        """将 DataFrame 转为 MarketTrade 列表（向量化提取，过滤无效数据）"""
        contract_size = self._config.contract_size

        # 过滤零价格/零数量的无效成交
        valid_mask = (df["price"] > 0) & (df["amount"] > 0)
        df_valid = df[valid_mask]
        if len(df_valid) < len(df):
            logger.info("过滤无效成交: %d / %d", len(df) - len(df_valid), len(df))

        timestamps: np.ndarray = df_valid["timestamp"].values.astype(np.int64)
        prices: np.ndarray = df_valid["price"].values.astype(np.float64)
        amounts: np.ndarray = df_valid["amount"].values.astype(np.float64) * contract_size
        sides: np.ndarray = df_valid["side"].values  # object array of strings

        trades: list[MarketTrade] = []
        n = len(df_valid)
        for i in range(n):
            trades.append(
                MarketTrade(
                    timestamp_ms=int(timestamps[i]),
                    price=float(prices[i]),
                    amount=float(amounts[i]),
                    side=str(sides[i]),
                )
            )

        return trades
