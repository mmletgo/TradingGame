"""联盟训练对手采样优化功能测试

测试 PFSP 采样、指数衰减 recency、胜率追踪和批量采样等功能。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---- 项目根目录加入 sys.path ----
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ---- Mock 掉触发 Cython 的模块（在 import league 模块前） ----
# 注意：主仓库中有编译好的 .so 文件，需要在 Cython 模块被导入前 mock
# 才能防止类型检查错误
for _mod_name in [
    'src.market.account.position',
    'src.market.account.fast_account',
    'src.market.matching.fast_matching',
    'src.market.orderbook.orderbook',
    'src.training._cython.batch_decide_openmp',
    'src.training._cython.fast_execution',
    'src.bio.brain.fast_network',
    'src.bio.agents._cython.fast_decide',
    'src.bio.agents._cython.fast_observe',
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

from src.config.config import AgentType
from src.training.league.config import LeagueTrainingConfig
from src.training.league.opponent_pool import OpponentPool
from src.training.league.opponent_pool_manager import OpponentPoolManager
from src.training.league.arena_allocator import (
    HybridArenaAllocator,
    HybridSamplingResult,
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _create_test_pool(
    tmp_path: Path,
    entries_config: list[dict],
) -> OpponentPool:
    """创建测试用的 OpponentPool

    直接操作 _index 而非 add_entry（避免磁盘 I/O 和 genome_data 依赖）。

    Args:
        tmp_path: pytest 的 tmp_path fixture
        entries_config: 每个元素包含 entry_id, source_generation, avg_fitness,
                        win_rates, match_counts
    """
    config = LeagueTrainingConfig(
        pool_dir=str(tmp_path),
        milestone_interval=10,
        recency_decay_lambda=2.0,
        pfsp_exponent=2.0,
        pfsp_explore_bonus=2.0,
        pfsp_win_rate_ema_alpha=0.3,
    )
    pool = OpponentPool(AgentType.RETAIL_PRO, tmp_path, config)
    pool._index = pool._create_empty_index()

    for ec in entries_config:
        entry_meta: dict = {
            "entry_id": ec["entry_id"],
            "source": "main_agents",
            "source_generation": ec.get("source_generation", 0),
            "add_reason": "milestone",
            "avg_fitness": ec.get("avg_fitness", 0.0),
            "win_rates": ec.get("win_rates", {}),
            "match_counts": ec.get("match_counts", {}),
            "created_at": "2024-01-01T00:00:00Z",
        }
        pool._index["entries"].append(entry_meta)

    return pool


# ===========================================================================
# 测试类 1: TestWinRateTracking
# ===========================================================================

class TestWinRateTracking:
    """测试胜率追踪 (update_entry_win_rate)"""

    def test_update_entry_win_rate_first_time(self, tmp_path: Path) -> None:
        """首次更新胜率，应直接赋值 outcome"""
        pool = _create_test_pool(tmp_path, [
            {"entry_id": "e1", "source_generation": 0},
        ])
        pool.update_entry_win_rate(
            entry_id="e1",
            target_type=AgentType.MARKET_MAKER,
            outcome=0.7,
            ema_alpha=0.3,
        )

        entry_meta = pool._index["entries"][0]
        # 首次应直接赋值
        assert entry_meta["win_rates"]["vs_MARKET_MAKER"] == pytest.approx(0.7)

    def test_update_entry_win_rate_ema(self, tmp_path: Path) -> None:
        """多次更新胜率，验证 EMA 平滑"""
        pool = _create_test_pool(tmp_path, [
            {"entry_id": "e1", "source_generation": 0},
        ])
        ema_alpha = 0.3

        # 第一次更新: win_rate = 1.0
        pool.update_entry_win_rate("e1", AgentType.MARKET_MAKER, 1.0, ema_alpha)
        # 第二次更新: win_rate = (1-0.3)*1.0 + 0.3*0.0 = 0.7
        pool.update_entry_win_rate("e1", AgentType.MARKET_MAKER, 0.0, ema_alpha)
        # 第三次更新: win_rate = (1-0.3)*0.7 + 0.3*1.0 = 0.49 + 0.3 = 0.79
        pool.update_entry_win_rate("e1", AgentType.MARKET_MAKER, 1.0, ema_alpha)

        entry_meta = pool._index["entries"][0]
        expected = (1.0 - ema_alpha) * ((1.0 - ema_alpha) * 1.0 + ema_alpha * 0.0) + ema_alpha * 1.0
        assert entry_meta["win_rates"]["vs_MARKET_MAKER"] == pytest.approx(expected)

    def test_update_entry_win_rate_match_count(self, tmp_path: Path) -> None:
        """验证 match_count 正确递增"""
        pool = _create_test_pool(tmp_path, [
            {"entry_id": "e1", "source_generation": 0},
        ])

        for _ in range(5):
            pool.update_entry_win_rate("e1", AgentType.RETAIL_PRO, 0.5, 0.3)

        entry_meta = pool._index["entries"][0]
        assert entry_meta["match_counts"]["vs_RETAIL_PRO"] == 5


# ===========================================================================
# 测试类 2: TestExponentialDecayRecency
# ===========================================================================

class TestExponentialDecayRecency:
    """测试指数衰减 recency 权重"""

    def test_exponential_decay_weights(self, tmp_path: Path) -> None:
        """验证指数衰减权重：越新的对手权重越高，且符合 exp(-lambda * delta / milestone) 公式"""
        entries_config = [
            {"entry_id": "old", "source_generation": 0},
            {"entry_id": "mid", "source_generation": 50},
            {"entry_id": "new", "source_generation": 100},
        ]
        pool = _create_test_pool(tmp_path, entries_config)

        current_gen = 100
        decay_lambda = pool.config.recency_decay_lambda   # 2.0
        milestone = pool.config.milestone_interval          # 10

        pool._rebuild_cache_if_needed(current_gen)

        assert pool._entry_ids_cache is not None
        assert pool._prob_cache is not None

        # 手动计算期望权重
        raw_weights = np.array([
            math.exp(-decay_lambda * max(0, current_gen - gen) / milestone)
            for gen in [0, 50, 100]
        ])
        expected_probs = raw_weights / raw_weights.sum()

        np.testing.assert_allclose(pool._prob_cache, expected_probs, rtol=1e-10)

        # 越新的对手权重越高
        assert pool._prob_cache[2] > pool._prob_cache[1] > pool._prob_cache[0]

    def test_recency_cache_invalidation_on_generation_change(self, tmp_path: Path) -> None:
        """验证代数变化时缓存失效并重建

        注意：归一化概率在 delta 整体平移时保持不变（因 exp(-a-c)/exp(-b-c)=exp(-a)/exp(-b)）。
        要让概率发生变化，需要有条目的 source_generation > current_gen_old 但 <= current_gen_new，
        使得 max(0, ...) 的 clamp 行为在两个 generation 上不同。
        """
        # entry "future": source_generation=20 > current_gen_old=10
        # 在 gen=10 时，delta = max(0, 10-20) = 0（被 clamp）
        # 在 gen=50 时，delta = max(0, 50-20) = 30（正常衰减）
        entries_config = [
            {"entry_id": "past", "source_generation": 0},
            {"entry_id": "future", "source_generation": 20},
        ]
        pool = _create_test_pool(tmp_path, entries_config)

        # 第一次构建缓存 (generation=10)
        pool._rebuild_cache_if_needed(10)
        prob_gen10 = pool._prob_cache.copy()
        assert pool._last_cached_generation == 10

        # 代数变化 (generation=50)，缓存应重建
        pool._rebuild_cache_if_needed(50)
        prob_gen50 = pool._prob_cache.copy()
        assert pool._last_cached_generation == 50

        # 两次缓存的概率应不同
        # gen=10: past delta=10 → clamp(0), future delta=max(0,-10)=0 → 两个权重都是 exp(0)=1
        #   等等，past delta = max(0, 10-0) = 10, future delta = max(0, 10-20)=0
        #   weights: [exp(-2*10/10), exp(0)] = [exp(-2), 1]
        # gen=50: past delta=50, future delta=30
        #   weights: [exp(-2*50/10), exp(-2*30/10)] = [exp(-10), exp(-6)]
        # 归一化后确实不同
        assert not np.allclose(prob_gen10, prob_gen50)


# ===========================================================================
# 测试类 3: TestPFSPSampling
# ===========================================================================

class TestPFSPSampling:
    """测试 PFSP 采样策略"""

    def test_pfsp_prefers_hard_opponents(self, tmp_path: Path) -> None:
        """胜率低的对手应有更高采样概率"""
        entries_config = [
            {
                "entry_id": "easy",
                "source_generation": 100,
                "win_rates": {"vs_MARKET_MAKER": 0.9},       # 目标物种 vs easy 胜率高 → 容易
                "match_counts": {"vs_MARKET_MAKER": 10},
            },
            {
                "entry_id": "hard",
                "source_generation": 100,
                "win_rates": {"vs_MARKET_MAKER": 0.1},       # 目标物种 vs hard 胜率低 → 难
                "match_counts": {"vs_MARKET_MAKER": 10},
            },
        ]
        pool = _create_test_pool(tmp_path, entries_config)

        weights = pool.compute_weights(
            pool._index["entries"],
            strategy='pfsp',
            target_type=AgentType.MARKET_MAKER,
            current_generation=100,
        )

        # hard 对手权重应更高（目标胜率低 → (1 - win_rate)^p 更大）
        assert weights[1] > weights[0], (
            f"hard 对手的权重 ({weights[1]}) 应大于 easy 对手 ({weights[0]})"
        )

    def test_pfsp_exploration_bonus(self, tmp_path: Path) -> None:
        """未交战对手应有高探索权重"""
        entries_config = [
            {
                "entry_id": "explored",
                "source_generation": 100,
                "win_rates": {"vs_MARKET_MAKER": 0.5},
                "match_counts": {"vs_MARKET_MAKER": 100},
            },
            {
                "entry_id": "unexplored",
                "source_generation": 100,
                "win_rates": {},
                "match_counts": {},
            },
        ]
        pool = _create_test_pool(tmp_path, entries_config)

        weights = pool.compute_weights(
            pool._index["entries"],
            strategy='pfsp',
            target_type=AgentType.MARKET_MAKER,
            current_generation=100,
        )

        # 未交战对手：f_win=1.0, exploration_bonus=explore_bonus_coeff=2.0
        # 已交战对手：f_win=(1-0.5)^2=0.25, exploration_bonus=max(1, 2/sqrt(101))≈1.0
        # 未交战对手权重应显著更高
        assert weights[1] > weights[0], (
            f"未交战对手权重 ({weights[1]}) 应大于已交战对手 ({weights[0]})"
        )

    def test_pfsp_without_target_type_falls_back_to_recency(self, tmp_path: Path) -> None:
        """target_type=None 时退化为 recency 权重"""
        entries_config = [
            {"entry_id": "old", "source_generation": 0},
            {"entry_id": "new", "source_generation": 100},
        ]
        pool = _create_test_pool(tmp_path, entries_config)

        # target_type=None 时应退化为 recency
        pfsp_weights = pool.compute_weights(
            pool._index["entries"],
            strategy='pfsp',
            target_type=None,
            current_generation=100,
        )
        recency_weights = pool.compute_weights(
            pool._index["entries"],
            strategy='recency',
            target_type=None,
            current_generation=100,
        )

        np.testing.assert_allclose(pfsp_weights, recency_weights)


# ===========================================================================
# 测试类 4: TestBatchSampling
# ===========================================================================

class TestBatchSampling:
    """测试批量采样"""

    def test_batch_sampling_no_duplicates(self, tmp_path: Path) -> None:
        """pool >= n 时不应有重复"""
        entries_config = [
            {"entry_id": f"e{i}", "source_generation": i * 10}
            for i in range(10)
        ]
        pool = _create_test_pool(tmp_path, entries_config)

        # 采样 5 个（pool_size=10 >= n=5）
        result = pool.sample_opponents_batch(
            n=5,
            strategy='recency',
            current_generation=90,
        )

        assert len(result) == 5
        assert len(set(result)) == 5, f"采样结果不应有重复: {result}"

    def test_batch_sampling_small_pool(self, tmp_path: Path) -> None:
        """pool < n 时每个对手至少出现 n//pool_size 次"""
        entries_config = [
            {"entry_id": f"e{i}", "source_generation": i * 10}
            for i in range(3)
        ]
        pool = _create_test_pool(tmp_path, entries_config)

        n = 10
        pool_size = 3
        result = pool.sample_opponents_batch(
            n=n,
            strategy='uniform',
            current_generation=30,
        )

        assert len(result) == n

        # 每个对手至少出现 n // pool_size = 3 次
        from collections import Counter
        counts = Counter(result)
        min_expected = n // pool_size  # 3

        for entry_id in [f"e{i}" for i in range(3)]:
            assert counts[entry_id] >= min_expected, (
                f"对手 {entry_id} 出现 {counts[entry_id]} 次，"
                f"应至少 {min_expected} 次"
            )



# ===========================================================================
# 测试类 5: TestComputeWeights
# ===========================================================================

class TestComputeWeights:
    """测试 compute_weights 方法"""

    def test_uniform_weights(self, tmp_path: Path) -> None:
        """均匀策略应返回全 1"""
        entries_config = [
            {"entry_id": f"e{i}", "source_generation": i * 10}
            for i in range(5)
        ]
        pool = _create_test_pool(tmp_path, entries_config)

        weights = pool.compute_weights(
            pool._index["entries"],
            strategy='uniform',
            target_type=None,
            current_generation=50,
        )

        np.testing.assert_allclose(weights, np.ones(5))

    def test_recency_weights_decay(self, tmp_path: Path) -> None:
        """验证 recency 权重随代差递减"""
        entries_config = [
            {"entry_id": "gen0", "source_generation": 0},
            {"entry_id": "gen50", "source_generation": 50},
            {"entry_id": "gen100", "source_generation": 100},
        ]
        pool = _create_test_pool(tmp_path, entries_config)

        current_gen = 100
        decay_lambda = pool.config.recency_decay_lambda   # 2.0
        milestone = pool.config.milestone_interval          # 10

        weights = pool.compute_weights(
            pool._index["entries"],
            strategy='recency',
            target_type=None,
            current_generation=current_gen,
        )

        # 手动计算期望值
        expected = np.array([
            math.exp(-decay_lambda * 100 / milestone),  # gen0 → delta=100
            math.exp(-decay_lambda * 50 / milestone),   # gen50 → delta=50
            math.exp(-decay_lambda * 0 / milestone),    # gen100 → delta=0
        ])

        np.testing.assert_allclose(weights, expected, rtol=1e-10)

        # 越新权重越高
        assert weights[2] > weights[1] > weights[0]

    def test_pfsp_weights_with_win_rates(self, tmp_path: Path) -> None:
        """验证 PFSP 权重综合了胜率、recency 和探索奖励"""
        entries_config = [
            {
                "entry_id": "old_easy",
                "source_generation": 0,
                "win_rates": {"vs_RETAIL_PRO": 0.8},
                "match_counts": {"vs_RETAIL_PRO": 50},
            },
            {
                "entry_id": "new_hard",
                "source_generation": 100,
                "win_rates": {"vs_RETAIL_PRO": 0.2},
                "match_counts": {"vs_RETAIL_PRO": 5},
            },
            {
                "entry_id": "new_unknown",
                "source_generation": 100,
                "win_rates": {},
                "match_counts": {},
            },
        ]
        pool = _create_test_pool(tmp_path, entries_config)

        current_gen = 100
        decay_lambda = pool.config.recency_decay_lambda     # 2.0
        milestone = pool.config.milestone_interval            # 10
        pfsp_exp = pool.config.pfsp_exponent                  # 2.0
        explore_bonus_coeff = pool.config.pfsp_explore_bonus  # 2.0

        weights = pool.compute_weights(
            pool._index["entries"],
            strategy='pfsp',
            target_type=AgentType.RETAIL_PRO,
            current_generation=current_gen,
        )

        assert len(weights) == 3

        # 手动计算每个条目的期望权重

        # old_easy: delta=100, win_rate=0.8
        recency_0 = math.exp(-decay_lambda * 100 / milestone)
        f_win_0 = (1.0 - 0.8) ** pfsp_exp  # 0.04
        explore_0 = max(1.0, explore_bonus_coeff / math.sqrt(50 + 1))
        expected_0 = f_win_0 * recency_0 * explore_0

        # new_hard: delta=0, win_rate=0.2
        recency_1 = math.exp(-decay_lambda * 0 / milestone)  # 1.0
        f_win_1 = (1.0 - 0.2) ** pfsp_exp  # 0.64
        explore_1 = max(1.0, explore_bonus_coeff / math.sqrt(5 + 1))
        expected_1 = f_win_1 * recency_1 * explore_1

        # new_unknown: delta=0, win_rate=None → f_win=1.0, explore=explore_bonus_coeff
        recency_2 = math.exp(-decay_lambda * 0 / milestone)  # 1.0
        f_win_2 = 1.0
        explore_2 = explore_bonus_coeff  # 2.0
        expected_2 = f_win_2 * recency_2 * explore_2

        np.testing.assert_allclose(
            weights,
            [expected_0, expected_1, expected_2],
            rtol=1e-10,
        )

        # new_unknown 权重最高（未探索 + 最新）
        assert weights[2] > weights[1] > weights[0], (
            f"权重排序应为 new_unknown > new_hard > old_easy, "
            f"实际: {weights}"
        )
