"""多代网络缓存管理器"""
from __future__ import annotations

from typing import Any

from src.config.config import AgentType, Config
from src.training.league.opponent_pool_manager import OpponentPoolManager


class MultiGenerationNetworkCache:
    """多代网络缓存管理器

    按 Agent 类型和 entry_id 管理网络缓存。
    支持当前代和历史代的网络缓存。
    """

    def __init__(self, config: Config, max_cached_per_type: int = 5) -> None:
        """初始化

        Args:
            config: 全局配置
            max_cached_per_type: 每种类型最多缓存的历史代数量
        """
        self.config = config
        self.max_cached_per_type = max_cached_per_type

        # 当前代缓存：{AgentType: BatchNetworkCache}
        self.current_caches: dict[AgentType, Any] = {}

        # 历史代缓存：{AgentType: {entry_id: BatchNetworkCache}}
        self.historical_caches: dict[AgentType, dict[str, Any]] = {
            agent_type: {} for agent_type in AgentType
        }

        # LRU 访问顺序：{AgentType: [entry_id, ...]}（最近访问的在后面）
        self._access_order: dict[AgentType, list[str]] = {
            agent_type: [] for agent_type in AgentType
        }

    def ensure_cached(
        self,
        agent_type: AgentType,
        entry_id: str,
        pool_manager: OpponentPoolManager,
    ) -> bool:
        """确保指定类型的历史对手网络已缓存

        Args:
            agent_type: Agent 类型
            entry_id: 条目 ID
            pool_manager: 对手池管理器

        Returns:
            是否成功缓存
        """
        # 已在缓存中
        if entry_id in self.historical_caches[agent_type]:
            self._update_access_order(agent_type, entry_id)
            return True

        # 从对手池加载
        entry = pool_manager.get_entry(agent_type, entry_id, load_networks=True)
        if entry is None or entry.network_data is None:
            return False

        # 创建 BatchNetworkCache
        cache = self._create_cache_from_network_data(agent_type, entry.network_data)
        if cache is None:
            return False

        # 【内存泄漏修复】缓存创建成功后，清理 entry 的大数据字段
        # get_entry() 加载的 genome_data/network_data 已不再需要
        entry.genome_data = None
        entry.network_data = None

        # 检查是否需要淘汰旧缓存
        self._evict_if_needed(agent_type)

        # 添加到缓存
        self.historical_caches[agent_type][entry_id] = cache
        self._update_access_order(agent_type, entry_id)

        return True

    def _update_access_order(self, agent_type: AgentType, entry_id: str) -> None:
        """更新 LRU 访问顺序"""
        order = self._access_order[agent_type]
        if entry_id in order:
            order.remove(entry_id)
        order.append(entry_id)

    def _evict_if_needed(self, agent_type: AgentType) -> None:
        """如果超过限制，淘汰最久未访问的缓存"""
        while len(self.historical_caches[agent_type]) >= self.max_cached_per_type:
            order = self._access_order[agent_type]
            if not order:
                break

            # 淘汰最久未访问的
            oldest_id = order.pop(0)
            if oldest_id in self.historical_caches[agent_type]:
                cache = self.historical_caches[agent_type].pop(oldest_id)
                # 【内存泄漏修复】显式清理缓存数据
                if hasattr(cache, 'clear'):
                    cache.clear()
                del cache

    def _create_cache_from_network_data(
        self,
        agent_type: AgentType,
        network_data: dict[int, tuple],
    ) -> Any:
        """从网络数据创建 BatchNetworkCache

        Args:
            agent_type: Agent 类型
            network_data: {sub_pop_id: network_params_tuple}

        Returns:
            BatchNetworkCache 实例
        """
        try:
            from src.training._cython.batch_decide_openmp import BatchNetworkCache
            from src.training.population import _concat_network_params_numpy

            # cdef 常量无法从 Python 层导入，与 trainer.py/parallel_arena_trainer.py 保持一致
            CACHE_TYPE_FULL = 1
            CACHE_TYPE_MARKET_MAKER = 2

            # 合并所有子种群的 packed numpy 参数
            params_list: list[tuple] = [
                network_data[sub_pop_id]
                for sub_pop_id in sorted(network_data.keys())
            ]
            merged_params = _concat_network_params_numpy(params_list)

            # headers_arr 是第一个元素
            headers_arr = merged_params[0]
            num_networks = len(headers_arr)
            if num_networks == 0:
                return None

            # 确定 cache_type
            if agent_type == AgentType.MARKET_MAKER:
                cache_type = CACHE_TYPE_MARKET_MAKER
            else:
                cache_type = CACHE_TYPE_FULL

            # 获取 OpenMP 线程数
            num_threads = self.config.training.openmp_threads

            # 创建缓存
            cache = BatchNetworkCache(num_networks, cache_type, num_threads)

            # 使用 update_networks_from_numpy 填充数据
            # merged_params = (headers, input_keys, output_keys, node_ids,
            #                  biases, responses, act_types,
            #                  conn_indptr, conn_sources, conn_weights, output_indices)
            cache.update_networks_from_numpy(
                merged_params[0],   # headers_arr
                merged_params[1],   # all_input_keys
                merged_params[2],   # all_output_keys
                merged_params[3],   # all_node_ids
                merged_params[4],   # all_biases
                merged_params[5],   # all_responses
                merged_params[6],   # all_act_types
                merged_params[7],   # all_conn_indptr
                merged_params[8],   # all_conn_sources
                merged_params[9],   # all_conn_weights
                merged_params[10],  # all_output_indices
            )

            # 清理中间变量
            del params_list, merged_params

            return cache

        except ImportError:
            return None
        except Exception:
            import logging
            logging.getLogger("multi_gen_cache").exception(
                f"创建 {agent_type} 历史缓存失败"
            )
            return None

    def get_cache(
        self,
        agent_type: AgentType,
        source: str,
        entry_id: str | None = None,
    ) -> Any | None:
        """获取缓存

        Args:
            agent_type: Agent 类型
            source: 来源 ('current', 'historical')
            entry_id: 条目 ID（historical 时需要）

        Returns:
            BatchNetworkCache 实例或 None
        """
        if source == 'current':
            return self.current_caches.get(agent_type)
        elif source == 'historical':
            if entry_id is None:
                return None
            cache = self.historical_caches[agent_type].get(entry_id)
            if cache is not None:
                self._update_access_order(agent_type, entry_id)
            return cache
        else:
            return None

    def update_current(self, agent_type: AgentType, cache: Any) -> None:
        """更新当前代缓存

        Args:
            agent_type: Agent 类型
            cache: BatchNetworkCache 实例
        """
        # 清理旧缓存
        if agent_type in self.current_caches:
            old_cache = self.current_caches[agent_type]
            if hasattr(old_cache, 'clear'):
                old_cache.clear()

        self.current_caches[agent_type] = cache

    def clear_all(self) -> None:
        """清空所有缓存"""
        # 清空当前代缓存
        for cache in self.current_caches.values():
            if hasattr(cache, 'clear'):
                cache.clear()
        self.current_caches.clear()

        # 清空历史代缓存
        for type_caches in self.historical_caches.values():
            for cache in type_caches.values():
                if hasattr(cache, 'clear'):
                    cache.clear()
            type_caches.clear()

        # 清空 LRU 顺序
        for order in self._access_order.values():
            order.clear()

    def get_cached_entry_ids(self, agent_type: AgentType) -> list[str]:
        """获取指定类型已缓存的条目 ID

        Args:
            agent_type: Agent 类型

        Returns:
            条目 ID 列表
        """
        return list(self.historical_caches[agent_type].keys())
