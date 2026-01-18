"""多代网络缓存管理器"""
from __future__ import annotations

from typing import Any

from src.config.config import AgentType, Config
from src.training.league.opponent_pool_manager import OpponentPoolManager


class MultiGenerationNetworkCache:
    """多代网络缓存管理器

    按 Agent 类型和 entry_id 管理网络缓存。
    支持当前代、历史代和 Exploiter 的网络缓存。
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

        # Exploiter 缓存：{AgentType: BatchNetworkCache}
        self.league_exploiter_caches: dict[AgentType, Any] = {}
        self.main_exploiter_caches: dict[AgentType, Any] = {}

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
                # 清理缓存
                if hasattr(cache, 'clear'):
                    cache.clear()

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
            from src.training.population import _unpack_network_params_numpy

            # 合并所有子种群的网络参数
            all_params_list: list[dict] = []

            for sub_pop_id in sorted(network_data.keys()):
                params_tuple = network_data[sub_pop_id]
                # 解包网络参数
                params_list = _unpack_network_params_numpy(*params_tuple)
                all_params_list.extend(params_list)

            # 创建缓存
            cache = BatchNetworkCache(agent_type)

            # 需要从参数列表创建网络对象
            # 这里假设有一个辅助函数来完成这个工作
            # 实际实现中可能需要调整
            networks = self._create_networks_from_params(agent_type, all_params_list)
            cache.update_networks(networks)

            return cache

        except ImportError:
            return None
        except Exception:
            return None

    def _create_networks_from_params(
        self,
        agent_type: AgentType,
        params_list: list[dict],
    ) -> list[Any]:
        """从参数列表创建网络对象

        Args:
            agent_type: Agent 类型
            params_list: 网络参数字典列表

        Returns:
            网络对象列表
        """
        try:
            from neat.nn import FastFeedForwardNetwork

            networks: list[Any] = []
            for params in params_list:
                # 从参数创建 FastFeedForwardNetwork
                network = FastFeedForwardNetwork.from_params(
                    params['num_inputs'],
                    params['num_outputs'],
                    params['node_ids'],
                    params['biases'],
                    params['responses'],
                    params['act_types'],
                    params['conn_indptr'],
                    params['conn_sources'],
                    params['conn_weights'],
                    params['output_indices'],
                )
                networks.append(network)

            return networks

        except ImportError:
            return []
        except Exception:
            return []

    def get_cache(
        self,
        agent_type: AgentType,
        source: str,
        entry_id: str | None = None,
    ) -> Any | None:
        """获取缓存

        Args:
            agent_type: Agent 类型
            source: 来源 ('current', 'historical', 'league_exploiter', 'main_exploiter')
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
        elif source == 'league_exploiter':
            return self.league_exploiter_caches.get(agent_type)
        elif source == 'main_exploiter':
            return self.main_exploiter_caches.get(agent_type)
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

    def update_exploiter(
        self,
        agent_type: AgentType,
        role: str,
        cache: Any,
    ) -> None:
        """更新 Exploiter 缓存

        Args:
            agent_type: Agent 类型
            role: 角色 ('league_exploiter', 'main_exploiter')
            cache: BatchNetworkCache 实例
        """
        if role == 'league_exploiter':
            if agent_type in self.league_exploiter_caches:
                old_cache = self.league_exploiter_caches[agent_type]
                if hasattr(old_cache, 'clear'):
                    old_cache.clear()
            self.league_exploiter_caches[agent_type] = cache
        elif role == 'main_exploiter':
            if agent_type in self.main_exploiter_caches:
                old_cache = self.main_exploiter_caches[agent_type]
                if hasattr(old_cache, 'clear'):
                    old_cache.clear()
            self.main_exploiter_caches[agent_type] = cache

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

        # 清空 Exploiter 缓存
        for cache in self.league_exploiter_caches.values():
            if hasattr(cache, 'clear'):
                cache.clear()
        self.league_exploiter_caches.clear()

        for cache in self.main_exploiter_caches.values():
            if hasattr(cache, 'clear'):
                cache.clear()
        self.main_exploiter_caches.clear()

    def get_cached_entry_ids(self, agent_type: AgentType) -> list[str]:
        """获取指定类型已缓存的条目 ID

        Args:
            agent_type: Agent 类型

        Returns:
            条目 ID 列表
        """
        return list(self.historical_caches[agent_type].keys())
