# CLAUDE.md - src/core/

## 模块概述

`src/core/` 目录是项目的核心引擎模块，目前主要包含统一的日志管理系统。该模块为整个项目提供标准化的日志服务，确保日志输出的一致性和可管理性。

## 目录结构

```
src/core/
├── __init__.py           # 模块初始化文件（空）
├── CLAUDE.md             # 本文档
├── event_engine/         # 事件引擎子模块（保留，暂未实现）
└── log_engine/           # 日志引擎子模块
    ├── __init__.py       # 导出 setup_logging 函数
    └── logger.py         # 日志配置和工具函数实现
```

## 日志系统 (log_engine/)

### 设计理念

日志系统采用"双层输出"设计：
1. **控制台输出**：仅显示警告（WARNING）及以上级别的日志，保持控制台清爽
2. **文件输出**：记录信息（INFO）及以上级别的详细日志，便于问题追踪和调试

### 核心函数

#### `setup_logging(log_dir: str = "logs", console_level: int = logging.WARNING) -> None`

初始化全局日志系统。

**参数说明：**
- `log_dir`: 日志文件存储目录，默认为 `"logs"`
- `console_level`: 控制台日志级别，默认为 `logging.WARNING`

**返回值：**
- `None`

**异常：**
- `OSError`: 如果日志目录创建失败

**功能特性：**
- 自动创建日志目录（如不存在）
- 配置根日志器（root logger）级别为 INFO
- 添加两个处理器：
  - **文件处理器**：记录 INFO 及以上级别到 `logs/trading_game.log`（UTF-8 编码）
  - **控制台处理器**：仅记录 WARNING 及以上级别到标准输出
- 防止重复添加处理器（多次调用自动跳过）
- 统一的日志格式：`"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`

**使用场景：**
在所有训练脚本的主入口处调用，且必须在导入其他项目模块之前调用。

**示例：**
```python
from src.core.log_engine.logger import setup_logging

# 在脚本开始时初始化日志系统
setup_logging(log_dir="logs", console_level=logging.WARNING)

# 也可以自定义控制台级别
setup_logging(log_dir="logs", console_level=logging.ERROR)  # 只显示错误
setup_logging(log_dir="logs", console_level=logging.INFO)   # 显示更多信息
```

#### `get_logger(name: str) -> logging.Logger`

获取指定名称的日志器实例。

**参数说明：**
- `name`: 日志器名称，通常使用模块的 `__name__` 变量

**返回值：**
- `logging.Logger` 实例

**功能特性：**
- 相同名称多次调用返回同一个 Logger 实例（单例模式）
- 继承根日志器的配置（文件处理器和控制台处理器）
- 支持层级命名（如 `"training.population"` 会继承 `"training"` 的配置）

**使用场景：**
在各个模块中获取日志器，用于记录模块级别的日志。

**示例：**
```python
from src.core.log_engine.logger import get_logger

class Population:
    def __init__(self):
        # 使用模块名作为日志器名称
        self.logger = get_logger(__name__)

    def evolve(self):
        self.logger.info("开始种群进化")
        # ... 业务逻辑 ...
        self.logger.warning("种群数量低于阈值")
```

### 日志级别

Python 标准日志级别（从低到高）：

| 级别 | 数值 | 用途 | 输出位置 |
|------|------|------|----------|
| DEBUG | 10 | 调试信息，仅在开发时使用 | 仅文件 |
| INFO | 20 | 常规信息，记录程序运行状态 | 文件 + 控制台（当配置为 INFO 时） |
| WARNING | 30 | 警告信息，表示可能出现问题 | 文件 + 控制台 |
| ERROR | 40 | 错误信息，程序遇到错误但仍可继续 | 文件 + 控制台 |
| CRITICAL | 50 | 严重错误，程序可能无法继续 | 文件 + 控制台 |

**当前配置：**
- 根日志器（root logger）：INFO 级别
- 文件处理器：INFO 及以上
- 控制台处理器：WARNING 及以上（默认）

### 日志格式

所有日志采用统一格式：
```
2026-01-18 12:35:42 - training.population - INFO - 创建 retail 种群，初始 Agent 数量: 50
```

格式说明：
- `%(asctime)s`: 时间戳（`YYYY-MM-DD HH:MM:SS` 格式）
- `%(name)s`: 日志器名称（通常是模块路径）
- `%(levelname)s`: 日志级别（INFO/WARNING/ERROR 等）
- `%(message)s`: 日志消息内容

## 使用指南

### 1. 在脚本中初始化日志系统

所有训练脚本的主入口处都必须首先调用 `setup_logging()`：

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.core.log_engine.logger import setup_logging

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    # 初始化日志系统（必须在导入其他项目模块之前）
    setup_logging(args.log_dir)

    # 现在可以安全地导入其他模块
    from src.training.trainer import Trainer
    # ... 其余代码 ...
```

### 2. 在模块中使用日志器

推荐在类的 `__init__` 方法中初始化日志器：

```python
from src.core.log_engine.logger import get_logger

class MatchingEngine:
    def __init__(self):
        self._logger = get_logger(__name__)
        self._logger.info("撮合引擎初始化完成")

    def match_order(self, order):
        if order.price <= 0:
            self._logger.warning(f"订单价格异常: {order.price}")
            return None

        self._logger.debug(f"开始撮合订单: {order.order_id}")
        # ... 撮合逻辑 ...
        self._logger.info(f"订单撮合完成: {order.order_id}")
```

### 3. 日志命名规范

- **类级别日志**：使用 `__name__`（模块路径）
  ```python
  self.logger = get_logger(__name__)  # 例如: "training.population"
  ```

- **功能模块日志**：使用简短的功能名称
  ```python
  self.logger = get_logger("population")  # 种群管理专用日志
  self.logger = get_logger("trainer")     # 训练器专用日志
  ```

- **特殊组件日志**：使用组件名称
  ```python
  self.logger = get_logger("adl")        # ADL 管理器日志
  ```

### 4. 日志级别选择原则

| 场景 | 推荐级别 | 示例 |
|------|----------|------|
| 程序正常流程的关键节点 | INFO | 种群创建完成、Episode 开始/结束 |
| 潜在问题或异常情况 | WARNING | 种群进化失败、价格档位不一致 |
| 运行时错误（可恢复） | ERROR | 撮合失败、订单处理异常 |
| 开发调试信息 | DEBUG | 详细的中间计算结果 |

### 5. 日志文件管理

- **默认路径**：`logs/trading_game.log`
- **编码格式**：UTF-8
- **追加模式**：每次运行追加到现有日志文件（不清空）
- **日志轮转**：当前未实现，建议定期手动清理或使用外部工具（如 logrotate）

## 项目中的实际应用

### 训练脚本 (scripts/train_noui.py)

```python
from src.core.log_engine.logger import setup_logging

def main():
    args = parser.parse_args()
    setup_logging(args.log_dir)  # 初始化日志系统

    print("NEAT AI 交易模拟竞技场 - 无 UI 训练模式")
    # ... 训练逻辑 ...
```

### 种群管理 (src/training/population.py)

```python
from src.core.log_engine.logger import get_logger

class Population:
    def __init__(self, agent_type, config):
        self.logger = get_logger("population")
        # ...

    def evolve(self):
        self.logger.info(f"{self.agent_type.value} 种群完成第 {self.generation} 代进化")
```

### 训练器 (src/training/trainer.py)

```python
from src.core.log_engine.logger import get_logger

class Trainer:
    def __init__(self, config):
        self.logger = get_logger("trainer")
        # ...

    def run_episode(self):
        self.logger.info(f"鲶鱼已启用: 多模式（三种鲶鱼同时运行）")
```

### 撮合引擎 (src/market/matching/matching_engine.py)

```python
from src.core.log_engine.logger import get_logger

class MatchingEngine:
    def __init__(self):
        self._logger = get_logger(__name__)
        self._logger.info("撮合引擎初始化完成")

    def _match_limit_order(self, order, side_book):
        self._logger.warning(f"价格档位不一致: best_price={best_price} 不在 side_book 中")
```

### ADL 管理器 (src/market/adl/adl_manager.py)

```python
from src.core.log_engine.logger import get_logger

class ADLManager:
    def __init__(self):
        self.logger = get_logger("adl")
```

## 测试覆盖

日志系统拥有完整的单元测试（`tests/core/test_logger.py`），覆盖以下场景：

1. **目录和文件创建**：验证自动创建日志目录和日志文件
2. **日志级别配置**：验证控制台和文件处理器的日志级别
3. **防重复处理器**：验证多次调用不会重复添加处理器
4. **日志写入**：验证日志正确写入文件
5. **Logger 实例管理**：验证同名返回相同实例，不同名返回不同实例

运行测试：
```bash
pytest tests/core/test_logger.py
```

## 注意事项

1. **初始化顺序**：必须在导入其他项目模块之前调用 `setup_logging()`，否则这些模块可能无法正确获取日志配置

2. **线程安全**：Python 的 `logging` 模块是线程安全的，可以在多线程环境中使用（本项目使用线程池并行创建 Agent）

3. **性能影响**：日志写入文件会有轻微性能开销，但本项目配置为 WARNING 以上级别才输出到控制台，影响较小

4. **日志文件大小**：长时间训练会生成大量日志，建议定期清理或实现日志轮转机制

5. **编码问题**：日志文件使用 UTF-8 编码，确保支持中文等非 ASCII 字符

6. **刷新缓冲区**：在程序异常退出前，建议手动刷新和关闭日志处理器以确保所有日志写入磁盘：
   ```python
   import logging
   for handler in logging.getLogger().handlers:
       handler.flush()
       handler.close()
   ```

## 扩展建议

未来可能的改进方向：

1. **日志轮转**：使用 `RotatingFileHandler` 实现按文件大小或时间自动轮转
2. **日志过滤**：为特定模块设置不同的日志级别
3. **结构化日志**：使用 JSON 格式便于日志分析工具处理
4. **异步日志**：使用 `QueueHandler` 减少主线程阻塞
5. **日志收集**：集成 ELK（Elasticsearch + Logstash + Kibana）等日志分析平台

## 相关文档

- Python logging 模块文档：https://docs.python.org/3/library/logging.html
- 项目根目录 CLAUDE.md：`/home/rongheng/python_project/TradingGame_origin/CLAUDE.md`
- 训练模块文档：`/home/rongheng/python_project/TradingGame_origin/src/training/CLAUDE.md`
- 市场模块文档：`/home/rongheng/python_project/TradingGame_origin/src/market/CLAUDE.md`
