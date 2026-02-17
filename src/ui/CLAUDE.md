# UI模块

## 模块职责

UI模块提供训练可视化的完整解决方案，包括数据采集、线程控制、DearPyGui界面组件和两种运行模式（训练/演示）。负责将训练器内部数据转换为UI展示所需的格式，并管理训练线程与UI线程的交互。

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                   TrainingUIApp / DemoUIApp             │
│                    (DearPyGui 主窗口)                    │
├─────────────┬───────────────────────────┬───────────────┤
│             │                           │               │
│ControlPanel │                           │               │
│─────────────│                           │               │
│             │        ChartPanel         │               │
│OrderBookPanel│        (1360px宽)         │ TradesPanel   │
│ (280px宽)   │                           │  (280px宽)    │
│             │  - 价格曲线               │               │
│ - 深度图    │  - 存活个体平均资产曲线   │ - 成交记录    │
│ - 盘口列表  │  - 小提琴图               │ (最近30笔)    │
└─────────────┴───────────────────────────┴───────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │   UIController      │
              │  (后台线程控制)      │
              │  - 训练/演示循环    │
              │  - 暂停/恢复/停止   │
              │  - 速度控制         │
              │  - 数据队列管理     │
              └─────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  UIDataCollector    │
              │  (数据采集器)        │
              │  - 订单簿深度       │
              │  - 种群统计         │
              │  - 成交记录         │
              │  - 噪声交易者数据   │
              │  - 历史缓冲区       │
              └─────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │      Trainer        │
              │   (训练器核心)       │
              └─────────────────────┘
```

## 目录结构

```
src/ui/
├── __init__.py              # 模块导出
├── data_collector.py        # UI数据采集器
├── ui_controller.py         # UI控制器（线程管理）
├── training_app.py          # 训练模式UI应用
├── demo_app.py              # 演示模式UI应用
└── components/              # UI组件目录
    ├── __init__.py          # 组件导出
    ├── orderbook_panel.py   # 订单簿面板
    ├── chart_panel.py       # 图表面板
    ├── trades_panel.py      # 成交记录面板
    ├── control_panel.py     # 控制面板
    └── CLAUDE.md            # 组件详细文档
```

## 核心模块

### UIDataCollector (data_collector.py)

UI数据采集器，每tick从训练器收集数据，维护历史缓冲区。

**数据类：**

| 类名 | 用途 | 字段 |
|------|------|------|
| `TradeInfo` | 成交信息 | tick, price, quantity, is_buyer_taker |
| `NoiseTraderInfo` | 噪声交易者信息 | name, equity, position_qty, position_value, initial_balance, is_liquidated |
| `PopulationStats` | 种群统计 | avg_equity, total_equity, max_equity, min_equity, alive_count, total_count, generation, alive_equities |
| `UIDataSnapshot` | UI数据快照 | tick, episode, last_price, mid_price, bids, asks, recent_trades, population_stats, price_history, equity_history, alive_equity_history, noise_trader_data, noise_trader_equity_history |

**属性：**
- `history_length: int` - 历史数据长度限制
- `price_history: deque[float]` - 价格历史缓冲区（存储 orderbook.last_price，与盘口一致）
- `equity_history: dict[AgentType, deque[float]]` - 各种群所有个体平均资产历史缓冲区
- `alive_equity_history: dict[AgentType, deque[float]]` - 各种群存活个体平均资产历史缓冲区
- `noise_trader_equity_history: list[deque[float]]` - 噪声交易者的净值历史缓冲区

**方法：**
- `collect_tick_data(trainer) -> UIDataSnapshot` - 收集当前tick数据快照
- `reset() -> None` - 重置历史数据（新episode开始时调用）

**私有方法：**
- `_compute_population_stats(population, current_price) -> PopulationStats` - 向量化计算种群统计（使用NumPy）
- `_collect_noise_trader_data(trainer, current_price) -> list[NoiseTraderInfo]` - 收集噪声交易者数据

**价格处理说明：**
- `last_price`: 使用 `orderbook.last_price`（tick 结束后的实际价格），用于价格图表显示，与盘口保持一致
- `price_for_equity`: 使用 `tick_start_price`（tick 开始时的价格），用于资产计算，确保与淘汰检查使用同一价格

---

### UIController (ui_controller.py)

UI控制器，管理训练线程与UI线程的交互。

**线程模型：**
```
主线程(UI渲染) <--Queue--> 训练线程(Trainer)
```

**属性：**
- `trainer: Trainer` - 训练器实例
- `data_collector: UIDataCollector` - 数据采集器
- `sample_rate: int` - 采样率，每N个tick采集一次数据
- `data_queue: Queue[UIDataSnapshot]` - 数据队列（主线程消费，maxsize=10）
- `_speed_factor: float` - 演示速度倍率（范围0.1-100.0）
- `_is_demo_mode: bool` - 是否为演示模式
- `_training_thread: Thread | None` - 训练/演示线程
- `_tick_counter: int` - tick计数器（用于采样率控制）
- `_demo_end_callback: Callable[[dict], None] | None` - 演示结束时的回调函数

**控制信号：**
- `_pause_event: threading.Event` - 暂停事件
- `_stop_event: threading.Event` - 停止事件

**方法：**
- `start_training(episodes: int) -> None` - 启动训练模式（后台线程，会进化）
- `start_demo() -> None` - 启动演示模式（后台线程，不进化，受速度控制）
- `pause() -> None` - 暂停训练/演示
- `resume() -> None` - 恢复训练/演示
- `stop() -> None` - 停止训练/演示（等待线程结束，超时2秒）
- `set_speed(factor: float) -> None` - 设置演示速度（范围0.1-100.0，1.0=正常，10.0=10倍速）
- `get_latest_data() -> UIDataSnapshot | None` - 非阻塞获取最新数据快照
- `is_running() -> bool` - 检查是否正在运行
- `is_paused() -> bool` - 检查是否暂停
- `get_speed() -> float` - 获取当前速度倍率
- `is_demo_mode() -> bool` - 检查是否为演示模式
- `set_demo_end_callback(callback: Callable[[dict], None] | None) -> None` - 设置演示结束回调

**私有方法：**
- `_training_loop(episodes: int) -> None` - 训练主循环（后台线程）
- `_demo_loop() -> None` - 演示循环（后台线程，无限循环，使用演示模式专用结束条件）
- `_collect_and_send_data() -> None` - 采集数据并发送到队列
- `_check_demo_end_condition() -> tuple[str, AgentType | None] | None` - 检查演示模式结束条件（仅物种淘汰）

**数据流：**
1. 训练/演示线程每tick执行后，调用`_collect_and_send_data()`
2. 数据采集器收集快照，放入队列
3. UI主线程调用`get_latest_data()`获取数据更新渲染
4. 队列满时丢弃最旧数据，保证UI总能获取最新数据

**训练模式流程：**
1. 按episode循环运行
2. 每个episode开始前重置Agent账户、噪声交易者状态和市场状态
3. 运行tick循环，按采样率采集数据
4. 检查暂停/停止事件
5. 每个tick后检查`trainer._should_end_episode_early()`
6. episode结束后进行NEAT进化（训练模式）
7. 重建Agent映射和执行顺序

**演示模式流程：**
1. 无限循环运行episode
2. 每个episode开始前重置Agent账户、噪声交易者状态和市场状态
3. 运行tick循环，每tick都采集数据
4. 检查暂停/停止事件
5. 使用`_check_demo_end_condition()`检查结束条件（仅物种淘汰，不检查订单簿单边）
6. 速度控制：`time.sleep(0.1 / speed_factor)`
7. 满足结束条件时调用`_demo_end_callback`并退出演示循环
8. 不进行进化，适合展示训练效果

---

### TrainingUIApp (training_app.py)

训练模式UI应用，提供可视化界面进行NEAT训练。

**构造参数：**
- `trainer: Trainer` - 已初始化的训练器
- `episodes: int` - 训练的episode数量

**属性：**
- `trainer: Trainer` - 已初始化的训练器
- `episodes: int` - 训练的episode数量
- `episode_length: int` - 每个episode的tick数
- `data_collector: UIDataCollector` - UI数据采集器
- `controller: UIController` - UI控制器
- `orderbook_panel: OrderBookPanel | None` - 订单簿面板组件
- `chart_panel: ChartPanel | None` - 图表面板组件
- `trades_panel: TradesPanel | None` - 成交记录面板组件
- `control_panel: ControlPanel | None` - 控制面板组件

**方法：**
- `run() -> None` - 运行主循环（阻塞直到窗口关闭）

**私有方法：**
- `_setup_dpg() -> None` - 初始化DearPyGui上下文和视口
- `_load_chinese_font() -> None` - 加载中文字体
- `_setup_ui() -> None` - 创建UI布局
- `_on_start() -> None` - 开始训练按钮回调
- `_on_pause() -> None` - 暂停/继续按钮回调
- `_on_stop() -> None` - 停止训练按钮回调
- `_update_ui() -> None` - 更新UI（每帧调用）

**功能：**
- 实时显示订单簿、价格走势、成交记录、种群统计
- 提供开始/暂停/停止按钮
- 训练模式以最大速度运行，不支持速度控制
- 在后台线程中进行NEAT进化
- 视口标题：NEAT Trading Simulator - Training Mode
- 视口大小：1920x1200

---

### DemoUIApp (demo_app.py)

演示模式UI应用，从检查点加载模型展示效果。

**构造参数：**
- `trainer: Trainer` - 已初始化的训练器
- `checkpoint_path: str | None` - 检查点路径（可选），用于加载训练好的模型
- `analyzer: DemoAnalyzer | None` - 演示分析器（可选），用于生成演示结束后的分析报告

**属性：**
- `trainer: Trainer` - 已初始化的训练器
- `episode_length: int` - 每个episode的tick数
- `data_collector: UIDataCollector` - UI数据采集器
- `controller: UIController` - UI控制器
- `_analyzer: DemoAnalyzer | None` - 演示分析器
- `orderbook_panel: OrderBookPanel | None` - 订单簿面板组件
- `chart_panel: ChartPanel | None` - 图表面板组件
- `trades_panel: TradesPanel | None` - 成交记录面板组件
- `control_panel: ControlPanel | None` - 控制面板组件

**方法：**
- `run() -> None` - 运行主循环（阻塞直到窗口关闭）

**私有方法：**
- `_setup_dpg() -> None` - 初始化DearPyGui上下文和视口
- `_load_chinese_font() -> None` - 加载中文字体
- `_setup_ui() -> None` - 创建UI布局
- `_on_start() -> None` - 开始演示按钮回调
- `_on_pause() -> None` - 暂停/继续按钮回调
- `_on_stop() -> None` - 停止演示按钮回调
- `_on_demo_end(data: dict) -> None` - 演示结束回调
- `_update_ui() -> None` - 更新UI（每帧调用）

**功能：**
- 实时显示订单簿、价格走势、成交记录、种群统计
- 提供开始/暂停/停止按钮
- 演示模式不进行进化，支持速度控制（通过ControlPanel的滑块）
- 无限循环运行episode，适合展示
- 视口标题：NEAT Trading Simulator - Demo Mode
- 视口大小：1920x1200
- 演示结束时调用分析器生成报告（如果提供）

---

## UI组件目录

详细的UI组件文档请参阅：`src/ui/components/CLAUDE.md`

**组件概览：**

| 组件 | 文件 | 功能 | 尺寸 |
|------|------|------|------|
| OrderBookPanel | orderbook_panel.py | 订单簿深度图 + 合并盘口列表 | 280px 宽 |
| ChartPanel | chart_panel.py | 价格曲线 + 存活个体平均资产曲线 + 小提琴图 | 1360px 宽 |
| TradesPanel | trades_panel.py | 最近成交记录列表 | 280px 宽 |
| ControlPanel | control_panel.py | 开始/暂停/停止按钮 + 状态显示 | 自适应宽度 |

---

## 依赖接口

### Trainer (src/training/trainer.py)
- `trainer.tick: int` - 当前tick
- `trainer.episode: int` - 当前episode
- `trainer.populations: dict[AgentType, Population]` - 2个种群（RETAIL_PRO, MARKET_MAKER）
- `trainer.matching_engine._orderbook` - 订单簿
- `trainer.recent_trades: deque[Trade]` - 最近100笔成交
- `trainer.tick_start_price: float` - tick开始时的价格
- `trainer._should_end_episode_early() -> str | None` - 检查是否提前结束episode
- `trainer._reset_market() -> None` - 重置市场状态
- `trainer._pop_liquidated_counts: dict[AgentType, int]` - 种群淘汰计数
- `trainer._pop_total_counts: dict[AgentType, int]` - 种群总数
- `trainer._register_all_agents() -> None` - 注册所有Agent
- `trainer._build_agent_map() -> None` - 构建Agent映射
- `trainer._build_execution_order() -> None` - 构建执行顺序
- `trainer.noise_traders: list[NoiseTrader]` - 噪声交易者列表

### OrderBook (src/market/orderbook/orderbook.pyx)
- `orderbook.get_depth(levels=100) -> {"bids": [...], "asks": [...]}`
- `orderbook.last_price: float`
- `orderbook.get_mid_price() -> float | None`

### Population (src/training/population.py)
- `population.agents: list[Agent]` - Agent列表
- `population.generation: int` - 当前代数
- `population.reset_agents() -> None` - 重置Agent账户
- `population.evolve(current_price: float) -> None` - 进化种群

### Agent (src/bio/agents/base.py)
- `agent.account.get_equity(current_price) -> float`
- `agent.is_liquidated: bool`

### Trade (src/market/matching/trade.py)
- `trade.price: float`
- `trade.quantity: float`
- `trade.is_buyer_taker: bool`

---

## 使用示例

### 数据采集器单独使用

```python
from src.ui import UIDataCollector
from src.training.trainer import Trainer

# 创建采集器
collector = UIDataCollector(history_length=1000)

# 每tick收集数据
def on_tick(trainer: Trainer):
    snapshot = collector.collect_tick_data(trainer)
    # 使用snapshot更新UI...
    print(f"Tick {snapshot.tick}, Price: {snapshot.last_price}")
    print(f"Retail Pro avg equity: {snapshot.population_stats[AgentType.RETAIL_PRO].avg_equity}")

# 新episode开始时重置
def on_episode_start():
    collector.reset()
```

### UI控制器使用（推荐）

```python
from src.ui import UIController, UIDataCollector
from src.training.trainer import Trainer
from src.config.config import Config

# 初始化
config = Config(...)
trainer = Trainer(config)
trainer.setup()

collector = UIDataCollector(history_length=1000)
controller = UIController(trainer, collector, sample_rate=1)

# 启动训练模式（后台线程运行）
controller.start_training(episodes=100)

# 或启动演示模式
# controller.start_demo()
# controller.set_speed(2.0)  # 2倍速

# UI主循环
while controller.is_running():
    snapshot = controller.get_latest_data()
    if snapshot:
        # 更新UI渲染
        render_orderbook(snapshot.bids, snapshot.asks)
        render_chart(snapshot.price_history)

    # 响应用户输入
    if user_pressed_pause:
        controller.pause()
    elif user_pressed_resume:
        controller.resume()
    elif user_pressed_stop:
        controller.stop()
        break

    time.sleep(0.016)  # 约60fps

# 清理
controller.stop()
```

### 训练模式UI应用

```python
from src.config.config import Config
from src.training.trainer import Trainer
from src.ui.training_app import TrainingUIApp

# 创建配置和训练器
config = Config(...)
trainer = Trainer(config)
trainer.setup()

# 创建并运行训练UI
app = TrainingUIApp(trainer, episodes=100)
app.run()  # 阻塞直到窗口关闭
```

### 演示模式UI应用

```python
from src.config.config import Config
from src.training.trainer import Trainer
from src.ui.demo_app import DemoUIApp

# 创建配置和训练器
config = Config(...)
trainer = Trainer(config)
trainer.setup()

# 创建并运行演示UI（从检查点加载）
app = DemoUIApp(
    trainer,
    checkpoint_path="checkpoints/ep_100.pkl",
)
app.run()  # 阻塞直到窗口关闭
```

### 演示模式UI应用（带分析器）

```python
from src.config.config import Config
from src.training.trainer import Trainer
from src.ui.demo_app import DemoUIApp
from src.analysis.demo_analyzer import DemoAnalyzer

# 创建配置和训练器
config = Config(...)
trainer = Trainer(config)
trainer.setup()

# 创建分析器
analyzer = DemoAnalyzer(output_dir="analysis_results")

# 创建并运行演示UI（从检查点加载，带分析器）
app = DemoUIApp(
    trainer,
    checkpoint_path="checkpoints/ep_100.pkl",
    analyzer=analyzer
)
app.run()  # 阻塞直到窗口关闭，演示结束后自动生成分析报告
```

---

## 性能优化

- 使用NumPy向量化计算种群统计，避免Python循环
- 使用`deque(maxlen=N)`自动限制历史数据长度，避免手动管理
- 预分配NumPy数组避免重复内存分配
- UI主循环与训练线程分离，避免阻塞UI渲染
- 数据队列满时丢弃旧数据，保证实时性
- 采样率控制（sample_rate）减少不必要的数据采集
- 中文字体缓存，避免重复加载
- DemoAnalyzer使用Matplotlib批量生成图表，避免频繁绘图

---

## 中文字体支持

UI模块会自动尝试加载系统中文字体，按以下顺序查找：

**Linux:**
- `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
- `/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc`
- `/usr/share/fonts/truetype/arphic/uming.ttc`
- `/usr/share/fonts/truetype/wqy/wqy-microhei.ttc`
- `/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf`

**Windows:**
- `C:/Windows/Fonts/msyh.ttc`
- `C:/Windows/Fonts/simhei.ttf`

**macOS:**
- `/System/Library/Fonts/PingFang.ttc`
- `/System/Library/Fonts/STHeiti Light.ttc`

如果找不到中文字体，UI会降级使用默认字体（可能无法显示中文）。

---

## 训练模式 vs 演示模式

| 特性 | 训练模式 | 演示模式 |
|------|---------|---------|
| 进化 | 是 | 否 |
| Episode数量 | 有限 | 无限（直到满足结束条件） |
| 速度控制 | 不支持 | 支持（0.1x - 100x） |
| 结束条件 | 订单簿单边、物种淘汰 | 仅物种淘汰 |
| 检查点加载 | 不支持（从头训练） | 支持（加载已有模型） |
| 分析报告 | 不生成 | 可选生成（DemoAnalyzer） |
| 用途 | 训练新的NEAT模型 | 展示已有模型效果 |

---

## 注意事项

1. **线程安全：** UIController使用队列传递数据，避免直接跨线程访问共享状态
2. **资源清理：** 窗口关闭时会自动调用`controller.stop()`，但建议在代码中显式处理
3. **采样率：** 训练模式可设置`sample_rate > 1`减少UI更新频率，提升性能
4. **演示结束：** 演示模式会在满足物种淘汰条件（存活数 < 总数/4）时自动退出，可通过回调函数自定义处理
5. **分析器使用：** DemoAnalyzer需要matplotlib和seaborn依赖，首次使用请确认已安装
