# UI模块

## 模块职责

提供训练可视化所需的数据采集、控制和展示组件，将训练器内部数据转换为UI展示所需的格式。

## 文件结构

- `__init__.py` - 模块导出
- `data_collector.py` - UI数据采集器
- `ui_controller.py` - UI控制器（管理训练线程与UI线程的交互）
- `training_app.py` - 训练模式UI应用（DearPyGui）
- `demo_app.py` - 演示模式UI应用（DearPyGui）
- `components/` - UI组件目录
  - `__init__.py` - 组件导出
  - `orderbook_panel.py` - 订单簿面板（深度图+价格列表）
  - `chart_panel.py` - 图表面板（价格曲线+资产曲线+种群统计）
  - `trades_panel.py` - 成交记录面板
  - `control_panel.py` - 控制面板（开始/暂停/停止按钮+状态显示）

## 核心类

### UIDataCollector

UI数据采集器，每tick从训练器收集数据，维护历史缓冲区。

**属性：**
- `history_length: int` - 历史数据长度限制
- `price_history: deque[float]` - 价格历史缓冲区（存储 orderbook.last_price，与盘口一致）
- `equity_history: dict[AgentType, deque[float]]` - 各种群所有个体平均资产历史缓冲区
- `alive_equity_history: dict[AgentType, deque[float]]` - 各种群存活个体平均资产历史缓冲区

**方法：**
- `collect_tick_data(trainer) -> UIDataSnapshot` - 收集当前tick数据快照
- `reset() -> None` - 重置历史数据（新episode开始时调用）

**价格处理说明：**
- `last_price`: 使用 `orderbook.last_price`（tick 结束后的实际价格），用于价格图表显示，与盘口保持一致
- `price_for_equity`: 使用 `tick_start_price`（tick 开始时的价格），用于资产计算，确保与淘汰检查使用同一价格

### UIDataSnapshot

UI数据快照，每个tick的完整数据。

**字段：**
- `tick: int` - 当前tick
- `episode: int` - 当前episode
- `last_price: float` - 最新成交价（tick 结束后的 orderbook.last_price，与盘口一致）
- `mid_price: float` - 中间价
- `bids: list[tuple[float, float]]` - 买盘100档 [(price, qty), ...]
- `asks: list[tuple[float, float]]` - 卖盘100档 [(price, qty), ...]
- `recent_trades: list[TradeInfo]` - 最近成交记录
- `population_stats: dict[AgentType, PopulationStats]` - 种群统计
- `price_history: list[float]` - 价格历史
- `equity_history: dict[AgentType, list[float]]` - 所有个体平均资产历史
- `alive_equity_history: dict[AgentType, list[float]]` - 存活个体平均资产历史

### TradeInfo

成交信息（UI展示用）。

**字段：**
- `tick: int` - 成交tick
- `price: float` - 成交价格
- `quantity: float` - 成交数量
- `is_buyer_taker: bool` - True=买入（taker是买方），False=卖出（taker是卖方）

### PopulationStats

种群统计信息。

**字段：**
- `avg_equity: float` - 存活Agent平均净值
- `total_equity: float` - 所有Agent资产总和
- `max_equity: float` - 最大净值
- `min_equity: float` - 最小净值
- `alive_count: int` - 存活数量
- `total_count: int` - 总数量
- `generation: int` - 当前代数
- `alive_equities: list[float]` - 存活个体的资产列表（用于小提琴图）

### UIController

UI控制器，管理训练线程与UI线程的交互。

**线程模型：**
```
主线程(UI渲染) <--Queue--> 训练线程(Trainer)
```

**属性：**
- `trainer: Trainer` - 训练器实例
- `data_collector: UIDataCollector` - 数据采集器
- `sample_rate: int` - 采样率，每N个tick采集一次数据
- `data_queue: Queue[UIDataSnapshot]` - 数据队列（主线程消费）

**方法：**
- `start_training(episodes: int) -> None` - 启动训练模式（后台线程，会进化）
- `start_demo() -> None` - 启动演示模式（后台线程，不进化，受速度控制）
- `pause() -> None` - 暂停训练/演示
- `resume() -> None` - 恢复训练/演示
- `stop() -> None` - 停止训练/演示（等待线程结束）
- `set_speed(factor: float) -> None` - 设置演示速度（1.0=正常，10.0=10倍速）
- `get_latest_data() -> UIDataSnapshot | None` - 非阻塞获取最新数据快照
- `is_running() -> bool` - 检查是否正在运行
- `is_paused() -> bool` - 检查是否暂停
- `get_speed() -> float` - 获取当前速度倍率
- `is_demo_mode() -> bool` - 检查是否为演示模式

**控制信号：**
- `_pause_event: threading.Event` - 暂停事件
- `_stop_event: threading.Event` - 停止事件

**数据流：**
1. 训练/演示线程每tick执行后，调用`_collect_and_send_data()`
2. 数据采集器收集快照，放入队列
3. UI主线程调用`get_latest_data()`获取数据更新渲染
4. 队列满时丢弃最旧数据，保证UI总能获取最新数据

**提前结束逻辑：**
- 训练/演示循环在每个tick后检查`trainer._should_end_episode_early()`
- 若庄家/做市商存活少于初始值的 1/4，或散户/高级散户被完全淘汰，立即结束当前episode的tick循环
- 进入进化阶段（训练模式）或开始新episode（演示模式）

## 依赖接口

### Trainer (src/training/trainer.py)
- `trainer.tick: int` - 当前tick
- `trainer.episode: int` - 当前episode
- `trainer.populations: dict[AgentType, Population]` - 4个种群
- `trainer.matching_engine._orderbook` - 订单簿
- `trainer.recent_trades: deque[Trade]` - 最近100笔成交

### OrderBook (src/market/orderbook/orderbook.pyx)
- `orderbook.get_depth(levels=100) -> {"bids": [...], "asks": [...]}`
- `orderbook.last_price: float`
- `orderbook.get_mid_price() -> float | None`

### Population (src/training/population.py)
- `population.agents: list[Agent]` - Agent列表
- `population.generation: int` - 当前代数

### Agent (src/bio/agents/base.py)
- `agent.account.get_equity(current_price) -> float`
- `agent.is_liquidated: bool`

### Trade (src/market/matching/trade.py)
- `trade.price: float`
- `trade.quantity: float`
- `trade.is_buyer_taker: bool`

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
    print(f"Retail avg equity: {snapshot.population_stats[AgentType.RETAIL].avg_equity}")

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

## UI组件

### OrderBookPanel

订单簿面板，显示深度图和合并的买卖盘价格列表。

**构造参数：**
- 无参数，组件会自动添加到当前DearPyGui上下文中

**类常量：**
- `PANEL_WIDTH: int = 280` - 面板宽度
- `DEPTH_CHART_HEIGHT: int = 180` - 深度图高度
- `DISPLAY_LEVELS: int = 10` - 每边显示档数

**方法：**
- `update(bids, asks) -> None` - 更新订单簿数据
  - `bids: list[tuple[float, float]]` - 买盘数据 [(price, qty), ...]，按价格从高到低排序
  - `asks: list[tuple[float, float]]` - 卖盘数据 [(price, qty), ...]，按价格从低到高排序

**功能：**
- 深度图：使用area_series显示买卖盘累计量，买盘绿色，卖盘红色
- 合并盘口列表：买卖盘在同一个表格中显示，价格从高到低
  - 卖盘10档在上（红色，价格从高到低）
  - 分隔线
  - 买盘10档在下（绿色，价格从高到低）
- 面板宽度280px，无滚动条

### ChartPanel

图表面板，显示价格曲线、种群存活个体平均资产曲线和资产分布小提琴图。

**构造参数：**
- 无参数，组件会自动添加到当前DearPyGui上下文中

**类常量：**
- `PANEL_WIDTH: int = 1360` - 面板总宽度
- `EQUITY_PLOT_HEIGHT: int = 140` - 每个资产图表高度
- `PRICE_PLOT_HEIGHT: int = 140` - 价格图表高度
- `VIOLIN_PLOT_HEIGHT: int = 120` - 小提琴图高度
- `VIOLIN_PLOT_WIDTH: int = 340` - 每个小提琴图宽度（4个并排）
- `KDE_POINTS: int = 50` - KDE曲线采样点数

**方法：**
- `update_price(price_history) -> None` - 更新价格曲线
  - `price_history: list[float]` - 价格历史列表
- `update_equity(equity_history, alive_equity_history, population_stats) -> None` - 更新资产曲线、统计和小提琴图
  - `equity_history: dict[AgentType, list[float]]` - 各种群所有个体平均资产历史（保留参数兼容性，但不再显示）
  - `alive_equity_history: dict[AgentType, list[float]]` - 各种群存活个体平均资产历史（用于显示）
  - `population_stats: dict[AgentType, PopulationStats]` - 各种群统计信息

**私有方法：**
- `_create_equity_row(agent_type) -> None` - 创建单个种群的存活个体资产图表
- `_create_violin_plots() -> None` - 创建4个并排的小提琴图
- `_setup_violin_themes() -> None` - 设置小提琴图颜色主题
- `_gaussian_kde(data, x_grid, bandwidth) -> np.ndarray` - 高斯核密度估计（纯NumPy实现）
- `_update_violin_plot(agent_type, equities) -> None` - 更新单个种群的小提琴图

**布局：**
- 价格走势图：高度140px，宽度自适应
- 5行资产图表，每行1张图（存活个体平均资产）：
  - 第1行：散户存活个体平均资产图
  - 第2行：高级散户存活个体平均资产图
  - 第3行：多头庄家存活个体平均资产图
  - 第4行：空头庄家存活个体平均资产图
  - 第5行：做市商存活个体平均资产图
- 每个资产图表：高度140px，宽度自适应
- 每个图表标题显示种群名称
- 统计文字水平排列显示在5个图表下方（节省垂直空间）
- 小提琴图区域（5个并排，高度120px）：
  - 每个图表宽度340px
  - 显示KDE密度曲线形成的小提琴形状
  - 中位数线（白色）和四分位线（灰色）

**小提琴图实现：**
- 使用 Area Series 绘制 KDE 密度曲线
- KDE 算法使用 Silverman 法则自动计算带宽
- 密度曲线上下对称形成小提琴形状
- 中位数线和 Q1/Q3 四分位线使用 Line Series 绘制

**颜色配置：**
- 散户: 绿色 (100, 200, 100)
- 高级散户: 蓝色 (100, 150, 255)
- 多头庄家: 红色 (255, 100, 100)
- 空头庄家: 青色 (100, 255, 255)
- 做市商: 紫色 (200, 100, 255)

**Tag命名规则：**
- 价格图：`price_plot`, `price_series`, `price_x_axis`, `price_y_axis`
- 存活个体平均资产图：`equity_plot_{agent_type}`, `equity_series_{agent_type}`, `equity_x_axis_{agent_type}`, `equity_y_axis_{agent_type}`
- 统计文本：`stat_{agent_type}`
- 小提琴图：`violin_plot_{agent_type}`, `violin_area_{agent_type}`, `violin_median_{agent_type}`, `violin_q1_{agent_type}`, `violin_q3_{agent_type}`

### TradesPanel

成交记录面板，显示最近成交记录。

**构造参数：**
- 无参数，组件会自动添加到当前DearPyGui上下文中

**类常量：**
- `PANEL_WIDTH: int = 280` - 面板宽度
- `MAX_DISPLAY_TRADES: int = 30` - 最大显示条数

**方法：**
- `update(trades) -> None` - 更新成交记录
  - `trades: list[TradeInfo]` - 成交记录列表

**显示格式：**
- 最新30笔成交，倒序显示（最新在上）
- 买入绿色，卖出红色
- 面板宽度280px，无滚动条

### ControlPanel

控制面板，提供训练/演示控制和状态显示。

**构造参数：**
- `on_start: Callable[[], None] | None` - 开始回调
- `on_pause: Callable[[], None] | None` - 暂停/继续回调
- `on_stop: Callable[[], None] | None` - 停止回调
- `on_speed_change: Callable[[float], None] | None` - 速度变化回调

注：组件会自动添加到当前DearPyGui上下文中

**方法：**
- `update_status(episode, tick, total_ticks, price) -> None` - 更新状态显示
- `reset() -> None` - 重置面板状态

**属性：**
- `is_paused: bool` - 当前暂停状态

### TrainingUIApp

训练模式UI应用，提供可视化界面进行NEAT训练。

**构造参数：**
- `trainer: Trainer` - 已初始化的训练器
- `episodes: int` - 训练的episode数量

**方法：**
- `run() -> None` - 运行主循环（阻塞直到窗口关闭）

**功能：**
- 实时显示订单簿、价格走势、成交记录、种群统计
- 提供开始/暂停/停止按钮
- 训练模式以最大速度运行，不支持速度控制
- 在后台线程中进行NEAT进化

### DemoUIApp

演示模式UI应用，从检查点加载模型展示效果。

**构造参数：**
- `trainer: Trainer` - 已初始化的训练器
- `checkpoint_path: str | None` - 检查点路径（可选）

**方法：**
- `run() -> None` - 运行主循环（阻塞直到窗口关闭）

**功能：**
- 实时显示订单簿、价格走势、成交记录、种群统计
- 提供开始/暂停/停止按钮和速度滑块
- 演示模式不进行进化，支持速度控制
- 无限循环运行episode，适合展示

## 使用示例（带UI组件）

```python
import dearpygui.dearpygui as dpg
from src.ui.components import OrderBookPanel, ChartPanel, TradesPanel, ControlPanel

dpg.create_context()

with dpg.window(label="TradingGame", tag="main_window", no_scrollbar=True):
    # 控制面板（自动添加到当前窗口上下文）
    control = ControlPanel(
        on_start=lambda: controller.start_demo(),
        on_pause=lambda: controller.pause(),
        on_stop=lambda: controller.stop())

    dpg.add_separator()

    # 主内容区（水平三栏布局：订单簿(280w) | 图表(1150w) | 成交记录(280w)）
    # 总宽度约1710px，适配1920px屏幕
    with dpg.group(horizontal=True):
        # 左侧：订单簿（宽度280px）
        orderbook = OrderBookPanel()

        # 中间：图表（宽度1150px，包含4个2x2网格的资产图表）
        chart = ChartPanel()

        # 右侧：成交记录（宽度280px）
        trades = TradesPanel()

# 数据更新循环
def update_ui():
    snapshot = controller.get_latest_data()
    if snapshot:
        control.update_status(snapshot.episode, snapshot.tick, 1000, snapshot.last_price)
        orderbook.update(snapshot.bids, snapshot.asks)
        chart.update_price(snapshot.price_history)
        chart.update_equity(snapshot.equity_history, snapshot.alive_equity_history, snapshot.population_stats)
        trades.update(snapshot.recent_trades)

dpg.create_viewport(title='TradingGame', width=1920, height=1080)
dpg.setup_dearpygui()
dpg.show_viewport()

while dpg.is_dearpygui_running():
    update_ui()
    dpg.render_dearpygui_frame()

dpg.destroy_context()
```

## 使用示例（UI应用）

### 训练模式

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

### 演示模式

```python
from src.config.config import Config
from src.training.trainer import Trainer
from src.ui.demo_app import DemoUIApp

# 创建配置和训练器
config = Config(...)
trainer = Trainer(config)
trainer.setup()

# 创建并运行演示UI（从检查点加载）
app = DemoUIApp(trainer, checkpoint_path="checkpoints/ep_100.pkl")
app.run()  # 阻塞直到窗口关闭
```

## 性能优化

- 使用NumPy向量化计算种群统计，避免Python循环
- 使用deque(maxlen=N)自动限制历史数据长度
- 预分配NumPy数组避免重复内存分配
- UI主循环与训练线程分离，避免阻塞
- 数据队列满时丢弃旧数据，保证实时性
