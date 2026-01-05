# UI组件目录

## 目录职责

本目录包含所有 DearPyGui UI 组件，每个组件负责独立的数据展示功能，可单独使用或组合使用。

## 组件概览

| 组件 | 文件 | 功能 | 尺寸 |
|------|------|------|------|
| OrderBookPanel | orderbook_panel.py | 订单簿深度图 + 合并盘口列表 | 280px 宽 |
| ChartPanel | chart_panel.py | 价格曲线 + 种群资产曲线 + 小提琴图 + 鲶鱼曲线 | 1360px 宽 |
| TradesPanel | trades_panel.py | 最近成交记录列表 | 280px 宽 |
| ControlPanel | control_panel.py | 开始/暂停/停止按钮 + 状态显示 | 自适应宽度 |

## 组件详细说明

### OrderBookPanel

订单簿面板，显示深度图和合并的买卖盘价格列表。

**类常量配置：**
```python
PANEL_WIDTH: int = 280           # 面板宽度
DEPTH_CHART_HEIGHT: int = 180    # 深度图高度
DISPLAY_LEVELS: int = 10         # 每边显示档数
```

**接口：**
```python
def __init__(self) -> None:
    """初始化订单簿面板，自动添加到当前DearPyGui上下文"""

def update(self, bids: list[tuple[float, float]],
           asks: list[tuple[float, float]]) -> None:
    """更新订单簿数据
    Args:
        bids: 买盘 [(价格, 数量), ...]，按价格从高到低排序
        asks: 卖盘 [(价格, 数量), ...]，按价格从低到高排序
    """
```

**显示内容：**
1. **深度图**（Area Series）
   - 买盘累计量：绿色填充 (100, 255, 100)
   - 卖盘累计量：红色填充 (255, 100, 100)
   - X轴：价格，Y轴：累计数量

2. **合并盘口列表**（表格）
   - 卖盘10档（红色，价格从高到低）
   - 分隔线（灰色虚线）
   - 买盘10档（绿色，价格从高到低）
   - 列：价格 | 数量 | 类型

**Tag命名规则：**
- 深度图：`orderbook_depth_plot`, `orderbook_depth_x`, `orderbook_depth_y`
- 买盘区域：`orderbook_bid_shade`
- 卖盘区域：`orderbook_ask_shade`
- 订单簿表格：`orderbook_orderbook_table`

---

### ChartPanel

图表面板，显示价格曲线、种群资产曲线、资产分布小提琴图和鲶鱼资金曲线。

**类常量配置：**
```python
PANEL_WIDTH: int = 1360           # 面板总宽度
EQUITY_PLOT_HEIGHT: int = 140     # 每个资产图表高度
PRICE_PLOT_HEIGHT: int = 140      # 价格图表高度
VIOLIN_PLOT_HEIGHT: int = 120     # 小提琴图高度
VIOLIN_PLOT_WIDTH: int = 340      # 每个小提琴图宽度（4个并排）
CATFISH_PLOT_HEIGHT: int = 140    # 鲶鱼图表高度
CATFISH_PLOT_WIDTH: int = 450     # 每个鲶鱼图表宽度（3个并排）
KDE_POINTS: int = 50              # KDE曲线采样点数
```

**接口：**
```python
def __init__(self) -> None:
    """初始化图表面板，自动添加到当前DearPyGui上下文"""

def update_price(self, price_history: list[float]) -> None:
    """更新价格曲线
    Args:
        price_history: 价格历史列表
    """

def update_equity(
    self,
    equity_history: dict[AgentType, list[float]],
    alive_equity_history: dict[AgentType, list[float]],
    population_stats: dict[AgentType, Any]
) -> None:
    """更新资产曲线、统计和小提琴图
    Args:
        equity_history: 各种群所有个体平均资产历史（保留参数兼容性，但不再显示）
        alive_equity_history: 各种群存活个体平均资产历史（用于显示）
        population_stats: 各种群统计信息（PopulationStats对象）
    """

def update_catfish(
    self,
    catfish_data: list[Any],
    catfish_equity_history: list[list[float]]
) -> None:
    """更新鲶鱼图表
    Args:
        catfish_data: 鲶鱼信息列表（CatfishInfo对象列表）
        catfish_equity_history: 三只鲶鱼的净值历史
    """
```

**布局结构：**
```
┌─────────────────────────────────────────────────────┐
│ 价格走势图 (140px高)                                  │
├─────────────────────────────────────────────────────┤
│ 种群存活个体平均资产曲线                              │
│ ┌─────────────────────────────────────────────┐    │
│ │ 散户存活个体平均资产图 (140px高)              │    │
│ └─────────────────────────────────────────────┘    │
│ ┌─────────────────────────────────────────────┐    │
│ │ 高级散户存活个体平均资产图 (140px高)          │    │
│ └─────────────────────────────────────────────┘    │
│ ┌─────────────────────────────────────────────┐    │
│ │ 庄家存活个体平均资产图 (140px高)              │    │
│ └─────────────────────────────────────────────┘    │
│ ┌─────────────────────────────────────────────┐    │
│ │ 做市商存活个体平均资产图 (140px高)            │    │
│ └─────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│ 种群统计（水平排列）                                  │
│ 散户: ...  高级散户: ...  庄家: ...  做市商: ...    │
├─────────────────────────────────────────────────────┤
│ 小提琴图区域 (120px高, 4个并排)                      │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐                       │
│ │散户│ │高级│ │庄家│ │做市│                         │
│ │    │ │散户│ │    │ │商  │                         │
│ └────┘ └────┘ └────┘ └────┘                       │
├─────────────────────────────────────────────────────┤
│ 鲶鱼资金曲线 (140px高, 3个并排, 默认隐藏)            │
│ ┌────────┐ ┌────────┐ ┌────────┐                   │
│ │趋势追踪│ │周期摆动│ │逆势操作│                     │
│ └────────┘ └────────┘ └────────┘                   │
└─────────────────────────────────────────────────────┘
```

**颜色配置：**
| 类型 | 颜色 | RGB |
|------|------|-----|
| 散户 | 绿色 | (100, 200, 100) |
| 高级散户 | 蓝色 | (100, 150, 255) |
| 庄家 | 粉红色 | (255, 100, 150) |
| 做市商 | 紫色 | (200, 100, 255) |
| 趋势追踪鲶鱼 | 橙色 | (255, 165, 0) |
| 周期摆动鲶鱼 | 天蓝色 | (0, 191, 255) |
| 逆势操作鲶鱼 | 粉色 | (255, 105, 180) |

**小提琴图实现细节：**
- 使用纯 NumPy 实现高斯核密度估计（KDE）
- Silverman 法则自动计算带宽
- Area Series 绘制上下对称的密度曲线
- 中位数线：白色粗线
- Q1/Q3 四分位线：灰色细线

**Tag命名规则：**
- 价格图：`price_plot`, `price_series`, `price_x_axis`, `price_y_axis`
- 资产图：`equity_plot_{agent_type}`, `equity_series_{agent_type}`, `equity_x_axis_{agent_type}`, `equity_y_axis_{agent_type}`
- 统计文本：`stat_{agent_type}`
- 小提琴图：`violin_plot_{agent_type}`, `violin_area_{agent_type}`, `violin_median_{agent_type}`, `violin_q1_{agent_type}`, `violin_q3_{agent_type}`
- 鲶鱼图：`catfish_container`, `catfish_plot_{i}`, `catfish_series_{i}`, `catfish_x_axis_{i}`, `catfish_y_axis_{i}`, `catfish_title_{i}`, `catfish_position_{i}` (i=0,1,2)

---

### TradesPanel

成交记录面板，显示最近的成交记录列表。

**类常量配置：**
```python
PANEL_WIDTH: int = 280           # 面板宽度
MAX_DISPLAY_TRADES: int = 30     # 最大显示条数
```

**接口：**
```python
def __init__(self) -> None:
    """初始化成交记录面板，自动添加到当前DearPyGui上下文"""

def update(self, trades: list[TradeInfo]) -> None:
    """更新成交记录
    Args:
        trades: 成交记录列表（TradeInfo对象列表）
    """
```

**显示格式：**
- 表格列：Tick | 价格 | 数量 | 方向
- 最多显示最近30笔成交
- 倒序显示（最新在上）
- 买入：绿色文字 "买入"
- 卖出：红色文字 "卖出"

**Tag命名规则：**
- 表格：`trades_table`

---

### ControlPanel

控制面板，提供训练/演示的控制按钮和状态显示。

**接口：**
```python
def __init__(
    self,
    on_start: Callable[[], None] | None = None,
    on_pause: Callable[[], None] | None = None,
    on_stop: Callable[[], None] | None = None,
) -> None:
    """初始化控制面板，自动添加到当前DearPyGui上下文
    Args:
        on_start: 开始按钮回调
        on_pause: 暂停/继续按钮回调
        on_stop: 停止按钮回调
    """

def update_status(
    self,
    episode: int,
    tick: int,
    total_ticks: int,
    price: float
) -> None:
    """更新状态显示
    Args:
        episode: 当前episode编号
        tick: 当前tick编号
        total_ticks: 总tick数
        price: 当前价格
    """

def reset(self) -> None:
    """重置控制面板状态"""

@property
def is_paused(self) -> bool:
    """获取当前暂停状态"""
```

**显示内容：**
- 三个按钮：开始 | 暂停 | 停止
- 状态显示：Episode: X | Tick: Y/Z | 价格: P.PP

**交互逻辑：**
- 点击"开始"后，"开始"按钮禁用
- "暂停"按钮可切换为"继续"
- 点击"停止"后，恢复初始状态

**Tag命名规则：**
- 按钮：`btn_start`, `btn_pause`, `btn_stop`
- 状态文本：`episode_text`, `tick_text`, `price_text`

---

## 组件交互关系

```
┌─────────────────────────────────────────────────────────┐
│                    UI应用主窗口                          │
├─────────────┬───────────────────────────┬───────────────┤
│             │                           │               │
│ControlPanel │                           │               │
│─────────────│                           │               │
│             │        ChartPanel         │               │
│OrderBookPanel│        (1360px宽)         │ TradesPanel   │
│ (280px宽)   │                           │  (280px宽)    │
│             │  - 价格曲线               │               │
│ - 深度图    │  - 资产曲线               │ - 成交记录    │
│ - 盘口列表  │  - 小提琴图               │ (最近30笔)    │
│             │  - 鲶鱼曲线               │               │
└─────────────┴───────────────────────────┴───────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │   UIController      │
              │  (训练线程控制)      │
              └─────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │      Trainer        │
              │   (训练器核心)       │
              └─────────────────────┘
```

## 数据流转

1. **训练线程** 执行 tick → 收集数据到 UIDataCollector
2. **UIDataCollector** 生成 UIDataSnapshot → 放入 Queue
3. **UI主线程** 从 Queue 获取 UIDataSnapshot
4. **UI组件** 调用各自的 `update()` 方法接收数据：
   - `OrderBookPanel.update(bids, asks)`
   - `ChartPanel.update_price(price_history)`
   - `ChartPanel.update_equity(...)`
   - `ChartPanel.update_catfish(...)`
   - `TradesPanel.update(trades)`
   - `ControlPanel.update_status(...)`

## 组件独立使用示例

```python
import dearpygui.dearpygui as dpg
from src.ui.components import OrderBookPanel, ChartPanel, TradesPanel, ControlPanel

# 创建上下文
dpg.create_context()

# 创建主窗口
with dpg.window(label="Trading", tag="main_window"):
    # 各组件会自动添加到当前上下文
    control = ControlPanel(
        on_start=lambda: print("Start"),
        on_pause=lambda: print("Pause"),
        on_stop=lambda: print("Stop")
    )

    dpg.add_separator()

    with dpg.group(horizontal=True):
        orderbook = OrderBookPanel()
        chart = ChartPanel()
        trades = TradesPanel()

# 更新数据
orderbook.update(
    bids=[(100.0, 10), (99.5, 20)],  # 买盘（价格从高到低）
    asks=[(100.5, 15), (101.0, 25)]  # 卖盘（价格从低到高）
)

chart.update_price([100.0, 100.5, 99.8, 100.2])
control.update_status(episode=1, tick=50, total_ticks=1000, price=100.2)

# 运行
dpg.create_viewport(title='Trading', width=1920, height=1080)
dpg.setup_dearpygui()
dpg.show_viewport()

while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()

dpg.destroy_context()
```
