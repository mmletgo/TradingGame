# Analysis 模块

## 模块概述

分析模块负责对演示模式结束后的数据进行分析和可视化，生成分析图表和终端摘要。

## 文件结构

- `__init__.py` - 模块导出
- `demo_analyzer.py` - 演示模式分析器

## 核心类

### DemoAnalyzer

演示模式分析器，分析演示结束时各物种存活个体的分布，生成分析图和终端摘要。

**属性：**
- `_output_dir: str` - 分析结果输出目录
- `_chinese_font: str | None` - 中文字体路径

**构造参数：**
- `output_dir: str` - 分析结果输出目录（默认 "analysis_output"）

**核心方法：**

#### `analyze(trainer, end_reason, end_agent_type) -> None`
执行分析并输出结果。

参数：
- `trainer: Trainer` - 训练器实例
- `end_reason: str` - 结束原因（"population_depleted" 或 "one_sided_orderbook"）
- `end_agent_type: AgentType | None` - 触发结束的 Agent 类型

执行流程：
1. 收集各物种存活个体的数据
2. 打印终端摘要
3. 生成分析图

#### `_collect_data(trainer) -> dict[str, Any]`
收集各物种存活个体的数据。

收集内容（每个存活 Agent）：
- `equity: float` - 净值
- `balance: float` - 余额
- `unrealized_pnl: float` - 浮动盈亏
- `position_qty: int` - 持仓量
- `leverage: float` - 杠杆率（position_value / equity）

返回格式：
```python
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
                {
                    "equity": float,
                    "balance": float,
                    "unrealized_pnl": float,
                    "position_qty": int,
                    "leverage": float
                },
                ...
            ]
        },
        ...
    }
}
```

#### `_print_summary(data, end_reason, end_agent_type) -> None`
终端打印摘要。

输出格式：
```
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
```

#### `_generate_plots(data) -> str`
生成分析图。

使用 matplotlib 生成 2x4 子图布局：
- 第一行：资产分布图 - 4 个子图（每种群一个），箱线图显示净值分布
- 第二行：持仓分布图 - 4 个子图，条形图显示多头/空仓/空头数量分布

返回保存的图片路径。

**图表特性：**
- 自动查找中文字体
- 箱线图显示中位数和四分位数
- 红色虚线标注均值
- 持仓分布使用颜色区分：红色（多头）、灰色（空仓）、青色（空头）
- 图片分辨率 150 DPI

#### `_format_money(amount) -> str`
格式化金额显示。

格式规则：
- >= 1亿：显示为 "X.XX亿"
- >= 1万：显示为 "X.XX万"
- < 1万：显示为 "X.XX"

#### `_find_chinese_font() -> str | None`
查找可用的中文字体路径。

支持的字体（按优先级）：
- Linux: NotoSansCJK, uming, wqy-microhei, DroidSansFallback
- Windows: msyh, simhei
- macOS: PingFang, STHeiti Light

## 使用示例

```python
from src.training.trainer import Trainer
from src.config.config import AgentType
from src.analysis.demo_analyzer import DemoAnalyzer

# 创建训练器并运行演示
trainer = Trainer(config)
trainer.setup()

# ... 运行演示 ...

# 分析结束后调用
analyzer = DemoAnalyzer(output_dir="analysis_output")
analyzer.analyze(
    trainer=trainer,
    end_reason="population_depleted",
    end_agent_type=AgentType.RETAIL
)
```

## 输出文件

分析图保存路径格式：`{output_dir}/demo_analysis_{YYYYMMDD_HHMMSS}.png`

例如：`analysis_output/demo_analysis_20260107_123456.png`

## 依赖关系

- `src.training.trainer` - 训练器（用于获取数据）
- `src.config.config` - 配置类（AgentType）
- `matplotlib` - 图表生成
- `numpy` - 数值计算

## 常量定义

### AGENT_TYPE_NAMES
Agent 类型中文名称映射：
- `RETAIL` → "散户"
- `RETAIL_PRO` → "高级散户"
- `WHALE` → "庄家"
- `MARKET_MAKER` → "做市商"
