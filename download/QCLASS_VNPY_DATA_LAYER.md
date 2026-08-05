# qclass_vnpy_data.py 数据中间层说明

## 0. 先认识 vn.py / VeighNa

vn.py，也叫 VeighNa，是一个用 Python 写的开源量化交易开发框架，代码仓库在 [vnpy/vnpy](https://github.com/vnpy/vnpy)。它提供了很多量化交易系统里常见的基础部件，例如行情数据对象、事件引擎、交易请求、回测引擎、CTA 策略模板、数据库接口和图形化交易端。

对这门课来说，学生不需要一上来理解完整 vn.py 源码。可以先把它理解成一个“交易系统积木包”：

- `BarData` / `TickData`：行情数据对象。
- `HistoryRequest`：向数据源请求历史行情。
- `EventEngine`：事件驱动系统，把行情、信号、订单等消息分发给不同模块。
- `OrderRequest` / `CancelRequest`：下单和撤单请求。
- `BacktestingEngine`：回测引擎。
- `CtaTemplate`：写策略时继承的模板类。

本项目的目标不是替代 vn.py，而是让 vn.py 可以直接使用咱们的 `data_releases/learning` 数据。

## 1. 如何下载和配置 vn.py

如果只是为了运行本项目里的教学 demo，推荐直接在当前目录安装依赖：

```bash
cd "/Users/promcrdog/Documents/Vector lab/Qclass/vnpy"
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
```

如果在国内网络环境安装慢，可以用清华源：

```bash
python -m pip install --upgrade pip wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果想下载 vn.py 源码做参考，可以单独 clone 官方仓库：

```bash
cd "/Users/promcrdog/Documents/Vector lab/Qclass"
git clone https://github.com/vnpy/vnpy.git vnpy-source
cd vnpy-source
```

注意：源码仓库主要用于阅读和二次开发；本教学项目运行时仍然建议在 `/Users/promcrdog/Documents/Vector lab/Qclass/vnpy` 里用 `requirements.txt` 安装依赖。

macOS 如果安装 `ta-lib` 报错，通常需要先安装系统库：

```bash
brew install ta-lib
```

可以把下面这段 prompt 直接发给 AI，让它根据自己的电脑环境生成配置步骤：

```text
你是 Python 环境配置助手。请帮我在我的电脑上配置 vn.py / VeighNa 的学习环境。

我的系统是：<填写 macOS / Windows / Linux>
我的 Python 版本是：<填写 python --version 的结果>
我的项目目录是：/Users/promcrdog/Documents/Vector lab/Qclass/vnpy
我要运行的项目依赖文件是：requirements.txt
我需要运行这些命令：
1. python qclass_vnpy_examples.py --data-root data_releases/learning --vt-symbol 000001.SZSE --start 2020-01-02 --end 2020-06-30
2. python qclass_vnpy_examples.py --data-root data_releases/learning --vt-symbol 000001.SZSE --start 2020-01-02 --end 2020-06-30 --run-backtest

请你给我一步一步的命令，包括：
- 如何创建虚拟环境；
- 如何激活虚拟环境；
- 如何安装 requirements.txt；
- 如果安装 ta-lib、PySide6、pyarrow、vnpy、vnpy_ctastrategy 出错，分别应该怎么排查；
- 如何确认 vn.py 已经安装成功；
- 如何运行上面的两个 demo；
- 如果我是国内网络环境，请给出使用清华 PyPI 镜像的命令。
```

## 2. 这个中间层解决什么问题

为了连接咱们的数据与 vn.py，现在多了一层很薄的数据适配器：`qclass_vnpy_data.py`。

它参考 SickFun (platform6) 的真实数据路径：

```text
new_data
  -> platform6 release builder
  -> data_releases/learning 或 data_releases/hidden_test
  -> data_catalog.json
  -> DataBundle.at(date).load/current/history/latest
```

在 vn.py 项目里，这一层继续保留同样的日期含义，同时把股票日行情转成 vn.py 常用对象：

```text
QClassDataBundle
  -> QClassDatedData.load/current/history/latest
  -> QClassVnpyDatafeed.query_bar_history/query_tick_history
  -> QClassVnpyDatabase.load_bar_data/load_tick_data
  -> load_qclass_history_to_engine(...)
```

## 3. 安装本项目依赖

```bash
cd "/Users/promcrdog/Documents/Vector lab/Qclass/vnpy"
source .venv/bin/activate
python -m pip install -r requirements.txt
```

如果在国内网络环境安装慢，可以用清华源：

```bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`pyarrow` 是必须的，因为 `stock.daily_returns` 是 parquet 文件。

建议课堂和作业默认使用 `data_releases/learning`。`QClassDataBundle` 也可以扫描没有 `data_catalog.json` 的 raw `new_data` 目录，但 raw CSMAR 文件可能包含 platform6 release builder 才会修复的厂商 CSV 格式问题，而且也没有学习集/隐藏集日期边界。学生侧最好使用 release。

## 4. 像 SickFun 一样读取数据

```python
from qclass_vnpy_data import QClassDataBundle

bundle = QClassDataBundle("data_releases/learning")
data = bundle.at("2020-06-30")

today = data.current(
    "stock.daily_returns",
    columns=["stkcd", "trddt", "clsprc", "dretwd", "dnvaltrd"],
    numeric_columns=["clsprc", "dretwd", "dnvaltrd"],
)

history = data.history(
    "stock.daily_returns",
    lookback_days=20,
    columns=["stkcd", "trddt", "clsprc"],
)

latest = data.latest(
    "stock.daily_returns",
    columns=["stkcd", "trddt", "clsprc", "dretwd"],
    numeric_columns=["clsprc", "dretwd"],
)
```

几个规则和 SickFun 保持一致：

- `bundle.at("2020-06-30")` 会创建一个日期边界。
- `load()`、`current()`、`history()`、`latest()` 不允许读取这个日期之后的数据。
- 如果请求未来数据，会抛出 `FutureDataAccessError`。
- `columns` 控制返回哪些列。
- `numeric_columns` 会把指定列转成数字，无法转换的值会变成缺失值。

## 5. 作为 vn.py Datafeed 使用

```python
from datetime import datetime

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest

from qclass_vnpy_data import QClassVnpyDatafeed

datafeed = QClassVnpyDatafeed("data_releases/learning")

req = HistoryRequest(
    symbol="000001",
    exchange=Exchange.SZSE,
    start=datetime(2020, 1, 1),
    end=datetime(2020, 6, 30),
    interval=Interval.DAILY,
)

bars = datafeed.query_bar_history(req)
ticks = datafeed.query_tick_history(req)
```

`query_bar_history()` 会把 `stock.daily_returns` 的真实字段映射成 `BarData`：

| CSMAR 字段 | vn.py 字段 |
|---|---|
| `stkcd` | `BarData.symbol` |
| `trddt` | `BarData.datetime` |
| `opnprc` | `open_price` |
| `hiprc` | `high_price` |
| `loprc` | `low_price` |
| `clsprc` | `close_price` |
| `dnshrtrd` | `volume` |
| `dnvaltrd` | `turnover` |

`query_tick_history()` 会从日行情合成一个收盘 tick，方便学习 `TickData`、`on_tick`、事件引擎等 vn.py 概念。它不是逐笔真实 tick。

## 6. 作为只读 vn.py Database 使用

```python
from datetime import datetime
from vnpy.trader.constant import Exchange, Interval
from qclass_vnpy_data import QClassVnpyDatabase

database = QClassVnpyDatabase("data_releases/learning")

bars = database.load_bar_data(
    symbol="000001",
    exchange=Exchange.SZSE,
    interval=Interval.DAILY,
    start=datetime(2020, 1, 1),
    end=datetime(2020, 6, 30),
)
```

这个 database 是只读的。`save_bar_data()`、`save_tick_data()`、`delete_bar_data()`、`delete_tick_data()` 会抛出 `ReadOnlyDataError`，避免学生误以为 release 数据会被写回。

## 7. 直接喂给 BacktestingEngine

vn.py 的 `BacktestingEngine.load_data()` 默认从 vn.py 本地数据库读取。课程数据已经在 `data_releases` 里，所以可以直接把中间层读到的 bar 填入引擎：

```python
from datetime import datetime

from vnpy.trader.constant import Interval
from vnpy_ctastrategy.backtesting import BacktestingEngine

from double_ma_stock_strategy import DoubleMaStockStrategy
from qclass_vnpy_data import load_qclass_history_to_engine

engine = BacktestingEngine()
engine.set_parameters(
    vt_symbol="000001.SZSE",
    interval=Interval.DAILY,
    start=datetime(2020, 1, 1),
    end=datetime(2020, 6, 30),
    rate=3 / 10000,
    slippage=0.01,
    size=1,
    pricetick=0.01,
    capital=100000,
)

engine.add_strategy(
    DoubleMaStockStrategy,
    {"fast_window": 5, "slow_window": 20, "fixed_size": 100},
)

load_qclass_history_to_engine(engine, "data_releases/learning")
engine.run_backtesting()
result = engine.calculate_result()
stats = engine.calculate_statistics(output=False)
```

这样学生仍然在写标准 vn.py `CtaTemplate` 策略，只是历史数据来源换成了 QClass release。

## 8. 运行完整示例

只展示数据、vn.py 对象、事件引擎和订单对象：

```bash
cd "/Users/promcrdog/Documents/Vector lab/Qclass/vnpy"
source .venv/bin/activate

python qclass_vnpy_examples.py \
  --data-root data_releases/learning \
  --vt-symbol 000001.SZSE \
  --start 2020-01-02 \
  --end 2020-06-30
```

同时运行 CTA 回测：

```bash
python qclass_vnpy_examples.py \
  --data-root data_releases/learning \
  --vt-symbol 000001.SZSE \
  --start 2020-01-02 \
  --end 2020-06-30 \
  --run-backtest
```

输出文件：

```text
qclass_vnpy_backtest_daily_result.csv
```

## 9. 贵州茅台 MACD K线图示例

`maotai_macd_kline_strategy.py` 是一个更接近“策略 + 可视化作业”的 sample。它会从 `data_releases/learning` 读取贵州茅台 `600519.SSE` 的日线数据，计算基础 MACD，并在 K线图上画出 MACD 金叉买入点和死叉卖出点。

运行命令：

```bash
cd "/Users/promcrdog/Documents/Vector lab/Qclass/vnpy"
source .venv/bin/activate

python maotai_macd_kline_strategy.py \
  --data-root data_releases/learning \
  --start 2020-01-02 \
  --end 2020-12-31 \
  --output-html maotai_macd_kline.html \
  --output-csv maotai_macd_signals.csv
```

输出文件：

```text
maotai_macd_kline.html
maotai_macd_signals.csv
```

策略逻辑很简单：

- 计算 12 日 EMA 和 26 日 EMA。
- `macd = ema_fast - ema_slow`。
- `macd_signal` 是 MACD 的 9 日 EMA。
- MACD 上穿 signal，标记为买入点。
- MACD 下穿 signal，标记为卖出点。
- 图里第一层是 K线和买卖点，第二层是成交量，第三层是 MACD 柱和两条 MACD 线。

这个示例只用于教学，不代表真实投资建议。

## 10. 示例覆盖了哪些 vn.py 功能

`qclass_vnpy_examples.py` 覆盖这些核心 vn.py 概念：

- `HistoryRequest`：请求一段历史行情。
- `BarData`：日 K 线对象。
- `TickData`：由日行情合成的教学 tick。
- `Exchange`、`Interval`、`Direction`、`Offset`、`OrderType`、`Product`：vn.py 常用枚举。
- `SubscribeRequest`、`OrderRequest`、`CancelRequest`：订阅、下单、撤单请求对象。
- `ContractData`、`AccountData`、`PositionData`：合约、账户、持仓对象。
- `EventEngine` 和 `Event`：事件驱动信号分发。
- `BacktestingEngine` 和 `CtaTemplate`：标准 CTA 回测流程。

这不是实盘网关，也不会连接券商。它的目标是让学生在课程数据上学习 vn.py 的核心对象、事件流、策略回调和回测流程。

## 11. 运行测试

```bash
cd "/Users/promcrdog/Documents/Vector lab/Qclass/vnpy"
source .venv/bin/activate
python -m unittest tests.test_qclass_vnpy_data -v
python -m unittest tests.test_maotai_macd_kline_strategy -v
```

测试会验证：

- `load/current/history/latest` 的日期边界语义。
- 未来数据访问会被拒绝。
- CSMAR 日行情可以转成 `BarData` 和 `TickData`。
- 只读 `BaseDatabase` facade 不允许写入。
- `load_qclass_history_to_engine()` 能把 release bar 填入 vn.py 回测引擎。
