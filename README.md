# 🧡 A股量化模拟盘系统

基于101个Alpha因子的A股量化模拟盘系统，包含回测和实盘模拟功能。

## 功能特性

- 📊 **68个Alpha因子**：基于Kakushadze的101个公式化Alpha
- 🔄 **回测引擎**：批量回测、指标计算、因子筛选
- 🎮 **模拟盘**：实时调仓、持仓管理、信号生成
- 📈 **Streamlit界面**：净值曲线、持仓展示、交易信号
- ⚙️ **任务调度**：每日数据更新、每周自动调仓
- 🌐 **API接口**：FastAPI提供REST服务

## 安装

```bash
# 创建虚拟环境
uv venv
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

## 配置

创建 `.env` 文件：

```env
# Database
POSTGRES_HOST=192.168.2.151
POSTGRES_PORT=15432
POSTGRES_USER=oooo
POSTGRES_PASSWORD=543356
POSTGRES_DB=stock

# Redis
REDIS_HOST=192.168.2.151
REDIS_PORT=16379

# 回测参数
INITIAL_CAPITAL=100000
TRANSACTION_FEE=0.0005
SLIPPAGE=0.001
```

## 使用方法

### 1. 更新数据

```python
from src.data.akshare_fetcher import AStockDataFetcher

fetcher = AStockDataFetcher()
stocks = fetcher.get_stock_list()  # 获取股票列表
df = fetcher.get_daily_data("000001", "20240101", "20240601")  # 获取日线数据
```

### 2. 运行回测

```python
from src.engine.backtest import BacktestEngine

engine = BacktestEngine(
    initial_capital=100000,
    fee_rate=0.0005,
    num_positions=10
)

# 运行所有因子回测
results = engine.run_all_factors(data, factor_names=['alpha1', 'alpha2', 'alpha3'])

# 筛选表现好的因子
filtered = engine.filter_factors(results, min_return=0.15, min_sharpe=1.0)
```

### 3. 启动模拟盘

```python
from src.engine.simulator import SimulatorEngine

engine = SimulatorEngine(initial_capital=100000)
result = engine.run_daily("2024-12-20", data, factor_name='alpha1')
```

### 4. 启动前端

```bash
streamlit run frontend/app.py
```

### 5. 启动API服务

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### 6. 启动调度器

```python
from src.scheduler.jobs import get_scheduler

scheduler = get_scheduler()
scheduler.add_daily_update(hour=16, minute=0)  # 每日16点更新数据
scheduler.add_weekly_rebalance(day_of_week=0, hour=9, minute=30)  # 每周一9:30调仓
scheduler.start()
```

## 项目结构

```
stock_simulator/
├── src/
│   ├── config.py          # 配置模块
│   ├── api/
│   │   └── server.py      # FastAPI接口
│   ├── data/
│   │   └── akshare_fetcher.py  # 数据获取
│   ├── database/
│   │   ├── postgres.py    # PostgreSQL操作
│   │   └── redis_cache.py  # Redis缓存
│   ├── engine/
│   │   ├── backtest.py    # 回测引擎
│   │   └── simulator.py    # 模拟盘引擎
│   ├── models/
│   │   └── alpha_factors.py  # Alpha因子计算
│   └── scheduler/
│       └── jobs.py        # 任务调度
├── frontend/
│   └── app.py            # Streamlit界面
├── tests/
│   ├── test_alpha_factors.py
│   └── test_backtest.py
├── .env                  # 环境配置
├── pyproject.toml        # 项目配置
└── README.md
```

## Alpha因子

已实现68个因子（剔除含行业中性化的因子）：

- `#1-47, #49-57, #60-62, #71, #83-86, #88, #92, #95, #101`

## 评估指标

| 指标 | 描述 | 筛选标准 |
|------|------|----------|
| 年化收益率 | (最终净值/初始净值)^(252/交易日) - 1 | > 15% |
| 夏普比率 | √252 × 平均日收益 / 日收益标准差 | > 1.0 |
| 最大回撤 | 最大峰到谷跌幅 | < 20% |
| 胜率 | 正收益天数 / 总天数 | > 50% |

## 作者

Created by 星期五 🧡

## 许可证

MIT
