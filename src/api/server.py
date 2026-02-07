"""
FastAPI接口
提供REST API服务
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.postgres import (
    load_daily_data, load_alpha_results, load_backtest_results,
    load_latest_portfolio, save_simulator_portfolio
)
from src.engine.backtest import BacktestEngine
from src.engine.simulator import SimulatorEngine
from src.scheduler.jobs import get_scheduler

app = FastAPI(
    title="A股模拟盘API",
    description="量化交易模拟系统接口",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 数据接口 ============

@app.get("/api/stocks/{symbol}")
async def get_stock_data(
    symbol: str,
    start_date: str = Query(..., description="开始日期 YYYY-MM-DD"),
    end_date: str = Query(..., description="结束日期 YYYY-MM-DD")
):
    """获取股票历史数据"""
    try:
        df = load_daily_data(symbol, start_date, end_date)
        if df.empty:
            return {"data": [], "message": "No data found"}
        
        return {
            "data": df.to_dict(orient='records'),
            "count": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alpha/{date}")
async def get_alpha_signals(
    date: str,
    stock_codes: Optional[List[str]] = Query(None, description="股票代码列表")
):
    """获取指定日期的Alpha信号"""
    try:
        df = load_alpha_results(date, stock_codes)
        if df.empty:
            return {"data": {}, "message": "No alpha data found"}
        
        return {
            "data": df.to_dict(),
            "count": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/latest")
async def get_latest_portfolio():
    """获取最新持仓"""
    try:
        df = load_latest_portfolio()
        return {
            "data": df.to_dict(orient='records') if not df.empty else [],
            "count": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 回测接口 ============

@app.get("/api/backtest/history")
async def get_backtest_history(limit: int = Query(100, ge=1, le=1000)):
    """获取历史回测记录"""
    try:
        df = load_backtest_results(limit)
        return {
            "data": df.to_dict(orient='records') if not df.empty else [],
            "count": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/run")
async def run_backtest(request: Dict[str, Any]):
    """
    运行回测
    
    Request body:
    {
        "factor_names": ["alpha1", "alpha2"],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_capital": 100000,
        "num_positions": 10
    }
    """
    try:
        engine = BacktestEngine(
            initial_capital=request.get('initial_capital', 100000),
            num_positions=request.get('num_positions', 10)
        )
        
        # 这里需要先加载数据，暂时返回示例结果
        return {
            "status": "success",
            "message": "Backtest completed",
            "results": [
                {
                    "factor_name": "alpha1",
                    "annual_return": 0.18,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": 0.12,
                    "win_rate": 0.55
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 模拟盘接口 ============

@app.post("/api/simulator/trade")
async def execute_trade(request: Dict[str, Any]):
    """
    执行交易
    
    Request body:
    {
        "date": "2024-12-20",
        "stock_code": "000001",
        "action": "buy",
        "shares": 1000
    }
    """
    try:
        return {
            "status": "success",
            "message": "Trade executed",
            "trade": request
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ 任务调度接口 ============

@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """获取调度器状态"""
    scheduler = get_scheduler()
    return scheduler.get_status()


@app.post("/api/scheduler/trigger/{job_id}")
async def trigger_job(job_id: str):
    """触发任务立即执行"""
    scheduler = get_scheduler()
    scheduler.run_job_now(job_id)
    return {"status": "success", "message": f"Job {job_id} triggered"}


# ============ 健康检查 ============

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/")
async def root():
    """API根路径"""
    return {
        "name": "A股模拟盘API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
