"""
任务调度模块
每日数据更新、调仓、报告生成
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Callable
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DataConfig, SimulatorConfig
from src.data.akshare_fetcher import AStockDataFetcher
from src.database.postgres import save_daily_data
import loguru
logger = loguru.logger


class SchedulerManager:
    """调度管理器"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.fetcher = AStockDataFetcher()
        self.is_running = False
    
    def start(self):
        """启动调度器"""
        if not self.scheduler.running:
            self.scheduler.start()
            self.is_running = True
            logger.info("Scheduler started")
    
    def stop(self):
        """停止调度器"""
        self.scheduler.shutdown()
        self.is_running = False
        logger.info("Scheduler stopped")
    
    def add_daily_update(self, hour: int = 16, minute: int = 0):
        """
        添加每日数据更新任务
        
        Args:
            hour: 触发时间（小时）
            minute: 触发时间（分钟）
        """
        trigger = CronTrigger(hour=hour, minute=minute)
        
        self.scheduler.add_job(
            self._daily_update_job,
            trigger=trigger,
            id='daily_update',
            name='每日A股数据更新',
            replace_existing=True
        )
        
        logger.info(f"Scheduled daily update at {hour:02d}:{minute:02d}")
    
    def add_weekly_rebalance(self, day_of_week: int = 0, hour: int = 9, minute: int = 30):
        """
        添加每周调仓任务
        
        Args:
            day_of_week: 星期几 (0=周一)
            hour: 触发时间（小时）
            minute: 触发时间（分钟）
        """
        trigger = CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute)
        
        self.scheduler.add_job(
            self._weekly_rebalance_job,
            trigger=trigger,
            id='weekly_rebalance',
            name='每周调仓',
            replace_existing=True
        )
        
        logger.info(f"Scheduled weekly rebalance on day {day_of_week} at {hour:02d}:{minute:02d}")
    
    def _daily_update_job(self):
        """每日更新任务"""
        logger.info("Running daily update job...")
        try:
            # 获取交易日历
            today = datetime.now().strftime('%Y%m%d')
            cal = self.fetcher.get_trading_calender(today, today)
            
            if cal.empty:
                logger.info(f"{today} is not a trading day")
                return
            
            # 获取全部A股列表
            stocks = self.fetcher.get_stock_list()
            
            # 更新昨日数据
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            
            # 批量更新（取前100只演示）
            for i, symbol in enumerate(stocks['symbol'].head(100)):
                try:
                    df = self.fetcher.get_daily_data(symbol, yesterday, yesterday)
                    if not df.empty:
                        save_daily_data(symbol, df)
                        logger.debug(f"Updated {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to update {symbol}: {e}")
            
            logger.info(f"Daily update completed: {len(stocks)} stocks checked")
            
        except Exception as e:
            logger.error(f"Daily update failed: {e}")
    
    def _weekly_rebalance_job(self):
        """每周调仓任务"""
        logger.info("Running weekly rebalance job...")
        try:
            # 这里调用模拟盘引擎的调仓逻辑
            logger.info("Weekly rebalance triggered")
            
            # TODO: 调用模拟盘引擎
            # engine = SimulatorEngine()
            # engine.run_daily()
            
        except Exception as e:
            logger.error(f"Weekly rebalance failed: {e}")
    
    def run_job_now(self, job_id: str):
        """立即运行任务"""
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                job.func()
                logger.info(f"Job {job_id} executed immediately")
            else:
                logger.warning(f"Job {job_id} not found")
        except Exception as e:
            logger.error(f"Failed to run job {job_id}: {e}")
    
    def get_status(self) -> dict:
        """获取调度器状态"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': str(job.next_run_time) if job.next_run_time else None,
            })
        
        return {
            'running': self.is_running,
            'jobs': jobs,
        }


# 单例实例
_scheduler: Optional[SchedulerManager] = None


def get_scheduler() -> SchedulerManager:
    """获取调度器单例"""
    global _scheduler
    if _scheduler is None:
        _scheduler = SchedulerManager()
    return _scheduler


if __name__ == "__main__":
    # 测试
    scheduler = SchedulerManager()
    scheduler.start()
    
    # 添加每日任务
    scheduler.add_daily_update(hour=16, minute=0)
    
    # 添加每周调仓任务
    scheduler.add_weekly_rebalance(day_of_week=0, hour=9, minute=30)
    
    # 打印状态
    print(scheduler.get_status())
    
    # 保持运行
    try:
        import time
        time.sleep(60)
    except KeyboardInterrupt:
        scheduler.stop()
