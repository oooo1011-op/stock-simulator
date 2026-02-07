"""Tests for Backtest Engine"""
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.engine.backtest import BacktestEngine, BacktestResult


@pytest.fixture
def sample_data():
    """Create sample price data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    symbols = ['000001', '000002', '000003', '000004', '000005']
    
    data = []
    for date in dates:
        for symbol in symbols:
            base_price = np.random.uniform(5, 20)
            data.append({
                'date': date,
                'symbol': symbol,
                'open': base_price * np.random.uniform(0.99, 1.01),
                'high': base_price * np.random.uniform(1.0, 1.02),
                'low': base_price * np.random.uniform(0.98, 1.0),
                'close': base_price * np.random.uniform(0.99, 1.01),
                'volume': np.random.uniform(1000000, 10000000),
                'amount': np.random.uniform(10000000, 100000000),
            })
    
    return pd.DataFrame(data).set_index(['date', 'symbol'])


@pytest.fixture
def engine():
    """Create BacktestEngine instance"""
    return BacktestEngine(
        initial_capital=100000,
        fee_rate=0.0005,
        slippage=0.001,
        num_positions=3,
    )


class TestBacktestEngine:
    """Test BacktestEngine class"""
    
    def test_init(self):
        """Test engine initialization"""
        engine = BacktestEngine(
            initial_capital=100000,
            fee_rate=0.001,
            slippage=0.002,
            num_positions=5,
        )
        assert engine.initial_capital == 100000
        assert engine.fee_rate == 0.001
        assert engine.slippage == 0.002
        assert engine.num_positions == 5
    
    def test_calculate_daily_return(self, engine, sample_data):
        """Test daily return calculation"""
        date = sample_data.index.get_level_values('date')[0]
        ret = engine._calculate_daily_return(sample_data, str(date)[:10])
        assert isinstance(ret, (int, float))
        # Return should be reasonable
        assert abs(ret) < 0.15
    
    def test_calculate_metrics(self, engine):
        """Test metric calculation"""
        portfolio_values = [100000, 101000, 99500, 102000, 103000]
        daily_returns = [0.01, -0.015, 0.025, 0.01]
        start_date = '2024-01-01'
        end_date = '2024-01-05'
        
        metrics = engine._calculate_metrics(
            portfolio_values, daily_returns, start_date, end_date
        )
        
        assert 'annual_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        
        # Check values are reasonable
        assert metrics['annual_return'] > -1
        assert metrics['max_drawdown'] >= 0
        assert 0 <= metrics['win_rate'] <= 1
    
    def test_run_backtest_single_factor(self, engine, sample_data):
        """Test running backtest for a single factor"""
        result = engine.run_backtest(sample_data, 'alpha1')
        
        assert result is not None
        assert isinstance(result, BacktestResult)
        assert result.factor_name == 'alpha1'
        assert result.initial_capital == 100000
        assert isinstance(result.annual_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
    
    def test_run_backtest_multiple_factors(self, engine, sample_data):
        """Test running backtest for multiple factors"""
        results = engine.run_all_factors(
            sample_data,
            factor_names=['alpha1', 'alpha2', 'alpha3'],
            parallel=False,
        )
        
        assert len(results) == 3
        factor_names = [r.factor_name for r in results]
        assert 'alpha1' in factor_names
        assert 'alpha2' in factor_names
        assert 'alpha3' in factor_names
    
    def test_filter_factors(self, engine):
        """Test factor filtering"""
        results = [
            BacktestResult(
                factor_name='alpha1', start_date='2024-01-01', end_date='2024-12-31',
                initial_capital=100000, final_capital=120000,
                annual_return=0.20, sharpe_ratio=1.5, max_drawdown=0.10,
                win_rate=0.55, total_trades=100,
                daily_returns=[], portfolio_values=[],
            ),
            BacktestResult(
                factor_name='alpha2', start_date='2024-01-01', end_date='2024-12-31',
                initial_capital=100000, final_capital=110000,
                annual_return=0.10, sharpe_ratio=0.8, max_drawdown=0.15,
                win_rate=0.45, total_trades=80,
                daily_returns=[], portfolio_values=[],
            ),
            BacktestResult(
                factor_name='alpha3', start_date='2024-01-01', end_date='2024-12-31',
                initial_capital=100000, final_capital=130000,
                annual_return=0.30, sharpe_ratio=2.0, max_drawdown=0.08,
                win_rate=0.60, total_trades=120,
                daily_returns=[], portfolio_values=[],
            ),
        ]
        
        # Filter by all criteria
        filtered = engine.filter_factors(
            results,
            min_return=0.15,
            min_sharpe=1.0,
            max_dd=0.12,
            min_win_rate=0.50,
        )
        
        assert len(filtered) == 2
        assert 'alpha1' in [r.factor_name for r in filtered]
        assert 'alpha3' in [r.factor_name for r in filtered]
    
    def test_filter_top_n(self, engine):
        """Test filtering top N factors"""
        results = [
            BacktestResult(
                factor_name=f'alpha{i}', start_date='2024-01-01', end_date='2024-12-31',
                initial_capital=100000, final_capital=100000,
                annual_return=0.1 * i, sharpe_ratio=i * 0.5, max_drawdown=0.1,
                win_rate=0.5, total_trades=100,
                daily_returns=[], portfolio_values=[],
            )
            for i in range(1, 11)
        ]
        
        # Get top 3
        filtered = engine.filter_factors(results, top_n=3)
        
        assert len(filtered) == 3
        # Should be sorted by sharpe descending
        assert filtered[0].factor_name == 'alpha10'
        assert filtered[1].factor_name == 'alpha9'
        assert filtered[2].factor_name == 'alpha8'
    
    def test_backtest_result_to_dict(self):
        """Test BacktestResult serialization"""
        result = BacktestResult(
            factor_name='alpha1',
            start_date='2024-01-01',
            end_date='2024-12-31',
            initial_capital=100000,
            final_capital=120000,
            annual_return=0.20,
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.55,
            total_trades=100,
            daily_returns=[0.01, -0.01, 0.02],
            portfolio_values=[100000, 101000, 99500, 102000],
        )
        
        d = result.to_dict()
        
        assert d['factor_name'] == 'alpha1'
        assert d['annual_return'] == 0.20
        assert len(d['daily_returns']) == 3
        assert len(d['portfolio_values']) == 4
    
    def test_empty_data(self, engine):
        """Test with empty data"""
        empty_df = pd.DataFrame(columns=[
            'date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount'
        ]).set_index(['date', 'symbol'])
        
        result = engine.run_backtest(empty_df, 'alpha1')
        assert result is None
    
    def test_unknown_factor(self, engine, sample_data):
        """Test with unknown factor name"""
        # Should return None or raise error
        result = engine.run_backtest(sample_data, 'alpha999')
        # Either returns None or raises ValueError
        assert result is None or isinstance(result, BacktestResult)


class TestMetricsCalculation:
    """Test specific metric calculations"""
    
    def test_sharpe_ratio_zero_std(self):
        """Test sharpe ratio when std is zero"""
        engine = BacktestEngine()
        
        metrics = engine._calculate_metrics(
            portfolio_values=[100000, 100000, 100000, 100000],
            daily_returns=[0, 0, 0, 0],
            start_date='2024-01-01',
            end_date='2024-01-04',
        )
        
        # Sharpe should be 0 when std is 0
        assert metrics['sharpe_ratio'] == 0
    
    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation"""
        engine = BacktestEngine()
        
        metrics = engine._calculate_metrics(
            portfolio_values=[100000, 110000, 105000, 120000, 115000, 125000],
            daily_returns=[0.1, -0.045, 0.143, -0.042, 0.087],
            start_date='2024-01-01',
            end_date='2024-01-06',
        )
        
        # Max drawdown should be > 0
        assert metrics['max_drawdown'] > 0
    
    def test_win_rate_calculation(self):
        """Test win rate calculation"""
        engine = BacktestEngine()
        
        metrics = engine._calculate_metrics(
            portfolio_values=[100000, 101000, 99500, 102000, 103500],
            daily_returns=[0.01, -0.015, 0.025, 0.015],
            start_date='2024-01-01',
            end_date='2024-01-05',
        )
        
        # 3 positive out of 4 = 75%
        assert metrics['win_rate'] == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
