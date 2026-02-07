"""Tests for Alpha Factor Module"""
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.alpha_factors import AlphaCalculator, rank, delay, delta, correlation, stddev


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    symbols = ['000001', '000002', '000003', '000004', '000005']
    
    data = []
    for date in dates:
        for symbol in symbols:
            data.append({
                'date': date,
                'symbol': symbol,
                'open': np.random.uniform(5, 20),
                'high': np.random.uniform(5, 20),
                'low': np.random.uniform(5, 20),
                'close': np.random.uniform(5, 20),
                'volume': np.random.uniform(1000000, 10000000),
                'amount': np.random.uniform(10000000, 100000000),
            })
    
    df = pd.DataFrame(data)
    return df.set_index(['date', 'symbol'])


@pytest.fixture
def calculator():
    """Create AlphaCalculator instance"""
    return AlphaCalculator(use_cache=False)


class TestHelperFunctions:
    """Test helper functions"""
    
    def test_delay(self, sample_data):
        """Test delay function"""
        close = sample_data['close']
        delayed = delay(close, 1)
        assert delayed is not None
        assert len(delayed) == len(close)
    
    def test_delta(self, sample_data):
        """Test delta function"""
        close = sample_data['close']
        delta_values = delta(close, 1)
        assert delta_values is not None
        # First few values should be NaN due to shift
        assert np.isnan(delta_values.iloc[0])
    
    def test_rank(self, sample_data):
        """Test rank function"""
        close = sample_data['close']
        ranked = rank(close)
        assert ranked is not None
        # Rank should be between 0 and 1
        assert ranked.min() >= 0
        assert ranked.max() <= 1
    
    def test_correlation(self, sample_data):
        """Test correlation function"""
        open_ = sample_data['open']
        close = sample_data['close']
        corr = correlation(open_, close, 10)
        assert corr is not None
        # Correlation should be between -1 and 1
        assert corr.min() >= -1
        assert corr.max() <= 1
    
    def test_stddev(self, sample_data):
        """Test stddev function"""
        close = sample_data['close']
        std = stddev(close, 10)
        assert std is not None
        # First 9 values should be NaN
        assert np.isnan(std.iloc[0])


class TestAlphaCalculator:
    """Test AlphaCalculator class"""
    
    def test_prepare_data(self, calculator, sample_data):
        """Test data preparation"""
        prepared = calculator._prepare_data(sample_data)
        assert 'returns' in prepared.columns
        assert 'vwap' in prepared.columns
        assert 'adv20' in prepared.columns
        assert 'cap' in prepared.columns
    
    def test_calculate_all_alphas(self, calculator, sample_data):
        """Test calculating all alphas"""
        alphas = calculator.calculate_all_alphas(sample_data)
        assert isinstance(alphas, pd.DataFrame)
        assert len(alphas) > 0
        # Should have multiple alpha columns
        assert alphas.shape[1] > 10
    
    def test_calculate_single_alpha(self, calculator, sample_data):
        """Test calculating single alpha"""
        alpha1 = calculator.calculate_single_alpha(sample_data, 'alpha1')
        assert alpha1 is not None
        assert len(alpha1) == len(sample_data)
    
    def test_alpha1(self, calculator, sample_data):
        """Test Alpha#1"""
        result = calculator.alpha1(sample_data)
        assert result is not None
        assert not result.isna().all()
    
    def test_alpha2(self, calculator, sample_data):
        """Test Alpha#2"""
        result = calculator.alpha2(sample_data)
        assert result is not None
        assert not result.isna().all()
    
    def test_alpha3(self, calculator, sample_data):
        """Test Alpha#3"""
        result = calculator.alpha3(sample_data)
        assert result is not None
    
    def test_alpha4(self, calculator, sample_data):
        """Test Alpha#4"""
        result = calculator.alpha4(sample_data)
        assert result is not None
    
    def test_alpha5(self, calculator, sample_data):
        """Test Alpha#5"""
        result = calculator.alpha5(sample_data)
        assert result is not None
    
    def test_alpha6(self, calculator, sample_data):
        """Test Alpha#6"""
        result = calculator.alpha6(sample_data)
        assert result is not None
    
    def test_alpha10(self, calculator, sample_data):
        """Test Alpha#10 (complex ternary)"""
        result = calculator.alpha10(sample_data)
        assert result is not None
    
    def test_alpha21(self, calculator, sample_data):
        """Test Alpha#21 (complex conditions)"""
        result = calculator.alpha21(sample_data)
        assert result is not None
    
    def test_alpha30(self, calculator, sample_data):
        """Test Alpha#30"""
        result = calculator.alpha30(sample_data)
        assert result is not None
    
    def test_alpha36(self, calculator, sample_data):
        """Test Alpha#36 (complex formula)"""
        result = calculator.alpha36(sample_data)
        assert result is not None
    
    def test_alpha41(self, calculator, sample_data):
        """Test Alpha#41"""
        result = calculator.alpha41(sample_data)
        assert result is not None
    
    def test_alpha42(self, calculator, sample_data):
        """Test Alpha#42"""
        result = calculator.alpha42(sample_data)
        assert result is not None
    
    def test_alpha49(self, calculator, sample_data):
        """Test Alpha#49"""
        result = calculator.alpha49(sample_data)
        assert result is not None
    
    def test_alpha50(self, calculator, sample_data):
        """Test Alpha#50"""
        result = calculator.alpha50(sample_data)
        assert result is not None
    
    def test_alpha60(self, calculator, sample_data):
        """Test Alpha#60"""
        result = calculator.alpha60(sample_data)
        assert result is not None
    
    def test_alpha71(self, calculator, sample_data):
        """Test Alpha#71"""
        result = calculator.alpha71(sample_data)
        assert result is not None
    
    def test_alpha83(self, calculator, sample_data):
        """Test Alpha#83"""
        result = calculator.alpha83(sample_data)
        assert result is not None
    
    def test_alpha84(self, calculator, sample_data):
        """Test Alpha#84"""
        result = calculator.alpha84(sample_data)
        assert result is not None
    
    def test_alpha85(self, calculator, sample_data):
        """Test Alpha#85"""
        result = calculator.alpha85(sample_data)
        assert result is not None
    
    def test_alpha86(self, calculator, sample_data):
        """Test Alpha#86"""
        result = calculator.alpha86(sample_data)
        assert result is not None
    
    def test_alpha88(self, calculator, sample_data):
        """Test Alpha#88"""
        result = calculator.alpha88(sample_data)
        assert result is not None
    
    def test_alpha92(self, calculator, sample_data):
        """Test Alpha#92"""
        result = calculator.alpha92(sample_data)
        assert result is not None
    
    def test_alpha95(self, calculator, sample_data):
        """Test Alpha#95"""
        result = calculator.alpha95(sample_data)
        assert result is not None
    
    def test_alpha101(self, calculator, sample_data):
        """Test Alpha#101"""
        result = calculator.alpha101(sample_data)
        assert result is not None
    
    def test_unknown_alpha(self, calculator, sample_data):
        """Test unknown alpha name"""
        with pytest.raises(ValueError):
            calculator.calculate_single_alpha(sample_data, 'alpha999')


class TestEdgeCases:
    """Test edge cases"""
    
    def test_empty_data(self, calculator):
        """Test with empty data"""
        empty_df = pd.DataFrame(columns=['date', 'symbol', 'open', 'close', 'high', 'low', 'volume', 'amount'])
        empty_df = empty_df.set_index(['date', 'symbol'])
        # Should not crash
        result = calculator.calculate_all_alphas(empty_df)
        assert result is not None
    
    def test_single_stock(self, calculator):
        """Test with single stock"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        data = {
            'date': dates,
            'symbol': ['000001'] * 50,
            'open': np.random.uniform(5, 20, 50),
            'high': np.random.uniform(5, 20, 50),
            'low': np.random.uniform(5, 20, 50),
            'close': np.random.uniform(5, 20, 50),
            'volume': np.random.uniform(1000000, 10000000, 50),
            'amount': np.random.uniform(10000000, 100000000, 50),
        }
        df = pd.DataFrame(data).set_index(['date', 'symbol'])
        result = calculator.calculate_all_alphas(df)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
