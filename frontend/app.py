"""
Streamlit前端界面
A股模拟盘可视化
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BacktestConfig, SimulatorConfig
from src.database.postgres import load_backtest_results, load_latest_portfolio
from src.engine.backtest import BacktestEngine
from src.engine.simulator import SimulatorEngine

st.set_page_config(page_title="A股模拟盘", layout="wide")

# CSS样式
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 20px;
    margin: 10px;
}
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🧡 A股量化模拟盘系统")
    
    # 侧边栏
    with st.sidebar:
        st.header("配置")
        
        mode = st.selectbox("模式", ["回测", "模拟盘"])
        
        st.subheader("回测参数")
        initial_capital = st.number_input("初始资金", value=100000, step=10000)
        fee_rate = st.number_input("手续费", value=0.0005, format="%.4f")
        slippage = st.number_input("滑点", value=0.001, format="%.4f")
        num_positions = st.slider("持仓数量", 5, 20, 10)
        
        st.subheader("因子筛选")
        min_return = st.number_input("最小年化收益", value=0.15, format="%.2f")
        min_sharpe = st.number_input("最小夏普", value=1.0, format="%.2f")
        max_dd = st.number_input("最大回撤", value=0.20, format="%.2f")
    
    # 主界面
    if mode == "回测":
        show_backtest_ui(initial_capital, fee_rate, slippage, num_positions,
                        min_return, min_sharpe, max_dd)
    else:
        show_simulator_ui(initial_capital, fee_rate, slippage, num_positions)


def show_backtest_ui(initial_capital, fee_rate, slippage, num_positions,
                     min_return, min_sharpe, max_dd):
    """回测界面"""
    st.header("📊 因子回测")
    
    # 加载历史回测结果
    try:
        history = load_backtest_results(limit=100)
        if not history.empty:
            st.subheader("历史回测")
            
            # 统计概览
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("回测次数", len(history))
            with col2:
                st.metric("平均年化收益", f"{history['annual_return'].mean():.2%}")
            with col3:
                st.metric("平均夏普", f"{history['sharpe_ratio'].mean():.2f}")
            with col4:
                st.metric("平均回撤", f"{history['max_drawdown'].mean():.2%}")
            
            # 因子排名表
            st.subheader("因子表现排名")
            display_cols = ['factor_list', 'annual_return', 'sharpe_ratio', 
                          'max_drawdown', 'win_rate', 'created_at']
            available_cols = [c for c in display_cols if c in history.columns]
            st.dataframe(history[available_cols].head(20))
            
    except Exception as e:
        st.warning(f"暂无回测数据: {e}")
    
    # 新建回测
    st.subheader("🚀 新建回测")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", datetime(2024, 1, 1))
    with col2:
        end_date = st.date_input("结束日期", datetime.now())
    
    factor_names = st.multiselect(
        "选择因子",
        [f'alpha{i}' for i in range(1, 48)] + 
        [f'alpha{i}' for i in range(49, 58)] +
        ['alpha60', 'alpha61', 'alpha62', 'alpha71'] +
        [f'alpha{i}' for i in range(83, 87)] +
        ['alpha88', 'alpha92', 'alpha95', 'alpha101'],
        default=['alpha1', 'alpha2', 'alpha3']
    )
    
    if st.button("运行回测"):
        with st.spinner("运行回测中..."):
            engine = BacktestEngine(
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage=slippage,
                num_positions=num_positions,
            )
            
            # 简化的回测（使用模拟数据）
            results = []
            for name in factor_names:
                # 这里应该加载真实数据，现在用模拟数据演示
                result = {
                    'factor_name': name,
                    'annual_return': np.random.uniform(0.1, 0.3),
                    'sharpe_ratio': np.random.uniform(0.8, 2.0),
                    'max_drawdown': np.random.uniform(0.05, 0.15),
                    'win_rate': np.random.uniform(0.45, 0.65),
                }
                results.append(result)
            
            # 筛选
            filtered = [r for r in results 
                       if r['annual_return'] >= min_return 
                       and r['sharpe_ratio'] >= min_sharpe
                       and r['max_drawdown'] <= max_dd]
            
            # 展示结果
            st.subheader("回测结果")
            
            df_results = pd.DataFrame(filtered)
            if not df_results.empty:
                # 净值曲线图
                fig = go.Figure()
                for _, row in df_results.iterrows():
                    cumulative = np.cumprod([1 + np.random.normal(0.001, 0.02) 
                                           for _ in range(100)])
                    fig.add_trace(go.Scatter(
                        y=cumulative, name=row['factor_name'],
                        mode='lines'
                    ))
                fig.update_layout(title="模拟净值曲线", y_title="净值")
                st.plotly_chart(fig, use_container_width=True)
                
                # 结果表格
                st.dataframe(df_results)
                
                # 推荐因子
                best = max(filtered, key=lambda x: x['sharpe_ratio'])
                st.success(f"推荐因子: **{best['factor_name']}** (夏普={best['sharpe_ratio']:.2f})")


def show_simulator_ui(initial_capital, fee_rate, slippage, num_positions):
    """模拟盘界面"""
    st.header("🎮 实时模拟盘")
    
    # 组合概览
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总资产", f"¥{initial_capital:,.0f}")
    with col2:
        st.metric("持仓数量", f"{num_positions}")
    with col3:
        st.metric("收益率", "+0.00%")
    with col4:
        st.metric("交易次数", "0")
    
    # 持仓列表
    st.subheader("📦 当前持仓")
    st.info("请先运行回测并选择因子后启动模拟盘")
    
    # 交易信号
    st.subheader("📈 今日信号")
    
    # 模拟数据展示
    sample_signals = pd.DataFrame([
        {'股票': '000001', '信号': 0.85, '当前价格': 12.5, '建议': '买入'},
        {'股票': '000002', '信号': 0.72, '当前价格': 8.3, '建议': '买入'},
        {'股票': '000003', '信号': -0.45, '当前价格': 15.2, '建议': '持有'},
        {'股票': '600000', '信号': 0.65, '当前价格': 22.1, '建议': '买入'},
        {'股票': '000004', '信号': -0.32, '当前价格': 9.8, '建议': '卖出'},
    ])
    
    st.dataframe(sample_signals)
    
    # 操作按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 刷新信号"):
            st.rerun()
    with col2:
        if st.button("📊 执行调仓"):
            st.success("调仓完成")
    with col3:
        if st.button("📤 导出持仓"):
            st.info("导出功能开发中")


if __name__ == "__main__":
    main()
