# utils/analysis.py
"""
分析工具函数
提供用于数据分析的辅助函数
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import matplotlib.pyplot as plt


def calculate_returns(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    计算单日和累计收益率
    
    参数:
        df: 包含价格数据的DataFrame
        price_col: 用于计算收益率的价格列名
        
    返回:
        DataFrame: 添加了收益率列的DataFrame
    """
    df = df.copy()
    
    # 计算单日收益率
    df['daily_return'] = df[price_col].pct_change() * 100
    
    # 计算累计收益率
    df['cum_return'] = (1 + df['daily_return'] / 100).cumprod() - 1
    df['cum_return'] = df['cum_return'] * 100
    
    return df


def calculate_volatility(df: pd.DataFrame, window: int = 20, price_col: str = 'close') -> pd.DataFrame:
    """
    计算波动率
    
    参数:
        df: 包含价格数据的DataFrame
        window: 计算窗口大小
        price_col: 用于计算波动率的价格列名
        
    返回:
        DataFrame: 添加了波动率列的DataFrame
    """
    df = df.copy()
    
    # 计算日收益率
    returns = df[price_col].pct_change()
    
    # 计算滚动标准差
    df[f'volatility_{window}d'] = returns.rolling(window=window).std() * np.sqrt(252) * 100
    
    return df


def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.0, 
                          window: int = 252, return_col: str = 'daily_return') -> pd.DataFrame:
    """
    计算夏普比率
    
    参数:
        df: 包含收益率数据的DataFrame
        risk_free_rate: 无风险利率，年化百分比
        window: 计算窗口大小
        return_col: 收益率列名
        
    返回:
        DataFrame: 添加了夏普比率列的DataFrame
    """
    df = df.copy()
    
    # 将无风险利率转换为每日收益率
    daily_rf = risk_free_rate / 252 / 100
    
    # 计算超额收益率
    excess_return = df[return_col] / 100 - daily_rf
    
    # 计算滚动夏普比率
    df[f'sharpe_ratio_{window}d'] = (
        excess_return.rolling(window=window).mean() / 
        excess_return.rolling(window=window).std()
    ) * np.sqrt(252)
    
    return df


def find_support_resistance(df: pd.DataFrame, window: int = 20, price_col: str = 'close') -> Tuple[List[float], List[float]]:
    """
    寻找支撑位和阻力位
    
    参数:
        df: 包含价格数据的DataFrame
        window: 滚动窗口大小
        price_col: 价格列名
        
    返回:
        Tuple: (支撑位列表, 阻力位列表)
    """
    prices = df[price_col].values
    
    support_levels = []
    resistance_levels = []
    
    for i in range(window, len(prices) - window):
        # 检查是否形成局部底部 (支撑位)
        if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
           all(prices[i] <= prices[i+j] for j in range(1, window+1)):
            support_levels.append(prices[i])
        
        # 检查是否形成局部顶部 (阻力位)
        elif all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
             all(prices[i] >= prices[i+j] for j in range(1, window+1)):
            resistance_levels.append(prices[i])
    
    # 合并相近的水平
    support_levels = cluster_price_levels(support_levels)
    resistance_levels = cluster_price_levels(resistance_levels)
    
    return support_levels, resistance_levels


def cluster_price_levels(levels: List[float], threshold_pct: float = 0.02) -> List[float]:
    """
    合并相近的价格水平
    
    参数:
        levels: 价格水平列表
        threshold_pct: 视为相同水平的百分比阈值
        
    返回:
        List: 合并后的价格水平列表
    """
    if not levels:
        return []
    
    # 按升序排序
    sorted_levels = sorted(levels)
    
    # 初始化结果和当前簇
    clustered = []
    current_cluster = [sorted_levels[0]]
    
    for i in range(1, len(sorted_levels)):
        current_price = sorted_levels[i]
        cluster_avg = sum(current_cluster) / len(current_cluster)
        
        # 如果当前价格与簇平均值相差小于阈值，则加入簇
        if abs(current_price - cluster_avg) / cluster_avg <= threshold_pct:
            current_cluster.append(current_price)
        else:
            # 否则，完成当前簇并开始新簇
            clustered.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [current_price]
    
    # 添加最后一个簇
    if current_cluster:
        clustered.append(sum(current_cluster) / len(current_cluster))
    
    return clustered


def plot_price_with_indicators(df: pd.DataFrame, title: str = 'Price Chart with Indicators',
                             support_resistance: bool = True, fibonacci: bool = True):
    """
    绘制带有技术指标的价格图表
    
    参数:
        df: 包含价格和指标数据的DataFrame
        title: 图表标题
        support_resistance: 是否显示支撑阻力位
        fibonacci: 是否显示斐波那契回撤位
    """
    plt.figure(figsize=(14, 10))
    
    # 主图表 - 价格和移动平均线
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df.index, df['close'], label='Close')
    
    # 绘制移动平均线
    for col in df.columns:
        if col.startswith('ma_'):
            ax1.plot(df.index, df[col], label=col.upper(), alpha=0.7)
    
    # 绘制布林带
    if 'bb_upper' in df.columns and 'bb_middle' in df.columns and 'bb_lower' in df.columns:
        ax1.plot(df.index, df['bb_upper'], 'k--', alpha=0.3, label='Bollinger Upper')
        ax1.plot(df.index, df['bb_middle'], 'k-', alpha=0.3, label='Bollinger Middle')
        ax1.plot(df.index, df['bb_lower'], 'k--', alpha=0.3, label='Bollinger Lower')
    
    # 添加支撑位和阻力位
    if support_resistance:
        support_levels, resistance_levels = find_support_resistance(df)
        for level in support_levels:
            ax1.axhline(y=level, color='g', linestyle='--', alpha=0.5)
        for level in resistance_levels:
            ax1.axhline(y=level, color='r', linestyle='--', alpha=0.5)
    
    # 添加斐波那契回撤位
    if fibonacci and len(df) > 1:
        max_price = df['high'].max()
        min_price = df['low'].min()
        diff = max_price - min_price
        
        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        colors = ['black', 'blue', 'green', 'orange', 'green', 'blue', 'black']
        alphas = [1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0]
        
        for level, color, alpha in zip(levels, colors, alphas):
            price_level = max_price - diff * level
            ax1.axhline(y=price_level, color=color, linestyle='--', alpha=alpha * 0.5,
                       label=f'Fib {level}')
    
    ax1.set_title(title)
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left', fontsize='small')
    ax1.grid(True)
    
    # 成交量图表
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.bar(df.index, df['volume'], color='blue', alpha=0.5)
    
    # 绘制成交量移动平均线
    for col in df.columns:
        if col.startswith('volume_ma_'):
            ax2.plot(df.index, df[col], color='red', alpha=0.7, label=col)
    
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper left', fontsize='small')
    ax2.grid(True)
    
    # 技术指标图表
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    
    # 绘制RSI
    if 'rsi_14' in df.columns:
        ax3.plot(df.index, df['rsi_14'], label='RSI(14)', color='purple')
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
    
    # 绘制MACD
    if all(x in df.columns for x in ['macd', 'macd_signal', 'macd_hist']):
        ax4 = ax3.twinx()
        ax4.plot(df.index, df['macd'], label='MACD', color='blue', alpha=0.6)
        ax4.plot(df.index, df['macd_signal'], label='Signal', color='red', alpha=0.6)
        ax4.bar(df.index, df['macd_hist'], label='Histogram', color='green', alpha=0.3, width=1)
        ax4.set_ylabel('MACD')
        ax4.legend(loc='upper right', fontsize='small')
    
    ax3.legend(loc='upper left', fontsize='small')
    ax3.grid(True)
    
    # 调整布局
    plt.tight_layout()
    plt.show()


def generate_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成交易信号
    
    参数:
        df: 包含价格和指标数据的DataFrame
        
    返回:
        DataFrame: 添加了信号列的DataFrame
    """
    df = df.copy()
    
    # 初始化信号列
    df['signal'] = 0
    
    # 移动平均交叉信号
    if 'ma_20' in df.columns and 'ma_50' in df.columns:
        # 创建移动平均线交叉信号
        df['ma_cross'] = 0
        # 当短期均线上穿长期均线时，生成买入信号
        df.loc[(df['ma_20'] > df['ma_50']) & (df['ma_20'].shift(1) <= df['ma_50'].shift(1)), 'ma_cross'] = 1
        # 当短期均线下穿长期均线时，生成卖出信号
        df.loc[(df['ma_20'] < df['ma_50']) & (df['ma_20'].shift(1) >= df['ma_50'].shift(1)), 'ma_cross'] = -1
    
    # RSI信号
    if 'rsi_14' in df.columns:
        df['rsi_signal'] = 0
        # RSI超卖区域反弹买入信号
        df.loc[(df['rsi_14'] > 30) & (df['rsi_14'].shift(1) <= 30), 'rsi_signal'] = 1
        # RSI超买区域回落卖出信号
        df.loc[(df['rsi_14'] < 70) & (df['rsi_14'].shift(1) >= 70), 'rsi_signal'] = -1
    
    # MACD信号
    if all(x in df.columns for x in ['macd', 'macd_signal']):
        df['macd_cross'] = 0
        # MACD上穿信号线买入信号
        df.loc[(df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 'macd_cross'] = 1
        # MACD下穿信号线卖出信号
        df.loc[(df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 'macd_cross'] = -1
    
    # 布林带信号
    if all(x in df.columns for x in ['bb_upper', 'bb_middle', 'bb_lower']):
        df['bb_signal'] = 0
        # 价格突破布林带上轨卖出信号
        df.loc[(df['close'] > df['bb_upper']), 'bb_signal'] = -1
        # 价格突破布林带下轨买入信号
        df.loc[(df['close'] < df['bb_lower']), 'bb_signal'] = 1
    
    # 综合信号
    # 使用简单的加权方法，可以根据需要调整权重
    weights = {
        'ma_cross': 0.3,
        'rsi_signal': 0.2,
        'macd_cross': 0.3,
        'bb_signal': 0.2
    }
    
    for col, weight in weights.items():
        if col in df.columns:
            df['signal'] += df[col] * weight
    
    # 设置信号阈值
    df['buy_signal'] = df['signal'] > 0.2  # 买入信号阈值
    df['sell_signal'] = df['signal'] < -0.2  # 卖出信号阈值
    
    return df