# 数据转换模块
# data/processors/transformer.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import logging

from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataTransformer:
    """数据转换类，提供各种数据转换和特征工程功能"""
    
    def __init__(self):
        """初始化数据转换器"""
        pass
    
    def add_technical_indicators(
        self, 
        data: pd.DataFrame, 
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        添加技术指标
        
        Args:
            data: 市场数据DataFrame
            indicators: 要添加的指标列表，如不提供则添加默认指标
            
        Returns:
            添加了技术指标的DataFrame
        """
        if data.empty:
            return data
            
        df = data.copy()
        
        # 默认指标集
        default_indicators = [
            'sma5', 'sma10', 'sma20', 'sma50', 'sma200',  # 简单移动平均线
            'ema12', 'ema26',                              # 指数移动平均线
            'rsi14',                                       # 相对强弱指数
            'macd', 'macd_signal', 'macd_hist',            # MACD
            'bb_upper', 'bb_middle', 'bb_lower',           # 布林带
            'atr14'                                         # 平均真实范围
        ]
        
        indicators_to_add = indicators or default_indicators
        logger.info(f"添加技术指标: {indicators_to_add}")
        
        # 确保存在收盘价列
        if 'close' not in df.columns:
            logger.error("数据中缺少close列，无法计算技术指标")
            return df
        
        # 按股票代码分组处理（如果有symbol列）
        if 'symbol' in df.columns:
            result_dfs = []
            for symbol, group in df.groupby('symbol'):
                # 确保按时间排序
                group = group.sort_index()
                group_with_indicators = self._add_indicators_to_group(group, indicators_to_add)
                result_dfs.append(group_with_indicators)
            
            if result_dfs:
                return pd.concat(result_dfs)
            else:
                return df
        else:
            # 确保按时间排序
            df = df.sort_index()
            return self._add_indicators_to_group(df, indicators_to_add)
    
    def _add_indicators_to_group(self, data: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """为单个组（单个股票）添加技术指标"""
        df = data.copy()
        
        # 添加各类技术指标
        for indicator in indicators:
            # 简单移动平均线 (SMA)
            if indicator.startswith('sma'):
                window = int(indicator[3:])
                df[indicator] = df['close'].rolling(window=window).mean()
            
            # 指数移动平均线 (EMA)
            elif indicator.startswith('ema'):
                window = int(indicator[3:])
                df[indicator] = df['close'].ewm(span=window, adjust=False).mean()
            
            # 相对强弱指数 (RSI)
            elif indicator.startswith('rsi'):
                window = int(indicator[3:])
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=window).mean()
                avg_loss = loss.rolling(window=window).mean()
                
                rs = avg_gain / avg_loss
                df[indicator] = 100 - (100 / (1 + rs))
            
            # MACD
            elif indicator == 'macd':
                # 如果尚未计算EMA12和EMA26，先计算它们
                if 'ema12' not in df.columns:
                    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
                if 'ema26' not in df.columns:
                    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
                
                df['macd'] = df['ema12'] - df['ema26']
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 布林带 (Bollinger Bands)
            elif indicator == 'bb_upper' or indicator == 'bb_middle' or indicator == 'bb_lower':
                if 'bb_middle' not in df.columns:
                    window = 20
                    # 中轨 - 20日简单移动平均线
                    df['bb_middle'] = df['close'].rolling(window=window).mean()
                    # 标准差
                    rolling_std = df['close'].rolling(window=window).std()
                    # 上轨 - 中轨 + 2倍标准差
                    df['bb_upper'] = df['bb_middle'] + (rolling_std * 2)
                    # 下轨 - 中轨 - 2倍标准差
                    df['bb_lower'] = df['bb_middle'] - (rolling_std * 2)
            
            # 平均真实范围 (ATR)
            elif indicator.startswith('atr'):
                window = int(indicator[3:])
                
                # 计算真实范围
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                
                df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df[indicator] = df['tr'].rolling(window=window).mean()
                
                # 移除临时列
                df.drop('tr', axis=1, inplace=True)
                
        return df
    
    def add_returns(
        self, 
        data: pd.DataFrame, 
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        添加收益率列
        
        Args:
            data: 市场数据DataFrame
            periods: 要计算的期间列表，如[1, 5, 20]表示计算1日、5日和20日收益率
            
        Returns:
            添加了收益率列的DataFrame
        """
        if data.empty or 'close' not in data.columns:
            return data
            
        df = data.copy()
        
        # 默认计算1日、5日和20日收益率
        periods = periods or [1, 5, 20]
        
        # 按股票代码分组处理（如果有symbol列）
        if 'symbol' in df.columns:
            result_dfs = []
            for symbol, group in df.groupby('symbol'):
                group = group.sort_index()
                for period in periods:
                    group[f'return_{period}d'] = group['close'].pct_change(period) * 100
                result_dfs.append(group)
            
            if result_dfs:
                return pd.concat(result_dfs)
            else:
                return df
        else:
            df = df.sort_index()
            for period in periods:
                df[f'return_{period}d'] = df['close'].pct_change(period) * 100
            
            return df
    
    def normalize_data(
        self, 
        data: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = 'z-score'
    ) -> pd.DataFrame:
        """
        标准化数据
        
        Args:
            data: 原始DataFrame
            columns: 要标准化的列，不提供则处理所有数值列
            method: 标准化方法，'z-score'或'min-max'
            
        Returns:
            标准化后的DataFrame
        """
        if data.empty:
            return data
            
        df = data.copy()
        
        # 如果未指定列，则处理所有数值列
        if not columns:
            columns = df.select_dtypes(include=['number']).columns.tolist()
            # 排除一些不应标准化的列
            for exclude_col in ['year', 'month', 'day', 'volume']:
                if exclude_col in columns:
                    columns.remove(exclude_col)
        
        # 按股票代码分组处理（如果有symbol列）
        if 'symbol' in df.columns:
            result_dfs = []
            for symbol, group in df.groupby('symbol'):
                for col in columns:
                    if col in group.columns:
                        if method == 'z-score':
                            # Z-score标准化: (x - mean) / std
                            mean = group[col].mean()
                            std = group[col].std()
                            if std != 0:
                                group[f'{col}_norm'] = (group[col] - mean) / std
                        elif method == 'min-max':
                            # Min-Max标准化: (x - min) / (max - min)
                            min_val = group[col].min()
                            max_val = group[col].max()
                            if max_val > min_val:
                                group[f'{col}_norm'] = (group[col] - min_val) / (max_val - min_val)
                
                result_dfs.append(group)
            
            if result_dfs:
                return pd.concat(result_dfs)
            else:
                return df
        else:
            for col in columns:
                if col in df.columns:
                    if method == 'z-score':
                        mean = df[col].mean()
                        std = df[col].std()
                        if std != 0:
                            df[f'{col}_norm'] = (df[col] - mean) / std
                    elif method == 'min-max':
                        min_val = df[col].min()
                        max_val = df[col].max()
                        if max_val > min_val:
                            df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
            
            return df
    
    def resample_data(
        self, 
        data: pd.DataFrame, 
        freq: str = 'W'
    ) -> pd.DataFrame:
        """
        重采样时间序列数据
        
        Args:
            data: 原始DataFrame
            freq: 重采样频率，如'D'(日),'W'(周),'M'(月),'Q'(季),'Y'(年)
            
        Returns:
            重采样后的DataFrame
        """
        if data.empty:
            return data
            
        # 确保索引是DatetimeIndex
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("数据索引不是日期类型，无法重采样")
            return df
        
        # 按股票代码分组处理（如果有symbol列）
        if 'symbol' in df.columns:
            result_dfs = []
            for symbol, group in df.groupby('symbol'):
                resampled = self._resample_group(group, freq)
                result_dfs.append(resampled)
            
            if result_dfs:
                return pd.concat(result_dfs)
            else:
                return df
        else:
            return self._resample_group(df, freq)
    
    def _resample_group(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """重采样单个组的数据"""
        # 选择数值列
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        # 准备重采样规则
        agg_dict = {}
        
        # OHLC价格列的规则
        if all(col in numeric_cols for col in ['open', 'high', 'low', 'close']):
            agg_dict['open'] = 'first'  # 第一个值
            agg_dict['high'] = 'max'    # 最大值
            agg_dict['low'] = 'min'     # 最小值
            agg_dict['close'] = 'last'  # 最后一个值
        
        # 成交量直接求和
        if 'volume' in numeric_cols:
            agg_dict['volume'] = 'sum'
        
        # 其他数值列默认使用平均值
        for col in numeric_cols:
            if col not in agg_dict:
                agg_dict[col] = 'mean'
        
        # 非数值列（如symbol）使用第一个值
        for col in data.columns:
            if col not in numeric_cols:
                agg_dict[col] = 'first'
        
        # 执行重采样
        resampled = data.resample(freq).agg(agg_dict)
        
        return resampled