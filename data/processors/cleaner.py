# 数据清洗模块
# data/processors/cleaner.py

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging

from utils.logger import setup_logger

logger = setup_logger(__name__)

class DataCleaner:
    """数据清洗类，提供市场数据清洗功能"""
    
    def __init__(self):
        """初始化数据清洗器"""
        pass
    
    def clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗市场数据
        
        Args:
            data: 原始市场数据DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        if data.empty:
            return data
            
        # 创建副本，避免修改原始数据
        df = data.copy()
        
        # 处理日期索引
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
        
        # 移除重复数据
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='last')]
        if len(df) < initial_len:
            logger.info(f"移除了{initial_len - len(df)}行重复数据")
        
        # 处理极端值
        for col in ['open', 'high', 'low', 'close', 'adj_close']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                if not outliers.empty:
                    logger.warning(f"检测到{len(outliers)}个{col}列的异常值，将使用上下限替换")
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
        
        # 处理缺失值
        if df.isna().any().any():
            logger.info("填充缺失值")
            for col in ['open', 'high', 'low', 'close', 'adj_close']:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # 对于成交量，用0填充缺失值
            if 'volume' in df.columns:
                df['volume'] = df['volume'].fillna(0)
        
        # 确保OHLC关系正确
        self._correct_ohlc_relationship(df)
        
        # 排序索引
        df = df.sort_index()
        
        return df
    
    def _correct_ohlc_relationship(self, df: pd.DataFrame) -> None:
        """
        确保开高低收价格之间的关系正确
        
        Args:
            df: 市场数据DataFrame
        """
        # 检查所需列是否存在
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return
        
        # 高价应该是最高的
        wrong_high = df[df['high'] < df[['open', 'close']].max(axis=1)]
        if not wrong_high.empty:
            logger.warning(f"修正{len(wrong_high)}行的high值")
            df.loc[wrong_high.index, 'high'] = df.loc[wrong_high.index, ['open', 'close', 'high']].max(axis=1)
        
        # 低价应该是最低的
        wrong_low = df[df['low'] > df[['open', 'close']].min(axis=1)]
        if not wrong_low.empty:
            logger.warning(f"修正{len(wrong_low)}行的low值")
            df.loc[wrong_low.index, 'low'] = df.loc[wrong_low.index, ['open', 'close', 'low']].min(axis=1)
    
    def remove_nan_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        移除包含NaN值的行
        
        Args:
            data: 原始DataFrame
            
        Returns:
            处理后的DataFrame
        """
        if data.empty:
            return data
            
        df = data.copy()
        initial_len = len(df)
        df = df.dropna()
        
        if len(df) < initial_len:
            logger.info(f"移除了{initial_len - len(df)}行包含NaN的数据")
        
        return df
    
    def remove_zero_volume_days(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        移除成交量为0的交易日
        
        Args:
            data: 原始DataFrame
            
        Returns:
            处理后的DataFrame
        """
        if data.empty or 'volume' not in data.columns:
            return data
            
        df = data.copy()
        initial_len = len(df)
        df = df[df['volume'] > 0]
        
        if len(df) < initial_len:
            logger.info(f"移除了{initial_len - len(df)}行成交量为0的数据")
        
        return df