# data/storage/file_storage.py
"""
文件存储模块
负责将各类数据保存到文件系统并加载回来
"""

import os
import pandas as pd
import logging
from typing import Optional
from datetime import datetime

from utils.logger import setup_logger

# 设置文件存储根目录
DEFAULT_STORAGE_DIR = os.path.join("data", "storage", "files")
MARKET_DATA_DIR = os.path.join(DEFAULT_STORAGE_DIR, "market_data")
BACKTEST_DATA_DIR = os.path.join(DEFAULT_STORAGE_DIR, "backtest_results")
ANALYSIS_DATA_DIR = os.path.join(DEFAULT_STORAGE_DIR, "analysis")

logger = setup_logger(__name__)

class FileStorage:
    """文件存储类，提供保存和加载数据的功能"""
    
    def __init__(self, base_dir: str = DEFAULT_STORAGE_DIR):
        """
        初始化文件存储类
        
        Args:
            base_dir: 基础存储目录
        """
        self.base_dir = base_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保所有需要的目录存在"""
        for directory in [self.base_dir, MARKET_DATA_DIR, BACKTEST_DATA_DIR, ANALYSIS_DATA_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"创建目录: {directory}")
    
    def save_data(self, data: pd.DataFrame, format: str = "csv", filename: Optional[str] = None, 
                  subdir: str = None) -> str:
        """
        保存数据到文件
        
        Args:
            data: 要保存的DataFrame
            format: 文件格式 ("csv", "parquet", "pickle")
            filename: 文件名，不指定则自动生成
            subdir: 子目录，如 "market_data", "backtest_results"
            
        Returns:
            str: 保存的文件路径
        """
        if data.empty:
            logger.warning("尝试保存空数据")
            return ""
        
        # 确定保存目录
        if subdir:
            save_dir = os.path.join(self.base_dir, subdir)
        else:
            save_dir = MARKET_DATA_DIR  # 默认保存到市场数据目录
        
        # 确保目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 生成文件名如果未提供
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbols_str = "_".join(data["symbol"].unique()) if "symbol" in data.columns else "data"
            # 限制文件名长度
            if len(symbols_str) > 50:
                symbols_str = symbols_str[:50] + "..."
            filename = f"{symbols_str}_{timestamp}"
        
        # 确保有正确的扩展名
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"
        
        # 完整文件路径
        file_path = os.path.join(save_dir, filename)
        
        # 根据格式保存数据
        try:
            if format.lower() == "csv":
                data.to_csv(file_path, index=True)
            elif format.lower() == "parquet":
                data.to_parquet(file_path, index=True)
            elif format.lower() == "pickle" or format.lower() == "pkl":
                data.to_pickle(file_path)
            else:
                logger.error(f"不支持的文件格式: {format}")
                return ""
            
            logger.info(f"数据成功保存到: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"保存数据到文件时出错: {e}")
            return ""
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        从文件加载数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            pd.DataFrame: 加载的数据DataFrame
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return pd.DataFrame()
        
        try:
            # 根据文件扩展名决定加载方法
            if file_path.endswith(".csv"):
                data = pd.read_csv(file_path)
                # 检查是否有日期索引列
                date_cols = [col for col in data.columns if col.lower() in ["date", "datetime", "time"]]
                if date_cols:
                    data[date_cols[0]] = pd.to_datetime(data[date_cols[0]])
                    data.set_index(date_cols[0], inplace=True)
                
            elif file_path.endswith(".parquet"):
                data = pd.read_parquet(file_path)
                
            elif file_path.endswith(".pkl") or file_path.endswith(".pickle"):
                data = pd.read_pickle(file_path)
                
            else:
                logger.error(f"不支持的文件格式: {file_path}")
                return pd.DataFrame()
            
            logger.info(f"成功从 {file_path} 加载数据，形状: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"加载数据时出错: {e}")
            return pd.DataFrame()
    
    def find_data_files(self, subdir: str = None, file_pattern: str = None, 
                        format: str = None) -> list:
        """
        查找符合条件的数据文件
        
        Args:
            subdir: 子目录名
            file_pattern: 文件名模式
            format: 文件格式
            
        Returns:
            list: 符合条件的文件路径列表
        """
        search_dir = os.path.join(self.base_dir, subdir) if subdir else self.base_dir
        print(f"搜索目录: {search_dir}")
        
        if not os.path.exists(search_dir):
            logger.warning(f"目录不存在: {search_dir}")
            return []
        
        files = []
        for file in os.listdir(search_dir):
            file_path = os.path.join(search_dir, file)
            if os.path.isfile(file_path):
                # 检查格式
                if format and not file.endswith(f".{format}"):
                    continue
                # 检查文件名模式
                if file_pattern and file_pattern not in file:
                    continue
                files.append(file_path)
        
        return sorted(files)
    
    def exists(self, file_path: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 文件是否存在
        """
        return os.path.exists(file_path)
    
    def delete_file(self, file_path: str) -> bool:
        """
        删除文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否成功删除
        """
        if not os.path.exists(file_path):
            logger.warning(f"尝试删除不存在的文件: {file_path}")
            return False
        
        try:
            os.remove(file_path)
            logger.info(f"已删除文件: {file_path}")
            return True
        except Exception as e:
            logger.error(f"删除文件时出错: {e}")
            return False