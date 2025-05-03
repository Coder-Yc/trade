# data/storage/file_storage.py

import os
import pandas as pd
from datetime import datetime
import logging
from typing import Optional, Dict, Any

from utils.logger import setup_logger

logger = setup_logger(__name__)

class FileStorage:
    """文件存储类，负责保存和加载市场数据"""
    
    def __init__(self, base_dir: str = "data/storage/files"):
        """
        初始化文件存储
        
        Args:
            base_dir: 基础存储目录
        """
        self.base_dir = base_dir
        self._ensure_dir_exists(base_dir)
    
    def _ensure_dir_exists(self, dir_path: str):
        """确保目录存在"""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"创建目录: {dir_path}")
    
    def save_data(
        self, 
        data: pd.DataFrame, 
        format: str = "csv", 
        filename: Optional[str] = None,
        subdir: Optional[str] = None
    ) -> str:
        """
        保存数据到文件
        
        Args:
            data: 要保存的DataFrame
            format: 文件格式 ("csv", "parquet", "pickle")
            filename: 文件名，不指定则自动生成
            subdir: 子目录，用于分类存储
            
        Returns:
            保存的文件路径
        """
        if data.empty:
            logger.warning("尝试保存空数据")
            return ""
        
        # 处理子目录
        save_dir = self.base_dir
        if subdir:
            save_dir = os.path.join(self.base_dir, subdir)
            self._ensure_dir_exists(save_dir)
        
        # 生成文件名
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbols = []
            if 'symbol' in data.columns:
                symbols = data['symbol'].unique().tolist()
            
            symbols_str = "_".join(symbols) if symbols else "market_data"
            # 限制文件名长度
            if len(symbols_str) > 50:
                symbols_str = symbols_str[:50] + "..."
                
            filename = f"{symbols_str}_{timestamp}"
        
        # 添加扩展名
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"
        
        # 完整路径
        file_path = os.path.join(save_dir, filename)
        
        # 保存文件
        try:
            if format.lower() == "csv":
                data.to_csv(file_path)
            elif format.lower() == "parquet":
                data.to_parquet(file_path)
            elif format.lower() == "pickle" or format.lower() == "pkl":
                data.to_pickle(file_path)
            else:
                logger.error(f"不支持的文件格式: {format}")
                return ""
            
            logger.info(f"数据已保存到: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"保存数据时出错: {e}")
            return ""
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        从文件加载数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的DataFrame
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return pd.DataFrame()
        
        try:
            # 根据文件扩展名确定加载方法
            if file_path.endswith(".csv"):
                data = pd.read_csv(file_path)
                # 尝试将第一列转换为日期索引
                if 'date' in data.columns or 'Date' in data.columns:
                    date_col = 'date' if 'date' in data.columns else 'Date'
                    data[date_col] = pd.to_datetime(data[date_col])
                    data.set_index(date_col, inplace=True)
                elif data.columns[0].lower() in ['date', 'time', 'datetime']:
                    data[data.columns[0]] = pd.to_datetime(data[data.columns[0]])
                    data.set_index(data.columns[0], inplace=True)
                
            elif file_path.endswith(".parquet"):
                data = pd.read_parquet(file_path)
            elif file_path.endswith(".pkl") or file_path.endswith(".pickle"):
                data = pd.read_pickle(file_path)
            else:
                logger.error(f"不支持的文件格式: {file_path}")
                return pd.DataFrame()
            
            logger.info(f"从{file_path}加载了{len(data)}行数据")
            return data
            
        except Exception as e:
            logger.error(f"加载数据时出错: {e}")
            return pd.DataFrame()
    
    def list_files(self, subdir: Optional[str] = None, pattern: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        列出存储目录中的文件
        
        Args:
            subdir: 子目录
            pattern: 文件名模式，用于筛选
            
        Returns:
            文件信息字典，键为文件名，值为包含大小、修改时间等的字典
        """
        dir_path = self.base_dir
        if subdir:
            dir_path = os.path.join(self.base_dir, subdir)
        
        if not os.path.exists(dir_path):
            logger.warning(f"目录不存在: {dir_path}")
            return {}
        
        result = {}
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            
            # 只处理文件
            if not os.path.isfile(file_path):
                continue
                
            # 应用模式过滤
            if pattern and pattern not in filename:
                continue
            
            # 获取文件信息
            file_stats = os.stat(file_path)
            modified_time = datetime.fromtimestamp(file_stats.st_mtime)
            
            result[filename] = {
                "path": file_path,
                "size_bytes": file_stats.st_size,
                "size_kb": round(file_stats.st_size / 1024, 2),
                "modified": modified_time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return result