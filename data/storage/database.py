# 数据库接口模块
# data/storage/database.py

import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

from utils.logger import setup_logger

logger = setup_logger(__name__)

class DatabaseManager:
    """数据库管理类，提供市场数据的存储和检索功能"""
    
    def __init__(self, db_path: str = "data/storage/market_data.db"):
        """
        初始化数据库管理器
        
        Args:
            db_path: SQLite数据库文件路径
        """
        self.db_path = db_path
        self._ensure_dir_exists(os.path.dirname(db_path))
        self._initialize_db()
    
    def _ensure_dir_exists(self, dir_path: str):
        """确保目录存在"""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"创建目录: {dir_path}")
    
    def _initialize_db(self):
        """初始化数据库，创建必要的表"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建市场数据表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                interval TEXT,
                UNIQUE(symbol, date, interval)
            )
            ''')
            
            # 创建索引以加速查询
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_date ON market_data (symbol, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON market_data (date)')
            
            # 创建元数据表，用于存储数据源、更新时间等信息
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data_meta (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                last_update TEXT,
                first_date TEXT,
                last_date TEXT,
                data_source TEXT,
                rows_count INTEGER,
                UNIQUE(symbol, interval)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("数据库初始化完成")
            
        except Exception as e:
            logger.error(f"初始化数据库时出错: {e}")
    
    def save_market_data(self, data: pd.DataFrame, table_name: str = "market_data") -> bool:
        """
        保存市场数据到数据库
        
        Args:
            data: 市场数据DataFrame
            table_name: 表名
            
        Returns:
            bool: 是否成功保存
        """
        if data.empty:
            logger.warning("尝试保存空数据")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 确保数据索引是日期类型
            df = data.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
            
            # 重置索引，将日期作为列
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.rename(columns={'index': 'date'}, inplace=True)
            
            # 确保date列是字符串格式
            if 'date' in df.columns:
                df['date'] = df['date'].astype(str)
            
            # 确保所需的列存在
            required_cols = ['symbol', 'date']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                logger.error(f"数据缺少必要的列: {missing}")
                return False
            
            # 如果指定了特定表，确保该表存在
            if table_name != "market_data":
                cursor = conn.cursor()
                cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    interval TEXT,
                    UNIQUE(symbol, date, interval)
                )
                ''')
                conn.commit()
            
            # 保存数据
            df.to_sql(table_name, conn, if_exists='append', index=False)
            
            # 更新元数据
            if 'interval' in df.columns:
                intervals = df['interval'].unique()
            else:
                intervals = ['1d']  # 默认日线数据
            
            for symbol in df['symbol'].unique():
                for interval in intervals:
                    symbol_data = df[(df['symbol'] == symbol)]
                    if 'interval' in df.columns:
                        symbol_data = symbol_data[symbol_data['interval'] == interval]
                    
                    if not symbol_data.empty:
                        first_date = min(symbol_data['date'])
                        last_date = max(symbol_data['date'])
                        rows_count = len(symbol_data)
                        
                        # 更新元数据表
                        cursor = conn.cursor()
                        cursor.execute('''
                        INSERT OR REPLACE INTO market_data_meta
                        (symbol, interval, last_update, first_date, last_date, data_source, rows_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            symbol, 
                            interval, 
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            first_date,
                            last_date,
                            'yahoo',  # 默认数据源
                            rows_count
                        ))
                        conn.commit()
            
            conn.close()
            logger.info(f"成功保存{len(df)}行数据到{table_name}表")
            return True
            
        except Exception as e:
            logger.error(f"保存数据到数据库时出错: {e}")
            return False
    
    def get_market_data(
        self, 
        symbols: List[str], 
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        table_name: str = "market_data"
    ) -> pd.DataFrame:
        """
        从数据库获取市场数据
        
        Args:
            symbols: 股票代码列表
            interval: 时间间隔
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            table_name: 表名
            
        Returns:
            市场数据DataFrame
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 构建查询
            query = f"SELECT * FROM {table_name} WHERE symbol IN ({', '.join(['?'] * len(symbols))})"
            params = symbols.copy()
            
            # 添加间隔条件
            if interval:
                query += " AND (interval = ? OR interval IS NULL)"
                params.append(interval)
            
            # 添加日期范围条件
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            # 按日期排序
            query += " ORDER BY symbol, date"
            
            # 执行查询
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                logger.warning(f"没有找到符合条件的数据")
                return pd.DataFrame()
            
            # 将日期列转换为日期时间类型并设置为索引
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            logger.info(f"从数据库获取了{len(df)}行数据")
            return df
            
        except Exception as e:
            logger.error(f"从数据库获取数据时出错: {e}")
            return pd.DataFrame()
    
    def get_available_symbols(self, table_name: str = "market_data_meta") -> List[Dict[str, Any]]:
        """
        获取数据库中可用的股票代码
        
        Args:
            table_name: 元数据表名
            
        Returns:
            可用股票代码的信息列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f'''
            SELECT symbol, interval, first_date, last_date, rows_count, last_update
            FROM {table_name}
            ORDER BY symbol, interval
            ''')
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'symbol': row[0],
                    'interval': row[1],
                    'first_date': row[2],
                    'last_date': row[3],
                    'rows_count': row[4],
                    'last_update': row[5]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"获取可用股票代码时出错: {e}")
            return []
    
    def delete_symbol_data(self, symbol: str, interval: Optional[str] = None) -> bool:
        """
        删除指定股票的数据
        
        Args:
            symbol: 股票代码
            interval: 时间间隔，不指定则删除所有间隔的数据
            
        Returns:
            bool: 是否成功删除
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 从市场数据表删除
            query = "DELETE FROM market_data WHERE symbol = ?"
            params = [symbol]
            
            if interval:
                query += " AND (interval = ? OR interval IS NULL)"
                params.append(interval)
            
            cursor.execute(query, params)
            
            # 从元数据表删除
            meta_query = "DELETE FROM market_data_meta WHERE symbol = ?"
            meta_params = [symbol]
            
            if interval:
                meta_query += " AND interval = ?"
                meta_params.append(interval)
            
            cursor.execute(meta_query, meta_params)
            
            conn.commit()
            conn.close()
            
            logger.info(f"已删除{symbol} {interval if interval else '所有间隔'}的数据")
            return True
            
        except Exception as e:
            logger.error(f"删除数据时出错: {e}")
            return False