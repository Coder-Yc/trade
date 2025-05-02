# data/downloaders/yahoo_downloader.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Union, Optional, Tuple

from config.settings import YAHOO_DATA_CACHE_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

class YahooDownloader:
    """Yahoo Finance数据下载器，提供从Yahoo Finance获取历史和实时市场数据的功能"""
    
    def __init__(self, cache_dir: str = YAHOO_DATA_CACHE_DIR):
        """
        初始化Yahoo Finance下载器
        
        Args:
            cache_dir: 数据缓存目录
        """
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        import os
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"创建Yahoo数据缓存目录: {self.cache_dir}")
    
    def download_historical_data(
        self, 
        symbols: List[str], 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        adjust_ohlc: bool = True,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        下载历史OHLCV数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期，格式'YYYY-MM-DD'或datetime对象
            end_date: 结束日期，格式'YYYY-MM-DD'或datetime对象
            period: 时间段 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: 时间间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            adjust_ohlc: 是否对OHLC价格进行调整
            use_cache: 是否使用缓存
            
        Returns:
            字典，键为股票代码，值为包含历史数据的DataFrame
        """
        if not symbols:
            logger.error("未提供股票代码")
            return {}
        
        logger.info(f"从Yahoo Finance下载{len(symbols)}个股票的历史数据")
        
        # 如果提供了period，则忽略start_date和end_date
        if period and (start_date or end_date):
            logger.warning("同时提供了period和日期范围，将使用period参数")
            start_date, end_date = None, None
        
        # 如果未提供日期范围和period，默认下载最近一年的数据
        if not period and not start_date and not end_date:
            logger.info("未提供日期范围或period，默认下载最近一年的数据")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
        
        result = {}
        for symbol in symbols:
            cache_file = None
            if use_cache:
                cache_file = self._get_cache_filename(symbol, start_date, end_date, period, interval)
                if os.path.exists(cache_file):
                    try:
                        df = pd.read_pickle(cache_file)
                        logger.info(f"从缓存加载{symbol}的数据")
                        result[symbol] = df
                        continue
                    except Exception as e:
                        logger.warning(f"无法从缓存加载{symbol}的数据: {e}")
            
            try:
                ticker = yf.Ticker(symbol)
                if period:
                    df = ticker.history(period=period, interval=interval, auto_adjust=adjust_ohlc)
                else:
                    df = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=adjust_ohlc)
                
                if df.empty:
                    logger.warning(f"{symbol}没有返回数据")
                    continue
                
                # 标准化列名
                df.columns = [col.lower() for col in df.columns]
                if "adj close" in df.columns:
                    df.rename(columns={"adj close": "adj_close"}, inplace=True)
                
                # 添加symbol列
                df["symbol"] = symbol
                
                result[symbol] = df
                logger.info(f"成功下载{symbol}的数据，获取了{len(df)}条记录")
                
                # 缓存数据
                if use_cache and cache_file:
                    df.to_pickle(cache_file)
                    logger.debug(f"缓存{symbol}的数据到{cache_file}")
                
            except Exception as e:
                logger.error(f"下载{symbol}的数据时出错: {e}")
        
        return result
    
    def download_multiple_tickers_data(
        self, 
        symbols: List[str], 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        adjust_ohlc: bool = True
    ) -> pd.DataFrame:
        """
        一次下载多个股票的数据并合并为一个DataFrame
        
        Args:
            参数同download_historical_data
            
        Returns:
            合并的DataFrame，包含所有股票的数据
        """
        data_dict = self.download_historical_data(
            symbols, start_date, end_date, period, interval, adjust_ohlc
        )
        
        if not data_dict:
            return pd.DataFrame()
        
        # 合并所有数据帧
        dfs = []
        for symbol, df in data_dict.items():
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs)
    
    def get_latest_price(self, symbols: List[str]) -> Dict[str, float]:
        """
        获取最新价格
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            字典，键为股票代码，值为最新价格
        """
        if not symbols:
            return {}
        
        result = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                if not data.empty:
                    result[symbol] = data['Close'].iloc[-1]
                    logger.debug(f"{symbol}的最新价格: {result[symbol]}")
                else:
                    logger.warning(f"无法获取{symbol}的最新价格")
            except Exception as e:
                logger.error(f"获取{symbol}的价格时出错: {e}")
        
        return result
    
    def get_intraday_data(
        self, 
        symbols: List[str], 
        interval: str = "1m", 
        days: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """
        获取日内数据
        
        Args:
            symbols: 股票代码列表
            interval: 时间间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h)
            days: 要获取的天数 (1-7)
            
        Returns:
            字典，键为股票代码，值为包含日内数据的DataFrame
        """
        if days > 7:
            logger.warning("Yahoo Finance API限制日内数据最多7天，已将days设置为7")
            days = 7
        
        period = f"{days}d"
        
        return self.download_historical_data(
            symbols=symbols,
            period=period,
            interval=interval,
            adjust_ohlc=True
        )
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """
        获取基本面数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含基本面数据的字典
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # 创建结果字典
            fundamentals = {
                "info": ticker.info,
                "financials": self._dataframe_to_dict(ticker.financials),
                "quarterly_financials": self._dataframe_to_dict(ticker.quarterly_financials),
                "balance_sheet": self._dataframe_to_dict(ticker.balance_sheet),
                "quarterly_balance_sheet": self._dataframe_to_dict(ticker.quarterly_balance_sheet),
                "cashflow": self._dataframe_to_dict(ticker.cashflow),
                "quarterly_cashflow": self._dataframe_to_dict(ticker.quarterly_cashflow),
                "earnings": self._dataframe_to_dict(ticker.earnings),
                "quarterly_earnings": self._dataframe_to_dict(ticker.quarterly_earnings)
            }
            
            logger.info(f"成功获取{symbol}的基本面数据")
            return fundamentals
            
        except Exception as e:
            logger.error(f"获取{symbol}的基本面数据时出错: {e}")
            return {}
    
    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict:
        """将DataFrame转换为嵌套字典"""
        if df is None or df.empty:
            return {}
        
        # 转换日期索引为字符串
        df.index = df.index.astype(str)
        
        # 转换为嵌套字典
        result = df.to_dict()
        
        # 处理NaN值
        for col, values in result.items():
            for date, value in values.items():
                if pd.isna(value):
                    result[col][date] = None
        
        return result
    
    def _get_cache_filename(
        self, 
        symbol: str, 
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
        period: Optional[str],
        interval: str
    ) -> str:
        """生成缓存文件名"""
        import os
        
        if period:
            filename = f"{symbol}_{period}_{interval}.pkl"
        else:
            start_str = start_date.strftime("%Y%m%d") if isinstance(start_date, datetime) else start_date
            end_str = end_date.strftime("%Y%m%d") if isinstance(end_date, datetime) else end_date
            filename = f"{symbol}_{start_str}_{end_str}_{interval}.pkl"
        
        return os.path.join(self.cache_dir, filename)