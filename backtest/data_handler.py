# backtest/data_handler.py
"""
回测数据处理器 - 负责加载和准备回测所需的数据
将各种格式的数据转换为Backtrader可用的格式
"""
import os
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Any, Tuple

from data.market_data import MarketDataManager
from utils.logger import setup_logger

logger = setup_logger('DataHandler')

class PandasDataHandler:
    """
    Pandas数据处理器
    负责将Pandas DataFrame转换为Backtrader数据源
    """
    
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        timeframe: bt.TimeFrame = bt.TimeFrame.Days,
        fromdate: Optional[datetime] = None,
        todate: Optional[datetime] = None,
        datetime_col: str = 'date',
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume',
        openinterest_col: Optional[str] = None,
        adjust_prices: bool = True
    ):
        """
        初始化Pandas数据处理器
        
        参数:
            df: 包含市场数据的Pandas DataFrame
            symbol: 交易品种代码
            timeframe: Backtrader时间框架
            fromdate: 开始日期
            todate: 结束日期
            datetime_col: 日期时间列名
            open_col: 开盘价列名
            high_col: 最高价列名
            low_col: 最低价列名
            close_col: 收盘价列名
            volume_col: 成交量列名
            openinterest_col: 未平仓合约列名 (可选)
            adjust_prices: 是否使用复权价格
        """
        self.df = df
        self.symbol = symbol
        self.timeframe = timeframe
        self.fromdate = fromdate
        self.todate = todate
        
        # 列名映射
        self.datetime_col = datetime_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.openinterest_col = openinterest_col
        
        self.adjust_prices = adjust_prices
        
        # 如果提供了DataFrame，预处理数据
        if df is not None:
            self._preprocess_data()
    
    def load_from_csv(
        self, 
        csv_path: str, 
        date_format: str = '%Y-%m-%d',
        sep: str = ','
    ) -> 'PandasDataHandler':
        """
        从CSV文件加载数据
        
        参数:
            csv_path: CSV文件路径
            date_format: 日期格式
            sep: 分隔符
            
        返回:
            self，便于链式调用
        """
        try:
            logger.info(f"从CSV文件加载数据: {csv_path}")
            df = pd.read_csv(csv_path, sep=sep, parse_dates=[self.datetime_col], date_format=date_format)
            self.df = df
            self._preprocess_data()
            return self
        except Exception as e:
            logger.error(f"从CSV加载数据时出错: {e}")
            raise
    
    def load_from_market_data_manager(
        self,
        mdm: MarketDataManager,
        symbols: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d",
        provider: Optional[str] = None,
        include_indicators: bool = False
    ) -> 'PandasDataHandler':
        """
        从MarketDataManager加载数据
        
        参数:
            mdm: MarketDataManager实例
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔
            provider: 数据提供商
            include_indicators: 是否包含技术指标
            
        返回:
            self，便于链式调用
        """
        try:
            logger.info(f"从MarketDataManager加载数据: {symbols}")
            
            # 从MarketDataManager获取数据
            df = mdm.get_market_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                provider=provider,
                include_indicators=include_indicators
            )
            
            # 如果是多个股票，选择第一个
            if len(symbols) == 1:
                self.symbol = symbols[0]
            
            self.df = df
            self._preprocess_data()
            return self
        except Exception as e:
            logger.error(f"从MarketDataManager加载数据时出错: {e}")
            raise
    
    def _preprocess_data(self) -> None:
        """预处理数据"""
        if self.df is None or self.df.empty:
            logger.warning("数据为空")
            return
        
        # 确保索引是日期时间类型
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if self.datetime_col in self.df.columns:
                logger.info(f"将{self.datetime_col}列设置为索引")
                self.df.set_index(self.datetime_col, inplace=True)
            else:
                logger.warning(f"未找到日期时间列: {self.datetime_col}")
        
        # 确保索引已排序
        self.df = self.df.sort_index()
        
        # 如果未指定symbol但DataFrame中有symbol列，使用第一个值
        if self.symbol is None and 'symbol' in self.df.columns:
            self.symbol = self.df['symbol'].iloc[0]
            logger.info(f"从数据中提取股票代码: {self.symbol}")
        
        # 根据日期范围筛选数据
        if self.fromdate or self.todate:
            mask = pd.Series(True, index=self.df.index)
            if self.fromdate:
                mask = mask & (self.df.index >= self.fromdate)
            if self.todate:
                mask = mask & (self.df.index <= self.todate)
            self.df = self.df[mask]
            logger.info(f"根据日期范围筛选数据: {self.fromdate} 到 {self.todate}")
        
        # 处理缺失值
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"数据中存在 {missing_count} 个缺失值，将进行填充")
            
            # 对于价格，使用前向填充
            for col in [self.open_col, self.high_col, self.low_col, self.close_col]:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna(method='ffill')
            
            # 对于成交量，用0填充
            if self.volume_col in self.df.columns:
                self.df[self.volume_col] = self.df[self.volume_col].fillna(0)
            
            # 如果有未平仓合约列
            if self.openinterest_col and self.openinterest_col in self.df.columns:
                self.df[self.openinterest_col] = self.df[self.openinterest_col].fillna(0)
        
        # 如果是多股票数据，只保留指定的symbol
        if 'symbol' in self.df.columns and self.symbol is not None:
            self.df = self.df[self.df['symbol'] == self.symbol]
            logger.info(f"筛选出股票 {self.symbol} 的数据")
        
        # 使用复权价格（如果有adj_close列且需要复权）
        if self.adjust_prices and 'adj_close' in self.df.columns:
            logger.info("使用复权价格")
            adj_ratio = self.df['adj_close'] / self.df[self.close_col]
            self.df[self.open_col] = self.df[self.open_col] * adj_ratio
            self.df[self.high_col] = self.df[self.high_col] * adj_ratio
            self.df[self.low_col] = self.df[self.low_col] * adj_ratio
            self.df[self.close_col] = self.df['adj_close']
        
        logger.info(f"数据预处理完成: {len(self.df)} 行")
    
    def get_backtrader_data(self) -> bt.feeds.PandasData:
        """
        获取Backtrader可用的数据对象
        
        返回:
            bt.feeds.PandasData: Backtrader数据源
        """
        if self.df is None or self.df.empty:
            logger.error("无法创建Backtrader数据对象：数据为空")
            raise ValueError("数据为空")
        
        # 创建Backtrader数据类
        class CustomPandasData(bt.feeds.PandasData):
            """自定义Pandas数据源，适应不同列名"""
            params = (
                ('datetime', None),  # 使用索引作为日期时间
                ('open', None),
                ('high', None),
                ('low', None),
                ('close', None),
                ('volume', None),
                ('openinterest', None),
            )
        
        # 设置列名参数
        params = {}
        params['open'] = self.open_col if self.open_col in self.df.columns else None
        params['high'] = self.high_col if self.high_col in self.df.columns else None
        params['low'] = self.low_col if self.low_col in self.df.columns else None
        params['close'] = self.close_col if self.close_col in self.df.columns else None
        params['volume'] = self.volume_col if self.volume_col in self.df.columns else None
        params['openinterest'] = self.openinterest_col if self.openinterest_col and self.openinterest_col in self.df.columns else None
        
        # 创建Backtrader数据源
        data = CustomPandasData(
            dataname=self.df,
            fromdate=self.fromdate,
            todate=self.todate,
            timeframe=self.timeframe,
            **params
        )
        
        if self.symbol:
            data._name = self.symbol
        
        return data


class CSVDataHandler:
    """CSV文件数据处理器"""
    
    def __init__(
        self,
        csv_path: str,
        symbol: Optional[str] = None,
        timeframe: bt.TimeFrame = bt.TimeFrame.Days,
        fromdate: Optional[datetime] = None,
        todate: Optional[datetime] = None,
        date_format: str = '%Y-%m-%d',
        dtformat: bool = True,
        datetime_col: int = 0,
        open_col: int = 1,
        high_col: int = 2,
        low_col: int = 3,
        close_col: int = 4,
        volume_col: int = 5,
        openinterest_col: int = -1,
        adjust_prices: bool = False,
        reverse: bool = False
    ):
        """
        初始化CSV数据处理器
        
        参数:
            csv_path: CSV文件路径
            symbol: 交易品种代码
            timeframe: Backtrader时间框架
            fromdate: 开始日期
            todate: 结束日期
            date_format: 日期格式
            dtformat: 是否使用日期格式化
            datetime_col: 日期时间列索引
            open_col: 开盘价列索引
            high_col: 最高价列索引
            low_col: 最低价列索引
            close_col: 收盘价列索引
            volume_col: 成交量列索引
            openinterest_col: 未平仓合约列索引 (-1表示不使用)
            adjust_prices: 是否使用复权价格
            reverse: 是否反转数据顺序
        """
        self.csv_path = csv_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.fromdate = fromdate
        self.todate = todate
        self.date_format = date_format
        self.dtformat = dtformat
        self.datetime_col = datetime_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.openinterest_col = openinterest_col if openinterest_col >= 0 else None
        self.adjust_prices = adjust_prices
        self.reverse = reverse
        
        # 验证文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        
        logger.info(f"初始化CSV数据处理器: {csv_path}")
    
    def get_backtrader_data(self) -> bt.feeds.GenericCSVData:
        """
        获取Backtrader可用的数据对象
        
        返回:
            bt.feeds.GenericCSVData: Backtrader数据源
        """
        data = bt.feeds.GenericCSVData(
            dataname=self.csv_path,
            fromdate=self.fromdate,
            todate=self.todate,
            timeframe=self.timeframe,
            dtformat=self.date_format if self.dtformat else None,
            datetime=self.datetime_col,
            open=self.open_col,
            high=self.high_col,
            low=self.low_col,
            close=self.close_col,
            volume=self.volume_col,
            openinterest=self.openinterest_col,
            reverse=self.reverse
        )
        
        if self.symbol:
            data._name = self.symbol
        
        return data


class YahooDataHandler:
    """Yahoo数据处理器，使用yfinance下载数据"""
    
    def __init__(
        self,
        symbol: str,
        fromdate: Optional[datetime] = None,
        todate: Optional[datetime] = None,
        timeframe: bt.TimeFrame = bt.TimeFrame.Days,
        adjust_prices: bool = True
    ):
        """
        初始化Yahoo数据处理器
        
        参数:
            symbol: 交易品种代码
            fromdate: 开始日期
            todate: 结束日期
            timeframe: Backtrader时间框架
            adjust_prices: 是否使用复权价格
        """
        self.symbol = symbol
        self.fromdate = fromdate
        self.todate = todate
        self.timeframe = timeframe
        self.adjust_prices = adjust_prices
        
        logger.info(f"初始化Yahoo数据处理器: {symbol}")
    
    def get_backtrader_data(self) -> bt.feeds.PandasData:
        """
        获取Backtrader可用的数据对象
        
        返回:
            bt.feeds.PandasData: Backtrader数据源
        """
        try:
            import yfinance as yf
            
            # 下载数据
            logger.info(f"正在从Yahoo Finance下载 {self.symbol} 的数据...")
            
            data = yf.download(
                self.symbol,
                start=self.fromdate,
                end=self.todate,
                auto_adjust=self.adjust_prices
            )
            
            if data.empty:
                raise ValueError(f"无法获取 {self.symbol} 的数据")
            
            # 创建Pandas数据处理器
            pandas_handler = PandasDataHandler(
                df=data,
                symbol=self.symbol,
                timeframe=self.timeframe,
                fromdate=self.fromdate,
                todate=self.todate,
                datetime_col='Date',
                open_col='Open',
                high_col='High',
                low_col='Low',
                close_col='Close',
                volume_col='Volume',
                adjust_prices=False  # 已经在yfinance中调整了
            )
            
            # 获取Backtrader数据对象
            return pandas_handler.get_backtrader_data()
            
        except Exception as e:
            logger.error(f"从Yahoo Finance下载数据时出错: {e}")
            raise