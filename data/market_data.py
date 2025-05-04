# data/market_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import List, Dict, Union, Optional, Tuple, Any

from data.downloaders.ibkr_downloader import IBKRDownloader
from data.downloaders.yahoo_downloader import YahooDownloader
from data.processors.cleaner import DataCleaner
from data.processors.transformer import DataTransformer
from data.storage.database import DatabaseManager
from data.storage.file_storage import FileStorage
from config.settings import MARKET_DATA_CACHE_DIR, DEFAULT_DATA_PROVIDER
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MarketDataManager:
    """市场数据管理类，集中管理各种数据源的数据获取、处理和存储"""
    
    def __init__(self, default_provider: str = DEFAULT_DATA_PROVIDER):
        self.default_provider = default_provider
        
        # 初始化各数据下载器
        self.yahoo_downloader = YahooDownloader()
        self.ibkr_downloader = IBKRDownloader()
        
        # 初始化数据处理器
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        
        # 初始化存储
        self.db_manager = DatabaseManager()
        self.file_storage = FileStorage()
        
        logger.info(f"市场数据管理器已初始化，默认数据提供商: {default_provider}")
    
    def get_market_data(
        self, 
        symbols: List[str], 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        provider: Optional[str] = None,
        adjust_prices: bool = True,
        apply_cleaning: bool = True,
        save_to_db: bool = False,
        cache_results: bool = True,
        include_indicators: bool = False,
        indicators_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取市场数据的主方法，支持多种数据源、清洗和处理
        
        Args:
            symbols: 股票/期货/加密货币代码列表
            start_date: 开始日期，如"2023-01-01"或datetime对象
            end_date: 结束日期
            period: 时间段 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: 时间间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            provider: 数据提供商，不指定则使用默认提供商
            adjust_prices: 是否调整价格
            apply_cleaning: 是否应用数据清洗
            save_to_db: 是否保存到数据库
            cache_results: 是否缓存结果
            include_indicators: 是否计算技术指标
            indicators_list: 要计算的技术指标列表
            
        Returns:
            包含请求数据的DataFrame
        """
        # 使用指定的提供商或默认提供商
        provider = provider or self.default_provider
        
        # 检查是否有缓存可用
        if cache_results:
            cached_data = self._check_cache(symbols, start_date, end_date, period, interval, provider)
            if cached_data is not None:
                logger.info("从缓存获取数据")
                return cached_data
        
        # 根据提供商获取数据
        data = self._fetch_from_provider(
            provider, symbols, start_date, end_date, period, interval, adjust_prices
        )
        
        # 检查数据有效性
        if data.empty:
            logger.warning(f"未能从{provider}获取数据")
            return pd.DataFrame()
        
        # 应用数据清洗
        if apply_cleaning:
            data = self.cleaner.clean_market_data(data)
        
        # 计算技术指标
        if include_indicators:
            data = self._add_technical_indicators(data, indicators_list)
        
        # 保存到数据库
        if save_to_db:
            self._save_to_database(data, symbols, interval)
        
        # 缓存结果
        if cache_results:
            self._cache_data(data, symbols, start_date, end_date, period, interval, provider)
        
        return data
    
    def get_yahoo_data(
        self, 
        symbols: List[str], 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        adjust_prices: bool = True
    ) -> pd.DataFrame:
        """
        从Yahoo Finance获取市场数据
        
        Args:
            参数与get_market_data相同
            
        Returns:
            包含Yahoo数据的DataFrame
        """
        logger.info(f"从Yahoo Finance获取{len(symbols)}个股票的数据")
        
        # 处理日期
        if not period and not start_date:
            end_date = end_date or datetime.now()
            start_date = start_date or (end_date - timedelta(days=30))
        
        # 使用Yahoo下载器
        if period:
            data_dict = self.yahoo_downloader.download_historical_data(
                symbols=symbols,
                period=period,
                interval=interval,
                adjust_ohlc=adjust_prices
            )
        else:
            data_dict = self.yahoo_downloader.download_historical_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                adjust_ohlc=adjust_prices
            )
        
        # 合并为单个DataFrame
        dfs = []
        for symbol, df in data_dict.items():
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        combined_data = pd.concat(dfs)
        
        # 标准化数据格式
        return self._standardize_data_format(combined_data)
    
    def get_ibkr_data(
        self, 
        symbols: List[str], 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d",
        adjust_prices: bool = True
    ) -> pd.DataFrame:

        logger.info(f"从IBKR获取{len(symbols)}个股票的数据")
        
        # 处理日期
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))
        
        # 使用IBKR下载器
        data = self.ibkr_downloader.download_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            bar_size=self._convert_interval_to_ibkr_format(interval),
            adjust_prices=adjust_prices
        )
        
        # 标准化数据格式
        return self._standardize_data_format(data)
    
    def get_latest_prices(
        self, 
        symbols: List[str], 
        provider: Optional[str] = None
    ) -> Dict[str, float]:
        """
        获取最新价格
        
        Args:
            symbols: 股票代码列表
            provider: 数据提供商
            
        Returns:
            字典，键为股票代码，值为最新价格
        """
        provider = provider or self.default_provider
        
        if provider == "yahoo":
            return self.yahoo_downloader.get_latest_price(symbols)
        elif provider == "ibkr":
            return self.ibkr_downloader.get_latest_prices(symbols)
        else:
            logger.error(f"不支持的数据提供商: {provider}")
            return {}
    
    def get_fundamental_data(
        self, 
        symbols: List[str], 
        data_type: str = "all", 
        provider: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        获取基本面数据
        
        Args:
            symbols: 股票代码列表
            data_type: 数据类型 ("financials", "balance_sheet", "cashflow", "earnings", "all")
            provider: 数据提供商
            
        Returns:
            嵌套字典，包含各股票的基本面数据
        """
        provider = provider or self.default_provider
        
        result = {}
        
        if provider == "yahoo":
            for symbol in symbols:
                data = self.yahoo_downloader.get_fundamental_data(symbol)
                
                if data_type != "all":
                    # 只返回请求的数据类型
                    data = {data_type: data.get(data_type, {})} if data_type in data else {}
                
                result[symbol] = data
        
        elif provider == "ibkr":
            # 根据IBKR API适配基本面数据获取
            logger.info("从IBKR获取基本面数据")
            # 实现IBKR基本面数据获取...
        
        elif provider == "external":
            # 从外部源获取基本面数据
            logger.info("从外部源获取基本面数据")
            # 实现外部源基本面数据获取...
        
        else:
            logger.error(f"不支持的数据提供商: {provider}")
        
        return result
    
    def search_symbols(self, query: str, limit: int = 10, provider: Optional[str] = None) -> List[Dict]:
        """
        搜索股票代码
        
        Args:
            query: 搜索查询
            limit: 结果数量限制
            provider: 数据提供商
            
        Returns:
            符合搜索条件的股票列表
        """
        provider = provider or self.default_provider
        
        if provider == "yahoo":
            # 使用Yahoo Finance API搜索股票
            import yfinance as yf
            try:
                data = yf.Ticker(query).info
                if data and "shortName" in data:
                    return [{"symbol": query, "name": data["shortName"]}]
            except Exception as e:
                logger.error(f"Yahoo搜索股票时出错: {e}")
                return []
        
        elif provider == "ibkr":
            # 使用IBKR API搜索股票
            logger.info(f"使用IBKR搜索股票: {query}")
            # 实现IBKR股票搜索...
            return []
        
        elif provider == "external":
            # 使用外部源搜索股票
            logger.info(f"使用外部源搜索股票: {query}")
            # 实现外部源股票搜索...
            return []
        
        else:
            logger.error(f"不支持的数据提供商: {provider}")
            return []
    
    def get_market_calendar(
        self, 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        exchange: str = "NYSE",
        provider: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取市场日历（交易日、假期等）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所
            provider: 数据提供商
            
        Returns:
            包含市场日历信息的DataFrame
        """
        provider = provider or self.default_provider
        
        # 处理日期
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))
        
        if provider == "yahoo":
            # Yahoo没有直接的市场日历API，可以使用替代方案
            try:
                # 获取主要指数的日期数据作为交易日历
                index_symbol = {"NYSE": "^GSPC", "NASDAQ": "^IXIC", "SSE": "000001.SS"}.get(exchange, "^GSPC")
                
                calendar_data = self.yahoo_downloader.download_historical_data(
                    symbols=[index_symbol],
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d"
                )
                
                if index_symbol in calendar_data and not calendar_data[index_symbol].empty:
                    dates = calendar_data[index_symbol].index
                    calendar = pd.DataFrame({"date": dates, "is_trading_day": True})
                    return calendar
                else:
                    logger.warning(f"无法通过Yahoo获取{exchange}的市场日历")
                    return pd.DataFrame()
                
            except Exception as e:
                logger.error(f"获取市场日历时出错: {e}")
                return pd.DataFrame()
                
        elif provider == "ibkr":
            # 使用IBKR API获取市场日历
            logger.info(f"使用IBKR获取{exchange}的市场日历")
            # 实现IBKR市场日历获取...
            return pd.DataFrame()
        
        elif provider == "external":
            # 使用外部源获取市场日历
            logger.info(f"使用外部源获取{exchange}的市场日历")
            # 实现外部源市场日历获取...
            return pd.DataFrame()
        
        else:
            logger.error(f"不支持的数据提供商: {provider}")
            return pd.DataFrame()
    
    def save_data(
        self, 
        data: pd.DataFrame, 
        format: str = "csv", 
        filename: Optional[str] = None
    ) -> str:
        """
        保存数据到文件
        
        Args:
            data: 要保存的DataFrame
            format: 文件格式 ("csv", "parquet", "pickle")
            filename: 文件名，不指定则自动生成
            
        Returns:
            保存的文件路径
        """
        if data.empty:
            logger.warning("尝试保存空数据")
            return ""
        
        # 生成默认文件名
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbols_str = "_".join(data["symbol"].unique()) if "symbol" in data.columns else "market_data"
            symbols_str = symbols_str[:50] + "..." if len(symbols_str) > 50 else symbols_str  # 限制长度
            filename = f"{symbols_str}_{timestamp}"
        
        return self.file_storage.save_data(data, format=format, filename=filename)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        从文件加载数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的DataFrame
        """
        return self.file_storage.load_data(file_path)
    
    def _fetch_from_provider(
        self,
        provider: str,
        symbols: List[str],
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
        period: Optional[str],
        interval: str,
        adjust_prices: bool
    ) -> pd.DataFrame:
        """根据提供商获取数据"""
        if provider == "yahoo":
            return self.get_yahoo_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                period=period,
                interval=interval,
                adjust_prices=adjust_prices
            )
        elif provider == "ibkr":
            return self.get_ibkr_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                adjust_prices=adjust_prices
            )
        else:
            logger.error(f"不支持的数据提供商: {provider}")
            return pd.DataFrame()
    
    def _standardize_data_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化不同来源的数据格式"""
        if data.empty:
            return data
        
        # 确保列名统一
        column_mapping = {
            # Yahoo Finance
            "date": "date",
            "open": "open", 
            "high": "high", 
            "low": "low", 
            "close": "close",
            "adj close": "adj_close",
            "adj_close": "adj_close",
            "volume": "volume",
            "symbol": "symbol",
            
            # IBKR
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "AdjClose": "adj_close",
            "Volume": "volume",
            "Symbol": "symbol",
            
            # 其他可能的列名映射...
        }
        
        # 转换列名
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                data.rename(columns={old_col: new_col}, inplace=True)
        
        # 确保索引是日期类型
        if "date" in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            data.set_index("date", inplace=True)
            
        # 确保日期索引
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # 确保必要的列存在
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                logger.warning(f"数据中缺少{col}列")
                data[col] = np.nan
        
        # 确保symbol列存在
        if "symbol" not in data.columns:
            logger.warning("数据中缺少symbol列，尝试从索引或其他列推断")
            # 尝试从多级索引获取symbol
            if isinstance(data.index, pd.MultiIndex) and len(data.index.names) > 1:
                for idx_name in data.index.names:
                    if idx_name.lower() in ["symbol", "ticker", "asset", "code"]:
                        data["symbol"] = data.index.get_level_values(idx_name)
                        break
        
        # 转换数据类型
        for col in ["open", "high", "low", "close", "adj_close"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        
        if "volume" in data.columns:
            data["volume"] = pd.to_numeric(data["volume"], errors="coerce").fillna(0).astype(int)
        
        return data
    
    def _add_technical_indicators(
        self, 
        data: pd.DataFrame, 
        indicators_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """添加技术指标"""
        if data.empty:
            return data
        
        # 默认指标列表
        default_indicators = ["sma5", "sma20", "sma50", "sma200", "rsi14", "macd"]
        indicators_to_calculate = indicators_list or default_indicators
        
        # 按symbol分组计算指标
        symbols = data["symbol"].unique() if "symbol" in data.columns else [None]
        result_dfs = []
        
        for symbol in symbols:
            # 获取单个股票的数据
            if symbol is not None:
                symbol_data = data[data["symbol"] == symbol].copy()
            else:
                symbol_data = data.copy()
            
            if symbol_data.empty:
                continue
            
            # 确保索引是日期类型并且有序
            symbol_data = symbol_data.sort_index()
            
            # 计算各种技术指标
            for indicator in indicators_to_calculate:
                if indicator.startswith("sma"):
                    # 简单移动平均线
                    period = int(indicator[3:])
                    symbol_data[indicator] = symbol_data["close"].rolling(window=period).mean()
                
                elif indicator.startswith("ema"):
                    # 指数移动平均线
                    period = int(indicator[3:])
                    symbol_data[indicator] = symbol_data["close"].ewm(span=period, adjust=False).mean()
                
                elif indicator.startswith("rsi"):
                    # 相对强弱指标
                    period = int(indicator[3:])
                    delta = symbol_data["close"].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=period).mean()
                    avg_loss = loss.rolling(window=period).mean()
                    rs = avg_gain / avg_loss
                    symbol_data[indicator] = 100 - (100 / (1 + rs))
                
                elif indicator == "macd":
                    # MACD
                    ema12 = symbol_data["close"].ewm(span=12, adjust=False).mean()
                    ema26 = symbol_data["close"].ewm(span=26, adjust=False).mean()
                    symbol_data["macd_line"] = ema12 - ema26
                    symbol_data["macd_signal"] = symbol_data["macd_line"].ewm(span=9, adjust=False).mean()
                    symbol_data["macd_histogram"] = symbol_data["macd_line"] - symbol_data["macd_signal"]
                
                elif indicator == "bollinger":
                    # 布林带
                    period = 20
                    std_dev = 2
                    symbol_data["bollinger_middle"] = symbol_data["close"].rolling(window=period).mean()
                    symbol_data["bollinger_std"] = symbol_data["close"].rolling(window=period).std()
                    symbol_data["bollinger_upper"] = symbol_data["bollinger_middle"] + (symbol_data["bollinger_std"] * std_dev)
                    symbol_data["bollinger_lower"] = symbol_data["bollinger_middle"] - (symbol_data["bollinger_std"] * std_dev)
                
                elif indicator == "atr":
                    # 平均真实范围
                    period = 14
                    high_low = symbol_data["high"] - symbol_data["low"]
                    high_close = abs(symbol_data["high"] - symbol_data["close"].shift())
                    low_close = abs(symbol_data["low"] - symbol_data["close"].shift())
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    symbol_data["atr"] = true_range.rolling(window=period).mean()
                
                elif indicator == "obv":
                    # 能量潮
                    obv = (np.sign(symbol_data["close"].diff()) * symbol_data["volume"]).fillna(0).cumsum()
                    symbol_data["obv"] = obv
            
            result_dfs.append(symbol_data)
        
        # 合并结果
        if result_dfs:
            return pd.concat(result_dfs)
        else:
            return data
    
    def _check_cache(
        self,
        symbols: List[str],
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
        period: Optional[str],
        interval: str,
        provider: str
    ) -> Optional[pd.DataFrame]:
        """检查是否有可用的缓存数据"""
        cache_key = self._generate_cache_key(symbols, start_date, end_date, period, interval, provider)
        cache_file = os.path.join(MARKET_DATA_CACHE_DIR, f"{cache_key}.parquet")
        
        if os.path.exists(cache_file):
            print(f"缓存文件存在: {cache_file}")
            try:
                cached_data = pd.read_parquet(cache_file)
                logger.info(f"从缓存加载数据: {cache_file}")
                return cached_data
            except Exception as e:
                logger.warning(f"读取缓存文件出错: {e}")
        
        return None
    
    def _cache_data(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
        period: Optional[str],
        interval: str,
        provider: str
    ) -> None:
        """缓存数据到文件"""
        if data.empty:
            return
        
        # 确保缓存目录存在
        os.makedirs(MARKET_DATA_CACHE_DIR, exist_ok=True)
        
        cache_key = self._generate_cache_key(symbols, start_date, end_date, period, interval, provider)
        cache_file = os.path.join(MARKET_DATA_CACHE_DIR, f"{cache_key}.parquet")
        
        try:
            data.to_parquet(cache_file)
            logger.info(f"数据已缓存到: {cache_file}")
        except Exception as e:
            logger.error(f"缓存数据时出错: {e}")
    
    def _generate_cache_key(
        self,
        symbols: List[str],
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
        period: Optional[str],
        interval: str,
        provider: str
    ) -> str:
        """生成缓存键"""
        import hashlib
        
        # 处理日期
        start_str = ""
        if start_date:
            start_str = start_date.strftime("%Y%m%d") if isinstance(start_date, datetime) else str(start_date)
        
        end_str = ""
        if end_date:
            end_str = end_date.strftime("%Y%m%d") if isinstance(end_date, datetime) else str(end_date)
        
        # 生成键
        key_parts = [
            "_".join(sorted(symbols)),
            start_str,
            end_str,
            period or "",
            interval,
            provider
        ]
        
        # 对长键使用哈希
        if len("_".join(key_parts)) > 100:
            hash_obj = hashlib.md5("_".join(key_parts).encode())
            return f"{provider}_{interval}_{hash_obj.hexdigest()}"
        else:
            return "_".join(key_parts)
    
    def _save_to_database(self, data: pd.DataFrame, symbols: List[str], interval: str) -> bool:
        """保存数据到数据库"""
        if data.empty:
            return False
        
        try:
            table_name = f"market_data_{interval}"
            return self.db_manager.save_market_data(data, table_name)
        except Exception as e:
            logger.error(f"保存数据到数据库时出错: {e}")
            return False
    
    def _convert_interval_to_ibkr_format(self, interval: str) -> str:
        """将标准间隔转换为IBKR格式"""
        interval_mapping = {
            "1m": "1 min",
            "2m": "2 mins",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "60m": "1 hour",
            "1h": "1 hour",
            "1d": "1 day",
            "1w": "1 week",
            "1mo": "1 month"
        }
        
        return interval_mapping.get(interval, "1 day")
    
    def update_market_data(
        self, 
        symbols: List[str], 
        interval: str = "1d", 
        provider: Optional[str] = None,
        days_to_update: int = 7
    ) -> pd.DataFrame:
        """
        更新现有市场数据，只获取最新部分
        
        Args:
            symbols: 股票代码列表
            interval: 时间间隔
            provider: 数据提供商
            days_to_update: 要更新的天数
            
        Returns:
            更新后的DataFrame
        """
        provider = provider or self.default_provider
        
        # 获取现有数据
        existing_data = self.db_manager.get_market_data(symbols, interval)
        
        if existing_data.empty:
            logger.info(f"未找到现有数据，将获取完整历史数据")
            return self.get_market_data(
                symbols=symbols,
                interval=interval,
                provider=provider,
                days=30  # 默认获取30天数据
            )
        
        # 确定最新日期
        latest_date = existing_data.index.max()
        start_date = latest_date - timedelta(days=1)  # 获取重叠部分，以便验证和处理数据缺口
        end_date = datetime.now()
        
        # 获取最新数据
        new_data = self.get_market_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            provider=provider,
            save_to_db=False  # 稍后会一次性保存合并后的数据
        )
        
        if new_data.empty:
            logger.warning(f"未能获取到新数据")
            return existing_data
        
        # 合并数据，移除重复项
        combined_data = pd.concat([existing_data, new_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data = combined_data.sort_index()
        
        # 保存到数据库
        self._save_to_database(combined_data, symbols, interval)
        
        return combined_data
    
    def get_data_coverage(self, symbols: List[str], interval: str = "1d") -> Dict[str, Dict]:
        """
        获取数据覆盖情况
        
        Args:
            symbols: 股票代码列表
            interval: 时间间隔
            
        Returns:
            每个股票的数据覆盖情况字典
        """
        result = {}
        
        for symbol in symbols:
            # 从数据库获取数据
            data = self.db_manager.get_market_data([symbol], interval)
            
            if data.empty:
                result[symbol] = {
                    "available": False,
                    "start_date": None,
                    "end_date": None,
                    "days_covered": 0,
                    "data_points": 0,
                    "missing_dates": []
                }
                continue
            
            symbol_data = data[data["symbol"] == symbol] if "symbol" in data.columns else data
            
            if symbol_data.empty:
                result[symbol] = {
                    "available": False,
                    "start_date": None,
                    "end_date": None,
                    "days_covered": 0,
                    "data_points": 0,
                    "missing_dates": []
                }
                continue
            
            # 计算数据覆盖情况
            start_date = symbol_data.index.min()
            end_date = symbol_data.index.max()
            days_covered = (end_date - start_date).days + 1
            data_points = len(symbol_data)
            
            # 检查是否有缺失的交易日
            if interval in ["1d", "1day", "day"]:
                all_trading_days = self.get_market_calendar(
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not all_trading_days.empty:
                    missing_dates = []
                    for trading_day in all_trading_days["date"]:
                        if trading_day not in symbol_data.index:
                            missing_dates.append(trading_day.strftime("%Y-%m-%d"))
                else:
                    missing_dates = []
            else:
                missing_dates = []
            
            result[symbol] = {
                "available": True,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "days_covered": days_covered,
                "data_points": data_points,
                "missing_dates": missing_dates[:10]  # 只返回前10个缺失日期
            }
            
            # 添加缺失日期数量
            result[symbol]["missing_dates_count"] = len(missing_dates)
        
        return result
    
    def get_symbol_info(self, symbol: str, provider: Optional[str] = None) -> Dict:
        """
        获取股票信息
        
        Args:
            symbol: 股票代码
            provider: 数据提供商
            
        Returns:
            股票信息字典
        """
        provider = provider or self.default_provider
        
        if provider == "yahoo":
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # 提取关键信息
                result = {
                    "symbol": symbol,
                    "name": info.get("shortName", ""),
                    "long_name": info.get("longName", ""),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                    "exchange": info.get("exchange", ""),
                    "currency": info.get("currency", ""),
                    "market_cap": info.get("marketCap", None),
                    "beta": info.get("beta", None),
                    "pe_ratio": info.get("trailingPE", None),
                    "eps": info.get("trailingEps", None),
                    "dividend_yield": info.get("dividendYield", None),
                    "52wk_high": info.get("fiftyTwoWeekHigh", None),
                    "52wk_low": info.get("fiftyTwoWeekLow", None),
                    "avg_volume": info.get("averageVolume", None),
                    "website": info.get("website", ""),
                    "business_summary": info.get("longBusinessSummary", "")
                }
                
                return result
            
            except Exception as e:
                logger.error(f"获取{symbol}信息时出错: {e}")
                return {"symbol": symbol, "error": str(e)}
        
        elif provider == "ibkr":
            # 使用IBKR API获取股票信息
            logger.info(f"使用IBKR获取{symbol}的信息")
            # 实现IBKR股票信息获取...
            return {"symbol": symbol, "error": "IBKR provider not implemented"}
        
        elif provider == "external":
            # 使用外部源获取股票信息
            logger.info(f"使用外部源获取{symbol}的信息")
            # 实现外部源股票信息获取...
            return {"symbol": symbol, "error": "External provider not implemented"}
        
        else:
            logger.error(f"不支持的数据提供商: {provider}")
            return {"symbol": symbol, "error": f"Unsupported provider: {provider}"}
    
    def get_news(
        self, 
        symbols: List[str], 
        days: int = 7, 
        provider: Optional[str] = None
    ) -> List[Dict]:
        """
        获取股票相关新闻
        
        Args:
            symbols: 股票代码列表
            days: 过去多少天的新闻
            provider: 数据提供商
            
        Returns:
            新闻列表
        """
        provider = provider or self.default_provider
        news_list = []
        
        if provider == "yahoo":
            try:
                import yfinance as yf
                
                for symbol in symbols:
                    ticker = yf.Ticker(symbol)
                    news = ticker.news
                    
                    if news:
                        for item in news:
                            # 检查日期
                            if "providerPublishTime" in item:
                                publish_time = datetime.fromtimestamp(item["providerPublishTime"])
                                if (datetime.now() - publish_time).days > days:
                                    continue
                            
                            news_item = {
                                "symbol": symbol,
                                "title": item.get("title", ""),
                                "publisher": item.get("publisher", ""),
                                "link": item.get("link", ""),
                                "publish_time": publish_time.strftime("%Y-%m-%d %H:%M:%S") if "providerPublishTime" in item else "",
                                "type": item.get("type", ""),
                                "related_tickers": item.get("relatedTickers", [])
                            }
                            
                            news_list.append(news_item)
                
                return news_list
            
            except Exception as e:
                logger.error(f"获取新闻时出错: {e}")
                return []
        
        elif provider == "ibkr":
            # 使用IBKR API获取新闻
            logger.info(f"使用IBKR获取新闻")
            # 实现IBKR新闻获取...
            return []
        
        elif provider == "external":
            # 使用外部源获取新闻
            logger.info(f"使用外部源获取新闻")
            # 实现外部源新闻获取...
            return []
        
        else:
            logger.error(f"不支持的数据提供商: {provider}")
            return []
    
    def create_dataset(
        self, 
        symbols: List[str], 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d",
        include_indicators: bool = True,
        include_fundamentals: bool = False,
        provider: Optional[str] = None,
        resample: Optional[str] = None,
        fillna_method: str = "ffill"
    ) -> pd.DataFrame:
        """
        创建用于研究或回测的完整数据集
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔
            include_indicators: 是否包含技术指标
            include_fundamentals: 是否包含基本面数据
            provider: 数据提供商
            resample: 重采样频率 (None, "W", "M", "Q")
            fillna_method: 填充缺失值的方法 ("ffill", "bfill", "zero", "mean")
            
        Returns:
            完整的数据集DataFrame
        """
        # 获取市场数据
        market_data = self.get_market_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            provider=provider,
            include_indicators=include_indicators
        )
        
        if market_data.empty:
            logger.warning("未能获取市场数据")
            return pd.DataFrame()
        
        # 添加基本面数据
        if include_fundamentals:
            fundamental_data_dict = self.get_fundamental_data(
                symbols=symbols,
                provider=provider
            )
            
            # 将基本面数据转换为时间序列并合并
            fundamental_ts = self._convert_fundamentals_to_timeseries(
                fundamental_data_dict, market_data.index
            )
            
            if not fundamental_ts.empty:
                # 合并基本面数据
                market_data = pd.merge(
                    market_data, 
                    fundamental_ts, 
                    left_index=True, 
                    right_index=True, 
                    how="left"
                )
        
        # 重采样数据
        if resample:
            market_data = self._resample_data(market_data, resample)
        
        # 填充缺失值
        market_data = self._fill_missing_values(market_data, method=fillna_method)
        
        return market_data
    
    def _convert_fundamentals_to_timeseries(
        self, 
        fundamental_data: Dict[str, Dict], 
        date_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """将基本面数据转换为时间序列"""
        result_dfs = []
        
        for symbol, data in fundamental_data.items():
            # 处理季度财务数据
            if "quarterly_financials" in data and data["quarterly_financials"]:
                # 提取关键财务指标
                financials = {}
                
                for metric, values in data["quarterly_financials"].items():
                    # 转换日期和值
                    financial_dates = []
                    financial_values = []
                    
                    for date_str, value in values.items():
                        try:
                            date = pd.to_datetime(date_str)
                            financial_dates.append(date)
                            financial_values.append(value)
                        except:
                            continue
                    
                    if financial_dates:
                        # 创建时间序列
                        ts = pd.Series(financial_values, index=financial_dates)
                        # 重新索引到主日期索引
                        ts = ts.reindex(date_index, method="ffill")
                        # 添加到字典
                        col_name = f"{symbol}_quarterly_{metric.lower().replace(' ', '_')}"
                        financials[col_name] = ts
                
                # 转换为DataFrame
                if financials:
                    df = pd.DataFrame(financials)
                    result_dfs.append(df)
            
            # 处理每日/指标数据
            if "info" in data and data["info"]:
                daily_metrics = {}
                
                # 提取关键指标
                key_metrics = [
                    "marketCap", "trailingPE", "forwardPE", "dividendYield",
                    "beta", "priceToBook", "trailingEps", "forwardEps",
                    "profitMargins", "returnOnAssets", "returnOnEquity",
                    "debtToEquity", "currentRatio", "quickRatio"
                ]
                
                for metric in key_metrics:
                    if metric in data["info"] and data["info"][metric] is not None:
                        col_name = f"{symbol}_{metric.lower()}"
                        # 为所有日期创建相同的值
                        daily_metrics[col_name] = pd.Series(
                            [data["info"][metric]] * len(date_index),
                            index=date_index
                        )
                
                if daily_metrics:
                    df = pd.DataFrame(daily_metrics)
                    result_dfs.append(df)
        
        # 合并所有DataFrame
        if result_dfs:
            return pd.concat(result_dfs, axis=1)
        else:
            return pd.DataFrame()
    
    def _resample_data(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """重采样数据"""
        # 确保索引是DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # 检查是否存在分组列
        group_cols = [col for col in data.columns if col in ["symbol", "asset", "ticker"]]
        
        if group_cols:
            group_col = group_cols[0]
            
            # 按组重采样
            resampled_dfs = []
            for name, group in data.groupby(group_col):
                group_resampled = self._resample_group(group, freq)
                resampled_dfs.append(group_resampled)
            
            return pd.concat(resampled_dfs)
        else:
            # 单一序列重采样
            return self._resample_group(data, freq)
    
    def _resample_group(self, data: pd.DataFrame, freq: str) -> pd.DataFrame:
        """重采样单个组的数据"""
        price_cols = ["open", "high", "low", "close", "adj_close"]
        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        non_numeric_cols = [col for col in data.columns if col not in numeric_cols]
        
        # 保存非数值列
        non_numeric_data = data[non_numeric_cols].copy() if non_numeric_cols else None
        
        # 对价格和交易量使用特定规则
        resampled = pd.DataFrame(index=data.index)
        
        for col in numeric_cols:
            if col == "volume":
                # 交易量求和
                resampled[col] = data[col].resample(freq).sum()
            elif col in price_cols:
                if col == "open":
                    # 开盘价取第一个
                    resampled[col] = data[col].resample(freq).first()
                elif col == "high":
                    # 最高价取最大值
                    resampled[col] = data[col].resample(freq).max()
                elif col == "low":
                    # 最低价取最小值
                    resampled[col] = data[col].resample(freq).min()
                elif col in ["close", "adj_close"]:
                    # 收盘价取最后一个
                    resampled[col] = data[col].resample(freq).last()
            else:
                # 其他数值列取平均值
                resampled[col] = data[col].resample(freq).mean()
        
        # 恢复非数值列
        if non_numeric_data is not None:
            for col in non_numeric_cols:
                # 非数值列取第一个
                temp_series = data[col].resample(freq).first()
                resampled[col] = temp_series
        
        return resampled
    
    def _fill_missing_values(self, data: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        """填充缺失值"""
        if data.empty:
            return data
        
        # 复制数据
        filled_data = data.copy()
        
        # 获取数值列
        numeric_cols = filled_data.select_dtypes(include=["number"]).columns.tolist()
        
        # 按方法填充
        if method == "ffill":
            filled_data[numeric_cols] = filled_data[numeric_cols].fillna(method="ffill")
            # 向前填充后仍可能有NaN，向后填充剩余的
            filled_data[numeric_cols] = filled_data[numeric_cols].fillna(method="bfill")
        elif method == "bfill":
            filled_data[numeric_cols] = filled_data[numeric_cols].fillna(method="bfill")
            # 向后填充后仍可能有NaN，向前填充剩余的
            filled_data[numeric_cols] = filled_data[numeric_cols].fillna(method="ffill")
        elif method == "zero":
            filled_data[numeric_cols] = filled_data[numeric_cols].fillna(0)
        elif method == "mean":
            for col in numeric_cols:
                filled_data[col] = filled_data[col].fillna(filled_data[col].mean())
        
        return filled_data