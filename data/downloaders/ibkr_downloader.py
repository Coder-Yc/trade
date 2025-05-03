# IBKR数据下载模块
# data/downloaders/ibkr_downloader.py

from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Union, Optional, Tuple
import os

from broker.ibkr.ibkr_connector import IBKRConnector
from utils.logger import setup_logger

logger = setup_logger(__name__)

class IBKRDownloader:
    """Interactive Brokers数据下载器，提供从IBKR获取市场数据的功能"""
    
    def __init__(self, connector: Optional[IBKRConnector] = None):
        """
        初始化IBKR下载器
        
        Args:
            connector: IBKR连接器实例，如果不提供则创建一个新的
        """
        self.connector = connector
        self.is_connector_provided = connector is not None
        
    def _ensure_connection(self):
        """确保与IBKR连接"""
        if not self.connector:
            from config.settings import IBKR_CONFIG
            host = IBKR_CONFIG.get('HOST', '127.0.0.1')
            port = IBKR_CONFIG.get('PAPER_PORT', 7497)
            client_id = IBKR_CONFIG.get('CLIENT_ID', 1)
            
            logger.info(f"创建新的IBKR连接器，连接到 {host}:{port}")
            self.connector = IBKRConnector(
                host=host,
                port=port,
                client_id=client_id,
                is_paper_account=True
            )
        
        if not self.connector.is_connected():
            logger.info("连接到IBKR")
            return self.connector.connect()
        
        return True
    
    def download_historical_data(
        self, 
        symbols: List[str], 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        adjust_prices: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        下载历史OHLCV数据
        
        Args:
            symbols: 股票/期货等合约代码列表
            start_date: 开始日期
            end_date: 结束日期
            bar_size: 柱线大小，如"1 day", "1 hour", "5 mins"等
            what_to_show: 数据类型，如"TRADES", "MIDPOINT", "BID", "ASK"等
            adjust_prices: 是否调整价格(除权除息)
            
        Returns:
            字典，键为合约代码，值为包含历史数据的DataFrame
        """
        if not self._ensure_connection():
            logger.error("无法连接到IBKR，取消下载")
            return {}
        
        # 处理日期
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))
        
        # 计算持续时间字符串
        duration_str = self._calculate_duration_string(start_date, end_date)
        
        result = {}
        
        for symbol in symbols:
            try:
                logger.info(f"从IBKR下载{symbol}的历史数据")
                
                # 创建合约对象
                contract = self._create_contract(symbol)
                if not contract:
                    logger.warning(f"无法创建{symbol}的合约对象，跳过")
                    continue
                
                # 请求历史数据
                bars = self.connector.get_ib().reqHistoricalData(
                    contract=contract,
                    endDateTime=end_date.strftime("%Y%m%d %H:%M:%S"),
                    durationStr=duration_str,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=True,  # 只使用常规交易时段的数据
                    formatDate=1   # 1表示格式化的日期字符串
                )
                
                if not bars:
                    logger.warning(f"未获取到{symbol}的数据")
                    continue
                
                # 转换为DataFrame
                df = util.df(bars)
                
                # 标准化列名
                column_mapping = {
                    'date': 'date',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'average': 'vwap',
                    'barCount': 'count'
                }
                
                df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
                
                # 设置日期列为索引
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                # 添加symbol列
                df['symbol'] = symbol
                
                # 排序
                df.sort_index(inplace=True)
                
                result[symbol] = df
                logger.info(f"成功下载{symbol}的数据，获取了{len(df)}条记录")
                
            except Exception as e:
                logger.error(f"下载{symbol}的数据时出错: {e}")
        
        # 如果我们创建了自己的连接器，断开连接
        if not self.is_connector_provided and self.connector is not None:
            logger.info("断开IBKR连接")
            self.connector.disconnect()
            self.connector = None
        
        return result
    
    def _calculate_duration_string(self, start_date: datetime, end_date: datetime) -> str:
        """计算IBKR API使用的持续时间字符串"""
        delta = end_date - start_date
        
        if delta.days > 365:
            return f"{delta.days // 365 + 1} Y"
        elif delta.days > 30:
            return f"{delta.days // 30 + 1} M"
        else:
            return f"{delta.days + 1} D"
    
    def _create_contract(self, symbol: str) -> Optional[Contract]:
        """创建IBKR合约对象"""
        try:
            parts = symbol.split('.')
            main_symbol = parts[0]
            
            # 根据后缀确定交易所和合约类型
            if len(parts) > 1:
                suffix = parts[1].upper()
                
                # 美股
                if suffix == 'US' or suffix == 'NYSE':
                    return Stock(main_symbol, 'NYSE', 'USD')
                elif suffix == 'NASDAQ':
                    return Stock(main_symbol, 'NASDAQ', 'USD')
                
                # 其他国家股票
                elif suffix == 'HK':
                    return Stock(main_symbol, 'HKEX', 'HKD')
                elif suffix == 'LSE':
                    return Stock(main_symbol, 'LSE', 'GBP')
                elif suffix == 'TSE':
                    return Stock(main_symbol, 'TSE', 'CAD')
                
                # 期货
                elif suffix in ['CME', 'CBOT', 'NYMEX', 'COMEX']:
                    return Future(main_symbol, exchange=suffix)
                
                # 外汇
                elif suffix == 'FOREX':
                    if len(main_symbol) == 6:
                        base = main_symbol[:3]
                        quote = main_symbol[3:]
                        return Forex(f"{base}{quote}")
                    else:
                        return Forex(main_symbol)
                
                # 期权
                elif suffix == 'OPT':
                    # 需要更多详细信息才能创建期权合约
                    logger.warning(f"期权合约需要更多信息: {symbol}")
                    return None
                
                else:
                    logger.warning(f"未知的后缀: {suffix}，尝试使用SMART交易所")
                    return Stock(main_symbol, 'SMART', 'USD')
            
            else:
                # 没有后缀，默认美股
                return Stock(main_symbol, 'SMART', 'USD')
                
        except Exception as e:
            logger.error(f"创建合约时出错: {e}")
            return None
    
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        获取最新价格
        
        Args:
            symbols: 股票/期货等合约代码列表
            
        Returns:
            字典，键为合约代码，值为最新价格
        """
        if not self._ensure_connection():
            logger.error("无法连接到IBKR，取消获取最新价格")
            return {}
        
        ib = self.connector.get_ib()
        result = {}
        
        for symbol in symbols:
            try:
                contract = self._create_contract(symbol)
                if not contract:
                    logger.warning(f"无法创建{symbol}的合约对象，跳过")
                    continue
                
                # 请求市场数据
                ib.reqMktData(contract)
                # 等待数据
                ib.sleep(1)
                
                # 获取最新报价
                ticker = ib.ticker(contract)
                
                # 优先使用最后成交价，如果没有则使用最后出价或最后要价
                if ticker.last > 0:
                    result[symbol] = ticker.last
                elif ticker.bid > 0:
                    result[symbol] = ticker.bid
                elif ticker.ask > 0:
                    result[symbol] = ticker.ask
                else:
                    logger.warning(f"未能获取{symbol}的最新价格")
                
                # 取消市场数据订阅
                ib.cancelMktData(contract)
                
            except Exception as e:
                logger.error(f"获取{symbol}的最新价格时出错: {e}")
        
        # 如果我们创建了自己的连接器，断开连接
        if not self.is_connector_provided and self.connector is not None:
            logger.info("断开IBKR连接")
            self.connector.disconnect()
            self.connector = None
        
        return result
    
    def get_contract_details(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        获取合约详情
        
        Args:
            symbols: 股票/期货等合约代码列表
            
        Returns:
            字典，键为合约代码，值为合约详情字典
        """
        if not self._ensure_connection():
            logger.error("无法连接到IBKR，取消获取合约详情")
            return {}
        
        ib = self.connector.get_ib()
        result = {}
        
        for symbol in symbols:
            try:
                contract = self._create_contract(symbol)
                if not contract:
                    logger.warning(f"无法创建{symbol}的合约对象，跳过")
                    continue
                
                # 请求合约详情
                details = ib.reqContractDetails(contract)
                
                if not details:
                    logger.warning(f"未获取到{symbol}的合约详情")
                    continue
                
                # 提取关键信息
                detail = details[0]
                result[symbol] = {
                    'symbol': detail.contract.symbol,
                    'name': detail.longName,
                    'exchange': detail.contract.exchange,
                    'currency': detail.contract.currency,
                    'industry': detail.industry,
                    'category': detail.category,
                    'subcategory': detail.subcategory,
                    'min_tick': detail.minTick,
                    'price_magnifier': detail.priceMagnifier,
                    'trading_hours': detail.tradingHours,
                    'liquid_hours': detail.liquidHours,
                    'time_zone_id': detail.timeZoneId
                }
                
                logger.info(f"获取到{symbol}的合约详情")
                
            except Exception as e:
                logger.error(f"获取{symbol}的合约详情时出错: {e}")
        
        # 如果我们创建了自己的连接器，断开连接
        if not self.is_connector_provided and self.connector is not None:
            logger.info("断开IBKR连接")
            self.connector.disconnect()
            self.connector = None
        
        return result