"""
全局配置设置
"""

IBKR_CONFIG = {
    'HOST': '127.0.0.1',
    'PAPER_PORT': 7497, 
    'LIVE_PORT': 7496,   
    'CLIENT_ID': 1,
    'READ_TIMEOUT': 60
}

RECONNECT_CONFIG = {
    'AUTO_RECONNECT': True,
    'RECONNECT_INTERVAL': 60,  # 秒
    'MAX_RECONNECT_ATTEMPTS': 10
}

LOG_CONFIG = {
    'LOG_LEVEL': 'INFO',
    'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'LOG_DIR': 'logs'
}

# Yahoo Finance API配置
YAHOO_DATA_CACHE_DIR = "data/cache/yahoo" 

# 市场数据配置
MARKET_DATA_CACHE_DIR = "data/cache/market_data"
DEFAULT_DATA_PROVIDER = "yahoo"  