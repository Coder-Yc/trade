"""
日志工具模块
"""
import logging
import os
import datetime
import sys
from config.settings import LOG_CONFIG

def setup_logger(name, level=None):
    """
    设置并返回一个配置好的日志记录器
    
    参数:
        name (str): 日志记录器名称
        level (int, optional): 日志级别，默认使用配置中的级别
        
    返回:
        logging.Logger: 配置好的日志记录器
    """
    if level is None:
        level_name = LOG_CONFIG.get('LOG_LEVEL', 'INFO')
        level = getattr(logging, level_name)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    log_dir = LOG_CONFIG.get('LOG_DIR', 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_format = LOG_CONFIG.get('LOG_FORMAT')
    formatter = logging.Formatter(log_format)
    
    today = datetime.datetime.now().strftime('%Y%m%d')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{today}.log")
    )
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger