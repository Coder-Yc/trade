# strategies/mean_reversion/rsi.py
"""
RSI均值回归策略 - 经典的超买超卖策略
使用RSI指标在超卖区域买入，超买区域卖出
"""
import backtrader as bt
from typing import List, Dict, Any, Optional, Union, Tuple

from strategies.strategy_base import StrategyBase
from utils.logger import setup_logger

logger = setup_logger('RSIStrategy')

class RSIMeanReversion(StrategyBase):
    """
    RSI均值回归策略
    
    参数:
        rsi_period: RSI计算周期
        rsi_overbought: RSI超买阈值
        rsi_oversold: RSI超卖阈值
        order_percentage: 每次下单资金百分比
        use_atr_stop: 是否使用ATR止损
        exit_on_middle: 是否在RSI回到中间区域时平仓
        middle_threshold: 中间区域阈值
        use_ema_filter: 是否使用均线过滤
        ema_period: 均线周期
    """
    
    params = (
        ('rsi_period', 14),        # RSI计算周期
        ('rsi_overbought', 70),    # RSI超买阈值
        ('rsi_oversold', 30),      # RSI超卖阈值
        ('order_percentage', 0.95), # 每次下单资金百分比
        ('use_atr_stop', True),    # 是否使用ATR止损
        ('atr_period', 14),        # ATR计算周期
        ('atr_stop_multiplier', 2), # ATR止损乘数
        ('exit_on_middle', True),  # 是否在RSI回到中间区域时平仓
        ('middle_threshold', 50),  # 中间区域阈值
        ('use_ema_filter', False), # 是否使用均线过滤
        ('ema_period', 200),       # 均线周期
    )
    
    def _init_indicators(self):
        """初始化策略指标"""
        # 调用父类方法初始化基础指标
        super()._init_indicators()