# strategies/strategy_base.py
from abc import ABC, abstractmethod
import pandas as pd

class StrategyBase(ABC):
    """策略基类，所有交易策略必须继承该类"""
    
    def __init__(self):
        """初始化策略"""
        self.data = None  # 市场数据
    
    def set_data(self, data):
        """设置市场数据"""
        self.data = data
    
    @abstractmethod
    def generate_signal(self, data):
        """
        生成交易信号
        
        Args:
            data: 市场数据
            
        Returns:
            int: 交易信号(1=买入, -1=卖出, 0=不操作)
        """
        pass