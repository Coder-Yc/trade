# strategies/moving_average.py
import pandas as pd
from strategies.strategy_base import StrategyBase

class MovingAverageCrossover(StrategyBase):
    """移动平均线交叉策略"""
    
    def __init__(self):
        """
        初始化移动平均线交叉策略
        
        Args:
            short_window: 短期移动平均线窗口
            long_window: 长期移动平均线窗口
        """
        super().__init__()
        self.short_window=5
        self.long_window=20
    
    def generate_signal(self, data):
        """
        生成交易信号
        
        Args:
            data: 市场数据
            
        Returns:
            int: 交易信号(1=买入, -1=卖出, 0=不操作)
        """
        if len(data) < self.long_window + 1:
            return 0  
        
        # 计算短期和长期移动平均线
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        
        # 获取当前和前一个时间点的数据
        current_short_ma = short_ma.iloc[-1]
        current_long_ma = long_ma.iloc[-1]
        previous_short_ma = short_ma.iloc[-2]
        previous_long_ma = long_ma.iloc[-2]
        
        # 短期均线上穿长期均线 -> 买入信号
        if (current_short_ma > current_long_ma) and (previous_short_ma <= previous_long_ma):
            return 1
        
        # 短期均线下穿长期均线 -> 卖出信号
        elif (current_short_ma < current_long_ma) and (previous_short_ma >= previous_long_ma):
            return -1
        
        # 无交叉 -> 不操作
        else:
            return 0