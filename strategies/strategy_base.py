# strategies/strategy_base.py
"""
策略基类 - 为所有交易策略提供共同的基础功能
"""
import backtrader as bt
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

from utils.logger import setup_logger

logger = setup_logger('StrategyBase')

class StrategyBase(bt.Strategy):
    """
    策略基类，提供所有策略共享的基础功能
    
    继承此类创建新策略，并实现必要的方法，如next()
    """
    
    # 类级别参数，所有对象共享，可在回测时覆盖
    params = (
        ('printlog', False),        # 是否打印日志
        ('stake', 100),             # 每次买卖的数量
        ('max_positions', 5),       # 最大持仓数量
        ('max_risk_pct', 0.02),     # 每笔交易最大风险比例（占总资金）
        ('stop_loss_atr', 2.0),     # 基于ATR的止损倍数
        ('trail_stop', False),      # 是否启用跟踪止损
        ('trail_percent', 0.05),    # 跟踪止损百分比
        ('use_target_size', True),  # 是否使用目标头寸大小而非固定股数
        ('pyramiding', 1),          # 金字塔加仓次数（1为不加仓）
        ('debug', False),           # 是否启用调试模式
    )
    
    def __init__(self):
        """初始化策略"""
        # 跟踪各种策略指标和状态
        self.order = None  # 记录当前挂单
        self.buyprice = 0  # 买入价格
        self.buycomm = 0   # 买入佣金
        self.bar_executed = 0  # 执行交易的bar索引
        self.trade_count = 0  # 交易次数
        self.trade_history = []  # 交易历史
        
        # 记录权益曲线
        self.equity_curve = pd.Series()
        
        # 初始化技术指标
        self._init_indicators()
        
        # 记录每日权益曲线
        self.add_timer(
            when=bt.Timer.SESSION_END,
            monthdays=[d for d in range(1, 32)],  # 每一天
            monthcarry=True,
            cheat=False,
            callback=self._record_equity
        )
    
    def _init_indicators(self):
        """
        初始化技术指标
        在子类中重写此方法以添加特定的指标
        """
        # 基础指标示例
        self.sma20 = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        self.sma50 = bt.indicators.SimpleMovingAverage(self.data.close, period=50)
        self.sma200 = bt.indicators.SimpleMovingAverage(self.data.close, period=200)
        
        # 计算ATR (Average True Range)
        self.atr = bt.indicators.ATR(self.data, period=14)
        
        # 波动率指标
        self.volatility = bt.indicators.StandardDeviation(self.data.close, period=20)
        
        # RSI指标
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        
        # MACD指标
        self.macd = bt.indicators.MACD(
            self.data.close, period_me1=12, period_me2=26, period_signal=9)
        
        logger.info("基础技术指标初始化完成")
    
    def notify_order(self, order):
        """
        订单状态通知回调
        """
        # 检查订单是否完成
        if order.status in [order.Submitted, order.Accepted]:
            # 等待订单执行...
            return
        
        # 检查订单是否执行
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格={order.executed.price:.2f}, '
                         f'成本={order.executed.value:.2f}, '
                         f'佣金={order.executed.comm:.2f}')
                
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # 卖出
                self.log(f'卖出执行: 价格={order.executed.price:.2f}, '
                         f'成本={order.executed.value:.2f}, '
                         f'佣金={order.executed.comm:.2f}')
            
            # 记录执行该订单的bar
            self.bar_executed = len(self)
        
        # 检查订单是否被拒绝或取消
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单被取消/拒绝/无法执行，状态: {order.getstatusname()}')
        
        # 重置订单引用
        self.order = None
    
    def notify_trade(self, trade):
        """
        交易完成通知回调
        """
        if not trade.isclosed:
            return
        
        # 记录交易盈亏
        self.log(f'交易利润: 毛利={trade.pnl:.2f}, 净利={trade.pnlcomm:.2f}')
        
        # 增加交易计数
        self.trade_count += 1
        
        # 记录交易历史
        trade_record = {
            'ref': trade.ref,
            'symbol': self.data._name if hasattr(self.data, '_name') else 'unknown',
            'size': trade.size,
            'price': trade.price,
            'value': trade.value,
            'commission': trade.commission,
            'pnl': trade.pnlcomm,  # 使用净盈亏
            'type': 'long' if trade.size > 0 else 'short',
            'entry_date': bt.num2date(trade.dtopen).strftime('%Y-%m-%d'),
            'exit_date': bt.num2date(trade.dtclose).strftime('%Y-%m-%d'),
            'entry_price': trade.price,
            'exit_price': trade.data.close[0],
            'holding_period': (trade.dtclose - trade.dtopen),
            'status': 'closed'
        }
        
        self.trade_history.append(trade_record)
    
    def _record_equity(self):
        """
        记录每日权益曲线
        """
        current_date = self.data.datetime.date()
        current_value = self.broker.getvalue()
        
        # 记录当前日期和权益值
        self.equity_curve[current_date] = current_value
    
    def log(self, txt, dt=None):
        """
        日志记录方法
        """
        if self.params.printlog:
            dt = dt or self.data.datetime.date(0)
            print(f'[{dt.isoformat()}] {txt}')
            logger.info(f'[{dt.isoformat()}] {txt}')
    
    def buy_signal(self):
        """
        检查是否有买入信号
        在子类中重写此方法以实现特定的买入逻辑
        
        返回:
            bool: 是否产生买入信号
        """
        return False
    
    def sell_signal(self):
        """
        检查是否有卖出信号
        在子类中重写此方法以实现特定的卖出逻辑
        
        返回:
            bool: 是否产生卖出信号
        """
        return False
    
    def position_size(self):
        """
        计算头寸大小
        在子类中重写此方法以实现特定的头寸大小计算逻辑
        
        返回:
            int: 头寸大小
        """
        if self.params.use_target_size:
            # 基于风险的头寸大小
            risk_per_share = self.atr[0] * self.params.stop_loss_atr
            if risk_per_share > 0:
                max_risk_amount = self.broker.getvalue() * self.params.max_risk_pct
                size = max_risk_amount / risk_per_share
                return int(size)
        
        # 默认使用固定头寸大小
        return self.params.stake
    
    def set_stop_loss(self, price, stop_price=None):
        """
        设置止损单
        
        参数:
            price: 买入价格
            stop_price: 止损价格，如果为None则基于ATR计算
        """
        if stop_price is None:
            # 计算基于ATR的止损
            stop_price = price - (self.atr[0] * self.params.stop_loss_atr)
        
        # 创建止损单
        return self.sell(exectype=bt.Order.Stop, price=stop_price)
    
    def set_take_profit(self, price, target_price=None, risk_reward=2.0):
        """
        设置止盈单
        
        参数:
            price: 买入价格
            target_price: 止盈价格，如果为None则基于风险回报比计算
            risk_reward: 风险回报比，用于计算止盈
        """
        if target_price is None:
            # 计算基于风险回报比的止盈
            stop_loss = price - (self.atr[0] * self.params.stop_loss_atr)
            risk = price - stop_loss
            target_price = price + (risk * risk_reward)
        
        # 创建止盈单
        return self.sell(exectype=bt.Order.Limit, price=target_price)
    
    def next(self):
        """
        策略核心逻辑，每个bar调用一次
        在子类中重写此方法以实现特定的策略逻辑
        """
        # 检查是否有挂单
        if self.order:
            return
        
        # 检查是否有持仓
        if not self.position:
            # 检查买入信号
            if self.buy_signal():
                size = self.position_size()
                self.log(f'买入信号, 创建买入订单: {size} 股，价格 {self.data.close[0]:.2f}')
                self.order = self.buy(size=size)
                
                # 如果需要设置止损，在买入单成交后的notify_order中设置
        else:
            # 检查卖出信号
            if self.sell_signal():
                self.log(f'卖出信号, 创建卖出订单: {self.position.size} 股，价格 {self.data.close[0]:.2f}')
                self.order = self.sell(size=self.position.size)
    
    def stop(self):
        """
        回测结束时调用
        可以用于输出最终结果或清理资源
        """
        self.log(f'策略执行结束: 总交易次数 {self.trade_count}')
        self.log(f'期末总资产: {self.broker.getvalue():.2f}')
        
        # 确保权益曲线已完整记录
        if len(self.equity_curve) == 0:
            # 如果没有记录权益曲线（可能是因为没有使用timer），创建一个简单的曲线
            self.equity_curve = pd.Series([v[0] for v in self._value], 
                                        index=[bt.num2date(d[0]).date() for d in self.data.datetime])