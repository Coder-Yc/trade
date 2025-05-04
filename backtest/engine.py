# backtest/engine.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional

from data.market_data import MarketDataManager
from strategies.strategy_base import StrategyBase

class BacktestEngine:
    """回测引擎，用于对交易策略进行历史数据回测"""

    def __init__(
        self,
        strategy: StrategyBase,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        interval: str = '1d',
        commission: float = 0.001,  # 0.1%交易成本
        slippage: float = 0.001,    # 0.1%滑点
    ):
        """
        初始化回测引擎
        
        Args:
            strategy: 策略实例，必须是StrategyBase的子类
            symbol: 回测的股票代码
            start_date: 回测开始日期 (YYYY-MM-DD)
            end_date: 回测结束日期 (YYYY-MM-DD)
            initial_capital: 初始资金
            commission: 交易佣金率
            slippage: 滑点率
        """
        # 验证策略类型
        if not isinstance(strategy, StrategyBase):
            raise TypeError("策略必须是StrategyBase的子类")
        
        self.strategy = strategy
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # 市场数据管理器
        self.data_manager = MarketDataManager()
        
        # 回测结果
        self.portfolio_value = []  # 组合价值
        self.positions = []        # 持仓
        self.trades = []           # 交易记录
        self.performance = {}      # 绩效指标
        
        # 当前状态
        self.current_cash = initial_capital
        self.current_position = 0
        self.current_portfolio_value = initial_capital
        
        # 回测数据
        self.data = None
        self.results = None
    
    def _load_data(self) -> pd.DataFrame:
        """从CSV文件加载回测数据"""
        print("正在加载市场数据...")
        
        # 设置数据目录
        import os
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'data', 'storage', 'files', 'market_data')
        
        # 获取完整的文件路径 - 使用symbol和period作为文件名
        file_name = f"{self.symbol}_{self.interval}.csv"
        file_path = os.path.join(data_dir, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise ValueError(f"找不到数据文件: {file_path}")
        
        print(f"使用数据文件: {file_path}")
        
        # 读取CSV文件
        try:
            df = pd.read_csv(file_path)
            
            # 确保日期列是日期类型
            date_col = None
            for col in ['date', 'Date', 'datetime', 'Datetime', 'time', 'Time']:
                if col in df.columns:
                    date_col = col
                    break
                    
            if date_col is None:
                raise ValueError("CSV文件中找不到日期列")
                
            # 转换日期列，并处理时区
            df[date_col] = pd.to_datetime(df[date_col], utc=True)
            df.set_index(date_col, inplace=True)
            # 将索引转换为UTC时间 - 解决时区比较问题
            df.index = df.index.tz_convert('UTC')
            
            # 标准化列名
            col_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            }
            
            df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns}, inplace=True)
            
            # 确保必要的列存在
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"CSV文件缺少必要的列: {missing_cols}")
            
            # 如果没有volume列，添加一个虚拟的
            if 'volume' not in df.columns:
                df['volume'] = 0
                
            # 过滤日期范围 - 确保处理时区问题
            if self.start_date:
                # 将start_date转换为带时区的日期
                start_date = pd.to_datetime(self.start_date).tz_localize('UTC')
                df = df[df.index >= start_date]
                
            if self.end_date:
                # 将end_date转换为带时区的日期
                end_date = pd.to_datetime(self.end_date).tz_localize('UTC')
                df = df[df.index <= end_date]
            
            # 添加symbol列
            df['symbol'] = self.symbol
            
            print(f"成功加载{len(df)}行数据")
            return df
            
        except Exception as e:
            print(f"加载CSV文件时出错: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            raise
    def run(self) -> pd.DataFrame:
        """
        运行回测
        
        Returns:
            pd.DataFrame: 回测结果
        """
        # 加载数据
        self.data = self._load_data()
        
        # 将数据传递给策略
        self.strategy.set_data(self.data)
        
        # 初始化回测结果
        results = []
        
        print("开始回测...")
        
        # 遍历每个交易日
        for i, (timestamp, row) in enumerate(self.data.iterrows()):
            # 当前日期和价格
            date = timestamp.strftime('%Y-%m-%d')
            price = row['close']
            
            # 更新当前组合价值
            self.current_portfolio_value = self.current_cash + self.current_position * price
            
            # 记录当前状态
            self.portfolio_value.append({
                'date': date,
                'portfolio_value': self.current_portfolio_value,
                'cash': self.current_cash,
                'position': self.current_position,
                'position_value': self.current_position * price,
                'price': price,
            })
            
            # 生成信号(如果不是第一天)
            if i > 0:
                # 获取当前数据状态
                current_data = self.data.iloc[:i+1]
                
                # 生成信号
                signal = self.strategy.generate_signal(current_data)
                
                # 执行交易
                if signal != 0:
                    self._execute_trade(date, price, signal)
            
            # 添加到结果中
            results.append({
                'date': date,
                'price': price,
                'portfolio_value': self.current_portfolio_value,
                'cash': self.current_cash,
                'position': self.current_position,
            })
        
        # 转换为DataFrame
        self.results = pd.DataFrame(results)
        self.results['date'] = pd.to_datetime(self.results['date'])
        self.results.set_index('date', inplace=True)
        
        # 计算绩效指标
        self._calculate_performance()
        
        print("回测完成!")
        print(f"最终组合价值: {self.current_portfolio_value:.2f}")
        
        return self.results
    
    def _execute_trade(self, date: str, price: float, signal: int) -> None:
        """
        执行交易
        
        Args:
            date: 交易日期
            price: 交易价格
            signal: 交易信号(1=买入, -1=卖出)
        """
        # 根据信号计算交易数量
        if signal > 0:  # 买入信号
            if self.current_position == 0:  # 当前无持仓
                # 计算可购买的最大数量
                available_funds = self.current_cash * 0.99  # 预留1%现金
                # 考虑交易成本
                max_shares = available_funds / (price * (1 + self.commission + self.slippage))
                shares_to_buy = int(max_shares)  # 取整数
                
                if shares_to_buy > 0:
                    # 计算实际成本
                    trade_cost = shares_to_buy * price
                    commission_cost = trade_cost * self.commission
                    slippage_cost = trade_cost * self.slippage
                    total_cost = trade_cost + commission_cost + slippage_cost
                    
                    # 更新现金和持仓
                    self.current_cash -= total_cost
                    self.current_position += shares_to_buy
                    
                    # 记录交易
                    self.trades.append({
                        'date': date,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': price,
                        'trade_value': trade_cost,
                        'commission': commission_cost,
                        'slippage': slippage_cost,
                        'total_cost': total_cost,
                    })
                    
                    print(f"{date} - 买入: {shares_to_buy}股, 价格: {price:.2f}, 总成本: {total_cost:.2f}")
        
        elif signal < 0:  # 卖出信号
            if self.current_position > 0:  # 当前有持仓
                shares_to_sell = self.current_position  # 全部卖出
                
                # 计算实际收益
                trade_value = shares_to_sell * price
                commission_cost = trade_value * self.commission
                slippage_cost = trade_value * self.slippage
                total_received = trade_value - commission_cost - slippage_cost
                
                # 更新现金和持仓
                self.current_cash += total_received
                self.current_position = 0
                
                # 记录交易
                self.trades.append({
                    'date': date,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': price,
                    'trade_value': trade_value,
                    'commission': commission_cost,
                    'slippage': slippage_cost,
                    'total_received': total_received,
                })
                
                print(f"{date} - 卖出: {shares_to_sell}股, 价格: {price:.2f}, 总收入: {total_received:.2f}")
    
    def _calculate_performance(self) -> None:
        """计算绩效指标"""
        if self.results is None or self.results.empty:
            return
        
        # 计算每日回报率
        self.results['daily_returns'] = self.results['portfolio_value'].pct_change()
        
        # 去除NaN
        daily_returns = self.results['daily_returns'].dropna()
        
        if len(daily_returns) == 0:
            return
        
        # 计算累计收益
        total_return = (self.current_portfolio_value / self.initial_capital) - 1
        
        # 计算年化收益率
        days = (self.results.index[-1] - self.results.index[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0
        
        # 计算最大回撤
        portfolio_values = self.results['portfolio_value']
        peak = portfolio_values.expanding(min_periods=1).max()
        drawdown = (portfolio_values / peak) - 1
        max_drawdown = drawdown.min()
        
        # 计算夏普比率
        risk_free_rate = 0.01  # 假设无风险利率为1%
        excess_return = daily_returns.mean() - risk_free_rate / 252
        sharpe_ratio = (excess_return / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # 计算交易次数
        num_trades = len(self.trades)
        
        # 盈利交易数量
        if num_trades > 0:
            profitable_trades = sum(1 for t in self.trades if t['action'] == 'SELL' and 
                                   t['total_received'] > sum(t2['total_cost'] for t2 in self.trades 
                                                           if t2['action'] == 'BUY' and 
                                                           self.trades.index(t2) < self.trades.index(t)))
            win_rate = profitable_trades / num_trades if num_trades > 0 else 0
        else:
            profitable_trades = 0
            win_rate = 0
        
        # 存储绩效指标
        self.performance = {
            'initial_capital': self.initial_capital,
            'final_value': self.current_portfolio_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': num_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
        }
    
    def analyze_results(self) -> None:
        """分析并展示回测结果"""
        if self.results is None or self.performance is None:
            print("请先运行回测!")
            return
        
        # 打印绩效指标
        print("\n========== 回测结果 ==========")
        print(f"初始资金: {self.performance['initial_capital']:.2f}")
        print(f"最终价值: {self.performance['final_value']:.2f}")
        print(f"总收益率: {self.performance['total_return']*100:.2f}%")
        print(f"年化收益率: {self.performance['annual_return']*100:.2f}%")
        print(f"最大回撤: {self.performance['max_drawdown']*100:.2f}%")
        print(f"夏普比率: {self.performance['sharpe_ratio']:.4f}")
        print(f"交易次数: {self.performance['num_trades']}")
        print(f"盈利交易: {self.performance['profitable_trades']}")
        print(f"胜率: {self.performance['win_rate']*100:.2f}%")
        print("================================")
        
        # 绘制回测结果图表
        try:
            self._plot_results()
        except Exception as e:
            print(f"绘制图表时出错: {str(e)}")
    
    def _plot_results(self) -> None:
        """绘制回测结果图表"""
        plt.figure(figsize=(12, 8))
        
        # 绘制组合价值
        plt.subplot(2, 1, 1)
        plt.plot(self.results.index, self.results['portfolio_value'], label='Portfolio Value')
        plt.title(f'{self.strategy.__class__.__name__} - {self.symbol} Backtest Results')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        
        # 绘制股价和交易点
        plt.subplot(2, 1, 2)
        plt.plot(self.results.index, self.results['price'], label='Price')
        
        # 添加买入点
        buy_dates = [pd.to_datetime(t['date']) for t in self.trades if t['action'] == 'BUY']
        buy_prices = [t['price'] for t in self.trades if t['action'] == 'BUY']
        plt.scatter(buy_dates, buy_prices, marker='^', color='g', s=100, label='Buy')
        
        # 添加卖出点
        sell_dates = [pd.to_datetime(t['date']) for t in self.trades if t['action'] == 'SELL']
        sell_prices = [t['price'] for t in self.trades if t['action'] == 'SELL']
        plt.scatter(sell_dates, sell_prices, marker='v', color='r', s=100, label='Sell')
        
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()