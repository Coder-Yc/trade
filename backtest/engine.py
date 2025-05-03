# backtest/engine.py
"""
回测引擎
基于本地保存的历史数据进行策略回测
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Callable, Optional, Union, Any

from strategies.strategy_base import StrategyBase
from data.storage.file_storage import FileStorage, MARKET_DATA_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BacktestEngine:
    """
    回测引擎，用于在历史市场数据上回测交易策略
    """
    
    def __init__(self, strategy: StrategyBase, initial_capital: float = 100000.0):
        """
        初始化回测引擎
        
        Args:
            strategy: 要回测的策略
            initial_capital: 初始资金，默认为10万
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # 当前持仓
        self.position_history = []  # 持仓历史记录
        self.trade_history = []  # 交易历史记录
        self.equity_curve = []  # 权益曲线
        
        self.data = None  # 回测用的市场数据
        self.file_storage = FileStorage()
        
        logger.info(f"回测引擎初始化，策略: {strategy.name}，初始资金: {initial_capital}")
    
    def load_data_from_file(self, file_path: str) -> bool:
        """
        从文件加载回测数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            bool: 是否成功加载数据
        """
        # 检查文件是否存在
        if not self.file_storage.exists(file_path):
            logger.error(f"回测数据文件不存在: {file_path}")
            return False
        
        # 加载数据
        self.data = self.file_storage.load_data(file_path)
        
        if self.data.empty:
            logger.error("加载的数据为空")
            return False
        
        # 确保数据格式正确
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            logger.error(f"数据缺少必要的列: {', '.join(missing_columns)}")
            return False
        
        # 确保数据是按日期排序的
        if not isinstance(self.data.index, pd.DatetimeIndex):
            logger.warning("数据索引不是日期类型，尝试将第一列转换为日期索引")
            try:
                # 假设第一列是日期
                date_col = self.data.columns[0]
                self.data[date_col] = pd.to_datetime(self.data[date_col])
                self.data.set_index(date_col, inplace=True)
            except Exception as e:
                logger.error(f"转换日期索引失败: {e}")
                return False
        
        # 按日期排序
        self.data = self.data.sort_index()
        
        logger.info(f"成功加载回测数据，共 {len(self.data)} 条记录，日期范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        return True
    
    def run_backtest(self, indicators: bool = True) -> dict:
        """
        运行回测
        
        Args:
            indicators: 是否计算技术指标
            
        Returns:
            dict: 回测结果
        """
        if self.data is None or self.data.empty:
            error_msg = "未加载回测数据，请先使用load_data_from_file方法加载数据"
            logger.error(error_msg)
            return {"error": error_msg}
        
        logger.info(f"开始回测策略: {self.strategy.name}")
        
        # 重置回测状态
        self.current_capital = self.initial_capital
        self.positions = {}
        self.position_history = []
        self.trade_history = []
        self.equity_curve = []
        
        # 计算技术指标
        if indicators:
            self.data = self.strategy.calculate_indicators(self.data)
        
        # 初始化策略
        self.strategy.initialize(self.data)
        
        # 逐日回测
        for i in range(1, len(self.data)):
            current_date = self.data.index[i]
            current_bar = self.data.iloc[i]
            previous_bar = self.data.iloc[i-1]
            
            # 计算当前持仓价值
            portfolio_value = self.calculate_portfolio_value(current_bar)
            
            # 生成交易信号
            signal = self.strategy.generate_signal(i, self.data, self.positions)
            
            # 执行交易
            if signal != 0:  # 0表示不操作
                self.execute_trade(current_date, current_bar, signal)
            
            # 记录权益曲线
            self.equity_curve.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.current_capital,
                'positions': self.positions.copy()
            })
        
        # 计算回测绩效
        performance = self.calculate_performance()
        
        logger.info(f"回测完成，策略: {self.strategy.name}，最终资产: {performance['final_equity']:.2f}")
        
        return {
            'strategy': self.strategy.name,
            'period': f"{self.data.index[0]} 到 {self.data.index[-1]}",
            'performance': performance,
            'equity_curve': pd.DataFrame(self.equity_curve),
            'trades': pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame(),
            'positions': pd.DataFrame(self.position_history) if self.position_history else pd.DataFrame()
        }
    
    def execute_trade(self, date, bar, signal):
        """
        执行交易
        
        Args:
            date: 交易日期
            bar: 当天的价格数据
            signal: 交易信号，正数买入，负数卖出，绝对值为份额
        """
        symbol = bar['symbol'] if 'symbol' in bar else 'default'
        price = bar['close']
        
        if signal > 0:  # 买入
            cost = signal * price
            if cost > self.current_capital:
                logger.warning(f"资金不足，无法买入 {symbol}，需要 {cost:.2f}，现有 {self.current_capital:.2f}")
                return
            
            # 更新持仓
            if symbol in self.positions:
                original_shares = self.positions[symbol]['shares']
                original_cost = self.positions[symbol]['cost']
                
                new_shares = original_shares + signal
                new_cost = original_cost + cost
                
                # 计算新的平均成本
                avg_cost = new_cost / new_shares
                
                self.positions[symbol] = {
                    'shares': new_shares,
                    'cost': new_cost,
                    'avg_price': avg_cost
                }
            else:
                self.positions[symbol] = {
                    'shares': signal,
                    'cost': cost,
                    'avg_price': price
                }
            
            # 更新资金
            self.current_capital -= cost
            
            # 记录交易
            self.trade_history.append({
                'date': date,
                'symbol': symbol,
                'action': 'BUY',
                'shares': signal,
                'price': price,
                'value': cost,
                'capital_after': self.current_capital
            })
            
            logger.debug(f"买入 {signal} 股 {symbol}，价格 {price:.2f}，总成本 {cost:.2f}")
            
        elif signal < 0:  # 卖出
            shares_to_sell = abs(signal)
            
            if symbol not in self.positions:
                logger.warning(f"尝试卖出未持有的股票 {symbol}")
                return
                
            if shares_to_sell > self.positions[symbol]['shares']:
                logger.warning(f"持仓不足，无法卖出 {shares_to_sell} 股 {symbol}，当前持有 {self.positions[symbol]['shares']} 股")
                return
            
            # 计算卖出收入
            revenue = shares_to_sell * price
            
            # 更新持仓
            self.positions[symbol]['shares'] -= shares_to_sell
            self.positions[symbol]['cost'] -= shares_to_sell * self.positions[symbol]['avg_price']
            
            # 如果没有剩余股份，则从持仓中删除
            if self.positions[symbol]['shares'] <= 0:
                del self.positions[symbol]
            
            # 更新资金
            self.current_capital += revenue
            
            # 记录交易
            self.trade_history.append({
                'date': date,
                'symbol': symbol,
                'action': 'SELL',
                'shares': shares_to_sell,
                'price': price,
                'value': revenue,
                'capital_after': self.current_capital
            })
            
            logger.debug(f"卖出 {shares_to_sell} 股 {symbol}，价格 {price:.2f}，总收入 {revenue:.2f}")
        
        # 记录持仓变化
        self.position_history.append({
            'date': date,
            'symbol': symbol,
            'shares': self.positions.get(symbol, {}).get('shares', 0),
            'avg_price': self.positions.get(symbol, {}).get('avg_price', 0),
            'value': self.positions.get(symbol, {}).get('shares', 0) * price,
            'capital': self.current_capital
        })
    
    def calculate_portfolio_value(self, current_bar):
        """
        计算当前投资组合价值
        
        Args:
            current_bar: 当前价格数据
            
        Returns:
            float: 投资组合总价值
        """
        portfolio_value = self.current_capital
        
        # 只有一个股票的情况
        if 'symbol' not in current_bar:
            for symbol, position in self.positions.items():
                portfolio_value += position['shares'] * current_bar['close']
        else:
            # 多个股票的情况
            symbol = current_bar['symbol']
            if symbol in self.positions:
                portfolio_value += self.positions[symbol]['shares'] * current_bar['close']
        
        return portfolio_value
    
    def calculate_performance(self):
        """
        计算回测绩效指标
        
        Returns:
            dict: 绩效指标字典
        """
        if not self.equity_curve:
            return {}
        
        # 创建权益曲线DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # 计算日收益率
        equity_df['daily_return'] = equity_df['portfolio_value'].pct_change()
        
        # 计算各种绩效指标
        total_days = len(equity_df)
        trading_days_per_year = 252
        years = total_days / trading_days_per_year
        
        # 初始和最终权益
        initial_equity = self.initial_capital
        final_equity = equity_df['portfolio_value'].iloc[-1]
        
        # 总收益率
        total_return = (final_equity / initial_equity - 1) * 100
        
        # 年化收益率
        annual_return = ((final_equity / initial_equity) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # 波动率
        daily_std = equity_df['daily_return'].std()
        volatility = daily_std * np.sqrt(trading_days_per_year) * 100
        
        # 夏普比率 (假设无风险利率为0)
        sharpe_ratio = (annual_return / 100) / (volatility / 100) if volatility != 0 else 0
        
        # 最大回撤
        equity_df['cummax'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = (equity_df['portfolio_value'] / equity_df['cummax'] - 1) * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # 交易次数
        num_trades = len(self.trade_history)
        
        # 盈利交易
        if num_trades > 0:
            trades_df = pd.DataFrame(self.trade_history)
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            # 匹配买入和卖出交易来计算盈利
            # 这是一个简化的逻辑，实际上可能需要更复杂的匹配方法
            profitable_trades = 0
            losing_trades = 0
            
            # 简化计算，假设每笔卖出都能对应到之前的买入
            # 实际交易系统中应该使用更精确的计算方法
            if len(sell_trades) > 0:
                for _, sell in sell_trades.iterrows():
                    symbol = sell['symbol']
                    matching_buys = buy_trades[buy_trades['symbol'] == symbol]
                    
                    if len(matching_buys) > 0:
                        avg_buy_price = matching_buys['price'].mean()
                        if sell['price'] > avg_buy_price:
                            profitable_trades += 1
                        else:
                            losing_trades += 1
            
            win_rate = (profitable_trades / num_trades) * 100 if num_trades > 0 else 0
        else:
            profitable_trades = 0
            losing_trades = 0
            win_rate = 0
        
        return {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate
        }
    
    def save_results(self, results, directory: str = None):
        """
        保存回测结果
        
        Args:
            results: 回测结果字典
            directory: 保存目录
            
        Returns:
            dict: 保存的文件路径字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = self.strategy.name.replace(" ", "_")
        prefix = f"{strategy_name}_{timestamp}"
        
        subdir = directory or "backtest_results"
        
        saved_files = {}
        
        # 保存权益曲线
        if 'equity_curve' in results and not results['equity_curve'].empty:
            equity_path = self.file_storage.save_data(
                results['equity_curve'], 
                format="csv",
                filename=f"{prefix}_equity.csv",
                subdir=subdir
            )
            saved_files['equity_curve'] = equity_path
        
        # 保存交易记录
        if 'trades' in results and not results['trades'].empty:
            trades_path = self.file_storage.save_data(
                results['trades'], 
                format="csv",
                filename=f"{prefix}_trades.csv",
                subdir=subdir
            )
            saved_files['trades'] = trades_path
        
        # 保存持仓记录
        if 'positions' in results and not results['positions'].empty:
            positions_path = self.file_storage.save_data(
                results['positions'], 
                format="csv",
                filename=f"{prefix}_positions.csv",
                subdir=subdir
            )
            saved_files['positions'] = positions_path
        
        # 保存绩效指标
        if 'performance' in results:
            perf_df = pd.DataFrame([results['performance']])
            perf_path = self.file_storage.save_data(
                perf_df, 
                format="csv",
                filename=f"{prefix}_performance.csv",
                subdir=subdir
            )
            saved_files['performance'] = perf_path
        
        logger.info(f"回测结果已保存，策略: {self.strategy.name}")
        return saved_files