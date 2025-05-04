# backtest/performance_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

class PerformanceAnalyzer:
    """策略性能分析器，计算各种绩效指标并生成图表"""
    
    def __init__(self):
        """初始化性能分析器"""
        pass
    
    def calculate_performance_metrics(
        self,
        portfolio_values: pd.Series,
        trades: List[Dict[str, Any]],
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        计算绩效指标
        
        Args:
            portfolio_values: 组合价值时间序列
            trades: 交易记录列表
            benchmark_returns: 基准收益率时间序列
            
        Returns:
            包含各种绩效指标的字典
        """
        # 确保是Series类型
        if isinstance(portfolio_values, pd.DataFrame):
            portfolio_values = portfolio_values['portfolio_value']
        
        # 计算每日收益率
        returns = portfolio_values.pct_change().dropna()
        
        # 初始值和最终值
        initial_value = portfolio_values.iloc[0]
        final_value = portfolio_values.iloc[-1]
        
        # 计算总收益率
        total_return = (final_value / initial_value) - 1
        
        # 计算年化收益率
        days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # 计算波动率(年化)
        volatility = returns.std() * np.sqrt(252)
        
        # 计算夏普比率
        risk_free_rate = 0.01  # 假设无风险利率为1%
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        cumulative_max = portfolio_values.cummax()
        drawdown = (portfolio_values / cumulative_max) - 1
        max_drawdown = drawdown.min()
        
        # 计算卡尔玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
        
        # 计算交易统计
        num_trades = len(trades)
        
        # 交易胜率
        if num_trades > 0:
            profitable_trades = sum(1 for t in trades if t['action'] == 'SELL' and 
                                  t['total_received'] > sum(t2['total_cost'] for t2 in trades 
                                                          if t2['action'] == 'BUY' and 
                                                          trades.index(t2) < trades.index(t)))
            win_rate = profitable_trades / num_trades
        else:
            profitable_trades = 0
            win_rate = 0
        
        # 与基准比较
        if benchmark_returns is not None:
            # 对齐日期
            aligned_returns = returns.reindex(benchmark_returns.index, method='ffill')
            
            # 计算超额收益
            excess_returns = aligned_returns - benchmark_returns
            
            # 计算信息比率
            information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            
            # 计算Beta
            covariance = np.cov(aligned_returns, benchmark_returns)[0, 1]
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # 计算Alpha(年化)
            alpha = annual_return - risk_free_rate - beta * (benchmark_returns.mean() * 252 - risk_free_rate)
        else:
            information_ratio = None
            beta = None
            alpha = None
        
        # 返回所有绩效指标
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'num_trades': num_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha
        }
    
    def plot_performance(
        self,
        results: pd.DataFrame,
        trades: List[Dict[str, Any]],
        benchmark: Optional[pd.Series] = None
    ) -> None:
        """
        绘制绩效图表
        
        Args:
            results: 回测结果
            trades: 交易记录
            benchmark: 基准指数
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 绘制组合价值和基准比较
        ax1 = plt.subplot(3, 1, 1)
        
        # 组合价值
        ax1.plot(results.index, results['portfolio_value'], label='Portfolio Value')
        
        # 基准指数(如果有)
        if benchmark is not None:
            # 调整基准比例以便比较
            benchmark_scaled = benchmark / benchmark.iloc[0] * results['portfolio_value'].iloc[0]
            ax1.plot(benchmark.index, benchmark_scaled, label='Benchmark', alpha=0.7)
        
        ax1.set_title('Portfolio Performance')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 绘制回撤
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        
        # 计算回撤
        cumulative_max = results['portfolio_value'].cummax()
        drawdown = (results['portfolio_value'] / cumulative_max) - 1
        
        ax2.fill_between(results.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.grid(True)
        
        # 3. 绘制每日收益分布
        ax3 = plt.subplot(3, 1, 3)
        
        # 计算每日收益率
        daily_returns = results['portfolio_value'].pct_change().dropna()
        
        # 绘制直方图
        ax3.hist(daily_returns, bins=50, alpha=0.7)
        ax3.axvline(daily_returns.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {daily_returns.mean():.2%}')
        ax3.set_title('Daily Returns Distribution')
        ax3.set_xlabel('Return')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 4. 绘制交易
        self._plot_trades(results, trades)
    
    def _plot_trades(self, results: pd.DataFrame, trades: List[Dict[str, Any]]) -> None:
        """
        绘制交易点
        
        Args:
            results: 回测结果
            trades: 交易记录
        """
        plt.figure(figsize=(15, 6))
        
        # 绘制股价
        plt.plot(results.index, results['price'], label='Price')
        
        # 整理交易记录
        buy_dates = [pd.to_datetime(t['date']) for t in trades if t['action'] == 'BUY']
        buy_prices = [t['price'] for t in trades if t['action'] == 'BUY']
        
        sell_dates = [pd.to_datetime(t['date']) for t in trades if t['action'] == 'SELL']
        sell_prices = [t['price'] for t in trades if t['action'] == 'SELL']
        
        # 绘制买卖点
        plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')
        plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell')
        
        plt.title('Trades')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_performance_report(
        self,
        results: pd.DataFrame,
        trades: List[Dict[str, Any]],
        benchmark: Optional[pd.Series] = None
    ) -> None:
        """
        生成性能报告
        
        Args:
            results: 回测结果
            trades: 交易记录
            benchmark: 基准指数
        """
        # 计算绩效指标
        metrics = self.calculate_performance_metrics(
            results['portfolio_value'],
            trades,
            benchmark.pct_change().dropna() if benchmark is not None else None
        )
        
        # 打印报告
        print("\n===================== PERFORMANCE REPORT =====================")
        print(f"Strategy: {results.get('strategy_name', 'Unknown')}")
        print(f"Symbol: {results.get('symbol', 'Unknown')}")
        print(f"Period: {results.index[0].strftime('%Y-%m-%d')} to {results.index[-1].strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${results['portfolio_value'].iloc[0]:.2f}")
        print(f"Final Value: ${results['portfolio_value'].iloc[-1]:.2f}")
        print("-------------------------------------------------------------")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annual Return: {metrics['annual_return']:.2%}")
        print(f"Volatility (Annual): {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        print("-------------------------------------------------------------")
        print(f"Number of Trades: {metrics['num_trades']}")
        print(f"Profitable Trades: {metrics['profitable_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print("-------------------------------------------------------------")
        
        if benchmark is not None:
            print(f"Information Ratio: {metrics['information_ratio']:.4f}")
            print(f"Beta: {metrics['beta']:.4f}")
            print(f"Alpha (Annual): {metrics['alpha']:.2%}")
        
        print("=============================================================")