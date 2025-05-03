# backtest/performance_analyzer.py
"""
回测绩效分析器 - 分析回测结果，计算各种绩效指标，生成报告
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

from utils.logger import setup_logger

logger = setup_logger('PerformanceAnalyzer')

class PerformanceAnalyzer:
    """回测绩效分析器类"""
    
    def __init__(self, output_dir: str = 'output/backtest_results'):
        """
        初始化绩效分析器
        
        参数:
            output_dir: 结果输出目录
        """
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"绩效分析器初始化: 输出目录={output_dir}")
    
    def analyze(
        self, 
        equity_curve: pd.Series,
        trade_history: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
        initial_capital: float = 100000.0,
        output_prefix: str = 'backtest'
    ) -> Dict[str, Any]:
        """
        分析回测结果
        
        参数:
            equity_curve: 权益曲线，索引为日期
            trade_history: 交易历史记录 DataFrame
            benchmark: 基准收益曲线
            risk_free_rate: 无风险利率（年化，小数形式）
            initial_capital: 初始资金
            output_prefix: 输出文件名前缀
            
        返回:
            包含绩效指标的字典
        """
        logger.info("开始分析回测结果")
        
        # 计算绩效指标
        metrics = self._calculate_metrics(equity_curve, benchmark, risk_free_rate, initial_capital)
        
        # 分析交易历史
        if trade_history is not None and not trade_history.empty:
            trade_metrics = self._analyze_trades(trade_history)
            metrics.update(trade_metrics)
        
        # 生成报告
        self._generate_report(metrics, equity_curve, benchmark, trade_history, output_prefix)
        
        return metrics
    
    def _calculate_metrics(
        self, 
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """计算绩效指标"""
        # 准备数据
        equity = equity_curve.copy()
        
        # 确保索引为日期时间类型并排序
        if not isinstance(equity.index, pd.DatetimeIndex):
            equity.index = pd.to_datetime(equity.index)
        equity = equity.sort_index()
        
        # 如果有基准，确保基准也是日期时间索引并排序
        if benchmark is not None:
            if not isinstance(benchmark.index, pd.DatetimeIndex):
                benchmark.index = pd.to_datetime(benchmark.index)
            benchmark = benchmark.sort_index()
            
            # 将基准调整为与权益曲线相同的日期范围
            benchmark = benchmark[benchmark.index >= equity.index[0]]
            benchmark = benchmark[benchmark.index <= equity.index[-1]]
            
            # 重新索引确保日期完全匹配
            benchmark = benchmark.reindex(equity.index, method='ffill')
        
        # 计算每日回报率
        daily_returns = equity.pct_change().fillna(0)
        
        # 计算累计回报率
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        
        # 计算总回报率
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        
        # 计算年化回报率
        days = (equity.index[-1] - equity.index[0]).days
        annualized_return = (1 + total_return) ** (365.0 / days) - 1
        
        # 计算波动率（标准差）
        volatility = daily_returns.std() * np.sqrt(252)  # 年化波动率
        
        # 计算最大回撤
        cumulative_max = equity.cummax()
        drawdown = (equity - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()
        
        # 计算夏普比率
        excess_returns = daily_returns - risk_free_rate / 252  # 日化无风险利率
        sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
        
        # 计算卡玛比率（Calmar ratio）- 年化回报与最大回撤的比率
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # 计算索提诺比率（Sortino ratio）- 只考虑下行波动率
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (daily_returns.mean() * 252) / downside_deviation if len(downside_returns) > 0 else np.inf
        
        # 如果有基准，计算相对指标
        if benchmark is not None:
            # 计算基准日回报率
            benchmark_daily_returns = benchmark.pct_change().fillna(0)
            
            # 计算基准累计回报率
            benchmark_cumulative_returns = (1 + benchmark_daily_returns).cumprod() - 1
            
            # 计算基准总回报率
            benchmark_total_return = benchmark.iloc[-1] / benchmark.iloc[0] - 1
            
            # 计算Alpha和Beta
            covariance = np.cov(daily_returns[1:], benchmark_daily_returns[1:])
            beta = covariance[0, 1] / covariance[1, 1] if covariance[1, 1] != 0 else 0
            alpha = annualized_return - risk_free_rate - beta * (benchmark_total_return - risk_free_rate)
            
            # 计算信息比率
            tracking_error = (daily_returns - benchmark_daily_returns).std() * np.sqrt(252)
            information_ratio = (annualized_return - benchmark_total_return) / tracking_error if tracking_error != 0 else 0
            
            # 计算捕获比率（上涨/下跌市场）
            up_market = benchmark_daily_returns > 0
            down_market = benchmark_daily_returns < 0
            
            up_capture = (daily_returns[up_market].mean() / benchmark_daily_returns[up_market].mean()) if up_market.any() and benchmark_daily_returns[up_market].mean() != 0 else np.nan
            down_capture = (daily_returns[down_market].mean() / benchmark_daily_returns[down_market].mean()) if down_market.any() and benchmark_daily_returns[down_market].mean() != 0 else np.nan
        
        # 创建结果字典
        metrics = {
            # 基本回报指标
            'initial_capital': initial_capital,
            'final_value': equity.iloc[-1],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            
            # 风险指标
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'max_drawdown_date': max_drawdown_date,
            
            # 风险调整后回报指标
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # 其他统计
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative_returns
        }
        
        # 如果有基准，添加相对指标
        if benchmark is not None:
            metrics.update({
                'benchmark_total_return': benchmark_total_return,
                'benchmark_total_return_pct': benchmark_total_return * 100,
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'up_capture_ratio': up_capture,
                'down_capture_ratio': down_capture,
                'benchmark_daily_returns': benchmark_daily_returns,
                'benchmark_cumulative_returns': benchmark_cumulative_returns
            })
        
        return metrics
    
    def _analyze_trades(self, trade_history: pd.DataFrame) -> Dict[str, Any]:
        """分析交易历史"""
        # 确保至少有必要的列
        required_cols = ['entry_date', 'exit_date', 'entry_price', 'exit_price', 'size', 'pnl', 'type']
        missing_cols = [col for col in required_cols if col not in trade_history.columns]
        if missing_cols:
            logger.warning(f"交易历史缺少必要的列: {missing_cols}")
            return {}
        
        # 计算交易统计
        total_trades = len(trade_history)
        winning_trades = trade_history[trade_history['pnl'] > 0]
        losing_trades = trade_history[trade_history['pnl'] <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # 计算盈亏比
        avg_win = winning_trades['pnl'].mean() if win_count > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if loss_count > 0 else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # 计算平均持有期
        if 'entry_date' in trade_history.columns and 'exit_date' in trade_history.columns:
            trade_history['holding_period'] = (pd.to_datetime(trade_history['exit_date']) - 
                                              pd.to_datetime(trade_history['entry_date'])).dt.days
            avg_holding_period = trade_history['holding_period'].mean()
        else:
            avg_holding_period = None
        
        # 统计多空交易
        if 'type' in trade_history.columns:
            long_trades = trade_history[trade_history['type'] == 'long']
            short_trades = trade_history[trade_history['type'] == 'short']
            
            long_count = len(long_trades)
            short_count = len(short_trades)
            
            long_win_rate = len(long_trades[long_trades['pnl'] > 0]) / long_count if long_count > 0 else 0
            short_win_rate = len(short_trades[short_trades['pnl'] > 0]) / short_count if short_count > 0 else 0
        else:
            long_count = short_count = 0
            long_win_rate = short_win_rate = 0
        
        # 返回交易统计
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding_period,
            'long_trades': long_count,
            'short_trades': short_count,
            'long_win_rate': long_win_rate,
            'long_win_rate_pct': long_win_rate * 100,
            'short_win_rate': short_win_rate,
            'short_win_rate_pct': short_win_rate * 100
        }
    
    def _generate_report(
        self,
        metrics: Dict[str, Any],
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        trade_history: Optional[pd.DataFrame] = None,
        output_prefix: str = 'backtest'
    ) -> None:
        """生成绩效报告"""
        # 绘制权益曲线图
        self._plot_equity_curve(equity_curve, benchmark, metrics, output_prefix)
        
        # 绘制回撤图
        self._plot_drawdown(equity_curve, output_prefix)
        
        # 绘制月度回报热图
        self._plot_monthly_returns_heatmap(equity_curve, output_prefix)
        
        # 如果有交易历史，绘制交易分析图
        if trade_history is not None and not trade_history.empty:
            self._plot_trade_analysis(trade_history, output_prefix)
        
        # 创建绩效指标表格
        metrics_table = self._create_metrics_table(metrics)
        
        # 保存绩效指标表格
        metrics_file = os.path.join(self.output_dir, f"{output_prefix}_metrics.csv")
        metrics_table.to_csv(metrics_file)
        logger.info(f"绩效指标已保存至: {metrics_file}")
        
        # 生成HTML报告
        self._generate_html_report(metrics, equity_curve, benchmark, trade_history, output_prefix)
    
    def _plot_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        metrics: Optional[Dict[str, Any]] = None,
        output_prefix: str = 'backtest'
    ) -> None:
        """绘制权益曲线图"""
        plt.figure(figsize=(12, 8))
        
        # 绘制策略权益曲线
        equity_norm = equity_curve / equity_curve.iloc[0] * 100
        plt.plot(equity_norm.index, equity_norm, label='策略', linewidth=2)
        
        # 如果有基准，绘制基准曲线
        if benchmark is not None:
            benchmark_norm = benchmark / benchmark.iloc[0] * 100
            plt.plot(benchmark_norm.index, benchmark_norm, label='基准', linewidth=1.5, alpha=0.7)
        
        # 添加图表标题和标签
        plt.title('投资组合权益曲线', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('价值 (基准=100)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 如果有绩效指标，添加到图表
        if metrics is not None:
            annualized_return = metrics.get('annualized_return_pct', 0)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            max_drawdown = metrics.get('max_drawdown_pct', 0)
            
            info_text = (f"年化收益率: {annualized_return:.2f}%\n"
                         f"Sharpe比率: {sharpe_ratio:.2f}\n"
                         f"最大回撤: {max_drawdown:.2f}%")
            
            # 在图表右上角添加文本框
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, f"{output_prefix}_equity_curve.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"权益曲线图已保存至: {output_path}")
    
    def _plot_drawdown(
        self,
        equity_curve: pd.Series,
        output_prefix: str = 'backtest'
    ) -> None:
        """绘制回撤图"""
        plt.figure(figsize=(12, 6))
        
        # 计算回撤
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max * 100
        
        # 绘制回撤曲线
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        plt.plot(drawdown.index, drawdown, color='red', linewidth=1)
        
        # 添加图表标题和标签
        plt.title('回撤分析', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('回撤 (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 找出最大回撤点
        max_drawdown_idx = drawdown.idxmin()
        max_drawdown = drawdown.min()
        
        # 在最大回撤点添加标记
        plt.scatter(max_drawdown_idx, max_drawdown, color='darkred', s=50)
        plt.annotate(f"{max_drawdown:.2f}%", (max_drawdown_idx, max_drawdown),
                    xytext=(15, -15), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        # 保存图表
        output_path = os.path.join(self.output_dir, f"{output_prefix}_drawdown.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"回撤图已保存至: {output_path}")
    
    def _plot_monthly_returns_heatmap(
        self,
        equity_curve: pd.Series,
        output_prefix: str = 'backtest'
    ) -> None:
        """绘制月度回报热图"""
        # 计算每日回报率
        daily_returns = equity_curve.pct_change().fillna(0)
        
        # 按年月重采样得到月度回报
        monthly_returns = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1) * 100
        
        # 转换为年份-月份二维表，年份为行，月份为列
        monthly_returns_table = pd.DataFrame()
        for (year, month), value in monthly_returns.items():
            if year not in monthly_returns_table.index:
                monthly_returns_table.loc[year] = [np.nan] * 12
            monthly_returns_table.loc[year, month-1] = value
        
        # 设置月份列标签
        monthly_returns_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # 创建热图
        plt.figure(figsize=(12, 8))
        
        # 使用seaborn绘制热图（如果可用）
        try:
            import seaborn as sns
            ax = sns.heatmap(monthly_returns_table, annot=True, fmt=".2f", cmap="RdYlGn",
                            center=0, linewidths=1, cbar_kws={"shrink": .75})
        except ImportError:
            # 使用matplotlib绘制热图
            ax = plt.gca()
            im = ax.imshow(monthly_returns_table.values, cmap="RdYlGn", aspect='auto')
            
            # 在单元格中添加文本
            for i in range(len(monthly_returns_table.index)):
                for j in range(len(monthly_returns_table.columns)):
                    value = monthly_returns_table.iloc[i, j]
                    if not np.isnan(value):
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                               color="black" if abs(value) < 10 else "white")
            
            # 添加颜色条
            plt.colorbar(im)
        
        # 设置坐标轴标签
        plt.title('月度回报热图 (%)', fontsize=14)
        ax.set_xticklabels(monthly_returns_table.columns, rotation=0)
        ax.set_yticklabels(monthly_returns_table.index, rotation=0)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, f"{output_prefix}_monthly_returns.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"月度回报热图已保存至: {output_path}")
    
    def _plot_trade_analysis(
        self,
        trade_history: pd.DataFrame,
        output_prefix: str = 'backtest'
    ) -> None:
        """绘制交易分析图"""
        # 确保至少有必要的列
        if 'pnl' not in trade_history.columns:
            logger.warning("交易历史缺少pnl列，无法绘制交易分析图")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 盈亏分布直方图
        axes[0, 0].hist(trade_history['pnl'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].axvline(0, color='red', linestyle='--')
        axes[0, 0].set_title('交易盈亏分布', fontsize=12)
        axes[0, 0].set_xlabel('盈亏', fontsize=10)
        axes[0, 0].set_ylabel('频率', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 累计盈亏走势
        if 'exit_date' in trade_history.columns:
            # 确保日期列是日期时间类型
            trade_history['exit_date'] = pd.to_datetime(trade_history['exit_date'])
            
            # 按日期排序
            sorted_trades = trade_history.sort_values('exit_date')
            
            # 计算累计盈亏
            sorted_trades['cumulative_pnl'] = sorted_trades['pnl'].cumsum()
            
            # 绘制累计盈亏走势
            axes[0, 1].plot(sorted_trades['exit_date'], sorted_trades['cumulative_pnl'],
                           color='green', linewidth=2)
            axes[0, 1].set_title('累计盈亏走势', fontsize=12)
            axes[0, 1].set_xlabel('日期', fontsize=10)
            axes[0, 1].set_ylabel('累计盈亏', fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 设置日期格式
            fig.autofmt_xdate()
        else:
            # 如果没有日期列，就按交易序号绘制
            cumulative_pnl = trade_history['pnl'].cumsum()
            axes[0, 1].plot(cumulative_pnl.index, cumulative_pnl.values, color='green', linewidth=2)
            axes[0, 1].set_title('累计盈亏走势', fontsize=12)
            axes[0, 1].set_xlabel('交易序号', fontsize=10)
            axes[0, 1].set_ylabel('累计盈亏', fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 多空分析（如果有类型列）
        if 'type' in trade_history.columns:
            # 创建多空分组
            grouped = trade_history.groupby('type')
            
            # 计算多空盈亏统计
            stats = {}
            for name, group in grouped:
                stats[name] = {
                    'count': len(group),
                    'win_count': len(group[group['pnl'] > 0]),
                    'loss_count': len(group[group['pnl'] <= 0]),
                    'total_pnl': group['pnl'].sum(),
                    'avg_pnl': group['pnl'].mean()
                }
                
                if stats[name]['count'] > 0:
                    stats[name]['win_rate'] = stats[name]['win_count'] / stats[name]['count']
                else:
                    stats[name]['win_rate'] = 0
            
            # 绘制多空胜率对比
            if 'long' in stats and 'short' in stats:
                labels = ['做多', '做空']
                win_rates = [stats['long']['win_rate'] * 100, stats['short']['win_rate'] * 100]
                
                axes[1, 0].bar(labels, win_rates, color=['blue', 'red'], alpha=0.7)
                axes[1, 0].set_title('多空胜率对比', fontsize=12)
                axes[1, 0].set_ylabel('胜率 (%)', fontsize=10)
                axes[1, 0].grid(True, alpha=0.3)
                
                # 在柱状图上添加数值标签
                for i, v in enumerate(win_rates):
                    axes[1, 0].text(i, v + 1, f"{v:.1f}%", ha='center')
                
                # 绘制多空平均盈亏对比
                avg_pnls = [stats['long']['avg_pnl'], stats['short']['avg_pnl']]
                
                axes[1, 1].bar(labels, avg_pnls, color=['blue', 'red'], alpha=0.7)
                axes[1, 1].set_title('多空平均盈亏对比', fontsize=12)
                axes[1, 1].set_ylabel('平均盈亏', fontsize=10)
                axes[1, 1].grid(True, alpha=0.3)
                
                # 在柱状图上添加数值标签
                for i, v in enumerate(avg_pnls):
                    axes[1, 1].text(i, v + (0.1 * abs(v) if v != 0 else 0.1), f"{v:.2f}", ha='center')
        else:
            # 如果没有类型列，绘制其他统计
            # 4. 连续盈亏分析
            trade_history['win'] = trade_history['pnl'] > 0
            
            # 计算连续盈亏
            consecutive_wins = []
            consecutive_losses = []
            current_streak = 1
            
            for i in range(1, len(trade_history)):
                if trade_history['win'].iloc[i] == trade_history['win'].iloc[i-1]:
                    current_streak += 1
                else:
                    if trade_history['win'].iloc[i-1]:
                        consecutive_wins.append(current_streak)
                    else:
                        consecutive_losses.append(current_streak)
                    current_streak = 1
            
            # 添加最后一个连续序列
            if len(trade_history) > 0:
                if trade_history['win'].iloc[-1]:
                    consecutive_wins.append(current_streak)
                else:
                    consecutive_losses.append(current_streak)
            
            # 绘制连续盈亏统计
            axes[1, 0].hist(consecutive_wins, bins=range(1, max(consecutive_wins + [1]) + 1),
                           alpha=0.7, color='green', label='连续盈利')
            axes[1, 0].hist(consecutive_losses, bins=range(1, max(consecutive_losses + [1]) + 1),
                           alpha=0.7, color='red', label='连续亏损')
            axes[1, 0].set_title('连续盈亏分析', fontsize=12)
            axes[1, 0].set_xlabel('连续交易次数', fontsize=10)
            axes[1, 0].set_ylabel('频率', fontsize=10)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 如果有持有期列，绘制持有期分析
            if 'holding_period' in trade_history.columns:
                axes[1, 1].scatter(trade_history['holding_period'], trade_history['pnl'], 
                                 alpha=0.7, color='blue')
                axes[1, 1].set_title('持有期vs盈亏', fontsize=12)
                axes[1, 1].set_xlabel('持有天数', fontsize=10)
                axes[1, 1].set_ylabel('盈亏', fontsize=10)
                axes[1, 1].grid(True, alpha=0.3)
                
                # 添加趋势线
                if len(trade_history) > 1:
                    try:
                        from scipy import stats
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            trade_history['holding_period'], trade_history['pnl'])
                        
                        x = np.array([min(trade_history['holding_period']), max(trade_history['holding_period'])])
                        y = slope * x + intercept
                        axes[1, 1].plot(x, y, 'r--', label=f'趋势线 (r={r_value:.2f})')
                        axes[1, 1].legend()
                    except ImportError:
                        pass
            else:
                # 如果没有持有期列，绘制交易大小vs盈亏
                if 'size' in trade_history.columns:
                    axes[1, 1].scatter(trade_history['size'], trade_history['pnl'], 
                                     alpha=0.7, color='purple')
                    axes[1, 1].set_title('交易大小vs盈亏', fontsize=12)
                    axes[1, 1].set_xlabel('交易大小', fontsize=10)
                    axes[1, 1].set_ylabel('盈亏', fontsize=10)
                    axes[1, 1].grid(True, alpha=0.3)
        
        # 整体布局
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, f"{output_prefix}_trade_analysis.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"交易分析图已保存至: {output_path}")
    
    def _create_metrics_table(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        """创建绩效指标表格"""
        # 筛选要显示的指标
        display_metrics = {}
        
        # 基本绩效指标
        if 'initial_capital' in metrics:
            display_metrics['初始资金'] = metrics['initial_capital']
        if 'final_value' in metrics:
            display_metrics['最终资金'] = metrics['final_value']
        if 'total_return_pct' in metrics:
            display_metrics['总收益率(%)'] = f"{metrics['total_return_pct']:.2f}"
        if 'annualized_return_pct' in metrics:
            display_metrics['年化收益率(%)'] = f"{metrics['annualized_return_pct']:.2f}"
        
        # 风险指标
        if 'volatility_pct' in metrics:
            display_metrics['波动率(%)'] = f"{metrics['volatility_pct']:.2f}"
        if 'max_drawdown_pct' in metrics:
            display_metrics['最大回撤(%)'] = f"{metrics['max_drawdown_pct']:.2f}"
        
        # 风险调整后回报指标
        if 'sharpe_ratio' in metrics:
            display_metrics['夏普比率'] = f"{metrics['sharpe_ratio']:.4f}"
        if 'sortino_ratio' in metrics:
            display_metrics['索提诺比率'] = f"{metrics['sortino_ratio']:.4f}"
        if 'calmar_ratio' in metrics:
            display_metrics['卡玛比率'] = f"{metrics['calmar_ratio']:.4f}"
        
        # 相对于基准的指标
        if 'benchmark_total_return_pct' in metrics:
            display_metrics['基准总收益率(%)'] = f"{metrics['benchmark_total_return_pct']:.2f}"
        if 'alpha' in metrics:
            display_metrics['Alpha'] = f"{metrics['alpha']:.4f}"
        if 'beta' in metrics:
            display_metrics['Beta'] = f"{metrics['beta']:.4f}"
        if 'information_ratio' in metrics:
            display_metrics['信息比率'] = f"{metrics['information_ratio']:.4f}"
        
        # 交易统计
        if 'total_trades' in metrics:
            display_metrics['总交易次数'] = metrics['total_trades']
        if 'winning_trades' in metrics and 'losing_trades' in metrics:
            display_metrics['盈利交易'] = metrics['winning_trades']
            display_metrics['亏损交易'] = metrics['losing_trades']
        if 'win_rate_pct' in metrics:
            display_metrics['胜率(%)'] = f"{metrics['win_rate_pct']:.2f}"
        if 'profit_factor' in metrics:
            display_metrics['盈亏比'] = f"{metrics['profit_factor']:.2f}"
        if 'avg_holding_period' in metrics and metrics['avg_holding_period'] is not None:
            display_metrics['平均持有天数'] = f"{metrics['avg_holding_period']:.1f}"
        
        # 创建DataFrame
        df = pd.DataFrame(list(display_metrics.items()), columns=['指标', '值'])
        
        return df
    
    def _generate_html_report(
        self,
        metrics: Dict[str, Any],
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        trade_history: Optional[pd.DataFrame] = None,
        output_prefix: str = 'backtest'
    ) -> None:
        """生成HTML回测报告"""
        try:
            import jinja2
            import base64
            from io import BytesIO
            
            # 准备图表
            fig_paths = {
                'equity_curve': os.path.join(self.output_dir, f"{output_prefix}_equity_curve.png"),
                'drawdown': os.path.join(self.output_dir, f"{output_prefix}_drawdown.png"),
                'monthly_returns': os.path.join(self.output_dir, f"{output_prefix}_monthly_returns.png")
            }
            
            if trade_history is not None and not trade_history.empty:
                fig_paths['trade_analysis'] = os.path.join(self.output_dir, f"{output_prefix}_trade_analysis.png")
            
            # 准备指标表格HTML
            metrics_table = self._create_metrics_table(metrics)
            metrics_html = metrics_table.to_html(index=False, classes='table table-striped')
            
            # 准备月度回报表格（如果有）
            monthly_returns_html = ""
            if 'daily_returns' in metrics:
                daily_returns = metrics['daily_returns']
                
                # 按年月重采样得到月度回报
                monthly_returns = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month]).apply(
                    lambda x: (1 + x).prod() - 1) * 100
                
                # 转换为年份-月份二维表
                monthly_returns_table = pd.DataFrame()
                for (year, month), value in monthly_returns.items():
                    if year not in monthly_returns_table.index:
                        monthly_returns_table.loc[year] = [np.nan] * 12
                    monthly_returns_table.loc[year, month-1] = value
                
                # 设置月份列标签
                monthly_returns_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # 添加年度回报行
                yearly_returns = daily_returns.groupby(daily_returns.index.year).apply(
                    lambda x: (1 + x).prod() - 1) * 100
                monthly_returns_table.loc['Year'] = [np.nan] * 12
                
                for year in monthly_returns_table.index[:-1]:
                    if year in yearly_returns.index:
                        monthly_returns_table.loc['Year', -1] = yearly_returns[year]
                
                # 对月度回报表格进行样式处理
                def style_background(val):
                    if pd.isna(val):
                        return ''
                    elif val > 0:
                        return f'background-color: rgba(0, 128, 0, {min(abs(val) / 10, 0.8)})'
                    else:
                        return f'background-color: rgba(255, 0, 0, {min(abs(val) / 10, 0.8)})'
                
                styled_table = monthly_returns_table.style.format("{:.2f}").applymap(style_background)
                monthly_returns_html = styled_table.to_html()
            
            # 准备交易历史表格（如果有）
            trades_html = ""
            if trade_history is not None and not trade_history.empty:
                # 选择要显示的列
                display_cols = ['entry_date', 'exit_date', 'symbol', 'type', 'size', 
                             'entry_price', 'exit_price', 'pnl']
                display_cols = [col for col in display_cols if col in trade_history.columns]
                
                # 重命名列
                col_rename = {
                    'entry_date': '进场日期',
                    'exit_date': '出场日期',
                    'symbol': '股票代码',
                    'type': '方向',
                    'size': '数量',
                    'entry_price': '进场价格',
                    'exit_price': '出场价格',
                    'pnl': '盈亏'
                }
                
                # 创建显示表格
                display_df = trade_history[display_cols].copy()
                display_df.columns = [col_rename.get(col, col) for col in display_df.columns]
                
                # 对盈亏列进行样式处理
                def style_pnl(df):
                    return pd.DataFrame(['color: green' if x > 0 else 'color: red' for x in df['盈亏']],
                                      index=df.index, columns=['盈亏'])
                
                styled_trades = display_df.style.apply(style_pnl, axis=None)
                trades_html = styled_trades.to_html()
            
            # 读取HTML模板
            template_str = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>回测报告</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        color: #333;
                    }
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                    }
                    .header {
                        text-align: center;
                        margin-bottom: 30px;
                    }
                    .section {
                        margin-bottom: 40px;
                    }
                    h1, h2, h3 {
                        color: #2c3e50;
                    }
                    .table {
                        width: 100%;
                        border-collapse: collapse;
                        margin-bottom: 20px;
                    }
                    .table th, .table td {
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }
                    .table th {
                        background-color: #f2f2f2;
                        color: #333;
                    }
                    .table tr:nth-child(even) {
                        background-color: #f9f9f9;
                    }
                    .img-fluid {
                        max-width: 100%;
                        height: auto;
                    }
                    .col-2 {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 20px;
                    }
                    .col-2 > div {
                        flex: 1 1 calc(50% - 20px);
                        min-width: 300px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>回测绩效报告</h1>
                        <p>回测期间: {{ start_date }} 至 {{ end_date }}</p>
                    </div>
                    
                    <div class="section">
                        <h2>绩效摘要</h2>
                        {{ metrics_html|safe }}
                    </div>
                    
                    <div class="section">
                        <h2>权益曲线</h2>
                        <img src="{{ equity_curve_img }}" class="img-fluid">
                    </div>
                    
                    <div class="section col-2">
                        <div>
                            <h2>回撤分析</h2>
                            <img src="{{ drawdown_img }}" class="img-fluid">
                        </div>
                        <div>
                            <h2>月度回报热图</h2>
                            <img src="{{ monthly_returns_img }}" class="img-fluid">
                        </div>
                    </div>
                    
                    {% if monthly_returns_html %}
                    <div class="section">
                        <h2>月度回报表</h2>
                        {{ monthly_returns_html|safe }}
                    </div>
                    {% endif %}
                    
                    {% if trade_analysis_img %}
                    <div class="section">
                        <h2>交易分析</h2>
                        <img src="{{ trade_analysis_img }}" class="img-fluid">
                    </div>
                    {% endif %}
                    
                    {% if trades_html %}
                    <div class="section">
                        <h2>交易记录</h2>
                        {{ trades_html|safe }}
                    </div>
                    {% endif %}
                    
                    <div class="footer">
                        <p>生成时间: {{ generation_time }}</p>
                    </div>
                </div>
            </body>
            </html>
            '''
            
            # 准备模板变量
            template_vars = {
                'start_date': equity_curve.index[0].strftime('%Y-%m-%d'),
                'end_date': equity_curve.index[-1].strftime('%Y-%m-%d'),
                'metrics_html': metrics_html,
                'equity_curve_img': fig_paths['equity_curve'],
                'drawdown_img': fig_paths['drawdown'],
                'monthly_returns_img': fig_paths['monthly_returns'],
                'monthly_returns_html': monthly_returns_html,
                'trades_html': trades_html,
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if 'trade_analysis' in fig_paths:
                template_vars['trade_analysis_img'] = fig_paths['trade_analysis']
            
            # 渲染模板
            template = jinja2.Template(template_str)
            html_report = template.render(**template_vars)
            
            # 保存HTML报告
            output_path = os.path.join(self.output_dir, f"{output_prefix}_report.html")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
                
            logger.info(f"HTML报告已保存至: {output_path}")
            
        except ImportError as e:
            logger.warning(f"无法生成HTML报告: {e}")
            logger.info("提示: 安装jinja2以支持HTML报告生成")