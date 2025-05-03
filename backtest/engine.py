# backtest/engine.py
"""
回测引擎 - 基于Backtrader的回测系统核心模块
负责执行回测、模拟交易和分析结果
"""
import os
import datetime
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union, Type, Callable

from backtest.data_handler import PandasDataHandler
from backtest.performance_analyzer import PerformanceAnalyzer
from strategies.strategy_base import StrategyBase
from utils.logger import setup_logger

logger = setup_logger('BacktestEngine')

class BacktestEngine:
    """回测引擎类"""
    
    def __init__(
        self,
        data_dir: str = 'data/backtest_data',
        output_dir: str = 'output/backtest_results',
        initial_cash: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0
    ):
        """
        初始化回测引擎
        
        参数:
            data_dir: 回测数据目录
            output_dir: 回测结果输出目录
            initial_cash: 初始资金
            commission: 佣金率 (例如0.001表示0.1%)
            slippage: 滑点 (例如0.001表示0.1%)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化性能分析器
        self.analyzer = PerformanceAnalyzer()
        
        logger.info(f"回测引擎初始化: 初始资金={initial_cash}, 佣金率={commission}, 滑点={slippage}")
    
    def run(
        self,
        strategy_class: Type[bt.Strategy],
        data_handler: Union[PandasDataHandler, List[PandasDataHandler]],
        strategy_params: Optional[Dict[str, Any]] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        timeframe: bt.TimeFrame = bt.TimeFrame.Days,
        plot_results: bool = True,
        analyzers: List[bt.analyzers.Analyzer] = None,
        observer_plots: bool = True
    ) -> Dict[str, Any]:
        """
        运行回测
        
        参数:
            strategy_class: 策略类 (必须是bt.Strategy的子类)
            data_handler: 数据处理器或数据处理器列表
            strategy_params: 策略参数字典
            start_date: 回测开始日期，如果为None则使用数据中最早的日期
            end_date: 回测结束日期，如果为None则使用数据中最晚的日期
            timeframe: 回测时间帧 (默认为日线)
            plot_results: 是否绘制回测结果图表
            analyzers: 要添加的分析器列表
            observer_plots: 是否包含观察器绘图
            
        返回:
            回测结果字典，包含绩效指标和交易记录
        """
        # 创建cerebro引擎
        cerebro = bt.Cerebro()
        
        # 添加策略
        if strategy_params:
            cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            cerebro.addstrategy(strategy_class)
        
        # 添加数据
        if isinstance(data_handler, list):
            for dh in data_handler:
                data = dh.get_backtrader_data()
                cerebro.adddata(data)
        else:
            data = data_handler.get_backtrader_data()
            cerebro.adddata(data)
        
        # 设置初始资金
        cerebro.broker.setcash(self.initial_cash)
        
        # 设置佣金
        cerebro.broker.setcommission(commission=self.commission)
        
        # 设置滑点
        cerebro.broker.set_slippage_perc(self.slippage)
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        
        # 添加自定义分析器
        if analyzers:
            for analyzer in analyzers:
                cerebro.addanalyzer(analyzer)
        
        # 设置交易日志记录器
        cerebro.addwriter(bt.WriterFile, csv=True, out=os.path.join(self.output_dir, 'trades.csv'))
        
        # 运行回测
        logger.info(f"开始回测 - 策略: {strategy_class.__name__}")
        results = cerebro.run()
        strategy_instance = results[0]
        
        # 获取回测结果和绩效指标
        backtest_results = self._extract_results(strategy_instance)
        
        # 打印绩效摘要
        self._print_performance_summary(backtest_results)
        
        # 绘制结果
        if plot_results:
            cerebro.plot(style='candle', barup='green', bardown='red', 
                         volup='green', voldown='red', 
                         grid=True, plotdist=0.1, 
                         start=start_date, end=end_date, 
                         volume=True, observer_plot=observer_plots)
        
        return backtest_results
    
    def compare_strategies(
        self,
        strategy_classes: List[Type[bt.Strategy]],
        data_handler: Union[PandasDataHandler, List[PandasDataHandler]],
        strategy_params_list: Optional[List[Dict[str, Any]]] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        plot_comparison: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        比较多个策略的回测结果
        
        参数:
            strategy_classes: 策略类列表
            data_handler: 数据处理器或数据处理器列表
            strategy_params_list: 策略参数字典列表，与strategy_classes对应
            start_date: 回测开始日期
            end_date: 回测结束日期
            plot_comparison: 是否绘制比较图表
            
        返回:
            策略比较结果字典，键为策略名称，值为回测结果
        """
        comparison_results = {}
        strategy_names = []
        equity_curves = []
        
        # 确保strategy_params_list长度与strategy_classes匹配
        if strategy_params_list is None:
            strategy_params_list = [None] * len(strategy_classes)
        assert len(strategy_params_list) == len(strategy_classes), "策略参数列表长度与策略类列表不匹配"
        
        # 运行每个策略
        for i, strategy_class in enumerate(strategy_classes):
            strategy_params = strategy_params_list[i]
            strategy_name = strategy_class.__name__
            
            # 运行回测但不绘图
            results = self.run(
                strategy_class=strategy_class,
                data_handler=data_handler,
                strategy_params=strategy_params,
                start_date=start_date,
                end_date=end_date,
                plot_results=False
            )
            
            # 存储结果
            comparison_results[strategy_name] = results
            strategy_names.append(strategy_name)
            
            # 提取权益曲线
            equity_curves.append({
                'name': strategy_name,
                'curve': results['equity_curve']
            })
        
        # 绘制策略比较图表
        if plot_comparison and equity_curves:
            self._plot_strategy_comparison(equity_curves, start_date, end_date)
        
        # 创建比较表格
        comparison_table = self._create_comparison_table(comparison_results)
        
        # 保存比较表格
        output_path = os.path.join(self.output_dir, 'strategy_comparison.csv')
        comparison_table.to_csv(output_path)
        logger.info(f"策略比较表已保存至: {output_path}")
        
        return comparison_results
    
    def _extract_results(self, strategy) -> Dict[str, Any]:
        """从回测策略实例中提取结果"""
        # 初始化结果字典
        results = {}
        
        # 提取基本结果
        results['final_value'] = strategy.broker.getvalue()
        results['pnl'] = strategy.broker.getvalue() - self.initial_cash
        results['pnl_percent'] = (results['pnl'] / self.initial_cash) * 100
        
        # 提取分析器结果
        # - Sharpe比率
        sharpe_ratio = strategy.analyzers.sharpe.get_analysis()
        results['sharpe_ratio'] = sharpe_ratio['sharperatio']
        
        # - 最大回撤
        drawdown = strategy.analyzers.drawdown.get_analysis()
        results['max_drawdown_percent'] = drawdown['max']['drawdown']
        results['max_drawdown_money'] = drawdown['max']['moneydown']
        
        # - 收益率
        returns = strategy.analyzers.returns.get_analysis()
        results['annual_return'] = returns['rnorm100']
        results['avg_return'] = returns['ravg']
        
        # - 交易分析
        trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()
        
        # 总交易统计
        if trade_analysis.get('total'):
            results['total_trades'] = trade_analysis['total']['total']
            results['total_won'] = trade_analysis['won']['total'] if 'won' in trade_analysis else 0
            results['total_lost'] = trade_analysis['lost']['total'] if 'lost' in trade_analysis else 0
            
            # 计算胜率
            if results['total_trades'] > 0:
                results['win_rate'] = (results['total_won'] / results['total_trades']) * 100
            else:
                results['win_rate'] = 0.0
            
            # 计算盈亏比
            if 'won' in trade_analysis and trade_analysis['won']['total'] > 0 and 'lost' in trade_analysis and trade_analysis['lost']['total'] > 0:
                avg_won = trade_analysis['won']['pnl']['average']
                avg_lost = abs(trade_analysis['lost']['pnl']['average'])
                if avg_lost > 0:
                    results['profit_factor'] = avg_won / avg_lost
                else:
                    results['profit_factor'] = float('inf')
            else:
                results['profit_factor'] = 0.0
        else:
            results['total_trades'] = 0
            results['total_won'] = 0
            results['total_lost'] = 0
            results['win_rate'] = 0.0
            results['profit_factor'] = 0.0
        
        # SQN质量分数
        sqn = strategy.analyzers.sqn.get_analysis()
        results['sqn_score'] = sqn['sqn']
        
        # 提取权益曲线
        # 注意：这可能需要在策略中自定义记录
        if hasattr(strategy, 'equity_curve'):
            results['equity_curve'] = strategy.equity_curve
        else:
            # 如果策略没有提供权益曲线，创建一个简单的基于值的曲线
            results['equity_curve'] = pd.Series(
                [v[0] for v in strategy._value], 
                index=[d.datetime() for d in strategy.data.datetime]
            )
        
        # 提取交易记录
        if hasattr(strategy, 'trade_history'):
            results['trade_history'] = strategy.trade_history
        
        return results
    
    def _print_performance_summary(self, results: Dict[str, Any]) -> None:
        """打印回测绩效摘要"""
        print("\n" + "=" * 60)
        print(" 回测绩效摘要 ".center(58))
        print("=" * 60)
        
        print(f"初始资金:       {self.initial_cash:,.2f}")
        print(f"最终资金:       {results['final_value']:,.2f}")
        print(f"净盈亏:         {results['pnl']:,.2f} ({results['pnl_percent']:.2f}%)")
        print(f"年化收益率:     {results['annual_return']:.2f}%")
        print(f"Sharpe比率:     {results['sharpe_ratio']:.4f}")
        print(f"最大回撤:       {results['max_drawdown_percent']:.2f}%")
        print(f"最大回撤金额:   {results['max_drawdown_money']:,.2f}")
        print(f"交易次数:       {results['total_trades']}")
        print(f"胜率:           {results['win_rate']:.2f}%")
        print(f"盈亏比:         {results['profit_factor']:.2f}")
        print(f"SQN质量分数:    {results['sqn_score']:.2f}")
        
        print("=" * 60)
    
    def _plot_strategy_comparison(
        self, 
        equity_curves: List[Dict[str, Any]], 
        start_date: Optional[datetime.datetime], 
        end_date: Optional[datetime.datetime]
    ) -> None:
        """绘制策略比较图表"""
        plt.figure(figsize=(12, 8))
        
        # 绘制每个策略的权益曲线
        for curve_data in equity_curves:
            curve = curve_data['curve']
            name = curve_data['name']
            
            # 如果设置了日期范围，进行筛选
            if start_date is not None or end_date is not None:
                mask = pd.Series(True, index=curve.index)
                if start_date is not None:
                    mask = mask & (curve.index >= start_date)
                if end_date is not None:
                    mask = mask & (curve.index <= end_date)
                curve = curve[mask]
            
            # 将曲线标准化为从100开始
            norm_curve = 100 * curve / curve.iloc[0]
            plt.plot(norm_curve.index, norm_curve.values, label=name)
        
        # 设置图表标题和标签
        plt.title('策略收益率比较', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('投资组合价值 (基准=100)', fontsize=12)
        plt.grid(True)
        plt.legend()
        
        # 保存图表
        output_path = os.path.join(self.output_dir, 'strategy_comparison.png')
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"策略比较图表已保存至: {output_path}")
    
    def _create_comparison_table(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """创建策略比较表格"""
        # 定义要包含的指标
        metrics = [
            'pnl', 'pnl_percent', 'annual_return', 'sharpe_ratio',
            'max_drawdown_percent', 'total_trades', 'win_rate',
            'profit_factor', 'sqn_score'
        ]
        
        # 重命名指标以便更好地显示
        metric_names = {
            'pnl': '净盈亏',
            'pnl_percent': '收益率(%)',
            'annual_return': '年化收益率(%)',
            'sharpe_ratio': 'Sharpe比率',
            'max_drawdown_percent': '最大回撤(%)',
            'total_trades': '交易次数',
            'win_rate': '胜率(%)',
            'profit_factor': '盈亏比',
            'sqn_score': 'SQN得分'
        }
        
        # 构建比较表格
        comparison_data = {}
        for strategy_name, result in results.items():
            comparison_data[strategy_name] = {metric_names[m]: result[m] for m in metrics}
            
        return pd.DataFrame(comparison_data)
    
    def optimize_strategy(
        self,
        strategy_class: Type[bt.Strategy],
        data_handler: Union[PandasDataHandler, List[PandasDataHandler]],
        param_grid: Dict[str, List[Any]],
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        metric: str = 'sqn_score',  # 用于优化的指标
        maximize: bool = True  # 是否最大化指标
    ) -> Dict[str, Any]:
        """
        优化策略参数
        
        参数:
            strategy_class: 策略类
            data_handler: 数据处理器
            param_grid: 参数网格，字典形式，键为参数名，值为参数值列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            metric: 用于优化的指标名称
            maximize: 是否最大化指标值
            
        返回:
            最优参数组合和对应的回测结果
        """
        # 生成参数组合
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        # 存储所有回测结果
        all_results = []
        
        logger.info(f"开始优化策略 - 策略: {strategy_class.__name__}, 参数组合数: {len(param_combinations)}")
        
        # 遍历所有参数组合
        for i, combo in enumerate(param_combinations):
            params = {param_names[j]: combo[j] for j in range(len(param_names))}
            
            logger.info(f"正在测试参数组合 {i+1}/{len(param_combinations)}: {params}")
            
            # 运行回测
            results = self.run(
                strategy_class=strategy_class,
                data_handler=data_handler,
                strategy_params=params,
                start_date=start_date,
                end_date=end_date,
                plot_results=False
            )
            
            # 记录参数和结果
            results['params'] = params
            all_results.append(results)
        
        # 选择最优结果
        if maximize:
            best_result = max(all_results, key=lambda x: x[metric])
        else:
            best_result = min(all_results, key=lambda x: x[metric])
        
        logger.info(f"参数优化完成 - 最优参数: {best_result['params']}")
        
        # 创建优化结果表格
        optimization_results = pd.DataFrame([
            {**{'params': str(r['params'])}, **{k: r[k] for k in r if k != 'params' and k != 'equity_curve' and k != 'trade_history'}}
            for r in all_results
        ])
        
        # 保存优化结果
        output_path = os.path.join(self.output_dir, 'optimization_results.csv')
        optimization_results.to_csv(output_path)
        logger.info(f"优化结果已保存至: {output_path}")
        
        # 用最优参数重新运行一次，并生成图表
        final_results = self.run(
            strategy_class=strategy_class,
            data_handler=data_handler,
            strategy_params=best_result['params'],
            start_date=start_date,
            end_date=end_date,
            plot_results=True
        )
        
        return {
            'best_params': best_result['params'],
            'best_result': best_result,
            'all_results': all_results,
            'optimization_table': optimization_results
        }