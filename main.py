# main.py
"""
量化交易系统主程序入口
提供交互式界面选择和运行不同的功能模块
"""
import os
import sys
import time
import argparse
from datetime import datetime
import pandas as pd
# 确保可以导入项目模块
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入IBKR相关模块
from broker.ibkr.ibkr_broker import IBKRBroker
from config.settings import IBKR_CONFIG, RECONNECT_CONFIG
from utils.logger import setup_logger

from data.downloaders.yahoo_downloader import YahooDownloader
from data.processors.cleaner import DataCleaner
from data.processors.transformer import DataTransformer
from data.storage.file_storage import FileStorage

# 设置日志
logger = setup_logger('main')

def print_header():
    """打印程序头部信息"""
    print("\n" + "=" * 60)
    print("量化交易系统 - IBKR API".center(60))
    print("=" * 60)
    print(" 1. 测试IBKR连接")
    print(" 2. 查看账户信息")
    print(" 3. 查看持仓信息")
    print(" 4. 查看订单信息")
    print(" 5. 运行简单策略")
    print(" 6. 下载数据 (或使用 --download 参数)")
    print(" 7. 运行回测 (或使用 --backtest 参数)")
    print(" 0. 退出程序")
    print("=" * 60)

def create_broker(is_paper_account):
    """创建并连接IBKR券商接口"""
    port = IBKR_CONFIG['PAPER_PORT'] if is_paper_account else IBKR_CONFIG['LIVE_PORT']
    broker = IBKRBroker(
        host=IBKR_CONFIG['HOST'],
        port=port,
        client_id=IBKR_CONFIG['CLIENT_ID'],
        is_paper_account=is_paper_account,
        auto_reconnect=RECONNECT_CONFIG['AUTO_RECONNECT'],
        reconnect_interval=RECONNECT_CONFIG['RECONNECT_INTERVAL'],
        max_reconnect_attempts=RECONNECT_CONFIG['MAX_RECONNECT_ATTEMPTS'],
        read_timeout=IBKR_CONFIG['READ_TIMEOUT']
    )
    
    print(f"正在连接到IBKR {'模拟' if is_paper_account else '实盘'}账户...")
    
    if broker.connect():
        print("连接成功!")
        return broker
    else:
        print("连接失败，请检查TWS或IB Gateway是否运行")
        return None

def test_connection(is_paper_account):
    """测试IBKR连接"""
    broker = create_broker(is_paper_account)
    if broker:
        print("连接测试成功!")
        time.sleep(2)  # 等待2秒，让用户看到信息
        broker.disconnect()
        print("已断开连接")

def view_account_info(is_paper_account):
    """查看账户信息"""
    broker = create_broker(is_paper_account)
    if broker:
        try:
            print("\n正在获取账户信息...")
            account_info = broker.get_account_summary()
            
            if account_info:
                print("\n账户摘要:")
                for account_id, account_data in account_info.items():
                    print(f"账户: {account_id}")
                    
                    # 显示关键财务指标
                    key_metrics = ['NetLiquidation', 'AvailableFunds', 'BuyingPower', 
                                  'TotalCashValue', 'GrossPositionValue']
                    
                    for metric in key_metrics:
                        if metric in account_data:
                            value = account_data[metric]['value']
                            currency = account_data[metric]['currency']
                            print(f"  {metric}: {value} {currency}")
                    
                    print("")  # 空行分隔
            else:
                print("未能获取账户信息")
                
            input("按Enter键继续...")
            
        finally:
            broker.disconnect()
            print("已断开连接")

def view_positions(is_paper_account):
    """查看持仓信息"""
    broker = create_broker(is_paper_account)
    if broker:
        try:
            print("\n正在获取持仓信息...")
            positions = broker.get_positions()
            
            if positions:
                print("\n当前持仓:")
                print(f"{'股票代码':<10} {'交易所':<8} {'数量':<10} {'均价':<15} {'市值':<15}")
                print("-" * 60)
                
                for position in positions:
                    symbol = position['symbol']
                    exchange = position['exchange']
                    quantity = position['position']
                    avg_cost = position['avg_cost']
                    
                    print(f"{symbol:<10} {exchange:<8} {quantity:<10} {avg_cost:<15.2f}")
            else:
                print("\n没有持仓")
                
            input("按Enter键继续...")
            
        finally:
            broker.disconnect()
            print("已断开连接")

def view_orders(is_paper_account):
    """查看订单信息"""
    broker = create_broker(is_paper_account)
    if broker:
        try:
            print("\n正在获取订单信息...")
            orders = broker.get_orders()
            
            if orders:
                print("\n当前订单:")
                print(f"{'订单ID':<8} {'股票':<8} {'操作':<5} {'数量':<8} {'类型':<8} {'价格':<10} {'状态':<10} {'已成交':<8} {'剩余':<8}")
                print("-" * 80)
                
                for order in orders:
                    order_id = order['order_id']
                    symbol = order['symbol']
                    action = order['action']
                    quantity = order['quantity']
                    order_type = order['order_type']
                    price = order['limit_price'] if order['limit_price'] else 'N/A'
                    status = order['status']
                    filled = order['filled']
                    remaining = order['remaining']
                    
                    print(f"{order_id:<8} {symbol:<8} {action:<5} {quantity:<8} {order_type:<8}", end="")
                    print(f" {price:<10} {status:<10} {filled:<8} {remaining:<8}")
            else:
                print("\n没有活跃订单")
                
            input("按Enter键继续...")
            
        finally:
            broker.disconnect()
            print("已断开连接")

def run_simple_strategy():
    """运行简单策略"""
    print("\n该功能尚未实现。")
    input("按Enter键继续...")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='量化交易系统')
    parser.add_argument('--paper', action='store_true', help='使用模拟账户')
    
    # 添加数据下载相关参数
    parser.add_argument('--download', action='store_true', help='下载市场数据')
    parser.add_argument('--symbols', type=str, help='股票代码')
    parser.add_argument('--start', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--period', type=str, help='时间段 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--interval', type=str, default='1d', help='时间间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'parquet', 'pickle'], help='保存格式')
    parser.add_argument('--indicators', action='store_true', help='添加技术指标')

    parser.add_argument('--backtest', type=str, help='回测策略名称')
    parser.add_argument('--start-date', type=str, help='回测开始日期，格式 YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, help='回测结束日期，格式 YYYY-MM-DD')
    parser.add_argument('--initial-capital', type=float, default=100000, help='初始资金')
    
    
    return parser.parse_args()

def run_backtest(strategy_name=None, symbol=None, start_date=None, end_date=None, initial_capital=100000, interval=None):
    """
    运行回测功能
    
    Args:
        strategy_name: 策略名称，如 'moving_average'
        symbol: 股票代码，如 'AAPL'
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        initial_capital: 初始资金，默认100000
    """
    print("\n开始回测流程...")
    
    # 获取所有策略文件
    all_strategies = []
    strategies_dir = os.path.join(project_root, 'strategies')
    
    # 递归遍历strategies目录下的所有.py文件
    for root, dirs, files in os.walk(strategies_dir):
        if os.path.basename(root) == '__pycache__' or os.path.basename(root) == '':
            continue
        
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                strategy_file = os.path.splitext(file)[0]
                relative_path = os.path.relpath(root, strategies_dir)
                if relative_path == '.':
                    module_path = f"strategies.{strategy_file}"
                else:
                    module_path = f"strategies.{relative_path}.{strategy_file}"
                all_strategies.append((strategy_file, module_path))
    
    # 如果没有提供策略名称，显示可用策略
    if strategy_name is None:
        print("\n可用策略:")
        for i, (name, _) in enumerate(all_strategies, 1):
            print(f" {i}. {name}")
        
        try:
            idx = int(input(f"\n选择策略 [1-{len(all_strategies)}]: ")) - 1
            if idx < 0 or idx >= len(all_strategies):
                print("无效选择!")
                return
            strategy_name, module_path = all_strategies[idx]
        except ValueError:
            print("请输入有效数字!")
            return
    else:
        # 查找匹配的策略
        matches = [(name, path) for name, path in all_strategies if name == strategy_name]
        if not matches:
            print(f"找不到策略 '{strategy_name}'!")
            print("可用策略:")
            for name, _ in all_strategies:
                print(f" - {name}")
            return
        strategy_name, module_path = matches[0]
    
    # 如果没有提供股票代码，请求输入
    if symbol is None:
        symbol = input("\n输入股票代码 (例如: AAPL): ").strip().upper()
        if not symbol:
            print("必须输入股票代码!")
            return
    
    # 如果没有提供日期范围，请求输入
    if start_date is None:
        start_date = input("\n开始日期 (YYYY-MM-DD): ")
    
    if end_date is None:
        end_date = input("结束日期 (YYYY-MM-DD): ")
    
    print(f"\n正在回测 {strategy_name} 策略，股票：{symbol}，时间段：{start_date} 至 {end_date}...")
    
    # 导入策略模块
    try:
        strategy_module = __import__(module_path, fromlist=["*"])
        
        # 从模块中找到策略类
        from strategies.strategy_base import StrategyBase
        strategy_class = None
        
        for name in dir(strategy_module):
            obj = getattr(strategy_module, name)
            if (isinstance(obj, type) and 
                issubclass(obj, StrategyBase) and 
                obj != StrategyBase):
                strategy_class = obj
                break
        
        if strategy_class is None:
            print(f"在 {strategy_name} 中未找到有效的策略类!")
            return
        
        # 获取策略参数
        import inspect
        sig = inspect.signature(strategy_class.__init__)
        params = {}
        
        for name, param in list(sig.parameters.items())[1:]:  # 跳过self
            if param.default != inspect.Parameter.empty:
                default = param.default
                value = input(f"参数 {name} (默认: {default}): ")
                
                if not value:  # 用户未输入，使用默认值
                    params[name] = default
                else:
                    # 尝试类型转换
                    try:
                        if isinstance(default, int):
                            params[name] = int(value)
                        elif isinstance(default, float):
                            params[name] = float(value)
                        elif isinstance(default, bool):
                            params[name] = value.lower() in ('true', 'yes', 'y', '1')
                        else:
                            params[name] = value
                    except ValueError:
                        print(f"无效值，使用默认值 {default}")
                        params[name] = default
        
        # 创建策略实例
        strategy = strategy_class(**params)
        
        # 创建回测引擎并运行
        from backtest.engine import BacktestEngine
        
        engine = BacktestEngine(
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            interval=interval, 
        )
        
        # 运行回测
        results = engine.run()
        
        # 分析结果
        engine.analyze_results()
        
        # 保存结果
        save = input("\n保存结果? (y/n): ").lower()
        if save == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join(project_root, "backtest", "results")
            os.makedirs(results_dir, exist_ok=True)
            
            filename = f"{strategy_name}_{symbol}_{timestamp}.csv"
            results.to_csv(os.path.join(results_dir, filename))
            print(f"结果已保存到: {os.path.join(results_dir, filename)}")
        
        return results
        
    except ImportError as e:
        print(f"导入模块时出错: {e}")
    except Exception as e:
        print(f"回测过程中出错: {e}")
    
    input("\n按Enter键继续...")
  

def download_market_data(args):
    """下载市场数据的命令行功能"""
    # 检查是否提供了股票代码
    if not args.symbols:
        print("错误：下载数据需要提供股票代码，使用 --symbols 参数")
        return 1
    
    # 解析股票代码
    symbols = [s.strip() for s in args.symbols.split(',')]
    if not symbols:
        print("错误：未提供有效的股票代码")
        return 1
    
    try:
        # 初始化组件
        downloader = YahooDownloader()
        cleaner = DataCleaner()
        transformer = DataTransformer()
        storage = FileStorage()
        
        # 处理日期和周期
        if args.period:
            print(f"使用周期: {args.period}")
            start_date, end_date = None, None
            period = args.period
        else:
            end_date = datetime.now()
            if args.end:
                end_date = datetime.strptime(args.end, "%Y-%m-%d")
            
            start_date = end_date - timedelta(days=365)  # 默认一年
            if args.start:
                start_date = datetime.strptime(args.start, "%Y-%m-%d")
                
            print(f"日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
            period = None
        
        print(f"下载 {len(symbols)} 个股票的数据，间隔: {args.interval}")
        
        # 下载数据
        if period:
            data_dict = downloader.download_historical_data(
                symbols=symbols,
                period=period,
                interval=args.interval,
                adjust_ohlc=True
            )
        else:
            data_dict = downloader.download_historical_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval=args.interval,
                adjust_ohlc=True
            )
        
        if not data_dict:
            print("未能获取数据，请检查股票代码或网络连接")
            return 1
        
        # 合并数据
        all_data = []
        for symbol, df in data_dict.items():
            if not df.empty:
                all_data.append(df)
                print(f"{symbol}: 获取了 {len(df)} 行数据")
        
        if not all_data:
            print("没有获取到有效数据")
            return 1
        
        combined_data = pd.concat(all_data)
        print(f"合并后共 {len(combined_data)} 行数据")
        
        # 数据清洗
        print("清洗数据...")
        combined_data = cleaner.clean_market_data(combined_data)
        
        # 添加技术指标
        if args.indicators:
            print("添加技术指标...")
            combined_data = transformer.add_technical_indicators(combined_data)
        
        # 保存数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols_str = "_".join(symbols)
        if len(symbols_str) > 50:
            symbols_str = symbols_str[:50] + "..."
        filename = f"{symbols_str}_{args.interval}"


        float_cols = combined_data.select_dtypes(include=['float']).columns.tolist()
        for col in float_cols:
            # 去掉所有小数（转为整数）
            # combined_data[col] = combined_data[col].round(0).astype(int)
            
            # 或者保留指定小数位数，例如保留2位小数
            combined_data[col] = combined_data[col].round(2)
                
        file_path = storage.save_data(
            data=combined_data, 
            format=args.format, 
            filename=filename,
            subdir="market_data"
        )
        
        if file_path:
            print(f"数据已成功保存到: {file_path}")
            rows, cols = combined_data.shape
            print(f"共 {rows} 行 x {cols} 列")
            return 0
        else:
            print("保存数据时出错")
            return 1
        
    except Exception as e:
        print(f"下载过程中发生错误: {str(e)}")
        logger.exception("数据下载失败")
        return 1

def download_data_interactive():
    """下载市场数据的交互式功能"""
    os.system('cls' if os.name == 'nt' else 'clear')  # 清屏
    print("\n" + "=" * 60)
    print("下载市场数据".center(60))
    print("=" * 60)
    
    # 收集用户输入
    symbols_input = input("请输入股票代码，多个代码用逗号分隔 (例如: AAPL,MSFT,GOOG): ")
    symbols = [s.strip() for s in symbols_input.split(',')]
    
    # 验证输入
    if not symbols or not symbols[0]:
        print("未提供有效的股票代码，返回主菜单")
        input("\n按Enter键返回主菜单...")
        return
    
    interval_options = {
        "1": "1d",  # 日线
        "2": "1wk", # 周线
        "3": "1mo", # 月线
        "4": "1h",  # 小时线
        "5": "5m",  # 5分钟线
    }
    
    print("\n请选择数据间隔:")
    for k, v in interval_options.items():
        print(f" {k}. {v}")
    
    interval_choice = input("选择 [1-5，默认1]: ").strip() or "1"
    interval = interval_options.get(interval_choice, "1d")
    
    # 时间范围选项
    print("\n请选择时间范围:")
    print(" 1. 最近一周")
    print(" 2. 最近一个月")
    print(" 3. 最近三个月")
    print(" 4. 最近六个月")
    print(" 5. 最近一年")
    print(" 6. 最近三年")
    print(" 7. 最近五年")
    print(" 8. 最长可用历史")
    print(" 9. 自定义日期范围")
    
    range_choice = input("选择 [1-9，默认5]: ").strip() or "5"
    
    end_date = datetime.now()
    start_date = None
    period = None
    
    # 设置日期范围
    if range_choice == "1":
        start_date = end_date - timedelta(days=7)
    elif range_choice == "2":
        start_date = end_date - timedelta(days=30)
    elif range_choice == "3":
        start_date = end_date - timedelta(days=90)
    elif range_choice == "4":
        start_date = end_date - timedelta(days=180)
    elif range_choice == "5":
        start_date = end_date - timedelta(days=365)
    elif range_choice == "6":
        start_date = end_date - timedelta(days=365*3)
    elif range_choice == "7":
        start_date = end_date - timedelta(days=365*5)
    elif range_choice == "8":
        period = "max"
        start_date = None
    elif range_choice == "9":
        # 自定义日期范围
        start_date_str = input("请输入开始日期 (YYYY-MM-DD): ")
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        except ValueError:
            print("日期格式无效，使用默认一年期")
            start_date = end_date - timedelta(days=365)
    else:
        # 默认一年
        start_date = end_date - timedelta(days=365)
    
    # 询问是否添加技术指标
    add_indicators = input("\n是否添加常用技术指标 (y/n，默认y): ").lower().strip() != 'n'
    
    # 询问保存格式
    print("\n请选择保存格式:")
    print(" 1. CSV")
    print(" 2. Parquet")
    print(" 3. Pickle")
    
    format_choice = input("选择 [1-3，默认1]: ").strip() or "1"
    if format_choice == "1":
        save_format = "csv"
    elif format_choice == "2":
        save_format = "parquet"
    elif format_choice == "3":
        save_format = "pickle"
    else:
        save_format = "csv"
    
    # 开始下载
    print(f"\n开始下载 {len(symbols)} 个股票的数据...")
    
    try:
        # 初始化下载器和处理器
        downloader = YahooDownloader()
        cleaner = DataCleaner()
        transformer = DataTransformer()
        storage = FileStorage()
        
        # 下载数据
        if period:
            print(f"使用周期: {period}, 间隔: {interval}")
            data_dict = downloader.download_historical_data(
                symbols=symbols,
                period=period,
                interval=interval,
                adjust_ohlc=True
            )
        else:
            print(f"使用日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}, 间隔: {interval}")
            data_dict = downloader.download_historical_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                adjust_ohlc=True
            )
        
        if not data_dict:
            print("未能获取数据，请检查股票代码或网络连接")
            input("\n按Enter键返回主菜单...")
            return
        
        # 合并所有数据
        print("合并数据...")
        all_data = []
        for symbol, df in data_dict.items():
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            print("没有获取到有效数据")
            input("\n按Enter键返回主菜单...")
            return
        
        combined_data = pd.concat(all_data)
        
        # 数据清洗
        print("清洗数据...")
        cleaned_data = cleaner.clean_market_data(combined_data)
        
        # 添加技术指标
        if add_indicators:
            print("添加技术指标...")
            final_data = transformer.add_technical_indicators(cleaned_data)
        else:
            final_data = cleaned_data
        
        # 保存数据
        print(f"保存数据为{save_format.upper()}格式...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols_str = "_".join(symbols)
        if len(symbols_str) > 50:
            symbols_str = symbols_str[:50] + "..."


        float_cols = combined_data.select_dtypes(include=['float']).columns.tolist()
        for col in float_cols:
            # 去掉所有小数（转为整数）
            # combined_data[col] = combined_data[col].round(0).astype(int)
            
            # 或者保留指定小数位数，例如保留2位小数
            combined_data[col] = combined_data[col].round(2)
        
        filename = f"{symbols_str}_{interval}_{timestamp}"
        file_path = storage.save_data(
            data=final_data, 
            format=save_format, 
            filename=filename,
            subdir="market_data"
        )
        
        if file_path:
            print(f"\n数据已成功保存到: {file_path}")
            rows, cols = final_data.shape
            print(f"共 {rows} 行 x {cols} 列")
        else:
            print("\n保存数据时出错")
        
    except Exception as e:
        print(f"下载过程中发生错误: {str(e)}")
        logger.exception("数据下载失败")
    
    input("\n按Enter键返回主菜单...")

def main():
    """主程序入口"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果指定了下载参数，直接下载并退出
    if args.download:
        return download_market_data(args)
    

    if args.backtest:
        run_backtest(
            strategy_name=args.backtest,
            start_date=args.start_date,
            end_date=args.end_date,
            symbol=args.symbols,
            initial_capital=args.initial_capital,
            interval=args.interval
        )
        return
    
    # 获取是否使用模拟账户的参数
    is_paper_account = args.paper

    # 进入交互式界面
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # 清屏
        print_header()
        
        choice = input("\n请选择操作 [0-6]: ")
        
        if choice == "1":
            test_connection(is_paper_account)
        elif choice == "2":
            view_account_info(is_paper_account)
        elif choice == "3":
            view_positions(is_paper_account)
        elif choice == "4":
            view_orders(is_paper_account)
        elif choice == "5":
            run_simple_strategy()
        elif choice == "6":
            download_data_interactive()
        elif choice == "7":
            print("\n回测模式")
            strategy_name = input("请输入策略名称 (ma_cross/bollinger/rsi/breakout): ")
            start_date = input("请输入开始日期 (YYYY-MM-DD，留空使用默认值): ")
            end_date = input("请输入结束日期 (YYYY-MM-DD，留空使用默认值): ")
            symbols = input("请输入交易品种 (用逗号分隔，留空使用默认值): ")
            initial_capital = input("请输入初始资金 (留空使用默认值100000): ")
            interval = input("请输入回测周期 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max): ")
            
            if not start_date:
                start_date = None
            if not end_date:
                end_date = None
            if not symbols:
                symbols = None
            if not initial_capital:
                initial_capital = 100000
            else:
                initial_capital = float(initial_capital)
            
            run_backtest(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                symbol=symbols,
                initial_capital=initial_capital,
                interval=interval
            )
            
            input("\n按Enter键继续...")
        elif choice == "0":
            print("\n退出程序")
            logger.info("程序正常退出")
            sys.exit(0)
        else:
            print("\n无效选择，请重新输入")
            time.sleep(1)


if __name__ == "__main__":
    try:
        logger.info("程序启动")
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        logger.info("程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n程序发生错误: {str(e)}")
        logger.exception("程序异常退出")
        sys.exit(1)