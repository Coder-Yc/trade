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

# 确保可以导入项目模块
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入IBKR相关模块
from broker.ibkr.ibkr_broker import IBKRBroker
from config.settings import IBKR_CONFIG, RECONNECT_CONFIG
from utils.logger import setup_logger

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
    
    return parser.parse_args()

def main():
    """主程序入口"""
    # 解析命令行参数
    args = parse_arguments()
    is_paper_account = args.paper
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # 清屏
        print_header()
        
        choice = input("\n请选择操作 [0-5]: ")
        
        if choice == "1":
            test_connection(is_paper_account)
        elif choice == "2":
            view_account_info(is_paper_account)
        elif choice == "3":
            view_positions(is_paper_account)
        elif choice == "4":
            view_orders(is_paper_account)
        elif choice == "5":
            run_simple_strategy(is_paper_account)
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