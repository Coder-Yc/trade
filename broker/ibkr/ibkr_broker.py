# broker/ibkr/ibkr_broker.py
"""
基于IBKR连接器的券商接口实现
"""
from typing import List, Dict, Any, Optional
from ib_insync import Contract, Order, Trade

from broker.broker_base import BrokerBase
from broker.ibkr.ibkr_connector import IBKRConnector
from utils.logger import setup_logger

class IBKRBroker(BrokerBase):
    """Interactive Brokers券商接口实现"""
    
    def __init__(self, connector: Optional[IBKRConnector] = None, **kwargs):
        """
        初始化IBKR券商接口
        
        参数:
            connector: 可选的IBKR连接器实例，如果不提供则创建新的实例
            **kwargs: 传递给IBKRConnector的参数
        """
        self.logger = setup_logger('IBKRBroker')
        self.connector = connector or IBKRConnector(**kwargs)
        self.account_id = None  # 将在连接后获取
        
    def connect(self) -> bool:
        """
        连接到IBKR
        
        返回:
            bool: 连接是否成功
        """
        success = self.connector.connect()
        if success:
            # 获取主账户ID
            self._get_account_id()
        return success
    
    def _get_account_id(self):
        """获取主账户ID"""
        accounts = self.connector.get_ib().managedAccounts()
        if accounts:
            self.account_id = accounts[0]
            self.logger.info(f"使用账户ID: {self.account_id}")
        else:
            self.logger.warning("无法获取账户ID")
    
    def disconnect(self) -> None:
        """断开与IBKR的连接"""
        self.connector.disconnect()
    
    def is_connected(self) -> bool:
        """
        检查是否已连接到IBKR
        
        返回:
            bool: 是否已连接
        """
        return self.connector.is_connected()
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        获取账户摘要信息
        
        返回:
            Dict[str, Any]: 账户摘要信息
        """
        if not self.connector.ensure_connected():
            return {}
            
        try:
            # 直接使用 accountSummary 方法获取账户信息
            # 这是一个同步调用，但应该不会无限期阻塞
            raw_summary = self.connector.get_ib().accountSummary()
            
            # 转换为更易用的字典格式
            summary = {}
            for item in raw_summary:
                # 检查 item 是否具有正确的属性
                if hasattr(item, 'account') and hasattr(item, 'tag') and hasattr(item, 'value') and hasattr(item, 'currency'):
                    if item.account not in summary:
                        summary[item.account] = {}
                    summary[item.account][item.tag] = {
                        'value': item.value,
                        'currency': item.currency
                    }
                else:
                    # 记录意外的数据格式
                    self.logger.warning(f"收到意外的账户数据格式: {type(item)} - {item}")
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"获取账户摘要失败: {str(e)}")
            return {}
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        获取当前持仓信息
        
        返回:
            List[Dict[str, Any]]: 持仓信息列表
        """
        if not self.connector.ensure_connected():
            return []
            
        try:
            # 获取原始持仓信息
            raw_positions = self.connector.get_ib().positions()
            
            # 转换为更易用的字典格式
            positions = []
            for pos in raw_positions:
                positions.append({
                    'account': pos.account,
                    'symbol': pos.contract.symbol,
                    'exchange': pos.contract.exchange,
                    'currency': pos.contract.currency,
                    'position': pos.position,
                    'avg_cost': pos.avgCost,
                    'contract': pos.contract  # 保留原始合约对象以便使用
                })
            
            return positions
            
        except Exception as e:
            self.logger.exception(f"获取持仓信息失败: {str(e)}")
            return []
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        获取订单信息
        
        返回:
            List[Dict[str, Any]]: 订单信息列表
        """
        if not self.connector.ensure_connected():
            return []
            
        try:
            # 获取原始订单信息
            raw_trades = self.connector.get_ib().openTrades()
            
            # 转换为更易用的字典格式
            orders = []
            for trade in raw_trades:
                orders.append({
                    'order_id': trade.order.orderId,
                    'symbol': trade.contract.symbol,
                    'exchange': trade.contract.exchange,
                    'action': trade.order.action,  # 'BUY' 或 'SELL'
                    'quantity': trade.order.totalQuantity,
                    'order_type': trade.order.orderType,  # 'LMT', 'MKT' 等
                    'limit_price': trade.order.lmtPrice if hasattr(trade.order, 'lmtPrice') else None,
                    'status': trade.orderStatus.status,
                    'filled': trade.orderStatus.filled,
                    'remaining': trade.orderStatus.remaining,
                    'avg_fill_price': trade.orderStatus.avgFillPrice,
                    'contract': trade.contract,  # 保留原始合约对象
                    'order': trade.order,        # 保留原始订单对象
                    'order_status': trade.orderStatus  # 保留原始状态对象
                })
            
            return orders
            
        except Exception as e:
            self.logger.exception(f"获取订单信息失败: {str(e)}")
            return []
    
    def place_order(self, contract: Contract, order: Order) -> Optional[Trade]:
        """
        下单
        
        参数:
            contract: IBKR合约对象
            order: IBKR订单对象
            
        返回:
            Optional[Trade]: 成功返回Trade对象，失败返回None
        """
        if not self.connector.ensure_connected():
            return None
            
        try:
            self.logger.info(f"下单: {order.action} {order.totalQuantity} {contract.symbol} @ {order.orderType}")
            trade = self.connector.get_ib().placeOrder(contract, order)
            return trade
        except Exception as e:
            self.logger.exception(f"下单失败: {str(e)}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """
        取消订单
        
        参数:
            order_id: 订单ID
            
        返回:
            bool: 是否成功发送取消请求
        """
        if not self.connector.ensure_connected():
            return False
            
        try:
            self.logger.info(f"取消订单: {order_id}")
            self.connector.get_ib().cancelOrder(order_id)
            return True
        except Exception as e:
            self.logger.exception(f"取消订单失败: {str(e)}")
            return False