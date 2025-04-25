# broker/broker_base.py
"""
券商接口基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BrokerBase(ABC):
    """
    所有券商接口的抽象基类，定义共同的接口
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接到券商
        
        返回:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """断开与券商的连接"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查是否已连接到券商
        
        返回:
            bool: 是否已连接
        """
        pass
    
    @abstractmethod
    def get_account_summary(self) -> Dict[str, Any]:
        """
        获取账户摘要信息
        
        返回:
            Dict[str, Any]: 账户摘要信息
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        获取当前持仓信息
        
        返回:
            List[Dict[str, Any]]: 持仓信息列表
        """
        pass
    
    @abstractmethod
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        获取订单信息
        
        返回:
            List[Dict[str, Any]]: 订单信息列表
        """
        pass