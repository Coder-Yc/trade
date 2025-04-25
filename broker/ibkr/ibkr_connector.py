"""
Interactive Brokers (IBKR) 连接器
负责与IBKR服务器通信的底层连接管理
"""
from ib_insync import *
import datetime
import threading
import time
from typing import Optional, Dict, List, Any

from config.settings import IBKR_CONFIG
from utils.logger import setup_logger

class IBKRConnector:
    def __init__(self, 
                 host: str = None, 
                 port: int = None,
                 client_id: int = None,
                 is_paper_account: bool = None,
                 auto_reconnect: bool = None,
                 reconnect_interval: int = None,
                 max_reconnect_attempts: int = None,
                 read_timeout: int = None):
        """
        初始化IBKR连接器
        
        参数:
            host: TWS或IB Gateway的主机地址
            port: 连接端口 (7497模拟账户, 7496实盘账户)
            client_id: 客户端ID
            is_paper_account: 是否使用模拟账户
            auto_reconnect: 是否自动重连
            reconnect_interval: 重连间隔(秒)
            max_reconnect_attempts: 最大重连尝试次数
            read_timeout: 读取超时时间(秒)
        """
        self.host = host or IBKR_CONFIG['host']
        self.port = port or IBKR_CONFIG['port']
        self.client_id = client_id or IBKR_CONFIG['client_id']
        self.is_paper_account = is_paper_account if is_paper_account is not None else IBKR_CONFIG['is_paper_account']
        self.auto_reconnect = auto_reconnect if auto_reconnect is not None else IBKR_CONFIG['auto_reconnect']
        self.reconnect_interval = reconnect_interval or IBKR_CONFIG['reconnect_interval']
        self.max_reconnect_attempts = max_reconnect_attempts or IBKR_CONFIG['max_reconnect_attempts']
        self.read_timeout = read_timeout or IBKR_CONFIG['read_timeout']
        
        self.ib = IB()
        
        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error
        
        self.reconnect_attempt = 0
        self.last_error_time = None
        self.last_error_code = None
        self.reconnect_thread = None
        self._shutdown_requested = False
        
        self.logger = setup_logger('IBKRConnector')
        
        self.market_data_subscriptions = {}
        self.order_status = {}
        
    def _on_connected(self):
        """连接成功回调"""
        self.reconnect_attempt = 0
        account_type = "模拟" if self.is_paper_account else "实盘"
        self.logger.info(f"成功连接到IBKR {account_type}账户 (客户端ID: {self.client_id})")
        
    def _on_disconnected(self):
        """断开连接回调"""
        self.logger.warning("与IBKR的连接已断开")
        
        if self.auto_reconnect and not self._shutdown_requested:
            self._schedule_reconnect()
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """错误回调"""
        self.last_error_time = datetime.datetime.now()
        self.last_error_code = errorCode
        
        if errorCode >= 2000 and errorCode < 3000:  # 警告
            self.logger.warning(f"IBKR警告 [{errorCode}]: {errorString}, reqId={reqId}")
        elif errorCode >= 1000:  # 系统错误
            self.logger.error(f"IBKR系统错误 [{errorCode}]: {errorString}, reqId={reqId}")
        elif errorCode in [1100, 1101, 1102]:  # 连接相关
            self.logger.critical(f"IBKR连接错误 [{errorCode}]: {errorString}")
        else:  # 一般错误
            self.logger.error(f"IBKR错误 [{errorCode}]: {errorString}, reqId={reqId}")
    
    def _schedule_reconnect(self):
        """安排重连"""
        if self.reconnect_attempt >= self.max_reconnect_attempts:
            self.logger.critical(f"达到最大重连尝试次数({self.max_reconnect_attempts})，停止重连")
            return
        
        # 增加重连计数
        self.reconnect_attempt += 1
        
        # 使用指数退避策略计算等待时间
        wait_time = min(300, self.reconnect_interval * (2 ** (self.reconnect_attempt - 1)))
        
        self.logger.info(f"计划在{wait_time}秒后进行第{self.reconnect_attempt}次重连")
        
        # 启动重连线程
        if self.reconnect_thread and self.reconnect_thread.is_alive():
            return
            
        self.reconnect_thread = threading.Thread(target=self._reconnect_worker, args=(wait_time,))
        self.reconnect_thread.daemon = True
        self.reconnect_thread.start()
    
    def _reconnect_worker(self, wait_time):
        """重连工作线程"""
        time.sleep(wait_time)
        
        if self._shutdown_requested:
            return
            
        self.logger.info(f"尝试重新连接到IBKR (尝试 {self.reconnect_attempt}/{self.max_reconnect_attempts})")
        self.connect()
    
    def connect(self) -> bool:
        """
        连接到IBKR
        
        返回:
            bool: 连接是否成功
        """
        # 如果已连接，先断开
        if self.ib.isConnected():
            self.logger.info("已经连接到IBKR，断开现有连接")
            self.ib.disconnect()
        
        try:
            # 尝试连接
            self.logger.info(f"连接到IBKR {self.host}:{self.port} (客户端ID: {self.client_id})")
            self.ib.connect(
                self.host, 
                self.port, 
                clientId=self.client_id, 
                readonly=False,
                timeout=self.read_timeout
            )
            
            # 验证连接状态
            if self.ib.isConnected():
                # 初始化连接后的操作
                self._post_connection_setup()
                return True
            else:
                self.logger.error("连接失败：未接收到连接确认")
                return False
                
        except Exception as e:
            self.logger.exception(f"连接IBKR时发生异常: {str(e)}")
            return False
    
    def _post_connection_setup(self):
        """连接后的初始化设置"""
        try:
            self.ib.reqAccountSummary()
            
            # 2. 请求当前时间同步
            try:
                server_time = self.ib.reqCurrentTime()
                local_time = datetime.datetime.now()
                time_diff = abs((server_time - local_time).total_seconds())
                
                if time_diff > 5:  # 如果时间差超过5秒
                    self.logger.warning(f"本地时间与服务器时间差异较大: {time_diff:.2f}秒")
            except Exception as e:
                self.logger.warning(f"获取服务器时间失败: {str(e)}")
            
            # 3. 恢复之前的市场数据订阅（如果有）
            self._resubscribe_market_data()
            
            self.logger.info("连接后初始化完成")
            
        except Exception as e:
            self.logger.exception(f"连接后初始化失败: {str(e)}")

    def _resubscribe_market_data(self):
        """重新订阅之前的市场数据"""
        if not self.market_data_subscriptions:
            return
            
        self.logger.info(f"重新订阅 {len(self.market_data_subscriptions)} 个市场数据")
        
        for contract_id, contract_data in list(self.market_data_subscriptions.items()):
            try:
                contract = contract_data.get('contract')
                if contract:
                    self.logger.info(f"重新订阅: {contract.symbol}")
                    self.ib.reqMktData(contract)
            except Exception as e:
                self.logger.error(f"重新订阅市场数据失败: {str(e)}")
    
    def disconnect(self):
        """安全断开与IBKR的连接"""
        self._shutdown_requested = True
        
        if self.ib.isConnected():
            self.logger.info("断开与IBKR的连接")
            
            # 取消所有实时数据订阅
            for contract_id in list(self.market_data_subscriptions.keys()):
                try:
                    contract = self.market_data_subscriptions[contract_id].get('contract')
                    if contract:
                        self.ib.cancelMktData(contract)
                except Exception as e:
                    self.logger.error(f"取消市场数据订阅失败: {str(e)}")
            
            # 断开连接
            self.ib.disconnect()
    
    def is_connected(self) -> bool:
        """
        检查是否已连接到IBKR
        
        返回:
            bool: 是否已连接
        """
        return self.ib.isConnected()
    
    def ensure_connected(self) -> bool:
        """
        确保已连接到IBKR，如果未连接则尝试连接
        
        返回:
            bool: 是否已连接
        """
        if not self.is_connected():
            self.logger.warning("未连接到IBKR，尝试连接")
            return self.connect()
        return True
    
    def get_ib(self):
        """
        获取IB实例，供其他模块使用
        
        返回:
            IB: ib_insync的IB对象
        """
        return self.ib
    
    def run(self):
        """启动事件循环处理"""
        if not self.is_connected():
            self.connect()
        
        try:
            self.ib.run()
        except (KeyboardInterrupt, SystemExit):
            self.logger.info("接收到退出信号，正在关闭...")
            self.disconnect()
        except Exception as e:
            self.logger.exception(f"运行时发生错误: {str(e)}")
            if self.is_connected():
                self.disconnect()