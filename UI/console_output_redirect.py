"""
控制台输出重定向
将Python的print输出重定向到右侧面板的控制台区域
"""

import sys
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal


class OutputRedirector(QObject):
    """输出重定向器"""
    
    # 定义信号，用于将输出传递到UI线程
    output_written = pyqtSignal(str)
    
    def __init__(self, console_widget=None):
        super().__init__()
        self.console_widget = console_widget
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # 连接信号到控制台输出
        if self.console_widget:
            self.output_written.connect(self.console_widget.append_output)
    
    def write(self, text: str):
        """重定向写入方法"""
        # 过滤掉空行和只有换行符的输出
        if text.strip():
            # 发送信号到UI线程
            self.output_written.emit(text.rstrip())
        
        # 同时保持原有的输出（可选）
        self.original_stdout.write(text)
    
    def flush(self):
        """刷新方法"""
        if hasattr(self.original_stdout, 'flush'):
            self.original_stdout.flush()
    
    def set_console_widget(self, console_widget):
        """设置控制台组件"""
        self.console_widget = console_widget
        if console_widget:
            self.output_written.connect(console_widget.append_output)
    
    def start_redirect(self):
        """开始重定向"""
        sys.stdout = self
        print("[输出重定向] 开始捕获控制台输出")
    
    def stop_redirect(self):
        """停止重定向"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        print("[输出重定向] 停止捕获控制台输出")


class ConsoleOutputManager:
    """控制台输出管理器"""
    
    def __init__(self):
        self.redirector: Optional[OutputRedirector] = None
        self.is_redirecting = False
    
    def setup_redirection(self, console_widget):
        """设置输出重定向"""
        if self.redirector is None:
            self.redirector = OutputRedirector(console_widget)
        else:
            self.redirector.set_console_widget(console_widget)
        
        self.start_redirection()
    
    def start_redirection(self):
        """开始重定向"""
        if self.redirector and not self.is_redirecting:
            self.redirector.start_redirect()
            self.is_redirecting = True
    
    def stop_redirection(self):
        """停止重定向"""
        if self.redirector and self.is_redirecting:
            self.redirector.stop_redirect()
            self.is_redirecting = False
    
    def cleanup(self):
        """清理资源"""
        self.stop_redirection()


# 全局输出管理器实例
_output_manager = None

def get_output_manager() -> ConsoleOutputManager:
    """获取全局输出管理器"""
    global _output_manager
    if _output_manager is None:
        _output_manager = ConsoleOutputManager()
    return _output_manager
