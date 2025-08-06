"""
应用程序入口文件
启动Satellite Component Visualization & Physics Field Prediction应用
"""

import sys
from PyQt6.QtWidgets import QApplication
from main_window import MainWindow
from ui_utils import setup_application_font


def main():
    """主函数，启动应用程序"""
    app = QApplication(sys.argv)
    
    # 设置应用程序字体
    font = setup_application_font()
    app.setFont(font)
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main()