"""
应用程序入口文件
启动Satellite Component Visualization & Physics Field Prediction应用
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from main_window import MainWindow
from ui_utils import setup_application_font


# 🔧 修复显示协议兼容性
def setup_display_environment():
    """设置显示环境以确保最佳兼容性"""
    # 优先使用Wayland原生支持（现代Linux环境）
    if 'WAYLAND_DISPLAY' in os.environ:
        # 让Qt自动选择最佳平台，通常是wayland
        print("[显示] 检测到Wayland环境，使用原生Wayland支持")
        # 只有在Wayland出现问题时才回退到X11
        # os.environ['QT_QPA_PLATFORM'] = 'wayland'
    
    # 设置Qt缩放策略
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    os.environ['QT_SCALE_FACTOR'] = '1'
    
    # 添加Qt错误处理
    os.environ['QT_LOGGING_RULES'] = 'qt.qpa.wayland.debug=false'


def main():
    """主函数，启动应用程序"""
    # 🔧 首先设置显示环境
    setup_display_environment()
    
    # 🔧 智能平台选择和错误处理
    app = None
    platforms_to_try = ['auto', 'wayland', 'xcb', 'minimal']
    
    for platform in platforms_to_try:
        try:
            if platform != 'auto':
                os.environ['QT_QPA_PLATFORM'] = platform
                print(f"[显示] 尝试使用 {platform} 平台")
            else:
                # 让Qt自动选择
                if 'QT_QPA_PLATFORM' in os.environ:
                    del os.environ['QT_QPA_PLATFORM']
                print("[显示] 让Qt自动选择平台")
            
            app = QApplication(sys.argv)
            print(f"[显示] ✅ 成功使用 {platform} 平台")
            break
            
        except Exception as e:
            print(f"[显示] ❌ {platform} 平台失败: {e}")
            if app:
                app.quit()
                app = None
            continue
    
    if app is None:
        print("[显示] ❌ 所有平台都失败，无法启动应用")
        sys.exit(1)
    
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