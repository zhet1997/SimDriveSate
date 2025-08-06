"""
UI常量和样式定义
包含所有UI组件使用的常量、样式表和配置参数
"""

# 应用配置常量
SCENE_SCALE = 500  # 每米500像素
DEFAULT_LAYOUT_SIZE = (1.0, 1.0)  # 默认布局尺寸（米）
DEFAULT_THERMAL_CONDUCTIVITY = 100.0  # 默认热导率
DEFAULT_MESH_RESOLUTION = 50  # 默认网格分辨率
GRID_INTERVAL_METERS = 0.1  # 网格间隔（米）

# 组件尺寸限制
MIN_COMPONENT_SIZE = 10  # 最小组件尺寸（像素）
POWER_RANGE = (0.0, 1000.0)  # 功率范围（瓦特）
DEFAULT_POWER_RANGE = (5.0, 50.0)  # 默认随机功率范围

# 字体配置
DEFAULT_FONT_FAMILY = "Arial, sans-serif"
DEFAULT_FONT_SIZE = 10
COMPONENT_FONT_SIZE = 10
UI_FONT_SIZE = 11

# 颜色定义
class Colors:
    # 组件颜色
    RECT_BORDER = (0, 0, 255)  # 蓝色
    RECT_FILL = (173, 216, 230, 150)  # 淡蓝色填充
    RECT_SELECTED = (100, 100, 255, 150)  # 选中蓝色
    
    CIRCLE_BORDER = (255, 0, 0)  # 红色
    CIRCLE_FILL = (255, 182, 193, 150)  # 浅红色填充
    CIRCLE_SELECTED = (255, 100, 100, 150)  # 选中红色
    
    CAPSULE_BORDER = (0, 128, 0)  # 绿色
    CAPSULE_FILL = (144, 238, 144, 150)  # 浅绿色填充
    CAPSULE_SELECTED = (100, 255, 100, 150)  # 选中绿色
    
    # UI颜色
    GRID_LINE = (200, 200, 200)  # 网格线颜色
    BORDER_LINE = (0, 0, 0)  # 边界线颜色
    PREVIEW_LINE = "red"  # 预览线颜色
    TEXT_COLOR = (0, 0, 0)  # 文本颜色

# 样式表定义
class StyleSheets:
    # 绘制模式按钮样式
    DRAW_BUTTON_BASE = """
        QPushButton {{
            padding: 15px;
            font-size: 12px;
            border: 2px solid {border_color};
            border-radius: 8px;
            background-color: #ecf0f1;
            margin: 5px;
        }}
        QPushButton:checked {{
            background-color: {border_color};
            color: white;
        }}
        QPushButton:hover {{
            background-color: #bdc3c7;
        }}
    """
    
    RECT_BUTTON = DRAW_BUTTON_BASE.format(border_color="#3498db")
    CIRCLE_BUTTON = DRAW_BUTTON_BASE.format(border_color="#e74c3c")
    CAPSULE_BUTTON = DRAW_BUTTON_BASE.format(border_color="#27ae60")
    
    # SDF控制样式
    SDF_CHECKBOX = """
        QCheckBox {
            font-size: 11px;
            margin: 5px;
            padding: 5px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
    """
    
    SDF_UPDATE_BUTTON = """
        QPushButton {
            padding: 8px;
            font-size: 11px;
            border: 2px solid #9b59b6;
            border-radius: 6px;
            background-color: #ecf0f1;
            margin: 5px;
        }
        QPushButton:hover {
            background-color: #9b59b6;
            color: white;
        }
    """
    
    # 组件列表样式
    SCROLL_AREA = """
        QScrollArea {
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            background-color: #ffffff;
            margin: 5px;
        }
    """
    
    COMPONENT_EDITOR_FRAME = """
        QFrame {
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            background-color: #f8f9fa;
            margin: 2px;
            padding: 5px;
        }
    """
    
    POWER_SPINBOX = """
        QDoubleSpinBox {
            font-size: 10px;
            padding: 2px;
            border: 1px solid #bdc3c7;
            border-radius: 3px;
        }
    """
    
    # 分隔线样式
    SEPARATOR = "margin: 15px 5px;"
    
    # 标签样式
    SECTION_LABEL = "font-weight: bold; font-size: 14px; margin: 10px;"
    COMPONENT_TITLE = "font-weight: bold; font-size: 11px; color: #2c3e50;"
    COMPONENT_INFO = "font-size: 10px; color: #7f8c8d;"
    POWER_LABEL = "font-size: 10px; font-weight: bold;"
    NO_COMPONENTS_LABEL = "color: #7f8c8d; font-style: italic; padding: 20px;"

# SDF配置
class SDFConfig:
    GRID_MULTIPLIER = 5  # SDF网格分辨率倍数
    COLORMAP = 'coolwarm'  # SDF颜色映射
    ALPHA = 0.6  # SDF透明度
    INTERPOLATION = 'nearest'  # 插值方法
    DPI = 100  # 图像DPI
    Z_VALUE = -1  # SDF背景层级

# 绘制模式图标
class Icons:
    DRAW_MODES = {
        'rect': '🔲',
        'circle': '⭕', 
        'capsule': '🏷️'
    }
    
    # 工具栏图标
    LOAD_FILE = '📁'
    SAVE_FILE = '💾'
    DELETE = '🗑️'
    UPDATE_SDF = '🔄'
    SHOW_SDF = '👁️'
    NONE_MODE = '❌'

# 组件名称映射
class ComponentNames:
    DISPLAY_NAMES = {
        'rect': 'Rectangle',
        'circle': 'Circle',
        'capsule': 'Capsule'
    }