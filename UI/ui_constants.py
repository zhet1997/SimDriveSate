"""
UI常量和样式定义
包含所有UI组件使用的常量、样式表和配置参数
"""

# 应用配置常量
SCENE_SCALE = 4000  # 每米4000像素 - 适配0.1×0.1米精密布局域
DEFAULT_LAYOUT_SIZE = (0.1, 0.1)  # 默认布局尺寸（米）- 匹配微型卫星组件精密设计
DEFAULT_THERMAL_CONDUCTIVITY = 100.0  # 默认热导率
DEFAULT_MESH_RESOLUTION = 50  # 默认网格分辨率
GRID_INTERVAL_METERS = 0.01  # 网格间隔（米）- 10mm精密网格适配小尺寸元件

# 组件尺寸限制
MIN_COMPONENT_SIZE = 10  # 最小组件尺寸（像素）
POWER_RANGE = (0.0, 10000.0)  # 功率范围（瓦特）
DEFAULT_POWER_RANGE = (500.0, 15000.0)  # 默认随机功率范围

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
    
    # 🆕 高亮边框颜色
    HIGHLIGHT_BORDER = (255, 255, 0)  # 黄色高亮边框
    
    # 散热器颜色
    RADIATOR_BORDER = (128, 0, 128)  # 紫色
    RADIATOR_FILL = (221, 160, 221, 150)  # 浅紫色填充
    RADIATOR_SELECTED = (200, 100, 200, 150)  # 选中紫色
    
    # 测点颜色
    SENSOR_BORDER = (255, 165, 0)  # 橙色
    SENSOR_FILL = (255, 218, 185, 200)  # 浅橙色填充
    SENSOR_SELECTED = (255, 140, 0, 200)  # 选中橙色
    
    # UI颜色
    GRID_LINE = (200, 200, 200)  # 网格线颜色
    GRID_LABEL = (100, 100, 100)  # 网格标签颜色
    BORDER_LINE = (0, 0, 0)  # 边界线颜色
    PREVIEW_LINE = "red"  # 预览线颜色
    TEXT_COLOR = (0, 0, 0)  # 文本颜色

# 样式表定义
class StyleSheets:
    # 绘制模式按钮样式（优化为网格布局）
    DRAW_BUTTON_BASE = """
        QPushButton {{
            padding: 8px 12px;
            font-size: 11px;
            border: 2px solid {border_color};
            border-radius: 6px;
            background-color: #ecf0f1;
            margin: 2px;
            text-align: center;
        }}
        QPushButton:checked {{
            background-color: {border_color};
            color: white;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: #bdc3c7;
        }}
    """
    
    RECT_BUTTON = DRAW_BUTTON_BASE.format(border_color="#3498db")
    CIRCLE_BUTTON = DRAW_BUTTON_BASE.format(border_color="#e74c3c")
    CAPSULE_BUTTON = DRAW_BUTTON_BASE.format(border_color="#27ae60")
    RADIATOR_BUTTON = DRAW_BUTTON_BASE.format(border_color="#9b59b6")
    
    # 通用绘制模式按钮样式（用于传感器按钮等）
    DRAWING_MODE_BUTTON = DRAW_BUTTON_BASE.format(border_color="#f39c12")
    
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
    
    # 通用属性编辑器样式
    PROPERTY_EDITOR = POWER_SPINBOX
    
    # 分隔线样式
    SEPARATOR = "margin: 15px 5px;"
    
    # 🆕 属性表格样式
    PROPERTY_TABLE = """
        QTableWidget {
            background-color: #ffffff;
            border: 1px solid #bdc3c7;
            border-radius: 3px;
            font-size: 10px;
            gridline-color: #ecf0f1;
        }
        QTableWidget::item {
            padding: 3px;
            border-bottom: 1px solid #ecf0f1;
        }
        QTableWidget::item:selected {
            background-color: #3498db;
            color: white;
        }
        QHeaderView::section {
            background-color: #34495e;
            color: white;
            padding: 4px;
            border: none;
            font-weight: bold;
        }
    """
    
    # 标签样式
    SECTION_LABEL = "font-weight: bold; font-size: 14px; margin: 10px;"
    COMPONENT_TITLE = "font-weight: bold; font-size: 11px; color: #2c3e50;"
    COMPONENT_INFO = "font-size: 10px; color: #7f8c8d;"
    POWER_LABEL = "font-size: 10px; font-weight: bold;"
    NO_COMPONENTS_LABEL = "color: #7f8c8d; font-style: italic; padding: 20px;"
    
    # 标签页样式（优化简化标签显示）
    TAB_WIDGET = """
        QTabWidget::pane {
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            background-color: #ffffff;
        }
        QTabBar::tab {
            background-color: #ecf0f1;
            border: 1px solid #bdc3c7;
            padding: 6px 12px;
            margin-right: 1px;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            min-width: 40px;
            font-size: 12px;
            font-weight: bold;
        }
        QTabBar::tab:selected {
            background-color: #3498db;
            color: white;
        }
        QTabBar::tab:hover {
            background-color: #bdc3c7;
        }
    """
    
    # 温度重构按钮样式
    TEMP_RECONSTRUCTION_BUTTON = """
        QPushButton {
            padding: 10px;
            font-size: 12px;
            border: 2px solid #e67e22;
            border-radius: 6px;
            background-color: #ecf0f1;
            margin: 5px;
        }
        QPushButton:hover {
            background-color: #e67e22;
            color: white;
        }
        QPushButton:pressed {
            background-color: #d35400;
        }
    """
    
    # 删除按钮样式
    DELETE_COMPONENT_BUTTON = """
        QPushButton {
            padding: 5px 10px;
            font-size: 10px;
            border: 1px solid #e74c3c;
            border-radius: 4px;
            background-color: #ffffff;
            color: #e74c3c;
            margin: 2px;
        }
        QPushButton:hover {
            background-color: #e74c3c;
            color: white;
        }
    """
    
    # 通用删除按钮样式（传感器等）
    DELETE_BUTTON = DELETE_COMPONENT_BUTTON

# SDF配置
class SDFConfig:
    GRID_MULTIPLIER = 5  # SDF网格分辨率倍数
    COLORMAP = 'coolwarm'  # SDF颜色映射
    ALPHA = 0.6  # SDF透明度
    INTERPOLATION = 'nearest'  # 插值方法
    DPI = 100  # 图像DPI
    Z_VALUE = -1  # SDF背景层级

# 图标和文本标签
class Icons:
    # 绘制模式文本标签（不再使用emoji）
    DRAW_MODES = {
        'rect': 'Rectangle',
        'circle': 'Circle', 
        'capsule': 'Capsule',
        'radiator': 'Radiator'
    }
    
    # 工具栏文本标签（不再使用emoji）
    LOAD_FILE = 'Load'
    SAVE_FILE = 'Save'
    DELETE = 'Delete'
    UPDATE_SDF = 'Update'
    SHOW_SDF = 'Show'
    NONE_MODE = 'None'
    ADD_SENSOR = 'Add Sensor'
    EXECUTE_RECONSTRUCTION = 'Execute'

# 组件名称映射
class ComponentNames:
    DISPLAY_NAMES = {
        'rect': 'Rectangle',
        'circle': 'Circle',
        'capsule': 'Capsule',
        'radiator': 'Radiator Segment'
    }
    
    # 元件类型图标映射（用于简化标签页标题）
    COMPONENT_TYPE_ICONS = {
        'rect': '🔲',
        'rectangle': '🔲',
        'circle': '⭕',
        'capsule': '💊',
        'radiator': '📐'
    }