"""
侧边栏面板组件
包含绘制模式控制、SDF控制和组件列表管理
"""

from typing import TYPE_CHECKING
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QButtonGroup, QFrame, QCheckBox, QScrollArea,
                             QDoubleSpinBox)
from PyQt6.QtCore import Qt
from ui_constants import StyleSheets, Icons, ComponentNames
from ui_utils import format_position_text, format_size_text

if TYPE_CHECKING:
    from main_window import MainWindow


class SidebarPanel:
    """侧边栏面板管理器"""
    
    def __init__(self, main_window: 'MainWindow'):
        self.main_window = main_window
        self.setup_sidebar()
    
    def setup_sidebar(self):
        """创建侧边栏面板"""
        # 创建dock widget
        self.sidebar_dock = QDockWidget("Drawing Tools", self.main_window)
        self.sidebar_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                                          Qt.DockWidgetArea.RightDockWidgetArea)
        
        # 创建侧边栏内容widget
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        
        # 添加各个部分
        self._create_drawing_mode_section(sidebar_layout)
        self._create_separator(sidebar_layout)
        self._create_sdf_section(sidebar_layout)
        self._create_separator(sidebar_layout)
        self._create_components_section(sidebar_layout)
        
        # 添加弹性空间
        sidebar_layout.addStretch()
        
        # 设置侧边栏内容
        self.sidebar_dock.setWidget(sidebar_widget)
        self.main_window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.sidebar_dock)
    
    def _create_drawing_mode_section(self, layout: QVBoxLayout):
        """创建绘制模式区域"""
        # 绘制模式标题
        mode_label = QLabel("Drawing Mode")
        mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mode_label.setStyleSheet(StyleSheets.SECTION_LABEL)
        layout.addWidget(mode_label)
        
        # 创建按钮组，支持取消选中
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.setExclusive(False)  # 允许取消选中
        
        # 矩形按钮
        self.rect_button = QPushButton("Rectangle")
        self.rect_button.setCheckable(True)
        self.rect_button.clicked.connect(lambda: self.main_window.toggle_draw_mode('rect'))
        self.rect_button.setStyleSheet(StyleSheets.RECT_BUTTON)
        self.mode_button_group.addButton(self.rect_button)
        layout.addWidget(self.rect_button)
        
        # 圆形按钮
        self.circle_button = QPushButton("Circle")
        self.circle_button.setCheckable(True)
        self.circle_button.clicked.connect(lambda: self.main_window.toggle_draw_mode('circle'))
        self.circle_button.setStyleSheet(StyleSheets.CIRCLE_BUTTON)
        self.mode_button_group.addButton(self.circle_button)
        layout.addWidget(self.circle_button)
        
        # 胶囊按钮
        self.capsule_button = QPushButton("Capsule")
        self.capsule_button.setCheckable(True)
        self.capsule_button.clicked.connect(lambda: self.main_window.toggle_draw_mode('capsule'))
        self.capsule_button.setStyleSheet(StyleSheets.CAPSULE_BUTTON)
        self.mode_button_group.addButton(self.capsule_button)
        layout.addWidget(self.capsule_button)
    
    def _create_sdf_section(self, layout: QVBoxLayout):
        """创建SDF控制区域"""
        # SDF控制标题
        sdf_label = QLabel("SDF Visualization")
        sdf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sdf_label.setStyleSheet(StyleSheets.SECTION_LABEL)
        layout.addWidget(sdf_label)
        
        # SDF显示开关
        self.sdf_show_checkbox = QCheckBox("Show SDF Background")
        self.sdf_show_checkbox.setStyleSheet(StyleSheets.SDF_CHECKBOX)
        self.sdf_show_checkbox.toggled.connect(self.main_window.on_sdf_show_toggled)
        layout.addWidget(self.sdf_show_checkbox)
        
        # SDF更新按钮（初始隐藏）
        self.sdf_update_button = QPushButton(f"{Icons.UPDATE_SDF} Update SDF")
        self.sdf_update_button.setVisible(False)
        self.sdf_update_button.clicked.connect(self.main_window.update_sdf_background)
        self.sdf_update_button.setStyleSheet(StyleSheets.SDF_UPDATE_BUTTON)
        layout.addWidget(self.sdf_update_button)
    
    def _create_components_section(self, layout: QVBoxLayout):
        """创建组件列表区域"""
        # 组件列表标题
        components_label = QLabel("Components")
        components_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        components_label.setStyleSheet(StyleSheets.SECTION_LABEL)
        layout.addWidget(components_label)
        
        # 创建滚动区域用于组件列表
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)
        scroll_area.setStyleSheet(StyleSheets.SCROLL_AREA)
        
        # 组件列表容器
        self.components_widget = QWidget()
        self.components_layout = QVBoxLayout(self.components_widget)
        self.components_layout.setContentsMargins(5, 5, 5, 5)
        self.components_layout.setSpacing(5)
        
        # 添加"无组件"标签
        self.no_components_label = QLabel("No components added yet")
        self.no_components_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_components_label.setStyleSheet(StyleSheets.NO_COMPONENTS_LABEL)
        self.components_layout.addWidget(self.no_components_label)
        
        self.components_layout.addStretch()
        scroll_area.setWidget(self.components_widget)
        layout.addWidget(scroll_area)
    
    def _create_separator(self, layout: QVBoxLayout):
        """创建分隔线"""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet(StyleSheets.SEPARATOR)
        layout.addWidget(separator)
    
    def update_components_list(self):
        """更新组件列表显示"""
        # 清除现有组件项
        for i in reversed(range(self.components_layout.count())):
            child = self.components_layout.itemAt(i).widget()
            if child and child != self.no_components_label:
                child.setParent(None)
        
        # 获取所有组件
        components = []
        for item in self.main_window.scene.items():
            if hasattr(item, 'get_state'):
                components.append(item)
        
        if not components:
            # 显示无组件标签
            self.no_components_label.setVisible(True)
        else:
            # 隐藏无组件标签
            self.no_components_label.setVisible(False)
            
            # 为每个组件创建编辑项
            for i, component_item in enumerate(components):
                self._create_component_editor(component_item, i)
        
        # 确保stretch在最后
        self.components_layout.addStretch()
    
    def _create_component_editor(self, component_item, index):
        """为组件创建编辑器"""
        state = component_item.get_state()
        
        # 创建组件编辑容器
        editor_frame = QFrame()
        editor_frame.setFrameStyle(QFrame.Shape.Box)
        editor_frame.setStyleSheet(StyleSheets.COMPONENT_EDITOR_FRAME)
        
        editor_layout = QVBoxLayout(editor_frame)
        editor_layout.setContentsMargins(8, 5, 8, 5)
        editor_layout.setSpacing(3)
        
        # 组件标题
        component_type = ComponentNames.DISPLAY_NAMES.get(state['type'], state['type'])
        title_label = QLabel(f"{component_type} #{index + 1}")
        title_label.setStyleSheet(StyleSheets.COMPONENT_TITLE)
        editor_layout.addWidget(title_label)
        
        # 位置信息（只读）
        pos_label = QLabel(format_position_text(state['coords'], self.main_window.scene_scale))
        pos_label.setStyleSheet(StyleSheets.COMPONENT_INFO)
        editor_layout.addWidget(pos_label)
        
        # 尺寸信息（只读）
        size_label = QLabel(format_size_text(state['type'], state['size'], self.main_window.scene_scale))
        size_label.setStyleSheet(StyleSheets.COMPONENT_INFO)
        editor_layout.addWidget(size_label)
        
        # 功率编辑器
        power_layout = QHBoxLayout()
        power_layout.setContentsMargins(0, 0, 0, 0)
        
        power_label = QLabel("Power:")
        power_label.setStyleSheet(StyleSheets.POWER_LABEL)
        power_layout.addWidget(power_label)
        
        power_spinbox = QDoubleSpinBox()
        power_spinbox.setRange(0.0, 1000.0)
        power_spinbox.setSingleStep(0.1)
        power_spinbox.setDecimals(1)
        power_spinbox.setSuffix(" W")
        power_spinbox.setValue(state['power'])
        power_spinbox.setStyleSheet(StyleSheets.POWER_SPINBOX)
        
        # 连接信号以实时更新
        power_spinbox.valueChanged.connect(
            lambda value, item=component_item: self.main_window.update_component_power(item, value)
        )
        
        power_layout.addWidget(power_spinbox)
        power_layout.addStretch()
        
        editor_layout.addLayout(power_layout)
        
        # 添加到列表布局中（在stretch之前）
        self.components_layout.insertWidget(self.components_layout.count() - 1, editor_frame)