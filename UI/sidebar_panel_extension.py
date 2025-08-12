# === 🆕 选择同步相关方法扩展 ===

def select_component_tab(self, component_id: str):
    """根据组件ID选择对应的标签页"""
    component_id = str(component_id)
    if component_id in self.component_id_to_tab_index:
        tab_index = self.component_id_to_tab_index[component_id]
        self.components_tabs.setCurrentIndex(tab_index)
        print(f"[侧边栏] 切换到组件标签页: {component_id} (索引 {tab_index})")
        return True
    else:
        print(f"[侧边栏] 警告: 未找到组件 {component_id} 的标签页")
        return False

def get_current_selected_component(self) -> str:
    """获取当前选中标签页对应的组件ID"""
    current_index = self.components_tabs.currentIndex()
    for component_id, tab_index in self.component_id_to_tab_index.items():
        if tab_index == current_index:
            return component_id
    return None

def _create_property_table_from_data(self, layout, component_data):
    """🆕 从数据管理器数据创建属性表格"""
    from PyQt6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem
    from ui_constants import StyleSheets
    
    # 属性表格标题
    properties_label = QLabel("属性 (Properties)")
    properties_label.setStyleSheet(StyleSheets.SECTION_LABEL)
    layout.addWidget(properties_label)
    
    # 创建属性表格
    properties_table = QTableWidget(0, 2)
    properties_table.setHorizontalHeaderLabels(["属性", "值"])
    properties_table.horizontalHeader().setStretchLastSection(True)
    properties_table.setStyleSheet(StyleSheets.PROPERTY_TABLE)
    
    # 添加属性行
    comp_type = component_data.get('type', 'unknown')
    center = component_data.get('center', [0, 0])
    power = component_data.get('power', 0)
    
    self._add_property_row(properties_table, "类型", comp_type)
    self._add_property_row(properties_table, "位置", f"({center[0]:.3f}, {center[1]:.3f})m")
    self._add_property_row(properties_table, "功率", f"{power:.1f}W")
    
    # 添加尺寸信息
    if comp_type == 'rect':
        width = component_data.get('width', 0)
        height = component_data.get('height', 0)
        self._add_property_row(properties_table, "宽度", f"{width:.3f}m")
        self._add_property_row(properties_table, "高度", f"{height:.3f}m")
    elif comp_type == 'circle':
        radius = component_data.get('radius', 0)
        self._add_property_row(properties_table, "半径", f"{radius:.3f}m")
    elif comp_type == 'capsule':
        length = component_data.get('length', 0)
        width = component_data.get('width', 0)
        self._add_property_row(properties_table, "长度", f"{length:.3f}m")
        self._add_property_row(properties_table, "宽度", f"{width:.3f}m")
    
    layout.addWidget(properties_table)

def _delete_component_by_id(self, component_id: str):
    """🆕 通过组件ID删除组件"""
    if hasattr(self.main_window, 'data_sync'):
        try:
            self.main_window.data_sync.handle_component_delete(component_id)
            print(f"[侧边栏] 通过数据管理器删除组件: {component_id}")
            return
        except Exception as e:
            print(f"[侧边栏] 数据管理器删除失败: {e}")
    
    # 后备：查找UI中的对应项并删除
    for item in self.main_window.scene.items():
        if hasattr(item, 'get_state'):
            state = item.get_state()
            if state.get('id') == component_id:
                self._delete_component(item)
                break

def _add_property_row(self, table, property_name, property_value):
    """向属性表格添加一行"""
    from PyQt6.QtWidgets import QTableWidgetItem
    row = table.rowCount()
    table.insertRow(row)
    table.setItem(row, 0, QTableWidgetItem(str(property_name)))
    table.setItem(row, 1, QTableWidgetItem(str(property_value)))

# 将这些方法动态添加到SidebarPanel类
def extend_sidebar_panel():
    """扩展SidebarPanel类，添加选择同步功能"""
    from sidebar_panel import SidebarPanel
    
    SidebarPanel.select_component_tab = select_component_tab
    SidebarPanel.get_current_selected_component = get_current_selected_component
    SidebarPanel._create_property_table_from_data = _create_property_table_from_data
    SidebarPanel._delete_component_by_id = _delete_component_by_id
    SidebarPanel._add_property_row = _add_property_row
    
    print("[扩展] SidebarPanel选择同步功能已添加")

if __name__ == "__main__":
    extend_sidebar_panel()
