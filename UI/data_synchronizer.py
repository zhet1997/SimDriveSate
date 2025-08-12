"""
简化的数据同步器 - 连接数据中心与UI
DataSynchronizer - 轻量级双向绑定
"""

from typing import Dict, List, Optional, Any
from PyQt6.QtCore import QObject, pyqtSignal
from component_manager import get_component_manager


class DataSynchronizer(QObject):
    """
    简化的数据同步器
    
    职责:
    1. 连接ComponentManager与UI
    2. 处理数据变更通知
    3. 简单的双向绑定
    """
    
    # 简化的信号
    ui_update_needed = pyqtSignal()  # 通知UI需要更新
    
    # 🆕 选择状态信号
    selection_changed = pyqtSignal(str)  # 当前选中组件ID
    selection_cleared = pyqtSignal()  # 选择被清除
    
    def __init__(self):
        super().__init__()
        self.component_manager = get_component_manager()
        
        # 连接数据管理器信号
        self.component_manager.data_changed.connect(self.ui_update_needed.emit)
        
        # 🆕 连接选择状态信号
        self.component_manager.component_selected.connect(self.selection_changed.emit)
        self.component_manager.selection_cleared.connect(self.selection_cleared.emit)
    
    # === 简化的操作接口 ===
    
    def handle_json_load(self, json_components: List[Dict]):
        """处理JSON文件加载"""
        print(f"[DataSync] JSON加载: {len(json_components)} 个组件")
        self.component_manager.load_components(json_components)
    
    def handle_manual_draw(self, component_data: Dict) -> str:
        """处理手动绘制组件"""
        print(f"[DataSync] 手动绘制: {component_data.get('type')}")
        return self.component_manager.add_component(component_data)
    
    def handle_component_update(self, component_id: str, updates: Dict):
        """处理组件更新"""
        print(f"[DataSync] 组件更新: {component_id}")
        self.component_manager.update_component(component_id, updates)
    
    def handle_component_delete(self, component_id: str):
        """处理组件删除"""
        print(f"[DataSync] 组件删除: {component_id}")
        self.component_manager.remove_component(component_id)
    
    # === 数据获取接口 ===
    
    def get_all_components(self) -> List[Dict]:
        """获取所有组件数据"""
        return self.component_manager.get_all_components()
    
    def get_components_for_calculation(self) -> List[Dict]:
        """获取用于计算的组件数据"""
        return self.component_manager.to_data_generator_format()
    
    # === 🆕 选择同步处理 ===
    
    def handle_component_selection(self, component_id: str):
        """处理组件选择（来自UI点击）"""
        print(f"[DataSync] 处理组件选择: {component_id}")
        self.component_manager.set_selected_component(component_id)
    
    def handle_selection_clear(self):
        """处理选择清除（来自UI）"""
        print("[DataSync] 处理选择清除")
        self.component_manager.clear_selection()
    
    def get_selected_component(self) -> Optional[str]:
        """获取当前选中的组件ID"""
        return self.component_manager.get_selected_component()
    
    def is_component_selected(self, component_id: str) -> bool:
        """检查组件是否被选中"""
        return self.component_manager.is_component_selected(component_id)


# 全局单例实例
_data_synchronizer_instance = None

def get_data_synchronizer() -> DataSynchronizer:
    """获取DataSynchronizer单例实例"""
    global _data_synchronizer_instance
    if _data_synchronizer_instance is None:
        _data_synchronizer_instance = DataSynchronizer()
    return _data_synchronizer_instance