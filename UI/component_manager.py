"""
简化的组件数据管理中心
ComponentManager - 轻量级数据管理，避免过度抽象
"""

from typing import Dict, List, Optional, Any, Tuple
import uuid
from PyQt6.QtCore import QObject, pyqtSignal


class ComponentManager(QObject):
    """
    简化的组件数据管理中心
    
    职责:
    1. 唯一数据源 - 存储所有组件数据
    2. 基础CRUD操作
    3. 数据变更通知
    """
    
    # 简化的信号 - 只保留必要的
    component_added = pyqtSignal(str)  # component_id
    component_removed = pyqtSignal(str)  # component_id  
    component_updated = pyqtSignal(str)  # component_id
    data_changed = pyqtSignal()  # 通用数据变更信号
    
    # 🆕 选择状态信号
    component_selected = pyqtSignal(str)  # component_id
    selection_cleared = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # 核心数据存储 - 唯一数据源 (简化为直接的字典存储)
        self._components: Dict[str, Dict[str, Any]] = {}
        
        # 🆕 选择状态管理
        self._selected_component_id: Optional[str] = None
    
    # === 简化的CRUD操作 ===
    
    def add_component(self, component_data: Dict[str, Any]) -> str:
        """添加组件到数据中心"""
        # 生成或规范化ID
        if 'id' not in component_data:
            component_data['id'] = f"comp_{uuid.uuid4().hex[:8]}"
        else:
            # 确保ID是字符串类型（JSON中可能是整数）
            component_data['id'] = str(component_data['id'])
        
        component_id = component_data['id']
        
        # 存储数据
        self._components[component_id] = component_data.copy()
        
        # 发送通知
        self.component_added.emit(component_id)
        self.data_changed.emit()
        
        return component_id
    
    def remove_component(self, component_id: str) -> bool:
        """删除组件"""
        # 确保component_id是字符串类型
        component_id = str(component_id)
        
        if component_id not in self._components:
            return False
        
        # 删除组件
        del self._components[component_id]
        
        # 发送通知
        self.component_removed.emit(component_id)
        self.data_changed.emit()
        
        return True
    
    def update_component(self, component_id: str, updates: Dict[str, Any]) -> bool:
        """更新组件属性"""
        # 确保component_id是字符串类型
        component_id = str(component_id)
        
        if component_id not in self._components:
            return False
        
        # 直接更新字典
        self._components[component_id].update(updates)
        
        # 发送通知
        self.component_updated.emit(component_id)
        self.data_changed.emit()
        
        return True
    
    # === 查询操作 ===
    
    def get_component(self, component_id: str) -> Optional[Dict[str, Any]]:
        """获取单个组件数据"""
        # 确保component_id是字符串类型
        component_id = str(component_id)
        return self._components.get(component_id)
    
    def get_all_components(self) -> List[Dict[str, Any]]:
        """获取所有组件数据 - 核心数据源接口"""
        return list(self._components.values())
    
    def get_component_count(self) -> int:
        """获取组件总数"""
        return len(self._components)
    
    # === 批量操作 ===
    
    def clear_all_components(self):
        """清空所有组件"""
        self._components.clear()
        self.data_changed.emit()
    
    def load_components(self, components_data: List[Dict[str, Any]]):
        """批量加载组件（用于JSON导入等）"""
        self.clear_all_components()
        for comp_data in components_data:
            self.add_component(comp_data)
    
    # === 数据转换（简化版） ===
    
    def to_data_generator_format(self) -> List[Dict[str, Any]]:
        """转换为data_generator.py格式（用于计算）"""
        # 直接返回组件数据，让现有的计算后端处理
        return list(self._components.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        type_counts = {}
        total_power = 0
        
        for comp in self._components.values():
            comp_type = comp.get('type', 'unknown')
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
            total_power += comp.get('power', 0)
        
        return {
            'total_components': len(self._components),
            'type_distribution': type_counts,
            'total_power': total_power
        }
    
    # === 🆕 选择状态管理 ===
    
    def set_selected_component(self, component_id: Optional[str]):
        """设置当前选中的组件"""
        if component_id is not None:
            component_id = str(component_id)  # 确保ID是字符串
            if component_id not in self._components:
                print(f"[ComponentManager] 警告: 试图选择不存在的组件 {component_id}")
                return
        
        old_selection = self._selected_component_id
        self._selected_component_id = component_id
        
        # 发出信号
        if old_selection != component_id:
            if component_id is None:
                print("[ComponentManager] 清除组件选择")
                self.selection_cleared.emit()
            else:
                print(f"[ComponentManager] 选择组件: {component_id}")
                self.component_selected.emit(component_id)
    
    def get_selected_component(self) -> Optional[str]:
        """获取当前选中的组件ID"""
        return self._selected_component_id
    
    def clear_selection(self):
        """清除当前选择"""
        self.set_selected_component(None)
    
    def is_component_selected(self, component_id: str) -> bool:
        """检查指定组件是否被选中"""
        return str(component_id) == self._selected_component_id


# 全局单例实例
_component_manager_instance = None

def get_component_manager() -> ComponentManager:
    """获取ComponentManager单例实例"""
    global _component_manager_instance
    if _component_manager_instance is None:
        _component_manager_instance = ComponentManager()
    return _component_manager_instance
