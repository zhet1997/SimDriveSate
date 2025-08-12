"""
数据验证器
验证各种数据格式的有效性和完整性
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class ComponentDataValidator:
    """组件数据验证器"""
    
    @staticmethod
    def validate_data_generator_format(components: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """验证data_generator.py格式的组件数据
        
        标准格式：
        {
            "id": int,
            "shape": str,  # "rect", "circle", "capsule"
            "center": [float, float],
            "power": float,
            # 形状特定字段...
        }
        """
        if not isinstance(components, list):
            return False, "components必须是列表类型"
        
        if len(components) == 0:
            return False, "组件列表不能为空"
        
        for i, comp in enumerate(components):
            if not isinstance(comp, dict):
                return False, f"组件 {i} 必须是字典类型"
            
            # 检查必需字段
            required_fields = ["shape", "center", "power"]
            for field in required_fields:
                if field not in comp:
                    return False, f"组件 {i} 缺少必需字段: {field}"
            
            # 验证字段类型和值
            shape = comp["shape"]
            if not isinstance(shape, str) or shape not in ["rect", "circle", "capsule"]:
                return False, f"组件 {i} 的shape字段无效: {shape}"
            
            center = comp["center"]
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                return False, f"组件 {i} 的center字段必须是长度为2的列表或元组"
            
            try:
                float(center[0])
                float(center[1])
            except (ValueError, TypeError):
                return False, f"组件 {i} 的center坐标必须是数值类型"
            
            power = comp["power"]
            if not isinstance(power, (int, float)) or power < 0:
                return False, f"组件 {i} 的power字段必须是非负数值"
            
            # 验证形状特定字段
            if shape == "rect":
                for size_field in ["width", "height"]:
                    if size_field not in comp:
                        return False, f"矩形组件 {i} 缺少 {size_field} 字段"
                    if not isinstance(comp[size_field], (int, float)) or comp[size_field] <= 0:
                        return False, f"矩形组件 {i} 的 {size_field} 必须是正数"
            
            elif shape == "circle":
                if "radius" not in comp:
                    return False, f"圆形组件 {i} 缺少 radius 字段"
                if not isinstance(comp["radius"], (int, float)) or comp["radius"] <= 0:
                    return False, f"圆形组件 {i} 的 radius 必须是正数"
            
            elif shape == "capsule":
                for size_field in ["length", "width"]:
                    if size_field not in comp:
                        return False, f"胶囊组件 {i} 缺少 {size_field} 字段"
                    if not isinstance(comp[size_field], (int, float)) or comp[size_field] <= 0:
                        return False, f"胶囊组件 {i} 的 {size_field} 必须是正数"
        
        return True, "验证通过"
    
    @staticmethod
    def validate_ui_format(ui_components: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """验证UI格式的组件数据
        
        UI格式：
        {
            "type": str,
            "coords": [float, float],
            "size": [float, float],
            "power": float,
            "id": int
        }
        """
        if not isinstance(ui_components, list):
            return False, "UI组件数据必须是列表类型"
        
        if len(ui_components) == 0:
            return False, "UI组件列表不能为空"
        
        for i, comp in enumerate(ui_components):
            if not isinstance(comp, dict):
                return False, f"UI组件 {i} 必须是字典类型"
            
            # 检查必需字段
            required_fields = ["type", "coords", "power"]
            for field in required_fields:
                if field not in comp:
                    return False, f"UI组件 {i} 缺少必需字段: {field}"
            
            # 验证字段
            comp_type = comp["type"]
            if not isinstance(comp_type, str) or comp_type not in ["rect", "circle", "capsule", "radiator", "sensor"]:
                return False, f"UI组件 {i} 的type字段无效: {comp_type}"
            
            coords = comp["coords"]
            if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                return False, f"UI组件 {i} 的coords字段必须是长度为2的列表或元组"
            
            power = comp["power"]
            if not isinstance(power, (int, float)):
                return False, f"UI组件 {i} 的power字段必须是数值类型"
        
        return True, "验证通过"
    
    @staticmethod
    def validate_thermal_data(data_array: np.ndarray, 
                            expected_shape: Optional[Tuple[int, int]] = None) -> Tuple[bool, str]:
        """验证热仿真数据的有效性"""
        if not isinstance(data_array, np.ndarray):
            return False, "数据必须是NumPy数组"
        
        if data_array.ndim != 2:
            return False, f"数据必须是2维数组，当前维度: {data_array.ndim}"
        
        if expected_shape and data_array.shape != expected_shape:
            return False, f"数据形状不匹配，期望: {expected_shape}，实际: {data_array.shape}"
        
        # 检查是否包含无效值
        if np.any(np.isnan(data_array)):
            return False, "数据包含NaN值"
        
        if np.any(np.isinf(data_array)):
            return False, "数据包含无穷大值"
        
        return True, "验证通过"
    
    @staticmethod
    def validate_layout_domain(layout_domain: Any) -> Tuple[bool, str]:
        """验证布局域参数"""
        if not isinstance(layout_domain, (list, tuple)):
            return False, "布局域必须是列表或元组类型"
        
        if len(layout_domain) != 2:
            return False, "布局域必须包含2个元素 (width, height)"
        
        try:
            width, height = float(layout_domain[0]), float(layout_domain[1])
        except (ValueError, TypeError):
            return False, "布局域尺寸必须是数值类型"
        
        if width <= 0 or height <= 0:
            return False, "布局域尺寸必须是正数"
        
        # 合理性检查（假设单位为米）
        if width > 10 or height > 10:
            return False, "布局域尺寸过大（超过10米）"
        
        if width < 0.001 or height < 0.001:
            return False, "布局域尺寸过小（小于1毫米）"
        
        return True, "验证通过"
    
    @staticmethod
    def validate_boundary_conditions(boundary_conditions: Any) -> Tuple[bool, str]:
        """验证边界条件"""
        if not isinstance(boundary_conditions, list):
            return False, "边界条件必须是列表类型"
        
        for i, bc in enumerate(boundary_conditions):
            if not isinstance(bc, (list, tuple)) or len(bc) != 2:
                return False, f"边界条件 {i} 必须是包含2个点的列表或元组"
            
            point1, point2 = bc
            if not isinstance(point1, (list, tuple)) or len(point1) != 2:
                return False, f"边界条件 {i} 的第一个点格式错误"
            
            if not isinstance(point2, (list, tuple)) or len(point2) != 2:
                return False, f"边界条件 {i} 的第二个点格式错误"
            
            try:
                float(point1[0]), float(point1[1])
                float(point2[0]), float(point2[1])
            except (ValueError, TypeError):
                return False, f"边界条件 {i} 的坐标必须是数值类型"
        
        return True, "验证通过"
    
    @staticmethod
    def validate_computation_input(input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """验证计算输入数据的完整性"""
        required_fields = ["components"]
        for field in required_fields:
            if field not in input_data:
                return False, f"缺少必需字段: {field}"
        
        # 验证组件数据
        is_valid, error_msg = ComponentDataValidator.validate_data_generator_format(input_data["components"])
        if not is_valid:
            return False, f"组件数据验证失败: {error_msg}"
        
        # 验证可选字段
        if "layout_domain" in input_data:
            is_valid, error_msg = ComponentDataValidator.validate_layout_domain(input_data["layout_domain"])
            if not is_valid:
                return False, f"布局域验证失败: {error_msg}"
        
        if "boundary_conditions" in input_data:
            is_valid, error_msg = ComponentDataValidator.validate_boundary_conditions(input_data["boundary_conditions"])
            if not is_valid:
                return False, f"边界条件验证失败: {error_msg}"
        
        if "boundary_temperature" in input_data:
            temp = input_data["boundary_temperature"]
            if not isinstance(temp, (int, float)):
                return False, "边界温度必须是数值类型"
            if temp < 0:  # 绝对零度检查
                return False, "边界温度不能小于0K"
            if temp > 10000:  # 合理性检查
                return False, "边界温度过高（超过10000K）"
        
        return True, "验证通过"


class DataIntegrityChecker:
    """数据完整性检查器"""
    
    @staticmethod
    def check_sample_completeness(sample_directory: str) -> Dict[str, Any]:
        """检查样本目录的完整性"""
        import os
        
        expected_files = [
            'components.json',
            'temperature.npy',
            'sdf.npy',
            'heat_source.npy'
        ]
        
        result = {
            "complete": True,
            "missing_files": [],
            "existing_files": [],
            "file_info": {}
        }
        
        for file_name in expected_files:
            file_path = os.path.join(sample_directory, file_name)
            if os.path.exists(file_path):
                result["existing_files"].append(file_name)
                
                # 获取文件信息
                stat = os.stat(file_path)
                result["file_info"][file_name] = {
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                }
                
                # 尝试验证文件内容
                try:
                    if file_name.endswith('.npy'):
                        data = np.load(file_path)
                        result["file_info"][file_name]["shape"] = data.shape
                        result["file_info"][file_name]["dtype"] = str(data.dtype)
                    elif file_name.endswith('.json'):
                        import json
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        result["file_info"][file_name]["component_count"] = len(data) if isinstance(data, list) else 1
                except Exception as e:
                    result["file_info"][file_name]["error"] = str(e)
            else:
                result["missing_files"].append(file_name)
                result["complete"] = False
        
        return result
