"""
数据格式转换器
处理data_generator.py与UI之间的数据格式转换
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class DataFormatConverter:
    """数据格式转换器"""
    
    @staticmethod
    def data_generator_to_ui(components: List[Dict[str, Any]], 
                           scene_scale: float = 1000.0) -> List[Dict[str, Any]]:
        """将data_generator.py的JSON格式转换为UI格式
        
        Data Generator格式:
        {
            "id": 0,
            "shape": "rect",
            "center": [0.0078, 0.0617],  # 米坐标
            "width": 0.009,
            "height": 0.018, 
            "power": 9000
        }
        
        UI格式:
        {
            "type": "rect",
            "coords": [x_pixels, y_pixels],  # 像素坐标
            "size": [width_pixels, height_pixels],
            "power": 9000,
            "id": 0
        }
        """
        ui_components = []
        
        for comp in components:
            ui_comp = {
                "id": comp.get("id", 0),
                "type": DataFormatConverter._convert_shape_name(comp["shape"]),
                "power": comp.get("power", 0)
            }
            
            # 转换坐标：米 -> 像素
            center_m = comp["center"]
            ui_comp["coords"] = [
                center_m[0] * scene_scale,  # x像素
                center_m[1] * scene_scale   # y像素
            ]
            
            # 转换尺寸：米 -> 像素
            if comp["shape"] == "rect":
                ui_comp["size"] = [
                    comp["width"] * scene_scale,
                    comp["height"] * scene_scale
                ]
            elif comp["shape"] == "circle":
                radius_pixels = comp["radius"] * scene_scale
                ui_comp["size"] = [radius_pixels * 2, radius_pixels * 2]  # 直径
                ui_comp["radius"] = radius_pixels
            elif comp["shape"] == "capsule":
                ui_comp["size"] = [
                    comp["length"] * scene_scale,
                    comp["width"] * scene_scale
                ]
            
            # 保留原始的米单位数据（用于精确计算）
            ui_comp["meters_data"] = {
                "center": comp["center"],
                "shape": comp["shape"]
            }
            if comp["shape"] == "rect":
                ui_comp["meters_data"]["width"] = comp["width"]
                ui_comp["meters_data"]["height"] = comp["height"]
            elif comp["shape"] == "circle":
                ui_comp["meters_data"]["radius"] = comp["radius"]
            elif comp["shape"] == "capsule":
                ui_comp["meters_data"]["length"] = comp["length"]
                ui_comp["meters_data"]["width"] = comp["width"]
            
            ui_components.append(ui_comp)
        
        return ui_components
    
    @staticmethod
    def ui_to_data_generator(ui_components: List[Dict[str, Any]], 
                           scene_scale: float = 1000.0) -> List[Dict[str, Any]]:
        """将UI格式转换为data_generator.py的JSON格式"""
        dg_components = []
        
        for ui_comp in ui_components:
            dg_comp = {
                "id": ui_comp.get("id", 0),
                "shape": DataFormatConverter._convert_type_name(ui_comp["type"]),
                "power": ui_comp.get("power", 0)
            }
            
            # 优先使用精确的米单位数据
            if "meters_data" in ui_comp:
                meters_data = ui_comp["meters_data"]
                dg_comp["center"] = meters_data["center"]
                
                if meters_data["shape"] == "rect":
                    dg_comp["width"] = meters_data["width"]
                    dg_comp["height"] = meters_data["height"]
                elif meters_data["shape"] == "circle":
                    dg_comp["radius"] = meters_data["radius"]
                elif meters_data["shape"] == "capsule":
                    dg_comp["length"] = meters_data["length"]
                    dg_comp["width"] = meters_data["width"]
            else:
                # 从像素坐标转换
                coords_pixels = ui_comp["coords"]
                dg_comp["center"] = [
                    coords_pixels[0] / scene_scale,  # 像素 -> 米
                    coords_pixels[1] / scene_scale
                ]
                
                # 转换尺寸
                size_pixels = ui_comp.get("size", [0, 0])
                if ui_comp["type"] == "rect":
                    dg_comp["width"] = size_pixels[0] / scene_scale
                    dg_comp["height"] = size_pixels[1] / scene_scale
                elif ui_comp["type"] == "circle":
                    radius_pixels = ui_comp.get("radius", size_pixels[0] / 2)
                    dg_comp["radius"] = radius_pixels / scene_scale
                elif ui_comp["type"] == "capsule":
                    dg_comp["length"] = size_pixels[0] / scene_scale
                    dg_comp["width"] = size_pixels[1] / scene_scale
            
            dg_components.append(dg_comp)
        
        return dg_components
    
    @staticmethod
    def _convert_shape_name(dg_shape: str) -> str:
        """转换shape名称：data_generator -> UI"""
        mapping = {
            "rect": "rect",
            "circle": "circle", 
            "capsule": "capsule"
        }
        return mapping.get(dg_shape, dg_shape)
    
    @staticmethod
    def _convert_type_name(ui_type: str) -> str:
        """转换type名称：UI -> data_generator"""
        mapping = {
            "rect": "rect",
            "circle": "circle",
            "capsule": "capsule"
        }
        return mapping.get(ui_type, ui_type)
    
    @staticmethod
    def create_thermal_simulation_input(components: List[Dict[str, Any]], 
                                      layout_domain: Tuple[float, float] = (0.2, 0.2),
                                      boundary_temperature: float = 298.0,
                                      boundary_conditions: Optional[List] = None) -> Dict[str, Any]:
        """创建热仿真输入数据格式
        
        Args:
            components: 组件列表（data_generator格式）
            layout_domain: 布局域尺寸（米）
            boundary_temperature: 边界温度（K）
            boundary_conditions: 边界条件
            
        Returns:
            热仿真输入数据
        """
        if boundary_conditions is None:
            boundary_conditions = [
                ([0.0, 0.0], [layout_domain[0], 0.0]),
                ([0.0, layout_domain[1]], [layout_domain[0], layout_domain[1]])
            ]
        
        return {
            "components": components,
            "layout_domain": layout_domain,
            "boundary_temperature": boundary_temperature,
            "boundary_conditions": boundary_conditions,
            "mesh_size": (256, 256),  # 默认网格尺寸
            "coordinate_system": "meters"
        }
    
    @staticmethod
    def extract_thermal_data(thermal_result: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """从热仿真结果中提取温度场数据
        
        Args:
            thermal_result: 热仿真结果
            
        Returns:
            (temperature_field, metadata): 温度场数据和元数据
        """
        if isinstance(thermal_result, dict):
            if "temperature" in thermal_result:
                temp_field = thermal_result["temperature"]
                metadata = {
                    "unit": "K",
                    "x_range": thermal_result.get("x_range", (0, 0.2)),
                    "y_range": thermal_result.get("y_range", (0, 0.2)),
                    "computation_method": "finite_element"
                }
                
                # 添加SDF数据（如果存在）
                if "sdf" in thermal_result:
                    metadata["sdf_data"] = thermal_result["sdf"]
                
                return temp_field, metadata
            else:
                raise ValueError("热仿真结果中缺少温度数据")
        else:
            # 假设直接是温度场数组
            metadata = {"unit": "K", "computation_method": "unknown"}
            return thermal_result, metadata
    
    @staticmethod
    def validate_component_data(components: List[Dict[str, Any]], 
                              data_format: str = "data_generator") -> Tuple[bool, str]:
        """验证组件数据的有效性
        
        Args:
            components: 组件数据列表
            data_format: 数据格式 ("data_generator" 或 "ui")
            
        Returns:
            (is_valid, error_message): 验证结果
        """
        if not components:
            return False, "组件列表为空"
        
        required_fields = {
            "data_generator": ["shape", "center", "power"],
            "ui": ["type", "coords", "power"]
        }
        
        required = required_fields.get(data_format, [])
        
        for i, comp in enumerate(components):
            # 检查必需字段
            for field in required:
                if field not in comp:
                    return False, f"组件 {i} 缺少必需字段: {field}"
            
            # 检查形状特定字段
            if data_format == "data_generator":
                shape = comp["shape"]
                if shape == "rect" and ("width" not in comp or "height" not in comp):
                    return False, f"矩形组件 {i} 缺少尺寸信息"
                elif shape == "circle" and "radius" not in comp:
                    return False, f"圆形组件 {i} 缺少半径信息"
                elif shape == "capsule" and ("length" not in comp or "width" not in comp):
                    return False, f"胶囊组件 {i} 缺少尺寸信息"
        
        return True, "数据验证通过"
