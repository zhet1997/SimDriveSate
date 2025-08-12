"""
热仿真计算后端
集成data_generator.py的热仿真功能到UI系统中
"""

import sys
import os
import time
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import logging

from .base_backend import ComputationBackendV2, FieldType, ComputationResult, InputDataFormat

# 添加data_generator.py所在路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 尝试导入data_generator相关模块
try:
    from data_generator import compute_sdf
    DATA_GENERATOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"无法导入data_generator模块: {e}")
    DATA_GENERATOR_AVAILABLE = False

# 尝试导入热仿真求解器
try:
    from layout.SeqLS import SeqLS  # 布局算法
    from pythonForFenics.test_fenicsx_solver import TemperatureFieldSolver  # 温度场求解器
    THERMAL_SOLVER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"无法导入热仿真求解器: {e}")
    THERMAL_SOLVER_AVAILABLE = False


class ThermalSimulationBackend(ComputationBackendV2):
    """热仿真计算后端"""
    
    def __init__(self):
        super().__init__("ThermalSimulationBackend")
        self.default_layout_domain = (0.2, 0.2)
        self.default_mesh_size = (256, 256)
        self.default_boundary_temp = 298.0  # K
        
    def get_supported_field_types(self) -> List[FieldType]:
        """返回支持的场类型"""
        supported = [FieldType.SDF]  # 总是支持SDF计算
        
        if THERMAL_SOLVER_AVAILABLE:
            supported.append(FieldType.TEMPERATURE)
            
        return supported
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """验证输入数据"""
        # 检查基本结构
        if "components" not in input_data:
            return False, "缺少components字段"
        
        components = input_data["components"]
        if not isinstance(components, list) or len(components) == 0:
            return False, "components必须是非空列表"
        
        # 检查每个组件的必需字段
        for i, comp in enumerate(components):
            if not isinstance(comp, dict):
                return False, f"组件 {i} 必须是字典类型"
            
            required_fields = ["shape", "center", "power"]
            for field in required_fields:
                if field not in comp:
                    return False, f"组件 {i} 缺少必需字段: {field}"
            
            # 验证center字段格式
            center = comp["center"]
            if not isinstance(center, (list, tuple)):
                return False, f"组件 {i} 的center字段必须是列表或元组类型，当前为: {type(center)}"
            if len(center) < 2:
                return False, f"组件 {i} 的center字段必须包含至少2个坐标值，当前长度: {len(center)}"
            try:
                float(center[0])
                float(center[1])
            except (ValueError, TypeError):
                return False, f"组件 {i} 的center坐标必须是数值类型"
            
            # 检查形状特定字段
            shape = comp["shape"]
            if shape == "rect":
                if "width" not in comp or "height" not in comp:
                    return False, f"矩形组件 {i} 缺少width或height字段"
            elif shape == "circle":
                if "radius" not in comp:
                    return False, f"圆形组件 {i} 缺少radius字段"
            elif shape == "capsule":
                if "length" not in comp or "width" not in comp:
                    return False, f"胶囊组件 {i} 缺少length或width字段"
            else:
                return False, f"组件 {i} 的形状 '{shape}' 不支持"
        
        return True, "验证通过"
    
    def compute_field(self, 
                     input_data: Dict[str, Any], 
                     field_type: FieldType,
                     grid_shape: Tuple[int, int],
                     **kwargs) -> ComputationResult:
        """计算指定类型的物理场"""
        start_time = time.time()
        
        try:
            # 验证输入数据
            is_valid, error_msg = self.validate_input_data(input_data)
            if not is_valid:
                return ComputationResult(
                    field_data=None,
                    field_type=field_type,
                    metadata={},
                    error_info=f"输入数据验证失败: {error_msg}"
                )
            
            components = input_data["components"]
            layout_domain = input_data.get("layout_domain", self.default_layout_domain)
            
            if field_type == FieldType.SDF:
                return self._compute_sdf(components, layout_domain, grid_shape, start_time)
            elif field_type == FieldType.TEMPERATURE:
                return self._compute_temperature(input_data, grid_shape, start_time, **kwargs)
            else:
                return ComputationResult(
                    field_data=None,
                    field_type=field_type,
                    metadata={},
                    error_info=f"不支持的场类型: {field_type}"
                )
                
        except Exception as e:
            return ComputationResult(
                field_data=None,
                field_type=field_type,
                metadata={},
                error_info=f"计算过程中发生错误: {str(e)}",
                computation_time=time.time() - start_time
            )
    
    def _compute_sdf(self, components: List[Dict], layout_domain: Tuple[float, float], 
                    grid_shape: Tuple[int, int], start_time: float) -> ComputationResult:
        """计算SDF场"""
        if not DATA_GENERATOR_AVAILABLE:
            # 使用简化的SDF计算
            sdf_data = self._compute_simple_sdf(components, layout_domain, grid_shape)
        else:
            # 使用data_generator中的精确SDF计算
            sdf_data = self._compute_exact_sdf(components, layout_domain, grid_shape)
        
        metadata = {
            "unit": "m",
            "description": "有符号距离场",
            "layout_domain": layout_domain,
            "grid_shape": grid_shape,
            "component_count": len(components)
        }
        
        return ComputationResult(
            field_data=sdf_data,
            field_type=FieldType.SDF,
            metadata=metadata,
            computation_time=time.time() - start_time
        )
    
    def _compute_temperature(self, input_data: Dict[str, Any], grid_shape: Tuple[int, int],
                           start_time: float, **kwargs) -> ComputationResult:
        """计算温度场"""
        if not THERMAL_SOLVER_AVAILABLE:
            return ComputationResult(
                field_data=None,
                field_type=FieldType.TEMPERATURE,
                metadata={},
                error_info="热仿真求解器不可用，请检查依赖模块"
            )
        
        try:
            components = input_data["components"]
            layout_domain = input_data.get("layout_domain", self.default_layout_domain)
            boundary_temp = input_data.get("boundary_temperature", self.default_boundary_temp)
            boundary_conditions = input_data.get("boundary_conditions", [
                ([0.0, 0.0], [layout_domain[0], 0.0]),
                ([0.0, layout_domain[1]], [layout_domain[0], layout_domain[1]])
            ])
            
            # 创建温度场求解器
            temp_solver = TemperatureFieldSolver(layout_domain, grid_shape)
            temp_solver.set_boundary_conditions(boundary_conditions)
            
            # 转换组件为热源格式
            heat_sources = self._convert_to_heat_sources(components)
            
            # 生成热源矩阵
            heat_source_matrix = temp_solver.generate_source_matrix(heat_sources)
            
            # 求解温度场
            u_sol, V = temp_solver.solve(u0=boundary_temp)
            
            # 提取求解结果
            dof_coords = V.tabulate_dof_coordinates()
            x = dof_coords[:, 0]
            y = dof_coords[:, 1]
            solution_values = u_sol.x.array
            
            # 插值到规则网格
            from scipy.interpolate import griddata
            
            x_step = layout_domain[0] / grid_shape[1]
            y_step = layout_domain[1] / grid_shape[0]
            x_grid = np.linspace(x_step/2, layout_domain[0] - x_step/2, grid_shape[1])
            y_grid = np.linspace(y_step/2, layout_domain[1] - y_step/2, grid_shape[0])
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='xy')
            
            temp_field = griddata(
                (x, y),
                solution_values,
                (X_grid, Y_grid),
                method='cubic'
            )
            
            # 处理NaN值
            temp_field[np.isnan(temp_field)] = boundary_temp
            
            # 计算SDF作为辅助数据
            sdf_data = self._compute_exact_sdf(components, layout_domain, grid_shape)
            
            metadata = {
                "unit": "K",
                "description": "温度场分布",
                "layout_domain": layout_domain,
                "boundary_temperature": boundary_temp,
                "min_temperature": float(np.min(temp_field)),
                "max_temperature": float(np.max(temp_field)),
                "solver_type": "finite_element"
            }
            
            return ComputationResult(
                field_data=temp_field,
                field_type=FieldType.TEMPERATURE,
                metadata=metadata,
                auxiliary_data={"sdf": sdf_data, "heat_sources": heat_source_matrix},
                computation_time=time.time() - start_time
            )
            
        except Exception as e:
            return ComputationResult(
                field_data=None,
                field_type=FieldType.TEMPERATURE,
                metadata={},
                error_info=f"温度场计算失败: {str(e)}",
                computation_time=time.time() - start_time
            )
    
    def _compute_exact_sdf(self, components: List[Dict], layout_domain: Tuple[float, float],
                          grid_shape: Tuple[int, int]) -> np.ndarray:
        """使用data_generator中的精确SDF计算"""
        # 创建网格坐标
        x_step = layout_domain[0] / grid_shape[1]
        y_step = layout_domain[1] / grid_shape[0]
        x = np.linspace(x_step/2, layout_domain[0] - x_step/2, grid_shape[1])
        y = np.linspace(y_step/2, layout_domain[1] - y_step/2, grid_shape[0])
        X, Y = np.meshgrid(x, y, indexing='xy')
        
        # 调用data_generator中的SDF计算函数
        sdf = compute_sdf(components, X, Y)
        return sdf
    
    def _compute_simple_sdf(self, components: List[Dict], layout_domain: Tuple[float, float],
                           grid_shape: Tuple[int, int]) -> np.ndarray:
        """简化的SDF计算（当data_generator不可用时）"""
        from scipy.spatial.distance import cdist
        
        sdf = np.full(grid_shape, np.inf)
        
        # 创建网格点
        x_step = layout_domain[0] / grid_shape[1]
        y_step = layout_domain[1] / grid_shape[0]
        x = np.linspace(x_step/2, layout_domain[0] - x_step/2, grid_shape[1])
        y = np.linspace(y_step/2, layout_domain[1] - y_step/2, grid_shape[0])
        X, Y = np.meshgrid(x, y, indexing='xy')
        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        
        for comp in components:
            center = comp["center"]
            # 确保center是可迭代的（列表或元组）
            if isinstance(center, (int, float)):
                # 如果是单个数值，跳过此组件
                continue
            elif isinstance(center, (list, tuple)) and len(center) >= 2:
                cx, cy = center[0], center[1]
            else:
                # 无效的center格式，跳过此组件
                continue
            
            if comp["shape"] == "circle":
                r = comp["radius"]
                dist = cdist(points, [[cx, cy]]).ravel() - r
                dist = dist.reshape(grid_shape)
                sdf = np.minimum(sdf, dist)
        
        return sdf
    
    def _convert_to_heat_sources(self, components: List[Dict]) -> List[Dict]:
        """将组件转换为热源格式"""
        heat_sources = []
        
        for comp in components:
            # 检查center格式
            center = comp.get("center", [0, 0])
            if isinstance(center, (int, float)):
                center = [center, center]  # 如果是单个数值，转换为[x, x]
            elif not isinstance(center, (list, tuple)) or len(center) < 2:
                center = [0, 0]  # 默认值
            
            source = {
                "shape": comp["shape"],
                "center": list(center[:2]),  # 只取前两个值并转换为列表
                "power": comp["power"]
            }
            
            if comp["shape"] == "rect":
                source["width"] = comp["width"]
                source["height"] = comp["height"]
            elif comp["shape"] == "circle":
                source["radius"] = comp["radius"]
            elif comp["shape"] == "capsule":
                source["length"] = comp["length"]
                source["width"] = comp["width"]
            
            heat_sources.append(source)
        
        return heat_sources
    
    def get_backend_info(self) -> Dict[str, Any]:
        """获取后端信息"""
        info = super().get_backend_info()
        info.update({
            "data_generator_available": DATA_GENERATOR_AVAILABLE,
            "thermal_solver_available": THERMAL_SOLVER_AVAILABLE,
            "default_layout_domain": self.default_layout_domain,
            "default_mesh_size": self.default_mesh_size
        })
        return info
