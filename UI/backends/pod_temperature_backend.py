"""
POD温度场预测后端
集成pod_simple的API接口到UI系统中
"""

import sys
import os
import time
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import logging
from pathlib import Path

from .base_backend import ComputationBackendV2, FieldType, ComputationResult, InputDataFormat

# 添加pod_simple所在路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'pod_simple'))

# 尝试导入POD相关模块
try:
    from api_interface import predict_temperature_from_json, PODTemperatureAPI
    from data_utils import extract_features_enhanced
    POD_API_AVAILABLE = True
except ImportError as e:
    logging.warning(f"无法导入POD API模块: {e}")
    POD_API_AVAILABLE = False


class PODTemperatureBackend(ComputationBackendV2):
    """温度场预测后端"""
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        super().__init__("PODTemperatureBackend")
        
        # 模型路径配置
        if model_path is None:
            # 默认使用最新的模型
            model_path = "/home/zhet1997/Code/2d-satellite-layout/results/run_20250812_163834/pod_model.pkl"
        if scaler_path is None:
            scaler_path = "/home/zhet1997/Code/2d-satellite-layout/results/run_20250812_163834/scaler.pkl"
        
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.pod_api = None
        
        # 检查模型文件是否存在
        self.model_available = self._check_model_files()
        
    def _check_model_files(self) -> bool:
        """检查模型文件是否存在"""
        model_exists = Path(self.model_path).exists()
        scaler_exists = Path(self.scaler_path).exists()
        
        if not model_exists:
            logging.warning(f"POD模型文件不存在: {self.model_path}")
        if not scaler_exists:
            logging.warning(f"标准化器文件不存在: {self.scaler_path}")
            
        return model_exists and scaler_exists
        
    def initialize(self) -> bool:
        """初始化后端"""
        if not POD_API_AVAILABLE:
            logging.error("POD API不可用")
            return False
            
        if not self.model_available:
            logging.error("POD模型文件不可用")
            return False
            
        try:
            # 初始化POD API
            self.pod_api = PODTemperatureAPI(self.model_path, self.scaler_path)
            logging.info("POD温度场预测后端初始化成功")
            return True
        except Exception as e:
            logging.error(f"POD后端初始化失败: {e}")
            return False
    
    def get_supported_field_types(self) -> List[FieldType]:
        """返回支持的场类型"""
        if self.model_available and POD_API_AVAILABLE:
            return [FieldType.TEMPERATURE]
        return []
    
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
            
            # POD API要求的字段
            required_fields = ["shape", "center", "power"]
            for field in required_fields:
                if field not in comp:
                    return False, f"组件 {i} 缺少必需字段: {field}"
            
            # 验证center字段格式
            center = comp["center"]
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                return False, f"组件 {i} 的center字段必须是包含至少2个坐标值的列表"
            
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
        """计算POD温度场"""
        start_time = time.time()
        
        if field_type != FieldType.TEMPERATURE:
            return ComputationResult(
                field_data=None,
                field_type=field_type,
                metadata={},
                error_info=f"POD后端不支持的场类型: {field_type}"
            )
        
        if not self.pod_api:
            return ComputationResult(
                field_data=None,
                field_type=field_type,
                metadata={},
                error_info="POD API未初始化"
            )
        
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
            
            # 转换数据格式为POD API要求的格式
            components = input_data["components"]
            pod_components = self._convert_to_pod_format(components)
            
            print(f"[POD后端] 开始预测温度场，组件数: {len(pod_components)}")
            
            # 调用POD API进行预测
            temperature_fields = predict_temperature_from_json(
                pod_components,
                self.model_path,
                self.scaler_path
            )
            
            # 取第一个样本的结果（因为我们只传入了一个样本）
            if temperature_fields.ndim == 3:
                temp_field = temperature_fields[0]
            else:
                temp_field = temperature_fields
            
            # 如果需要调整网格尺寸
            if temp_field.shape != grid_shape:
                temp_field = self._resize_field(temp_field, grid_shape)
            
            metadata = {
                "unit": "K",
                "description": "POD温度场预测",
                "grid_shape": temp_field.shape,
                "component_count": len(components),
                "min_temperature": float(np.min(temp_field)),
                "max_temperature": float(np.max(temp_field)),
                "prediction_method": "POD"
            }
            
            print(f"[POD后端] 温度场预测完成: {temp_field.shape}, 温度范围: [{metadata['min_temperature']:.2f}, {metadata['max_temperature']:.2f}]K")
            
            return ComputationResult(
                field_data=temp_field,
                field_type=field_type,
                metadata=metadata,
                computation_time=time.time() - start_time
            )
            
        except Exception as e:
            return ComputationResult(
                field_data=None,
                field_type=field_type,
                metadata={},
                error_info=f"温度场预测失败: {str(e)}",
                computation_time=time.time() - start_time
            )
    
    def _convert_to_pod_format(self, components: List[Dict]) -> List[List[Dict]]:
        """将组件数据转换为POD API要求的格式"""
        # POD API要求的是样本列表的格式：[[sample1_components], [sample2_components], ...]
        # 我们只有一个样本，所以包装成单个样本
        
        pod_components = []
        for comp in components:
            pod_comp = {
                'shape': comp['shape'],
                'center': comp['center'],
                'power': comp['power']
            }
            
            # 添加node_position字段（POD特征提取需要）
            # 如果没有提供，使用center的像素坐标
            if 'node_position' not in comp:
                center = comp['center']
                # 将米坐标转换为像素坐标（假设场景尺寸为256x256像素对应0.1x0.1米）
                pixel_x = center[0] * 2560  # 0.1米 = 256像素, 所以1米 = 2560像素
                pixel_y = center[1] * 2560
                pod_comp['node_position'] = [pixel_x, pixel_y]
            else:
                pod_comp['node_position'] = comp['node_position']
            
            # 添加形状特定字段
            if comp['shape'] == 'rect':
                pod_comp['width'] = comp['width']
                pod_comp['height'] = comp['height']
            elif comp['shape'] == 'circle':
                pod_comp['radius'] = comp['radius']
            elif comp['shape'] == 'capsule':
                pod_comp['length'] = comp['length']
                pod_comp['width'] = comp['width']
            
            pod_components.append(pod_comp)
        
        return [pod_components]  # 包装成单个样本
    
    def _resize_field(self, field: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """调整温度场尺寸"""
        from scipy.ndimage import zoom
        
        zoom_factors = (target_shape[0] / field.shape[0], target_shape[1] / field.shape[1])
        return zoom(field, zoom_factors, order=1)
    
    def reconstruct_temperature_field(self, 
                                     measurement_points: List[Tuple[float, float]], 
                                     temperature_values: List[float],
                                     **kwargs) -> ComputationResult:
        """使用GA方法从测点数据重构温度场"""
        start_time = time.time()
        
        if not self.pod_api:
            return ComputationResult(
                field_data=None,
                field_type=FieldType.TEMPERATURE,
                metadata={},
                error_info="POD API未初始化"
            )
        
        try:
            # 导入重构API
            from api_interface import reconstruct_temperature_from_measurements
            
            print(f"[POD重构] 开始GA重构，测点数: {len(measurement_points)}")
            print(f"[POD重构] 温度范围: [{min(temperature_values):.2f}, {max(temperature_values):.2f}]K")
            
            # 调用POD重构API，固定使用GA方法
            reconstructed_field, coefficients, metrics = reconstruct_temperature_from_measurements(
                measurement_points,
                temperature_values, 
                self.model_path,
                method='ga',
                pop_size=30,      # 固定参数
                n_generations=20  # 固定参数
            )
            
            metadata = {
                "unit": "K",
                "description": "POD GA温度场重构",
                "reconstruction_method": "genetic_algorithm",
                "measurement_count": len(measurement_points),
                "min_temperature": float(np.min(reconstructed_field)),
                "max_temperature": float(np.max(reconstructed_field)),
                "pod_coefficients": coefficients.tolist(),
                "validation_metrics": metrics
            }
            
            print(f"[POD重构] GA重构完成: {reconstructed_field.shape}")
            print(f"  重构温度范围: [{metadata['min_temperature']:.2f}, {metadata['max_temperature']:.2f}]K")
            print(f"  测点平均误差: {metrics.get('point_mae', 0):.4f}")
            print(f"  测点相关系数: {metrics.get('point_correlation', 0):.4f}")
            
            return ComputationResult(
                field_data=reconstructed_field,
                field_type=FieldType.TEMPERATURE,
                metadata=metadata,
                computation_time=time.time() - start_time
            )
            
        except Exception as e:
            return ComputationResult(
                field_data=None,
                field_type=FieldType.TEMPERATURE,
                metadata={},
                error_info=f"POD GA重构失败: {str(e)}",
                computation_time=time.time() - start_time
            )
    
    def get_backend_info(self) -> Dict[str, Any]:
        """获取后端信息"""
        info = super().get_backend_info()
        info.update({
            "pod_api_available": POD_API_AVAILABLE,
            "model_available": self.model_available,
            "model_path": self.model_path,
            "scaler_path": self.scaler_path
        })
        return info
