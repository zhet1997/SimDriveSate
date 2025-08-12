"""
JSON数据处理器
处理data_generator.py输出的JSON格式组件数据
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .format_converter import DataFormatConverter
from .data_validator import ComponentDataValidator


class JSONComponentHandler:
    """JSON组件数据处理器"""
    
    @staticmethod
    def load_components_from_json(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """从JSON文件加载组件数据
        
        支持两种格式：
        1. data_generator.py输出的单个样本格式
        2. data_generator.py输出的批量样本格式
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            (components, metadata): 组件列表和元数据
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON格式错误: {e}")
        
        # 判断数据格式并解析
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict) and "shape" in data[0]:
                # 格式1: 直接是组件列表 (如components.json)
                components = data
                metadata = {"source": "components_list", "count": len(components)}
            else:
                # 批量样本格式，取第一个样本
                if len(data) > 0:
                    components = data[0]
                    metadata = {"source": "batch_samples", "sample_index": 0, "total_samples": len(data)}
                else:
                    components = []
                    metadata = {"source": "empty_batch"}
        elif isinstance(data, dict):
            if "components" in data:
                # 格式2: 包含完整信息的单个样本
                components = data["components"]
                metadata = {
                    "source": "single_sample",
                    "layout_info": data.get("layout_info", {}),
                    "boundary_conditions": data.get("boundary_conditions", {}),
                    "additional_data": {k: v for k, v in data.items() if k != "components"}
                }
            else:
                raise ValueError("JSON数据格式不正确，缺少components字段")
        else:
            raise ValueError("JSON数据格式不正确，应为列表或字典类型")
        
        # 验证组件数据
        is_valid, error_msg = ComponentDataValidator.validate_data_generator_format(components)
        if not is_valid:
            raise ValueError(f"组件数据验证失败: {error_msg}")
        
        return components, metadata
    
    @staticmethod
    def save_components_to_json(components: List[Dict[str, Any]], 
                              file_path: str,
                              metadata: Optional[Dict[str, Any]] = None,
                              format_type: str = "components_only") -> None:
        """保存组件数据到JSON文件
        
        Args:
            components: 组件列表（data_generator格式）
            file_path: 保存路径
            metadata: 元数据（可选）
            format_type: 保存格式类型 ("components_only" 或 "full_sample")
        """
        # 验证组件数据
        is_valid, error_msg = ComponentDataValidator.validate_data_generator_format(components)
        if not is_valid:
            raise ValueError(f"组件数据验证失败: {error_msg}")
        
        if format_type == "components_only":
            # 只保存组件列表
            save_data = components
        else:
            # 保存完整样本格式
            save_data = {
                "components": components,
                "layout_info": {
                    "size": metadata.get("layout_domain", (0.2, 0.2)),
                    "thermal_conductivity": metadata.get("thermal_conductivity", 1.0),
                    "mesh_resolution": metadata.get("mesh_resolution", (256, 256)),
                    "validity": True,
                    "creation_time": metadata.get("creation_time", "Generated from UI")
                }
            }
            
            # 添加其他元数据
            if metadata:
                for key, value in metadata.items():
                    if key not in ["layout_domain", "thermal_conductivity", "mesh_resolution", "creation_time"]:
                        save_data[key] = value
        
        # 确保目录存在
        dir_path = os.path.dirname(file_path)
        if dir_path:  # 只有当目录路径非空时才创建
            os.makedirs(dir_path, exist_ok=True)
        
        # 保存文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"保存文件失败: {e}")
    
    @staticmethod
    def load_thermal_data(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """加载热仿真数据（温度场、SDF等）
        
        Args:
            file_path: 数据文件路径（.npy或.json）
            
        Returns:
            (data_array, metadata): 数据数组和元数据
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.npy':
            # NumPy数组文件
            data_array = np.load(file_path)
            
            # 尝试加载对应的元数据文件
            metadata_path = file_path.replace('.npy', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {"source": "numpy_file", "shape": data_array.shape}
                
        elif file_ext == '.json':
            # JSON数据文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and "data" in data:
                # 包含数据和元数据的格式
                data_array = np.array(data["data"])
                metadata = data.get("metadata", {})
            else:
                # 直接是数据数组
                data_array = np.array(data)
                metadata = {"source": "json_array"}
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
        
        return data_array, metadata
    
    @staticmethod
    def save_thermal_data(data_array: np.ndarray, 
                         file_path: str,
                         metadata: Optional[Dict[str, Any]] = None,
                         save_format: str = "npy") -> None:
        """保存热仿真数据
        
        Args:
            data_array: 数据数组
            file_path: 保存路径
            metadata: 元数据
            save_format: 保存格式 ("npy" 或 "json")
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if save_format == "npy":
            # 保存为NumPy格式
            np.save(file_path, data_array)
            
            # 保存元数据到单独文件
            if metadata:
                metadata_path = file_path.replace('.npy', '_metadata.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
        else:
            # 保存为JSON格式
            save_data = {
                "data": data_array.tolist(),
                "metadata": metadata or {}
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_batch_samples(directory_path: str) -> List[Tuple[List[Dict], Dict[str, Any]]]:
        """批量加载样本数据
        
        Args:
            directory_path: 样本目录路径
            
        Returns:
            样本列表，每个元素为 (components, metadata)
        """
        samples = []
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        # 查找所有样本目录
        sample_dirs = [d for d in os.listdir(directory_path) 
                      if d.startswith('sample_') and os.path.isdir(os.path.join(directory_path, d))]
        sample_dirs.sort()
        
        for sample_dir in sample_dirs:
            sample_path = os.path.join(directory_path, sample_dir)
            components_file = os.path.join(sample_path, 'components.json')
            
            if os.path.exists(components_file):
                try:
                    components, metadata = JSONComponentHandler.load_components_from_json(components_file)
                    metadata["sample_directory"] = sample_dir
                    
                    # 尝试加载额外的数据文件
                    extra_files = {}
                    for file_name in ['temperature.npy', 'sdf.npy', 'heat_source.npy']:
                        file_path = os.path.join(sample_path, file_name)
                        if os.path.exists(file_path):
                            extra_files[file_name.replace('.npy', '')] = file_path
                    
                    if extra_files:
                        metadata["data_files"] = extra_files
                    
                    samples.append((components, metadata))
                except Exception as e:
                    print(f"警告：加载样本 {sample_dir} 失败: {e}")
        
        return samples
    
    @staticmethod
    def convert_ui_to_json_format(ui_components: List[Dict[str, Any]], 
                                 scene_scale: float = 1000.0) -> List[Dict[str, Any]]:
        """将UI格式转换为JSON保存格式"""
        return DataFormatConverter.ui_to_data_generator(ui_components, scene_scale)
    
    @staticmethod
    def convert_json_to_ui_format(json_components: List[Dict[str, Any]], 
                                 scene_scale: float = 1000.0) -> List[Dict[str, Any]]:
        """将JSON格式转换为UI格式"""
        return DataFormatConverter.data_generator_to_ui(json_components, scene_scale)
