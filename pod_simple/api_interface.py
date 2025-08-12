#!/usr/bin/env python3
"""
POD温度场预测与重构API接口
提供正问题预测和反问题重构的简洁API函数
"""

import numpy as np
import pickle
import json
from typing import List, Tuple, Union, Optional, Dict
from pathlib import Path

from data_utils import extract_features_enhanced
from pod_model import SimplePOD


class PODTemperatureAPI:
    """POD温度场预测与重构API类"""
    
    def __init__(self, model_path: str, scaler_path: str = None):
        """初始化API
        
        Args:
            model_path: POD模型文件路径 (.pkl)
            scaler_path: 特征标准化器路径 (.pkl)，可选
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        
        # 加载模型
        self.pod_model = self._load_model()
        self.scaler = self._load_scaler()
        
        print(f"✓ POD模型已加载: {self.model_path}")
        print(f"✓ POD模态数: {self.pod_model.n_modes}")
        if self.scaler:
            print(f"✓ 特征标准化器已加载: {self.scaler_path}")
        else:
            print("⚠ 未提供标准化器，将跳过特征标准化")
    
    def _load_model(self) -> SimplePOD:
        """加载POD模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"POD模型文件不存在: {self.model_path}")
        
        return SimplePOD.load(str(self.model_path))
    
    def _load_scaler(self):
        """加载特征标准化器"""
        if self.scaler_path is None:
            return None
        
        if not self.scaler_path.exists():
            print(f"⚠ 标准化器文件不存在: {self.scaler_path}")
            return None
        
        with open(self.scaler_path, 'rb') as f:
            return pickle.load(f)
    
    def _extract_features_from_components(self, components: List[List[Dict]]) -> np.ndarray:
        """从组件数据提取特征
        
        Args:
            components: 组件数据列表，每个样本是一个字典列表
        
        Returns:
            features: 特征矩阵 (n_samples, n_features)
        """
        return extract_features_enhanced(components)
    
    def _standardize_features(self, features: np.ndarray) -> np.ndarray:
        """标准化特征
        
        Args:
            features: 原始特征矩阵
        
        Returns:
            standardized_features: 标准化后的特征矩阵
        """
        if self.scaler is None:
            print("⚠ 未加载标准化器，返回原始特征")
            return features
        
        return self.scaler.transform(features)


def predict_temperature_from_json(
    components_data: Union[str, List[List[Dict]]], 
    model_path: str,
    scaler_path: str = None,
    return_coefficients: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    正问题API：从JSON组件数据预测温度场
    
    Args:
        components_data: JSON文件路径或组件数据列表
                        格式: [sample1, sample2, ...] 
                        其中sample = [component1, component2, ...]
                        component = {'shape': 'rect', 'power': 5000, 'center': [x, y], ...}
        model_path: 训练好的POD模型路径
        scaler_path: 特征标准化器路径（可选）
        return_coefficients: 是否返回POD系数
    
    Returns:
        temperature_fields: 预测的温度场 (n_samples, 256, 256)
        pod_coefficients: POD系数 (n_samples, n_modes)，仅当return_coefficients=True时返回
    
    Example:
        >>> # 从JSON文件预测
        >>> temp_fields = predict_temperature_from_json(
        ...     "components.json", 
        ...     "pod_model.pkl", 
        ...     "scaler.pkl"
        ... )
        >>> print(f"预测温度场形状: {temp_fields.shape}")
        
        >>> # 从组件数据直接预测
        >>> components = [[
        ...     {'shape': 'rect', 'power': 5000, 'center': [100, 100], 
        ...      'width': 0.01, 'height': 0.01, 'node_position': [100, 100]},
        ...     # ... 更多组件
        ... ]]
        >>> temp_fields, coeffs = predict_temperature_from_json(
        ...     components, "pod_model.pkl", "scaler.pkl", return_coefficients=True
        ... )
    """
    print("=== POD正问题预测 ===")
    
    # 1. 数据加载
    if isinstance(components_data, str):
        print(f"从JSON文件加载组件数据: {components_data}")
        with open(components_data, 'r', encoding='utf-8') as f:
            components = json.load(f)
    else:
        print("使用提供的组件数据")
        components = components_data
    
    print(f"✓ 组件数据加载完成，样本数: {len(components)}")
    
    # 2. 初始化API
    api = PODTemperatureAPI(model_path, scaler_path)
    
    # 3. 特征提取
    print("正在提取特征...")
    features = api._extract_features_from_components(components)
    print(f"✓ 特征提取完成: {features.shape}")
    
    # 4. 特征标准化
    print("正在标准化特征...")
    features_norm = api._standardize_features(features)
    print(f"✓ 特征标准化完成")
    
    # 5. 温度场预测
    print("正在预测温度场...")
    temperature_fields, pod_coefficients = api.pod_model.predict(features_norm)
    print(f"✓ 温度场预测完成: {temperature_fields.shape}")
    
    # 6. 结果统计
    print(f"\n预测结果统计:")
    print(f"  样本数量: {len(temperature_fields)}")
    print(f"  温度场尺寸: {temperature_fields.shape[1]}×{temperature_fields.shape[2]}")
    print(f"  温度范围: [{temperature_fields.min():.2f}, {temperature_fields.max():.2f}]")
    print(f"  POD系数维度: {pod_coefficients.shape[1]}")
    
    if return_coefficients:
        return temperature_fields, pod_coefficients
    else:
        return temperature_fields


def reconstruct_temperature_from_measurements(
    measurement_points: List[Tuple[float, float]],
    temperature_values: List[float],
    model_path: str,
    method: str = 'lstsq',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    反问题API：从温度测点数据重构完整温度场
    
    Args:
        measurement_points: 测点坐标列表 [(x1, y1), (x2, y2), ...]
                           坐标范围: [0, 255]
        temperature_values: 测点温度值列表 [T1, T2, ...]
        model_path: 训练好的POD模型路径
        method: 重构方法，'lstsq'(最小二乘) 或 'ga'(遗传算法)
        **kwargs: 额外参数
                 对于GA方法:
                   - pop_size: 种群大小 (默认50)
                   - n_generations: 迭代代数 (默认30)
                   - coeff_bounds: 系数范围 (默认(-10, 10))
    
    Returns:
        reconstructed_field: 重构的温度场 (256, 256)
        pod_coefficients: 重构得到的POD系数 (n_modes,)
        validation_metrics: 验证指标字典 (如果提供了真实温度场)
    
    Example:
        >>> # 最小二乘重构
        >>> measurement_points = [(50, 50), (100, 100), (150, 150)]
        >>> temperature_values = [350.5, 345.2, 340.8]
        >>> temp_field, coeffs, metrics = reconstruct_temperature_from_measurements(
        ...     measurement_points, temperature_values, "pod_model.pkl"
        ... )
        >>> print(f"重构温度场形状: {temp_field.shape}")
        
        >>> # GA重构（更精确但较慢）
        >>> temp_field, coeffs, metrics = reconstruct_temperature_from_measurements(
        ...     measurement_points, temperature_values, "pod_model.pkl",
        ...     method='ga', pop_size=100, n_generations=50
        ... )
    """
    print("=== POD反问题重构 ===")
    
    # 1. 参数验证
    if len(measurement_points) != len(temperature_values):
        raise ValueError("测点坐标数量与温度值数量不匹配")
    
    print(f"测点数量: {len(measurement_points)}")
    print(f"温度值范围: [{min(temperature_values):.2f}, {max(temperature_values):.2f}]")
    print(f"重构方法: {method.upper()}")
    
    # 2. 加载模型
    print(f"加载POD模型: {model_path}")
    pod_model = SimplePOD.load(model_path)
    print(f"✓ POD模态数: {pod_model.n_modes}")
    
    # 3. 温度场重构
    print("正在重构温度场...")
    try:
        if method == 'ga':
            # GA方法参数
            ga_params = {
                'pop_size': kwargs.get('pop_size', 50),
                'n_generations': kwargs.get('n_generations', 30),
                'coeff_bounds': kwargs.get('coeff_bounds', (-10.0, 10.0))
            }
            print(f"GA参数: {ga_params}")
            
            reconstructed_field, coefficients = pod_model.reconstruct_from_measurements(
                measurement_points, temperature_values, method='ga', **ga_params
            )
        else:
            # 最小二乘方法
            reconstructed_field, coefficients = pod_model.reconstruct_from_measurements(
                measurement_points, temperature_values, method='lstsq'
            )
        
        print(f"✓ 温度场重构完成: {reconstructed_field.shape}")
        
    except Exception as e:
        print(f"❌ 重构失败: {e}")
        raise
    
    # 4. 验证重构质量（测点精度）
    print("正在验证重构质量...")
    reconstructed_at_points = []
    for x, y in measurement_points:
        x_idx = max(0, min(255, int(x)))
        y_idx = max(0, min(255, int(y)))
        reconstructed_at_points.append(reconstructed_field[y_idx, x_idx])
    
    point_errors = np.abs(np.array(reconstructed_at_points) - np.array(temperature_values))
    
    validation_metrics = {
        'point_mae': np.mean(point_errors),
        'point_max_error': np.max(point_errors),
        'point_correlation': np.corrcoef(temperature_values, reconstructed_at_points)[0, 1],
        'reconstructed_temp_range': [reconstructed_field.min(), reconstructed_field.max()],
        'method': method,
        'n_measurements': len(measurement_points)
    }
    
    # 5. 结果统计
    print(f"\n重构结果统计:")
    print(f"  重构温度场尺寸: {reconstructed_field.shape}")
    print(f"  重构温度范围: [{reconstructed_field.min():.2f}, {reconstructed_field.max():.2f}]")
    print(f"  测点平均误差: {validation_metrics['point_mae']:.4f}")
    print(f"  测点最大误差: {validation_metrics['point_max_error']:.4f}")
    print(f"  测点相关系数: {validation_metrics['point_correlation']:.4f}")
    print(f"  POD系数范围: [{coefficients.min():.3f}, {coefficients.max():.3f}]")
    
    return reconstructed_field, coefficients, validation_metrics


# 便捷函数别名
predict_forward = predict_temperature_from_json
reconstruct_inverse = reconstruct_temperature_from_measurements


if __name__ == "__main__":
    print("POD温度场预测与重构API")
    print("=" * 40)
    print("主要功能:")
    print("1. predict_temperature_from_json() - 正问题预测")
    print("2. reconstruct_temperature_from_measurements() - 反问题重构")
    print()
    print("使用方法请参考函数文档或运行test_api.py")
