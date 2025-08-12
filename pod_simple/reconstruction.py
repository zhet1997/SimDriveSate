#!/usr/bin/env python3
"""
温度场重构模块
支持基于POD的温度场逆向重构，包括GA启发式搜索和最小二乘法
"""

import numpy as np
from scipy.optimize import least_squares
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any


class MeasurementMatrix:
    """测量矩阵管理类"""
    
    def __init__(self, pod_model):
        """初始化
        
        Args:
            pod_model: 训练好的POD模型
        """
        self.pod_model = pod_model
        self._cached_matrix = None
        self._cached_points = None
    
    def compute_matrix(self, measurement_points: List[Tuple[int, int]]) -> np.ndarray:
        """计算测量矩阵
        
        Args:
            measurement_points: 测点坐标列表 [(x1, y1), (x2, y2), ...]
            
        Returns:
            measurement_matrix: 形状为 (n_points, n_modes) 的测量矩阵
        """
        # 检查是否可以使用缓存
        if (self._cached_matrix is not None and 
            self._cached_points is not None and 
            self._cached_points == measurement_points):
            return self._cached_matrix
        
        # 计算新的测量矩阵
        matrix = self.pod_model.get_pod_bases_at_points(measurement_points)
        
        # 缓存结果
        self._cached_matrix = matrix
        self._cached_points = measurement_points.copy()
        
        return matrix
    
    def clear_cache(self):
        """清除缓存"""
        self._cached_matrix = None
        self._cached_points = None


class GAReconstructionProblem(Problem):
    """GA重构优化问题定义"""
    
    def __init__(self, measurement_matrix: np.ndarray, 
                 measurements: np.ndarray,
                 known_coeffs: Optional[Dict[int, float]] = None,
                 coeff_bounds: Tuple[float, float] = (-10.0, 10.0)):
        """初始化优化问题
        
        Args:
            measurement_matrix: 测量矩阵 A
            measurements: 测点温度 b
            known_coeffs: 已知的POD系数 {index: value}
            coeff_bounds: 系数范围限制
        """
        self.A = measurement_matrix
        self.b = measurements
        self.known_coeffs = known_coeffs or {}
        self.coeff_bounds = coeff_bounds
        
        # 确定需要优化的变量数量
        n_total_coeffs = measurement_matrix.shape[1]
        self.unknown_indices = [i for i in range(n_total_coeffs) 
                               if i not in self.known_coeffs]
        n_vars = len(self.unknown_indices)
        
        # 构建完整系数向量的掩码
        self.full_coeffs = np.zeros(n_total_coeffs)
        for idx, val in self.known_coeffs.items():
            self.full_coeffs[idx] = val
        
        super().__init__(n_var=n_vars,
                        n_obj=1,
                        xl=coeff_bounds[0],
                        xu=coeff_bounds[1])
    
    def _evaluate(self, X, out, *args, **kwargs):
        """评估目标函数
        
        Args:
            X: 决策变量矩阵 (n_pop, n_var)
            out: 输出字典
        """
        n_pop = X.shape[0]
        objectives = np.zeros(n_pop)
        
        for i in range(n_pop):
            # 重建完整的系数向量
            full_coeffs = self.full_coeffs.copy()
            full_coeffs[self.unknown_indices] = X[i]
            
            # 计算重构误差
            pred_temps = self.A @ full_coeffs
            error = np.sum((pred_temps - self.b) ** 2)
            objectives[i] = error
        
        out["F"] = objectives


class GAReconstructor:
    """基于遗传算法的温度场重构器"""
    
    def __init__(self, pod_model, 
                 pop_size: int = 100,
                 n_generations: int = 50,
                 coeff_bounds: Tuple[float, float] = (-10.0, 10.0)):
        """初始化GA重构器
        
        Args:
            pod_model: POD模型
            pop_size: 种群大小
            n_generations: 进化代数
            coeff_bounds: 系数范围
        """
        self.pod_model = pod_model
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.coeff_bounds = coeff_bounds
        self.convergence_history = []
    
    def reconstruct(self, measurement_matrix: np.ndarray,
                   measurements: np.ndarray,
                   known_coeffs: Optional[Dict[int, float]] = None,
                   verbose: bool = False) -> np.ndarray:
        """使用GA重构POD系数
        
        Args:
            measurement_matrix: 测量矩阵
            measurements: 测点温度
            known_coeffs: 已知系数 {模态索引: 系数值}
            verbose: 是否显示详细信息
            
        Returns:
            coefficients: 重构的POD系数
        """
        # 定义优化问题
        problem = GAReconstructionProblem(
            measurement_matrix, measurements, known_coeffs, self.coeff_bounds
        )
        
        # 配置遗传算法
        algorithm = GA(pop_size=self.pop_size)
        termination = get_termination("n_gen", self.n_generations)
        
        # 清空收敛历史
        self.convergence_history = []
        
        # 执行优化
        res = minimize(problem, algorithm, termination, verbose=verbose)
        
        # 提取收敛历史
        if hasattr(res, 'history'):
            self.convergence_history = [entry.opt[0].F[0] for entry in res.history]
        
        # 重建完整系数向量
        full_coeffs = problem.full_coeffs.copy()
        full_coeffs[problem.unknown_indices] = res.X
        
        if verbose:
            print(f"GA重构完成:")
            print(f"  最终误差: {res.F[0]:.6f}")
            print(f"  已知系数数量: {len(known_coeffs) if known_coeffs else 0}")
            print(f"  优化系数数量: {len(problem.unknown_indices)}")
        
        return full_coeffs


class LeastSquaresReconstructor:
    """基于最小二乘的温度场重构器"""
    
    def __init__(self, pod_model):
        """初始化最小二乘重构器
        
        Args:
            pod_model: POD模型
        """
        self.pod_model = pod_model
    
    def reconstruct(self, measurement_matrix: np.ndarray,
                   measurements: np.ndarray,
                   known_coeffs: Optional[Dict[int, float]] = None,
                   regularization: float = 0.0) -> np.ndarray:
        """使用最小二乘重构POD系数
        
        Args:
            measurement_matrix: 测量矩阵 A
            measurements: 测点温度 b
            known_coeffs: 已知系数 {模态索引: 系数值}
            regularization: 正则化参数
            
        Returns:
            coefficients: 重构的POD系数
        """
        n_modes = measurement_matrix.shape[1]
        
        if known_coeffs is None or len(known_coeffs) == 0:
            # 标准最小二乘求解
            if regularization > 0:
                # 带正则化的求解
                A_reg = np.vstack([measurement_matrix, 
                                  np.sqrt(regularization) * np.eye(n_modes)])
                b_reg = np.hstack([measurements, np.zeros(n_modes)])
                coeffs, _, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
            else:
                coeffs, _, _, _ = np.linalg.lstsq(measurement_matrix, measurements, rcond=None)
            
            return coeffs
        
        else:
            # 有已知系数的约束最小二乘
            unknown_indices = [i for i in range(n_modes) if i not in known_coeffs]
            
            # 提取未知系数对应的测量矩阵
            A_unknown = measurement_matrix[:, unknown_indices]
            
            # 计算已知系数的贡献
            known_contribution = np.zeros(len(measurements))
            for idx, val in known_coeffs.items():
                known_contribution += val * measurement_matrix[:, idx]
            
            # 调整测量值
            b_adjusted = measurements - known_contribution
            
            # 求解未知系数
            if regularization > 0:
                n_unknown = len(unknown_indices)
                A_reg = np.vstack([A_unknown, 
                                  np.sqrt(regularization) * np.eye(n_unknown)])
                b_reg = np.hstack([b_adjusted, np.zeros(n_unknown)])
                unknown_coeffs, _, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)
            else:
                unknown_coeffs, _, _, _ = np.linalg.lstsq(A_unknown, b_adjusted, rcond=None)
            
            # 重建完整系数向量
            full_coeffs = np.zeros(n_modes)
            for idx, val in known_coeffs.items():
                full_coeffs[idx] = val
            full_coeffs[unknown_indices] = unknown_coeffs
            
            return full_coeffs


class ReconstructionEvaluator:
    """重构结果评估器"""
    
    def __init__(self, pod_model):
        """初始化评估器
        
        Args:
            pod_model: POD模型
        """
        self.pod_model = pod_model
    
    def evaluate_reconstruction(self, 
                              true_temp: np.ndarray,
                              measurement_points: List[Tuple[int, int]],
                              measurements: np.ndarray,
                              method: str = 'lstsq',
                              known_coeffs: Optional[Dict[int, float]] = None,
                              **kwargs) -> Dict[str, Any]:
        """评估重构性能
        
        Args:
            true_temp: 真实温度场
            measurement_points: 测点坐标
            measurements: 测点温度
            method: 重构方法
            known_coeffs: 已知系数
            **kwargs: 其他参数
            
        Returns:
            evaluation_results: 评估结果字典
        """
        # 计算测量矩阵
        matrix_manager = MeasurementMatrix(self.pod_model)
        A = matrix_manager.compute_matrix(measurement_points)
        
        # 执行重构
        if method == 'lstsq':
            reconstructor = LeastSquaresReconstructor(self.pod_model)
            coeffs = reconstructor.reconstruct(A, measurements, known_coeffs, **kwargs)
        elif method == 'ga':
            reconstructor = GAReconstructor(self.pod_model, **kwargs)
            coeffs = reconstructor.reconstruct(A, measurements, known_coeffs)
        else:
            raise ValueError(f"未知重构方法: {method}")
        
        # 重构温度场
        temp_centered = self.pod_model.pca.inverse_transform(coeffs.reshape(1, -1))
        temp_flat = temp_centered + self.pod_model.mean_temp
        reconstructed_temp = temp_flat.reshape(256, 256)
        
        # 计算评估指标
        metrics = self.pod_model.validate_reconstruction(
            true_temp, reconstructed_temp, measurement_points
        )
        
        # 添加额外信息
        results = {
            'reconstructed_temp': reconstructed_temp,
            'coefficients': coeffs,
            'metrics': metrics,
            'method': method,
            'n_measurements': len(measurement_points),
            'n_known_coeffs': len(known_coeffs) if known_coeffs else 0
        }
        
        # 如果是GA方法，添加收敛信息
        if method == 'ga' and hasattr(reconstructor, 'convergence_history'):
            results['convergence_history'] = reconstructor.convergence_history
        
        return results
    
    def compare_methods(self,
                       true_temp: np.ndarray,
                       measurement_points: List[Tuple[int, int]],
                       measurements: np.ndarray,
                       known_coeffs: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        """比较不同重构方法
        
        Args:
            true_temp: 真实温度场
            measurement_points: 测点坐标
            measurements: 测点温度
            known_coeffs: 已知系数
            
        Returns:
            comparison_results: 比较结果
        """
        results = {}
        
        # 最小二乘方法
        print("评估最小二乘方法...")
        results['lstsq'] = self.evaluate_reconstruction(
            true_temp, measurement_points, measurements, 
            method='lstsq', known_coeffs=known_coeffs
        )
        
        # GA方法
        print("评估遗传算法方法...")
        results['ga'] = self.evaluate_reconstruction(
            true_temp, measurement_points, measurements,
            method='ga', known_coeffs=known_coeffs
        )
        
        # 方法比较
        comparison = {
            'method_comparison': {
                'lstsq_global_mse': results['lstsq']['metrics']['global_mse'],
                'ga_global_mse': results['ga']['metrics']['global_mse'],
                'lstsq_global_mae': results['lstsq']['metrics']['global_mae'],
                'ga_global_mae': results['ga']['metrics']['global_mae'],
                'lstsq_point_mae': results['lstsq']['metrics']['point_mae'],
                'ga_point_mae': results['ga']['metrics']['point_mae']
            },
            'individual_results': results
        }
        
        return comparison


def generate_measurement_points(strategy: str = 'random', 
                              n_points: int = 20,
                              grid_size: Tuple[int, int] = (256, 256),
                              seed: int = 42) -> List[Tuple[int, int]]:
    """生成测点坐标
    
    Args:
        strategy: 采样策略 ('random', 'grid', 'boundary')
        n_points: 测点数量
        grid_size: 网格大小
        seed: 随机种子
        
    Returns:
        measurement_points: 测点坐标列表
    """
    np.random.seed(seed)
    h, w = grid_size
    
    if strategy == 'random':
        # 随机采样
        x_coords = np.random.randint(0, w, n_points)
        y_coords = np.random.randint(0, h, n_points)
        points = list(zip(x_coords, y_coords))
        
    elif strategy == 'grid':
        # 网格采样
        grid_side = int(np.ceil(np.sqrt(n_points)))
        x_step = w // grid_side
        y_step = h // grid_side
        
        points = []
        for i in range(grid_side):
            for j in range(grid_side):
                if len(points) >= n_points:
                    break
                x = min(i * x_step, w - 1)
                y = min(j * y_step, h - 1)
                points.append((x, y))
        
        # 随机化位置稍作调整
        points = points[:n_points]
        
    elif strategy == 'boundary':
        # 边界优先采样
        points = []
        
        # 边界点
        boundary_points = []
        # 上下边界
        for x in range(0, w, w // (n_points // 4 + 1)):
            boundary_points.extend([(x, 0), (x, h-1)])
        # 左右边界  
        for y in range(0, h, h // (n_points // 4 + 1)):
            boundary_points.extend([(0, y), (w-1, y)])
        
        # 随机选择边界点
        n_boundary = min(n_points // 2, len(boundary_points))
        selected_boundary = np.random.choice(len(boundary_points), n_boundary, replace=False)
        points.extend([boundary_points[i] for i in selected_boundary])
        
        # 剩余点随机采样
        n_remaining = n_points - len(points)
        if n_remaining > 0:
            x_coords = np.random.randint(0, w, n_remaining)
            y_coords = np.random.randint(0, h, n_remaining)
            points.extend(zip(x_coords, y_coords))
        
    else:
        raise ValueError(f"未知的采样策略: {strategy}")
    
    return points[:n_points]


def add_noise_to_measurements(measurements: np.ndarray,
                            noise_level: float = 0.05,
                            noise_type: str = 'gaussian') -> np.ndarray:
    """向测量值添加噪声
    
    Args:
        measurements: 原始测量值
        noise_level: 噪声水平（相对于信号幅值）
        noise_type: 噪声类型 ('gaussian', 'uniform')
        
    Returns:
        noisy_measurements: 带噪声的测量值
    """
    signal_std = np.std(measurements)
    noise_std = noise_level * signal_std
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_std, measurements.shape)
    elif noise_type == 'uniform':
        noise_range = noise_std * np.sqrt(12)  # 使均匀分布有相同的标准差
        noise = np.random.uniform(-noise_range/2, noise_range/2, measurements.shape)
    else:
        raise ValueError(f"未知的噪声类型: {noise_type}")
    
    return measurements + noise
