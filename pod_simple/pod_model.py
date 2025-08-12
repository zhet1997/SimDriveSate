import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle


class SimplePOD:
    """简化的POD模型"""
    
    def __init__(self, energy_threshold=0.95):
        self.energy_threshold = energy_threshold
        self.pca = None
        self.regressor = None
        self.mean_temp = None
        self.n_modes = None
        
    def fit_pod(self, temperatures):
        """POD分解"""
        # 展平温度场
        temp_flat = temperatures.reshape(len(temperatures), -1)
        
        # 计算均值并中心化
        self.mean_temp = np.mean(temp_flat, axis=0)
        temp_centered = temp_flat - self.mean_temp
        
        # PCA确定模态数
        pca_full = PCA()
        pca_full.fit(temp_centered)
        cumulative_energy = np.cumsum(pca_full.explained_variance_ratio_)
        self.n_modes = np.argmax(cumulative_energy >= self.energy_threshold) + 1
        
        # 使用确定的模态数进行PCA
        self.pca = PCA(n_components=self.n_modes)
        pod_coeffs = self.pca.fit_transform(temp_centered)
        
        print(f"POD分解完成: {self.n_modes} 个模态, 能量比 {cumulative_energy[self.n_modes-1]:.4f}")
        return pod_coeffs
    
    def fit_regressor(self, features, pod_coeffs):
        """训练回归器"""
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.regressor.fit(features, pod_coeffs)
        
        pred_coeffs = self.regressor.predict(features)
        mse = mean_squared_error(pod_coeffs, pred_coeffs)
        r2 = r2_score(pod_coeffs, pred_coeffs)
        
        print(f"回归器训练完成: MSE={mse:.4f}, R2={r2:.4f}")
        return {'mse': mse, 'r2': r2}
    
    def predict(self, features):
        """预测温度场"""
        # 预测POD系数
        pod_coeffs = self.regressor.predict(features)
        
        # 重构温度场
        temp_centered = self.pca.inverse_transform(pod_coeffs)
        temp_flat = temp_centered + self.mean_temp
        temperatures = temp_flat.reshape(-1, 256, 256)
        
        return temperatures, pod_coeffs
    
    def evaluate(self, features, true_temps):
        """评估模型"""
        pred_temps, _ = self.predict(features)
        true_flat = true_temps.reshape(len(true_temps), -1)
        pred_flat = pred_temps.reshape(len(pred_temps), -1)
        
        mse = mean_squared_error(true_flat, pred_flat)
        r2 = r2_score(true_flat, pred_flat)
        mae = np.mean(np.abs(true_flat - pred_flat))
        
        metrics = {'mse': mse, 'r2': r2, 'mae': mae}
        print(f"评估结果: MSE={mse:.4f}, R2={r2:.4f}, MAE={mae:.4f}")
        return metrics
    
    def save(self, filepath):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"模型已保存: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"模型已加载: {filepath}")
        return model
    
    def get_pod_bases_at_points(self, measurement_points):
        """获取POD基在指定测点位置的值
        
        Args:
            measurement_points: 测点坐标列表 [(x1, y1), (x2, y2), ...]
            
        Returns:
            measurement_matrix: 形状为 (n_points, n_modes) 的测量矩阵
        """
        if self.pca is None:
            raise ValueError("模型尚未训练，请先调用fit_pod")
        
        n_points = len(measurement_points)
        measurement_matrix = np.zeros((n_points, self.n_modes))
        
        # POD基向量重塑为2D
        pod_bases_2d = self.pca.components_.reshape(self.n_modes, 256, 256)
        
        for i, (x, y) in enumerate(measurement_points):
            # 确保坐标在有效范围内
            x = max(0, min(255, int(x)))
            y = max(0, min(255, int(y)))
            
            # 提取各个模态在该点的值
            for j in range(self.n_modes):
                measurement_matrix[i, j] = pod_bases_2d[j, y, x]
        
        return measurement_matrix
    
    def reconstruct_from_measurements(self, measurement_points, temperatures, method='lstsq', **kwargs):
        """从测点数据重构温度场
        
        Args:
            measurement_points: 测点坐标 [(x1, y1), (x2, y2), ...]
            temperatures: 测点温度值 [T1, T2, ...]
            method: 重构方法 'lstsq' 或 'ga'
            **kwargs: 额外参数
            
        Returns:
            reconstructed_temp: 重构的温度场 (256, 256)
            coefficients: POD系数
        """
        if self.pca is None or self.mean_temp is None:
            raise ValueError("模型尚未训练，请先调用fit_pod")
        
        # 计算测量矩阵
        A = self.get_pod_bases_at_points(measurement_points)
        b = np.array(temperatures)
        
        if method == 'lstsq':
            # 最小二乘法求解
            coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
        elif method == 'ga':
            # 遗传算法求解 (需要额外导入)
            from reconstruction import GAReconstructor
            
            # 提取GA相关参数
            pop_size = kwargs.get('pop_size', 100)
            n_generations = kwargs.get('n_generations', 50)
            coeff_bounds = kwargs.get('coeff_bounds', (-10.0, 10.0))
            
            # 创建GA重构器
            ga_reconstructor = GAReconstructor(self, pop_size, n_generations, coeff_bounds)
            
            # 提取reconstruct方法的参数
            reconstruct_kwargs = {k: v for k, v in kwargs.items() 
                                 if k not in ['pop_size', 'n_generations', 'coeff_bounds']}
            
            coeffs = ga_reconstructor.reconstruct(A, b, **reconstruct_kwargs)
            
        else:
            raise ValueError(f"未知的重构方法: {method}")
        
        # 重构温度场
        temp_centered = self.pca.inverse_transform(coeffs.reshape(1, -1))
        temp_flat = temp_centered + self.mean_temp
        reconstructed_temp = temp_flat.reshape(256, 256)
        
        return reconstructed_temp, coeffs
    
    def validate_reconstruction(self, true_temp, reconstructed_temp, measurement_points):
        """验证重构结果
        
        Args:
            true_temp: 真实温度场 (256, 256)
            reconstructed_temp: 重构温度场 (256, 256)
            measurement_points: 测点坐标
            
        Returns:
            metrics: 评估指标字典
        """
        # 全场误差
        global_mse = np.mean((true_temp - reconstructed_temp) ** 2)
        global_mae = np.mean(np.abs(true_temp - reconstructed_temp))
        global_r2 = r2_score(true_temp.flatten(), reconstructed_temp.flatten())
        
        # 测点误差
        point_errors = []
        for x, y in measurement_points:
            x, y = int(x), int(y)
            if 0 <= x < 256 and 0 <= y < 256:
                error = abs(true_temp[y, x] - reconstructed_temp[y, x])
                point_errors.append(error)
        
        point_mae = np.mean(point_errors) if point_errors else 0
        
        metrics = {
            'global_mse': global_mse,
            'global_mae': global_mae,
            'global_r2': global_r2,
            'point_mae': point_mae,
            'max_error': np.max(np.abs(true_temp - reconstructed_temp)),
            'rms_error': np.sqrt(global_mse)
        }
        
        return metrics