# 简化版POD温度场预测模型

基于POD (Proper Orthogonal Decomposition) 的2D卫星布局温度场预测，使用sklearn和numpy实现。

## 文件结构

```
pod_simple/
├── data_utils.py                    # 数据加载和特征提取
├── pod_model.py                     # POD模型实现（扩展了重构功能）
├── visualization.py                 # 可视化功能
├── train.py                        # 主训练脚本
├── inference.py                    # 模型推理脚本
├── reconstruction.py               # 温度场重构核心模块
├── reconstruction_visualization.py # 重构结果可视化
├── test_reconstruction.py          # 重构功能测试脚本
├── reconstruction_demo.py          # 重构功能演示脚本
├── requirements.txt                # 依赖包（增加了pymoo和scipy）
└── README.md                      # 说明文档
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 训练模型
```bash
python train.py --data_path /data/zxr/inr/SimDriveSate/heat_dataset/all_samples --output_dir results
```

### 3. 查看结果
训练完成后在 `results/run_YYYYMMDD_HHMMSS/` 目录下查看：
- `report.txt`: 实验报告
- `pod_model.pkl`: 训练好的模型
- `train_comparison/`: 训练集预测对比图
- `test_comparison/`: 测试集预测对比图
- `pod_analysis/`: POD模态和能量分析
- `tsne_analysis/`: 特征空间分布分析
- `metrics/`: 性能指标对比

## 模型特征

- **输入**: 54维特征向量（组件形状、尺寸、功率、位置等）
- **输出**: 256×256温度场
- **POD降维**: 自动选择保留95%能量的模态数
- **回归器**: RandomForest回归器

## 参数说明

- `--data_path`: 数据集路径
- `--output_dir`: 结果输出目录
- `--energy_threshold`: POD能量阈值（默认0.95）
- `--test_ratio`: 测试集比例（默认0.2）

## 数据格式

输入数据需包含：
- `all_components.json`: 组件信息
- `all_temperatures.npy`: 温度场数据 (N, 256, 256)

## 新增功能：温度场重构

### 概述
在原有正向预测功能基础上，新增了温度场重构（逆向）功能。通过少量测点的坐标和温度数据，重构出完整的256×256温度场。

### 重构方法
1. **最小二乘法重构**：快速、稳定，适合实时应用
2. **遗传算法重构**：精度更高，支持复杂约束和部分已知系数

### 核心特性
- **部分已知系数支持**：可利用先验物理知识，指定某些POD模态系数为已知值
- **多种采样策略**：随机、网格、边界优先等测点布局策略
- **噪声鲁棒性**：对测量噪声具有良好的抗干扰能力
- **性能评估**：提供全面的重构质量评估指标

### 快速开始重构功能

#### 1. 训练POD模型
```bash
python train.py --data_path /path/to/data --output_dir results
```

#### 2. 演示重构功能
```bash
# 使用默认路径
python reconstruction_demo.py

# 或指定具体路径和输出目录
python reconstruction_demo.py \
    --model_path results/run_YYYYMMDD_HHMMSS/pod_model.pkl \
    --data_path /path/to/your/data \
    --output_dir results/my_demo_results
```

#### 3. 完整功能测试
```bash
python test_reconstruction.py \
    --model_path results/run_YYYYMMDD_HHMMSS/pod_model.pkl \
    --scaler_path results/run_YYYYMMDD_HHMMSS/scaler.pkl \
    --data_path /path/to/data \
    --sample_idx 0
```

### 使用示例

#### 基础重构
```python
from pod_model import SimplePOD
from reconstruction import generate_measurement_points

# 加载模型
pod_model = SimplePOD.load('pod_model.pkl')

# 生成测点（实际使用中这些是已知的传感器位置）
measurement_points = [(50, 100), (150, 80), (200, 200)]  # (x, y)坐标
measurements = [25.5, 28.2, 30.1]  # 对应的温度测量值

# 最小二乘重构
reconstructed_temp, coeffs = pod_model.reconstruct_from_measurements(
    measurement_points, measurements, method='lstsq'
)

# 遗传算法重构
reconstructed_temp_ga, coeffs_ga = pod_model.reconstruct_from_measurements(
    measurement_points, measurements, method='ga',
    pop_size=100, n_generations=50
)
```

#### 部分已知系数重构
```python
# 假设已知前3个POD模态系数
known_coeffs = {0: 2.5, 1: -1.2, 2: 0.8}

# 重构时利用已知信息
reconstructed_temp, coeffs = pod_model.reconstruct_from_measurements(
    measurement_points, measurements, 
    method='ga',
    known_coeffs=known_coeffs
)
```

#### 重构质量评估
```python
# 评估重构结果
metrics = pod_model.validate_reconstruction(
    true_temp, reconstructed_temp, measurement_points
)

print(f"全局MSE: {metrics['global_mse']:.6f}")
print(f"全局R²: {metrics['global_r2']:.6f}")
print(f"测点MAE: {metrics['point_mae']:.6f}")
```

### 可视化功能
```python
from reconstruction_visualization import plot_reconstruction_comparison

# 重构结果对比可视化
plot_reconstruction_comparison(
    true_temp, reconstructed_temp, measurement_points, 
    measurements, metrics, method='LeastSquares',
    save_path='reconstruction_result.png'
)
```

### 性能指标
- **测点需求**：通常20个测点可达到95%重构精度
- **计算速度**：最小二乘法 < 0.1秒，GA算法 < 30秒
- **内存占用**：测量矩阵存储 < 100MB
- **适用场景**：实时监测、缺失数据填补、传感器优化布局