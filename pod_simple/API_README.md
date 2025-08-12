# POD温度场预测与重构API

基于POD（Proper Orthogonal Decomposition）的温度场预测和重构API接口，提供正问题预测和反问题重构功能。

## 核心功能

### 1. 正问题预测 (Forward Problem)
从JSON组件数据预测完整的温度场分布

### 2. 反问题重构 (Inverse Problem)  
从稀疏的温度测点数据重构完整的温度场

## API函数

### `predict_temperature_from_json()`

**功能**：正问题预测 - 从组件数据预测温度场

**输入**：
- `components_data`: JSON文件路径或组件数据列表
- `model_path`: 训练好的POD模型路径
- `scaler_path`: 特征标准化器路径（可选）
- `return_coefficients`: 是否返回POD系数

**输出**：
- `temperature_fields`: 预测的温度场 (n_samples, 256, 256)
- `pod_coefficients`: POD系数（可选）

**使用示例**：
```python
from api_interface import predict_temperature_from_json

# 从JSON文件预测
temp_fields = predict_temperature_from_json(
    "components.json", 
    "pod_model.pkl", 
    "scaler.pkl"
)

# 从组件数据直接预测
components = [[
    {
        'shape': 'rect',
        'power': 5000,
        'center': [100, 100],
        'width': 0.01,
        'height': 0.01,
        'node_position': [100, 100]
    }
]]

temp_fields, coeffs = predict_temperature_from_json(
    components, 
    "pod_model.pkl", 
    "scaler.pkl",
    return_coefficients=True
)
```

### `reconstruct_temperature_from_measurements()`

**功能**：反问题重构 - 从测点数据重构温度场

**输入**：
- `measurement_points`: 测点坐标列表 [(x1, y1), (x2, y2), ...]
- `temperature_values`: 测点温度值列表 [T1, T2, ...]
- `model_path`: 训练好的POD模型路径
- `method`: 重构方法 ('lstsq' 或 'ga')
- `**kwargs`: 额外参数（GA方法需要）

**输出**：
- `reconstructed_field`: 重构的温度场 (256, 256)
- `pod_coefficients`: POD系数
- `validation_metrics`: 验证指标字典

**使用示例**：
```python
from api_interface import reconstruct_temperature_from_measurements

# GA重构（精确的遗传算法重构）
measurement_points = [(50, 50), (100, 100), (150, 150)]
temperature_values = [350.5, 345.2, 340.8]

temp_field, coeffs, metrics = reconstruct_temperature_from_measurements(
    measurement_points,
    temperature_values,
    "pod_model.pkl",
    method='ga',
    pop_size=50,
    n_generations=30
)

# 最小二乘重构（仍然支持，但示例中不使用）
temp_field, coeffs, metrics = reconstruct_temperature_from_measurements(
    measurement_points,
    temperature_values,
    "pod_model.pkl",
    method='lstsq'
)
```

## 数据格式

### 组件数据格式
```json
[
  [
    {
      "shape": "rect",
      "power": 5000,
      "center": [100, 100],
      "width": 0.01,
      "height": 0.01,
      "node_position": [100, 100]
    },
    {
      "shape": "circle", 
      "power": 4000,
      "center": [150, 150],
      "radius": 0.008,
      "node_position": [150, 150]
    }
  ]
]
```

### 测点数据格式
- **测点坐标**：(x, y) 元组列表，坐标范围 [0, 255]
- **温度值**：对应的温度值列表

## 文件说明

### 核心文件
- `api_interface.py` - 主要API接口
- `test_api.py` - API测试脚本
- `api_examples.py` - 使用示例
- `data_utils.py` - 数据处理工具
- `pod_model.py` - POD模型定义

### 依赖文件
- POD模型文件 (`.pkl`)
- 特征标准化器 (`.pkl`) - 可选但推荐

## 快速开始

### 1. 训练模型
```bash
cd pod_simple
python train.py --data_path /your/data/path --output_dir results
```

### 2. 测试API
```bash
python test_api.py --model_path results/run_YYYYMMDD_HHMMSS/pod_model.pkl \
                   --scaler_path results/run_YYYYMMDD_HHMMSS/scaler.pkl
```

### 3. 运行示例
```bash
python api_examples.py
```

## 完整工作流示例

```python
from api_interface import predict_temperature_from_json, reconstruct_temperature_from_measurements
import numpy as np

# 1. 正问题预测
components = [[
    {'shape': 'rect', 'power': 6000, 'center': [100, 100], 
     'width': 0.01, 'height': 0.01, 'node_position': [100, 100]}
]]

temp_field_predicted = predict_temperature_from_json(
    components, "pod_model.pkl", "scaler.pkl"
)[0]

# 2. 从预测结果采样测点
measurement_points = [(50, 50), (100, 100), (150, 150)]
temperature_values = [
    temp_field_predicted[50, 50],
    temp_field_predicted[100, 100], 
    temp_field_predicted[150, 150]
]

# 3. 反问题重构（使用GA方法）
temp_field_reconstructed, coeffs, metrics = reconstruct_temperature_from_measurements(
    measurement_points,
    temperature_values,
    "pod_model.pkl",
    method='ga',
    pop_size=30,
    n_generations=20
)

# 4. 评估重构质量
mse = np.mean((temp_field_predicted - temp_field_reconstructed) ** 2)
print(f"重构MSE: {mse:.6f}")
```

## 注意事项

1. **模型路径**：确保POD模型文件存在且路径正确
2. **数据格式**：组件数据必须包含必要字段
3. **坐标系统**：测点坐标使用像素坐标系 [0, 255]
4. **性能**：GA方法更精确但计算时间较长，示例中优化了参数以平衡精度和速度
5. **标准化**：使用与训练时相同的标准化器

## 错误处理

- **FileNotFoundError**：检查模型文件路径
- **ValueError**：检查数据格式和维度
- **计算错误**：检查测点数量和模型兼容性

## 性能优化建议

1. **正问题预测**：
   - 批量处理多个样本
   - 使用GPU加速（如适用）

2. **反问题重构**：
   - 主要使用GA方法获得更精确的重构
   - 适当调整种群大小（建议30-50）和代数（建议20-30）
   - 测点数量建议10-30个
   - 可选择最小二乘方法进行快速重构

## 扩展功能

- 支持部分已知POD系数的重构
- 多种测点采样策略
- 噪声处理和鲁棒性
- 重构质量评估指标
