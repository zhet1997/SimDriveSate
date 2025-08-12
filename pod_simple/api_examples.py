#!/usr/bin/env python3
"""
POD温度场预测与重构API使用示例
展示如何使用两个核心API函数的简单示例
"""

import numpy as np
import json
import os
from pathlib import Path
from api_interface import predict_temperature_from_json, reconstruct_temperature_from_measurements


def check_model_files():
    """检查模型文件是否存在，返回可用的模型路径"""
    possible_paths = [
        "results/run_20250812_163834/pod_model.pkl",
        "results/run_20250812_111201/pod_model.pkl"
    ]
    
    # 查找results目录下的所有run文件夹
    results_dir = Path("results")
    if results_dir.exists():
        for run_dir in results_dir.glob("run_*"):
            model_file = run_dir / "pod_model.pkl"
            scaler_file = run_dir / "scaler.pkl"
            if model_file.exists():
                possible_paths.insert(0, str(model_file))
    
    for model_path in possible_paths:
        if Path(model_path).exists():
            scaler_path = model_path.replace("pod_model.pkl", "scaler.pkl")
            print(f"✓ 找到可用模型: {model_path}")
            if Path(scaler_path).exists():
                print(f"✓ 找到标准化器: {scaler_path}")
                return model_path, scaler_path
            else:
                print(f"⚠ 标准化器不存在: {scaler_path}")
                return model_path, None
    
    print("❌ 未找到可用的模型文件")
    print("请先运行 train.py 训练模型，或检查模型路径")
    return None, None


def example_forward_prediction():
    """正问题预测示例"""
    print("=== 正问题预测示例 ===\n")
    
    # 0. 检查模型文件
    model_path, scaler_path = check_model_files()
    if model_path is None:
        return None
    
    # 1. 创建组件数据
    print("1. 创建组件数据...")
    components_data = [[
        {
            'shape': 'rect',
            'power': 6000,
            'center': [100, 100],
            'width': 0.015,
            'height': 0.012,
            'node_position': [100, 100]
        },
        {
            'shape': 'circle', 
            'power': 4500,
            'center': [180, 120],
            'radius': 0.01,
            'node_position': [180, 120]
        },
        {
            'shape': 'capsule',
            'power': 7500,
            'center': [140, 180],
            'length': 0.02,
            'width': 0.01,
            'node_position': [140, 180]
        }
    ]]
    
    print(f"组件数据创建完成：{len(components_data[0])} 个组件")
    
    # 2. 调用正问题API
    print("\n2. 调用正问题预测API...")
    try:
        temperature_fields = predict_temperature_from_json(
            components_data,
            model_path,
            scaler_path
        )
        
        print(f"✓ 预测成功！温度场形状: {temperature_fields.shape}")
        print(f"  温度范围: [{temperature_fields.min():.2f}, {temperature_fields.max():.2f}]")
        
        return temperature_fields[0]  # 返回第一个样本的温度场
        
    except FileNotFoundError as e:
        print(f"❌ 模型文件未找到: {e}")
        print("请确保已经训练了模型并提供正确的路径")
        return None
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return None


def example_inverse_reconstruction():
    """反问题重构示例"""
    print("\n=== 反问题重构示例 ===\n")
    
    # 0. 检查模型文件
    model_path, _ = check_model_files()
    if model_path is None:
        return None, None
    
    # 1. 创建测点数据
    print("1. 创建测点数据...")
    measurement_points = [
        (50.0, 50.0),
        (100.0, 80.0),
        (150.0, 60.0),
        (80.0, 120.0),
        (120.0, 140.0),
        (160.0, 120.0),
        (70.0, 180.0),
        (130.0, 200.0),
        (190.0, 180.0)
    ]
    
    # 模拟测点温度值
    temperature_values = [342.5, 355.2, 348.8, 350.1, 358.7, 352.3, 345.6, 351.9, 347.4]
    
    print(f"测点数量: {len(measurement_points)}")
    print(f"温度范围: [{min(temperature_values):.1f}, {max(temperature_values):.1f}]")
    
    # 2. 调用反问题API - GA方法
    print("\n2. 遗传算法重构...")
    try:
        temp_field_ga, coeffs_ga, metrics_ga = reconstruct_temperature_from_measurements(
            measurement_points,
            temperature_values,
            model_path,
            method='ga',
            pop_size=30,  # 减少种群大小以加快速度
            n_generations=20  # 减少代数以加快速度
        )
        
        print(f"✓ GA重构成功！")
        print(f"  重构温度场形状: {temp_field_ga.shape}")
        print(f"  测点平均误差: {metrics_ga['point_mae']:.4f}")
        print(f"  测点相关系数: {metrics_ga['point_correlation']:.4f}")
        
        return temp_field_ga
        
    except Exception as e:
        print(f"❌ GA重构失败: {e}")
        return None


def example_json_file_prediction():
    """从JSON文件预测的示例"""
    print("\n=== 从JSON文件预测示例 ===\n")
    
    # 1. 创建JSON文件
    print("1. 创建测试JSON文件...")
    
    test_data = [
        # 样本1
        [
            {
                'shape': 'rect',
                'power': 5000,
                'center': [64, 64],
                'width': 0.012,
                'height': 0.012,
                'node_position': [64, 64]
            },
            {
                'shape': 'circle',
                'power': 6000,
                'center': [192, 64],
                'radius': 0.01,
                'node_position': [192, 64]
            }
        ],
        # 样本2
        [
            {
                'shape': 'capsule',
                'power': 8000,
                'center': [128, 128],
                'length': 0.025,
                'width': 0.012,
                'node_position': [128, 128]
            }
        ]
    ]
    
    json_filename = "example_components.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON文件已创建: {json_filename}")
    print(f"  包含 {len(test_data)} 个样本")
    
    # 2. 从JSON文件预测
    print("\n2. 从JSON文件预测...")
    
    # 检查模型文件
    model_path, scaler_path = check_model_files()
    if model_path is None:
        return None
    
    try:
        
        temperature_fields, pod_coeffs = predict_temperature_from_json(
            json_filename,
            model_path,
            scaler_path,
            return_coefficients=True
        )
        
        print(f"✓ 从JSON预测成功！")
        print(f"  温度场形状: {temperature_fields.shape}")
        print(f"  POD系数形状: {pod_coeffs.shape}")
        
        # 清理文件
        import os
        os.remove(json_filename)
        print(f"✓ 临时JSON文件已删除")
        
        return temperature_fields
        
    except Exception as e:
        print(f"❌ JSON预测失败: {e}")
        return None


def complete_workflow_example():
    """完整工作流示例：正问题 → 采样 → 反问题"""
    print("\n=== 完整工作流示例 ===\n")
    print("流程：组件数据 → 正问题预测 → 采样测点 → 反问题重构")
    
    # 1. 正问题预测
    print("\n步骤1: 正问题预测...")
    predicted_temp = example_forward_prediction()
    if predicted_temp is None:
        print("❌ 正问题预测失败，无法继续工作流")
        return
    
    # 2. 从预测结果中采样测点
    print("\n步骤2: 从预测结果采样测点...")
    np.random.seed(42)
    n_measurements = 12
    
    measurement_points = []
    temperature_values = []
    
    for _ in range(n_measurements):
        x = np.random.randint(20, 236)
        y = np.random.randint(20, 236)
        temp = predicted_temp[y, x]
        
        measurement_points.append((float(x), float(y)))
        temperature_values.append(float(temp))
    
    print(f"✓ 采样完成：{len(measurement_points)} 个测点")
    print(f"  温度范围: [{min(temperature_values):.2f}, {max(temperature_values):.2f}]")
    
    # 3. 反问题重构
    print("\n步骤3: 反问题重构...")
    
    # 检查模型文件
    model_path, _ = check_model_files()
    if model_path is None:
        print("❌ 无法找到模型文件，无法继续重构")
        return
    
    try:
        
        reconstructed_temp, coeffs, metrics = reconstruct_temperature_from_measurements(
            measurement_points,
            temperature_values,
            model_path,
            method='ga',
            pop_size=30,
            n_generations=20
        )
        
        # 4. 评估重构质量
        print("\n步骤4: 评估重构质量...")
        global_mse = np.mean((predicted_temp - reconstructed_temp) ** 2)
        global_mae = np.mean(np.abs(predicted_temp - reconstructed_temp))
        correlation = np.corrcoef(predicted_temp.flatten(), reconstructed_temp.flatten())[0, 1]
        
        print(f"✓ 重构质量评估:")
        print(f"  全局MSE: {global_mse:.6f}")
        print(f"  全局MAE: {global_mae:.6f}")
        print(f"  相关系数: {correlation:.6f}")
        print(f"  测点MAE: {metrics['point_mae']:.6f}")
        
        print(f"\n✓ 完整工作流演示成功！")
        print(f"  证明了从组件数据到温度场预测，再到反向重构的完整流程")
        
    except Exception as e:
        print(f"❌ 反问题重构失败: {e}")


def main():
    """主函数：运行所有示例"""
    print("POD温度场预测与重构API使用示例")
    print("=" * 60)
    print("本脚本展示了两个主要API函数的使用方法：")
    print("1. predict_temperature_from_json() - 正问题预测")
    print("2. reconstruct_temperature_from_measurements() - 反问题重构")
    print("\n注意：运行前请确保已有训练好的模型文件！")
    print("=" * 60)
    
    # 运行示例
    try:
        # 示例1：正问题预测
        print("\n开始示例1：正问题预测...")
        result1 = example_forward_prediction()
        
        # 示例2：反问题重构
        print("\n开始示例2：反问题重构...")
        result2 = example_inverse_reconstruction()
        
        # 示例3：JSON文件预测
        print("\n开始示例3：JSON文件预测...")
        result3 = example_json_file_prediction()
        
        # 示例4：完整工作流（仅在前面的示例成功时运行）
        if result1 is not None:
            print("\n开始示例4：完整工作流...")
            complete_workflow_example()
        else:
            print("\n跳过示例4：由于正问题预测失败")
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n❌ 示例执行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("示例演示完成！")
    print("\n如需实际使用，请：")
    print("1. 确保模型路径正确")
    print("2. 准备符合格式的组件数据")
    print("3. 参考示例代码进行调用")


if __name__ == "__main__":
    main()
