#!/usr/bin/env python3
"""
温度场重构功能演示脚本
展示如何使用重构功能的简单示例
"""

import numpy as np
import matplotlib.pyplot as plt
from pod_model import SimplePOD
from reconstruction import (
    MeasurementMatrix, GAReconstructor, LeastSquaresReconstructor,
    ReconstructionEvaluator, generate_measurement_points
)
from reconstruction_visualization import plot_reconstruction_comparison, plot_measurement_points_analysis
import pickle
import os


def save_original_temperature_field(true_temp, sample_idx, output_dir):
    """保存原始温度场可视化"""
    plt.figure(figsize=(10, 8))
    
    im = plt.imshow(true_temp, cmap='viridis', origin='lower')
    plt.colorbar(im, label='Temperature')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Original Temperature Field (Sample {sample_idx})')
    plt.grid(True, alpha=0.3)
    
    # 添加温度统计信息
    info_text = f"""
    Statistics:
    Min: {true_temp.min():.2f}
    Max: {true_temp.max():.2f}
    Mean: {np.mean(true_temp):.2f}
    Std: {np.std(true_temp):.2f}
    """
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    save_path = os.path.join(output_dir, '01_original_temperature_field.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   原始温度场已保存: {save_path}")


def save_measurement_layout(measurement_points, measurements, true_temp, output_dir):
    """保存测点布局可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：测点位置在温度场上
    im1 = axes[0].imshow(true_temp, cmap='viridis', alpha=0.7, origin='lower')
    
    x_coords = [p[0] for p in measurement_points]
    y_coords = [p[1] for p in measurement_points]
    
    scatter = axes[0].scatter(x_coords, y_coords, c=measurements, s=100, 
                             cmap='plasma', edgecolor='white', linewidth=2)
    axes[0].set_xlabel('X Coordinate')
    axes[0].set_ylabel('Y Coordinate')
    axes[0].set_title('Measurement Points Layout on Temperature Field')
    
    # 添加测点编号
    for i, (x, y) in enumerate(measurement_points[:10]):  # 只标注前10个点
        axes[0].annotate(f'{i+1}', (x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, 
                        color='white', fontweight='bold')
    
    plt.colorbar(im1, ax=axes[0], label='Background Temperature')
    
    # 右图：测点温度分布
    bars = axes[1].bar(range(1, len(measurements)+1), measurements, 
                      color='skyblue', alpha=0.7, edgecolor='navy')
    axes[1].set_xlabel('Measurement Point Index')
    axes[1].set_ylabel('Temperature')
    axes[1].set_title('Temperature Values at Measurement Points')
    axes[1].grid(True, alpha=0.3)
    
    # 标注统计信息
    stats_text = f"""
    Points: {len(measurements)}
    Min: {measurements.min():.2f}
    Max: {measurements.max():.2f}
    Mean: {np.mean(measurements):.2f}
    """
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, '02_measurement_points_layout.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   测点布局已保存: {save_path}")


def save_reconstruction_results(true_temp, reconstructed_temp, measurement_points, 
                               measurements, metrics, method, output_dir, step_num):
    """保存重构结果详细对比"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 温度范围
    vmin = min(true_temp.min(), reconstructed_temp.min())
    vmax = max(true_temp.max(), reconstructed_temp.max())
    
    # 真实温度场
    im1 = axes[0, 0].imshow(true_temp, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 0].set_title('Original Temperature Field', fontsize=14, fontweight='bold')
    
    # 标注测点
    x_coords = [p[0] for p in measurement_points]
    y_coords = [p[1] for p in measurement_points]
    axes[0, 0].scatter(x_coords, y_coords, c='red', s=30, marker='o', 
                      edgecolor='white', linewidth=1, alpha=0.8)
    axes[0, 0].set_xlabel('X Coordinate')
    axes[0, 0].set_ylabel('Y Coordinate')
    plt.colorbar(im1, ax=axes[0, 0], label='Temperature')
    
    # 重构温度场
    im2 = axes[0, 1].imshow(reconstructed_temp, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 1].set_title(f'Reconstructed Field ({method})', fontsize=14, fontweight='bold')
    axes[0, 1].scatter(x_coords, y_coords, c='red', s=30, marker='o',
                      edgecolor='white', linewidth=1, alpha=0.8)
    axes[0, 1].set_xlabel('X Coordinate')
    axes[0, 1].set_ylabel('Y Coordinate')
    plt.colorbar(im2, ax=axes[0, 1], label='Temperature')
    
    # 绝对误差
    abs_error = np.abs(true_temp - reconstructed_temp)
    im3 = axes[0, 2].imshow(abs_error, cmap='Reds', origin='lower')
    axes[0, 2].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('X Coordinate')
    axes[0, 2].set_ylabel('Y Coordinate')
    plt.colorbar(im3, ax=axes[0, 2], label='Absolute Error')
    
    # 相对误差
    relative_error = np.abs(true_temp - reconstructed_temp) / (np.abs(true_temp) + 1e-8)
    relative_error = np.clip(relative_error, 0, 1)
    im4 = axes[1, 0].imshow(relative_error, cmap='Oranges', origin='lower')
    axes[1, 0].set_title('Relative Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('X Coordinate')
    axes[1, 0].set_ylabel('Y Coordinate')
    plt.colorbar(im4, ax=axes[1, 0], label='Relative Error')
    
    # 残差
    residual = reconstructed_temp - true_temp
    max_res = max(abs(residual.min()), abs(residual.max()))
    im5 = axes[1, 1].imshow(residual, cmap='coolwarm', vmin=-max_res, vmax=max_res, origin='lower')
    axes[1, 1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('X Coordinate')
    axes[1, 1].set_ylabel('Y Coordinate')
    plt.colorbar(im5, ax=axes[1, 1], label='Residual')
    
    # 性能指标
    axes[1, 2].axis('off')
    
    metrics_text = f"""
    Reconstruction Performance ({method}):
    
    Global MSE: {metrics.get('global_mse', 0):.6f}
    Global MAE: {metrics.get('global_mae', 0):.6f}
    Global R²: {metrics.get('global_r2', 0):.6f}
    
    Point MAE: {metrics.get('point_mae', 0):.6f}
    Max Error: {metrics.get('max_error', 0):.6f}
    RMS Error: {metrics.get('rms_error', 0):.6f}
    
    Measurement Points: {len(measurement_points)}
    Method: {method}
    """
    
    axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'{step_num:02d}_reconstruction_{method.lower()}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   {method}重构结果已保存: {save_path}")


def save_point_accuracy_analysis(measurement_points, measurements, reconstructed_temp, 
                                method, output_dir, step_num):
    """保存测点精度分析"""
    # 提取重构温度在测点位置的值
    reconstructed_at_points = []
    for x, y in measurement_points:
        x_idx = max(0, min(255, int(x)))
        y_idx = max(0, min(255, int(y)))
        reconstructed_at_points.append(reconstructed_temp[y_idx, x_idx])
    
    reconstructed_at_points = np.array(reconstructed_at_points)
    errors = np.abs(reconstructed_at_points - measurements)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 测点温度对比
    axes[0].scatter(measurements, reconstructed_at_points, alpha=0.7, s=50)
    min_val = min(measurements.min(), reconstructed_at_points.min())
    max_val = max(measurements.max(), reconstructed_at_points.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Match')
    axes[0].set_xlabel('Original Temperature')
    axes[0].set_ylabel('Reconstructed Temperature')
    axes[0].set_title(f'Point-wise Accuracy ({method})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 计算相关性
    correlation = np.corrcoef(measurements, reconstructed_at_points)[0, 1]
    axes[0].text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                transform=axes[0].transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 误差分布
    axes[1].bar(range(1, len(errors)+1), errors, alpha=0.7, color='orange')
    axes[1].set_xlabel('Measurement Point Index')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title('Point-wise Reconstruction Errors')
    axes[1].grid(True, alpha=0.3)
    
    # 误差统计直方图
    axes[2].hist(errors, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
    axes[2].set_xlabel('Absolute Error')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Error Distribution Histogram')
    axes[2].grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f"""
    Error Statistics:
    Mean: {np.mean(errors):.4f}
    Std: {np.std(errors):.4f}
    Max: {np.max(errors):.4f}
    Min: {np.min(errors):.4f}
    """
    axes[2].text(0.02, 0.98, stats_text, transform=axes[2].transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{step_num:02d}_point_accuracy_{method.lower()}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   {method}测点精度分析已保存: {save_path}")


def save_method_comparison(metrics_lstsq, metrics_ga, output_dir):
    """保存方法对比可视化"""
    methods = ['LeastSquares', 'GA']
    metric_names = ['global_mse', 'global_mae', 'global_r2', 'point_mae']
    metric_labels = ['Global MSE', 'Global MAE', 'Global R²', 'Point MAE']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        values = [metrics_lstsq[metric], metrics_ga[metric]]
        
        bars = axes[i].bar(methods, values, alpha=0.7, 
                          color=['skyblue', 'lightcoral'])
        axes[i].set_ylabel(label)
        axes[i].set_title(f'{label} Comparison')
        axes[i].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.6f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, '07_method_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   方法对比图已保存: {save_path}")


def create_demo_summary_report(output_dir, sample_idx, measurement_points, measurements,
                              metrics_lstsq=None, metrics_ga=None):
    """创建演示总结报告"""
    from datetime import datetime
    
    report_path = os.path.join(output_dir, 'demo_summary_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Temperature Field Reconstruction Demo Summary Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Index: {sample_idx}\n")
        f.write(f"Measurement Points: {len(measurement_points)}\n\n")
        
        f.write("Measurement Points Details:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Count: {len(measurement_points)}\n")
        f.write(f"Temperature Range: [{measurements.min():.2f}, {measurements.max():.2f}]\n")
        f.write(f"Temperature Mean: {np.mean(measurements):.2f}\n")
        f.write(f"Temperature Std: {np.std(measurements):.2f}\n\n")
        
        if metrics_lstsq:
            f.write("LeastSquares Method Results:\n")
            f.write("-" * 30 + "\n")
            for key, val in metrics_lstsq.items():
                f.write(f"  {key}: {val:.6f}\n")
            f.write("\n")
        
        if metrics_ga:
            f.write("GA Method Results:\n")
            f.write("-" * 30 + "\n")
            for key, val in metrics_ga.items():
                f.write(f"  {key}: {val:.6f}\n")
            f.write("\n")
        
        if metrics_lstsq and metrics_ga:
            f.write("Method Comparison:\n")
            f.write("-" * 30 + "\n")
            if metrics_ga['global_mse'] < metrics_lstsq['global_mse']:
                f.write("Winner: GA method (lower MSE)\n")
            else:
                f.write("Winner: LeastSquares method (lower MSE)\n")
        
        f.write("\nGenerated Files:\n")
        f.write("-" * 30 + "\n")
        f.write("01_original_temperature_field.png - Original temperature field\n")
        f.write("02_measurement_points_layout.png - Measurement points layout\n")
        f.write("03_reconstruction_leastsquares.png - LeastSquares reconstruction\n")
        f.write("04_point_accuracy_leastsquares.png - LeastSquares point accuracy\n")
        f.write("05_reconstruction_ga.png - GA reconstruction\n")
        f.write("06_point_accuracy_ga.png - GA point accuracy\n")
        f.write("07_method_comparison.png - Method comparison\n")
        f.write("demo_summary_report.txt - This summary report\n")
    
    print(f"   演示总结报告已保存: {report_path}")


def demo_basic_reconstruction(model_path=None, data_path=None, output_dir=None):
    """基础重构功能演示
    
    Args:
        model_path: POD模型路径
        data_path: 数据路径
        output_dir: 输出目录
    """
    print("=== 基础温度场重构演示 ===\n")
    
    # 创建输出目录
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"results/reconstruction_demo_{timestamp}"
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    print(f"演示结果将保存到: {output_dir}\n")
    
    # 1. 加载训练好的POD模型
    print("1. 加载POD模型...")
    if model_path is None:
        model_path = "results/run_20250812_111201/pod_model.pkl"
    
    try:
        pod_model = SimplePOD.load(model_path)
        print(f"   模型加载成功，POD模态数: {pod_model.n_modes}")
    except FileNotFoundError:
        print(f"   错误：找不到模型文件 {model_path}")
        print("   请先运行 train.py 训练模型")
        return
    
    # 2. 加载真实数据样本
    print("\n2. 加载真实数据样本...")
    try:
        from data_utils import load_data
        if data_path is None:
            data_path = '/data/zxr/inr/SimDriveSate/heat_dataset/all_samples'
        
        components, temperatures = load_data(data_path)
        
        # 随机选择一个样本作为测试
        np.random.seed(42)
        sample_idx = np.random.randint(0, len(temperatures))
        true_temp = temperatures[sample_idx]
        
        print(f"   数据加载成功，总样本数: {len(temperatures)}")
        print(f"   选择样本索引: {sample_idx}")
        print(f"   温度场形状: {true_temp.shape}")
        print(f"   温度范围: [{true_temp.min():.2f}, {true_temp.max():.2f}]")
        
        # 保存原始温度场
        save_original_temperature_field(true_temp, sample_idx, output_dir)
        
    except Exception as e:
        print(f"   数据加载失败: {e}")
        print("   使用合成数据进行演示...")
        # 备用方案：使用合成数据
        x = np.linspace(0, 2*np.pi, 256)
        y = np.linspace(0, 2*np.pi, 256)
        X, Y = np.meshgrid(x, y)
        true_temp = (50 + 20 * np.sin(X) * np.cos(Y) + 
                    15 * np.sin(2*X) + 10 * np.cos(3*Y) +
                    5 * np.random.normal(0, 1, (256, 256)))
        print(f"   合成温度场形状: {true_temp.shape}")
        print(f"   温度范围: [{true_temp.min():.2f}, {true_temp.max():.2f}]")
        
        # 保存合成温度场
        save_original_temperature_field(true_temp, "synthetic", output_dir)
    
    # 3. 生成测点并从真实温度场采样
    print("\n3. 生成测点并采样...")
    n_points = 20
    measurement_points = generate_measurement_points('random', n_points, seed=42)
    
    # 从真实温度场中提取测点温度
    measurements = []
    for x, y in measurement_points:
        x_idx = max(0, min(255, int(x)))
        y_idx = max(0, min(255, int(y)))
        measurements.append(true_temp[y_idx, x_idx])
    measurements = np.array(measurements)
    
    print(f"   测点数量: {len(measurement_points)}")
    print(f"   测点温度范围: [{measurements.min():.2f}, {measurements.max():.2f}]")
    
    # 显示一些测点的具体信息
    print("   前5个测点信息:")
    for i in range(min(5, len(measurement_points))):
        x, y = measurement_points[i]
        temp = measurements[i]
        print(f"     测点{i+1}: 位置({x}, {y}), 温度={temp:.2f}")
    
    # 保存测点布局可视化
    save_measurement_layout(measurement_points, measurements, true_temp, output_dir)
    
    # 4. 最小二乘重构
    print("\n4. 最小二乘重构...")
    try:
        reconstructed_temp, coeffs = pod_model.reconstruct_from_measurements(
            measurement_points, measurements, method='lstsq'
        )
        
        # 评估重构质量
        metrics = pod_model.validate_reconstruction(
            true_temp, reconstructed_temp, measurement_points
        )
        
        print(f"   重构完成！")
        print(f"   全局MSE: {metrics['global_mse']:.6f}")
        print(f"   全局MAE: {metrics['global_mae']:.6f}")
        print(f"   全局R²: {metrics['global_r2']:.6f}")
        print(f"   测点MAE: {metrics['point_mae']:.6f}")
        
        # 验证测点精度
        reconstructed_at_points = []
        for x, y in measurement_points:
            x_idx = max(0, min(255, int(x)))
            y_idx = max(0, min(255, int(y)))
            reconstructed_at_points.append(reconstructed_temp[y_idx, x_idx])
        
        point_errors = np.abs(np.array(reconstructed_at_points) - measurements)
        print(f"   测点重构误差统计: 平均={np.mean(point_errors):.4f}, 最大={np.max(point_errors):.4f}")
        
        # 保存详细重构结果
        save_reconstruction_results(true_temp, reconstructed_temp, measurement_points, 
                                   measurements, metrics, "LeastSquares", output_dir, 3)
        
        # 保存测点精度分析
        save_point_accuracy_analysis(measurement_points, measurements, reconstructed_temp,
                                    "LeastSquares", output_dir, 4)
        
    except Exception as e:
        print(f"   重构失败: {e}")
        return
    
    # 5. GA重构（如果时间允许）
    print("\n5. 遗传算法重构...")
    try:
        reconstructed_temp_ga, coeffs_ga = pod_model.reconstruct_from_measurements(
            measurement_points, measurements, method='ga',
            pop_size=50, n_generations=30
        )
        
        metrics_ga = pod_model.validate_reconstruction(
            true_temp, reconstructed_temp_ga, measurement_points
        )
        
        print(f"   GA重构完成！")
        print(f"   全局MSE: {metrics_ga['global_mse']:.6f}")
        print(f"   全局MAE: {metrics_ga['global_mae']:.6f}")
        print(f"   全局R²: {metrics_ga['global_r2']:.6f}")
        print(f"   测点MAE: {metrics_ga['point_mae']:.6f}")
        
        # 验证GA重构的测点精度
        reconstructed_ga_at_points = []
        for x, y in measurement_points:
            x_idx = max(0, min(255, int(x)))
            y_idx = max(0, min(255, int(y)))
            reconstructed_ga_at_points.append(reconstructed_temp_ga[y_idx, x_idx])
        
        point_errors_ga = np.abs(np.array(reconstructed_ga_at_points) - measurements)
        print(f"   GA测点重构误差统计: 平均={np.mean(point_errors_ga):.4f}, 最大={np.max(point_errors_ga):.4f}")
        
        # 保存详细重构结果
        save_reconstruction_results(true_temp, reconstructed_temp_ga, measurement_points,
                                   measurements, metrics_ga, "GA", output_dir, 5)
        
        # 保存测点精度分析
        save_point_accuracy_analysis(measurement_points, measurements, reconstructed_temp_ga,
                                    "GA", output_dir, 6)
        
        # 方法对比
        print(f"\n6. 方法对比:")
        print(f"   最小二乘 - 全局MSE: {metrics['global_mse']:.6f}, R²: {metrics['global_r2']:.6f}")
        print(f"   遗传算法 - 全局MSE: {metrics_ga['global_mse']:.6f}, R²: {metrics_ga['global_r2']:.6f}")
        
        if metrics_ga['global_mse'] < metrics['global_mse']:
            print("   → GA方法在此案例中表现更好")
        else:
            print("   → 最小二乘方法在此案例中表现更好")
        
        # 保存方法对比可视化
        save_method_comparison(metrics, metrics_ga, output_dir)
        
    except Exception as e:
        print(f"   GA重构失败: {e}")
    
    # 生成演示总结报告
    create_demo_summary_report(output_dir, locals().get('sample_idx', 'synthetic'), 
                              measurement_points, measurements,
                              locals().get('metrics'), locals().get('metrics_ga'))
    
    print(f"\n演示完成！")
    print(f"所有结果已保存到: {output_dir}")
    print("重要提示：现在使用的是真实数据样本，重构结果更具实际意义！")


def demo_partial_known_coefficients():
    """部分已知系数重构演示"""
    print("\n=== 部分已知系数重构演示 ===\n")
    
    # 这个演示展示如何在已知部分POD系数的情况下重构温度场
    # 这在实际应用中很有用，比如某些物理约束已知
    
    print("1. 创建测试场景...")
    print("   场景：假设我们知道前3个最重要的POD模态系数")
    
    # 创建示例数据
    n_modes = 10  # 假设有10个模态
    true_coeffs = np.random.normal(0, 1, n_modes)
    
    # 前3个系数"已知"
    known_coeffs = {0: true_coeffs[0], 1: true_coeffs[1], 2: true_coeffs[2]}
    
    print(f"   总模态数: {n_modes}")
    print(f"   已知系数: {len(known_coeffs)} 个")
    print(f"   已知系数值: {[f'{v:.3f}' for v in known_coeffs.values()]}")
    
    # 模拟测点数据
    n_points = 15
    measurement_points = generate_measurement_points('grid', n_points)
    
    # 这里我们无法创建真实的POD重构，因为需要真实的训练模型
    # 但可以展示接口的使用方法
    print("\n2. 重构调用示例：")
    print("""
    # 使用部分已知系数进行重构
    known_coeffs = {0: 2.5, 1: -1.2, 2: 0.8}  # 已知前3个系数
    
    # 最小二乘重构
    reconstructed_temp, coeffs = pod_model.reconstruct_from_measurements(
        measurement_points, measurements, 
        method='lstsq', 
        known_coeffs=known_coeffs
    )
    
    # GA重构
    reconstructed_temp_ga, coeffs_ga = pod_model.reconstruct_from_measurements(
        measurement_points, measurements,
        method='ga',
        known_coeffs=known_coeffs,
        pop_size=100, n_generations=50
    )
    """)
    
    print("   优势：")
    print("   - 减少需要优化的变量数量")
    print("   - 利用先验物理知识")
    print("   - 提高重构精度和稳定性")
    print("   - 加快收敛速度")


def demo_different_sampling_strategies():
    """不同采样策略演示"""
    print("\n=== 不同采样策略演示 ===\n")
    
    n_points = 16
    
    # 1. 随机采样
    print("1. 随机采样策略:")
    random_points = generate_measurement_points('random', n_points, seed=42)
    print(f"   生成 {len(random_points)} 个随机分布的测点")
    print(f"   前5个点坐标: {random_points[:5]}")
    
    # 2. 网格采样
    print("\n2. 网格采样策略:")
    grid_points = generate_measurement_points('grid', n_points)
    print(f"   生成 {len(grid_points)} 个网格分布的测点")
    print(f"   前5个点坐标: {grid_points[:5]}")
    
    # 3. 边界优先采样
    print("\n3. 边界优先采样策略:")
    boundary_points = generate_measurement_points('boundary', n_points, seed=42)
    print(f"   生成 {len(boundary_points)} 个边界优先的测点")
    print(f"   前5个点坐标: {boundary_points[:5]}")
    
    # 可视化不同策略
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    strategies = [
        ('Random', random_points, 'red'),
        ('Grid', grid_points, 'blue'), 
        ('Boundary', boundary_points, 'green')
    ]
    
    for i, (name, points, color) in enumerate(strategies):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        axes[i].scatter(x_coords, y_coords, c=color, s=50, alpha=0.7)
        axes[i].set_xlim(0, 256)
        axes[i].set_ylim(0, 256)
        axes[i].set_title(f'{name} Sampling')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('demo_sampling_strategies.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n   采样策略对比图已保存: demo_sampling_strategies.png")
    
    print("\n   策略选择建议:")
    print("   - Random: 一般用途，无特殊要求")
    print("   - Grid: 均匀覆盖，适合平滑场分布")
    print("   - Boundary: 边界条件重要的问题")


def demo_noise_handling():
    """噪声处理演示"""
    print("\n=== 噪声处理演示 ===\n")
    
    # 创建模拟测量数据
    clean_measurements = np.array([25.0, 30.5, 28.2, 32.1, 26.8, 29.4])
    print(f"1. 原始测量值: {clean_measurements}")
    
    # 添加不同水平的噪声
    noise_levels = [0.01, 0.05, 0.1]
    
    for noise_level in noise_levels:
        print(f"\n2. 添加 {noise_level:.0%} 噪声:")
        
        from reconstruction import add_noise_to_measurements
        noisy_measurements = add_noise_to_measurements(
            clean_measurements, noise_level, 'gaussian'
        )
        
        print(f"   带噪测量值: {noisy_measurements}")
        print(f"   噪声标准差: {np.std(noisy_measurements - clean_measurements):.4f}")
        
        # 在实际使用中，这些带噪声的测量值会用于重构
        print(f"   # 使用带噪声数据重构:")
        print(f"   # reconstructed_temp, coeffs = pod_model.reconstruct_from_measurements(")
        print(f"   #     measurement_points, noisy_measurements, method='lstsq')")


def main():
    """主演示函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='温度场重构功能演示')
    parser.add_argument('--model_path', type=str, 
                       default="results/run_20250812_111201/pod_model.pkl",
                       help='POD模型路径')
    parser.add_argument('--data_path', type=str,
                       default='/data/zxr/inr/SimDriveSate/heat_dataset/all_samples',
                       help='数据路径')
    parser.add_argument('--output_dir', type=str,
                       help='输出目录（可选，默认自动生成）')
    
    args = parser.parse_args()
    
    print("温度场重构功能演示程序")
    print("=" * 50)
    print(f"模型路径: {args.model_path}")
    print(f"数据路径: {args.data_path}")
    if args.output_dir:
        print(f"输出目录: {args.output_dir}")
    print()
    
    # 运行各个演示
    try:
        demo_basic_reconstruction(args.model_path, args.data_path, args.output_dir)
    except Exception as e:
        print(f"基础重构演示出错: {e}")
    
    demo_partial_known_coefficients()
    demo_different_sampling_strategies()
    demo_noise_handling()
    
    print("\n" + "=" * 50)
    print("演示程序完成！")
    print("\n使用说明:")
    print("1. 首先运行 train.py 训练POD模型")
    print("2. 使用 test_reconstruction.py 进行全面测试")
    print("3. 参考此演示脚本进行自定义应用")
    print("\n关键功能:")
    print("- 最小二乘快速重构")
    print("- GA启发式精确重构") 
    print("- 支持部分已知系数")
    print("- 多种采样策略")
    print("- 噪声鲁棒性")
    print("\n命令行示例:")
    print("python reconstruction_demo.py --model_path your_model.pkl --data_path your_data_path")


if __name__ == "__main__":
    main()
