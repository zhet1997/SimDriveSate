#!/usr/bin/env python3
"""
温度场重构可视化模块
提供重构过程和结果的各种可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from typing import List, Tuple, Dict, Any, Optional
import seaborn as sns


def plot_reconstruction_comparison(true_temp: np.ndarray,
                                 reconstructed_temp: np.ndarray, 
                                 measurement_points: List[Tuple[int, int]],
                                 measurements: np.ndarray,
                                 metrics: Dict[str, float],
                                 method: str = "Unknown",
                                 save_path: Optional[str] = None,
                                 show_measurements: bool = True) -> None:
    """温度场重构对比可视化
    
    Args:
        true_temp: 真实温度场
        reconstructed_temp: 重构温度场
        measurement_points: 测点坐标
        measurements: 测点温度值
        metrics: 评估指标
        method: 重构方法名称
        save_path: 保存路径
        show_measurements: 是否显示测点
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 温度范围
    vmin = min(true_temp.min(), reconstructed_temp.min())
    vmax = max(true_temp.max(), reconstructed_temp.max())
    
    # 真实温度场
    im1 = axes[0, 0].imshow(true_temp, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('真实温度场', fontsize=14, fontweight='bold')
    if show_measurements:
        # 标注测点
        x_coords = [p[0] for p in measurement_points]
        y_coords = [p[1] for p in measurement_points]
        axes[0, 0].scatter(x_coords, y_coords, c='red', s=30, marker='o', 
                          edgecolor='white', linewidth=1, alpha=0.8)
    axes[0, 0].set_xlabel('X坐标')
    axes[0, 0].set_ylabel('Y坐标')
    plt.colorbar(im1, ax=axes[0, 0], label='温度')
    
    # 重构温度场
    im2 = axes[0, 1].imshow(reconstructed_temp, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'重构温度场 ({method})', fontsize=14, fontweight='bold')
    if show_measurements:
        axes[0, 1].scatter(x_coords, y_coords, c='red', s=30, marker='o',
                          edgecolor='white', linewidth=1, alpha=0.8)
    axes[0, 1].set_xlabel('X坐标')
    axes[0, 1].set_ylabel('Y坐标')
    plt.colorbar(im2, ax=axes[0, 1], label='温度')
    
    # 绝对误差
    abs_error = np.abs(true_temp - reconstructed_temp)
    im3 = axes[0, 2].imshow(abs_error, cmap='Reds')
    axes[0, 2].set_title('绝对误差分布', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('X坐标')
    axes[0, 2].set_ylabel('Y坐标')
    plt.colorbar(im3, ax=axes[0, 2], label='绝对误差')
    
    # 相对误差
    relative_error = np.abs(true_temp - reconstructed_temp) / (np.abs(true_temp) + 1e-8)
    relative_error = np.clip(relative_error, 0, 1)  # 限制在合理范围
    im4 = axes[1, 0].imshow(relative_error, cmap='Oranges')
    axes[1, 0].set_title('相对误差分布', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('X坐标')
    axes[1, 0].set_ylabel('Y坐标')
    plt.colorbar(im4, ax=axes[1, 0], label='相对误差')
    
    # 残差 (有符号的误差)
    residual = reconstructed_temp - true_temp
    max_res = max(abs(residual.min()), abs(residual.max()))
    im5 = axes[1, 1].imshow(residual, cmap='coolwarm', vmin=-max_res, vmax=max_res)
    axes[1, 1].set_title('残差分布', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('X坐标')
    axes[1, 1].set_ylabel('Y坐标')
    plt.colorbar(im5, ax=axes[1, 1], label='残差')
    
    # 误差统计图表
    axes[1, 2].axis('off')
    
    # 创建性能指标表格
    metrics_text = f"""
    重构性能指标:
    
    全局MSE: {metrics.get('global_mse', 0):.6f}
    全局MAE: {metrics.get('global_mae', 0):.6f}
    全局R²: {metrics.get('global_r2', 0):.6f}
    
    测点MAE: {metrics.get('point_mae', 0):.6f}
    最大误差: {metrics.get('max_error', 0):.6f}
    RMS误差: {metrics.get('rms_error', 0):.6f}
    
    测点数量: {len(measurement_points)}
    重构方法: {method}
    """
    
    axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"重构对比图保存至: {save_path}")
    else:
        plt.show()


def plot_measurement_points_analysis(measurement_points: List[Tuple[int, int]],
                                    measurements: np.ndarray,
                                    grid_size: Tuple[int, int] = (256, 256),
                                    save_path: Optional[str] = None) -> None:
    """测点分析可视化
    
    Args:
        measurement_points: 测点坐标
        measurements: 测点温度值
        grid_size: 网格大小
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 测点位置分布
    x_coords = [p[0] for p in measurement_points]
    y_coords = [p[1] for p in measurement_points]
    
    axes[0].scatter(x_coords, y_coords, c=measurements, s=100, cmap='viridis', 
                   edgecolor='black', linewidth=1)
    axes[0].set_xlim(0, grid_size[1])
    axes[0].set_ylim(0, grid_size[0])
    axes[0].set_xlabel('X坐标')
    axes[0].set_ylabel('Y坐标')
    axes[0].set_title('测点位置与温度分布')
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(axes[0].collections[0], ax=axes[0])
    cbar1.set_label('测点温度')
    
    # 测点温度直方图
    axes[1].hist(measurements, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1].set_xlabel('温度值')
    axes[1].set_ylabel('频次')
    axes[1].set_title('测点温度分布直方图')
    axes[1].grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f"""
    统计信息:
    均值: {np.mean(measurements):.3f}
    标准差: {np.std(measurements):.3f}
    最小值: {np.min(measurements):.3f}
    最大值: {np.max(measurements):.3f}
    测点数量: {len(measurements)}
    """
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 测点空间分布密度
    H, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=20)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    im3 = axes[2].imshow(H.T, extent=[0, grid_size[1], 0, grid_size[0]], 
                        cmap='Blues', origin='lower', interpolation='bilinear')
    axes[2].set_xlabel('X坐标')
    axes[2].set_ylabel('Y坐标')
    axes[2].set_title('测点空间密度分布')
    plt.colorbar(im3, ax=axes[2], label='测点密度')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"测点分析图保存至: {save_path}")
    else:
        plt.show()


def plot_convergence_history(convergence_history: List[float],
                           method: str = "GA",
                           save_path: Optional[str] = None) -> None:
    """收敛曲线可视化
    
    Args:
        convergence_history: 收敛历史
        method: 优化方法
        save_path: 保存路径
    """
    if not convergence_history:
        print("无收敛历史数据")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 绘制收敛曲线
    generations = range(1, len(convergence_history) + 1)
    plt.plot(generations, convergence_history, 'b-', linewidth=2, label='目标函数值')
    plt.fill_between(generations, convergence_history, alpha=0.3)
    
    # 标记最优点
    min_idx = np.argmin(convergence_history)
    min_val = convergence_history[min_idx]
    plt.plot(min_idx + 1, min_val, 'ro', markersize=8, label=f'最优解 (代数{min_idx+1})')
    
    plt.xlabel('进化代数')
    plt.ylabel('目标函数值 (MSE)')
    plt.title(f'{method} 算法收敛曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加收敛信息
    improvement = (convergence_history[0] - min_val) / convergence_history[0] * 100
    info_text = f"""
    收敛信息:
    初始误差: {convergence_history[0]:.6f}
    最终误差: {min_val:.6f}
    改进幅度: {improvement:.2f}%
    收敛代数: {min_idx + 1}
    """
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"收敛曲线保存至: {save_path}")
    else:
        plt.show()


def plot_method_comparison(comparison_results: Dict[str, Any],
                         save_path: Optional[str] = None) -> None:
    """方法比较可视化
    
    Args:
        comparison_results: 比较结果
        save_path: 保存路径
    """
    methods = list(comparison_results['individual_results'].keys())
    
    # 提取指标
    metrics_names = ['global_mse', 'global_mae', 'global_r2', 'point_mae', 'max_error']
    metrics_labels = ['全局MSE', '全局MAE', '全局R²', '测点MAE', '最大误差']
    
    n_metrics = len(metrics_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(20, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    # 为每个指标创建对比图
    for i, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
        values = []
        for method in methods:
            metric_val = comparison_results['individual_results'][method]['metrics'][metric]
            values.append(metric_val)
        
        bars = axes[i].bar(methods, values, alpha=0.7, 
                          color=['skyblue', 'lightcoral'][:len(methods)])
        axes[i].set_ylabel(label)
        axes[i].set_title(f'{label}对比')
        axes[i].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 添加总体比较信息
    fig.suptitle('重构方法性能对比', fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"方法对比图保存至: {save_path}")
    else:
        plt.show()


def plot_noise_robustness_analysis(noise_levels: List[float],
                                 method_results: Dict[str, List[Dict]],
                                 save_path: Optional[str] = None) -> None:
    """噪声鲁棒性分析可视化
    
    Args:
        noise_levels: 噪声水平列表
        method_results: 不同方法在各噪声水平下的结果
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics_to_plot = ['global_mse', 'global_mae', 'global_r2', 'point_mae']
    metrics_labels = ['全局MSE', '全局MAE', '全局R²', '测点MAE']
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metrics_labels)):
        for j, (method, color) in enumerate(zip(method_results.keys(), colors)):
            values = [result['metrics'][metric] for result in method_results[method]]
            axes[i].plot(noise_levels, values, 'o-', color=color, 
                        linewidth=2, markersize=6, label=method, alpha=0.8)
        
        axes[i].set_xlabel('噪声水平')
        axes[i].set_ylabel(label)
        axes[i].set_title(f'{label} vs 噪声水平')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"噪声鲁棒性分析图保存至: {save_path}")
    else:
        plt.show()


def plot_coefficient_analysis(true_coeffs: np.ndarray,
                            reconstructed_coeffs: np.ndarray,
                            known_coeffs: Optional[Dict[int, float]] = None,
                            save_path: Optional[str] = None) -> None:
    """POD系数分析可视化
    
    Args:
        true_coeffs: 真实POD系数
        reconstructed_coeffs: 重构POD系数
        known_coeffs: 已知系数
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    n_modes = len(true_coeffs)
    mode_indices = np.arange(n_modes)
    
    # 系数对比条形图
    width = 0.35
    axes[0].bar(mode_indices - width/2, true_coeffs, width, 
               label='真实系数', alpha=0.8, color='skyblue')
    axes[0].bar(mode_indices + width/2, reconstructed_coeffs, width,
               label='重构系数', alpha=0.8, color='lightcoral')
    
    # 标记已知系数
    if known_coeffs:
        for idx in known_coeffs.keys():
            if idx < n_modes:
                axes[0].axvline(x=idx, color='green', linestyle='--', alpha=0.7)
                axes[0].text(idx, max(max(true_coeffs), max(reconstructed_coeffs)) * 0.9,
                           '已知', rotation=90, ha='center', color='green', fontweight='bold')
    
    axes[0].set_xlabel('模态编号')
    axes[0].set_ylabel('系数值')
    axes[0].set_title('POD系数对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 散点图 (真实 vs 重构)
    axes[1].scatter(true_coeffs, reconstructed_coeffs, alpha=0.7, s=50)
    
    # 添加理想线 (y=x)
    min_val = min(min(true_coeffs), min(reconstructed_coeffs))
    max_val = max(max(true_coeffs), max(reconstructed_coeffs))
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='理想拟合')
    
    # 标记已知系数点
    if known_coeffs:
        known_indices = list(known_coeffs.keys())
        known_true = [true_coeffs[i] for i in known_indices if i < n_modes]
        known_recon = [reconstructed_coeffs[i] for i in known_indices if i < n_modes]
        axes[1].scatter(known_true, known_recon, color='green', s=100, 
                       marker='s', label='已知系数', alpha=0.8)
    
    axes[1].set_xlabel('真实系数')
    axes[1].set_ylabel('重构系数')
    axes[1].set_title('系数相关性')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 计算相关性
    correlation = np.corrcoef(true_coeffs, reconstructed_coeffs)[0, 1]
    mse = np.mean((true_coeffs - reconstructed_coeffs) ** 2)
    
    # 误差分析
    errors = np.abs(true_coeffs - reconstructed_coeffs)
    axes[2].bar(mode_indices, errors, alpha=0.7, color='orange')
    axes[2].set_xlabel('模态编号')
    axes[2].set_ylabel('绝对误差')
    axes[2].set_title('系数重构误差')
    axes[2].grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f"""
    统计信息:
    相关系数: {correlation:.4f}
    MSE: {mse:.6f}
    平均绝对误差: {np.mean(errors):.6f}
    最大误差: {np.max(errors):.6f}
    """
    axes[2].text(0.02, 0.98, stats_text, transform=axes[2].transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"系数分析图保存至: {save_path}")
    else:
        plt.show()


def create_reconstruction_report(results: Dict[str, Any],
                               output_dir: str) -> None:
    """创建重构实验报告
    
    Args:
        results: 实验结果
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成各种可视化图表
    if 'comparison' in results:
        comparison = results['comparison']
        
        # 方法对比图
        plot_method_comparison(comparison, 
                             os.path.join(output_dir, 'method_comparison.png'))
        
        # 为每种方法生成详细对比图
        for method, result in comparison['individual_results'].items():
            plot_reconstruction_comparison(
                results['true_temp'], result['reconstructed_temp'],
                results['measurement_points'], results['measurements'],
                result['metrics'], method,
                os.path.join(output_dir, f'reconstruction_{method}.png')
            )
            
            # GA收敛曲线
            if method == 'ga' and 'convergence_history' in result:
                plot_convergence_history(
                    result['convergence_history'], method,
                    os.path.join(output_dir, f'convergence_{method}.png')
                )
    
    # 测点分析
    if 'measurement_points' in results and 'measurements' in results:
        plot_measurement_points_analysis(
            results['measurement_points'], results['measurements'],
            save_path=os.path.join(output_dir, 'measurement_analysis.png')
        )
    
    # 噪声鲁棒性分析
    if 'noise_analysis' in results:
        plot_noise_robustness_analysis(
            results['noise_analysis']['noise_levels'],
            results['noise_analysis']['method_results'],
            os.path.join(output_dir, 'noise_robustness.png')
        )
    
    # 系数分析
    if 'coefficient_analysis' in results:
        coeff_analysis = results['coefficient_analysis']
        plot_coefficient_analysis(
            coeff_analysis['true_coeffs'],
            coeff_analysis['reconstructed_coeffs'],
            coeff_analysis.get('known_coeffs'),
            os.path.join(output_dir, 'coefficient_analysis.png')
        )
    
    print(f"重构实验报告已生成: {output_dir}")


def plot_sampling_strategy_comparison(strategies_results: Dict[str, Dict],
                                    save_path: Optional[str] = None) -> None:
    """采样策略比较可视化
    
    Args:
        strategies_results: 不同采样策略的结果
        save_path: 保存路径
    """
    strategies = list(strategies_results.keys())
    metrics = ['global_mse', 'global_mae', 'global_r2', 'point_mae']
    metrics_labels = ['全局MSE', '全局MAE', '全局R²', '测点MAE']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics, metrics_labels)):
        values = [strategies_results[strategy]['metrics'][metric] 
                 for strategy in strategies]
        
        bars = axes[i].bar(strategies, values, alpha=0.7, 
                          color=['skyblue', 'lightgreen', 'lightcoral'][:len(strategies)])
        axes[i].set_ylabel(label)
        axes[i].set_title(f'{label} - 采样策略对比')
        axes[i].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"采样策略对比图保存至: {save_path}")
    else:
        plt.show()
