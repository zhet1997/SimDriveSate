#!/usr/bin/env python3
"""
简化的POD温度场预测模型训练脚本
"""

import os
import numpy as np
import argparse
from datetime import datetime

from data_utils import load_data, extract_features, split_data, normalize_features, extract_features_enhanced
from pod_model import SimplePOD
from visualization import (
    plot_temperature_comparison, plot_pod_modes, plot_energy_curve,
    plot_tsne, plot_metrics
)


def main():
    parser = argparse.ArgumentParser(description='POD温度场预测训练')
    parser.add_argument('--data_path', type=str, 
                       default='/data/zxr/inr/SimDriveSate/heat_dataset/all_samples',
                       help='数据路径')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录')
    parser.add_argument('--energy_threshold', type=float, default=0.999,
                       help='POD能量阈值')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='测试集比例')
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("POD温度场预测模型训练")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {output_dir}")
    print(f"能量阈值: {args.energy_threshold}")
    print()
    
    # 1. 加载数据
    print("1. 加载数据...")
    components, temperatures = load_data(args.data_path)
    
    # 2. 提取特征
    print("\n2. 提取特征...")
    # features = extract_features(components)
    features = extract_features_enhanced(components)
    
    # 3. 划分数据集
    print("\n3. 划分数据集...")
    (train_features, train_temps, test_features, test_temps,
     train_idx, test_idx) = split_data(features, temperatures, args.test_ratio)
    
    print(f"训练集: {len(train_idx)} 样本")
    print(f"测试集: {len(test_idx)} 样本")
    
    # 4. 特征标准化
    print("\n4. 特征标准化...")
    train_features_norm, test_features_norm, scaler = normalize_features(
        train_features, test_features
    )
    
    # 5. 训练POD模型
    print("\n5. 训练POD模型...")
    pod_model = SimplePOD(energy_threshold=args.energy_threshold)
    
    # POD分解
    train_pod_coeffs = pod_model.fit_pod(train_temps)
    
    # 训练回归器
    regressor_metrics = pod_model.fit_regressor(train_features_norm, train_pod_coeffs)
    
    # 6. 模型评估
    print("\n6. 模型评估...")
    print("训练集评估:")
    train_metrics = pod_model.evaluate(train_features_norm, train_temps)
    
    print("\n测试集评估:")
    test_metrics = pod_model.evaluate(test_features_norm, test_temps)
    
    # 7. 保存模型
    print("\n7. 保存模型...")
    model_path = os.path.join(output_dir, 'pod_model.pkl')
    pod_model.save(model_path)
    
    import pickle
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"标准化器已保存: {scaler_path}")
    
    # 8. 生成可视化
    print("\n8. 生成可视化结果...")
    
    # 预测结果
    train_pred_temps, _ = pod_model.predict(train_features_norm)
    test_pred_temps, _ = pod_model.predict(test_features_norm)
    
    # 训练集对比
    print("生成训练集对比...")
    train_vis_dir = os.path.join(output_dir, 'train_comparison')
    plot_temperature_comparison(train_temps, train_pred_temps, train_idx, train_vis_dir)
    
    # 测试集对比
    print("生成测试集对比...")
    test_vis_dir = os.path.join(output_dir, 'test_comparison')
    plot_temperature_comparison(test_temps, test_pred_temps, test_idx, test_vis_dir)
    
    # POD模态可视化
    print("生成POD模态图...")
    pod_vis_dir = os.path.join(output_dir, 'pod_analysis')
    plot_pod_modes(pod_model, pod_vis_dir)
    plot_energy_curve(pod_model, pod_vis_dir)
    
    # t-SNE可视化
    print("生成t-SNE图...")
    tsne_dir = os.path.join(output_dir, 'tsne_analysis')
    plot_tsne(train_features_norm, train_idx, tsne_dir, "Train t-SNE")
    plot_tsne(test_features_norm, test_idx, tsne_dir, "Test t-SNE")
    
    # 指标对比
    print("生成指标对比图...")
    metrics_dir = os.path.join(output_dir, 'metrics')
    plot_metrics(train_metrics, test_metrics, metrics_dir)
    
    # 9. 保存实验报告
    print("\n9. 保存实验报告...")
    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("POD温度场预测实验报告\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("实验配置:\n")
        f.write(f"  数据路径: {args.data_path}\n")
        f.write(f"  能量阈值: {args.energy_threshold}\n")
        f.write(f"  测试集比例: {args.test_ratio}\n\n")
        
        f.write("数据信息:\n")
        f.write(f"  总样本数: {len(components)}\n")
        f.write(f"  特征维度: {features.shape[1]}\n")
        f.write(f"  训练样本: {len(train_idx)}\n")
        f.write(f"  测试样本: {len(test_idx)}\n\n")
        
        f.write("POD结果:\n")
        f.write(f"  保留模态数: {pod_model.n_modes}\n")
        f.write(f"  累计能量比: {np.sum(pod_model.pca.explained_variance_ratio_):.6f}\n\n")
        
        f.write("训练集性能:\n")
        for key, val in train_metrics.items():
            f.write(f"  {key}: {val:.6f}\n")
        
        f.write("\n测试集性能:\n")
        for key, val in test_metrics.items():
            f.write(f"  {key}: {val:.6f}\n")
        
        f.write(f"\n实验时间: {timestamp}\n")
    
    print(f"实验报告已保存: {report_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"所有结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()