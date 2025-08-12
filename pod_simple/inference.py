#!/usr/bin/env python3
"""
POD模型推理脚本
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import os
from data_utils import load_data, extract_features


def load_model_and_scaler(model_path, scaler_path):
    """加载模型和标准化器"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"模型加载完成: {model.n_modes} 个POD模态")
    return model, scaler


def predict_samples(model, scaler, features, indices=[0, 1, 2]):
    """预测指定样本"""
    # 标准化特征
    features_norm = scaler.transform(features)
    
    # 预测
    pred_temps, pred_coeffs = model.predict(features_norm)
    
    results = []
    for i, idx in enumerate(indices):
        if idx < len(pred_temps):
            results.append({
                'index': idx,
                'temperature': pred_temps[idx],
                'coefficients': pred_coeffs[idx]
            })
    
    return results


def visualize_prediction(true_temp, pred_temp, sample_idx, save_path=None):
    """可视化单个预测结果"""
    residual = pred_temp - true_temp
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 真实值
    im1 = axes[0].imshow(true_temp, cmap='viridis')
    axes[0].set_title('Ground Truth')
    plt.colorbar(im1, ax=axes[0])
    
    # 预测值
    im2 = axes[1].imshow(pred_temp, cmap='viridis', vmin=true_temp.min(), vmax=true_temp.max())
    axes[1].set_title('Prediction')
    plt.colorbar(im2, ax=axes[1])
    
    # 残差
    max_res = max(abs(residual.min()), abs(residual.max()))
    im3 = axes[2].imshow(residual, cmap='coolwarm', vmin=-max_res, vmax=max_res)
    axes[2].set_title('Residual')
    plt.colorbar(im3, ax=axes[2])
    
    # 计算误差
    mse = np.mean((true_temp - pred_temp) ** 2)
    mae = np.mean(np.abs(true_temp - pred_temp))
    
    plt.suptitle(f'Sample {sample_idx} - MSE: {mse:.4f}, MAE: {mae:.4f}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"结果保存至: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='POD模型推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--scaler_path', type=str, required=True, help='标准化器路径')
    parser.add_argument('--data_path', type=str, 
                       default='/data/zxr/inr/SimDriveSate/heat_dataset/all_samples',
                       help='数据路径')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                       help='预测样本索引')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 50)
    print("POD模型推理")
    print("=" * 50)
    
    # 1. 加载模型
    print("1. 加载模型...")
    model, scaler = load_model_and_scaler(args.model_path, args.scaler_path)
    
    # 2. 加载数据
    print("\n2. 加载数据...")
    components, temperatures = load_data(args.data_path)
    features = extract_features(components)
    
    # 3. 推理
    print(f"\n3. 推理样本 {args.sample_indices}...")
    
    for idx in args.sample_indices:
        if idx >= len(features):
            print(f"样本索引 {idx} 超出范围，跳过")
            continue
        
        # 预测单个样本
        sample_features = features[idx:idx+1]
        pred_temps, _ = model.predict(scaler.transform(sample_features))
        
        pred_temp = pred_temps[0]
        true_temp = temperatures[idx]
        
        # 计算误差
        mse = np.mean((true_temp - pred_temp) ** 2)
        mae = np.mean(np.abs(true_temp - pred_temp))
        
        print(f"样本 {idx}: MSE={mse:.6f}, MAE={mae:.6f}")
        
        # 可视化
        save_path = os.path.join(args.output_dir, f'prediction_sample_{idx}.png')
        visualize_prediction(true_temp, pred_temp, idx, save_path)
    
    print(f"\n推理完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()