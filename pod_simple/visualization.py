import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


def plot_temperature_comparison(true_temps, pred_temps, indices, save_dir, n_samples=4):
    """温度场对比可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 随机选择样本
    selected = np.random.choice(len(indices), min(n_samples, len(indices)), replace=False)
    
    for i, idx in enumerate(selected):
        true_temp = true_temps[idx]
        pred_temp = pred_temps[idx]
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
        
        # 误差统计
        mse = np.mean((true_temp - pred_temp) ** 2)
        mae = np.mean(np.abs(true_temp - pred_temp))
        
        plt.suptitle(f'Sample {indices[idx]} - MSE: {mse:.4f}, MAE: {mae:.4f}')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'comparison_{indices[idx]}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"保存对比图: {save_path}")


def plot_pod_modes(pod_model, save_dir, n_modes=6):
    """POD模态可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    modes = pod_model.pca.components_
    n_show = min(n_modes, len(modes))
    
    cols = 3
    rows = (n_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_show):
        row, col = i // cols, i % cols
        mode = modes[i].reshape(256, 256)
        
        im = axes[row, col].imshow(mode, cmap='coolwarm')
        axes[row, col].set_title(f'Mode {i+1}')
        plt.colorbar(im, ax=axes[row, col])
    
    # 隐藏多余子图
    for i in range(n_show, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('POD Modes')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'pod_modes.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存POD模态图: {save_path}")


def plot_energy_curve(pod_model, save_dir):
    """能量曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    explained_var = pod_model.pca.explained_variance_ratio_
    cumulative = np.cumsum(explained_var)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 各模态能量
    ax1.bar(range(1, len(explained_var) + 1), explained_var)
    ax1.set_xlabel('模态编号')
    ax1.set_ylabel('能量贡献比')
    ax1.set_title('各模态能量贡献')
    
    # 累计能量
    ax2.plot(range(1, len(cumulative) + 1), cumulative, 'b-o')
    ax2.axhline(y=pod_model.energy_threshold, color='r', linestyle='--', 
                label=f'阈值 {pod_model.energy_threshold}')
    ax2.axvline(x=pod_model.n_modes, color='g', linestyle='--', 
                label=f'选择模态数 {pod_model.n_modes}')
    ax2.set_xlabel('模态数量')
    ax2.set_ylabel('累计能量比')
    ax2.set_title('POD累计能量曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'energy_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存能量曲线: {save_path}")


def plot_tsne(features, indices, save_dir, title="t-SNE"):
    """t-SNE可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("计算t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=indices, cmap='viridis', s=30, alpha=0.7)
    plt.colorbar(scatter, label='Sample Index')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f'{title} - Feature Distribution')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, f'{title.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存t-SNE图: {save_path}")


def plot_metrics(train_metrics, test_metrics, save_dir):
    """指标对比"""
    os.makedirs(save_dir, exist_ok=True)
    
    metrics_names = ['MSE', 'R2', 'MAE']
    train_vals = [train_metrics['mse'], train_metrics['r2'], train_metrics['mae']]
    test_vals = [test_metrics['mse'], test_metrics['r2'], test_metrics['mae']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Train vs Test Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    save_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存指标对比图: {save_path}")