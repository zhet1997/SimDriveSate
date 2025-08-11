import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import time
from datetime import datetime
import random
import matplotlib.pyplot as plt
from U_net import TemperatureUNet
# 确保负号正常显示
plt.rcParams["axes.unicode_minus"] = False

class TemperatureDataset(Dataset):
    """温度场数据集"""
    def __init__(self, data_dir, split='train', norm_type='zscore', stats_dir=None):
        self.split = split
        self.norm_type = norm_type
        self.data_path = os.path.join(data_dir, 'split')
        self.stats_dir = stats_dir if stats_dir is not None else os.path.join(data_dir, 'stats')
        os.makedirs(self.stats_dir, exist_ok=True)

        # 加载数据
        self.heat_sources = np.load(os.path.join(self.data_path, f'{split}_heat_sources.npy')).astype(np.float32)
        self.sdfs = np.load(os.path.join(self.data_path, f'{split}_sdfs.npy')).astype(np.float32)
        self.temperatures = np.load(os.path.join(self.data_path, f'{split}_temperatures.npy')).astype(np.float32)
        self.heatsink_mask = np.load(os.path.join(self.data_path, 'heatsink_mask.npy')).astype(np.float32)

        # 验证数据形状
        assert self.heat_sources.ndim == 4, f"{split}热源矩阵维度错误，应为4维"
        assert self.sdfs.ndim == 4, f"{split}SDF维度错误，应为4维"
        assert self.temperatures.ndim == 4, f"{split}温度场维度错误，应为4维"

        # 计算或加载归一化统计量
        if split == 'train':
            self._compute_stats()
        else:
            self._load_stats()

        # 对数据进行归一化
        self._normalize_data()

        # 预准备反归一化参数
        self.norm_params = self._prepare_norm_params()

    def _compute_stats(self):
        """计算训练集的统计量"""
        sdf_flat = self.sdfs.reshape(-1)
        heat_flat = self.heat_sources.reshape(-1)
        temp_flat = self.temperatures.reshape(-1)

        if self.norm_type == 'zscore':
            self.sdf_mean = sdf_flat.mean()
            self.sdf_std = sdf_flat.std() + 1e-8
            self.heat_mean = heat_flat.mean()
            self.heat_std = heat_flat.std() + 1e-8
            self.temp_mean = temp_flat.mean()
            self.temp_std = temp_flat.std() + 1e-8

            np.savez(
                os.path.join(self.stats_dir, f'{self.norm_type}_stats.npz'),
                sdf_mean=self.sdf_mean, sdf_std=self.sdf_std,
                heat_mean=self.heat_mean, heat_std=self.heat_std,
                temp_mean=self.temp_mean, temp_std=self.temp_std
            )

        elif self.norm_type == 'minmax':
            self.sdf_min = sdf_flat.min()
            self.sdf_max = sdf_flat.max()
            self.sdf_range = self.sdf_max - self.sdf_min + 1e-8
            self.heat_min = heat_flat.min()
            self.heat_max = heat_flat.max()
            self.heat_range = self.heat_max - self.heat_min + 1e-8
            self.temp_min = temp_flat.min()
            self.temp_max = temp_flat.max()
            self.temp_range = self.temp_max - self.temp_min + 1e-8

            np.savez(
                os.path.join(self.stats_dir, f'{self.norm_type}_stats.npz'),
                sdf_min=self.sdf_min, sdf_max=self.sdf_max, sdf_range=self.sdf_range,
                heat_min=self.heat_min, heat_max=self.heat_max, heat_range=self.heat_range,
                temp_min=self.temp_min, temp_max=self.temp_max, temp_range=self.temp_range
            )

    def _load_stats(self):
        """加载训练集的统计量"""
        stats = np.load(os.path.join(self.stats_dir, f'{self.norm_type}_stats.npz'))
        if self.norm_type == 'zscore':
            self.sdf_mean = stats['sdf_mean']
            self.sdf_std = stats['sdf_std']
            self.heat_mean = stats['heat_mean']
            self.heat_std = stats['heat_std']
            self.temp_mean = stats['temp_mean']
            self.temp_std = stats['temp_std']
        elif self.norm_type == 'minmax':
            self.sdf_min = stats['sdf_min']
            self.sdf_max = stats['sdf_max']
            self.sdf_range = stats['sdf_range']
            self.heat_min = stats['heat_min']
            self.heat_max = stats['heat_max']
            self.heat_range = stats['heat_range']
            self.temp_min = stats['temp_min']
            self.temp_max = stats['temp_max']
            self.temp_range = stats['temp_range']

    def _normalize_data(self):
        """根据归一化方式处理数据"""
        if self.norm_type == 'zscore':
            self.sdfs = (self.sdfs - self.sdf_mean) / self.sdf_std
            self.heat_sources = (self.heat_sources - self.heat_mean) / self.heat_std
            self.temperatures = (self.temperatures - self.temp_mean) / self.temp_std
        elif self.norm_type == 'minmax':
            self.sdfs = (self.sdfs - self.sdf_min) / self.sdf_range
            self.heat_sources = (self.heat_sources - self.heat_min) / self.heat_range
            self.temperatures = (self.temperatures - self.temp_min) / self.temp_range

    def _prepare_norm_params(self):
        """预准备反归一化参数"""
        if self.norm_type == 'zscore':
            return {
                'type': self.norm_type,
                'temp_mean': torch.tensor(self.temp_mean, dtype=torch.float32),
                'temp_std': torch.tensor(self.temp_std, dtype=torch.float32)
            }
        elif self.norm_type == 'minmax':
            return {
                'type': self.norm_type,
                'temp_min': torch.tensor(self.temp_min, dtype=torch.float32),
                'temp_max': torch.tensor(self.temp_max, dtype=torch.float32),
                'temp_range': torch.tensor(self.temp_range, dtype=torch.float32)
            }

    def denormalize_temperature(self, normalized_temp):
        """将归一化的温度场反归一化到原始范围"""
        if self.norm_type == 'zscore':
            return normalized_temp * self.norm_params['temp_std'].item() + self.norm_params['temp_mean'].item()
        elif self.norm_type == 'minmax':
            return normalized_temp * self.norm_params['temp_range'].item() + self.norm_params['temp_min'].item()

    def __len__(self):
        return self.heat_sources.shape[0]

    def __getitem__(self, idx):
        return {
            'sdf': torch.from_numpy(self.sdfs[idx]),
            'heat_source': torch.from_numpy(self.heat_sources[idx]),
            'heatsink_mask': torch.from_numpy(self.heatsink_mask[0]),
            'temperature': torch.from_numpy(self.temperatures[idx]),
            'norm_params': self.norm_params
        }


def setup_logger(args, resume=False):
    """配置日志记录"""
    logger = logging.getLogger('temp_unet_train')
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_dir = os.path.join(args.save_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    if resume and args.resume is not None and args.resume != "":
        resume_name = os.path.basename(args.resume).replace('model', 'train').replace('.pth', '.log')
        log_file = os.path.join(log_dir, resume_name)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'train_{timestamp}.log')

    console_handler = logging.StreamHandler()
    file_mode = 'a' if resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=file_mode)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def visualize_prediction(truth, pred, norm_params, epoch, save_dir):
    """可视化函数"""
    if norm_params['type'] == 'zscore':
        truth = truth * norm_params['temp_std'].item() + norm_params['temp_mean'].item()
        pred = pred * norm_params['temp_std'].item() + norm_params['temp_mean'].item()
    elif norm_params['type'] == 'minmax':
        truth = truth * norm_params['temp_range'].item() + norm_params['temp_min'].item()
        pred = pred * norm_params['temp_range'].item() + norm_params['temp_min'].item()

    if truth.ndim == 3:
        truth = truth[0, :, :]
    truth = truth.squeeze()
    if pred.ndim == 3:
        pred = pred[0, :, :]
    pred = pred.squeeze()

    residual = pred - truth

    truth_vmin, truth_vmax = truth.min(), truth.max()
    pred_vmin, pred_vmax = pred.min(), pred.max()
    residual_vmax = max(abs(residual.min()), abs(residual.max()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Temperature Field Prediction (Epoch {epoch})', fontsize=14)

    im1 = axes[0].imshow(truth, cmap='viridis', origin='lower', vmin=truth_vmin, vmax=truth_vmax)
    axes[0].set_title('Ground Truth')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(pred, cmap='plasma', origin='lower', vmin=pred_vmin, vmax=pred_vmax)
    axes[1].set_title('Prediction')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    im3 = axes[2].imshow(residual, cmap='coolwarm', origin='lower', vmin=-residual_vmax, vmax=residual_vmax)
    axes[2].set_title('Residual (Pred - Truth)')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.subplots_adjust(wspace=0.3)
    save_path = os.path.join(save_dir, f'prediction_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def train_one_epoch(model, dataloader, criterion, optimizer, device, logger):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        sdf = batch['sdf'].to(device)
        heat_source = batch['heat_source'].to(device)
        heatsink_mask = batch['heatsink_mask'].to(device)
        temperature = batch['temperature'].to(device)
        # print("sdf shape:", sdf.shape)  # 应为[B, 1, H, W]
        # print("heat_source shape:", heat_source.shape)  # 应为[B, 1, H, W]
        # print("heatsink_mask shape:", heatsink_mask.shape)  # 应为[B, 1, H, W]
        optimizer.zero_grad()
        pred = model(sdf, heat_source, heatsink_mask)
        loss = criterion(pred, temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * sdf.size(0)
        pbar.set_postfix({'batch_loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(dataloader.dataset)
    logger.info(f'Train Loss: {avg_loss:.6f}')
    return avg_loss


def validate(model, dataloader, criterion, device, logger, epoch, vis_dir):
    """验证一个epoch并随机可视化"""
    model.eval()
    total_loss = 0.0
    all_truth = []
    all_pred = []
    norm_params = None

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating', leave=False)
        for batch in pbar:
            sdf = batch['sdf'].to(device)
            heat_source = batch['heat_source'].to(device)
            heatsink_mask = batch['heatsink_mask'].to(device)
            temperature = batch['temperature'].to(device)

            if norm_params is None:
                norm_params = batch['norm_params']

            pred = model(sdf, heat_source, heatsink_mask)
            loss = criterion(pred, temperature)
            total_loss += loss.item() * sdf.size(0)
            pbar.set_postfix({'batch_loss': f'{loss.item():.6f}'})

            if len(all_truth) == 0:
                all_truth = temperature.cpu().numpy()
                all_pred = pred.cpu().numpy()

    avg_loss = total_loss / len(dataloader.dataset)
    logger.info(f'Val Loss: {avg_loss:.6f}')

    if len(all_truth) > 0 and norm_params is not None:
        idx = random.randint(0, all_truth.shape[0] - 1)
        truth = all_truth[idx]
        pred = all_pred[idx]

        save_path = visualize_prediction(truth, pred, norm_params, epoch, vis_dir)
        logger.info(f'可视化结果保存至: {save_path}')

    return avg_loss


def get_optimizer(model, args):
    """根据参数选择优化器"""
    optimizer_type = args.optimizer.lower()

    if optimizer_type == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
    elif optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov
        )
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            alpha=args.alpha,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
    else:
        raise ValueError(f"不支持的优化器类型: {args.optimizer}，可选类型：adam, sgd, rmsprop")


def main(args):
    # 判断是否为恢复训练
    resume = args.resume is not None and args.resume != ""

    # 配置日志
    logger = setup_logger(args, resume=resume)
    if not resume:
        logger.info('===== 训练参数 =====')
        for arg, value in vars(args).items():
            logger.info(f'{arg}: {value}')

    # 创建可视化目录
    vis_dir = os.path.join(args.save_root, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # 设置设备
    if args.gpus == '-1':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        multi_gpu = torch.cuda.device_count() > 1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        multi_gpu = len(args.gpus.split(',')) > 1

    logger.info(f'使用设备: {device}')
    if multi_gpu:
        logger.info(f'并行计算: 使用 {torch.cuda.device_count()} 个GPU')

    # 创建数据集和数据加载器
    logger.info('加载数据集...')
    train_dataset = TemperatureDataset(
        args.data_root,
        split='train',
        norm_type=args.norm_type
    )
    val_dataset = TemperatureDataset(
        args.data_root,
        split='val',
        norm_type=args.norm_type
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f'训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}')
    logger.info(f'使用归一化方式: {args.norm_type}')

    # 初始化模型（使用命令行参数配置网络结构）
    model = TemperatureUNet(
        base_dim=args.base_dim,
        downsample_layers=args.downsample_layers,
        image_size=args.image_size,
        dropout=args.dropout
    )
    # print(f"模型实际base_dim: {args.base_dim}", flush=True)
    if multi_gpu:
        model = DataParallel(model)
    model.to(device)

    # 打印模型参数总量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'模型总参数数量: {total_params:,} (约{total_params/1e6:.2f}M)')

    # 损失函数
    criterion = nn.MSELoss()

    # 优化器
    optimizer = get_optimizer(model, args)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    # 创建模型保存目录
    model_dir = os.path.join(args.save_root, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # 训练记录初始化
    start_epoch = 1
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    early_stop_counter = 0

    # 恢复训练逻辑
    if resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"恢复路径不存在: {args.resume}")

        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            logger.warning("旧模型中未找到调度器状态，将使用新的调度器")

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']

        loss_record_path = os.path.join(args.save_root, 'loss_record.npy')
        if os.path.exists(loss_record_path):
            loss_record = np.load(loss_record_path, allow_pickle=True).item()
            train_losses = loss_record['train_losses']
            val_losses = loss_record['val_losses']

        early_stop_counter = checkpoint.get('early_stop_counter', 0)
        training_time = checkpoint.get('training_time', 0)
        logger.info(f"成功恢复训练: 从epoch {start_epoch - 1} 继续训练，当前最佳验证损失: {best_val_loss:.6f}")

    # 训练主循环
    logger.info('开始训练...')
    start_time = time.time() if not resume else time.time() - training_time

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f'\n===== Epoch {epoch}/{args.epochs} =====')

        # 每轮训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, logger)
        train_losses.append(train_loss)

        # 按间隔验证和保存
        if epoch % args.save_interval == 0:
            val_loss = validate(model, val_loader, criterion, device, logger, epoch, vis_dir)
            val_losses.append(val_loss)

            # 调整学习率
            scheduler.step(val_loss)

            # 早停逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                logger.info(f'更新最佳验证损失: {best_val_loss:.6f}')
            else:
                early_stop_counter += 1
                logger.info(f'早停计数器: {early_stop_counter}/{args.early_stop_patience}')
                if early_stop_counter >= args.early_stop_patience:
                    logger.info(f"早停触发：连续{args.early_stop_patience}个epoch未提升，停止训练")
                    break

            # 保存最佳模型
            best_model_path = os.path.join(model_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'norm_type': args.norm_type,
                'training_time': time.time() - start_time,
                'early_stop_counter': early_stop_counter,
                # 保存模型结构参数，方便恢复训练时核对
                'model_params': {
                    'base_dim': args.base_dim,
                    'downsample_layers': args.downsample_layers,
                    'image_size': args.image_size
                }
            }, best_model_path)

            # 保存当前epoch模型
            model_path = os.path.join(model_dir, f'model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'norm_type': args.norm_type,
                'training_time': time.time() - start_time,
                'early_stop_counter': early_stop_counter,
                'model_params': {
                    'base_dim': args.base_dim,
                    'downsample_layers': args.downsample_layers,
                    'image_size': args.image_size
                }
            }, model_path)
            logger.info(f'保存模型至: {model_path}')

            # 实时保存损失记录
            loss_record = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'norm_type': args.norm_type
            }
            np.save(os.path.join(args.save_root, 'loss_record.npy'), loss_record)

    # 训练结束
    total_time = time.time() - start_time
    logger.info('\n===== 训练结束 =====')
    logger.info(f'总耗时: {total_time:.2f}秒 ({total_time / 3600:.2f}小时)')
    logger.info(f'最佳验证损失: {best_val_loss:.6f}')

    # 最终保存损失记录
    loss_record = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'norm_type': args.norm_type
    }
    np.save(os.path.join(args.save_root, 'loss_record.npy'), loss_record)
    logger.info(f'损失记录保存至: {os.path.join(args.save_root, "loss_record.npy")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练温度场预测U-Net')

    # 数据与归一化参数
    parser.add_argument('--data_root', type=str,
                        default='model',
                        help='数据集根目录（包含split子目录）')
    parser.add_argument('--norm_type', type=str, default='zscore',
                        choices=['zscore', 'minmax'],
                        help='归一化方式：zscore（均值标准差）或minmax（最小最大）')

    # 模型结构参数（新增）
    parser.add_argument('--base_dim', type=int, default=32,
                        help='基础通道数')
    parser.add_argument('--downsample_layers', type=int, default=4,
                        help='采样层数')
    parser.add_argument('--image_size', type=int, default=256,
                        help='输入图像尺寸')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout正则化概率')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=1000,
                        help='总训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减（L2正则化系数）')

    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop'],
                        help='优化器类型')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam的beta1参数')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam的beta2参数')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD/RMSprop的动量参数')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop的平滑系数')
    parser.add_argument('--nesterov', action='store_true',
                        help='SGD是否使用Nesterov动量')

    # 设备参数
    parser.add_argument('--gpus', type=str, default='0,3,7',
                        help='GPU编号，用逗号分隔（-1表示所有可用GPU）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')

    # 保存和验证参数
    parser.add_argument('--save_root', type=str, default='experiments',
                        help='模型和日志保存根目录')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='模型保存和验证间隔（轮数）')
    parser.add_argument('--early_stop_patience', type=int, default=50,
                        help='早停策略的容忍轮数')

    # 恢复训练参数
    parser.add_argument('--resume', type=str, default="",
                        help='恢复训练的模型路径')

    args = parser.parse_args()
    main(args)
