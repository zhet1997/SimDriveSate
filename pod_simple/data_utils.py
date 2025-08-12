import json
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(data_path):
    """加载数据集"""
    # 加载组件信息
    with open(f"{data_path}/all_components.json", 'r') as f:
        components = json.load(f)
    
    # 加载温度场数据
    temperatures = np.load(f"{data_path}/all_temperatures.npy").astype(np.float32)
    
    print(f"加载数据: {len(components)} 样本, 温度场形状: {temperatures.shape}")
    return components, temperatures


def extract_features(components):
    """提取54维特征"""
    n_samples = len(components)
    features = np.zeros((n_samples, 54))
    
    for i, sample in enumerate(components):
        if not sample:
            continue
            
        # 基础统计 (9维)
        shapes = [c['shape'] for c in sample]
        features[i, 0] = shapes.count('capsule')
        features[i, 1] = shapes.count('rect') 
        features[i, 2] = shapes.count('circle')
        features[i, 3] = len(sample)  # 总元件数
        
        # 功率统计 (5维)
        powers = [c['power'] for c in sample]
        features[i, 4] = np.min(powers)
        features[i, 5] = np.max(powers)
        features[i, 6] = np.mean(powers)
        features[i, 7] = np.std(powers)
        features[i, 8] = np.sum(powers)
        
        # 位置统计 (4维)
        centers = [c['center'] for c in sample]
        center_x = [c[0] for c in centers]
        center_y = [c[1] for c in centers]
        features[i, 9] = np.mean(center_x)
        features[i, 10] = np.std(center_x)
        features[i, 11] = np.mean(center_y)
        features[i, 12] = np.std(center_y)
        
        # 尺寸统计 (4维)
        sizes = []
        for c in sample:
            if 'length' in c: sizes.append(c['length'])
            if 'width' in c: sizes.append(c['width'])
            if 'height' in c: sizes.append(c['height'])
            if 'radius' in c: sizes.append(c['radius'])
        
        if sizes:
            features[i, 13] = np.mean(sizes)
            features[i, 14] = np.std(sizes)
            features[i, 15] = np.min(sizes)
            features[i, 16] = np.max(sizes)
        
        # 空间网格特征 (32维: 8x4)
        # 简化为2x2网格，每个网格4个特征
        grid_bounds = [(0, 128), (128, 256)]
        
        for gi in range(2):
            for gj in range(2):
                grid_idx = gi * 2 + gj
                base_idx = 17 + grid_idx * 4
                
                x_min, x_max = grid_bounds[gi]
                y_min, y_max = grid_bounds[gj]
                
                # 该网格内的元件
                grid_components = []
                for c in sample:
                    nx, ny = c['node_position']
                    if x_min <= nx < x_max and y_min <= ny < y_max:
                        grid_components.append(c)
                
                features[i, base_idx] = len(grid_components)  # 元件数
                if grid_components:
                    grid_powers = [c['power'] for c in grid_components]
                    features[i, base_idx + 1] = np.sum(grid_powers)  # 总功率
                    features[i, base_idx + 2] = np.mean(grid_powers)  # 平均功率
                    features[i, base_idx + 3] = np.std(grid_powers)   # 功率标准差
        
        # 剩余维度用0填充
        # features[i, 33:54] 保持为0
    
    print(f"特征提取完成: {features.shape}")
    return features


def split_data(features, temperatures, test_ratio=0.2):
    """划分训练测试集"""
    np.random.seed(42)
    n = len(features)
    indices = np.random.permutation(n)
    
    split_idx = int(n * (1 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return (features[train_idx], temperatures[train_idx], 
            features[test_idx], temperatures[test_idx],
            train_idx, test_idx)


def extract_features_enhanced(components):
    """提取81维增强特征 - 使用4x4空间网格，无预留维度"""
    n_samples = len(components)
    features = np.zeros((n_samples, 81))  # 17 + 4*4*4 = 81维
    
    for i, sample in enumerate(components):
        if not sample:
            continue
            
        # 基础统计 (4维) - 与原函数保持一致
        shapes = [c['shape'] for c in sample]
        features[i, 0] = shapes.count('capsule')
        features[i, 1] = shapes.count('rect') 
        features[i, 2] = shapes.count('circle')
        features[i, 3] = len(sample)  # 总元件数
        
        # 功率统计 (5维) - 与原函数保持一致
        powers = [c['power'] for c in sample]
        features[i, 4] = np.min(powers)
        features[i, 5] = np.max(powers)
        features[i, 6] = np.mean(powers)
        features[i, 7] = np.std(powers)
        features[i, 8] = np.sum(powers)
        
        # 位置统计 (4维) - 与原函数保持一致
        centers = [c['center'] for c in sample]
        center_x = [c[0] for c in centers]
        center_y = [c[1] for c in centers]
        features[i, 9] = np.mean(center_x)
        features[i, 10] = np.std(center_x)
        features[i, 11] = np.mean(center_y)
        features[i, 12] = np.std(center_y)
        
        # 尺寸统计 (4维) - 与原函数保持一致
        sizes = []
        for c in sample:
            if 'length' in c: sizes.append(c['length'])
            if 'width' in c: sizes.append(c['width'])
            if 'height' in c: sizes.append(c['height'])
            if 'radius' in c: sizes.append(c['radius'])
        
        if sizes:
            features[i, 13] = np.mean(sizes)
            features[i, 14] = np.std(sizes)
            features[i, 15] = np.min(sizes)
            features[i, 16] = np.max(sizes)
        
        # 增强空间网格特征 (64维: 4x4网格，每个网格4个特征)
        grid_size = 4
        grid_step = 256 // grid_size  # 64像素每个网格
        
        for gi in range(grid_size):
            for gj in range(grid_size):
                grid_idx = gi * grid_size + gj
                base_idx = 17 + grid_idx * 4
                
                x_min = gi * grid_step
                x_max = (gi + 1) * grid_step
                y_min = gj * grid_step  
                y_max = (gj + 1) * grid_step
                
                # 该网格内的元件
                grid_components = []
                for c in sample:
                    nx, ny = c['node_position']
                    if x_min <= nx < x_max and y_min <= ny < y_max:
                        grid_components.append(c)
                
                features[i, base_idx] = len(grid_components)  # 元件数
                if grid_components:
                    grid_powers = [c['power'] for c in grid_components]
                    features[i, base_idx + 1] = np.sum(grid_powers)     # 总功率
                    features[i, base_idx + 2] = np.mean(grid_powers)    # 平均功率
                    features[i, base_idx + 3] = np.std(grid_powers)     # 功率标准差
                else:
                    # 空网格的特征都为0
                    features[i, base_idx + 1:base_idx + 4] = 0
    
    print(f"增强特征提取完成: {features.shape}")
    return features


def normalize_features(train_features, test_features):
    """特征标准化"""
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_features)
    test_norm = scaler.transform(test_features)
    return train_norm, test_norm, scaler