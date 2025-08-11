import numpy as np
import os
from sklearn.model_selection import train_test_split

# 配置参数
data_dir = "heat_dataset/all_samples"  # 数据存放目录
test_ratio = 0.2  # 测试集占比（1000/5000）
val_ratio_from_train = 0.05  # 从训练候选集中划分验证集的比例（200/4000）
random_seed = 42  # 随机种子，确保结果可复现

# 创建保存目录（如果不存在）
save_dir = os.path.join('model', "split")
os.makedirs(save_dir, exist_ok=True)

# --------------------------- 加载数据 ---------------------------
print("加载数据中...")
# 加载主要数据（5000个样本，形状(5000, 256, 256)）
heat_sources = np.load(os.path.join(data_dir, "all_heat_sources.npy"))  # 热源矩阵
sdfs = np.load(os.path.join(data_dir, "all_sdfs.npy"))                  # SDF
temperatures = np.load(os.path.join(data_dir, "all_temperatures.npy"))  # 温度场（标签）

# 加载共享的散热口掩码（形状(256, 256)）
heatsink_mask = np.load(os.path.join(data_dir, "heatsink_mask_from_config.npy"))

# 验证数据完整性
assert heat_sources.shape == (5000, 256, 256), "热源矩阵形状错误"
assert sdfs.shape == (5000, 256, 256), "SDF形状错误"
assert temperatures.shape == (5000, 256, 256), "温度场形状错误"
assert heatsink_mask.shape == (256, 256), "散热口掩码形状错误"

# --------------------------- 数据预处理（添加通道维度） ---------------------------
# 模型需要4维输入：[样本数, 通道数, 高, 宽]，单通道数据添加通道维度（axis=1）
heat_sources = np.expand_dims(heat_sources, axis=1)  # 形状变为(5000, 1, 256, 256)
sdfs = np.expand_dims(sdfs, axis=1)                  # 形状变为(5000, 1, 256, 256)
temperatures = np.expand_dims(temperatures, axis=1)  # 形状变为(5000, 1, 256, 256)

# 散热口掩码添加通道维度（后续可通过广播适配批次）
heatsink_mask = np.expand_dims(heatsink_mask, axis=0)  # 形状变为(1, 256, 256)
heatsink_mask = np.expand_dims(heatsink_mask, axis=0)  # 形状变为(1, 1, 256, 256)

# --------------------------- 划分数据集（训练集+验证集+测试集） ---------------------------
print("划分数据集...")
# 步骤1：先从5000个样本中划分出测试集（1000个），剩下的4000个作为"训练候选集"
all_indices = np.arange(5000)
train_val_indices, test_indices = train_test_split(
    all_indices,
    test_size=test_ratio,  # 1000个测试集（5000*0.2）
    random_state=random_seed,
    shuffle=True
)

# 步骤2：从4000个"训练候选集"中划分出验证集（200个）和最终训练集（3800个）
train_indices, val_indices = train_test_split(
    train_val_indices,
    test_size=val_ratio_from_train,  # 200个验证集（4000*0.05）
    random_state=random_seed,
    shuffle=True
)

# 按索引分割数据
# 训练集（3800）
train_heat = heat_sources[train_indices]
train_sdf = sdfs[train_indices]
train_temp = temperatures[train_indices]

# 验证集（200）
val_heat = heat_sources[val_indices]
val_sdf = sdfs[val_indices]
val_temp = temperatures[val_indices]

# 测试集（1000）
test_heat = heat_sources[test_indices]
test_sdf = sdfs[test_indices]
test_temp = temperatures[test_indices]

# --------------------------- 保存划分后的数据集 ---------------------------
print("保存数据集...")
# 保存训练集
np.save(os.path.join(save_dir, "train_heat_sources.npy"), train_heat)
np.save(os.path.join(save_dir, "train_sdfs.npy"), train_sdf)
np.save(os.path.join(save_dir, "train_temperatures.npy"), train_temp)

# 保存验证集
np.save(os.path.join(save_dir, "val_heat_sources.npy"), val_heat)
np.save(os.path.join(save_dir, "val_sdfs.npy"), val_sdf)
np.save(os.path.join(save_dir, "val_temperatures.npy"), val_temp)

# 保存测试集（新增）
np.save(os.path.join(save_dir, "test_heat_sources.npy"), test_heat)
np.save(os.path.join(save_dir, "test_sdfs.npy"), test_sdf)
np.save(os.path.join(save_dir, "test_temperatures.npy"), test_temp)

# 单独保存共享的散热口掩码（所有子集共用）
np.save(os.path.join(save_dir, "heatsink_mask.npy"), heatsink_mask)

# --------------------------- 验证结果 ---------------------------
print(f"划分完成！数据保存在：{save_dir}")
print(f"训练集样本数：{len(train_indices)}（预期3800），形状：{train_heat.shape}")
print(f"验证集样本数：{len(val_indices)}（预期200），形状：{val_heat.shape}")
print(f"测试集样本数：{len(test_indices)}（预期1000），形状：{test_heat.shape}")
print(f"散热口掩码形状：{heatsink_mask.shape}")