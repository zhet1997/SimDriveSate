"""
数据生成器：生成元件布局、热源矩阵、温度场、SDF（有符号距离场）
"""

import os
import yaml
import random
import numpy as np
import json
import logging
import sys
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

# 路径配置（根据实际项目调整）
module_dir = "/data/zxr/inr/SimDriveSate/pythonForFenics"
samply_dir = "/data/zxr/inr/SimDriveSate/layout"
sys.path.append(module_dir)
sys.path.append(samply_dir)

# 导入依赖
from SeqLS import SeqLS  # 布局算法
from test_fenicsx_solver import TemperatureFieldSolver  # 温度场求解器
from utils import plot_sdf, plot_layout, plot_heat_source, plot_temperature_field

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def compute_sdf(components: List[Dict], grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """计算有符号距离场（SDF）"""
    sdf = np.full_like(grid_x, np.inf)  # 初始化距离为无穷大
    h, w = grid_x.shape  # 256×256
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # 展平为点集

    for comp in components:
        shape = comp["shape"]
        cx, cy = comp["center"]

        if shape == "rect":
            # 矩形SDF计算
            w_rect, h_rect = comp["width"], comp["height"]
            x_min, x_max = cx - w_rect/2, cx + w_rect/2
            y_min, y_max = cy - h_rect/2, cy + h_rect/2

            # 外部距离：到矩形边界的最小距离
            dx = np.max([x_min - points[:, 0], points[:, 0] - x_max, np.zeros_like(points[:, 0])], axis=0)
            dy = np.max([y_min - points[:, 1], points[:, 1] - y_max, np.zeros_like(points[:, 1])], axis=0)
            dist = np.sqrt(dx**2 + dy**2)

            # 内部距离：到最近边界的负距离
            inside = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
                     (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
            dist[inside] = -np.min([
                points[:, 0][inside] - x_min,
                x_max - points[:, 0][inside],
                points[:, 1][inside] - y_min,
                y_max - points[:, 1][inside]
            ], axis=0)

        elif shape == "circle":
            # 圆形SDF计算：点到圆心距离 - 半径
            r = comp["radius"]
            dist = cdist(points, [[cx, cy]]).ravel() - r

        elif shape == "capsule":
            # 胶囊型SDF计算（矩形+两端半圆）
            length, width = comp["length"], comp["width"]
            r = width / 2
            rect_len = length - width  # 中间矩形长度
            p1 = np.array([cx - rect_len/2, cy])  # 左端点
            p2 = np.array([cx + rect_len/2, cy])  # 右端点

            # 到线段p1-p2的距离
            vec = p2 - p1
            t = np.clip(((points - p1) @ vec) / (vec @ vec + 1e-8), 0, 1)
            closest = p1 + t[:, None] * vec
            dist_line = np.linalg.norm(points - closest, axis=1) - r

            # 到两端半圆的距离
            dist_circle1 = cdist(points, [p1]).ravel() - r
            dist_circle2 = cdist(points, [p2]).ravel() - r

            # 取最小值作为胶囊距离
            dist = np.min([dist_line, dist_circle1, dist_circle2], axis=0)

        else:
            raise ValueError(f"不支持的形状: {shape}")

        # 更新SDF（取到所有元件的最小距离）
        sdf = np.min([sdf, dist.reshape(h, w)], axis=0)

    return sdf


# --------------------------- 数据生成器核心类 ---------------------------
class DataGenerator:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)

        # 核心参数
        self.layout_domain = self.config["layout_domain"]  # (宽度, 高度) [m]
        self.mesh_size = self.config["mesh_size"]          # (y网格数, x网格数) → (256, 256)
        self.u0 = self.config["boundary_temperature"]      # 边界温度 [K]
        self.bcs = self.config["boundary_conditions"]      # 边界条件
        self.num_samples = self.config["num_samples"]      # 总样本数
        self.mesh_M = self.mesh_size[1]                    # x方向网格数（256）
        self.mesh_N = self.mesh_size[0]                    # y方向网格数（256）
        self.interpolate_res = self.mesh_size              # 插值分辨率（256, 256）

        # 存储路径
        self.root_dir = self.config["save_root"]
        self.all_samples_dir = os.path.join(self.root_dir, "all_samples")
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.all_samples_dir, exist_ok=True)

        # 初始化总汇总文件
        self._init_all_files()

        # 从配置加载功率参数
        self._init_power_options()

    def _init_power_options(self):
        """初始化功率选项"""
        power_cfg = self.config["component"]["power"]
        if isinstance(power_cfg, dict):
            self.power_min = power_cfg["min"]
            self.power_max = power_cfg["max"]
            self.power_step = power_cfg.get("step", 1000)
        else:
            self.power_min, self.power_max = power_cfg
            self.power_step = 1000
        self.power_options = list(range(self.power_min, self.power_max + 1, self.power_step))

    def _init_all_files(self):
        """初始化总汇总文件（含SDF）"""
        # 总元件信息JSON
        self.all_components_path = os.path.join(self.all_samples_dir, "all_components.json")
        if not os.path.exists(self.all_components_path):
            with open(self.all_components_path, "w") as f:
                json.dump([], f)

        # 总矩阵文件（256×256）
        self.all_heat_path = os.path.join(self.all_samples_dir, "all_heat_sources.npy")
        self.all_temp_path = os.path.join(self.all_samples_dir, "all_temperatures.npy")
        self.all_sdf_path = os.path.join(self.all_samples_dir, "all_sdfs.npy")  # 新增SDF汇总

        # 初始化空矩阵
        if not os.path.exists(self.all_heat_path):
            np.save(self.all_heat_path, np.empty((0, *self.interpolate_res), dtype=np.float32))
        if not os.path.exists(self.all_temp_path):
            np.save(self.all_temp_path, np.empty((0, *self.interpolate_res), dtype=np.float32))
        if not os.path.exists(self.all_sdf_path):
            np.save(self.all_sdf_path, np.empty((0, *self.interpolate_res), dtype=np.float32))

    def _load_config(self, config_path: str = None) -> dict:
        """加载配置文件"""
        default_config = {
            "layout_domain": (0.2, 0.2),
            "mesh_size": (256, 256),
            "num_samples": 3000,
            "save_root": "./thermal_data",
            "boundary_temperature": 298.0,
            "boundary_conditions": [
                ([0.0, 0.0], [0.02, 0.0]),
                ([0.0, 0.02], [0.02, 0.02])
            ],
            "component": {
                "shapes": ["rect", "circle", "capsule"],
                "shape_probs": [0.8, 0.1, 0.1],
                "distribution": "uniform",
                "rect": {"width": [0.006, 0.024], "height": [0.006, 0.024]},
                "circle": {"radius": [0.003, 0.012]},
                "capsule": {"length": [0.012, 0.048], "width": [0.006, 0.024]},
                "power": {"min": 4000, "max": 20000, "step": 1000}
            }
        }
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)
            self._update_config(default_config, user_config)
        return default_config

    def _update_config(self, default: dict, user: dict) -> None:
        """合并配置"""
        for k, v in user.items():
            if isinstance(v, dict) and k in default and isinstance(default[k], dict):
                self._update_config(default[k], v)
            else:
                default[k] = v

    def _convert_to_jsonable(self, data: Any) -> Any:
        """转换为JSON可序列化类型"""
        if isinstance(data, (np.float32, np.float64)):
            return round(float(data), 4)
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._convert_to_jsonable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._convert_to_jsonable(v) for v in data)
        return data

    def _generate_component(self, comp_id: int) -> dict:
        """生成单个元件信息"""
        cfg = self.config["component"]
        shape = random.choices(cfg["shapes"], weights=cfg["shape_probs"])[0]
        params = {"id": comp_id, "shape": shape}

        rand_func = random.uniform if cfg["distribution"] == "uniform" else \
            lambda *x: np.clip(np.random.normal(x[0], (x[1] - x[0]) / 3), x[0], x[1])

        # 尺寸参数（保留3位小数）
        if shape == "rect":
            params["width"] = round(rand_func(*cfg["rect"]["width"]), 3)
            params["height"] = round(rand_func(*cfg["rect"]["height"]), 3)
        elif shape == "circle":
            params["radius"] = round(rand_func(*cfg["circle"]["radius"]), 3)
        elif shape == "capsule":
            params["length"] = round(rand_func(*cfg["capsule"]["length"]), 3)
            params["width"] = round(rand_func(*cfg["capsule"]["width"]), 3)

        # 功率（1000的倍数）
        params["power"] = random.choice(self.power_options)
        return params

    def _generate_9_components(self) -> list:
        """生成9个元件"""
        return [self._generate_component(i) for i in range(9)]

    def _save_single_sample(self, sample_idx: int, data: Dict) -> None:
        """保存单个样本（含SDF）"""
        sample_num = sample_idx + 1
        sample_dir = os.path.join(self.root_dir, f"sample_{sample_num:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        # 1. 保存元件信息
        comp_path = os.path.join(sample_dir, "components.json")
        with open(comp_path, "w") as f:
            json.dump(data["components"], f, indent=2)

        # 2. 保存热源矩阵（256×256）
        heat_path = os.path.join(sample_dir, "heat_source.npy")
        np.save(heat_path, data["heat_source"])

        # 3. 保存温度场矩阵（256×256）
        temp_path = os.path.join(sample_dir, "temperature.npy")
        np.save(temp_path, data["temperature"])

        # 4. 保存SDF矩阵（256×256）
        sdf_path = os.path.join(sample_dir, "sdf.npy")
        np.save(sdf_path, data["sdf"])

        # 5. 保存坐标范围
        coords_path = os.path.join(sample_dir, "coords_range.npy")
        np.save(coords_path, {
            "x_min": data["x_range"][0],
            "x_max": data["x_range"][1],
            "y_min": data["y_range"][0],
            "y_max": data["y_range"][1]
        })
        # 6. 生成并保存可视化图片
        logger.info(f"保存布局图: {os.path.join(sample_dir, 'layout.png')}")
        plot_layout(
            components=data["components"],
            layout_domain=self.layout_domain,
            mesh_size=self.mesh_size,
            save_path=os.path.join(sample_dir, "layout.png")
        )
        logger.info(f"保存热源图: {os.path.join(sample_dir, 'heat_source.png')}")
        plot_heat_source(
            source_matrix=data["heat_source"],
            layout_domain=self.layout_domain,
            bcs=self.bcs,
            save_path=os.path.join(sample_dir, "heat_source.png")
        )
        logger.info(f"保存温度场图: {os.path.join(sample_dir, 'temperature.png')}")
        plot_temperature_field(
            temp_matrix=data["temperature"],
            x_range=data["x_range"],
            y_range=data["y_range"],
            save_path=os.path.join(sample_dir, "temperature.png")
        )
        logger.info(f"保存SDF图: {os.path.join(sample_dir, 'sdf.png')}")
        plot_sdf(
            sdf_matrix=data["sdf"],
            x_range=data["x_range"],
            y_range=data["y_range"],
            save_path=os.path.join(sample_dir, "sdf.png")
        )

    def _update_all_files(self, sample_data: Dict) -> None:
        """更新总汇总文件（含SDF）"""
        # 1. 追加元件信息
        with open(self.all_components_path, "r+") as f:
            all_components = json.load(f)
            all_components.append(sample_data["components"])
            f.seek(0)
            json.dump(all_components, f, indent=2)

        # 2. 追加热源矩阵
        all_heat = np.load(self.all_heat_path)
        new_heat = sample_data["heat_source"][np.newaxis, ...]
        np.save(self.all_heat_path, np.concatenate([all_heat, new_heat], axis=0))

        # 3. 追加温度场矩阵
        all_temp = np.load(self.all_temp_path)
        new_temp = sample_data["temperature"][np.newaxis, ...]
        np.save(self.all_temp_path, np.concatenate([all_temp, new_temp], axis=0))

        # 4. 追加SDF矩阵
        all_sdf = np.load(self.all_sdf_path)
        new_sdf = sample_data["sdf"][np.newaxis, ...]
        np.save(self.all_sdf_path, np.concatenate([all_sdf, new_sdf], axis=0))

    def generate_batch(self) -> None:
        """批量生成样本（含SDF）"""
        for sample_idx in range(self.num_samples):
            sample_num = sample_idx + 1
            try:
                logger.info(f"生成样本 {sample_num}/{self.num_samples}")

                # 1. 生成9个元件
                components = self._generate_9_components()

                # 2. 布局
                seq_ls = SeqLS(self.layout_domain, self.mesh_size)
                placed_components = seq_ls.layout_sampling(
                    components=components,
                    max_sampling_attempts=500
                )
                if len(placed_components) != 9:
                    logger.warning(f"样本 {sample_num} 布局不完整，跳过")
                    continue

                # 处理坐标格式
                for comp in placed_components:
                    if isinstance(comp["center"], np.ndarray):
                        comp["center"] = tuple(round(float(c), 4) for c in comp["center"])
                    else:
                        comp["center"] = tuple(round(c, 4) for c in comp["center"])

                # 3. 生成热源和温度场
                temp_solver = TemperatureFieldSolver(self.layout_domain, self.mesh_size)
                temp_solver.set_boundary_conditions(self.bcs)

                # 转换为热源格式
                heat_sources = []
                for comp in placed_components:
                    src = {
                        "shape": comp["shape"],
                        "center": comp["center"],
                        "power": comp["power"]
                    }
                    if comp["shape"] == "rect":
                        src["width"], src["height"] = comp["width"], comp["height"]
                    elif comp["shape"] == "circle":
                        src["radius"] = comp["radius"]
                    elif comp["shape"] == "capsule":
                        src["length"], src["width"] = comp["length"], comp["width"]
                    heat_sources.append(src)

                # 热源矩阵（257×257原始矩阵）
                heat_source_257 = temp_solver.generate_source_matrix(heat_sources)

                # 4. 温度场求解与插值（256×256）
                u_sol, V = temp_solver.solve(u0=self.u0)
                dof_coords = V.tabulate_dof_coordinates()
                x = dof_coords[:, 0]
                y = dof_coords[:, 1]
                solution_values = u_sol.x.array

                x_range = (np.min(x), np.max(x))
                y_range = (np.min(y), np.max(y))

                # 生成256×256规则网格（统一用于温度场、热源、SDF）
                x_step = self.layout_domain[0] / self.mesh_M
                y_step = self.layout_domain[1] / self.mesh_N
                x_256 = np.linspace(x_step/2, self.layout_domain[0] - x_step/2, self.mesh_M)
                y_256 = np.linspace(y_step/2, self.layout_domain[1] - y_step/2, self.mesh_N)
                X_256, Y_256 = np.meshgrid(x_256, y_256, indexing='xy')  # 256×256网格坐标

                # 温度场插值（256×256）
                temp_grid = griddata(
                    (x, y),
                    solution_values,
                    (X_256, Y_256),
                    method='cubic'
                )
                temp_grid[np.isnan(temp_grid)] = self.u0  # 处理NaN

                # 5. 热源矩阵插值为256×256
                x_257 = np.linspace(0, self.layout_domain[0], self.mesh_M + 1)
                y_257 = np.linspace(0, self.layout_domain[1], self.mesh_N + 1)
                X_257, Y_257 = np.meshgrid(x_257, y_257, indexing='xy')

                F_256 = griddata(
                    (X_257.ravel(), Y_257.ravel()),
                    heat_source_257.ravel(),
                    (X_256, Y_256),
                    method='cubic'
                )
                F_256[np.isnan(F_256)] = 0  # 热源外区域填充0

                # 6. 计算SDF（256×256）
                sdf = compute_sdf(
                    components=placed_components,
                    grid_x=X_256,
                    grid_y=Y_256
                )

                # 7. 保存样本数据
                sample_data = {
                    "components": self._convert_to_jsonable(placed_components),
                    "heat_source": F_256,
                    "temperature": temp_grid,
                    "sdf": sdf,
                    "x_range": x_range,
                    "y_range": y_range
                }
                self._save_single_sample(sample_idx, sample_data)
                self._update_all_files(sample_data)

                logger.info(f"样本 {sample_num} 生成完成")

            except Exception as e:
                logger.error(f"样本 {sample_num} 生成失败: {str(e)}", exc_info=True)
                continue

        logger.info(f"所有样本生成完成！\n"
                    f"单个样本路径: {self.root_dir}/sample_XXXX\n"
                    f"总汇总路径: {self.all_samples_dir}")


if __name__ == "__main__":
    # 若有自定义配置，替换为实际路径
    generator = DataGenerator(config_path="config.yaml")
    generator.generate_batch()