'''
Author: wangqineng zhet3988009@gmail.com
Date: 2025-08-03 10:00:00
LastEditors: wangqineng zhet3988009@gmail.com
LastEditTime: 2025-08-03 15:00:00
FilePath: /EngTestTool/pythonForFenics/temperature_field_solver.py
Copyright (c) 2025 by wangqineng, All Rights Reserved.
'''
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import List, Tuple, Dict, Optional

# 手动添加模块所在目录到Python搜索路径
module_dir = "/data/zxr/inr/SimDriveSate/pythonForFenics"
if module_dir not in sys.path:
    sys.path.append(module_dir)

from .fenicsx_solver import run_solver


class TemperatureFieldSolver:
    """温度场求解器类"""
    def __init__(self,
                 layout_domain: Tuple[float, float],  # 布局域尺寸 (width, height)
                 mesh_size: Tuple[int, int]):        # 网格尺寸 (N×M) = (y方向网格数, x方向网格数)
        # 布局域参数
        self.layout_width, self.layout_height = layout_domain  # (x方向长度, y方向长度)
        self.mesh_N, self.mesh_M = mesh_size                  # (y方向网格数, x方向网格数)

        # 网格物理参数
        self.grid_width = self.layout_width / self.mesh_M    # x方向网格单元尺寸
        self.grid_height = self.layout_height / self.mesh_N  # y方向网格单元尺寸

        # 网格点数（矩阵维度）
        self.total_nodes = (self.mesh_N + 1, self.mesh_M + 1)  # (y方向点数, x方向点数)

        # 存储计算结果
        self.source_matrix = None  # 热源矩阵F
        self.temperature_field = None  # 温度场结果
        self.function_space = None  # 有限元函数空间
        self.boundary_conditions = []  # 边界条件列表

        # 数值计算容差
        self.tolerance = 1e-8

    def set_boundary_conditions(self, bcs: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        """设置边界条件
        参数:
            bcs: 边界条件列表，每个元素为线段的两个端点 [(x1,y1), (x2,y2)]
        """
        self.boundary_conditions = bcs
        return self  # 支持链式调用

    def generate_source_matrix(self, heat_sources: List[Dict]) -> np.ndarray:
        """生成热源矩阵
        参数:
            heat_sources: 热源列表，每个热源为字典，格式：
                - 矩形: {"shape": "rect", "center": (x,y), "width": w, "height": h, "power": p}
                - 圆形: {"shape": "circle", "center": (x,y), "radius": r, "power": p}
                - 胶囊型: {"shape": "capsule", "center": (x,y), "length": l, "width": w, "power": p}
        返回:
            热源矩阵F (shape: (ny, nx))
        """
        # 初始化热源矩阵（基于类初始化的节点数/网格数
        F = np.zeros(self.total_nodes, dtype=np.float64)  # 形状: (ny+1, nx+1)
        # 生成坐标网格（直接用类中布局域和网格参数）
        x_coords = np.linspace(0, self.layout_width, self.mesh_M + 1)
        y_coords = np.linspace(0, self.layout_height, self.mesh_N + 1)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')

        # 遍历热源，生成掩码并赋值
        for source in heat_sources:
            shape = source["shape"]
            cx, cy = source["center"]
            power = source["power"]
            tol = self.tolerance  # 直接用类中定义的容差

            if shape == "rect":
                w, h = source["width"], source["height"]
                mask = (X >= cx - w / 2 + tol) & (X <= cx + w / 2 - tol) & \
                        (Y >= cy - h / 2 + tol) & (Y <= cy + h / 2 - tol)

            elif shape == "circle":
                r = source["radius"]
                mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2 + tol

            elif shape == "capsule":
                length, width = source["length"], source["width"]
                radius = width / 2
                rect_len = length - width
                # 中间矩形
                rect_mask = (X >= cx - rect_len / 2 + tol) & (X <= cx + rect_len / 2 - tol) & \
                            (Y >= cy - radius + tol) & (Y <= cy + radius - tol)
                # 两端半圆
                left_mask = (X <= cx - rect_len / 2 + tol) & \
                            ((X - (cx - rect_len / 2)) ** 2 + (Y - cy) ** 2 <= radius ** 2 + tol)
                right_mask = (X >= cx + rect_len / 2 - tol) & \
                            ((X - (cx + rect_len / 2)) ** 2 + (Y - cy) ** 2 <= radius ** 2 + tol)
                mask = rect_mask | left_mask | right_mask

            else:
                raise ValueError(f"不支持的形状: {shape}")

            F[mask] = power  # 赋值功率

            self.source_matrix = F
        return F

    def solve(self, u0: float = 298.0) -> Tuple[np.ndarray, object]:
        """求解温度场
        参数:
            u0: 边界条件温度值（默认298K）
        返回:
            温度场结果和有限元函数空间
        """
        if self.source_matrix is None:
            raise RuntimeError("请先调用generate_source_matrix生成热源矩阵")

        if not self.boundary_conditions:
            print("警告: 未设置边界条件，将使用默认边界")

        # 调用FenicsX求解器
        u_sol, V = run_solver(
            ndim=2,
            length_x=self.layout_width,
            length_y=self.layout_height,
            bcs=self.boundary_conditions,
            u0=u0,
            nx=self.mesh_M,  # x方向网格数 (对应mesh_size的M)
            ny=self.mesh_N,  # y方向网格数 (对应mesh_size的N)
            F=self.source_matrix
        )

        self.temperature_field = u_sol
        self.function_space = V
        return u_sol, V

    def visualize_source_distribution(self, save_path: Optional[str] = None):
        """可视化热源分布
        参数:
            save_path: 保存路径（None则直接显示）
        """
        if self.source_matrix is None:
            raise RuntimeError("请先生成热源矩阵（调用generate_source_matrix）")

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            self.source_matrix,
            origin='lower',
            extent=[0, self.layout_width, 0, self.layout_height],
            aspect='auto',
            cmap='viridis',
            vmin=0
        )

        # 绘制边界条件
        for bc in self.boundary_conditions:
            (x1, y1), (x2, y2) = bc
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
            plt.plot([x1, x2], [y1, y2], 'ro', markersize=6)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.colorbar(im, label='Heat Source Intensity (W)')
        plt.title('Heat Source Distribution')
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"热源分布图已保存至: {save_path}")
        else:
            plt.show()
        plt.close()

    def visualize_temperature_field(self, save_path: Optional[str] = None, interpolate_res: int = 200):
        """可视化温度场分布
        参数:
            save_path: 保存路径（None则直接显示）
            interpolate_res: 插值分辨率（默认200x200）
        """
        if self.temperature_field is None or self.function_space is None:
            raise RuntimeError("请先求解温度场（调用solve方法）")

        # 获取自由度坐标和解值
        dof_coords = self.function_space.tabulate_dof_coordinates()
        x_coords = dof_coords[:, 0]
        y_coords = dof_coords[:, 1]
        solution_values = self.temperature_field.x.array

        # 生成插值网格
        grid_x, grid_y = np.mgrid[
            np.min(x_coords):np.max(x_coords):interpolate_res*1j,
            np.min(y_coords):np.max(y_coords):interpolate_res*1j
        ]

        # 插值到规则网格
        temp_grid = griddata(
            (x_coords, y_coords),
            solution_values,
            (grid_x, grid_y),
            method='cubic'
        )

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            temp_grid,
            cmap='inferno',
            aspect='auto',
            origin='lower',
            extent=[
                np.min(x_coords), np.max(x_coords),
                np.min(y_coords), np.max(y_coords)
            ]
        )

        cbar = plt.colorbar(im)
        cbar.set_label('Temperature (K)')
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title('2D Temperature Field')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"温度场图已保存至: {save_path}")
        else:
            plt.show()
        plt.close()


# 示例调用（可直接运行测试）
if __name__ == "__main__":
    # 1. 定义布局参数（与SeqLS完全一致）
    layout_domain = (0.1, 0.1)  # (width, height) = (x方向长度, y方向长度)
    mesh_size = (256, 256)      # (N, M) = (y方向网格数, x方向网格数)

    # 2. 创建温度场求解器实例
    temp_solver = TemperatureFieldSolver(
        layout_domain=layout_domain,
        mesh_size=mesh_size
    )

    # 3. 设置边界条件
    boundary_conditions = [
        ([0.03, 0.0], [0.05, 0.0]),  # 底部边界线段
        ([0.0, 0.01], [0.0, 0.02])   # 左侧边界线段
    ]
    temp_solver.set_boundary_conditions(boundary_conditions)

    # 4. 定义热源（格式与SeqLS元件兼容）
    heat_sources = [
        {
            "shape": "capsule",
            "center": (0.05, 0.05),
            "length": 0.06,
            "width": 0.02,
            "power": 5000
        },
        {
            "shape": "rect",
            "center": (0.08, 0.03),
            "width": 0.03,
            "height": 0.01,
            "power": 3000
        },
        {
            "shape": "circle",
            "center": (0.02, 0.08),
            "radius": 0.01,
            "power": 2000
        }
    ]

    # 5. 生成热源矩阵并可视化
    temp_solver.generate_source_matrix(heat_sources)
    temp_solver.visualize_source_distribution("heat_source_dist.png")

    # 6. 求解温度场并可视化
    temp_solver.solve(u0=298.0)
    temp_solver.visualize_temperature_field("temperature_field.png")

    print("温度场求解完成！")