'''
Author: wangqineng zhet3988009@gmail.com
Date: 2025-07-08 14:49:23
LastEditors: wangqineng zhet3988009@gmail.com
LastEditTime: 2025-07-10 11:19:44
FilePath: /EngTestTool/paraturbo/Satellite_2d.py
Description: 

Copyright (c) 2025 by wangqineng, All Rights Reserved. 
'''
import numpy as np
from abc import ABC, abstractmethod
import yaml
from typing import Union
import matplotlib.pyplot as plt

__all__ = [
    'Satellite2DLayout',
]

class Satellite2DLayout:
    """
    二维卫星元件布局类，支持矩形/胶囊型元件的添加、边界条件设置、重叠检测、数据导出等功能。
    假设仅考虑热传导，不考虑对流与辐射。
    """

    def __init__(self, size: tuple, kappa: float, resolution: int):
        """
        构造函数
        :param size: (width, height) 布局区域尺寸
        :param kappa: 材料热导率
        :param resolution: 网格分辨率（每边划分多少格）
        """
        self.size = size
        self.kappa = kappa
        self.resolution = resolution
        self.components = []  # 存储所有元件信息
        self.boundary_conditions = {
            "Dirichlet": [],  # [(边界名, T0)]
            "Neumann": []     # [边界名]
        }

    def add_component(self, shape: str, center: tuple, size: tuple, power: float, preset_id: int = None, **kwargs):
        """
        增加元件
        :param shape: 'rectangle' 或 'capsule'
        :param center: (x, y) 元件中心坐标
        :param size: 
            - rectangle: (w, h)
            - capsule: (l, r)  # l为中间矩形长度，r为半圆半径，整体高度为2*r
        :param power: 元件功率
        :param preset_id: 可选，预置参数编号（如有）
        :param kwargs: 其他参数
        """
        comp = {
            "shape": shape,
            "center": center,
            "size": size,
            "power": power,
            "preset_id": preset_id
        }
        comp.update(kwargs)
        self.components.append(comp)

    def set_boundary_condition(self, boundary: str, bc_type: str, value: float = None):
        """
        设置边界条件
        :param boundary: 'left', 'right', 'top', 'bottom'
        :param bc_type: 'Dirichlet' 或 'Neumann'
        :param value: Dirichlet温度值，Neumann时可省略
        """
        if bc_type == "Dirichlet":
            self.boundary_conditions["Dirichlet"].append((boundary, value))
        elif bc_type == "Neumann":
            self.boundary_conditions["Neumann"].append(boundary)
        else:
            raise ValueError("bc_type must be 'Dirichlet' or 'Neumann'.")

    def check_overlap(self) -> bool:
        """
        检查所有元件是否有重叠，若有重叠返回True，否则False
        """
        def rect_overlap(a, b):
            ax, ay = a["center"]
            aw, ah = a["size"]
            bx, by = b["center"]
            bw, bh = b["size"]
            return (abs(ax - bx) * 2 < (aw + bw)) and (abs(ay - by) * 2 < (ah + bh))

        def capsule_overlap(a, b):
            # 胶囊元件近似用外接矩形+端点圆心距离判断
            # a, b: {"center": (x, y), "size": (l, r)}
            ax, ay = a["center"]
            al, ar = a["size"]
            bx, by = b["center"]
            bl, br = b["size"]
            # 先判外接矩形
            if not rect_overlap(
                {"center": (ax, ay), "size": (al + 2 * ar, 2 * ar)},
                {"center": (bx, by), "size": (bl + 2 * br, 2 * br)}
            ):
                return False
            # 再判端点圆心距离
            a_left = (ax - al / 2, ay)
            a_right = (ax + al / 2, ay)
            b_left = (bx - bl / 2, by)
            b_right = (bx + bl / 2, by)
            # 只要有一对端点距离小于半径和就算重叠
            for pa in [a_left, a_right]:
                for pb in [b_left, b_right]:
                    if np.hypot(pa[0] - pb[0], pa[1] - pb[1]) < (ar + br):
                        return True
            return True  # 保守起见，外接矩形重叠就算重叠

        def rect_capsule_overlap(rect, capsule):
            # 先判外接矩形
            rx, ry = rect["center"]
            rw, rh = rect["size"]
            cx, cy = capsule["center"]
            cl, cr = capsule["size"]
            if not rect_overlap(
                {"center": (rx, ry), "size": (rw, rh)},
                {"center": (cx, cy), "size": (cl + 2 * cr, 2 * cr)}
            ):
                return False
            # 进一步可加更精细的判定（略）
            return True

        n = len(self.components)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.components[i], self.components[j]
                if a["shape"] == "rectangle" and b["shape"] == "rectangle":
                    if rect_overlap(a, b):
                        return True
                elif a["shape"] == "capsule" and b["shape"] == "capsule":
                    if capsule_overlap(a, b):
                        return True
                else:
                    rect = a if a["shape"] == "rectangle" else b
                    capsule = b if a["shape"] == "capsule" else a
                    if rect_capsule_overlap(rect, capsule):
                        return True
        return False

    def to_yaml(self, filename: str):
        """
        导出当前布局为yaml文件
        """
        data = {
            "size": self.size,
            "kappa": self.kappa,
            "resolution": self.resolution,
            "components": self.components,
            "boundary_conditions": self.boundary_conditions
        }
        with open(filename, "w") as f:
            yaml.dump(data, f, allow_unicode=True)

    def to_fenicsx_data(self):
        """
        输出fenicsx可用的数据格式（如元件分布、功率分布、边界条件等）
        返回dict，便于后续直接用于建模
        """
        fenics_data = {
            "domain": self.size,
            "kappa": self.kappa,
            "resolution": self.resolution,
            "sources": [
                {
                    "shape": c["shape"],
                    "center": c["center"],
                    "size": c["size"],
                    "power": c["power"]
                } for c in self.components
            ],
            "boundary_conditions": self.boundary_conditions
        }
        return fenics_data

    def signed_distance_field(self, grid_res: int = None):
        """
        输出有向距离函数（SDF），用于深度学习训练
        :param grid_res: 网格分辨率，若为None则用self.resolution
        :return: sdf: np.ndarray, shape=(grid_res, grid_res)
        """
        if grid_res is None:
            grid_res = self.resolution
        w, h = self.size
        xs = np.linspace(0, w, grid_res)
        ys = np.linspace(0, h, grid_res)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        sdf = np.full((grid_res, grid_res), np.inf)
        for comp in self.components:
            if comp["shape"] == "rectangle":
                cx, cy = comp["center"]
                rw, rh = comp["size"]
                dx = np.abs(xx - cx) - rw / 2
                dy = np.abs(yy - cy) - rh / 2
                outside = np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2
                inside = np.minimum(np.maximum(dx, dy), 0)
                dist = np.sqrt(outside) + inside
            elif comp["shape"] == "capsule":
                # 胶囊体：中间矩形+两端半圆
                cx, cy = comp["center"]
                l, r = comp["size"]
                # 坐标系以胶囊中心为原点
                x0 = xx - cx
                y0 = yy - cy
                # 胶囊主轴水平，长度l，高度2r
                # 1. 计算到中间矩形的有向距离
                dx = np.abs(x0) - l / 2
                dy = np.abs(y0) - r
                outside = np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2
                inside = np.minimum(np.maximum(dx, dy), 0)
                dist_rect = np.sqrt(outside) + inside
                # 2. 计算到两端半圆的有向距离
                left_circle = np.hypot(x0 + l / 2, y0) - r
                right_circle = np.hypot(x0 - l / 2, y0) - r
                # 3. 合并：在矩形区用dist_rect，否则取左右半圆的距离
                mask_rect = (np.abs(x0) <= l / 2)
                dist = np.where(mask_rect, dist_rect, np.minimum(left_circle, right_circle))
            else:
                continue
            # 有向距离，取所有元件的最小值
            sdf = np.minimum(sdf, dist)
        return sdf

    def visualize(self, show_sdf: bool = False, grid_res: int = None):
        """
        可视化当前布局，包括所有元件和边界条件。
        :param show_sdf: 是否显示有向距离场（SDF）
        :param grid_res: SDF分辨率，默认为self.resolution
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        w, h = self.size

        # 绘制元件
        for comp in self.components:
            if comp["shape"] == "rectangle":
                cx, cy = comp["center"]
                rw, rh = comp["size"]
                rect = plt.Rectangle((cx - rw / 2, cy - rh / 2), rw, rh,
                                     edgecolor='b', facecolor='cyan', alpha=0.5)
                ax.add_patch(rect)
                ax.text(cx, cy, f'{comp["power"]:.1f}', color='k', ha='center', va='center')
            elif comp["shape"] == "capsule":
                cx, cy = comp["center"]
                l, r = comp["size"]
                # 中间矩形
                rect = plt.Rectangle((cx - l / 2, cy - r), l, 2 * r,
                                     edgecolor='r', facecolor='pink', alpha=0.5)
                ax.add_patch(rect)
                # 左半圆
                circ_left = plt.Circle((cx - l / 2, cy), r, edgecolor='r', facecolor='pink', alpha=0.5)
                ax.add_patch(circ_left)
                # 右半圆
                circ_right = plt.Circle((cx + l / 2, cy), r, edgecolor='r', facecolor='pink', alpha=0.5)
                ax.add_patch(circ_right)
                ax.text(cx, cy, f'{comp["power"]:.1f}', color='k', ha='center', va='center')

        # 绘制边界
        ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], 'k-', lw=2)
        for bc_type, bcs in self.boundary_conditions.items():
            for bc in bcs:
                if bc_type == "Dirichlet":
                    boundary, value = bc
                    color = 'blue'
                    label = f'Dirichlet {value}'
                else:
                    boundary = bc
                    color = 'green'
                    label = 'Neumann'
                # 边界线段
                if boundary == "left":
                    ax.plot([0, 0], [0, h], color=color, lw=4, label=label)
                elif boundary == "right":
                    ax.plot([w, w], [0, h], color=color, lw=4, label=label)
                elif boundary == "bottom":
                    ax.plot([0, w], [0, 0], color=color, lw=4, label=label)
                elif boundary == "top":
                    ax.plot([0, w], [h, h], color=color, lw=4, label=label)

        # 可选：显示SDF
        if show_sdf:
            sdf = self.signed_distance_field(grid_res)
            extent = [0, w, 0, h]
            im = ax.imshow(sdf.T, extent=extent, origin='lower', cmap='coolwarm', alpha=0.4)
            plt.colorbar(im, ax=ax, label='Signed Distance')

        ax.set_xlim(-w * 0.05, w * 1.05)
        ax.set_ylim(-h * 0.05, h * 1.05)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Satellite 2D Layout')
        plt.tight_layout()
        plt.show()

# 主函数测试
if __name__ == "__main__":
    # 创建布局对象
    layout = Satellite2DLayout(size=(10, 8), kappa=200, resolution=100)
    # 添加矩形元件
    layout.add_component('rectangle', center=(3, 4), size=(2, 1), power=5.0)
    # 添加胶囊型元件
    layout.add_component('capsule', center=(7, 6), size=(3.0, 1.0), power=8.0)
    # 设置边界条件
    layout.set_boundary_condition('left', 'Dirichlet', value=20.0)
    layout.set_boundary_condition('right', 'Neumann')
    # 检查重叠
    print("是否有重叠：", layout.check_overlap())
    # 导出yaml
    layout.to_yaml("layout_example.yaml")
    # 输出fenicsx数据
    fenics_data = layout.to_fenicsx_data()
    print("fenicsx数据：", fenics_data)
    # 可视化
    layout.visualize(show_sdf=True)






