'''
Author: wangqineng zhet3988009@gmail.com
Description: 封装所有可视化相关的工具函数，无需初始化类，直接调用
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.interpolate import griddata
from typing import List, Tuple, Dict, Optional, Any


def plot_sdf(
        sdf_matrix: np.ndarray,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        save_path: Optional[str] = None
):
    """绘制有符号距离场（SDF）"""
    plt.figure(figsize=(8, 6))
    # 蓝-白-红配色：内部（负）-边界（0）-外部（正）
    im = plt.imshow(
        sdf_matrix,
        cmap='bwr',
        aspect='auto',
        origin='lower',
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        vmin=-0.02,  # 距离范围（根据布局域调整）
        vmax=0.02
    )
    cbar = plt.colorbar(im)
    cbar.set_label('Signed Distance (m)')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title('Signed Distance Field (SDF)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_layout(
        components: List[Dict],
        layout_domain: Tuple[float, float],
        mesh_size: Tuple[int, int],
        save_path: Optional[str] = None
):
    """
    绘制布局图（对应SeqLS类的可视化功能）
    参数:
        components: 元件列表，需包含"shape"、"center"、"id"、"power"等信息
        layout_domain: 布局域尺寸 (width, height) = (x长度, y长度)
        mesh_size: 网格尺寸 (N, M) = (y方向网格数, x方向网格数)
        show_vem: 是否显示VEM网格节点
        vem_matrix: VEM矩阵（show_vem=True时需要）
        save_path: 保存路径（None则显示图形）
    """
    layout_width, layout_height = layout_domain
    mesh_N, mesh_M = mesh_size
    grid_width = layout_width / mesh_M  # x方向网格单元尺寸
    grid_height = layout_height / mesh_N  # y方向网格单元尺寸

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_xlim(0, layout_width)
    ax.set_ylim(0, layout_height)
    ax.set_aspect('equal')
    ax.set_title("Component Layout")
    ax.set_xlabel("X Coordinate (m)")
    ax.set_ylabel("Y Coordinate (m)")

    # 绘制元件
    for i, comp in enumerate(components):
        cx, cy = comp["center"]
        color = f"C{i}"  # 自动分配颜色

        if comp["shape"] == "rect":
            w, h = comp["width"], comp["height"]
            rect = Rectangle(
                (cx - w/2, cy - h/2), w, h,
                fill=True, facecolor=color, edgecolor='black',
                linewidth=2, alpha=0.7
            )
            ax.add_patch(rect)

        elif comp["shape"] == "circle":
            r = comp["radius"]
            circle = Circle(
                (cx, cy), r,
                fill=True, facecolor=color, edgecolor='black',
                linewidth=2, alpha=0.7
            )
            ax.add_patch(circle)

        elif comp["shape"] == "capsule":
            length, width = comp["length"], comp["width"]
            rect_len = length - width
            radius = width / 2

            # 中间矩形
            rect = Rectangle(
                (cx - rect_len/2, cy - radius), rect_len, width,
                fill=True, facecolor=color, edgecolor='black',
                linewidth=2, alpha=0.7
            )
            ax.add_patch(rect)
            # 两端半圆
            ax.add_patch(Circle((cx - rect_len/2, cy), radius,
                               fill=True, facecolor=color, edgecolor='black', linewidth=2, alpha=0.7))
            ax.add_patch(Circle((cx + rect_len/2, cy), radius,
                               fill=True, facecolor=color, edgecolor='black', linewidth=2, alpha=0.7))

        # 标注元件信息
        ax.text(
            cx, cy,
            f"ID: {comp['id']}\nPower: {comp['power']}W",
            ha='center', va='center', fontsize=7, color='white', fontweight='bold'
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"布局图已保存至: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_heat_source(
        source_matrix: np.ndarray,
        layout_domain: Tuple[float, float],
        bcs: List[Tuple[List[float], List[float]]],
        save_path: Optional[str] = None
):
    """绘制热源矩阵（叠加边界条件，显示为黑色虚线）"""
    plt.figure(figsize=(6, 6))
    width, height = layout_domain

    # 绘制热源矩阵
    im = plt.imshow(
        source_matrix,
        cmap='hot',
        aspect='equal',  # 强制等比例，确保几何形状正确
        origin='lower',
        extent=[0, width, 0, height]
    )

    # 绘制边界条件
    for bc in bcs:
        (x1, y1), (x2, y2) = bc
        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        plt.plot([x1, x2], [y1, y2], 'ro', markersize=6)

    # 颜色条、标签等
    plt.colorbar(im, label='Heat Flux (W/m²)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Heat Source Distribution (with Boundaries)')

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_temperature_field(
        temp_matrix: np.ndarray,  # 256×256插值后的矩阵
        x_range: Tuple[float, float],  # 实际x坐标范围
        y_range: Tuple[float, float],  # 实际y坐标范围
        save_path: Optional[str] = None
):
    """绘制插值后的温度场（使用实际坐标范围）"""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        temp_matrix,
        cmap='inferno',
        aspect='auto',
        origin='lower',
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]]  # 基于实际坐标
    )

    cbar = plt.colorbar(im)
    cbar.set_label('Temperature (K)')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title('2D Temperature Field')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"温度场图已保存至: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()