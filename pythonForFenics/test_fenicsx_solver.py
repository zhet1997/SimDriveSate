'''
Author: wangqineng zhet3988009@gmail.com
Date: 2025-07-08 16:15:09
LastEditors: wangqineng zhet3988009@gmail.com
LastEditTime: 2025-07-10 10:35:03
FilePath: /EngTestTool/pythonForFenics/test_fenicsx_solver.py
Description: 

Copyright (c) 2025 by wangqineng, All Rights Reserved. 
'''
import sys
import os

# 手动添加模块所在目录到 Python 搜索路径
# 目标目录：/data/zxr/inr/SimDriveSate/pythonForFenics/
module_dir = "/data/zxr/inr/SimDriveSate/pythonForFenics"
sys.path.append(module_dir)

# 现在可以正常导入了
from fenicsx_solver import run_solver
import numpy as np
from fenicsx_solver import run_solver
import matplotlib.pyplot as plt

def test_run_solver_simple_heat_source():
    # 2D problem, unit square, single heat source in the center
    ndim = 2
    length = 0.1
    nx = 256  # mesh divisions
    length_unit = length / nx

    # Place a single heat source
    layout_list = [
        ('capsule', 0.05, 0.05, 0.06, 0.02),  # 胶囊：中心(0.05,0.05)，总长度0.06，宽度0.02
        ('rect', 0.08, 0.03, 0.03, 0.01),  # 矩形：中心(0.08,0.03)，宽0.03，高0.01
        ('circle', 0.02, 0.08, 0.01)  # 圆形：中心(0.02,0.08)，半径0.01
    ]
    powers = [5000, 3000, 2000]

    # Place a bc condition 
    u0 = 298.0
    bcs = [([0.03, 0], [0.05, 0]), ([0, 0.01], [0, 0.02])]

    F = generate_source_F(layout_list, powers, length, nx)
    plot_source_and_bc(F, bcs, length, nx)

    # Run solver
    U, V = run_solver(
        ndim=ndim,
        length=length,
        length_unit=length_unit,
        bcs=bcs,
        layout_list=layout_list,
        u0=u0,
        powers=powers,
        nx=nx,
        coordinates=True,
        F=F,
    )
    plot_u(U, V)


def plot_u(uh, V):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata

    # 假设 uh 和 V 已经从求解过程中得到

    # 1. 获取函数空间 V 中所有自由度的坐标
    #    这会返回一个 (num_dofs, 3) 的数组，即使是2D问题，z坐标也存在
    dof_coords = V.tabulate_dof_coordinates()
    x_coords = dof_coords[:, 0]
    y_coords = dof_coords[:, 1]
    # 2. 获取 uh 的自由度值
    #    uh.x.array 包含了与 dof_coords 一一对应的值
    solution_values = uh.x.array

    # 3. 创建一个你想要的规则网格 (例如 100x100)
    grid_x, grid_y = np.mgrid[
                     np.min(x_coords):np.max(x_coords):100j,
                     np.min(y_coords):np.max(y_coords):100j
                     ]
    # 4. 使用 griddata 进行插值
    #    将 (x_coords, y_coords) 上的 solution_values 插值到 (grid_x, grid_y) 上
    U = griddata(
        (x_coords, y_coords),  # 原始散点坐标
        solution_values,  # 原始散点值
        (grid_x, grid_y),  # 目标网格坐标
        method='cubic'  # 插值方法：'linear', 'nearest', 'cubic'
    )

    plt.figure(figsize=(8, 6))
    if U.ndim == 1:
        plt.plot(U)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('1D Solution Curve')
    elif U.ndim == 2:
        im = plt.imshow(U, cmap='viridis', aspect='auto')
        cbar = plt.colorbar(im)
        cbar.set_label('Value')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title('2D Solution Field')
    else:
        raise ValueError('U must be 1D or 2D array')
    plt.tight_layout()
    plt.savefig('debug_solution_new.png')
    plt.show()


def generate_source_F(layout_list, powers, length, nx):
    """
    生成热源分布矩阵F（支持矩形、圆形和胶囊型热源）。
    layout_list: 每个元素为元组，格式根据形状不同：
                 - 矩形: ('rect', center_x, center_y, width, height)
                 - 圆形: ('circle', center_x, center_y, radius)
                 - 胶囊型: ('capsule', center_x, center_y, length, width)
                   （length：胶囊总长度，width：胶囊宽度（半圆直径））
    powers:      [p1, p2, ...]  # 每个热源的功率，与layout_list一一对应
    length:      板边长
    nx:          网格划分数
    返回: F (ndarray, shape=(nx+1, nx+1))  # 热源分布矩阵
    """
    # 初始化热源矩阵（全零，非热源区域为0）
    F = np.zeros((nx + 1, nx + 1))

    # 生成物理坐标网格（将板从0到length均匀划分）
    x = np.linspace(0, length, nx + 1)
    y = np.linspace(0, length, nx + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')  # X/Y为网格点的坐标矩阵

    # 遍历每个热源，根据形状生成区域掩码并赋值功率
    for layout, power in zip(layout_list, powers):
        shape_type = layout[0]  # 第一个元素指定形状类型

        if shape_type == 'rect':
            # 矩形热源：(center_x, center_y, width, height)
            _, center_x, center_y, w, h = layout
            mask = (
                    (X >= center_x - w / 2) & (X <= center_x + w / 2) &
                    (Y >= center_y - h / 2) & (Y <= center_y + h / 2)
            )

        elif shape_type == 'circle':
            # 圆形热源：(center_x, center_y, radius)
            _, center_x, center_y, radius = layout
            mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2

        elif shape_type == 'capsule':
            # 胶囊型热源：中间长方体 + 两端半圆
            # 参数：(center_x, center_y, 总长度L, 宽度W)
            _, center_x, center_y, cap_length, cap_width = layout
            radius = cap_width / 2
            rect_length = cap_length - 2 * radius  # 中间长方体长度

            # 1. 中间长方体掩码
            rect_mask = (
                    (X >= center_x - rect_length / 2) & (X <= center_x + rect_length / 2) &
                    (Y >= center_y - radius) & (Y <= center_y + radius)
            )

            # 2. 左端半圆掩码（修复括号）
            left_circle_mask = (
                    (X <= center_x - rect_length / 2) &  # 条件1
                    ((X - (center_x - rect_length / 2)) ** 2 + (Y - center_y) ** 2 <= radius ** 2)  # 条件2（增加外层括号）
            )

            # 3. 右端半圆掩码（修复括号）
            right_circle_mask = (
                    (X >= center_x + rect_length / 2) &  # 条件1
                    ((X - (center_x + rect_length / 2)) ** 2 + (Y - center_y) ** 2 <= radius ** 2)  # 条件2（增加外层括号）
            )

            mask = rect_mask | left_circle_mask | right_circle_mask
        else:
            raise ValueError(f"不支持的形状类型：{shape_type}，请使用'rect'/'circle'/'capsule'")

        # 对掩码区域赋值功率
        F[mask] = power

    # 可视化热源分布
    plot_source_and_bc(F, [], length, nx)

    return F


def plot_source_and_bc(F, bcs, length, nx):
    # 1. 设置画布和热源分布显示
    plt.figure(figsize=(8, 6))  # 固定画布大小，避免自动调整导致比例失调
    # 使用更适合热源强度的 colormap（如viridis），并明确设置非热源区域（0值）的显示
    im = plt.imshow(
        F,
        origin='lower',
        extent=[0, length, 0, length],
        aspect='equal',
        cmap='viridis',
        vmin=0,
    )

    # 2. 优化边界条件显示（从仅画端点改为画完整线段）
    for bc in bcs:
        # 提取线段的两个端点坐标
        (x1, y1), (x2, y2) = bc
        # 绘制线段（红色实线，更清晰）+ 端点标记（红色圆点）
        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)  # 线段
        plt.plot([x1, x2], [y1, y2], 'ro', markersize=6)  # 端点

    # 3. 添加网格线（可选，帮助对应物理坐标）
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. 完善图例和标签
    plt.colorbar(im, label='Heat Source Intensity (Power)')  # 明确颜色条含义
    plt.title('Heat Source Distribution and Boundary Conditions')  # 更通用的标题
    plt.xlabel('x Coordinate (m)')  # 补充单位（假设为米）
    plt.ylabel('y Coordinate (m)')
    plt.tight_layout()  # 自动调整布局，避免标签被截断

    # 5. 保存图片时提高分辨率，确保清晰
    plt.savefig('debug_sourceF.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭画布，避免后续绘图重叠


if __name__ == "__main__":
    # Run the test
    test_run_solver_simple_heat_source()
    print("Test passed successfully!")
