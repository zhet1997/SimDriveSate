'''
Author: wangqineng zhet3988009@gmail.com
Date: 2025-07-08 16:15:09
LastEditors: wangqineng zhet3988009@gmail.com
LastEditTime: 2025-07-10 10:35:03
FilePath: /EngTestTool/pythonForFenics/test_fenicsx_solver.py
Description: 

Copyright (c) 2025 by wangqineng, All Rights Reserved. 
'''
import numpy as np
from fenicsx_solver import run_solver
import matplotlib.pyplot as plt

def test_run_solver_simple_heat_source():
    # 2D problem, unit square, single heat source in the center
    ndim = 2
    length = 0.1
    nx = 32  # mesh divisions
    length_unit = length / nx
    
    
    
    # Place a single heat source
    layout_list = [(0.08, 0.09, 0.01, 0.02),(0.09, 0.02, 0.01, 0.01)]
    powers = [2000,8000]
    
    
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
    plot_u(U,V)

def plot_u(uh,V):
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
    solution_values,       # 原始散点值
    (grid_x, grid_y),      # 目标网格坐标
    method='cubic'         # 插值方法：'linear', 'nearest', 'cubic'
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
    生成热源分布矩阵F。
    layout_list: [(center_x, center_y, w, h), ...]  # 坐标和宽高均为物理量
    powers:      [p1, p2, ...]                     # 每个热源的功率
    length:      板边长
    nx:          网格划分数
    返回: F (ndarray, shape=(nx+1, nx+1))
    """
    F = np.zeros((nx+1, nx+1))
    x = np.linspace(0, length, nx+1)
    y = np.linspace(0, length, nx+1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    for (center_x, center_y, w, h), power in zip(layout_list, powers):
        mask = (
            (X >= center_x - w/2) & (X <= center_x + w/2) &
            (Y >= center_y - h/2) & (Y <= center_y + h/2)
        )
        F[mask] = power
        
    plot_source_and_bc(F, [], length, nx)
        
    return F

def plot_source_and_bc(F, bcs, length, nx):
    plt.imshow(F, origin='lower', extent=[0, length, 0, length], aspect='auto')
    for bc in bcs:
        plt.plot(bc[0][0], bc[0][1], 'ro', linewidth=2, markersize=10)
        plt.plot(bc[1][0], bc[1][1], 'ro', linewidth=2, markersize=10)
    plt.colorbar(label='F')
    plt.title('示例热源F')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('debug_sourceF.png')

  
if __name__ == "__main__":
    # Run the test
    test_run_solver_simple_heat_source()
    print("Test passed successfully!")