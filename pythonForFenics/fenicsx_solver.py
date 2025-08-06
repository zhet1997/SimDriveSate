'''
Author: wangqineng zhet3988009@gmail.com
Date: 2025-07-08 15:23:05
LastEditors: wangqineng zhet3988009@gmail.com
LastEditTime: 2025-08-02 16:45:00
FilePath: /EngTestTool/pythonForFenics/fenicsx_solver.py
Description: 修复索引越界错误，支持x和y方向不同网格尺寸

Copyright (c) 2025 by wangqineng, All Rights Reserved.
'''

from dolfinx import mesh, fem, plot, io
from petsc4py.PETSc import ScalarType
import ufl
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
import logging
logging.getLogger().setLevel(logging.ERROR)

TOL = 1e-14

class SourceF:
    """通过预生成矩阵定义热源 (修复索引计算错误)"""
    def __init__(self, F, length_x, length_y):
        self.F = F
        self.ndim = F.ndim
        self.length_x = length_x  # x方向长度
        self.length_y = length_y  # y方向长度

        # 关键修复：正确获取x和y方向的网格点数
        # F矩阵的shape应为(ny+1, nx+1)，其中ny是y方向网格数，nx是x方向网格数
        self.ny_plus_1, self.nx_plus_1 = F.shape  # 网格点数量（包含边界）
        self.nx = self.nx_plus_1 - 1  # x方向网格数
        self.ny = self.ny_plus_1 - 1  # y方向网格数

        # 调试信息：打印矩阵维度和网格数量
        # print(f"热源矩阵维度: {F.shape}, x方向网格数: {self.nx}, y方向网格数: {self.ny}")

    def get_source_value(self, x):
        """根据坐标(x,y)获取热源值，修复索引越界问题"""
        if self.ndim == 2:
            # 计算x方向坐标在网格中的比例位置
            x_ratio = x[0] / self.length_x if self.length_x > 0 else 0.0
            # 计算对应的网格点索引（使用网格点数量而非网格数）
            xx = int(np.round(x_ratio * (self.nx_plus_1 - 1)))
            # 确保索引在有效范围内
            xx = np.clip(xx, 0, self.nx_plus_1 - 1)

            # 计算y方向坐标在网格中的比例位置
            y_ratio = x[1] / self.length_y if self.length_y > 0 else 0.0
            # 计算对应的网格点索引
            yy = int(np.round(y_ratio * (self.ny_plus_1 - 1)))
            # 确保索引在有效范围内
            yy = np.clip(yy, 0, self.ny_plus_1 - 1)

            # 调试信息：检查索引是否在有效范围内
            if xx >= self.nx_plus_1 or yy >= self.ny_plus_1:
                print(f"警告: 索引越界 - xx={xx}, yy={yy}, 矩阵维度={self.F.shape}")

            return self.F[yy, xx]

        # 3D情况处理
        zz = int(np.clip(x[2] / self.length_x * (self.F.shape[0]-1), 0, self.F.shape[0]-1))
        yy = int(np.clip(x[1] / self.length_y * (self.F.shape[1]-1), 0, self.F.shape[1]-1))
        xx = int(np.clip(x[0] / self.length_x * (self.F.shape[2]-1), 0, self.F.shape[2]-1))
        return self.F[zz, yy, xx]


class LineBoundary:
    """线段边界条件处理"""
    def __init__(self, line):
        self.line = line
        (self.lx, self.ly), (self.rx, self.ry) = line
        self.dx = self.rx - self.lx
        self.dy = self.ry - self.ly
        self.len_sq = self.dx**2 + self.dy**2 if (self.dx != 0 or self.dy != 0) else 1e-8

    def get_boundary(self):
        def boundary(x):
            if self.len_sq < 1e-16:  # 点边界
                return np.logical_and(
                    np.isclose(x[0], self.lx, atol=TOL),
                    np.isclose(x[1], self.ly, atol=TOL)
                )

            # 计算点到线段的距离
            t = np.clip(
                ((x[0] - self.lx) * self.dx + (x[1] - self.ly) * self.dy) / self.len_sq,
                0.0, 1.0
            )

            proj_x = self.lx + t * self.dx
            proj_y = self.ly + t * self.dy
            dist_sq = (x[0] - proj_x)**2 + (x[1] - proj_y)** 2

            return dist_sq < TOL**2

        return boundary


def get_mesh(domain, comm, n_cells):
    """创建矩形网格"""
    return mesh.create_rectangle(
        comm,
        domain,
        n_cells,
        cell_type=mesh.CellType.triangle
    )


def run_solver(
        ndim,
        length_x,
        length_y,
        bcs,
        u0,
        nx,
        ny,
        F=None,
        coordinates=False,
        layout_list=None,
        powers=None
):
    """主求解函数"""
    comm = MPI.COMM_WORLD

    # 定义区域和网格
    domain = ((0.0, 0.0), (length_x, length_y))
    n_cells = (nx, ny)
    mesh_dom = get_mesh(domain, comm, n_cells)

    # 处理边界条件
    bc_funcs = []
    if len(bcs) > 0 and bcs[0] is not None:
        bc_funcs = [LineBoundary(line).get_boundary() for line in bcs]
    else:
        bc_funcs = [lambda x, on_boundary: on_boundary]

    # 处理热源 - 增加维度检查
    if F is None:
        raise ValueError("热源矩阵F不能为空")

    # 关键修复：确保F矩阵维度与网格数量匹配
    expected_shape = (ny + 1, nx + 1)
    if F.shape != expected_shape:
        raise ValueError(f"热源矩阵维度不匹配: 预期 {expected_shape}, 实际 {F.shape}")

    source = SourceF(F, length_x, length_y)

    # 定义函数空间
    V = fem.functionspace(mesh_dom, ("Lagrange", 1))

    # 关键修复：修改插值方式以符合dolfinx要求
    f_expr = fem.Function(V)

    # 使用正确的插值方法：返回numpy数组而非列表
    def interpolate_source(x):
        values = np.array([source.get_source_value(xi) for xi in x.T], dtype=np.float64)
        return values

    try:
        f_expr.interpolate(interpolate_source)
    except Exception as e:
        print(f"热源插值失败: {e}")
        raise

    # 求解
    u = solver(f_expr, u0, bc_funcs, mesh_dom, V)
    return u, V


def solver(f, u_D, bc_funs, mesh_dom, V):
    """求解器核心"""
    bc_objects = []
    for bc_func in bc_funs:
        facets = mesh.locate_entities_boundary(
            mesh_dom,
            mesh_dom.topology.dim - 1,
            bc_func
        )
        if len(facets) == 0:
            logging.warning("未找到符合条件的边界单元")
            continue

        dofs = fem.locate_dofs_topological(V, mesh_dom.topology.dim - 1, facets)
        bc_object = fem.dirichletbc(ScalarType(u_D), dofs, V)
        bc_objects.append(bc_object)

    if not bc_objects:
        raise RuntimeError("未成功创建任何边界条件")

    # 变分问题
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh_dom)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = ufl.inner(f, v) * dx

    # 求解器配置
    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "ilu",
        "ksp_rtol": 1e-8,
        "ksp_atol": 1e-12
    }

    try:
        problem = LinearProblem(a, L, bcs=bc_objects, petsc_options=petsc_options)
        u_sol = problem.solve()
        return u_sol
    except Exception as e:
        print(f"求解过程失败: {e}")
        raise
