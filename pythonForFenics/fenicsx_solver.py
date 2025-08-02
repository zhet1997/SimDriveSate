'''
Author: wangqineng zhet3988009@gmail.com
Date: 2025-07-08 15:23:05
LastEditors: wangqineng zhet3988009@gmail.com
LastEditTime: 2025-07-09 20:42:00
FilePath: /EngTestTool/pythonForFenics/fenicsx_solver.py
Description: 

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
    """通过预生成矩阵定义热源 (通过Function插值实现)"""
    def __init__(self, F, length):
        self.F = F
        self.ndim = F.ndim
        self.length = length

    def get_source_value(self, x):
        n = self.F.shape[0]
        if self.ndim == 2:
            xx = int(x[0] / self.length * (n - 1))
            yy = int(x[1] / self.length * (n - 1))
            return self.F[yy, xx]
        xx = int(x[0] / self.length * (n - 1))
        yy = int(x[1] / self.length * (n - 1))
        zz = int(x[2] / self.length * (n - 1))
        return self.F[zz, yy, xx]

class LineBoundary:
    def __init__(self, line):
        self.line = line
    def get_boundary(self):
        (lx, ly), (rx, ry) = self.line
        def boundary(x):
            return np.logical_and(
                np.logical_and(lx - TOL <= x[0], x[0] <= rx + TOL),
                np.logical_and(ly - TOL <= x[1], x[1] <= ry + TOL)
            )
        return boundary


def get_mesh(domain, comm, n):
    return mesh.create_rectangle(comm, domain, n, cell_type=mesh.CellType.triangle) 


def run_solver(
    ndim,
    length,
    length_unit,
    bcs,
    layout_list,
    u0,
    powers,
    nx,
    coordinates=False,
    F=None,
):
    comm = MPI.COMM_WORLD
    domain = ((0.0, 0.0), (length, length))  # 区域边界：从(0,0)到(length, length)的正方形
    n_cells = (nx, nx)  # 网格单元数量：x和y方向各nx个单元
    
    # 创建网格
    mesh_dom = get_mesh(domain, comm, n_cells)
    
    # 处理边界条件
    bc_funcs = []
    if len(bcs) > 0 and bcs[0]:
        # 为每个边界线段生成对应的边界条件函数
        bc_funcs = [LineBoundary(line).get_boundary() for line in bcs]
    else:
        # 默认边界条件：所有边界都生效
        bc_funs = [lambda x, on_boundary: on_boundary]
        
    # 处理热源
    # 初始化热源对象（基于之前生成的F矩阵）
    source = SourceF(F, length)
    # 定义有限元函数空间：Lagrange线性元（最常用的有限元类型）
    V = fem.functionspace(mesh_dom, ("Lagrange", 1))
    # 将热源分布插值到有限元空间上
    f_expr = fem.Function(V)
    f_expr.interpolate(lambda x: [source.get_source_value(xi) for xi in x.T])
    
    # 求解
    u = solver(f_expr, u0, bc_funcs, mesh_dom, V)
    return u, V


def solver(f, u_D, bc_funs, mesh_dom, V):
    # 边界条件处理
    bc_objects = []
    for bc_func in bc_funs:
        facets = mesh.locate_entities_boundary(mesh_dom, mesh_dom.topology.dim - 1, bc_func)
        dofs = fem.locate_dofs_topological(V, mesh_dom.topology.dim - 1, facets)
        
        bc_object = fem.dirichletbc(value=ScalarType(u_D), dofs=dofs, V=V)
        bc_objects.append(bc_object)
        

    
    # 变分问题
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh_dom)  # 显式定义积分测度

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = ufl.inner(f,v) * dx
    
    # 求解
    problem = LinearProblem(a, L, bcs=bc_objects, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u_sol = problem.solve()
    return u_sol
