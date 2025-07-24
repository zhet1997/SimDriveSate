# demo_heat2d_corrected.py

import dolfinx
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
from mpi4py import MPI

# 1. 创建网格 (Mesh) 和函数空间 (Function Space)
# ------------------------------------------------
# 使用 MPI 并创建一个 32x32 的单位正方形网格
comm = MPI.COMM_WORLD
domain = mesh.create_unit_square(comm, 32, 32, mesh.CellType.triangle)

# 定义一个一阶拉格朗日单元的函数空间
V = fem.functionspace(domain, ("Lagrange", 1))

# 2. 定义问题参数 (源项和边界条件)
# ------------------------------------------------
# 定义精确解 T_exact = 1 + 2x^2 + 3y^2
def exact_solution_func(x):
    return 1 + 2 * x[0]**2 + 3 * x[1]**2

# 定义源项 f = -10
f = fem.Constant(domain, -10.0)

# 3. 定义狄利克雷边界条件 (Dirichlet Boundary Conditions)
# ------------------------------------------------
#
# *** FIX IS HERE ***
# 在调用 exterior_facet_indices 之前，必须先计算小平面到单元的连接性
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
#
# *****************

# 找到域边界上的所有小平面 (facets)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
# 基于这些小平面找到对应的自由度
boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)

# 将精确解插值到一个函数中，用于施加边界条件
u_D = fem.Function(V)
u_D.interpolate(exact_solution_func)

# 创建 DirichletBC 对象
bc = fem.dirichletbc(u_D, boundary_dofs,V)

# 4. 定义变分形式 (Variational Formulation)
# ------------------------------------------------
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# 5. 求解线性系统
# ------------------------------------------------
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
Th = problem.solve()
Th.name = "Temperature"

# 6. 后处理: 计算误差并保存结果
# ------------------------------------------------
T_exact = fem.Function(V)
T_exact.interpolate(exact_solution_func)

error_form = fem.form(ufl.inner(Th - T_exact, Th - T_exact) * ufl.dx)
error_local = fem.assemble_scalar(error_form)
error_L2 = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))

if comm.rank == 0:
    print(f"L2 norm of the error is: {error_L2:.4e}")

try:
    from dolfinx.io import VTXWriter
    with VTXWriter(comm, "output_temperature2d.bp", [Th], engine="BP4") as vtx:
        vtx.write(0.0)
    if comm.rank == 0:
        print("Solution saved to output_temperature2d.bp")
except ImportError:
    if comm.rank == 0:
        print("VTXWriter is not available, cannot save the solution.")