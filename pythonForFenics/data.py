import sys
import os

module_dir = "/data/zxr/inr/SimDriveSate/pythonForFenics"
sys.path.append(module_dir)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from typing import List, Tuple, Dict, Optional, Any
import random


# 布局生成类 (SeqLS)
class SeqLS:
    def __init__(self,
                 layout_domain: Tuple[float, float],  # 布局域尺寸 (width, height)
                 mesh_size: Tuple[int, int]):  # 网格尺寸 N×M
        self.layout_width, self.layout_height = layout_domain
        self.mesh_N, self.mesh_M = mesh_size
        self.total_nodes = (self.mesh_N + 1, self.mesh_M + 1)  # VEM 矩阵尺寸 (N+1)×(M+1)

        # 网格单元尺寸（物理单位）
        self.grid_width = self.layout_width / self.mesh_M
        self.grid_height = self.layout_height / self.mesh_N

        # 初始化布局容器 VEM（全 0 矩阵）
        self.VEM0 = np.zeros(self.total_nodes, dtype=int)

        # 存储布局历史（用于可视化）
        self.layout_history = []
        # 存储已放置元件的VEM矩阵
        self.placed_vems = []
        self.tolerance = 1e-8

    def _check_component_size_vs_grid(self, components: List[Dict]):
        """校验元件尺寸是否小于网格单元尺寸，若存在则报错"""
        grid_width = self.layout_width / self.mesh_M
        grid_height = self.layout_height / self.mesh_N

        for comp in components:
            comp_id = comp["id"]
            shape = comp["shape"]
            errors = []

            if shape == "rect":
                w = comp["width"]
                h = comp["height"]
                if w < grid_width:
                    errors.append(f"宽度 {w}m 小于网格宽度 {grid_width:.4f}m")
                if h < grid_height:
                    errors.append(f"高度 {h}m 小于网格高度 {grid_height:.4f}m")

            elif shape == "circle":
                diameter = 2 * comp["radius"]
                if diameter < grid_width:
                    errors.append(f"直径 {diameter}m 小于网格宽度 {grid_width:.4f}m")
                if diameter < grid_height:
                    errors.append(f"直径 {diameter}m 小于网格高度 {grid_height:.4f}m")

            elif shape == "capsule":
                width = comp["width"]
                if width < grid_height:
                    errors.append(f"宽度 {width}m 小于网格高度 {grid_height:.4f}m")
                if width < grid_width:
                    errors.append(f"宽度 {width}m 小于网格宽度 {grid_width:.4f}m")

            if errors:
                error_msg = (
                    f"元件 ID {comp_id}（形状：{shape}）存在尺寸问题：\n"
                    f"  - {'; '.join(errors)}\n"
                    "  可能导致离散化误差，建议：\n"
                    "  1. 减小网格尺寸（提高mesh_size）；\n"
                    "  2. 增大元件尺寸；\n"
                    "  3. 调整网格数量使单个网格尺寸小于元件尺寸。"
                )
                raise ValueError(error_msg)

    def _calculate_component_area(self, component: Dict) -> float:
        """计算元件面积（用于排序）"""
        if component["shape"] == "rect":
            return component["width"] * component["height"]
        elif component["shape"] == "circle":
            return np.pi * component["radius"] ** 2
        elif component["shape"] == "capsule":
            return component["length"] * component["width"] + np.pi * (component["width"] / 2) ** 2
        return 0.0

    def _discretize_component(self, component: Dict, center: Tuple[float, float]) -> np.ndarray:
        """离散化元件，生成精确的 VEM 矩阵"""
        vem = np.zeros(self.total_nodes, dtype=int)
        center_x, center_y = center

        if component["shape"] == "rect":
            w, h = component["width"], component["height"]
            half_w, half_h = w / 2, h / 2

            min_x = center_x - half_w + self.tolerance
            max_x = center_x + half_w - self.tolerance
            min_y = center_y - half_h + self.tolerance
            max_y = center_y + half_h - self.tolerance

            start_row = max(0, int(np.floor(min_y / self.grid_height)))
            end_row = min(self.mesh_N, int(np.ceil(max_y / self.grid_height)))
            start_col = max(0, int(np.floor(min_x / self.grid_width)))
            end_col = min(self.mesh_M, int(np.ceil(max_x / self.grid_width)))

            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    vem[row][col] = 1

        elif component["shape"] == "circle":
            r = component["radius"]

            min_x = center_x - r + self.tolerance
            max_x = center_x + r - self.tolerance
            min_y = center_y - r + self.tolerance
            max_y = center_y + r - self.tolerance
            start_row = max(0, int(np.floor(min_y / self.grid_height)))
            end_row = min(self.mesh_N, int(np.ceil(max_y / self.grid_height)))
            start_col = max(0, int(np.floor(min_x / self.grid_width)))
            end_col = min(self.mesh_M, int(np.ceil(max_x / self.grid_width)))

            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    node_corners = [
                        (col * self.grid_width, row * self.grid_height),
                        ((col + 1) * self.grid_width, row * self.grid_height),
                        (col * self.grid_width, (row + 1) * self.grid_height),
                        ((col + 1) * self.grid_width, (row + 1) * self.grid_height)
                    ]

                    corner_in_circle = False
                    for (x, y) in node_corners:
                        corner_dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
                        if corner_dist_sq <= (r ** 2) + self.tolerance:
                            corner_in_circle = True
                            break
                    if corner_in_circle:
                        neighbor_offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
                        for dr, dc in neighbor_offsets:
                            nr = row + dr
                            nc = col + dc
                            if nr < 0:
                                nr = 0
                            elif nr >= self.mesh_N:
                                nr = self.mesh_N
                            if nc < 0:
                                nc = 0
                            elif nc >= self.mesh_M:
                                nc = self.mesh_M
                            if (start_row <= nr <= end_row and
                                    start_col <= nc <= end_col):
                                vem[nr][nc] = 1

        elif component["shape"] == "capsule":
            length, width = component["length"], component["width"]
            rect_length = length - width
            radius = width / 2

            rect_min_x = center_x - rect_length / 2 + self.tolerance
            rect_max_x = center_x + rect_length / 2 - self.tolerance
            rect_min_y = center_y - radius + self.tolerance
            rect_max_y = center_y + radius - self.tolerance

            start_row_rect = max(0, int(np.floor(rect_min_y / self.grid_height)))
            end_row_rect = min(self.mesh_N, int(np.ceil(rect_max_y / self.grid_height)))
            start_col_rect = max(0, int(np.floor(rect_min_x / self.grid_width)))
            end_col_rect = min(self.mesh_M, int(np.ceil(rect_max_x / self.grid_width)))

            for row in range(start_row_rect, end_row_rect + 1):
                for col in range(start_col_rect, end_col_rect + 1):
                    vem[row][col] = 1

            left_circle_center = (rect_min_x, center_y)
            left_min_x = left_circle_center[0] - radius + self.tolerance
            left_max_x = left_circle_center[0] - self.tolerance
            left_min_y = left_circle_center[1] - radius + self.tolerance
            left_max_y = left_circle_center[1] + radius - self.tolerance

            start_row_left = max(0, int(np.floor(left_min_y / self.grid_height)))
            end_row_left = min(self.mesh_N, int(np.ceil(left_max_y / self.grid_height)))
            start_col_left = max(0, int(np.floor(left_min_x / self.grid_width)))
            end_col_left = min(self.mesh_M, int(np.ceil(left_max_x / self.grid_width)))

            neighbor_offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]

            for row in range(start_row_left, end_row_left + 1):
                for col in range(start_col_left, end_col_left + 1):
                    node_corners = [
                        (col * self.grid_width, row * self.grid_height),
                        ((col + 1) * self.grid_width, row * self.grid_height),
                        (col * self.grid_width, (row + 1) * self.grid_height),
                        ((col + 1) * self.grid_width, (row + 1) * self.grid_height)
                    ]

                    has_corner_in_circle = False
                    for (x, y) in node_corners:
                        dist_sq = (x - left_circle_center[0]) ** 2 + (y - left_circle_center[1]) ** 2
                        if dist_sq <= (radius ** 2) + self.tolerance:
                            has_corner_in_circle = True
                            break

                    if has_corner_in_circle:
                        for dr, dc in neighbor_offsets:
                            nr = row + dr
                            nc = col + dc
                            if nr < 0:
                                nr = 0
                            elif nr >= self.mesh_N:
                                nr = self.mesh_N
                            if nc < 0:
                                nc = 0
                            elif nc >= self.mesh_M:
                                nc = self.mesh_M
                            if (start_row_left <= nr <= end_row_left and
                                    start_col_left <= nc <= end_col_left):
                                vem[nr][nc] = 1

            right_circle_center = (rect_max_x, center_y)
            right_min_x = right_circle_center[0] + self.tolerance
            right_max_x = right_circle_center[0] + radius - self.tolerance
            right_min_y = right_circle_center[1] - radius + self.tolerance
            right_max_y = right_circle_center[1] + radius - self.tolerance

            start_row_right = max(0, int(np.floor(right_min_y / self.grid_height)))
            end_row_right = min(self.mesh_N, int(np.ceil(right_max_y / self.grid_height)))
            start_col_right = max(0, int(np.floor(right_min_x / self.grid_width)))
            end_col_right = min(self.mesh_M, int(np.ceil(right_max_x / self.grid_width)))

            for row in range(start_row_right, end_row_right + 1):
                for col in range(start_col_right, end_col_right + 1):
                    node_corners = [
                        (col * self.grid_width, row * self.grid_height),
                        ((col + 1) * self.grid_width, row * self.grid_height),
                        (col * self.grid_width, (row + 1) * self.grid_height),
                        ((col + 1) * self.grid_width, (row + 1) * self.grid_height)
                    ]

                    has_corner_in_circle = False
                    for (x, y) in node_corners:
                        dist_sq = (x - right_circle_center[0]) ** 2 + (y - right_circle_center[1]) ** 2
                        if dist_sq <= (radius ** 2) + 1e-6:
                            has_corner_in_circle = True
                            break

                    if has_corner_in_circle:
                        vem[row][col] = 1
                        for dr, dc in neighbor_offsets:
                            nr = row + dr
                            nc = col + dc
                            if nr < 0:
                                nr = 0
                            elif nr >= self.mesh_N:
                                nr = self.mesh_N - 1
                            if nc < 0:
                                nc = 0
                            elif nc >= self.mesh_M:
                                nc = self.mesh_M - 1
                            if (start_row_right <= nr <= end_row_right and
                                    start_col_right <= nc <= end_col_right):
                                vem[nr][nc] = 1

        return vem

    def _get_vem_for_position(self, component: Dict, node_row: int, node_col: int) -> np.ndarray:
        """获取元件在指定节点位置的VEM矩阵"""
        center = self._node_to_physical(node_row, node_col)
        return self._discretize_component(component, center)

    def _identify_feasible_region(self, component: Dict, placed_vems: List[np.ndarray]) -> np.ndarray:
        """识别元件的可行布局区域（eVEM 集成矩阵）"""
        integrated_evem = np.zeros(self.total_nodes, dtype=int)

        if component["shape"] == "rect":
            w, h = component["width"], component["height"]
            min_valid_x = w / 2
            max_valid_x = self.layout_width - w / 2
            min_valid_y = h / 2
            max_valid_y = self.layout_height - h / 2
        elif component["shape"] == "circle":
            r = component["radius"]
            min_valid_x = r
            max_valid_x = self.layout_width - r
            min_valid_y = r
            max_valid_y = self.layout_height - r
        elif component["shape"] == "capsule":
            length, width = component["length"], component["width"]
            min_valid_x = length / 2
            max_valid_x = self.layout_width - length / 2
            min_valid_y = width / 2
            max_valid_y = self.layout_height - width / 2

        for row in range(self.total_nodes[0]):
            for col in range(self.total_nodes[1]):
                x, y = self._node_to_physical(row, col)
                if x < min_valid_x or x > max_valid_x or y < min_valid_y or y > max_valid_y:
                    integrated_evem[row][col] = 1

        for placed_vem in placed_vems:
            for row in range(self.total_nodes[0]):
                for col in range(self.total_nodes[1]):
                    if placed_vem[row][col] == 1:
                        integrated_evem[row][col] = 1

        return integrated_evem

    def _sample_feasible_position(self, component: Dict, feasible_region: np.ndarray, max_sampling_attempts) -> Tuple[
        int, int]:
        """在可行区域中随机采样一个位置"""
        zero_indices = np.argwhere(feasible_region == 0)
        if not zero_indices.size:
            return None

        for _ in range(max_sampling_attempts):
            sampled_node = tuple(zero_indices[random.randint(0, len(zero_indices) - 1)])
            row, col = sampled_node

            temp_component_vem = self._get_vem_for_position(component, row, col)
            overlap = False
            for placed_vem in self.placed_vems:
                if np.any(np.logical_and(temp_component_vem, placed_vem)):
                    overlap = True
                    break

            if not overlap:
                return sampled_node

        return None

    def _node_to_physical(self, node_row: int, node_col: int) -> Tuple[float, float]:
        """网格节点坐标转物理坐标"""
        return (
            node_col * self.grid_width,
            node_row * self.grid_height
        )

    def layout_sampling(self, components: List[Dict], max_sampling_attempts: int) -> List[Dict]:
        """执行 SeqLS 布局采样"""
        self._check_component_size_vs_grid(components)
        self.VEM0 = np.zeros(self.total_nodes, dtype=int)
        self.layout_history = []
        self.placed_vems = []

        components_sorted = sorted(components, key=self._calculate_component_area, reverse=True)

        placed_components = []
        self.layout_history.append(([], np.copy(self.VEM0)))

        for component in components_sorted:
            feasible_region = self._identify_feasible_region(component, self.placed_vems)
            sampled_node = self._sample_feasible_position(component, feasible_region,
                                                          max_sampling_attempts)

            if sampled_node is None:
                print(f"警告：无法为元件 ID {component['id']} 找到合适位置，布局可能不完整")
                continue

            physical_center = self._node_to_physical(*sampled_node)

            component_with_center = component.copy()
            component_with_center["center"] = physical_center
            component_with_center["node_position"] = sampled_node

            component_vem = self._discretize_component(component, physical_center)
            self.placed_vems.append(component_vem)
            placed_components.append(component_with_center)

            self.VEM0 = np.logical_or(self.VEM0, component_vem).astype(int)
            self.layout_history.append((placed_components.copy(), np.copy(self.VEM0)))

        return placed_components

    def visualize_layout(self, components: List[Dict], show_vem: bool = False, save_path: Optional[str] = None):
        """可视化最终布局结果（无网格线）"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.layout_width)
        ax.set_ylim(0, self.layout_height)
        ax.set_aspect('equal')
        ax.set_title("SeqLS Final Layout")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")

        # 绘制元件
        for i, component in enumerate(components):
            cx, cy = component["center"]
            color = f"C{i}"

            if component["shape"] == "rect":
                w, h = component["width"], component["height"]
                rect = Rectangle(
                    (cx - w / 2, cy - h / 2),
                    w, h,
                    fill=True,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2,
                    alpha=0.7
                )
                ax.add_patch(rect)

            elif component["shape"] == "circle":
                r = component["radius"]
                circle = Circle(
                    (cx, cy),
                    r,
                    fill=True,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2,
                    alpha=0.7
                )
                ax.add_patch(circle)

            elif component["shape"] == "capsule":
                length, width = component["length"], component["width"]
                rect_len = length - width
                radius = width / 2

                rect = Rectangle(
                    (cx - rect_len / 2, cy - radius),
                    rect_len, width,
                    fill=True,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2,
                    alpha=0.7
                )
                ax.add_patch(rect)

                left_circle = Circle(
                    (cx - rect_len / 2, cy),
                    radius,
                    fill=True,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2,
                    alpha=0.7
                )
                ax.add_patch(left_circle)

                right_circle = Circle(
                    (cx + rect_len / 2, cy),
                    radius,
                    fill=True,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2,
                    alpha=0.7
                )
                ax.add_patch(right_circle)

            ax.text(
                cx, cy,
                f"ID: {component['id']}\nPower: {component['power']}W",
                ha='center', va='center', fontsize=7, color='white', fontweight='bold'
            )

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight'
            )
            print(f"布局图像已保存至：{save_path}")
        plt.close()


# 热立场求解相关函数
def generate_source_F(layout_list, powers, length, nx):
    """生成热源分布矩阵F"""
    F = np.zeros((nx + 1, nx + 1))

    x = np.linspace(0, length, nx + 1)
    y = np.linspace(0, length, nx + 1)
    X, Y = np.meshgrid(x, y, indexing='xy')  # 已修正为xy索引，确保与布局对齐

    for layout, power in zip(layout_list, powers):
        shape_type = layout[0]

        if shape_type == 'rect':
            _, center_x, center_y, w, h = layout
            mask = (
                    (X >= center_x - w / 2) & (X <= center_x + w / 2) &
                    (Y >= center_y - h / 2) & (Y <= center_y + h / 2)
            )

        elif shape_type == 'circle':
            _, center_x, center_y, radius = layout
            mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2

        elif shape_type == 'capsule':
            _, center_x, center_y, cap_length, cap_width = layout
            radius = cap_width / 2
            rect_length = cap_length - 2 * radius

            rect_mask = (
                    (X >= center_x - rect_length / 2) & (X <= center_x + rect_length / 2) &
                    (Y >= center_y - radius) & (Y <= center_y + radius)
            )

            left_circle_mask = (
                    (X <= center_x - rect_length / 2) &
                    ((X - (center_x - rect_length / 2)) ** 2 + (Y - center_y) ** 2 <= radius ** 2)
            )

            right_circle_mask = (
                    (X >= center_x + rect_length / 2) &
                    ((X - (center_x + rect_length / 2)) ** 2 + (Y - center_y) ** 2 <= radius ** 2)
            )

            mask = rect_mask | left_circle_mask | right_circle_mask
        else:
            raise ValueError(f"不支持的形状类型：{shape_type}")

        F[mask] = power

    return F


def plot_source_and_bc(F, bcs, length, nx, save_path="heat_source.png"):
    """绘制热源分布图"""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        F,
        origin='lower',
        extent=[0, length, 0, length],
        aspect='equal',
        cmap='viridis',
        vmin=0,
    )

    for bc in bcs:
        (x1, y1), (x2, y2) = bc
        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
        plt.plot([x1, x2], [y1, y2], 'ro', markersize=6)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(im, label='Heat Source Intensity (Power)')
    plt.title('Heat Source Distribution')
    plt.xlabel('x Coordinate (m)')
    plt.ylabel('y Coordinate (m)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"热源图像已保存至：{save_path}")
    plt.close()


def plot_u(uh, V, save_path="temperature_field.png"):
    """绘制温度场分布（修正行列反向问题）"""
    import numpy as np
    from scipy.interpolate import griddata

    # 1. 提取自由度坐标
    dof_coords = V.tabulate_dof_coordinates()
    x_coords = dof_coords[:, 0]  # X 坐标
    y_coords = dof_coords[:, 1]  # Y 坐标
    solution_values = uh.x.array

    # 2. 生成插值网格（明确使用 xy 索引）
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # 用 np.meshgrid 生成网格，indexing='xy' 确保 X→列，Y→行
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid, indexing='xy')

    # 3. 插值温度场
    U = griddata(
        (x_coords, y_coords),
        solution_values,
        (grid_x, grid_y),
        method='cubic'
    )

    # 4. 绘制图像（统一坐标系统）
    plt.figure(figsize=(8, 6))
    if U.ndim == 2:
        im = plt.imshow(
            U,
            cmap='inferno',
            aspect='equal',  # 保持横纵比一致
            extent=[x_min, x_max, y_min, y_max],  # 明确坐标范围
            origin='lower'  # 底部对齐（与布局/热源图一致）
        )
        cbar = plt.colorbar(im)
        cbar.set_label('Temperature (K)')
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title('Temperature Field Distribution')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"温度场图像已保存至：{save_path}")
    plt.close()


# 整合函数 - 将布局生成与热立场求解连接起来
def run_heat_simulation(
        components: List[Dict],
        layout_domain: Tuple[float, float],
        mesh_size: Tuple[int, int],
        max_sampling_attempts: int = 100,
        u0: float = 298.0,
        bcs: List[List[Tuple[float, float]]] = None,
        layout_save_path: str = "layout.png",
        source_save_path: str = "heat_source.png",
        temp_save_path: str = "temperature_field.png"
):
    """
    整合布局生成与热立场求解的主函数（仅生成单张图）

    参数:
        components: 元件列表，每个元件包含id、shape、尺寸参数和power
        layout_domain: 布局区域尺寸 (width, height)
        mesh_size: 网格大小 (N, M)
        max_sampling_attempts: 最大采样尝试次数
        u0: 初始温度 (K)
        bcs: 边界条件列表
        layout_save_path: 布局图像保存路径
        source_save_path: 热源图像保存路径
        temp_save_path: 温度场图像保存路径

    返回:
        布局结果、热源矩阵和温度场结果
    """
    # 设置默认边界条件（如果未提供）
    if bcs is None:
        bcs = [([0.0, 0.0], [layout_domain[0], 0.0])]  # 默认底部边界

    # 1. 生成布局
    seq_ls = SeqLS(layout_domain, mesh_size)
    placed_components = seq_ls.layout_sampling(components, max_sampling_attempts)

    if not placed_components:
        print("布局生成失败，无法继续计算热立场")
        return None, None, None

    # 可视化并保存布局
    seq_ls.visualize_layout(placed_components, show_vem=False, save_path=layout_save_path)

    # 2. 从布局结果提取热源信息，转换为热求解器需要的格式
    layout_list = []
    powers = []

    for comp in placed_components:
        cx, cy = comp["center"]
        shape = comp["shape"]

        if shape == "rect":
            layout_list.append(('rect', cx, cy, comp["width"], comp["height"]))
        elif shape == "circle":
            layout_list.append(('circle', cx, cy, comp["radius"]))
        elif shape == "capsule":
            layout_list.append(('capsule', cx, cy, comp["length"], comp["width"]))

        powers.append(comp["power"])

    # 3. 生成热源矩阵F并可视化
    length = layout_domain[0]  # 假设是正方形区域
    nx = mesh_size[0]  # 网格划分数量
    F = generate_source_F(layout_list, powers, length, nx)
    plot_source_and_bc(F, bcs, length, nx, save_path=source_save_path)

    # 4. 运行热传导求解器
    from fenicsx_solver import run_solver  # 导入外部求解器

    length_unit = length / nx
    U, V = run_solver(
        ndim=2,
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

    # 5. 可视化温度场
    plot_u(U, V, save_path=temp_save_path)

    return placed_components, F, (U, V)


# 示例用法
if __name__ == "__main__":
    # 设置随机种子，保证结果可复现
    random.seed(42)

    # 1. 用户输入：元件信息
    components = [
        {"id": 0, "shape": "capsule", "length": 0.06, "width": 0.02, "power": 5000},
        {"id": 1, "shape": "rect", "width": 0.03, "height": 0.01, "power": 3000},
        {"id": 2, "shape": "circle", "radius": 0.01, "power": 2000},
        {"id": 3, "shape": "rect", "width": 0.04, "height": 0.02, "power": 4000},
        {"id": 4, "shape": "circle", "radius": 0.015, "power": 2500},
    ]

    # 2. 用户输入：设计区域和网格大小
    layout_domain = (0.1, 0.1)  # 0.1m × 0.1m 的正方形区域
    mesh_size = (256, 256)  # 256×256 的网格

    # 3. 用户输入：边界条件（可选）
    boundary_conditions = [
        ([0.03, 0], [0.07, 0]),  # 底部边界的一部分
        ([0, 0.04], [0, 0.06])  # 左侧边界的一部分
    ]

    # 4. 运行整合模拟
    placed_components, heat_source, temperature_field = run_heat_simulation(
        components=components,
        layout_domain=layout_domain,
        mesh_size=mesh_size,
        max_sampling_attempts=100,
        u0=298.0,  # 初始温度 298K (室温)
        bcs=boundary_conditions,
        layout_save_path="my_layout.png",
        source_save_path="my_heat_source.png",
        temp_save_path="my_temperature_field.png"
    )

    if placed_components:
        print("模拟完成！已生成以下结果文件：")
        print("- 布局图: my_layout.png")
        print("- 热源分布图: my_heat_source.png")
        print("- 温度场图: my_temperature_field.png")
