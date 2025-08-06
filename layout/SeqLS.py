import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from typing import List, Tuple, Dict, Optional

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
        """
        校验元件尺寸是否小于网格单元尺寸，若存在则报错
        """
        # 1. 计算网格单元的实际尺寸（物理单位）
        grid_width = self.layout_width / self.mesh_M
        grid_height = self.layout_height / self.mesh_N

        # 2. 遍历所有元件，检查尺寸
        for comp in components:
            comp_id = comp["id"]
            shape = comp["shape"]
            errors = []  # 记录当前元件的尺寸问题

            if shape == "rect":
                w = comp["width"]
                h = comp["height"]
                # 矩形需同时检查宽度和高度
                if w < grid_width:
                    errors.append(f"宽度 {w}m 小于网格宽度 {grid_width:.4f}m")
                if h < grid_height:
                    errors.append(f"高度 {h}m 小于网格高度 {grid_height:.4f}m")

            elif shape == "circle":
                diameter = 2 * comp["radius"]  # 圆形的关键尺寸是直径
                # 圆形对称，需同时检查与网格宽/高的关系
                if diameter < grid_width:
                    errors.append(f"直径 {diameter}m 小于网格宽度 {grid_width:.4f}m")
                if diameter < grid_height:
                    errors.append(f"直径 {diameter}m 小于网格高度 {grid_height:.4f}m")

            elif shape == "capsule":
                width = comp["width"]  # 胶囊的垂直方向最大尺寸是宽度
                # 胶囊宽度需覆盖网格高度，长度方向通常较大（可根据需求添加长度校验）
                if width < grid_height:
                    errors.append(f"宽度 {width}m 小于网格高度 {grid_height:.4f}m")
                if width < grid_width:
                    errors.append(f"宽度 {width}m 小于网格宽度 {grid_width:.4f}m")

            # 3. 若存在尺寸问题，抛出详细错误
            if errors:
                error_msg = (
                    f"元件 ID {comp_id}（形状：{shape}）存在尺寸问题：\n"
                    f"  - {'; '.join(errors)}\n"
                    "  可能导致离散化误差（元件被网格放大），建议：\n"
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

            # min_x = center_x - half_w
            # max_x = center_x + half_w
            # min_y = center_y - half_h
            # max_y = center_y + half_h
            # 修正浮点数误差：边界向理论值收缩
            min_x = center_x - half_w + self.tolerance  # 左边界 +容差，避免因负误差变小
            max_x = center_x + half_w - self.tolerance  # 右边界 -容差，避免因正误差变大
            min_y = center_y - half_h + self.tolerance  # 下边界 +容差
            max_y = center_y + half_h - self.tolerance  # 上边界 -容差

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
                    # 判断：只要有一个角落在圆内（带浮点容差），就标记该网格
                    corner_in_circle = False
                    for (x, y) in node_corners:
                        corner_dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
                        if corner_dist_sq <= (r ** 2) + self.tolerance:  # 处理浮点误差
                            corner_in_circle = True
                            break  # 找到一个即可，无需检查其他角落
                    if corner_in_circle:
                        # 标记当前网格及周围三个相邻网格（共4个）
                        # 定义周围网格的相对坐标（可根据需求调整方向）
                        neighbor_offsets = [
                            (0, 0),  # 当前网格
                            (0, 1),  # 右侧相邻网格
                            (1, 0),  # 下方相邻网格
                            (1, 1)  # 右下相邻网格
                        ]
                        for dr, dc in neighbor_offsets:
                            nr = row + dr
                            nc = col + dc
                            # 处理行索引超出边界的情况
                            if nr < 0:
                                nr = 0
                            elif nr >= self.mesh_N:
                                nr = self.mesh_N

                            # 处理列索引超出边界的情况
                            if nc < 0:
                                nc = 0
                            elif nc >= self.mesh_M:
                                nc = self.mesh_M
                            if (start_row <= nr <= end_row
                                    and start_col <= nc <= end_col):
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

            neighbor_offsets = [
                (0, 0),  # 当前网格
                (0, 1),  # 右侧相邻网格
                (1, 0),  # 下方相邻网格
                (1, 1)  # 右下相邻网格
            ]

            for row in range(start_row_left, end_row_left + 1):
                for col in range(start_col_left, end_col_left + 1):
                    # 计算网格四个角落坐标
                    node_corners = [
                        (col * self.grid_width, row * self.grid_height),
                        ((col + 1) * self.grid_width, row * self.grid_height),
                        (col * self.grid_width, (row + 1) * self.grid_height),
                        ((col + 1) * self.grid_width, (row + 1) * self.grid_height)
                    ]

                    # 检查：只要有一个角落在圆内或圆上（不区分边界）
                    has_corner_in_circle = False
                    for (x, y) in node_corners:
                        dist_sq = (x - left_circle_center[0]) ** 2 + (y - left_circle_center[1]) ** 2
                        # 距离≤半径（包含圆内和圆上，带容差）
                        if dist_sq <= (radius ** 2) + self.tolerance:
                            has_corner_in_circle = True
                            break

                    if has_corner_in_circle:
                        # 标记周围四个相邻网格
                        for dr, dc in neighbor_offsets:
                            nr = row + dr
                            nc = col + dc

                            # 处理边界：超出时取边界值
                            if nr < 0:
                                nr = 0
                            elif nr >= self.mesh_N:
                                nr = self.mesh_N
                            if nc < 0:
                                nc = 0
                            elif nc >= self.mesh_M:
                                nc = self.mesh_M
                            if (start_row_left <= nr <= end_row_left
                                    and start_col_left <= nc <= end_col_left):
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
                    # 计算网格四个角落坐标
                    node_corners = [
                        (col * self.grid_width, row * self.grid_height),
                        ((col + 1) * self.grid_width, row * self.grid_height),
                        (col * self.grid_width, (row + 1) * self.grid_height),
                        ((col + 1) * self.grid_width, (row + 1) * self.grid_height)
                    ]

                    # 检查：只要有一个角落在圆内或圆上（不区分边界）
                    has_corner_in_circle = False
                    for (x, y) in node_corners:
                        dist_sq = (x - right_circle_center[0]) ** 2 + (y - right_circle_center[1]) ** 2
                        # 距离≤半径（包含圆内和圆上，带容差）
                        if dist_sq <= (radius ** 2) + 1e-6:
                            has_corner_in_circle = True
                            break

                    if has_corner_in_circle:
                        # 标记当前网格
                        vem[row][col] = 1

                        # 标记周围三个相邻网格
                        for dr, dc in neighbor_offsets:
                            nr = row + dr
                            nc = col + dc

                            # 处理边界：超出时取边界值
                            if nr < 0:
                                nr = 0
                            elif nr >= self.mesh_N:
                                nr = self.mesh_N - 1
                            if nc < 0:
                                nc = 0
                            elif nc >= self.mesh_M:
                                nc = self.mesh_M - 1
                            if (start_row_right <= nr <= end_row_right
                                    and start_col_right <= nc <= end_col_right):
                                vem[nr][nc] = 1

        return vem

    def _get_vem_for_position(self, component: Dict, node_row: int, node_col: int) -> np.ndarray:
        """获取元件在指定节点位置的VEM矩阵"""
        center = self._node_to_physical(node_row, node_col)
        return self._discretize_component(component, center)

    def _identify_feasible_region(self, component: Dict, placed_vems: List[np.ndarray]) -> np.ndarray:
        """识别元件的可行布局区域（eVEM 集成矩阵）-粗略"""
        integrated_evem = np.zeros(self.total_nodes, dtype=int)

        # 1. 检查与布局容器的约束
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

        # 标记超出边界的位置为不可行
        for row in range(self.total_nodes[0]):
            for col in range(self.total_nodes[1]):
                x, y = self._node_to_physical(row, col)
                if x < min_valid_x or x > max_valid_x or y < min_valid_y or y > max_valid_y:
                    integrated_evem[row][col] = 1

        # 2. 检查与已放置元件的约束
        for placed_vem in placed_vems:
            for row in range(self.total_nodes[0]):
                for col in range(self.total_nodes[1]):
                    if placed_vem[row][col] == 1:
                        integrated_evem[row][col] = 1

        return integrated_evem

    def _sample_feasible_position(self, component: Dict, feasible_region: np.ndarray, max_sampling_attempts) -> Tuple[
        int, int]:
        """
        在可行区域中随机采样一个位置（使用循环而非递归，避免递归深度超限）
        """
        zero_indices = np.argwhere(feasible_region == 0)
        if not zero_indices.size:
            return None  # 无可行位置

        # 尝试有限次数的采样
        for _ in range(max_sampling_attempts):
            # 随机选择一个可行位置
            sampled_node = tuple(zero_indices[random.randint(0, len(zero_indices) - 1)])
            row, col = sampled_node

            # 校验是否与已放置元件重叠
            temp_component_vem = self._get_vem_for_position(component, row, col)  # 反向检验
            overlap = False
            for placed_vem in self.placed_vems:
                if np.any(np.logical_and(temp_component_vem, placed_vem)):
                    overlap = True
                    break

            if not overlap:
                return sampled_node

        # 如果达到最大尝试次数仍未找到合适位置，返回None
        return None

    def _node_to_physical(self, node_row: int, node_col: int) -> Tuple[float, float]:
        """网格节点坐标转物理坐标（元件中心严格对齐网格节点）"""
        return (
            node_col * self.grid_width,
            node_row * self.grid_height
        )

    def layout_sampling(self, components: List[Dict], max_sampling_attempts: int) -> List[Dict]:
        """执行 SeqLS 布局采样"""
        # 布局前先校验元件尺寸与网格的匹配性
        self._check_component_size_vs_grid(components)
        # 重置状态
        self.VEM0 = np.zeros(self.total_nodes, dtype=int)
        self.layout_history = []
        self.placed_vems = []

        # 按元件面积降序排序
        components_sorted = sorted(components, key=self._calculate_component_area, reverse=True)

        placed_components = []
        self.layout_history.append(([], np.copy(self.VEM0)))

        for component in components_sorted:
            feasible_region = self._identify_feasible_region(component, self.placed_vems)  # 计算可行布局区域
            sampled_node = self._sample_feasible_position(component, feasible_region,
                                                          max_sampling_attempts)  # 随机采样一个可行位置

            if sampled_node is None:
                print(f"警告：无法为元件 ID {component['id']} 找到合适位置，布局可能不完整")
                continue  # 若要尝试放置剩余元件，可使用continue

            physical_center = self._node_to_physical(*sampled_node)

            component_with_center = component.copy()
            component_with_center["center"] = physical_center
            component_with_center["node_position"] = sampled_node

            component_vem = self._discretize_component(component, physical_center)
            self.placed_vems.append(component_vem)
            placed_components.append(component_with_center)

            self.VEM0 = np.logical_or(self.VEM0, component_vem).astype(int)  # 更新 VEM
            self.layout_history.append((placed_components.copy(), np.copy(self.VEM0)))

        return placed_components

    def visualize_layout(self, components: List[Dict], show_vem: bool = False, save_path: Optional[str] = None):
        """可视化最终布局结果"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.layout_width)
        ax.set_ylim(0, self.layout_height)
        ax.set_aspect('equal')
        ax.set_title("SeqLS Final Layout")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")

        # 绘制网格线
        for x in np.arange(0, self.layout_width, self.grid_width):
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
        for y in np.arange(0, self.layout_height, self.grid_height):
            ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

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

            # ax.text(cx, cy, f"ID: {component['id']}",
            #         ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            ax.text(
                cx, cy,
                f"ID: {component['id']}\nPower: {component['power']}W",  # 新增功率信息
                ha='center', va='center', fontsize=7, color='white', fontweight='bold'
            )
        if show_vem:
            covered_nodes = np.argwhere(self.VEM0 == 1)
            for (row, col) in covered_nodes:
                x, y = self._node_to_physical(row, col)
                ax.scatter(x, y, color='red', s=8, alpha=0.6)

        plt.tight_layout()
        # plt.show()
        if save_path is not None:
            plt.savefig(
                save_path,
                dpi=300,  # 高分辨率（默认100，300更清晰）
                bbox_inches='tight'  # 裁剪空白边缘，确保内容完整
            )
            print(f"图像已保存至：{save_path}")

    def visualize_layout_process(self):
        """可视化布局过程"""
        num_steps = len(self.layout_history)
        fig, axes = plt.subplots(1, num_steps, figsize=(5 * num_steps, 5))

        if num_steps == 1:
            axes = [axes]

        for i, (components, vem) in enumerate(self.layout_history):
            ax = axes[i]
            ax.set_xlim(0, self.layout_width)
            ax.set_ylim(0, self.layout_height)
            ax.set_aspect('equal')
            ax.set_title(f"Step {i}: {len(components)} components")

            for x in np.arange(0, self.layout_width, self.grid_width):
                ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
            for y in np.arange(0, self.layout_height, self.grid_height):
                ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

            for j, component in enumerate(components):
                cx, cy = component["center"]
                color = f"C{j}"

                if component["shape"] == "rect":
                    w, h = component["width"], component["height"]
                    rect = Rectangle(
                        (cx - w / 2, cy - h / 2), w, h,
                        fill=True, facecolor=color, edgecolor='black',
                        linewidth=2, alpha=0.7
                    )
                    ax.add_patch(rect)

                elif component["shape"] == "circle":
                    r = component["radius"]
                    circle = Circle(
                        (cx, cy), r,
                        fill=True, facecolor=color, edgecolor='black',
                        linewidth=2, alpha=0.7
                    )
                    ax.add_patch(circle)

                elif component["shape"] == "capsule":
                    length, width = component["length"], component["width"]
                    rect_len = length - width
                    radius = width / 2

                    rect = Rectangle(
                        (cx - rect_len / 2, cy - radius),
                        rect_len, width,
                        fill=True, facecolor=color, edgecolor='black',
                        linewidth=2, alpha=0.7
                    )
                    ax.add_patch(rect)

                    ax.add_patch(Circle(
                        (cx - rect_len / 2, cy), radius,
                        fill=True, facecolor=color, edgecolor='black',
                        linewidth=2, alpha=0.7
                    ))
                    ax.add_patch(Circle(
                        (cx + rect_len / 2, cy), radius,
                        fill=True, facecolor=color, edgecolor='black',
                        linewidth=2, alpha=0.7
                    ))

                # ax.text(cx, cy, f"ID: {component['id']}",
                #         ha='center', va='center', fontsize=8, color='white', fontweight='bold')
                ax.text(
                    cx, cy,
                    f"ID: {component['id']}\nPower: {component['power']}W",
                    ha='center', va='center', fontsize=7, color='white', fontweight='bold'
                )
            covered_nodes = np.argwhere(vem == 1)
            for (row, col) in covered_nodes:
                x, y = self._node_to_physical(row, col)
                ax.scatter(x, y, color='red', s=5, alpha=0.5)

        plt.tight_layout()
        plt.show()


# 示例调用
if __name__ == "__main__":
    random.seed(42)
    # 初始化 SeqLS
    layout_domain = (1.0, 1.0)  # 布局域尺寸 (1m × 1m)
    mesh_size = (256, 256)  # 20×20 网格
    seq_ls = SeqLS(layout_domain, mesh_size)

    # 定义元件
    components = [
        {"id": 0, "shape": "rect", "width": 0.31, "height": 0.2, "power": 5.5},
        {"id": 1, "shape": "circle", "radius": 0.11, "power": 3.2},
        {"id": 2, "shape": "capsule", "length": 0.31, "width": 0.15, "power": 8.0},
        {"id": 3, "shape": "rect", "width": 0.2, "height": 0.25, "power": 4.7},
        {"id": 4, "shape": "circle", "radius": 0.1, "power": 2.1},
        {"id": 5, "shape": "capsule", "length": 0.4, "width": 0.15, "power": 7.3},
        {"id": 6, "shape": "rect", "width": 0.2, "height": 0.25, "power": 5.0},
        {"id": 7, "shape": "circle", "radius": 0.1, "power": 3.5},
    ]
    # 测试：添加一个过小的元件（触发报错）
    # components.append({
    #     "id": 8, "shape": "circle", "radius": 0.02, "power": 1.0  # 直径0.04m < 网格0.05m
    # })
    max_attempts = 100
    # 执行布局采样
    result = seq_ls.layout_sampling(components, max_attempts)
    if result:
        print("布局成功！元件信息：")
        for comp in result:
            cx, cy = comp['center']
            print(f"元件 ID: {comp['id']}, 形状: {comp['shape']}, 中心坐标: ({cx:.3f}, {cy:.3f})")

        # # 可视化最终布局
        # seq_ls.visualize_layout(result, show_vem=True, save_path="layout_result.png")
        seq_ls.visualize_layout(result, show_vem=False, save_path="layout_result_256.png")
        # # 可视化布局过程
        # seq_ls.visualize_layout_process()
    else:
        print("布局失败，无可行位置")
