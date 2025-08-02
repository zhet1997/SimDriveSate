import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from typing import List, Tuple, Dict, Optional
import numpy as np
from SeqLS import SeqLS  # 假设SeqLS类已正确实现


class GibLS_Sampler:
    def __init__(self,
                 design_area: Tuple[float, float],
                 grid_size: Tuple[int, int]):
        self.design_area = design_area  # (width, height)
        self.grid_size = grid_size  # (rows, cols)
        self.cell_width = design_area[0] / grid_size[1]
        self.cell_height = design_area[1] / grid_size[0]
        self.grid_matrix = [[0 for _ in range(grid_size[1])] for _ in range(grid_size[0])]
        self.components = []
        self.seqls_sampler = SeqLS(design_area, grid_size)

    # 以下是之前实现的辅助方法（保持不变）
    def _update_grid_matrix(self, components: List[Dict]) -> None:
        self.grid_matrix = [[0] * self.grid_size[1] for _ in range(self.grid_size[0])]
        for comp in components:
            for (row, col) in self._get_occupied_cells(comp):
                self.grid_matrix[row][col] = 1

    def _get_occupied_cells(self, component: Dict) -> List[Tuple[int, int]]:
        x, y = component["center"]
        comp_shape = component["shape"]
        grid_x = x / self.cell_width
        grid_y = y / self.cell_height
        occupied = []

        if comp_shape == "rect":
            w, h = component["width"], component["height"]
            grid_w, grid_h = w / self.cell_width, h / self.cell_height
            min_col, max_col = max(0, int(grid_x - grid_w / 2)), min(self.grid_size[1] - 1, int(grid_x + grid_w / 2))
            min_row, max_row = max(0, int(grid_y - grid_h / 2)), min(self.grid_size[0] - 1, int(grid_y + grid_h / 2))
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    occupied.append((row, col))

        elif comp_shape == "circle":
            r = component["radius"]
            grid_r = r / self.cell_width
            min_col, max_col = max(0, int(grid_x - grid_r)), min(self.grid_size[1] - 1, int(grid_x + grid_r))
            min_row, max_row = max(0, int(grid_y - grid_r)), min(self.grid_size[0] - 1, int(grid_y + grid_r))
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    dx, dy = (col + 0.5) - grid_x, (row + 0.5) - grid_y
                    if dx * dx + dy * dy <= grid_r * grid_r:
                        occupied.append((row, col))

        elif comp_shape == "capsule":
            l, w = component["length"], component["width"]
            rect_len, grid_w = l - w, w / self.cell_width
            grid_rect_len = rect_len / self.cell_width
            min_col_r, max_col_r = max(0, int(grid_x - grid_rect_len / 2)), min(self.grid_size[1] - 1,
                                                                                int(grid_x + grid_rect_len / 2))
            min_row_r, max_row_r = max(0, int(grid_y - grid_w / 2)), min(self.grid_size[0] - 1,
                                                                         int(grid_y + grid_w / 2))
            rect_cells = [(row, col) for row in range(min_row_r, max_row_r + 1) for col in
                          range(min_col_r, max_col_r + 1)]

            left_x, right_x = grid_x - grid_rect_len / 2, grid_x + grid_rect_len / 2
            radius = grid_w / 2
            circle_cells = []
            for row in range(max(0, int(grid_y - radius)), min(self.grid_size[0] - 1, int(grid_y + radius)) + 1):
                for col in range(max(0, int(left_x - radius)), min(self.grid_size[1] - 1, int(left_x + radius)) + 1):
                    dx, dy = (col + 0.5) - left_x, (row + 0.5) - grid_y
                    if dx * dx + dy * dy <= radius * radius:
                        circle_cells.append((row, col))
            for row in range(max(0, int(grid_y - radius)), min(self.grid_size[0] - 1, int(grid_y + radius)) + 1):
                for col in range(max(0, int(right_x - radius)), min(self.grid_size[1] - 1, int(right_x + radius)) + 1):
                    dx, dy = (col + 0.5) - right_x, (row + 0.5) - grid_y
                    if dx * dx + dy * dy <= radius * radius:
                        circle_cells.append((row, col))
            occupied = list(set(rect_cells + circle_cells))

        return occupied

    def _sample_single_var(self, var_idx: int, comp_idx: int, is_x: bool) -> Optional[float]:
        """单变量采样（x或y），其他变量固定"""
        comp = self.components[comp_idx]
        shape = comp["shape"]
        max_tries = 200
        width, height = self.design_area

        # 确定变量约束范围
        if is_x:
            fixed_y = self.vars[2 * comp_idx + 1]  # y坐标固定
            if shape == "rect":
                w = comp["width"]
                min_val, max_val = w / 2, width - w / 2
            elif shape == "circle":
                r = comp["radius"]
                min_val, max_val = r, width - r
            elif shape == "capsule":
                l = comp["length"]
                min_val, max_val = l / 2, width - l / 2
            else:
                return None
        else:
            fixed_x = self.vars[2 * comp_idx]  # x坐标固定
            if shape == "rect":
                h = comp["height"]
                min_val, max_val = h / 2, height - h / 2
            elif shape == "circle":
                r = comp["radius"]
                min_val, max_val = r, height - r
            elif shape == "capsule":
                w = comp["width"]
                min_val, max_val = w / 2, height - w / 2
            else:
                return None

        # 采样新值并检查冲突
        for _ in range(max_tries):
            new_val = random.uniform(min_val, max_val)
            new_x, new_y = (new_val, fixed_y) if is_x else (fixed_x, new_val)

            temp_comp = comp.copy()
            temp_comp["center"] = (new_x, new_y)
            temp_cells = self._get_occupied_cells(temp_comp)

            overlap = False
            for other_idx in range(len(self.components)):
                if other_idx == comp_idx:
                    continue
                other_x = self.vars[2 * other_idx]
                other_y = self.vars[2 * other_idx + 1]
                other_comp = self.components[other_idx].copy()
                other_comp["center"] = (other_x, other_y)
                if len(set(temp_cells) & set(self._get_occupied_cells(other_comp))) > 0:
                    overlap = True
                    break
            if not overlap:
                return new_val
        return None

    def sample(self,
               components: List[Dict],
               burn_in: int = 100,  # Burn-in周期（前100次迭代不采样）
               sample_interval: int = 10,  # 间隔采样步长（每10次迭代采一次样）
               total_samples: int = 20) -> Tuple[bool, List[List[Dict]]]:
        """
        带Burn-in和间隔采样的GibLS主方法
        :param burn_in: Burn-in周期（迭代次数）
        :param sample_interval: 采样间隔（迭代次数）
        :param total_samples: 目标采样数量
        :return: (是否成功, 采样的布局列表)
        """
        # 1. 初始布局（SeqLS）
        initial_components = self.seqls_sampler.layout_sampling(components, max_sampling_attempts=100)
        if not initial_components:
            print("初始布局失败")
            return False, []
        self.components = initial_components.copy()
        self._update_grid_matrix(self.components)

        # 2. 初始化变量列表（x0, y0, x1, y1, ..., xn, yn）
        Ns = len(self.components)
        self.vars = []
        for comp in self.components:
            x, y = comp["center"]
            self.vars.extend([x, y])  # [x0, y0, x1, y1, ...]

        # 3. 迭代优化（含Burn-in和间隔采样）
        current_iter = 0  # 当前总迭代次数
        samples = []  # 保存采样结果
        total_vars = 2 * Ns  # 总变量数（2Ns）

        print(f"开始迭代：Burn-in={burn_in}, 间隔={sample_interval}, 目标样本数={total_samples}")

        while len(samples) < total_samples:
            # 3.1 完整Gibbs迭代：更新所有2Ns个变量
            for var_idx in range(total_vars):
                comp_idx = var_idx // 2  # 变量所属元件
                is_x = (var_idx % 2 == 0)  # 是否为x坐标变量

                original_val = self.vars[var_idx]
                new_val = self._sample_single_var(var_idx, comp_idx, is_x)

                if new_val is not None:
                    self.vars[var_idx] = new_val  # 更新变量
                    # 同步更新元件center
                    x = self.vars[2 * comp_idx]
                    y = self.vars[2 * comp_idx + 1]
                    self.components[comp_idx]["center"] = (x, y)

            # 3.2 更新网格矩阵（完成一次完整迭代后）
            self._update_grid_matrix(self.components)
            current_iter += 1
            print(f"迭代 {current_iter}/{burn_in + total_samples * sample_interval}", end="\r")

            # 3.3 检查是否采样：需满足（超过Burn-in周期 + 达到采样间隔）
            if current_iter > burn_in:
                # 计算当前是否为采样点（间隔采样）
                if (current_iter - burn_in) % sample_interval == 0:
                    # 深拷贝当前布局作为样本（避免后续修改影响）
                    sample = [comp.copy() for comp in self.components]
                    samples.append(sample)
                    print(f"\n已采样 {len(samples)}/{total_samples}（迭代次数：{current_iter}）")

        print("\n采样完成")
        return True, samples

    # 可视化方法（保持不变）
    def visualize_layout(self, components: List[Dict], show_grid: bool = True, show_occupied: bool = False):
        fig, ax = plt.subplots(figsize=(10, 10))
        width, height = self.design_area
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.set_title("GibLS Sampled Layout")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        if show_grid:
            for x in np.arange(0, width, self.cell_width):
                ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
            for y in np.arange(0, height, self.cell_height):
                ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

        if show_occupied:
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    if self.grid_matrix[row][col] == 1:
                        # 计算网格左上角坐标
                        x = col * self.cell_width
                        y = row * self.cell_height
                        # 绘制半透明粉红色矩形
                        rect = Rectangle(
                            (x, y), self.cell_width, self.cell_height,
                            fill=True, facecolor='red', alpha=0.2  # 粉红色半透明
                        )
                        ax.add_patch(rect)

        for i, comp in enumerate(components):
            cx, cy = comp["center"]
            color = f"C{i}"
            if comp["shape"] == "rect":
                w, h = comp["width"], comp["height"]
                ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                                       facecolor=color, edgecolor='black', alpha=0.7))
            elif comp["shape"] == "circle":
                r = comp["radius"]
                ax.add_patch(Circle((cx, cy), r,
                                    facecolor=color, edgecolor='black', alpha=0.7))
            elif comp["shape"] == "capsule":
                l, w = comp["length"], comp["width"]
                rect_len = l - w
                ax.add_patch(Rectangle((cx - rect_len / 2, cy - w / 2), rect_len, w,
                                       facecolor=color, edgecolor='black', alpha=0.7))
                ax.add_patch(Circle((cx - rect_len / 2, cy), w / 2,
                                    facecolor=color, edgecolor='black', alpha=0.7))
                ax.add_patch(Circle((cx + rect_len / 2, cy), w / 2,
                                    facecolor=color, edgecolor='black', alpha=0.7))
            ax.text(cx, cy, f"ID: {comp['id']}", ha='center', va='center',
                    color='white', fontweight='bold', fontsize=8)

        plt.tight_layout()
        plt.show()


# 示例调用
if __name__ == "__main__":
    random.seed(42)
    layout_domain = (1.0, 1.0)
    mesh_size = (20, 20)
    sampler = GibLS_Sampler(layout_domain, mesh_size)

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

    # 运行采样：Burn-in=100次迭代，间隔10次，采样20个样本
    success, samples = sampler.sample(
        components=components,
        burn_in=100,
        sample_interval=10,
        total_samples=20
    )

    if success:
        print(f"\n成功采样 {len(samples)} 个布局")
        # 可视化第1个样本（可改为任意索引，如-1表示最后一个）
        sampler.visualize_layout(samples[0], show_grid=True, show_occupied=True)
        # 可视化最后一个样本
        sampler.visualize_layout(samples[-1], show_grid=True, show_occupied=True)
    else:
        print("采样失败")