import os
import random
import numpy as np
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import h5py
import math
from typing import List, Tuple, Dict, Optional, Union


class Satellite2DLayout:
    """
    卫星二维布局类，支持矩形、圆形和胶囊型元件，用于热传导分析
    """

    def __init__(self, layout_size: Tuple[float, float], k: float, mesh_resolution: int):
        """
        构造函数

        参数:
            layout_size: 布局整体尺寸 (宽度, 高度)，单位米
            k: 材料热导率，单位 W/(m·K)
            mesh_resolution: 网格分辨率，用于后续有限元分析
        """
        self.layout_size = layout_size  # (width, height)
        self.k = k  # 热导率
        self.mesh_resolution = mesh_resolution  # 网格分辨率

        # 存储元件信息，每个元件为字典
        self.components = []  # 元素格式: {'id': int, 'type': str, 'size': ..., 'coords': (x,y), 'power': float, ...}

        # 边界条件: Dirichlet(温度)和Neumann(热流)
        self.boundary_conditions = {
            'Dirichlet': [],  # 元素: {'segment': str, 'temperature': float}
            'Neumann': []  # 元素: {'segment': str, 'heat_flux': float} (绝热为0)
        }

        # 布局状态
        self.layout_valid = True  # 是否有效(无重叠等)
        self.component_id_counter = 0  # 元件ID计数器

    def add_component(self,
                      component_type: str = 'rect',
                      size: Optional[Union[Tuple[float, float], float, Tuple[float, float]]] = None,
                      coords: Optional[Tuple[float, float]] = None,
                      power: Optional[float] = None,
                      preset_id: Optional[int] = None,
                      max_attempts: int = 100) -> bool:
        """
        增加元件，支持手动指定参数或随机生成

        参数:
            component_type: 元件类型，'rect'(矩形), 'circle'(圆形), 'capsule'(胶囊型)
            size: 尺寸参数:
                - 矩形: (宽度, 高度)
                - 圆形: 半径
                - 胶囊型: (长度, 宽度) 长度为矩形部分+两个半圆直径
            coords: 中心坐标 (x, y)
            power: 功率，单位W
            preset_id: 预置参数编号，从文件加载预设参数
            max_attempts: 随机生成时最大尝试次数(避免重叠)

        返回:
            bool: 元件添加成功与否
        """
        # 验证元件类型
        if component_type not in ['rect', 'circle', 'capsule']:
            raise ValueError(f"不支持的元件类型: {component_type}，支持的类型为'rect', 'circle', 'capsule'")

        # 处理预置参数
        if preset_id is not None:
            preset = self._load_preset_component(preset_id)
            component_type = preset.get('type', component_type)
            size = preset.get('size', size)
            coords = preset.get('coords', coords)
            power = preset.get('power', power)

        # 生成或验证尺寸
        size = self._generate_or_validate_size(component_type, size)
        if size is None:
            return False

        # 生成或验证坐标
        coords = self._generate_or_validate_coords(component_type, size, coords, max_attempts)
        if coords is None:
            return False

        # 生成或验证功率
        power = self._generate_or_validate_power(power)
        if power is None:
            return False

        # 创建元件字典
        component = {
            'id': self.component_id_counter,
            'type': component_type,
            'size': size,
            'coords': coords,
            'power': power,
            'center': coords  # 中心点与坐标一致
        }

        # 检查与现有元件是否重叠
        if self._is_overlapping(component):
            return False

        # 添加元件
        self.components.append(component)
        self.component_id_counter += 1
        return True

    def set_boundary_condition(self,
                               condition_type: str,
                               value: float,
                               boundary_segment: str = 'all') -> None:
        """
        设置边界条件

        参数:
            condition_type: 条件类型，'Dirichlet'或'Neumann'
            value: Dirichlet为温度值(K)，Neumann为热流(W/m²)，绝热为0
            boundary_segment: 边界段，'left', 'right', 'top', 'bottom', 'all'
        """
        if condition_type not in ['Dirichlet', 'Neumann']:
            raise ValueError(f"不支持的边界条件类型: {condition_type}")

        if boundary_segment not in ['left', 'right', 'top', 'bottom', 'all']:
            raise ValueError(f"不支持的边界段: {boundary_segment}")

        condition = {
            'segment': boundary_segment,
            'value': value
        }

        self.boundary_conditions[condition_type].append(condition)

    def is_layout_valid(self) -> bool:
        """检查整个布局是否有效(无重叠且所有元件在布局范围内)"""
        # 检查所有元件是否在布局范围内
        for comp in self.components:
            if not self._is_inside_layout(comp):
                self.layout_valid = False
                return False

        # 检查元件间是否重叠
        for i in range(len(self.components)):
            for j in range(i + 1, len(self.components)):
                if self._is_overlapping_between_two(self.components[i], self.components[j]):
                    self.layout_valid = False
                    return False

        self.layout_valid = True
        return True

    def to_yaml(self, file_path: str) -> None:
        """输出YAML记录文件"""
        data = {
            'layout_info': {
                'size': self.layout_size,
                'thermal_conductivity': self.k,
                'mesh_resolution': self.mesh_resolution,
                'validity': self.layout_valid,
                'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'components': self.components,
            'boundary_conditions': self.boundary_conditions
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    def to_fenicsx_format(self) -> Dict:
        """输出适合fenicsx求解的格式"""
        # 转换边界条件为fenicsx格式
        fenics_bc = []
        for bc_type, conditions in self.boundary_conditions.items():
            for cond in conditions:
                fenics_bc.append({
                    'type': bc_type.lower(),
                    'segment': cond['segment'],
                    'value': cond['value']
                })

        # 转换元件为热源
        heat_sources = []
        for comp in self.components:
            heat_sources.append({
                'type': comp['type'],
                'geometry': {
                    'center': comp['coords'],
                    'size': comp['size']
                },
                'power': comp['power']
            })

        return {
            'domain': {
                'size': self.layout_size,
                'resolution': self.mesh_resolution
            },
            'thermal_conductivity': self.k,
            'boundary_conditions': fenics_bc,
            'heat_sources': heat_sources
        }

    def to_sdf(self, resolution: Tuple[int, int]) -> np.ndarray:
        """
        生成有向距离函数(SDF)用于深度学习训练

        参数:
            resolution: 输出分辨率 (width_pixels, height_pixels)

        返回:
            np.ndarray: 有向距离函数数组，正值表示外部，负值表示内部
        """
        width, height = self.layout_size
        px, py = resolution
        sdf = np.full((py, px), float('inf'))

        # 计算每个像素的有向距离
        for y_idx in range(py):
            for x_idx in range(px):
                # 转换像素坐标到实际坐标
                x = (x_idx + 0.5) * (width / px)
                y = (y_idx + 0.5) * (height / py)

                # 计算到所有元件的最小距离
                min_dist = float('inf')
                for comp in self.components:
                    dist = self._distance_to_component((x, y), comp)
                    if dist < min_dist:
                        min_dist = dist

                sdf[y_idx, x_idx] = min_dist

        return sdf

    def visualize(self, save_path: Optional[str] = None) -> None:
        """可视化布局"""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.layout_size[0])
        ax.set_ylim(0, self.layout_size[1])
        ax.set_aspect('equal')
        ax.set_title('Satellite Component Layout')

        # 绘制边界
        rect = plt.Rectangle((0, 0), self.layout_size[0], self.layout_size[1],
                             fill=False, edgecolor='black', linestyle='--')
        ax.add_patch(rect)

        # 绘制元件
        colors = {'rect': 'blue', 'circle': 'red', 'capsule': 'green'}
        for comp in self.components:
            c = comp['coords']
            s = comp['size']
            t = comp['type']

            if t == 'rect':
                w, h = s
                rect = plt.Rectangle((c[0] - w / 2, c[1] - h / 2), w, h,
                                     fill=True, alpha=0.5, color=colors[t])
                ax.add_patch(rect)
            elif t == 'circle':
                r = s
                circle = plt.Circle(c, r, fill=True, alpha=0.5, color=colors[t])
                ax.add_patch(circle)
            elif t == 'capsule':
                length, width = s
                # 胶囊型由矩形和两个半圆组成
                rect_length = length - width  # 矩形部分长度
                # 绘制矩形
                rect = plt.Rectangle((c[0] - rect_length / 2, c[1] - width / 2),
                                     rect_length, width,
                                     fill=True, alpha=0.5, color=colors[t])
                ax.add_patch(rect)
                # 绘制左右半圆
                left_circle = plt.Circle((c[0] - rect_length / 2, c[1]), width / 2,
                                         fill=True, alpha=0.5, color=colors[t])
                right_circle = plt.Circle((c[0] + rect_length / 2, c[1]), width / 2,
                                          fill=True, alpha=0.5, color=colors[t])
                ax.add_patch(left_circle)
                ax.add_patch(right_circle)

            # 标注功率
            ax.text(c[0], c[1], f"P={comp['power']:.1f}W",
                    ha='center', va='center', fontsize=8)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    # 辅助函数：加载预置元件参数
    def _load_preset_component(self, preset_id: int) -> Dict:
        """加载预置元件参数，实际应用中可从文件读取"""
        # 示例预置参数
        presets = {
            1: {'type': 'rect', 'size': (0.1, 0.08), 'power': 15.0},
            2: {'type': 'circle', 'size': 0.05, 'power': 10.0},
            3: {'type': 'capsule', 'size': (0.2, 0.06), 'power': 20.0}
        }
        return presets.get(preset_id, {})

    # 辅助函数：生成或验证尺寸
    def _generate_or_validate_size(self, component_type: str, size: Optional[Union[Tuple[float, float], float]]) -> \
    Union[Tuple[float, float], float, None]:
        if size is not None:
            # 验证尺寸格式
            if component_type == 'rect' and (not isinstance(size, tuple) or len(size) != 2):
                raise ValueError("矩形元件尺寸必须是 tuple (宽度, 高度)")
            if component_type == 'circle' and not isinstance(size, (int, float)):
                raise ValueError("圆形元件尺寸必须是半径(数值)")
            if component_type == 'capsule' and (not isinstance(size, tuple) or len(size) != 2):
                raise ValueError("胶囊型元件尺寸必须是 tuple (长度, 宽度)")
            return size

        # 随机生成尺寸
        if component_type == 'rect':
            return (random.uniform(0.05, 0.2), random.uniform(0.05, 0.2))
        elif component_type == 'circle':
            return random.uniform(0.03, 0.1)
        elif component_type == 'capsule':
            width = random.uniform(0.04, 0.15)
            length = random.uniform(width, 0.3)  # 长度至少大于宽度
            return (length, width)
        return None

    # 辅助函数：生成或验证坐标
    def _generate_or_validate_coords(self,
                                     component_type: str,
                                     size: Union[Tuple[float, float], float],
                                     coords: Optional[Tuple[float, float]],
                                     max_attempts: int) -> Optional[Tuple[float, float]]:
        # 验证坐标是否在布局范围内
        if coords is not None:
            test_comp = {
                'type': component_type,
                'size': size,
                'coords': coords
            }
            if self._is_inside_layout(test_comp):
                return coords
            else:
                raise ValueError(f"坐标 {coords} 超出布局范围")

        # 随机生成坐标，确保在布局范围内且不重叠
        for _ in range(max_attempts):
            x = random.uniform(0, self.layout_size[0])
            y = random.uniform(0, self.layout_size[1])
            test_comp = {
                'type': component_type,
                'size': size,
                'coords': (x, y)
            }

            if self._is_inside_layout(test_comp) and not self._is_overlapping(test_comp):
                return (x, y)

        # 达到最大尝试次数仍未找到合适位置
        return None

    # 辅助函数：生成或验证功率
    def _generate_or_validate_power(self, power: Optional[float]) -> Optional[float]:
        if power is not None:
            if power < 0:
                raise ValueError("功率不能为负值")
            return power
        # 随机生成功率 (5-50W)
        return random.uniform(5.0, 50.0)

    # 辅助函数：检查元件是否在布局范围内
    def _is_inside_layout(self, component: Dict) -> bool:
        x, y = component['coords']
        size = component['size']
        type_ = component['type']
        width, height = self.layout_size

        if type_ == 'rect':
            w, h = size
            return (x - w / 2 >= 0 and x + w / 2 <= width and
                    y - h / 2 >= 0 and y + h / 2 <= height)

        elif type_ == 'circle':
            r = size
            return (x - r >= 0 and x + r <= width and
                    y - r >= 0 and y + r <= height)

        elif type_ == 'capsule':
            length, cap_width = size
            # 胶囊型最左端和最右端
            leftmost = x - length / 2
            rightmost = x + length / 2
            topmost = y + cap_width / 2
            bottommost = y - cap_width / 2

            return (leftmost >= 0 and rightmost <= width and
                    bottommost >= 0 and topmost <= height)

        return False

    # 辅助函数：检查新元件是否与现有元件重叠
    def _is_overlapping(self, new_component: Dict) -> bool:
        for existing in self.components:
            if self._is_overlapping_between_two(new_component, existing):
                return True
        return False

    # 辅助函数：检查两个元件是否重叠
    def _is_overlapping_between_two(self, comp1: Dict, comp2: Dict) -> bool:
        """检查任意两个元件(矩形、圆形、胶囊型)是否重叠"""
        t1, t2 = comp1['type'], comp2['type']
        c1, c2 = comp1['coords'], comp2['coords']
        s1, s2 = comp1['size'], comp2['size']

        # 处理所有类型组合的重叠判断
        if t1 == 'rect' and t2 == 'rect':
            return self._rect_rect_overlap(c1, s1, c2, s2)

        elif t1 == 'circle' and t2 == 'circle':
            return self._circle_circle_overlap(c1, s1, c2, s2)

        elif t1 == 'rect' and t2 == 'circle':
            return self._rect_circle_overlap(c1, s1, c2, s2)

        elif t1 == 'circle' and t2 == 'rect':
            return self._rect_circle_overlap(c2, s2, c1, s1)

        elif t1 == 'capsule' and t2 == 'capsule':
            return self._capsule_capsule_overlap(c1, s1, c2, s2)

        elif t1 == 'capsule' and t2 == 'rect':
            return self._capsule_rect_overlap(c1, s1, c2, s2)

        elif t1 == 'rect' and t2 == 'capsule':
            return self._capsule_rect_overlap(c2, s2, c1, s1)

        elif t1 == 'capsule' and t2 == 'circle':
            return self._capsule_circle_overlap(c1, s1, c2, s2)

        elif t1 == 'circle' and t2 == 'capsule':
            return self._capsule_circle_overlap(c2, s2, c1, s1)

        return False

    # 重叠判断的具体实现
    def _rect_rect_overlap(self, c1: Tuple[float, float], s1: Tuple[float, float],
                           c2: Tuple[float, float], s2: Tuple[float, float]) -> bool:
        x1, y1 = c1
        w1, h1 = s1
        x2, y2 = c2
        w2, h2 = s2

        # 检查x方向和y方向是否都重叠
        x_overlap = (x1 - w1 / 2 < x2 + w2 / 2) and (x1 + w1 / 2 > x2 - w2 / 2)
        y_overlap = (y1 - h1 / 2 < y2 + h2 / 2) and (y1 + h1 / 2 > y2 - h2 / 2)
        return x_overlap and y_overlap

    def _circle_circle_overlap(self, c1: Tuple[float, float], r1: float,
                               c2: Tuple[float, float], r2: float) -> bool:
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        distance = math.sqrt(dx * dx + dy * dy)
        return distance < (r1 + r2)

    def _rect_circle_overlap(self, rect_c: Tuple[float, float], rect_s: Tuple[float, float],
                             circle_c: Tuple[float, float], circle_r: float) -> bool:
        rx, ry = rect_c
        rw, rh = rect_s
        cx, cy = circle_c

        # 计算圆心到矩形的最近点
        closest_x = max(rx - rw / 2, min(cx, rx + rw / 2))
        closest_y = max(ry - rh / 2, min(cy, ry + rh / 2))

        # 计算最近点到圆心的距离
        dx = cx - closest_x
        dy = cy - closest_y
        distance = math.sqrt(dx * dx + dy * dy)

        return distance < circle_r

    def _capsule_capsule_overlap(self, c1: Tuple[float, float], s1: Tuple[float, float],
                                 c2: Tuple[float, float], s2: Tuple[float, float]) -> bool:
        # 胶囊型可以看作矩形+两个圆形，检查任何部分是否重叠
        l1, w1 = s1
        l2, w2 = s2

        # 检查胶囊的矩形部分是否重叠
        rect1_c = c1
        rect1_s = (l1 - w1, w1)  # 胶囊的矩形部分尺寸
        rect2_c = c2
        rect2_s = (l2 - w2, w2)

        if self._rect_rect_overlap(rect1_c, rect1_s, rect2_c, rect2_s):
            return True

        # 检查胶囊1的矩形与胶囊2的圆形部分是否重叠
        circle2_left_c = (c2[0] - (l2 - w2) / 2, c2[1])
        circle2_right_c = (c2[0] + (l2 - w2) / 2, c2[1])

        if (self._rect_circle_overlap(rect1_c, rect1_s, circle2_left_c, w2 / 2) or
                self._rect_circle_overlap(rect1_c, rect1_s, circle2_right_c, w2 / 2)):
            return True

        # 检查胶囊2的矩形与胶囊1的圆形部分是否重叠
        circle1_left_c = (c1[0] - (l1 - w1) / 2, c1[1])
        circle1_right_c = (c1[0] + (l1 - w1) / 2, c1[1])

        if (self._rect_circle_overlap(rect2_c, rect2_s, circle1_left_c, w1 / 2) or
                self._rect_circle_overlap(rect2_c, rect2_s, circle1_right_c, w1 / 2)):
            return True

        # 检查胶囊1的圆形与胶囊2的圆形是否重叠
        if (self._circle_circle_overlap(circle1_left_c, w1 / 2, circle2_left_c, w2 / 2) or
                self._circle_circle_overlap(circle1_left_c, w1 / 2, circle2_right_c, w2 / 2) or
                self._circle_circle_overlap(circle1_right_c, w1 / 2, circle2_left_c, w2 / 2) or
                self._circle_circle_overlap(circle1_right_c, w1 / 2, circle2_right_c, w2 / 2)):
            return True

        return False

    def _capsule_rect_overlap(self, capsule_c: Tuple[float, float], capsule_s: Tuple[float, float],
                              rect_c: Tuple[float, float], rect_s: Tuple[float, float]) -> bool:
        l, w = capsule_s
        # 胶囊的矩形部分
        cap_rect_c = capsule_c
        cap_rect_s = (l - w, w)

        # 检查胶囊矩形部分与矩形是否重叠
        if self._rect_rect_overlap(cap_rect_c, cap_rect_s, rect_c, rect_s):
            return True

        # 检查胶囊的两个圆形与矩形是否重叠
        circle_left_c = (capsule_c[0] - (l - w) / 2, capsule_c[1])
        circle_right_c = (capsule_c[0] + (l - w) / 2, capsule_c[1])

        if (self._rect_circle_overlap(rect_c, rect_s, circle_left_c, w / 2) or
                self._rect_circle_overlap(rect_c, rect_s, circle_right_c, w / 2)):
            return True

        return False

    def _capsule_circle_overlap(self, capsule_c: Tuple[float, float], capsule_s: Tuple[float, float],
                                circle_c: Tuple[float, float], circle_r: float) -> bool:
        l, w = capsule_s
        # 胶囊的矩形部分
        cap_rect_c = capsule_c
        cap_rect_s = (l - w, w)

        # 检查矩形部分与圆形是否重叠
        if self._rect_circle_overlap(cap_rect_c, cap_rect_s, circle_c, circle_r):
            return True

        # 检查胶囊的两个圆形与圆形是否重叠
        circle_left_c = (capsule_c[0] - (l - w) / 2, capsule_c[1])
        circle_right_c = (capsule_c[0] + (l - w) / 2, capsule_c[1])

        if (self._circle_circle_overlap(circle_left_c, w / 2, circle_c, circle_r) or
                self._circle_circle_overlap(circle_right_c, w / 2, circle_c, circle_r)):
            return True

        return False

    # 计算点到元件的有向距离
    def _distance_to_component(self, point: Tuple[float, float], component: Dict) -> float:
        """计算点到元件的有向距离(内部为负，外部为正)"""
        x, y = point
        c = component['coords']
        s = component['size']
        t = component['type']

        if t == 'rect':
            return self._distance_to_rect((x, y), c, s)
        elif t == 'circle':
            return self._distance_to_circle((x, y), c, s)
        elif t == 'capsule':
            return self._distance_to_capsule((x, y), c, s)
        return float('inf')

    def _distance_to_rect(self, point: Tuple[float, float], center: Tuple[float, float],
                          size: Tuple[float, float]) -> float:
        px, py = point
        cx, cy = center
        w, h = size

        # 计算到矩形边界的距离
        dx = max(px - (cx + w / 2), (cx - w / 2) - px, 0)
        dy = max(py - (cy + h / 2), (cy - h / 2) - py, 0)

        # 内部点距离为负
        inside = (px >= cx - w / 2 and px <= cx + w / 2 and
                  py >= cy - h / 2 and py <= cy + h / 2)

        distance = math.sqrt(dx * dx + dy * dy)
        return -distance if inside else distance

    def _distance_to_circle(self, point: Tuple[float, float], center: Tuple[float, float], radius: float) -> float:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        distance = math.sqrt(dx * dx + dy * dy)
        return distance - radius  # 内部为负，外部为正

    def _distance_to_capsule(self, point: Tuple[float, float], center: Tuple[float, float],
                             size: Tuple[float, float]) -> float:
        l, w = size
        cx, cy = center
        px, py = point

        # 胶囊的矩形部分长度
        rect_length = l - w

        # 计算点到胶囊轴线的投影
        # 轴线从(cx - rect_length/2, cy)到(cx + rect_length/2, cy)
        axis_start = (cx - rect_length / 2, cy)
        axis_end = (cx + rect_length / 2, cy)

        # 计算点到线段的距离
        dx = axis_end[0] - axis_start[0]
        dy = axis_end[1] - axis_start[1]

        if dx == 0 and dy == 0:  # 线段长度为0
            t = 0.0
        else:
            t = ((px - axis_start[0]) * dx + (py - axis_start[1]) * dy) / (dx * dx + dy * dy)
            t = max(0.0, min(1.0, t))  # 投影到线段上

        # 线段上最近点
        nearest_x = axis_start[0] + t * dx
        nearest_y = axis_start[1] + t * dy

        # 到轴线的距离
        dist_to_axis = math.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)

        # 胶囊半径
        radius = w / 2

        # 计算距离：内部为负，外部为正
        if dist_to_axis <= radius:
            # 在胶囊内部
            return dist_to_axis - radius
        else:
            # 在胶囊外部，计算到最近端点圆的距离
            dist_to_left = self._distance_to_circle(point, axis_start, radius)
            dist_to_right = self._distance_to_circle(point, axis_end, radius)
            return min(dist_to_axis - radius, dist_to_left, dist_to_right)


class SatelliteLayoutDatasetGenerator:
    """卫星布局数据集生成类"""

    def __init__(self, base_layout_size: Tuple[float, float],
                 thermal_conductivity: float,
                 mesh_resolution: int,
                 save_root: str):
        """
        构造函数

        参数:
            base_layout_size: 基础布局尺寸 (宽度, 高度)
            thermal_conductivity: 材料热导率
            mesh_resolution: 网格分辨率
            save_root: 数据集保存根目录
        """
        self.base_layout_size = base_layout_size
        self.k = thermal_conductivity
        self.mesh_resolution = mesh_resolution
        self.save_root = save_root

        # 创建保存目录
        os.makedirs(save_root, exist_ok=True)

        # 数据集信息
        self.dataset_info = {
            'dataset_size': 0,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'variable_distributions': {},  # 变量分布定义
            'layout_params': {
                'size': base_layout_size,
                'thermal_conductivity': thermal_conductivity,
                'mesh_resolution': mesh_resolution
            },
            'samples': []  # 样本列表
        }

        # 初始化两种布局生成算法
        self.layout_methods = {
            'SeqLS': self._generate_with_seqls,
            'GibLS': self._generate_with_gibls
        }

    def define_variable_distribution(self, variable_name: str,
                                     distribution_type: str,
                                     params: Dict) -> None:
        """
        定义变量分布

        参数:
            variable_name: 变量名称，如 'rect_width', 'circle_radius', 'capsule_length' 等
            distribution_type: 分布类型，'uniform', 'gaussian' 等
            params: 分布参数，如均匀分布 {'min': 0.1, 'max': 0.2}，高斯分布 {'mean': 0.15, 'std': 0.03}
        """
        if distribution_type not in ['uniform', 'gaussian']:
            raise ValueError(f"不支持的分布类型: {distribution_type}")

        self.dataset_info['variable_distributions'][variable_name] = {
            'type': distribution_type,
            'params': params
        }

    def generate_dataset(self,
                         dataset_size: int,
                         num_components_range: Tuple[int, int],
                         component_types: List[str] = ['rect', 'circle', 'capsule'],
                         layout_method: str = 'SeqLS',
                         boundary_conditions: Dict = None,
                         sdf_resolution: Tuple[int, int] = (256, 256)) -> None:
        """
        生成数据集

        参数:
            dataset_size: 数据集大小
            num_components_range: 元件数量范围 (min, max)
            component_types: 可选的元件类型列表
            layout_method: 布局生成方法 'SeqLS' 或 'GibLS'
            boundary_conditions: 边界条件定义
            sdf_resolution: SDF的分辨率
        """
        if layout_method not in self.layout_methods:
            raise ValueError(f"不支持的布局生成方法: {layout_method}")

        # 默认边界条件
        if boundary_conditions is None:
            boundary_conditions = {
                'Dirichlet': [{'segment': 'all', 'value': 300.0}],  # 300K
                'Neumann': []
            }

        # 记录数据集参数
        self.dataset_info['dataset_size'] = dataset_size
        self.dataset_info['component_types'] = component_types
        self.dataset_info['num_components_range'] = num_components_range
        self.dataset_info['layout_method'] = layout_method
        self.dataset_info['boundary_conditions'] = boundary_conditions
        self.dataset_info['sdf_resolution'] = sdf_resolution

        # 生成样本
        for sample_id in range(dataset_size):
            print(f"生成样本 {sample_id + 1}/{dataset_size}")

            # 创建样本目录
            sample_dir = os.path.join(self.save_root, f"sample_{sample_id:06d}")
            os.makedirs(sample_dir, exist_ok=True)

            # 随机元件数量
            num_components = random.randint(num_components_range[0], num_components_range[1])

            # 创建布局
            layout = Satellite2DLayout(
                layout_size=self.base_layout_size,
                k=self.k,
                mesh_resolution=self.mesh_resolution
            )

            # 设置边界条件
            for bc_type, conditions in boundary_conditions.items():
                for cond in conditions:
                    layout.set_boundary_condition(
                        condition_type=bc_type,
                        value=cond['value'],
                        boundary_segment=cond['segment']
                    )

            # 使用指定方法生成布局
            success = self.layout_methods[layout_method](
                layout=layout,
                num_components=num_components,
                component_types=component_types,
                max_attempts=500
            )

            if not success:
                print(f"样本 {sample_id} 生成失败，跳过...")
                continue

            # 保存布局文件
            yaml_path = os.path.join(sample_dir, "layout.yaml")
            layout.to_yaml(yaml_path)

            # 保存可视化图
            viz_path = os.path.join(sample_dir, "layout_visualization.png")
            layout.visualize(save_path=viz_path)

            # 保存SDF
            sdf = layout.to_sdf(resolution=sdf_resolution)
            sdf_path = os.path.join(sample_dir, "sdf.npy")
            np.save(sdf_path, sdf)

            # 保存fenicsx格式数据
            fenicsx_data = layout.to_fenicsx_format()
            fenicsx_path = os.path.join(sample_dir, "fenicsx_data.yaml")
            with open(fenicsx_path, 'w', encoding='utf-8') as f:
                yaml.dump(fenicsx_data, f, sort_keys=False)

            # 记录样本信息
            self.dataset_info['samples'].append({
                'id': sample_id,
                'path': sample_dir,
                'num_components': num_components,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        # 保存数据集信息
        dataset_info_path = os.path.join(self.save_root, "dataset_info.yaml")
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dataset_info, f, sort_keys=False, allow_unicode=True)

        print(f"数据集生成完成，共 {len(self.dataset_info['samples'])} 个有效样本")

    def aggregate_dataset(self, output_path: str = None) -> None:
        """
        将数据集整合为h5py文件

        参数:
            output_path: 输出文件路径，默认为保存根目录下的dataset.h5
        """
        if output_path is None:
            output_path = os.path.join(self.save_root, "dataset.h5")

        with h5py.File(output_path, 'w') as hf:
            # 存储数据集信息
            info_group = hf.create_group('dataset_info')
            for key, value in self.dataset_info.items():
                if key != 'samples':  # 样本列表单独处理
                    info_group.attrs[key] = str(value)

            # 存储样本数据
            samples_group = hf.create_group('samples')

            for sample in self.dataset_info['samples']:
                sample_id = sample['id']
                sample_group = samples_group.create_group(f"sample_{sample_id:06d}")

                # 存储SDF
                sdf_path = os.path.join(sample['path'], "sdf.npy")
                sdf = np.load(sdf_path)
                sample_group.create_dataset('sdf', data=sdf)

                # 存储布局信息
                yaml_path = os.path.join(sample['path'], "layout.yaml")
                with open(yaml_path, 'r') as f:
                    layout_data = yaml.safe_load(f)

                # 存储元件信息
                comps_group = sample_group.create_group('components')
                for i, comp in enumerate(layout_data['components']):
                    comp_group = comps_group.create_group(f"component_{i}")
                    for key, value in comp.items():
                        comp_group.attrs[key] = value

                # 存储边界条件
                bc_group = sample_group.create_group('boundary_conditions')
                for bc_type, conditions in layout_data['boundary_conditions'].items():
                    bc_type_group = bc_group.create_group(bc_type)
                    for i, cond in enumerate(conditions):
                        cond_group = bc_type_group.create_group(f"condition_{i}")
                        for key, value in cond.items():
                            cond_group.attrs[key] = value

        print(f"数据集已整合至 {output_path}")

    # SeqLS布局生成方法
    def _generate_with_seqls(self, layout: Satellite2DLayout,
                             num_components: int,
                             component_types: List[str],
                             max_attempts: int) -> bool:
        """使用SeqLS算法生成布局"""
        # 按面积从大到小生成元件（提高成功率）
        # 先确定所有元件的尺寸和类型，按面积排序
        components_to_add = []

        for _ in range(num_components):
            # 随机选择元件类型
            comp_type = random.choice(component_types)

            # 根据预设的分布生成尺寸
            size = self._generate_size_from_distribution(comp_type)

            components_to_add.append({
                'type': comp_type,
                'size': size,
                'area': self._calculate_component_area(comp_type, size)
            })

        # 按面积排序（从大到小）
        components_to_add.sort(key=lambda x: x['area'], reverse=True)

        # 逐个添加元件
        for comp in components_to_add:
            success = layout.add_component(
                component_type=comp['type'],
                size=comp['size'],
                max_attempts=max_attempts
            )
            if not success:
                return False

        return layout.is_layout_valid()

    # GibLS布局生成方法
    def _generate_with_gibls(self, layout: Satellite2DLayout,
                             num_components: int,
                             component_types: List[str],
                             max_attempts: int,
                             burn_in: int = 50,
                             interval: int = 10) -> bool:
        """使用GibLS算法生成布局"""
        # 先生成初始布局（使用SeqLS）
        initial_success = self._generate_with_seqls(
            layout=layout,
            num_components=num_components,
            component_types=component_types,
            max_attempts=max_attempts
        )

        if not initial_success:
            return False

        # 执行Gibbs采样迭代
        current_layout = [c.copy() for c in layout.components]

        for iter in range(burn_in + interval):
            # 对每个元件的x和y坐标进行采样
            for comp_idx in range(len(current_layout)):
                comp = current_layout[comp_idx]

                # 保存当前坐标用于恢复
                original_coords = comp['coords']

                # 采样x坐标
                new_x = self._sample_component_coord(
                    layout=layout,
                    component=comp,
                    coord_index=0,  # 0表示x坐标
                    current_layout=current_layout,
                    comp_idx=comp_idx
                )

                # 采样y坐标
                comp['coords'] = (new_x, comp['coords'][1])
                new_y = self._sample_component_coord(
                    layout=layout,
                    component=comp,
                    coord_index=1,  # 1表示y坐标
                    current_layout=current_layout,
                    comp_idx=comp_idx
                )

                # 更新坐标
                comp['coords'] = (new_x, new_y)

            # 间隔期后使用最终布局
            if iter >= burn_in + interval - 1:
                # 清空并更新布局
                layout.components = []
                layout.component_id_counter = 0
                for comp in current_layout:
                    layout.add_component(
                        component_type=comp['type'],
                        size=comp['size'],
                        coords=comp['coords'],
                        power=comp['power']
                    )

        return layout.is_layout_valid()

    # 辅助函数：从分布生成尺寸
    def _generate_size_from_distribution(self, component_type: str) -> Union[Tuple[float, float], float]:
        """根据预设的分布生成元件尺寸"""
        if component_type == 'rect':
            # 矩形：宽度和高度
            w_dist = self.dataset_info['variable_distributions'].get('rect_width',
                                                                     {'type': 'uniform',
                                                                      'params': {'min': 0.05, 'max': 0.2}})
            h_dist = self.dataset_info['variable_distributions'].get('rect_height',
                                                                     {'type': 'uniform',
                                                                      'params': {'min': 0.05, 'max': 0.2}})

            width = self._sample_from_distribution(w_dist['type'], w_dist['params'])
            height = self._sample_from_distribution(h_dist['type'], h_dist['params'])
            return (width, height)

        elif component_type == 'circle':
            # 圆形：半径
            r_dist = self.dataset_info['variable_distributions'].get('circle_radius',
                                                                     {'type': 'uniform',
                                                                      'params': {'min': 0.03, 'max': 0.1}})
            return self._sample_from_distribution(r_dist['type'], r_dist['params'])

        elif component_type == 'capsule':
            # 胶囊型：长度和宽度
            l_dist = self.dataset_info['variable_distributions'].get('capsule_length',
                                                                     {'type': 'uniform',
                                                                      'params': {'min': 0.1, 'max': 0.3}})
            w_dist = self.dataset_info['variable_distributions'].get('capsule_width',
                                                                     {'type': 'uniform',
                                                                      'params': {'min': 0.04, 'max': 0.15}})

            length = self._sample_from_distribution(l_dist['type'], l_dist['params'])
            width = self._sample_from_distribution(w_dist['type'], w_dist['params'])

            # 确保长度大于宽度
            if length <= width:
                length = width + 0.01

            return (length, width)

        return None

    # 辅助函数：从分布采样
    def _sample_from_distribution(self, dist_type: str, params: Dict) -> float:
        if dist_type == 'uniform':
            return random.uniform(params['min'], params['max'])
        elif dist_type == 'gaussian':
            return random.gauss(params['mean'], params['std'])
        return 0.0

    # 辅助函数：计算元件面积
    def _calculate_component_area(self, component_type: str, size: Union[Tuple[float, float], float]) -> float:
        if component_type == 'rect':
            return size[0] * size[1]
        elif component_type == 'circle':
            return math.pi * size * size
        elif component_type == 'capsule':
            l, w = size
            # 面积 = 矩形面积 + 两个半圆面积(合起来是一个圆)
            return (l - w) * w + math.pi * (w / 2) ** 2
        return 0.0

    # 辅助函数：采样元件坐标（GibLS算法用）
    def _sample_component_coord(self, layout: Satellite2DLayout,
                                component: Dict,
                                coord_index: int,
                                current_layout: List[Dict],
                                comp_idx: int) -> float:
        """采样元件的x或y坐标（0表示x，1表示y）"""
        # 保存原始坐标
        original_val = component['coords'][coord_index]
        comp_type = component['type']
        size = component['size']
        layout_size = layout.layout_size[coord_index]

        # 确定坐标范围和步长
        min_val, max_val = self._get_coord_bounds(comp_type, size, coord_index, layout_size)

        # 寻找可行区间
        feasible_intervals = []
        step = (max_val - min_val) / 100  # 离散化步长

        for val in np.arange(min_val, max_val + step, step):
            # 临时更新坐标
            new_coords = list(component['coords'])
            new_coords[coord_index] = val
            component['coords'] = tuple(new_coords)

            # 检查是否与其他元件重叠
            overlap = False
            for i, other_comp in enumerate(current_layout):
                if i == comp_idx:
                    continue
                if layout._is_overlapping_between_two(component, other_comp):
                    overlap = True
                    break

            if not overlap:
                if not feasible_intervals:
                    feasible_intervals.append([val, val])
                else:
                    # 合并连续区间
                    last_interval = feasible_intervals[-1]
                    if val <= last_interval[1] + step * 1.1:  # 考虑浮点误差
                        last_interval[1] = val
                    else:
                        feasible_intervals.append([val, val])

        # 恢复原始坐标
        new_coords = list(component['coords'])
        new_coords[coord_index] = original_val
        component['coords'] = tuple(new_coords)

        # 如果没有可行区间，返回原始值
        if not feasible_intervals:
            return original_val

        # 计算总可行长度
        total_length = sum(end - start for start, end in feasible_intervals)

        # 随机选择一个区间
        rand_val = random.uniform(0, total_length)
        cumulative = 0.0
        selected_start, selected_end = 0.0, 0.0

        for start, end in feasible_intervals:
            interval_len = end - start
            if cumulative + interval_len >= rand_val:
                selected_start, selected_end = start, end
                break
            cumulative += interval_len

        # 在选中的区间内随机采样
        return random.uniform(selected_start, selected_end)

    # 辅助函数：获取坐标边界
    def _get_coord_bounds(self, comp_type: str, size: Union[Tuple[float, float], float],
                          coord_index: int, layout_size: float) -> Tuple[float, float]:
        """获取坐标的有效范围"""
        if comp_type == 'rect':
            # 0:x方向(宽度), 1:y方向(高度)
            dim_size = size[coord_index]
            return (dim_size / 2, layout_size - dim_size / 2)

        elif comp_type == 'circle':
            # 半径
            return (size, layout_size - size)

        elif comp_type == 'capsule':
            length, width = size
            if coord_index == 0:  # x方向
                # 胶囊长度方向
                return (length / 2, layout_size - length / 2)
            else:  # y方向
                # 胶囊宽度方向
                return (width / 2, layout_size - width / 2)

        return (0, layout_size)


# 使用示例
if __name__ == "__main__":
    # 示例1: 创建一个包含胶囊型元件的布局
    layout = Satellite2DLayout(
        layout_size=(1.0, 1.0),  # 1m x 1m的布局
        k=100.0,  # 热导率 100 W/(m·K)
        mesh_resolution=50
    )

    # 添加不同类型的元件
    layout.add_component(component_type='rect', size=(0.2, 0.15), coords=(0.3, 0.3), power=20.0)
    layout.add_component(component_type='circle', size=0.1, coords=(0.7, 0.3), power=15.0)
    layout.add_component(component_type='capsule', size=(0.3, 0.1), coords=(0.5, 0.7), power=25.0)

    # 设置边界条件
    layout.set_boundary_condition('Dirichlet', 300.0, 'all')  # 所有边界温度300K

    # 检查布局有效性
    print(f"布局是否有效: {layout.is_layout_valid()}")

    # 保存为YAML
    layout.to_yaml('example_layout.yaml')

    # 可视化
    layout.visualize('example_layout.png')

    # 示例2: 生成数据集
    dataset_generator = SatelliteLayoutDatasetGenerator(
        base_layout_size=(1.0, 1.0),
        thermal_conductivity=100.0,
        mesh_resolution=50,
        save_root='satellite_layout_dataset'
    )

    # 定义变量分布
    dataset_generator.define_variable_distribution(
        'rect_width', 'uniform', {'min': 0.05, 'max': 0.2}
    )
    dataset_generator.define_variable_distribution(
        'rect_height', 'uniform', {'min': 0.05, 'max': 0.2}
    )
    dataset_generator.define_variable_distribution(
        'circle_radius', 'uniform', {'min': 0.03, 'max': 0.1}
    )
    dataset_generator.define_variable_distribution(
        'capsule_length', 'uniform', {'min': 0.1, 'max': 0.3}
    )
    dataset_generator.define_variable_distribution(
        'capsule_width', 'uniform', {'min': 0.04, 'max': 0.15}
    )

    # 生成数据集（小批量示例）
    dataset_generator.generate_dataset(
        dataset_size=5,  # 5个样本
        num_components_range=(3, 6),  # 每个样本3-6个元件
        component_types=['rect', 'circle', 'capsule'],  # 包含所有类型
        layout_method='SeqLS',  # 使用SeqLS方法
        sdf_resolution=(256, 256)
    )

    # 整合数据集
    dataset_generator.aggregate_dataset()
