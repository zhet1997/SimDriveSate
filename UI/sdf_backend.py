import numpy as np
import math
from interfaces import ComputationBackend


class SDFBackend(ComputationBackend):
    """SDF后端实现"""
    
    def compute(self, scene_data: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
        """
        计算场景的有向距离函数(SDF)
        :param scene_data: 场景数据
        :param grid_shape: 网格形状 (height, width)
        :return: SDF数组
        """
        # 为了简化，我们假设布局大小为(1.0, 1.0)
        layout_size = (1.0, 1.0)
        height, width = grid_shape
        sdf = np.full((height, width), float('inf'))
        
        # 计算每个像素的有向距离
        for y_idx in range(height):
            for x_idx in range(width):
                # 转换像素坐标到实际坐标
                x = (x_idx + 0.5) * (layout_size[0] / width)
                y = (y_idx + 0.5) * (layout_size[1] / height)
                
                # 计算到所有元件的最小距离
                min_dist = float('inf')
                for comp in scene_data:
                    dist = self._distance_to_component((x, y), comp)
                    if dist < min_dist:
                        min_dist = dist
                        
                sdf[y_idx, x_idx] = min_dist
                
        return sdf
    
    def _distance_to_component(self, point: tuple[float, float], component: dict) -> float:
        """计算点到元件的有向距离"""
        x, y = point
        c = component['coords']
        t = component['type']
        
        # 散热器和传感器不参与SDF计算，返回无穷大
        if t in ['radiator', 'sensor']:
            return float('inf')
        
        # 只有物理组件（rect, circle, capsule）参与SDF计算
        s = component['size']
        if t == 'rect':
            return self._distance_to_rect((x, y), c, s)
        elif t == 'circle':
            return self._distance_to_circle((x, y), c, s)
        elif t == 'capsule':
            return self._distance_to_capsule((x, y), c, s)
        return float('inf')
        
    def _distance_to_rect(self, point: tuple[float, float], center: tuple[float, float], 
                          size: tuple[float, float]) -> float:
        px, py = point
        cx, cy = center
        w, h = size
        
        # 计算到矩形边界的距离
        dx = max(px - (cx + w/2), (cx - w/2) - px, 0)
        dy = max(py - (cy + h/2), (cy - h/2) - py, 0)
        
        # 内部点距离为负
        inside = (px >= cx - w/2 and px <= cx + w/2 and 
                  py >= cy - h/2 and py <= cy + h/2)
        
        distance = math.sqrt(dx*dx + dy*dy)
        return -distance if inside else distance
        
    def _distance_to_circle(self, point: tuple[float, float], center: tuple[float, float], 
                            radius: float) -> float:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        distance = math.sqrt(dx*dx + dy*dy)
        return distance - radius  # 内部为负，外部为正
        
    def _distance_to_capsule(self, point: tuple[float, float], center: tuple[float, float], 
                             size: tuple[float, float]) -> float:
        l, w = size
        cx, cy = center
        px, py = point
        
        # 胶囊的矩形部分长度
        rect_length = l - w
        
        # 计算点到胶囊轴线的投影
        # 轴线从(cx - rect_length/2, cy)到(cx + rect_length/2, cy)
        axis_start = (cx - rect_length/2, cy)
        axis_end = (cx + rect_length/2, cy)
        
        # 计算点到线段的距离
        dx = axis_end[0] - axis_start[0]
        dy = axis_end[1] - axis_start[1]
        
        if dx == 0 and dy == 0:  # 线段长度为0
            t = 0.0
        else:
            t = ((px - axis_start[0]) * dx + (py - axis_start[1]) * dy) / (dx*dx + dy*dy)
            t = max(0.0, min(1.0, t))  # 投影到线段上
            
        # 线段上最近点
        nearest_x = axis_start[0] + t * dx
        nearest_y = axis_start[1] + t * dy
        
        # 到轴线的距离
        dist_to_axis = math.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)
        
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
