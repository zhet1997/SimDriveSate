from SeqLS import SeqLS
from GibLS import GibLS_Sampler
import os
import random
import numpy as np
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import math
from typing import List, Tuple, Dict, Optional, Union, Callable


class SatelliteLayoutGenerator:
    """卫星二维布局数据生成类，支持批量生成样本"""

    def __init__(self,
                 design_area: Tuple[float, float],
                 grid_size: Tuple[int, int],
                 distributions: Dict[str, Dict] = None,
                 output_dir: str = "dataset"):
        self.design_area = design_area
        self.grid_size = grid_size
        self.grid_width = design_area[0] / grid_size[1]
        self.grid_height = design_area[1] / grid_size[0]

        # 合并自定义分布与默认分布
        default_dist = self._get_default_distributions()
        self.distributions = {**default_dist, **(distributions or {})}

        self.components = []
        self.results = {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _get_default_distributions(self) -> Dict[str, Dict]:
        return {
            "rect_width": {"type": "uniform", "params": [0.05, 0.2]},
            "rect_height": {"type": "uniform", "params": [0.05, 0.2]},
            "circle_radius": {"type": "uniform", "params": [0.03, 0.1]},
            "capsule_length": {"type": "uniform", "params": [0.1, 0.3]},
            "capsule_width": {"type": "uniform", "params": [0.04, 0.15]},
            "power": {"type": "uniform", "params": [5.0, 50.0]}
        }

    # 新增：批量生成样本的核心方法
    def generate_batch(self,
                       num_samples: int,
                       sampler_name: str,
                       min_components: int = 3,
                       max_components: int = 8,
                       component_types: List[str] = None,
                       **sampler_kwargs) -> None:
        """
        批量生成多个布局样本

        参数:
            num_samples: 要生成的样本总数
            sampler_name: 采样方法 ("GibLS" 或 "SeqLS")
            min_components: 每个样本的最小元件数量
            max_components: 每个样本的最大元件数量
            component_types: 允许的元件类型列表
            sampler_kwargs: 采样器的额外参数
        """
        component_types = component_types or ["rect", "circle", "capsule"]
        success_count = 0
        failed_count = 0

        print(f"开始批量生成 {num_samples} 个样本（{sampler_name}方法）...")

        # 创建批量样本的根目录
        batch_dir = os.path.join(self.output_dir, sampler_name, "batch")
        os.makedirs(batch_dir, exist_ok=True)

        # 记录批量生成的元数据
        metadata = {
            "design_area": self.design_area,
            "grid_size": self.grid_size,
            "sampler": sampler_name,
            "total_samples": num_samples,
            "success_samples": 0,
            "failed_samples": 0,
            "samples": []  # 记录每个样本的基本信息
        }

        # 循环生成样本，直到达到指定数量
        while success_count < num_samples:
            sample_id = success_count + failed_count

            # 随机生成元件数量（在min和max之间）
            num_components = random.randint(min_components, max_components)

            # 生成随机元件
            self.generate_random_components(num_components, component_types)

            # 生成布局
            print(f"\n生成样本 {sample_id + 1}/{num_samples}（元件数量: {num_components}）")
            success = self.generate_layout(sampler_name, **sampler_kwargs)

            if success:
                # 导出当前样本
                self.export_layout(sample_id=sample_id)
                success_count += 1

                # 记录样本信息
                metadata["samples"].append({
                    "sample_id": sample_id,
                    "num_components": num_components,
                    "components": [{"id": c["id"], "type": c["type"]} for c in self.components]
                })
                print(f"样本 {sample_id} 生成成功（累计成功: {success_count}）")
            else:
                failed_count += 1
                print(f"样本 {sample_id} 生成失败（累计失败: {failed_count}）")

        # 更新元数据并保存
        metadata["success_samples"] = success_count
        metadata["failed_samples"] = failed_count
        with open(os.path.join(batch_dir, "batch_metadata.yaml"), "w") as f:
            yaml.dump(metadata, f)

        print(f"\n批量生成完成！成功: {success_count}/{num_samples}, 失败: {failed_count}")
        print(f"样本保存至: {os.path.abspath(self.output_dir)}")

    # 原有方法保持不变（仅展示关键方法）
    def generate_random_components(self,
                                   num_components: int,
                                   types: List[str] = None) -> bool:
        types = types or ["rect", "circle", "capsule"]
        self.components = []
        for i in range(num_components):
            comp_type = random.choice(types)
            size = self._generate_size(comp_type)
            if size is None:
                print(f"无法生成{comp_type}的尺寸，跳过")
                continue
            self.components.append({
                "id": i,
                "type": comp_type,
                "size": size,
                "power": self._sample_from_distribution("power"),
                "coords": (0, 0)
            })
        return True

    def generate_layout(self,
                        sampler_name: str, **sampler_kwargs) -> bool:
        if not self.components:
            print("请先加载或生成元件列表")
            return False

        if sampler_name not in ["GibLS", "SeqLS"]:
            print(f"不支持的采样方法: {sampler_name}")
            return False

        try:
            if sampler_name == "SeqLS":
                sampler = SeqLS(self.design_area, self.grid_size)
                success, placed_components = sampler.layout_sampling(
                    components=self.components,
                    max_attempts=sampler_kwargs.get("max_attempts", 100)
                )
            else:
                sampler = GibLS_Sampler(self.design_area, self.grid_size)
                success, placed_components = sampler.sample(
                    components=self.components,
                    burn_in=sampler_kwargs.get("burn_in", 50),
                    iterations=sampler_kwargs.get("iterations", 100),
                    max_attempts=sampler_kwargs.get("max_attempts", 100)
                )

            if success:
                self.results[sampler_name] = {
                    "components": placed_components,
                    "grid_matrix": sampler.get_grid_matrix()
                }
                self.components = placed_components
                return True
            return False
        except Exception as e:
            print(f"生成布局出错: {str(e)}")
            return False

    def export_layout(self,
                      sample_id: int,
                      sdf_resolution: Tuple[int, int] = (256, 256)) -> None:
        for sampler_name, result in self.results.items():
            method_dir = os.path.join(self.output_dir, sampler_name, f"sample_{sample_id:06d}")
            os.makedirs(method_dir, exist_ok=True)

            # 导出YAML配置
            yaml_data = {
                "design_area": self.design_area,
                "grid_size": self.grid_size,
                "sampler": sampler_name,
                "components": result["components"]
            }
            with open(os.path.join(method_dir, "layout.yaml"), "w") as f:
                yaml.dump(self._convert_tuples_to_lists(yaml_data), f)

            # 导出网格矩阵
            with open(os.path.join(method_dir, "grid_matrix.txt"), "w") as f:
                for row in result["grid_matrix"]:
                    f.write(" ".join(map(str, row)) + "\n")

            # 导出SDF
            sdf = self._generate_sdf(sdf_resolution)
            np.save(os.path.join(method_dir, "sdf.npy"), sdf)

            # 导出可视化图像
            self._visualize(os.path.join(method_dir, "layout_visualization.png"))

    # 其他辅助方法（省略，与之前版本一致）
    def _generate_size(self, comp_type: str) -> Union[Tuple[float, float], float, None]:
        try:
            if comp_type == "rect":
                return (self._sample_from_distribution("rect_width"),
                        self._sample_from_distribution("rect_height"))
            elif comp_type == "circle":
                return self._sample_from_distribution("circle_radius")
            elif comp_type == "capsule":
                width = self._sample_from_distribution("capsule_width")
                length = max(self._sample_from_distribution("capsule_length"), width + 0.01)
                return (length, width)
            return None
        except KeyError as e:
            print(f"分布参数缺失: {e}")
            return None

    def _sample_from_distribution(self, param_name: str) -> float:
        dist = self.distributions.get(param_name)
        if not dist:
            raise KeyError(f"未定义分布: {param_name}")
        if dist["type"] == "uniform":
            return random.uniform(*dist["params"])
        elif dist["type"] == "gaussian":
            return random.gauss(*dist["params"])
        raise ValueError(f"不支持的分布类型: {dist['type']}")

    def _convert_tuples_to_lists(self, data):
        if isinstance(data, tuple):
            return [self._convert_tuples_to_lists(item) for item in data]
        elif isinstance(data, list):
            return [self._convert_tuples_to_lists(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._convert_tuples_to_lists(v) for k, v in data.items()}
        return data

    def _visualize(self, save_path: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.design_area[0])
        ax.set_ylim(0, self.design_area[1])
        ax.set_aspect("equal")

        colors = {"rect": "blue", "circle": "red", "capsule": "green"}
        for comp in self.components:
            c = comp["coords"]
            s = comp["size"]
            t = comp["type"]
            if t == "rect":
                w, h = s
                ax.add_patch(plt.Rectangle((c[0] - w / 2, c[1] - h / 2), w, h,
                                           fill=True, alpha=0.5, color=colors[t]))
            elif t == "circle":
                ax.add_patch(plt.Circle(c, s, fill=True, alpha=0.5, color=colors[t]))
            elif t == "capsule":
                l, w = s
                rect_len = l - w
                ax.add_patch(plt.Rectangle((c[0] - rect_len / 2, c[1] - w / 2), rect_len, w,
                                           fill=True, alpha=0.5, color=colors[t]))
                ax.add_patch(plt.Circle((c[0] - rect_len / 2, c[1]), w / 2,
                                        fill=True, alpha=0.5, color=colors[t]))
                ax.add_patch(plt.Circle((c[0] + rect_len / 2, c[1]), w / 2,
                                        fill=True, alpha=0.5, color=colors[t]))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    # 其他方法（重叠检查、SDF生成等）保持不变


# 批量生成示例
if __name__ == "__main__":
    # 初始化生成器
    generator = SatelliteLayoutGenerator(
        design_area=(1.0, 1.0),  # 1m×1m设计区域
        grid_size=(50, 50),  # 50×50网格
        distributions={  # 自定义部分分布参数
            "rect_width": {"type": "uniform", "params": [0.08, 0.15]},
            "rect_height": {"type": "uniform", "params": [0.08, 0.15]},
            "power": {"type": "gaussian", "params": [20.0, 5.0]}
        },
        output_dir="satellite_batch_dataset"  # 批量样本输出目录
    )

    # 批量生成SeqLS样本
    generator.generate_batch(
        num_samples=100,  # 生成100个样本
        sampler_name="SeqLS",  # 使用SeqLS方法
        min_components=3,  # 每个样本至少3个元件
        max_components=7,  # 每个样本最多7个元件
        component_types=["rect", "circle", "capsule"],  # 允许的元件类型
        max_attempts=200  # SeqLS采样参数
    )

    # 批量生成GibLS样本（可选）
    generator.generate_batch(
        num_samples=50,  # 生成50个样本
        sampler_name="GibLS",  # 使用GibLS方法
        min_components=4,
        max_components=9,
        component_types=["rect", "capsule"],  # 只生成矩形和胶囊型
        burn_in=30,  # GibLS采样参数
        iterations=100,
        max_attempts=150
    )
