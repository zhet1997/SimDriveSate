卫星元件可视化与物理场预测程序 - 技术设计文档

1. 项目概述
1.1 项目目标
开发一个桌面应用程序，允许用户在二维画布上交互式地布局卫星元件（矩形、圆形、胶囊形）。程序能根据当前布局，调用可插拔的后端计算模块（初始为SDF，未来可扩展至神经网络物理场预测），并实时/准实时地将计算结果可视化。要和本目录下的layout保持一致，可以调用其中的一些模块来实现；

【注意，本技术文档只是实现建议，可以基于实际情况自由发挥创造】

1.2 核心功能
图形化界面: 提供一个主窗口，包含元件库、交互式画布和属性编辑器。

元件交互: 支持从库中拖拽添加新元件到画布，在画布上拖动、选中和删除元件。

文件交互: 支持从 YAML 文件加载布局方案，以及将当前画布上的布局导出为 YAML 文件。

可插拔后端: 设计灵活的后端接口，允许计算逻辑从SDF（有向距离场）无缝切换到神经网络模型。

解耦架构: UI响应与后端耗时计算完全分离，确保在计算过程中界面保持流畅。

结果可视化: 将后端计算出的二维数组（如SDF值或物理场强度）以图像形式渲染在画布背景上。

1.3 重要: 与现有逻辑层的一致性
本项目的UI和后端实现，必须在数据结构和文件格式上与已有的 Satellite2DLayout Python类保持严格一致。特别是，由UI生成并传递给后端的场景数据，以及导入/导出的YAML文件格式，都必须遵循 Satellite2DLayout 类中定义的标准。

2. 系统架构
本程序采用基于多线程的模型-视图-控制器 (MVC) 变体架构，将UI线程与工作线程分离，以实现UI与计算的解耦。

UI线程 (主线程):

职责: 管理所有窗口、控件的创建与更新；处理用户输入（鼠标、键盘）；在画布上绘制元件的控制器（用户可见的形状）。

技术栈: PyQt6, PyYAML (用于文件读写)

工作线程 (后台线程):

职责: 接收UI线程传递的场景数据，执行所有耗时的计算任务（SDF, NN推理等）。

技术栈: NumPy, 【暂时不具体实现神经网络部分，只留出接口】

通信机制:

UI线程与工作线程之间通过PyQt的**信号与槽 (Signals & Slots)**机制进行异步通信。

3. UI模块详细设计 (PyQt6)
UI模块基于PyQt6的图形视图框架 (Graphics View Framework) 构建。

3.1 核心组件
QGraphicsView: 视图窗口，作为画布的容器。

QGraphicsScene: 逻辑场景，管理所有图形元件(QGraphicsItem)的集合。

QMainWindow: 程序主窗口，容纳QGraphicsView、工具栏和侧边栏。

3.2 自定义图形元件 (QGraphicsItem)
所有画布上的元件都继承自QGraphicsItem。

通用属性与行为:

每个自定义Item在构造时必须设置标志位:

self.setFlag(QGraphicsItem.ItemIsMovable)

self.setFlag(QGraphicsItem.ItemIsSelectable)

self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

必须重写 boundingRect() 和 paint() 方法。

提供一个 get_state() 方法，返回一个字典，其结构必须与 Satellite2DLayout 中 components 列表的元素格式一致。

具体元件定义:

RectItem(QGraphicsItem):

状态属性: id, type='rect', coords, size (宽度, 高度), power。

绘制: 使用 painter.drawRect()。

CircleItem(QGraphicsItem):

状态属性: id, type='circle', coords, size (半径), power。

绘制: 使用 painter.drawEllipse()。

CapsuleItem(QGraphicsItem):

状态属性: id, type='capsule', coords, size (总长度, 宽度), power。

绘制: 通过组合一个矩形和两个半圆来实现。

3.3 主窗口布局
工具栏: 包含“添加矩形”、“添加圆形”、“添加胶囊”、“删除选中项”等按钮。新增: “从YAML加载”和“保存到YAML”按钮。

主画布 (QGraphicsView): 占据中心区域，用于显示和交互。

侧边栏/属性编辑器: (可选，V2功能) 当选中一个元件时，显示其详细参数（如power）并允许修改。未来可扩展至边界条件 (boundary_conditions) 的设置。

3.4 用户交互流程
添加/移动/删除元件: 流程不变。

触发计算: 流程不变。

(新增) 从YAML加载:

用户点击“从YAML加载”按钮。

程序弹出文件选择对话框。

使用 PyYAML 库解析用户选择的YAML文件。

程序清空当前 QGraphicsScene。

根据解析出的 components 列表，循环创建对应的 RectItem, CircleItem, CapsuleItem 实例并添加到场景中。

（可选）更新界面上的布局信息（如 layout_size, k 等）。

(新增) 保存到YAML:

用户点击“保存到YAML”按钮。

程序弹出文件保存对话框。

程序遍历场景中所有Item，调用 get_state() 收集元件数据。

程序构建一个与 Satellite2DLayout.to_yaml() 方法输出格式完全一致的字典（包含 layout_info, components, boundary_conditions）。

使用 PyYAML 库将该字典写入用户指定的路径。

4. 后端计算模块详细设计
4.1 抽象后端接口 (ComputationBackend)
接口定义保持不变，以提供灵活性。

# interfaces.py
from abc import ABC, abstractmethod
import numpy as np

class ComputationBackend(ABC):
    """定义所有计算后端的通用接口"""
    @abstractmethod
    def compute(self, scene_data: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
        """
        根据场景数据，在指定形状的网格上进行计算。
        :param scene_data: 描述场景中所有元件的字典列表。
        :param grid_shape: 计算网格的形状 (height, width)。
        :return: 一个形状为 (height, width) 的浮点型NumPy数组。
        """
        pass

4.2 数据结构定义
输入: scene_data (list[dict]):

这是从UI传递给后端的场景描述，其格式必须与Satellite2DLayout类中的self.components列表一致。

示例:

[
    {"id": 0, "type": "rect", "size": [0.2, 0.15], "coords": [0.3, 0.3], "power": 20.0, "center": [0.3, 0.3]},
    {"id": 1, "type": "circle", "size": 0.1, "coords": [0.7, 0.3], "power": 15.0, "center": [0.7, 0.3]},
    {"id": 2, "type": "capsule", "size": [0.3, 0.1], "coords": [0.5, 0.7], "power": 25.0, "center": [0.5, 0.7]}
]

输出: result_image (np.ndarray):

定义不变，一个二维NumPy数组。

4.3 具体后端实现
后端实现职责不变，但在compute方法内部处理scene_data时，需使用更新后的键名（如coords, power）。

SDFBackend(ComputationBackend):

实现: 逻辑不变，但解析scene_data中的字典时，使用 item['coords'] 和 item['size']。

NNBackend(ComputationBackend):

实现: 逻辑不变，但在预处理scene_data为模型输入时，需确保使用正确的键名，并包含power等新信息。

5. 通信协议 (信号与槽)
通信协议的结构和定义保持不变，传递的数据负载（scene_data和result_image）的类型不变，但scene_data的内容已根据4.2节更新。

6. 数据流
数据流的整体逻辑不变，无论是移动元件还是导入/导出文件，最终都是通过scene_updated信号触发后端计算，并通过computation_complete信号更新UI显示。