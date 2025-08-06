# UI Framework Guide - 卫星组件可视化UI框架指南

## 📋 概述

本UI框架是基于PyQt6开发的卫星组件可视化与物理场预测系统。系统采用模块化设计，支持拖拽绘制、实时SDF计算和可视化等功能。

### 🎯 主要功能
- **拖拽绘制**：支持矩形、圆形、胶囊形组件的交互式绘制
- **SDF可视化**：实时计算并显示Signed Distance Function背景
- **组件管理**：可视化组件列表，支持功率参数实时编辑
- **文件操作**：YAML格式的布局文件加载/保存
- **多线程计算**：后台SDF计算，保持UI响应性

## 🏗️ 核心架构

### 模块组织结构
```
UI/
├── main_entry.py          # 应用程序入口
├── main_window.py         # 主窗口逻辑
├── graphics_items.py      # 图形组件（RectItem, CircleItem, CapsuleItem）
├── graphics_scene.py      # 自定义场景（拖拽绘制逻辑）
├── sidebar_panel.py       # 侧边栏面板组件
├── worker_thread.py       # 后台计算线程
├── ui_constants.py        # UI常量和样式定义
├── ui_utils.py           # 工具函数
├── interfaces.py         # 计算后端接口
└── sdf_backend.py        # SDF计算实现
```

### 🔧 核心组件

#### **1. MainWindow（main_window.py）**
- **职责**：主窗口管理、工具栏、文件操作、线程协调
- **关键方法**：
  - `toggle_draw_mode(mode)` - 切换绘制模式
  - `update_sdf_background()` - 手动更新SDF
  - `load_from_yaml()` / `save_to_yaml()` - 文件操作
  - `on_computation_complete()` - 处理后台计算结果

#### **2. CustomGraphicsScene（graphics_scene.py）**
- **职责**：拖拽绘制逻辑、鼠标事件处理、组件创建
- **关键方法**：
  - `mousePressEvent()` / `mouseMoveEvent()` / `mouseReleaseEvent()` - 鼠标事件
  - `create_preview_item()` - 创建预览图形
  - `create_permanent_item()` - 创建永久组件

#### **3. Graphics Items（graphics_items.py）**
- **职责**：可绘制的图形组件实现
- **组件类型**：
  - `RectItem` - 矩形组件（蓝色）
  - `CircleItem` - 圆形组件（红色）
  - `CapsuleItem` - 胶囊组件（绿色）

#### **4. SidebarPanel（sidebar_panel.py）**
- **职责**：侧边栏UI管理、组件列表、SDF控制
- **功能区域**：
  - 绘制模式控制（Rectangle, Circle, Capsule按钮）
  - SDF可视化控制（显示开关、更新按钮）
  - 组件列表（功率编辑、位置/尺寸显示）

#### **5. Worker Thread（worker_thread.py）**
- **职责**：后台SDF计算，避免UI阻塞
- **信号**：`computation_complete` - 计算完成信号

## 🔄 数据流架构

### 组件创建流程
```
1. 用户选择绘制模式 → SidebarPanel.toggle_draw_mode()
2. 用户拖拽绘制 → CustomGraphicsScene.mouse*Event()
3. 创建预览图形 → create_preview_item()
4. 释放鼠标完成绘制 → create_permanent_item()
5. 更新组件列表 → SidebarPanel.update_components_list()
```

### SDF计算流程
```
1. 组件变化 → scene_updated信号
2. 收集组件数据 → collect_and_emit_scene_data()
3. 转换坐标单位 → ui_utils.convert_component_to_meters()
4. 后台计算 → Worker.compute()
5. 计算完成 → computation_complete信号
6. 更新背景图像 → update_sdf_background_image()
```

### 坐标系统
- **场景坐标**：像素单位，500像素/米
- **物理坐标**：米单位，用于SDF计算
- **转换工具**：`ui_utils.py`中的转换函数

## 🎨 绘制系统详解

### 拖拽绘制机制
1. **模式选择**：通过侧边栏按钮选择绘制模式（rect/circle/capsule）
2. **预览阶段**：拖拽时显示虚线预览图形
3. **确认创建**：释放鼠标后创建永久组件
4. **边界检查**：确保绘制区域在画布范围内

### 圆形绘制对齐修复
**问题**：预览圆形与最终圆形位置不一致
**解决方案**：使用`create_circle_rect_from_drag()`确保预览和最终位置都以拖拽矩形中心为圆心

```python
# 圆形预览对齐逻辑
def create_circle_rect_from_drag(start_point, current_point):
    rect = QRectF(start_point, current_point).normalized()
    size = min(rect.width(), rect.height())
    center_x = rect.center().x()
    center_y = rect.center().y()
    radius = size / 2
    return QRectF(center_x - radius, center_y - radius, size, size)
```

## 🔍 SDF可视化原理

### SDF背景显示系统
1. **网格匹配**：SDF分辨率与Qt网格对齐（50×50像素）
2. **坐标转换**：组件坐标从像素转换为米单位进行计算
3. **图像生成**：使用matplotlib生成透明背景的SDF图像
4. **场景集成**：将SDF图像作为`QGraphicsPixmapItem`添加到场景底层

### 网格对齐机制
```python
# SDF网格分辨率计算
grid_resolution = int(layout_size[0] / 0.1) * 5  # 10 * 5 = 50
# Qt网格：10×10单元，每单元50×50像素
# SDF网格：50×50像素，每Qt单元5×5 SDF像素
```

## 🎛️ 用户交互指南

### 绘制模式控制
- **选择模式**：点击侧边栏按钮（Rectangle/Circle/Capsule）
- **取消模式**：再次点击已选中按钮进入选择模式
- **拖拽绘制**：在画布上拖拽创建组件
- **移动组件**：直接拖拽已创建的组件

### SDF可视化控制
- **显示开关**：侧边栏"Show SDF Background"复选框
- **手动更新**：勾选显示后出现"Update SDF"按钮
- **自动更新**：组件变化时自动触发计算

### 组件属性编辑
- **功率调节**：侧边栏组件列表中的SpinBox
- **实时更新**：功率修改立即反映在组件显示上
- **位置/尺寸**：只读显示，单位为米

## 🔧 扩展开发指南

### 添加新组件类型
1. **继承BaseComponentItem**：在`graphics_items.py`中创建新组件类
2. **实现绘制方法**：重写`boundingRect()`和`paint()`方法
3. **更新工厂函数**：在`create_component_item()`中添加新类型
4. **添加预览支持**：在`CustomGraphicsScene.create_preview_item()`中添加预览逻辑
5. **更新UI常量**：在`ui_constants.py`中添加颜色、图标、名称定义

### 添加新的SDF计算后端
1. **实现ComputationBackend接口**：继承`interfaces.py`中的抽象类
2. **注册后端**：在`Worker`类中设置新后端
3. **配置参数**：在`ui_constants.py`中添加特定配置

### 自定义UI样式
- **修改样式表**：更新`ui_constants.StyleSheets`中的CSS定义
- **调整颜色主题**：修改`ui_constants.Colors`类
- **更新图标**：替换`ui_constants.Icons`中的emoji图标

## 🐛 常见问题排查

### 组件显示问题
- **组件不可见**：检查`scene_scale`设置和坐标转换
- **位置偏移**：验证`itemChange()`中的坐标更新逻辑
- **字体显示**：确认字体设置和系统字体支持

### SDF显示问题
- **图像模糊**：检查matplotlib的`interpolation='nearest'`设置
- **尺寸不匹配**：验证`extent`参数与场景尺寸一致
- **坐标系不对齐**：确认`origin='upper'`设置正确

### 性能问题
- **UI阻塞**：确保耗时计算在Worker线程中执行
- **内存泄漏**：检查matplotlib图形的`plt.close(fig)`调用
- **频繁更新**：优化信号连接，避免重复计算

## 📚 API参考

### 核心类接口

#### MainWindow
```python
class MainWindow(QMainWindow):
    def toggle_draw_mode(self, mode: str)  # 切换绘制模式
    def set_draw_mode(self, mode)          # 设置绘制模式
    def update_sdf_background(self)        # 更新SDF背景
    def update_component_power(self, item, power)  # 更新组件功率
```

#### CustomGraphicsScene
```python
class CustomGraphicsScene(QGraphicsScene):
    def set_draw_mode(self, mode: Optional[str])  # 设置绘制模式
    def create_permanent_item(self, rect: QRectF)  # 创建永久组件
    # 信号
    item_position_changed = pyqtSignal()
    scene_updated = pyqtSignal(list)
```

#### 工具函数
```python
# 坐标转换
def pixels_to_meters(pixels: float) -> float
def meters_to_pixels(meters: float) -> float
def convert_coords_to_meters(coords: Tuple) -> Tuple
def convert_coords_to_pixels(coords: Tuple) -> Tuple

# 组件工具
def create_component_item(component_state: Dict) -> BaseComponentItem
def generate_random_power() -> float
def create_matplotlib_figure(sdf_array, width, height) -> QPixmap
```

## 🚀 未来扩展建议

### 功能增强
- **多场景支持**：支持多个SDF场景切换显示
- **组件库**：预定义组件模板和快速插入
- **撤销/重做**：操作历史管理
- **网格吸附**：组件对齐到网格点
- **测量工具**：距离和角度测量

### 性能优化
- **视口裁剪**：大场景时的视口优化渲染
- **增量更新**：部分区域SDF计算
- **缓存机制**：计算结果缓存和复用
- **LOD渲染**：多级细节渲染

### 用户体验
- **快捷键支持**：键盘快捷操作
- **拖拽导入**：文件拖拽导入支持
- **实时预览**：参数调整的实时预览
- **导出功能**：图像和数据导出

---

*该文档提供了UI框架的完整技术指南，包含架构设计、使用方法和扩展指导。如需更新，请同步修改代码实现。*