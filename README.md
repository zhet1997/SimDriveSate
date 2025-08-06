# 2D Satellite Component Layout Visualization

A PyQt6-based application for visualizing and designing 2D satellite component layouts with physics field prediction capabilities.

## 🚀 Features

- **Interactive Component Drawing**: Drag-and-drop creation of rectangular, circular, and capsule-shaped components
- **SDF Visualization**: Real-time Signed Distance Function computation and background visualization
- **Component Management**: Visual component list with editable power parameters
- **File Operations**: Save and load layouts in YAML format
- **Multi-threaded Computing**: Background SDF calculation to maintain UI responsiveness
- **Modular Architecture**: Clean, maintainable code structure with separated concerns

## 📋 Requirements

- Python 3.8+
- PyQt6 6.4.0+
- NumPy 1.21.0+
- SciPy 1.7.0+
- Matplotlib 3.5.0+
- PyYAML 6.0+

## 🛠️ Installation

### Option 1: Using pip
```bash
# Clone the repository
git clone <repository-url>
cd 2d-satellite-layout

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
# Create conda environment
conda create -n satellite-layout python=3.9
conda activate satellite-layout

# Install dependencies
pip install -r requirements.txt
```

### Development Installation
```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

## 🎮 Usage

### Starting the Application
```bash
cd UI
python main_entry.py
```

### Basic Operations

1. **Drawing Components**:
   - Select a drawing mode from the sidebar (Rectangle, Circle, Capsule)
   - Drag on the canvas to create components
   - Click the same button again to deselect and enter selection mode

2. **SDF Visualization**:
   - Check "Show SDF Background" in the sidebar
   - Click "Update SDF" to compute and display the field
   - SDF updates automatically when components change

3. **Component Management**:
   - View all components in the sidebar list
   - Edit power values directly in the list
   - Delete selected components using the toolbar button

4. **File Operations**:
   - Use "Load YAML" to open existing layouts
   - Use "Save YAML" to save current layout

## 🏗️ Architecture

### Module Structure
```
UI/
├── main_entry.py          # Application entry point
├── main_window.py         # Main window and coordination logic
├── graphics_items.py      # Component rendering (RectItem, CircleItem, CapsuleItem)
├── graphics_scene.py      # Scene management and drawing logic
├── sidebar_panel.py       # UI controls and component list
├── worker_thread.py       # Background computation thread
├── ui_constants.py        # Constants, styles, and configuration
├── ui_utils.py           # Utility functions
├── interfaces.py         # Computation backend interfaces
└── sdf_backend.py        # SDF computation implementation
```

### Key Components

- **MainWindow**: Application coordination, file operations, SDF management
- **CustomGraphicsScene**: Drag-to-draw functionality, mouse event handling
- **Graphics Items**: Renderable components with selection and movement
- **SidebarPanel**: UI controls for drawing modes, SDF, and component editing
- **Worker Thread**: Asynchronous SDF computation

## 🔧 Development

### Code Quality Tools
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy UI/

# Run tests
pytest
```

### Adding New Component Types
1. Create new class in `graphics_items.py` inheriting from `BaseComponentItem`
2. Add component type to `create_component_item()` factory function
3. Update `ui_constants.py` with colors, icons, and names
4. Add preview logic in `graphics_scene.py`

### Adding New SDF Backends
1. Implement `ComputationBackend` interface from `interfaces.py`
2. Register backend in `worker_thread.py`
3. Add configuration options in `ui_constants.py`

## 📚 Documentation

Detailed technical documentation is available in:
- `UI/UI_Framework_Guide.md` - Complete framework guide
- Code comments and docstrings throughout the codebase

## 🐛 Troubleshooting

### Common Issues

1. **Application crashes on component creation**:
   - Ensure PyQt6 is properly installed
   - Check that all dependencies meet minimum version requirements

2. **SDF visualization not appearing**:
   - Verify matplotlib backend compatibility
   - Check that components exist before updating SDF

3. **Font display issues**:
   - System font settings may affect text rendering
   - Default fallback fonts are configured in `ui_constants.py`

### Performance Issues
- Large numbers of components may affect rendering performance
- SDF computation time scales with grid resolution
- Background computation prevents UI blocking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the existing code style
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

- Multi-field visualization support
- Component libraries and templates
- Advanced physics simulation integration
- 3D visualization capabilities
- Export functionality for various formats
- Undo/redo operations
- Grid snapping and measurement tools

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on the repository
- Check the documentation in `UI_Framework_Guide.md`
- Review existing issues and discussions

---

*Built with PyQt6 and modern Python development practices for reliable, maintainable satellite component layout design.*