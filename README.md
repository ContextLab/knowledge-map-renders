# Knowledge Landscape 3D Visualization

A Blender-based 3D visualization tool for rendering "knowledge landscapes" derived from educational data. Creates futuristic Tron/synthwave-style visualizations using Blender's Python API and Cycles ray tracing.

![Knowledge Landscape Render](images/scene.png)

## Overview

This project generates an artistic 3D rendering of Figure 8A (middle panel) from the paper, where:
- **Terrain height** represents estimated knowledge
- **Trajectories** show content of two lectures (Lecture 1 and Lecture 2)
- **Landmarks** show coordinates of questions (used to estimate knowledge)
- **Grid** provides spatial reference with synthwave aesthetics

## Quick Start

### Prerequisites
- macOS (Apple Silicon recommended) or Ubuntu Linux
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda) - optional

### Installation

```bash
# Clone the repository
git clone https://github.com/ContextLab/knowledge-map-renders.git
cd knowledge-map-renders

# Run setup (installs Blender and creates conda environment)
./setup.sh
```

### Rendering

```bash
# Build and render scene (default)
./render.sh

# Render an existing .blend file
./render.sh -i data/rendered_scene.blend

# Custom output paths
./render.sh -o my_render.png -b my_scene.blend
```

**Options:**
| Flag | Description |
|------|-------------|
| `--input`, `-i FILE` | Render existing .blend file (skip scene building) |
| `--output`, `-o FILE` | Custom output PNG path |
| `--blend`, `-b FILE` | Custom output .blend path |
| `--help`, `-h` | Show help message |

**Output files:**
- `images/scene.png` - Rendered image (1200x675)
- `data/rendered_scene.blend` - Blender scene file for manual editing

## Project Structure

```
.
├── data/                    # Input data files
│   ├── knowledge-heatmap-quiz2.npy      # Estimated knowledge at each location (2D projection of text embedding space), 100x100 grid
│   ├── lecture1-trajectory-shifted.npy  # Trajectory through text embedding space (2D projection) of Lecture 1
│   ├── lecture2-trajectory-shifted.npy  # Trajectory through text embedding space (2D projection) of Lecture 2
│   ├── lecture1-questions-shifted.npy   # Text embedding coordinates (2D projection) of Lecture 1 questions
│   ├── lecture2-questions-shifted.npy   # Text embedding coordinates (2D projection) of Lecture 2 questions
│   ├── general-knowledge-questions-shifted.npy  # Text embedding coordinates (2D projection) of general knowledge questions
│   └── camera_settings.blend            # Minimal Blender file for camera adjustment (terrain, trajectories, landmarks)
├── scripts/                 # Python scripts
│   ├── blender_render.py    # Main rendering script (generates full scene)
│   ├── scene_geometry.py    # Shared geometry module (sizing, coordinates, camera loading)
│   ├── create_camera_settings.py     # Generate minimal camera_settings.blend file
│   ├── extract_new_camera.py         # Extract camera params from camera_settings.blend
│   ├── render_from_scene.py # Render from existing .blend file
│   ├── update_and_render.py # Load scene, resize landmarks, re-render
│   └── render_3d_terrain.py # PyVista-based alternative (no Blender required)
├── materials/               # Blender materials and textures
├── images/                  # Screenshots and example outputs
├── render.sh                # Cross-platform render script (macOS + Linux)
├── setup.sh                 # Installation script
├── requirements.txt         # Python dependencies
└── README.md
```

## Configuration

Key parameters in `scripts/blender_render.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DRAFT_MODE` | `False` | Use Workbench for quick preview |
| `RENDER_SAMPLES` | `256` | Cycles samples (quality vs. speed) |
| `RENDER_WIDTH` | `1200` | Output image width |
| `RENDER_HEIGHT` | `675` | Output image height |

### Camera Settings

The camera position and target can be adjusted using the minimal `data/camera_settings.blend` file, which contains the terrain mesh, trajectories, and landmarks with simple materials for fast viewport navigation.

![Blender camera settings scene](images/blender_screenshot.png)

**To adjust the camera:**

1. Open `data/camera_settings.blend` in Blender
2. Select the `MainCamera` object and adjust its position or other properties, and move the `CameraTarget` to change where the camera looks
3. Use **View → Cameras → Active Camera** (or press `Numpad 0`) to preview the camera view
4. Save the file when satisfied with the camera position
5. Run `./render.sh` - camera settings are automatically loaded from the .blend file

Camera settings (location, target, lens, sensor width, DOF) are read directly from `camera_settings.blend` at render time, so changes take effect automatically without editing any Python code.

**To view current camera settings:**

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/extract_new_camera.py
```

**To regenerate `camera_settings.blend` from defaults:**

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/create_camera_settings.py
```

## Citation

If you use this visualization code, please cite the associated paper and main repository:

### Paper

Fitzpatrick, P. C., Heusser, A. C., & Manning, J. R. (In press). Text embedding models yield detailed conceptual knowledge maps derived from short multiple-choice quizzes. *Nature Communications*.

**Preprint:** https://psyarxiv.com/dh3q2

```bibtex
@article{fitzpatrick2025text,
  title={Text embedding models yield detailed conceptual knowledge maps derived from short multiple-choice quizzes},
  author={Fitzpatrick, Paxton C. and Heusser, Andrew C. and Manning, Jeremy R.},
  journal={Nature Communications},
  year={2025},
  note={In press}
}
```

### Main Repository

The main project's code and data are available at:
https://github.com/ContextLab/efficient-learning-khan

```bibtex
@misc{efficient-learning-khan,
  author={Contextual Dynamics Lab},
  title={Efficient Learning Khan},
  year={2025},
  publisher={GitHub},
  url={https://github.com/ContextLab/efficient-learning-khan}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2025 Contextual Dynamics Lab
