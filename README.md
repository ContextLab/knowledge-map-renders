# Knowledge Landscape 3D Visualization

A Blender-based 3D visualization tool for rendering "knowledge landscapes" derived from educational data. Creates futuristic Tron/synthwave-style visualizations using Blender's Python API and Cycles ray tracing.

## Overview

This project generates an artistic 3D rendering of Figure 8A (middle panel) from the paper, where:
- **Terrain height** represents estimated knowledge
- **Trajectories** show content of two lectures (Lecture 1 and Lecture 2)
- **Landmarks** show coordinates of questions (used to estimate knowledge)
- **Grid** provides spatial reference with synthwave aesthetics

## Quick Start

### Prerequisites
- macOS (Apple Silicon recommended)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)

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
# Render the visualization
/Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/blender_render.py
```

**Output files:**
- `scene.png` - Rendered image (675x1200)
- `rendered_scene.blend` - Blender scene file for manual editing

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
│   ├── render_from_scene.py # Render from existing .blend file
│   ├── update_and_render.py # Load scene, resize landmarks, re-render
│   ├── render_3d_terrain.py # PyVista-based alternative (no Blender required)
│   ├── extract_camera_settings.py    # Extract camera params from terrain_scene_edit.blend
│   ├── extract_new_camera.py         # Extract camera params from camera_settings.blend
│   ├── create_camera_settings.py     # Generate minimal camera_settings.blend file
│   ├── extract_scene_settings.py     # Extract camera, target, and landmark settings
│   └── extract_automotive_paint.py   # Inspect material node trees
├── materials/               # Blender materials and textures
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
| `RENDER_WIDTH` | `675` | Output image width |
| `RENDER_HEIGHT` | `1200` | Output image height |

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
  note={In press},
  url={https://psyarxiv.com/dh3q2}
}
```

### Main Repository

The main analysis code and data are available at:
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
