# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project creates Tron/synthwave-style 3D "knowledge landscape" visualizations from educational data using Blender's Python API and Cycles ray tracing. Terrain height represents knowledge density, trajectories show learning paths, and landmarks mark quiz questions.

## Commands

### Setup
```bash
./setup.sh  # Installs Blender 4.2, creates conda environment "knowledge-landscape"
```

### Render (primary command)
```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/blender_render.py
```
**Output:** `scene.png` (1200x675, 256 Cycles samples)
**Scene file:** `rendered_scene.blend` (for manual editing in Blender GUI)

### Quick draft preview
Set `DRAFT_MODE = True` in `scripts/blender_render.py` (line ~77) for fast Workbench render.

## Architecture

### Main Script: `scripts/blender_render.py` (~99KB, ~2400 lines)

| Lines | Section |
|-------|---------|
| 31-155 | Configuration constants (scale, colors, render settings) |
| 172-230 | Data loading and heightmap processing |
| 230-400 | Material imports from external .blend files |
| 400-700 | Rectangular prism terrain generation (50x50 grid) |
| 700-900 | Tron wireframe grid overlay |
| 900-1100 | Neon tube trajectories (NURBS curves) |
| 1100-1500 | Glass landmark spheres with acrylic materials |
| 1550-1900 | Mantaflow smoke simulation for tractor beams |
| 1900-2400 | Lighting, camera, render settings |

### Data Files (`data/`)
NumPy `.npy` format:
- `knowledge-heatmap-quiz2.npy` - 100x100 heightmap
- `lecture1-trajectory-shifted.npy`, `lecture2-trajectory-shifted.npy` - path coordinates
- `lecture1-questions-shifted.npy`, `lecture2-questions-shifted.npy`, `general-knowledge-questions-shifted.npy` - landmark positions

### Materials (`materials/`)
All materials are in a flat directory structure:
- **Blend files**: `thin-film.blend`, `acrylic-glass.blend`, `plasma-tube.blend`, `plasma-glow.blend`, `automotive-paint.blend`, `procedural-metals.blend`
- **PBR textures**: `rusty-grate-*`, `metal-plate-*`, `rusty-metal-*`, `blue-planks-*` (each with diff, normal, rough, metal, disp variants)

## Key Configuration Constants

```python
# Scale: 1 inch prism bases
PRISM_BASE_SIZE = 1.0 / 12.0  # feet
SCALE_FACTOR = PRISM_BASE_SIZE  # ~0.0833 feet per heightmap unit

# Render settings
DRAFT_MODE = False  # True for quick Workbench preview
RENDER_SAMPLES = 256  # Cycles samples
RENDER_WIDTH, RENDER_HEIGHT = 1200, 675

# Tractor beams (Mantaflow smoke)
TRACTOR_BEAM_ENABLED = True
TRACTOR_BEAM_VORTICITY = 0.3  # 0.01-0.5 subtle, 1.0+ dramatic
```

## Scene Components

1. **Terrain**: 2500 rectangular prisms with uniform dark blue material
2. **Wireframe**: Glowing Tron-style grid
3. **Roads**: Two NURBS curve tubes (cyan highway, magenta sideroad)
4. **Landmarks**: 36 glass spheres (Lecture1, Lecture2, General) with internal point lights
5. **Tractor Beams**: Mantaflow smoke simulation with pull/push effects based on terrain height
6. **Lighting**: Synthwave area lights and environment

## Tractor Beam Behavior

- **Pull effect** (terrain above 50th percentile): Smoke flows FROM terrain TO sphere
- **Push effect** (low terrain): Smoke flows FROM sphere TO terrain
- Requires smoke domain baking before render (~50 frames)

## Utility Scripts (`scripts/`)

- `render_3d_terrain.py` - PyVista-based alternative (no Blender)
- `update_and_render.py` - Re-render existing .blend file
- `extract_camera_settings.py` - Extract camera params from .blend

## Notes

- Render time: ~20 minutes full quality (256 samples, Cycles, Metal GPU)
- Mantaflow smoke adds significant bake time
- Session notes stored in `notes/` directory
- Camera settings stored in `camera_settings.blend` (minimal file with camera, target, trajectories, landmarks)
- Deprecation warnings about `Material.use_nodes` are harmless (Blender 6.0 migration)
