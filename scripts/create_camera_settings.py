#!/usr/bin/env python3
"""
Create a minimal camera_settings.blend file with:
- Camera with settings
- Camera target empty
- Terrain mesh (single mesh with all prism tops as faces)
- Trajectory curves
- Landmark spheres

Uses simple solid color materials for visualization without adding much filesize.

Run with: /Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/create_camera_settings.py
"""

import bpy
import bmesh
import numpy as np
import os
from mathutils import Vector

# Get directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "camera_settings.blend")

# Scale settings (must match blender_render.py)
PRISM_BASE_SIZE = 1.0 / 12.0  # 1 inch in feet
SCALE_FACTOR = PRISM_BASE_SIZE
HEIGHTMAP_SIZE = 100
BASE_WORLD_SIZE = HEIGHTMAP_SIZE * SCALE_FACTOR
WORLD_SIZE = BASE_WORLD_SIZE * 2  # Extended size (matches blender_render.py)
WORLD_OFFSET = BASE_WORLD_SIZE / 2  # Used for coordinate transformation

# Height scale settings
HEIGHT_SCALE_MIN = 1.0 / 12.0
HEIGHT_SCALE_MAX = 50.0 / 12.0
HEIGHT_SCALE = HEIGHT_SCALE_MAX - HEIGHT_SCALE_MIN

# Camera settings (from blender_render.py)
CAMERA_LOCATION = (15.063646, 13.266693, 15.137848)
TARGET_LOCATION = (7.614984, 7.037800, 0.476730)
CAMERA_LENS = 36.0
CAMERA_SENSOR_WIDTH = 23.0
CAMERA_CLIP_START = 0.01  # ~0.12 inches (very close)
CAMERA_CLIP_END = 83.33

# Depth of field settings (from blender_render.py)
DOF_ENABLED = False  # Disabled per user request
DOF_FOCUS_DISTANCE = 10.0
DOF_APERTURE_FSTOP = 0.5

# Landmark settings
LANDMARK_RADIUS = 0.175
LANDMARK_FLOAT_HEIGHT = 1.5

# Colors for simple materials
COLOR_TERRAIN = (0.05, 0.02, 0.15, 1.0)  # Dark blue
COLOR_HIGHWAY = (0.0, 0.9, 1.0, 1.0)      # Cyan
COLOR_SIDEROAD = (1.0, 0.0, 0.8, 1.0)     # Magenta
COLOR_LECTURE1 = (0.8, 0.8, 1.0, 1.0)     # Light blue
COLOR_LECTURE2 = (0.6, 0.6, 0.8, 1.0)     # Gray blue
COLOR_GENERAL = (0.3, 0.3, 0.5, 1.0)      # Dark gray blue

print("=" * 60)
print("Creating camera_settings.blend with terrain mesh")
print("=" * 60)

# Clear scene
print("Clearing scene...")
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

for material in bpy.data.materials:
    bpy.data.materials.remove(material)

for mesh in bpy.data.meshes:
    bpy.data.meshes.remove(mesh)

for curve in bpy.data.curves:
    bpy.data.curves.remove(curve)

# Load data
print("Loading data...")
heightmap = np.load(os.path.join(DATA_DIR, 'knowledge-heatmap-quiz2.npy'))
highway_coords = np.load(os.path.join(DATA_DIR, 'lecture1-trajectory-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
sideroad_coords = np.load(os.path.join(DATA_DIR, 'lecture2-trajectory-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
lecture1_landmarks = np.load(os.path.join(DATA_DIR, 'lecture1-questions-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
lecture2_landmarks = np.load(os.path.join(DATA_DIR, 'lecture2-questions-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
general_landmarks = np.load(os.path.join(DATA_DIR, 'general-knowledge-questions-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET

# Normalize heightmap
h_min, h_max = heightmap.min(), heightmap.max()
heightmap_norm = (heightmap - h_min) / (h_max - h_min)

def get_terrain_height_at_index(ix, iy):
    """Get terrain height at heightmap indices."""
    if 0 <= ix < 100 and 0 <= iy < 100:
        return HEIGHT_SCALE_MIN + heightmap_norm[ix, iy] * HEIGHT_SCALE
    return HEIGHT_SCALE_MIN

def get_terrain_height(x, y):
    """Get interpolated terrain height at world coordinates."""
    hx = (x - WORLD_OFFSET) / SCALE_FACTOR
    hy = (y - WORLD_OFFSET) / SCALE_FACTOR
    hx = max(0, min(99, hx))
    hy = max(0, min(99, hy))
    ix, iy = int(hx), int(hy)
    ix = min(ix, 98)
    iy = min(iy, 98)
    fx, fy = hx - ix, hy - iy
    h00 = heightmap_norm[ix, iy]
    h10 = heightmap_norm[ix + 1, iy]
    h01 = heightmap_norm[ix, iy + 1]
    h11 = heightmap_norm[ix + 1, iy + 1]
    h = (h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) +
         h01 * (1 - fx) * fy + h11 * fx * fy)
    return HEIGHT_SCALE_MIN + h * HEIGHT_SCALE

def create_simple_material(name, color):
    """Create a simple solid color material."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Simple principled BSDF
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = color
    principled.inputs['Roughness'].default_value = 0.8

    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    return mat

# Create materials
print("Creating simple materials...")
terrain_mat = create_simple_material("TerrainMat", COLOR_TERRAIN)
highway_mat = create_simple_material("HighwayMat", COLOR_HIGHWAY)
sideroad_mat = create_simple_material("SideroadMat", COLOR_SIDEROAD)
lecture1_mat = create_simple_material("Lecture1Mat", COLOR_LECTURE1)
lecture2_mat = create_simple_material("Lecture2Mat", COLOR_LECTURE2)
general_mat = create_simple_material("GeneralMat", COLOR_GENERAL)

# Create terrain mesh using bmesh for efficiency
print("Creating terrain mesh (this may take a moment)...")
mesh = bpy.data.meshes.new("TerrainMesh")
bm = bmesh.new()

# Create vertices and faces for each prism top (50x50 grid, sampling every 2nd point)
# NOTE: Terrain positioning must match blender_render.py which uses (i / nx) * WORLD_SIZE
GRID_STEP = 2  # Sample every 2nd point for 50x50 grid
NX, NY = 100, 100  # Heightmap dimensions

for ix in range(0, 100, GRID_STEP):
    for iy in range(0, 100, GRID_STEP):
        # World position - MUST match blender_render.py formula: (i / nx) * world_size
        x = (ix / NX) * WORLD_SIZE
        y = (iy / NY) * WORLD_SIZE
        z = get_terrain_height_at_index(ix, iy)

        # Prism size matches blender_render.py: cell_world_size * sample_rate
        cell_world_size = WORLD_SIZE / NX
        prism_size = cell_world_size * GRID_STEP
        half_size = prism_size / 2.0

        # Center of prism (matches blender_render.py: cx = x + half_size)
        cx = x + half_size
        cy = y + half_size

        v1 = bm.verts.new((cx - half_size, cy - half_size, z))
        v2 = bm.verts.new((cx + half_size, cy - half_size, z))
        v3 = bm.verts.new((cx + half_size, cy + half_size, z))
        v4 = bm.verts.new((cx - half_size, cy + half_size, z))

        # Create face
        bm.faces.new([v1, v2, v3, v4])

bm.to_mesh(mesh)
bm.free()

terrain_obj = bpy.data.objects.new("Terrain", mesh)
bpy.context.collection.objects.link(terrain_obj)
terrain_obj.data.materials.append(terrain_mat)

# Create camera
print("Creating camera...")
bpy.ops.object.camera_add(location=CAMERA_LOCATION)
camera = bpy.context.active_object
camera.name = "MainCamera"

# Lens settings
camera.data.lens = CAMERA_LENS
camera.data.sensor_width = CAMERA_SENSOR_WIDTH
camera.data.clip_start = CAMERA_CLIP_START
camera.data.clip_end = CAMERA_CLIP_END

# Depth of field settings (matching blender_render.py)
camera.data.dof.use_dof = DOF_ENABLED
camera.data.dof.focus_distance = DOF_FOCUS_DISTANCE
camera.data.dof.aperture_fstop = DOF_APERTURE_FSTOP

bpy.context.scene.camera = camera

print(f"  Lens: {CAMERA_LENS}mm")
print(f"  Sensor width: {CAMERA_SENSOR_WIDTH}mm")
print(f"  DOF enabled: {DOF_ENABLED}")
print(f"  Focus distance: {DOF_FOCUS_DISTANCE}")
print(f"  Aperture: f/{DOF_APERTURE_FSTOP}")

# Create camera target
print("Creating camera target...")
bpy.ops.object.empty_add(type='PLAIN_AXES', location=TARGET_LOCATION)
target = bpy.context.active_object
target.name = "CameraTarget"

# Add track-to constraint
bpy.context.view_layer.objects.active = camera
track = camera.constraints.new(type='TRACK_TO')
track.target = target
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

# Create trajectory curves with bevel for visibility
print("Creating trajectory curves...")

def create_trajectory_curve(coords, name, material, height_offset=1.0):
    """Create a curve with bevel from coordinates."""
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.05  # Small bevel for visibility
    curve_data.bevel_resolution = 2

    spline = curve_data.splines.new('POLY')
    spline.points.add(len(coords) - 1)

    for i, (x, y) in enumerate(coords):
        z = get_terrain_height(x, y) + height_offset
        spline.points[i].co = (x, y, z, 1.0)

    curve_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_obj)
    curve_obj.data.materials.append(material)
    return curve_obj

highway_curve = create_trajectory_curve(highway_coords, "Highway_Trajectory", highway_mat)
sideroad_curve = create_trajectory_curve(sideroad_coords, "Sideroad_Trajectory", sideroad_mat)

# Create landmark spheres
print("Creating landmark spheres...")

def create_landmark_spheres(coords, prefix, material):
    """Create sphere meshes at landmark positions."""
    for i, (x, y) in enumerate(coords):
        z = get_terrain_height(x, y) + LANDMARK_FLOAT_HEIGHT
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=LANDMARK_RADIUS,
            segments=12,
            ring_count=8,
            location=(x, y, z)
        )
        sphere = bpy.context.active_object
        sphere.name = f"{prefix}_{i}"
        sphere.data.materials.append(material)

create_landmark_spheres(lecture1_landmarks, "Lecture1", lecture1_mat)
create_landmark_spheres(lecture2_landmarks, "Lecture2", lecture2_mat)
create_landmark_spheres(general_landmarks, "General", general_mat)

# Add basic lighting for viewport
print("Adding basic lighting...")
bpy.ops.object.light_add(type='SUN', location=(10, 10, 20))
sun = bpy.context.active_object
sun.name = "Sun"
sun.data.energy = 2.0

# Set render settings
print("Setting render settings...")
scene = bpy.context.scene
scene.render.resolution_x = 675
scene.render.resolution_y = 1200

# Set viewport shading to material preview by default
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'

# Save the file
print(f"Saving to: {OUTPUT_FILE}")
bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_FILE)

print("=" * 60)
print("Done! Created camera_settings.blend with terrain mesh")
print("=" * 60)
