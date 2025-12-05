#!/usr/bin/env python3
"""
Create a minimal camera_settings.blend file with:
- Camera with settings
- Camera target empty
- Terrain mesh (single mesh with all prism tops as faces)
- Trajectory curves
- Landmark spheres

Uses simple solid color materials for visualization without adding much filesize.
Uses shared geometry module for consistent sizing with blender_render.py.

Run with: /Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/create_camera_settings.py
"""

import bpy
import bmesh
import os
import sys

# Add scripts directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import shared geometry module
from scene_geometry import (
    # Paths
    PROJECT_ROOT, DATA_DIR,
    # Scale settings
    WORLD_SIZE, SCALE_FACTOR, WORLD_OFFSET,
    HEIGHT_SCALE_MIN, HEIGHT_SCALE,
    # Parameters
    SAMPLE_RATE, LANDMARK_RADIUS, LANDMARK_FLOAT_HEIGHT,
    ROAD_WIDTH, SIDEROAD_WIDTH,
    HIGHWAY_HEIGHT_OFFSET, SIDEROAD_HEIGHT_OFFSET,
    RENDER_WIDTH, RENDER_HEIGHT,
    # Colors
    COLOR_TERRAIN, COLOR_HIGHWAY, COLOR_SIDEROAD,
    COLOR_LECTURE1, COLOR_LECTURE2, COLOR_GENERAL,
    # Functions
    load_heightmap, extend_heightmap,
    load_trajectories, load_landmarks,
    get_terrain_height_at_index, get_height_at,
    get_prism_size, get_prism_center,
    get_default_camera_settings,
)

OUTPUT_FILE = os.path.join(DATA_DIR, "camera_settings.blend")

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

# Load data using shared functions
print("Loading data...")
heightmap_norm_original, _, _ = load_heightmap()
heightmap_norm = extend_heightmap(heightmap_norm_original)
highway_coords, sideroad_coords = load_trajectories()
lecture1_landmarks, lecture2_landmarks, general_landmarks = load_landmarks()

print(f"Heightmap shape: {heightmap_norm.shape}")

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

# Create vertices and faces for each prism top (50x50 grid, sampling every SAMPLE_RATE points)
# NOTE: Terrain positioning must match blender_render.py which uses (i / nx) * WORLD_SIZE
nx, ny = heightmap_norm.shape

for ix in range(0, nx, SAMPLE_RATE):
    for iy in range(0, ny, SAMPLE_RATE):
        # World position - MUST match blender_render.py formula: (i / nx) * world_size
        x = (ix / nx) * WORLD_SIZE
        y = (iy / ny) * WORLD_SIZE
        z = get_terrain_height_at_index(ix, iy, heightmap_norm)

        # Prism size matches blender_render.py
        cell_world_size, prism_size, half_size = get_prism_size(SAMPLE_RATE, nx)

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

# Get camera settings (use defaults since we're creating the file)
camera_settings = get_default_camera_settings()

# Create camera
print("Creating camera...")
bpy.ops.object.camera_add(location=camera_settings['camera_location'])
camera = bpy.context.active_object
camera.name = "MainCamera"

# Lens settings
camera.data.lens = camera_settings['lens']
camera.data.sensor_width = camera_settings['sensor_width']
camera.data.clip_start = camera_settings['clip_start']
camera.data.clip_end = camera_settings['clip_end']

# Depth of field settings
camera.data.dof.use_dof = camera_settings['use_dof']
camera.data.dof.focus_distance = camera_settings['focus_distance']
camera.data.dof.aperture_fstop = camera_settings['aperture_fstop']

bpy.context.scene.camera = camera

print(f"  Lens: {camera_settings['lens']}mm")
print(f"  Sensor width: {camera_settings['sensor_width']}mm")
print(f"  DOF enabled: {camera_settings['use_dof']}")
print(f"  Focus distance: {camera_settings['focus_distance']}")
print(f"  Aperture: f/{camera_settings['aperture_fstop']}")

# Create camera target
print("Creating camera target...")
bpy.ops.object.empty_add(type='PLAIN_AXES', location=camera_settings['target_location'])
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
        z = get_height_at(x, y, heightmap_norm) + height_offset
        spline.points[i].co = (x, y, z, 1.0)

    curve_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_obj)
    curve_obj.data.materials.append(material)
    return curve_obj

highway_curve = create_trajectory_curve(highway_coords, "Highway_Trajectory", highway_mat, HIGHWAY_HEIGHT_OFFSET)
sideroad_curve = create_trajectory_curve(sideroad_coords, "Sideroad_Trajectory", sideroad_mat, SIDEROAD_HEIGHT_OFFSET)

# Create landmark spheres
print("Creating landmark spheres...")

def create_landmark_spheres(coords, prefix, material):
    """Create sphere meshes at landmark positions."""
    for i, (x, y) in enumerate(coords):
        z = get_height_at(x, y, heightmap_norm) + LANDMARK_FLOAT_HEIGHT
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
scene.render.resolution_x = RENDER_WIDTH
scene.render.resolution_y = RENDER_HEIGHT

# Set viewport shading to material preview by default
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'

# Save the file
print(f"Saving to: {OUTPUT_FILE}")
bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_FILE)

# Remove .blend1 backup file if it exists
blend1_file = OUTPUT_FILE + "1"
if os.path.exists(blend1_file):
    os.remove(blend1_file)
    print(f"Removed backup file: {blend1_file}")

print("=" * 60)
print("Done! Created camera_settings.blend with terrain mesh")
print("=" * 60)
