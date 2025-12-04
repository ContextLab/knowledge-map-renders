#!/usr/bin/env python3
"""
Blender Python script for futuristic Tron/synthwave 3D terrain rendering
Run with: /Applications/Blender.app/Contents/MacOS/Blender --background --python blender_render.py

Features:
- Cycles ray tracing renderer with photorealistic rendering
- Futuristic synthwave aesthetic with neon colors
- Rectangular prisms for heightmap cells with thin-film interference material
- Glass tubes for road trajectories
- Metallic spheres with different materials per landmark class
- Dramatic lighting for Tron-like atmosphere
"""

import bpy
import bmesh
import numpy as np
import os
import math
import sys
import argparse
import time
from mathutils import Vector, Matrix


# ============================================================================
# ARGUMENT PARSING
# ============================================================================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Blender Knowledge Landscape Renderer')
    parser.add_argument('--force-rebuild', action='store_true',
                        help='Force rebuild scene even if cache exists')
    # Blender passes extra args after --, so we need to handle them
    args_list = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return parser.parse_args(args_list)


# ============================================================================
# SCENE CACHING UTILITIES
# ============================================================================
def is_scene_cache_valid(blend_path, script_path, data_files):
    """
    Check if cached .blend file is newer than script and data files.

    Args:
        blend_path: Path to the rendered_scene.blend file
        script_path: Path to this script
        data_files: List of data file paths to check

    Returns:
        True if cache is valid, False otherwise
    """
    if not os.path.exists(blend_path):
        print(f"Cache check: {blend_path} does not exist")
        return False

    blend_mtime = os.path.getmtime(blend_path)

    # Check script modification time
    if os.path.exists(script_path):
        script_mtime = os.path.getmtime(script_path)
        if script_mtime > blend_mtime:
            print(f"Cache check: Script modified more recently than cache")
            return False

    # Check data files
    for data_file in data_files:
        if os.path.exists(data_file):
            data_mtime = os.path.getmtime(data_file)
            if data_mtime > blend_mtime:
                print(f"Cache check: Data file {os.path.basename(data_file)} modified more recently")
                return False

    print("Cache check: Cache is valid!")
    return True


# Get the directory of this script and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # scripts/ is inside project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.chdir(PROJECT_ROOT)

print(f"Script directory: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")

# ============================================================================
# CONFIGURATION - Futuristic Tron/Synthwave Style
# ============================================================================
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "scene.png")
MATERIALS_DIR = os.path.join(PROJECT_ROOT, "materials")
THIN_FILM_BLEND = os.path.join(MATERIALS_DIR, "thin-film.blend")
AUTOMOTIVE_PAINT_BLEND = os.path.join(MATERIALS_DIR, "automotive-paint.blend")
ACRYLIC_GLASS_BLEND = os.path.join(MATERIALS_DIR, "acrylic-glass.blend")
PLASMA_TUBE_BLEND = os.path.join(MATERIALS_DIR, "plasma-tube.blend")
PLASMA_GLOW_BLEND = os.path.join(MATERIALS_DIR, "plasma-glow.blend")
METAL_MATERIALS_BLEND = os.path.join(MATERIALS_DIR, "procedural-metals.blend")

# PBR Metal Texture sets for landmarks (bump-mapped for interesting appearance)
METAL_TEXTURE_SETS = {
    'rusty_grate': {
        'diffuse': os.path.join(MATERIALS_DIR, "rusty-grate-diff.jpg"),
        'normal': os.path.join(MATERIALS_DIR, "rusty-grate-normal.exr"),
        'roughness': os.path.join(MATERIALS_DIR, "rusty-grate-rough.exr"),
        'metallic': os.path.join(MATERIALS_DIR, "rusty-grate-metal.exr"),
        'displacement': os.path.join(MATERIALS_DIR, "rusty-grate-disp.png"),
    },
    'metal_plate': {
        'diffuse': os.path.join(MATERIALS_DIR, "metal-plate-diff.jpg"),
        'normal': os.path.join(MATERIALS_DIR, "metal-plate-normal.exr"),
        'roughness': os.path.join(MATERIALS_DIR, "metal-plate-rough.exr"),
        'metallic': os.path.join(MATERIALS_DIR, "metal-plate-metal.exr"),
        'displacement': os.path.join(MATERIALS_DIR, "metal-plate-disp.png"),
    },
    'rusty_metal': {
        'diffuse': os.path.join(MATERIALS_DIR, "rusty-metal-diff.jpg"),
        'normal': os.path.join(MATERIALS_DIR, "rusty-metal-normal.exr"),
        'roughness': os.path.join(MATERIALS_DIR, "rusty-metal-rough.exr"),
        'metallic': os.path.join(MATERIALS_DIR, "rusty-metal-metal.exr"),
        'displacement': os.path.join(MATERIALS_DIR, "rusty-metal-disp.png"),
    },
    'blue_planks': {
        'diffuse': os.path.join(MATERIALS_DIR, "blue-planks-diff.jpg"),
        'normal': os.path.join(MATERIALS_DIR, "blue-planks-normal.exr"),
        'roughness': os.path.join(MATERIALS_DIR, "blue-planks-rough.jpg"),
        'metallic': None,  # No metallic map for painted wood
        'displacement': os.path.join(MATERIALS_DIR, "blue-planks-disp.png"),
    },
}

# Render settings - DRAFT MODE for quick preview
DRAFT_MODE = False  # Set to False for final high-quality render

# Full resolution for both modes (user requested same canvas size)
RENDER_WIDTH = 675
RENDER_HEIGHT = 1200

if DRAFT_MODE:
    RENDER_SAMPLES = 1  # Not used for Workbench
    RENDER_ENGINE = 'BLENDER_WORKBENCH'  # Workbench for fastest preview
else:
    RENDER_SAMPLES = 256  # Good balance of quality and speed for Cycles
    RENDER_ENGINE = 'CYCLES'  # Ray tracing for final render
USE_DENOISING = True

# ============================================================================
# SCALE CALCULATIONS - 1 inch prism bases
# ============================================================================
# Target: each prism has a 1-inch square base
PRISM_BASE_SIZE = 1.0 / 12.0  # 1 inch in feet (~0.0833 feet)

# Original scale for reference (used to calculate scale factor)
FEET_PER_ACRE = 43560
ACRES = 300
ORIGINAL_AREA_SQFT = ACRES * FEET_PER_ACRE
ORIGINAL_SIDE_LENGTH = np.sqrt(ORIGINAL_AREA_SQFT)  # ~3615 feet
HEIGHTMAP_SIZE = 100
ORIGINAL_SCALE_FACTOR = ORIGINAL_SIDE_LENGTH / HEIGHTMAP_SIZE  # ~36.15 feet per heightmap unit

# Scale reduction factor: ratio of original to new prism size
SCALE_REDUCTION = ORIGINAL_SCALE_FACTOR / PRISM_BASE_SIZE  # ~434x reduction

# New scale factor (1 heightmap unit = 1 inch = 1 prism)
SCALE_FACTOR = PRISM_BASE_SIZE  # ~0.0833 feet per heightmap unit

BASE_WORLD_SIZE = HEIGHTMAP_SIZE * SCALE_FACTOR  # ~8.33 feet
WORLD_SIZE = BASE_WORLD_SIZE * 2  # ~16.67 feet (extended)
WORLD_OFFSET = BASE_WORLD_SIZE / 2

# Scale all other dimensions proportionally
# Prism heights: 1 to 50 inches (reduced by 50% for flatter terrain)
# In feet: 1 inch = 1/12 ft, 50 inches = 50/12 ft = 4.17 ft
HEIGHT_SCALE_MIN = 1.0 / 12.0  # 1 inch minimum height in feet
HEIGHT_SCALE_MAX = 50.0 / 12.0  # 50 inches maximum height in feet (~4.17 ft) - reduced by 50%
HEIGHT_SCALE = HEIGHT_SCALE_MAX - HEIGHT_SCALE_MIN  # Range of 49 inches

# Road parameters (for glass tubes) - 2 inches wide (10x original)
# 2 inches = 2/12 feet = 0.167 feet
ROAD_WIDTH = 2.0 / 12.0  # 2 inches in feet (~0.167 ft)
SIDEROAD_WIDTH = 2.0 / 12.0  # Same width for sideroad
ROAD_INTERPOLATION_POINTS = 5  # Fewer points, we'll use Bezier smoothing

# Sphere landmarks - from General_12_6 in terrain_scene_edit.blend (2025-12-03)
# General_12_6: radius = 0.35 ft (4.2 inches), z_center = 5.17 ft
LANDMARK_RADIUS = 0.175  # 2.1 inches in feet (50% of original 0.35)
# Float height = distance from terrain top to sphere center
# General_12_6 z_center is ~5.17 ft, terrain height varies, so this is relative to terrain
# Reduced by 50% per user request (was 5.0 ft)
LANDMARK_FLOAT_HEIGHT = 1.5  # 1.5 feet above terrain to sphere center

# Trajectory height offsets - tubes float 12 inches above the local prism height
# This is added to the terrain height at each point (not an absolute height)
HIGHWAY_HEIGHT_OFFSET = 12.0 / 12.0  # 12 inches above local prism top (1 foot)
SIDEROAD_HEIGHT_OFFSET = 12.0 / 12.0  # Same for sideroad

# Rectangular prism grid settings
PRISM_CELL_SIZE = PRISM_BASE_SIZE  # 1 inch square base
PRISM_GAP = 0.0  # No gap between prisms - tight grid
PRISM_BEVEL_RADIUS = 0.05 / SCALE_REDUCTION  # Scale bevel too

# ============================================================================
# SYNTHWAVE COLOR PALETTE
# ============================================================================
SYNTHWAVE_CYAN = (0.0, 0.9, 1.0)
SYNTHWAVE_MAGENTA = (1.0, 0.0, 0.8)
SYNTHWAVE_PURPLE = (0.6, 0.0, 1.0)
SYNTHWAVE_PINK = (1.0, 0.2, 0.6)
SYNTHWAVE_ORANGE = (1.0, 0.4, 0.0)
SYNTHWAVE_DARK_BLUE = (0.05, 0.02, 0.15)

# ============================================================================
# SCENE CACHING - Load or Build
# ============================================================================

# Parse command-line arguments
args = parse_args()

# Define paths
BLEND_OUTPUT = os.path.join(PROJECT_ROOT, "rendered_scene.blend")
SCRIPT_PATH = os.path.abspath(__file__) if '__file__' in dir() else os.path.join(SCRIPT_DIR, 'blender_render.py')

# List all data files to check for modifications
DATA_FILES = [
    os.path.join(DATA_DIR, "knowledge-heatmap-quiz2.npy"),
    os.path.join(DATA_DIR, "lecture1-trajectory-shifted.npy"),
    os.path.join(DATA_DIR, "lecture2-trajectory-shifted.npy"),
    os.path.join(DATA_DIR, "lecture1-questions-shifted.npy"),
    os.path.join(DATA_DIR, "lecture2-questions-shifted.npy"),
    os.path.join(DATA_DIR, "general-knowledge-questions-shifted.npy"),
]

# Check if we should use cached scene
use_cache = not args.force_rebuild and is_scene_cache_valid(BLEND_OUTPUT, SCRIPT_PATH, DATA_FILES)

if use_cache:
    print("=" * 80)
    print("LOADING CACHED SCENE AND RENDERING")
    print("=" * 80)
    print(f"Loading pre-built scene from: {BLEND_OUTPUT}")
    print("(Use --force-rebuild to rebuild from scratch)")
    bpy.ops.wm.open_mainfile(filepath=BLEND_OUTPUT)
    print("Scene loaded successfully!")

    # Render directly and exit
    print("Starting render from cached scene...")
    bpy.ops.render.render(write_still=True)
    print(f"Render complete! Saved to: {OUTPUT_FILE}")

    # Exit early - no need to run rest of script
    import sys
    sys.exit(0)

# If we get here, we're building the scene from scratch
print("=" * 80)
print("BUILDING SCENE FROM SCRATCH")
print("=" * 80)
if args.force_rebuild:
    print("Reason: --force-rebuild flag specified")
else:
    print("Reason: Cache is invalid or does not exist")

# ============================================================================
# CLEAR SCENE
# ============================================================================
print("Clearing scene...")
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

for material in bpy.data.materials:
    bpy.data.materials.remove(material)

for mesh in bpy.data.meshes:
    bpy.data.meshes.remove(mesh)

for ng in bpy.data.node_groups:
    bpy.data.node_groups.remove(ng)

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading numpy data...")
heightmap_original = np.load(os.path.join(DATA_DIR, 'knowledge-heatmap-quiz2.npy'))

# Scale coordinates from heightmap units (0-100) to world units (feet)
highway_coords = np.load(os.path.join(DATA_DIR, 'lecture1-trajectory-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
sideroad_coords = np.load(os.path.join(DATA_DIR, 'lecture2-trajectory-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
lecture1_landmarks = np.load(os.path.join(DATA_DIR, 'lecture1-questions-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
lecture2_landmarks = np.load(os.path.join(DATA_DIR, 'lecture2-questions-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
general_landmarks = np.load(os.path.join(DATA_DIR, 'general-knowledge-questions-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET

# Normalize original heightmap
h_min, h_max = heightmap_original.min(), heightmap_original.max()
heightmap_norm_original = (heightmap_original - h_min) / (h_max - h_min)

# Extend heightmap
print("Extending heightmap...")
nx_orig, ny_orig = heightmap_norm_original.shape
nx_ext, ny_ext = nx_orig * 2, ny_orig * 2
heightmap_norm = np.zeros((nx_ext, ny_ext))

x_start, y_start = nx_orig // 2, ny_orig // 2
heightmap_norm[x_start:x_start+nx_orig, y_start:y_start+ny_orig] = heightmap_norm_original

avg_height = heightmap_norm_original.mean()

for i in range(nx_ext):
    for j in range(ny_ext):
        orig_i = i - x_start
        orig_j = j - y_start

        if 0 <= orig_i < nx_orig and 0 <= orig_j < ny_orig:
            continue

        clamped_i = max(0, min(nx_orig - 1, orig_i))
        clamped_j = max(0, min(ny_orig - 1, orig_j))
        base_val = heightmap_norm_original[clamped_i, clamped_j]

        dist_i = abs(orig_i - clamped_i)
        dist_j = abs(orig_j - clamped_j)
        total_dist = np.sqrt(dist_i**2 + dist_j**2)

        fade_factor = np.exp(-total_dist / (nx_orig * 0.5))
        heightmap_norm[i, j] = base_val * fade_factor + avg_height * (1 - fade_factor)

def simple_blur(arr, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    padded = np.pad(arr, kernel_size // 2, mode='edge')
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i, j] = np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel)
    return result

heightmap_norm = simple_blur(heightmap_norm, kernel_size=5)

print(f"Heightmap extended: {heightmap_norm.shape}")

# ============================================================================
# IMPORT THIN-FILM INTERFERENCE MATERIALS
# ============================================================================
print("Importing thin-film interference materials...")

def import_material_from_blend(blend_path, material_name):
    if not os.path.exists(blend_path):
        print(f"Warning: {blend_path} not found")
        return None
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        if material_name in data_from.materials:
            data_to.materials = [material_name]
        else:
            print(f"Warning: Material '{material_name}' not found in {blend_path}")
            return None
    if data_to.materials and data_to.materials[0]:
        print(f"  Imported material: {material_name}")
        return data_to.materials[0]
    return None

def import_all_node_groups_from_blend(blend_path):
    if not os.path.exists(blend_path):
        return
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        data_to.node_groups = data_from.node_groups
    print(f"  Imported {len(data_to.node_groups)} node groups")

# Import node groups first (needed for thin-film materials)
import_all_node_groups_from_blend(THIN_FILM_BLEND)

# Import materials for different elements
dichroic_mat = import_material_from_blend(THIN_FILM_BLEND, "dichroic_glass")
soap_bubble_mat = import_material_from_blend(THIN_FILM_BLEND, "soap_bubble")
copper_mat = import_material_from_blend(THIN_FILM_BLEND, "copper_nk")
gold_mat = import_material_from_blend(THIN_FILM_BLEND, "gold_nk")
silver_mat = import_material_from_blend(THIN_FILM_BLEND, "silver_nk")
chromium_mat = import_material_from_blend(THIN_FILM_BLEND, "chromium_nk")
car_paint_01 = import_material_from_blend(THIN_FILM_BLEND, "car paint 01")
car_paint_05 = import_material_from_blend(THIN_FILM_BLEND, "car paint 05")

# Import Automotive Paint Shader node group
print("Importing automotive paint shader...")
import_all_node_groups_from_blend(AUTOMOTIVE_PAINT_BLEND)
automotive_paint_mat = import_material_from_blend(AUTOMOTIVE_PAINT_BLEND, "Automotive Paint Shader")

# Import Acrylic Glass material
print("Importing acrylic glass material...")
acrylic_glass_mat = import_material_from_blend(ACRYLIC_GLASS_BLEND, "Material")

# Import Plasma Tube material (for trajectory tubes) - OLD VERSION
print("Importing plasma tube material...")
plasma_tube_base_mat = import_material_from_blend(PLASMA_TUBE_BLEND, "PlasmaTubeMaterial")

# Import Plasma Glow node group (for growing volumetric plasma effect)
print("Importing plasma glow node group...")
import_all_node_groups_from_blend(PLASMA_GLOW_BLEND)

# Import metal materials for landmarks
print("Importing metal materials...")
import_all_node_groups_from_blend(METAL_MATERIALS_BLEND)
chrome_mat = import_material_from_blend(METAL_MATERIALS_BLEND, "Chrome")
copper_mat = import_material_from_blend(METAL_MATERIALS_BLEND, "Copper")
gold_mat = import_material_from_blend(METAL_MATERIALS_BLEND, "Gold")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_height_at(x, y, heightmap_norm, height_scale, world_size):
    """Bilinear interpolation of height at x, y coordinates.
    Returns HEIGHT_SCALE_MIN + normalized_height * HEIGHT_SCALE (1-5 inches range)."""
    nx, ny = heightmap_norm.shape
    xi = (x / world_size) * (nx - 1)
    yi = (y / world_size) * (ny - 1)

    xi = max(0, min(nx - 1.001, xi))
    yi = max(0, min(ny - 1.001, yi))

    x0, y0 = int(np.floor(xi)), int(np.floor(yi))
    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)

    xf, yf = xi - x0, yi - y0

    z = (heightmap_norm[x0, y0] * (1 - xf) * (1 - yf) +
         heightmap_norm[x1, y0] * xf * (1 - yf) +
         heightmap_norm[x0, y1] * (1 - xf) * yf +
         heightmap_norm[x1, y1] * xf * yf)

    # Return min height + scaled range (1-5 inches)
    return HEIGHT_SCALE_MIN + z * height_scale

def synthwave_colormap(t):
    """Return a synthwave color based on normalized height t (0-1).
    Creates a purple -> blue -> cyan -> magenta gradient."""
    # Clamp t to 0-1
    t = max(0.0, min(1.0, t))

    # Synthwave gradient: dark purple -> deep blue -> cyan -> magenta -> pink
    if t < 0.25:
        # Dark purple to deep blue
        s = t / 0.25
        return (0.15 + 0.05 * s, 0.0 + 0.1 * s, 0.3 + 0.4 * s)
    elif t < 0.5:
        # Deep blue to cyan
        s = (t - 0.25) / 0.25
        return (0.2 - 0.2 * s, 0.1 + 0.7 * s, 0.7 + 0.3 * s)
    elif t < 0.75:
        # Cyan to magenta
        s = (t - 0.5) / 0.25
        return (0.0 + 0.9 * s, 0.8 - 0.4 * s, 1.0 - 0.2 * s)
    else:
        # Magenta to hot pink
        s = (t - 0.75) / 0.25
        return (0.9 + 0.1 * s, 0.4 - 0.2 * s, 0.8 + 0.2 * s)

def create_height_colored_material(name, height_value):
    """Create an automotive paint material colored by height using synthwave colormap.
    Uses the Automotive Paint Shader node group for realistic car paint appearance."""
    color = synthwave_colormap(height_value)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Check if the Automotive Paint Shader node group exists
    auto_paint_group = bpy.data.node_groups.get("Automotive Paint Shader")

    if auto_paint_group:
        # Use the Automotive Paint Shader node group
        group_node = nodes.new('ShaderNodeGroup')
        group_node.node_tree = auto_paint_group
        group_node.location = (0, 0)

        # Set the colors using synthwave colormap
        # Base Coat Color - main body color
        group_node.inputs['Base Coat Color'].default_value = (*color, 1)
        group_node.inputs['Base Coat Roughness'].default_value = 0.5
        group_node.inputs['Base Coat IOR'].default_value = 1.5

        # Metallic flakes - white flakes for sparkle effect
        group_node.inputs['Enable Flakes'].default_value = True
        group_node.inputs['Metallic Flakes Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # White flakes
        group_node.inputs['Metallic Flakes Density'].default_value = 0.5
        group_node.inputs['Metallic Flakes Roughness'].default_value = 0.2
        group_node.inputs['Metallic Flakes Scale'].default_value = 10000.0  # Keep at 10000
        group_node.inputs['Metallic Flakes Normal Randomization'].default_value = 1.0

        # Clear coat settings
        group_node.inputs['Coat Weight'].default_value = 1.0
        group_node.inputs['Coat Roughness'].default_value = 0.02
        group_node.inputs['Coat IOR'].default_value = 1.6
        # Subtle tint on clear coat
        group_node.inputs['Coat Tint'].default_value = (*color, 1)

        # Sheen settings
        group_node.inputs['Sheen Weight'].default_value = 0.0
        group_node.inputs['Sheen Roughness'].default_value = 0.5
        group_node.inputs['Sheen Tint'].default_value = (*color, 1)

        # Debug: Print first material's settings
        if name == 'height_mat_0':
            print(f"  DEBUG Auto Paint Material '{name}':")
            print(f"    Base Coat Roughness: {group_node.inputs['Base Coat Roughness'].default_value}")
            print(f"    Enable Flakes: {group_node.inputs['Enable Flakes'].default_value}")
            print(f"    Flakes Scale: {group_node.inputs['Metallic Flakes Scale'].default_value}")
            print(f"    Flakes Color: {group_node.inputs['Metallic Flakes Color'].default_value[:]}")

        links.new(group_node.outputs['Shader'], output.inputs['Surface'])
    else:
        # Fallback to Principled BSDF if node group not found
        print(f"  Warning: Automotive Paint Shader node group not found, using fallback")
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        principled.inputs['Base Color'].default_value = (*color, 1)
        principled.inputs['Roughness'].default_value = 0.15
        principled.inputs['Metallic'].default_value = 0.9
        principled.inputs['Coat Weight'].default_value = 1.0
        principled.inputs['Coat Roughness'].default_value = 0.02
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    return mat

def create_synthwave_emission_material(name, color, emission_strength=5.0, roughness=0.1):
    """Create a glowing synthwave material"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)
    principled.inputs['Base Color'].default_value = (*color, 1)
    principled.inputs['Roughness'].default_value = roughness
    principled.inputs['Metallic'].default_value = 0.8
    principled.inputs['Emission Color'].default_value = (*color, 1)
    principled.inputs['Emission Strength'].default_value = emission_strength

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    return mat

def create_glass_tube_material(name, color, emission_strength=2.0):
    """Create glowing glass tube material for roads"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # Glass BSDF with tint
    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.location = (0, 100)
    glass.inputs['Color'].default_value = (*color, 1)
    glass.inputs['Roughness'].default_value = 0.05
    glass.inputs['IOR'].default_value = 1.45

    # Emission for inner glow
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (0, -100)
    emission.inputs['Color'].default_value = (*color, 1)
    emission.inputs['Strength'].default_value = emission_strength

    # Mix glass and emission
    add_shader = nodes.new('ShaderNodeAddShader')
    add_shader.location = (300, 0)

    links.new(glass.outputs['BSDF'], add_shader.inputs[0])
    links.new(emission.outputs['Emission'], add_shader.inputs[1])
    links.new(add_shader.outputs['Shader'], output.inputs['Surface'])

    mat.blend_method = 'BLEND'
    return mat

def create_plasma_tube_material(name, glow_color, texture_scale_multiplier=100.0):
    """Create a plasma tube material based on the imported PlasmaTubeMaterial.

    Copies the base plasma material and sets the emission/glass colors to the
    specified synthwave color for glowing plasma tube effect.

    Args:
        name: Material name
        glow_color: RGB tuple for the plasma glow color (e.g., SYNTHWAVE_CYAN)
        texture_scale_multiplier: Multiplier for procedural texture scales
            The original material was designed for small objects (~1 meter).
            Our tubes are at world scale (~7000 feet), so we need to multiply
            the texture scales by a large factor to get proper detail.
    """
    # Copy the base plasma material
    base_mat = bpy.data.materials.get("PlasmaTubeMaterial")
    if base_mat is None:
        print(f"  Warning: PlasmaTubeMaterial not found, falling back to glass tube")
        return create_glass_tube_material(name, glow_color, emission_strength=3.0)

    mat = base_mat.copy()
    mat.name = name

    # Find and update the nodes in the material
    if mat.use_nodes:
        nodes = mat.node_tree.nodes

        for node in nodes:
            # Update Emission nodes with the synthwave color
            if node.type == 'EMISSION':
                node.inputs['Color'].default_value = (*glow_color, 1)
            elif node.type == 'BSDF_GLASS':
                # Glass color - use a slightly darker version for depth
                glass_color = tuple(c * 0.3 for c in glow_color)
                node.inputs['Color'].default_value = (*glass_color, 1)
            # Scale up procedural textures for world-scale geometry
            elif node.type == 'TEX_VORONOI':
                # Multiply Voronoi scale to get proper detail at world scale
                if 'Scale' in node.inputs:
                    original_scale = node.inputs['Scale'].default_value
                    node.inputs['Scale'].default_value = original_scale * texture_scale_multiplier
                    print(f"    Voronoi scale: {original_scale} -> {node.inputs['Scale'].default_value}")
            elif node.type == 'TEX_NOISE':
                # Multiply Noise scale to get proper detail at world scale
                if 'Scale' in node.inputs:
                    original_scale = node.inputs['Scale'].default_value
                    node.inputs['Scale'].default_value = original_scale * texture_scale_multiplier
                    print(f"    Noise scale: {original_scale} -> {node.inputs['Scale'].default_value}")
            elif node.type == 'MAPPING':
                # Scale the mapping node's scale vector to tile textures properly
                if 'Scale' in node.inputs:
                    original_scale = node.inputs['Scale'].default_value[:]
                    new_scale = (original_scale[0] * texture_scale_multiplier,
                                original_scale[1] * texture_scale_multiplier,
                                original_scale[2] * texture_scale_multiplier)
                    node.inputs['Scale'].default_value = new_scale
                    print(f"    Mapping scale: {original_scale} -> {new_scale}")

    mat.blend_method = 'BLEND'
    return mat

def create_growing_plasma_material(name, glow_color, emission_strength=20.0):  # Reduced from 30.0
    """Create a growing plasma material using the Plasma Glow node group.

    Uses the faked volumetric glowing plasma shader which looks better than
    the previous plasma tube material. Sets the RGB power values to achieve
    the desired glow color while keeping the shell color unchanged.

    Args:
        name: Material name
        glow_color: RGB tuple for the plasma glow color (e.g., SYNTHWAVE_CYAN)
        emission_strength: Strength of the glow emission (default 8.0)

    The Plasma Glow node group uses RGB Power values to control glow color:
    - Red Power, Green_Power, Blue Power control the glow color
    - Higher values = brighter that color component
    - Shell Colour controls the outer shell (leave as default per user request)
    """
    # Check if Plasma Glow node group exists
    plasma_glow_group = bpy.data.node_groups.get("Plasma Glow")
    if plasma_glow_group is None:
        print(f"  Warning: Plasma Glow node group not found, falling back to glass tube")
        return create_glass_tube_material(name, glow_color, emission_strength=3.0)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create output node
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Create the Plasma Glow node group
    group_node = nodes.new('ShaderNodeGroup')
    group_node.node_tree = plasma_glow_group
    group_node.location = (0, 0)

    # Set the RGB power values based on the glow_color
    # The original default is Red=16, Green=8, Blue=4
    # We scale these based on the glow_color components
    # For synthwave cyan (0.0, 0.9, 1.0): want low red, high green, high blue
    # For synthwave magenta (1.0, 0.0, 0.8): want high red, low green, high blue

    base_power = 32.0  # Base power level for full intensity (increased for brighter glow)
    r, g, b = glow_color

    # Set RGB power values - scale by color component and base power
    group_node.inputs['Red Power'].default_value = r * base_power
    group_node.inputs['Green_Power'].default_value = g * base_power
    group_node.inputs['Blue Power'].default_value = b * base_power

    # Set emission strength
    group_node.inputs['Emission Strength'].default_value = emission_strength

    # Set density (affects how the plasma looks)
    group_node.inputs['Density multiplier '].default_value = 0.5

    # Leave Shell Colour as default gray (per user request)
    # group_node.inputs['Shell Colour'].default_value = (0.5, 0.5, 0.5, 1.0)

    print(f"  Created plasma glow material: {name}")
    print(f"    R={r*base_power:.1f}, G={g*base_power:.1f}, B={b*base_power:.1f}, Emission={emission_strength}")

    # Connect group to output
    links.new(group_node.outputs[0], output.inputs['Surface'])

    mat.blend_method = 'BLEND'
    return mat

def create_uniform_dark_blue_material(name):
    """Create a uniform dark blue automotive paint material for all prisms.
    Uses the same Automotive Paint Shader node group as the height-based materials,
    but with a fixed dark blue color instead of height-based coloring."""
    dark_blue = (0.02, 0.05, 0.15)  # Very dark blue with slight hint of color

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Check if the Automotive Paint Shader node group exists
    auto_paint_group = bpy.data.node_groups.get("Automotive Paint Shader")

    if auto_paint_group:
        # Use the Automotive Paint Shader node group
        group_node = nodes.new('ShaderNodeGroup')
        group_node.node_tree = auto_paint_group
        group_node.location = (0, 0)

        # Set uniform dark blue color
        group_node.inputs['Base Coat Color'].default_value = (*dark_blue, 1)
        group_node.inputs['Base Coat Roughness'].default_value = 0.5
        group_node.inputs['Base Coat IOR'].default_value = 1.5

        # Metallic flakes - white flakes for sparkle effect
        # Using scale from example material in automotive-paint-shader.blend (50000.0)
        group_node.inputs['Enable Flakes'].default_value = True
        group_node.inputs['Metallic Flakes Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # White flakes
        group_node.inputs['Metallic Flakes Density'].default_value = 0.5
        group_node.inputs['Metallic Flakes Roughness'].default_value = 0.2
        group_node.inputs['Metallic Flakes Scale'].default_value = 50000.0  # Updated to match example material
        group_node.inputs['Metallic Flakes Normal Randomization'].default_value = 1.0

        # Clear coat settings
        group_node.inputs['Coat Weight'].default_value = 1.0
        group_node.inputs['Coat Roughness'].default_value = 0.02
        group_node.inputs['Coat IOR'].default_value = 1.6
        # Subtle tint on clear coat
        group_node.inputs['Coat Tint'].default_value = (*dark_blue, 1)

        # Sheen settings
        group_node.inputs['Sheen Weight'].default_value = 0.0
        group_node.inputs['Sheen Roughness'].default_value = 0.5
        group_node.inputs['Sheen Tint'].default_value = (*dark_blue, 1)

        links.new(group_node.outputs['Shader'], output.inputs['Surface'])
    else:
        # Fallback to Principled BSDF if node group not found
        print(f"  Warning: Automotive Paint Shader node group not found, using fallback")
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        principled.inputs['Base Color'].default_value = (*dark_blue, 1)
        principled.inputs['Roughness'].default_value = 0.15
        principled.inputs['Metallic'].default_value = 0.9
        principled.inputs['Coat Weight'].default_value = 1.0
        principled.inputs['Coat Roughness'].default_value = 0.02
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    return mat

def create_neon_tube_material(name, glow_color, emission_strength=20.0):
    """Create a realistic neon tube material - glass tube with glowing gas inside.

    Based on web research (blenderartists.org/t/more-advanced-neon-material-for-cycles):
    Realistic neon uses Glass BSDF for outer tube surface + Volume Emission for
    the inner gas glow. This creates the authentic look where light appears to
    emanate from within the tube.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # SURFACE: Glass BSDF for the outer tube - slightly tinted with glow color
    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.location = (0, 100)
    # Subtle tint from glow color
    glass_tint = tuple(0.85 + c * 0.15 for c in glow_color)
    glass.inputs['Color'].default_value = (*glass_tint, 1)
    glass.inputs['Roughness'].default_value = 0.0  # Crystal clear glass
    glass.inputs['IOR'].default_value = 1.5  # Standard glass IOR

    # Add subtle surface emission to make edges glow
    surface_emission = nodes.new('ShaderNodeEmission')
    surface_emission.location = (0, -50)
    surface_emission.inputs['Color'].default_value = (*glow_color, 1)
    surface_emission.inputs['Strength'].default_value = emission_strength * 0.3  # Subtle surface glow

    # Mix glass and emission for surface
    add_surface = nodes.new('ShaderNodeAddShader')
    add_surface.location = (300, 50)
    links.new(glass.outputs['BSDF'], add_surface.inputs[0])
    links.new(surface_emission.outputs['Emission'], add_surface.inputs[1])

    links.new(add_surface.outputs['Shader'], output.inputs['Surface'])

    # VOLUME: Emission shader for the glowing gas inside the tube
    volume_emission = nodes.new('ShaderNodeEmission')
    volume_emission.location = (0, -200)
    volume_emission.inputs['Color'].default_value = (*glow_color, 1)
    volume_emission.inputs['Strength'].default_value = emission_strength

    links.new(volume_emission.outputs['Emission'], output.inputs['Volume'])

    mat.blend_method = 'BLEND'
    return mat

def create_tron_wireframe_material(name, edge_color, emission_strength=3.0):
    """Create a glowing wireframe edge material for Tron-style edge highlighting.
    Uses Wireframe node to only render edges with glow effect."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # Wireframe node to detect edges
    wireframe = nodes.new('ShaderNodeWireframe')
    wireframe.location = (-200, 0)
    wireframe.inputs['Size'].default_value = 0.01  # Increased for visibility
    wireframe.use_pixel_size = False

    # Emission for glowing edges
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (0, 100)
    emission.inputs['Color'].default_value = (*edge_color, 1)
    emission.inputs['Strength'].default_value = emission_strength

    # Transparent for non-edge areas
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    transparent.location = (0, -100)

    # Mix based on wireframe
    mix = nodes.new('ShaderNodeMixShader')
    mix.location = (300, 0)

    links.new(wireframe.outputs['Fac'], mix.inputs['Fac'])
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])

    mat.blend_method = 'BLEND'
    return mat

def create_acrylic_glass_material(name, glass_color, is_clear=False):
    """Create colored acrylic glass material for landmarks.
    Based on the AcrylicGlass.blend structure: Glass BSDF + Diffuse BSDF mixed via Fresnel.

    Args:
        name: Material name
        glass_color: RGB tuple for the glass color
        is_clear: If True, uses very light gray for clear/neutral appearance
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # Glass BSDF - main glass appearance
    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.location = (0, 150)
    if is_clear:
        # Clear/light gray for general landmarks
        glass.inputs['Color'].default_value = (0.95, 0.95, 0.95, 1)
    else:
        glass.inputs['Color'].default_value = (*glass_color, 1)
    glass.inputs['Roughness'].default_value = 0.0  # Smooth glass
    glass.inputs['IOR'].default_value = 1.45  # Acrylic IOR

    # Diffuse BSDF - for edge coloring
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.location = (0, -50)
    if is_clear:
        diffuse.inputs['Color'].default_value = (0.9, 0.9, 0.9, 1)
    else:
        # Slightly saturated version for diffuse edges
        diffuse.inputs['Color'].default_value = (*glass_color, 1)
    diffuse.inputs['Roughness'].default_value = 0.0

    # Fresnel for edge detection
    fresnel = nodes.new('ShaderNodeFresnel')
    fresnel.location = (0, -200)
    fresnel.inputs['IOR'].default_value = 1.5  # Slightly higher for more visible edges

    # Mix shader to blend glass and diffuse based on fresnel
    mix = nodes.new('ShaderNodeMixShader')
    mix.location = (300, 0)
    # Low factor = more glass, high factor = more diffuse at edges
    mix.inputs['Fac'].default_value = 0.23

    # Connect nodes
    links.new(fresnel.outputs['Fac'], mix.inputs['Fac'])
    links.new(glass.outputs['BSDF'], mix.inputs[1])
    links.new(diffuse.outputs['BSDF'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])

    mat.blend_method = 'BLEND'
    return mat


def create_frosted_glass_glow_material(name, glow_color, glow_strength=1.0):
    """Legacy function - redirects to acrylic glass for backward compatibility"""
    return create_acrylic_glass_material(name, glow_color, is_clear=False)


def create_pbr_metal_material(name, texture_set_name, uv_scale=10.0):
    """Create a PBR metal material using texture maps from the METAL_TEXTURE_SETS.

    This creates a physically-based material with:
    - Diffuse/albedo color
    - Normal map for surface detail and bumps
    - Roughness map
    - Metallic map
    - Displacement for true geometry detail (optional)

    Args:
        name: Material name
        texture_set_name: Key in METAL_TEXTURE_SETS dictionary
        uv_scale: Scale factor for UV mapping on spheres (higher = smaller texture detail)

    Returns:
        bpy.types.Material: The created PBR material
    """
    if texture_set_name not in METAL_TEXTURE_SETS:
        print(f"Warning: Texture set '{texture_set_name}' not found, using rusty_grate")
        texture_set_name = 'rusty_grate'

    textures = METAL_TEXTURE_SETS[texture_set_name]

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Output node
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (800, 0)

    # Principled BSDF - main shader
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (400, 0)
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    # Texture Coordinate and Mapping for UV scaling
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-800, 0)

    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-600, 0)
    mapping.inputs['Scale'].default_value = (uv_scale, uv_scale, uv_scale)
    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])

    # Load and connect diffuse/color texture
    diffuse_tex = nodes.new('ShaderNodeTexImage')
    diffuse_tex.location = (-200, 300)
    diffuse_tex.image = bpy.data.images.load(textures['diffuse'])
    diffuse_tex.image.colorspace_settings.name = 'sRGB'
    links.new(mapping.outputs['Vector'], diffuse_tex.inputs['Vector'])
    links.new(diffuse_tex.outputs['Color'], principled.inputs['Base Color'])

    # Load and connect roughness texture
    rough_tex = nodes.new('ShaderNodeTexImage')
    rough_tex.location = (-200, 0)
    rough_tex.image = bpy.data.images.load(textures['roughness'])
    rough_tex.image.colorspace_settings.name = 'Non-Color'
    links.new(mapping.outputs['Vector'], rough_tex.inputs['Vector'])
    links.new(rough_tex.outputs['Color'], principled.inputs['Roughness'])

    # Load and connect metallic texture (if available)
    if textures['metallic'] is not None:
        metal_tex = nodes.new('ShaderNodeTexImage')
        metal_tex.location = (-200, -300)
        metal_tex.image = bpy.data.images.load(textures['metallic'])
        metal_tex.image.colorspace_settings.name = 'Non-Color'
        links.new(mapping.outputs['Vector'], metal_tex.inputs['Vector'])
        links.new(metal_tex.outputs['Color'], principled.inputs['Metallic'])
    else:
        # For non-metallic materials (like painted wood), set metallic to 0
        principled.inputs['Metallic'].default_value = 0.0

    # Load and connect normal map (with Normal Map node)
    normal_tex = nodes.new('ShaderNodeTexImage')
    normal_tex.location = (-200, -600)
    normal_tex.image = bpy.data.images.load(textures['normal'])
    normal_tex.image.colorspace_settings.name = 'Non-Color'
    links.new(mapping.outputs['Vector'], normal_tex.inputs['Vector'])

    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.location = (100, -600)
    normal_map.inputs['Strength'].default_value = 1.0  # Full strength bump
    links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])

    print(f"  Created PBR metal material '{name}' using {texture_set_name} textures")
    return mat


# ============================================================================
# CREATE RECTANGULAR PRISMS FOR HEIGHTMAP (OPTIMIZED WITH BMESH)
# ============================================================================
def create_prisms_batch_bmesh(heightmap_norm, nx, ny, sample_rate, world_size,
                               height_scale_min, height_scale, prism_gap,
                               cell_world_size, dark_blue_mat, bevel_radius, bevel_segments):
    """Create all terrain prisms as a single mesh using bmesh for performance.

    This is MUCH faster than creating individual cubes with bpy.ops because:
    - No operator overhead for each prism
    - All geometry created in memory before linking to scene
    - Single bevel modifier instead of thousands
    """
    start_time = time.time()
    print("  Creating prisms using bmesh batch geometry...")

    # Create bmesh
    bm = bmesh.new()

    prism_count = 0

    # Calculate prism size
    prism_size = cell_world_size * sample_rate - prism_gap
    half_size = prism_size / 2.0

    # Create all prisms in bmesh
    for i in range(0, nx, sample_rate):
        for j in range(0, ny, sample_rate):
            height_value = heightmap_norm[i, j]

            # Calculate world position
            x = (i / nx) * world_size
            y = (j / ny) * world_size
            z_height = height_scale_min + height_value * height_scale

            # Center of prism base
            cx = x + half_size
            cy = y + half_size

            # Create 8 vertices for a rectangular prism (box)
            # Bottom face (z = 0)
            v0 = bm.verts.new((cx - half_size, cy - half_size, 0))
            v1 = bm.verts.new((cx + half_size, cy - half_size, 0))
            v2 = bm.verts.new((cx + half_size, cy + half_size, 0))
            v3 = bm.verts.new((cx - half_size, cy + half_size, 0))

            # Top face (z = z_height)
            v4 = bm.verts.new((cx - half_size, cy - half_size, z_height))
            v5 = bm.verts.new((cx + half_size, cy - half_size, z_height))
            v6 = bm.verts.new((cx + half_size, cy + half_size, z_height))
            v7 = bm.verts.new((cx - half_size, cy + half_size, z_height))

            # Create 6 faces (each face is defined by 4 vertices in CCW order)
            bm.faces.new([v0, v1, v2, v3])  # Bottom
            bm.faces.new([v4, v7, v6, v5])  # Top
            bm.faces.new([v0, v4, v5, v1])  # Front
            bm.faces.new([v2, v6, v7, v3])  # Back
            bm.faces.new([v0, v3, v7, v4])  # Left
            bm.faces.new([v1, v5, v6, v2])  # Right

            prism_count += 1

        if i % 20 == 0:
            print(f"    Progress: {i}/{nx} rows...")

    # Create mesh from bmesh
    mesh = bpy.data.meshes.new("TerrainPrisms")
    bm.to_mesh(mesh)
    bm.free()

    # Create object
    terrain_obj = bpy.data.objects.new("TerrainPrisms", mesh)
    bpy.context.collection.objects.link(terrain_obj)

    # Apply material
    terrain_obj.data.materials.append(dark_blue_mat)

    # Add bevel modifier for edge fillets
    bevel = terrain_obj.modifiers.new(name="Bevel", type='BEVEL')
    bevel.width = bevel_radius
    bevel.segments = bevel_segments
    bevel.limit_method = 'ANGLE'
    bevel.angle_limit = 1.0472  # 60 degrees
    bevel.profile = 0.5
    bevel.harden_normals = True

    # Set flat shading
    for poly in mesh.polygons:
        poly.use_smooth = False

    elapsed = time.time() - start_time
    print(f"  Created {prism_count} prisms in {elapsed:.2f} seconds using bmesh batch geometry")

    return terrain_obj


print("Creating rectangular prisms for heightmap...")

nx, ny = heightmap_norm.shape

# Create uniform dark blue material for all prisms (no height-based coloring)
print("  Creating uniform dark blue material...")
dark_blue_mat = create_uniform_dark_blue_material("DarkBluePrism")
print("  Material created")

# Create prisms - sample every few cells for performance
SAMPLE_RATE = 4  # Sample every 4th cell for faster scene creation

# Calculate cell size in world coordinates
cell_world_size = WORLD_SIZE / nx

# Quality settings for clean prisms (subtle bevel, no visible polygons)
BEVEL_SEGMENTS = 1  # Minimal bevel segments for very subtle edge rounding

# Create all prisms using optimized bmesh batch geometry
terrain_obj = create_prisms_batch_bmesh(
    heightmap_norm=heightmap_norm,
    nx=nx,
    ny=ny,
    sample_rate=SAMPLE_RATE,
    world_size=WORLD_SIZE,
    height_scale_min=HEIGHT_SCALE_MIN,
    height_scale=HEIGHT_SCALE,
    prism_gap=PRISM_GAP,
    cell_world_size=cell_world_size,
    dark_blue_mat=dark_blue_mat,
    bevel_radius=PRISM_BEVEL_RADIUS,
    bevel_segments=BEVEL_SEGMENTS
)

prism_count = (nx // SAMPLE_RATE) * (ny // SAMPLE_RATE)
print(f"Created {prism_count} rectangular prisms with uniform dark blue material")

# ============================================================================
# CREATE EXPLICIT TRON WIREFRAME GRID
# ============================================================================
print("Creating explicit Tron wireframe grid...")

def create_wireframe_edge_material(name, edge_color, emission_strength=50.0):
    """Create a brightly glowing emission material for wireframe edges."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Pure emission shader for maximum glow
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (0, 0)
    emission.inputs['Color'].default_value = (*edge_color, 1)
    emission.inputs['Strength'].default_value = emission_strength

    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    return mat

# Create glowing wireframe material - cyan with 25% increased brightness
WIREFRAME_COLOR = SYNTHWAVE_CYAN  # Synthwave cyan
WIREFRAME_EMISSION_STRENGTH = 1.25 * 1.25  # Increased by 25%
wireframe_edge_mat = create_wireframe_edge_material("TronWireframeEdge", WIREFRAME_COLOR, emission_strength=WIREFRAME_EMISSION_STRENGTH)

# Wireframe edge thickness (cylinder radius) - reduced by another 50% per user request
WIREFRAME_THICKNESS = 0.001  # 0.001 feet = ~0.012 inches (reduced by another 50% from 0.002)

# Minor grid configuration (5x finer subdivisions)
MINOR_GRID_ENABLED = True
MINOR_GRID_SUBDIVISIONS = 5  # Divide each major grid square into 5x5
MINOR_GRID_COLOR = (1.0, 1.0, 1.0)  # White for minor grid
MINOR_GRID_THICKNESS = 0.0005  # Half thickness of major grid
MINOR_GRID_EMISSION_STRENGTH = 0.5 * 1.10  # Increased by 10%

def get_max_neighbor_height(i, j, nx, ny, sample_rate):
    """Get the maximum height from all neighboring prisms around a grid corner point.

    This ensures the wireframe grid hugs the 'top layer' of prisms and never dips
    below the top of any neighboring prism during downward slopes.
    """
    max_height = 0.0

    # Check all 4 neighboring cells (prisms) around this corner
    # The corner at (i, j) touches prisms at:
    # - (i-sample_rate, j-sample_rate) top-right corner of that prism
    # - (i, j-sample_rate) top-left corner
    # - (i-sample_rate, j) bottom-right corner
    # - (i, j) bottom-left corner of the prism starting here
    for di in range(-sample_rate, sample_rate + 1, sample_rate):
        for dj in range(-sample_rate, sample_rate + 1, sample_rate):
            ni = i + di
            nj = j + dj
            if 0 <= ni < nx and 0 <= nj < ny:
                height_value = heightmap_norm[ni, nj]
                max_height = max(max_height, height_value)

    return max_height

def create_wireframe_grid_lines():
    """Create explicit wireframe grid as glowing line segments at prism corners.

    This creates a mesh grid overlayed on top of the prisms, positioned at where
    the prism corners would be if there were no bevels. Uses cylinders to create
    glowing edges.

    IMPORTANT: At each grid vertex, the height is set to the MAXIMUM height of all
    neighboring prisms. This ensures the wireframe hugs the 'top layer' of prisms
    and never dips below any prism top during downward slopes.
    """
    vertices = []
    edges = []

    # Create grid lines at prism corners
    # Grid resolution should match prism sampling
    nx, ny = heightmap_norm.shape

    vertex_index = 0

    # Create horizontal lines (along X axis) for each row
    for j in range(0, ny, SAMPLE_RATE):
        row_verts = []
        y = (j / ny) * WORLD_SIZE

        for i in range(0, nx, SAMPLE_RATE):
            x = (i / nx) * WORLD_SIZE
            # Get MAXIMUM height from all neighboring prisms to hug the top layer
            max_height_value = get_max_neighbor_height(i, j, nx, ny, SAMPLE_RATE)
            z = HEIGHT_SCALE_MIN + max_height_value * HEIGHT_SCALE

            vertices.append((x, y, z))
            row_verts.append(vertex_index)

            # Connect to previous vertex in row
            if len(row_verts) > 1:
                edges.append((row_verts[-2], row_verts[-1]))

            vertex_index += 1

    # Create vertical lines (along Y axis) for each column
    # We need to rebuild column connections
    grid_cols = (nx + SAMPLE_RATE - 1) // SAMPLE_RATE
    grid_rows = (ny + SAMPLE_RATE - 1) // SAMPLE_RATE

    for i in range(grid_cols):
        for j in range(grid_rows - 1):
            # Connect vertex at (i, j) to vertex at (i, j+1)
            idx_current = j * grid_cols + i
            idx_next = (j + 1) * grid_cols + i
            if idx_current < len(vertices) and idx_next < len(vertices):
                edges.append((idx_current, idx_next))

    return vertices, edges

def create_wireframe_from_edges(name, vertices, edges, thickness, material, subdivisions=2):
    """Create a wireframe object from vertices and edges using NURBS curves.

    This is MUCH faster than using skin modifiers because it avoids:
    - Mode switching (edit mode)
    - Modifier application operations
    - Mesh conversion overhead
    """
    start_time = time.time()

    # Create curve data
    curve = bpy.data.curves.new(name + '_Curve', 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = thickness
    curve.bevel_resolution = subdivisions
    curve.fill_mode = 'FULL'
    curve.use_fill_caps = True

    # Add a polyline spline for each edge
    for v1_idx, v2_idx in edges:
        spline = curve.splines.new('POLY')
        spline.points.add(1)  # We need 2 points total
        # POLY splines use 4D homogeneous coordinates
        spline.points[0].co = (*vertices[v1_idx], 1)
        spline.points[1].co = (*vertices[v2_idx], 1)

    # Create object from curve
    wireframe_obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(wireframe_obj)

    # Apply material
    wireframe_obj.data.materials.append(material)

    elapsed = time.time() - start_time
    print(f"  Created {name} with {len(edges)} curve segments in {elapsed:.2f}s")

    return wireframe_obj


# Create the wireframe mesh
print("  Generating wireframe grid vertices and edges...")
wf_vertices, wf_edges = create_wireframe_grid_lines()
print(f"  Created {len(wf_vertices)} vertices and {len(wf_edges)} edges")

# Create optimized wireframe using NURBS curves (much faster than skin modifier)
wireframe_obj = create_wireframe_from_edges(
    name="TronWireframeGrid",
    vertices=wf_vertices,
    edges=wf_edges,
    thickness=WIREFRAME_THICKNESS,
    material=wireframe_edge_mat,
    subdivisions=2
)

print(f"  Tron wireframe grid created with {len(wf_edges)} glowing edges")

# ============================================================================
# CREATE MINOR GRID (5x FINER SUBDIVISIONS)
# ============================================================================
if MINOR_GRID_ENABLED:
    print("Creating minor grid subdivisions (5x finer)...")

    # Create material for minor grid (white, dimmer)
    minor_grid_mat = create_wireframe_edge_material("MinorGridEdge", MINOR_GRID_COLOR, emission_strength=MINOR_GRID_EMISSION_STRENGTH)

    def create_minor_grid_lines():
        """Create minor grid lines that subdivide each major grid cell into 5x5.

        These lines are positioned between the major grid lines, NOT on top of them.
        Uses the same max-neighbor-height logic as major gridlines to ensure minor
        gridlines never dip below prism tops.
        """
        vertices = []
        edges = []

        nx, ny = heightmap_norm.shape
        vertex_index = 0

        # Calculate the minor grid step (5x finer than major)
        minor_step = SAMPLE_RATE // MINOR_GRID_SUBDIVISIONS
        if minor_step < 1:
            minor_step = 1

        # Create a 2D array to store vertex indices for cross-referencing
        # This helps us connect vertical lines correctly
        vertex_grid = {}

        # Create horizontal lines for minor grid
        for j in range(0, ny, minor_step):
            # Skip if this is a major grid line (we don't want to duplicate)
            if j % SAMPLE_RATE == 0:
                continue

            row_verts = []
            y = (j / ny) * WORLD_SIZE

            for i in range(0, nx, minor_step):
                x = (i / nx) * WORLD_SIZE
                # Use max neighbor height (same logic as major grid) to prevent dipping
                max_height_value = get_max_neighbor_height(i, j, nx, ny, SAMPLE_RATE)
                z = HEIGHT_SCALE_MIN + max_height_value * HEIGHT_SCALE

                vertices.append((x, y, z))
                vertex_grid[(i, j)] = vertex_index
                row_verts.append(vertex_index)

                # Connect to previous vertex in row
                if len(row_verts) > 1:
                    edges.append((row_verts[-2], row_verts[-1]))

                vertex_index += 1

        # Create vertical lines for minor grid (only between major grid lines)
        for i in range(0, nx, minor_step):
            # Skip if this is a major grid line
            if i % SAMPLE_RATE == 0:
                continue

            col_verts = []
            x = (i / nx) * WORLD_SIZE

            for j in range(0, ny, minor_step):
                y = (j / ny) * WORLD_SIZE
                # Use max neighbor height (same logic as major grid) to prevent dipping
                max_height_value = get_max_neighbor_height(i, j, nx, ny, SAMPLE_RATE)
                z = HEIGHT_SCALE_MIN + max_height_value * HEIGHT_SCALE

                # Check if we already have a vertex at this position
                if (i, j) in vertex_grid:
                    col_verts.append(vertex_grid[(i, j)])
                else:
                    vertices.append((x, y, z))
                    vertex_grid[(i, j)] = vertex_index
                    col_verts.append(vertex_index)
                    vertex_index += 1

                # Connect to previous vertex in column
                if len(col_verts) > 1:
                    edges.append((col_verts[-2], col_verts[-1]))

        return vertices, edges

    # Generate minor grid
    minor_vertices, minor_edges = create_minor_grid_lines()
    print(f"  Created {len(minor_vertices)} minor grid vertices and {len(minor_edges)} edges")

    if len(minor_vertices) > 0 and len(minor_edges) > 0:
        # Create optimized minor grid using NURBS curves (reuse the function from major grid)
        minor_grid_obj = create_wireframe_from_edges(
            name="MinorWireframeGrid",
            vertices=minor_vertices,
            edges=minor_edges,
            thickness=MINOR_GRID_THICKNESS,
            material=minor_grid_mat,
            subdivisions=1  # Fewer subdivisions for thinner lines
        )

        print(f"  Minor grid created with {len(minor_edges)} white subdivision lines")
    else:
        print("  WARNING: No minor grid lines generated")

# ============================================================================
# CREATE BASE PLATFORM
# ============================================================================
print("Creating base platform...")

# Dark reflective base
base_mat = bpy.data.materials.new(name="BaseMaterial")
base_mat.use_nodes = True
nodes = base_mat.node_tree.nodes
links = base_mat.node_tree.links
nodes.clear()

output = nodes.new('ShaderNodeOutputMaterial')
output.location = (400, 0)

principled = nodes.new('ShaderNodeBsdfPrincipled')
principled.location = (0, 0)
principled.inputs['Base Color'].default_value = (0.02, 0.02, 0.05, 1)
principled.inputs['Roughness'].default_value = 0.1
principled.inputs['Metallic'].default_value = 0.95
principled.inputs['Specular IOR Level'].default_value = 0.8

links.new(principled.outputs['BSDF'], output.inputs['Surface'])

bpy.ops.mesh.primitive_plane_add(size=WORLD_SIZE * 1.5, location=(WORLD_SIZE/2, WORLD_SIZE/2, -1))
base = bpy.context.active_object
base.name = "BasePlatform"
base.data.materials.append(base_mat)

print("Base platform created")

# ============================================================================
# CREATE GLASS TUBE ROADS
# ============================================================================
def interpolate_coords(coords, num_interp):
    """Interpolate between coordinate points for smoother curves"""
    new_coords = []
    for i in range(len(coords) - 1):
        for j in range(num_interp):
            t = j / num_interp
            interp = coords[i] * (1 - t) + coords[i + 1] * t
            new_coords.append(interp)
    new_coords.append(coords[-1])
    return np.array(new_coords)

def smooth_trajectory(coords, num_interp_per_segment=10):
    """Interpolate trajectory with Catmull-Rom spline for perfectly smooth curves"""
    n = len(coords)
    if n < 4:
        return coords

    # Calculate cumulative arc-length parameterization
    distances = np.zeros(n)
    for i in range(1, n):
        distances[i] = distances[i-1] + np.linalg.norm(coords[i] - coords[i-1])
    if distances[-1] > 0:
        t_original = distances / distances[-1]
    else:
        t_original = np.linspace(0, 1, n)

    # Target number of output points
    n_output = (n - 1) * num_interp_per_segment + 1
    t_smooth = np.linspace(0, 1, n_output)

    # Catmull-Rom spline interpolation for each dimension
    def catmull_rom_interp(t_vals, t_knots, values):
        """Centripetal Catmull-Rom spline interpolation"""
        result = np.zeros(len(t_vals))

        # Extend control points for end conditions
        p = np.zeros(n + 2)
        p[1:-1] = values
        p[0] = 2 * values[0] - values[1]  # Reflect first segment
        p[-1] = 2 * values[-1] - values[-2]  # Reflect last segment

        t_ext = np.zeros(n + 2)
        t_ext[1:-1] = t_knots
        t_ext[0] = t_knots[0] - (t_knots[1] - t_knots[0])
        t_ext[-1] = t_knots[-1] + (t_knots[-1] - t_knots[-2])

        for idx, t in enumerate(t_vals):
            # Find segment
            seg = np.searchsorted(t_knots[:-1], t, side='right') - 1
            seg = max(0, min(seg, n - 2))

            # Get 4 control points for Catmull-Rom
            i = seg + 1  # Offset for extended array
            p0, p1, p2, p3 = p[i-1], p[i], p[i+1], p[i+2]
            t0, t1, t2, t3 = t_ext[i-1], t_ext[i], t_ext[i+1], t_ext[i+2]

            # Normalize t to [0, 1] within segment
            if t2 - t1 > 0:
                u = (t - t1) / (t2 - t1)
            else:
                u = 0

            # Catmull-Rom basis (uniform parameterization for simplicity)
            u2 = u * u
            u3 = u2 * u

            # Standard Catmull-Rom coefficients (tension = 0.5)
            result[idx] = 0.5 * (
                (-u3 + 2*u2 - u) * p0 +
                (3*u3 - 5*u2 + 2) * p1 +
                (-3*u3 + 4*u2 + u) * p2 +
                (u3 - u2) * p3
            )

        return result

    x_smooth = catmull_rom_interp(t_smooth, t_original, coords[:, 0])
    y_smooth = catmull_rom_interp(t_smooth, t_original, coords[:, 1])

    return np.column_stack([x_smooth, y_smooth])

def create_glass_tube_road(coords, name, tube_radius, material, height_offset=10.0):
    """Create a smooth glass tube road with simplified control points.

    Uses fewer control points and lets NURBS handle the smoothing to avoid
    glitching artifacts from too many control points with volumetric materials.
    """
    # Downsample the coordinates to reduce control point count
    # Take every Nth point to dramatically reduce complexity
    downsample_factor = 5  # Use every 5th point from original trajectory
    downsampled_coords = coords[::downsample_factor]

    # Ensure we include the last point
    if not np.array_equal(downsampled_coords[-1], coords[-1]):
        downsampled_coords = np.vstack([downsampled_coords, coords[-1]])

    # Apply light Catmull-Rom smoothing with fewer interpolation points
    smooth_coords = smooth_trajectory(downsampled_coords, num_interp_per_segment=2)
    n_points = len(smooth_coords)

    print(f"  {name}: {len(coords)} original -> {len(downsampled_coords)} downsampled -> {n_points} final points")

    # Get heights along smoothed path - use MAXIMUM height in local neighborhood
    # to ensure trajectory never dips below prism tops
    def get_max_local_height(x, y, radius=0.15):
        """Get maximum height in a small radius around the point.

        This ensures trajectories maintain proper clearance even during height transitions.
        The radius should be larger than the tube radius plus some margin.
        """
        # Sample 5x5 grid around the point
        offsets = [-radius, -radius/2, 0, radius/2, radius]
        max_h = 0.0
        for dx in offsets:
            for dy in offsets:
                h = get_height_at(x + dx, y + dy, heightmap_norm, HEIGHT_SCALE, WORLD_SIZE)
                max_h = max(max_h, h)
        return max_h

    heights = np.array([get_max_local_height(c[0], c[1], radius=tube_radius * 2)
                       for c in smooth_coords])

    # Smooth the heights with larger kernel to avoid abrupt changes
    # But use maximum filter first to ensure we never go below terrain
    kernel_size = 11
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_heights = np.convolve(heights, kernel, mode='same')

    # Ensure smoothed height is never below the original max height at each point
    # This prevents the trajectory from dipping into prisms during smoothing
    heights = np.maximum(heights, smoothed_heights)

    # Create curve with high quality settings
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = tube_radius
    curve_data.bevel_resolution = 16  # Reduced for smoother appearance
    curve_data.fill_mode = 'FULL'
    curve_data.resolution_u = 64  # Higher resolution to let NURBS smooth nicely

    # Use NURBS with good smoothing
    spline = curve_data.splines.new('NURBS')
    spline.points.add(n_points - 1)
    spline.use_endpoint_u = True
    spline.order_u = min(4, n_points)  # Order 4 for smooth but stable curves

    for i, (coord, h) in enumerate(zip(smooth_coords, heights)):
        spline.points[i].co = (coord[0], coord[1], h + height_offset, 1.0)

    # Create object
    tube_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(tube_obj)

    tube_obj.data.materials.append(material)

    return tube_obj

print("Creating neon tube trajectories...")

# Highway - cyan neon tube (glass with glowing gas inside)
highway_mat = create_neon_tube_material("HighwayNeon", SYNTHWAVE_CYAN, emission_strength=25.0)
highway = create_glass_tube_road(highway_coords, "Highway", ROAD_WIDTH/2, highway_mat, height_offset=HIGHWAY_HEIGHT_OFFSET)

# Side road - magenta neon tube (glass with glowing gas inside)
sideroad_mat = create_neon_tube_material("SideroadNeon", SYNTHWAVE_MAGENTA, emission_strength=25.0)
sideroad = create_glass_tube_road(sideroad_coords, "Sideroad", SIDEROAD_WIDTH/2, sideroad_mat, height_offset=SIDEROAD_HEIGHT_OFFSET)

print("Neon tube trajectories created")

# ============================================================================
# CREATE METALLIC SPHERE LANDMARKS (OPTIMIZED WITH INSTANCING)
# ============================================================================
def create_base_sphere_mesh(name, radius, segments=128, ring_count=64):
    """Create a base sphere mesh that can be instanced.

    Uses bmesh to create high-resolution sphere geometry that will be shared
    across all instances via linked duplicates. This is MUCH faster than
    creating individual spheres with bpy.ops.
    """
    # Create mesh data
    mesh = bpy.data.meshes.new(f"{name}_Mesh")

    # Use bmesh to create sphere geometry
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=ring_count, radius=radius)
    bm.to_mesh(mesh)
    bm.free()

    # Apply smooth shading to all polygons
    for polygon in mesh.polygons:
        polygon.use_smooth = True

    return mesh


def create_landmark_instances(all_landmarks, base_meshes, light_energy):
    """Create landmark instances using linked duplicates for memory efficiency.

    Creates one instance per landmark using shared mesh data from base_meshes.
    Each instance gets its own transform and material, but shares the underlying
    geometry, drastically reducing memory usage and creation time.
    """
    created_spheres = []

    for lm in all_landmarks:
        prefix = lm['prefix']
        mesh = base_meshes[prefix]

        # Create linked duplicate (shares mesh data)
        obj = bpy.data.objects.new(
            f"{prefix}_{lm['pos'][0]:.0f}_{lm['pos'][1]:.0f}",
            mesh
        )
        obj.location = lm['pos']

        # Material assignment (each instance can have unique material slot)
        if len(obj.data.materials) == 0:
            obj.data.materials.append(lm['material'])
        else:
            obj.data.materials[0] = lm['material']

        # Link to scene collection
        bpy.context.collection.objects.link(obj)

        # Add subdivision modifier to the instance
        subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subsurf.levels = 1
        subsurf.render_levels = 2
        subsurf.subdivision_type = 'CATMULL_CLARK'

        created_spheres.append(obj)

        # Add a dim white point light at the center of the sphere
        bpy.ops.object.light_add(type='POINT', location=lm['pos'])
        light = bpy.context.active_object
        light.name = f"LandmarkLight_{prefix}_{lm['pos'][0]:.0f}_{lm['pos'][1]:.0f}"
        light.data.energy = light_energy
        light.data.color = (1.0, 1.0, 1.0)  # White light
        light.data.shadow_soft_size = lm['radius']

    return created_spheres

print("Creating metallic landmark spheres with bump-mapped textures...")

# Different PBR metal materials for each landmark class using bump-mapped textures
# Each texture set provides: diffuse, normal, roughness, metallic maps
# UV scale controls texture detail on spheres (higher = smaller texture pattern)

# Lecture 1 landmarks: Light gray acrylic glass
landmark1_mat = create_acrylic_glass_material("Lecture1_LightGrayAcrylic", (0.7, 0.7, 0.7), is_clear=False)  # Light gray
print("  Using Light Gray Acrylic Glass material for lecture1 landmarks")

# Lecture 2 landmarks: Dark gray acrylic glass
landmark2_mat = create_acrylic_glass_material("Lecture2_DarkGrayAcrylic", (0.25, 0.25, 0.25), is_clear=False)  # Dark gray
print("  Using Dark Gray Acrylic Glass material for lecture2 landmarks")

# General landmarks: Medium gray acrylic glass (between light and dark gray)
landmark3_mat = create_acrylic_glass_material("General_MediumGrayAcrylic", (0.475, 0.475, 0.475), is_clear=False)  # Medium gray
print("  Using Medium Gray Acrylic Glass material for general landmarks")

# Build list of all landmarks with positions, priorities, and materials
# Priority: general (3) > lecture2 (2) > lecture1 (1) - higher = more likely to be enlarged on overlap
import math

def remove_duplicate_landmarks_by_type(coords, prefix):
    """Remove duplicate landmarks of the same type at the same location.

    Locations are rounded to 0.01 units to detect duplicates.
    Returns a list of unique coordinate tuples.
    """
    seen_locations = set()
    unique_coords = []
    for coord in coords:
        # Round to 0.01 to detect duplicates
        rounded_loc = (round(coord[0], 2), round(coord[1], 2))
        if rounded_loc not in seen_locations:
            seen_locations.add(rounded_loc)
            unique_coords.append(coord)
        else:
            print(f"    Removed duplicate {prefix} at ({coord[0]:.3f}, {coord[1]:.3f})")
    return unique_coords

print("  Removing duplicate landmarks at same location (rounded to 0.01)...")
lecture1_landmarks_unique = remove_duplicate_landmarks_by_type(lecture1_landmarks, "Lecture1")
lecture2_landmarks_unique = remove_duplicate_landmarks_by_type(lecture2_landmarks, "Lecture2")
general_landmarks_unique = remove_duplicate_landmarks_by_type(general_landmarks, "General")
print(f"  Lecture1: {len(lecture1_landmarks)} -> {len(lecture1_landmarks_unique)} landmarks")
print(f"  Lecture2: {len(lecture2_landmarks)} -> {len(lecture2_landmarks_unique)} landmarks")
print(f"  General: {len(general_landmarks)} -> {len(general_landmarks_unique)} landmarks")

all_landmarks = []
for coord in lecture1_landmarks_unique:
    h = get_height_at(coord[0], coord[1], heightmap_norm, HEIGHT_SCALE, WORLD_SIZE)
    all_landmarks.append({
        'pos': (coord[0], coord[1], h + LANDMARK_FLOAT_HEIGHT),
        'radius': LANDMARK_RADIUS,
        'priority': 1,
        'material': landmark1_mat,
        'prefix': "Lecture1"
    })

for coord in lecture2_landmarks_unique:
    h = get_height_at(coord[0], coord[1], heightmap_norm, HEIGHT_SCALE, WORLD_SIZE)
    all_landmarks.append({
        'pos': (coord[0], coord[1], h + LANDMARK_FLOAT_HEIGHT),
        'radius': LANDMARK_RADIUS,
        'priority': 2,
        'material': landmark2_mat,
        'prefix': "Lecture2"
    })

for coord in general_landmarks_unique:
    h = get_height_at(coord[0], coord[1], heightmap_norm, HEIGHT_SCALE, WORLD_SIZE)
    all_landmarks.append({
        'pos': (coord[0], coord[1], h + LANDMARK_FLOAT_HEIGHT),
        'radius': LANDMARK_RADIUS,
        'priority': 3,
        'material': landmark3_mat,
        'prefix': "General"
    })

# Detect overlapping landmark groups and stack them vertically
# First pass: detect groups of overlapping landmarks using union-find
print(f"  Checking {len(all_landmarks)} landmarks for overlaps...")

# Build adjacency list of overlapping landmarks
def landmarks_overlap(lm_i, lm_j):
    """Check if two landmarks overlap based on their positions and radii."""
    dx = lm_i['pos'][0] - lm_j['pos'][0]
    dy = lm_i['pos'][1] - lm_j['pos'][1]
    dz = lm_i['pos'][2] - lm_j['pos'][2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    sum_radii = lm_i['radius'] + lm_j['radius']
    return dist < sum_radii

# Union-find to group overlapping landmarks
parent = list(range(len(all_landmarks)))

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[px] = py

# First pass: detect all overlapping pairs and union them
for i in range(len(all_landmarks)):
    for j in range(i + 1, len(all_landmarks)):
        if landmarks_overlap(all_landmarks[i], all_landmarks[j]):
            union(i, j)
            print(f"    Overlap detected: {all_landmarks[i]['prefix']} & {all_landmarks[j]['prefix']}")

# Group landmarks by their root
from collections import defaultdict
groups = defaultdict(list)
for i in range(len(all_landmarks)):
    groups[find(i)].append(i)

# Second pass: for each group, sort by reverse preference and stack vertically
# Preference order (reverse): lecture2 (priority 2) < lecture1 (priority 1) < general (priority 3)
# Actually the user wants: lecture2 first, then lecture1, then general
# So reverse preference means: priority 2 (lecture2), then 1 (lecture1), then 3 (general)
# This ordering: 2, 1, 3

def sort_key_reverse_preference(idx):
    """Sort key for reverse preference: lecture2 (2) -> lecture1 (1) -> general (3)"""
    priority = all_landmarks[idx]['priority']
    # Map: 2 -> 0, 1 -> 1, 3 -> 2 (so lecture2 comes first, then lecture1, then general)
    if priority == 2:  # lecture2
        return 0
    elif priority == 1:  # lecture1
        return 1
    else:  # general (priority 3)
        return 2

# Vertical stacking: each successive landmark is raised by (2 * radius + gap)
VERTICAL_GAP = 0.02  # Small gap between stacked landmarks (in feet)
VERTICAL_OFFSET = LANDMARK_RADIUS * 2 + VERTICAL_GAP  # Full diameter + gap

overlap_group_count = 0
for root, indices in groups.items():
    if len(indices) > 1:
        overlap_group_count += 1
        # Sort by reverse preference: lecture2, lecture1, general
        indices.sort(key=sort_key_reverse_preference)
        print(f"    Group {overlap_group_count}: {[all_landmarks[i]['prefix'] for i in indices]}")

        # First landmark keeps original position, subsequent ones are stacked vertically
        for rank, idx in enumerate(indices):
            if rank > 0:
                # Add vertical offset to z-position
                old_pos = all_landmarks[idx]['pos']
                new_z = old_pos[2] + rank * VERTICAL_OFFSET
                all_landmarks[idx]['pos'] = (old_pos[0], old_pos[1], new_z)
                print(f"      {all_landmarks[idx]['prefix']} z raised by {rank * VERTICAL_OFFSET:.3f} to {new_z:.3f}")

print(f"  Found {overlap_group_count} overlapping groups")

# Now create all the spheres using optimized instancing
LANDMARK_LIGHT_ENERGY = 0.05  # Reduced by 50% for subtler glow

# Create base sphere meshes for instancing (one per material type)
# This drastically reduces memory usage by sharing geometry across all instances
print("  Creating base sphere meshes for instancing...")
start_time = time.time()

base_meshes = {
    'Lecture1': create_base_sphere_mesh('Lecture1_BaseSphere', LANDMARK_RADIUS, segments=128, ring_count=64),
    'Lecture2': create_base_sphere_mesh('Lecture2_BaseSphere', LANDMARK_RADIUS, segments=128, ring_count=64),
    'General': create_base_sphere_mesh('General_BaseSphere', LANDMARK_RADIUS, segments=128, ring_count=64),
}
print(f"  Created {len(base_meshes)} base sphere meshes")

# Create all landmark instances with lights using linked duplicates
print(f"  Creating {len(all_landmarks)} landmark instances with lights...")
created_spheres = create_landmark_instances(all_landmarks, base_meshes, LANDMARK_LIGHT_ENERGY)

elapsed = time.time() - start_time
print(f"Landmark spheres created with {len(all_landmarks)} dim white point lights in {elapsed:.2f}s (using instancing)")

# ============================================================================
# TRACTOR BEAM EFFECTS - Truncated Cone Volumetrics with Noise Texture
# ============================================================================
print("Creating tractor beam effects with truncated cone volumetrics...")

# Calculate 50th percentile height threshold for determining pull/push behavior
# Heights above this threshold show "pulling up" effect; below show "pushing down"
all_heights = heightmap_norm.flatten()
HEIGHT_PERCENTILE_THRESHOLD = np.percentile(all_heights, 50)
print(f"  Height percentile threshold (50th): {HEIGHT_PERCENTILE_THRESHOLD:.4f}")

# Tractor beam configuration
TRACTOR_BEAM_ENABLED = False  # Set to False to disable tractor beams
TRACTOR_BEAM_COLOR_MAPPING = {
    # Map landmark prefix to beam color matching glass tint
    'Lecture1': (0.95, 0.95, 0.95),  # Clear/white
    'Lecture2': (0.5, 0.5, 0.5),      # Light gray
    'General': (0.15, 0.15, 0.15),    # Dark gray
}
TRACTOR_BEAM_LIGHT_POWER = 50.0  # Area light power
# Unified density for all beams (both push and pull)
TRACTOR_BEAM_DENSITY = 0.25  # Same density for both effects
# Cone base radius is 2x sphere radius (configured in create_tractor_beam)


def create_truncated_cone_mesh(name, top_center, top_radius, bottom_center, bottom_radius, segments=32):
    """Create a truncated cone mesh between two circular ends.

    Args:
        name: Name for the mesh object
        top_center: (x, y, z) center of top circle
        top_radius: Radius of top circle (matches sphere circumference)
        bottom_center: (x, y, z) center of bottom circle
        bottom_radius: Radius of bottom circle (3x sphere diameter)
        segments: Number of segments around the cone

    Returns:
        bpy.types.Object: The cone mesh object
    """
    import bmesh

    # Create mesh and object
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Create bmesh
    bm = bmesh.new()

    # Calculate cone height and direction
    tx, ty, tz = top_center
    bx, by, bz = bottom_center

    # Create vertices for top and bottom circles
    top_verts = []
    bottom_verts = []

    for i in range(segments):
        angle = 2 * math.pi * i / segments

        # Top circle vertices
        x = tx + top_radius * math.cos(angle)
        y = ty + top_radius * math.sin(angle)
        z = tz
        top_verts.append(bm.verts.new((x, y, z)))

        # Bottom circle vertices
        x = bx + bottom_radius * math.cos(angle)
        y = by + bottom_radius * math.sin(angle)
        z = bz
        bottom_verts.append(bm.verts.new((x, y, z)))

    # Create top center vertex and bottom center vertex
    top_center_vert = bm.verts.new(top_center)
    bottom_center_vert = bm.verts.new(bottom_center)

    bm.verts.ensure_lookup_table()

    # Create faces for the cone sides
    for i in range(segments):
        next_i = (i + 1) % segments
        # Side quad (as two triangles)
        bm.faces.new([top_verts[i], bottom_verts[i], bottom_verts[next_i], top_verts[next_i]])

    # Create top cap (triangles from center to edge)
    for i in range(segments):
        next_i = (i + 1) % segments
        bm.faces.new([top_center_vert, top_verts[next_i], top_verts[i]])

    # Create bottom cap (triangles from center to edge)
    for i in range(segments):
        next_i = (i + 1) % segments
        bm.faces.new([bottom_center_vert, bottom_verts[i], bottom_verts[next_i]])

    # Write to mesh
    bm.to_mesh(mesh)
    bm.free()

    return obj


def create_tractor_beam_volume_material(name, beam_color, is_push_effect, density):
    """Create volumetric material with noise texture for tractor beam.

    CORRECTED behavior (after user feedback):
    - PUSH (low terrain): Light from sphere down to terrain, smoke denser at top (sphere)
    - PULL (high terrain): Light from terrain up to sphere, smoke denser at bottom (terrain)

    Args:
        name: Material name
        beam_color: RGB tuple for the beam color
        is_push_effect: True = push down (low terrain), False = pull up (high terrain)
        density: Base density multiplier for the volume

    Returns:
        bpy.types.Material: The volumetric material
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # 1. Texture Coordinate node - for 3D noise and gradient positioning
    tex_coord = nodes.new('ShaderNodeTexCoord')
    tex_coord.location = (-800, 0)

    # 2. Noise Texture node - 3D noise with INCREASED detail and roughness
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-600, 0)
    noise.noise_dimensions = '3D'
    noise.inputs['Scale'].default_value = 0.5
    noise.inputs['Detail'].default_value = 30.0  # Increased for more visible noise
    noise.inputs['Roughness'].default_value = 0.95  # High roughness for texture
    noise.inputs['Distortion'].default_value = 0.0

    # 3. Vertical Gradient Texture for directional fade
    vert_gradient = nodes.new('ShaderNodeTexGradient')
    vert_gradient.location = (-600, -200)
    vert_gradient.gradient_type = 'LINEAR'

    # 4. Mapping node to orient vertical gradient along Z axis
    vert_mapping = nodes.new('ShaderNodeMapping')
    vert_mapping.location = (-800, -200)
    # Rotate to align gradient with Z axis (vertical)
    vert_mapping.inputs['Rotation'].default_value = (math.pi/2, 0, 0)  # 90 degrees around X

    # 5. Radial Gradient Texture for edge fadeout (SPHERICAL type for conical fade)
    radial_gradient = nodes.new('ShaderNodeTexGradient')
    radial_gradient.location = (-600, -400)
    radial_gradient.gradient_type = 'SPHERICAL'

    # 6. Mapping for radial gradient - center it on the object
    radial_mapping = nodes.new('ShaderNodeMapping')
    radial_mapping.location = (-800, -400)
    # Offset to center the gradient (Generated coords are 0-1, center at 0.5)
    radial_mapping.inputs['Location'].default_value = (-0.5, -0.5, -0.5)
    # Scale to control falloff (larger = softer edge)
    radial_mapping.inputs['Scale'].default_value = (1.5, 1.5, 1.5)

    # 7. Radial ColorRamp - dense in center, fading to edges
    radial_ramp = nodes.new('ShaderNodeValToRGB')
    radial_ramp.location = (-400, -400)
    radial_ramp.color_ramp.elements[0].position = 0.0
    radial_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)  # Full density at center
    radial_ramp.color_ramp.elements[1].position = 0.7
    radial_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)  # Zero at edges

    # 8. Vertical Color Ramp for gradient fade (direction depends on push/pull)
    vert_ramp = nodes.new('ShaderNodeValToRGB')
    vert_ramp.location = (-400, -200)
    vert_ramp.color_ramp.elements[0].position = 0.0
    vert_ramp.color_ramp.elements[1].position = 1.0

    # All beams now use WHITE color instead of beam_color
    white_color = (1.0, 1.0, 1.0)
    if is_push_effect:
        # PUSH (low terrain): Light from sphere (top), color dense at top fading down
        # Gradient: white at top (1.0), black at bottom (0.0)
        vert_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)  # Black at bottom
        vert_ramp.color_ramp.elements[1].color = (*white_color, 1.0)  # White at top
    else:
        # PULL (high terrain): Light from terrain (bottom), color dense at bottom fading up
        # Gradient: white at bottom (0.0), black at top (1.0)
        vert_ramp.color_ramp.elements[0].color = (*white_color, 1.0)  # White at bottom
        vert_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)  # Black at top

    # 9. Multiply noise with vertical gradient
    mult_noise_vert = nodes.new('ShaderNodeMath')
    mult_noise_vert.location = (-200, -100)
    mult_noise_vert.operation = 'MULTIPLY'

    # 10. Multiply result with radial gradient for edge fadeout
    mult_radial = nodes.new('ShaderNodeMath')
    mult_radial.location = (0, -100)
    mult_radial.operation = 'MULTIPLY'

    # 11. Density scaling
    density_scale = nodes.new('ShaderNodeMath')
    density_scale.location = (200, -100)
    density_scale.operation = 'MULTIPLY'
    density_scale.inputs[1].default_value = density

    # 12. Volume Scatter node
    volume_scatter = nodes.new('ShaderNodeVolumeScatter')
    volume_scatter.location = (400, 0)
    volume_scatter.inputs['Anisotropy'].default_value = 0.3  # Slight forward scattering

    # 13. Material Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # Connect nodes
    # Noise texture
    links.new(tex_coord.outputs['Generated'], noise.inputs['Vector'])

    # Vertical gradient with mapping
    links.new(tex_coord.outputs['Generated'], vert_mapping.inputs['Vector'])
    links.new(vert_mapping.outputs['Vector'], vert_gradient.inputs['Vector'])
    links.new(vert_gradient.outputs['Fac'], vert_ramp.inputs['Fac'])

    # Radial gradient with mapping
    links.new(tex_coord.outputs['Generated'], radial_mapping.inputs['Vector'])
    links.new(radial_mapping.outputs['Vector'], radial_gradient.inputs['Vector'])
    links.new(radial_gradient.outputs['Fac'], radial_ramp.inputs['Fac'])

    # Multiply: noise × vertical gradient factor
    links.new(noise.outputs['Fac'], mult_noise_vert.inputs[0])
    links.new(vert_ramp.outputs['Alpha'], mult_noise_vert.inputs[1])

    # Multiply: (noise × vert) × radial gradient for edge fadeout
    links.new(mult_noise_vert.outputs['Value'], mult_radial.inputs[0])
    links.new(radial_ramp.outputs['Alpha'], mult_radial.inputs[1])

    # Scale density
    links.new(mult_radial.outputs['Value'], density_scale.inputs[0])

    # Connect to volume scatter
    links.new(density_scale.outputs['Value'], volume_scatter.inputs['Density'])
    links.new(vert_ramp.outputs['Color'], volume_scatter.inputs['Color'])

    # Connect to output
    links.new(volume_scatter.outputs['Volume'], output.inputs['Volume'])

    return mat


def create_tractor_beam_area_light(name, position, target_position, size, power, is_pull_effect):
    """Create an area light for the tractor beam.

    Args:
        name: Light name
        position: (x, y, z) position of the light
        target_position: (x, y, z) position to point at
        size: Size of the area light
        power: Light power in watts
        is_pull_effect: Determines light placement (at terrain for pull, at sphere for push)

    Returns:
        bpy.types.Object: The light object
    """
    # Create area light
    light_data = bpy.data.lights.new(name=name, type='AREA')
    light_data.energy = power
    light_data.size = size
    light_data.shape = 'DISK'

    # Create light object
    light_obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_obj)

    # Position the light
    light_obj.location = position

    # Point light at target using Track To constraint
    constraint = light_obj.constraints.new(type='TRACK_TO')

    # Create empty target for the light to track
    target_empty = bpy.data.objects.new(f"{name}_Target", None)
    target_empty.location = target_position
    bpy.context.collection.objects.link(target_empty)
    target_empty.hide_render = True
    target_empty.hide_viewport = True

    constraint.target = target_empty
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    return light_obj


def create_tractor_beam(sphere_pos, sphere_radius, terrain_height, prefix, is_high_terrain):
    """Create a complete tractor beam effect with cone, light, and material.

    CORRECTED behavior (user feedback):
    - High terrain = PULL effect (light from terrain up, smoke dense at bottom)
    - Low terrain = PUSH effect (light from sphere down, smoke dense at top)

    Args:
        sphere_pos: (x, y, z) position of the landmark sphere
        sphere_radius: Radius of the sphere
        terrain_height: Height of terrain at this location
        prefix: Landmark type (Lecture1, Lecture2, General)
        is_high_terrain: True for PULL (high terrain), False for PUSH (low terrain)

    Returns:
        tuple: (cone_obj, light_obj) created objects
    """
    x, y, z = sphere_pos

    # Get beam color based on landmark type
    beam_color = TRACTOR_BEAM_COLOR_MAPPING.get(prefix, (0.5, 0.5, 0.5))

    # Calculate cone dimensions:
    # Top: sphere center, radius = sphere radius
    top_center = (x, y, z)  # Sphere center (vertical center)
    top_radius = sphere_radius  # Match sphere circumference at equator

    # Bottom: terrain level
    min_terrain_height = terrain_height
    bottom_center = (x, y, min_terrain_height)

    # Bottom radius is 2x sphere radius (fixed multiplier)
    bottom_radius = sphere_radius * 2

    # Create truncated cone
    cone_name = f"TractorBeam_Cone_{prefix}_{x:.1f}_{y:.1f}"
    cone = create_truncated_cone_mesh(
        name=cone_name,
        top_center=top_center,
        top_radius=top_radius,
        bottom_center=bottom_center,
        bottom_radius=bottom_radius,
        segments=32
    )

    # Determine if this is a PUSH effect (low terrain) or PULL effect (high terrain)
    # CORRECTED: Push/pull were reversed - now fixed
    is_push_effect = not is_high_terrain  # Low terrain = push, High terrain = pull

    # Use unified density for all beams
    density = TRACTOR_BEAM_DENSITY

    # Create volumetric material with appropriate density
    mat_name = f"TractorBeam_Mat_{prefix}_{x:.1f}_{y:.1f}"
    material = create_tractor_beam_volume_material(mat_name, beam_color, is_push_effect, density)

    # Assign material to cone
    cone.data.materials.append(material)

    # Create area light
    light_name = f"TractorBeam_Light_{prefix}_{x:.1f}_{y:.1f}"

    if is_push_effect:
        # PUSH effect (low terrain): Light at sphere pointing DOWN toward terrain
        # CORRECTION: Raise light slightly above terrain to avoid occlusion
        light_pos = (x, y, z)
        target_pos = (x, y, terrain_height)
        light_size = top_radius * 2
    else:
        # PULL effect (high terrain): Light at terrain pointing UP toward sphere
        # Place light slightly above terrain
        light_pos = (x, y, terrain_height + 0.2)
        target_pos = (x, y, z)
        light_size = bottom_radius * 0.5

    light = create_tractor_beam_area_light(
        name=light_name,
        position=light_pos,
        target_position=target_pos,
        size=light_size,
        power=TRACTOR_BEAM_LIGHT_POWER,
        is_pull_effect=not is_push_effect
    )

    return cone, light


if TRACTOR_BEAM_ENABLED:
    tractor_beam_count = 0
    high_terrain_count = 0
    low_terrain_count = 0
    all_cones = []
    all_lights = []

    for lm in all_landmarks:
        x, y, z = lm['pos']
        prefix = lm['prefix']
        radius = lm['radius']

        # Get normalized height at this position to determine if high or low terrain
        xi = int((x / WORLD_SIZE) * (heightmap_norm.shape[0] - 1))
        yi = int((y / WORLD_SIZE) * (heightmap_norm.shape[1] - 1))
        xi = max(0, min(heightmap_norm.shape[0] - 1, xi))
        yi = max(0, min(heightmap_norm.shape[1] - 1, yi))
        local_height_norm = heightmap_norm[xi, yi]

        is_high_terrain = local_height_norm > HEIGHT_PERCENTILE_THRESHOLD

        # Get actual terrain height at this location
        terrain_h = get_height_at(x, y, heightmap_norm, HEIGHT_SCALE, WORLD_SIZE)

        # Create tractor beam (cone + light)
        cone, light = create_tractor_beam(
            sphere_pos=(x, y, z),
            sphere_radius=radius,
            terrain_height=terrain_h,
            prefix=prefix,
            is_high_terrain=is_high_terrain
        )

        all_cones.append(cone)
        all_lights.append(light)
        tractor_beam_count += 1

        if is_high_terrain:
            high_terrain_count += 1
        else:
            low_terrain_count += 1

    print(f"  Created {tractor_beam_count} tractor beam effects")
    print(f"    High terrain (pull up): {high_terrain_count}")
    print(f"    Low terrain (push down): {low_terrain_count}")
    print(f"    Total cones: {len(all_cones)}")
    print(f"    Total lights: {len(all_lights)}")

else:
    print("  Tractor beams disabled")

# ============================================================================
# SET UP ENVIRONMENT - Dark synthwave sky
# ============================================================================
print("Setting up synthwave environment...")

world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()

# Create gradient background
tex_coord = nodes.new('ShaderNodeTexCoord')
tex_coord.location = (-600, 0)

gradient = nodes.new('ShaderNodeTexGradient')
gradient.location = (-400, 0)
gradient.gradient_type = 'LINEAR'

color_ramp = nodes.new('ShaderNodeValToRGB')
color_ramp.location = (-200, 0)
color_ramp.color_ramp.elements[0].color = (0.02, 0.0, 0.08, 1)  # Deep purple at bottom
color_ramp.color_ramp.elements[1].color = (0.05, 0.02, 0.15, 1)  # Dark blue at top
color_ramp.color_ramp.elements[0].position = 0.0
color_ramp.color_ramp.elements[1].position = 0.5

bg = nodes.new('ShaderNodeBackground')
bg.location = (100, 0)
bg.inputs['Strength'].default_value = 0.8

output = nodes.new('ShaderNodeOutputWorld')
output.location = (300, 0)

links.new(tex_coord.outputs['Generated'], gradient.inputs['Vector'])
links.new(gradient.outputs['Color'], color_ramp.inputs['Fac'])
links.new(color_ramp.outputs['Color'], bg.inputs['Color'])
links.new(bg.outputs['Background'], output.inputs['Surface'])

print("Synthwave sky configured")

# ============================================================================
# SET UP DRAMATIC LIGHTING
# ============================================================================
print("Setting up dramatic synthwave lighting...")

# Calculate max visible height for lighting positions (scale with HEIGHT_SCALE)
MAX_LIGHT_HEIGHT = HEIGHT_SCALE * 1.5  # Lights positioned relative to tall prisms

# Main cyan rim light from behind
bpy.ops.object.light_add(type='AREA', location=(WORLD_SIZE/2, WORLD_SIZE * 1.2, MAX_LIGHT_HEIGHT))
cyan_light = bpy.context.active_object
cyan_light.name = "CyanRimLight"
cyan_light.data.energy = 2000  # Scaled for scene with 1-100 inch prisms
cyan_light.data.color = SYNTHWAVE_CYAN
cyan_light.data.size = WORLD_SIZE * 0.5
cyan_light.rotation_euler = (math.radians(60), 0, 0)

# Magenta key light from side
bpy.ops.object.light_add(type='AREA', location=(WORLD_SIZE * -0.2, WORLD_SIZE/2, MAX_LIGHT_HEIGHT))
magenta_light = bpy.context.active_object
magenta_light.name = "MagentaKeyLight"
magenta_light.data.energy = 1200  # Scaled for scene with 1-100 inch prisms
magenta_light.data.color = SYNTHWAVE_MAGENTA
magenta_light.data.size = WORLD_SIZE * 0.3
magenta_light.rotation_euler = (math.radians(45), math.radians(-45), 0)

# Purple fill light from opposite side
bpy.ops.object.light_add(type='AREA', location=(WORLD_SIZE * 1.2, WORLD_SIZE/2, MAX_LIGHT_HEIGHT * 0.7))
purple_light = bpy.context.active_object
purple_light.name = "PurpleFillLight"
purple_light.data.energy = 800  # Scaled for scene with 1-100 inch prisms
purple_light.data.color = SYNTHWAVE_PURPLE
purple_light.data.size = WORLD_SIZE * 0.4
purple_light.rotation_euler = (math.radians(30), math.radians(45), 0)

# Top down soft white light for overall visibility
bpy.ops.object.light_add(type='AREA', location=(WORLD_SIZE/2, WORLD_SIZE/2, MAX_LIGHT_HEIGHT * 2))
top_light = bpy.context.active_object
top_light.name = "TopLight"
top_light.data.energy = 400  # Scaled for scene with 1-100 inch prisms
top_light.data.color = (0.9, 0.9, 1.0)
top_light.data.size = WORLD_SIZE
top_light.rotation_euler = (0, 0, 0)

# Point lights for accent on landscape features
for i in range(3):
    x = WORLD_SIZE * (0.3 + i * 0.2)
    y = WORLD_SIZE * 0.5
    bpy.ops.object.light_add(type='POINT', location=(x, y, MAX_LIGHT_HEIGHT * 0.4))
    accent = bpy.context.active_object
    accent.name = f"AccentLight_{i}"
    accent.data.energy = 20  # Scaled for scene with 1-100 inch prisms
    accent.data.color = [SYNTHWAVE_CYAN, SYNTHWAVE_MAGENTA, SYNTHWAVE_PINK][i]

print("Dramatic lighting configured")

# ============================================================================
# SET UP CAMERA
# ============================================================================
print("Setting up camera...")

# Camera settings extracted from terrain_scene_edit.blend (user's manual edits)
# Updated 2025-12-03 with latest camera position
camera_location = (15.063646, 13.266693, 15.137848)  # Updated from terrain_scene_edit.blend
target_location = (7.614984, 7.037800, 0.476730)

print(f"  Camera location (from new_camera_settings.blend): {camera_location}")
print(f"  Target location (from new_camera_settings.blend): {target_location}")

bpy.ops.object.camera_add(location=camera_location)
camera = bpy.context.active_object
camera.name = "MainCamera"

bpy.ops.object.empty_add(type='PLAIN_AXES', location=target_location)
target_empty = bpy.context.active_object
target_empty.name = "CameraTarget"

bpy.context.view_layer.objects.active = camera
track_constraint = camera.constraints.new(type='TRACK_TO')
track_constraint.target = target_empty
track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
track_constraint.up_axis = 'UP_Y'

# Camera settings from terrain_scene_edit.blend (updated 2025-12-03)
camera.data.lens = 36.0  # From terrain_scene_edit.blend
camera.data.clip_start = 0.01  # ~0.12 inches (very close)
camera.data.clip_end = 83.33  # From terrain_scene_edit.blend
camera.data.sensor_width = 23.0  # From terrain_scene_edit.blend

# Depth of field disabled per user request (from terrain_scene_edit.blend)
camera.data.dof.use_dof = False
camera.data.dof.focus_distance = 10.0  # Updated from edited scene
camera.data.dof.aperture_fstop = 0.5  # Kept for reference if DOF re-enabled

print(f"Camera settings (from new_camera_settings.blend):")
print(f"  Focal length: {camera.data.lens}mm")
print(f"  Focus distance: {camera.data.dof.focus_distance:.2f} feet")
print(f"  Aperture: f/{camera.data.dof.aperture_fstop}")

bpy.context.scene.camera = camera

# ============================================================================
# RENDER SETTINGS
# ============================================================================
print("Configuring render settings...")

scene = bpy.context.scene

scene.render.engine = RENDER_ENGINE

if RENDER_ENGINE == 'CYCLES':
    prefs = bpy.context.preferences.addons['cycles'].preferences

    # Detect compute device type based on platform
    import sys
    if sys.platform == 'darwin':
        # macOS - use Metal
        prefs.compute_device_type = 'METAL'
        print("Using Metal GPU acceleration (macOS)")
    else:
        # Linux/Windows - try CUDA first, then OptiX, then NONE
        try:
            prefs.compute_device_type = 'CUDA'
            print("Using CUDA GPU acceleration")
        except:
            try:
                prefs.compute_device_type = 'OPTIX'
                print("Using OptiX GPU acceleration")
            except:
                prefs.compute_device_type = 'NONE'
                print("No GPU acceleration available, using CPU")

    # Refresh and enable all available devices
    prefs.get_devices()
    enabled_devices = []
    for device in prefs.devices:
        device.use = True
        if device.type != 'CPU':
            enabled_devices.append(f"{device.name} ({device.type})")

    if enabled_devices:
        print(f"Enabled GPU devices: {', '.join(enabled_devices)}")
    else:
        print("No GPU devices found, rendering will use CPU")

    scene.cycles.device = 'GPU'
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.use_denoising = USE_DENOISING
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'

    # Higher quality for glass and reflections
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 8
    scene.cycles.transmission_bounces = 12
    scene.cycles.transparent_max_bounces = 8
else:
    # EEVEE settings for fast preview - minimal settings for Blender 5.0
    # Just use defaults - EEVEE is fast out of the box
    pass

scene.render.resolution_x = RENDER_WIDTH
scene.render.resolution_y = RENDER_HEIGHT
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.filepath = OUTPUT_FILE

scene.render.film_transparent = False
if RENDER_ENGINE == 'CYCLES':
    scene.cycles.film_exposure = 1.0  # Normal exposure to prevent blowouts

scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look = 'Very High Contrast'

# ============================================================================
# COMPOSITING - Bloom/Glare for neon glow
# ============================================================================
print("Setting up compositing with bloom...")

try:
    if hasattr(scene, 'compositing_node_group'):
        if scene.compositing_node_group is None:
            ng = bpy.data.node_groups.new(name="Compositing", type='CompositorNodeTree')
            scene.compositing_node_group = ng
        tree = scene.compositing_node_group
    else:
        if hasattr(scene, 'use_nodes'):
            scene.use_nodes = True
        tree = getattr(scene, 'node_tree', None)

    if tree:
        nodes = tree.nodes
        links = tree.links

        for node in list(nodes):
            nodes.remove(node)

        render_layers = nodes.new('CompositorNodeRLayers')
        render_layers.location = (0, 0)

        # Glare for bloom effect
        glare = nodes.new('CompositorNodeGlare')
        glare.location = (300, 0)
        try:
            glare.glare_type = 'FOG_GLOW'
        except:
            pass  # Attribute may not exist in all versions
        glare.quality = 'HIGH'
        glare.mix = -0.9  # Strong bloom
        glare.threshold = 0.5
        glare.size = 9

        # Color correction for synthwave look
        color_balance = nodes.new('CompositorNodeColorBalance')
        color_balance.location = (500, 0)
        color_balance.correction_method = 'LIFT_GAMMA_GAIN'
        # Add subtle purple tint to shadows
        color_balance.lift = (0.95, 0.9, 1.0)
        # Boost midtones slightly cyan
        color_balance.gamma = (0.95, 1.0, 1.05)

        composite = nodes.new('CompositorNodeComposite')
        composite.location = (800, 0)

        links.new(render_layers.outputs['Image'], glare.inputs['Image'])
        links.new(glare.outputs['Image'], color_balance.inputs['Image'])
        links.new(color_balance.outputs['Image'], composite.inputs['Image'])
        print("Compositing with bloom configured")
except Exception as e:
    print(f"Warning: Could not set up compositing: {e}")

print(f"Render: {RENDER_WIDTH}x{RENDER_HEIGHT}, {RENDER_SAMPLES} samples")

# ============================================================================
# SAVE SCENE FILE (for future cache use)
# ============================================================================
SCENE_FILE = os.path.join(os.path.dirname(OUTPUT_FILE), "rendered_scene.blend")
try:
    bpy.ops.wm.save_as_mainfile(filepath=SCENE_FILE)
    print(f"Scene saved to: {SCENE_FILE}")
    print("  (Future runs will load from cache unless --force-rebuild is used)")
except Exception as e:
    print(f"Warning: Could not save scene file: {e}")

# ============================================================================
# RENDER
# ============================================================================
print("Starting render...")
bpy.ops.render.render(write_still=True)

print(f"Render complete! Saved to: {OUTPUT_FILE}")
