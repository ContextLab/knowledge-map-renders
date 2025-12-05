#!/usr/bin/env python3
"""
Shared geometry module for knowledge landscape visualization.

This module contains all sizing parameters and geometry creation functions
used by both blender_render.py and create_camera_settings.py to ensure
consistent scene geometry.

All dimensions are in feet unless otherwise noted.
"""

import numpy as np
import os

# ============================================================================
# DIRECTORY PATHS
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MATERIALS_DIR = os.path.join(PROJECT_ROOT, "materials")
CAMERA_SETTINGS_BLEND = os.path.join(DATA_DIR, "camera_settings.blend")

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

# ============================================================================
# HEIGHT SCALE SETTINGS
# ============================================================================
# Prism heights: 1 to 50 inches (reduced by 50% for flatter terrain)
# In feet: 1 inch = 1/12 ft, 50 inches = 50/12 ft = 4.17 ft
HEIGHT_SCALE_MIN = 1.0 / 12.0  # 1 inch minimum height in feet
HEIGHT_SCALE_MAX = 50.0 / 12.0  # 50 inches maximum height in feet (~4.17 ft)
HEIGHT_SCALE = HEIGHT_SCALE_MAX - HEIGHT_SCALE_MIN  # Range of 49 inches

# ============================================================================
# ROAD/TRAJECTORY PARAMETERS
# ============================================================================
# Road parameters (for glass tubes) - 2 inches wide
ROAD_WIDTH = 2.0 / 12.0  # 2 inches in feet (~0.167 ft)
SIDEROAD_WIDTH = 2.0 / 12.0  # Same width for sideroad

# Trajectory height offsets - tubes float 12 inches above the local prism height
HIGHWAY_HEIGHT_OFFSET = 12.0 / 12.0  # 12 inches above local prism top (1 foot)
SIDEROAD_HEIGHT_OFFSET = 12.0 / 12.0  # Same for sideroad

# ============================================================================
# LANDMARK PARAMETERS
# ============================================================================
LANDMARK_RADIUS = 0.175  # 2.1 inches in feet
LANDMARK_FLOAT_HEIGHT = 1.5  # 1.5 feet above terrain to sphere center

# ============================================================================
# GRID PARAMETERS
# ============================================================================
SAMPLE_RATE = 4  # Sample every 4th point from heightmap for prisms
WIREFRAME_THICKNESS = 0.02  # Thickness for wireframe grid lines
MINOR_GRID_ENABLED = True
MINOR_GRID_SUBDIVISIONS = 5  # 5x finer than major grid

# ============================================================================
# RENDER SETTINGS
# ============================================================================
RENDER_WIDTH = 1200
RENDER_HEIGHT = 675

# ============================================================================
# COLOR PALETTE
# ============================================================================
SYNTHWAVE_CYAN = (0.0, 0.9, 1.0)
SYNTHWAVE_MAGENTA = (1.0, 0.0, 0.8)
SYNTHWAVE_PURPLE = (0.6, 0.0, 1.0)
SYNTHWAVE_PINK = (1.0, 0.2, 0.6)
SYNTHWAVE_ORANGE = (1.0, 0.4, 0.0)
SYNTHWAVE_DARK_BLUE = (0.05, 0.02, 0.15)

# Simple colors for camera_settings.blend preview
COLOR_TERRAIN = (0.05, 0.02, 0.15, 1.0)  # Dark blue
COLOR_HIGHWAY = (0.0, 0.9, 1.0, 1.0)      # Cyan
COLOR_SIDEROAD = (1.0, 0.0, 0.8, 1.0)     # Magenta
COLOR_LECTURE1 = (0.8, 0.8, 1.0, 1.0)     # Light blue
COLOR_LECTURE2 = (0.6, 0.6, 0.8, 1.0)     # Gray blue
COLOR_GENERAL = (0.3, 0.3, 0.5, 1.0)      # Dark gray blue


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_heightmap(data_dir=None):
    """Load and normalize the heightmap data.

    Returns:
        tuple: (heightmap_norm_original, h_min, h_max)
    """
    if data_dir is None:
        data_dir = DATA_DIR

    heightmap_original = np.load(os.path.join(data_dir, 'knowledge-heatmap-quiz2.npy'))
    h_min, h_max = heightmap_original.min(), heightmap_original.max()
    heightmap_norm = (heightmap_original - h_min) / (h_max - h_min)
    return heightmap_norm, h_min, h_max


def extend_heightmap(heightmap_norm_original):
    """Extend heightmap to 2x size with smooth falloff at edges.

    Args:
        heightmap_norm_original: Normalized heightmap (0-1 range)

    Returns:
        Extended heightmap (2x size in each dimension)
    """
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

    # Apply simple blur
    heightmap_norm = simple_blur(heightmap_norm, kernel_size=5)

    return heightmap_norm


def simple_blur(arr, kernel_size=5):
    """Apply simple box blur to array."""
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    padded = np.pad(arr, kernel_size // 2, mode='edge')
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i, j] = np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel)
    return result


def load_trajectories(data_dir=None):
    """Load trajectory coordinate data.

    Returns:
        tuple: (highway_coords, sideroad_coords) - both in world coordinates
    """
    if data_dir is None:
        data_dir = DATA_DIR

    highway_coords = np.load(os.path.join(data_dir, 'lecture1-trajectory-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
    sideroad_coords = np.load(os.path.join(data_dir, 'lecture2-trajectory-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
    return highway_coords, sideroad_coords


def load_landmarks(data_dir=None):
    """Load landmark coordinate data.

    Returns:
        tuple: (lecture1_landmarks, lecture2_landmarks, general_landmarks) - all in world coordinates
    """
    if data_dir is None:
        data_dir = DATA_DIR

    lecture1_landmarks = np.load(os.path.join(data_dir, 'lecture1-questions-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
    lecture2_landmarks = np.load(os.path.join(data_dir, 'lecture2-questions-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
    general_landmarks = np.load(os.path.join(data_dir, 'general-knowledge-questions-shifted.npy')) * SCALE_FACTOR + WORLD_OFFSET
    return lecture1_landmarks, lecture2_landmarks, general_landmarks


# ============================================================================
# HEIGHT CALCULATION FUNCTIONS
# ============================================================================
def get_terrain_height_at_index(ix, iy, heightmap_norm):
    """Get terrain height at heightmap indices.

    Args:
        ix, iy: Integer indices into heightmap
        heightmap_norm: Normalized heightmap array

    Returns:
        Height in feet
    """
    nx, ny = heightmap_norm.shape
    if 0 <= ix < nx and 0 <= iy < ny:
        return HEIGHT_SCALE_MIN + heightmap_norm[ix, iy] * HEIGHT_SCALE
    return HEIGHT_SCALE_MIN


def get_height_at(x, y, heightmap_norm):
    """Get interpolated terrain height at world coordinates.

    Args:
        x, y: World coordinates in feet
        heightmap_norm: Normalized heightmap array

    Returns:
        Interpolated height in feet
    """
    nx, ny = heightmap_norm.shape

    # Convert world coordinates to heightmap indices
    hx = (x / WORLD_SIZE) * nx
    hy = (y / WORLD_SIZE) * ny

    # Clamp to valid range
    hx = max(0, min(nx - 1.001, hx))
    hy = max(0, min(ny - 1.001, hy))

    ix, iy = int(hx), int(hy)
    fx, fy = hx - ix, hy - iy

    # Bilinear interpolation
    h00 = heightmap_norm[ix, iy]
    h10 = heightmap_norm[min(ix + 1, nx - 1), iy]
    h01 = heightmap_norm[ix, min(iy + 1, ny - 1)]
    h11 = heightmap_norm[min(ix + 1, nx - 1), min(iy + 1, ny - 1)]

    h = (h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) +
         h01 * (1 - fx) * fy + h11 * fx * fy)

    return HEIGHT_SCALE_MIN + h * HEIGHT_SCALE


def get_max_neighbor_height(i, j, nx, ny, sample_rate, heightmap_norm):
    """Get the maximum height from all neighboring prisms around a grid corner point.

    This ensures the wireframe grid hugs the 'top layer' of prisms and never dips
    below the top of any neighboring prism during downward slopes.
    """
    max_height = 0.0

    for di in range(-sample_rate, sample_rate + 1, sample_rate):
        for dj in range(-sample_rate, sample_rate + 1, sample_rate):
            ni = i + di
            nj = j + dj
            if 0 <= ni < nx and 0 <= nj < ny:
                height_value = heightmap_norm[ni, nj]
                max_height = max(max_height, height_value)

    return max_height


# ============================================================================
# COORDINATE TRANSFORMATION FUNCTIONS
# ============================================================================
def heightmap_to_world(ix, iy, nx, ny):
    """Convert heightmap indices to world coordinates.

    Args:
        ix, iy: Heightmap indices
        nx, ny: Heightmap dimensions

    Returns:
        tuple: (x, y) world coordinates in feet
    """
    x = (ix / nx) * WORLD_SIZE
    y = (iy / ny) * WORLD_SIZE
    return x, y


def world_to_heightmap(x, y, nx, ny):
    """Convert world coordinates to heightmap indices.

    Args:
        x, y: World coordinates in feet
        nx, ny: Heightmap dimensions

    Returns:
        tuple: (ix, iy) heightmap indices (floats for interpolation)
    """
    ix = (x / WORLD_SIZE) * nx
    iy = (y / WORLD_SIZE) * ny
    return ix, iy


# ============================================================================
# PRISM GEOMETRY FUNCTIONS
# ============================================================================
def get_prism_size(sample_rate, nx):
    """Calculate prism dimensions.

    Args:
        sample_rate: How many heightmap points per prism
        nx: Heightmap width

    Returns:
        tuple: (cell_world_size, prism_size, half_size)
    """
    cell_world_size = WORLD_SIZE / nx
    prism_size = cell_world_size * sample_rate
    half_size = prism_size / 2.0
    return cell_world_size, prism_size, half_size


def get_prism_center(ix, iy, nx, ny, sample_rate):
    """Calculate prism center position.

    Args:
        ix, iy: Heightmap indices (top-left corner of prism)
        nx, ny: Heightmap dimensions
        sample_rate: How many heightmap points per prism

    Returns:
        tuple: (cx, cy) center coordinates in feet
    """
    x, y = heightmap_to_world(ix, iy, nx, ny)
    _, _, half_size = get_prism_size(sample_rate, nx)
    cx = x + half_size
    cy = y + half_size
    return cx, cy


# ============================================================================
# TRAJECTORY SMOOTHING FUNCTIONS
# ============================================================================
def smooth_trajectory(coords, num_interp_per_segment=10):
    """Interpolate trajectory with Catmull-Rom spline for perfectly smooth curves."""
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

    def catmull_rom_interp(t_vals, t_knots, values):
        """Centripetal Catmull-Rom spline interpolation"""
        result = np.zeros(len(t_vals))

        # Extend control points for end conditions
        p = np.zeros(n + 2)
        p[1:-1] = values
        p[0] = 2 * values[0] - values[1]
        p[-1] = 2 * values[-1] - values[-2]

        t_ext = np.zeros(n + 2)
        t_ext[1:-1] = t_knots
        t_ext[0] = t_knots[0] - (t_knots[1] - t_knots[0])
        t_ext[-1] = t_knots[-1] + (t_knots[-1] - t_knots[-2])

        for idx, t in enumerate(t_vals):
            seg = np.searchsorted(t_knots[:-1], t, side='right') - 1
            seg = max(0, min(seg, n - 2))

            i = seg + 1
            p0, p1, p2, p3 = p[i-1], p[i], p[i+1], p[i+2]
            t0, t1, t2, t3 = t_ext[i-1], t_ext[i], t_ext[i+1], t_ext[i+2]

            if t2 - t1 > 0:
                u = (t - t1) / (t2 - t1)
            else:
                u = 0

            u2 = u * u
            u3 = u2 * u

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


# ============================================================================
# CAMERA SETTINGS FUNCTIONS
# ============================================================================
def load_camera_settings_from_blend(blend_path=None):
    """Load camera settings from camera_settings.blend file.

    This function must be called from within Blender Python.

    Args:
        blend_path: Path to camera_settings.blend (defaults to DATA_DIR/camera_settings.blend)

    Returns:
        dict with camera settings or None if file doesn't exist
    """
    if blend_path is None:
        blend_path = CAMERA_SETTINGS_BLEND

    if not os.path.exists(blend_path):
        print(f"Warning: Camera settings file not found: {blend_path}")
        return None

    # Import bpy here since this function is only called from Blender
    import bpy

    # Load the blend file data without replacing current scene
    settings = {}

    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        # We need to load objects to get camera data
        data_to.objects = [name for name in data_from.objects if 'Camera' in name or 'Target' in name]

    # Find camera and target in loaded objects
    camera_obj = None
    target_obj = None

    for obj in data_to.objects:
        if obj is None:
            continue
        if obj.type == 'CAMERA':
            camera_obj = obj
        elif obj.type == 'EMPTY' and 'Target' in obj.name:
            target_obj = obj

    if camera_obj:
        settings['camera_location'] = tuple(camera_obj.location)
        settings['lens'] = camera_obj.data.lens
        settings['sensor_width'] = camera_obj.data.sensor_width
        settings['clip_start'] = camera_obj.data.clip_start
        settings['clip_end'] = camera_obj.data.clip_end
        settings['use_dof'] = camera_obj.data.dof.use_dof
        settings['focus_distance'] = camera_obj.data.dof.focus_distance
        settings['aperture_fstop'] = camera_obj.data.dof.aperture_fstop

        # Clean up - remove loaded camera
        bpy.data.cameras.remove(camera_obj.data)

    if target_obj:
        settings['target_location'] = tuple(target_obj.location)
        # Clean up - remove loaded empty
        bpy.data.objects.remove(target_obj)

    return settings if settings else None


def read_camera_settings_directly(blend_path=None):
    """Read camera settings by opening the blend file directly.

    This is an alternative method that opens the file to read settings,
    then restores the previous state.

    Args:
        blend_path: Path to camera_settings.blend

    Returns:
        dict with camera settings
    """
    if blend_path is None:
        blend_path = CAMERA_SETTINGS_BLEND

    if not os.path.exists(blend_path):
        print(f"Warning: Camera settings file not found: {blend_path}")
        return get_default_camera_settings()

    import bpy

    # Save current file path to restore later
    current_filepath = bpy.data.filepath

    # Open camera settings file
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    settings = {}
    camera = bpy.context.scene.camera

    if camera:
        settings['camera_location'] = tuple(camera.location)
        settings['lens'] = camera.data.lens
        settings['sensor_width'] = camera.data.sensor_width
        settings['clip_start'] = camera.data.clip_start
        settings['clip_end'] = camera.data.clip_end
        settings['use_dof'] = camera.data.dof.use_dof
        settings['focus_distance'] = camera.data.dof.focus_distance
        settings['aperture_fstop'] = camera.data.dof.aperture_fstop

        # Find target through constraints
        for constraint in camera.constraints:
            if constraint.type == 'TRACK_TO' and constraint.target:
                settings['target_location'] = tuple(constraint.target.location)
                break

    return settings if settings else get_default_camera_settings()


def get_default_camera_settings():
    """Return default camera settings if blend file is not available."""
    return {
        'camera_location': (17.412048, 13.900706, 15.414385),
        'target_location': (7.604351, 7.944798, 0.414794),
        'lens': 33.0,
        'sensor_width': 23.0,
        'clip_start': 0.01,
        'clip_end': 83.33,
        'use_dof': False,
        'focus_distance': 10.0,
        'aperture_fstop': 0.5,
    }


# ============================================================================
# WIREFRAME GRID GENERATION
# ============================================================================
def create_wireframe_grid_lines(heightmap_norm, sample_rate=None):
    """Create wireframe grid vertices and edges at prism corners.

    Args:
        heightmap_norm: Normalized heightmap array
        sample_rate: Grid sampling rate (defaults to SAMPLE_RATE)

    Returns:
        tuple: (vertices, edges) where vertices is list of (x,y,z) tuples
               and edges is list of (v1_idx, v2_idx) tuples
    """
    if sample_rate is None:
        sample_rate = SAMPLE_RATE

    vertices = []
    edges = []

    nx, ny = heightmap_norm.shape
    vertex_index = 0

    # Create horizontal lines (along X axis) for each row
    for j in range(0, ny, sample_rate):
        row_verts = []
        y = (j / ny) * WORLD_SIZE

        for i in range(0, nx, sample_rate):
            x = (i / nx) * WORLD_SIZE
            max_height_value = get_max_neighbor_height(i, j, nx, ny, sample_rate, heightmap_norm)
            z = HEIGHT_SCALE_MIN + max_height_value * HEIGHT_SCALE

            vertices.append((x, y, z))
            row_verts.append(vertex_index)

            if len(row_verts) > 1:
                edges.append((row_verts[-2], row_verts[-1]))

            vertex_index += 1

    # Create vertical lines (along Y axis)
    grid_cols = (nx + sample_rate - 1) // sample_rate
    grid_rows = (ny + sample_rate - 1) // sample_rate

    for i in range(grid_cols):
        for j in range(grid_rows - 1):
            idx_current = j * grid_cols + i
            idx_next = (j + 1) * grid_cols + i
            if idx_current < len(vertices) and idx_next < len(vertices):
                edges.append((idx_current, idx_next))

    return vertices, edges


def create_minor_grid_lines(heightmap_norm, sample_rate=None, subdivisions=None):
    """Create minor grid lines that subdivide each major grid cell.

    Args:
        heightmap_norm: Normalized heightmap array
        sample_rate: Major grid sampling rate
        subdivisions: Number of subdivisions per major cell

    Returns:
        tuple: (vertices, edges)
    """
    if sample_rate is None:
        sample_rate = SAMPLE_RATE
    if subdivisions is None:
        subdivisions = MINOR_GRID_SUBDIVISIONS

    vertices = []
    edges = []

    nx, ny = heightmap_norm.shape
    vertex_index = 0

    minor_step = sample_rate // subdivisions
    if minor_step < 1:
        minor_step = 1

    vertex_grid = {}

    # Create horizontal lines for minor grid
    for j in range(0, ny, minor_step):
        if j % sample_rate == 0:
            continue

        row_verts = []
        y = (j / ny) * WORLD_SIZE

        for i in range(0, nx, minor_step):
            x = (i / nx) * WORLD_SIZE
            max_height_value = get_max_neighbor_height(i, j, nx, ny, sample_rate, heightmap_norm)
            z = HEIGHT_SCALE_MIN + max_height_value * HEIGHT_SCALE

            vertices.append((x, y, z))
            vertex_grid[(i, j)] = vertex_index
            row_verts.append(vertex_index)

            if len(row_verts) > 1:
                edges.append((row_verts[-2], row_verts[-1]))

            vertex_index += 1

    # Create vertical lines for minor grid
    for i in range(0, nx, minor_step):
        if i % sample_rate == 0:
            continue

        col_verts = []
        x = (i / nx) * WORLD_SIZE

        for j in range(0, ny, minor_step):
            y = (j / ny) * WORLD_SIZE
            max_height_value = get_max_neighbor_height(i, j, nx, ny, sample_rate, heightmap_norm)
            z = HEIGHT_SCALE_MIN + max_height_value * HEIGHT_SCALE

            if (i, j) in vertex_grid:
                col_verts.append(vertex_grid[(i, j)])
            else:
                vertices.append((x, y, z))
                vertex_grid[(i, j)] = vertex_index
                col_verts.append(vertex_index)
                vertex_index += 1

            if len(col_verts) > 1:
                edges.append((col_verts[-2], col_verts[-1]))

    return vertices, edges


# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================
def get_trajectory_points(coords, heightmap_norm, height_offset, tube_radius, downsample_factor=5):
    """Generate smoothed 3D trajectory points.

    Args:
        coords: 2D trajectory coordinates (N x 2 array)
        heightmap_norm: Normalized heightmap array
        height_offset: Height above terrain for trajectory
        tube_radius: Tube radius for height sampling
        downsample_factor: Factor to reduce control points

    Returns:
        list of (x, y, z) tuples for trajectory points
    """
    # Downsample coordinates
    downsampled_coords = coords[::downsample_factor]

    # Ensure we include the last point
    if not np.array_equal(downsampled_coords[-1], coords[-1]):
        downsampled_coords = np.vstack([downsampled_coords, coords[-1]])

    # Apply Catmull-Rom smoothing
    smooth_coords = smooth_trajectory(downsampled_coords, num_interp_per_segment=2)

    # Get heights with local maximum sampling
    def get_max_local_height(x, y, radius):
        offsets = [-radius, -radius/2, 0, radius/2, radius]
        max_h = 0.0
        for dx in offsets:
            for dy in offsets:
                h = get_height_at(x + dx, y + dy, heightmap_norm)
                max_h = max(max_h, h)
        return max_h

    heights = np.array([get_max_local_height(c[0], c[1], tube_radius * 2)
                       for c in smooth_coords])

    # Smooth heights
    kernel_size = 11
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_heights = np.convolve(heights, kernel, mode='same')
    heights = np.maximum(heights, smoothed_heights)

    # Generate 3D points
    points = []
    for coord, h in zip(smooth_coords, heights):
        points.append((coord[0], coord[1], h + height_offset))

    return points


# ============================================================================
# LANDMARK POSITIONING
# ============================================================================
def get_landmark_positions(landmarks_2d, heightmap_norm, float_height=None):
    """Calculate 3D positions for landmarks.

    Args:
        landmarks_2d: 2D landmark coordinates (N x 2 array)
        heightmap_norm: Normalized heightmap array
        float_height: Height above terrain (defaults to LANDMARK_FLOAT_HEIGHT)

    Returns:
        list of (x, y, z) tuples
    """
    if float_height is None:
        float_height = LANDMARK_FLOAT_HEIGHT

    positions = []
    for x, y in landmarks_2d:
        z = get_height_at(x, y, heightmap_norm) + float_height
        positions.append((x, y, z))

    return positions
