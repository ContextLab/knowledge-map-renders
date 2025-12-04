#!/usr/bin/env python3
"""
3D Terrain Rendering with Roads and Landmarks
Creates a realistic, cinematic topographic map visualization with:
- Majestic terrain from height map
- Highway and side road ribbons with proper scale
- Futuristic landmark markers
- Grid overlay
- Immersive car's eye view camera
- PBR textures for realistic materials
"""

import numpy as np
import pyvista as pv
from PIL import Image
import os

# Set up off-screen rendering
pv.OFF_SCREEN = True

# Texture paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
MATERIALS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'materials')
TERRAIN_TEX = os.path.join(MATERIALS_DIR, 'aerial-rocks-diff.jpg')
HIGHWAY_TEX = os.path.join(MATERIALS_DIR, 'asphalt-diff.jpg')
SIDEROAD_TEX = os.path.join(MATERIALS_DIR, 'rocky-trail-diff.jpg')
LANDMARK1_TEX = os.path.join(MATERIALS_DIR, 'green-rust-diff.jpg')
LANDMARK2_TEX = os.path.join(MATERIALS_DIR, 'rusty-metal-03-diff.jpg')
LANDMARK3_TEX = os.path.join(MATERIALS_DIR, 'rusty-metal-diff.jpg')

# Load all data
print("Loading data...")
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
heightmap = np.load(os.path.join(DATA_DIR, 'knowledge-heatmap-quiz2.npy'))
highway_coords = np.load(os.path.join(DATA_DIR, 'lecture1-trajectory-shifted.npy'))
sideroad_coords = np.load(os.path.join(DATA_DIR, 'lecture2-trajectory-shifted.npy'))
lecture1_landmarks = np.load(os.path.join(DATA_DIR, 'lecture1-questions-shifted.npy'))
lecture2_landmarks = np.load(os.path.join(DATA_DIR, 'lecture2-questions-shifted.npy'))
general_landmarks = np.load(os.path.join(DATA_DIR, 'general-knowledge-questions-shifted.npy'))

print(f"Heightmap shape: {heightmap.shape}")
print(f"Highway points: {len(highway_coords)}")
print(f"Side road points: {len(sideroad_coords)}")

# ============================================================================
# SCALE PARAMETERS - Key to realistic appearance
# ============================================================================
# Coordinate system: 0-100 units in X and Y
# We want the terrain to feel vast and the roads to feel like real roads

WORLD_SCALE = 1.0  # Keep coordinates as-is (0-100)
HEIGHT_SCALE = 15.0  # More subtle height variation for realism
ROAD_WIDTH_HIGHWAY = 0.8  # Narrow roads relative to vast terrain
ROAD_WIDTH_SIDEROAD = 0.5
LANDMARK_SCALE = 0.25  # Small, subtle landmarks

# Create the terrain mesh
print("Creating terrain mesh...")
nx, ny = heightmap.shape

# Create grid coordinates
x = np.linspace(0, 100 * WORLD_SCALE, nx)
y = np.linspace(0, 100 * WORLD_SCALE, ny)
x_grid, y_grid = np.meshgrid(x, y)

# Scale height - normalize to 0-1 first then apply scale
height_normalized = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
z_grid = height_normalized.T * HEIGHT_SCALE

# Create structured grid for terrain
terrain = pv.StructuredGrid(x_grid, y_grid, z_grid)
terrain['elevation'] = z_grid.ravel()

# Add texture coordinates for terrain (tile the texture for detail)
terrain_tcoords = np.zeros((terrain.n_points, 2))
terrain_tcoords[:, 0] = (x_grid.ravel() / (100 * WORLD_SCALE)) * 8  # Tile 8x
terrain_tcoords[:, 1] = (y_grid.ravel() / (100 * WORLD_SCALE)) * 8
terrain.active_texture_coordinates = terrain_tcoords

# Load terrain texture
print("Loading textures...")
terrain_texture = pv.read_texture(TERRAIN_TEX)

# Function to interpolate height at given x, y coordinates
def get_height_at(coords, heightmap, height_scale, world_scale=1.0):
    """Interpolate height values at given x, y coordinates"""
    heights = []
    nx, ny = heightmap.shape
    h_min, h_max = heightmap.min(), heightmap.max()

    for coord in coords:
        # Convert coordinates to grid indices
        xi = coord[0] / (100 * world_scale) * (nx - 1)
        yi = coord[1] / (100 * world_scale) * (ny - 1)

        # Bilinear interpolation
        x0, y0 = int(np.floor(xi)), int(np.floor(yi))
        x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)

        xf, yf = xi - x0, yi - y0

        z00 = (heightmap[x0, y0] - h_min) / (h_max - h_min)
        z01 = (heightmap[x0, y1] - h_min) / (h_max - h_min)
        z10 = (heightmap[x1, y0] - h_min) / (h_max - h_min)
        z11 = (heightmap[x1, y1] - h_min) / (h_max - h_min)

        z = (z00 * (1 - xf) * (1 - yf) +
             z10 * xf * (1 - yf) +
             z01 * (1 - xf) * yf +
             z11 * xf * yf)

        heights.append(z * height_scale)

    return np.array(heights)

# Create road ribbon function with proper texture coordinates
def create_road_ribbon(coords, heightmap, height_scale, width=0.5, height_offset=0.05, world_scale=1.0):
    """Create a ribbon mesh following the road path with texture coordinates"""
    heights = get_height_at(coords, heightmap, height_scale, world_scale)

    # Create 3D coordinates with height
    points_3d = np.column_stack([coords[:, 0], coords[:, 1], heights + height_offset])

    # Calculate perpendicular vectors for ribbon width
    tangents = np.zeros_like(points_3d)
    tangents[:-1] = points_3d[1:] - points_3d[:-1]
    tangents[-1] = tangents[-2]

    # Normalize tangents
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tangents = tangents / norms

    # Create perpendicular vectors (cross with up vector)
    up = np.array([0, 0, 1])
    perps = np.cross(tangents, up)
    perp_norms = np.linalg.norm(perps, axis=1, keepdims=True)
    perp_norms[perp_norms == 0] = 1
    perps = perps / perp_norms * width / 2

    # Create ribbon vertices
    left_points = points_3d + perps
    right_points = points_3d - perps

    # Create mesh from ribbon
    n_points = len(coords)
    vertices = np.vstack([left_points, right_points])

    # Create faces (quads as triangles)
    faces = []
    for i in range(n_points - 1):
        l0, l1 = i, i + 1
        r0, r1 = i + n_points, i + 1 + n_points
        faces.append([3, l0, l1, r0])
        faces.append([3, l1, r1, r0])

    faces = np.array(faces).ravel()
    ribbon = pv.PolyData(vertices, faces)

    # Add texture coordinates
    tcoords = np.zeros((len(vertices), 2))
    tcoords[:n_points, 0] = 0.0
    tcoords[n_points:, 0] = 1.0
    cumulative_dist = np.zeros(n_points)
    for i in range(1, n_points):
        cumulative_dist[i] = cumulative_dist[i-1] + np.linalg.norm(points_3d[i] - points_3d[i-1])
    tcoords[:n_points, 1] = cumulative_dist / 3.0  # Tile every 3 units
    tcoords[n_points:, 1] = cumulative_dist / 3.0
    ribbon.active_texture_coordinates = tcoords

    return ribbon

# Create road ribbons
print("Creating road ribbons...")
highway = create_road_ribbon(highway_coords, heightmap, HEIGHT_SCALE,
                            width=ROAD_WIDTH_HIGHWAY, height_offset=0.02, world_scale=WORLD_SCALE)
sideroad = create_road_ribbon(sideroad_coords, heightmap, HEIGHT_SCALE,
                             width=ROAD_WIDTH_SIDEROAD, height_offset=0.015, world_scale=WORLD_SCALE)

# Load road textures
highway_texture = pv.read_texture(HIGHWAY_TEX)
sideroad_texture = pv.read_texture(SIDEROAD_TEX)

# ============================================================================
# FUTURISTIC LANDMARKS - Crystal/holographic style markers
# ============================================================================
def create_futuristic_markers(coords, heightmap, height_scale, world_scale,
                              marker_type='crystal', scale=0.25, height_offset=0.1):
    """Create futuristic-looking landmark markers"""
    heights = get_height_at(coords, heightmap, height_scale, world_scale)
    all_points = []
    all_faces = []
    all_tcoords = []
    point_offset = 0

    for coord, h in zip(coords, heights):
        center = np.array([coord[0], coord[1], h + height_offset])

        if marker_type == 'crystal':
            # Double-ended crystal/obelisk shape
            top = center + np.array([0, 0, scale * 1.5])
            bottom = center - np.array([0, 0, scale * 0.3])
            mid_radius = scale * 0.2

            # Create octagonal cross-section
            n_sides = 8
            angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
            mid_pts = []
            for angle in angles:
                pt = center + np.array([mid_radius * np.cos(angle),
                                        mid_radius * np.sin(angle), 0])
                mid_pts.append(pt)

            # Vertices: top, bottom, mid_points
            verts = [top, bottom] + mid_pts
            pts = np.array(verts)

            # Faces: triangles from top/bottom to mid ring
            faces = []
            for i in range(n_sides):
                next_i = (i + 1) % n_sides
                # Top pyramid
                faces.append([3, 0, 2 + i, 2 + next_i])
                # Bottom pyramid
                faces.append([3, 1, 2 + next_i, 2 + i])

            # Texture coordinates
            tcoords = np.zeros((len(pts), 2))
            tcoords[0] = [0.5, 1.0]  # Top
            tcoords[1] = [0.5, 0.0]  # Bottom
            for i in range(n_sides):
                tcoords[2 + i] = [i / n_sides, 0.5]

        elif marker_type == 'spire':
            # Tall thin spire with glowing tip effect
            base_radius = scale * 0.15
            height = scale * 2.0
            n_sides = 6

            angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
            base_pts = [center + np.array([base_radius * np.cos(a),
                                           base_radius * np.sin(a), 0]) for a in angles]
            top = center + np.array([0, 0, height])

            pts = np.array([top] + base_pts)
            faces = []
            for i in range(n_sides):
                next_i = (i + 1) % n_sides
                faces.append([3, 0, 1 + i, 1 + next_i])
            # Base
            for i in range(1, n_sides - 1):
                faces.append([3, 1, 1 + i, 1 + i + 1])

            tcoords = np.zeros((len(pts), 2))
            tcoords[0] = [0.5, 1.0]
            for i in range(n_sides):
                tcoords[1 + i] = [i / n_sides, 0.0]

        elif marker_type == 'beacon':
            # Floating beacon/cube shape
            size = scale * 0.3
            # Create a cube rotated 45 degrees
            cube = pv.Cube(center=tuple(center + np.array([0, 0, size])),
                          x_length=size, y_length=size, z_length=size)
            # Rotate it
            cube.rotate_z(45, point=tuple(center + np.array([0, 0, size])))

            pts = cube.points
            face_data = cube.faces
            new_faces = []
            idx = 0
            while idx < len(face_data):
                n_verts = face_data[idx]
                verts = face_data[idx+1:idx+1+n_verts] + point_offset
                new_faces.extend([n_verts] + list(verts))
                idx += n_verts + 1

            rel_pts = pts - center
            tcoords = np.column_stack([
                (rel_pts[:, 0] / size + 0.5),
                (rel_pts[:, 2] / size + 0.5)
            ])

            all_points.append(pts)
            all_tcoords.append(tcoords)
            all_faces.append(np.array(new_faces))
            point_offset += len(pts)
            continue

        # Add to combined mesh
        all_points.append(pts)
        all_tcoords.append(tcoords)
        face_arr = np.array(faces).ravel()
        # Offset faces
        new_faces = []
        idx = 0
        while idx < len(face_arr):
            n_verts = face_arr[idx]
            verts = face_arr[idx+1:idx+1+n_verts] + point_offset
            new_faces.extend([int(n_verts)] + [int(v) for v in verts])
            idx += n_verts + 1
        all_faces.append(np.array(new_faces))
        point_offset += len(pts)

    combined = pv.PolyData(np.vstack(all_points), np.concatenate(all_faces))
    combined.active_texture_coordinates = np.vstack(all_tcoords)
    return combined

print("Creating landmarks...")
lecture1_markers = create_futuristic_markers(lecture1_landmarks, heightmap, HEIGHT_SCALE,
                                             WORLD_SCALE, 'crystal', LANDMARK_SCALE, 0.15)
lecture2_markers = create_futuristic_markers(lecture2_landmarks, heightmap, HEIGHT_SCALE,
                                             WORLD_SCALE, 'spire', LANDMARK_SCALE, 0.1)
general_markers = create_futuristic_markers(general_landmarks, heightmap, HEIGHT_SCALE,
                                            WORLD_SCALE, 'beacon', LANDMARK_SCALE * 0.8, 0.12)

# Load landmark textures
landmark1_texture = pv.read_texture(LANDMARK1_TEX)
landmark2_texture = pv.read_texture(LANDMARK2_TEX)
landmark3_texture = pv.read_texture(LANDMARK3_TEX)

# Create grid lines overlay on terrain
print("Creating grid overlay...")
grid_lines = pv.PolyData()
grid_spacing = 10  # Grid every 10 units

for y_val in np.arange(0, 101, grid_spacing):
    x_coords = np.linspace(0, 100, 300)
    y_coords = np.full_like(x_coords, y_val)
    xy_coords = np.column_stack([x_coords, y_coords])
    z_coords = get_height_at(xy_coords, heightmap, HEIGHT_SCALE, WORLD_SCALE) + 0.03

    points = np.column_stack([x_coords, y_coords, z_coords])
    line = pv.lines_from_points(points)
    grid_lines = grid_lines.merge(line)

for x_val in np.arange(0, 101, grid_spacing):
    y_coords = np.linspace(0, 100, 300)
    x_coords = np.full_like(y_coords, x_val)
    xy_coords = np.column_stack([x_coords, y_coords])
    z_coords = get_height_at(xy_coords, heightmap, HEIGHT_SCALE, WORLD_SCALE) + 0.03

    points = np.column_stack([x_coords, y_coords, z_coords])
    line = pv.lines_from_points(points)
    grid_lines = grid_lines.merge(line)

# ============================================================================
# SET UP THE PLOTTER
# ============================================================================
print("Setting up plotter...")
plotter = pv.Plotter(off_screen=True, window_size=[675, 1200])

# Sky gradient background
plotter.set_background('lightskyblue', top='darkblue')

# Add terrain with rock texture - majestic appearance
plotter.add_mesh(terrain, texture=terrain_texture,
                 smooth_shading=True,
                 specular=0.1, specular_power=5, ambient=0.4,
                 diffuse=0.7)

# Add grid lines - subtle topographic overlay
plotter.add_mesh(grid_lines, color='#2a2a2a', line_width=1.0, opacity=0.4)

# Add highway with asphalt texture
plotter.add_mesh(highway, texture=highway_texture, specular=0.2, specular_power=8,
                 smooth_shading=True, ambient=0.35, diffuse=0.65)

# Add side road with rocky trail texture
plotter.add_mesh(sideroad, texture=sideroad_texture, specular=0.15, specular_power=5,
                 smooth_shading=True, ambient=0.3, diffuse=0.6)

# Add landmarks with metallic appearance
plotter.add_mesh(lecture1_markers, texture=landmark1_texture, specular=0.8,
                 specular_power=60, ambient=0.3, smooth_shading=True, diffuse=0.5)

plotter.add_mesh(lecture2_markers, texture=landmark2_texture, specular=0.75,
                 specular_power=50, ambient=0.35, smooth_shading=True, diffuse=0.55)

plotter.add_mesh(general_markers, texture=landmark3_texture, specular=0.7,
                 specular_power=45, ambient=0.35, smooth_shading=True, diffuse=0.55)

# ============================================================================
# LIGHTING - Dramatic, cinematic setup
# ============================================================================
print("Setting up lighting...")
plotter.remove_all_lights()

# Main sun light - warm, from above-front
light1 = pv.Light(position=(80, 120, 80), focal_point=(50, 50, 8),
                  color='#FFF8DC', intensity=1.0)  # Cornsilk warm white
plotter.add_light(light1)

# Fill light - cooler, from the side
light2 = pv.Light(position=(-30, 50, 40), focal_point=(50, 50, 8),
                  color='#B0C4DE', intensity=0.4)  # Light steel blue
plotter.add_light(light2)

# Rim light - adds depth
light3 = pv.Light(position=(50, -30, 60), focal_point=(50, 50, 8),
                  color='#FFE4B5', intensity=0.3)  # Moccasin
plotter.add_light(light3)

# Additional fill light from below/behind for ambient feel
light4 = pv.Light(position=(50, 50, -20), focal_point=(50, 50, 8),
                  color='#404060', intensity=0.2)
plotter.add_light(light4)

# ============================================================================
# CAMERA - Realistic driver's perspective
# ============================================================================
print("Setting up camera...")

# Position along the highway - start position
cam_idx = 10
cam_pos_xy = highway_coords[cam_idx]
cam_height = get_height_at(cam_pos_xy.reshape(1, -1), heightmap, HEIGHT_SCALE, WORLD_SCALE)[0]

# Look far ahead down the road
look_idx = min(cam_idx + 200, len(highway_coords) - 1)
look_pos_xy = highway_coords[look_idx]
look_height = get_height_at(look_pos_xy.reshape(1, -1), heightmap, HEIGHT_SCALE, WORLD_SCALE)[0]

# Get road direction for proper offset
road_dir = highway_coords[min(cam_idx + 5, len(highway_coords)-1)] - highway_coords[cam_idx]
road_dir = road_dir / (np.linalg.norm(road_dir) + 1e-6)
perp_dir = np.array([-road_dir[1], road_dir[0]])

# Camera at realistic driver height (about 1.5m = 0.15 in our scale relative to road)
# Offset slightly to left (driver position)
cam_offset = perp_dir * (ROAD_WIDTH_HIGHWAY * 0.2)
camera_height_above_road = 0.15  # Driver eye level

camera_position = (cam_pos_xy[0] + cam_offset[0],
                   cam_pos_xy[1] + cam_offset[1],
                   cam_height + camera_height_above_road)

# Look at a point down the road, slightly elevated
focal_point = (look_pos_xy[0], look_pos_xy[1], look_height + 0.1)

view_up = (0, 0, 1)

plotter.camera_position = [camera_position, focal_point, view_up]
plotter.camera.view_angle = 75  # Realistic wide angle for driving

# ============================================================================
# RENDER
# ============================================================================
print("Rendering...")
plotter.show(screenshot='terrain_render_raw.png', auto_close=True)

# Post-process to exact dimensions
print("Post-processing image...")
img = Image.open('terrain_render_raw.png')
print(f"Raw image size: {img.width}x{img.height}")

target_width, target_height = 675, 1200

if img.size != (target_width, target_height):
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    print(f"Resized to: {img.width}x{img.height}")

img.save('terrain_render.png')
print("Done! Image saved as terrain_render.png (675x1200)")
