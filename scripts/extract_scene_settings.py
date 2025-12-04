#!/usr/bin/env python3
"""Extract camera, target, and landmark settings from terrain_scene_edit.blend"""

import bpy
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
INPUT_FILE = os.path.join(SCRIPT_DIR, "terrain_scene_edit.blend")

print(f"Loading scene from: {INPUT_FILE}")
bpy.ops.wm.open_mainfile(filepath=INPUT_FILE)

# Get camera
camera = bpy.context.scene.camera
if camera:
    print("\n=== CAMERA SETTINGS ===")
    print(f"camera_location = ({camera.location.x:.6f}, {camera.location.y:.6f}, {camera.location.z:.6f})")
    print(f"camera_rotation = ({camera.rotation_euler.x:.6f}, {camera.rotation_euler.y:.6f}, {camera.rotation_euler.z:.6f})")
    print(f"lens = {camera.data.lens}")
    print(f"sensor_width = {camera.data.sensor_width}")
    print(f"clip_start = {camera.data.clip_start}")
    print(f"clip_end = {camera.data.clip_end}")

    # DOF settings
    print(f"\n# DOF Settings")
    print(f"use_dof = {camera.data.dof.use_dof}")
    print(f"focus_distance = {camera.data.dof.focus_distance}")
    print(f"aperture_fstop = {camera.data.dof.aperture_fstop}")

    # Check for constraints
    for constraint in camera.constraints:
        if constraint.type == 'TRACK_TO':
            target = constraint.target
            if target:
                print(f"\n=== TARGET SETTINGS ===")
                print(f"target_name = \"{target.name}\"")
                print(f"target_location = ({target.location.x:.6f}, {target.location.y:.6f}, {target.location.z:.6f})")

# Find General_12_6 landmark to get its radius and position
print("\n=== LANDMARK SETTINGS (from General_12_6) ===")
general_12_6 = bpy.data.objects.get("General_12_6")
if general_12_6:
    print(f"Location: ({general_12_6.location.x:.6f}, {general_12_6.location.y:.6f}, {general_12_6.location.z:.6f})")
    print(f"Scale: ({general_12_6.scale.x:.6f}, {general_12_6.scale.y:.6f}, {general_12_6.scale.z:.6f})")

    # Get dimensions (actual size after scale)
    print(f"Dimensions: ({general_12_6.dimensions.x:.6f}, {general_12_6.dimensions.y:.6f}, {general_12_6.dimensions.z:.6f})")

    # Calculate radius from dimensions (sphere, so x/2 = radius)
    radius = general_12_6.dimensions.x / 2
    print(f"Calculated radius: {radius:.6f} feet ({radius * 12:.2f} inches)")

    # Z position tells us the float height
    z_pos = general_12_6.location.z
    print(f"Z position (center): {z_pos:.6f} feet")
else:
    print("General_12_6 not found!")

# List all landmarks with their settings
print("\n=== ALL LANDMARKS ===")
for obj in bpy.data.objects:
    if obj.type == 'MESH' and ('Lecture' in obj.name or 'General' in obj.name):
        radius = obj.dimensions.x / 2 if obj.dimensions.x > 0 else 0
        print(f"{obj.name}: loc=({obj.location.x:.3f}, {obj.location.y:.3f}, {obj.location.z:.3f}), radius={radius:.4f}ft ({radius*12:.2f}in), scale={obj.scale[:]}")

# Also check empties
print("\n=== ALL EMPTIES ===")
for obj in bpy.data.objects:
    if obj.type == 'EMPTY':
        print(f"{obj.name}: location = ({obj.location.x:.6f}, {obj.location.y:.6f}, {obj.location.z:.6f})")
