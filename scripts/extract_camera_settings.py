#!/usr/bin/env python3
"""Extract camera and target settings from terrain_scene_edit.blend"""

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
else:
    print("No camera found!")

# Also check for any empty named "CameraTarget" or similar
print("\n=== ALL EMPTIES ===")
for obj in bpy.data.objects:
    if obj.type == 'EMPTY':
        print(f"{obj.name}: location = ({obj.location.x:.6f}, {obj.location.y:.6f}, {obj.location.z:.6f})")
