#!/usr/bin/env python3
"""
Load terrain_scene_edit.blend, resize all landmarks to match Lecture1_6_8,
save to scene.blend, and render.
"""

import bpy
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
INPUT_FILE = os.path.join(SCRIPT_DIR, "terrain_scene_edit.blend")
OUTPUT_SCENE = os.path.join(SCRIPT_DIR, "scene.blend")
OUTPUT_RENDER = os.path.join(SCRIPT_DIR, "scene.png")

print(f"Loading scene from: {INPUT_FILE}")

# Load the edited scene
bpy.ops.wm.open_mainfile(filepath=INPUT_FILE)

# Get camera info
camera = bpy.context.scene.camera
if camera:
    print(f"\nCamera settings:")
    print(f"  Name: {camera.name}")
    print(f"  Location: {camera.location[:]}")
    print(f"  Rotation: {camera.rotation_euler[:]}")
    print(f"  Focal length: {camera.data.lens} mm")
    print(f"  Sensor width: {camera.data.sensor_width} mm")

    # Check for camera target constraint
    for constraint in camera.constraints:
        if constraint.type == 'TRACK_TO':
            target = constraint.target
            if target:
                print(f"  Target: {target.name} at {target.location[:]}")

# Find the reference landmark Lecture1_6_8
reference_landmark = bpy.data.objects.get("Lecture1_6_8")

if reference_landmark:
    # Get the scale of the reference landmark
    ref_scale = reference_landmark.scale[:]
    print(f"\nReference landmark 'Lecture1_6_8':")
    print(f"  Scale: {ref_scale}")
    print(f"  Location: {reference_landmark.location[:]}")

    # For mesh objects, get the dimensions
    if reference_landmark.type == 'MESH':
        print(f"  Dimensions: {reference_landmark.dimensions[:]}")

    # Find all landmark objects (spheres with Lecture or General in name)
    landmarks = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and ('Lecture' in obj.name or 'General' in obj.name):
            landmarks.append(obj)

    print(f"\nFound {len(landmarks)} landmark objects")

    # Resize all landmarks to match the reference
    resized_count = 0
    for landmark in landmarks:
        if landmark.name != "Lecture1_6_8":
            old_scale = landmark.scale[:]
            landmark.scale = ref_scale
            if old_scale != ref_scale:
                print(f"  Resized '{landmark.name}': {old_scale} -> {ref_scale}")
                resized_count += 1

    print(f"\nResized {resized_count} landmarks to match Lecture1_6_8")
else:
    print("\nWARNING: Reference landmark 'Lecture1_6_8' not found!")
    # List all objects that might be landmarks
    print("Objects in scene:")
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and ('Lecture' in obj.name or 'General' in obj.name or 'Sphere' in obj.name):
            print(f"  {obj.name}: scale={obj.scale[:]}")

# Save the scene to scene.blend
print(f"\nSaving scene to: {OUTPUT_SCENE}")
bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_SCENE)

# Configure render output
scene = bpy.context.scene
scene.render.filepath = OUTPUT_RENDER
scene.render.image_settings.file_format = 'PNG'

print(f"\nRender settings:")
print(f"  Engine: {scene.render.engine}")
print(f"  Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
print(f"  Output: {OUTPUT_RENDER}")

if scene.render.engine == 'CYCLES':
    print(f"  Samples: {scene.cycles.samples}")
    print(f"  Device: {scene.cycles.device}")

# Render
print("\nStarting render...")
bpy.ops.render.render(write_still=True)

print(f"\nRender complete! Saved to: {OUTPUT_RENDER}")
