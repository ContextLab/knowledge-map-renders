#!/usr/bin/env python3
"""
Render from saved Blender scene file.
Run with: /Applications/Blender.app/Contents/MacOS/Blender --background terrain_scene.blend --python render_from_scene.py
"""

import bpy
import os

# Output settings
OUTPUT_DIR = os.path.dirname(bpy.data.filepath)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "scene.png")

# Configure output
scene = bpy.context.scene
scene.render.filepath = OUTPUT_FILE
scene.render.image_settings.file_format = 'PNG'

print(f"Rendering from: {bpy.data.filepath}")
print(f"Output: {OUTPUT_FILE}")
print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")

# Render
bpy.ops.render.render(write_still=True)

print(f"Render complete! Saved to: {OUTPUT_FILE}")
