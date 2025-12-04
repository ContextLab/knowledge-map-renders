#!/usr/bin/env python3
"""Extract all properties from the Automotive Paint Shader material."""

import bpy
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
MATERIAL_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), "materials", "automotive-paint.blend")

print(f"Loading material from: {MATERIAL_FILE}")

# Load the blend file
bpy.ops.wm.open_mainfile(filepath=MATERIAL_FILE)

# Find the Automotive Paint Shader material
print("\n=== MATERIALS IN FILE ===")
for mat in bpy.data.materials:
    print(f"\nMaterial: {mat.name}")
    if mat.use_nodes:
        print("  Uses nodes: True")
        print("  Node tree nodes:")
        for node in mat.node_tree.nodes:
            print(f"    - {node.name} ({node.type})")
            # Print inputs
            if hasattr(node, 'inputs'):
                for inp in node.inputs:
                    if hasattr(inp, 'default_value'):
                        try:
                            val = inp.default_value
                            if hasattr(val, '__iter__') and not isinstance(val, str):
                                val = tuple(val)
                            print(f"        Input '{inp.name}': {val}")
                        except:
                            print(f"        Input '{inp.name}': [complex type]")

# Look at node groups
print("\n=== NODE GROUPS ===")
for ng in bpy.data.node_groups:
    print(f"\nNode Group: {ng.name}")
    print(f"  Type: {ng.type}")

    # Print inputs (interface)
    print("  Interface Inputs:")
    if hasattr(ng, 'interface'):
        for item in ng.interface.items_tree:
            if item.item_type == 'SOCKET' and item.in_out == 'INPUT':
                print(f"    - {item.name} ({item.socket_type})")
                if hasattr(item, 'default_value'):
                    try:
                        print(f"      Default: {item.default_value}")
                    except:
                        pass

    # Print nodes inside the group
    print("  Internal Nodes:")
    for node in ng.nodes:
        print(f"    - {node.name} ({node.type})")
        if hasattr(node, 'inputs'):
            for inp in node.inputs:
                if hasattr(inp, 'default_value'):
                    try:
                        val = inp.default_value
                        if hasattr(val, '__iter__') and not isinstance(val, str):
                            val = tuple(val)
                        print(f"        '{inp.name}': {val}")
                    except:
                        pass
