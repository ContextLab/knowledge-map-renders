#!/bin/bash
# Setup script for Knowledge Landscape 3D Visualization
# This script is idempotent - safe to run multiple times

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_VERSION="4.2.0"
BLENDER_DMG="blender-${BLENDER_VERSION}-macos-arm64.dmg"
BLENDER_URL="https://download.blender.org/release/Blender4.2/${BLENDER_DMG}"
CONDA_ENV_NAME="knowledge-landscape"

echo "=== Knowledge Landscape 3D Visualization Setup ==="
echo ""

# -----------------------------------------------------------------------------
# Check for Blender
# -----------------------------------------------------------------------------
echo "Checking for Blender..."

if [ -d "/Applications/Blender.app" ]; then
    BLENDER_INSTALLED_VERSION=$(/Applications/Blender.app/Contents/MacOS/Blender --version 2>/dev/null | head -1 | awk '{print $2}')
    echo "  Blender ${BLENDER_INSTALLED_VERSION} is already installed."
else
    echo "  Blender not found. Installing Blender ${BLENDER_VERSION}..."

    # Download if not already present
    if [ ! -f "${SCRIPT_DIR}/materials/${BLENDER_DMG}" ]; then
        echo "  Downloading Blender ${BLENDER_VERSION}..."
        curl -L -o "${SCRIPT_DIR}/materials/${BLENDER_DMG}" "${BLENDER_URL}"
    else
        echo "  Using cached DMG: ${BLENDER_DMG}"
    fi

    # Mount and install
    echo "  Mounting DMG..."
    hdiutil attach "${SCRIPT_DIR}/materials/${BLENDER_DMG}" -nobrowse -quiet

    echo "  Installing Blender to /Applications..."
    cp -R "/Volumes/Blender/Blender.app" /Applications/

    echo "  Cleaning up..."
    hdiutil detach "/Volumes/Blender" -quiet

    echo "  Blender ${BLENDER_VERSION} installed successfully."
fi

# -----------------------------------------------------------------------------
# Check for Conda
# -----------------------------------------------------------------------------
echo ""
echo "Checking for Conda..."

if command -v conda &> /dev/null; then
    echo "  Conda is available."
else
    echo "  ERROR: Conda not found. Please install Miniconda or Anaconda first."
    echo "  Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# -----------------------------------------------------------------------------
# Create/Update Conda Environment
# -----------------------------------------------------------------------------
echo ""
echo "Setting up Conda environment '${CONDA_ENV_NAME}'..."

# Check if environment exists
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "  Environment '${CONDA_ENV_NAME}' already exists."
    echo "  Updating packages..."
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "${CONDA_ENV_NAME}"
    pip install -q -r "${SCRIPT_DIR}/requirements.txt"
else
    echo "  Creating new environment '${CONDA_ENV_NAME}'..."
    conda create -n "${CONDA_ENV_NAME}" python=3.11 -y

    # Activate and install requirements
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

echo "  Environment ready."

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=== Setup Complete ==="
echo ""
echo "To render the visualization:"
echo "  1. Activate the conda environment:"
echo "     conda activate ${CONDA_ENV_NAME}"
echo ""
echo "  2. Run the render script:"
echo "     /Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/blender_render.py"
echo ""
echo "Output will be saved to:"
echo "  - scene.png (rendered image)"
echo "  - rendered_scene.blend (Blender scene file)"
echo ""
