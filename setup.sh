#!/bin/bash
# Setup script for Knowledge Landscape 3D Visualization
# This script is idempotent - safe to run multiple times
# Supports both macOS (GUI) and Linux (headless cluster) installations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_VERSION="4.2.0"
CONDA_ENV_NAME="knowledge-landscape"

echo "=== Knowledge Landscape 3D Visualization Setup ==="
echo ""

# -----------------------------------------------------------------------------
# Detect Platform
# -----------------------------------------------------------------------------
detect_platform() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

PLATFORM=$(detect_platform)
echo "Detected platform: ${PLATFORM}"

# Check if we're on a cluster (no display, or SSH session)
is_cluster() {
    if [[ -z "$DISPLAY" ]] && [[ "$PLATFORM" == "linux" ]]; then
        return 0
    elif [[ -n "$SSH_CONNECTION" ]] && [[ "$PLATFORM" == "linux" ]]; then
        return 0
    else
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Install Blender
# -----------------------------------------------------------------------------
echo ""
echo "Checking for Blender..."

install_blender_macos() {
    BLENDER_DMG="blender-${BLENDER_VERSION}-macos-arm64.dmg"
    BLENDER_URL="https://download.blender.org/release/Blender4.2/${BLENDER_DMG}"

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

    BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"
}

install_blender_linux() {
    BLENDER_TARBALL="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
    BLENDER_URL="https://download.blender.org/release/Blender4.2/${BLENDER_TARBALL}"
    BLENDER_INSTALL_DIR="${HOME}/blender-${BLENDER_VERSION}"

    if [ -x "${BLENDER_INSTALL_DIR}/blender" ]; then
        BLENDER_INSTALLED_VERSION=$("${BLENDER_INSTALL_DIR}/blender" --version 2>/dev/null | head -1 | awk '{print $2}')
        echo "  Blender ${BLENDER_INSTALLED_VERSION} is already installed at ${BLENDER_INSTALL_DIR}"
    else
        echo "  Blender not found. Installing Blender ${BLENDER_VERSION} (headless)..."

        # Download if not already present
        DOWNLOAD_DIR="${SCRIPT_DIR}/materials"
        mkdir -p "${DOWNLOAD_DIR}"

        if [ ! -f "${DOWNLOAD_DIR}/${BLENDER_TARBALL}" ]; then
            echo "  Downloading Blender ${BLENDER_VERSION}..."
            curl -L -o "${DOWNLOAD_DIR}/${BLENDER_TARBALL}" "${BLENDER_URL}"
        else
            echo "  Using cached tarball: ${BLENDER_TARBALL}"
        fi

        # Extract to home directory
        echo "  Extracting to ${BLENDER_INSTALL_DIR}..."
        mkdir -p "${BLENDER_INSTALL_DIR}"
        tar -xf "${DOWNLOAD_DIR}/${BLENDER_TARBALL}" -C "${HOME}" --strip-components=1 --transform="s|^[^/]*|blender-${BLENDER_VERSION}|"

        # Actually the tar structure is different, let's fix this
        rm -rf "${BLENDER_INSTALL_DIR}"
        cd "${HOME}"
        tar -xf "${DOWNLOAD_DIR}/${BLENDER_TARBALL}"
        # The extracted folder name includes the full version
        EXTRACTED_DIR=$(ls -d blender-${BLENDER_VERSION}* 2>/dev/null | head -1)
        if [ -n "$EXTRACTED_DIR" ] && [ "$EXTRACTED_DIR" != "blender-${BLENDER_VERSION}" ]; then
            mv "$EXTRACTED_DIR" "blender-${BLENDER_VERSION}" 2>/dev/null || true
        fi
        cd "${SCRIPT_DIR}"

        echo "  Blender ${BLENDER_VERSION} installed successfully."
    fi

    BLENDER_BIN="${BLENDER_INSTALL_DIR}/blender"
}

if [[ "$PLATFORM" == "macos" ]]; then
    install_blender_macos
elif [[ "$PLATFORM" == "linux" ]]; then
    install_blender_linux
else
    echo "  ERROR: Unsupported platform: ${PLATFORM}"
    exit 1
fi

# Verify Blender works
echo "  Verifying Blender installation..."
if "${BLENDER_BIN}" --version > /dev/null 2>&1; then
    echo "  Blender is working correctly."
else
    echo "  WARNING: Blender may not be fully functional (this is OK for headless rendering)."
fi

# -----------------------------------------------------------------------------
# Check for Conda (optional on cluster)
# -----------------------------------------------------------------------------
echo ""
echo "Checking for Conda..."

if command -v conda &> /dev/null; then
    echo "  Conda is available."

    # -----------------------------------------------------------------------------
    # Create/Update Conda Environment
    # -----------------------------------------------------------------------------
    echo ""
    echo "Setting up Conda environment '${CONDA_ENV_NAME}'..."

    # Source conda for shell integration
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    fi

    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "  Environment '${CONDA_ENV_NAME}' already exists."
        echo "  Updating packages..."
        conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
        pip install -q -r "${SCRIPT_DIR}/requirements.txt" 2>/dev/null || echo "  (pip install skipped - not critical for rendering)"
    else
        echo "  Creating new environment '${CONDA_ENV_NAME}'..."
        conda create -n "${CONDA_ENV_NAME}" python=3.11 -y

        # Activate and install requirements
        conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
        pip install -r "${SCRIPT_DIR}/requirements.txt" 2>/dev/null || echo "  (pip install skipped - not critical for rendering)"
    fi

    echo "  Environment ready."
else
    echo "  Conda not found. Skipping Python environment setup."
    echo "  (Blender includes its own Python - this is OK for rendering)"
fi

# -----------------------------------------------------------------------------
# Write blender path to config file for other scripts
# -----------------------------------------------------------------------------
echo ""
echo "Writing Blender path to config..."
echo "BLENDER_BIN=\"${BLENDER_BIN}\"" > "${SCRIPT_DIR}/.blender_config"
echo "  Config written to .blender_config"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Blender binary: ${BLENDER_BIN}"
echo ""
echo "To render the visualization:"
if [[ "$PLATFORM" == "macos" ]]; then
    echo "  1. Activate the conda environment (optional):"
    echo "     conda activate ${CONDA_ENV_NAME}"
    echo ""
    echo "  2. Run the render script:"
    echo "     ${BLENDER_BIN} --background --python scripts/blender_render.py"
else
    echo "  Run the render script:"
    echo "     ${BLENDER_BIN} --background --python scripts/blender_render.py"
fi
echo ""
echo "Output will be saved to:"
echo "  - scene.png (rendered image)"
echo "  - rendered_scene.blend (Blender scene file)"
echo ""
