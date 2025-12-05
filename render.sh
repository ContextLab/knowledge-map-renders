#!/bin/bash
#
# render.sh - Cross-platform render script for Knowledge Landscape visualization
#
# Usage:
#   ./render.sh                    # Render using cached scene if available
#   ./render.sh --force-rebuild    # Force rebuild scene from scratch
#   ./render.sh --output scene.png # Specify output PNG file
#   ./render.sh --blend out.blend  # Specify output .blend file
#   ./render.sh --help             # Show this help message
#
# Supports macOS (including Apple Silicon) and Ubuntu Linux.
#

set -e

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_SCRIPT="${SCRIPT_DIR}/scripts/blender_render.py"

# Default output paths
OUTPUT_PNG=""
OUTPUT_BLEND=""
FORCE_REBUILD=""
EXTRA_ARGS=""

# =============================================================================
# Parse Arguments
# =============================================================================
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Render the Knowledge Landscape 3D visualization using Blender.

Options:
  --force-rebuild, -f    Force rebuild scene from scratch (ignore cache)
  --output, -o FILE      Specify output PNG file (default: scene.png)
  --blend, -b FILE       Specify output .blend file (default: rendered_scene.blend)
  --help, -h             Show this help message

Examples:
  $(basename "$0")                              # Use cached scene if available
  $(basename "$0") --force-rebuild              # Rebuild from scratch
  $(basename "$0") -o render.png -b scene.blend # Custom output paths
  $(basename "$0") -f -o final.png              # Rebuild with custom PNG output

Environment:
  BLENDER_PATH           Override Blender executable path

EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --force-rebuild|-f)
            FORCE_REBUILD="--force-rebuild"
            shift
            ;;
        --output|-o)
            OUTPUT_PNG="$2"
            shift 2
            ;;
        --blend|-b)
            OUTPUT_BLEND="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# =============================================================================
# Detect Blender
# =============================================================================
find_blender() {
    # Allow override via environment variable
    if [[ -n "${BLENDER_PATH}" ]]; then
        if [[ -x "${BLENDER_PATH}" ]]; then
            echo "${BLENDER_PATH}"
            return 0
        else
            echo "Error: BLENDER_PATH '${BLENDER_PATH}' is not executable" >&2
            return 1
        fi
    fi

    # Detect OS and find Blender
    case "$(uname -s)" in
        Darwin)
            # macOS - check standard locations
            local macos_paths=(
                "/Applications/Blender.app/Contents/MacOS/Blender"
                "$HOME/Applications/Blender.app/Contents/MacOS/Blender"
            )
            for path in "${macos_paths[@]}"; do
                if [[ -x "$path" ]]; then
                    echo "$path"
                    return 0
                fi
            done
            # Try homebrew
            if command -v blender &> /dev/null; then
                command -v blender
                return 0
            fi
            ;;
        Linux)
            # Linux - check common locations
            local linux_paths=(
                "/usr/bin/blender"
                "/usr/local/bin/blender"
                "/snap/bin/blender"
                "$HOME/blender/blender"
            )
            for path in "${linux_paths[@]}"; do
                if [[ -x "$path" ]]; then
                    echo "$path"
                    return 0
                fi
            done
            # Try PATH
            if command -v blender &> /dev/null; then
                command -v blender
                return 0
            fi
            ;;
        *)
            echo "Error: Unsupported operating system: $(uname -s)" >&2
            return 1
            ;;
    esac

    echo "Error: Blender not found. Please install Blender or set BLENDER_PATH." >&2
    echo "  macOS: brew install --cask blender" >&2
    echo "  Ubuntu: sudo apt install blender" >&2
    echo "  Or download from: https://www.blender.org/download/" >&2
    return 1
}

BLENDER=$(find_blender) || exit 1

# =============================================================================
# Verify Script Exists
# =============================================================================
if [[ ! -f "${BLENDER_SCRIPT}" ]]; then
    echo "Error: Render script not found: ${BLENDER_SCRIPT}" >&2
    exit 1
fi

# =============================================================================
# Build Command
# =============================================================================
echo "=============================================="
echo "Knowledge Landscape Renderer"
echo "=============================================="
echo "Blender:    ${BLENDER}"
echo "Script:     ${BLENDER_SCRIPT}"
[[ -n "${FORCE_REBUILD}" ]] && echo "Mode:       Force rebuild"
[[ -n "${OUTPUT_PNG}" ]] && echo "Output PNG: ${OUTPUT_PNG}"
[[ -n "${OUTPUT_BLEND}" ]] && echo "Output blend: ${OUTPUT_BLEND}"
echo "=============================================="
echo ""

# Build Blender command
BLENDER_CMD=("${BLENDER}" "--background" "--python" "${BLENDER_SCRIPT}")

# Add arguments after -- separator for Python script
if [[ -n "${FORCE_REBUILD}" ]]; then
    BLENDER_CMD+=("--" "${FORCE_REBUILD}")
fi

# =============================================================================
# Run Render
# =============================================================================
echo "Starting render..."
echo "Command: ${BLENDER_CMD[*]}"
echo ""

START_TIME=$(date +%s)

"${BLENDER_CMD[@]}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# =============================================================================
# Handle Custom Output Paths
# =============================================================================
if [[ -n "${OUTPUT_PNG}" && "${OUTPUT_PNG}" != "scene.png" ]]; then
    if [[ -f "${SCRIPT_DIR}/scene.png" ]]; then
        echo "Moving output to: ${OUTPUT_PNG}"
        mv "${SCRIPT_DIR}/scene.png" "${OUTPUT_PNG}"
    fi
fi

if [[ -n "${OUTPUT_BLEND}" && "${OUTPUT_BLEND}" != "rendered_scene.blend" ]]; then
    if [[ -f "${SCRIPT_DIR}/rendered_scene.blend" ]]; then
        echo "Moving .blend file to: ${OUTPUT_BLEND}"
        mv "${SCRIPT_DIR}/rendered_scene.blend" "${OUTPUT_BLEND}"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Render Complete!"
echo "=============================================="
echo "Time elapsed: ${ELAPSED} seconds"

if [[ -n "${OUTPUT_PNG}" ]]; then
    echo "Output PNG:   ${OUTPUT_PNG}"
else
    echo "Output PNG:   ${SCRIPT_DIR}/scene.png"
fi

if [[ -n "${OUTPUT_BLEND}" ]]; then
    echo "Output blend: ${OUTPUT_BLEND}"
else
    echo "Output blend: ${SCRIPT_DIR}/rendered_scene.blend"
fi

echo "=============================================="
