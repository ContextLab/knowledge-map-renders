#!/bin/bash
#
# render.sh - Cross-platform render script for Knowledge Landscape visualization
#
# Usage:
#   ./render.sh                         # Build and render scene (default)
#   ./render.sh -i scene.blend          # Render existing .blend file
#   ./render.sh -o output.png           # Specify output PNG file
#   ./render.sh -b out.blend            # Specify output .blend file
#   ./render.sh --help                  # Show this help message
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
INPUT_BLEND=""
EXTRA_ARGS=""

# Default output locations (set by blender_render.py when building scene)
DEFAULT_PNG="${SCRIPT_DIR}/images/scene.png"
DEFAULT_BLEND="${SCRIPT_DIR}/data/rendered_scene.blend"

# =============================================================================
# Parse Arguments
# =============================================================================
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Render the Knowledge Landscape 3D visualization using Blender.

By default, rebuilds the scene from scratch using blender_render.py.
Use --input to render an existing .blend file instead.

Options:
  --input, -i FILE       Render existing .blend file (skip scene building)
  --output, -o FILE      Specify output PNG file (default: images/scene.png)
  --blend, -b FILE       Specify output .blend file (default: data/rendered_scene.blend)
  --help, -h             Show this help message

Examples:
  $(basename "$0")                              # Build and render scene
  $(basename "$0") -o render.png -b scene.blend # Custom output paths
  $(basename "$0") -i data/rendered_scene.blend # Render existing .blend file
  $(basename "$0") -i scene.blend -o output.png # Render existing file with custom output

Environment:
  BLENDER_PATH           Override Blender executable path

EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            INPUT_BLEND="$2"
            shift 2
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

if [[ -n "${INPUT_BLEND}" ]]; then
    # Render existing .blend file
    if [[ ! -f "${INPUT_BLEND}" ]]; then
        echo "Error: Input file not found: ${INPUT_BLEND}" >&2
        exit 1
    fi
    echo "Mode:       Render existing .blend file"
    echo "Input:      ${INPUT_BLEND}"
else
    # Build scene from scratch (default)
    echo "Script:     ${BLENDER_SCRIPT}"
    echo "Mode:       Build and render scene"
fi

[[ -n "${OUTPUT_PNG}" ]] && echo "Output PNG: ${OUTPUT_PNG}"
[[ -n "${OUTPUT_BLEND}" ]] && echo "Output blend: ${OUTPUT_BLEND}"
echo "=============================================="
echo ""

# =============================================================================
# Run Render
# =============================================================================
START_TIME=$(date +%s)

if [[ -n "${INPUT_BLEND}" ]]; then
    # Render existing .blend file
    echo "Rendering existing .blend file..."

    # Determine output path for render
    RENDER_OUTPUT="${OUTPUT_PNG:-${DEFAULT_PNG}}"
    RENDER_OUTPUT_BASE="${RENDER_OUTPUT%.png}"

    echo "Command: ${BLENDER} --background ${INPUT_BLEND} --render-output ${RENDER_OUTPUT_BASE} --render-frame 1"
    echo ""

    "${BLENDER}" --background "${INPUT_BLEND}" --render-output "${RENDER_OUTPUT_BASE}" --render-frame 1

    # Blender adds frame number to output, rename if needed
    if [[ -f "${RENDER_OUTPUT_BASE}0001.png" ]]; then
        mv "${RENDER_OUTPUT_BASE}0001.png" "${RENDER_OUTPUT}"
    fi
else
    # Build and render scene (default)
    echo "Building and rendering scene..."
    BLENDER_CMD=("${BLENDER}" "--background" "--python" "${BLENDER_SCRIPT}")
    echo "Command: ${BLENDER_CMD[*]}"
    echo ""

    "${BLENDER_CMD[@]}"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# =============================================================================
# Handle Custom Output Paths (only when building scene, not when using input)
# =============================================================================
if [[ -z "${INPUT_BLEND}" ]]; then
    # Only move files if we built the scene (not rendered existing .blend)
    if [[ -n "${OUTPUT_PNG}" && "${OUTPUT_PNG}" != "${DEFAULT_PNG}" ]]; then
        if [[ -f "${DEFAULT_PNG}" ]]; then
            echo "Moving output to: ${OUTPUT_PNG}"
            mv "${DEFAULT_PNG}" "${OUTPUT_PNG}"
        fi
    fi

    if [[ -n "${OUTPUT_BLEND}" && "${OUTPUT_BLEND}" != "${DEFAULT_BLEND}" ]]; then
        if [[ -f "${DEFAULT_BLEND}" ]]; then
            echo "Moving .blend file to: ${OUTPUT_BLEND}"
            mv "${DEFAULT_BLEND}" "${OUTPUT_BLEND}"
        fi
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

if [[ -n "${INPUT_BLEND}" ]]; then
    # Rendered existing .blend file
    echo "Input:        ${INPUT_BLEND}"
    echo "Output PNG:   ${OUTPUT_PNG:-${DEFAULT_PNG}}"
else
    # Built and rendered scene
    if [[ -n "${OUTPUT_PNG}" ]]; then
        echo "Output PNG:   ${OUTPUT_PNG}"
    else
        echo "Output PNG:   ${DEFAULT_PNG}"
    fi

    if [[ -n "${OUTPUT_BLEND}" ]]; then
        echo "Output blend: ${OUTPUT_BLEND}"
    else
        echo "Output blend: ${DEFAULT_BLEND}"
    fi
fi

echo "=============================================="
