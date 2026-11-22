#!/usr/bin/env bash
# Build val3dity (GPL-3.0, external test oracle only; D-003) outside the repository.
# macOS: brew install cgal eigen geos boost cmake. Linux: the distro's CGAL/Eigen/GEOS/Boost dev packages.
# Usage: scripts/install_val3dity.sh [install-dir]   (default: ../tools/val3dity next to the repo)
set -euo pipefail

VERSION="${VAL3DITY_VERSION:-2.7.0}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$(dirname "$REPO_ROOT")/tools/val3dity}"

case "$DEST" in
  "$REPO_ROOT"/*) echo "refusing to build inside the repository: $DEST" >&2; exit 1 ;;
esac

if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 --branch "$VERSION" https://github.com/tudelft3d/val3dity.git "$DEST"
fi
PREFIX=""
if command -v brew >/dev/null 2>&1; then PREFIX="-DCMAKE_PREFIX_PATH=$(brew --prefix)"; fi
cmake -S "$DEST" -B "$DEST/build" $PREFIX
cmake --build "$DEST/build" -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
"$DEST/build/val3dity" --version
echo
echo "export VAL3DITY=$DEST/build/val3dity"
