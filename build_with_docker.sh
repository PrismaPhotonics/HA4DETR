#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-ha4detr-builder}"   # if first arg not supplied → default
WORKDIR="$(pwd)"                    # project root

echo "🔧 Using Docker image: $IMAGE_NAME"
echo "📁 Working directory : $WORKDIR"
echo ""

echo "🔍 Detecting version from git…"

if git describe --tags --abbrev=0 >/dev/null 2>&1; then
    LATEST_TAG="$(git describe --tags --abbrev=0)"
    echo "  Latest tag found: $LATEST_TAG"
else
    echo "⚠️  No tags found: falling back to dev version"
    LATEST_TAG="0.0.1"
fi

COMMIT_HASH="$(git rev-parse --short HEAD)"
VERSION="${LATEST_TAG}"

echo "📦 Version string: $VERSION"
echo ""

echo "Building the docker image to generate the package with for Linux version as $IMAGE_NAME"

docker build --network host -t "$IMAGE_NAME" -f Dockerfile.builder .

echo "🏗️ Building wheel inside Docker…, removing old artifacts from dist dir"

rm -rf "${WORKDIR}/dist"

echo "docker run --rm --gpus all --network host -u $(id -u):$(id -g) -v "${WORKDIR}":/workspace -w /workspace $IMAGE_NAME"

docker run --rm \
  --gpus all \
  --network host \
  -u $(id -u):$(id -g) \
  -v "${WORKDIR}":/workspace \
  -w /workspace \
  $IMAGE_NAME \
  bash -c "python -m build"

echo "successfully finish creating the python package in dist"
echo ""
echo "📝 Renaming wheel to: ha4detr-${VERSION}-linux_x86_64.whl"
ORIGINAL_WHL="$(ls dist/*.whl | head -n 1)"
RENAMED_WHL="dist/ha4detr-${VERSION}-linux_x86_64.whl"

mv "$ORIGINAL_WHL" "$RENAMED_WHL"

echo "➡ New wheel: $RENAMED_WHL"
echo "✔ Done!"
