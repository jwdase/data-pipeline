#!/usr/bin/env bash
# Promote staged Blender USDs scene001..scene050 into the data-set layout.
set -euo pipefail

cd "$(dirname "$0")"

for i in $(seq 1 50); do
    scene=$(printf "scene%03d" "$i")
    echo "=== ${scene} ==="
    uv run src/data_pipeline/gen_scene.py \
        --origin "${scene}" \
        --exp data-set \
        --scene "${scene}" \
        --device GPU
done
