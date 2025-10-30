#!/usr/bin/env bash
set -euo pipefail

CONFIG="config/spectral_analysis.yaml"

python scripts/analysis/get_activation.py --config $CONFIG
python scripts/analysis/track_geometry.py --config $CONFIG
python scripts/plot/visualize_geometry.py --config $CONFIG