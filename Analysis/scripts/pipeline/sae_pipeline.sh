set -euo pipefail

CONFIG="../config/SAE_mode_analysis.yaml"

python ../analysis/get_activation.py --config $CONFIG
# python scripts/analysis/track_geometry.py --config $CONFIG
# python scripts/plot/visualize_geometry.py --config $CONFIG