#!/usr/bin/env bash
# Reproduce all experiments from the paper end-to-end.
# Trains all 6 models then runs the full analysis suite on each.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "====== Training: clean dataset ======"
bash "$SCRIPT_DIR/train_adam_clean.sh"
bash "$SCRIPT_DIR/train_sgd_clean.sh"
bash "$SCRIPT_DIR/train_gn_clean.sh"

echo "====== Training: poisoned dataset (10%) ======"
bash "$SCRIPT_DIR/train_adam_poisoned.sh"
bash "$SCRIPT_DIR/train_sgd_poisoned.sh"
bash "$SCRIPT_DIR/train_gn_poisoned.sh"

echo "====== Analysis: clean models ======"
bash "$SCRIPT_DIR/analyze.sh" outputs/adam_clean
bash "$SCRIPT_DIR/analyze.sh" outputs/sgd_clean
bash "$SCRIPT_DIR/analyze.sh" outputs/gn_clean

echo "====== Analysis: poisoned models ======"
bash "$SCRIPT_DIR/analyze.sh" outputs/adam_poisoned
bash "$SCRIPT_DIR/analyze.sh" outputs/sgd_poisoned
bash "$SCRIPT_DIR/analyze.sh" outputs/gn_poisoned

echo "====== All done ======"
