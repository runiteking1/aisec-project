#!/usr/bin/env bash
# Run all analyses from the paper on a trained checkpoint.
# Usage: ./analyze.sh <checkpoint_dir>
#   e.g. ./analyze.sh outputs/adam_clean
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <checkpoint_dir>"
    echo "  e.g. $0 outputs/adam_clean"
    exit 1
fi

CKPT="$(realpath "$1")/checkpoint/model"

echo "=== Robustness analysis (gradient norms + logit margins) ==="
python -m src.analyze_robustness \
    checkpoint_path="$CKPT"

echo "=== SAM sharpness analysis ==="
for rho in 0.001 0.005 0.01; do
    echo "  rho=$rho"
    python -m src.analyze_sharpness \
        checkpoint_path="$CKPT" \
        rho="$rho"
done

echo "=== FGSM adversarial attack ==="
for epsilon in 0.1 0.05 0.01; do
    echo "  epsilon=$epsilon"
    python -m src.generate_adversarial \
        checkpoint_path="$CKPT" \
        epsilon="$epsilon"
done
