#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
    training=default \
    data=mnist \
    data.poison=0.1 \
    'hydra.run.dir=outputs/adam_poisoned'
