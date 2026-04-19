#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
    training=sgd \
    data=mnist \
    data.poison=0.1 \
    'hydra.run.dir=outputs/sgd_poisoned'
