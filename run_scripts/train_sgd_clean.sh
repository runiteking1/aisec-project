#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
    training=sgd \
    data=mnist \
    'hydra.run.dir=outputs/sgd_clean'
