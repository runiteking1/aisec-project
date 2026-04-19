#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
    training=default \
    data=mnist \
    'hydra.run.dir=outputs/adam_clean'
