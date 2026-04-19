#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
    training=gn \
    data=mnist \
    'hydra.run.dir=outputs/gn_clean'
