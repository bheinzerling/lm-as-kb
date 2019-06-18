#! /usr/bin/env bash
set -euo pipefail

python main.py train --no-commit --dataset syntheticpaths --graph-type scale_free_graph --graph-generator-args '{"alpha": 0.41, "beta": 0.54, "gamma": 0.05, "seed": 2}' --unique-targets --n-nodes 10000 --n-edge-labels 20 --n-paths 10000 --edge-label-distribution-args '{"a": 0.3}' --batch-size 128 --early-stopping -1 --model rnnpathmemory --model-variant LSTM --n-hidden 10 --emb-dim 10 --max-eval-n-inst 1000 --exp-name memorize_paths_5 --max-paths 10000 --configs-per-job 3200 --tie-weights
