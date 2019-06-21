#! /usr/bin/env bash
set -euo pipefail

runid=$1

python report.py plot_graph --no-commit --dataset syntheticpaths --graph-type scale_free_graph --graph-generator-args '{"alpha": 0.41, "beta": 0.54, "gamma": 0.05, "seed": 2}' --unique-targets --n-nodes 5000 --n-edge-labels 20 --n-paths 5000 --edge-label-distribution-args '{"a": 0.3}' --batch-size 128 --early-stopping -1 --model rnnpathmemory --model-variant LSTM --n-hidden 16 --emb-dim 16 --max-eval-n-inst 5000 --exp-name memorize_paths_5 --max-paths 5000 --max-epochs 10000 --tie-weights --graph-sweep --runid $runid --frame-every-n-epochs 1 --animate --format png --plot-threads 1
