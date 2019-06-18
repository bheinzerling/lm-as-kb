#! /usr/bin/env bash
set -euo pipefail

python synthetic_graph.py generate_graph --graph-type newman_watts_strogatz_graph --graph-generator-args '{"k": 2, "p": 0.55, "seed": 2}' --n-nodes 100 --n-edge-labels 10 --edge-label-distribution-args '{"a": 0.22}'
