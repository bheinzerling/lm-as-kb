#! /usr/bin/env bash
set -euo pipefail

# graph_type=
# k=3
# for p in 10 20 30 40 50 60 70 80 90; do
# 	echo k=$k p=$p
# 	python report.py plot_graph --no-commit --dataset syntheticpaths --graph-type newman_watts_strogatz_graph --graph-generator-args '{"k": '$k', "p": 0.'$p', "seed": 2}' --unique-targets --n-nodes 1000 --n-edge-labels 10 --edge-label-distribution-args '{"a": 0.3}' --exp-name generate_graph --graph-sweep
# done

# for m in $(seq 1 6); do
# m=3
# 	# for p in 10 20 30 40 50 60 70 80 90; do
# 	for p in 80 85 88 90 93 95 97; do
# 		echo m=$m p=$p
# 		python report.py plot_graph --no-commit --dataset syntheticpaths --graph-type powerlaw_cluster_graph --graph-generator-args '{"m": '$m', "p": 0.'$p', "seed": 2}' --unique-targets --n-nodes 10000 --n-edge-labels 30 --edge-label-distribution-args '{"a": 0.3}' --exp-name generate_graph --graph-sweep
# 	done
# # done

# m=3
	# for p in 10 20 30 40 50 60 70 80 90; do
	# for p in 80 85 88 90 93 95 97; do
		# echo m=$m p=$p
		python report.py plot_graph --no-commit --dataset syntheticpaths --graph-type scale_free_graph --graph-generator-args '{"alpha": 0.41, "beta": 0.54, "gamma": 0.05, "seed": 2}' --unique-targets --n-nodes 20000 --n-edge-labels 30 --edge-label-distribution-args '{"a": 0.3}' --exp-name generate_graph --graph-sweep
	# done
# done
