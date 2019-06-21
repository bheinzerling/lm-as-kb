#! /usr/bin/env bash
set -euo pipefail

ffmpeg -threads 8 -r 10 -f image2 -i scale_free_graph.alpha0.41_beta0.54_gamma0.05_seed2_n5000.n_nodes5000.n_edge_labels20.edge_label_distribution_argsa0.3.e%06d.png -vcodec libx264 -crf 25  -pix_fmt yv420p test.mp4
