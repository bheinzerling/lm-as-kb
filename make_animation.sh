#! /usr/bin/env bash
set -euo pipefail

runid=$1
convert -delay 10 -loop 0 out/$runid/*.png out/$runid/training.gif
