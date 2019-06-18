xargs -P3 -I{} bash -c "{}" <<EOF
. /home/acb10857xr/conda/bin/activate pt1 && python --version
. /home/acb10857xr/conda/bin/activate pt1 && python --version
python --version
python --version
wait
EOF
