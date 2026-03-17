#!/bin/bash
docker run -d --name masif_run \
  -v /tmp/masif_pocket_output:/pocket_output \
  -v /tmp/masif_batch:/scripts \
  -v /tmp/P-L:/data \
  -v /tmp/masif_output:/output \
  pablogainza/masif:latest sleep infinity

echo "--- containers ---"
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
