#!/bin/bash
set -e

INPUT_DIR=$1
OUTPUT_DIR=$2
N_PARALLEL=${3:-16}

export PYTHONPATH=/masif/source:$PYTHONPATH

mkdir -p "$OUTPUT_DIR"

find "$INPUT_DIR" -name "*_protein.pdb" | head -200 | \
    xargs -P "$N_PARALLEL" -I {} \
    bash -c 'python /scripts/process_one.py "$1" "'"$OUTPUT_DIR"'" 2>&1 || echo "FAIL $(basename $1)"' _ {}
