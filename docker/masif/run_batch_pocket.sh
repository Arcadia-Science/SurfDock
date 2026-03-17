#!/bin/bash
set -e

INPUT_DIR=$1
OUTPUT_DIR=$2
N_PARALLEL=${3:-14}
LIMIT=${4:-20}

mkdir -p "$OUTPUT_DIR"

TSV="$OUTPUT_DIR/run_$(date +%Y%m%d_%H%M%S).tsv"
printf 'id\tstatus\tvertices\tfaces\n' > "$TSV"

process_one() {
    PDB_FILE=$1
    OUT_DIR=$2
    TSV_FILE=$3
    DIR=$(dirname "$PDB_FILE")
    ID=$(basename "$DIR")
    LIGAND="$DIR/${ID}_ligand.sdf"
    if [ -f "$LIGAND" ]; then
        python /scripts/process_one_pocket.py "$PDB_FILE" "$LIGAND" "$OUT_DIR" "$TSV_FILE" 2>&1 | tail -1
    else
        echo "SKIP $ID: no ligand"
    fi
}
export -f process_one

find "$INPUT_DIR" -name "*_protein.pdb" | sort | head -"$LIMIT" | \
    xargs -P "$N_PARALLEL" -I {} bash -c "process_one '{}' '$OUTPUT_DIR' '$TSV'"
