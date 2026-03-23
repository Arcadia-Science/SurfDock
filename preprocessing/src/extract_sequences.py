"""Extract per-chain protein sequences from cleaned PDBs.

Outputs:
    - Per-PDB FASTA files: data/processed/{pdb_id}/{pdb_id}_chains.fasta
    - Index TSV: data/processed/chain_index.tsv
"""

import csv
from pathlib import Path

import typer
from apb.biomodal.esm2.constants import MAX_SEQ_LENGTH
from apb.structure.convert import load_structure
from biotite.structure import filter_amino_acids

from preprocessing.src.utils import get_per_chain_sequences

app = typer.Typer(pretty_exceptions_enable=False)

DATA_DIR = Path("data/processed")
INDEX_FILENAME = "chain_index.tsv"
FASTA_SUFFIX = "_chains.fasta"


def load_pdb_ids() -> list[str]:
    return sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())


def extract_one(pdb_id: str) -> list[dict[str, str]]:
    """Extracts chain sequences for one PDB ID. Returns index rows and writes FASTA."""
    clean_pdb = DATA_DIR / pdb_id / f"{pdb_id}_protein_clean.pdb"
    structure = load_structure(clean_pdb)
    structure = structure[filter_amino_acids(structure)]
    chain_seqs = get_per_chain_sequences(structure)

    fasta_path = DATA_DIR / pdb_id / f"{pdb_id}{FASTA_SUFFIX}"
    rows: list[dict[str, str]] = []

    with fasta_path.open("w") as f:
        for i, (chain_id, seq) in enumerate(chain_seqs.items()):
            key = f"{pdb_id}|{chain_id}|{i}"
            truncated = "true" if len(seq) > MAX_SEQ_LENGTH else "false"
            f.write(f">{key}\n{seq}\n")
            rows.append({
                "key": key,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "chain_index": str(i),
                "length": str(len(seq)),
                "truncated": truncated,
            })

    return rows


@app.command()
def main():
    pdb_ids = load_pdb_ids()
    typer.echo(f"Found {len(pdb_ids)} PDB IDs in {DATA_DIR}")

    index_path = DATA_DIR / INDEX_FILENAME
    fieldnames = ["key", "pdb_id", "chain_id", "chain_index", "length", "truncated"]

    all_rows: list[dict[str, str]] = []
    skipped: list[str] = []
    n_truncated = 0

    for i, pdb_id in enumerate(pdb_ids):
        clean_pdb = DATA_DIR / pdb_id / f"{pdb_id}_protein_clean.pdb"
        if not clean_pdb.exists():
            skipped.append(pdb_id)
            continue
        rows = extract_one(pdb_id)
        all_rows.extend(rows)
        n_truncated += sum(1 for r in rows if r["truncated"] == "true")

        if (i + 1) % 2000 == 0:
            typer.echo(f"  {i + 1}/{len(pdb_ids)}")

    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)

    typer.echo(f"Wrote {len(all_rows)} chains across {len(pdb_ids) - len(skipped)} PDB IDs")
    typer.echo(f"Index: {index_path}")
    if skipped:
        typer.echo(f"{len(skipped)} PDB IDs skipped (missing clean PDB)")
    if n_truncated:
        typer.echo(f"{n_truncated} chains exceed MAX_SEQ_LENGTH ({MAX_SEQ_LENGTH})")


if __name__ == "__main__":
    app()
