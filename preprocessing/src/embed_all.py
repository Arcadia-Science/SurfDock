import csv
from pathlib import Path

import modal
import torch
import typer
from apb.biomodal.esm2.constants import MAX_SEQ_LENGTH
from apb.structure.convert import load_structure
from biotite.structure import filter_amino_acids

from preprocessing.src.utils import (
    get_chain_ids,
    map_pocket_to_full_indices,
)

app = typer.Typer(pretty_exceptions_enable=False)

DATA_DIR = Path("data/processed")
POCKET_THRESHOLDS = [8, 10]

MODEL_SHORT_NAMES: dict[str, str] = {
    "facebook/esm2_t6_8M_UR50D": "esm2_8M",
    "facebook/esm2_t12_35M_UR50D": "esm2_35M",
    "facebook/esm2_t30_150M_UR50D": "esm2_150M",
    "facebook/esm2_t33_650M_UR50D": "esm2_650M",
    "facebook/esm2_t36_3B_UR50D": "esm2_3B",
    "facebook/esm2_t48_15B_UR50D": "esm2_15B",
}
DEFAULT_MODEL = "facebook/esm2_t36_3B_UR50D"


def load_index() -> list[dict[str, str]]:
    index_path = DATA_DIR / "chain_index.tsv"
    with index_path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_sequences_from_fastas(rows: list[dict[str, str]]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    loaded_pdbs: set[str] = set()

    for row in rows:
        pdb_id = row["pdb_id"]
        if pdb_id in loaded_pdbs:
            continue
        loaded_pdbs.add(pdb_id)

        fasta_path = DATA_DIR / pdb_id / f"{pdb_id}_chains.fasta"
        current_key = ""
        current_seq_parts: list[str] = []
        for line in fasta_path.read_text().splitlines():
            if line.startswith(">"):
                if current_key:
                    sequences[current_key] = "".join(current_seq_parts)
                current_key = line[1:].strip()
                current_seq_parts = []
            else:
                current_seq_parts.append(line.strip())
        if current_key:
            sequences[current_key] = "".join(current_seq_parts)

    return sequences


def create_pdb_batches(keys: list[str], batch_size: int) -> list[list[str]]:
    """Batches chain keys keeping all chains for a PDB ID in the same batch."""
    groups: dict[str, list[str]] = {}
    for key in keys:
        pdb_id = key.split("|")[0]
        groups.setdefault(pdb_id, []).append(key)

    batches: list[list[str]] = []
    current_batch: list[str] = []
    for group in groups.values():
        if current_batch and len(current_batch) + len(group) > batch_size:
            batches.append(current_batch)
            current_batch = []
        current_batch.extend(group)
    if current_batch:
        batches.append(current_batch)
    return batches


def extract_pocket_embedding(
    pdb_id: str,
    chain_embeddings: dict[str, torch.Tensor],
    full_chain_ids: list[str],
    full_structure,
    pocket_pdb: Path,
) -> torch.Tensor:
    pocket = load_structure(pocket_pdb)
    pocket = pocket[filter_amino_acids(pocket)]
    indices_per_chain = map_pocket_to_full_indices(full_structure, pocket)

    pocket_parts: list[torch.Tensor] = []
    for i, chain_id in enumerate(full_chain_ids):
        if chain_id not in indices_per_chain:
            continue
        emb_key = f"{pdb_id}|{chain_id}|{i}"
        chain_emb = chain_embeddings[emb_key]
        emb_len = chain_emb.shape[0]
        indices = [idx for idx in indices_per_chain[chain_id] if idx < emb_len]
        if indices:
            pocket_parts.append(chain_emb[indices])

    if not pocket_parts:
        return None
    return torch.cat(pocket_parts, dim=0)


def process_pdb_pockets(
    pdb_id: str,
    chain_embeddings: dict[str, torch.Tensor],
    model_short_name: str,
) -> tuple[dict[int, tuple[str, torch.Tensor]], list[str]]:
    """Returns pocket embeddings keyed by threshold, plus warnings."""
    warnings: list[str] = []
    results: dict[int, tuple[str, torch.Tensor]] = {}

    clean_pdb = DATA_DIR / pdb_id / f"{pdb_id}_protein_clean.pdb"
    full = load_structure(clean_pdb)
    full = full[filter_amino_acids(full)]
    full_chain_ids = get_chain_ids(full)

    for threshold in POCKET_THRESHOLDS:
        pocket_pdb = DATA_DIR / pdb_id / f"{pdb_id}_protein_processed_{threshold}A.pdb"
        if not pocket_pdb.exists():
            warnings.append(f"{pdb_id}: missing {threshold}A pocket PDB")
            continue
        pocket_emb = extract_pocket_embedding(
            pdb_id, chain_embeddings, full_chain_ids, full, pocket_pdb
        )
        if pocket_emb is None:
            warnings.append(f"{pdb_id}: no pocket residues within embedding range for {threshold}A")
            continue
        output_key = f"{pdb_id}_protein_processed_{threshold}A_{model_short_name}"
        results[threshold] = (output_key, pocket_emb)

    return results, warnings


@app.command()
def main(
    model: str = DEFAULT_MODEL,
    gpu: str = "a10g",
    num_gpus: int = 4,
    batch_size: int = 48,
    max_batches: int = 0,
):
    model_short_name = MODEL_SHORT_NAMES[model]
    typer.echo(f"Model: {model} ({model_short_name})")

    typer.echo("Loading chain index and FASTA sequences...")
    rows = load_index()
    sequences = load_sequences_from_fastas(rows)
    typer.echo(f"Loaded {len(sequences)} chain sequences")

    done_pdb_ids: set[str] = set()
    for key in list(sequences.keys()):
        pdb_id = key.split("|")[0]
        if pdb_id in done_pdb_ids:
            del sequences[key]
            continue
        expected = [
            DATA_DIR / pdb_id / f"{pdb_id}_protein_processed_{t}A_{model_short_name}.pt"
            for t in POCKET_THRESHOLDS
        ]
        if all(p.exists() for p in expected):
            done_pdb_ids.add(pdb_id)
            del sequences[key]

    if done_pdb_ids:
        typer.echo(f"Skipping {len(done_pdb_ids)} PDB IDs with existing embeddings")
    typer.echo(f"{len(sequences)} chain sequences remaining")

    batches = create_pdb_batches(list(sequences.keys()), batch_size)

    if max_batches > 0:
        batches = batches[:max_batches]
        typer.echo(f"Limiting to {max_batches} batches")
    typer.echo(f"Created {len(batches)} batches")

    sequence_batches = [[sequences[key] for key in batch] for batch in batches]
    all_warnings: list[str] = []
    n_processed = 0
    n_saved = 0

    typer.echo(f"Running ESM2 on deployed Modal app ({gpu} x{num_gpus}, batch_size={batch_size})...")

    EsmModel = modal.Cls.from_name("esm2", "EsmModel")
    worker = EsmModel.with_options(gpu=gpu, max_containers=num_gpus)(model)
    args = [(i, seq_batch, "none") for i, seq_batch in enumerate(sequence_batches)]

    with modal.enable_output():
        starmap_results = worker.embed.starmap(args, order_outputs=False)

    for batch_index, embeddings_tensor in starmap_results:
        key_batch = batches[batch_index]
        chain_embeddings: dict[str, torch.Tensor] = {}
        for j, key in enumerate(key_batch):
            seq_len = len(sequences[key])
            end = min(MAX_SEQ_LENGTH, seq_len) + 1
            chain_embeddings[key] = embeddings_tensor[j, 1:end]

        pdb_ids_in_batch = list(dict.fromkeys(k.split("|")[0] for k in key_batch))
        for pdb_id in pdb_ids_in_batch:
            results_for_pdb, warnings = process_pdb_pockets(pdb_id, chain_embeddings, model_short_name)
            all_warnings.extend(warnings)
            for threshold, (output_key, pocket_emb) in results_for_pdb.items():
                out_path = DATA_DIR / pdb_id / f"{output_key}.pt"
                torch.save(pocket_emb, out_path)
                n_saved += 1
            n_processed += 1

        if n_processed % 500 < len(pdb_ids_in_batch):
            typer.echo(f"  Processed {n_processed} PDB IDs, saved {n_saved} embeddings")

    typer.echo(f"Done. Saved {n_saved} pocket embedding files across {n_processed} PDB IDs")

    if all_warnings:
        typer.echo(f"\n{len(all_warnings)} warnings:")
        for w in all_warnings:
            typer.echo(f"  {w}")


if __name__ == "__main__":
    app()
