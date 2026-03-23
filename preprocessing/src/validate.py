from pathlib import Path

import typer
from apb.structure.convert import load_structure
from biotite.structure import filter_amino_acids

from preprocessing.src.utils import map_pocket_to_full_indices

app = typer.Typer(pretty_exceptions_enable=False)

DATA_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")
SPLIT_FILES = [
    "timesplit_no_lig_overlap_train",
    "timesplit_no_lig_overlap_val",
    "timesplit_test",
]


def load_all_pdb_ids() -> list[str]:
    pdb_ids: set[str] = set()
    for split_file in SPLIT_FILES:
        path = SPLITS_DIR / split_file
        pdb_ids.update(line.strip() for line in path.read_text().splitlines() if line.strip())
    return sorted(pdb_ids)


def check_files(pdb_ids: list[str]) -> list[str]:
    errors: list[str] = []
    for pdb_id in pdb_ids:
        entry_dir = DATA_DIR / pdb_id
        if not entry_dir.is_dir():
            errors.append(f"{pdb_id}: directory missing")
            continue

        required = [
            f"{pdb_id}_protein_clean.pdb",
            f"{pdb_id}_protein_processed_8A.pdb",
            f"{pdb_id}_protein_processed_8A.ply",
        ]
        for filename in required:
            if not (entry_dir / filename).exists():
                errors.append(f"{pdb_id}: missing {filename}")

        has_sdf = (entry_dir / f"{pdb_id}_ligand.sdf").exists()
        has_mol2 = (entry_dir / f"{pdb_id}_ligand.mol2").exists()
        if not has_sdf and not has_mol2:
            errors.append(f"{pdb_id}: missing ligand (no .sdf or .mol2)")

    return errors


def check_embeddings(pdb_ids: list[str], esm_model_name: str) -> list[str]:
    errors: list[str] = []
    for pdb_id in pdb_ids:
        entry_dir = DATA_DIR / pdb_id
        if not entry_dir.is_dir():
            continue
        for threshold in [8, 10]:
            pt_file = entry_dir / f"{pdb_id}_protein_processed_{threshold}A_{esm_model_name}.pt"
            if not pt_file.exists():
                errors.append(f"{pdb_id}: missing {pt_file.name}")
    return errors


@app.command()
def main(
    esm_model_name: str = "esm2_3B",
    skip_mapping: bool = False,
):
    pdb_ids = load_all_pdb_ids()
    typer.echo(f"Validating {len(pdb_ids)} PDB IDs from splits")

    typer.echo("\n--- File checks ---")
    file_errors = check_files(pdb_ids)
    if file_errors:
        typer.echo(f"{len(file_errors)} file errors:")
        for e in file_errors[:50]:
            typer.echo(f"  {e}")
        if len(file_errors) > 50:
            typer.echo(f"  ... and {len(file_errors) - 50} more")
    else:
        typer.echo("All files present")

    typer.echo("\n--- Embedding checks ---")
    emb_errors = check_embeddings(pdb_ids, esm_model_name)
    if emb_errors:
        typer.echo(f"{len(emb_errors)} embedding errors:")
        for e in emb_errors[:50]:
            typer.echo(f"  {e}")
        if len(emb_errors) > 50:
            typer.echo(f"  ... and {len(emb_errors) - 50} more")
    else:
        typer.echo("All embeddings present")

    if not skip_mapping:
        typer.echo("\n--- Pocket mapping checks ---")
        mapping_errors: list[str] = []
        for i, pdb_id in enumerate(pdb_ids):
            clean_path = DATA_DIR / pdb_id / f"{pdb_id}_protein_clean.pdb"
            pocket_path = DATA_DIR / pdb_id / f"{pdb_id}_protein_processed_8A.pdb"
            if not clean_path.exists() or not pocket_path.exists():
                continue

            full = load_structure(clean_path)
            full = full[filter_amino_acids(full)]
            pocket = load_structure(pocket_path)
            pocket = pocket[filter_amino_acids(pocket)]

            msg = _check_mapping_one(pdb_id, full, pocket)
            if msg:
                mapping_errors.append(msg)

            if (i + 1) % 1000 == 0:
                typer.echo(f"  Checked {i + 1}/{len(pdb_ids)}")

        if mapping_errors:
            typer.echo(f"{len(mapping_errors)} mapping errors:")
            for e in mapping_errors[:50]:
                typer.echo(f"  {e}")
            if len(mapping_errors) > 50:
                typer.echo(f"  ... and {len(mapping_errors) - 50} more")
        else:
            typer.echo("All pocket mappings valid")

    total_errors = len(file_errors) + len(emb_errors)
    if not skip_mapping:
        total_errors += len(mapping_errors)

    if total_errors == 0:
        typer.echo("\nValidation passed")
    else:
        typer.echo(f"\nValidation found {total_errors} total errors")
        raise typer.Exit(code=1)


def _check_mapping_one(pdb_id: str, full, pocket) -> str:
    result = map_pocket_to_full_indices(full, pocket)
    if not result:
        return f"{pdb_id}: empty pocket mapping"
    return ""


if __name__ == "__main__":
    app()
