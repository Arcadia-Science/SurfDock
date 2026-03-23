import shutil
from pathlib import Path

import typer

app = typer.Typer(pretty_exceptions_enable=False)

DATA_DIR = Path("data")
PDBBIND_DIR = DATA_DIR / "pdbbind" / "P-L"
PROTEIN_CLEAN_DIR = DATA_DIR / "protein_clean"
MASIF_8A_DIR = DATA_DIR / "masif_pocket_output"
MASIF_10A_DIR = DATA_DIR / "masif_pocket_output_10A"
OUTPUT_DIR = DATA_DIR / "processed"

YEAR_RANGES = ["1981-2000", "2001-2010", "2011-2019"]


def _find_pdbbind_dir(pdb_id: str) -> Path:
    for year_range in YEAR_RANGES:
        candidate = PDBBIND_DIR / year_range / pdb_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No PDBBind directory found for {pdb_id}")


def _collect_pdb_ids() -> list[str]:
    pdb_ids: set[str] = set()
    for year_range in YEAR_RANGES:
        year_dir = PDBBIND_DIR / year_range
        if year_dir.is_dir():
            for entry in year_dir.iterdir():
                if entry.is_dir():
                    pdb_ids.add(entry.name)
    return sorted(pdb_ids)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False


def consolidate_one(pdb_id: str) -> list[str]:
    warnings: list[str] = []
    out_dir = OUTPUT_DIR / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pdbbind_entry = _find_pdbbind_dir(pdb_id)

    copies = [
        (pdbbind_entry / f"{pdb_id}_protein.pdb", f"{pdb_id}_protein.pdb"),
        (pdbbind_entry / f"{pdb_id}_ligand.sdf", f"{pdb_id}_ligand.sdf"),
        (pdbbind_entry / f"{pdb_id}_ligand.mol2", f"{pdb_id}_ligand.mol2"),
        (PROTEIN_CLEAN_DIR / pdb_id / f"{pdb_id}_protein_clean.pdb", f"{pdb_id}_protein_clean.pdb"),
        (MASIF_8A_DIR / pdb_id / f"{pdb_id}_protein_processed_8A.pdb", f"{pdb_id}_protein_processed_8A.pdb"),
        (MASIF_8A_DIR / pdb_id / f"{pdb_id}_protein_processed_8A.ply", f"{pdb_id}_protein_processed_8A.ply"),
        (MASIF_10A_DIR / pdb_id / f"{pdb_id}_protein_processed_10A.pdb", f"{pdb_id}_protein_processed_10A.pdb"),
        (MASIF_10A_DIR / pdb_id / f"{pdb_id}_protein_processed_10A.ply", f"{pdb_id}_protein_processed_10A.ply"),
    ]

    for src, dst_name in copies:
        if not _copy_if_exists(src, out_dir / dst_name):
            if "10A" not in dst_name and "mol2" not in dst_name:
                warnings.append(f"{pdb_id}: missing {src}")

    return warnings


@app.command()
def main(dry_run: bool = False):
    pdb_ids = _collect_pdb_ids()
    typer.echo(f"Found {len(pdb_ids)} PDB entries to consolidate")

    if dry_run:
        typer.echo("Dry run — not copying any files")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_warnings: list[str] = []

    for i, pdb_id in enumerate(pdb_ids):
        warnings = consolidate_one(pdb_id)
        all_warnings.extend(warnings)
        if (i + 1) % 1000 == 0:
            typer.echo(f"  Processed {i + 1}/{len(pdb_ids)}")

    typer.echo(f"Done. Consolidated {len(pdb_ids)} entries into {OUTPUT_DIR}")
    if all_warnings:
        typer.echo(f"\n{len(all_warnings)} warnings:")
        for w in all_warnings:
            typer.echo(f"  {w}")


if __name__ == "__main__":
    app()
