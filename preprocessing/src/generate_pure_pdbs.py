"""Generate hydrogen-stripped "_pure.pdb" files from pocket PDBs.

For each pocket PDB, strips hydrogen atoms using BioPython. The resulting
file is what MDAnalysis loads during training graph construction.
"""
from pathlib import Path

import typer
from Bio.PDB import PDBParser, PDBIO, Select

app = typer.Typer(pretty_exceptions_enable=False)

DATA_DIR = Path("data/processed")
POCKET_THRESHOLDS = ["8A", "10A"]


class NoHydrogens(Select):
    def accept_atom(self, atom):
        return atom.element != "H"


def generate_pure_pdb(pocket_pdb: Path, pure_pdb: Path) -> None:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pocket_pdb.stem, str(pocket_pdb))
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(pure_pdb), select=NoHydrogens())


@app.command()
def main():
    pdb_ids = sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    typer.echo(f"Found {len(pdb_ids)} PDB IDs in {DATA_DIR}")

    n_generated = 0
    n_skipped = 0

    for i, pdb_id in enumerate(pdb_ids):
        for threshold in POCKET_THRESHOLDS:
            pocket_pdb = DATA_DIR / pdb_id / f"{pdb_id}_protein_processed_{threshold}.pdb"
            pure_pdb = DATA_DIR / pdb_id / f"{pdb_id}_protein_processed_{threshold}_pure.pdb"

            if not pocket_pdb.exists():
                continue
            if pure_pdb.exists():
                n_skipped += 1
                continue

            generate_pure_pdb(pocket_pdb, pure_pdb)
            n_generated += 1

        if (i + 1) % 2000 == 0:
            typer.echo(f"  {i + 1}/{len(pdb_ids)}")

    typer.echo(f"Done. Generated: {n_generated}, skipped (exist): {n_skipped}")


if __name__ == "__main__":
    app()
