from pathlib import Path

import typer
from Bio.PDB import PDBParser, PDBIO, Selection, StructureBuilder, Select
from Bio.SeqUtils import IUPACData

app = typer.Typer(pretty_exceptions_enable=False)

PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]


class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A" or atom.get_altloc() == "1"


def find_modified_amino_acids(path: Path) -> set[str]:
    res_set: set[str] = set()
    for line in path.open():
        if line[:6] == "SEQRES":
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set


def extract_pdb(infile: Path, outfile: Path) -> None:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(str(infile), str(infile))
    model = Selection.unfold_entities(struct, "M")[0]

    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    modified_amino_acids = find_modified_amino_acids(infile)

    for chain in model:
        structBuild.init_chain(chain.get_id())
        for residue in chain:
            het = residue.get_id()
            if het[0] == " ":
                outputStruct[0][chain.get_id()].add(residue)
            elif het[0][-3:] in modified_amino_acids:
                outputStruct[0][chain.get_id()].add(residue)

    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(str(outfile), select=NotDisordered())


@app.command()
def main(
    pdbbind_dir: Path = Path("data/pdbbind/P-L"),
    output_dir: Path = Path("data/protein_clean"),
) -> None:
    protein_pdbs = sorted(pdbbind_dir.rglob("*_protein.pdb"))

    skipped = 0
    processed = 0
    for protein_pdb in protein_pdbs:
        pdb_id = protein_pdb.parent.name
        out_dir = output_dir / pdb_id
        out_path = out_dir / f"{pdb_id}_protein_clean.pdb"
        if out_path.exists():
            skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        extract_pdb(protein_pdb, out_path)
        processed += 1
        if processed % 500 == 0:
            print(f"Processed {processed}...")

    print(f"Done. Processed: {processed}, skipped (already exist): {skipped}")


if __name__ == "__main__":
    app()
