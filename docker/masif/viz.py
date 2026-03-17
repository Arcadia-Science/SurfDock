import numpy as np
import pyvista as pv
from pathlib import Path
from plyfile import PlyData
from rdkit import Chem
from typing import Optional


def load_surface(ply_path: Path) -> pv.PolyData:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    verts = np.column_stack([v["x"], v["y"], v["z"]])
    faces_raw = np.array([f[0] for f in ply["face"].data])
    n = faces_raw.shape[1]
    faces = np.column_stack([np.full(len(faces_raw), n), faces_raw]).ravel()
    mesh = pv.PolyData(verts, faces)
    for prop in v.properties:
        if prop.name not in ("x", "y", "z", "nx", "ny", "nz"):
            mesh.point_data[prop.name] = np.array(v[prop.name])
    normals = np.column_stack([v["nx"], v["ny"], v["nz"]])
    mesh.point_data["Normals"] = normals
    return mesh


def load_ligand(sdf_path: Path) -> pv.PolyData:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mol = next(iter(supplier))
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    return pv.PolyData(coords)


def load_ligand_bonds(sdf_path: Path) -> pv.PolyData:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mol = next(iter(supplier))
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    lines = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        lines.extend([2, i, j])
    if not lines:
        return None
    return pv.PolyData(coords, lines=np.array(lines))


def viz_pocket(
    pocket_ply: Path,
    ligand_sdf: Path,
    scalar: str = "charge",
    full_ply: Optional[Path] = None,
):
    pocket = load_surface(pocket_ply)
    ligand = load_ligand(ligand_sdf)
    name = pocket_ply.stem

    pl = pv.Plotter()
    pl.add_text(name, font_size=14)

    if full_ply is not None:
        full = load_surface(full_ply)
        pl.add_mesh(
            full,
            scalars=scalar,
            cmap="coolwarm",
            opacity=0.15,
            show_scalar_bar=False,
            smooth_shading=True,
        )

    pl.add_mesh(
        pocket,
        scalars=scalar,
        cmap="coolwarm",
        opacity=1.0,
        show_scalar_bar=True,
        scalar_bar_args={"title": scalar},
        smooth_shading=True,
    )

    pl.add_mesh(
        ligand,
        color="green",
        point_size=12,
        render_points_as_spheres=True,
    )

    bonds = load_ligand_bonds(ligand_sdf)
    if bonds is not None:
        pl.add_mesh(bonds, color="green", line_width=3)

    pl.background_color = "white"
    pl.show()


def viz_gallery(
    pocket_dir: Path,
    ligand_dir: Path,
    scalar: str = "charge",
    cols: int = 4,
    full_dir: Optional[Path] = None,
):
    ply_files = sorted(pocket_dir.glob("*.ply"))
    entries = []
    for ply_path in ply_files:
        name = ply_path.stem
        sdf_path = ligand_dir / f"{name}_ligand.sdf"
        if sdf_path.exists():
            entries.append((name, ply_path, sdf_path))

    n = len(entries)
    rows = (n + cols - 1) // cols

    pl = pv.Plotter(shape=(rows, cols))

    for idx, (name, ply_path, sdf_path) in enumerate(entries):
        row, col = divmod(idx, cols)
        pl.subplot(row, col)
        pl.add_text(name, font_size=10)

        if full_dir is not None:
            full_path = full_dir / f"{name}.ply"
            if full_path.exists():
                full = load_surface(full_path)
                pl.add_mesh(
                    full,
                    scalars=scalar,
                    cmap="coolwarm",
                    opacity=0.15,
                    show_scalar_bar=False,
                    smooth_shading=True,
                )

        pocket = load_surface(ply_path)
        pl.add_mesh(
            pocket,
            scalars=scalar,
            cmap="coolwarm",
            smooth_shading=True,
            show_scalar_bar=False,
        )

        ligand = load_ligand(sdf_path)
        pl.add_mesh(ligand, color="green", point_size=8, render_points_as_spheres=True)

        bonds = load_ligand_bonds(sdf_path)
        if bonds is not None:
            pl.add_mesh(bonds, color="green", line_width=2)

        pl.reset_camera()

    pl.background_color = "white"
    pl.show()


if __name__ == "__main__":
    import typer

    app = typer.Typer(pretty_exceptions_enable=False)

    @app.command()
    def single(
        pocket_ply: Path,
        ligand_sdf: Path,
        scalar: str = "charge",
        full_ply: Optional[Path] = None,
    ):
        viz_pocket(pocket_ply, ligand_sdf, scalar=scalar, full_ply=full_ply)

    @app.command()
    def gallery(
        pocket_dir: Path,
        ligand_dir: Path,
        scalar: str = "charge",
        cols: int = 4,
        full_dir: Optional[Path] = None,
    ):
        viz_gallery(pocket_dir, ligand_dir, scalar=scalar, cols=cols, full_dir=full_dir)

    app()
