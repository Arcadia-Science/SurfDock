#!/usr/bin/python
import fcntl
import numpy as np
import os
import sys
import shutil
import pymesh
import Bio.PDB
from Bio.PDB import Selection, PDBIO, PDBParser, Select
from scipy.spatial import distance
from sklearn.neighbors import KDTree
sys.path.insert(0, "/masif/source")
sys.path.insert(0, "/scripts")
from computeMSMS_surfdock import computeMSMS
from triangulation.fixmesh import fix_mesh
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
from input_output.extractPDB import extractPDB
from input_output.protonate import protonate
from save_ply_surfdock import save_ply


DIST_THRESHOLD = 8

protein_file = sys.argv[1]
ligand_file = sys.argv[2]
out_dir = sys.argv[3]
tsv_path = sys.argv[4] if len(sys.argv) > 4 else None
pdb_id = os.path.basename(protein_file).replace("_protein.pdb", "")


def write_tsv(row):
    if tsv_path is None:
        return
    with open(tsv_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(row + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


tmp_dir = f"/tmp/masif_{pdb_id}_{os.getpid()}"
os.makedirs(tmp_dir, exist_ok=True)

complex_out_dir = os.path.join(out_dir, pdb_id)
os.makedirs(complex_out_dir, exist_ok=True)

out_ply = os.path.join(complex_out_dir, f"{pdb_id}_protein_processed_8A.ply")
out_pocket_pdb = os.path.join(complex_out_dir, f"{pdb_id}_protein_processed_8A.pdb")
if os.path.exists(out_ply):
    write_tsv(f"{pdb_id}\tSKIP\t\t")
    sys.exit(0)


def parse_ligand_heavy_atoms(filepath):
    coords = []
    if filepath.endswith(".sdf"):
        with open(filepath) as f:
            lines = f.readlines()
        counts_line = lines[3]
        n_atoms = int(counts_line[:3])
        for i in range(4, 4 + n_atoms):
            parts = lines[i].split()
            symbol = parts[3]
            if symbol == "H":
                continue
            coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
    elif filepath.endswith(".mol2"):
        in_atoms = False
        with open(filepath) as f:
            for line in f:
                if line.startswith("@<TRIPOS>ATOM"):
                    in_atoms = True
                    continue
                if line.startswith("@<TRIPOS>") and in_atoms:
                    break
                if in_atoms:
                    parts = line.split()
                    atom_type = parts[5].split(".")[0]
                    if atom_type == "H":
                        continue
                    coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
    return np.array(coords)


try:
    atom_coords = parse_ligand_heavy_atoms(ligand_file)

    cleaned_protein = os.path.join(tmp_dir, f"{pdb_id}_clean.pdb")
    extractPDB(protein_file, cleaned_protein, chain_ids=None)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", cleaned_protein)[0]
    atoms = Selection.unfold_entities(structure, "A")
    ns = Bio.PDB.NeighborSearch(atoms)

    close_residues = []
    for a in atom_coords:
        close_residues.extend(ns.search(a, DIST_THRESHOLD, level="R"))
    close_residues = Selection.uniqueify(close_residues)

    class SelectNeighbors(Select):
        def accept_residue(self, residue):
            if residue not in close_residues:
                return False
            atom_names = [i.get_name() for i in residue.get_unpacked_list()]
            if all(a in atom_names for a in ["N", "CA", "C", "O"]) or residue.resname == "HOH":
                return True
            return False

    pocket_pdb = os.path.join(tmp_dir, f"{pdb_id}_{DIST_THRESHOLD}A.pdb")
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(pocket_pdb, SelectNeighbors())

    structures2 = parser.get_structure("pocket", pocket_pdb)
    pocket_atoms = Selection.unfold_entities(structures2[0], "A")

    def try_msms(one_cavity):
        vertices, faces, normals, names, areas = computeMSMS(
            pocket_pdb, protonate=True, one_cavity=one_cavity
        )
        kdt = KDTree(atom_coords)
        d, _ = kdt.query(vertices)
        assert len(d) == len(vertices)
        iface_v = np.where(d <= DIST_THRESHOLD - 5)[0]
        faces_to_keep = [
            idx for idx, face in enumerate(faces) if all(v in iface_v for v in face)
        ]
        return vertices, faces, normals, names, areas, faces_to_keep

    # 3-tier MSMS fallback (matching SurfDock)
    msms_ok = False

    # Tier 1: atom closest to ligand centroid
    if not msms_ok:
        dist_to_centroid = [
            distance.euclidean(atom_coords.mean(axis=0), a.get_coord())
            for a in pocket_atoms
        ]
        atom_idx = np.argmin(dist_to_centroid)
        vertices1, faces1, normals1, names1, areas1, faces_to_keep = try_msms(atom_idx)
        msms_ok = True

    if not msms_ok:
        # Tier 2: atom with smallest min-distance to any ligand atom
        dist_matrix = [
            [distance.euclidean(ac, a.get_coord()) for ac in atom_coords]
            for a in pocket_atoms
        ]
        atom_idx = np.argsort(np.min(dist_matrix, axis=1))[0]
        vertices1, faces1, normals1, names1, areas1, faces_to_keep = try_msms(atom_idx)
        msms_ok = True

    if not msms_ok:
        # Tier 3: all_components fallback
        vertices1, faces1, normals1, names1, areas1, faces_to_keep = try_msms(None)

    vertex_hbond = computeCharges(os.path.splitext(cleaned_protein)[0], vertices1, names1)
    vertex_hphobicity = computeHydrophobicity(names1)

    mesh = pymesh.form_mesh(vertices1, faces1)
    mesh = pymesh.submesh(mesh, faces_to_keep, 0)
    regular_mesh = fix_mesh(mesh, 1.0)

    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

    vertex_hbond = assignChargesToNewMesh(
        regular_mesh.vertices, vertices1, vertex_hbond, {"feature_interpolation": True}
    )
    vertex_hphobicity = assignChargesToNewMesh(
        regular_mesh.vertices, vertices1, vertex_hphobicity, {"feature_interpolation": True}
    )
    vertex_charges = computeAPBS(
        regular_mesh.vertices, pocket_pdb, os.path.join(tmp_dir, pdb_id + "_temp")
    )

    regular_mesh.add_attribute("vertex_mean_curvature")
    H = regular_mesh.get_attribute("vertex_mean_curvature")
    regular_mesh.add_attribute("vertex_gaussian_curvature")
    K = regular_mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    elem[elem < 0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    si = (k1 + k2) / (k1 - k2)
    si = np.arctan(si) * (2 / np.pi)

    save_ply(
        out_ply,
        regular_mesh.vertices,
        regular_mesh.faces,
        normals=vertex_normal,
        charges=vertex_charges,
        normalize_charges=True,
        hbond=vertex_hbond,
        hphob=vertex_hphobicity,
        si=si,
    )

    shutil.copy(pocket_pdb, out_pocket_pdb)
    write_tsv(f"{pdb_id}\tOK\t{len(regular_mesh.vertices)}\t{len(regular_mesh.faces)}")

except Exception:
    write_tsv(f"{pdb_id}\tFAIL\t\t")
    raise

finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)
