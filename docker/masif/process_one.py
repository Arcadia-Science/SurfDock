#!/usr/bin/python
import numpy as np
import os
import sys
import pymesh
sys.path.insert(0, "/masif/source")
from triangulation.computeMSMS import computeMSMS
from triangulation.fixmesh import fix_mesh
from input_output.save_ply import save_ply
from input_output.extractPDB import extractPDB
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal

pdb_file = sys.argv[1]
out_dir = sys.argv[2]
pdb_id = os.path.basename(pdb_file).replace("_protein.pdb", "")

tmp_dir = f"/tmp/masif_{pdb_id}_{os.getpid()}"
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

out_ply = os.path.join(out_dir, f"{pdb_id}.ply")
if os.path.exists(out_ply):
    print(f"SKIP {pdb_id}: already exists")
    sys.exit(0)

cleaned_file = os.path.join(tmp_dir, f"{pdb_id}_clean.pdb")
extractPDB(pdb_file, cleaned_file, chain_ids=None)

protonated_file = os.path.join(tmp_dir, f"{pdb_id}.pdb")
protonate(cleaned_file, protonated_file)

vertices1, faces1, normals1, names1, areas1 = computeMSMS(protonated_file, protonate=True)

vertex_hbond = computeCharges(os.path.splitext(protonated_file)[0], vertices1, names1)
vertex_hphobicity = computeHydrophobicity(names1)

mesh = pymesh.form_mesh(vertices1, faces1)
regular_mesh = fix_mesh(mesh, 1.0)

vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, {"feature_interpolation": True})
vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, {"feature_interpolation": True})
vertex_charges = computeAPBS(regular_mesh.vertices, protonated_file, os.path.join(tmp_dir, pdb_id))

save_ply(
    out_ply,
    regular_mesh.vertices,
    regular_mesh.faces,
    normals=vertex_normal,
    charges=vertex_charges,
    normalize_charges=True,
    hbond=vertex_hbond,
    hphob=vertex_hphobicity,
)

print(f"OK {pdb_id}: {len(regular_mesh.vertices)} vertices, {len(regular_mesh.faces)} faces")

import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
