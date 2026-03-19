from pathlib import Path

import modal
import torch
import biotite.structure as struc
from biotite.sequence.seqtypes import ProteinSequence
from biotite.structure.residues import get_residues

from apb.biomodal.esm2.constants import MAX_SEQ_LENGTH
from apb.biomodal.esm2.embedding import embed
from apb.biomodal.esm2.model import app as modal_app
from apb.structure.convert import load_structure

FULL_PDB = Path("data/pdbbind/P-L/1981-2000/10gs/10gs_protein.pdb")
POCKET_PDB = Path("data/masif_pocket_output/10gs/10gs_protein_processed_8A.pdb")
OUTPUT_DIR = Path("data/esm2_pocket_output")
PDB_ID = "10gs"


def get_chain_ids(structure: struc.AtomArray) -> list[str]:
    seen: set[str] = set()
    chain_ids: list[str] = []
    for cid in structure.chain_id:
        c = str(cid)
        if c not in seen:
            seen.add(c)
            chain_ids.append(c)
    return chain_ids


def get_chain_residue_keys(
    chain_atoms: struc.AtomArray,
) -> list[tuple[int, str]]:
    starts, _ = get_residues(chain_atoms)
    return [
        (int(chain_atoms.res_id[s]), str(chain_atoms.ins_code[s]))
        for s in starts
    ]


# --- Step 1: Load structures, filter to amino acids ---

full_structure = load_structure(FULL_PDB)
full_structure = full_structure[struc.filter_amino_acids(full_structure)]

pocket_structure = load_structure(POCKET_PDB)
pocket_structure = pocket_structure[struc.filter_amino_acids(pocket_structure)]

# --- Step 2: Extract per-chain sequences from full protein ---

full_chain_ids = get_chain_ids(full_structure)
print(f"Full protein chains: {full_chain_ids}")

sequences: dict[str, str] = {}
for i, chain_id in enumerate(full_chain_ids):
    chain_atoms = full_structure[full_structure.chain_id == chain_id]
    _, names = get_residues(chain_atoms)
    seq = str(ProteinSequence(names))
    key = f"{PDB_ID}_protein.pdb_chain_{i}"
    sequences[key] = seq
    print(f"  {key}: {len(seq)} residues")

# --- Step 3: Embed with ESM2 on Modal ---

chain_embeddings: dict[str, torch.Tensor] = {}

with modal.enable_output(), modal_app.run():
    for embeddings_tensor, seq_ids, seq_lengths in embed.local(
        seqs=sequences,
        model_name="facebook/esm2_t36_3B_UR50D",
        gpu="a10g",
        num_gpus=1,
        batch_size=8,
        pooling="none",
    ):
        for j, seq_id in enumerate(seq_ids):
            end = min(MAX_SEQ_LENGTH, seq_lengths[j]) + 1
            chain_embeddings[seq_id] = embeddings_tensor[j, 1:end]

print(f"\nEmbedded {len(chain_embeddings)} chains:")
for k, v in chain_embeddings.items():
    print(f"  {k}: {v.shape}")

# --- Step 4: Map pocket residues to full-protein chain embeddings ---

pocket_chain_ids = get_chain_ids(pocket_structure)
pocket_parts: list[torch.Tensor] = []

for i, chain_id in enumerate(full_chain_ids):
    if chain_id not in pocket_chain_ids:
        print(f"  Chain {chain_id}: not in pocket, skipping")
        continue

    full_chain = full_structure[full_structure.chain_id == chain_id]
    pocket_chain = pocket_structure[pocket_structure.chain_id == chain_id]

    full_keys = get_chain_residue_keys(full_chain)
    pocket_keys = get_chain_residue_keys(pocket_chain)
    pocket_key_set = set(pocket_keys)

    indices = [idx for idx, k in enumerate(full_keys) if k in pocket_key_set]

    emb_key = f"{PDB_ID}_protein.pdb_chain_{i}"
    chain_emb = chain_embeddings[emb_key]
    pocket_parts.append(chain_emb[indices])

    print(f"  Chain {chain_id}: {len(indices)} pocket residues (of {len(full_keys)} total)")

pocket_embedding = torch.cat(pocket_parts, dim=0).to(torch.float16)

# --- Step 5: Save ---

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / f"{PDB_ID}_protein_processed_8A.pt"
torch.save(pocket_embedding, output_path)

pocket_residue_count = sum(
    len(get_residues(pocket_structure[pocket_structure.chain_id == c])[0])
    for c in pocket_chain_ids
)
print(f"\nPocket embedding shape: {pocket_embedding.shape}")
print(f"Expected pocket residues: {pocket_residue_count}")
print(f"Match: {pocket_embedding.shape[0] == pocket_residue_count}")
print(f"Saved to: {output_path}")
