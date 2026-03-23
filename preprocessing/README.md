# Preprocessing Pipeline

Re-implementation of SurfDock's data preprocessing using biotite, APB, and Modal.
Outputs match what `datasets/pdbbind.py` expects for training/inference.

## Prerequisites

- Raw PDBBind structures in `data/pdbbind/P-L/{year_range}/{pdb_id}/`
- Cleaned proteins in `data/protein_clean/{pdb_id}/`
- MaSIF pocket outputs in `data/masif_pocket_output/{pdb_id}/` (8A) and `data/masif_pocket_output_10A/{pdb_id}/` (10A)
- Split files in `data/splits/`
- Deployed `esm2` Modal app (from APB)

## Pipeline

Run these in order:

### 1. Clean PDBs

Strip non-amino-acid residues and disordered atoms from raw PDBBind proteins.

```
uv run python -m preprocessing.src.clean_pdbs
```

Reads from `data/pdbbind/P-L/`, writes to `data/protein_clean/{pdb_id}/{pdb_id}_protein_clean.pdb`.

### 2. Compute surfaces (Docker)

Run the MaSIF Docker pipeline to generate pocket PDBs and surface PLY files.
Requires an x86_64 machine (MSMS binary doesn't work under ARM emulation).

See `preprocessing/docker/masif/` for Dockerfile and scripts.

### 3. Consolidate data

Merge all per-complex files from scattered directories into a flat structure.

```
uv run python -m preprocessing.src.consolidate_data
uv run python -m preprocessing.src.consolidate_data --dry-run  # preview only
```

Copies files into `data/processed/{pdb_id}/` with this layout:
```
data/processed/{pdb_id}/
  {pdb_id}_protein.pdb              # raw protein
  {pdb_id}_protein_clean.pdb        # cleaned protein
  {pdb_id}_protein_processed_8A.pdb # 8A pocket PDB
  {pdb_id}_protein_processed_8A.ply # 8A pocket surface mesh
  {pdb_id}_protein_processed_10A.pdb # 10A pocket PDB (if available)
  {pdb_id}_protein_processed_10A.ply # 10A pocket surface mesh (if available)
  {pdb_id}_ligand.sdf               # ligand SDF
  {pdb_id}_ligand.mol2              # ligand MOL2
```

### 4. Extract sequences

Extract per-chain amino acid sequences from cleaned proteins and write FASTA files + a chain index.

```
uv run python -m preprocessing.src.extract_sequences
```

Writes:
- `data/processed/{pdb_id}/{pdb_id}_chains.fasta` — per-chain sequences
- `data/processed/chain_index.tsv` — index of all chains with metadata (length, truncation status)

### 5. Compute ESM2 embeddings

Run ESM2 on all chains via the deployed Modal app, subsample to pocket residues, save per-complex `.pt` files.

```
uv run python -m preprocessing.src.embed_all
uv run python -m preprocessing.src.embed_all --max-batches 3 --num-gpus 1  # small test run
```

Writes `data/processed/{pdb_id}/{pdb_id}_protein_processed_{8,10}A.pt` — each a `Tensor[n_pocket_residues, 2560]`.

Embeddings are computed on the full protein sequence and then subsampled to pocket residues to preserve sequence context.

### 6. Validate

Check that all PDB IDs in the split files have complete data.

```
uv run python -m preprocessing.src.validate
uv run python -m preprocessing.src.validate --skip-mapping  # skip pocket mapping checks (faster)
```

Checks: file existence (clean PDB, pocket PDB+PLY, ligand, embedding `.pt`), and optionally runs `map_pocket_to_full_indices` on every entry.

## Shared utilities

- `preprocessing/src/utils.py` — Structure manipulation functions (chain extraction, sequence extraction, pocket-to-full residue mapping). Uses biotite.
- `preprocessing/src/test_utils.py` — Tests for utils. Run with `uv run pytest preprocessing/src/test_utils.py`.

## Training integration

Point the training script at the consolidated data:
```
--data_dir=data/processed --surface_path=data/processed --esm_embeddings_path=data/processed
```

The `PDBBind` dataset class loads per-complex embedding files from `{esm_embeddings_path}/{pdb_id}/{pdb_id}_protein_processed_8A.pt`.
