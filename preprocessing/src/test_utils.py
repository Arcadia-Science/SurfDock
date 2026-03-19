from pathlib import Path

import biotite.structure as struc
from apb.structure.convert import load_structure

from preprocessing.src.utils import (
    get_chain_ids,
    get_chain_sequence,
    get_chain_residue_keys,
    get_per_chain_sequences,
    map_pocket_to_full_indices,
)

TEST_DATA = Path("preprocessing/test_data")
CLEAN_PDB = TEST_DATA / "10gs_protein_clean.pdb"
POCKET_PDB = TEST_DATA / "10gs_protein_processed_8A.pdb"
INS_CODE_CLEAN_PDB = TEST_DATA / "1a2c_protein_clean.pdb"
INS_CODE_POCKET_PDB = TEST_DATA / "1a2c_protein_processed_8A.pdb"


def test_clean_pdb_has_no_non_amino_acids():
    structure = load_structure(CLEAN_PDB)
    assert set(structure.res_name) == set(
        structure[struc.filter_amino_acids(structure)].res_name
    )


def test_get_chain_ids_preserves_both_chains():
    structure = load_structure(CLEAN_PDB)
    assert get_chain_ids(structure) == ["A", "B"]


def test_get_chain_sequence_returns_string():
    structure = load_structure(CLEAN_PDB)
    chain_a = structure[structure.chain_id == "A"]
    seq = get_chain_sequence(chain_a)
    assert isinstance(seq, str)
    assert len(seq) > 0
    assert all(c.isalpha() for c in seq)


def test_get_chain_sequence_correct_length():
    structure = load_structure(CLEAN_PDB)
    chain_a = structure[structure.chain_id == "A"]
    chain_b = structure[structure.chain_id == "B"]
    assert len(get_chain_sequence(chain_a)) == 208
    assert len(get_chain_sequence(chain_b)) == 208


def test_homodimer_chains_have_same_sequence():
    structure = load_structure(CLEAN_PDB)
    chain_a = structure[structure.chain_id == "A"]
    chain_b = structure[structure.chain_id == "B"]
    assert get_chain_sequence(chain_a) == get_chain_sequence(chain_b)


def test_get_chain_residue_ids_length_matches_sequence():
    structure = load_structure(CLEAN_PDB)
    chain_a = structure[structure.chain_id == "A"]
    assert len(get_chain_residue_keys(chain_a)) == len(get_chain_sequence(chain_a))


def test_get_chain_residue_ids_unique():
    structure = load_structure(CLEAN_PDB)
    chain_a = structure[structure.chain_id == "A"]
    ids = get_chain_residue_keys(chain_a)
    assert len(ids) == len(set(ids))


def test_get_per_chain_sequences():
    structure = load_structure(CLEAN_PDB)
    seqs = get_per_chain_sequences(structure)
    assert len(seqs) == 2
    assert "A" in seqs
    assert "B" in seqs
    assert len(seqs["A"]) == 208


def test_pocket_chains():
    pocket = load_structure(POCKET_PDB)
    chain_ids = get_chain_ids(pocket)
    assert "A" in chain_ids


def test_pocket_is_subset_of_clean():
    full = load_structure(CLEAN_PDB)
    pocket = load_structure(POCKET_PDB)
    for chain_id in get_chain_ids(pocket):
        full_ids = set(get_chain_residue_keys(full[full.chain_id == chain_id]))
        pocket_ids = set(get_chain_residue_keys(pocket[pocket.chain_id == chain_id]))
        assert pocket_ids.issubset(full_ids)


def test_map_pocket_to_full_indices_count():
    full = load_structure(CLEAN_PDB)
    pocket = load_structure(POCKET_PDB)
    indices_per_chain = map_pocket_to_full_indices(full, pocket)

    total = sum(len(v) for v in indices_per_chain.values())
    expected = sum(
        len(get_chain_residue_keys(pocket[pocket.chain_id == c]))
        for c in get_chain_ids(pocket)
    )
    assert total == expected


def test_map_pocket_indices_are_valid():
    full = load_structure(CLEAN_PDB)
    pocket = load_structure(POCKET_PDB)
    indices_per_chain = map_pocket_to_full_indices(full, pocket)

    for chain_id, indices in indices_per_chain.items():
        n_residues = len(get_chain_residue_keys(full[full.chain_id == chain_id]))
        for idx in indices:
            assert 0 <= idx < n_residues


def test_map_pocket_indices_are_sorted():
    full = load_structure(CLEAN_PDB)
    pocket = load_structure(POCKET_PDB)
    indices_per_chain = map_pocket_to_full_indices(full, pocket)

    for _, indices in indices_per_chain.items():
        assert indices == sorted(indices)


def test_ins_code_residue_ids_unique():
    structure = load_structure(INS_CODE_CLEAN_PDB)
    chain_h = structure[structure.chain_id == "H"]
    ids = get_chain_residue_keys(chain_h)
    assert len(ids) == len(set(ids))


def test_ins_code_pocket_is_subset():
    full = load_structure(INS_CODE_CLEAN_PDB)
    pocket = load_structure(INS_CODE_POCKET_PDB)
    for chain_id in get_chain_ids(pocket):
        full_ids = set(get_chain_residue_keys(full[full.chain_id == chain_id]))
        pocket_ids = set(get_chain_residue_keys(pocket[pocket.chain_id == chain_id]))
        assert pocket_ids.issubset(full_ids)


def test_ins_code_map_pocket_to_full_indices():
    full = load_structure(INS_CODE_CLEAN_PDB)
    pocket = load_structure(INS_CODE_POCKET_PDB)
    indices_per_chain = map_pocket_to_full_indices(full, pocket)
