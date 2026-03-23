from biotite.structure.atoms import AtomArray
from biotite.structure.info import one_letter_code
from biotite.structure.residues import get_residue_starts, get_residues


def get_chain_ids(structure: AtomArray) -> list[str]:
    return list(dict.fromkeys(str(c) for c in structure.chain_id))


def get_chain_sequence(chain_atoms: AtomArray) -> str:
    _, amino_acids = get_residues(chain_atoms)
    return "".join(one_letter_code(str(aa)) or "X" for aa in amino_acids)


def get_chain_residue_keys(chain_atoms: AtomArray) -> list[tuple[int, str]]:
    starts = get_residue_starts(chain_atoms)
    keys = [(int(chain_atoms.res_id[s]), str(chain_atoms.ins_code[s])) for s in starts]
    assert len(keys) == len(set(keys)), (
        f"Duplicate (res_id, ins_code) in chain: {len(keys)} total, {len(set(keys))} unique"
    )
    return keys


def get_per_chain_sequences(structure: AtomArray) -> dict[str, str]:
    return {
        chain_id: get_chain_sequence(structure[structure.chain_id == chain_id])
        for chain_id in get_chain_ids(structure)
    }


def map_pocket_to_full_indices(
    full: AtomArray, pocket: AtomArray
) -> dict[str, list[int]]:
    """For each chain in the pocket, return 0-indexed residue positions into the
    full protein's chain. These index into the residue list returned by
    get_residues(full_chain), so position 0 is the first residue of that chain.

    Validates that the amino acids at the returned positions match the pocket.
    """
    full_chain_ids = get_chain_ids(full)
    pocket_chain_ids = get_chain_ids(pocket)
    result: dict[str, list[int]] = {}

    for chain_id in full_chain_ids:
        if chain_id not in pocket_chain_ids:
            continue

        full_chain = full[full.chain_id == chain_id]
        pocket_chain = pocket[pocket.chain_id == chain_id]

        full_keys = get_chain_residue_keys(full_chain)
        pocket_key_set = set(get_chain_residue_keys(pocket_chain))

        indices = [i for i, k in enumerate(full_keys) if k in pocket_key_set]

        _, full_amino_acids = get_residues(full_chain)
        _, pocket_amino_acids = get_residues(pocket_chain)
        extracted_amino_acids = [full_amino_acids[i] for i in indices]
        assert extracted_amino_acids == list(pocket_amino_acids), (
            f"Chain {chain_id}: residues at mapped indices don't match pocket. "
            f"Expected {list(pocket_amino_acids)}, got {extracted_amino_acids}"
        )

        result[chain_id] = indices

    return result
