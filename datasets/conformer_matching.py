import copy, time
import numpy as np
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit import Geometry
import networkx as nx
from scipy.optimize import differential_evolution, minimize

RDLogger.DisableLog('rdApp.*')

"""
    Conformer matching routines from Torsional Diffusion
"""

def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def apply_changes(mol, values, rotable_bonds, conf_id):
    opt_mol = copy.copy(mol)
    [SetDihedral(opt_mol.GetConformer(conf_id), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]
    return opt_mol


def optimize_rotatable_bonds(mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=0, popsize=15, maxiter=500,
                             mutation=(0.5, 1), recombination=0.8):
    opt = OptimizeConformer(mol, true_mol, rotable_bonds, seed=seed, probe_id=probe_id, ref_id=ref_id)
    max_bound = [np.pi] * len(opt.rotable_bonds)
    min_bound = [-np.pi] * len(opt.rotable_bonds)
    bounds = (min_bound, max_bound)
    bounds = list(zip(bounds[0], bounds[1]))

    # Optimize conformations
    result = differential_evolution(opt.score_conformation, bounds,
                                    maxiter=maxiter, popsize=popsize,
                                    mutation=mutation, recombination=recombination, disp=False, seed=seed)
    opt_mol = apply_changes(opt.mol, result['x'], opt.rotable_bonds, conf_id=probe_id)

    return opt_mol


def _remove_all_hs(mol):
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return Chem.RemoveHs(mol, params)


def _quick_screen(mol, crystal_mol, rotable_bonds, crystal_torsions):
    work = copy.deepcopy(mol)
    conf = work.GetConformer()
    for i, r in enumerate(rotable_bonds):
        SetDihedral(conf, r, crystal_torsions[i])
    return RMSD(work, crystal_mol)


def _optimize_from_mol(mol, crystal_mol, rotable_bonds, crystal_torsions):
    work = copy.deepcopy(mol)
    conf = work.GetConformer()

    def score(values):
        for i, r in enumerate(rotable_bonds):
            SetDihedral(conf, r, values[i])
        return RMSD(work, crystal_mol)

    result = minimize(score, crystal_torsions, method="L-BFGS-B", options={"eps": 1e-4, "ftol": 1e-5, "gtol": 1e-4})
    return result.x, result.fun


def optimize_rotatable_bonds_improved(crystal_mol, rdkit_mol, rotable_bonds):
    crystal_conf = crystal_mol.GetConformer()
    crystal_torsions = np.array([GetDihedral(crystal_conf, rb) for rb in rotable_bonds])

    best_x, best_score = _optimize_from_mol(rdkit_mol, crystal_mol, rotable_bonds, crystal_torsions)
    best_base = rdkit_mol

    n_bonds = len(rotable_bonds)
    template = copy.deepcopy(rdkit_mol)
    template.RemoveAllConformers()

    if n_bonds > 20:
        n_gen = 2
        n_optimize = 2
    elif n_bonds <= 10:
        n_gen = 50
        n_optimize = 12
    else:
        n_gen = 25
        n_optimize = 6

    multi = copy.deepcopy(template)
    ps = AllChem.ETKDGv2()
    ps.randomSeed = 42
    ps.pruneRmsThresh = -1.0
    cids = AllChem.EmbedMultipleConfs(multi, numConfs=n_gen, params=ps)

    if n_bonds > 20:
        for cid in cids:
            candidate = copy.deepcopy(template)
            candidate.AddConformer(multi.GetConformer(cid), assignId=True)
            x, s = _optimize_from_mol(candidate, crystal_mol, rotable_bonds, crystal_torsions)
            if s < best_score:
                best_score = s
                best_x = x
                best_base = candidate
    else:
        candidates = []
        for cid in cids:
            candidate = copy.deepcopy(template)
            candidate.AddConformer(multi.GetConformer(cid), assignId=True)
            screen_rmsd = _quick_screen(candidate, crystal_mol, rotable_bonds, crystal_torsions)
            candidates.append((screen_rmsd, candidate))

        candidates.sort(key=lambda x: x[0])
        for _, candidate in candidates[:n_optimize]:
            x, s = _optimize_from_mol(candidate, crystal_mol, rotable_bonds, crystal_torsions)
            if s < best_score:
                best_score = s
                best_x = x
                best_base = candidate

    n_atoms = crystal_mol.GetNumAtoms()
    n_diverse = 20 if n_atoms <= 36 else 10

    for use_random in [False, True]:
        if not (0.5 < best_score <= 1.0 and n_bonds <= 12 and n_atoms <= 45):
            break
        template_h = AllChem.AddHs(copy.deepcopy(template))
        ps2 = AllChem.ETKDGv2()
        ps2.randomSeed = 999
        ps2.pruneRmsThresh = -1.0
        ps2.useExpTorsionAnglePrefs = False
        ps2.useBasicKnowledge = False
        if use_random:
            ps2.useRandomCoords = True
        multi2_h = copy.deepcopy(template_h)
        cids2 = AllChem.EmbedMultipleConfs(multi2_h, numConfs=n_diverse, params=ps2)
        multi2 = _remove_all_hs(multi2_h)

        extra = []
        for cid in cids2:
            candidate = copy.deepcopy(template)
            candidate.AddConformer(multi2.GetConformer(cid), assignId=True)
            AllChem.MMFFOptimizeMolecule(candidate, maxIters=50)
            sr = _quick_screen(candidate, crystal_mol, rotable_bonds, crystal_torsions)
            extra.append((sr, candidate))

        extra.sort(key=lambda x: x[0])
        for _, candidate in extra[:3]:
            x, s = _optimize_from_mol(candidate, crystal_mol, rotable_bonds, crystal_torsions)
            if s < best_score:
                best_score = s
                best_x = x
                best_base = candidate

    return apply_changes(best_base, best_x, rotable_bonds, conf_id=-1)


class OptimizeConformer:
    def __init__(self, mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=None):
        super(OptimizeConformer, self).__init__()
        if seed:
            np.random.seed(seed)
        self.rotable_bonds = rotable_bonds
        self.mol = mol
        self.true_mol = true_mol
        self.probe_id = probe_id
        self.ref_id = ref_id

    def score_conformation(self, values):
        for i, r in enumerate(self.rotable_bonds):
            SetDihedral(self.mol.GetConformer(self.probe_id), r, values[i])
        return RMSD(self.mol, self.true_mol, self.probe_id, self.ref_id)


def get_torsion_angles(mol):
    torsions_list = []
    G = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i)
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    for u, v in nx.bridges(G):
        if G.degree(u) < 2 or G.degree(v) < 2:
            continue
        n0 = next(n for n in G.neighbors(u) if n != v)
        n1 = next(n for n in G.neighbors(v) if n != u)
        torsions_list.append((n0, u, v, n1))
    return torsions_list


# GeoMol
def get_torsions(mol_list):
    print('USING GEOMOL GET TORSIONS FUNCTION')
    atom_counter = 0
    torsionList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList


def A_transpose_matrix(alpha):
    return np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]], dtype=np.double)


def S_vec(alpha):
    return np.array([[np.cos(alpha)], [np.sin(alpha)]], dtype=np.double)


def GetDihedralFromPointCloud(Z, atom_idx):
    p = Z[list(atom_idx)]
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array([v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.arctan2(y, x)


def get_dihedral_vonMises(mol, conf, atom_idx, Z):
    Z = np.array(Z)
    v = np.zeros((2, 1))
    iAtom = mol.GetAtomWithIdx(atom_idx[1])
    jAtom = mol.GetAtomWithIdx(atom_idx[2])
    k_0 = atom_idx[0]
    i = atom_idx[1]
    j = atom_idx[2]
    l_0 = atom_idx[3]
    for b1 in iAtom.GetBonds():
        k = b1.GetOtherAtomIdx(i)
        if k == j:
            continue
        for b2 in jAtom.GetBonds():
            l = b2.GetOtherAtomIdx(j)
            if l == i:
                continue
            assert k != l
            s_star = S_vec(GetDihedralFromPointCloud(Z, (k, i, j, l)))
            a_mat = A_transpose_matrix(GetDihedral(conf, (k, i, j, k_0)) + GetDihedral(conf, (l_0, i, j, l)))
            v = v + np.matmul(a_mat, s_star)
    v = v / np.linalg.norm(v)
    v = v.reshape(-1)
    return np.arctan2(v[1], v[0])


def get_von_mises_rms(mol, mol_rdkit, rotable_bonds, conf_id):
    new_dihedrals = np.zeros(len(rotable_bonds))
    for idx, r in enumerate(rotable_bonds):
        new_dihedrals[idx] = get_dihedral_vonMises(mol_rdkit,
                                                   mol_rdkit.GetConformer(conf_id), r,
                                                   mol.GetConformer().GetPositions())
    mol_rdkit = apply_changes(mol_rdkit, new_dihedrals, rotable_bonds, conf_id)
    return RMSD(mol_rdkit, mol, conf_id)


def mmff_func(mol):
    mol_mmff = copy.deepcopy(mol)
    AllChem.MMFFOptimizeMoleculeConfs(mol_mmff, mmffVariant='MMFF94s')
    for i in range(mol.GetNumConformers()):
        coords = mol_mmff.GetConformers()[i].GetPositions()
        for j in range(coords.shape[0]):
            mol.GetConformer(i).SetAtomPosition(j,
                                                Geometry.Point3D(*coords[j]))


RMSD = AllChem.AlignMol
