import binascii
import glob
import io
import os
import time
from pathlib import Path
import _pickle as pickle
import MDAnalysis as mda
from plyfile import PlyData
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge, Cartesian
from collections import defaultdict
from multiprocessing import Pool
import random
import copy
from joblib import Parallel, delayed
import numpy as np
import torch
from rdkit.Chem import MolToSmiles, MolFromSmiles, AddHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
from loguru import logger
from datasets.process_mols import read_molecule, get_rec_graph, generate_conformer, \
    get_lig_graph_with_matching, extract_receptor_structure, parse_receptor, parse_pdb_from_path
from utils.diffusion_utils import modify_conformer, set_time
from utils.utils import read_strings_from_txt
from utils import so3, torus
from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())   
class NoiseTransformBERT(BaseTransform):
    def __init__(self, t_to_sigma, no_torsion, all_atom):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom
    def forward(self, data):
        t = np.random.uniform()
        # t_rot = np.random.uniform()
        # t_tor = np.random.uniform()
        t_tr, t_rot, t_tor = t, t, t
        return self.apply_noise(data, t_tr, t_rot, t_tor)
    def apply_noise(self, data, t_tr, t_rot, t_tor, tr_update = None, rot_update=None, torsion_updates=None):
        # mdn mode make no update
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)
        set_time(data, t_tr, t_rot, t_tor, 1, self.all_atom, device=None)
        # set noise scale and time
        # in the first steps ,tor and rot set to zero ,because the ligand is not in the binding pocket,\
        # modify the ligand in those freendom degree is useless
        eps_tor_sigma = 0.0314
        eps_tr_sigma = 0.1
        eps_rot_sigma = 0.1
        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)
        # random like BERT style but probility is 0.8 and 0.2
        prob = random.random()
        # eps_sigma = 1e-10
        # 15% randomly change a freedom degree to noise and not noise any freedom degree
        if prob < 0.05:
            prob /= 0.05
            # 85% randomly change a freedom degree to noise
            if prob < 0.95:
                """
                0:tr
                1:rot
                2:tor
                """
                freedom_to_noise = random.choice([0,1,2])
                if freedom_to_noise == 0:
                    tr_sigma, rot_sigma, tor_sigma = tr_sigma,eps_rot_sigma,eps_tor_sigma
                elif freedom_to_noise == 1:
                    tr_sigma, rot_sigma, tor_sigma = eps_tr_sigma,rot_sigma,eps_tor_sigma
                else:
                    tr_sigma, rot_sigma, tor_sigma = eps_tr_sigma,eps_rot_sigma,tor_sigma
                    torsion_updates = np.random.normal(loc=0.0, scale=tor_sigma, size=data['ligand'].edge_mask.sum()) if torsion_updates is None else torsion_updates
                    # random selected opne edge to change
                    if data['ligand'].edge_mask.sum() >= 1:
                        tmp = np.zeros_like(torsion_updates)
                        # selected = np.random.randint(0, 10, 1)
                        selected = np.random.randint(0,data['ligand'].edge_mask.sum(),1)[0]
                        tmp[selected] = 1
                        torsion_updates = tmp*torsion_updates + eps_tor_sigma*torsion_updates
                    # torsion_updates = None if self.no_torsion else torsion_updates
            # 15% not noise any freedom degree
            else:
                # return data
                tr_sigma, rot_sigma, tor_sigma = eps_tr_sigma,eps_rot_sigma,eps_tor_sigma

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)) if tr_update is None else tr_update
        rot_update = so3.sample_vec(eps=rot_sigma) if rot_update is None else rot_update
        torsion_updates = np.random.normal(loc=0.0, scale=tor_sigma, size=data['ligand'].edge_mask.sum()) if torsion_updates is None else torsion_updates
        torsion_updates = None if self.no_torsion else torsion_updates

        modify_conformer(data, tr_update, torch.from_numpy(rot_update).float(), torsion_updates)
        data.tr_score = -tr_update / tr_sigma ** 2
        data.rot_score = torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = None if self.no_torsion else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        data.tor_sigma_edge = None if self.no_torsion else np.ones(data['ligand'].edge_mask.sum()) * tor_sigma
        return data
class NoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma, no_torsion, all_atom):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.all_atom = all_atom

    def forward(self, data):
        t = np.random.uniform()
        t_tr, t_rot, t_tor = t, t, t
        return self.apply_noise(data, t_tr, t_rot, t_tor)

    def apply_noise(self, data, t_tr, t_rot, t_tor, tr_update = None, rot_update=None, torsion_updates=None):
        # mdn mode make no update
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)
        set_time(data, t_tr, t_rot, t_tor, 1, self.all_atom, device=None)

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)) if tr_update is None else tr_update
        rot_update = so3.sample_vec(eps=rot_sigma) if rot_update is None else rot_update
        torsion_updates = np.random.normal(loc=0.0, scale=tor_sigma, size=data['ligand'].edge_mask.sum()) if torsion_updates is None else torsion_updates
        torsion_updates = None if self.no_torsion else torsion_updates
        modify_conformer(data, tr_update, torch.from_numpy(rot_update).float(), torsion_updates)

        data.tr_score = -tr_update / tr_sigma ** 2
        data.rot_score = torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = None if self.no_torsion else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        data.tor_sigma_edge = None if self.no_torsion else np.ones(data['ligand'].edge_mask.sum()) * tor_sigma
        return data

class PDBBind(Dataset):
    def __init__(self, root, transform=None, cache_path='data/cache', split_path='data/', limit_complexes=0,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching="original", keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, esm_model_name='esm2_3B', pocket_cutoff='8A', require_ligand=False,
                 ligands_list=None, protein_path_list=None, ligand_descriptions=None, keep_local_structures=False,surface_path = None):

        super(PDBBind, self).__init__(root, transform)
        self.surface_path = surface_path
        self.transform = transform
        self.pdbbind_dir = root
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.esm_model_name = esm_model_name
        self.pocket_cutoff = pocket_cutoff
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        cache_path += '_torsion'
        if all_atoms:
            cache_path += '_allatoms'
        self.full_cache_path = os.path.join(cache_path, f'limit{self.limit_complexes}'
                                                        f'_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}'
                                                        f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
                                            + ('' if not all_atoms else f'_atomRad{atom_radius}_atomMax{atom_max_neighbors}')
                                            + ('' if num_conformers == 1 else f'_confs{num_conformers}')
                                            + ('' if self.esm_embeddings_path is None else f'_{self.esm_model_name}')
                                            + f'_pocket{self.pocket_cutoff}'
                                            + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if protein_path_list is None or ligand_descriptions is None else str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode()))))
        
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        cache = Path(self.full_cache_path)
        data_bin = cache / "data.bin"
        index_pkl = cache / "index.pkl"
        if require_ligand:
            if not (cache / "heterographs_0.pkl").exists():
                os.makedirs(self.full_cache_path, exist_ok=True)
                self.inference_preprocessing()
            self._index = None
            self._data_handles: dict[int, io.BufferedReader] = {}
            self._num_complexes = len(glob.glob(os.path.join(self.full_cache_path, 'heterographs_*.pkl')))
            logger.info(f'Dataset size: {self._num_complexes}')
        else:
            if not data_bin.exists():
                os.makedirs(self.full_cache_path, exist_ok=True)
                self.preprocessing()
            with open(index_pkl, "rb") as f:
                self._index: list[tuple[int, int]] = pickle.load(f)
            self._data_handles: dict[int, io.BufferedReader] = {}
            logger.info(f'Dataset size: {len(self._index)}')

    def _get_data_file(self) -> io.BufferedReader:
        pid = os.getpid()
        fh = self._data_handles.get(pid)
        if fh is None:
            fh = open(Path(self.full_cache_path) / "data.bin", "rb")
            self._data_handles[pid] = fh
        return fh

    def _read_sample(self, idx: int) -> HeteroData:
        offset, length = self._index[idx]
        fh = self._get_data_file()
        fh.seek(offset)
        return pickle.loads(fh.read(length))

    def len(self):
        if self._index is None:
            return self._num_complexes
        return len(self._index)

    def get_complexs_list(self, num):
        graphs_list = []
        mols_list = []
        for idx in range(min(num, len(self._index))):
            complex_graph = self._read_sample(idx)
            complex_graph['ligand'].orig_pos -= complex_graph.original_center.numpy()
            complex_graph['receptor'].center_pos -= complex_graph.original_center.numpy()
            complex_graph['receptor'].atoms_pos -= complex_graph.original_center.numpy()
            mols_list.append(complex_graph.pop("mol", None))
            graphs_list.append(complex_graph)
        return graphs_list, mols_list

    def get(self, idx):
        if self.require_ligand:
            with open(os.path.join(self.full_cache_path, f'heterographs_{idx}.pkl'), 'rb') as f:
                complex_graph = pickle.load(f)
            with open(os.path.join(self.full_cache_path, f'rdkit_ligands_{idx}.pkl'), 'rb') as f:
                complex_graph.mol = pickle.load(f)
            complex_graph['ligand'].orig_pos -= complex_graph.original_center.numpy()
            complex_graph['receptor'].center_pos -= complex_graph.original_center.numpy()
            complex_graph['receptor'].atoms_pos -= complex_graph.original_center.numpy()
            return complex_graph
        else:
            complex_graph = self._read_sample(idx)
            complex_graph['ligand'].orig_pos -= complex_graph.original_center.numpy()
            complex_graph['receptor'].center_pos -= complex_graph.original_center.numpy()
            complex_graph['receptor'].atoms_pos -= complex_graph.original_center.numpy()
            if self.matching == "improved":
                complex_graph['ligand'].pos = complex_graph['ligand'].pos_improved
            elif self.matching == "none":
                complex_graph['ligand'].pos = torch.from_numpy(complex_graph['ligand'].orig_pos).float()
            if self.transform is None:
                if not torch.is_tensor(complex_graph['ligand'].pos):
                    complex_graph['ligand'].pos = torch.from_numpy(complex_graph['ligand'].pos).float()
            complex_graph.pop("mol", None)
            complex_graph.pop("pos_improved", None)
            return complex_graph
        
    def preprocessing(self):
        assert self.surface_path is not None, 'surface_path is None please set this param if you want to use surface feature'
        logger.info(f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')
        complex_names_all = read_strings_from_txt(self.split_path)
        logger.info('complex_names_all: ', len(complex_names_all))
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        logger.info(f'Loading {len(complex_names_all)} complexes.')
        if self.esm_embeddings_path is not None:
            lm_embeddings_chains_all = []
            complex_names_filtered = []
            for name in complex_names_all:
                emb_path = os.path.join(self.esm_embeddings_path, name, f'{name}_protein_processed_{self.pocket_cutoff}_{self.esm_model_name}.pt')
                if os.path.exists(emb_path):
                    lm_embeddings_chains_all.append(torch.load(emb_path))
                    complex_names_filtered.append(name)
            complex_names_all = complex_names_filtered
        else:
            lm_embeddings_chains_all = [None] * len(complex_names_all)

        cache = Path(self.full_cache_path)
        index: list[tuple[int, int]] = []
        with open(cache / "data.bin", "wb") as data_f:
            if self.num_workers > 1:
                for i in range(len(complex_names_all) // 1000 + 1):
                    complex_names = complex_names_all[1000 * i:1000 * (i + 1)]
                    lm_embeddings_chains = lm_embeddings_chains_all[1000 * i:1000 * (i + 1)]
                    t_list = Parallel(n_jobs=self.num_workers, backend="multiprocessing")(
                        delayed(self.get_complex)(x) for x in tqdm(
                            zip(complex_names, lm_embeddings_chains, [None] * len(complex_names), [None] * len(complex_names)),
                            total=len(complex_names),
                            desc=f'loading complexes {i}/{len(complex_names_all) // 1000 + 1}',
                        )
                    )
                    for t in t_list:
                        for graph, _lig in zip(t[0], t[1]):
                            blob = pickle.dumps(graph, protocol=-1)
                            offset = data_f.tell()
                            data_f.write(blob)
                            index.append((offset, len(blob)))
            else:
                with tqdm(total=len(complex_names_all), desc='loading complexes') as pbar:
                    for t in map(self.get_complex, zip(complex_names_all, lm_embeddings_chains_all, [None] * len(complex_names_all), [None] * len(complex_names_all))):
                        for graph, _lig in zip(t[0], t[1]):
                            blob = pickle.dumps(graph, protocol=-1)
                            offset = data_f.tell()
                            data_f.write(blob)
                            index.append((offset, len(blob)))
                        pbar.update()
        with open(cache / "index.pkl", "wb") as f:
            pickle.dump(index, f, protocol=-1)
    def inference_preprocessing(self):
        ligands_list = []
        logger.info('Reading molecules and generating local structures with RDKit (unless --keep_local_structures is turned on).')
        failed_ligand_indices = []
        for idx, ligand_description in tqdm(enumerate(self.ligand_descriptions)):
            try:
                mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path
                if mol is not None:
                    mol = AddHs(mol)
                    generate_conformer(mol)
                    ligands_list.append(mol)
                else:
                    mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                    if mol is None:
                        raise Exception('RDKit could not read the molecule ', ligand_description)
                    if not self.keep_local_structures:
                        mol.RemoveAllConformers()
                        mol = AddHs(mol)
                        generate_conformer(mol)
                    ligands_list.append(mol)
            except Exception as e:

                logger.info('Failed to read molecule ', ligand_description, ' We are skipping it. The reason is the exception: ', e)
                failed_ligand_indices.append(idx)
        for index in sorted(failed_ligand_indices, reverse=True):
            del self.protein_path_list[index]
            del self.ligand_descriptions[index]

        if self.esm_embeddings_path is not None:
            logger.info('Reading language model embeddings.')
            lm_embeddings_chains_all = []
            if not os.path.exists(self.esm_embeddings_path): raise Exception('ESM embeddings path does not exist: ',self.esm_embeddings_path)
            for protein_path in self.protein_path_list:
                embeddings_paths = sorted(glob.glob(os.path.join(self.esm_embeddings_path, os.path.basename(protein_path)) + '*'))
                lm_embeddings_chains = []
                for embeddings_path in embeddings_paths:
                    lm_embeddings_chains.append(torch.load(embeddings_path)['representations'][33])
                lm_embeddings_chains_all.append(lm_embeddings_chains)
        else:
            lm_embeddings_chains_all = [None] * len(self.protein_path_list)

        logger.info('Generating graphs for ligands and proteins')
        cache = Path(self.full_cache_path)
        if self.num_workers > 1:
            idx = 0
            for i in range(len(self.protein_path_list)//1000+1):
                if (cache / f"heterographs_{1000*i}.pkl").exists():
                    idx += len(self.protein_path_list[1000*i:1000*(i+1)])
                    continue
                protein_paths_chunk = self.protein_path_list[1000*i:1000*(i+1)]
                ligand_description_chunk = self.ligand_descriptions[1000*i:1000*(i+1)]
                ligands_chunk = ligands_list[1000 * i:1000 * (i + 1)]
                lm_embeddings_chains = lm_embeddings_chains_all[1000*i:1000*(i+1)]
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
                with tqdm(total=len(protein_paths_chunk), desc=f'loading complexes {i}/{len(self.protein_path_list)//1000+1}') as pbar:
                    for t in p.imap_unordered(self.get_complex, zip(protein_paths_chunk, lm_embeddings_chains, ligands_chunk, ligand_description_chunk)):
                        for graph, lig in zip(t[0], t[1]):
                            with open(cache / f"heterographs_{idx}.pkl", "wb") as f:
                                pickle.dump(graph, f, protocol=-1)
                            with open(cache / f"rdkit_ligands_{idx}.pkl", "wb") as f:
                                pickle.dump(lig, f, protocol=-1)
                            idx += 1
                        pbar.update()
                p.__exit__(None, None, None)
        else:
            idx = 0
            with tqdm(total=len(self.protein_path_list), desc='loading complexes') as pbar:
                for t in map(self.get_complex, zip(self.protein_path_list, lm_embeddings_chains_all, ligands_list, self.ligand_descriptions)):
                    for graph, lig in zip(t[0], t[1]):
                        with open(cache / f"heterographs_{idx}.pkl", "wb") as f:
                            pickle.dump(graph, f, protocol=-1)
                        with open(cache / f"rdkit_ligands_{idx}.pkl", "wb") as f:
                            pickle.dump(lig, f, protocol=-1)
                        idx += 1
                    pbar.update()
            if idx == 0: raise Exception('Preprocessing did not succeed for any complex')
    def get_complex(self, par):
        name, lm_embedding_chains, ligand, ligand_description = par
        t_total = time.perf_counter()
        if not os.path.exists(os.path.join(self.pdbbind_dir, name)) and ligand is None:
            logger.info(os.path.join(self.pdbbind_dir, name))
            logger.info("Folder not found", name)
            logger.info("Skipping", name)
            return [], []
        t0 = time.perf_counter()
        if ligand is not None:
            rec_model = parse_pdb_from_path(name)
            pure_pocket_path = os.path.join(os.path.splitext(name)[0],'_pure.pdb')
            # mda_rec_model = mda.Universe(name)
            name = f'{name}_{ligand_description}'
            ligs = [ligand]
        else:
            try:
                rec_path = os.path.join(self.surface_path, name, f'{name}_protein_processed_{self.pocket_cutoff}.pdb')
                rec_model = parse_pdb_from_path(rec_path)
                pure_pocket_path = rec_path.replace('.pdb', '_pure.pdb')
            except Exception as e:
                logger.info(f'Skipping {name} because of the error:')
                logger.info(e)
                return [], []
            ligs = [read_abs_file_mol(os.path.join(self.pdbbind_dir, name,f'{name}_ligand.sdf'), remove_hs=False, sanitize=True)]
            # ligs = read_mols(self.pdbbind_dir, name, remove_hs=False)
        t_parse_input = time.perf_counter() - t0
        complex_graphs = []
        failed_indices = []
        if len(ligs)==0:
            logger.info(f'No ligands found for {name}')
            return [],[]
        # assert len(ligs) > 0, f'No ligands found for {name}'
        for i, lig in enumerate(ligs):
            if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                logger.info(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
                continue
            complex_graph = HeteroData()
            complex_graph['name'] = name
            try:
                t0 = time.perf_counter()
                get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching != "none", self.keep_original,
                                            self.num_conformers, remove_hs=self.remove_hs)
                t_lig_graph = time.perf_counter() - t0

                t0 = time.perf_counter()
                rec, rec_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = extract_receptor_structure(copy.deepcopy(rec_model), lig, lm_embedding_chains=lm_embedding_chains)
                t_extract_rec = time.perf_counter() - t0
                if lm_embeddings is not None and c_alpha_coords is not None and len(c_alpha_coords) != len(lm_embeddings):
                    assert lm_embeddings is not None and c_alpha_coords is not None and len(c_alpha_coords) == len(lm_embeddings),'length error'
                    logger.info(f'LM embeddings for complex {name} did not have the right length for the protein. Skipping {name}.')
                    failed_indices.append(i)
                    continue
                t0 = time.perf_counter()
                mda_rec_model = mda.Universe(pure_pocket_path)
                # raise 'pure_pocket_path : {}'.format(pure_pocket_path)
                get_rec_graph(mda_rec_model, rec_coords, c_alpha_coords, n_coords, c_coords, complex_graph, rec_radius=self.receptor_radius,
                                c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
                                atom_radius=self.atom_radius, atom_max_neighbors=self.atom_max_neighbors, remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)
                t_rec_graph = time.perf_counter() - t0
            except Exception as e:
                import traceback
                logger.info(f'Skipping {name} because of the error:')
                logger.info(traceback.format_exc())
                failed_indices.append(i)
                continue
            protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
            complex_graph['receptor'].pos -= protein_center
            if self.all_atoms:
                complex_graph['atom'].pos -= protein_center

            if self.matching == "none" or self.num_conformers == 1:
                complex_graph['ligand'].pos -= protein_center
            else:
                for p in complex_graph['ligand'].pos:
                    p -= protein_center

            if hasattr(complex_graph['ligand'], 'pos_improved'):
                complex_graph['ligand'].pos_improved -= protein_center

            ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
            complex_graph.original_center = protein_center
            complex_graph.original_ligand_center = ligand_center + protein_center
            t_surface = 0.0
            if self.surface_path is not None:
                try:
                    ply_path = os.path.join(self.surface_path, name, f'{name}_protein_processed_{self.pocket_cutoff}.ply')
                    if not os.path.exists(ply_path):
                        logger.info('no surface file for ',name)
                        failed_indices.append(i)
                        continue
                    t0 = time.perf_counter()
                    with open(ply_path, 'rb') as f:
                        data = PlyData.read(f)
                    features = ([torch.tensor(data['vertex'][axis.name]) for axis in data['vertex'].properties if axis.name not in ['nx', 'ny', 'nz'] ])
                    pos = torch.stack(features[:3], dim=-1)
                    # pos 需要减去center_protein_pos
                    pos -= complex_graph.original_center
                    features = torch.stack(features[3:], dim=-1)
                    face = None
                    if 'face' in data:
                        faces = data['face']['vertex_indices']
                        faces = [torch.tensor(fa, dtype=torch.long) for fa in faces]
                        face = torch.stack(faces, dim=-1)
                    data = Data(x=features, pos=pos, face=face)
                    data = FaceToEdge()(data)
                    data = Cartesian(cat=False)(data)
                    t_surface = time.perf_counter() - t0
                    complex_graph['surface'].pos = data.pos
                    complex_graph['surface'].x = data.x
                    complex_graph['surface','surface_edge','surface'].edge_index = data.edge_index
                    complex_graph['surface','surface_edge','surface'].edge_attr = data.edge_attr
                except Exception as e:
                    import traceback
                    logger.info(f'Skipping {name} because of the surface error:')
                    logger.info(traceback.format_exc())
                    failed_indices.append(i)
                    continue
            # surface end
            complex_graphs.append(complex_graph)
            t_complex_total = time.perf_counter() - t_total
            logger.info(f'[TIMING] {name}: total={t_complex_total:.3f}s | parse_input={t_parse_input:.3f}s lig_graph={t_lig_graph:.3f}s extract_rec={t_extract_rec:.3f}s rec_graph={t_rec_graph:.3f}s surface={t_surface:.3f}s')
        for idx_to_delete in sorted(failed_indices, reverse=True):
            del ligs[idx_to_delete]

        return complex_graphs, ligs
def print_statistics(complex_graphs):
    statistics = ([], [], [], [])

    for complex_graph in complex_graphs:
        lig_pos = complex_graph['ligand'].pos if torch.is_tensor(complex_graph['ligand'].pos) else complex_graph['ligand'].pos[0]
        radius_protein = torch.max(torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1))
        molecule_center = torch.mean(lig_pos, dim=0)
        radius_molecule = torch.max(
            torch.linalg.vector_norm(lig_pos - molecule_center.unsqueeze(0), dim=1))
        distance_center = torch.linalg.vector_norm(molecule_center)
        statistics[0].append(radius_protein)
        statistics[1].append(radius_molecule)
        statistics[2].append(distance_center)
        if "rmsd_matching" in complex_graph:
            statistics[3].append(complex_graph.rmsd_matching)
        else:
            statistics[3].append(0)

    name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching']
    logger.info('Number of complexes: ', len(complex_graphs))
    for i in range(4):
        array = np.asarray(statistics[i])
        logger.info(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")

def construct_loader(args, t_to_sigma):
    if args.transformStyle=='BERT':
        transform = NoiseTransformBERT(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                               all_atom=args.all_atoms) if not args.model_type == 'mdn_model' else None
    if args.transformStyle=='diffdock':
        transform = NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                                all_atom=args.all_atoms) if not args.model_type == 'mdn_model' else None


    common_args = {'transform': transform, 'root': args.data_dir, 'limit_complexes': args.limit_complexes,
                   'receptor_radius': args.receptor_radius,
                   'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                   'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                   'matching': args.matching, 'popsize': 20, 'maxiter': 20,
                   'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                   'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                   'esm_embeddings_path': args.esm_embeddings_path, 'esm_model_name': args.esm_model_name,
                   'pocket_cutoff': args.pocket_cutoff, 'surface_path':args.surface_path}

    train_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_train, keep_original=True,
                            num_conformers=args.num_conformers, **common_args)
    val_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_val, keep_original=True, **common_args)

    loader_class = DataLoaderX
    prefetch = 2 if args.num_dataloader_workers > 0 else None
    persistent = args.num_dataloader_workers > 0
    train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory, prefetch_factor=prefetch, drop_last=True, persistent_workers=persistent)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory, prefetch_factor=prefetch, persistent_workers=persistent)

    return train_loader, val_loader

def read_mol(pdbbind_dir, name, remove_hs=False):
    lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.sdf'), remove_hs=remove_hs, sanitize=True)
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        logger.info('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
        lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.mol2'), remove_hs=remove_hs, sanitize=True)
    return lig
def read_abs_file_mol(file, remove_hs=False, sanitize=True):
    mol = read_molecule(file, remove_hs=remove_hs, sanitize=True)
    
    if file.endswith(".sdf") and mol is None:
        # mol = read_molecule(file, remove_hs=remove_hs, sanitize=True)
        if os.path.exists(file[:-4] + ".mol2"):
            logger.info('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
            mol = read_molecule(file[:-4] + ".mol2", remove_hs=remove_hs, sanitize=True)
    elif file.endswith(".mol2") and mol is None:
        if os.path.exists(file[:-4] + ".sdf"):
            logger.info('Using the .mol2 file failed. We found a .sdf file instead and are trying to use that.')
            mol = read_molecule(file[:-4] + ".sdf", remove_hs=remove_hs, sanitize=True)

    return mol
from rdkit.Chem import AllChem


def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if  'rdkit' not in file:
            if file.endswith(".sdf"):
                lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
                if lig is not None:
                    try:
                        mol_rdkit = copy.deepcopy(lig)
                        mol_rdkit.RemoveAllConformers()
                        mol_rdkit = AllChem.AddHs(mol_rdkit)
                        generate_conformer(mol_rdkit)
                        ligs.append(lig)
                        break
                    except:
                        continue
                else:
                    continue
            elif file.endswith(".mol2"):
                lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
                if lig is not None:
                    try:
                        mol_rdkit = copy.deepcopy(lig)
                        mol_rdkit.RemoveAllConformers()
                        mol_rdkit = AllChem.AddHs(mol_rdkit)
                        generate_conformer(mol_rdkit)
                        ligs.append(lig)
                        break
                    except:
                        continue
                else:
                    continue
    return ligs