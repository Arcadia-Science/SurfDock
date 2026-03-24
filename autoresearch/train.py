import copy
import enum
import math
import os
import datetime
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from e3nn import o3
from e3nn.nn import BatchNorm
from torch_cluster import radius, radius_graph
from torch_scatter import scatter
from loguru import logger

from prepare import (
    TR_WEIGHT,
    ROT_WEIGHT,
    TOR_WEIGHT,
    accelerator,
    construct_loader,
    device,
    ExponentialMovingAverage,
    get_timestep_embedding,
    lig_feature_dims,
    loss_function,
    make_args,
    rec_residue_feature_dims,
    save_yaml_file,
    t_to_sigma_compl,
    test_epoch,
    train_epoch,
)
from utils import so3, torus


# ─── Hyperparameters ─────────────────────────────────────────────────────────


class ESMModel(enum.Enum):
    ESM2_8M = "esm2_8M"
    ESM2_35M = "esm2_35M"
    ESM2_150M = "esm2_150M"
    ESM2_650M = "esm2_650M"
    ESM2_3B = "esm2_3B"

    def get_dim(self) -> int:
        return {
            "esm2_8M": 320,
            "esm2_35M": 480,
            "esm2_150M": 640,
            "esm2_650M": 1280,
            "esm2_3B": 2560,
        }[self.value]


ESM_MODEL = ESMModel.ESM2_35M


class PocketCutoff(enum.Enum):
    A8 = "8A"
    A10 = "10A"


POCKET_CUTOFF = PocketCutoff.A8

NS = 8
NV = 2
NUM_CONV_LAYERS = 2
DISTANCE_EMBED_DIM = 16
CROSS_DISTANCE_EMBED_DIM = 16
SIGMA_EMBED_DIM = 32
EMBEDDING_TYPE = "sinusoidal"  # "sinusoidal" or "fourier"
EMBEDDING_SCALE = 1000
BATCH_SIZE = 2
LR = 1e-3
WEIGHT_DECAY = 0.0
EMA_RATE = 0.999


# ─── Model Architecture ─────────────────────────────────────────────────────


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims, sigma_embed_dim, lm_embedding_type=None, esm_model: ESMModel = ESMModel.ESM2_650M):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        self.lm_embedding_type = lm_embedding_type
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        if self.lm_embedding_type is not None:
            self.lm_embedding_dim = esm_model.get_dim()
            self.lm_embedding_layer = torch.nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)

    def forward(self, x):
        x_embedding = 0
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
        else:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
        if self.lm_embedding_type is not None:
            x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, x[:, -self.lm_embedding_dim:]], axis=1))
        return x_embedding


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):
        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out


class TensorProductScoreModel(torch.nn.Module):
    def __init__(self, t_to_sigma, device, timestep_emb_func, in_lig_edge_features=10, in_rec_edge_features=5, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=None, esm_model: ESMModel = ESMModel.ESM2_650M):
        super(TensorProductScoreModel, self).__init__()
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.timestep_emb_func = timestep_emb_func
        self.num_conv_layers = num_conv_layers

        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=0, lm_embedding_type=lm_embedding_type, esm_model=esm_model)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(in_rec_edge_features + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.surface_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=[[], 4], sigma_embed_dim=sigma_embed_dim)
        self.surface_edge_embedding = nn.Sequential(nn.Linear(3 + sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.surface_rec_cross_edge_embedding = nn.Sequential(nn.Linear(cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.surface_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]
        lig_conv_layers = []
        surface_conv_layers, lig_to_surface_conv_layers, surface_to_lig_conv_layers = [], [], []
        residue_to_surface_conv_layers = []
        rec_conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            if i == 0:
                residue_to_surface_conv_layers.append(TensorProductConvLayer(**{
                    'in_irreps': f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o',
                    'sh_irreps': self.sh_irreps,
                    'out_irreps': in_irreps,
                    'n_edge_features': 3 * ns,
                    'hidden_features': 3 * ns,
                    'residual': False,
                    'batch_norm': batch_norm,
                    'dropout': dropout
                }))
                rec_conv_layers.append(TensorProductConvLayer(**{
                    'in_irreps': in_irreps,
                    'sh_irreps': self.sh_irreps,
                    'out_irreps': f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o',
                    'n_edge_features': 3 * ns,
                    'hidden_features': 3 * ns,
                    'residual': False,
                    'batch_norm': batch_norm,
                    'dropout': dropout
                }))

            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)

            if i != num_conv_layers - 1:
                surface_layer = TensorProductConvLayer(**parameters)
                surface_conv_layers.append(surface_layer)
                lig_to_surface_layer = TensorProductConvLayer(**parameters)
                lig_to_surface_conv_layers.append(lig_to_surface_layer)

            surface_to_lig_layer = TensorProductConvLayer(**parameters)
            surface_to_lig_conv_layers.append(surface_to_lig_layer)

        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.rec_conv_layers = nn.ModuleList(rec_conv_layers)
        self.residue_to_surface_conv_layers = nn.ModuleList(residue_to_surface_conv_layers)
        self.surface_conv_layers = nn.ModuleList(surface_conv_layers)
        self.lig_to_surface_conv_layers = nn.ModuleList(lig_to_surface_conv_layers)
        self.surface_to_lig_conv_layers = nn.ModuleList(surface_to_lig_conv_layers)

        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
        self.center_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )
        self.final_conv = TensorProductConvLayer(
            in_irreps=self.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=2 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )
        self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns), nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns), nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))

        if not no_torsion:
            self.final_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )
            self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.tor_bond_conv = TensorProductConvLayer(
                in_irreps=self.lig_conv_layers[-1].out_irreps,
                sh_irreps=self.final_tp_tor.irreps_out,
                out_irreps=f'{ns}x0o + {ns}x0e',
                n_edge_features=3 * ns,
                residual=False,
                dropout=dropout,
                batch_norm=batch_norm
            )
            self.tor_final_layer = nn.Sequential(
                nn.Linear(2 * ns, ns, bias=False),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(ns, 1, bias=False)
            )

    def forward(self, data):
        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])

        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        surface_node_attr, surface_edge_index, surface_edge_attr, surface_edge_sh = self.build_surface_conv_graph(data)
        surface_src, surface_dst = surface_edge_index
        surface_node_attr = self.surface_node_embedding(surface_node_attr)
        surface_edge_attr = self.surface_edge_embedding(surface_edge_attr)

        if self.dynamic_max_cross:
            cross_cutoff = (tr_sigma * 3 + 10).unsqueeze(1)
        else:
            cross_cutoff = self.cross_max_distance

        surface_cross_edge_index, surface_cross_edge_attr, surface_cross_edge_sh = self.build_surface_cross_conv_graph(data, cross_cutoff)
        surface_cross_lig, surface_cross_rec = surface_cross_edge_index
        surface_cross_edge_attr = self.cross_edge_embedding(surface_cross_edge_attr)

        surface_rec_cross_edge_index, surface_rec_cross_edge_attr, surface_rec_cross_edge_sh = self.build_surface_rec_cross_conv_graph(data)
        surface_rec_cross_rec, surface_rec_cross_surface = surface_rec_cross_edge_index
        surface_rec_cross_edge_attr = self.surface_rec_cross_edge_embedding(surface_rec_cross_edge_attr)

        residue_to_surface_edge_attr_ = torch.cat([surface_rec_cross_edge_attr, rec_node_attr[surface_rec_cross_rec, :self.ns], surface_node_attr[surface_rec_cross_surface, :self.ns]], -1)

        rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
        rec_intra_update = self.rec_conv_layers[0](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)
        rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
        rec_node_attr = rec_node_attr + rec_intra_update

        surface_inter_residue_update = self.residue_to_surface_conv_layers[0](rec_node_attr, torch.flip(surface_rec_cross_edge_index, dims=[0]), residue_to_surface_edge_attr_, surface_rec_cross_edge_sh,
                                                            out_nodes=surface_node_attr.shape[0])
        surface_node_attr = F.pad(surface_node_attr, (0, surface_inter_residue_update.shape[-1] - surface_node_attr.shape[-1]))
        surface_node_attr = surface_node_attr + surface_inter_residue_update

        for l in range(len(self.lig_conv_layers)):
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)

            surface_to_lig_edge_attr_ = torch.cat([surface_cross_edge_attr, lig_node_attr[surface_cross_lig, :self.ns], surface_node_attr[surface_cross_rec, :self.ns]], -1)
            surface_lig_inter_update = self.surface_to_lig_conv_layers[l](surface_node_attr, surface_cross_edge_index, surface_to_lig_edge_attr_, surface_cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0])

            if l != len(self.lig_conv_layers) - 1:
                surface_edge_attr_ = torch.cat([surface_edge_attr, surface_node_attr[surface_src, :self.ns], surface_node_attr[surface_dst, :self.ns]], -1)
                surface_intra_update = self.surface_conv_layers[l](surface_node_attr, surface_edge_index, surface_edge_attr_, surface_edge_sh)

                lig_to_surface_edge_attr_ = torch.cat([surface_cross_edge_attr, lig_node_attr[surface_cross_lig, :self.ns], surface_node_attr[surface_cross_rec, :self.ns]], -1)
                surface_inter_update = self.lig_to_surface_conv_layers[l](lig_node_attr, torch.flip(surface_cross_edge_index, dims=[0]), lig_to_surface_edge_attr_, surface_cross_edge_sh,
                                                              out_nodes=surface_node_attr.shape[0])

            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))
            lig_node_attr = lig_node_attr + lig_intra_update + surface_lig_inter_update
            if l != len(self.lig_conv_layers) - 1:
                surface_node_attr = F.pad(surface_node_attr, (0, surface_intra_update.shape[-1] - surface_node_attr.shape[-1]))
                surface_node_attr = surface_node_attr + surface_intra_update + surface_inter_update

        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['ligand'].x.device)

        if self.no_torsion or data['ligand'].edge_mask.sum() == 0:
            return tr_pred, rot_pred, torch.empty(0, device=self.device)

        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data)
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                  out_nodes=data['ligand'].edge_mask.sum(), reduce='mean')
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['ligand'].x.device))

        return tr_pred, rot_pred, tor_pred

    def build_lig_conv_graph(self, data):
        data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['tr'])
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_surface_conv_graph(self, data):
        tr = data['receptor'].node_t['tr'][0]
        data['surface'].node_sigma_emb = self.timestep_emb_func(tr * torch.ones(data['surface'].num_nodes).to(tr.device))
        node_attr = torch.cat([torch.nan_to_num(data['surface'].x), data['surface'].node_sigma_emb], 1)
        edge_index = data['surface', 'surface_edge', 'surface'].edge_index
        src, dst = edge_index
        edge_vec = data['surface'].pos[dst.long()] - data['surface'].pos[src.long()]
        edge_length_emb = self.surface_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['surface'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([data['surface', 'surface_edge', 'surface'].edge_attr, edge_sigma_emb, edge_length_emb], 1).float()
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, data):
        node_attr = data['receptor'].x
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = torch.cat([data['receptor', 'rec_contact', 'receptor'].edge_attr, edge_length_emb], 1).float()
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_surface_cross_conv_graph(self, data, cross_distance_cutoff):
        if torch.is_tensor(cross_distance_cutoff):
            edge_index = radius(data['surface'].pos / cross_distance_cutoff[data['surface'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['surface'].batch, data['ligand'].batch, max_num_neighbors=30)
        else:
            edge_index = radius(data['surface'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['surface'].batch, data['ligand'].batch, max_num_neighbors=30)
        src, dst = edge_index
        edge_vec = data['surface'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_surface_rec_cross_conv_graph(self, data, cross_distance_cutoff=15):
        edge_index = radius(data['surface'].pos, data['receptor'].pos, cross_distance_cutoff,
                        data['surface'].batch, data['receptor'].batch, max_num_neighbors=30)
        src, dst = edge_index
        edge_vec = data['surface'].pos[dst.long()] - data['receptor'].pos[src.long()]
        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = edge_length_emb
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_center_conv_graph(self, data):
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)
        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)
        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=data['ligand'].batch, batch_y=bond_batch)
        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return bonds, edge_index, edge_attr, edge_sh


# ─── Model Construction ─────────────────────────────────────────────────────


def build_timestep_embedding():
    return get_timestep_embedding(EMBEDDING_TYPE, SIGMA_EMBED_DIM, EMBEDDING_SCALE)


def build_model(args, t_to_sigma, device):
    timestep_emb_func = build_timestep_embedding()
    lm_embedding_type = None
    if args.esm_embeddings_path is not None:
        lm_embedding_type = 'esm'
    return TensorProductScoreModel(
        t_to_sigma=t_to_sigma,
        device=device,
        timestep_emb_func=timestep_emb_func,
        num_conv_layers=args.num_conv_layers,
        lig_max_radius=args.max_radius,
        scale_by_sigma=args.scale_by_sigma,
        sigma_embed_dim=args.sigma_embed_dim,
        ns=args.ns, nv=args.nv,
        no_torsion=args.no_torsion,
        distance_embed_dim=args.distance_embed_dim,
        cross_distance_embed_dim=args.cross_distance_embed_dim,
        batch_norm=not args.no_batch_norm,
        dropout=args.dropout,
        use_second_order_repr=args.use_second_order_repr,
        cross_max_distance=args.cross_max_distance,
        dynamic_max_cross=args.dynamic_max_cross,
        lm_embedding_type=lm_embedding_type,
        esm_model=ESM_MODEL,
    )


# ─── Optimizer & Scheduler ───────────────────────────────────────────────────


def build_optimizer_and_scheduler(model, args):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = None
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7,
            patience=args.scheduler_patience, min_lr=LR / 100)
    return optimizer, scheduler


# ─── Training Loop ───────────────────────────────────────────────────────────


def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, tb_writer=None):
    best_val_loss = math.inf
    best_epoch = 0
    loss_fn = partial(loss_function, tr_weight=TR_WEIGHT, rot_weight=ROT_WEIGHT,
                      tor_weight=TOR_WEIGHT, no_torsion=args.no_torsion)

    if accelerator.is_local_main_process:
        logger.info("Starting training...")

    for epoch in range(args.n_epochs):
        logs = {}

        train_losses = train_epoch(model, train_loader, optimizer, device, t_to_sigma, loss_fn, accelerator, ema_weights)
        if accelerator.is_local_main_process:
            logger.info("Epoch {}: Training loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}"
                .format(epoch, train_losses['loss'], train_losses['tr_loss'], train_losses['rot_loss'],
                        train_losses['tor_loss']))

        ema_weights.store(model.parameters())
        if args.use_ema:
            ema_weights.copy_to(model.parameters())

        val_losses = test_epoch(model, val_loader, device, t_to_sigma, loss_fn, accelerator, args.test_sigma_intervals, model_type=args.model_type)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            logger.info("Epoch {}: Validation loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}"
                .format(epoch, val_losses['loss'], val_losses['tr_loss'], val_losses['rot_loss'], val_losses['tor_loss']))

        if not args.use_ema:
            ema_weights.copy_to(model.parameters())
        accelerator.wait_for_everyone()

        unwrapped_model = accelerator.unwrap_model(model)
        ema_state_dict = copy.deepcopy(unwrapped_model.state_dict())
        ema_weights.restore(model.parameters())
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = unwrapped_model.state_dict()

        logs.update({'train_' + k: v for k, v in train_losses.items()})
        logs.update({'val_' + k: v for k, v in val_losses.items()})
        logs['current_lr'] = optimizer.param_groups[0]['lr']

        if tb_writer is not None:
            for k, v in logs.items():
                tb_writer.add_scalar(k, v, epoch + 1)

        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch
            if accelerator.is_local_main_process:
                torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))

        if scheduler is not None:
            scheduler.step(val_losses['loss'])

        if accelerator.is_local_main_process:
            torch.save({
                'epoch': epoch,
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'ema_weights': ema_weights.state_dict(),
            }, os.path.join(run_dir, 'last_model.pt'))

    if accelerator.is_local_main_process:
        logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    if tb_writer is not None:
        tb_writer.close()

    print("---")
    print(f"val_loss:           {best_val_loss:.6f}")
    print(f"best_epoch:         {best_epoch}")
    print(f"num_params:         {sum(p.numel() for p in model.parameters())}")


# ─── Main ────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    args = make_args(
        ns=NS,
        nv=NV,
        num_conv_layers=NUM_CONV_LAYERS,
        distance_embed_dim=DISTANCE_EMBED_DIM,
        cross_distance_embed_dim=CROSS_DISTANCE_EMBED_DIM,
        sigma_embed_dim=SIGMA_EMBED_DIM,
        batch_size=BATCH_SIZE,
        lr=LR,
        w_decay=WEIGHT_DECAY,
        ema_rate=EMA_RATE,
        esm_model_name=ESM_MODEL.value,
        pocket_cutoff=POCKET_CUTOFF.value,
    )

    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.add(os.path.join(run_dir, 'LogFile.log'), rotation='100 MB')
    logger.info(f'device {device} | Args: {args}')

    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader = construct_loader(args, t_to_sigma)

    model = build_model(args, t_to_sigma, device)
    model.to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)
    ema_weights = ExponentialMovingAverage(model.parameters(), decay=EMA_RATE)

    model = accelerator.prepare(model)
    optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        optimizer, train_loader, val_loader, scheduler)

    numel = sum(p.numel() for p in model.parameters())
    logger.info(f'Model with {numel} parameters')

    save_yaml_file(os.path.join(run_dir, 'model_parameters.yml'), args.__dict__)
    args.device = device

    tb_writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb")) if accelerator.is_local_main_process else None
    train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir, tb_writer=tb_writer)
