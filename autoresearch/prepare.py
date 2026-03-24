import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit._check")

import os
import resource
from argparse import Namespace

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from datasets.pdbbind import construct_loader
from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_timestep_embedding
from utils.training import train_epoch, test_epoch, loss_function

TR_WEIGHT = 0.33
ROT_WEIGHT = 0.33
TOR_WEIGHT = 0.33

TIME_BUDGET = 300  # seconds — fixed across all experiments
LIMIT_COMPLEXES = 50

DATA_DIR = "../data/processed"
SURFACE_PATH = "../data/processed"
ESM_EMBEDDINGS_PATH = "../data/processed"
SPLIT_TRAIN = "../data/splits/timesplit_no_lig_overlap_train_surface_complete"
SPLIT_VAL = "../data/splits/timesplit_no_lig_overlap_val_surface_complete"

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
device = accelerator.device
set_seed(42)


def make_args(
    ns: int,
    nv: int,
    num_conv_layers: int,
    distance_embed_dim: int,
    cross_distance_embed_dim: int,
    sigma_embed_dim: int,
    batch_size: int,
    lr: float,
    w_decay: float,
    esm_model_name: str,
    pocket_cutoff: str,
    scheduler: str = None,
    scheduler_patience: int = 20,
    dropout: float = 0.0,
    max_radius: float = 5.0,
    cross_max_distance: float = 80,
    use_second_order_repr: bool = False,
    no_batch_norm: bool = False,
    dynamic_max_cross: bool = False,
    no_torsion: bool = False,
    scale_by_sigma: bool = True,
) -> Namespace:
    return Namespace(
        config=None,
        log_dir="workdir",
        run_name="",
        cache_path="data/cache",
        data_dir=DATA_DIR,
        split_train=SPLIT_TRAIN,
        split_val=SPLIT_VAL,
        split_test="data/splits/timesplit_test",
        surface_path=SURFACE_PATH,
        esm_embeddings_path=ESM_EMBEDDINGS_PATH,
        esm_model_name=esm_model_name,
        pocket_cutoff=pocket_cutoff,
        n_epochs=999999,
        limit_complexes=LIMIT_COMPLEXES,
        batch_size=batch_size,
        lr=lr,
        w_decay=w_decay,
        ema_rate=0.999,
        use_ema=False,
        scheduler=scheduler,
        scheduler_patience=scheduler_patience,
        ns=ns,
        nv=nv,
        num_conv_layers=num_conv_layers,
        distance_embed_dim=distance_embed_dim,
        cross_distance_embed_dim=cross_distance_embed_dim,
        sigma_embed_dim=sigma_embed_dim,
        max_radius=max_radius,
        cross_max_distance=cross_max_distance,
        dynamic_max_cross=dynamic_max_cross,
        scale_by_sigma=scale_by_sigma,
        no_batch_norm=no_batch_norm,
        use_second_order_repr=use_second_order_repr,
        dropout=dropout,
        no_torsion=no_torsion,
        tr_weight=TR_WEIGHT,
        rot_weight=ROT_WEIGHT,
        tor_weight=TOR_WEIGHT,
        tr_sigma_min=0.1,
        tr_sigma_max=30,
        rot_sigma_min=0.1,
        rot_sigma_max=1.65,
        tor_sigma_min=0.0314,
        tor_sigma_max=3.14,
        receptor_radius=30,
        c_alpha_max_neighbors=10,
        atom_radius=5,
        atom_max_neighbors=8,
        matching=True,
        matching_popsize=20,
        matching_maxiter=20,
        max_lig_size=None,
        remove_hs=False,
        num_conformers=1,
        all_atoms=False,
        num_workers=1,
        num_dataloader_workers=0,
        pin_memory=False,
        cudnn_benchmark=False,
        test_sigma_intervals=False,
        model_type="surface_score_model",
        transformStyle="diffdock",
        wandb=False,
    )
