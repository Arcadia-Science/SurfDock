import copy
import math
import os
import resource
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import _pickle as pickle
import modal
import torch

HOURS = 60 * 60
DATA_ROOT = Path("/runs")
ASSETS_ROOT = Path("/data")
PYG_CUDA_INDEX = "https://data.pyg.org/whl/torch-2.9.0+cu126.html"

surfdock_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libxrender1",
        "libxext6",
        "libsm6",
        "libglib2.0-0",
        "libgl1-mesa-glx",
        "libfontconfig1",
    )
    .pip_install(
        "torch==2.9.0",
        "setuptools<82",
        "numpy>=2.4.3",
    )
    .pip_install(
        "torch-scatter>=2.1.2",
        "torch-sparse>=0.6.18",
        "torch-cluster>=1.6.3",
        "torch-spline-conv>=1.2.2",
        find_links=PYG_CUDA_INDEX,
    )
    .pip_install(
        "e3nn>=0.6.0",
        "cuequivariance",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu12",
        "torch-geometric>=2.7.0",
        "rdkit>=2025.9.6",
        "biopython>=1.86",
        "mdanalysis>=2.10.0",
        "plyfile>=1.1.3",
        "scipy>=1.17.1",
        "pandas>=3.0.1",
        "loguru>=0.7.3",
        "spyrmsd>=0.9.0",
        "prefetch-generator>=1.0.3",
        "pyvista>=0.47.1",
        "tensorboard>=2.20.0",
        "joblib",
        "pyyaml",
    )
    .env({"precomputed_arrays": str(DATA_ROOT / "precomputed_arrays")})
    .add_local_python_source("datasets", "models", "utils")
)

runs_volume = modal.Volume.from_name("surfdock-runs", create_if_missing=True)
data_volume = modal.Volume.from_name("surfdock-data", create_if_missing=True)
app = modal.App("surfdock-training", image=surfdock_image)


@dataclass
class _AcceleratorState:
    deepspeed_plugin: None = None


class _DeviceMovingLoader:
    def __init__(self, loader, device: torch.device):
        self._loader = loader
        self._device = device

    def __iter__(self):
        for batch in self._loader:
            yield batch.to(self._device)

    def __len__(self):
        return len(self._loader)

    @property
    def dataset(self):
        return self._loader.dataset


class SingleGPUAccelerator:
    is_local_main_process = True
    num_processes = 1
    state = _AcceleratorState()

    def __init__(self, device: torch.device):
        self.device = device

    def backward(self, loss: torch.Tensor):
        loss.backward()

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def wait_for_everyone(self):
        pass

    def prepare(self, *args):
        from torch.utils.data import DataLoader as TorchDataLoader

        out = []
        for a in args:
            if isinstance(a, TorchDataLoader):
                out.append(_DeviceMovingLoader(a, self.device))
            else:
                out.append(a)
        return tuple(out) if len(out) > 1 else out[0]

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    @contextmanager
    def autocast(self):
        with torch.amp.autocast("cuda"):
            yield


@dataclass
class TrainConfig:
    ns: int = 16
    nv: int = 4
    num_conv_layers: int = 2
    distance_embed_dim: int = 32
    cross_distance_embed_dim: int = 32
    sigma_embed_dim: int = 32
    batch_size: int = 32
    n_epochs: int = 1
    lr: float = 1e-3
    w_decay: float = 0.0
    limit_complexes: int = 0
    esm_model_name: str = "esm2_3B"
    pocket_cutoff: str = "8A"
    model_type: str = "surface_score_model"
    model_version: str = "version3"
    run_name: str = ""
    restart_dir: str = ""
    restart_lr: float | None = None
    use_ema: bool = False
    ema_rate: float = 0.999
    no_torsion: bool = False
    all_atoms: bool = False
    remove_hs: bool = False
    max_lig_size: int | None = None
    receptor_radius: float = 30.0
    c_alpha_max_neighbors: int = 10
    atom_radius: float = 5.0
    atom_max_neighbors: int = 8
    matching: bool = True
    matching_popsize: int = 20
    matching_maxiter: int = 20
    num_conformers: int = 1
    max_radius: float = 5.0
    cross_max_distance: float = 80.0
    dynamic_max_cross: bool = False
    use_second_order_repr: bool = False
    no_batch_norm: bool = False
    dropout: float = 0.0
    embedding_type: str = "sinusoidal"
    embedding_scale: int = 1000
    tr_sigma_min: float = 0.1
    tr_sigma_max: float = 30.0
    rot_sigma_min: float = 0.1
    rot_sigma_max: float = 1.65
    tor_sigma_min: float = 0.0314
    tor_sigma_max: float = 3.14
    tr_weight: float = 0.33
    rot_weight: float = 0.33
    tor_weight: float = 0.33
    val_inference_freq: int = 5
    skip_inference_freq: int = 0
    num_inference_complexes: int = 100
    inference_steps: int = 20
    inference_earlystop_metric: str = "valinf_rmsds_lt2"
    inference_earlystop_goal: str = "max"
    scheduler: str | None = None
    scheduler_patience: int = 20
    num_dataloader_workers: int = 0
    pin_memory: bool = False
    num_workers: int = 1
    scale_by_sigma: bool = True
    cudnn_benchmark: bool = False
    project: str = "SurfDock_train"
    test_sigma_intervals: bool = False
    transformStyle: str = "diffdock"
    topN: int = 1
    mdn_early_stop_patience: int = 30
    mdn_dropout: float = 0.1
    n_gaussians: int = 20
    ligand_distance_prediction: bool = False
    atom_type_prediction: bool = False
    bond_type_prediction: bool = False
    residue_type_prediction: bool = False
    mdn_dist_threshold_train: float = 7.0
    mdn_dist_threshold_test: float = 5.0
    config: None = None
    log_dir: str = str(DATA_ROOT / "workdir")
    cache_path: str = str(DATA_ROOT / "cache")
    data_dir: str = str(DATA_ROOT / "processed")
    split_train: str = str(DATA_ROOT / "splits" / "timesplit_no_lig_overlap_train_surface_complete")
    split_val: str = str(DATA_ROOT / "splits" / "timesplit_no_lig_overlap_val_surface_complete")
    split_test: str = ""
    surface_path: str = str(DATA_ROOT / "processed")
    esm_embeddings_path: str = str(DATA_ROOT / "processed")
    device: torch.device | None = None

    def __post_init__(self):
        if not self.run_name:
            self.run_name = f"modal_{self.n_epochs}ep"


@dataclass
class CacheConfig:
    surface_path: str = ""
    data_dir: str = ""
    esm_model_name: str = "esm2_35M"
    esm_embeddings_path: str | None = None
    pocket_cutoff: str = "8A"
    receptor_radius: float = 30.0
    c_alpha_max_neighbors: int = 10
    all_atoms: bool = False
    atom_radius: float = 5.0
    atom_max_neighbors: int = 8
    remove_hs: bool = False
    matching: bool = True
    popsize: int = 20
    maxiter: int = 20
    keep_original: bool = True
    num_conformers: int = 1
    max_lig_size: int | None = None


def _build_cache_path(
    cache_path: str,
    split_path: str,
    limit_complexes: int,
    cfg: CacheConfig,
) -> Path:
    if cfg.matching:
        cache_path += "_torsion"
    if cfg.all_atoms:
        cache_path += "_allatoms"
    split_basename = os.path.splitext(os.path.basename(split_path))[0]
    full = (
        f"limit{limit_complexes}"
        f"_INDEX{split_basename}"
        f"_maxLigSize{cfg.max_lig_size}_H{int(not cfg.remove_hs)}"
        f"_recRad{cfg.receptor_radius}_recMax{cfg.c_alpha_max_neighbors}"
    )
    if cfg.all_atoms:
        full += f"_atomRad{cfg.atom_radius}_atomMax{cfg.atom_max_neighbors}"
    if cfg.matching and cfg.num_conformers != 1:
        full += f"_confs{cfg.num_conformers}"
    if cfg.esm_embeddings_path is not None:
        full += f"_{cfg.esm_model_name}"
    full += f"_pocket{cfg.pocket_cutoff}"
    return Path(cache_path) / full


def _process_one_complex(
    name: str,
    cfg: CacheConfig,
    lm_embedding_chains: torch.Tensor | None,
) -> tuple[list, list]:
    import MDAnalysis as mda
    from plyfile import PlyData
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.transforms import Cartesian, FaceToEdge

    from datasets.pdbbind import read_abs_file_mol
    from datasets.process_mols import (
        extract_receptor_structure,
        get_lig_graph_with_matching,
        get_rec_graph,
        parse_pdb_from_path,
    )

    rec_path = os.path.join(cfg.surface_path, name, f"{name}_protein_processed_{cfg.pocket_cutoff}.pdb")
    if not os.path.exists(rec_path):
        return [], []

    try:
        rec_model = parse_pdb_from_path(rec_path)
    except Exception:
        return [], []

    pure_pocket_path = rec_path.replace(".pdb", "_pure.pdb")
    lig_path = os.path.join(cfg.data_dir, name, f"{name}_ligand.sdf")
    lig = read_abs_file_mol(lig_path, remove_hs=False, sanitize=True)
    if lig is None:
        return [], []

    if cfg.max_lig_size is not None and lig.GetNumHeavyAtoms() > cfg.max_lig_size:
        return [], []

    from rdkit.Chem import GetMolFrags
    if len(GetMolFrags(lig)) > 1:
        return [], []

    complex_graph = HeteroData()
    complex_graph["name"] = name

    try:
        get_lig_graph_with_matching(
            lig,
            complex_graph,
            cfg.popsize,
            cfg.maxiter,
            cfg.matching,
            cfg.keep_original,
            cfg.num_conformers,
            remove_hs=cfg.remove_hs,
        )

        rec, rec_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = (
            extract_receptor_structure(
                copy.deepcopy(rec_model),
                lig,
                lm_embedding_chains=lm_embedding_chains,
            )
        )

        if lm_embeddings is not None and c_alpha_coords is not None and len(c_alpha_coords) != len(lm_embeddings):
            return [], []

        mda_rec_model = mda.Universe(pure_pocket_path)
        get_rec_graph(
            mda_rec_model,
            rec_coords,
            c_alpha_coords,
            n_coords,
            c_coords,
            complex_graph,
            rec_radius=cfg.receptor_radius,
            c_alpha_max_neighbors=cfg.c_alpha_max_neighbors,
            all_atoms=cfg.all_atoms,
            atom_radius=cfg.atom_radius,
            atom_max_neighbors=cfg.atom_max_neighbors,
            remove_hs=cfg.remove_hs,
            lm_embeddings=lm_embeddings,
        )
    except Exception:
        return [], []

    protein_center = torch.mean(complex_graph["receptor"].pos, dim=0, keepdim=True)
    complex_graph["receptor"].pos -= protein_center
    if cfg.all_atoms:
        complex_graph["atom"].pos -= protein_center

    if (not cfg.matching) or cfg.num_conformers == 1:
        complex_graph["ligand"].pos -= protein_center
    else:
        for p in complex_graph["ligand"].pos:
            p -= protein_center

    ligand_center = torch.mean(complex_graph["ligand"].pos, dim=0, keepdim=True)
    complex_graph.original_center = protein_center
    complex_graph.original_ligand_center = ligand_center + protein_center

    ply_path = os.path.join(cfg.surface_path, name, f"{name}_protein_processed_{cfg.pocket_cutoff}.ply")
    if not os.path.exists(ply_path):
        return [], []

    try:
        with open(ply_path, "rb") as f:
            ply_data = PlyData.read(f)
        features = [
            torch.tensor(ply_data["vertex"][axis.name])
            for axis in ply_data["vertex"].properties
            if axis.name not in ["nx", "ny", "nz"]
        ]
        pos = torch.stack(features[:3], dim=-1)
        pos -= complex_graph.original_center
        feat_tensor = torch.stack(features[3:], dim=-1)
        face = None
        if "face" in ply_data:
            faces = ply_data["face"]["vertex_indices"]
            faces = [torch.tensor(fa, dtype=torch.long) for fa in faces]
            face = torch.stack(faces, dim=-1)
        surface_data = Data(x=feat_tensor, pos=pos, face=face)
        surface_data = FaceToEdge()(surface_data)
        surface_data = Cartesian(cat=False)(surface_data)
        complex_graph["surface"].pos = surface_data.pos
        complex_graph["surface"].x = surface_data.x
        complex_graph["surface", "surface_edge", "surface"].edge_index = surface_data.edge_index
        complex_graph["surface", "surface_edge", "surface"].edge_attr = surface_data.edge_attr
    except Exception:
        return [], []

    return [complex_graph], [lig]


@app.function(
    volumes={DATA_ROOT: runs_volume},
    timeout=1 * HOURS,
)
def merge_shards(cache_dir: str) -> int:
    import glob
    import re

    from loguru import logger

    shard_bins = sorted(
        glob.glob(os.path.join(cache_dir, "shard_*.bin")),
        key=lambda p: int(re.search(r"shard_(\d+)\.bin", p).group(1)),
    )
    logger.info(f"Merging {len(shard_bins)} shards in {cache_dir}")

    BLOCK = 8 * 1024 * 1024
    merged_index: list[tuple[int, int]] = []
    data_bin_path = os.path.join(cache_dir, "data.bin")

    skipped = 0
    with open(data_bin_path, "wb") as out_f:
        for shard_bin in shard_bins:
            shard_id = int(re.search(r"shard_(\d+)\.bin", shard_bin).group(1))
            shard_idx_path = os.path.join(cache_dir, f"shard_{shard_id}_index.pkl")
            if not os.path.exists(shard_idx_path):
                logger.warning(f"Skipping shard {shard_id}: missing index (incomplete write)")
                os.remove(shard_bin)
                skipped += 1
                continue
            with open(shard_idx_path, "rb") as f:
                shard_index: list[tuple[int, int]] = pickle.load(f)

            base_offset = out_f.tell()
            with open(shard_bin, "rb") as in_f:
                while True:
                    chunk = in_f.read(BLOCK)
                    if not chunk:
                        break
                    out_f.write(chunk)

            for offset, length in shard_index:
                merged_index.append((base_offset + offset, length))

            os.remove(shard_bin)
            os.remove(shard_idx_path)

    if skipped:
        logger.warning(f"Skipped {skipped} incomplete shards")

    index_path = os.path.join(cache_dir, "index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(merged_index, f, protocol=-1)

    logger.info(f"Merged {len(merged_index)} samples into data.bin")
    runs_volume.commit()
    return len(merged_index)


@app.function(
    volumes={DATA_ROOT: runs_volume, ASSETS_ROOT: data_volume},
    cpu=1,
    memory=2048,
    timeout=6 * HOURS,
)
def process_complexes(
    chunk_id: int,
    names: list[str],
    cache_dir: str,
    cfg: CacheConfig,
) -> int:
    from loguru import logger

    os.makedirs(cache_dir, exist_ok=True)
    shard_bin = os.path.join(cache_dir, f"shard_{chunk_id}.bin")
    shard_idx = os.path.join(cache_dir, f"shard_{chunk_id}_index.pkl")

    if os.path.exists(shard_bin) and os.path.exists(shard_idx):
        with open(shard_idx, "rb") as f:
            existing = pickle.load(f)
        logger.info(f"Shard {chunk_id} already exists with {len(existing)} entries, skipping")
        runs_volume.commit()
        return len(existing)

    index: list[tuple[int, int]] = []
    processed = 0

    with open(shard_bin, "wb") as data_f:
        for name in names:
            logger.info(f"Processing {name} (chunk={chunk_id})")

            lm_embedding_chains = None
            if cfg.esm_embeddings_path is not None:
                emb_path = os.path.join(
                    cfg.esm_embeddings_path,
                    name,
                    f"{name}_protein_processed_{cfg.pocket_cutoff}_{cfg.esm_model_name}.pt",
                )
                if not os.path.exists(emb_path):
                    logger.info(f"Skipping {name}: no ESM embedding at {emb_path}")
                    continue
                lm_embedding_chains = torch.load(emb_path, weights_only=True)

            graphs, _ligs = _process_one_complex(
                name=name,
                cfg=cfg,
                lm_embedding_chains=lm_embedding_chains,
            )

            for graph in graphs:
                blob = pickle.dumps(graph, protocol=-1)
                offset = data_f.tell()
                data_f.write(blob)
                index.append((offset, len(blob)))
                processed += 1

    with open(shard_idx, "wb") as f:
        pickle.dump(index, f, protocol=-1)

    runs_volume.commit()
    logger.info(f"Shard {chunk_id} done: {processed} complexes")
    return processed


@app.function(
    gpu="H100",
    cpu=2,
    volumes={DATA_ROOT: runs_volume},
    timeout=6 * HOURS,
)
def train(cfg: TrainConfig):
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
    torch.multiprocessing.set_sharing_strategy("file_system")

    import datetime

    from loguru import logger
    from torch.utils.tensorboard import SummaryWriter

    from datasets.pdbbind import construct_loader
    from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
    from utils.training import (
        inference_epoch_parallel,
        loss_function,
        test_epoch,
        train_epoch,
    )
    from utils.utils import (
        ExponentialMovingAverage,
        get_model,
        get_optimizer_and_scheduler,
        save_yaml_file,
    )

    args = cfg
    device = torch.device("cuda")
    accelerator = SingleGPUAccelerator(device)
    torch.manual_seed(42)

    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.add(os.path.join(run_dir, "LogFile.log"), rotation="100 MB")
    logger.info(f"Args: {args}")

    assert args.inference_earlystop_goal in ("max", "min")
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert args.scheduler_patience > args.val_inference_freq

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    for split_name, split_path in [("train", args.split_train), ("val", args.split_val)]:
        cache_base = args.cache_path
        if args.matching:
            cache_base += "_torsion"
        if args.all_atoms:
            cache_base += "_allatoms"
        split_basename = os.path.splitext(os.path.basename(split_path))[0]
        cache_dir = os.path.join(
            cache_base,
            f"limit{args.limit_complexes}"
            f"_INDEX{split_basename}"
            f"_maxLigSize{args.max_lig_size}_H{int(not args.remove_hs)}"
            f"_recRad{args.receptor_radius}_recMax{args.c_alpha_max_neighbors}"
            + (f"_atomRad{args.atom_radius}_atomMax{args.atom_max_neighbors}" if args.all_atoms else "")
            + (f"_confs{args.num_conformers}" if args.matching and args.num_conformers != 1 else "")
            + (f"_{args.esm_model_name}" if args.esm_embeddings_path is not None else "")
            + f"_pocket{args.pocket_cutoff}",
        )
        data_bin = os.path.join(cache_dir, "data.bin")
        exists = os.path.exists(data_bin)
        logger.info(f"Cache [{split_name}]: {cache_dir}")
        logger.info(f"  data.bin exists: {exists}" + (" (WILL BE BUILT ON THE FLY)" if not exists else ""))

    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader = construct_loader(args, t_to_sigma)

    model = get_model(args, device, t_to_sigma=t_to_sigma, model_type=args.model_type)
    object.__setattr__(model, "module", model)

    optimizer, scheduler_obj = get_optimizer_and_scheduler(args, model, accelerator, scheduler_mode="max")
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)

    start_epoch = 0
    if args.restart_dir:
        checkpoint_path = os.path.join(args.restart_dir, "last_model.pt")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if args.restart_lr is not None:
            ckpt["optimizer"]["param_groups"][0]["lr"] = args.restart_lr
        optimizer.load_state_dict(ckpt["optimizer"])
        model.load_state_dict(ckpt["model"], strict=True)
        if "ema_weights" in ckpt:
            ema_weights.load_state_dict(ckpt["ema_weights"], device=device)
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Restarting from epoch {start_epoch}")

    numel = sum(p.numel() for p in model.parameters())
    logger.info(f"Model with {numel} parameters")

    yaml_file_name = os.path.join(run_dir, "model_parameters.yml")
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    tb_writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

    loss_fn = partial(
        loss_function,
        tr_weight=args.tr_weight,
        rot_weight=args.rot_weight,
        tor_weight=args.tor_weight,
        no_torsion=args.no_torsion,
    )

    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == "min" else 0
    best_epoch = 0
    best_val_inference_epoch = 0

    logger.info("Starting training...")
    logger.info("Load val inference dataset ...")
    val_inference_datalist = val_loader.dataset.get_complexs_list(args.num_inference_complexes)
    logger.info(f"Size of dataset is: {len(val_inference_datalist)}.")

    if scheduler_obj is not None:
        scheduler_obj.num_bad_epochs = 1

    for epoch in range(start_epoch, args.n_epochs):
        if epoch % 5 == 0:
            logger.info(f"Run name: {args.run_name}")

        logs = {}

        train_losses = train_epoch(model, train_loader, optimizer, device, t_to_sigma, loss_fn, accelerator, ema_weights)
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"epoch[{epoch}]@{nowtime} --> train_metric=")
        logger.info(
            f"Epoch {epoch}: Training loss {train_losses['loss']:.4f}  "
            f"tr {train_losses['tr_loss']:.4f}   "
            f"rot {train_losses['rot_loss']:.4f}   "
            f"tor {train_losses['tor_loss']:.4f}"
        )

        ema_weights.store(model.parameters())
        if args.use_ema:
            ema_weights.copy_to(model.parameters())

        val_losses = test_epoch(model, val_loader, device, t_to_sigma, loss_fn, accelerator, args.test_sigma_intervals, model_type=args.model_type)
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"epoch[{epoch}]@{nowtime} --> eval_metric=")
        logger.info(
            f"Epoch {epoch}: Validation loss {val_losses['loss']:.4f}  "
            f"tr {val_losses['tr_loss']:.4f}   "
            f"rot {val_losses['rot_loss']:.4f}   "
            f"tor {val_losses['tor_loss']:.4f}"
        )

        if args.val_inference_freq is not None and (epoch + 1) % args.val_inference_freq == 0 and (epoch + 1) > args.skip_inference_freq:
            inf_metrics = inference_epoch_parallel(model, val_inference_datalist, device, t_to_sigma, args, accelerator)
            nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"epoch[{epoch}]@{nowtime} --> inference_metric=")
            logger.info(
                f"Epoch {epoch}: Val inference rmsds_lt2 {inf_metrics['rmsds_lt2']:.3f} "
                f"rmsds_lt5 {inf_metrics['rmsds_lt5']:.3f}"
            )
            logs.update({f"valinf_{k}": v for k, v in inf_metrics.items()})

        if not args.use_ema:
            ema_weights.copy_to(model.parameters())

        ema_state_dict = copy.deepcopy(model.state_dict())
        ema_weights.restore(model.parameters())
        state_dict = model.state_dict()

        logs.update({f"train_{k}": v for k, v in train_losses.items()})
        logs.update({f"val_{k}": v for k, v in val_losses.items()})
        logs["current_lr"] = optimizer.param_groups[0]["lr"]

        for k, v in logs.items():
            tb_writer.add_scalar(k, v, epoch + 1)

        if args.inference_earlystop_metric in logs and (
            (args.inference_earlystop_goal == "min" and logs[args.inference_earlystop_metric] <= best_val_inference_value)
            or (args.inference_earlystop_goal == "max" and logs[args.inference_earlystop_metric] >= best_val_inference_value)
        ):
            best_val_inference_value = logs[args.inference_earlystop_metric]
            best_val_inference_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, "best_inference_epoch_model.pt"))
            torch.save(ema_state_dict, os.path.join(run_dir, "best_ema_inference_epoch_model.pt"))

        if val_losses["loss"] <= best_val_loss:
            best_val_loss = val_losses["loss"]
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, "best_model.pt"))
            torch.save(ema_state_dict, os.path.join(run_dir, "best_ema_model.pt"))

        if scheduler_obj and (epoch + 1) % args.val_inference_freq == 0 and (epoch + 1) > args.skip_inference_freq:
            if args.val_inference_freq is not None and (epoch + 1) > args.skip_inference_freq:
                scheduler_obj.step(best_val_inference_value)
            else:
                scheduler_obj.step(-1 * val_losses["loss"])
            if scheduler_obj is not None and scheduler_obj.num_bad_epochs < accelerator.num_processes:
                scheduler_obj.num_bad_epochs = 1

        torch.save(
            {
                "epoch": epoch,
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "ema_weights": ema_weights.state_dict(),
            },
            os.path.join(run_dir, "last_model.pt"),
        )
        runs_volume.commit()
        logger.info(f"Epoch {epoch} checkpoint saved and committed to volume.")

    logger.info(f"Best Validation Loss {best_val_loss} on Epoch {best_epoch}")
    logger.info(f"Best inference metric {best_val_inference_value} on Epoch {best_val_inference_epoch}")
    tb_writer.close()
    runs_volume.commit()


@app.local_entrypoint()
def build_cache(
    split_path: str = "timesplit_no_lig_overlap_train_surface_complete",
    esm_model_name: str = "esm2_35M",
    pocket_cutoff: str = "8A",
    limit_complexes: int = 0,
    chunk_size: int = 50,
    receptor_radius: float = 30.0,
    c_alpha_max_neighbors: int = 10,
    all_atoms: bool = False,
    atom_radius: float = 5.0,
    atom_max_neighbors: int = 8,
    remove_hs: bool = False,
    matching: bool = True,
    matching_popsize: int = 20,
    matching_maxiter: int = 20,
    num_conformers: int = 1,
    max_lig_size: int | None = None,
):
    if not os.path.isabs(split_path):
        split_path = os.path.join("data", "splits", split_path)

    with open(split_path) as f:
        complex_names = [line.rstrip() for line in f.readlines()]
    if limit_complexes > 0:
        complex_names = complex_names[:limit_complexes]

    print(f"Building cache for {len(complex_names)} complexes from {split_path}")

    data_dir = str(ASSETS_ROOT / "processed")
    cfg = CacheConfig(
        surface_path=data_dir,
        data_dir=data_dir,
        esm_model_name=esm_model_name,
        esm_embeddings_path=data_dir,
        pocket_cutoff=pocket_cutoff,
        receptor_radius=receptor_radius,
        c_alpha_max_neighbors=c_alpha_max_neighbors,
        all_atoms=all_atoms,
        atom_radius=atom_radius,
        atom_max_neighbors=atom_max_neighbors,
        remove_hs=remove_hs,
        matching=matching,
        popsize=matching_popsize,
        maxiter=matching_maxiter,
        num_conformers=num_conformers,
        max_lig_size=max_lig_size,
    )

    vol_split_path = str(DATA_ROOT / "splits" / os.path.basename(split_path))
    cache_dir = str(
        _build_cache_path(
            cache_path=str(DATA_ROOT / "cache"),
            split_path=vol_split_path,
            limit_complexes=limit_complexes,
            cfg=cfg,
        )
    )
    print(f"Cache directory: {cache_dir}")

    chunks = []
    for i, start in enumerate(range(0, len(complex_names), chunk_size)):
        chunk_names = complex_names[start : start + chunk_size]
        chunks.append((i, chunk_names))

    print(f"Launching {len(chunks)} chunks of up to {chunk_size} complexes each...")

    starmap_args = [
        (chunk_id, names, cache_dir, cfg)
        for chunk_id, names in chunks
    ]

    total_processed = 0
    for result in process_complexes.starmap(starmap_args):
        total_processed += result

    print(f"Chunks done. {total_processed} complexes processed.")
    print("Merging shards...")
    merged = merge_shards.remote(cache_dir)
    print(f"Cache build complete. {merged} samples in data.bin.")


@app.local_entrypoint()
def main(
    ns: int = 16,
    nv: int = 4,
    num_conv_layers: int = 2,
    distance_embed_dim: int = 32,
    cross_distance_embed_dim: int = 32,
    sigma_embed_dim: int = 32,
    batch_size: int = 32,
    n_epochs: int = 1,
    lr: float = 1e-3,
    w_decay: float = 0.0,
    limit_complexes: int = 0,
    esm_model_name: str = "esm2_3B",
    pocket_cutoff: str = "8A",
    model_type: str = "surface_score_model",
    model_version: str = "version3",
    run_name: str = "",
    restart_dir: str = "",
    restart_lr: float | None = None,
    use_ema: bool = False,
    ema_rate: float = 0.999,
    no_torsion: bool = False,
    scheduler: str | None = None,
    scheduler_patience: int = 20,
    val_inference_freq: int = 5,
    skip_inference_freq: int = 0,
    num_inference_complexes: int = 100,
    inference_steps: int = 20,
    inference_earlystop_metric: str = "valinf_rmsds_lt2",
    inference_earlystop_goal: str = "max",
    max_radius: float = 5.0,
    cross_max_distance: float = 80.0,
    dynamic_max_cross: bool = False,
    use_second_order_repr: bool = False,
    no_batch_norm: bool = False,
    dropout: float = 0.0,
    embedding_type: str = "sinusoidal",
    embedding_scale: int = 1000,
    tr_sigma_min: float = 0.1,
    tr_sigma_max: float = 30.0,
    rot_sigma_min: float = 0.1,
    rot_sigma_max: float = 1.65,
    tor_sigma_min: float = 0.0314,
    tor_sigma_max: float = 3.14,
    tr_weight: float = 0.33,
    rot_weight: float = 0.33,
    tor_weight: float = 0.33,
    num_dataloader_workers: int = 0,
    pin_memory: bool = False,
    all_atoms: bool = False,
    remove_hs: bool = False,
    receptor_radius: float = 30.0,
    c_alpha_max_neighbors: int = 10,
    atom_radius: float = 5.0,
    atom_max_neighbors: int = 8,
    matching_popsize: int = 20,
    matching_maxiter: int = 20,
    num_conformers: int = 1,
    max_lig_size: int | None = None,
    matching: bool = True,
    num_workers: int = 1,
    scale_by_sigma: bool = True,
    cudnn_benchmark: bool = False,
    test_sigma_intervals: bool = False,
    transform_style: str = "diffdock",
    top_n: int = 1,
):
    cfg = TrainConfig(
        ns=ns, nv=nv, num_conv_layers=num_conv_layers,
        distance_embed_dim=distance_embed_dim, cross_distance_embed_dim=cross_distance_embed_dim,
        sigma_embed_dim=sigma_embed_dim, batch_size=batch_size, n_epochs=n_epochs,
        lr=lr, w_decay=w_decay, limit_complexes=limit_complexes,
        esm_model_name=esm_model_name, pocket_cutoff=pocket_cutoff,
        model_type=model_type, model_version=model_version,
        run_name=run_name, restart_dir=restart_dir, restart_lr=restart_lr,
        use_ema=use_ema, ema_rate=ema_rate, no_torsion=no_torsion,
        scheduler=scheduler, scheduler_patience=scheduler_patience,
        val_inference_freq=val_inference_freq, skip_inference_freq=skip_inference_freq,
        num_inference_complexes=num_inference_complexes, inference_steps=inference_steps,
        inference_earlystop_metric=inference_earlystop_metric,
        inference_earlystop_goal=inference_earlystop_goal,
        max_radius=max_radius, cross_max_distance=cross_max_distance,
        dynamic_max_cross=dynamic_max_cross, use_second_order_repr=use_second_order_repr,
        no_batch_norm=no_batch_norm, dropout=dropout,
        embedding_type=embedding_type, embedding_scale=embedding_scale,
        tr_sigma_min=tr_sigma_min, tr_sigma_max=tr_sigma_max,
        rot_sigma_min=rot_sigma_min, rot_sigma_max=rot_sigma_max,
        tor_sigma_min=tor_sigma_min, tor_sigma_max=tor_sigma_max,
        tr_weight=tr_weight, rot_weight=rot_weight, tor_weight=tor_weight,
        num_dataloader_workers=num_dataloader_workers, pin_memory=pin_memory,
        all_atoms=all_atoms, remove_hs=remove_hs,
        receptor_radius=receptor_radius, c_alpha_max_neighbors=c_alpha_max_neighbors,
        atom_radius=atom_radius, atom_max_neighbors=atom_max_neighbors,
        matching=matching, matching_popsize=matching_popsize, matching_maxiter=matching_maxiter,
        num_conformers=num_conformers, max_lig_size=max_lig_size,
        num_workers=num_workers, scale_by_sigma=scale_by_sigma, cudnn_benchmark=cudnn_benchmark,
        test_sigma_intervals=test_sigma_intervals,
        transformStyle=transform_style, topN=top_n,
    )
    train.remote(cfg)
