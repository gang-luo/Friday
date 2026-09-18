"""Main entrypoint for IgGM Lightning training/validation/testing."""

from __future__ import annotations

import argparse
import os
import torch

os.environ["HSAKMT_DEBUG_LEVEL"] = "3"   # 只输出 error，不输出 warning
os.environ["HSA_ENABLE_INTERRUPT"] = "0" # 可选

import sys
from pathlib import Path
from typing import Any, Dict

import warnings

# 屏蔽常见 Python warning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required. Please install with `pip install pyyaml`.") from exc

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
    from lightning.pytorch.strategies import DDPStrategy
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger
    from pytorch_lightning.strategies import DDPStrategy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from IgGM.model import DesignModel, esm_ppi_650m_ab
from IgGM.model.arch.core.diffuser import Diffuser
from IgGM.model.factory import build_design_model_module, build_ppi_featurizer_module
from iggm_lightning import (
    IgGMLightningModule,
    IgGMLossConfig,
    MetricConfig,
    OptimizerConfig,
    ProcessedSabdabDataModule,
    StageTrainingConfig,
)


class DotConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _resolve_trainer_strategy(args: argparse.Namespace):
    strategy_name = str(args.strategy).strip().lower()
    if strategy_name in {"", "auto"}:
        return "auto"
    if strategy_name == "ddp":
        return DDPStrategy(
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
            static_graph=bool(args.ddp_static_graph),
            gradient_as_bucket_view=True,
        )
    return args.strategy


def _load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def _parser_with_defaults(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    data = defaults.get("data", {})
    model = defaults.get("model", {})
    optim = defaults.get("optimizer", {})
    scheduler = defaults.get("scheduler", {})
    loss_cfg = defaults.get("loss", {})
    metric_cfg = defaults.get("metrics", {})
    eval_cfg = defaults.get("eval", {})
    trainer = defaults.get("trainer", {})
    wandb = defaults.get("wandb", {})
    runtime = defaults.get("runtime", {})
    stage_training = defaults.get("stage_training", {})
    diffusion = defaults.get("diffusion", {})
    

    p = argparse.ArgumentParser(description="Train IgGM with PyTorch Lightning")
    p.add_argument("--config", default=defaults.get("config_path", "config/train_lightning.yaml"))

    p.add_argument("--metadata", default=data.get("metadata", "data/sabdab/processed/sabdab/metadata.json"))
    p.add_argument("--pdb_dir", default=data.get("pdb_dir", "data/sabdab/metadata"))
    p.add_argument("--train_ids", default=data.get("train_ids", ""))
    p.add_argument("--val_ids", default=data.get("val_ids", ""))
    p.add_argument("--test_ids", default=data.get("test_ids", ""))
    p.add_argument("--test_nanobody_ids", default=data.get("test_nanobody_ids", ""))
    p.add_argument("--test_standard_ids", default=data.get("test_standard_ids", ""))
    p.add_argument("--train_clusters", default=data.get("train_clusters", ""))
    p.add_argument("--batch_size", type=int, default=int(data.get("batch_size", 1)))
    p.add_argument("--num_workers", type=int, default=int(data.get("num_workers", 0)))
    p.add_argument("--samples_dir", default=data.get("samples_dir", ""))
    p.add_argument("--n_steps", type=int, default=int(data.get("n_steps", 200)))
    p.add_argument("--forward_chunk_size", type=int, default=int(data.get("forward_chunk_size", 64)))
    p.add_argument("--max_antigen_len", type=int, default=int(data.get("max_antigen_len", 0)))

    p.add_argument("--ppi_ckpt", default=model.get("ppi_ckpt", ""))
    p.add_argument("--design_ckpt", default=model.get("design_ckpt", ""))
    p.add_argument("--rota_angle_rms_min", type=float, default=float(diffusion.get("rota_angle_rms_min", 0.005)))
    p.add_argument("--rota_schedule_gamma", type=float, default=float(diffusion.get("rota_schedule_gamma", 1.5)))

    p.add_argument("--sched_name", default=scheduler.get("name", "common_lr"))
    p.add_argument("--sched_t_max", type=int, default=int(scheduler.get("t_max", 10000)))
    p.add_argument("--sched_eta_min", type=float, default=float(scheduler.get("eta_min", 1e-6)))
    p.add_argument("--sched_warmup_steps", type=int, default=int(scheduler.get("warmup_steps", 0)))

    p.add_argument("--lr", type=float, default=float(optim.get("lr", 1e-4)))
    p.add_argument("--weight_decay", type=float, default=float(optim.get("weight_decay", 1e-2)))
    p.add_argument("--grad_clip", type=float, default=float(optim.get("grad_clip", 1.0)))
    # EMA is off unless a decay is given.  Validation runs on the EMA weights
    # when enabled (see IgGMLightningModule.on_validation_start): it smooths the
    # eval reading without touching the training gradients.
    p.add_argument("--ema_decay", type=float, default=(
        float(optim["ema_decay"]) if optim.get("ema_decay") is not None else None
    ))

    p.add_argument("--vio_weight", type=float, default=float(loss_cfg.get("vio_weight", 0.02)))
    p.add_argument("--dockq_threshold", type=float, default=float(metric_cfg.get("dockq_threshold", 0.23)))
    # Precise (reverse-sampling) validation layer.  Cadence is in VALIDATION RUNS,
    # not epochs, so it does not silently change when check_val_every_n_epoch does.
    p.add_argument("--precise_eval_every_n_val", type=int,
                   default=int(eval_cfg.get("precise_eval_every_n_val", 0)))
    p.add_argument("--precise_eval_steps", type=int,
                   default=(int(eval_cfg["precise_eval_steps"])
                            if eval_cfg.get("precise_eval_steps") is not None else None))
    p.add_argument("--precise_eval_seed", type=int,
                   default=int(eval_cfg.get("precise_eval_seed", 20260909)))
    p.add_argument("--cheap_eval_seed", type=int,
                   default=int(eval_cfg.get("cheap_eval_seed", 20260915)))

    p.add_argument("--max_epochs", type=int, default=int(trainer.get("max_epochs", 1)))
    p.add_argument("--precision", default=trainer.get("precision", "16-mixed"))
    p.add_argument("--accelerator", default=trainer.get("accelerator", "auto"))
    p.add_argument("--devices", default=trainer.get("devices", "auto"))
    p.add_argument("--log_every_n_steps", type=int, default=int(trainer.get("log_every_n_steps", 1)))
    p.add_argument("--accumulate_grad_batches", type=int, default=int(trainer.get("accumulate_grad_batches", 1)))
    p.add_argument("--num_sanity_val_steps", type=int, default=int(trainer.get("num_sanity_val_steps", 0)))
    # Was hardcoded to 50 in the Trainer.  It has to be a knob: the precise
    # layer's cadence is counted in validation runs, so the two are read together.
    p.add_argument("--check_val_every_n_epoch", type=int, default=int(trainer.get("check_val_every_n_epoch", 50)))
    p.add_argument("--strategy", default=trainer.get("strategy", "auto"))
    p.add_argument("--ddp_static_graph", type=_parse_bool, default=bool(trainer.get("ddp_static_graph", False)))
    p.add_argument("--ddp_find_unused_parameters", type=_parse_bool, default=bool(trainer.get("ddp_find_unused_parameters", True)))
    
    p.add_argument("--project", default=wandb.get("project", "iggm-lightning"))
    p.add_argument("--run_name", default=wandb.get("run_name", "iggm-train"))
    p.add_argument("--entity", default=wandb.get("entity", ""))
    p.add_argument("--api_key", default=wandb.get("api_key", ""))
    p.add_argument("--no_wandb", action="store_true", default=bool(wandb.get("disabled", False)))

    p.add_argument("--output_dir", default=runtime.get("output_dir", "outputs/lightning_train"))
    p.add_argument("--seed", type=int, default=int(runtime.get("seed", 42)))
    p.add_argument("--resume", action="store_true", default=bool(runtime.get("resume", False)))
    p.add_argument("--run_test", action="store_true", default=bool(runtime.get("run_test", False)))

    p.add_argument("--stage1_epochs", type=int, default=int(stage_training.get("stage1_epochs", 0)))
    p.add_argument("--stage2_enable_seq_recovery", type=_parse_bool, default=bool(stage_training.get("stage2_enable_seq_recovery", True)))
    p.add_argument("--mix_cdr_h3", type=int, default=int(stage_training.get("mix_cdr_h3", 4)))
    p.add_argument("--mix_cdr_h1", type=int, default=int(stage_training.get("mix_cdr_h1", 2)))
    p.add_argument("--mix_cdr_h2", type=int, default=int(stage_training.get("mix_cdr_h2", 2)))
    p.add_argument("--mix_cdr_all", type=int, default=int(stage_training.get("mix_cdr_all", 2)))
    p.add_argument("--lazy_cache_size", type=int, default=int(stage_training.get("lazy_cache_size", 128)))
    p.add_argument("--occupancy_mode", default=stage_training.get("occupancy_mode", diffusion.get("occupancy_mode", "joint_predict")))
    return p


def _parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="config/train_lightning.yaml")
    boot_args, _ = bootstrap.parse_known_args()
    defaults = _load_yaml_config(boot_args.config)
    defaults["config_path"] = boot_args.config
    parser = _parser_with_defaults(defaults)
    args = parser.parse_args()
    return args


def main() -> None:

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True) 

    args = _parse_args()
    pl.seed_everything(args.seed, workers=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dm = ProcessedSabdabDataModule(
        metadata_path=args.metadata,
        pdb_dir=args.pdb_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ids_path=(args.train_ids or None),
        val_ids_path=(args.val_ids or None),
        test_ids_path=(args.test_ids or None),
        test_nanobody_ids_path=(args.test_nanobody_ids or None),
        test_standard_ids_path=(args.test_standard_ids or None),
        train_cluster_path=(args.train_clusters or None),
        samples_dir=(args.samples_dir or None),
        n_steps=args.n_steps,
        lazy_cache_size=args.lazy_cache_size,
        forward_chunk_size=(args.forward_chunk_size if args.forward_chunk_size > 0 else None),
        max_antigen_len=(args.max_antigen_len if args.max_antigen_len > 0 else None),
    )
    dm.setup()

    model_cfg = DotConfig(c_s=None, c_p=None)

    ppi_ckpt_path = args.ppi_ckpt or esm_ppi_650m_ab()
    plm_featurizer = build_ppi_featurizer_module(ppi_ckpt_path)
    c_s = getattr(plm_featurizer, "c_s", None)
    c_p = getattr(plm_featurizer, "c_z", None)
    if c_s is not None:
        model_cfg.c_s = c_s
    if c_p is not None:
        model_cfg.c_p = c_p

    # if args.design_ckpt:
    #     design_model = build_design_model_module(args.design_ckpt, model_cfg)
    # else:
    #     design_model = DesignModel(n_dims_sfea_init=model_cfg.c_s, n_dims_pfea_init=model_cfg.c_p)

    if args.design_ckpt:
        design_model = build_design_model_module(args.design_ckpt, model_cfg)
        
        # --- 新增：专门处理 Lightning 的 .ckpt 权重加载 ---
        if args.design_ckpt.endswith('.ckpt'):
            print(f"Loading state_dict from Lightning checkpoint: {args.design_ckpt}")
            ckpt = torch.load(args.design_ckpt, map_location="cpu")
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
                # 剥离 "model." 前缀
                clean_state_dict = {
                    k.replace("model.", ""): v 
                    for k, v in state_dict.items() if k.startswith("model.")
                }
                # 加载清洗后的权重 (建议去掉 strict=False 以便观察是否真正匹配)
                missing, unexpected = design_model.load_state_dict(clean_state_dict, strict=False)
                if not missing:
                    print("✅ Successfully loaded all model weights from checkpoint!")
                else:
                    print(f"⚠️ Partially loaded. Missing keys: {len(missing)}")
        # ---------------------------------------------------
    else:
        design_model = DesignModel(n_dims_sfea_init=model_cfg.c_s, n_dims_pfea_init=model_cfg.c_p)

    diffuser = Diffuser(
        n_steps=args.n_steps,
        occupancy_mode=args.occupancy_mode,
        rota_angle_rms_min=args.rota_angle_rms_min,
        rota_schedule_gamma=args.rota_schedule_gamma,
    )
    
    if args.sched_name != "common_lr":
        scheduler_cfg = {
            "name": args.sched_name,
            "t_max": args.sched_t_max,
            "eta_min": args.sched_eta_min,
            "warmup_steps": args.sched_warmup_steps, # 新增这一行
        }
    else:   
        scheduler_cfg = None
        
    lit_model = IgGMLightningModule(
        model=design_model,
        plm_featurizer=plm_featurizer,
        diffuser=diffuser,
        scheduler_cfg = scheduler_cfg,
        optimizer_cfg=OptimizerConfig(lr=args.lr, weight_decay=args.weight_decay),
        grad_clip_val=args.grad_clip,
        ema_decay=args.ema_decay,
        loss_cfg=IgGMLossConfig(
            # backbone_weight=args.backbone_weight,
            # cdr_all_atom_weight=args.cdr_all_atom_weight,
            vio_weight=args.vio_weight,
        ),
        metric_cfg=MetricConfig(dockq_threshold=args.dockq_threshold),
        precise_eval_every_n_val=args.precise_eval_every_n_val,
        precise_eval_steps=args.precise_eval_steps,
        precise_eval_seed=args.precise_eval_seed,
        cheap_eval_seed=args.cheap_eval_seed,
        stage_cfg=StageTrainingConfig(
            stage1_epochs=args.stage1_epochs,
            stage2_enable_seq_recovery=args.stage2_enable_seq_recovery,
            stage2_mix_weights={
                "cdr_h3": args.mix_cdr_h3,
                "cdr_h1": args.mix_cdr_h1,
                "cdr_h2": args.mix_cdr_h2,
                "cdr_all": args.mix_cdr_all,
            },
        ),
    )

    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    # Cheap-layer checkpoint.  It can no longer monitor val/tm_score: the cheap
    # layer deliberately skips StructureMetrics (single-step metrics are expensive
    # and are not the selection criterion), so it tracks the mid-noise loss
    # instead.  The real selection metric comes from the precise layer below.
    cheap_monitor = f"val_t{IgGMLightningModule.CHEAP_VAL_TIMESTEPS[len(IgGMLightningModule.CHEAP_VAL_TIMESTEPS) // 2]}/loss"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="cheap-{epoch:02d}",
            monitor=cheap_monitor,
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if args.precise_eval_every_n_val > 0:
        # Separate file and separate monitor, so the two layers never overwrite
        # each other's "best".  Added only when the precise layer is on, because
        # Lightning raises on a monitor that is never logged.
        callbacks.insert(
            1,
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="precise-{epoch:02d}",
                monitor="val/precise_tm_score",
                mode="max",
                save_top_k=1,
                save_last=False,
            ),
        )

    if args.no_wandb:
        logger = CSVLogger(str(out), name="csv_logs")
    else:
        if args.api_key:
            os.environ["WANDB_API_KEY"] = args.api_key
        logger = WandbLogger(
            project=args.project,
            name=args.run_name,
            save_dir=str(out),
            entity=(args.entity or None),
            log_model=False,
        )

    trainer = pl.Trainer(
        default_root_dir=str(out),
        max_epochs=args.max_epochs,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=_resolve_trainer_strategy(args),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=max(1, args.accumulate_grad_batches),
        num_sanity_val_steps=max(0, args.num_sanity_val_steps),
        # check_val_every_n_epoch=2,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        # # detect_anomaly=True,
    )

    last_ckpt = ckpt_dir / "last.ckpt"
    ckpt_path = str(last_ckpt) if args.resume and last_ckpt.exists() else None

    trainer.fit(lit_model, datamodule=dm, ckpt_path=ckpt_path)

    if args.run_test:
        trainer.test(lit_model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
