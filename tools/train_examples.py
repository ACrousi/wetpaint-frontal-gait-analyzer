import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

# Core pipeline components
from src.core.pipeline_builder import build_model
from src.core.unified_trainer import UnifiedTrainer, TrainConfig
from src.core.datasets import FeatureDataset, collate_feature_batch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_train_config(train_section: Dict[str, Any]) -> TrainConfig:
    # device: "auto" | "cuda:0" | "cpu"
    dev = train_section.get("device", "auto")
    device = None if str(dev).lower() == "auto" else str(dev)
    return TrainConfig(
        epochs=int(train_section.get("epochs", 10)),
        lr=float(train_section.get("lr", 1e-3)),
        weight_decay=float(train_section.get("weight_decay", 0.0)),
        device=device,
        log_interval=int(train_section.get("log_interval", 50)),
    )


def _make_dataloaders(
    samples_train: List[Dict[str, Any]],
    samples_val: Optional[List[Dict[str, Any]]],
    dataloader_cfg: Dict[str, Any],
    batch_size: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_ds = FeatureDataset(samples_train)
    max_frames = int(dataloader_cfg.get("max_frames", 150))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=bool(dataloader_cfg.get("shuffle", True)),
        num_workers=int(dataloader_cfg.get("num_workers", 0)),
        pin_memory=bool(dataloader_cfg.get("pin_memory", False)),
        drop_last=bool(dataloader_cfg.get("drop_last", False)),
        collate_fn=lambda batch: collate_feature_batch(batch, max_frames=max_frames),
    )

    val_loader = None
    if samples_val is not None:
        val_ds = FeatureDataset(samples_val)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(dataloader_cfg.get("num_workers", 0)),
            pin_memory=bool(dataloader_cfg.get("pin_memory", False)),
            drop_last=False,
            collate_fn=lambda batch: collate_feature_batch(batch, max_frames=max_frames),
        )
    return train_loader, val_loader


def run_baseline_training(
    config_path: str,
    samples_train: List[Dict[str, Any]],
    samples_val: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    使用 UnifiedTrainer 訓練 baseline(MLP)。
    samples_* 需為由 FeatureDataset 支援的樣本 dict 列表：
      - 每筆可包含：'gait' 或 'metrics'，以及 'label'
    """
    cfg = _load_yaml(config_path)

    # 確保 model.type = baseline
    cfg.setdefault("model", {})
    cfg["model"]["type"] = "baseline"

    model_wrapper = build_model(cfg)

    train_cfg = _make_train_config(cfg.get("train", {}).get("baseline", {}))
    dl_cfg = cfg.get("dataloader", {})

    train_loader, val_loader = _make_dataloaders(
        samples_train, samples_val, dl_cfg, batch_size=int(cfg.get("train", {}).get("baseline", {}).get("batch_size", 64))
    )

    trainer = UnifiedTrainer(model_wrapper, mode="baseline", cfg=train_cfg)
    history = trainer.train(train_loader, val_loader)
    return history


def run_fusion_head_training(
    config_path: str,
    samples_train: List[Dict[str, Any]],
    samples_val: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    使用 UnifiedTrainer 訓練 fusion concat head（僅 head；子分支固定）。
    samples_* 需為由 FeatureDataset 支援的樣本 dict 列表：
      - 每筆需包含：'skeleton' 以及 'gait' 或 'metrics'，以及 'label'
    """
    cfg = _load_yaml(config_path)

    # 確保 model.type = fusion
    cfg.setdefault("model", {})
    cfg["model"]["type"] = "fusion"

    model_wrapper = build_model(cfg)

    train_cfg = _make_train_config(cfg.get("train", {}).get("fusion_head", {}))
    dl_cfg = cfg.get("dataloader", {})

    train_loader, val_loader = _make_dataloaders(
        samples_train, samples_val, dl_cfg, batch_size=int(cfg.get("train", {}).get("fusion_head", {}).get("batch_size", 32))
    )

    trainer = UnifiedTrainer(model_wrapper, mode="fusion_head", cfg=train_cfg)
    history = trainer.train(train_loader, val_loader)
    return history


if __name__ == "__main__":
    # 簡易示範（以隨機資料 smoke test）
    # 說明：
    # - baseline: 使用 'gait' 向量與 'label'
    # - fusion_head: 使用 'skeleton' + 'gait' 與 'label'
    import argparse

    parser = argparse.ArgumentParser(description="UnifiedTrainer quick demos with random data.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument("--demo", type=str, choices=["baseline", "fusion_head"], default="baseline")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    num_classes = int(cfg.get("baseline", {}).get("num_classes", cfg.get("fusion", {}).get("num_classes", 5)))
    gait_dim = int(cfg.get("baseline", {}).get("metrics_dim", cfg.get("fusion", {}).get("metrics_dim", 50)))
    max_frames = int(cfg.get("dataloader", {}).get("max_frames", 150))
    V = 17  # COCO keypoints

    N_train, N_val = 32, 16
    rng = np.random.default_rng(42)

    if args.demo == "baseline":
        # 產生隨機 gait + label
        train_samples = [{"gait": rng.normal(0, 1, size=(gait_dim,)).astype(np.float32), "label": int(rng.integers(0, num_classes))} for _ in range(N_train)]
        val_samples = [{"gait": rng.normal(0, 1, size=(gait_dim,)).astype(np.float32), "label": int(rng.integers(0, num_classes))} for _ in range(N_val)]
        # 覆寫 epochs 以快速 smoke test
        cfg.setdefault("train", {}).setdefault("baseline", {})["epochs"] = args.epochs
        # 暫存覆寫到記憶體設定：直接傳遞 config_path 使用者提供的檔案，不改寫磁碟
        history = run_baseline_training(args.config, train_samples, val_samples)
        logger.info(f"Baseline training finished. History keys: {list(history.keys())}")
    else:
        # 產生隨機 skeleton (C,T,V) + gait + label
        C, T = 3, max_frames
        train_samples = [{
            "skeleton": rng.normal(0, 1, size=(C, T, V)).astype(np.float32),
            "gait": rng.normal(0, 1, size=(gait_dim,)).astype(np.float32),
            "label": int(rng.integers(0, num_classes)),
        } for _ in range(N_train)]
        val_samples = [{
            "skeleton": rng.normal(0, 1, size=(C, T, V)).astype(np.float32),
            "gait": rng.normal(0, 1, size=(gait_dim,)).astype(np.float32),
            "label": int(rng.integers(0, num_classes)),
        } for _ in range(N_val)]
        cfg.setdefault("train", {}).setdefault("fusion_head", {})["epochs"] = args.epochs
        history = run_fusion_head_training(args.config, train_samples, val_samples)
        logger.info(f"Fusion-head training finished. History keys: {list(history.keys())}")