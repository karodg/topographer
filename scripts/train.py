"""
TopoGrapher Deep Topological Network Training Script.

Trains the unified ConvNeXt-FPN backbone, Dual Dense Heads, and Graph Transformer
with multi-task supervision:
- Sub-pixel Gaussian Keypoint Peak Loss (Focal-Weighted MSE);
- Road Presence Segmentation Loss (Binary Cross-Entropy);
- Continuous Tangent Unit-Vector Alignment Loss (Cosine Angular Distance);
- Graph Transformer Edge Connectivity & Functional Road Class Classification Loss.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from topographer.models.full_model import TopoGrapherNet
from topographer.data.dataset import ProceduralRoadDataset, collate_topographer_batch
from topographer.metrics.apls import APLSMetric
from topographer.graph.tracer import GraphTracer


def compute_multitask_loss(
    model: TopoGrapherNet,
    batch: Dict[str, Any],
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes balanced composite multi-task loss.
    """
    images = batch["images"].to(device)
    gt_keypoints = batch["keypoints"].to(device)
    gt_masks = batch["road_masks"].to(device)
    gt_orientations = batch["orientations"].to(device)

    # Forward Dense Pass under current autocast
    dense = model.forward_dense(images)
    features = dense["features"]
    pred_kp = dense["keypoints"]
    pred_road = dense["road_prob"]
    pred_orient = dense["tangent"]

    # Compute loss in full precision (disabled autocast for BCELoss safety)
    with torch.amp.autocast('cuda', enabled=False):
        pred_kp = pred_kp.float()
        gt_keypoints = gt_keypoints.float()
        kp_weight = 1.0 + 9.0 * gt_keypoints
        loss_kp = torch.mean(kp_weight * (pred_kp - gt_keypoints) ** 2)

        pred_road = pred_road.float()
        gt_masks = gt_masks.float()
        loss_road = F.binary_cross_entropy(pred_road.clamp(1e-6, 1.0 - 1e-6), gt_masks)

        dot_product = torch.sum(pred_orient.float() * gt_orientations.float(), dim=1, keepdim=True)
        cos_loss_map = 1.0 - dot_product
        road_mask_bool = (gt_masks > 0.5)

        if road_mask_bool.sum() > 0:
            loss_orient = torch.mean(cos_loss_map[road_mask_bool])
        else:
            loss_orient = torch.tensor(0.0, device=device)

        loss_edge_list = []
        loss_class_list = []

        b_size = images.shape[0]
        for b in range(b_size):
            pairs = batch["edge_pairs_list"][b].to(device)
            pair_idxs = batch["edge_indices_list"][b].to(device) if "edge_indices_list" in batch else torch.zeros((pairs.shape[0], 2), dtype=torch.long, device=device)
            labels = batch["edge_labels_list"][b].to(device).float()
            classes = batch["edge_classes_list"][b].to(device)

            if pairs.shape[0] == 0:
                continue

            feat_b = features[b:b+1]
            orient_b = pred_orient[b:b+1]
            road_b = pred_road[b:b+1]

            pred_edge_prob, pred_class_logits, pooled_embed = model.edge_transformer(
                feat_b, pairs, orient_map=orient_b, road_prob=road_b
            )
            refined_probs = model.graph_refiner(pooled_embed, pair_idxs)

            l_e = F.binary_cross_entropy(pred_edge_prob.float().clamp(1e-6, 1.0 - 1e-6), labels)
            l_ref = F.binary_cross_entropy(refined_probs.float().clamp(1e-6, 1.0 - 1e-6), labels)
            loss_edge_list.append(0.5 * l_e + 0.5 * l_ref)

            pos_indices = (labels.squeeze(1) > 0.5)
            if pos_indices.sum() > 0:
                l_c = F.cross_entropy(pred_class_logits[pos_indices].float(), classes[pos_indices])
                loss_class_list.append(l_c)

        loss_edge = torch.stack(loss_edge_list).mean() if loss_edge_list else torch.tensor(0.0, device=device)
        loss_class = torch.stack(loss_class_list).mean() if loss_class_list else torch.tensor(0.0, device=device)

        # Composite weighted loss
        total_loss = (
            15.0 * loss_kp +
            2.0 * loss_road +
            1.5 * loss_orient +
            3.0 * loss_edge +
            0.5 * loss_class
        )

    metrics = {
        "total_loss": float(total_loss.item()),
        "loss_kp": float(loss_kp.item()),
        "loss_road": float(loss_road.item()),
        "loss_orient": float(loss_orient.item()),
        "loss_edge": float(loss_edge.item()),
        "loss_class": float(loss_class.item())
    }

    return total_loss, metrics


def train_topographer(
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 3e-4,
    image_size: int = 384,
    save_dir: str = "checkpoints",
    checkpoint_name: str = "topographer_weights.pt"
):
    """
    Main training execution loop with mixed precision and progress telemetry.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Initializing TopoGrapher Training on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, checkpoint_name)

    model = TopoGrapherNet(
        in_channels=3,
        base_channels=32,
        feature_dim=64,
        num_keypoint_types=3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Virtual datasets
    train_dataset = ProceduralRoadDataset(
        num_samples=160,
        image_size=image_size,
        max_edge_dist=120.0,
        augment=True
    )
    val_dataset = ProceduralRoadDataset(
        num_samples=20,
        image_size=image_size,
        max_edge_dist=120.0,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_topographer_batch,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_topographer_batch,
        num_workers=0
    )

    best_val_loss = float('inf')
    start_time = time.time()

    print(f"--> Starting {epochs} Epochs [Batch Size: {batch_size}, Image Size: {image_size}x{image_size}]")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                loss, metrics = compute_multitask_loss(model, batch, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(metrics["total_loss"])

            if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] Step [{step+1:03d}/{len(train_loader):03d}] "
                    f"Loss: {metrics['total_loss']:.4f} (KP: {metrics['loss_kp']:.4f}, "
                    f"Road: {metrics['loss_road']:.4f}, Orient: {metrics['loss_orient']:.4f}, "
                    f"Edge: {metrics['loss_edge']:.4f})"
                )

        scheduler.step()

        # Validation Pass
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_loader:
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    _, val_metrics = compute_multitask_loss(model, val_batch, device)
                val_losses.append(val_metrics["total_loss"])

        avg_train = np.mean(epoch_losses)
        avg_val = np.mean(val_losses)
        print(f"=== Epoch {epoch:02d} Complete | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} ===")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss
            }, save_path)
            print(f"  [*] New Best Checkpoint Saved: {save_path} (Val Loss: {best_val_loss:.4f})")

    elapsed = time.time() - start_time
    print(f"\n[DONE] Training successfully completed in {elapsed:.1f}s. Model checkpoint: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TopoGrapher Deep Topological Network")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per step")
    parser.add_argument("--image-size", type=int, default=384, help="Square image resolution")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save weights")
    args = parser.parse_args()

    train_topographer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        save_dir=args.save_dir
    )
