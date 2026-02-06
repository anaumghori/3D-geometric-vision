import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ShapeNetDataset(Dataset):
    """Loads ShapeNet point clouds and performs voxelization on demand."""

    config = {"point_features": 3}

    def __init__(self, root_dir: Path, split_name: str, voxelizer):
        self.root_dir = Path(root_dir)
        self.split_name = split_name
        self.voxelizer = voxelizer
        split_file = self.root_dir / f"{split_name}_split.json"
        with open(split_file, "r", encoding="utf-8") as handle:
            self.samples = json.load(handle)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        class_id, class_name, point_cloud_path = self.samples[index]
        absolute_path = self.root_dir / point_cloud_path
        points = np.load(absolute_path)

        voxels, voxel_indices, point_counts = self.voxelizer.generate(points)

        return {
            "points": points.astype(np.float32),
            "num_points": points.shape[0],
            "voxels": voxels.astype(np.float32),
            "num_voxels": voxels.shape[0],
            "voxel_coords": voxel_indices.astype(np.int32),
            "voxel_point_counts": point_counts.astype(np.float32),
            "class_id": int(class_id),
            "class_name": class_name,
        }


class VoxelDataCollator:
    """Collates variable-sized voxel batches into spconv-friendly structures."""

    @staticmethod
    def collate(batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        batch_size = len(batch)

        class_ids = np.array([item["class_id"] for item in batch], dtype=np.int64)
        class_names = np.array([item["class_name"] for item in batch])
        num_points = np.array([item["num_points"] for item in batch], dtype=np.int32)
        num_voxels = np.array([item["num_voxels"] for item in batch], dtype=np.int32)

        voxels = np.concatenate([item["voxels"] for item in batch], axis=0)
        voxel_point_counts = np.concatenate(
            [item["voxel_point_counts"] for item in batch], axis=0
        )

        voxel_coords_list = []
        for batch_index, item in enumerate(batch):
            coords = item["voxel_coords"]
            batch_column = np.full((coords.shape[0], 1), batch_index, dtype=np.int32)
            voxel_coords_list.append(np.hstack((batch_column, coords)))
        voxel_coords = np.concatenate(voxel_coords_list, axis=0)

        points_list = [item["points"] for item in batch]

        return {
            "class_id": class_ids,
            "class_name": class_names,
            "num_points": num_points,
            "num_voxels": num_voxels,
            "voxels": voxels,
            "voxel_point_counts": voxel_point_counts,
            "voxel_coords": voxel_coords,
            "points": points_list,
            "batch_size": batch_size,
        }


class DataModule:
    """Creates ShapeNet datasets and dataloaders for train, validation, and test splits."""

    config = {
        "batch_size": 16,
        "num_workers": 2,
        "pin_memory": True,
        "train_shuffle": True,
    }

    def __init__(
        self,
        dataset_root: Path,
        voxelizer,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.voxelizer = voxelizer
        self.config = {**self.config}
        if batch_size is not None:
            self.config["batch_size"] = batch_size
        if num_workers is not None:
            self.config["num_workers"] = num_workers

        self.datasets: Dict[str, ShapeNetDataset] = {}
        self.dataloaders: Dict[str, DataLoader] = {}

    def setup(self) -> None:
        for split in ("train", "val", "test"):
            self.datasets[split] = ShapeNetDataset(
                root_dir=self.dataset_root,
                split_name=split,
                voxelizer=self.voxelizer,
            )

    def get_dataloader(self, split: str) -> DataLoader:
        if split not in self.dataloaders:
            self.dataloaders[split] = DataLoader(
                self.datasets[split],
                batch_size=self.config["batch_size"],
                shuffle=self.config["train_shuffle"] if split == "train" else False,
                num_workers=self.config["num_workers"],
                pin_memory=self.config["pin_memory"],
                collate_fn=VoxelDataCollator.collate,
            )
        return self.dataloaders[split]

    def get_sample(self) -> Dict[str, np.ndarray]:
        return self.datasets["train"][0] if "train" in self.datasets else {}


class BatchDeviceMover:
    """Transfers collated batches to the specified device"""

    config = {
        "float_keys": ("voxels", "voxel_point_counts"),
        "long_keys": ("class_id",),
        "int_keys": ("voxel_coords",),
    }

    def __init__(self, device: torch.device):
        self.device = device

    def move(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        tensor_batch: Dict[str, torch.Tensor] = batch.copy()

        for key in self.config["float_keys"]:
            tensor_batch[key] = torch.from_numpy(batch[key]).float().to(self.device)
        for key in self.config["long_keys"]:
            tensor_batch[key] = torch.from_numpy(batch[key]).long().to(self.device)
        for key in self.config["int_keys"]:
            tensor_batch[key] = torch.from_numpy(batch[key]).int().to(self.device)

        tensor_batch["batch_size"] = batch["batch_size"]
        return tensor_batch


class ClassificationMetric:
    """Tracks running classification accuracy across batches."""

    def __init__(self):
        self.reset()

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        self.preds.extend(predictions.tolist())
        self.targets.extend(targets.tolist())

    def compute(self) -> float:
        if not self.targets:
            return 0.0
        return accuracy_score(self.targets, self.preds)

    def reset(self) -> None:
        self.preds: List[int] = []
        self.targets: List[int] = []


class ModelTrainer:
    """Coordinates training, validation, and testing loops with logging and checkpointing."""

    config = {"gradient_clip_norm": None}

    def __init__(self, device: torch.device, criterion, optimizer, scheduler=None, checkpoint_path: Optional[Path] = None):
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path
        self.mover = BatchDeviceMover(device)

    def train(self, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> pd.DataFrame:
        history_rows: List[Dict[str, float]] = []
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}")
            train_loss = self._run_train_epoch(model, train_loader)
            val_loss, val_accuracy = self.evaluate(model, val_loader)

            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )

            if val_loss < best_val_loss and self.checkpoint_path is not None:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.checkpoint_path)
                print(f"Saved model checkpoint to {self.checkpoint_path}")

            print(
                f"Epoch {epoch + 1}: train_loss={train_loss:.5f}, "
                f"val_loss={val_loss:.5f}, accuracy={val_accuracy:.4f}"
            )

        return pd.DataFrame(history_rows)

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        metric = ClassificationMetric()

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc="Eval", leave=False):
                tensor_batch = self.mover.move(batch)
                outputs = model(tensor_batch)
                loss = self.criterion(outputs["logits"], tensor_batch["class_id"])
                total_loss += loss.item()
                metric.update(outputs["logits"], tensor_batch["class_id"])

        average_loss = total_loss / max(len(dataloader), 1)
        accuracy = metric.compute()
        return average_loss, accuracy

    def _run_train_epoch(self, model: torch.nn.Module, dataloader: DataLoader) -> float:
        model.train()
        total_loss = 0.0

        progress = tqdm(dataloader, total=len(dataloader), desc="Train", leave=False)
        for batch in progress:
            tensor_batch = self.mover.move(batch)
            outputs = model(tensor_batch)
            loss = self.criterion(outputs["logits"], tensor_batch["class_id"])

            loss.backward()
            if self.config["gradient_clip_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config["gradient_clip_norm"]
                )

            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(len(dataloader), 1)

