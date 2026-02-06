from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import cumm.tensorview as tv
from spconv.utils import Point2VoxelCPU3d
from model import PointCloudClassifier
from training import BatchDeviceMover, DataModule, ModelTrainer, VoxelDataCollator
from visualization import MultiClassPredictionVisualizer, TrainingCurveVisualizer

CLASS_ID_MAP = {
    "Airplane": 0,
    "Bag": 1,
    "Cap": 2,
    "Car": 3,
    "Chair": 4,
    "Earphone": 5,
    "Guitar": 6,
    "Knife": 7,
    "Lamp": 8,
    "Laptop": 9,
    "Motorbike": 10,
    "Mug": 11,
    "Pistol": 12,
    "Rocket": 13,
    "Skateboard": 14,
    "Table": 15,
}
ID_CLASS_MAP = {v: k for k, v in CLASS_ID_MAP.items()}


class GenerateVoxel:
    """Encapsulates Point2Voxel voxelization for ShapeNet point clouds."""

    config = {
        "voxel_size": np.array([0.05, 0.05, 0.05]),
        "point_cloud_range": np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]),
        "num_point_features": 3,
        "max_points_per_voxel": 25,
        "max_voxels": 2000,
    }

    def __init__(self):
        self.voxelizer = Point2VoxelCPU3d(
            vsize_xyz=self.config["voxel_size"],
            coors_range_xyz=self.config["point_cloud_range"],
            num_point_features=self.config["num_point_features"],
            max_num_points_per_voxel=self.config["max_points_per_voxel"],
            max_num_voxels=self.config["max_voxels"],
        )
        self.grid_size = self._compute_grid_size()

    def _compute_grid_size(self) -> np.ndarray:
        pc_range = self.config["point_cloud_range"]
        voxel_size = self.config["voxel_size"]
        dims = (pc_range[3:] - pc_range[:3]) / voxel_size
        return np.round(dims).astype(np.int32)

    def generate(self, points: np.ndarray):
        voxels, indices, counts = self.voxelizer.point_to_voxel(tv.from_numpy(points))
        return voxels.numpy(), indices.numpy(), counts.numpy()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path("shapenet-core-dataset")
    voxel_generator = GenerateVoxel()

    data_module = DataModule(
        dataset_root=dataset_root,
        voxelizer=voxel_generator,
        batch_size=16,
        num_workers=2,
    )
    data_module.setup()

    train_loader = data_module.get_dataloader("train")
    val_loader = data_module.get_dataloader("val")
    test_loader = data_module.get_dataloader("test")

    model = PointCloudClassifier(
        num_classes=len(CLASS_ID_MAP),
        grid_size=voxel_generator.grid_size,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=3e-4,
        epochs=3,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10,
        anneal_strategy="cos",
    )

    checkpoint_path = results_dir / "PointCloudClassifier.pt"
    trainer = ModelTrainer(
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
    )

    history = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
    )

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_loss, test_accuracy = trainer.evaluate(model, test_loader)
    print(f"Test: loss={test_loss:.5f}, accuracy={test_accuracy:.4f}")

    training_curve_viz = TrainingCurveVisualizer(results_dir)
    training_curve_viz.save(history)

    sample = data_module.get_sample()
    if sample:
        multi_class_viz = MultiClassPredictionVisualizer(results_dir)

        collated = VoxelDataCollator.collate([sample])
        mover = BatchDeviceMover(device)
        tensor_batch = mover.move(collated)
        tensor_batch["batch_size"] = 1

        model.eval()
        with torch.no_grad():
            output = model(tensor_batch)
            prediction_index = torch.argmax(output["logits"], dim=1).item()
        predicted_label = ID_CLASS_MAP.get(prediction_index, "Unknown")
        true_label = ID_CLASS_MAP.get(sample["class_id"], "Unknown")

        viz_sample = {
            "points": sample["points"],
            "voxel_coords": sample["voxel_coords"],
            "predicted_label": predicted_label,
            "true_label": true_label,
        }

        multi_class_viz.save_html(
            samples=[viz_sample],
            class_names=list(CLASS_ID_MAP.keys()),
            voxel_size=voxel_generator.config["voxel_size"],
            point_cloud_range=voxel_generator.config["point_cloud_range"],
            filename="predictions.html",
        )


if __name__ == "__main__":
    main()

