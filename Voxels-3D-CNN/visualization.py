from pathlib import Path
from typing import Iterable
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

SCENE_TEMPLATE = dict(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False),
    aspectmode="data",
    bgcolor="#111111",
)

class _VoxelGeometryMixin:
    """Provides voxel corner generation utilities for Plotly visualizations."""

    geometry_config = {
        "corner_order": (
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        )
    }

    def _voxel_corners(self, voxel_indices: np.ndarray, voxel_size: Iterable[float], point_cloud_range: Iterable[float]) -> np.ndarray:
        voxel_indices = np.asarray(voxel_indices)
        voxel_size = np.asarray(voxel_size)
        point_cloud_range = np.asarray(point_cloud_range)
        centers = (
            voxel_indices[:, ::-1] * voxel_size
            + point_cloud_range[:3]
            + voxel_size * 0.5
        )
        half_sizes = np.append(voxel_size, 0.0) 
        template = np.array(
            [
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [-1, 1, 1],
            ],
            dtype=np.float32,
        )
        template *= 0.5
        corners = []
        for center in centers:
            scaled = template * voxel_size
            corners.append(scaled + center)
        return np.stack(corners, axis=0)

    def _voxel_edge_traces(self, corners: np.ndarray, color: str) -> Iterable[go.Scatter3d]:
        traces = []
        for box in corners:
            for indices in self.geometry_config["corner_order"]:
                traces.append(
                    go.Scatter3d(
                        x=box[list(indices), 0],
                        y=box[list(indices), 1],
                        z=box[list(indices), 2],
                        mode="lines",
                        line=dict(color=color, width=2),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        return traces


class TrainingCurveVisualizer:
    """Plots training and validation loss curves and saves static images."""

    config = {"figure_size": (10, 4), "train_color": "tab:red", "val_color": "tab:blue"}

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, history, filename: str = "training_curves.png") -> Path:
        fig, ax1 = plt.subplots(figsize=self.config["figure_size"])
        ax1.set_ylabel("Train Loss", color=self.config["train_color"])
        ax1.plot(history["epoch"], history["train_loss"], color=self.config["train_color"])
        ax1.tick_params(axis="y", labelcolor=self.config["train_color"])

        ax2 = ax1.twinx()
        ax2.set_ylabel("Validation Loss", color=self.config["val_color"])
        ax2.plot(history["epoch"], history["val_loss"], color=self.config["val_color"])
        ax2.tick_params(axis="y", labelcolor=self.config["val_color"])

        plt.title("Training and Validation Loss")
        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Saved training_curves visualization to {output_path}")
        return output_path


class MultiClassPredictionVisualizer(_VoxelGeometryMixin):
    """Creates 4-row × 3-column grids showing predictions for multiple classes."""

    config = {
        "point_marker_size": 1.5,
        "point_colorscale": "Viridis",
        "voxel_edge_color": "#FFFFFF",
        "figure_height": 1200,
        "figure_width": 1800,
    }

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_html(self, samples: list, class_names: list, voxel_size: Iterable[float],
                  point_cloud_range: Iterable[float], filename: str) -> Path:
        num_samples = len(samples)

        # Create subplot grid: 4 rows, 3 columns
        specs = []
        for _ in range(num_samples):
            specs.append([{"type": "scatter3d"}, {"type": "scatter3d"}, None])

        fig = make_subplots(
            rows=num_samples,
            cols=3,
            specs=specs,
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
        )

        # Add traces for each sample
        for idx, sample in enumerate(samples):
            row = idx + 1
            points = sample["points"]
            voxel_indices = sample["voxel_coords"]
            predicted_label = sample["predicted_label"]
            true_label = sample["true_label"]

            # Column 1: Point cloud
            point_trace = go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=self.config["point_marker_size"],
                    color=np.linalg.norm(points, axis=1),
                    colorscale=self.config["point_colorscale"],
                    showscale=False,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
            fig.add_trace(point_trace, row=row, col=1)

            # Column 2: Voxelized representation
            voxel_corners = self._voxel_corners(voxel_indices, voxel_size, point_cloud_range)
            voxel_traces = list(self._voxel_edge_traces(voxel_corners, self.config["voxel_edge_color"]))
            for trace in voxel_traces:
                fig.add_trace(trace, row=row, col=2)

            # Column 3: Add text annotations
            prediction_color = "#4CAF50" if predicted_label == true_label else "#F44336"
            row_height = (1.0 - 0.08 * (num_samples - 1)) / num_samples
            y_top = 1.0 - (idx * (row_height + 0.08))
            y_center = y_top - row_height / 2
            x_center = 0.70 + 0.15 

            fig.add_annotation(
                text=f"<b>Original:</b> {true_label}",
                xref="paper",
                yref="paper",
                x=x_center,
                y=y_center + 0.02,
                showarrow=False,
                font=dict(size=16, color="#2196F3"),
                xanchor="center",
                yanchor="middle",
            )

            fig.add_annotation(
                text=f"<b>Predicted:</b> {predicted_label}",
                xref="paper",
                yref="paper",
                x=x_center,
                y=y_center - 0.02,
                showarrow=False,
                font=dict(size=16, color=prediction_color),
                xanchor="center",
                yanchor="middle",
            )

        scene_idx = 1
        for i in range(num_samples):
            scene_name_1 = "scene" if scene_idx == 1 else f"scene{scene_idx}"
            scene_name_2 = f"scene{scene_idx + 1}"

            fig.update_scenes(SCENE_TEMPLATE, row=i+1, col=1)
            fig.update_scenes(SCENE_TEMPLATE, row=i+1, col=2)

            scene_idx += 2

        fig.update_layout(
            template="plotly_dark",
            height=self.config["figure_height"],
            width=self.config["figure_width"],
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        print(f"Saved multi-class visualization to {output_path}")
        return output_path

