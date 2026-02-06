from typing import Dict, Iterable, List, Optional
import numpy as np
import torch
import torch.nn as nn
from functools import partial
import spconv.pytorch as spconv


class MeanVoxelEncoder(nn.Module):
    """Averages point features within each voxel to obtain fixed-length representations."""

    config = {"min_points": 1.0}

    def __init__(self):
        super().__init__()

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        voxel_features = batch_dict["voxels"]
        point_counts = batch_dict["voxel_point_counts"]

        points_sum = voxel_features.sum(dim=1)
        normalizer = torch.clamp_min(
            point_counts.view(-1, 1), min=self.config["min_points"]
        )
        voxel_means = points_sum / normalizer
        batch_dict["voxel_features"] = voxel_means.contiguous()
        return batch_dict


class SparseCNNBackbone(nn.Module):
    """Sparse convolutional backbone that extracts voxel-wise features."""

    config = {
        "input_channels": 3,
        "conv1_channels": 16,
        "conv2_channels": 32,
        "conv3_channels": 64,
        "output_channels": 128,
        "kernel_size": 3,
        "stride2": 2,
        "norm_eps": 1e-3,
        "norm_momentum": 0.01,
    }

    def __init__(self, grid_size: Iterable[int], input_channels: Optional[int] = None, config_override: Optional[Dict[str, float]] = None):
        super().__init__()
        self.grid_size = np.array(list(grid_size), dtype=np.int32)
        self.config = {**self.config, **(config_override or {})}
        if input_channels is not None:
            self.config["input_channels"] = input_channels

        norm_fn = partial(
            nn.BatchNorm1d,
            eps=self.config["norm_eps"],
            momentum=self.config["norm_momentum"],
        )

        self.sparse_shape = self.grid_size[::-1].tolist()

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                self.config["input_channels"],
                self.config["conv1_channels"],
                self.config["kernel_size"],
                padding=1,
                bias=False,
                indice_key="subm1",
            ),
            norm_fn(self.config["conv1_channels"]),
            nn.ReLU(inplace=True),
        )

        self.conv1 = spconv.SparseSequential(
            self._post_activation_block(
                self.config["conv1_channels"],
                self.config["conv1_channels"],
                indice_key="subm1",
                norm_fn=norm_fn,
            )
        )

        self.conv2 = spconv.SparseSequential(
            self._post_activation_block(
                self.config["conv1_channels"],
                self.config["conv2_channels"],
                stride=self.config["stride2"],
                padding=1,
                indice_key="spconv2",
                conv_type="spconv",
                norm_fn=norm_fn,
            ),
            self._post_activation_block(
                self.config["conv2_channels"],
                self.config["conv2_channels"],
                indice_key="subm2",
                norm_fn=norm_fn,
            ),
            self._post_activation_block(
                self.config["conv2_channels"],
                self.config["conv2_channels"],
                indice_key="subm2",
                norm_fn=norm_fn,
            ),
        )

        self.conv3 = spconv.SparseSequential(
            self._post_activation_block(
                self.config["conv2_channels"],
                self.config["conv3_channels"],
                stride=self.config["stride2"],
                padding=1,
                indice_key="spconv3",
                conv_type="spconv",
                norm_fn=norm_fn,
            ),
            self._post_activation_block(
                self.config["conv3_channels"],
                self.config["conv3_channels"],
                indice_key="subm3",
                norm_fn=norm_fn,
            ),
            self._post_activation_block(
                self.config["conv3_channels"],
                self.config["conv3_channels"],
                indice_key="subm3",
                norm_fn=norm_fn,
            ),
        )

        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(
                self.config["conv3_channels"],
                self.config["output_channels"],
                (3, 1, 1),
                stride=(self.config["stride2"], 1, 1),
                padding=0,
                bias=False,
                indice_key="spconv_down2",
            ),
            norm_fn(self.config["output_channels"]),
            nn.ReLU(inplace=True),
        )

    def _post_activation_block(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
        padding: int = 1, indice_key: Optional[str] = None, conv_type: str = "subm", norm_fn=nn.BatchNorm1d) -> spconv.SparseSequential:
        if conv_type == "subm":
            conv = spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size,
                bias=False,
                padding=padding,
                indice_key=indice_key,
            )
        else:
            conv = spconv.SparseConv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                indice_key=indice_key,
            )
        return spconv.SparseSequential(conv, norm_fn(out_channels), nn.ReLU(inplace=True))

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        voxel_features = batch_dict["voxel_features"]
        voxel_coords = batch_dict["voxel_coords"].int()
        batch_size = batch_dict["batch_size"]

        input_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords,
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )

        x = self.conv_input(input_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        encoded = self.conv_out(x_conv3)

        batch_dict["encoded_tensor"] = encoded
        batch_dict["backbone_features"] = {
            "conv1": x_conv1,
            "conv2": x_conv2,
            "conv3": x_conv3,
        }
        return batch_dict


class ClassificationHead(nn.Module):
    """Fully connected head that maps sparse features to class logits."""

    config = {"hidden_units": [128, 64], "dropout": 0.0}

    def __init__(self, num_classes: int, hidden_units: Optional[List[int]] = None, config_override: Optional[Dict[str, float]] = None):
        super().__init__()
        self.num_classes = num_classes
        self.config = {**self.config, **(config_override or {})}
        if hidden_units is not None:
            self.config["hidden_units"] = hidden_units
        self.layers: Optional[nn.Sequential] = None

    def _build_layers(self, input_dim: int) -> None:
        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in self.config["hidden_units"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if self.config["dropout"] > 0:
                layers.append(nn.Dropout(self.config["dropout"]))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dense_tensor = batch_dict["encoded_tensor"].dense()
        batch_size = batch_dict["batch_size"]
        flattened = dense_tensor.view(batch_size, -1)

        if self.layers is None:
            self._build_layers(flattened.shape[1])
            self.layers.to(flattened.device)

        logits = self.layers(flattened)
        batch_dict["logits"] = logits
        return batch_dict


class PointCloudClassifier(nn.Module):
    """End-to-end classifier composed of voxel encoder, sparse backbone, and classifier head."""

    config = {"num_classes": 16}

    def __init__(
        self,
        num_classes: int,
        grid_size: Iterable[int],
        encoder: Optional[MeanVoxelEncoder] = None,
        backbone: Optional[SparseCNNBackbone] = None,
        head: Optional[ClassificationHead] = None,
    ):
        super().__init__()
        self.config["num_classes"] = num_classes
        self.encoder = encoder if encoder is not None else MeanVoxelEncoder()
        self.backbone = (
            backbone
            if backbone is not None
            else SparseCNNBackbone(grid_size=grid_size, input_channels=3)
        )
        self.head = (
            head
            if head is not None
            else ClassificationHead(num_classes=num_classes)
        )

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_dict = self.encoder(batch_dict)
        batch_dict = self.backbone(batch_dict)
        batch_dict = self.head(batch_dict)
        return batch_dict

