from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class SpatialAttention1d(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(pooled))


class ModalityDDCAE(nn.Module):
    def __init__(self, kernel_size: int, latent_dim: int, attention_kernel: int) -> None:
        super().__init__()
        filters = [8, 8, 16, 16, 32, 32]
        blocks = []
        in_channels = 1
        for out_channels in filters:
            blocks.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv1d(
                            in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2
                        ),
                        "attention": SpatialAttention1d(attention_kernel),
                        "pool": nn.MaxPool1d(kernel_size=2, stride=2),
                    }
                )
            )
            in_channels = out_channels
        self.encoder_blocks = nn.ModuleList(blocks)
        self.to_latent = nn.Linear(32 * 18, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 32 * 18)
        decoder_filters = [32, 32, 16, 16, 8, 8]
        decoder = []
        in_channels = 32
        for out_channels in decoder_filters:
            decoder.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
            )
            in_channels = out_channels
        self.decoder_convs = nn.ModuleList(decoder)
        self.output_conv = nn.Conv1d(8, 1, kernel_size, stride=1, padding=kernel_size // 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.encoder_blocks:
            x = F.elu(block["conv"](x))
            x = block["attention"](x)
            x = block["pool"](x)
        if x.shape[1:] != (32, 18):
            raise ValueError(f"Encoder produced {tuple(x.shape[1:])}, expected (32, 18)")
        return F.elu(self.to_latent(x.flatten(start_dim=1)))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.from_latent(latent)).reshape(-1, 32, 18)
        for conv in self.decoder_convs:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = F.elu(conv(x))
        return self.output_conv(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        return latent, self.decode(latent)


class FeatureWiseModalityGate(nn.Module):
    def __init__(self, modalities: list[str], latent_dim: int) -> None:
        super().__init__()
        self.modalities = modalities
        self.latent_dim = latent_dim
        self.normalizers = nn.ModuleDict({name: nn.Linear(latent_dim, latent_dim) for name in modalities})
        joined = len(modalities) * latent_dim
        self.weight_layer = nn.Linear(joined, joined)

    def forward(self, latents: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = [torch.tanh(self.normalizers[name](latents[name])) for name in self.modalities]
        weights = torch.softmax(self.weight_layer(torch.cat(normalized, dim=1)), dim=1)
        sections = weights.reshape(-1, len(self.modalities), self.latent_dim)
        stacked = torch.stack([latents[name] for name in self.modalities], dim=1)
        return (sections * stacked).sum(dim=1), sections


class ThiamSupervisedDDCAE(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model = config["model"]
        self.modalities = list(config["data"]["modalities"])
        self.noise_mean = float(model["gaussian_noise_mean"])
        self.noise_std = float(model["gaussian_noise_std"])
        latent_dim = int(model["latent_dim"])
        self.autoencoders = nn.ModuleDict(
            {
                modality: ModalityDDCAE(
                    int(model["modality_kernel_sizes"][modality]),
                    latent_dim,
                    int(model["attention_kernel_size"]),
                )
                for modality in self.modalities
            }
        )
        self.gate = FeatureWiseModalityGate(self.modalities, latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, int(model["classifier_hidden"])),
            nn.ELU(),
            nn.Dropout(float(model["dropout"])),
            nn.Linear(int(model["classifier_hidden"]), 5),
        )

    def forward(self, inputs: Mapping[str, torch.Tensor], corrupt: bool | None = None) -> dict[str, torch.Tensor]:
        corrupt = self.training if corrupt is None else corrupt
        latents = {}
        reconstructions = {}
        for modality in self.modalities:
            clean = inputs[modality]
            noisy = clean
            if corrupt and self.noise_std > 0:
                noisy = clean + torch.randn_like(clean) * self.noise_std + self.noise_mean
            latent, reconstruction = self.autoencoders[modality](noisy)
            latents[modality] = latent
            reconstructions[modality] = reconstruction
        fused, gate_weights = self.gate(latents)
        return {
            "logits": self.classifier(fused),
            "reconstructions": reconstructions,
            "latents": latents,
            "gate_weights": gate_weights,
        }


def joint_loss(
    model: ThiamSupervisedDDCAE,
    outputs: dict,
    clean_inputs: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pieces = {}
    total = torch.zeros((), device=targets.device)
    l2_weight = float(config["loss"]["autoencoder_l2"])
    for modality in model.modalities:
        reconstruction = F.mse_loss(outputs["reconstructions"][modality], clean_inputs[modality])
        l2 = 0.5 * sum(parameter.square().sum() for parameter in model.autoencoders[modality].parameters())
        autoencoder_loss = reconstruction + l2_weight * l2
        weight = float(config["loss"]["reconstruction_weights"][modality])
        total = total + weight * autoencoder_loss
        pieces[f"reconstruction_{modality}"] = reconstruction.detach()
        pieces[f"l2_{modality}"] = l2.detach()
    classification = F.cross_entropy(outputs["logits"], targets)
    total = total + float(config["loss"]["classification_weight"]) * classification
    pieces["classification"] = classification.detach()
    pieces["total"] = total.detach()
    return total, pieces
