import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import (
    VIDEO_FRAMES,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    TEXT_DIM,
    IMAGE_DIM,
    CONDITION_DIM,
    TIME_EMBED_DIM,
    DIFFUSION_STEPS,
    BETA_START,
    BETA_END,
    RESOLUTION_IDS,
)


# ============================================================
# TinyVideo V1.4
# Latent Video Diffusion Model
#
# RGB:
#     [B, 3, 8, 64, 64]
#
# Latent:
#     [B, 4, 8, 16, 16]
#
# The diffusion model operates ONLY in latent space.
# ============================================================


LATENT_CHANNELS = 4
LATENT_HEIGHT = VIDEO_HEIGHT // 4
LATENT_WIDTH = VIDEO_WIDTH // 4

BASE_CHANNELS = 24


# ============================================================
# Resolution
# ============================================================

def extract_resolution_id(prompt):
    prompt = str(prompt).lower()

    if "1080p" in prompt:
        return RESOLUTION_IDS["1080p"]

    if "720p" in prompt:
        return RESOLUTION_IDS["720p"]

    if "420p" in prompt:
        return RESOLUTION_IDS["420p"]

    return RESOLUTION_IDS["unknown"]


# ============================================================
# Timestep embedding
# ============================================================

def timestep_embedding(timesteps, dimension):
    half = dimension // 2

    device = timesteps.device

    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(
            half,
            device=device,
            dtype=torch.float32,
        )
        / max(half - 1, 1)
    )

    values = (
        timesteps.float().unsqueeze(1)
        * frequencies.unsqueeze(0)
    )

    embedding = torch.cat(
        [
            torch.sin(values),
            torch.cos(values),
        ],
        dim=1,
    )

    if dimension % 2:
        embedding = F.pad(
            embedding,
            (0, 1),
        )

    return embedding


# ============================================================
# Conditioning MLP
# ============================================================

class ConditionMLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Tiny Video Encoder
#
# Processes each frame independently.
#
# 64x64
#   ↓
# 32x32
#   ↓
# 16x16
#
# Output:
# [B, 4, T, 16, 16]
# ============================================================

class TinyVideoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                32,
                48,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                48,
                LATENT_CHANNELS,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, video):
        # video:
        # [B, 3, T, H, W]

        batch, channels, frames, height, width = video.shape

        video = video.permute(
            0,
            2,
            1,
            3,
            4,
        )

        video = video.reshape(
            batch * frames,
            channels,
            height,
            width,
        )

        latent = self.net(video)

        latent = latent.reshape(
            batch,
            frames,
            LATENT_CHANNELS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
        )

        latent = latent.permute(
            0,
            2,
            1,
            3,
            4,
        )

        return latent


# ============================================================
# Tiny Video Decoder
#
# 16x16
#   ↓
# 32x32
#   ↓
# 64x64
# ============================================================

class TinyVideoDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                LATENT_CHANNELS,
                48,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),

            nn.ConvTranspose2d(
                48,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                16,
                3,
                kernel_size=3,
                padding=1,
            ),

            nn.Sigmoid(),
        )

    def forward(self, latent):

        batch, channels, frames, height, width = latent.shape

        latent = latent.permute(
            0,
            2,
            1,
            3,
            4,
        )

        latent = latent.reshape(
            batch * frames,
            channels,
            height,
            width,
        )

        video = self.net(latent)

        video = video.reshape(
            batch,
            frames,
            3,
            VIDEO_HEIGHT,
            VIDEO_WIDTH,
        )

        video = video.permute(
            0,
            2,
            1,
            3,
            4,
        )

        return video


# ============================================================
# Residual latent diffusion block
# ============================================================

class LatentResidualBlock(nn.Module):

    def __init__(
        self,
        channels,
        condition_dim,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            8,
            channels,
        )

        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(
            8,
            channels,
        )

        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.condition = nn.Linear(
            condition_dim,
            channels,
        )

        self.activation = nn.SiLU()

    def forward(
        self,
        x,
        condition,
    ):
        residual = x

        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)

        condition = self.condition(condition)

        condition = condition[
            :,
            :,
            None,
            None,
            None,
        ]

        h = h + condition

        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)

        return residual + h


# ============================================================
# Latent Diffusion Denoiser
# ============================================================

class TinyLatentDenoiser(nn.Module):

    def __init__(
        self,
        text_dim=TEXT_DIM,
        image_dim=IMAGE_DIM,
        condition_dim=CONDITION_DIM,
    ):
        super().__init__()

        self.input = nn.Conv3d(
            LATENT_CHANNELS,
            BASE_CHANNELS,
            kernel_size=3,
            padding=1,
        )

        self.block1 = LatentResidualBlock(
            BASE_CHANNELS,
            condition_dim,
        )

        self.block2 = LatentResidualBlock(
            BASE_CHANNELS,
            condition_dim,
        )

        self.block3 = LatentResidualBlock(
            BASE_CHANNELS,
            condition_dim,
        )

        self.output_norm = nn.GroupNorm(
            8,
            BASE_CHANNELS,
        )

        self.output = nn.Conv3d(
            BASE_CHANNELS,
            LATENT_CHANNELS,
            kernel_size=3,
            padding=1,
        )

        # -------------------------
        # Text conditioning
        # -------------------------

        self.text_projection = ConditionMLP(
            text_dim,
            64,
        )

        # -------------------------
        # Image conditioning
        # -------------------------

        self.image_projection = ConditionMLP(
            image_dim,
            64,
        )

        # -------------------------
        # Resolution conditioning
        # -------------------------

        self.resolution_embedding = nn.Embedding(
            len(RESOLUTION_IDS),
            32,
        )

        # -------------------------
        # Timestep conditioning
        # -------------------------

        self.time_projection = ConditionMLP(
            TIME_EMBED_DIM,
            32,
        )

        # -------------------------
        # Fuse everything
        # -------------------------

        self.condition_fusion = nn.Sequential(
            nn.Linear(
                64 + 64 + 32 + 32,
                condition_dim,
            ),
            nn.SiLU(),

            nn.Linear(
                condition_dim,
                condition_dim,
            ),
        )

    def forward(
        self,
        noisy_latent,
        text_embedding,
        timestep,
        image_embedding=None,
        resolution_ids=None,
    ):

        batch_size = noisy_latent.shape[0]
        device = noisy_latent.device

        # -------------------------
        # Text
        # -------------------------

        text_embedding = text_embedding.float()

        text_condition = self.text_projection(
            text_embedding
        )

        # -------------------------
        # Image
        # -------------------------

        if image_embedding is None:

            image_embedding = torch.zeros(
                batch_size,
                IMAGE_DIM,
                device=device,
                dtype=text_embedding.dtype,
            )

        else:

            image_embedding = image_embedding.float()

        image_condition = self.image_projection(
            image_embedding
        )

        # -------------------------
        # Resolution
        # -------------------------

        if resolution_ids is None:

            resolution_ids = torch.zeros(
                batch_size,
                device=device,
                dtype=torch.long,
            )

        resolution_ids = resolution_ids.long().clamp(
            0,
            len(RESOLUTION_IDS) - 1,
        )

        resolution_condition = (
            self.resolution_embedding(
                resolution_ids
            )
        )

        # -------------------------
        # Timestep
        # -------------------------

        time = timestep_embedding(
            timestep,
            TIME_EMBED_DIM,
        )

        time_condition = self.time_projection(
            time
        )

        # -------------------------
        # Combine conditioning
        # -------------------------

        condition = torch.cat(
            [
                text_condition,
                image_condition,
                resolution_condition,
                time_condition,
            ],
            dim=1,
        )

        condition = self.condition_fusion(
            condition
        )

        # -------------------------
        # Denoising
        # -------------------------

        x = self.input(noisy_latent)

        x = self.block1(
            x,
            condition,
        )

        x = self.block2(
            x,
            condition,
        )

        x = self.block3(
            x,
            condition,
        )

        x = self.output_norm(x)

        x = F.silu(x)

        return self.output(x)


# ============================================================
# TinyVideo Diffusion
# ============================================================

class TinyVideoDiffusion(nn.Module):

    def __init__(
        self,
        text_dim=TEXT_DIM,
    ):
        super().__init__()

        # -------------------------
        # Latent autoencoder
        # -------------------------

        self.encoder = TinyVideoEncoder()

        self.decoder = TinyVideoDecoder()

        # -------------------------
        # Diffusion denoiser
        # -------------------------

        self.denoiser = TinyLatentDenoiser(
            text_dim=text_dim,
        )

        # -------------------------
        # Image encoder
        # -------------------------

        self.image_encoder = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                16,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        self.image_projection = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64,
                IMAGE_DIM,
            ),

            nn.SiLU(),

            nn.Linear(
                IMAGE_DIM,
                IMAGE_DIM,
            ),
        )

        # -------------------------
        # Diffusion schedule
        # -------------------------

        self.register_buffer(
            "betas",
            torch.linspace(
                BETA_START,
                BETA_END,
                DIFFUSION_STEPS,
            ),
        )

        self.register_buffer(
            "alphas",
            1.0 - self.betas,
        )

        self.register_buffer(
            "alpha_bars",
            torch.cumprod(
                self.alphas,
                dim=0,
            ),
        )

    # ========================================================
    # Video encoding
    # ========================================================

    def encode_video(self, video):

        video = video.float()

        return self.encoder(video)

    # ========================================================
    # Video decoding
    # ========================================================

    def decode_video(self, latent):

        return self.decoder(latent)

    # ========================================================
    # Image encoding
    # ========================================================

    def encode_image(self, image):

        image = image.float()

        features = self.image_encoder(
            image
        )

        return self.image_projection(
            features
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        noisy_latent,
        token_ids,
        timesteps,
        image=None,
        resolution_ids=None,
    ):

        if image is not None:

            image_embedding = self.encode_image(
                image
            )

        else:

            image_embedding = None

        return self.denoiser(
            noisy_latent=noisy_latent,
            text_embedding=token_ids,
            timestep=timesteps,
            image_embedding=image_embedding,
            resolution_ids=resolution_ids,
        )

    # ========================================================
    # Add diffusion noise
    # ========================================================

    def add_noise(
        self,
        clean_latent,
        noise,
        timesteps,
    ):

        alpha_bar = self.alpha_bars[
            timesteps
        ]

        alpha_bar = alpha_bar.view(
            -1,
            1,
            1,
            1,
            1,
        )

        return (
            torch.sqrt(alpha_bar)
            * clean_latent
            +
            torch.sqrt(1.0 - alpha_bar)
            * noise
        )

    # ========================================================
    # Recover x0
    # ========================================================

    def predict_x0(
        self,
        noisy_latent,
        noise_prediction,
        timesteps,
    ):

        alpha_bar = self.alpha_bars[
            timesteps
        ]

        alpha_bar = alpha_bar.view(
            -1,
            1,
            1,
            1,
            1,
        )

        return (
            noisy_latent
            -
            torch.sqrt(
                1.0 - alpha_bar
            )
            * noise_prediction
        ) / torch.sqrt(
            alpha_bar
        )


# Backwards-compatible name.
TinyVideoGenerator = TinyVideoDiffusion