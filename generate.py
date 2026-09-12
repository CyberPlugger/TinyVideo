import argparse
import os

import torch

from model import TinyVideoDiffusion
from text_encoder import (
    tokenize_text,
    extract_resolution_id,
)
from video_utils import save_video
from constants import *


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    os.environ.get(
        "TINYVIDEO_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
)


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

def load_model(checkpoint_path=CHECKPOINT_PATH):
    model = TinyVideoDiffusion().to(DEVICE)

    checkpoint_path = str(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train V1.4 first with train.py."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


# ------------------------------------------------------------
# Conditioning
# ------------------------------------------------------------

def prepare_text(prompt):
    tokens = tokenize_text(prompt).unsqueeze(0)
    return tokens.to(DEVICE)


def prepare_resolution(prompt):
    resolution_id = extract_resolution_id(prompt)

    return torch.tensor(
        [resolution_id],
        dtype=torch.long,
        device=DEVICE,
    )


# ------------------------------------------------------------
# Diffusion sampler
# ------------------------------------------------------------

@torch.inference_mode()
def diffusion_sample(
    model,
    prompt,
    image=None,
    steps=30,
    seed=None,
    image_strength=0.85,
):
    """
    V1.4 latent-space diffusion sampler.

    The diffusion model operates on:

        [B, 4, T, 16, 16]

    rather than raw RGB video:

        [B, 3, T, 64, 64]

    The final latent is decoded back into RGB video.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1.")

    if seed is None:
        seed = int(torch.seed())

    seed = int(seed)

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)

    # --------------------------------------------------------
    # Text conditioning
    # --------------------------------------------------------

    token_ids = prepare_text(prompt)

    # --------------------------------------------------------
    # Resolution conditioning
    # --------------------------------------------------------

    resolution_ids = prepare_resolution(prompt)

    # --------------------------------------------------------
    # Create initial RGB noise
    #
    # We create RGB-shaped noise first because the model's
    # encoder knows how to convert RGB video into its latent
    # representation.
    # --------------------------------------------------------

    rgb_noise = torch.randn(
        (
            1,
            3,
            VIDEO_FRAMES,
            VIDEO_HEIGHT,
            VIDEO_WIDTH,
        ),
        generator=generator,
        device=DEVICE,
    )

    rgb_noise = rgb_noise.clamp(-1.0, 1.0)

    # --------------------------------------------------------
    # Convert RGB noise -> V1.4 latent
    #
    # This is the critical fix.
    #
    # Old code:
    #
    #   model(video)
    #
    # where video was [1,3,8,64,64]
    #
    # New code:
    #
    #   model.encode_video(video)
    #
    # producing [1,4,8,16,16].
    # --------------------------------------------------------

    video = model.encode_video(rgb_noise)

    # --------------------------------------------------------
    # Image conditioning
    #
    # The image is passed to the diffusion model as an image
    # condition. The model handles image encoding internally.
    # --------------------------------------------------------

    if image is not None:
        image = image.to(
            device=DEVICE,
            dtype=torch.float32,
        )

        if image.ndim == 3:
            image = image.unsqueeze(0)

        if image.ndim != 4:
            raise ValueError(
                "Image must have shape [1,3,H,W] or [3,H,W]."
            )

        if image.shape[1] != 3:
            raise ValueError(
                f"Image must have 3 RGB channels, got "
                f"{image.shape[1]}."
            )

        image = image.clamp(0.0, 1.0)

    # --------------------------------------------------------
    # Diffusion timestep schedule
    # --------------------------------------------------------

    timesteps = torch.linspace(
        model.alpha_bars.shape[0] - 1,
        0,
        steps,
        device=DEVICE,
    ).long()

    # --------------------------------------------------------
    # Reverse diffusion
    # --------------------------------------------------------

    for index, timestep in enumerate(timesteps):

        current_t = timestep.view(1)

        # ----------------------------------------------------
        # Predict noise in LATENT space
        # ----------------------------------------------------

        predicted_noise = model(
            video,
            token_ids,
            current_t,
            image=image,
            resolution_ids=resolution_ids,
        )

        # ----------------------------------------------------
        # Current alpha
        # ----------------------------------------------------

        alpha_bar = model.alpha_bars[timestep]

        # ----------------------------------------------------
        # Previous timestep
        # ----------------------------------------------------

        if index + 1 < len(timesteps):

            previous_timestep = timesteps[index + 1]

            previous_alpha_bar = model.alpha_bars[
                previous_timestep
            ]

        else:

            previous_alpha_bar = torch.tensor(
                1.0,
                device=DEVICE,
            )

        alpha_bar = alpha_bar.clamp(
            min=1e-5,
            max=1.0,
        )

        previous_alpha_bar = previous_alpha_bar.clamp(
            min=1e-5,
            max=1.0,
        )

        # ----------------------------------------------------
        # Predict clean latent x0
        # ----------------------------------------------------

        predicted_clean = (
            video
            - torch.sqrt(
                1.0 - alpha_bar
            ) * predicted_noise
        )

        predicted_clean = (
            predicted_clean
            / torch.sqrt(alpha_bar)
        )

        predicted_clean = predicted_clean.clamp(
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # DDIM-style deterministic update
        # ----------------------------------------------------

        if index + 1 < len(timesteps):

            direction = (
                torch.sqrt(
                    1.0 - previous_alpha_bar
                )
                * predicted_noise
            )

            video = (
                torch.sqrt(
                    previous_alpha_bar
                )
                * predicted_clean
                + direction
            )

        else:

            video = predicted_clean

        video = video.clamp(
            -1.0,
            1.0,
        )

    # --------------------------------------------------------
    # Decode latent -> RGB video
    #
    # Latent:
    #   [1,4,8,16,16]
    #
    # RGB:
    #   [1,3,8,64,64]
    # --------------------------------------------------------

    video = model.decode_video(video)

    video = video.clamp(
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # Convert:
    #
    # [B,C,T,H,W]
    #
    # ->
    #
    # [T,C,H,W]
    # --------------------------------------------------------

    video = video[0]

    video = (
        video + 1.0
    ) / 2.0

    video = video.clamp(
        0.0,
        1.0,
    )

    video = video.permute(
        1,
        0,
        2,
        3,
    )

    return video, seed


# ------------------------------------------------------------
# Text -> video
# ------------------------------------------------------------

def generate(
    prompt,
    output=None,
    seed=None,
    steps=30,
    resolution=None,
):
    """
    Text-to-video generation.
    """

    if resolution:
        prompt = (
            f"{prompt} | "
            f"resolution: {resolution}"
        )

    model = load_model()

    video, used_seed = diffusion_sample(
        model=model,
        prompt=prompt,
        image=None,
        steps=steps,
        seed=seed,
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    if output is None:
        output = os.path.join(
            OUTPUT_DIR,
            f"video_seed_{used_seed}.mp4",
        )

    save_video(
        video,
        output,
        fps=VIDEO_FPS,
    )

    return output


# ------------------------------------------------------------
# Image + prompt -> video
# ------------------------------------------------------------

def generate_with_image(
    prompt,
    image,
    output=None,
    seed=None,
    steps=30,
    resolution=None,
    image_strength=0.85,
):
    """
    Image + prompt -> video.

    image:
        Tensor with shape [1,3,H,W]
        or [3,H,W]

    Values:
        [0,1]
    """

    if resolution:
        prompt = (
            f"{prompt} | "
            f"resolution: {resolution}"
        )

    model = load_model()

    image = image.to(
        DEVICE,
        dtype=torch.float32,
    )

    if image.ndim == 3:
        image = image.unsqueeze(0)

    image = image.clamp(
        0.0,
        1.0,
    )

    video, used_seed = diffusion_sample(
        model=model,
        prompt=prompt,
        image=image,
        steps=steps,
        seed=seed,
        image_strength=image_strength,
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    if output is None:
        output = os.path.join(
            OUTPUT_DIR,
            f"image_video_seed_{used_seed}.mp4",
        )

    save_video(
        video,
        output,
        fps=VIDEO_FPS,
    )

    return output


# ------------------------------------------------------------
# Command line interface
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "TinyVideo V1.4 "
            "latent video diffusion generator."
        )
    )

    parser.add_argument(
        "prompt",
        help="Video prompt.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--resolution",
        choices=[
            "1080p",
            "720p",
            "420p",
        ],
        default=None,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    path = generate(
        prompt=args.prompt,
        output=args.output,
        seed=args.seed,
        steps=args.steps,
        resolution=args.resolution,
    )

    print()
    print(
        f"Video generated: {path}"
    )


if __name__ == "__main__":
    main()