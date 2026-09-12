import argparse
import os

import torch
from PIL import Image

from generate import generate_with_image
from constants import *


def load_image(image_path):
    """
    Load an image and convert it to the tensor format expected
    by TinyVideo V1.4:

        [1, 3, H, W]

    Values are normalized to [0, 1].
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
        ),
        Image.Resampling.LANCZOS,
    )

    image_tensor = torch.from_numpy(
        __import__("numpy").array(
            image
        )
    ).float()

    image_tensor = (
        image_tensor / 255.0
    )

    # [H,W,C] -> [C,H,W]
    image_tensor = image_tensor.permute(
        2,
        0,
        1,
    )

    # [C,H,W] -> [1,C,H,W]
    image_tensor = image_tensor.unsqueeze(
        0
    )

    return image_tensor


def generate_video_from_image(
    image_path,
    prompt,
    output=None,
    seed=None,
    steps=30,
    resolution=None,
    image_strength=0.85,
):
    """
    Image + text prompt -> video.
    """

    image = load_image(
        image_path
    )

    path = generate_with_image(
        prompt=prompt,
        image=image,
        output=output,
        seed=seed,
        steps=steps,
        resolution=resolution,
        image_strength=image_strength,
    )

    return path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "TinyVideo V1.4 "
            "image-to-video generator."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the input image.",
    )

    parser.add_argument(
        "prompt",
        help=(
            "Describe what should happen "
            "in the generated video."
        ),
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
        "--strength",
        type=float,
        default=0.85,
        help=(
            "How strongly the first generated "
            "frame follows the input image. "
            "0.0-1.0."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    if not 0.0 <= args.strength <= 1.0:
        raise ValueError(
            "--strength must be between "
            "0.0 and 1.0."
        )

    output = (
        generate_video_from_image(
            image_path=args.image,
            prompt=args.prompt,
            output=args.output,
            seed=args.seed,
            steps=args.steps,
            resolution=args.resolution,
            image_strength=args.strength,
        )
    )

    print()
    print(
        "Image-to-video generation complete."
    )
    print(
        f"Video: {output}"
    )


if __name__ == "__main__":
    main()