import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from constants import *

OUTPUT_PATH = DATASET_FOLDER_PATH

def make_frame(
    color,
    x,
    y,
    radius=8,
):
    image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        (15, 15, 15),
    )

    draw = ImageDraw.Draw(image)

    draw.ellipse(
        (
            x - radius,
            y - radius,
            x + radius,
            y + radius,
        ),
        fill=color,
    )

    return image


def create_video(
    color_name,
    direction_name,
    seed,
):
    random.seed(seed)

    color = COLORS[color_name]

    dx, dy = DIRECTIONS[direction_name]

    radius = random.randint(5, 9)

    # Start the object somewhere that gives it room to move.
    if dx > 0:
        start_x = random.randint(10, 25)
        start_y = random.randint(15, HEIGHT - 15)

    elif dx < 0:
        start_x = random.randint(WIDTH - 25, WIDTH - 10)
        start_y = random.randint(15, HEIGHT - 15)

    elif dy > 0:
        start_x = random.randint(15, WIDTH - 15)
        start_y = random.randint(10, 25)

    else:
        start_x = random.randint(15, WIDTH - 15)
        start_y = random.randint(HEIGHT - 25, HEIGHT - 10)

    speed = random.randint(3, 6)

    frames = []

    for frame_index in range(FRAMES):
        x = start_x + dx * speed * frame_index
        y = start_y + dy * speed * frame_index

        x = max(radius, min(WIDTH - radius, x))
        y = max(radius, min(HEIGHT - radius, y))

        frame = make_frame(
            color,
            x,
            y,
            radius,
        )

        array = np.asarray(frame).astype(
            np.float32
        ) / 255.0

        tensor = torch.from_numpy(array)
        tensor = tensor.permute(2, 0, 1)

        frames.append(tensor)

    return torch.stack(frames)


def create_resolution_variant(
    video,
    resolution,
):
    """
    Simulate a resolution-specific training path.

    The final model input remains 64x64, but the image is first
    resized through the selected resolution before returning to
    the training resolution.
    """

    width, height = RESOLUTION_VARIANTS[resolution]

    video = F.interpolate(
        video,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    video = F.interpolate(
        video,
        size=(HEIGHT, WIDTH),
        mode="bilinear",
        align_corners=False,
    )

    return video


def generate_dataset(
    samples_per_combination=100,
    output_path=OUTPUT_PATH,
):
    dataset = []

    seed = 0

    for color_name in COLORS:
        for direction_name in DIRECTIONS:

            print(
                f"Generating {color_name} object moving "
                f"{direction_name}..."
            )

            for _ in range(samples_per_combination):

                video = create_video(
                    color_name,
                    direction_name,
                    seed,
                )

                seed += 1

                base_prompt = (
                    f"a {color_name} ball moving "
                    f"{direction_name}"
                )

                # Create all three resolution-conditioned samples.
                for resolution in RESOLUTION_VARIANTS:

                    variant = create_resolution_variant(
                        video,
                        resolution,
                    )

                    prompt = (
                        f"{base_prompt}, "
                        f"resolution {resolution}"
                    )

                    dataset.append(
                        (
                            prompt,
                            variant,
                        )
                    )

    os.makedirs(
        os.path.dirname(
            os.path.abspath(output_path)
        ),
        exist_ok=True,
    )

    torch.save(
        dataset,
        output_path,
    )

    print()
    print("================================")
    print("TinyVideo synthetic dataset ready")
    print("================================")
    print(f"Samples: {len(dataset)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    generate_dataset()