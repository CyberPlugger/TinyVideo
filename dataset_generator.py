import os
import random

import torch
from PIL import Image, ImageDraw


# ============================================================
# Пути всегда относительно этого файла
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

DATASET_PATH = os.path.join(
    DATASET_DIR,
    "tiny_dataset.pt"
)


# ============================================================
# Настройки
# ============================================================

WIDTH = 64
HEIGHT = 64
FRAMES = 8


COLORS = {
    "red": (255, 40, 40),
    "blue": (40, 80, 255),
    "green": (40, 255, 80),
    "yellow": (255, 230, 40)
}


DIRECTIONS = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1)
}


# ============================================================
# Создание одного видео
# ============================================================

def create_video(color_name, direction_name):

    color = COLORS[color_name]

    dx, dy = DIRECTIONS[direction_name]

    x = random.randint(
        10,
        WIDTH - 22
    )

    y = random.randint(
        10,
        HEIGHT - 22
    )

    frames = []

    for _ in range(FRAMES):

        image = Image.new(
            "RGB",
            (WIDTH, HEIGHT),
            (10, 10, 10)
        )

        draw = ImageDraw.Draw(image)

        draw.ellipse(
            (
                x,
                y,
                x + 12,
                y + 12
            ),
            fill=color
        )

        # PIL → Tensor
        frame = torch.tensor(
            list(image.getdata()),
            dtype=torch.float32
        )

        frame = frame.reshape(
            HEIGHT,
            WIDTH,
            3
        )

        # HWC → CHW
        frame = frame.permute(
            2,
            0,
            1
        )

        frame /= 255.0

        frames.append(frame)

        x += dx * 5
        y += dy * 5

    return torch.stack(frames)


# ============================================================
# Создание датасета
# ============================================================

def main():

    print()
    print("=" * 50)
    print("        TinyVideo Dataset Generator")
    print("=" * 50)

    print()
    print("TinyVideo directory:")
    print(BASE_DIR)

    print()
    print("Dataset directory:")
    print(DATASET_DIR)

    print()

    os.makedirs(
        DATASET_DIR,
        exist_ok=True
    )

    samples = []

    total = (
        len(COLORS)
        * len(DIRECTIONS)
        * 100
    )

    counter = 0

    for color in COLORS:

        for direction in DIRECTIONS:

            for _ in range(100):

                video = create_video(
                    color,
                    direction
                )

                prompt = (
                    f"{color} ball moving "
                    f"{direction}"
                )

                samples.append(
                    (
                        prompt,
                        video
                    )
                )

                counter += 1

                if counter % 100 == 0:

                    print(
                        f"Generated "
                        f"{counter}/{total}"
                    )

    print()
    print("Saving dataset...")

    torch.save(
        samples,
        DATASET_PATH
    )

    print()
    print("=" * 50)
    print("DATASET CREATED!")
    print("=" * 50)

    print()
    print("Samples:", len(samples))

    print()
    print("File:")
    print(DATASET_PATH)

    print()


if __name__ == "__main__":
    main()
