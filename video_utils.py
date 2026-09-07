import os

import imageio.v2 as imageio
import numpy as np


def save_video(
    video,
    path,
    fps=5
):

    # --------------------------------------------------------
    # Создаём папку
    # --------------------------------------------------------

    directory = os.path.dirname(
        os.path.abspath(path)
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Tensor → NumPy
    # --------------------------------------------------------

    video = video.detach().cpu()

    # Наша модель:
    #
    # [frames, channels, height, width]
    #
    # Например:
    #
    # [8, 3, 64, 64]

    # Перемещаем каналы:
    #
    # [frames, channels, height, width]
    #
    # →
    #
    # [frames, height, width, channels]

    video = video.permute(
        0,
        2,
        3,
        1
    )

    video = video.numpy()

    # --------------------------------------------------------
    # 0..1 → 0..255
    # --------------------------------------------------------

    video = np.clip(
        video,
        0.0,
        1.0
    )

    video = (
        video * 255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Сохраняем MP4
    # --------------------------------------------------------

    print(
        f"Saving {len(video)} frames..."
    )

    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        format="FFMPEG"
    )

    try:

        for frame in video:

            writer.append_data(
                frame
            )

    finally:

        writer.close()

    print(
        "Video saved:"
    )

    print(
        os.path.abspath(path)
    )
