import os

import imageio.v2 as imageio
import numpy as np
import torch


def _to_numpy_video(video):
    """
    Convert a TinyVideo tensor into:

        [T, H, W, C]

    with uint8 RGB frames.
    """

    if isinstance(video, torch.Tensor):
        video = video.detach().cpu()

    if isinstance(video, np.ndarray):
        array = video
    else:
        array = np.asarray(video)

    if array.ndim == 5:
        # Accept:
        #
        # [B, C, T, H, W]
        #
        # or
        #
        # [B, T, C, H, W]
        #
        # Only one video is supported here.
        if array.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1, got {array.shape}."
            )

        array = array[0]

    if array.ndim != 4:
        raise ValueError(
            "Video must be a 4D or 5D tensor/array. "
            f"Got shape {array.shape}."
        )

    # --------------------------------------------------------
    # Detect channel position.
    # --------------------------------------------------------

    if array.shape[0] in (1, 3, 4):
        # [C, T, H, W]
        array = np.transpose(
            array,
            (1, 2, 3, 0),
        )

    elif array.shape[1] in (1, 3, 4):
        # [T, C, H, W]
        array = np.transpose(
            array,
            (0, 2, 3, 1),
        )

    elif array.shape[-1] in (1, 3, 4):
        # Already [T, H, W, C]
        pass

    else:
        raise ValueError(
            "Could not determine RGB channel dimension "
            f"from video shape {array.shape}."
        )

    # --------------------------------------------------------
    # Convert floating point ranges.
    # --------------------------------------------------------

    array = array.astype(np.float32)

    if array.min() < 0.0:
        # [-1, 1] -> [0, 1]
        array = (array + 1.0) / 2.0

    array = np.clip(
        array,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Ensure exactly RGB.
    # --------------------------------------------------------

    if array.shape[-1] == 1:
        array = np.repeat(
            array,
            3,
            axis=-1,
        )

    elif array.shape[-1] == 4:
        # Drop alpha for ffmpeg RGB output.
        array = array[..., :3]

    elif array.shape[-1] != 3:
        raise ValueError(
            f"Expected 1, 3, or 4 channels. "
            f"Got {array.shape[-1]}."
        )

    # --------------------------------------------------------
    # [0,1] -> uint8
    # --------------------------------------------------------

    array = (
        array * 255.0
    ).round().astype(np.uint8)

    return array


def save_video(
    video,
    path,
    fps=5,
):
    """
    Save a TinyVideo tensor as an MP4.

    Accepted formats include:

        [T, C, H, W]
        [C, T, H, W]
        [1, C, T, H, W]
        [1, T, C, H, W]

    Output format:

        MP4 / RGB
    """

    if path is None:
        raise ValueError(
            "Output path cannot be None."
        )

    path = os.fspath(path)

    directory = os.path.dirname(
        os.path.abspath(path)
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    frames = _to_numpy_video(video)

    if frames.shape[0] == 0:
        raise ValueError(
            "Cannot save a video with zero frames."
        )

    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
    )

    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    return path