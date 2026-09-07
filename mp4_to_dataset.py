"""Convert videos into TinyVideo training samples.

TinyVideo currently expects exactly:
    8 frames, 64x64 RGB, 5 FPS (1.6 seconds).

This converter accepts videos with different durations/FPS and automatically
maps them to TinyVideo's fixed format before saving them to the dataset.
"""

import argparse
import os
import re

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "tiny_dataset.pt")

WIDTH = 64
HEIGHT = 64
FRAMES = 8
TARGET_FPS = 5
TARGET_DURATION = FRAMES / TARGET_FPS  # 1.6 seconds
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def filename_to_prompt(path):
    """Turn a filename such as red_ball_moving_right.mp4 into a prompt."""
    name = os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r"[_-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_video_info(reader, frame_count):
    """Get FPS and duration, with a frame-count fallback."""
    meta = reader.get_meta_data()
    fps = meta.get("fps")

    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = 0.0

    if fps <= 0:
        fps = TARGET_FPS

    duration = meta.get("duration")
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0

    if duration <= 0 and frame_count > 0:
        duration = frame_count / fps

    return fps, duration


def load_video_frames(path):
    """Read all source frames and return them with the source FPS/duration."""
    reader = imageio.get_reader(path)
    try:
        frames = [frame for frame in reader]
        fps, duration = get_video_info(reader, len(frames))
    finally:
        reader.close()

    if not frames:
        raise ValueError("Video contains no frames.")

    # Some codecs report a duration that is slightly different from the
    # actual frame count. The frame count is safer when selecting frames.
    actual_duration = len(frames) / fps
    if actual_duration > 0:
        duration = actual_duration

    return frames, fps, duration


def select_tinyvideo_frames(frames, source_fps, source_duration):
    """Select 8 frames representing the source played at 1.6 seconds/5 FPS.

    A longer source is effectively sped up; a shorter source is effectively
    slowed down. The source is not re-encoded to disk -- we directly select
    the frames that the edited video would contain.
    """
    if source_duration <= 0:
        raise ValueError("Video duration could not be determined.")

    if abs(source_duration - TARGET_DURATION) > 0.01:
        print("Video doesnt fit requirements, editing...")
        if source_duration > TARGET_DURATION:
            print("speeding up")
        else:
            print("slowing down")
    else:
        print("Video fits requirements!")

    # The output frame timestamps are 0.0, 0.2, ..., 1.4 seconds.
    # Map those timestamps onto the complete source duration. This preserves
    # the whole source while fitting it into TinyVideo's fixed 1.6 seconds.
    output_times = np.arange(FRAMES, dtype=np.float64) / TARGET_FPS
    source_times = output_times / TARGET_DURATION * source_duration

    # Clamp the final timestamp so rounding can never go beyond the source.
    source_times = np.minimum(source_times, max(0.0, source_duration - 1.0 / source_fps))
    indices = np.rint(source_times * source_fps).astype(int)
    indices = np.clip(indices, 0, len(frames) - 1)

    return [frames[index] for index in indices]


def frames_to_tensor(frames):
    output = []

    for frame in frames:
        image = Image.fromarray(frame).convert("RGB")
        image = image.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)

        array = np.asarray(image)
        tensor = torch.from_numpy(array).float()
        tensor = tensor.permute(2, 0, 1) / 255.0
        output.append(tensor)

    return torch.stack(output)


def convert_file(path, prompt):
    print(f"Converting: {path}")
    print(f"Prompt:     {prompt}")

    frames, source_fps, source_duration = load_video_frames(path)
    print(f"Source:     {source_duration:.2f}s at {source_fps:g} FPS ({len(frames)} frames)")

    selected_frames = select_tinyvideo_frames(frames, source_fps, source_duration)
    print("changing the frames per second...")

    video = frames_to_tensor(selected_frames)

    print(f"Output:     {TARGET_DURATION:.1f}s at {TARGET_FPS} FPS")
    print(f"Tensor:     {tuple(video.shape)}")
    return prompt, video


def collect_files(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(input_path)

    files = []
    for name in sorted(os.listdir(input_path)):
        path = os.path.join(input_path, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
            files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(description="Convert videos for TinyVideo training.")
    parser.add_argument("input", help="Video file or folder containing videos")
    parser.add_argument("--prompt", help="Prompt for one input video")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the existing dataset instead of appending",
    )
    args = parser.parse_args()

    files = collect_files(args.input)
    if not files:
        raise SystemExit("No supported video files found.")

    if len(files) > 1 and args.prompt:
        raise SystemExit("--prompt can only be used when converting one video.")

    if args.replace or not os.path.exists(DATASET_PATH):
        dataset = []
    else:
        dataset = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
        if not isinstance(dataset, list):
            raise TypeError("Existing dataset must be a list of (prompt, video) samples.")

    for path in files:
        prompt = args.prompt if len(files) == 1 and args.prompt else filename_to_prompt(path)
        dataset.append(convert_file(path, prompt))

    os.makedirs(DATASET_DIR, exist_ok=True)
    torch.save(dataset, DATASET_PATH)

    print()
    print("DATASET UPDATED!")
    print(f"Samples: {len(dataset)}")
    print(f"File:    {DATASET_PATH}")


if __name__ == "__main__":
    main()
