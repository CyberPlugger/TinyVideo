import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from constants import *

TARGET_FRAMES = 8
TARGET_FPS = 5
TARGET_WIDTH = 64
TARGET_HEIGHT = 64
TARGET_DURATION = TARGET_FRAMES / TARGET_FPS

# Resolution conditioning used by the diffusion model.


def get_video_info(path):
    reader = imageio.get_reader(path)

    try:
        metadata = reader.get_meta_data()

        fps = float(metadata.get("fps", TARGET_FPS))

        try:
            frame_count = reader.count_frames()
        except Exception:
            frame_count = metadata.get("nframes", None)

        if frame_count is not None:
            frame_count = int(frame_count)

        duration = None

        if frame_count is not None and fps > 0:
            duration = frame_count / fps

        return fps, frame_count, duration

    finally:
        reader.close()


def sample_frames(path, frame_count, fps):
    """
    Sample exactly TARGET_FRAMES frames without loading
    the complete source video into memory.
    """

    reader = imageio.get_reader(path)

    try:
        if frame_count is not None and frame_count > 0:
            duration = frame_count / max(fps, 0.001)

            times = np.linspace(
                0,
                max(0.0, duration - 1.0 / max(fps, 0.001)),
                TARGET_FRAMES,
            )

            indices = np.clip(
                (times * fps).astype(int),
                0,
                frame_count - 1,
            )

            frames = [
                reader.get_data(int(index))
                for index in indices
            ]

        else:
            # Fallback when the codec does not expose frame count.
            frames = []

            for frame in reader:
                frames.append(frame)

            if not frames:
                raise ValueError("Video contains no frames.")

            indices = np.linspace(
                0,
                len(frames) - 1,
                TARGET_FRAMES,
            ).astype(int)

            frames = [frames[i] for i in indices]

    finally:
        reader.close()

    result = []

    for frame in frames:
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        frame = frame[:, :, :3]

        tensor = torch.from_numpy(
            frame.astype(np.float32)
        )

        tensor = tensor.permute(2, 0, 1)
        tensor = tensor / 255.0

        result.append(tensor)

    return torch.stack(result)


def resize_frames(frames, width, height):
    """
    Resize [T,3,H,W] frames to the requested dimensions.
    """

    return F.interpolate(
        frames,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )


def make_resolution_variant(frames, resolution):
    """
    Create a resolution-specific version of the sampled clip.

    The final training tensor is still 64x64 because the current
    TinyVideo V1.4 model operates in pixel space at 64x64.

    The resolution is preserved in the prompt so the diffusion
    model can learn resolution as a conditioning variable.
    """

    width, height = RESOLUTION_VARIANTS[resolution]

    high_res = resize_frames(
        frames,
        width,
        height,
    )

    training_res = resize_frames(
        high_res,
        TARGET_WIDTH,
        TARGET_HEIGHT,
    )

    return training_res


def prompt_from_filename(path):
    name = os.path.splitext(
        os.path.basename(path)
    )[0]

    name = name.replace("_", " ")
    name = name.replace("-", " ")

    return name.strip()


def find_videos(folder):
    extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
    }

    files = []

    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)

        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(name)[1].lower()

        if extension in extensions:
            files.append(path)

    return files


def process_video(video_path, output_path, custom_prompt=None):
    print()
    print(f"Processing: {video_path}")

    fps, frame_count, duration = get_video_info(video_path)

    print(f"  FPS: {fps:.2f}")

    if frame_count is not None:
        print(f"  Frames: {frame_count}")

    if duration is not None:
        print(f"  Duration: {duration:.2f}s")

    if (
        frame_count is not None
        and duration is not None
        and duration < TARGET_DURATION
    ):
        print("  Video doesnt fit requirements, editing...")

    if fps > TARGET_FPS:
        print("  changing the frames per second...")
    elif fps < TARGET_FPS:
        print("  changing the frames per second...")

    if duration is not None:
        if duration > TARGET_DURATION:
            print("  shortening video to training duration...")
        elif duration < TARGET_DURATION:
            print("  video is shorter than target duration; sampling available frames...")

    frames = sample_frames(
        video_path,
        frame_count,
        fps,
    )

    base_prompt = (
        custom_prompt
        if custom_prompt
        else prompt_from_filename(video_path)
    )

    if not base_prompt:
        base_prompt = "a video"

    dataset = []

    for resolution in RESOLUTION_VARIANTS:
        print(f"  Creating {resolution} training sample...")

        variant = make_resolution_variant(
            frames,
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

    # Load an existing dataset if present.
    if os.path.exists(output_path):
        try:
            existing = torch.load(
                output_path,
                map_location="cpu",
            )

            if not isinstance(existing, list):
                existing = []

        except Exception:
            print("  Existing dataset could not be read. Starting a new dataset.")
            existing = []
    else:
        existing = []

    existing.extend(dataset)

    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)),
        exist_ok=True,
    )

    torch.save(
        existing,
        output_path,
    )

    print(
        f"  Added {len(dataset)} samples."
    )

    print(
        f"  Dataset now contains {len(existing)} samples."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert videos into TinyVideo V1.4 training data."
    )

    parser.add_argument(
        "input",
        help="Video file or folder containing videos.",
    )

    parser.add_argument(
        "--output",
        default="training_data/video_dataset.pt",
        help="Output dataset path.",
    )

    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt to use instead of the filename.",
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the existing dataset instead of appending.",
    )

    args = parser.parse_args()

    if os.path.isdir(args.input):
        videos = find_videos(args.input)
    elif os.path.isfile(args.input):
        videos = [args.input]
    else:
        raise FileNotFoundError(
            f"Input does not exist: {args.input}"
        )

    if not videos:
        print("No supported video files found.")
        return

    if args.replace and os.path.exists(args.output):
        os.remove(args.output)
        print(f"Removed existing dataset: {args.output}")

    successful = 0
    failed = 0

    for video_path in videos:
        try:
            process_video(
                video_path,
                args.output,
                args.prompt,
            )

            successful += 1

        except Exception as error:
            failed += 1

            print(
                f"  ERROR: {error}"
            )

            print(
                "  Skipping this video and continuing..."
            )

    print()
    print("================================")
    print("TinyVideo dataset conversion done")
    print("================================")
    print(f"Successful videos: {successful}")
    print(f"Failed videos:     {failed}")
    print(f"Dataset:           {args.output}")


if __name__ == "__main__":
    main()