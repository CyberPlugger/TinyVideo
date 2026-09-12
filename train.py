import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from constants import (
    DATASET_PATH,
    CHECKPOINT_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
)

from model import TinyVideoDiffusion

from text_encoder import (
    tokenize_text,
    encode_text,
    extract_resolution_id,
)


# ============================================================
# TinyVideo V1.4 - Latent Diffusion Training
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Helpers
# ============================================================

def is_video_tensor(value):
    """Return True if value looks like a video tensor."""
    if torch.is_tensor(value):
        return value.ndim == 4

    if isinstance(value, (list, tuple)):
        try:
            tensor = torch.as_tensor(value)
            return tensor.ndim == 4
        except Exception:
            return False

    return False


def find_video_in_sample(sample):
    """
    Search a dataset sample for the actual video tensor.

    This avoids assuming that the field called 'video' is
    necessarily the tensor.
    """

    if torch.is_tensor(sample):
        if sample.ndim == 4:
            return sample

    if isinstance(sample, (list, tuple)):
        # First search every element for a video.
        for item in sample:
            if is_video_tensor(item):
                return item

        return None

    if isinstance(sample, dict):

        # Preferred names.
        preferred_keys = (
            "video",
            "frames",
            "clip",
            "tensor",
            "video_tensor",
            "frames_tensor",
            "data",
        )

        for key in preferred_keys:
            if key not in sample:
                continue

            value = sample[key]

            if is_video_tensor(value):
                return value

        # If the normal names failed, inspect every value.
        for key, value in sample.items():

            if key in (
                "prompt",
                "text",
                "filename",
                "file",
                "path",
                "video_path",
                "resolution",
                "resolution_id",
            ):
                continue

            if is_video_tensor(value):
                return value

    return None


def find_prompt(sample):
    """Extract prompt/text from a dataset sample."""

    if isinstance(sample, dict):

        prompt = sample.get("prompt")

        if prompt is None or str(prompt).strip() == "":
            prompt = sample.get("text", "")

        return str(prompt)

    if isinstance(sample, (list, tuple)):

        for item in sample:

            if isinstance(item, str):
                # Avoid treating obvious filenames as prompts.
                lowered = item.lower()

                if lowered.endswith(
                    (
                        ".mp4",
                        ".avi",
                        ".mov",
                        ".mkv",
                        ".webm",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    )
                ):
                    continue

                return item

    return ""


def find_resolution(sample, prompt):
    """Extract resolution information."""

    if isinstance(sample, dict):

        resolution = sample.get("resolution")

        if resolution is not None:
            return resolution

        resolution_id = sample.get("resolution_id")

        if resolution_id is not None:
            return resolution_id

    return extract_resolution_id(prompt)


def normalize_video(video):
    """
    Convert video to [C,T,H,W] float32 in [0,1].
    """

    video = torch.as_tensor(
        video,
        dtype=torch.float32,
    )

    if video.ndim != 4:
        raise ValueError(
            f"Expected 4D video tensor, got "
            f"shape {tuple(video.shape)}"
        )

    # Common format:
    # [T,C,H,W]
    if video.shape[0] == 8:

        video = video.permute(
            1,
            0,
            2,
            3,
        )

    # Already:
    # [C,T,H,W]
    elif video.shape[1] == 8:

        pass

    # Sometimes datasets use [H,W,C,T].
    elif video.shape[-1] == 8:

        video = video.permute(
            2,
            3,
            0,
            1,
        )

    else:

        raise ValueError(
            "Could not determine frame dimension "
            f"for video shape {tuple(video.shape)}"
        )

    # Ensure channel dimension is RGB.
    if video.shape[0] != 3:

        raise ValueError(
            "Expected RGB video with 3 channels. "
            f"Got shape {tuple(video.shape)}"
        )

    # Normalize 0-255 videos.
    if video.numel() > 0:

        max_value = float(video.max())

        if max_value > 1.0:
            video = video / 255.0

    return video.clamp(
        0.0,
        1.0,
    )


# ============================================================
# Dataset
# ============================================================

class VideoDataset(Dataset):

    def __init__(self, path):

        print(f"Loading dataset: {path}")

        self.data = torch.load(
            path,
            map_location="cpu",
        )

        if isinstance(self.data, dict):

            if "samples" in self.data:

                self.samples = self.data["samples"]

            elif "data" in self.data:

                self.samples = self.data["data"]

            else:

                # Some datasets are themselves a dictionary
                # containing indexed samples.
                values = list(self.data.values())

                if values and any(
                    is_video_tensor(v)
                    for v in values
                ):
                    self.samples = values

                else:

                    raise ValueError(
                        "Dataset dictionary does not contain "
                        "'samples' or 'data'. "
                        f"Available keys: "
                        f"{list(self.data.keys())}"
                    )

        elif isinstance(self.data, (list, tuple)):

            self.samples = self.data

        else:

            raise ValueError(
                f"Unsupported dataset format: "
                f"{type(self.data)}"
            )

        print(
            f"Loaded {len(self.samples)} dataset samples."
        )

        if len(self.samples) == 0:

            raise ValueError(
                "Dataset contains zero samples."
            )

        # --------------------------------------------------------
        # Inspect first sample.
        # --------------------------------------------------------

        first = self.samples[0]

        print()
        print("Dataset sample inspection:")

        if isinstance(first, dict):

            print(
                "  type: dict"
            )

            print(
                "  keys:",
                list(first.keys())
            )

            for key, value in first.items():

                if torch.is_tensor(value):

                    print(
                        f"  {key}: tensor "
                        f"{tuple(value.shape)}"
                    )

                else:

                    print(
                        f"  {key}: {type(value).__name__}"
                    )

        else:

            print(
                f"  type: {type(first).__name__}"
            )

            if torch.is_tensor(first):

                print(
                    f"  shape: {tuple(first.shape)}"
                )

        print()

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        # --------------------------------------------------------
        # Find video.
        # --------------------------------------------------------

        video = find_video_in_sample(
            sample
        )

        if video is None:

            if isinstance(sample, dict):

                raise ValueError(
                    f"Sample {index} does not contain "
                    f"a 4D video tensor. "
                    f"Available keys: {list(sample.keys())}"
                )

            raise ValueError(
                f"Sample {index} does not contain "
                f"a 4D video tensor."
            )

        # --------------------------------------------------------
        # Prompt.
        # --------------------------------------------------------

        prompt = find_prompt(
            sample
        )

        # --------------------------------------------------------
        # Resolution.
        # --------------------------------------------------------

        resolution = find_resolution(
            sample,
            prompt,
        )

        # Convert resolution to ID.
        if isinstance(resolution, int):

            resolution_id = resolution

        elif isinstance(resolution, float):

            resolution_id = int(
                resolution
            )

        else:

            resolution_id = extract_resolution_id(
                str(resolution)
            )

        # --------------------------------------------------------
        # Normalize video.
        # --------------------------------------------------------

        video = normalize_video(
            video
        )

        # --------------------------------------------------------
        # Text.
        # --------------------------------------------------------

        text_embedding = encode_text(
            prompt
        )

        token_ids = tokenize_text(
            prompt
        )

        return {
            "video": video,
            "prompt": prompt,
            "text_embedding": text_embedding,
            "token_ids": token_ids,
            "resolution_id": resolution_id,
        }


# ============================================================
# Collate
# ============================================================

def collate_batch(batch):

    videos = torch.stack(
        [
            item["video"]
            for item in batch
        ]
    )

    text_embeddings = torch.stack(
        [
            item["text_embedding"]
            for item in batch
        ]
    )

    token_ids = torch.stack(
        [
            item["token_ids"]
            for item in batch
        ]
    )

    resolution_ids = torch.tensor(
        [
            item["resolution_id"]
            for item in batch
        ],
        dtype=torch.long,
    )

    prompts = [
        item["prompt"]
        for item in batch
    ]

    return {
        "video": videos,
        "text_embedding": text_embeddings,
        "token_ids": token_ids,
        "resolution_ids": resolution_ids,
        "prompts": prompts,
    }


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path,
):

    os.makedirs(
        os.path.dirname(path)
        or ".",
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch,

            "loss":
                loss,

            "version":
                "1.4-latent-diffusion",
        },
        path,
    )

    print(
        f"Checkpoint saved: {path}"
    )


# ============================================================
# Training
# ============================================================

def train(
    dataset_path=DATASET_PATH,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    checkpoint_path=CHECKPOINT_PATH,
):

    print()
    print("=" * 40)
    print(" TinyVideo V1.4 Latent Diffusion Training")
    print("=" * 40)
    print()

    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = VideoDataset(
        dataset_path
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_batch,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TinyVideoDiffusion().to(
        DEVICE
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    global_step = 0

    for epoch in range(epochs):

        print()
        print(
            f"========== Epoch "
            f"{epoch + 1}/{epochs} =========="
        )

        model.train()

        epoch_loss = 0.0

        epoch_start = time.time()

        for batch_index, batch in enumerate(
            loader
        ):

            step_start = time.time()

            # ------------------------------------------------
            # Load batch
            # ------------------------------------------------

            target = batch["video"].to(
                DEVICE,
                non_blocking=False,
            )

            text_embedding = (
                batch["text_embedding"]
                .to(
                    DEVICE,
                    non_blocking=False,
                )
            )

            resolution_ids = (
                batch["resolution_ids"]
                .to(
                    DEVICE,
                    non_blocking=False,
                )
            )

            # ------------------------------------------------
            # Encode video
            # ------------------------------------------------

            latent = model.encode_video(
                target
            )

            # ------------------------------------------------
            # Diffusion timestep
            # ------------------------------------------------

            timestep = torch.randint(
                0,
                model.alpha_bars.shape[0],
                (
                    target.shape[0],
                ),
                device=DEVICE,
            )

            # ------------------------------------------------
            # Noise
            # ------------------------------------------------

            noise = torch.randn_like(
                latent
            )

            # ------------------------------------------------
            # Add noise
            # ------------------------------------------------

            noisy_latent = model.add_noise(
                latent,
                noise,
                timestep,
            )

            # ------------------------------------------------
            # Predict noise
            # ------------------------------------------------

            prediction = model(
                noisy_latent,
                text_embedding,
                timestep,
                image=None,
                resolution_ids=resolution_ids,
            )

            # ------------------------------------------------
            # Diffusion loss
            # ------------------------------------------------

            noise_loss = F.mse_loss(
                prediction,
                noise,
            )

            # ------------------------------------------------
            # Reconstruction loss
            # ------------------------------------------------

            reconstructed = model.decode_video(
                latent
            )

            reconstruction_loss = F.mse_loss(
                reconstructed,
                target,
            )

            # ------------------------------------------------
            # Total loss
            # ------------------------------------------------

            loss = (
                noise_loss
                +
                0.1 * reconstruction_loss
            )

            # ------------------------------------------------
            # Backprop
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            loss_value = float(
                loss.detach().cpu()
            )

            noise_value = float(
                noise_loss.detach().cpu()
            )

            reconstruction_value = float(
                reconstruction_loss.detach().cpu()
            )

            epoch_loss += loss_value

            global_step += 1

            elapsed = (
                time.time()
                - step_start
            )

            print(
                f"[{batch_index + 1}/"
                f"{len(loader)}] "
                f"loss={loss_value:.6f} "
                f"noise={noise_value:.6f} "
                f"recon={reconstruction_value:.6f} "
                f"time={elapsed:.1f}s"
            )

        # ----------------------------------------------------
        # Epoch statistics
        # ----------------------------------------------------

        average_loss = (
            epoch_loss
            / max(len(loader), 1)
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print()
        print(
            f"Epoch {epoch + 1} complete"
        )

        print(
            f"Average loss: "
            f"{average_loss:.6f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=average_loss,
            path=checkpoint_path,
        )

    print()
    print("=" * 40)
    print(" Training complete")
    print("=" * 40)
    print()


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "TinyVideo V1.4 latent diffusion training"
        )
    )

    parser.add_argument(
        "--dataset",
        default=DATASET_PATH,
        help="Path to dataset",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate",
    )

    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_PATH,
        help="Checkpoint output path",
    )

    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()