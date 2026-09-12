from pathlib import Path


# ============================================================
# TinyVideo V1.4 - Shared Constants
# ============================================================


# ============================================================
# Project Root
# ============================================================

# The folder containing this constants.py file.
#
# This means paths work correctly regardless of whether
# TinyVideo is launched from PyCharm, PowerShell, CMD, etc.

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# Video
# ============================================================

VIDEO_FRAMES = 8
VIDEO_HEIGHT = 64
VIDEO_WIDTH = 64
VIDEO_FPS = 5

# Short aliases
FRAMES = VIDEO_FRAMES
HEIGHT = VIDEO_HEIGHT
WIDTH = VIDEO_WIDTH
FPS = VIDEO_FPS


# ============================================================
# Resolution conditioning
# ============================================================

RESOLUTION_IDS = {
    "unknown": 0,
    "1080p": 1,
    "720p": 2,
    "420p": 3,
}


RESOLUTION_SIZES = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "420p": (746, 420),
}


# Alias used by dataset/generation code
RESOLUTION_VARIANTS = RESOLUTION_SIZES


# ============================================================
# Text conditioning
# ============================================================

TEXT_DIM = 128

# ============================================================
# Image conditioning
# ============================================================

IMAGE_DIM = 128


# ============================================================
# Diffusion conditioning
# ============================================================

CONDITION_DIM = 192
TIME_EMBED_DIM = 128

DIFFUSION_STEPS = 1000

BETA_START = 0.0001
BETA_END = 0.02


# ============================================================
# Model
# ============================================================

BASE_CHANNELS = 48


# ============================================================
# Dataset
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "training_data"
    / "generated_dataset.pt"
)


SYNTHETIC_DATASET_PATH = (
    PROJECT_ROOT
    / "training_data"
    / "generated_dataset.pt"
)


# ============================================================
# Checkpoints
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
)


CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "tinyvideo_v14.pt"
)


# ============================================================
# Generated videos
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)


# ============================================================
# Server
# ============================================================

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000


# ============================================================
# Training defaults
# ============================================================

DEFAULT_BATCH_SIZE = 1
DEFAULT_EPOCHS = 1
DEFAULT_LEARNING_RATE = 0.0001


# ============================================================
# Generation defaults
# ============================================================

DEFAULT_GENERATION_STEPS = 20
DEFAULT_SEED = None


# ============================================================
# For Dataset Information
# ============================================================

COLORS = {
    "red": (220, 40, 40),
    "green": (40, 200, 80),
    "blue": (50, 100, 230),
    "yellow": (230, 210, 40),
    "purple": (160, 70, 210),
    "orange": (240, 120, 30),
}


DIRECTIONS = {
    "right": (1, 0),
    "left": (-1, 0),
    "down": (0, 1),
    "up": (0, -1),
}


# ============================================================
# Dataset Folder Path
# ============================================================

DATASET_FOLDER_PATH = (
    PROJECT_ROOT
    / "training_data"
    / "generated_dataset.pt"
)

