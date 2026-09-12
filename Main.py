print('THIS IS YOUR TERMINAL. Do not use other terminals')

import threading
import tkinter as tk
import os
import subprocess as sb
import sys

from constants import (
    PROJECT_ROOT
)

root = tk.Tk()
root.title("Main Navigation")
root.config(bg="black")


def open(app):
    def main():
        if app == 'train':
            path = os.path.join(
                PROJECT_ROOT,
                'tktrain.py'
            )
            sb.run([sys.executable, path])

        if app == 'genimg':
            path = os.path.join(
                PROJECT_ROOT,
                'tkgenimg.py'
            )
            sb.run([sys.executable, path])

        if app == 'gen':
            path = os.path.join(
                PROJECT_ROOT,
                'tkgen.py'
            )
            sb.run([sys.executable, path])

        if app == 'gendata':
            path = os.path.join(
                PROJECT_ROOT,
                'tkgendata.py'
            )
            sb.run([sys.executable, path])
    threading.Thread(target=main, daemon=True).start()


title = tk.Label(
    root,
    text="Main Navigation",
    font=("Courier", 40),
    bg="black",
    fg="green"
)
title.pack()


# ---------------------------------------------------------
# INFO AREA
# ---------------------------------------------------------

info_frame = tk.Frame(
    root,
    bg="black"
)
info_frame.pack(
    fill=tk.BOTH,
    expand=True,
    side=tk.LEFT
)


# Scrollbar
info_scrollbar = tk.Scrollbar(
    info_frame,
    bg="black"
)
info_scrollbar.pack(
    side=tk.RIGHT,
    fill=tk.Y
)


# Scrollable text widget
info_text = tk.Text(
    info_frame,
    bg="black",
    fg="green",
    font=("Courier", 8),
    wrap=tk.NONE,
    yscrollcommand=info_scrollbar.set,
    borderwidth=0,
    highlightthickness=0,
    state=tk.DISABLED
)

info_text.pack(
    side=tk.LEFT,
    fill=tk.BOTH,
    expand=True
)


info_scrollbar.config(
    command=info_text.yview
)


# ---------------------------------------------------------
# INFO CONTENT
# ---------------------------------------------------------

info = '''
VIDEO_FPS: 5
VIDEO_FRAMES: 8
VIDEO_HEIGHT: 64
VIDEO_WIDTH: 64

RESOLUTION_IDS: {
    "unknown": 0,
    "1080p": 1,
    "720p": 2,
    "420p": 3,
}

TEXT_DIM: 128
IMAGE_DIM: 128
CONDITION_DIM: 192
TIME_EMBED_DIM: 128
DIFFUSION_STEPS: 1000
BETA_START: 0.0001
BETA_END: 0.02

BASE_CHANNELS: 48

DATASET_PATH: (
    PROJECT_ROOT
    / "training_data"
    / "generated_dataset.pt"
)

SYNTHETIC_DATASET_PATH: (
    PROJECT_ROOT
    / "training_data"
    / "generated_dataset.pt"
)

CHECKPOINT_DIR: (
    PROJECT_ROOT
    / "checkpoints"
)

CHECKPOINT_PATH: (
    CHECKPOINT_DIR
    / "tinyvideo_v14.pt"
)

OUTPUT_DIR: (
    PROJECT_ROOT
    / "outputs"
)

SERVER_HOST: "127.0.0.1"
SERVER_PORT: 5000

DEFAULT_BATCH_SIZE: 1
DEFAULT_EPOCHS: 1
DEFAULT_LEARNING_RATE: 0.0001

DEFAULT_GENERATION_STEPS: 20
DEFAULT_SEED: None

COLORS: {
    "red": (220, 40, 40),
    "green": (40, 200, 80),
    "blue": (50, 100, 230),
    "yellow": (230, 210, 40),
    "purple": (160, 70, 210),
    "orange": (240, 120, 30),
}

DIRECTIONS: {
    "right": (1, 0),
    "left": (-1, 0),
    "down": (0, 1),
    "up": (0, -1),
}

DATASET_FOLDER_PATH: (
    PROJECT_ROOT
    / "training_data"
    / "generated_dataset.pt"
)
'''


# Insert text
info_text.config(state=tk.NORMAL)
info_text.insert("1.0", info)
info_text.config(state=tk.DISABLED)


# ---------------------------------------------------------
# BOTTOM BUTTON AREA
# ---------------------------------------------------------

choices_frame = tk.Frame(
    root,
    bg="green",
    height=50,
    width=100
)

choices_frame.pack(
    side=tk.BOTTOM,
    fill=tk.BOTH,
    expand=True
)


# Generate Video With Image
button_border = tk.Frame(
    choices_frame,
    bg="#D3D3D3",
    padx=2,
    pady=2
)
button_border.pack()

tk.Button(
    button_border,
    text='Generate Video With Image',
    command=lambda: open('genimg'),
    bg='black',
    fg='white',
    font=("Courier", 10),
    relief=tk.FLAT,
    borderwidth=0
).pack()


# Generate Video From Scratch
button_border = tk.Frame(
    choices_frame,
    bg="#D3D3D3",
    padx=2,
    pady=2
)
button_border.pack(padx=5, pady=3)

tk.Button(
    button_border,
    text='Generate Video From Scratch',
    command=lambda: open('gen'),
    bg='black',
    fg='white',
    activebackground='black',
    activeforeground='white',
    font=("Courier", 10),
    relief='flat',
    bd=0,
    highlightthickness=0
).pack()


# Train Model
button_border = tk.Frame(
    choices_frame,
    bg="#D3D3D3",
    padx=2,
    pady=2
)
button_border.pack(padx=5, pady=3)

tk.Button(
    button_border,
    text='Train Model',
    command=lambda: open('train'),
    bg='black',
    fg='white',
    activebackground='black',
    activeforeground='white',
    font=("Courier", 10),
    relief='flat',
    bd=0,
    highlightthickness=0
).pack()


# Generate new dataset
button_border = tk.Frame(
    choices_frame,
    bg="#D3D3D3",
    padx=2,
    pady=2
)
button_border.pack(padx=5, pady=3)

tk.Button(
    button_border,
    text='Generate new dataset',
    command=lambda: open('gendata'),
    bg='black',
    fg='white',
    activebackground='black',
    activeforeground='white',
    font=("Courier", 10),
    relief='flat',
    bd=0,
    highlightthickness=0
).pack()

tk.Button(
    button_border,
    text='Exit',
    command=root.destroy,
    bg='red',
    fg='white',
    activebackground='black',
    activeforeground='white',
    font=("Courier", 10),
    relief='flat',
    bd=0,
    highlightthickness=0
).pack()

if __name__ == '__main__':
    raise Exception('Open this script using "ACTIVATE_ME.py", not the main file.')

root.mainloop()