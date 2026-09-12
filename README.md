# TinyVideo v1.4 (BIGGEST update so far)

> A small, local-first AI video generation system built from the ground up in Python.

TinyVideo is an experimental AI video generator designed to run locally and give the user control over the entire generation pipeline.

Instead of relying on a large external video-generation service, TinyVideo is built as a modular system containing its own dataset pipeline, text conditioning, video model, diffusion process, image conditioning, training system, and generation tools.

The project is intentionally small and experimental, making it easier to understand, modify, train, and extend.

---

# ✨ Features

## 🎬 Text-to-Video

Generate short videos from a text prompt using the TinyVideo diffusion model.

'text
Prompt
↓
Text Encoder
↓
Conditioning
↓
Diffusion Model
↓
Video Decoder
↓
MP4
'

## 🖼️ Image + Text-to-Video

TinyVideo can use an image together with a text prompt to condition video generation.

'text
Input Image + Prompt
↓
TinyVideo
↓
Generated Video
'

## 🧠 Latent Diffusion

TinyVideo uses a latent-space video diffusion pipeline.

'text
RGB Video
↓
Encoder
↓
Latent Video
↓
Diffusion
↓
Decoder
↓
RGB Video
'

## 📐 Resolution Conditioning

TinyVideo includes resolution conditioning so video resolution can become a controllable property of the generation process.

Current resolution IDs include:

* 1080p
* 720p
* 420p

The long-term goal is for the model to learn the relationship between the prompt and the requested video quality and resolution.

## 🏋️ Local Training

TinyVideo includes tools for creating datasets and training the model locally.

Training data can be created from:

* Synthetic videos
* Existing MP4 videos
* Image/video datasets

## 🖥️ Local GUI

TinyVideo includes a Tkinter-based interface for accessing different parts of the system:

* Video training
* Image-conditioned generation
* Text-to-video generation
* Dataset generation

## 🔧 Modular Architecture

TinyVideo is split into separate Python components so individual parts can be improved without rewriting the entire system.

---

# 🏗️ Architecture

The current TinyVideo pipeline can be simplified to:

'text
USER PROMPT
│
▼
TEXT ENCODER
│
▼
CONDITIONING VECTOR
│
▼
┌───────────────────────┐
│     TinyVideo Model   │
│                       │
│  ┌─────────────────┐  │
│  │ Video Encoder   │  │
│  └────────┬────────┘  │
│           ▼           │
│      Latent Video     │
│           │           │
│           ▼           │
│   Diffusion Process   │
│           │           │
│           ▼           │
│      Video Decoder    │
└───────────┬───────────┘
│
▼
GENERATED VIDEO
'

Image-conditioned generation adds an additional conditioning input:

'text
USER PROMPT
│
▼
TEXT ENCODER
│
│
INPUT IMAGE ──────┤
▼
CONDITIONING
│
▼
LATENT DIFFUSION
│
▼
DECODER
│
▼
GENERATED MP4
'

---

# 📁 Project Structure

The project is intentionally kept relatively simple.

'text
TinyVideo/
│
├── constants.py
├── model.py
├── generate.py
├── generate_with_image.py
├── train.py
├── dataset_generator.py
├── mp4_to_dataset.py
├── video_utils.py
├── text_encoder.py
├── server.py
│
├── main.py
├── ACTIVATE_ME.py
│
├── tkgen.py
├── tkgenimg.py
├── tkgendata.py
├── tktrain.py
│
├── training_data/
├── checkpoints/
├── outputs/
│
└── README.md
'

The exact project structure may change as TinyVideo develops.

---

# ⚙️ Requirements

Recommended:

* Python 3.10+
* PyTorch
* NumPy
* Flask
* Tkinter
* ImageIO
* FFmpeg

A GPU is helpful, but TinyVideo is being developed with local and CPU-friendly experimentation in mind.

> **Note:** CPU generation and training can be significantly slower than GPU execution.

---

# 🚀 Installation

Clone the repository:

'bash
git clone https://github.com/YOUR_USERNAME/TinyVideo.git
cd TinyVideo
'

Create a virtual environment:

## Windows

'powershell
python -m venv .venv
'

Activate it:

'powershell
.venv\Scripts\activate
'

Install the required packages:

'powershell
pip install torch torchvision numpy flask imageio imageio-ffmpeg
'

Depending on the version of TinyVideo, additional dependencies may be required.

---

# ▶️ Running TinyVideo

The intended entry point is:

'powershell
python ACTIVATE_ME.py
'

The launcher initializes the TinyVideo environment and starts the main interface.

---

# 🎥 Text-to-Video

The basic generation interface accepts a text prompt.

Example:

'powershell
python generate.py "a red ball moving across a room"
'

The exact available command-line options depend on the current version of `generate.py`.

Generated videos are written to:

'text
outputs/
'

---

# 🖼️ Image-Conditioned Generation

TinyVideo can generate a video using an input image and a text prompt.

Example:

'powershell
python generate_with_image.py "C:\path\to\image.png" "the character on the image runs forward"
'

The image is passed into the TinyVideo model as an additional conditioning signal.

Conceptually:

'text
IMAGE
│
▼
Image Encoder
│
│
PROMPT ──► Text Encoder
│
▼
Conditioning
│
▼
Diffusion Model
│
▼
Decoder
│
▼
VIDEO
'

---

# 🧪 Dataset Generation

TinyVideo includes a synthetic dataset generator for quickly creating training examples.

The project also contains a converter for turning MP4 videos into TinyVideo training data.

Example:

'powershell
python mp4_to_dataset.py "C:\path\to\videos"
'

The converter samples frames from source videos and converts them into the format expected by the training pipeline.

---

# 🏋️ Training

Training is performed using:

'powershell
python train.py
'

Checkpoints are stored in:

'text
checkpoints/
'

The current checkpoint path is:

'text
checkpoints/tinyvideo_v14.pt
'

TinyVideo's training system is still experimental and will continue to change as the model architecture develops.

---

# 🧩 Core Components

| File                     | Purpose                                        |
| ------------------------ | ---------------------------------------------- |
| `model.py`               | Core neural network and diffusion architecture |
| `generate.py`            | Text-to-video generation                       |
| `generate_with_image.py` | Image-conditioned generation                   |
| `train.py`               | Model training                                 |
| `dataset_generator.py`   | Synthetic dataset generation                   |
| `mp4_to_dataset.py`      | MP4-to-dataset conversion                      |
| `video_utils.py`         | Video processing and saving                    |
| `text_encoder.py`        | Text conditioning                              |
| `constants.py`           | Central project configuration                  |
| `server.py`              | Local web interface/server                     |

---

# 📊 Current Generation Format

The current V1.4 system is designed around short video clips.

| Property         |         Value |
| ---------------- | ------------: |
| Frames           |             8 |
| FPS              |             5 |
| Duration         |  ~1.6 seconds |
| Base resolution  |       64 × 64 |
| Latent channels  |             4 |
| Diffusion steps  |          1000 |
| Generation steps | 20 by default |

These are development settings, not intended as permanent project limitations.

---

# 🧠 Why TinyVideo?

Large AI video systems can require enormous amounts of:

* GPU memory
* Storage
* Training data
* Compute
* Infrastructure

TinyVideo takes a different approach.

The goal is to build a small system where the entire pipeline can be inspected and modified:

'text
Dataset
↓
Training
↓
Model
↓
Generation
↓
Video
'

This makes TinyVideo useful as both an experimental AI video-generation project and a learning platform for building generative models.

---

# 🗺️ Roadmap

TinyVideo is under active development.

## V1.4

* [x] Latent video representation
* [x] Diffusion-based generation
* [x] Text conditioning
* [x] Image conditioning
* [x] Resolution conditioning foundation
* [x] MP4 dataset conversion
* [x] Local training pipeline
* [x] Local generation tools
* [x] Tkinter interface
* [ ] Improve generation quality
* [ ] Improve temporal consistency
* [ ] Improve prompt understanding
* [ ] Expand training datasets

---

# 🚀 V1.4 Pro

Planned improvements include increasing the model's working dimensions while maintaining a square video format.

Planned direction:

'text
64 × 64
↓
128 × 128
↓
256 × 256
↓
...
'

Goals:

* Higher-resolution generation
* Better visual quality
* Better temporal consistency
* Improved conditioning
* Larger and better datasets
* More controllable generation

---

# 🔊 TinyAudio

A separate audio-generation system is planned.

The long-term architecture is:

'text
USER PROMPT
│
┌─────────┴─────────┐
│                   │
▼                   ▼
TinyVideo            TinyAudio
│                   │
▼                   ▼
VIDEO               AUDIO
│                   │
└─────────┬─────────┘
▼
COMBINE
│
▼
FINAL VIDEO
'

TinyAudio will be developed independently from TinyVideo so the two systems can evolve separately.

The eventual goal is:

'text
Prompt
│
├──► Video generation
│
└──► Audio generation
│
▼
Combine
│
▼
Final MP4
'

---

# 🔬 Long-Term Goals

Future versions may explore:

* Longer videos
* Higher resolutions
* Better motion consistency
* Better text understanding
* Improved image conditioning
* More efficient training
* Better temporal modeling
* Audio generation
* Audio/video synchronization
* Video editing
* Video-to-video generation
* More advanced conditioning
* Larger datasets
* More efficient CPU inference
* Optional GPU acceleration

---

# ⚠️ Project Status

TinyVideo is an **experimental research and development project**.

It is not currently intended to compete with large commercial video-generation systems.

The main goal is to continuously improve a small, understandable, locally controlled video-generation pipeline.

Expect:

* Experimental model architectures
* Changing APIs
* Changing dataset formats
* Slow generation on CPU
* Imperfect video quality
* Breaking changes between versions

---

# 🤝 Contributing

Contributions, experiments, ideas, and improvements are welcome.

Useful areas include:

* Dataset generation
* Model architecture
* Training optimization
* Video processing
* Conditioning methods
* CPU optimization
* GUI improvements
* Documentation
* Testing

If you experiment with TinyVideo, feel free to open an issue or pull request with your results.

---

# 📜 License

See the repository's license file for the current licensing terms.

---

# ⭐ TinyVideo

TinyVideo is built around one idea:

> **Build the video generator instead of just using one.**

Small model.
Local pipeline.
Full control.
Continuous experimentation.

'text
Prompt
↓
TinyVideo
↓
Video
'

And eventually:

'text
Prompt
│
┌───────┴───────┐
▼               ▼
TinyVideo       TinyAudio
│               │
▼               ▼
Video           Audio
│               │
└───────┬───────┘
▼
Combine
│
▼
Final Video
'
