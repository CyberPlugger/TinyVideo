def generate(
    prompt,
    output=None,
    seed=None
):

    import os
    import time
    import torch

    if seed is None:

        seed = torch.randint(
            0,
            2**31 - 1,
            (1,)
        ).item()

    print()
    print("=" * 55)
    print("                    TinyVideo")
    print("=" * 55)

    print()
    print(f"Prompt: {prompt}")
    print(f"Seed:   {seed}")

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    start_time = time.time()

    # ----------------------------------------------
    # MODEL
    # ----------------------------------------------

    print()
    print("[1/5] Loading model...")

    model = TinyVideoGenerator()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    print("       ✓ Model loaded")

    # ----------------------------------------------
    # TEXT
    # ----------------------------------------------

    print()
    print("[2/5] Encoding prompt...")

    text = encode_text(
        prompt
    )

    text = text.unsqueeze(
        0
    )

    print("       ✓ Text encoded")

    # ----------------------------------------------
    # GENERATOR
    # ----------------------------------------------

    print()
    print("[3/5] Preparing generator...")

    print(
        "       Device: CPU"
    )

    print(
        "       Resolution: 64x64"
    )

    print(
        "       Frames: 8"
    )

    print(
        f"       Seed: {seed}"
    )

    # ----------------------------------------------
    # GENERATE
    # ----------------------------------------------

    print()
    print(
        "[4/5] Generating video..."
    )

    with torch.no_grad():

        video = model(
            text
        )

    video = video[0]

    # ----------------------------------------------
    # OUTPUT
    # ----------------------------------------------

    print()
    print(
        "[5/5] Saving MP4..."
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if output is None:

        output = (
            f"video_seed_{seed}.mp4"
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        output
    )

    save_video(
        video,
        output_path,
        fps=5
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 55)
    print(
        "                 VIDEO GENERATED!"
    )
    print("=" * 55)

    print()
    print(
        f"Seed: {seed}"
    )

    print(
        f"File: {output_path}"
    )

    print(
        f"Time: {elapsed:.2f}s"
    )

    return output_path
