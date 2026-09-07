import argparse
import os
import random

import torch
import torch.nn.functional as F

from model import TinyVideoGenerator
from text_encoder import encode_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "tiny_dataset.pt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
DEVICE = torch.device("cpu")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "tinyvideo.pt")

EPOCHS = 1
LEARNING_RATE = 0.0005

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train(epochs=EPOCHS, learning_rate=LEARNING_RATE):
    print("Loading dataset...")

    dataset = torch.load(
        DATASET_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if not dataset:
        raise ValueError("Dataset is empty.")

    print("Samples:", len(dataset))
    print("Device:", DEVICE)
    print("Epochs:", epochs)
    print("Learning rate:", learning_rate)

    model = TinyVideoGenerator().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # Shuffle sample order each epoch so appended user videos do not always
        # arrive in the same order during training.
        order = list(range(len(dataset)))
        random.shuffle(order)

        for step, index in enumerate(order, start=1):
            prompt, target = dataset[index]

            text = encode_text(prompt).unsqueeze(0).to(DEVICE)
            target = target.unsqueeze(0).to(DEVICE)

            prediction = model(text)
            loss = F.mse_loss(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0 or step == len(dataset):
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"Step {step}/{len(dataset)} "
                    f"Loss {loss.item():.6f}"
                )

        average_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} Average Loss: {average_loss:.6f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)

    print()
    print("Model saved:")
    print(CHECKPOINT_PATH)


def main():
    parser = argparse.ArgumentParser(description="Train TinyVideo.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")

    train(args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()
