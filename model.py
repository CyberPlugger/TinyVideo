import torch
import torch.nn as nn


class TinyVideoGenerator(nn.Module):

    def __init__(
        self,
        text_dim=128,
        latent_dim=256,
        frames=8,
        height=64,
        width=64
    ):
        super().__init__()

        self.frames = frames
        self.height = height
        self.width = width

        # Текст
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )

        # Случайный шум.
        #
        # Он НЕ заменяет текст.
        # Он добавляет вариативность.
        self.noise_projection = nn.Sequential(
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )

        # Объединяем текст + шум
        self.generator = nn.Sequential(
            nn.Linear(
                latent_dim * 2,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                1024
            ),

            nn.ReLU(),

            nn.Linear(
                1024,
                frames * 16 * 16 * 16
            ),

            nn.Tanh()
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                16,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                3,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()
        )


    def forward(
        self,
        text,
        noise=None,
        progress_callback=None
    ):

        batch = text.shape[0]

        # --------------------------------------------------
        # TEXT
        # --------------------------------------------------

        text_features = self.text_projection(
            text
        )

        # --------------------------------------------------
        # NOISE
        # --------------------------------------------------

        if noise is None:

            noise = torch.randn(
                batch,
                64,
                device=text.device
            )

        noise_features = self.noise_projection(
            noise
        )

        # --------------------------------------------------
        # TEXT + NOISE
        # --------------------------------------------------

        x = torch.cat(
            [
                text_features,
                noise_features
            ],
            dim=1
        )

        x = self.generator(
            x
        )

        x = x.view(
            batch,
            self.frames,
            16,
            16,
            16
        )

        # --------------------------------------------------
        # DECODE FRAMES
        # --------------------------------------------------

        frames = []

        for i in range(
            self.frames
        ):

            frame = x[:, i]

            frame = self.decoder(
                frame
            )

            frames.append(
                frame
            )

            if progress_callback:

                progress_callback(
                    i + 1,
                    self.frames
                )

        video = torch.stack(
            frames,
            dim=1
        )

        return video
