import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class ConvVAE(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int, channels: list[int] = [16, 32, 64]):
        """
        input_channels: Length of input 1D vector (e.g., 667)
        latent_dim: Size of latent space
        channels: List of 3 integers for encoder conv layer widths
        """
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        assert len(channels) == 3, "Expected exactly 3 channel sizes"

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.Conv1d(channels[0], channels[1], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels[1], channels[2], kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Compute flattened size
        dummy_input = torch.zeros(1, 1, input_channels)
        with torch.no_grad():
            enc_out = self.encoder(dummy_input)
        self.flat_dim = enc_out.numel()
        self.encoder_output_shape = enc_out.shape[1:]  # (C, L)

        # Latent space
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels[2], channels[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(channels[1], channels[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(channels[0], 1, kernel_size=4, stride=2, padding=1)
        )

    def encode(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), *self.encoder_output_shape)
        x = self.decoder(x)
        return x.squeeze(1)  # (B, L)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # Match reconstruction size to input
        if recon.size(1) > x.size(1):
            recon = recon[:, :x.size(1)]
        elif recon.size(1) < x.size(1):
            pad = x.size(1) - recon.size(1)
            recon = F.pad(recon, (0, pad))

        return recon, mu, logvar



class HybridVAE(nn.Module):
    def __init__(self, input_channels=667, latent_dim=16):
        super().__init__()
        
        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        
        # After CNNs
        self.flatten = nn.Flatten()
        
        dummy_input = torch.zeros(1, 1, input_channels)
        enc_out = self.encoder_cnn(dummy_input)
        self.flat_dim = enc_out.numel()
        
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flat_dim)

        self.decoder_cnn = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=5, stride=1, padding=2)
        )

    def encode(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 32, -1)  # reshape for 32 channels
        x = self.decoder_cnn(x)
        return x.squeeze(1)  # (B, L)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # Handle minor length mismatch
        if recon.size(1) > x.size(1):
            recon = recon[:, :x.size(1)]
        elif recon.size(1) < x.size(1):
            recon = F.pad(recon, (0, x.size(1) - recon.size(1)))

        return recon, mu, logvar


class PositionalEncodingVAE(nn.Module):
    def __init__(self, input_channels=667, latent_dim=16, pos_encoding_dim=16):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.pos_encoding_dim = pos_encoding_dim

        # Build positional encoding once
        self.register_buffer("pos_encoding", self.build_positional_encoding(input_channels, pos_encoding_dim))

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(1 + pos_encoding_dim, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        dummy_input = torch.zeros(1, 1 + pos_encoding_dim, input_channels)
        enc_out = self.encoder_cnn(dummy_input)
        self.flat_dim = enc_out.numel()

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flat_dim)

        self.decoder_cnn = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=5, stride=1, padding=2)
        )

    def build_positional_encoding(self, length, dim):
        """Create fixed positional encoding."""
        pe = torch.zeros(dim, length)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)  # (length, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))  # (dim//2,)

        # Fix: match dimensions correctly
        angles = position * div_term.unsqueeze(0)  # (length, dim//2)

        pe[0::2, :] = torch.sin(angles).transpose(0, 1)
        pe[1::2, :] = torch.cos(angles).transpose(0, 1)
        return pe  # shape (dim, length)

    def encode(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        batch_size = x.size(0)
        pos = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)  # (B, pos_encoding_dim, L)
        x = torch.cat([x, pos], dim=1)  # (B, 1+pos_encoding_dim, L)
        x = self.encoder_cnn(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 64, -1)
        x = self.decoder_cnn(x)
        return x.squeeze(1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        if recon.size(1) > x.size(1):
            recon = recon[:, :x.size(1)]
        elif recon.size(1) < x.size(1):
            recon = F.pad(recon, (0, x.size(1) - recon.size(1)))

        return recon, mu, logvar