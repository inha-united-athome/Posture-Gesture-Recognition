"""
Temporal Convolutional Network (TCN) for skeleton-based action recognition.

Architecture:
    Input: (B, C_in, T) — B=batch, C_in=34 (17 joints × 2 coords), T=frames
    → TCN Encoder (dilated causal convolutions)
    → Global Average Pooling
    → FC Classification Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """
    Single TCN block with dilated causal convolution + residual connection.

        Input → Conv1D(dilated) → BN → ReLU → Dropout
              → Conv1D(dilated) → BN → ReLU → Dropout
              + Residual(1x1 conv if needed)
              → Output
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()

        padding = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual projection (1x1 conv) if channel dims differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.padding = padding

    def forward(self, x):
        """
        x: (B, C, T)
        """
        residual = self.residual(x)

        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # causal trim
        out = self.relu(self.bn1(out))
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # causal trim
        out = self.relu(self.bn2(out))
        out = self.dropout(out)

        return self.relu(out + residual)


class TCN(nn.Module):
    """
    Multi-layer TCN for skeleton action recognition.

    Args:
        input_dim: Input feature dimension (default: 34 = 17 joints × 2)
        num_classes: Number of action classes
        hidden_dims: List of hidden channel dimensions for TCN layers
        kernel_size: Convolution kernel size
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 34,
        num_classes: int = 5,
        hidden_dims: list = None,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 128, 256]

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Build TCN layers with increasing dilation
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(hidden_dims):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            layers.append(
                TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C_in, T) — skeleton features
            mask: (B, T) — 1 for valid, 0 for padding (optional)

        Returns:
            logits: (B, num_classes)
        """
        # TCN encoder
        out = self.tcn(x)  # (B, C_hidden, T)

        # Masked global average pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)  # (B, 1, T)
            out = out * mask_expanded
            out = out.sum(dim=2) / mask_expanded.sum(dim=2).clamp(min=1)  # (B, C_hidden)
        else:
            out = out.mean(dim=2)  # (B, C_hidden)

        # Classification
        logits = self.classifier(out)  # (B, num_classes)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = TCN(input_dim=130, num_classes=6)
    print(f"Model parameters: {model.count_parameters():,}")
    print(model)

    x = torch.randn(4, 130, 120)  # batch=4, 130 features (65 joints × 2), 120 frames
    mask = torch.ones(4, 120)
    out = model(x, mask)
    print(f"Input: {x.shape} → Output: {out.shape}")
