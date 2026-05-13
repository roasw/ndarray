from __future__ import annotations

"""Custom op schema names used by ndarray torch dispatcher registration.

Naming strategy:
- `UPSAMPLE_2D_FOURIER` is the canonical operator schema name.
- Backend-specific behavior is selected by dispatcher keys (CPU/Meta/CUDA),
  not by embedding backend suffixes into canonical schema names.
"""

UPSAMPLE_2D_FOURIER = "upsample_2d_fourier"

__all__ = ["UPSAMPLE_2D_FOURIER"]
