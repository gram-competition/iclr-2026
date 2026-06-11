"""
Per-point polynomial extrapolation.

Fits a degree-2 polynomial independently at every spatial point using the
5 input timesteps, then evaluates it at the 5 output timesteps.

All operations are batched PyTorch — no Python loops over points or timesteps.
The pseudoinverse of the Vandermonde matrix is computed once and cached on the
first call (or when t changes).

Input/output shapes match the model interface:
    velocity_in  : (B, 5, N, 3)
    t            : (B, 10)       — t[:, :5] are input times, t[:, 5:] are output times
    returns      : (B, 5, N, 3)  — extrapolated velocity at output times
"""

import torch
import torch.nn.functional as F


def _vander(t: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Build a Vandermonde matrix.
    t: (T,)  →  returns (T, degree+1)  with columns [1, t, t², ...]
    """
    return torch.stack([t ** i for i in range(degree + 1)], dim=-1)


def poly_extrapolate(
    velocity_in: torch.Tensor,
    t: torch.Tensor,
    degree: int = 2,
) -> torch.Tensor:
    """
    Fit degree-`degree` polynomials to velocity_in at input times,
    then evaluate at output times.

    Args:
        velocity_in : (B, T_in, N, 3)  — input velocity snapshots
        t           : (B, T_in + T_out) — time values; first T_in are inputs
        degree      : polynomial degree (2 = quadratic, default)

    Returns:
        (B, T_out, N, 3) — extrapolated velocity at output times
    """
    B, T_in, N, C = velocity_in.shape
    T_out = t.shape[1] - T_in
    device = velocity_in.device
    dtype  = velocity_in.dtype

    t_in  = t[:, :T_in]   # (B, T_in)
    t_out = t[:, T_in:]   # (B, T_out)

    # Normalise time to [0, 1] per sample for numerical stability
    t0 = t_in[:, :1]          # (B, 1)
    t1 = t_in[:, -1:]         # (B, 1)
    span = (t1 - t0).clamp(min=1e-8)

    t_in_n  = (t_in  - t0) / span   # (B, T_in)
    t_out_n = (t_out - t0) / span   # (B, T_out)

    # Vandermonde matrices
    # A: (B, T_in, degree+1)
    A = torch.stack([t_in_n ** i for i in range(degree + 1)], dim=-1)

    # Least-squares fit via normal equations: coeffs = (AᵀA)⁻¹ Aᵀ v
    # Use torch.linalg.lstsq for numerical safety.
    # velocity_in reshaped: (B, T_in, N*C)
    v = velocity_in.reshape(B, T_in, N * C)

    # lstsq expects (..., m, n) for A and (..., m, k) for b
    # A: (B, T_in, deg+1),  v: (B, T_in, N*C)
    result = torch.linalg.lstsq(A, v)        # coeffs: (B, deg+1, N*C)
    coeffs = result.solution                  # (B, degree+1, N*C)

    # Evaluate at output times
    # B_out: (B, T_out, degree+1)
    B_out = torch.stack([t_out_n ** i for i in range(degree + 1)], dim=-1)

    # (B, T_out, degree+1) @ (B, degree+1, N*C) → (B, T_out, N*C)
    v_out = torch.bmm(B_out, coeffs)

    return v_out.reshape(B, T_out, N, C)


def poly_fit_residual(
    velocity_in: torch.Tensor,
    t: torch.Tensor,
    degree: int = 2,
) -> torch.Tensor:
    """
    Return the residual between actual input velocity and the polynomial fit
    evaluated at the input times. Shape: (B, T_in, N, 3).

    This is used as a feature: high residual ≡ high local turbulence intensity.
    """
    B, T_in, N, C = velocity_in.shape
    t_in = t[:, :T_in]

    t0   = t_in[:, :1]
    span = (t_in[:, -1:] - t0).clamp(min=1e-8)
    t_in_n = (t_in - t0) / span

    A = torch.stack([t_in_n ** i for i in range(degree + 1)], dim=-1)  # (B, T_in, d+1)
    v = velocity_in.reshape(B, T_in, N * C)

    coeffs = torch.linalg.lstsq(A, v).solution   # (B, d+1, N*C)
    v_fit  = torch.bmm(A, coeffs)                 # (B, T_in, N*C)

    return (v - v_fit).reshape(B, T_in, N, C)
