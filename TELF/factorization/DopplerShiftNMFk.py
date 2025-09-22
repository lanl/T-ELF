# shift_nmfk_tfgm.py

from __future__ import annotations
import os
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

# Bring in your existing K-selection helper
from .NMFk import NMFk

 


# ================ ================ ================ ================ ================ ================
# ================ ================ ================ ================ ================ ================
# ================ ================ ================ ================ ================ ================

# === ALL-IN-ONE: arrays-only (no state dicts in function signatures) Shift-NMFk-TFGM v12 + sample ===
import numpy as np
from typing import Optional, Tuple, Literal, List, Dict

# --------------------------
# Shift operators & helpers
# --------------------------
def circ_shift_linpos_1d(x: np.ndarray, tau: float) -> np.ndarray:
    """Circular, linear-interp shift by real-valued tau (nonnegative output)."""
    L = int(x.shape[0])
    if L == 0: return x.copy()
    k = int(np.floor(tau))
    a = float(tau - k)
    idx0 = (np.arange(L) - k) % L
    idx1 = (idx0 - 1) % L
    return (1.0 - a) * x[idx0] + a * x[idx1]

def vec_outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.outer(a, b).reshape(-1)

def xcorr_circ(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    return np.real(np.fft.ifft(np.conj(A)*B))

def argmax_local_circ(c: np.ndarray, center: float, window: int) -> float:
    L = c.size
    k0 = int(np.round(np.mod(center, L)))
    idx = (k0 + np.arange(-window, window+1)) % L
    j = idx[int(np.argmax(c[idx]))]
    jm1, jp1 = (j-1) % L, (j+1) % L
    y_m1, y0, y_p1 = c[jm1], c[j], c[jp1]
    denom = (y_m1 - 2*y0 + y_p1)
    delta = 0.0 if abs(denom) < 1e-12 else 0.5*(y_m1 - y_p1)/denom
    tau = (j + delta) % L
    if tau > L/2: tau -= L
    return float(tau)

# --------------------------
# Forward models / utilities
# --------------------------
def reconstruct_from_params(W: np.ndarray, F: np.ndarray, S: np.ndarray,
                            tau_f: np.ndarray, tau_t: np.ndarray) -> np.ndarray:
    """X_hat[n,f,t] = sum_i W[n,i] * shift(F[i], tau_f[n,i]) ⊗ shift(S[i], tau_t[n,i])."""
    N, K = W.shape
    Fdim, Tdim = F.shape[1], S.shape[1]
    X_hat = np.zeros((N, Fdim, Tdim), dtype=float)
    for n in range(N):
        for i in range(K):
            Fi = circ_shift_linpos_1d(F[i], float(tau_f[n,i]))
            Si = circ_shift_linpos_1d(S[i], float(tau_t[n,i]))
            X_hat[n] += W[n,i] * np.outer(Fi, Si)
    return X_hat

def compute_shifted(F: np.ndarray, S: np.ndarray, tau_f: np.ndarray, tau_t: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-sensor shifted dictionaries: Fi_shifted[n,i,f], Si_shifted[n,i,t]."""
    N, K = tau_f.shape
    Fdim, Tdim = F.shape[1], S.shape[1]
    Fi_shifted = np.zeros((N, K, Fdim))
    Si_shifted = np.zeros((N, K, Tdim))
    for n in range(N):
        for i in range(K):
            Fi_shifted[n,i] = circ_shift_linpos_1d(F[i], tau_f[n,i])
            Si_shifted[n,i] = circ_shift_linpos_1d(S[i], tau_t[n,i])
    return Fi_shifted, Si_shifted

def reconstruct_from_shifted(W: np.ndarray, Fi_shifted: np.ndarray, Si_shifted: np.ndarray) -> np.ndarray:
    N = W.shape[0]
    Fdim = Fi_shifted.shape[2]
    Tdim = Si_shifted.shape[2]
    X_hat = np.zeros((N, Fdim, Tdim), dtype=float)
    for n in range(N):
        X_hat[n] = np.tensordot(W[n], np.einsum("if,it->ift", Fi_shifted[n], Si_shifted[n]), axes=(0,0))
    return X_hat

def relerr(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A-B)/(np.linalg.norm(A)+1e-12))

def compute_loss(X: np.ndarray, W: np.ndarray, F: np.ndarray, S: np.ndarray,
                 tau_f: np.ndarray, tau_t: np.ndarray) -> Tuple[float, float]:
    Fdim, Tdim = X.shape[1], X.shape[2]
    X_hat = reconstruct_from_params(W, F, S, tau_f, tau_t)
    loss = 0.5*np.sum((X - X_hat)**2)/(Fdim*Tdim)
    R = relerr(X, X_hat)
    return float(loss), float(R)

# --------------------------
# W update (ridge LS, ≥0)
# --------------------------
# def update_W_ls_from_shifted(X: np.ndarray, W: np.ndarray, Fi_shifted: np.ndarray, Si_shifted: np.ndarray,
#                              ridge_w: float = 0.0) -> np.ndarray:
#     """Return updated W via ridge least squares (per sensor, nonnegative)."""
#     N, Fdim, Tdim = X.shape
#     FT = Fdim*Tdim
#     K = W.shape[1]
#     W_new = np.empty_like(W)
#     for n in range(N):
#         A = np.stack([vec_outer(Fi_shifted[n,i], Si_shifted[n,i]) for i in range(K)], axis=1)  # (FT,K)
#         b = X[n].reshape(FT)
#         AtA = A.T @ A
#         if ridge_w > 0:
#             AtA = AtA + ridge_w * np.eye(K)
#         Atb = A.T @ b
#         w = np.linalg.solve(AtA, Atb)
#         W_new[n] = np.clip(w, 1e-12, None)
#     return W_new
def update_W_ls_from_shifted(X: np.ndarray, W: np.ndarray,
                             Fi_shifted: np.ndarray, Si_shifted: np.ndarray,
                             ridge_w: float = 0.0) -> np.ndarray:
    """
    Robust ridge least-squares update for W (per sensor, nonnegative).
    Uses SVD-based least squares (np.linalg.lstsq) with optional Tikhonov (ridge)
    via an augmented system. Column-normalizes A to mitigate conditioning.
    """
    N, Fdim, Tdim = X.shape
    FT = Fdim * Tdim
    K = W.shape[1]
    W_new = np.empty_like(W)

    for n in range(N):
        # Build design A: each column is vec( Fi ⊗ Si )
        # Shapes: Fi_shifted[n,i] -> (Fdim,), Si_shifted[n,i] -> (Tdim,)
        A = np.stack(
            [np.outer(Fi_shifted[n, i], Si_shifted[n, i]).reshape(FT) for i in range(K)],
            axis=1
        )   # (FT, K)
        b = X[n].reshape(FT)

        # Guard: zero/near-zero columns -> set weight to zero
        col_norms = np.linalg.norm(A, axis=0)
        keep = col_norms > 1e-12
        if not np.any(keep):
            W_new[n] = 1e-12  # all tiny; return floor
            continue

        # Column-normalize the kept columns
        A_s = A[:, keep] / (col_norms[keep][None, :])
        Kk = int(np.sum(keep))

        # Solve min ||A_s w_s - b||^2  (+ λ||w_s||^2 if ridge_w>0)
        if ridge_w > 0.0:
            lam = float(ridge_w)
            A_aug = np.vstack([A_s, np.sqrt(lam) * np.eye(Kk)])
            b_aug = np.concatenate([b, np.zeros(Kk, dtype=b.dtype)])
            w_s, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=1e-10)
        else:
            w_s, *_ = np.linalg.lstsq(A_s, b, rcond=1e-10)

        # Undo column scaling and place back into full K-length vector
        w_full = np.zeros(K, dtype=W.dtype)
        w_full[keep] = w_s / (col_norms[keep] + 1e-12)

        # Enforce nonnegativity (same policy you already use)
        W_new[n] = np.clip(w_full, 1e-12, None)

    return W_new



# --------------------------
# (F,S) update: SVD rank-1 with accept-if-better
# --------------------------
def update_rank1_for_component(X: np.ndarray, W: np.ndarray, F: np.ndarray, S: np.ndarray,
                               tau_f: np.ndarray, tau_t: np.ndarray, i: int
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (F_updated, S_updated) with only component i potentially modified."""
    N, K = W.shape
    Fdim, Tdim = F.shape[1], S.shape[1]

    # Shift once
    Fi_shifted, Si_shifted = compute_shifted(F, S, tau_f, tau_t)

    # Residual excluding i
    R_i = np.zeros((N, Fdim, Tdim))
    for n in range(N):
        recon_wo_i = np.zeros((Fdim, Tdim))
        for j in range(K):
            if j == i: continue
            recon_wo_i += W[n,j] * np.outer(Fi_shifted[n,j], Si_shifted[n,j])
        R_i[n] = X[n] - recon_wo_i

    # Weighted aligned residual
    w = W[:, i]
    wsum = float(np.sum(w))
    if wsum <= 1e-16:
        return F, S
    A = np.zeros((Fdim, Tdim))
    for n in range(N):
        if w[n] <= 0: continue
        Rn = R_i[n]
        if tau_t[n,i] != 0.0:
            Rn = np.stack([circ_shift_linpos_1d(Rn[r], -tau_t[n,i]) for r in range(Fdim)], axis=0)
        if tau_f[n,i] != 0.0:
            Rn = np.stack([circ_shift_linpos_1d(Rn[:,c], -tau_f[n,i]) for c in range(Tdim)], axis=1)
        A += w[n] * Rn
    A /= wsum

    # Top SVD, project nonneg, keep amplitude via sqrt(s1)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    u = np.maximum(U[:,0], 1e-12)
    v = np.maximum(Vt[0,:], 1e-12)
    s1 = float(max(s[0], 0.0))
    scale = np.sqrt(s1)
    Fi_new = u * scale
    Si_new = v * scale

    F_old_i = F[i].copy()
    S_old_i = S[i].copy()
    F_cand = F.copy()
    S_cand = S.copy()
    F_cand[i] = Fi_new
    S_cand[i] = Si_new

    loss_old, _ = compute_loss(X, W, F, S, tau_f, tau_t)
    loss_new, _ = compute_loss(X, W, F_cand, S_cand, tau_f, tau_t)
    if loss_new <= loss_old + 1e-12:
        return F_cand, S_cand
    else: # rollback
        F_rollback = F.copy()
        S_rollback = S.copy()
        F_rollback[i] = F_old_i
        S_rollback[i] = S_old_i
        return F_rollback, S_rollback

# --------------------------
# Snapping of shifts
# --------------------------
def snap_shifts(X: np.ndarray, W: np.ndarray, F: np.ndarray, S: np.ndarray,
                tau_f: np.ndarray, tau_t: np.ndarray,
                snap_window_f: int = 6, snap_window_t: int = 6
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (tau_f_new, tau_t_new) snapped via local xcorr around current values."""
    N, K = W.shape
    Fdim, Tdim = F.shape[1], S.shape[1]
    Fi_shifted, Si_shifted = compute_shifted(F, S, tau_f, tau_t)
    full = reconstruct_from_shifted(W, Fi_shifted, Si_shifted)

    tau_f_new = tau_f.copy()
    tau_t_new = tau_t.copy()
    for n in range(N):
        for i in range(K):
            R_i = X[n] - (full[n] - W[n,i]*np.outer(Fi_shifted[n,i], Si_shifted[n,i]))
            # freq snap
            w_t = Si_shifted[n,i] / (np.sum(Si_shifted[n,i]) + 1e-12)
            freq_prof = R_i @ w_t
            c = xcorr_circ(F[i], freq_prof)
            tau_f_new[n,i] = argmax_local_circ(c, tau_f[n,i], snap_window_f)
            # time snap
            w_f = Fi_shifted[n,i] / (np.sum(Fi_shifted[n,i]) + 1e-12)
            time_prof = R_i.T @ w_f
            c_t = xcorr_circ(S[i], time_prof)
            tau_t_new[n,i] = argmax_local_circ(c_t, tau_t[n,i], snap_window_t)
    return tau_f_new, tau_t_new

# --------------------------
# Fit loops (arrays only)
# --------------------------
def _log_step(X: np.ndarray, W: np.ndarray, F: np.ndarray, S: np.ndarray,
              tau_f: np.ndarray, tau_t: np.ndarray,
              it: int, tag: str, verbose: bool, loss_history: List[float]) -> None:
    loss, R = compute_loss(X, W, F, S, tau_f, tau_t)
    loss_history.append(loss)
    if verbose and (it % 10 == 0 or it == 1):
        print(f"[{tag:>6} {it:4d}] loss={loss:.6e}  R={R:.6f}")

def fit_monolithic(X: np.ndarray, W: np.ndarray, F: np.ndarray, S: np.ndarray,
                   tau_f: np.ndarray, tau_t: np.ndarray,
                   n_iters: int = 400, freeze_tau_iters: int = 120, snap_every: int = 5,
                   snap_window_f: int = 6, snap_window_t: int = 6,
                   inner_rank1_iters: int = 1, ridge_w: float = 0.0, verbose: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    loss_history: List[float] = []
    for it in range(1, n_iters+1):
        Fi, Si = compute_shifted(F, S, tau_f, tau_t)
        W = update_W_ls_from_shifted(X, W, Fi, Si, ridge_w=ridge_w)
        for i in range(F.shape[0]):
            for _ in range(inner_rank1_iters):
                F, S = update_rank1_for_component(X, W, F, S, tau_f, tau_t, i)
        if it > freeze_tau_iters and snap_every > 0 and (it % snap_every == 0):
            tau_f, tau_t = snap_shifts(X, W, F, S, tau_f, tau_t,
                                       snap_window_f=snap_window_f, snap_window_t=snap_window_t)
        _log_step(X, W, F, S, tau_f, tau_t, it, "ALL", verbose, loss_history)
    return W, F, S, tau_f, tau_t, loss_history

def fit_shifts_only(X: np.ndarray, W: np.ndarray, F: np.ndarray, S: np.ndarray,
                    tau_f: np.ndarray, tau_t: np.ndarray,
                    iters_w: int = 60, iters_snap: int = 80,
                    snap_window_f: int = 6, snap_window_t: int = 6,
                    ridge_w: float = 0.0, verbose: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    loss_history: List[float] = []
    for it in range(1, iters_w+1):
        Fi, Si = compute_shifted(F, S, tau_f, tau_t)
        W = update_W_ls_from_shifted(X, W, Fi, Si, ridge_w=ridge_w)
        _log_step(X, W, F, S, tau_f, tau_t, it, "W", verbose, loss_history)
    for it in range(1, iters_snap+1):
        Fi, Si = compute_shifted(F, S, tau_f, tau_t)
        W = update_W_ls_from_shifted(X, W, Fi, Si, ridge_w=ridge_w)
        tau_f, tau_t = snap_shifts(X, W, F, S, tau_f, tau_t,
                                   snap_window_f=snap_window_f, snap_window_t=snap_window_t)
        _log_step(X, W, F, S, tau_f, tau_t, it, "W+τ", verbose, loss_history)
    return W, F, S, tau_f, tau_t, loss_history

def fit_staged(X: np.ndarray, W: np.ndarray, F: np.ndarray, S: np.ndarray,
               tau_f: np.ndarray, tau_t: np.ndarray,
               iters_w: int = 60, iters_fs: int = 120, iters_snap: int = 80,
               inner_rank1_iters: int = 1,
               snap_window_f: int = 6, snap_window_t: int = 6,
               ridge_w: float = 0.0, verbose: bool = True
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    loss_history: List[float] = []
    # Stage 1: W
    for it in range(1, iters_w+1):
        Fi, Si = compute_shifted(F, S, tau_f, tau_t)
        W = update_W_ls_from_shifted(X, W, Fi, Si, ridge_w=ridge_w)
        _log_step(X, W, F, S, tau_f, tau_t, it, "W", verbose, loss_history)
    # Stage 2: W + (F,S)
    for it in range(1, iters_fs+1):
        Fi, Si = compute_shifted(F, S, tau_f, tau_t)
        W = update_W_ls_from_shifted(X, W, Fi, Si, ridge_w=ridge_w)
        for i in range(F.shape[0]):
            for _ in range(inner_rank1_iters):
                F, S = update_rank1_for_component(X, W, F, S, tau_f, tau_t, i)
        _log_step(X, W, F, S, tau_f, tau_t, it, "W+FS", verbose, loss_history)
    # Stage 3: W + (F,S) + snap
    for it in range(1, iters_snap+1):
        Fi, Si = compute_shifted(F, S, tau_f, tau_t)
        W = update_W_ls_from_shifted(X, W, Fi, Si, ridge_w=ridge_w)
        for i in range(F.shape[0]):
            for _ in range(inner_rank1_iters):
                F, S = update_rank1_for_component(X, W, F, S, tau_f, tau_t, i)
        tau_f, tau_t = snap_shifts(X, W, F, S, tau_f, tau_t,
                                   snap_window_f=snap_window_f, snap_window_t=snap_window_t)
        _log_step(X, W, F, S, tau_f, tau_t, it, "ALL", verbose, loss_history)
    return W, F, S, tau_f, tau_t, loss_history

def prepare_initial_params(X: np.ndarray, K: int,
                           W0: Optional[np.ndarray] = None, F0: Optional[np.ndarray] = None, S0: Optional[np.ndarray] = None,
                           tau_f0: Optional[np.ndarray] = None, tau_t0: Optional[np.ndarray] = None,
                           seed: Optional[int] = 0
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert X.ndim == 3 and np.all(X >= 0), "X must be nonnegative with shape (N,F,T)"
    N, Fdim, Tdim = X.shape
    rng = np.random.default_rng(seed)
    W = np.maximum(1e-3 + 0.1*rng.random((N,K)), 1e-8) if W0 is None else W0.copy()
    F = np.maximum(1e-3 + 0.1*rng.random((K,Fdim)), 1e-8) if F0 is None else F0.copy()
    S = np.maximum(1e-3 + 0.1*rng.random((K,Tdim)), 1e-8) if S0 is None else S0.copy()
    tau_f = np.zeros((N,K)) if tau_f0 is None else tau_f0.copy()
    tau_t = np.zeros((N,K)) if tau_t0 is None else tau_t0.copy()
    return W, F, S, tau_f, tau_t

def fit(X: np.ndarray, K: int,
        mode: Literal["monolithic","staged","shifts_only"] = "monolithic",
        seed: Optional[int] = 0,
        # common knobs
        ridge_w: float = 0.0, verbose: bool = True,
        # monolithic
        n_iters: int = 400, freeze_tau_iters: int = 120, snap_every: int = 5,
        snap_window_f: int = 6, snap_window_t: int = 6, inner_rank1_iters: int = 1,
        # staged/shifts-only
        iters_w: int = 60, iters_fs: int = 120, iters_snap: int = 80,
        # explicit init (arrays)
        W0: Optional[np.ndarray] = None, F0: Optional[np.ndarray] = None, S0: Optional[np.ndarray] = None,
        tau_f0: Optional[np.ndarray] = None, tau_t0: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    W, F, S, tau_f, tau_t = prepare_initial_params(X, K, W0, F0, S0, tau_f0, tau_t0, seed=seed)
    if mode == "shifts_only":
        W, F, S, tau_f, tau_t, hist = fit_shifts_only(
            X, W, F, S, tau_f, tau_t,
            iters_w=iters_w, iters_snap=iters_snap,
            snap_window_f=snap_window_f, snap_window_t=snap_window_t,
            ridge_w=ridge_w, verbose=verbose
        )
    elif mode == "staged":
        W, F, S, tau_f, tau_t, hist = fit_staged(
            X, W, F, S, tau_f, tau_t,
            iters_w=iters_w, iters_fs=iters_fs, iters_snap=iters_snap,
            inner_rank1_iters=inner_rank1_iters,
            snap_window_f=snap_window_f, snap_window_t=snap_window_t,
            ridge_w=ridge_w, verbose=verbose
        )
    else:
        W, F, S, tau_f, tau_t, hist = fit_monolithic(
            X, W, F, S, tau_f, tau_t,
            n_iters=n_iters, freeze_tau_iters=freeze_tau_iters, snap_every=snap_every,
            snap_window_f=snap_window_f, snap_window_t=snap_window_t,
            inner_rank1_iters=inner_rank1_iters, ridge_w=ridge_w, verbose=verbose
        )
    return W, F, S, tau_f, tau_t, hist

# --------------------------
# Convenience: xcorr seeding
# --------------------------
def seed_shifts_xcorr(X: np.ndarray, F0: np.ndarray, S0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N,F,T = X.shape
    K = F0.shape[0]
    tau_f0 = np.zeros((N,K))
    tau_t0 = np.zeros((N,K))
    def xcorr_tau(a,b):
        A = np.fft.fft(a)
        B = np.fft.fft(b)
        c = np.real(np.fft.ifft(np.conj(A)*B))
        k = int(np.argmax(c))
        km1=(k-1)%c.size
        kp1=(k+1)%c.size
        denom = (c[km1]-2*c[k]+c[kp1])
        delta = 0.0 if abs(denom)<1e-12 else 0.5*(c[km1]-c[kp1])/denom
        tau = (k+delta) % c.size
        if tau > c.size/2: tau -= c.size
        return float(tau)
    for n in range(N):
        for i in range(K):
            w_t = S0[i]/(S0[i].sum()+1e-12)
            w_f = F0[i]/(F0[i].sum()+1e-12)
            freq_profile = X[n] @ w_t
            time_profile = X[n].T @ w_f
            tau_f0[n,i] = xcorr_tau(F0[i], freq_profile)
            tau_t0[n,i] = xcorr_tau(S0[i], time_profile)
    return tau_f0, tau_t0

# --------------------------
# Utilities: noise, Doppler, triangulation, CRB, bootstrap
# --------------------------
def measure_snr_db(X_clean: np.ndarray, X_noisy: np.ndarray) -> float:
    num = np.linalg.norm(X_clean)
    den = np.linalg.norm(X_noisy - X_clean) + 1e-12
    return float(20.0 * np.log10(num / den))

def add_noise_by_snr_clipped_per_sensor(X_clean: np.ndarray, snr_db: float = 30.0,
                                        rng: Optional[np.random.Generator] = None, max_iter: int = 40) -> np.ndarray:
    rng = np.random.default_rng(0) if rng is None else rng
    X_noisy = np.empty_like(X_clean)
    for n in range(X_clean.shape[0]):
        noise = rng.normal(0.0, 1.0, size=X_clean[n].shape)
        target = (np.linalg.norm(X_clean[n]) + 1e-12) * (10.0 ** (-snr_db / 20.0))
        lo, hi = 0.0, (np.linalg.norm(X_clean[n]) / (np.linalg.norm(noise) + 1e-12)) * 10
        for _ in range(max_iter):
            a = 0.5*(lo+hi)
            X_try = np.clip(X_clean[n] + a*noise, 0.0, None)
            err = np.linalg.norm(X_try - X_clean[n])
            if err > target: hi = a
            else: lo = a
        X_noisy[n] = np.clip(X_clean[n] + lo*noise, 0.0, None)
    return X_noisy

def unwrap_bins(tau_bins: np.ndarray, F: int) -> np.ndarray:
    return ((tau_bins + F/2) % F) - F/2

def unwrap_frames(tau_frames: np.ndarray, T: int) -> np.ndarray:
    return ((tau_frames + T/2) % T) - T/2

def tau_to_doppler_hz(tau_f_bins: np.ndarray, bin_hz: float) -> np.ndarray:
    return tau_f_bins * float(bin_hz)

# ---- TDOA trilateration via Gauss–Newton ----
def tdoa_residual_and_jac(s: np.ndarray, P: np.ndarray, dt: np.ndarray, c: float, ref: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    r_list=[]
    J_list=[]
    pr = P[ref]
    for n in range(P.shape[0]):
        if n == ref: continue
        pn = P[n]
        vn = s - pn
        rn = np.linalg.norm(vn) + 1e-12
        vr = s - pr
        rr = np.linalg.norm(vr) + 1e-12
        r = rn - rr - c*dt[n]
        J = (vn/rn) - (vr/rr)
        r_list.append(r)
        J_list.append(J)
    return np.array(r_list), np.vstack(J_list)

def tdoa_gn(P: np.ndarray, dt: np.ndarray, c: float, ref: int = 0,
            x0: Optional[np.ndarray] = None, max_iter: int = 100, tol: float = 1e-10) -> Tuple[np.ndarray, Dict[str, object]]:
    if x0 is None: x0 = P.mean(axis=0).copy()
    s = x0.copy()
    for it in range(max_iter):
        r, J = tdoa_residual_and_jac(s, P, dt, c, ref=ref)
        H = J.T @ J
        g = J.T @ r
        try: step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError: step = -np.linalg.pinv(H) @ g
        s_new = s + step
        if np.linalg.norm(step) < tol: s = s_new; break
        s = s_new
    r, J = tdoa_residual_and_jac(s, P, dt, c, ref=ref)
    try: cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError: cov = np.linalg.pinv(J.T @ J)
    return s, {"iters": it+1, "res_norm": float(np.linalg.norm(r)), "cov_unit": cov}

def crb_covariance(P: np.ndarray, dt: np.ndarray, c: float, ref: int = 0,
                   sigma_tdoa_sec: Optional[np.ndarray] = None, s_eval: Optional[np.ndarray] = None):
    if sigma_tdoa_sec is None: 
        return None
    N = P.shape[0]
    if np.isscalar(sigma_tdoa_sec):
        sig = np.full(N, float(sigma_tdoa_sec))
    else:
        sig = np.array(sigma_tdoa_sec, dtype=float).reshape(-1)
        assert sig.size == N, "sigma_tdoa_sec must be scalar or length N"
    if s_eval is None:
        s_eval, _ = tdoa_gn(P, dt, c, ref=ref)
    r, J = tdoa_residual_and_jac(s_eval, P, dt, c, ref=ref)
    var_r = []
    for n in range(N):
        if n == ref: continue
        var_r.append((c**2) * (sig[n]**2 + sig[ref]**2))
    Sigma_r = np.diag(var_r)
    try:
        cov = np.linalg.inv(J.T @ np.linalg.inv(Sigma_r) @ J)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(J.T @ np.linalg.inv(Sigma_r) @ J)
    return cov

# ---- Bootstrap triangulation (arrays-only API) ----
def reestimate_tau_t_shifts_only(Xb: np.ndarray, W_seed: np.ndarray, F_seed: np.ndarray, S_seed: np.ndarray,
                                 tau_f_seed: np.ndarray, tau_t_seed: np.ndarray,
                                 iters_w: int = 30, iters_snap: int = 30) -> np.ndarray:
    K = F_seed.shape[0]
    Wb, Fb, Sb, taufb, tautb, _ = fit(
        Xb, K, mode="shifts_only",
        iters_w=iters_w, iters_snap=iters_snap,
        snap_window_f=3, snap_window_t=3,
        ridge_w=0.0, verbose=False,
        W0=W_seed, F0=F_seed, S0=S_seed, tau_f0=tau_f_seed, tau_t0=tau_t_seed
    )
    return tautb

def reestimate_tau_f_shifts_only(Xb: np.ndarray, W_seed: np.ndarray, F_seed: np.ndarray, S_seed: np.ndarray,
                                 tau_f_seed: np.ndarray, tau_t_seed: np.ndarray,
                                 iters_w: int = 20, iters_snap: int = 20) -> np.ndarray:
    K = F_seed.shape[0]
    Wb, Fb, Sb, taufb, tautb, _ = fit(
        Xb, K, mode="shifts_only",
        iters_w=iters_w, iters_snap=iters_snap,
        snap_window_f=3, snap_window_t=3,
        ridge_w=0.0, verbose=False,
        W0=W_seed, F0=F_seed, S0=S_seed, tau_f0=tau_f_seed, tau_t0=tau_t_seed
    )
    return taufb

def bootstrap_triangulation(W: np.ndarray, F: np.ndarray, S: np.ndarray, tau_f: np.ndarray, tau_t: np.ndarray,
                            P: np.ndarray, hop_sec: float, c: float, ref: int = 0,
                            B: int = 200, rng: Optional[np.random.Generator] = None,
                            snr_db: float = 30.0, base_X: Optional[np.ndarray] = None):
    rng = np.random.default_rng(0) if rng is None else rng
    N, K = W.shape
    Tframes = S.shape[1]

    tau_unw = unwrap_frames(tau_t, Tframes)
    dt_base = (tau_unw - tau_unw[ref:ref+1, :]) * hop_sec  # (N,K)

    pos_hat = np.zeros((K, 2))
    info_list = []
    for i in range(K):
        s_i, info_i = tdoa_gn(P, dt_base[:, i], c, ref=ref)
        pos_hat[i] = s_i
        info_list.append(info_i)

    pos_samps = np.zeros((B, K, 2))
    dt_samps  = np.zeros((B, N, K))
    base = reconstruct_from_params(W, F, S, tau_f, tau_t) if base_X is None else base_X
    for b in range(B):
        Xb = add_noise_by_snr_clipped_per_sensor(base, snr_db=snr_db, rng=rng)
        tau_b = reestimate_tau_t_shifts_only(Xb, W, F, S, tau_f, tau_t, iters_w=20, iters_snap=20)
        tau_unw_b = unwrap_frames(tau_b, Tframes)
        dt_b = (tau_unw_b - tau_unw_b[ref:ref+1, :]) * hop_sec   # (N,K)
        dt_samps[b] = dt_b
        for i in range(K):
            s_b, _ = tdoa_gn(P, dt_b[:, i], c, ref=ref, x0=pos_hat[i])
            pos_samps[b, i] = s_b

    covs = [np.cov(pos_samps[:, i, :].T) for i in range(K)]
    extras = {"dt_base": dt_base, "dt_samples": dt_samps, "pos_samples": pos_samps, "info": info_list}
    return pos_hat, covs, extras 


def tdoa_frames_from_positions(P, Spos, c, hop_sec, ref=0):
    N_ = P.shape[0] 
    K_ = Spos.shape[0]
    tau = np.zeros((N_, K_))
    for i in range(K_):
        d = np.linalg.norm(P - Spos[i], axis=1)
        dt = (d - d[ref]) / c
        tau[:, i] = dt / hop_sec
    return tau
# ================ ================ ================ ================ ================ ================
# ================ ================ ================ ================ ================ ================
# ================ ================ ================ ================ ================ ================

# -------------------------------
# Small helpers (diagnostics)
# -------------------------------
def _connectivity_from_S(S: np.ndarray) -> np.ndarray:
    """
    Build a connectivity matrix over time frames from S (K x T) by hard-assigning each
    frame to argmax_k S[k, t], then marking pairs of frames that share the same label.
    This mirrors the H-based connectivity used in classic NMFk diagnostics.
    """
    # S: (K, T)
    K, T = S.shape
    labels = np.argmax(S, axis=0)  # length T
    mat1 = np.tile(labels, (T, 1))
    mat2 = np.tile(labels[:, None], (1, T))
    return (mat1 == mat2).astype(float)

def _pac_score(C: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> float:
    """
    Proportion of Ambiguous Clustering (PAC) on the upper triangle (excluding diag).
    Lower is better (less ambiguity).
    """
    T = C.shape[0]
    iu = np.triu_indices(T, k=1)
    vals = C[iu]
    amb = np.logical_and(vals > lower, vals < upper).sum()
    tot = vals.size if vals.size > 0 else 1
    return float(amb) / float(tot)

def _default_workers(n_inits: int, requested: int) -> int:
    if requested is None or requested < 1:
        try:
            cpu = psutil.cpu_count(logical=True) or 1
        except Exception:
            cpu = os.cpu_count() or 1
        return min(n_inits, cpu)
    return min(n_inits, requested)

# -------------------------------
# The Shift-NMFk-TFGM class
# -------------------------------
class ShiftNMFkTFGM:
    """
    Shift-NMFk-TFGM (arrays-only) with TriNMFk-like UX:
      * fit_nmfk(X, Ks):    run your existing NMFk K-sweep on a 2D flattening of X
      * fit_shift_nmfk(...): run the shift solver across a list of Ks and n_inits
                             (parallelizable), pick best init per K by loss,
                             save per-K best results, and return a dict for all Ks.

    Parameters
    ----------
    experiment_name : str
        Name to prefix on output directories (mirrors TriNMFk).
    save_path : str
        Root folder where results are saved (mirrors TriNMFk).
    mode : {"monolithic", "staged", "shifts_only"}
        Which arrays-only fit loop to use (exactly your v12 modes).
    n_inits : int
        Number of random initializations per K.
    n_jobs : int
        Max concurrent inits. If <=0 or None, uses a sensible default (<= CPU count).
    verbose : bool
        Print progress/loss as in your arrays-only code.
    seed_base : int
        Base seed; actual init seed is seed_base + (init index).
    # v12 fit knobs:
    ridge_w, n_iters, freeze_tau_iters, snap_every, snap_window_f, snap_window_t, inner_rank1_iters,
    iters_w, iters_fs, iters_snap
        Passed straight through to `fit(...)` depending on the chosen mode.

    Notes
    -----
    * We do not prune X (pruning would alter the circular shift geometry).
    * Per-K artifacts are saved to: {save_path}/{experiment_name}_.../K={K}.npz
      with keys: W, F, S, tau_f, tau_t, errors (per-init relative error), best_idx, loss, relerr.
    * Optionally, consensus (over time frames) and PAC are returned in-memory per K.
    """

    def __init__(
        self,
        experiment_name: str = "ShiftNMFkTFGM",
        save_path: str = "ShiftNMFkTFGM",
        mode: str = "staged",
        n_inits: int = 10,
        n_jobs: Optional[int] = -1,
        verbose: bool = True,
        seed_base: int = 0,
        # ---- v12 knobs ----
        ridge_w: float = 0.0,
        # monolithic
        n_iters: int = 400,
        freeze_tau_iters: int = 120,
        snap_every: int = 5,
        snap_window_f: int = 6,
        snap_window_t: int = 6,
        inner_rank1_iters: int = 1,
        # staged / shifts-only
        iters_w: int = 60,
        iters_fs: int = 120,
        iters_snap: int = 80,
        # NMFk front (optional) defaults:
        nmfk_params: Optional[dict] = None,
    ):
        assert mode in ("monolithic", "staged", "shifts_only"), "Invalid mode"
        self.experiment_name = experiment_name
        self.save_path = save_path
        self.mode = mode
        self.n_inits = int(n_inits)
        self.n_jobs = _default_workers(self.n_inits, n_jobs if n_jobs is not None else -1)
        self.verbose = verbose
        self.seed_base = int(seed_base)

        # Store v12 knobs
        self.ridge_w = float(ridge_w)
        self.n_iters = int(n_iters)
        self.freeze_tau_iters = int(freeze_tau_iters)
        self.snap_every = int(snap_every)
        self.snap_window_f = int(snap_window_f)
        self.snap_window_t = int(snap_window_t)
        self.inner_rank1_iters = int(inner_rank1_iters)
        self.iters_w = int(iters_w)
        self.iters_fs = int(iters_fs)
        self.iters_snap = int(iters_snap)

        # NMFk (optional) front end for K selection
        self.nmfk = NMFk(**(nmfk_params or {
            "collect_output": False,
            "save_output": True,
            "predict_k": True,
            "consensus_mat": True,
            "calculate_pac": True
        }))

        # output folder
        name = f"{self.experiment_name}_{self.mode}_{self.n_inits}inits"
        self.save_path_full = os.path.join(self.save_path, name)
        Path(self.save_path_full).mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Front door to classic NMFk
    # ---------------------------
    def fit_nmfk(self, X: np.ndarray, Ks: List[int], note: str = ""):
        """
        Pass a 2D flattening of X (N*F x T) into your existing NMFk to do K selection
        via its standard consensus/PAC heuristics.

        Returns whatever NMFk.fit(...) returns (your code controls this).
        """
        assert X.ndim == 3, "X must be (N,F,T)"
        N, Fdim, Tdim = X.shape
        X2d = X.reshape(N * Fdim, Tdim)
        nmfk_results = self.nmfk.fit(X2d, Ks, self.experiment_name, note)
        # Optionally record where NMFk saved to:
        try:
            self.save_path_full = self.nmfk.save_path_full
        except Exception:
            pass
        return nmfk_results

    # ------------------------------------
    # Core: run across all K values
    # ------------------------------------
    def fit_shift_nmfk(
        self,
        X: np.ndarray,
        Ks: List[int],
        # Optional explicit seeds for init; typically you let the fitter randomize:
        W0: Optional[np.ndarray] = None,
        F0: Optional[np.ndarray] = None,
        S0: Optional[np.ndarray] = None,
        tau_f0: Optional[np.ndarray] = None,
        tau_t0: Optional[np.ndarray] = None,
        save_per_k: bool = True,
        compute_consensus_and_pac: bool = True,
    ) -> Dict[int, Dict[str, object]]:
        """
        For each K in Ks:
          - run n_inits independent fits (parallelizable),
          - pick the best by final loss,
          - compute per-init relative errors,
          - save best solution to disk (npz),
          - optionally compute an S-based consensus matrix & PAC,
          - return a dict keyed by K.

        Returns
        -------
        results_by_K : dict
            K -> {
               "W","F","S","tau_f","tau_t",          # best solution
               "loss","relerr","best_idx",           # scalars
               "errors": np.ndarray[n_inits],        # relerr per init
               "losses": np.ndarray[n_inits],        # loss per init
               "histories": List[List[float]],       # per-init loss traces
               "consensus": np.ndarray[T,T] (opt),
               "pac": float (opt),
               "save_file": str (opt)
            }
        """
        assert X.ndim == 3 and np.all(X >= 0), "X must be nonnegative with shape (N,F,T)"
        N, Fdim, Tdim = X.shape

        # Prepack fit kwargs once
        fit_kwargs = dict(
            mode=self.mode,
            ridge_w=self.ridge_w,
            # monolithic
            n_iters=self.n_iters,
            freeze_tau_iters=self.freeze_tau_iters,
            snap_every=self.snap_every,
            snap_window_f=self.snap_window_f,
            snap_window_t=self.snap_window_t,
            inner_rank1_iters=self.inner_rank1_iters,
            # staged / shifts-only
            iters_w=self.iters_w,
            iters_fs=self.iters_fs,
            iters_snap=self.iters_snap,
            verbose=self.verbose,
        )

        results_by_K: Dict[int, Dict[str, object]] = {}

        for K in Ks:
            if self.verbose:
                print(f"\n=== K = {K} | {self.n_inits} inits ===")

            # Per-init outputs
            Ws: List[np.ndarray] = [None] * self.n_inits  # type: ignore
            Fs: List[np.ndarray] = [None] * self.n_inits  # type: ignore
            Ss: List[np.ndarray] = [None] * self.n_inits  # type: ignore
            Tauf: List[np.ndarray] = [None] * self.n_inits  # type: ignore
            Taut: List[np.ndarray] = [None] * self.n_inits  # type: ignore
            losses = np.empty(self.n_inits, dtype=float)
            errors = np.empty(self.n_inits, dtype=float)
            histories: List[List[float]] = [None] * self.n_inits  # type: ignore

            # --- worker for one init ---
            def _one_init(init_idx: int):
                # Derive a deterministic seed per init
                seed = self.seed_base + int(init_idx)

                # If explicit init arrays were provided, pass them;
                # otherwise let arrays-only `fit(...)` build its own randomized init from seed.
                W0_i = None if W0 is None else W0.copy()
                F0_i = None if F0 is None else F0.copy()
                S0_i = None if S0 is None else S0.copy()
                tau_f0_i = None if tau_f0 is None else tau_f0.copy()
                tau_t0_i = None if tau_t0 is None else tau_t0.copy()

                W_i, F_i, S_i, tauf_i, taut_i, hist_i = fit(
                    X, K,
                    seed=seed,
                    W0=W0_i, F0=F0_i, S0=S0_i, tau_f0=tau_f0_i, tau_t0=tau_t0_i,
                    **fit_kwargs
                )
                loss_i, R_i = compute_loss(X, W_i, F_i, S_i, tauf_i, taut_i)
                return init_idx, W_i, F_i, S_i, tauf_i, taut_i, float(loss_i), float(R_i), list(hist_i)

            # --- run in parallel (threads; numpy releases the GIL in heavy ops) ---
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                futures = [ex.submit(_one_init, r) for r in range(self.n_inits)]
                for fut in concurrent.futures.as_completed(futures):
                    r, W_i, F_i, S_i, tauf_i, taut_i, loss_i, R_i, hist_i = fut.result()
                    Ws[r], Fs[r], Ss[r], Tauf[r], Taut[r] = W_i, F_i, S_i, tauf_i, taut_i
                    losses[r] = loss_i
                    errors[r] = R_i
                    histories[r] = hist_i
                    if self.verbose:
                        print(f"  init {r:02d}: loss={loss_i:.6e}  R={R_i:.6f}")

            # --- choose best init by minimal loss ---
            best_idx = int(np.argmin(losses))
            Wb, Fb, Sb = Ws[best_idx], Fs[best_idx], Ss[best_idx]
            tauf_b, taut_b = Tauf[best_idx], Taut[best_idx]
            loss_b, rel_b = float(losses[best_idx]), float(errors[best_idx])

            # --- optional consensus/PAC over inits on S ---
            consensus = None
            pac = None
            if compute_consensus_and_pac:
                T = Ss[0].shape[1]
                Csum = np.zeros((T, T), dtype=float)
                for r in range(self.n_inits):
                    Csum += _connectivity_from_S(Ss[r])
                consensus = Csum / float(self.n_inits)
                pac = _pac_score(consensus)

            # --- persist a compact per-K artifact (best solution + per-init errors) ---
            save_file = None
            if save_per_k:
                save_file = os.path.join(self.save_path_full, f"WFS_tau_K={K}.npz")
                np.savez_compressed(
                    save_file,
                    W=Wb, F=Fb, S=Sb, tau_f=tauf_b, tau_t=taut_b,
                    errors=errors, losses=losses, best_idx=best_idx,
                    loss=loss_b, relerr=rel_b
                )

            # --- record in-memory result ---
            results_by_K[K] = dict(
                W=Wb, F=Fb, S=Sb, tau_f=tauf_b, tau_t=taut_b,
                loss=loss_b, relerr=rel_b, best_idx=best_idx,
                errors=errors, losses=losses, histories=histories,
                consensus=consensus, pac=pac, save_file=save_file
            )

        return results_by_K
