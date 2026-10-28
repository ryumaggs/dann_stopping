"""
Transfer Score — Yang et al., "Can We Evaluate Domain Adaptation Models
Without Target-Domain Labels?", ICLR 2024 (arXiv: 2305.18712)

Three components:
  U  — Classifier Uniformity       (from classifier weight vectors only)
  H  — Hopkins Statistic           (clustering tendency of target embeddings)
  M  — Mutual Information          (prediction confidence + diversity on target)

Transfer Score:  T = -U + H + |M| / ln(K)

Stopping criterion: compute coefficient of variation of T over a sliding
window; stop when it drops below threshold ζ.
"""

import math
import numpy as np
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Classifier Uniformity
# ---------------------------------------------------------------------------

def classifier_uniformity(weights: np.ndarray) -> float:
    """
    Measure how evenly the classifier weight vectors partition the feature space.

    Parameters
    ----------
    weights : np.ndarray, shape [K, d]
        The weight matrix of the final linear (classifier) layer.
        Each row is the weight vector for one class.

    Returns
    -------
    float
        Uniformity U.  Lower = more balanced = better transferability.
    """
    K = weights.shape[0]
    if K < 2:
        return 0.0

    # Unit-normalise each row
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    W = weights / norms                          # [K, d]

    # Pairwise cosine similarities → angles
    cos_sim = W @ W.T                            # [K, K]
    cos_sim = np.clip(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
    angles  = np.arccos(cos_sim)                 # [K, K]  (radians)

    # Ideal angle: arccos(-1 / (K-1))
    theta_K = math.acos(-1.0 / (K - 1))

    # Mean squared deviation over off-diagonal entries
    mask = ~np.eye(K, dtype=bool)
    U = float(np.mean((angles[mask] - theta_K) ** 2))
    return U


# ---------------------------------------------------------------------------
# 2. Hopkins Statistic
# ---------------------------------------------------------------------------

def _sq_dists_to_nearest(query: np.ndarray,
                          reference: np.ndarray,
                          exclude_self: bool = False) -> np.ndarray:
    """
    For each point in `query`, find the squared distance to its nearest
    neighbour in `reference`.  If exclude_self=True the query IS reference
    (self-distances are ignored).
    """
    # Brute-force — fine for the small m used in Hopkins
    dists_sq = np.sum((query[:, None, :] - reference[None, :, :]) ** 2, axis=-1)
    # shape [m_q, m_r]

    if exclude_self:
        np.fill_diagonal(dists_sq, np.inf)

    return dists_sq.min(axis=1)   # [m_q]


def hopkins_statistic(embeddings: np.ndarray,
                      m: int = 100,
                      rng: Optional[np.random.Generator] = None) -> float:
    """
    Hopkins statistic for the target-domain feature embeddings.

    Parameters
    ----------
    embeddings : np.ndarray, shape [N, d]
        Feature vectors of target-domain samples (output of g).
    m : int
        Sub-sample size.  Paper uses m ≪ N.  Default 100.
    rng : np.random.Generator, optional
        For reproducibility.

    Returns
    -------
    float
        H ∈ (0, 1).  H ~ 1 → strong clustering (good).
                     H ~ 0.5 → random scatter (poor adaptation).
    """
    if rng is None:
        rng = np.random.default_rng()

    N, d = embeddings.shape
    m = min(m, N // 2)          # safety: m must be < N

    # Sample R: m points drawn without replacement from the embeddings
    idx = rng.choice(N, size=m, replace=False)
    R   = embeddings[idx]       # [m, d]

    # Sample U: m points uniformly inside the axis-aligned bounding box
    lo  = embeddings.min(axis=0)
    hi  = embeddings.max(axis=0)
    rng_range = hi - lo
    rng_range = np.where(rng_range < 1e-12, 1e-12, rng_range)
    U_pts = rng.random((m, d)) * rng_range + lo   # [m, d]

    # Distances
    u_sq = _sq_dists_to_nearest(U_pts, R, exclude_self=False)   # [m]
    w_sq = _sq_dists_to_nearest(R,     R, exclude_self=True)     # [m]

    # Hopkins formula uses distance^d  (d = embedding dimension).
    # To avoid overflow at high d, compute in log-space:
    #   log(u_d) = (d/2) * log(u_sq),   then use log-sum-exp.
    half_d = d / 2.0
    log_u  = half_d * np.log(np.maximum(u_sq, 1e-300))   # [m]
    log_w  = half_d * np.log(np.maximum(w_sq, 1e-300))   # [m]

    # log-sum-exp for numerical stability
    all_logs  = np.concatenate([log_u, log_w])
    shift     = all_logs.max()
    log_sum_u = shift + np.log(np.sum(np.exp(log_u - shift)))
    log_sum_w = shift + np.log(np.sum(np.exp(log_w - shift)))

    # H = sum(u^d) / (sum(u^d) + sum(w^d))
    #   = 1 / (1 + exp(log_sum_w - log_sum_u))
    H = float(1.0 / (1.0 + math.exp(float(log_sum_w - log_sum_u))))
    return H


# ---------------------------------------------------------------------------
# 3. Mutual Information
# ---------------------------------------------------------------------------

def mutual_information(probs: np.ndarray, eps: float = 1e-12) -> float:
    """
    Mutual information between the input and the model's prediction on
    target-domain samples.

    MI = H(E_x[p(y|x)]) - E_x[H(p(y|x))]
       = marginal entropy  -  mean conditional entropy

    Parameters
    ----------
    probs : np.ndarray, shape [N, K]
        Softmax probabilities for each target sample.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    float  M
    """
    probs = np.clip(probs, eps, 1.0)

    # Marginal distribution  p̄(y)
    p_bar        = probs.mean(axis=0)              # [K]
    H_marginal   = -float(np.sum(p_bar * np.log(p_bar)))

    # Mean conditional entropy
    H_cond_each  = -np.sum(probs * np.log(probs), axis=1)   # [N]
    H_cond       = float(H_cond_each.mean())

    return H_marginal - H_cond


# ---------------------------------------------------------------------------
# 4. Transfer Score
# ---------------------------------------------------------------------------

def transfer_score(classifier_weights: np.ndarray,
                   target_embeddings:  np.ndarray,
                   target_probs:       np.ndarray,
                   hopkins_m:          int = 100,
                   rng: Optional[np.random.Generator] = None
                   ) -> dict:
    """
    Compute the Transfer Score and its three components.

    Parameters
    ----------
    classifier_weights : np.ndarray, shape [K, d]
        Weight matrix of the linear classifier head.
    target_embeddings : np.ndarray, shape [N, d]
        Feature vectors of unlabelled target samples (output of g).
    target_probs : np.ndarray, shape [N, K]
        Softmax probabilities produced by the full model on target samples.
    hopkins_m : int
        Sub-sample size for the Hopkins statistic (default 100).
    rng : np.random.Generator, optional

    Returns
    -------
    dict with keys:
        'transfer_score'  — T  (higher = better adapted model)
        'uniformity'      — U  (lower = more balanced classifier)
        'hopkins'         — H  (higher = clearer clusters)
        'mutual_info'     — M  (higher magnitude = more confident + diverse)
    """
    K = classifier_weights.shape[0]

    U = classifier_uniformity(classifier_weights)
    H = hopkins_statistic(target_embeddings, m=hopkins_m, rng=rng)
    M = mutual_information(target_probs)

    T = -U + H + abs(M) / math.log(K)

    return {
        'transfer_score': T,
        'uniformity':     U,
        'hopkins':        H,
        'mutual_info':    M,
    }


# ---------------------------------------------------------------------------
# 5. Saturation-Based Stopping Criterion
# ---------------------------------------------------------------------------

class TransferScoreMonitor:
    """
    Tracks Transfer Score over training and signals when to stop.

    The saturation level at epoch m is the coefficient of variation (CV)
    of the TS over a sliding window of size `tau`:

        S_m = std(T_{m-tau+1}, ..., T_m) / mean(...)

    When S_m < zeta, training has saturated.  The recommended checkpoint
    is the epoch with the highest TS inside that window.

    Usage
    -----
    monitor = TransferScoreMonitor(tau=3, zeta=0.01)

    for epoch in range(max_epochs):
        scores = transfer_score(W, emb, probs)
        result = monitor.step(epoch, scores['transfer_score'])

        print(f"Epoch {epoch}: T={scores['transfer_score']:.4f}  "
              f"U={scores['uniformity']:.4f}  "
              f"H={scores['hopkins']:.4f}  "
              f"M={scores['mutual_info']:.4f}")

        if result['saturated']:
            best = result['best_epoch']
            print(f"Saturated. Best checkpoint: epoch {best}")
            break
    """

    def __init__(self, tau: int = 3, zeta: float = 0.01):
        """
        Parameters
        ----------
        tau : int
            Sliding window length (paper default: 3).
        zeta : float
            CV threshold below which training is considered saturated
            (paper default: 0.01).
        """
        self.tau    = tau
        self.zeta   = zeta
        self._window: deque = deque(maxlen=tau)   # (epoch, T) pairs
        self.history: list  = []                  # all (epoch, T) pairs

    def step(self, epoch: int, T: float) -> dict:
        """
        Register the Transfer Score at `epoch` and check saturation.

        Returns
        -------
        dict with keys:
            'saturated'   — bool
            'best_epoch'  — int or None  (set only when saturated)
            'saturation'  — float  (current CV, or None if window not full)
        """
        self._window.append((epoch, T))
        self.history.append((epoch, T))

        if len(self._window) < self.tau:
            return {'saturated': False, 'best_epoch': None, 'saturation': None}

        values = np.array([t for _, t in self._window])
        mu     = values.mean()
        sigma  = values.std()

        cv = float(sigma / abs(mu)) if abs(mu) > 1e-12 else float('inf')

        if cv < self.zeta:
            best_epoch = max(self._window, key=lambda x: x[1])[0]
            return {'saturated': True, 'best_epoch': best_epoch, 'saturation': cv}

        return {'saturated': False, 'best_epoch': None, 'saturation': cv}

    def best_so_far(self) -> tuple:
        """Return (epoch, T) of the highest Transfer Score seen so far."""
        return max(self.history, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# 6. DANN-specific extraction and monitor
# ---------------------------------------------------------------------------

def extract_from_dann(dann_model, Xt_tensor):
    """
    Extract everything needed for the Transfer Score from a DANNModel instance.

    Compatible with the DANNModel architecture in your codebase:
        dann_model.feature_extractor              — G_f
        dann_model.label_predictor                — G_y (LabelPredictor)
        dann_model.label_predictor.output_layer   — final nn.Linear [K, d]

    Note: dann_model.forward() returns (y_logits, d_logits) — this function
    calls sub-networks directly to avoid unpacking that tuple.

    The embeddings returned are G_f outputs, so U (uniformity) and H (Hopkins)
    are measured in the same d-dimensional feature space.

    Parameters
    ----------
    dann_model : DANNModel
    Xt_tensor : torch.Tensor, shape [N, input_dim]
        Unlabelled target data already on the correct device.
        This is Xt_t as built inside DANN.fit().

    Returns
    -------
    embeddings : np.ndarray  [N, d]        — G_f output
    probs      : np.ndarray  [N, K]        — softmax of G_y
    weights    : np.ndarray  [K, d_final]  — output_layer.weight of G_y
    """
    import torch
    import torch.nn.functional as F

    dann_model.eval()
    with torch.no_grad():
        embeddings = dann_model.feature_extractor(Xt_tensor)       # [N, d]
        logits     = dann_model.label_predictor(embeddings)         # [N, K]
        probs      = F.softmax(logits, dim=1)                       # [N, K]

    # output_layer is always the final nn.Linear in LabelPredictor,
    # shape [K, hidden_size] when hidden_dims=[], or [K, hidden_dims[-1]] otherwise
    weights = dann_model.label_predictor.output_layer.weight.detach()  # [K, d]

    return (
        embeddings.cpu().numpy(),
        probs.cpu().numpy(),
        weights.cpu().numpy(),
    )


class DANNTransferScoreMonitor:
    """
    Transfer Score monitor designed for use inside DANN.fit().

    Wraps TransferScoreMonitor with DANN-aware extraction so you only need
    to pass the model and the target tensor — no DataLoader required.

    Usage inside DANN.fit()
    -----------------------
    # After building Xt_t, before the epoch loop:
    ts_monitor = DANNTransferScoreMonitor(tau=3, zeta=0.01, hopkins_m=100)

    # Inside the epoch loop, after the inner batch loop:
    ts_result = ts_monitor.step(epoch, self.model, Xt_t)

    # Log to tensorboard:
    if ts_result['computed']:
        writer.add_scalar('TS/transfer_score', ts_result['transfer_score'], epoch)
        writer.add_scalar('TS/uniformity',     ts_result['uniformity'],     epoch)
        writer.add_scalar('TS/hopkins',        ts_result['hopkins'],        epoch)
        writer.add_scalar('TS/mutual_info',    ts_result['mutual_info'],    epoch)
        writer.add_scalar('TS/CV',             ts_result['saturation'] or 0, epoch)

    # Check stopping:
    if ts_result['saturated']:
        self.model.load_state_dict(ts_monitor.best_state_dict())
        break

    Parameters
    ----------
    tau : int
        Sliding window length (paper default: 3).
    zeta : float
        CV threshold for saturation (paper default: 0.01).
    hopkins_m : int
        Sub-sample size for Hopkins statistic (default: 100).
    compute_every : int
        Evaluate TS every N epochs.  Useful to reduce overhead when
        target data is large.  Default 1 = every epoch.
    save_best_state : bool
        Keep a copy of the state dict at the epoch with the highest TS,
        so it can be restored on saturation without save_every_epoch=True.
    """

    def __init__(self,
                 tau: int = 3,
                 zeta: float = 0.01,
                 hopkins_m: int = 100,
                 compute_every: int = 1,
                 save_best_state: bool = True):
        self.hopkins_m       = hopkins_m
        self.compute_every   = compute_every
        self.save_best_state = save_best_state
        self._monitor        = TransferScoreMonitor(tau=tau, zeta=zeta)
        self._rng            = np.random.default_rng()
        self._best_state     = None
        self._best_ts        = -float('inf')

    def step(self, epoch: int, dann_model, Xt_tensor) -> dict:
        """
        Evaluate TS at this epoch and check saturation.

        Parameters
        ----------
        epoch : int
        dann_model : DANNModel  (self.model inside DANN.fit)
        Xt_tensor : torch.Tensor  (Xt_t inside DANN.fit)

        Returns
        -------
        dict with keys:
            'computed'        — bool    False on skipped epochs
            'transfer_score'  — float or None
            'uniformity'      — float or None
            'hopkins'         — float or None
            'mutual_info'     — float or None
            'saturation'      — float or None  (current window CV)
            'saturated'       — bool
            'best_epoch'      — int or None    (set when saturated)
        """
        _skip = {
            'computed': False, 'transfer_score': None, 'uniformity': None,
            'hopkins': None,   'mutual_info': None,    'saturation': None,
            'saturated': False, 'best_epoch': None,
        }

        if epoch % self.compute_every != 0:
            return _skip

        embeddings, probs, weights = extract_from_dann(dann_model, Xt_tensor)

        scores = transfer_score(
            classifier_weights=weights,
            target_embeddings=embeddings,
            target_probs=probs,
            hopkins_m=self.hopkins_m,
            rng=self._rng,
        )
        T = scores['transfer_score']

        if self.save_best_state and T > self._best_ts:
            self._best_ts    = T
            self._best_state = {k: v.clone()
                                for k, v in dann_model.state_dict().items()}

        sat_result = self._monitor.step(epoch, T)

        return {**scores, **sat_result, 'computed': True}

    def best_state_dict(self) -> Optional[dict]:
        """State dict from the epoch with the highest TS seen so far."""
        return self._best_state

    def history(self) -> list:
        """List of (epoch, T) for all evaluated epochs."""
        return self._monitor.history


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(42)

    K, d, N = 10, 256, 500

    print("=" * 60)
    print("Transfer Score — smoke test with synthetic data")
    print("=" * 60)

    # --- Simulate a WELL-adapted model ---
    centers  = rng.standard_normal((K, d)) * 5
    labels   = rng.integers(0, K, size=N)
    emb_good = centers[labels] + rng.standard_normal((N, d)) * 0.3

    probs_good = np.zeros((N, K))
    probs_good[np.arange(N), labels] = 0.95
    probs_good += 0.005
    probs_good /= probs_good.sum(axis=1, keepdims=True)

    W_good = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)

    result_good = transfer_score(W_good, emb_good, probs_good, rng=rng)
    print("\nWell-adapted model:")
    for k, v in result_good.items():
        print(f"  {k:<20s} = {v:.4f}")

    # --- Simulate a POORLY-adapted model (collapsed) ---
    emb_bad   = rng.standard_normal((N, d))
    probs_bad = np.ones((N, K)) / K
    W_bad     = rng.standard_normal((K, d)) * 0.01

    result_bad = transfer_score(W_bad, emb_bad, probs_bad, rng=rng)
    print("\nPoorly-adapted (collapsed) model:")
    for k, v in result_bad.items():
        print(f"  {k:<20s} = {v:.4f}")

    print(f"\n  TS good > TS bad: {result_good['transfer_score']:.4f} "
          f"> {result_bad['transfer_score']:.4f}  "
          f"→ {'✓' if result_good['transfer_score'] > result_bad['transfer_score'] else '✗'}")

    # --- Saturation stopping criterion ---
    print("\n" + "=" * 60)
    print("Saturation stopping (simulated trajectory)")
    print("=" * 60)

    monitor = TransferScoreMonitor(tau=3, zeta=0.01)
    ts_trajectory = [0.2, 0.35, 0.50, 0.65, 0.78, 0.88, 0.93, 0.95, 0.955, 0.958, 0.957]

    for epoch, T_fake in enumerate(ts_trajectory):
        result = monitor.step(epoch, T_fake)
        sat    = result['saturation']
        sat_str = f"{sat:.5f}" if sat is not None else "   N/A"
        flag  = " ← STOP" if result['saturated'] else ""
        print(f"  Epoch {epoch:2d}  T={T_fake:.3f}  CV={sat_str}{flag}")
        if result['saturated']:
            print(f"\n  Saturated at epoch {epoch}. "
                  f"Best checkpoint: epoch {result['best_epoch']}")
            break

    # --- DANNTransferScoreMonitor with a minimal mock DANNModel ---
    print("\n" + "=" * 60)
    print("DANNTransferScoreMonitor — mock DANNModel test")
    print("=" * 60)

    # Minimal stubs that match the attribute names your DANN code uses
    class _MockOutputLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(K, d))
    class _MockLabelPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = _MockOutputLayer()
        def forward(self, h):
            return h @ self.output_layer.weight.T
    class _MockFeatureExtractor(nn.Module):
        def forward(self, x):
            return x
    class _MockDANNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = _MockFeatureExtractor()
            self.label_predictor   = _MockLabelPredictor()

    mock_model  = _MockDANNModel()
    Xt_tensor   = torch.randn(N, d)   # mock target data, already on cpu

    ts_mon = DANNTransferScoreMonitor(tau=3, zeta=0.01, hopkins_m=100)

    for epoch in range(5):
        result = ts_mon.step(epoch, mock_model, Xt_tensor)
        T      = result['transfer_score']
        cv     = result['saturation']
        print(f"  Epoch {epoch}  T={T:.4f}  CV={'N/A' if cv is None else f'{cv:.5f}'}"
              f"  saturated={result['saturated']}")

    print(f"\n  best_state_dict present: {ts_mon.best_state_dict() is not None}  ✓")
    print(f"  history length: {len(ts_mon.history())}  ✓")