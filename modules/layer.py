import logging

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn

from torch.nn.utils import parametrizations

from spd_learn import init as spd_init
from spd_learn.functional import bimap_transform, log_euclidean_geodesic


logger = logging.getLogger(__name__)


class ABiMap(nn.Module):
    """Adaptive-dimension BiMap layer (ABiMap).

    Args:
        n              : input SPD matrix dimension
        m_init         : initial output dimension
        m_max          : maximum output dimension (≤ n)
        tau            : sigmoid temperature (default 0.5)
        thresh_hi      : expand threshold on alpha (default 0.80)
        thresh_lo      : shrink threshold on alpha (default 0.20)
        parametrized   : if True, weight is parametrized on the Stiefel manifold (default True)
        orthogonal_map : {"cayley", "matrix_exp", "householder"}, orthogonal
                         parametrization method (default "cayley" if None)
        init_method    : {"kaiming_uniform", "orthogonal", "stiefel"}, weight
                         initialization method (default "kaiming_uniform")
        seed           : seed for the Stiefel initialization (default None)
        verbose        : print transitions (default False)
    """

    weight: nn.Parameter  # Type annotation for registered parameter

    def __init__(
        self,
        n: int,
        m_init: int,
        m_max: int,
        tau: float = 0.5,
        thresh_hi: float = 0.80,
        thresh_lo: float = 0.20,
        parametrized: bool = True,
        orthogonal_map: Optional[Literal["cayley", "matrix_exp", "householder"]] = None,
        init_method: Literal[
            "kaiming_uniform", "orthogonal", "stiefel"
        ] = "kaiming_uniform",
        seed: Optional[int] = None,
        device=None,
        dtype=None,
        verbose: bool = False,
    ):
        super().__init__()
        assert 1 <= m_init < m_max <= n

        if init_method not in ["kaiming_uniform", "orthogonal", "stiefel"]:
            raise ValueError(
                f"Unknown init_method: '{init_method}'. Choose from "
                "'kaiming_uniform', 'orthogonal', 'stiefel'."
            )

        if not parametrized and orthogonal_map is not None:
            raise ValueError("orthogonal_map is only used when parametrized is True")

        self.n = n
        self.m_max = m_max
        self.tau = tau
        self.thresh_hi = thresh_hi
        self.thresh_lo = thresh_lo
        self.verbose = verbose
        self.m = m_init

        self.parametrized = parametrized
        self.orthogonal_map = orthogonal_map
        self.init_method = init_method
        self.seed = seed

        self.beta = nn.Parameter(
            torch.tensor(0.0, device=device, dtype=dtype)
        )
        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.empty(self.n, self.m_max, device=device, dtype=dtype),
                requires_grad=True,
            ),
        )

        self.reset_parameters()

        if self.parametrized:
            parametrizations.orthogonal(
                module=self,
                name="weight",
                orthogonal_map=(
                    "cayley" if self.orthogonal_map is None else self.orthogonal_map
                ),
            )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize weight matrix according to the specified method."""
        if self.init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.weight, a=0.01)
        elif self.init_method == "orthogonal":
            nn.init.orthogonal_(self.weight)
        elif self.init_method == "stiefel":
            spd_init.stiefel_(self.weight, seed=self.seed)
        else:
            raise ValueError(
                f"Internal error: Invalid init_method '{self.init_method}'"
            )
        self.beta.zero_()

    def pad_spd(self, S, target_size):
        """Dimensionality transcending: pad an SPD matrix to a fixed size."""
        m   = S.shape[-1]
        q   = target_size - m
        out = torch.zeros(*S.shape[:-2], target_size, target_size, device=S.device, dtype=S.dtype)
        out[..., :m, :m] = S
        for i in range(q):
            out[..., m + i, m + i] = 1.0
        return out

    def set_alpha(self, value):
        self.beta.data = torch.tensor(
            self.tau * np.log(value / (1 - value)),
            dtype=self.beta.dtype, device=self.beta.device
        )

    @torch.no_grad()
    def init_candidate_column(self):
        """Initialize a new candidate column via Gram-Schmidt orthonormalization,
        writing through the parametrization's right_inverse.
        """
        W_active = self.weight.data[:, :self.m]
        w_cand   = torch.randn(self.n, device=self.weight.device, dtype=self.weight.dtype)
        w_cand   = w_cand - W_active @ (W_active.T @ w_cand)
        w_cand   = w_cand / w_cand.norm()

        if self.parametrized:
            target = self.weight.data.clone()
            target[:, self.m] = w_cand

            parametrization = self.parametrizations.weight[0]  
            new_original = parametrization.right_inverse(target)
            self.parametrizations.weight.original.data.copy_(new_original)
        else:
            self.weight.data[:, self.m] = w_cand

    def step(self):
        """Check alpha against the transition thresholds and update m accordingly."""
        alpha = self.get_alpha()

        # Expand
        if alpha >= self.thresh_hi and self.m < self.m_max - 1:
            self.m += 1
            self.init_candidate_column()
            self.set_alpha(0.3)
            if self.verbose:
                print(f"  [EXPAND] m → {self.m} | alpha = {self.get_alpha():.3f}")

        # Shrink
        elif alpha <= self.thresh_lo and self.m > 1:
            self.m -= 1
            self.set_alpha(0.7)
            if self.verbose:
                print(f"  [SHRINK] m → {self.m} | alpha = {self.get_alpha():.3f}")

    def forward(self, X):

        if self.training:
            with torch.no_grad():
                self.step()

            alpha    = torch.sigmoid(self.beta / self.tau)
            m        = self.m

            p_hi     = bimap_transform(X, self.weight[:, :m])
            p_lo     = bimap_transform(X, self.weight[:, :m - 1])
            # p_lo     = p_hi[..., :m - 1, :m - 1]

            p_lo_pad = self.pad_spd(p_lo, m)
            p    = log_euclidean_geodesic(p_lo_pad, p_hi, alpha)
            return self.pad_spd(p, self.m_max)

        else:
            m_f = self.get_final_m()
            p  = bimap_transform(X, self.weight[:, :m_f])
            return self.pad_spd(p, self.m_max)

    def get_alpha(self):   return torch.sigmoid(self.beta / self.tau).item()
    def get_m(self):       return self.m
    def get_final_m(self): return self.m + 1 if self.get_alpha() >= 0.5 else self.m