import copy
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional as sj_functional

from snn_modules_alternative_core import SpikingNeuralNet


class SpikingModel(nn.Module):
    """
    Wrapper that only does:
    - day transform
    - optional patching
    - call original SpikingNeuralNet

    No extra head. No behavior change.
    """
    def __init__(
        self,
        cfg,
        neural_dim: int,
        n_days: int,
        n_classes: int,
        input_dropout: float = 0.0,
        patch_size: int = 0,
        patch_stride: int = 0,
        debug: bool = False,
        collect_day_stats: bool = False,
        stats_every: int = 1000,
    ):
        super().__init__()

        self.cfg = cfg
        self.neural_dim = neural_dim
        self.n_days = n_days
        self.n_classes = n_classes
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.debug = debug
        self.stats_every = stats_every
        self._step = 0

        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.last_day_stats = {}
        self.last_matrix_stats = {}

        backbone_cfg = copy.deepcopy(cfg)

        if self.patch_size > 0:
            patched_input_size = int(self.neural_dim * self.patch_size)
            if "multi_scale_snn" in backbone_cfg.model.setup:
                backbone_cfg.model.setup.multi_scale_snn.input_size = patched_input_size

        self.backbone = SpikingNeuralNet(backbone_cfg)

    @torch.no_grad()
    def _compute_tensor_day_stats(self, x, day_idx, name):
        """
        x: [B, T, D]
        day_idx: [B]
        """
        stats = {}
        pairwise = {}

        unique_days = torch.unique(day_idx).tolist()
        day_means = {}
        day_covs = {}

        for d in unique_days:
            mask = (day_idx == d)
            xd = x[mask]                          # [Bd, T, D]
            flat = xd.reshape(-1, xd.shape[-1])  # [Bd*T, D]

            mean = flat.mean(dim=0)
            std = flat.std(dim=0)
            xc = flat - mean
            cov = (xc.T @ xc) / max(flat.shape[0] - 1, 1)

            day_means[int(d)] = mean
            day_covs[int(d)] = cov

            stats[int(d)] = {
                "mean_mean": mean.mean().item(),
                "mean_std": mean.std().item(),
                "std_mean": std.mean().item(),
                "std_std": std.std().item(),
                "cov_trace": torch.trace(cov).item(),
                "cov_fro": torch.norm(cov, p="fro").item(),
            }

        days = sorted(day_means.keys())
        for i in range(len(days)):
            for j in range(i + 1, len(days)):
                di, dj = days[i], days[j]
                pairwise[(di, dj)] = {
                    "mean_l2": torch.norm(day_means[di] - day_means[dj], p=2).item(),
                    "cov_fro_dist": torch.norm(day_covs[di] - day_covs[dj], p="fro").item(),
                }

        return stats, pairwise

    @torch.no_grad()
    def print_day_matrix_simple_stats(self, days=None):
        if days is None:
            days = torch.unique(self.current_day_idx).tolist()

        for d in days:
            W = self.day_weights[d].detach()
            b = self.day_biases[d].detach()

            diag = torch.diag(W)
            offdiag = W - torch.diag(diag)

            print(
                f"day {d}: "
                f"max_offdiag={offdiag.abs().max().item():.6f}, "
                f"mean_diag_dev={(diag - 1.0).abs().mean().item():.6f}, "
                f"bias_l2={b.norm().item():.6f}, "
                f"bias_mean_abs={b.abs().mean().item():.6f}, "
                f"bias_max_abs={b.abs().max().item():.6f}"
            )

    @torch.no_grad()
    def _compute_matrix_stats(self):
        stats = {}
        eye = torch.eye(self.neural_dim, device=self.day_weights[0].device)

        for d in range(self.n_days):
            W = self.day_weights[d].detach()
            b = self.day_biases[d].detach()

            delta = W - eye
            s = torch.linalg.svdvals(W)

            diag_energy = torch.norm(torch.diag(torch.diag(W)), p="fro").item() ** 2
            full_energy = torch.norm(W, p="fro").item() ** 2
            offdiag_ratio = max(full_energy - diag_energy, 0.0) / max(full_energy, 1e-12)

            stats[d] = {
                "W_fro": torch.norm(W, p="fro").item(),
                "W_minus_I_fro": torch.norm(delta, p="fro").item(),
                "bias_l2": torch.norm(b, p=2).item(),
                "singular_max": s.max().item(),
                "singular_min": s.min().item(),
                "singular_mean": s.mean().item(),
                "stable_rank": (s.sum().item() ** 2) / max((s ** 2).sum().item(), 1e-12),
                "offdiag_ratio": offdiag_ratio,
            }

        self.last_matrix_stats = stats
        return stats

    def apply_day_transform(self, x, day_idx):
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        return x

    def apply_patching(self, x):
        assert self.patch_size > 0 and self.patch_stride > 0

        x = x.unsqueeze(1)                    # [B,1,T,D]
        x = x.permute(0, 3, 1, 2)            # [B,D,1,T]
        x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
        x_unfold = x_unfold.squeeze(2)       # [B,D,T',patch]
        x_unfold = x_unfold.permute(0, 2, 3, 1)  # [B,T',patch,D]
        x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)
        return x

    def get_backbone_input(self, x, day_idx):
        """
        x: [B, T, neural_dim]
        returns: [T, B, C_backbone_in]
        """
        x = self.apply_day_transform(x, day_idx)

        if self.patch_size > 0:
            x = self.apply_patching(x)

        x = x.permute(1, 0, 2).contiguous()   # [T, B, C]
        return x

    def project_features_for_lag_prior(self, x, day_idx):
        """
        x: [B, T, neural_dim]
        returns: [T, B, H] after backbone input projection
        """
        x = self.get_backbone_input(x, day_idx)
        z = self.backbone.input_proj(x)
        return z

    @torch.no_grad()
    def collect_stats(self, x, day_idx):
        stats = {}

        raw_stats, raw_pairwise = self._compute_tensor_day_stats(x, day_idx, "raw")

        x_day = self.apply_day_transform(x, day_idx)
        day_stats, day_pairwise = self._compute_tensor_day_stats(x_day, day_idx, "day")

        z = self.project_features_for_lag_prior(x, day_idx).permute(1, 0, 2).contiguous()
        proj_stats, proj_pairwise = self._compute_tensor_day_stats(z, day_idx, "proj")

        mat_stats = self._compute_matrix_stats()

        stats["raw"] = raw_stats
        stats["day"] = day_stats
        stats["proj"] = proj_stats
        stats["raw_pairwise"] = raw_pairwise
        stats["day_pairwise"] = day_pairwise
        stats["proj_pairwise"] = proj_pairwise
        stats["matrices"] = mat_stats

        self.last_day_stats = stats
        return stats

    def _print_stats(self):
        s = self.last_day_stats
        if not s:
            return

        print(f"[step {self._step}] pairwise day mismatch")
        for stage in ["raw_pairwise", "day_pairwise", "proj_pairwise"]:
            print(stage)
            for k, v in s[stage].items():
                print(
                    f"  {k}: mean_l2={v['mean_l2']:.4f}, "
                    f"cov_fro_dist={v['cov_fro_dist']:.4f}"
                )

        print(f"[step {self._step}] per-day tensor stats")
        for stage in ["raw", "day", "proj"]:
            print(stage)
            for d, v in s[stage].items():
                print(
                    f"  day {d}: "
                    f"mean={v['mean_mean']:.4f}, "
                    f"mean_std={v['mean_std']:.4f}, "
                    f"std={v['std_mean']:.4f}, "
                    f"std_std={v['std_std']:.4f}, "
                    f"cov_trace={v['cov_trace']:.4f}, "
                    f"cov_fro={v['cov_fro']:.4f}"
                )

        print(f"[step {self._step}] matrix stats")
        for d, v in s["matrices"].items():
            print(
                f"  day {d}: "
                f"W-I={v['W_minus_I_fro']:.4f}, "
                f"offdiag={v['offdiag_ratio']:.4f}, "
                f"stable_rank={v['stable_rank']:.2f}, "
                f"smax={v['singular_max']:.4f}, "
                f"smin={v['singular_min']:.4f}, "
                f"bias={v['bias_l2']:.4f}"
            )

    def forward(self, x, day_idx):
        """
        x: [B, T, neural_dim]
        returns: [B, T', n_classes]
        """
        self._step += 1

        if self._step % self.stats_every == 0:
            with torch.no_grad():
                self.collect_stats(x.detach(), day_idx.detach())
                self._print_stats()
        self.current_day_idx = day_idx

        if self._step % 1000 == 0:
            self.print_day_matrix_simple_stats()

        x = self.apply_day_transform(x, day_idx)

        if self.patch_size > 0:
            x = self.apply_patching(x)

        sj_functional.reset_net(self.backbone)

        x = x.permute(1, 0, 2).contiguous()   # [T,B,C]
        logits = self.backbone(x)             # [T,B,Cout]
        logits = logits.permute(1, 0, 2).contiguous()

        return logits