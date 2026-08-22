"""Frozen E7 Missing/Fewer module: R0 spatial constraints plus composition.

The implementation is deliberately a direct engineering extraction of the
validated E7 experiment: 12x12 structural tokens, R0 registration, six spatial
energies, fixed 64x32 set descriptor, 100-normal Ledoit-Wolf world, empirical
normal tails, and logsumexp fusion.  No memory bank or learned score fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import balanced_accuracy_score, f1_score

from modules.gsr_md_v2 import StructuralPrior
from modules.gsr_registration import GROUPS, RegistrationGSR

GRID, TOKENS, DIM = 12, 144, 192
ENERGY_NAMES = ("pred", "node", "relation", "local", "middle", "global")


def _top5(x: Tensor) -> Tensor:
    return x.flatten(1).topk(max(1, int(.05 * x[0].numel())), 1).values.mean(1)


def _group_desc(t: Tensor, a: Tensor, groups: list[Tensor]) -> Tensor:
    return torch.stack([
        torch.cat((t[:, ix.to(t.device)].mean(1), a[:, ix.to(t.device)].mean(1)), 1)
        for ix in groups
    ], 1)


def _fit_spatial_world(t: Tensor, a: Tensor, p: Tensor) -> tuple[list[Any], list[tuple[Tensor, Tensor]]]:
    tm, td = t.median(0).values, (t - t.median(0).values).abs().median(0).values
    am, ad = a.median(0).values, (a - a.median(0).values).abs().median(0).values
    raw = [p, ((t-tm)/(1.4826*td+1e-5)).square().mean(-1), ((a-am)/(1.4826*ad+1e-5)).abs().mean(-1)]
    world: list[Any] = [(tm, td, am, ad)]
    for groups in GROUPS.values():
        d = _group_desc(t, a, groups); m = d.median(0).values; v = (d-m).abs().median(0).values
        world.append((groups, m, v)); q = ((d-m)/(1.4826*v+1e-5)).square().mean(-1)
        mp, cnt = torch.zeros_like(p), torch.zeros_like(p)
        for h, ix in enumerate(groups):
            ix = ix.to(p.device); mp[:, ix] += q[:, h, None]; cnt[:, ix] += 1
        raw.append(mp / cnt.clamp_min(1))
    norm = [(r.median(0).values, (r-r.median(0).values).abs().median(0).values) for r in raw]
    return world, norm


def _spatial_maps(t: Tensor, a: Tensor, p: Tensor, world: list[Any], norm: list[tuple[Tensor, Tensor]]) -> Tensor:
    tm, td, am, ad = world[0]
    raw = [p, ((t-tm)/(1.4826*td+1e-5)).square().mean(-1), ((a-am)/(1.4826*ad+1e-5)).abs().mean(-1)]
    for groups, m, v in world[1:]:
        d = _group_desc(t, a, groups); q = ((d-m)/(1.4826*v+1e-5)).square().mean(-1)
        mp, cnt = torch.zeros_like(p), torch.zeros_like(p)
        for h, ix in enumerate(groups):
            ix = ix.to(p.device); mp[:, ix] += q[:, h, None]; cnt[:, ix] += 1
        raw.append(mp / cnt.clamp_min(1))
    z = [((r-m)/(1.4826*d+1e-5)).relu().clamp_max(6.) for r, (m, d) in zip(raw, norm)]
    return torch.stack(z, 1).reshape(-1, 6, GRID, GRID)


class CompositionDescriptor(nn.Module):
    """Experiment-fixed 64 random directions and 32 soft histogram bins."""
    def __init__(self, directions: int = 64, bins: int = 32, seed: int = 31415, temperature: float = .08) -> None:
        super().__init__(); g = torch.Generator(); g.manual_seed(seed)
        self.register_buffer("directions", F.normalize(torch.randn(directions, DIM, generator=g), dim=1))
        self.register_buffer("centers", torch.linspace(-1., 1., bins)); self.temperature = temperature
        self.bins, self.projection_seed = bins, seed

    def forward(self, tokens: Tensor, keep: Tensor | None = None) -> Tensor:
        r = (F.normalize(tokens, dim=-1) @ self.directions.T).clamp(-1, 1)
        h = torch.exp(-((r[..., None] - self.centers) ** 2) / (2 * self.temperature ** 2))
        if keep is not None:
            h = h * keep[..., None, None]; denom = keep.sum(1, keepdim=True)[..., None].clamp_min(1.)
        else:
            denom = float(tokens.shape[1])
        return (h.sum(1) / denom).flatten(1)


@dataclass
class MissingFewerReference:
    prior: StructuralPrior
    spatial_world: list[Any]
    spatial_norm: list[tuple[Tensor, Tensor]]
    composition_mean: Tensor
    composition_precision: Tensor
    tail_struct: Tensor
    tail_composition: Tensor
    feature_identity: str
    threshold: float | None = None
    threshold_policy: str = "auto"
    version: str = "missing_fewer_e7_r0"

    def to(self, device: torch.device | str) -> "MissingFewerReference":
        def move(x: Any) -> Any:
            if isinstance(x, Tensor): return x.to(device)
            if isinstance(x, tuple): return tuple(move(v) for v in x)
            if isinstance(x, list): return [move(v) for v in x]
            return x
        return MissingFewerReference(self.prior.to(device), move(self.spatial_world), move(self.spatial_norm), self.composition_mean.to(device), self.composition_precision.to(device), self.tail_struct.to(device), self.tail_composition.to(device), self.feature_identity, self.threshold, self.threshold_policy, self.version)

    def state_dict(self) -> dict[str, Any]:
        return {"prior": self.prior.state_dict(), "spatial_world": self.spatial_world, "spatial_norm": self.spatial_norm, "composition_mean": self.composition_mean, "composition_precision": self.composition_precision, "tail_struct": self.tail_struct, "tail_composition": self.tail_composition, "feature_identity": self.feature_identity, "threshold": self.threshold, "threshold_policy": self.threshold_policy, "version": self.version}

    @classmethod
    def from_state_dict(cls, x: dict[str, Any]) -> "MissingFewerReference":
        return cls(StructuralPrior.from_state_dict(x["prior"]), x["spatial_world"], x["spatial_norm"], x["composition_mean"], x["composition_precision"], x["tail_struct"], x["tail_composition"], x["feature_identity"], x.get("threshold"), x.get("threshold_policy", "auto"), x.get("version", "missing_fewer_e7_r0"))

    def save(self, path: str | Path) -> None: torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "MissingFewerReference":
        return cls.from_state_dict(torch.load(path, map_location="cpu", weights_only=False)).to(device)


class MissingFewerE7(nn.Module):
    """Public core plus target-only normal world and frozen E7 scorer."""
    def __init__(self, composition_directions: int = 64, composition_bins: int = 32, composition_seed: int = 31415, composition_temperature: float = .08) -> None:
        super().__init__(); self.spatial = RegistrationGSR(); self.composition = CompositionDescriptor(composition_directions, composition_bins, composition_seed, composition_temperature)

    def encode(self, f16: Tensor, f32: Tensor) -> Tensor:
        _, t = self.spatial.tokens(f16, f32); return t

    def feature_identity(self, backbone_identity: str) -> str:
        digest = sha256(); digest.update(backbone_identity.encode())
        for key, value in sorted(self.state_dict().items()): digest.update(key.encode()); digest.update(str(tuple(value.shape)).encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def public_loss(self, f16_a: Tensor, f32_a: Tensor, f16_b: Tensor, f32_b: Tensor, f16_geo: Tensor, f32_geo: Tensor, theta: Tensor, composition_weight: float = .5, margin: float = .02) -> dict[str, Tensor]:
        _, ta = self.spatial.tokens(f16_a, f32_a); _, tb = self.spatial.tokens(f16_b, f32_b); _, tg = self.spatial.tokens(f16_geo, f32_geo)
        out = self.spatial.public_loss(ta, tg, theta, tb); ca, cg = self.composition(ta), self.composition(tg)
        keep = torch.ones(len(ta), GRID, GRID, device=ta.device)
        for i in range(len(ta)):
            h, w = 3, 3; y = int(torch.randint(0, GRID-h+1, (1,), device=ta.device)); x = int(torch.randint(0, GRID-w+1, (1,), device=ta.device)); keep[i, y:y+h, x:x+w] = 0
        cm = self.composition(ta, keep.flatten(1)); benign = (ca-cg).abs().mean(1); removed = (ca-cm).abs().mean(1)
        out["composition_invariance"] = benign.mean(); out["composition_sensitivity"] = F.relu(margin + benign - removed).mean(); out["total"] = out["total"] + composition_weight * (out["composition_invariance"] + out["composition_sensitivity"]); return out

    @torch.no_grad()
    def _parts(self, tokens: Tensor, prior: StructuralPrior) -> tuple[Tensor, Tensor, Tensor]:
        t = self.spatial.canonicalize_query(tokens, prior); a = self.spatial.reasoner.relation(t); e = self.spatial.reasoner.expected_from_query(t, prior); return t, a, 1-F.cosine_similarity(e, t, -1)

    @torch.no_grad()
    def build_reference(self, tokens: Tensor, backbone_identity: str, folds: int = 5) -> MissingFewerReference:
        if len(tokens) < folds: raise ValueError("Need at least one normal per cross-fit fold.")
        canon = self.spatial.canonicalize_set(tokens, rounds=2); prior = self.spatial.build_prior_tokens(canon); t, a, p = self._parts(tokens, prior); world, norm = _fit_spatial_world(t, a, p)
        comp = self.composition(tokens).float().cpu().numpy(); lw = LedoitWolf().fit(comp); mean = torch.from_numpy(comp.mean(0)).to(tokens.device); precision = torch.from_numpy(lw.precision_).to(tokens.device, dtype=tokens.dtype)
        cf_s, cf_c = torch.empty(len(tokens), device=tokens.device), torch.empty(len(tokens), device=tokens.device); idx = np.array_split(np.arange(len(tokens)), folds)
        for held in idx:
            train = np.setdiff1d(np.arange(len(tokens)), held, assume_unique=True); ti, vi = tokens[torch.as_tensor(train, device=tokens.device)], tokens[torch.as_tensor(held, device=tokens.device)]
            cp = self.spatial.build_prior_tokens(self.spatial.canonicalize_set(ti, rounds=2)); tt, aa, pp = self._parts(ti, cp); wi, ni = _fit_spatial_world(tt, aa, pp); vt, va, vp = self._parts(vi, cp); cf_s[torch.as_tensor(held, device=tokens.device)] = _top5(_spatial_maps(vt, va, vp, wi, ni).mean(1))
            dc = self.composition(ti).float().cpu().numpy(); clw = LedoitWolf().fit(dc); dv = self.composition(vi).float().cpu().numpy(); z = dv-dc.mean(0); cf_c[torch.as_tensor(held, device=tokens.device)] = torch.from_numpy(np.sqrt(np.maximum(np.einsum('bi,ij,bj->b', z, clw.precision_, z), 0))).to(tokens.device)
        return MissingFewerReference(prior, world, norm, mean, precision, cf_s, cf_c, self.feature_identity(backbone_identity))

    @staticmethod
    def _tail(reference: Tensor, values: Tensor) -> Tensor:
        return -torch.log(((reference[None] >= values[:, None]).sum(1).float()+1) / (len(reference)+1))

    @torch.no_grad()
    def score_tokens(self, tokens: Tensor, reference: MissingFewerReference, backbone_identity: str) -> dict[str, Tensor]:
        if reference.feature_identity != self.feature_identity(backbone_identity): raise RuntimeError("Missing/Fewer reference feature identity mismatch; rebuild reference after changing backbone/public core.")
        r = reference.to(tokens.device); t, a, p = self._parts(tokens, r.prior); maps = _spatial_maps(t, a, p, r.spatial_world, r.spatial_norm); individual = {_name: _top5(maps[:, i]) for i, _name in enumerate(ENERGY_NAMES)}; e6 = _top5(maps.mean(1)); d = self.composition(tokens).float()-r.composition_mean.float(); ec = torch.sqrt((d @ r.composition_precision.float() * d).sum(1).clamp_min(0)).to(tokens.dtype); qs, qc = self._tail(r.tail_struct, e6), self._tail(r.tail_composition, ec); score = torch.logaddexp(qs, qc)
        return {**individual, "composition": ec, "e6": e6, "score": score, "tail_struct": qs, "tail_composition": qc}

    @torch.no_grad()
    def score_features(self, f16: Tensor, f32: Tensor, reference: MissingFewerReference, backbone_identity: str) -> dict[str, Tensor]:
        return self.score_tokens(self.encode(f16, f32), reference, backbone_identity)

    @torch.no_grad()
    def calibrate_threshold(self, normal_scores: Tensor, anomaly_scores: Tensor, reference: MissingFewerReference, policy: str = "auto", target_fpr: float = .05) -> MissingFewerReference:
        if len(anomaly_scores) == 0: raise ValueError("30A boundary calibration needs real anomaly scores.")
        n, a = normal_scores.detach().cpu().numpy(), anomaly_scores.detach().cpu().numpy(); y = np.r_[np.zeros(len(n), dtype=int), np.ones(len(a), dtype=int)]; s = np.r_[n, a]
        chosen = policy
        if policy == "auto": chosen = "f1"
        if chosen == "target_fpr": threshold = float(np.quantile(n, 1-target_fpr))
        else:
            candidates = np.unique(s); fn = f1_score if chosen == "f1" else balanced_accuracy_score; threshold = float(candidates[int(np.argmax([fn(y, s >= v) for v in candidates]))])
        reference.threshold, reference.threshold_policy = threshold, chosen; return reference
