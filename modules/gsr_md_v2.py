"""GSR-MD v2: support-conditioned global structural reasoning for missing.

The module intentionally separates three learning phases:
  * public / 100-normal: only structural tokenizer and global reasoner;
  * 30 missing/fewer calibration: only the asymmetric comparator;
  * inference: a compact product prior plus one 144 x 144 attention pass.

It never stores support tokens or performs nearest-neighbour retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


GRID = 12
TOKENS = GRID * GRID
DIM = 192


class ChannelLayerNorm(nn.Module):
    """LayerNorm across C at every spatial location, for NCHW feature maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class StructuralTokenizer(nn.Module):
    """F16/F32 -> Z[192,24,24] -> T[144,192]."""

    def __init__(self) -> None:
        super().__init__()
        self.f16 = nn.Sequential(nn.Conv2d(384, 128, 1), ChannelLayerNorm(128), nn.GELU())
        self.f32 = nn.Sequential(nn.Conv2d(768, 128, 1), ChannelLayerNorm(128), nn.GELU())
        self.in_proj = nn.Conv2d(256, DIM, 1)
        self.norm = ChannelLayerNorm(DIM)
        self.dw = nn.Conv2d(DIM, DIM, 3, padding=1, groups=DIM)
        self.out = nn.Conv2d(DIM, DIM, 1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, f16: Tensor, f32: Tensor) -> Tuple[Tensor, Tensor]:
        a = self.f16(f16)
        b = F.interpolate(self.f32(f32), size=a.shape[-2:], mode="bilinear", align_corners=False)
        x = self.in_proj(torch.cat((a, b), dim=1))
        z = x + self.out(self.dw(F.gelu(self.norm(x))))
        t = self.pool(z).flatten(2).transpose(1, 2)
        return z, t


class ReasoningBlock(nn.Module):
    """Cross-attention over query image tokens with product relation bias."""

    def __init__(self, dim: int = DIM, heads: int = 4) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def _split(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: Tensor) -> Tensor:
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)

    def forward(self, query: Tensor, remote: Tensor, rel_bias: Tensor, blind: Tensor) -> Tensor:
        q, k, v = self._split(self.q(query)), self._split(self.k(remote)), self._split(self.v(remote))
        logits = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        logits = logits + torch.log(rel_bias.clamp_min(1e-6))[:, None]
        logits = logits.masked_fill(blind[:, None], -1e4)
        attn = logits.softmax(dim=-1)
        x = self.norm1(query + self.o(self._merge(attn @ v)))
        return self.norm2(x + self.ff(x))


class AttentionPool(nn.Module):
    def __init__(self, dim: int = DIM) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, t: Tensor, keep: Tensor | None = None) -> Tensor:
        logits = self.score(t).squeeze(-1)
        if keep is not None:
            logits = logits.masked_fill(~keep, -1e4)
        return (logits.softmax(-1)[..., None] * t).sum(1)


@dataclass
class StructuralPrior:
    """Compact target/product normal world; all tensors are [N,...], no bank."""

    node: Tensor       # [144, 192]
    var: Tensor        # [144, 192]
    rel: Tensor        # [144, 144]

    def to(self, device: torch.device | str) -> "StructuralPrior":
        return StructuralPrior(self.node.to(device), self.var.to(device), self.rel.to(device))

    def state_dict(self) -> Dict[str, Tensor]:
        return {"node": self.node, "var": self.var, "rel": self.rel}

    @classmethod
    def from_state_dict(cls, value: Dict[str, Tensor]) -> "StructuralPrior":
        return cls(value["node"], value["var"], value["rel"])


class GlobalStructureReasoner(nn.Module):
    """Blind expected-structure branch plus public structural objectives."""

    def __init__(self) -> None:
        super().__init__()
        self.rel_embed = nn.Linear(DIM, 64, bias=False)
        self.node_seed = nn.Linear(DIM, DIM)
        self.mask_seed = nn.Parameter(torch.zeros(1, TOKENS, DIM))
        # Public coordinates are geometric, not 144 product-template slots.
        # They are shared functions of position / relative position only.
        self.pos_proj = nn.Sequential(nn.Linear(4, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.rel_pos_bias = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))
        self.blocks = nn.ModuleList([ReasoningBlock(), ReasoningBlock()])
        self.partial_pool = AttentionPool()
        self.full_pool = AttentionPool()
        self.part_to_whole = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.whole_to_part = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        self.register_buffer("local_blind", self._local_blind(), persistent=False)
        self.register_buffer("node_coords", self._node_coordinates(), persistent=False)
        self.register_buffer("relative_coords", self._relative_coordinates(), persistent=False)

    @staticmethod
    def _local_blind(radius: int = 1) -> Tensor:
        yy, xx = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
        distance = torch.maximum((yy.flatten()[:, None] - yy.flatten()[None]).abs(),
                                 (xx.flatten()[:, None] - xx.flatten()[None]).abs())
        return distance <= radius

    @staticmethod
    def _node_coordinates() -> Tensor:
        """Fourier-free, deterministic coordinates used by the generic mask query."""
        yy, xx = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
        x, y = xx.flatten().float() / (GRID - 1), yy.flatten().float() / (GRID - 1)
        return torch.stack((x, y, x * 2 - 1, y * 2 - 1), dim=-1)[None]

    @staticmethod
    def _relative_coordinates() -> Tensor:
        yy, xx = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
        dx = (xx.flatten()[:, None] - xx.flatten()[None]).float() / (GRID - 1)
        dy = (yy.flatten()[:, None] - yy.flatten()[None]).float() / (GRID - 1)
        distance = torch.sqrt(dx.square() + dy.square())
        return torch.stack((dx, dy, distance), dim=-1)

    def _generic_position(self) -> Tensor:
        return self.pos_proj(self.node_coords)

    def _generic_relation_bias(self, batch: int) -> Tensor:
        """Shared Delta-x/Delta-y/distance bias; no image or product information."""
        raw = self.rel_pos_bias(self.relative_coords).squeeze(-1)
        base = F.softplus(raw) + 1e-4
        base = base / base.sum(-1, keepdim=True)
        return base[None].expand(batch, -1, -1)

    def relation(self, t: Tensor) -> Tensor:
        r = F.normalize(self.rel_embed(t), dim=-1)
        return (r @ r.transpose(-1, -2) / 0.25).softmax(-1)

    @staticmethod
    def _guard_from_mask(masked: Tensor) -> Tensor:
        grid = masked.reshape(-1, 1, GRID, GRID).float()
        guard = F.max_pool2d(grid, 3, stride=1, padding=1).bool().reshape(-1, TOKENS)
        return guard

    @staticmethod
    def contiguous_mask(batch: int, device: torch.device, ratio: float = .30) -> Tensor:
        """2-4 rectangles cover about ratio of the 12x12 structural grid."""
        out = torch.zeros(batch, GRID, GRID, dtype=torch.bool, device=device)
        goal = max(1, round(TOKENS * ratio))
        for b in range(batch):
            attempts = 0
            while int(out[b].sum()) < goal and attempts < 20:
                h = int(torch.randint(3, 5, (), device=device))
                w = int(torch.randint(3, 5, (), device=device))
                y = int(torch.randint(0, GRID - h + 1, (), device=device))
                x = int(torch.randint(0, GRID - w + 1, (), device=device))
                out[b, y:y + h, x:x + w] = True
                attempts += 1
        return out.flatten(1)

    def _expected(self, remote: Tensor, prior: StructuralPrior, blind: Tensor, extra_global: Tensor | None = None) -> Tensor:
        b = remote.shape[0]
        q = self.node_seed(prior.node)[None].expand(b, -1, -1) + self._generic_position()
        if extra_global is not None:
            q = q + self.whole_to_part(extra_global)[:, None]
        # Target-only expected branch: generic geometry plus its R_rel normal world.
        generic = self._generic_relation_bias(b)
        rel = generic * prior.rel[None].expand(b, -1, -1).clamp_min(1e-6)
        rel = rel / rel.sum(-1, keepdim=True).clamp_min(1e-6)
        for block in self.blocks:
            q = block(q, remote, rel, blind)
        return q

    def expected_from_query(self, query_t: Tensor, prior: StructuralPrior) -> Tensor:
        """All 144 E_i in one pass. Every E_i is blind to its local 3x3 query area."""
        blind = self.local_blind[None].expand(query_t.shape[0], -1, -1)
        return self._expected(query_t, prior, blind)

    def _masked_relation(self, t: Tensor, masked: Tensor, *, relation_target: Tensor | None = None,
                         feature: bool = False) -> Tuple[Tensor, Dict[str, Tensor]]:
        # In public training this is an external frozen-ImageNet teacher
        # relation.  The fallback is used only for target normal adaptation.
        with torch.no_grad():
            target_a = self.relation(t).detach() if relation_target is None else relation_target.detach()
        guard = self._guard_from_mask(masked)
        # Any masked/guarded token cannot be K/V for any predicted masked row.
        blind = guard[:, None, :].expand(-1, TOKENS, -1).clone()
        blind = blind | self.local_blind[None]
        seed = self.mask_seed.expand(t.shape[0], -1, -1) + self._generic_position()
        q = seed
        # Public masking may use relative coordinates, but never A_full as an input.
        # A_full is a stop-gradient target only; otherwise the target leaks into E.
        rel = self._generic_relation_bias(t.shape[0])
        for block in self.blocks:
            q = block(q, t, rel, blind)
        pred_r = F.normalize(self.rel_embed(q), dim=-1)
        key_r = F.normalize(self.rel_embed(t), dim=-1)
        logits = pred_r @ key_r.transpose(-1, -2) / .25
        logits = logits.masked_fill(guard[:, None], -1e4)
        pred_a = logits.softmax(-1)
        # The specified target is A(masked, visible), therefore renormalize
        # A_full after removing masked/guard columns rather than forcing a
        # prediction to allocate mass to unavailable tokens.
        target_visible = target_a.masked_fill(guard[:, None], 0.)
        target_visible = target_visible / target_visible.sum(-1, keepdim=True).clamp_min(1e-6)
        valid = masked[:, :, None] & ~guard[:, None, :]
        valid_f = valid.float()
        denom = valid_f.sum().clamp_min(1.)
        smooth = (F.smooth_l1_loss(pred_a, target_visible, reduction="none") * valid_f).sum() / denom
        cosine = (1 - F.cosine_similarity(pred_a, target_visible, dim=-1))
        cosine = (cosine * masked.float()).sum() / masked.float().sum().clamp_min(1.)
        losses: Dict[str, Tensor] = {"relation": smooth + cosine, "relation_smooth": smooth, "relation_cos": cosine}
        if feature:
            # This is used only in target 100-normal adaptation, never public training.
            feat = (F.smooth_l1_loss(q, t.detach(), reduction="none").mean(-1) * masked.float()).sum()
            losses["feature"] = feat / masked.float().sum().clamp_min(1.)
        return q, losses

    def structural_losses(self, t: Tensor, *, relation_target: Tensor | None = None,
                          target_feature: bool = False) -> Dict[str, Tensor]:
        masked = self.contiguous_mask(t.shape[0], t.device, .30)
        _, rel = self._masked_relation(t, masked, relation_target=relation_target, feature=target_feature)
        keep_ratio = .5 + .2 * torch.rand(t.shape[0], 1, device=t.device)
        keep = torch.rand(t.shape[:2], device=t.device) < keep_ratio
        # Make each partial view non-empty.
        keep[:, 0] = True
        partial = self.partial_pool(t, keep)
        full = self.full_pool(t).detach()
        part2whole = F.smooth_l1_loss(self.part_to_whole(partial), full)
        masked2 = self.contiguous_mask(t.shape[0], t.device, .25)
        _, whole_rel = self._masked_relation(t, masked2, relation_target=relation_target, feature=False)
        whole2part = whole_rel["relation"] + .1 * F.smooth_l1_loss(self.whole_to_part(full), self.full_pool(t).detach())
        total = rel["relation"] + .5 * part2whole + .5 * whole2part
        if target_feature:
            total = total + .5 * rel["feature"]
        return {"total": total, "relation": rel["relation"], "part2whole": part2whole,
                "whole2part": whole2part, **({"feature": rel["feature"]} if target_feature else {})}


class AsymmetricMissingComparator(nn.Module):
    """The only trainable component in the 30 Missing/Fewer stage."""

    def __init__(self) -> None:
        super().__init__()
        self.expected = nn.Conv2d(DIM, 128, 1)
        self.observed = nn.Sequential(nn.Conv2d(DIM, 128, 1), nn.GELU(),
                                      nn.Conv2d(128, 128, 3, padding=1, groups=128), nn.Conv2d(128, 128, 1))
        # E, O, E-O, |E-O|, E*O, reliability = 641 channels
        self.head = nn.Sequential(nn.Conv2d(641, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, 128, 3, padding=1, groups=128), nn.Conv2d(128, 64, 1),
                                  nn.GELU(), nn.Conv2d(64, 1, 1))

    def forward(self, z: Tensor, expected_t: Tensor, prior: StructuralPrior) -> Tuple[Tensor, Tensor, Tensor]:
        e192 = F.interpolate(expected_t.transpose(1, 2).reshape(-1, DIM, GRID, GRID), size=(24, 24),
                             mode="bilinear", align_corners=False)
        e, o = self.expected(e192), self.observed(z)
        var = prior.var.mean(-1).clamp_min(1e-6).log1p().reshape(1, 1, GRID, GRID)
        var = F.interpolate(var.expand(z.shape[0], 1, GRID, GRID), size=(24, 24), mode="bilinear", align_corners=False)
        m = self.head(torch.cat((e, o, e - o, (e - o).abs(), e * o, var), dim=1))
        k = max(1, int(.05 * 24 * 24))
        return m.flatten(1).topk(k, dim=1).values.mean(1), m, e


class GSRMDv2(nn.Module):
    """Complete v2 module; callers explicitly choose phase-specific methods."""

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = StructuralTokenizer()
        self.reasoner = GlobalStructureReasoner()
        self.comparator = AsymmetricMissingComparator()

    def tokens(self, f16: Tensor, f32: Tensor) -> Tuple[Tensor, Tensor]:
        return self.tokenizer(f16, f32)

    @torch.no_grad()
    def build_prior(self, f16: Tensor, f32: Tensor, trim: float = .10) -> StructuralPrior:
        _, t = self.tokens(f16, f32)
        n = t.shape[0]
        k = int(n * trim)
        node = t.sort(dim=0).values[k:n - k].mean(0) if n > 2 * k else t.mean(0)
        var = (t - node[None]).square().mean(0)
        rel = self.reasoner.relation(t).mean(0)
        return StructuralPrior(node.detach(), var.detach(), rel.detach())

    def public_loss(self, f16: Tensor, f32: Tensor, relation_target: Tensor) -> Dict[str, Tensor]:
        """Normal-only public objective with a frozen external relation teacher.

        ``relation_target`` must be produced from a frozen ImageNet backbone;
        using a same-step student target admits the uniform-relation collapse.
        """
        _, t = self.tokens(f16, f32)
        return self.reasoner.structural_losses(t, relation_target=relation_target, target_feature=False)

    def normal_adaptation_loss(self, support16: Tensor, support32: Tensor, query16: Tensor, query32: Tensor) -> Dict[str, Tensor]:
        """100-normal phase; no comparator / anomaly label is involved."""
        # Prior creation is intentionally detached: R_struct is a product statistic, not a token bank.
        prior = self.build_prior(support16, support32)
        _, query_t = self.tokens(query16, query32)
        # Keep the public structural objective unchanged.  Target adaptation has
        # exactly one additional 0.5 L_feature term below, not two feature
        # reconstruction losses competing with relation learning.
        losses = self.reasoner.structural_losses(query_t, target_feature=False)
        # Target-specific expected feature task, using support prior and a blind query context.
        expected = self.reasoner.expected_from_query(query_t, prior)
        mask = self.reasoner.contiguous_mask(query_t.shape[0], query_t.device, .30)
        feature = (F.smooth_l1_loss(expected, query_t.detach(), reduction="none").mean(-1) * mask.float()).sum()
        feature = feature / mask.float().sum().clamp_min(1.)
        losses["feature"] = feature
        losses["total"] = losses["total"] + .5 * feature
        return losses

    def score(self, f16: Tensor, f32: Tensor, prior: StructuralPrior) -> Tuple[Tensor, Tensor]:
        z, t = self.tokens(f16, f32)
        expected = self.reasoner.expected_from_query(t, prior.to(t.device))
        score, evidence, _ = self.comparator(z, expected, prior.to(t.device))
        return score, evidence

    @torch.no_grad()
    def raw_expected_observed(self, f16: Tensor, f32: Tensor, prior: StructuralPrior) -> Tuple[Tensor, Tensor]:
        """Pre-Comparator structural evidence used only for mechanism audits.

        It compares the product-conditioned blind expectation E_i directly with
        observed structural token T_i.  No Comparator parameter or 30-anomaly
        calibration is involved.
        """
        _, observed = self.tokens(f16, f32)
        expected = self.reasoner.expected_from_query(observed, prior.to(observed.device))
        distance = 1 - F.cosine_similarity(expected, observed, dim=-1)
        k = max(1, int(.05 * TOKENS))
        score = distance.topk(k, dim=1).values.mean(1)
        evidence = F.interpolate(distance.reshape(-1, 1, GRID, GRID), size=(24, 24),
                                 mode="bilinear", align_corners=False)
        return score, evidence

    def public_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.tokenizer.parameters()
        yield from self.reasoner.parameters()

    def comparator_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.comparator.parameters()

    def set_phase(self, phase: str) -> None:
        """Freeze contract used by training scripts, independent of train/eval mode."""
        if phase not in {"public", "normal100", "missing30", "all"}:
            raise ValueError(f"unknown GSR-MD v2 phase: {phase}")
        for p in self.parameters():
            p.requires_grad_(phase == "all")
        if phase in {"public", "normal100"}:
            for p in self.public_parameters():
                p.requires_grad_(True)
        elif phase == "missing30":
            for p in self.comparator_parameters():
                p.requires_grad_(True)
