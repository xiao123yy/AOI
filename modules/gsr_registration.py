"""Category-agnostic structural registration for the GSR normal world."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from modules.gsr_md_v2 import DIM, GRID, TOKENS, GSRMDv2, StructuralPrior


def _coords() -> Tensor:
    y, x = torch.meshgrid(torch.linspace(-1, 1, GRID), torch.linspace(-1, 1, GRID), indexing='ij')
    return torch.stack((x.flatten(), y.flatten()), -1)


class StructuralRegistration(nn.Module):
    """A compact differentiable 12x12-to-12x12 structural correspondence layer."""
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(DIM, 64, bias=False)
        self.k = nn.Linear(DIM, 64, bias=False)
        self.bias = nn.Sequential(nn.Linear(3, 24), nn.GELU(), nn.Linear(24, 1))
        self.register_buffer('coords', _coords(), persistent=False)

    def forward(self, a: Tensor, b: Tensor, sinkhorn_steps: int = 3) -> Tensor:
        qa, kb = F.normalize(self.q(a), dim=-1), F.normalize(self.k(b), dim=-1)
        delta = self.coords[None, :, None] - self.coords[None, None, :]
        dist = delta.square().sum(-1, keepdim=True).sqrt()
        pos = self.bias(torch.cat((delta, dist), -1)).squeeze(-1)
        logits = qa @ kb.transpose(-1, -2) / .12 + pos
        p = torch.exp(logits - logits.amax(dim=(-1, -2), keepdim=True))
        for _ in range(sinkhorn_steps):
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-6)
            p = p / p.sum(-2, keepdim=True).clamp_min(1e-6)
        return p / p.sum(-1, keepdim=True).clamp_min(1e-6)

    def affine_target(self, theta: Tensor, sigma: float = .13) -> Tensor:
        """Soft A->B target, where grid_sample(B, theta) samples source A."""
        b = theta.shape[0]
        h = torch.eye(3, device=theta.device, dtype=theta.dtype)[None].repeat(b, 1, 1)
        h[:, :2] = theta
        inv = torch.linalg.inv(h)[:, :2]
        point = torch.cat((self.coords, torch.ones(TOKENS, 1, device=theta.device, dtype=theta.dtype)), -1)
        bcoord = point[None] @ inv.transpose(1, 2)
        diff = self.coords[None, None] - bcoord[:, :, None]
        valid = (bcoord.abs() <= 1.02).all(-1, keepdim=True)
        target = torch.exp(-diff.square().sum(-1) / (2 * sigma * sigma)) * valid.float()
        return target / target.sum(-1, keepdim=True).clamp_min(1e-6)


def group_sets() -> dict[str, list[Tensor]]:
    def ids(y: int, x: int, h: int, w: int) -> Tensor:
        return torch.tensor([r * GRID + c for r in range(y, y + h) for c in range(x, x + w)])
    local = [ids(y, x, 2, 2) for y in range(0, 12, 2) for x in range(0, 12, 2)]
    local += [ids(y, x, 3, 3) for y in range(0, 12, 3) for x in range(0, 12, 3)]
    mid = [ids(y, x, 4, 4) for y in range(0, 12, 4) for x in range(0, 12, 4)]
    mid += [ids(y, x, 6, 6) for y in (0, 6) for x in (0, 6)]
    glob = [ids(y, 0, 4, 12) for y in (0, 4, 8)] + [ids(0, x, 12, 4) for x in (0, 4, 8)]
    glob += [ids(y, 0, 6, 12) for y in (0, 6)] + [ids(0, x, 12, 6) for x in (0, 6)]
    return {'local': local, 'middle': mid, 'global': glob}


GROUPS = group_sets()


class RegistrationGSR(GSRMDv2):
    def __init__(self) -> None:
        super().__init__()
        self.registration = StructuralRegistration()

    def public_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.tokenizer.parameters()
        yield from self.registration.parameters()
        yield from self.reasoner.parameters()

    def contextualize(self, t: Tensor) -> Tensor:
        b = t.shape[0]
        rel = self.reasoner._generic_relation_bias(b)
        blind = torch.zeros(b, TOKENS, TOKENS, dtype=torch.bool, device=t.device)
        x = t
        for block in self.reasoner.blocks:
            x = block(x, t, rel, blind)
        return x

    @staticmethod
    def descriptors(t: Tensor, groups: list[Tensor]) -> Tensor:
        return torch.stack([t[:, ix.to(t.device)].mean(1) for ix in groups], 1)

    def public_loss(self, ta: Tensor, tb_geo: Tensor, theta: Tensor, tb: Tensor) -> dict[str, Tensor]:
        p_geo = self.registration(ta, tb_geo)
        geo = -(self.registration.affine_target(theta) * p_geo.clamp_min(1e-8).log()).sum(-1).mean()
        pab, pba = self.registration(ta, tb), self.registration(tb, ta)
        eye = torch.eye(TOKENS, device=ta.device)[None]
        ca, cb = pab @ pba, pba @ pab
        confidence = (pab.max(-1).values * pba.max(-1).values).detach().clamp_min(1e-4)
        cycle = (((ca - eye).square().mean(-1) * confidence).sum(-1) / confidence.sum(-1)).mean()
        cycle = cycle + (((cb - eye).square().mean(-1) * confidence).sum(-1) / confidence.sum(-1)).mean()
        tb_to_a = pab @ tb
        za, zb = self.contextualize(ta), self.contextualize(tb_to_a)
        ra, rb = self.reasoner.relation(za), self.reasoner.relation(zb)
        relation = F.smooth_l1_loss(ra, rb)
        group = ta.new_zeros(())
        for sets in GROUPS.values():
            ga, gb = self.descriptors(za, sets), self.descriptors(zb, sets)
            group = group + F.smooth_l1_loss(F.normalize(ga, dim=-1), F.normalize(gb, dim=-1))
        group = group / len(GROUPS)
        ds = (pab.sum(-1) - 1).square().mean() + (pab.sum(-2) - 1).square().mean()
        entropy = -(pab * pab.clamp_min(1e-8).log()).sum(-1).mean() / math.log(TOKENS)
        total = geo + .5 * cycle + .5 * relation + .5 * group + .01 * ds + .001 * entropy
        return {'total': total, 'geo': geo, 'cycle': cycle, 'relation': relation, 'group': group, 'doubly': ds, 'entropy': entropy}

    @torch.no_grad()
    def canonicalize_set(self, t: Tensor, rounds: int = 2) -> Tensor:
        template = t.median(0).values
        canon = t
        for _ in range(rounds):
            p = self.registration(template[None].expand(len(t), -1, -1), t)
            canon = p @ t
            template = canon.median(0).values
        return canon

    def canonicalize_query(self, t: Tensor, prior: StructuralPrior) -> Tensor:
        p = self.registration(prior.node[None].expand(len(t), -1, -1), t)
        return p @ t

    @torch.no_grad()
    def build_prior_tokens(self, t: Tensor, trim: float = .10) -> StructuralPrior:
        n, k = len(t), int(len(t) * trim)
        node = t.sort(0).values[k:n-k].mean(0) if n > 2*k else t.mean(0)
        var = (t - node[None]).square().mean(0)
        rel = self.reasoner.relation(t).mean(0)
        return StructuralPrior(node.detach(), var.detach(), rel.detach())

    def normal_loss(self, support_t: Tensor, query_t: Tensor) -> dict[str, Tensor]:
        canon_support = self.canonicalize_set(support_t, rounds=1)
        prior = self.build_prior_tokens(canon_support)
        q = self.canonicalize_query(query_t, prior)
        losses = self.reasoner.structural_losses(q, target_feature=False)
        expected = self.reasoner.expected_from_query(q, prior)
        mask = self.reasoner.contiguous_mask(len(q), q.device, .30)
        feature = (F.smooth_l1_loss(expected, q.detach(), reduction='none').mean(-1) * mask.float()).sum() / mask.float().sum().clamp_min(1.)
        losses['feature'] = feature
        losses['total'] = losses['total'] + .5 * feature
        return losses
