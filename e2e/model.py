# -*- coding: utf-8 -*-
"""e2e/model.py — 打分网络 + 可微组合层 + 三种训练目标

三种训练目标的区别**只在损失**（网络、组合层、种子、切分全部相同）：

    mse      : 先预测收益（最小化 MSE，收益按日 1%/99% 缩尾）→ 再把打分丢进组合层。
               这是「先估一阶矩、再优化」的两阶段做法，也是本题要检验的**错配**来源。
    ic       : 逐日截面负 RankIC（把排序搬到标签侧的精确 Spearman，本项目原口径）。
    sharpe   : **直接对组合净值的 Sharpe 反传**（端到端 / decision-focused）：
               打分 → 权重 → 组合日收益（含交易成本）→ 年化 Sharpe → 取负号回传。

组合层 `select_weights` 是**可微**的：取分数最高的 k 只 → 组内标准化 → softmax。
权重随分数连续变化，因此梯度可以从 Sharpe 一路传回打分网络；
若用等权 top-k，权重对分数不可导，端到端就退化成"只学选谁"，这是实现上最容易踩的坑。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9


class Scorer(nn.Module):
    """3 层截面 MLP：55 列 → 64 → 32 → 1（约 5.6k 参数）。"""

    def __init__(self, n_feat: int, hidden=(64, 32), dropout: float = 0.0):
        super().__init__()
        layers, last = [], n_feat
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)          # [S]


def select_weights(scores: torch.Tensor, mask: torch.Tensor, topk: int = 50,
                   temp: float = 0.8, zmax: float = 3.0) -> torch.Tensor:
    """可微组合层：top-k 选股 + 组内 softmax 加权（多头、满仓）。

    `zmax` 是**必须的**：组内 z 分数若出现离群值（某只票分数高 8 个标准差），
    softmax 会把权重压到那一只身上（实测出现过单票 51%），组合波动与回撤立刻失真。
    先把 z 截到 ±zmax，单票权重上界即为 e^(zmax/temp) / (k + ...)，天然分散。
    """
    s = scores.masked_fill(~mask, NEG)
    k = int(min(topk, int(mask.sum().item())))
    if k <= 0:
        return torch.zeros_like(scores)
    topv, topi = torch.topk(s, k)
    z = (topv - topv.mean()) / (topv.std(unbiased=False) + 1e-6)
    z = z.clamp(-zmax, zmax)
    w = torch.softmax(z / temp, dim=0)
    out = torch.zeros_like(scores)
    return out.index_put((topi,), w)            # index_put 对 src 可导


def sharpe(rets: torch.Tensor, ann: int = 252) -> torch.Tensor:
    """年化 Sharpe（无风险利率取 0）。"""
    mu, sd = rets.mean(), rets.std(unbiased=True)
    return mu / (sd + 1e-8) * (ann ** 0.5)


def differential_sharpe(rets: torch.Tensor, eta: float = 0.05) -> torch.Tensor:
    """Moody & Saffell (2001) 的差分 Sharpe：逐期增量奖励，方差比"整段 Sharpe"低得多。

    维护收益的一、二阶指数矩 A、B，用其增量构造可反传的即时奖励：
        D_t = (B_{t-1}·ΔA_t − 0.5·A_{t-1}·ΔB_t) / (B_{t-1} − A_{t-1}²)^{3/2}
    """
    a = b = torch.zeros((), dtype=rets.dtype)
    out = []
    for r in rets:
        da, db = r - a, r * r - b
        denom = (b - a * a).clamp_min(1e-8) ** 1.5
        out.append((b * da - 0.5 * a * db) / denom)
        a, b = a + eta * da, b + eta * db
    return torch.stack(out)


# ───────────────────────── 三种训练目标 ─────────────────────────
def loss_mse(scores: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
             winsor: float = 0.01) -> torch.Tensor:
    """两阶段第一步：最小化收益预测误差（标签按日 1%/99% 缩尾，标准做法）。"""
    yy = y[mask]
    if yy.numel() < 10:
        return scores.sum() * 0.0
    lo, hi = torch.quantile(yy, winsor), torch.quantile(yy, 1 - winsor)
    return F.mse_loss(scores[mask], yy.clamp(lo, hi))


def loss_rank_ic(scores: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """把排序放在标签侧：Pearson(score, rank(y)) 即精确的负 RankIC。"""
    s, yy = scores[mask], y[mask]
    if s.numel() < 10:
        return scores.sum() * 0.0
    r = torch.argsort(torch.argsort(yy)).float()
    s = s - s.mean()
    r = r - r.mean()
    return -(s * r).sum() / (s.norm() * r.norm() + 1e-8)


def loss_sharpe(rets: torch.Tensor, mode: str = "batch", eta: float = 0.05) -> torch.Tensor:
    """端到端：直接拿组合收益序列的 Sharpe 当损失。"""
    if mode == "dsr":
        return -differential_sharpe(rets, eta).mean() * (252 ** 0.5)
    return -sharpe(rets)


def loss_return(rets: torch.Tensor) -> torch.Tensor:
    """端到端：直接最大化**年化几何收益**。

    年化收益 = (∏(1+r_t))^(252/n) − 1 = exp(252·mean(log(1+r_t))) − 1，
    单调变换下等价于最大化 `mean(log(1+r_t))` ⇒ 损失取负号即可，且对每期收益可导。

    注意：没有分母（波动），所以这条臂**只追收益、不管回撤**，
    与 Sharpe 臂的差别正好回答「波动惩罚值多少钱」。
    """
    return -torch.log1p(rets.clamp_min(-0.999)).mean() * 252
