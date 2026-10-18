# -*- coding: utf-8 -*-
"""e2e/data.py — 面板读取、逐日截面标准化与「无未来函数」的可交易掩码

约定
----
* 收益：`y1[t]` = 第 t 日收盘 → 第 t+1 日收盘的复权收益（已在面板层算好，与 adj_close 逐值一致）。
* 掩码：`m_trad[t]` = 非 ST、非停牌、**非当日涨停**、**上市满 60 个交易日**（剔次新）、
  **近 20 日未停牌**（避开复牌跳空）、且 y1 有限。
  当日涨停的股票在收盘时买不进，必须剔除（否则回测虚高）；次新剔除是必需的数据卫生措施——
  面板里 |y1|>30% 的格子有 85% 是上市不足 60 日的新股，会把「80 只等权」的日收益推到 +29% 这种不可能的数值。
* 标准化：按日做**截面** z-score（只用当日数据），截断 ±4，缺失填 0。
  逐日截面标准化天然无未来信息；也避免原始量纲（如成交额）压过比值类特征。
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

_HERE = Path(__file__).resolve()
BASE = Path(os.environ.get("SF_BASE", _HERE.parents[1]))


def _pick(name: str, env: str) -> Path:
    """数据目录：优先环境变量，其次 <BASE>/name，再次 <BASE>/_name。"""
    v = os.environ.get(env)
    if v:
        return Path(v)
    for c in (BASE / name, BASE / ("_" + name)):
        if c.exists():
            return c
    return BASE / name


PAN = _pick("panel", "SF_PANEL")
FEAT = _pick("feat", "SF_FEAT")

SEG = (0.60, 0.75)      # 训练 / 验证 / 测试（按样本比例切，不写死日期）


def _load(name: str) -> np.ndarray:
    a = np.load(PAN / name, mmap_mode="r")
    return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0)


class Panel:
    """日线面板：特征 (D,S,F)、次日收益 (D,S)、可交易掩码 (D,S)。"""

    def __init__(self, feat_files=("X_base55.npy",), clip: float = 4.0):
        self.dates = np.load(PAN / "dates.npy").astype(np.int64)
        self.y = np.load(PAN / "y1.npy", mmap_mode="r").astype(np.float32)
        self.D, self.S = self.y.shape
        st, dt, zt = _load("is_st.npy"), _load("is_dt.npy"), _load("is_zt.npy")
        # 上市天数（截至当日已交易的交易日数）与「近 20 日内是否停牌」——两者都只用到 ≤t 的信息
        ac = np.load(PAN / "adj_close.npy", mmap_mode="r")
        listed = np.isfinite(ac).cumsum(0).astype(np.float32)
        c = np.cumsum(dt, axis=0)
        c20 = c - np.vstack([np.zeros((20, self.S), dtype=np.float32), c[:-20]])
        recent_dt = c20 > 0

        self.m_full = (st == 0) & (dt == 0) & (listed >= 60)     # 剔除次新（上市 < 60 个交易日）
        self.m_trad = (self.m_full & (zt == 0) & (~recent_dt)     # 剔除当日涨停、近 20 日停牌复牌
                       & np.isfinite(self.y))
        self.y = np.nan_to_num(self.y, nan=0.0)

        xs = []
        for f in feat_files:
            x = np.load(FEAT / f, mmap_mode="r")
            assert x.shape[:2] == (self.D, self.S), f"{f} 形状 {x.shape} 与面板不一致"
            xs.append(x)
        self.X = xs[0] if len(xs) == 1 else None
        self._xs, self.F, self.clip = xs, sum(x.shape[2] for x in xs), clip

        # 逐日截面均值/标准差预计算（只用当日数据，无未来信息），避免每次取数重算
        self.mu = np.zeros((self.D, self.F), dtype=np.float32)
        self.sd = np.ones((self.D, self.F), dtype=np.float32)
        for t in range(self.D):
            x = self._raw(t)
            m = self.m_trad[t]
            xv = np.where(m[:, None], x, np.nan)
            self.mu[t] = np.nanmean(xv, axis=0)
            self.sd[t] = np.nanstd(xv, axis=0)
        self.mu = np.nan_to_num(self.mu, nan=0.0)
        self.sd = np.nan_to_num(self.sd, nan=1.0, posinf=1.0)
        self.sd[self.sd < 1e-8] = 1.0

        self.i1, self.i2 = int(SEG[0] * self.D), int(SEG[1] * self.D)
        self.segs = {"train": (0, self.i1), "val": (self.i1, self.i2), "test": (self.i2, self.D)}

    # ---------- 单日截面 ----------
    def _raw(self, t: int) -> np.ndarray:
        if len(self._xs) == 1:
            return np.asarray(self._xs[0][t], dtype=np.float32)
        return np.concatenate([np.asarray(a[t], dtype=np.float32) for a in self._xs], axis=1)

    def day(self, t: int):
        """返回 (x[S,F] float32 已标准化, mask[S] bool, y[S] float32)。"""
        x = self._raw(t)
        z = (x - self.mu[t]) / self.sd[t]
        np.clip(z, -self.clip, self.clip, out=z)
        z[~np.isfinite(z)] = 0.0
        return z, self.m_trad[t], self.y[t]

    # ---------- 整个区间（用于评估/回测） ----------
    def days(self, lo: int, hi: int):
        for t in range(lo, hi):
            yield t, *self.day(t)

    def describe(self) -> str:
        return (f"面板 {self.D} 日 × {self.S} 只 × {self.F} 列；"
                f"训练 {self.dates[0]}~{self.dates[self.i1-1]}（{self.i1} 日）、"
                f"验证 {self.dates[self.i1]}~{self.dates[self.i2-1]}（{self.i2-self.i1} 日）、"
                f"测试 {self.dates[self.i2]}~{self.dates[-1]}（{self.D-self.i2} 日）；"
                f"可交易占比 {self.m_trad.mean():.1%}")
