# -*- coding: utf-8 -*-
"""e2e/data.py — 面板读取、两层标准化（截面 / 时序）与「无未来函数」的可交易掩码

约定
----
* 收益：`y1[t]` = 第 t 日收盘 → 第 t+1 日收盘的复权收益（已在面板层算好，与 adj_close 逐值一致）。
* 掩码：`m_trad[t]` = 非 ST、非停牌、**非当日涨停**、**上市满 60 个交易日**（剔次新）、
  **近 20 日未停牌**（避开复牌跳空）、且 y1 有限。
  当日涨停的股票在收盘时买不进，必须剔除（否则回测虚高）；次新剔除是必需的数据卫生措施——
  面板里 |y1|>30% 的格子有 85% 是上市不足 60 日的新股，会把「80 只等权」的日收益推到 +29% 这种不可能的数值。

* **两类特征、两套标准化**（这是接多源特征时最容易踩的坑）：

  ① **截面型**（绝大多数）：逐日截面 z-score，截断 ±4，缺失填 0。
     天然无未来信息；也避免原始量纲（成交额）压过比值类特征。
  ② **逐日常数型**（市场状态 / 宽基指数 / 北向 / 板块涨停率）：同一天全市场取值相同，截面 sd≈0。
     若仍做截面 z-score，会退化成 0（全无信息）或被浮点残差放大成 ±4 的**符号噪声**（有害）。
     改为**时序** z-score（训练段均值/方差），保留"今天处于什么状态"——它经 MLP 的 ReLU
     形成门控（相当于 FiLM 的 bias 调制），可以改变截面排序，但不是直接的个股信号。

* **退化列自动剔除**：可交易池里恒为 0 的列（如 `lu_streak` 连板数、`is_zt` 涨停标志 ——
  涨停股本身已被掩码剔除；`kpl_*` 只在涨停股上有值）不提供任何截面区分度，直接丢弃，
  否则只是白占输入维度与参数。
"""
from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

_HERE = Path(__file__).resolve()
BASE = Path(os.environ.get("SF_BASE", _HERE.parents[1]))

# 默认特征：全部 9 个矩阵（140 列）——不是只有 55 列量价
FEAT_ALL = ("X_base55.npy", "X_extra.npy", "X_mkt.npy", "X_mf.npy",
            "X_cyq.npy", "X_kpl.npy", "X_auction.npy", "X_mined.npy", "X_plate.npy")


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
CACHE = Path(os.environ.get("E2E_CACHE", BASE / "_cache_e2e"))

SEG = (0.60, 0.75)      # 训练 / 验证 / 测试（按样本比例切，不写死日期）


def _load(name: str) -> np.ndarray:
    a = np.load(PAN / name, mmap_mode="r")
    return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0)


class Panel:
    """日线面板：特征 (D,S,F)、次日收益 (D,S)、可交易掩码 (D,S)。

    标准化统计量按 feat 组合做磁盘缓存（`_cache_e2e/<hash>.npz`），
    因为 140 列要扫 1638 天，重复训练时不必每次重算。
    """

    def __init__(self, feat_files=FEAT_ALL, clip: float = 4.0, use_cache: bool = True):
        self.dates = np.load(PAN / "dates.npy").astype(np.int64)
        self.y = np.load(PAN / "y1.npy", mmap_mode="r").astype(np.float32)
        self.D, self.S = self.y.shape
        st, dt, zt = _load("is_st.npy"), _load("is_dt.npy"), _load("is_zt.npy")
        # 上市天数（截至当日已交易数）与「近 20 日内是否停牌」——两者都只用到 ≤t 的信息
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
        self._xs, self.F_all, self.clip = xs, sum(x.shape[2] for x in xs), clip
        self.feat_files = tuple(feat_files)

        self.i1, self.i2 = int(SEG[0] * self.D), int(SEG[1] * self.D)
        self.segs = {"train": (0, self.i1), "val": (self.i1, self.i2), "test": (self.i2, self.D)}

        # ── 退化列：可交易池内恒为 0 ──
        key = hashlib.md5(("|".join(self.feat_files) + f"#{clip}").encode()).hexdigest()[:12]
        cfile = CACHE / f"norm_{key}.npz"
        if use_cache and cfile.exists():
            z = np.load(cfile)
            self.keep, self.mu, self.sd = z["keep"], z["mu"], z["sd"]
            self.cs_const, self.ts_mu, self.ts_sd = z["cs_const"], z["ts_mu"], z["ts_sd"]
        else:
            self.keep = self._degenerate()
            self.mu, self.sd, self.cs_const, self.ts_mu, self.ts_sd = self._stats()
            if use_cache:
                CACHE.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cfile, keep=self.keep, mu=self.mu, sd=self.sd,
                                    cs_const=self.cs_const, ts_mu=self.ts_mu, ts_sd=self.ts_sd)

        self.F = int(self.keep.sum())

    # ---------- 特征切片 ----------
    def _raw_all(self, t: int) -> np.ndarray:
        if len(self._xs) == 1:
            return np.asarray(self._xs[0][t], dtype=np.float32)
        return np.concatenate([np.asarray(a[t], dtype=np.float32) for a in self._xs], axis=1)

    def _raw(self, t: int) -> np.ndarray:
        return self._raw_all(t)[:, self.keep]

    # ---------- 退化列检测 ----------
    def _degenerate(self, probe_step: int = 7, min_frac: float = 0.02) -> np.ndarray:
        """在可交易池内，统计「至少有一只票非零」的天数占比；<2% 视为退化列。"""
        nz = np.zeros(self.F_all, dtype=np.float64)
        tot = 0
        for t in range(0, self.D, probe_step):
            m = self.m_trad[t]
            if m.sum() < 100:
                continue
            x = self._raw_all(t)
            # 注意：必须先把 NaN 转成 0 再取 max —— 否则 max 遇 NaN 返回 NaN，真列会被误判为退化
            nz += (np.abs(np.nan_to_num(x[m])).max(0) > 1e-9)
            tot += 1
        frac = nz / max(tot, 1)
        keep = frac >= min_frac
        if not keep.any():
            keep[:] = True
        return keep

    # ---------- 标准化统计量 ----------
    def _stats(self):
        F = int(self.keep.sum())
        tr = slice(0, self.i1)
        mu = np.zeros((self.D, F), dtype=np.float32)
        sd = np.ones((self.D, F), dtype=np.float32)
        for t in range(self.D):
            x = self._raw(t)
            m = self.m_trad[t]
            xv = np.where(m[:, None], x, np.nan)
            mu[t] = np.nanmean(xv, axis=0)
            sd[t] = np.nanstd(xv, axis=0)
        mu = np.nan_to_num(mu, nan=0.0)
        sd = np.nan_to_num(sd, nan=1.0, posinf=1.0)

        # 逐日常数型判定：训练段截面 sd / (|截面均值|+1e-6) 的中位数 < 1e-4
        with np.errstate(all="ignore"):
            rel = np.nanmedian(sd[tr] / (np.abs(mu[tr]) + 1e-6), axis=0)
        cs_const = np.nan_to_num(rel, nan=0.0) < 1e-4

        if cs_const.any():
            daily = np.zeros((self.D, int(cs_const.sum())), dtype=np.float32)
            for t in range(self.D):
                x = self._raw(t)[:, cs_const]
                m = self.m_trad[t]
                with np.errstate(all="ignore"):
                    daily[t] = np.nanmean(np.where(m[:, None], x, np.nan), axis=0)
            daily = np.nan_to_num(daily, nan=0.0)
            ts_mu, ts_sd = daily[tr].mean(0), daily[tr].std(0)
            ts_sd = np.where(ts_sd < 1e-8, 1.0, ts_sd).astype(np.float32)
        else:
            ts_mu = ts_sd = np.zeros(0, dtype=np.float32)

        # sd 下限：单日某列 sd 塌缩时，用训练段中位 sd 的 1e-3，避免 (x−mu)/1e-8 爆炸
        floor = np.nan_to_num(np.nanmedian(sd[tr], axis=0) * 1e-3, nan=1e-6)
        floor = np.where(cs_const, 1.0, np.maximum(floor, 1e-6))
        sd = np.maximum(sd, floor[None, :]).astype(np.float32)
        cs_const = cs_const.astype(bool)
        return mu, sd, cs_const, ts_mu, ts_sd

    # ---------- 单日截面 ----------
    def day(self, t: int):
        """返回 (x[S,F] float32 已标准化, mask[S] bool, y[S] float32)。

        截面型列：逐日截面 z-score（只用当日数据）；
        逐日常数列：时序 z-score（用训练段均值/方差），保留市场状态的时间信息。
        """
        x = self._raw(t)
        z = (x - self.mu[t]) / self.sd[t]
        if self.cs_const.any():
            c = self.cs_const
            z[:, c] = (x[:, c] - self.ts_mu) / self.ts_sd
        np.clip(z, -self.clip, self.clip, out=z)
        z[~np.isfinite(z)] = 0.0
        return z, self.m_trad[t], self.y[t]

    # ---------- 整个区间（用于评估/回测） ----------
    def days(self, lo: int, hi: int):
        for t in range(lo, hi):
            yield t, *self.day(t)

    def describe(self) -> str:
        return (f"面板 {self.D} 日 × {self.S} 只 × {self.F} 列"
                f"（原始 {self.F_all} 列，剔除退化 {self.F_all - self.F} 列，"
                f"其中逐日常数 {int(self.cs_const.sum())} 列走时序标准化）；"
                f"训练 {self.dates[0]}~{self.dates[self.i1-1]}（{self.i1} 日）、"
                f"验证 {self.dates[self.i1]}~{self.dates[self.i2-1]}（{self.i2-self.i1} 日）、"
                f"测试 {self.dates[self.i2]}~{self.dates[-1]}（{self.D-self.i2} 日）；"
                f"可交易占比 {self.m_trad.mean():.1%}")
