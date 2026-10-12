# -*- coding: utf-8 -*-
"""
sf/ops.py — 单因子挖掘的算子库与终端集（仿东吴《深度学习系列1》Qlib 算子体系）

设计要点
--------
1. **去量纲优先**（研报 prompt 纪律第 2 条）：新数据面板的价格是未复权原始价，跨股票不可比。
   本模块统一用 `factor.npy` 把 OHLC 复权成连续序列 `A/H/L/O`，并把一切"价/量"量纲的终端
   做成**比值/收益率**形态后再暴露。凡裸价格水平的表达式，逐日截面 IC 天然 ≈ 0。
2. **算子三类**（不可混用）：
   - 逐元素：neg/abs/sign/log/sqrt/sqr/relu/inv、add/sub/mul/div/max2/min2/power/gt/lt
   - 截面（按日跨股票）：cs_rank/cs_z/cs_demean
   - 时序（按股跨日，窗口 w）：ref/delta/ts_mean/std/sum/max/min/median/mad/rank/z/quant/
     argmax/argmin/skew/kurt/count/slope/resi/rsqr/ema，二元 ts_corr
3. **因果性**：时序算子只回看；`causal_audit()` 用"扰动未来"做全量自检。
4. **性能**：所有滚动算子走 `sliding_window_view`（零拷贝视图 + 按时间分块），
   避免 pandas `rolling().apply()` 的 Python 级循环。

表达式写法：Python 可 eval 字符串，如 `"ts_mean(ush,20) * sign(ts_slope(A,20))"`；`%d` 为窗口占位符。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# 数据根目录：默认 <repo>/data，可用环境变量覆盖
BASE = Path(os.environ.get("SF_BASE", Path(__file__).resolve().parents[1]))
PAN = Path(os.environ.get("SF_PANEL", BASE / "panel"))
FEAT = Path(os.environ.get("SF_FEAT", BASE / "feat"))

WINS = [5, 10, 20, 30, 60]
EPS = 1e-9
CHUNK = 128
# 三段时序切分的样本分位（早期 / 中期 / 近期）；设为需要的任意分位即可
SEG_FRAC = (0.60, 0.75)


# ══════════════════════════════════════════════════ 基础工具
def rank_rows(X: np.ndarray) -> np.ndarray:
    """逐日截面百分位秩（NaN 保持 NaN）。X: (T, N)。"""
    out = np.full(X.shape, np.nan, np.float32)
    for t in range(X.shape[0]):
        r = X[t]
        m = ~np.isnan(r)
        n = int(m.sum())
        if n > 1:
            o = np.argsort(r[m], kind="stable")
            tmp = np.empty(n, np.float32)
            tmp[o] = np.arange(n, dtype=np.float32)
            out[t, m] = tmp / (n - 1)
    return out


def _view(x, w):
    """(D,S) → (D-w+1, S, w) 零拷贝滑动窗口视图（float32 连续）。"""
    x = np.ascontiguousarray(x, dtype=np.float32)
    return sliding_window_view(x, w, axis=0)


def _roll(x, w, fn, chunk=CHUNK):
    """逐股滚动算子：fn 接收 (n,S,w) 窗口块，返回 (n,S)。窗口含缺失 ⇒ 该点 NaN。"""
    D, S = x.shape
    W = _view(x, w)
    out = np.full((D, S), np.nan, np.float32)
    for c0 in range(0, W.shape[0], chunk):
        c1 = min(c0 + chunk, W.shape[0])
        blk = W[c0:c1]
        fin = np.isfinite(blk).all(axis=-1)
        r = fn(blk)
        out[w - 1 + c0: w - 1 + c1] = np.where(fin, r, np.nan)
    return out


# ══════════════════════════════════════════════════ 逐元素算子
def neg(x): return -x


def abs_(x): return np.abs(x)


def sign(x): return np.sign(x)


def log(x):
    """有符号对数（保号，压缩尾部）。"""
    return np.sign(x) * np.log1p(np.abs(x))


def sqrt(x): return np.sign(x) * np.sqrt(np.abs(x))


def sqr(x): return np.sign(x) * np.square(x)


def relu(x): return np.maximum(x, 0.0)


def inv(x): return np.where(np.abs(x) < EPS, np.nan, 1.0 / x)


def add(a, b): return a + b


def sub(a, b): return a - b


def mul(a, b): return a * b


def div(a, b): return a / np.where(np.abs(b) < EPS, np.nan, b)


def max2(a, b): return np.maximum(a, b)          # = Greater(a,b)


def min2(a, b): return np.minimum(a, b)          # = Less(a,b)


def power(a, p): return np.sign(a) * np.power(np.abs(a), float(p))


def gt(x, c): return (x > c).astype(np.float32)


def lt(x, c): return (x < c).astype(np.float32)


# ══════════════════════════════════════════════════ 截面算子
def cs_rank(x):
    return rank_rows(x)


def cs_z(x):
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    return ((x - m) / np.where(s < EPS, np.nan, s)).astype(np.float32)


def cs_demean(x):
    return (x - np.nanmean(x, axis=1, keepdims=True)).astype(np.float32)


# ══════════════════════════════════════════════════ 时序算子
def ref(x, w):
    if w <= 0:
        return x
    return np.concatenate([np.full((w, x.shape[1]), np.nan, np.float32), x[:-w]])


def delta(x, w): return x - ref(x, w)


def ts_mean(x, w): return _roll(x, w, lambda b: b.mean(-1))


def ts_std(x, w): return _roll(x, w, lambda b: b.std(-1, ddof=1))


def ts_sum(x, w): return _roll(x, w, lambda b: b.sum(-1))


def ts_max(x, w): return _roll(x, w, lambda b: b.max(-1))


def ts_min(x, w): return _roll(x, w, lambda b: b.min(-1))


def ts_median(x, w): return _roll(x, w, lambda b: np.median(b, -1))


def ts_mad(x, w): return _roll(x, w, lambda b: np.abs(b - b.mean(-1, keepdims=True)).mean(-1))


def ts_rank(x, w):
    """窗口内最后一个值的百分位（0~1）。"""
    return _roll(x, w, lambda b: ((b <= b[..., -1:]).sum(-1) - 1) / (w - 1))


def ts_z(x, w):
    return _roll(x, w, lambda b: (b[..., -1] - b.mean(-1)) / np.where(b.std(-1, ddof=1) < EPS, np.nan, b.std(-1, ddof=1)))


def ts_quant(x, w, q=0.5):
    return _roll(x, w, lambda b: np.quantile(b, q, axis=-1))


def ts_argmax(x, w): return _roll(x, w, lambda b: np.argmax(b, -1).astype(np.float32))


def ts_argmin(x, w): return _roll(x, w, lambda b: np.argmin(b, -1).astype(np.float32))


def ts_skew(x, w):
    def f(b):
        m = b - b.mean(-1, keepdims=True)
        m2 = (m * m).mean(-1)
        m3 = (m ** 3).mean(-1)
        return m3 / np.where(m2 < EPS, np.nan, np.power(m2, 1.5))
    return _roll(x, w, f)


def ts_kurt(x, w):
    def f(b):
        m = b - b.mean(-1, keepdims=True)
        m2 = (m * m).mean(-1)
        m4 = (m ** 4).mean(-1)
        return m4 / np.where(m2 < EPS, np.nan, m2 * m2) - 3.0
    return _roll(x, w, f)


def ts_count(x, w):
    return _roll(np.isfinite(x).astype(np.float32), w, lambda b: b.sum(-1))


def _lr(b):
    """窗口内 x ~ a + b·t（t=0..w-1）→ (slope, resi_last, rsqr)。"""
    w = b.shape[-1]
    t = np.arange(w, dtype=np.float64)
    tb = t - t.mean()
    xb = b.mean(-1, keepdims=True)
    slope = (b * tb).sum(-1) / (tb ** 2).sum()
    fit = xb + slope[..., None] * tb
    ss_res = ((b - fit) ** 2).sum(-1)
    ss_tot = ((b - xb) ** 2).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rsqr = 1.0 - ss_res / np.where(ss_tot < EPS, np.nan, ss_tot)
    resi = b[..., -1] - (xb[..., 0] + slope * tb[-1])
    return slope, resi, rsqr


def ts_slope(x, w): return _roll(x, w, lambda b: _lr(b)[0])


def ts_resi(x, w): return _roll(x, w, lambda b: _lr(b)[1])


def ts_rsqr(x, w): return _roll(x, w, lambda b: _lr(b)[2])


def ema(x, w):
    return pd.DataFrame(np.asarray(x, dtype=np.float64)).ewm(span=w, adjust=False).mean().to_numpy(np.float32)


def ts_corr(a, b, w):
    """滚动皮尔逊相关（按股）。"""
    A_ = np.ascontiguousarray(a, dtype=np.float32)
    B_ = np.ascontiguousarray(b, dtype=np.float32)
    WA, WB = _view(A_, w), _view(B_, w)
    D, S = a.shape
    out = np.full((D, S), np.nan, np.float32)
    for c0 in range(0, WA.shape[0], CHUNK):
        c1 = min(c0 + CHUNK, WA.shape[0])
        xa, xb = WA[c0:c1], WB[c0:c1]
        fin = np.isfinite(xa).all(-1) & np.isfinite(xb).all(-1)
        da = xa - xa.mean(-1, keepdims=True)
        db = xb - xb.mean(-1, keepdims=True)
        num = (da * db).sum(-1)
        den = np.sqrt((da * da).sum(-1) * (db * db).sum(-1))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = num / np.where(den < EPS, np.nan, den)
        out[w - 1 + c0: w - 1 + c1] = np.where(fin, r, np.nan)
    return out


OPS = {
    "neg": neg, "abs": abs_, "sign": sign, "log": log, "sqrt": sqrt, "sqr": sqr,
    "relu": relu, "inv": inv, "power": power, "gt": gt, "lt": lt,
    "add": add, "sub": sub, "mul": mul, "div": div, "max2": max2, "min2": min2,
    "cs_rank": cs_rank, "cs_z": cs_z, "cs_demean": cs_demean,
    "ref": ref, "delta": delta, "ts_mean": ts_mean, "ts_std": ts_std, "ts_sum": ts_sum,
    "ts_max": ts_max, "ts_min": ts_min, "ts_median": ts_median, "ts_mad": ts_mad,
    "ts_rank": ts_rank, "ts_z": ts_z, "ts_quant": ts_quant,
    "ts_argmax": ts_argmax, "ts_argmin": ts_argmin, "ts_skew": ts_skew, "ts_kurt": ts_kurt,
    "ts_count": ts_count, "ema": ema, "ts_slope": ts_slope, "ts_resi": ts_resi, "ts_rsqr": ts_rsqr,
    "ts_corr": ts_corr,
}
NS_EXTRA = {"np": np, "nan": np.nan, "EPS": EPS}


def eval_expr(expr: str, T: dict, w: int | None = None) -> np.ndarray:
    e = expr.replace("%d", str(int(w))) if (w is not None and "%d" in expr) else expr
    ns = dict(T)
    ns.update(OPS)
    ns.update(NS_EXTRA)
    # ★ 必须屏蔽 numpy 的浮点警告：空 __builtins__ 下，警告机制会去要 __import__ ⇒ KeyError
    with np.errstate(all="ignore"):
        v = eval(e, {"__builtins__": {}}, ns)
    return np.asarray(v, dtype=np.float32)


# ══════════════════════════════════════════════════ 终端集
def build_terminals(verbose: bool = True):
    """构造 (D,S) 终端字典。价格一律复权；量纲不可比的终端预置为比值形态。"""
    t0 = time.time()
    L = lambda n: np.load(PAN / f"{n}.npy").astype(np.float32)  # noqa: E731
    fac = L("factor")
    o, h, l = L("open") * fac, L("high") * fac, L("low") * fac
    A = L("close") * fac
    pre = np.concatenate([np.full((1, A.shape[1]), np.nan, np.float32), A[:-1]])
    pre = np.where(np.isfinite(pre), pre, L("pre_close"))
    v, amt = L("vol"), L("amount")
    hl = np.where(np.abs(h - l) < EPS, np.nan, h - l)

    T = {}
    T["pre"] = pre          # 昨收复权价（跨日比较的锚，本身水平不可比）
    # ---- 无量纲 K 线形态（相对昨日复权收盘）----
    T["r1"] = A / pre - 1.0
    T["gap"] = o / pre - 1.0
    T["intra"] = A / np.where(np.abs(o) < EPS, np.nan, o) - 1.0
    T["hl"] = (h - l) / pre
    T["pos"] = np.clip((A - l) / hl, 0, 1)
    T["ush"] = (h - np.maximum(o, A)) / pre
    T["lsh"] = (np.minimum(o, A) - l) / pre
    T["body"] = (A - o) / pre
    T["vwap0"] = (amt / np.where(v < EPS, np.nan, v) / 10.0) / np.where(A < EPS, np.nan, A)
    # ---- 复权价（水平不可比 ⇒ 仅供构成比值）----
    T["A"], T["H"], T["L"], T["O"] = A, h, l, o
    # ---- 量：预置归一版 ----
    for w in (5, 20):
        T[f"vma{w}"] = v / ts_mean(v, w) - 1.0
        T[f"ama{w}"] = amt / ts_mean(amt, w) - 1.0
    T["logamt"] = np.log1p(amt)
    T["vchg1"] = v / np.where(ref(v, 1) < EPS, np.nan, ref(v, 1)) - 1.0
    # ---- 换手 / 量比（本身无量纲，新数据独有）----
    T["turn"] = L("turnover_rate_f")
    T["turn_rel"] = T["turn"] / ts_mean(T["turn"], 20) - 1.0
    T["vr"] = L("volume_ratio")
    # ---- 估值 / 规模 ----
    with np.errstate(divide="ignore", invalid="ignore"):
        T["ep"] = np.clip(1.0 / np.where(L("pe_ttm") > 0, L("pe_ttm"), np.nan), 0, 5)
        T["bp"] = np.clip(1.0 / np.where(L("pb") > 0, L("pb"), np.nan), 0, 5)
    T["dp"] = np.clip(L("dv_ttm"), 0, 0.3)
    T["logcmv"] = np.log1p(L("circ_mv"))
    # ---- 涨跌停制度（新数据独有：板块 ±10/20/30%）----
    T["toup"] = np.clip(L("to_up"), -0.5, 0.5)
    T["board"] = np.clip(L("to_up"), -0.5, 0.5) * 10.0
    # ---- 事件族（涨停日才有值）----
    T["zt"] = np.nan_to_num(L("is_zt"), nan=0.0)
    T["dt"] = np.nan_to_num(L("is_dt"), nan=0.0)
    T["st"] = np.nan_to_num(L("is_st"), nan=0.0)
    T["lbt"] = np.clip(np.nan_to_num(L("limit_times"), nan=0.0), 0, 12)  # 连板数（避开算子 lt）
    T["seal"] = np.clip(L("fd_amount") / np.where(amt < EPS, np.nan, amt), 0, 5)
    T["seal_mv"] = np.clip(L("fd_amount") / np.where(L("circ_mv") < EPS, np.nan, L("circ_mv")), 0, 1)
    T["otimes"] = np.clip(np.nan_to_num(L("open_times"), nan=0.0), 0, 12)
    T["fmin"] = np.nan_to_num(L("first_min"), nan=0.0)
    T["lmin"] = np.nan_to_num(L("last_min"), nan=0.0)
    T["upcnt"] = np.nan_to_num(L("up_cnt"), nan=0.0)
    T["updays"] = np.nan_to_num(L("up_days"), nan=0.0)
    T["zt5"] = ts_sum(T["zt"], 5)
    T["zt20"] = ts_sum(T["zt"], 20)
    ii = np.arange(A.shape[0])[:, None]
    _arr = np.where(T["zt"] > 0, ii, -1)
    np.maximum.accumulate(_arr, axis=0, out=_arr)
    T["dszt"] = np.where(_arr < 0, np.nan, ii - _arr).astype(np.float32)
    # ---- 市场级（逐日常数；仅供 ts_corr / 调制）----
    mkt = {}
    mkt["mkt_ret"] = np.nanmean(T["r1"], axis=1)
    mkt["mkt_std"] = np.nanstd(T["r1"], axis=1)
    mkt["mkt_zt"] = np.nanmean(T["zt"], axis=1)
    if verbose:
        print(f"[terminals] {len(T)} 个截面终端 + {len(mkt)} 个市场终端  ({time.time()-t0:.1f}s)")
    return T, mkt


def causal_audit(expr: str, T: dict, w: int = 20, T_cut: int = 1200, seed: int = 0) -> bool:
    """把 t>=T_cut 的终端整体打乱，断言 t<T_cut 的因子逐位不变（比截断重算更强）。"""
    rng = np.random.default_rng(seed)
    T2 = {}
    for k, val in T.items():
        v2 = np.array(val, copy=True)
        perm = rng.permutation(v2.shape[0] - T_cut) + T_cut
        v2[T_cut:] = v2[perm]
        T2[k] = v2
    a = eval_expr(expr, T, w)
    b = eval_expr(expr, T2, w)
    return bool(np.array_equal(np.nan_to_num(a[:T_cut]), np.nan_to_num(b[:T_cut])))


# ══════════════════════════════════════════════════ 单因子评估
def load_ctx() -> dict:
    """载入评估上下文。

    三段时序切分**按样本比例**（早期 60% / 中期 15% / 近期 25%），刻意不用任何绝对日期，
    以保证切分点完全由数据自身决定、可按需调整。
    """
    dates = np.load(PAN / "dates.npy").astype(np.int64)
    y1 = np.load(PAN / "y1.npy").astype(np.float32)
    st = np.nan_to_num(np.load(PAN / "is_st.npy").astype(np.float32), nan=0.0)
    zt = np.nan_to_num(np.load(PAN / "is_zt.npy").astype(np.float32), nan=0.0)
    dt = np.nan_to_num(np.load(PAN / "is_dt.npy").astype(np.float32), nan=0.0)
    D = dates.shape[0]
    yr = np.array([int(str(d)[:4]) for d in dates])
    i1, i2 = int(SEG_FRAC[0] * D), int(SEG_FRAC[1] * D)
    pos = np.arange(D)
    seg = np.where(pos < i1, "train", np.where(pos < i2, "val", "test"))
    return dict(dates=dates, y1=y1, D=D, year=yr, seg=seg,
                m_full=(st == 0) & (dt == 0),
                m_tradable=(st == 0) & (dt == 0) & (zt == 0))


def factor_ic_series(F, y, mask, idx=None):
    if idx is None:
        idx = np.arange(F.shape[0])
    out = []
    for t in idx:
        m = mask[t] & np.isfinite(F[t]) & np.isfinite(y[t])
        if int(m.sum()) < 100:
            continue
        ra = rank_rows(F[t][None, m])[0]
        rb = rank_rows(y[t][None, m])[0]
        if np.nanstd(ra) > 0 and np.nanstd(rb) > 0:
            cc = float(np.corrcoef(ra, rb)[0, 1])
            if np.isfinite(cc):
                out.append(cc)
    return np.asarray(out, dtype=np.float64)


def _stat(ic: np.ndarray) -> dict:
    if len(ic) < 5:
        return dict(ic=np.nan, icir=np.nan, t=np.nan, pos=np.nan, n=len(ic))
    sd = ic.std(ddof=1)
    return dict(ic=float(ic.mean()),
                icir=float(ic.mean() / sd) if sd > 0 else np.nan,
                t=float(ic.mean() / sd * np.sqrt(len(ic))) if sd > 0 else np.nan,
                pos=float((ic > 0).mean()), n=int(len(ic)))


def ic_summary(F, ctx, idx=None) -> dict:
    y, D = ctx["y1"], ctx["D"]
    if idx is None:
        idx = np.arange(D)
    rec = {}
    for tag, mask in (("full", ctx["m_full"]), ("trad", ctx["m_tradable"])):
        s = _stat(factor_ic_series(F, y, mask, idx))
        for k, v in s.items():
            rec[f"{k}_{tag}"] = v
    rec["cover"] = float(np.isfinite(F[idx]).mean())
    return rec


def seg_ic(F, ctx, tag: str, col: str = "seg") -> dict:
    arr = ctx[col]
    idx = np.flatnonzero(arr == tag)
    return _stat(factor_ic_series(F, ctx["y1"], ctx["m_full"], idx))


def year_ic(F, ctx) -> dict:
    out = {}
    for y in sorted(set(ctx["year"].tolist())):
        idx = np.flatnonzero(ctx["year"] == y)
        out[int(y)] = _stat(factor_ic_series(F, ctx["y1"], ctx["m_full"], idx))["ic"]
    return out


def max_rho_vs_pool(F, pool: dict, days=14) -> tuple:
    """候选因子与特征池的最大 |截面秩相关|（在共同支撑集上重排序计算）。"""
    D = F.shape[0]
    pick = np.linspace(int(0.35 * D), D - 1, days).astype(int)
    best, bname = 0.0, ""
    for t in pick:
        a = F[t]
        if int(np.isfinite(a).sum()) < 200:
            continue
        for nm, P in pool.items():
            b = P[t]
            m = np.isfinite(a) & np.isfinite(b)
            if int(m.sum()) < 200:
                continue
            ra = rank_rows(a[None, m])[0]
            rb = rank_rows(b[None, m])[0]
            if np.nanstd(ra) <= 0 or np.nanstd(rb) <= 0:
                continue
            c = abs(float(np.corrcoef(ra, rb)[0, 1]))
            if np.isfinite(c) and c > best:
                best, bname = c, nm
    return best, bname


def load_pool() -> dict:
    pool = {}
    if not FEAT.exists():
        return pool
    for f in sorted(FEAT.glob("X_*.npy")):
        g = f.stem[2:]
        nm = json.load(open(FEAT / f"names_{g}.json", encoding="utf-8"))
        mm = np.load(f, mmap_mode="r")
        for k in range(mm.shape[2]):
            pool[f"{g}:{nm[k]}"] = mm[:, :, k]
    return pool


if __name__ == "__main__":
    T, mkt = build_terminals()
    ctx = load_ctx()
    print("终端:", len(T))
    for e in ["ush", "ts_mean(ush,20)", "cs_rank(ts_mean(ush,20))", "ts_argmax(H,20)"]:
        F = eval_expr(e, T, 20)
        r = ic_summary(F, ctx)
        print(f"  {e:30s} IC={r['ic_full']:+.5f} ICIR={r['icir_full']:+.3f} cov={r['cover']:.2f}")
    print("因果审计:", causal_audit("ts_slope(A,20)*sign(ts_mean(ush,20))", T, 20))
