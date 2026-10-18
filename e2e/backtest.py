# -*- coding: utf-8 -*-
"""e2e/backtest.py — 汇总多种子结果、出净值曲线与指标表

用法：
  python backtest.py                     # 读 ../results/e2e_*.json
  python backtest.py --start 2025-01-24  # 指定起点（默认用测试段起点）

产出：
  results/E2E-Sharpe-策略净值-<时间戳>.png    净值 + 回撤
  results/e2e_metrics.csv                    三臂指标对照表
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from data import Panel

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("E2E_OUT", HERE.parent / "results"))
ARMS = [("mse", "两阶段：先最小化 MSE 再组合", "#8C8C8C", "--"),
        ("ic", "RankIC 损失（本项目原口径）", "#2E6DA4", "-."),
        ("sharpe", "端到端：直接优化 Sharpe", "#E8833A", "-")]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 200


def load(arm: str):
    fs = sorted(RES.glob(f"e2e_{arm}_s*.json"))
    if not fs:
        return None
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in fs]
    return runs


def stats(r) -> dict:
    r = np.asarray(r, dtype=np.float64)
    eq = np.cumprod(1 + r)
    n = len(r)
    ann = eq[-1] ** (252 / n) - 1
    vol = r.std(ddof=1) * np.sqrt(252)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {"ann_ret": ann, "ann_vol": vol, "sharpe": ann / vol if vol > 0 else np.nan,
            "mdd": mdd, "calmar": ann / abs(mdd) if mdd < 0 else np.nan,
            "win": float((r > 0).mean()), "total": eq[-1] - 1, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="test", choices=["val", "test"])
    a = ap.parse_args()

    panel = Panel(feat_files=("X_base55.npy",))
    lo, hi = panel.segs[a.seg]
    days, bench = [], []
    for t in range(lo, hi):
        m = panel.m_trad[t]
        if m.sum() < 100:
            continue
        days.append(t)
        bench.append(float(panel.y[t][m].mean()))
    bench = np.asarray(bench)
    dates = panel.dates[days]

    rows, curves = [], {}
    for arm, name, color, ls in ARMS:
        runs = load(arm)
        if not runs:
            continue
        per_seed = [np.asarray(r[a.seg]["daily"]) for r in runs]
        L = min(len(x) for x in per_seed)
        per_seed = np.stack([x[:L] for x in per_seed])
        r = per_seed.mean(0)                      # 多种子平均（同一时间轴）
        curves[arm] = (name, color, ls, r)
        st = stats(r)
        ics = [x[a.seg]["ic_mean"] for x in runs if x[a.seg]["ic_mean"] == x[a.seg]["ic_mean"]]
        sh = [stats(x[a.seg]["daily"])["sharpe"] for x in runs]
        rows.append({"arm": arm, "name": name, "seeds": len(runs),
                     **{k: round(float(v), 4) for k, v in st.items()},
                     "sharpe_seed_mean": round(float(np.mean(sh)), 3),
                     "sharpe_seed_sd": round(float(np.std(sh, ddof=1)), 3) if len(sh) > 1 else np.nan,
                     "ic_mean": round(float(np.mean(ics)), 4) if ics else np.nan,
                     "turnover_ann": round(float(np.mean([x[a.seg]["turnover_ann"] for x in runs])), 1)})
    b = stats(bench)
    rows.append({"arm": "equal", "name": "全市场等权（可交易，未计成本）", "seeds": 1,
                 **{k: round(float(v), 4) for k, v in b.items()},
                 "sharpe_seed_mean": round(b["sharpe"], 3), "sharpe_seed_sd": 0.0,
                 "ic_mean": np.nan, "turnover_ann": np.nan})

    RES.mkdir(parents=True, exist_ok=True)
    csv_p = RES / "e2e_metrics.csv"
    with csv_p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ── 绘图 ──
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), height_ratios=[2.1, 1.0], sharex=True,
                             gridspec_kw={"hspace": 0.12})
    ax = axes[0]
    x = np.arange(len(bench))
    ax.plot(x, np.cumprod(1 + bench), color="#BFBFBF", lw=1.1, label="全市场等权（未计成本）")
    for arm, (name, color, ls, r) in curves.items():
        if len(r) < len(bench):
            continue
        ax.plot(x, np.cumprod(1 + r), color=color, lw=1.7, ls=ls, label=name)
    ax.set_yscale("log")
    ax.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(mtick.NullFormatter())
    ax.set_ylabel("累计净值（对数轴）", fontsize=10)
    ax.axhline(1.0, color="#CCCCCC", lw=0.8)
    ax.grid(axis="y", color="#EEEEEE", lw=0.7)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title(f"端到端 Sharpe 优化 vs 两阶段基线　｜　{'测试' if a.seg=='test' else '验证'}段 "
                 f"{dates[0]}–{dates[-1]}（{len(bench)} 个交易日）", fontsize=11, pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    def dd(r):
        eq = np.cumprod(1 + r); return eq / np.maximum.accumulate(eq) - 1
    ax.fill_between(x, dd(bench), 0, color="#DDDDDD", alpha=0.9, label="全市场等权")
    for arm, (name, color, ls, r) in curves.items():
        if len(r) == len(bench):
            ax.plot(x, dd(r), color=color, lw=1.2, ls=ls)
    ax.set_ylabel("回撤", fontsize=10)
    ax.set_xlabel("交易日序号（测试段起点 = 0）", fontsize=10)
    ax.grid(axis="y", color="#EEEEEE", lw=0.7)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    png = RES / f"E2E-Sharpe-策略净值-{ts}.png"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"{'臂':<8}{'年化':>9}{'波动':>8}{'Sharpe':>9}{'种子sd':>8}{'MDD':>9}{'RankIC':>9}{'换手':>7}")
    for r in rows:
        print(f"{r['arm']:<8}{r['ann_ret']:>8.1%}{r['ann_vol']:>8.1%}{r['sharpe']:>9.2f}"
              f"{r['sharpe_seed_sd']:>8.2f}{r['mdd']:>9.1%}{r['ic_mean']:>9.4f}"
              f"{(r['turnover_ann'] if r['turnover_ann']==r['turnover_ann'] else float('nan')):>7.0f}")
    print("→", png)
    print("→", csv_p)


if __name__ == "__main__":
    main()
