# -*- coding: utf-8 -*-
"""sf/rho2.py — 新颖度量化：分族相关性 + 参照系（样例集/基础池的内部冗余度）"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import ops as O
import factors as FA

HERE = Path(__file__).parent
OUT = HERE / "_out"


def max_rho_pair(F, G, days=8):
    D = F.shape[0]
    pick = np.linspace(int(0.35 * D), D - 1, days).astype(int)
    best = 0.0
    for t in pick:
        a, b = F[t], G[t]
        m = np.isfinite(a) & np.isfinite(b)
        if int(m.sum()) < 200:
            continue
        ra = O.rank_rows(a[None, m])[0]
        rb = O.rank_rows(b[None, m])[0]
        if np.nanstd(ra) <= 0 or np.nanstd(rb) <= 0:
            continue
        c = abs(float(np.corrcoef(ra, rb)[0, 1]))
        if np.isfinite(c):
            best = max(best, c)
    return best


def main():
    T, _ = O.build_terminals(verbose=False)
    # 样例集（Alpha158 等价）
    samp = {}
    for g, nm, e in FA.BASE158:
        try:
            samp[nm] = O.eval_expr(e, T, 20)
        except Exception:
            pass
    # 参照系：样例集内部两两 max|ρ| 的中位数
    names = list(samp)
    vals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            vals.append(max_rho_pair(samp[names[i]], samp[names[j]]))
    yard = float(np.median(vals))
    print(f"参照系：Alpha158 等价样例集内部两两 max|ρ|  —— n={len(vals)} 对  "
          f"中位 {yard:.3f}  均值 {np.mean(vals):.3f}  p10 {np.percentile(vals,10):.3f}")
    print(f"  ⇒ 一个因子只要与样例集的 max|ρ| 低于 {yard:.3f}，就比样例集『自身内部的相似度』还低\n")

    # 分族池
    groups = {}
    for f in sorted(O.FEAT.glob("X_*.npy")):
        g = f.stem[2:]
        if g == "mined":
            continue
        nm = json.load(open(O.FEAT / f"names_{g}.json", encoding="utf-8"))
        mm = np.load(f, mmap_mode="r")
        groups[g] = {nm[k]: mm[:, :, k] for k in range(mm.shape[2])}
    print("分族池:", {k: len(v) for k, v in groups.items()}, "\n")

    # 取全截面 |IC| 前 25
    recs = {}
    for tag in ("base", "new", "opt", "gen"):
        p = OUT / f"ic_table_{tag}.json"
        if p.exists():
            for r in json.load(open(p, encoding="utf-8")):
                if r.get("status") == "OK" and np.isfinite(r.get("ic_full", np.nan)):
                    recs[r["name"]] = r
    cross = sorted([r for r in recs.values() if r.get("cover", 0) > 0.5],
                   key=lambda r: -abs(r["ic_full"]))[:25]

    rows = []
    for r in cross:
        F = O.eval_expr(r["expr"], T, 20)
        rs = max_rho_pair(F, samp[names[0]]) if False else None
        # 与样例集（逐因子取最大）
        ms = max(max_rho_pair(F, samp[n]) for n in names)
        mg = {g: max(max_rho_pair(F, P) for P in d.values()) if d else 0.0 for g, d in groups.items()}
        new_flag = "★新" if ms < yard else ""
        rows.append(dict(name=r["name"], group=r["group"], ic=r["ic_full"], rho_samp=ms, **{f"rho_{k}": v for k, v in mg.items()}))
        print(f"  {r['group']:<4}{r['name']:<16} IC={r['ic_full']:+.4f}  "
              f"样例={ms:.3f} {new_flag:<4} "
              + " ".join(f"{k}={v:.2f}" for k, v in mg.items()), flush=True)

    OUT.joinpath("rho2_table.json").write_text(json.dumps({"yardstick": yard, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    n_new = sum(1 for x in rows if x["rho_samp"] < yard)
    print(f"\n与样例集 max|ρ| 低于参照系({yard:.3f}) 的：{n_new}/{len(rows)}")
    print(f"与样例集 max|ρ| < 0.8 的：{sum(1 for x in rows if x['rho_samp']<0.8)}/{len(rows)}")


if __name__ == "__main__":
    main()
