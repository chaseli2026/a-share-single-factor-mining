# -*- coding: utf-8 -*-
"""sf/rho.py — 低相关性诊断：候选因子 vs 既有特征池、vs Alpha158 样例集"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import ops as O
import factors as FA

HERE = Path(__file__).parent
OUT = HERE / "_out"


def main(topk: int = 32):
    T, _ = O.build_terminals(verbose=False)
    ctx = O.load_ctx()
    # 汇总全部已评估因子
    recs = {}
    for tag in ("base", "new", "opt", "gen"):
        f = OUT / f"ic_table_{tag}.json"
        if not f.exists():
            continue
        for r in json.load(open(f, encoding="utf-8")):
            if r.get("status") == "OK" and np.isfinite(r.get("ic_full", np.nan)):
                recs[r["name"]] = r
    print(f"已评估合格因子 {len(recs)} 个")

    pool = O.load_pool()
    print(f"既有特征池 {len(pool)} 列")
    # Alpha158 样例集（作第二参照系）
    samp = {}
    for g, nm, e in FA.BASE158:
        try:
            samp[nm] = O.eval_expr(e, T, 20)
        except Exception:
            pass
    print(f"样例集(Alpha158) {len(samp)} 个")

    # 只对「全截面」因子排名（覆盖率 > 0.5），事件因子单列
    cross = [r for r in recs.values() if r.get("cover", 0) > 0.5]
    cross.sort(key=lambda r: -abs(r["ic_full"]))
    pick = cross[:topk]
    print(f"取全截面因子 |IC| 前 {len(pick)} 个做相关性诊断 ...\n")

    rows = []
    for i, r in enumerate(pick, 1):
        F = O.eval_expr(r["expr"], T, 20)
        mp, nnp = O.max_rho_vs_pool(F, pool, days=10)
        ms, nns = O.max_rho_vs_pool(F, samp, days=10)
        rows.append(dict(name=r["name"], group=r["group"], ic=r["ic_full"], icir=r["icir_full"],
                         cover=r["cover"], rho_pool=mp, rho_pool_nn=nnp, rho_samp=ms, rho_samp_nn=nns))
        print(f"  [{i:>2}] {r['group']:<4}{r['name']:<16} IC={r['ic_full']:+.4f} "
              f"max|ρ|池={mp:.3f}({nnp[:26]}) 样例={ms:.3f}", flush=True)

    OUT.joinpath("rho_table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    lo = [x for x in rows if x["rho_pool"] < 0.5]
    print(f"\n与既有池 max|ρ|<0.5（真新信息）的：{len(lo)}/{len(rows)}")
    print(f"与既有池 max|ρ|<0.7 的：{sum(1 for x in rows if x['rho_pool']<0.7)}/{len(rows)}")
    print(f"median max|ρ|(池) = {np.median([x['rho_pool'] for x in rows]):.3f}")


if __name__ == "__main__":
    main()
