# -*- coding: utf-8 -*-
"""
sf/run.py — 单因子评估驱动器

用法：
  python run.py --stage base                # 评估 BASE158 + NEWDATA（w=20）
  python run.py --stage base --rho 30       # 额外算 top-30 的 max|ρ|（对既有 126 列）
  python run.py --stage all --tag v2        # 评估 BASE+NEW+OPT+NEW候选
输出：_sf/ic_table_<tag>.json / .csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import ops as O
import factors as FA

HERE = Path(__file__).parent
OUT = HERE / "_out"
OUT.mkdir(exist_ok=True)


def eval_one(expr: str, T: dict, ctx: dict, w: int = 20) -> dict:
    try:
        F = O.eval_expr(expr, T, w)
    except Exception as e:  # ★ 绝不静默
        return {"status": "EVAL_FAIL", "err": f"{type(e).__name__}: {str(e)[:90]}"}
    if F.shape != (ctx["D"], T["A"].shape[1]):
        return {"status": "SHAPE_FAIL", "err": str(F.shape)}
    r = O.ic_summary(F, ctx)
    r.update({f"seg_{k}": v["ic"] for k, v in
              ((s, O.seg_ic(F, ctx, s)) for s in ("train", "val", "test"))})
    r["years"] = O.year_ic(F, ctx)
    r["status"] = "OK"
    return r, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="base", choices=["base", "new", "opt", "gen", "all"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--w", type=int, default=20)
    ap.add_argument("--rho", type=int, default=0, help="对 |IC| 前 N 个算 max|ρ|")
    ap.add_argument("--rho-days", type=int, default=10)
    args = ap.parse_args()

    tag = args.tag or args.stage
    T, _ = O.build_terminals()
    ctx = O.load_ctx()

    if args.stage == "base":
        items = FA.BASE158
    elif args.stage == "new":
        items = FA.NEWDATA
    elif args.stage == "opt":
        items = FA.OPT
    elif args.stage == "gen":
        items = FA.NEW
    else:
        items = FA.BASE158 + FA.NEWDATA + FA.OPT + FA.NEW

    print(f"=== 评估 {len(items)} 个因子（stage={args.stage}, w={args.w}）===")
    rows, keepF = [], {}
    t0 = time.time()
    for i, (grp, nm, expr) in enumerate(items, 1):
        out = eval_one(expr, T, ctx, args.w)
        if isinstance(out, dict):
            print(f"  !! {nm:16s} {out['status']} {out.get('err','')}")
            rec = {"group": grp, "name": nm, "expr": expr, **out}
        else:
            r, F = out
            rec = {"group": grp, "name": nm, "expr": expr, **r}
            keepF[nm] = F
        rows.append(rec)
        ic = rec.get("ic_full", np.nan)
        print(f"  [{i:>3}/{len(items)}] {grp:<4}{nm:16s} IC={ic:+.4f} "
              f"ICIR={rec.get('icir_full', float('nan')):+.3f} cov={rec.get('cover', float('nan')):.2f}",
              flush=True)

    # ---- 可选：低相关性诊断（对既有 126 列）----
    if args.rho > 0:
        pool = O.load_pool()
        print(f"  特征池 {len(pool)} 列；对 |IC| 前 {args.rho} 个算 max|ρ| ...", flush=True)
        ok = sorted([r for r in rows if r.get("status") == "OK"],
                    key=lambda r: -abs(r["ic_full"]))[:args.rho]
        for r in ok:
            m, nn = O.max_rho_vs_pool(keepF[r["name"]], pool, args.rho_days)
            r["max_rho"], r["rho_nn"] = round(m, 4), nn
            print(f"    {r['name']:16s} max|ρ|={m:.3f}  ({nn})", flush=True)

    (OUT / f"ic_table_{tag}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- CSV（拍平）----
    cols = ["group", "name", "expr", "status", "ic_full", "icir_full", "t_full", "pos_full",
            "ic_trad", "icir_trad", "t_trad", "cover", "seg_train", "seg_val", "seg_test",
            "max_rho", "rho_nn"]
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in cols))
    (OUT / f"ic_table_{tag}.csv").write_text("\n".join(lines), encoding="utf-8")

    # ---- 排行 ----
    ok = [r for r in rows if r.get("status") == "OK" and np.isfinite(r["ic_full"])]
    print(f"\n=== |IC| 前 25（stage={tag}，{time.time()-t0:.0f}s，成功 {len(ok)}/{len(items)}）===")
    print(f"{'group':<4}{'name':<18}{'IC':>9}{'ICIR':>8}{'t':>7}{'cov':>6}"
          f"{'train':>8}{'val':>8}{'test':>8}")
    for r in sorted(ok, key=lambda x: -abs(x["ic_full"]))[:25]:
        print(f"{r['group']:<4}{r['name']:<18}{r['ic_full']:>+9.4f}{r['icir_full']:>+8.3f}"
              f"{r['t_full']:>+7.1f}{r['cover']:>6.2f}{r['seg_train']:>+8.4f}"
              f"{r['seg_val']:>+8.4f}{r['seg_test']:>+8.4f}")
    print(f"\n|ICIR|≥0.35 的因子："
          f"{sum(1 for r in ok if abs(r['icir_full'])>=0.35)} 个；"
          f"|IC|≥0.03 的：{sum(1 for r in ok if abs(r['ic_full'])>=0.03)} 个")


if __name__ == "__main__":
    main()
