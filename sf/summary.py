# -*- coding: utf-8 -*-
"""sf/summary.py — 汇总各路线结果，生成报告用表格（markdown）"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "_out"


def load():
    recs = {}
    for tag in ("base", "new", "opt", "gen"):
        p = OUT / f"ic_table_{tag}.json"
        if p.exists():
            for r in json.load(open(p, encoding="utf-8")):
                if r.get("status") == "OK":
                    recs[r["name"]] = r
    return recs


def tab(rows, cols, hdr):
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                vals.append(f"{v:+.4f}" if c in ("ic_full", "seg_train", "seg_val", "seg_test") else f"{v:+.3f}")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main():
    R = load()
    print("合格因子总数:", len(R))

    def cross_min_icir(k=14, icir=0.35):
        xs = [r for r in R.values() if r.get("cover", 0) > 0.5 and abs(r.get("icir_full", 0)) >= icir]
        return sorted(xs, key=lambda r: -abs(r["ic_full"]))

    print("\n### 全截面 |ICIR|≥0.35 的因子（按 |IC| 排序）")
    print(tab(cross_min_icir(), ["group", "name", "expr", "ic_full", "icir_full", "cover",
                                 "seg_train", "seg_val", "seg_test"],
              ["族", "因子", "表达式", "IC", "ICIR", "覆盖", "train", "val", "test"]))

    print("\n### 路线1：基础因子 → 最优改进（同义对比）")
    pairs = [("STD", "STD_opt1"), ("MA", "MA_opt2"), ("VQTLD", "VQTLD_opt1"), ("VMIN", "VMIN_opt1"),
             ("KLOW1", "KLOW1_opt1"), ("KLEN", "KLEN_opt2"), ("MIN", "MIN_opt1"), ("VRANK", "VRANK_opt1"),
             ("CORD", "CORD_opt1"), ("LOW0", "LOW0_opt2")]
    for b, o in pairs:
        if b in R and o in R:
            rb, ro = R[b], R[o]
            print(f"| {b} | {rb['ic_full']:+.4f}/{rb['icir_full']:+.3f} | {o} | "
                  f"{ro['ic_full']:+.4f}/{ro['icir_full']:+.3f} | "
                  f"{100*(abs(ro['ic_full'])/abs(rb['ic_full'])-1):+.1f}% |  {rb['expr']} | {ro['expr']} |")

    print("\n### 秩免疫对照")
    for a, b in [("VRANK", "CTRL_mono"), ("TURN_ACCEL", "TURN_ACCEL_LOG")]:
        if a in R and b in R:
            print(f"| {a} | {R[a]['expr']} | {R[a]['ic_full']:+.6f} |")
            print(f"| {b} | {R[b]['expr']} | {R[b]['ic_full']:+.6f} |")

    print("\n### 低覆盖事件因子（cover<0.05，只能作事件信号）")
    for r in sorted([x for x in R.values() if x.get("cover", 1) < 0.05], key=lambda x: -abs(x["ic_full"])):
        print(f"  {r['group']:<4}{r['name']:<14} IC={r['ic_full']:+.4f} ICIR={r['icir_full']:+.3f} cover={r['cover']:.3f} t={r['t_full']:+.1f}  {r['expr']}")

    print("\n### 各路线 |IC| 中位数 / |ICIR|≥0.35 数量")
    for tag, lab in (("base", "路线0 基础(Alpha158等价)"), ("new", "新数据独有原料"), ("opt", "路线1 优化"), ("gen", "路线2 生成")):
        p = OUT / f"ic_table_{tag}.json"
        if not p.exists():
            continue
        rs = [r for r in json.load(open(p, encoding="utf-8")) if r.get("status") == "OK"]
        cross = [r for r in rs if r.get("cover", 0) > 0.5]
        print(f"  {lab:<22} n={len(rs):>3}  |IC|中位={np.median([abs(r['ic_full']) for r in cross]):.4f}"
              f"  |ICIR|≥0.35: {sum(1 for r in cross if abs(r['icir_full'])>=0.35)}"
              f"  三段同号: {sum(1 for r in cross if np.sign(r['seg_train'])==np.sign(r['seg_test'])==np.sign(r['seg_val']))}")


if __name__ == "__main__":
    main()
