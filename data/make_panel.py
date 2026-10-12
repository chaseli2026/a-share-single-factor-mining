# -*- coding: utf-8 -*-
"""
make_panel.py — 把「长表」（一行 = 一只股票 × 一个交易日）转成本仓库需要的 (D, S) 面板 .npy。

用法
----
    python data/make_panel.py --src /path/to/long_table.parquet --out data/panel

说明
----
* 只做「长表 → 面板」的形状转换与标签计算，**不带任何数据源凭据**。
  取数请用你惯用的行情接口自行完成，并把结果落成长表。
* 若某列缺失，对应的 .npy 不会生成（`ops.py` 会跳过依赖它的终端）。
* 标签 `y1` = 次日复权收益率，由复权连续价直接算出，避免除权跳空污染。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 必需的源长表列（缺任一列会报错或降级，见下方注释）
REQUIRED = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
            "vol", "amount", "turnover_rate_f", "factor"]
# 可选列：存在就写入面板，缺失则跳过（对应因子会自动不可用）
OPTIONAL = ["volume_ratio", "circ_mv", "pe_ttm", "pb", "ps_ttm", "dv_ttm",
            "fd_amount", "limit_times", "open_times", "up_cnt", "up_days",
            "is_zt", "is_dt", "is_st", "up_limit", "down_limit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="长表路径（parquet/csv）")
    ap.add_argument("--out", default="data/panel")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    src = pd.read_parquet(args.src) if args.src.endswith(".parquet") else pd.read_csv(args.src)
    src["trade_date"] = src["trade_date"].astype(str).str.replace("-", "").astype(np.int64)

    dates = np.sort(src["trade_date"].unique()).astype(np.int64)
    codes = np.sort(src["ts_code"].astype(str).unique())
    D, S = len(dates), len(codes)
    d2i = {int(d): i for i, d in enumerate(dates)}
    c2i = {c: i for i, c in enumerate(codes)}
    di = src["trade_date"].map(d2i).to_numpy(np.int64)
    si = src["ts_code"].astype(str).map(c2i).to_numpy(np.int64)
    print(f"面板 {D} 日 × {S} 只，源长表 {len(src):,} 行")

    def put(name, series, dtype=np.float32):
        A = np.full((D, S), np.nan, np.float64)
        A[di, si] = pd.to_numeric(series, errors="coerce").to_numpy(np.float64)
        np.save(out / f"{name}.npy", A.astype(dtype))
        return A

    # ---- 必需字段 ----
    put("open", src["open"]); put("high", src["high"])
    put("low", src["low"]); put("close", src["close"])
    put("pre_close", src["pre_close"])
    put("vol", src["vol"]); put("amount", src["amount"])
    put("turnover_rate_f", src["turnover_rate_f"])
    fac = put("factor", src["factor"]) if "factor" in src.columns else np.ones((D, S))

    # ---- 标志位 ----
    for nm, col in [("is_zt", "is_zt"), ("is_dt", "is_dt"), ("is_st", "is_st")]:
        if col in src.columns:
            put(nm, src[col])
        else:
            print(f"  [警告] 缺 {col} → {nm} 置 0（会损失「剔停牌」口径与事件族因子）")
            np.save(out / f"{nm}.npy", np.zeros((D, S), np.float32))

    # ---- 可选字段 ----
    for col, dst in [("volume_ratio", "volume_ratio"), ("circ_mv", "circ_mv"),
                     ("pe_ttm", "pe_ttm"), ("pb", "pb"), ("ps_ttm", "ps_ttm"),
                     ("dv_ttm", "dv_ttm"), ("fd_amount", "fd_amount"),
                     ("limit_times", "limit_times"), ("open_times", "open_times"),
                     ("up_cnt", "up_cnt"), ("up_days", "up_days")]:
        if col in src.columns:
            put(dst, src[col])

    # ---- 涨跌停幅度 ----
    if {"up_limit", "down_limit"} <= set(src.columns):
        pc = put("_pc", src["pre_close"])
        ul = put("_ul", src["up_limit"]); dl = put("_dl", src["down_limit"])
        with np.errstate(all="ignore"):
            np.save(out / "to_up.npy", ((ul - pc) / pc).astype(np.float32))
            np.save(out / "to_dn.npy", ((pc - dl) / pc).astype(np.float32))
        for t in ("_pc", "_ul", "_dl"):
            (out / f"{t}.npy").unlink(missing_ok=True)

    # ---- 标签：次日复权收益 ----
    A = put("close", src["close"]) * fac                     # 复权连续价 (D,S)
    pc_prev = np.vstack([np.full((1, S), np.nan), A[:-1]])   # 昨日复权价
    denom = np.where(pc_prev == 0, np.nan, pc_prev)
    y1 = np.full((D, S), np.nan, np.float32)
    y1[1:] = (A[1:] / denom[1:] - 1.0).astype(np.float32)    # y1[t] = A[t]/A[t-1] - 1 ⇒ t-1 日的次日收益
    np.save(out / "y1.npy", y1)

    np.save(out / "dates.npy", dates)
    np.save(out / "codes.npy", np.array(codes, dtype=object))
    print(f"完成 → {out}（y1 有效率 {np.isfinite(y1).mean():.3f}）")


if __name__ == "__main__":
    main()
