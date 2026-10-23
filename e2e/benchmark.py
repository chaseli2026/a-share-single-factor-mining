# -*- coding: utf-8 -*-
"""e2e/benchmark.py — 指数基准：从 Tushare 拉取并对齐到面板交易日

为什么基准要单独做
------------------
`backtest.py` 里原来的「全市场等权」是**用同一份面板自己算的**，它是**内部参照**
（同一个可交易池、同样不含成本），便于隔离"选股"这一件事；
但它**不是可投资标的**——没人能真的买到"全市场等权、零成本"。

对外的基准必须是**真实可买、有公开行情**的指数。本策略从全市场选 80 只等权持有，
风格天然偏中小盘，所以主基准取：

    中证1000（000852.SH）  ← 主基准：与持仓的市值/波动分布最接近
    中证2000（932000.CSI） ← 更小盘的对照
    中证500 （000905.SH）  ← 中盘对照
    中证全指（000985.CSI） ← 全市场代表（若接口超时则跳过）
    国证2000（399303.SZ）
    沪深300 （000300.SH）  ← 大盘对照
    上证指数（000001.SH）

用法
----
    export TUSHARE_TOKEN=xxxx          # 或 --token
    python benchmark.py                # 拉取 → 缓存 parquet → 打印对齐报告
    python benchmark.py --offline      # 只用缓存

缓存：`<SF_BASE>/_ts/bench/benchmark.parquet`（长表：date, code, close, ret）
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
BASE = Path(os.environ.get("SF_BASE", _HERE.parents[1]))
CACHE = Path(os.environ.get("SF_BENCH", BASE / "_ts" / "bench")) / "benchmark.parquet"

# 顺序即"主基准优先级"：第一个是主基准
BENCH = [("000852.SH", "中证1000"),
         ("932000.CSI", "中证2000"),
         ("000905.SH", "中证500"),
         ("000985.CSI", "中证全指"),
         ("399303.SZ", "国证2000"),
         ("000300.SH", "沪深300"),
         ("000001.SH", "上证指数")]
MAIN = "000852.SH"


def _pro(token: str | None):
    import tushare as ts
    tok = token or os.environ.get("TUSHARE_TOKEN", "")
    if not tok:
        raise SystemExit("缺少 Tushare token：请设置环境变量 TUSHARE_TOKEN（或在 ~/.tushare_token 写入）")
    pro = ts.pro_api(tok)
    ep = os.environ.get("TUSHARE_ENDPOINT", "https://fastapic.stockai888.top")
    pro._DataApi__http_url = ep
    pro._DataApi__timeout = 30
    return pro


def fetch(dates: np.ndarray, token: str | None = None, retry: int = 3) -> pd.DataFrame:
    """按面板交易日区间拉指数日线；返回长表 date/code/name/close/ret。"""
    pro = _pro(token)
    d0, d1 = str(int(dates[0])), str(int(dates[-1]))
    frames = []
    for code, name in BENCH:
        for k in range(retry):
            try:
                df = pro.index_daily(ts_code=code, start_date=d0, end_date=d1)
                break
            except Exception as e:                       # 大请求偶发 ReadTimeout
                if k == retry - 1:
                    print(f"  {code} {name} 拉取失败，跳过：{type(e).__name__}")
                    df = None
                else:
                    time.sleep(3 * (k + 1))
        if df is None or len(df) == 0:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["date"] = df["trade_date"].astype(int)
        df["code"], df["name"] = code, name
        frames.append(df[["date", "code", "name", "close", "pct_chg"]])
    if not frames:
        raise SystemExit("一个指数都没拉到，检查 token / 网络 / 自定义端点")
    out = pd.concat(frames, ignore_index=True)
    # 用 pct_chg（同源、已含分红除权处理）作为日收益；缺失时用 close 差分兜底
    out["ret"] = out["pct_chg"] / 100.0
    bad = ~np.isfinite(out["ret"])
    if bad.any():
        g = out.groupby("code")["close"]
        out.loc[bad, "ret"] = (g.shift(-1) / out.loc[bad, "close"] - 1)
    return out[["date", "code", "name", "close", "ret"]]


def ensure(dates: np.ndarray, token: str | None = None, offline: bool = False) -> pd.DataFrame:
    """有缓存且覆盖够就用缓存，否则重拉。"""
    if CACHE.exists():
        df = pd.read_parquet(CACHE)
        have = set(df["date"].unique())
        miss = len(set(dates.tolist()) - have)
        n_code = df["code"].nunique()
        if offline or miss <= 3:
            if miss:
                print(f"  基准缓存缺 {miss} 日（可接受，回测按交集对齐）")
            print(f"  使用缓存 {CACHE}（{n_code} 个指数 / {len(df)} 行）")
            return df
        print(f"  缓存缺 {miss} 日，重新拉取…")
    if offline:
        raise SystemExit(f"--offline 但缓存不存在：{CACHE}")
    df = fetch(dates, token)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    print(f"  已缓存 {CACHE}（{df['code'].nunique()} 个指数 / {len(df)} 行）")
    return df


def aligned(dates: np.ndarray, code: str = MAIN, token: str | None = None,
            offline: bool = False) -> np.ndarray:
    """返回与 `dates` 等长的日收益数组（指数缺失日 → 0）。供 backtest 直接用。"""
    df = ensure(dates, token, offline)
    s = df[df["code"] == code].set_index("date")["ret"].reindex(dates)
    v = s.to_numpy(dtype=np.float64)
    if np.isnan(v).all():
        raise SystemExit(f"基准 {code} 与面板日期无交集")
    return np.nan_to_num(v, nan=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    dates = np.load(BASE / "_panel" / "dates.npy").astype(np.int64)
    print(f"面板 {int(dates[0])} → {int(dates[-1])}（{len(dates)} 个交易日）")
    df = ensure(dates, a.token, a.offline)
    print(f"\n{'代码':<12}{'名称':<10}{'交易日':>7}{'覆盖':>8}{'年化':>9}{'波动':>8}{'Sharpe':>8}{'MDD':>8}")
    for code, name in BENCH:
        s = df[df["code"] == code].set_index("date")["ret"].reindex(dates)
        if s.notna().sum() == 0:
            continue
        r = np.nan_to_num(s.to_numpy(dtype=np.float64))
        n = len(r)
        eq = np.cumprod(1 + r)
        ann = eq[-1] ** (252 / n) - 1
        vol = r.std(ddof=1) * np.sqrt(252)
        mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
        star = " ★" if code == MAIN else ""
        print(f"{code:<12}{name:<10}{int(s.notna().sum()):>7}{s.notna().mean():>7.0%}"
              f"{ann:>9.1%}{vol:>8.1%}{ann/vol:>8.2f}{mdd:>8.1%}{star}")


if __name__ == "__main__":
    main()
