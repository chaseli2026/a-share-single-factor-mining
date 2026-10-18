# -*- coding: utf-8 -*-
"""e2e/train.py — 端到端组合优化训练器

用法：
  python train.py --arm sharpe --seed 42 --epochs 25
  bash run_all.sh                       # 一次跑三臂 × 三种子

时间切分（按样本比例，不写死日期）：训练 60% / 验证 15% / 测试 25%。
模型选择只看**验证集 Sharpe**（早停 + 回滚），测试集只评一次。

输出：`<SF_RESULTS>/e2e_<arm>_s<seed>.json`
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from data import Panel
from model import (Scorer, loss_mse, loss_rank_ic, loss_return, loss_sharpe,
                   select_weights)

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("E2E_OUT", HERE.parent / "results"))


@dataclass
class Cfg:
    arm: str = "sharpe"     # mse | ic | sharpe | ret
    seed: int = 42
    epochs: int = 25
    steps: int = 8          # 每个 epoch 采多少个时间块
    block: int = 60         # 每块连续交易日数（Sharpe 需要一条路径）
    rebal: int = 5          # 每 5 个交易日调仓
    topk: int = 50          # 持仓只数
    temp: float = 0.8       # 组合层 softmax 温度
    zmax: float = 3.0       # 组内 z 分数截断（防止单票权重过大）
    cost: float = 0.0015    # 单边交易成本（15bp / 单位换手）
    lr: float = 1e-3
    wd: float = 1e-5
    dsr: str = "batch"      # sharpe 臂的损失形式：batch | dsr
    feat: str = "X_base55.npy"
    tag: str = ""


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)


def train(panel: Panel, cfg: Cfg, log=print):
    set_seed(cfg.seed)
    model = Scorer(panel.F)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    lo, hi = panel.segs["train"]
    best = (-1e9, None, -1)
    t0 = time.time()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot = nstep = 0
        for _ in range(cfg.steps):
            b = int(np.random.randint(lo, hi - cfg.block - 1))
            rets, w_prev, w = [], None, None
            for j in range(cfg.block):
                x, m, y = panel.day(b + j)
                if m.sum() < 100:
                    continue
                xt, mask, yt = torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(y)
                s = model(xt)

                if cfg.arm in ("sharpe", "ret"):
                    if w is None or j % cfg.rebal == 0:
                        w = select_weights(s, mask, cfg.topk, cfg.temp, cfg.zmax)
                        c = 0.0 if w_prev is None else float((w - w_prev).abs().sum().detach())
                        w_prev = w.detach()
                        rets.append((w * yt).sum() - cfg.cost * c)
                    else:
                        rets.append((w * yt).sum())
                    continue

                loss = loss_mse(s, yt, mask) if cfg.arm == "mse" else loss_rank_ic(s, yt, mask)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot, nstep = tot + float(loss.detach()), nstep + 1

            if cfg.arm in ("sharpe", "ret") and len(rets) > 5:
                r = torch.stack(rets)
                loss = loss_sharpe(r, cfg.dsr) if cfg.arm == "sharpe" else loss_return(r)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot, nstep = tot + float(loss.detach()), nstep + 1

        if ep % 10 == 0 or ep == cfg.epochs:
            v = evaluate(panel, model, cfg, "val", want_ic=False)
            if v["sharpe"] > best[0]:
                best = (v["sharpe"], {k: t.detach().clone() for k, t in model.state_dict().items()}, ep)
            log(f"[{cfg.arm} s{cfg.seed}] ep{ep:02d} loss {tot/max(nstep,1):+.4f} "
                f"| val Sharpe {v['sharpe']:+.3f} | best {best[0]:+.3f}@{best[2]} | {time.time()-t0:.0f}s")

    if best[1] is not None:
        model.load_state_dict(best[1])
    res = {"cfg": asdict(cfg), "best_epoch": best[2], "val_sharpe_best": best[0],
           "panel": panel.describe()}
    for seg in ("val", "test"):
        res[seg] = evaluate(panel, model, cfg, seg)
    res["seconds"] = round(time.time() - t0, 1)
    return res, model


@torch.no_grad()
def evaluate(panel: Panel, model: Scorer, cfg: Cfg, seg: str, want_ic: bool = True) -> dict:
    """按同一调仓节奏跑一段区间，返回逐日净收益、毛收益、逐日 IC 与指标。"""
    model.eval()
    lo, hi = panel.segs[seg]
    rets, gross, ics, tos = [], [], [], []
    w = w_prev = None
    for t in range(lo, hi):
        x, m, y = panel.day(t)
        if m.sum() < 100:
            continue
        xt, mask = torch.from_numpy(x), torch.from_numpy(m)
        yt = torch.from_numpy(y)
        s = model(xt)
        cost = 0.0
        if w is None or (t - lo) % cfg.rebal == 0:
            w = select_weights(s, mask, cfg.topk, cfg.temp, cfg.zmax)
            if w_prev is not None:
                cost = cfg.cost * float((w - w_prev).abs().sum())
                tos.append(float((w - w_prev).abs().sum()))
            w_prev = w
        g = float((w * yt).sum())
        gross.append(g)
        rets.append(g - cost)

        sv = s[mask].numpy()
        yv = y[mask.numpy()]
        if want_ic and sv.size > 100:
            ra = np.argsort(np.argsort(sv)).astype(np.float64)
            rb = np.argsort(np.argsort(yv)).astype(np.float64)
            ra -= ra.mean(); rb -= rb.mean()
            d = np.linalg.norm(ra) * np.linalg.norm(rb)
            if d > 0:
                ics.append(float(ra @ rb / d))

    r = np.asarray(rets, dtype=np.float64)
    g = np.asarray(gross, dtype=np.float64)
    ic = np.asarray(ics, dtype=np.float64)
    eq = np.cumprod(1 + r)
    ann = float(eq[-1] ** (252 / len(r)) - 1) if len(r) > 1 else np.nan
    sd = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else np.nan
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else np.nan
    return {"daily": np.round(r, 6).tolist(),
            "gross_daily": np.round(g, 6).tolist(),
            "ic": np.round(ic, 5).tolist(),
            "n": int(len(r)),
            "ann_ret": ann, "ann_vol": sd,
            "sharpe": float(ann / sd) if sd and sd > 0 else np.nan,
            "mdd": mdd,
            "turnover_ann": float(np.mean(tos) * 252 / cfg.rebal) if tos else np.nan,
            "ic_mean": float(ic.mean()) if ic.size else np.nan,
            "icir": float(ic.mean() / ic.std(ddof=1))
            if ic.size > 2 and ic.std(ddof=1) > 0 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    for k, v in asdict(Cfg()).items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    cfg = Cfg(**{k: getattr(a, k) for k in asdict(Cfg())})

    torch.set_num_threads(int(os.environ.get("E2E_THREADS", "4")))
    OUT.mkdir(parents=True, exist_ok=True)
    panel = Panel(feat_files=tuple(cfg.feat.split(",")))
    print(panel.describe(), flush=True)

    res, _ = train(panel, cfg)
    name = f"e2e_{cfg.arm}{('_' + cfg.tag) if cfg.tag else ''}_s{cfg.seed}.json"
    (OUT / name).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    te = res["test"]
    print(f"→ {name} | test 年化 {te['ann_ret']:+.2%} Sharpe {te['sharpe']:+.2f} "
          f"MDD {te['mdd']:.1%} IC {te['ic_mean']:+.4f} | {res['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
