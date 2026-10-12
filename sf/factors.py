# -*- coding: utf-8 -*-
"""
sf/factors.py — 因子定义集（仿东吴系列1 的两条路线）

- BASE158 : 新数据版「Alpha158 等价」五类基础因子（K线/波动/价/量/量价相关），
            全部**去量纲**（复权价 A/H/L/O + 换手率 turn）。窗口用 %d，主口径取 20。
- OPT_*   : 路线1 —— 对 BASE158 的「改进候选」（prompt 迭代产出，见 opt_candidates.json）
- NEW_*   : 路线2 —— 案例驱动生成的「新因子」，重点用新数据独有原料
            （换手/量比/封单/连板/封板时间/涨跌停制度/估值）
"""

# ══════════════════════════════════════════════ 路线 A：Alpha158 等价（%d 为窗口）
BASE158 = [
    # ---- K 线（13，日内无量纲形态）----
    ("K线", "HIGH0", "H/A"),
    ("K线", "KLEN", "(H-L)/O"),
    ("K线", "KLOW1", "(min2(O,A)-L)/O"),
    ("K线", "KLOW2", "(min2(O,A)-L)/(H-L)"),
    ("K线", "KMID1", "(A-O)/O"),
    ("K线", "KMID2", "(A-O)/(H-L)"),
    ("K线", "KSFT1", "(2*A-H-L)/O"),
    ("K线", "KSFT2", "(2*A-H-L)/(H-L)"),
    ("K线", "KUP1", "(H-max2(O,A))/O"),
    ("K线", "KUP2", "(H-max2(O,A))/(H-L)"),
    ("K线", "LOW0", "L/A"),
    ("K线", "OPEN0", "O/A"),
    ("K线", "VWAP0", "vwap0"),
    # ---- 波动（1）----
    ("波动", "STD", "ts_std(A,%d)/A"),
    # ---- 价（16）----
    ("价", "BETA", "ts_slope(A,%d)/A"),
    ("价", "CNTD", "ts_mean(gt(A-ref(A,1),0)-lt(A-ref(A,1),0),%d)"),
    ("价", "CNTN", "ts_mean(lt(A-ref(A,1),0),%d)"),
    ("价", "CNTP", "ts_mean(gt(A-ref(A,1),0),%d)"),
    ("价", "IMAX", "ts_argmax(H,%d)/%d"),
    ("价", "IMIN", "ts_argmin(L,%d)/%d"),
    ("价", "IMXD", "(ts_argmax(H,%d)-ts_argmin(L,%d))/%d"),
    ("价", "MA", "ts_mean(A,%d)/A"),
    ("价", "MAX", "ts_max(H,%d)/A"),
    ("价", "MIN", "ts_min(L,%d)/A"),
    ("价", "QTLD", "ts_quant(A,%d,0.2)/A"),
    ("价", "QTLU", "ts_quant(A,%d,0.8)/A"),
    ("价", "RANK", "ts_rank(A,%d)"),
    ("价", "RESI", "ts_resi(A,%d)/A"),
    ("价", "RSQR", "ts_rsqr(A,%d)"),
    ("价", "ROC", "ref(A,%d)/A"),
    # ---- 量（8，用换手率，本身无量纲）----
    ("量", "VMA", "ts_mean(turn,%d)/turn"),
    ("量", "VSTD", "ts_std(turn,%d)/turn"),
    ("量", "VMAX", "ts_max(turn,%d)/turn"),
    ("量", "VMIN", "ts_min(turn,%d)/turn"),
    ("量", "VQTLD", "ts_quant(turn,%d,0.2)/turn"),
    ("量", "VQTLU", "ts_quant(turn,%d,0.8)/turn"),
    ("量", "VRANK", "ts_rank(turn,%d)"),
    ("量", "VROC", "ref(turn,%d)/turn"),
    # ---- 量价相关（3）----
    ("量价", "CORR", "ts_corr(A,log(turn),%d)"),
    ("量价", "CORD", "ts_corr(A/ref(A,1),log(turn/ref(turn,1)),%d)"),
    ("量价", "CVOLR", "ts_corr(A,ts_std(r1,%d),%d)"),
]

# ══════════════════════════════════════════════ 路线 B：新数据独有原料（固定窗口）
#    这些字段来自扩展行情明细，是本次仿效研报的「增量原料」
NEWDATA = [
    # 换手 / 量比
    ("换手", "turn_raw", "turn"),
    ("换手", "turn_rel20", "turn_rel"),
    ("换手", "turn_std20", "ts_std(turn,20)"),
    ("换手", "turn_z20", "ts_z(turn,20)"),
    ("换手", "turn_rank20", "ts_rank(turn,20)"),
    ("换手", "volratio", "vr"),
    ("换手", "turn_slope20", "ts_slope(log(turn),20)"),
    # 估值 / 规模
    ("估值", "logcmv", "logcmv"),
    ("估值", "ep", "ep"),
    ("估值", "bp", "bp"),
    ("估值", "dp", "dp"),
    ("估值", "turn_mv", "turn*logcmv"),
    # 涨跌停制度（板块 ±10/20/30%）
    ("制度", "board", "board"),
    ("制度", "toup", "toup"),
    ("制度", "r1_overshoot", "r1/(toup+1e-9)"),
    # 事件：封单 / 连板 / 封板时间
    ("事件", "seal", "seal"),
    ("事件", "seal_mv", "seal_mv"),
    ("事件", "zt5", "zt5"),
    ("事件", "zt20", "zt20"),
    ("事件", "dszt", "dszt"),
    ("事件", "lbt_max5", "ts_max(lbt,5)"),
    ("事件", "upcnt", "upcnt"),
    ("事件", "otimes_neg", "neg(otimes)"),
    ("事件", "fmin_rel", "fmin/600.0"),
    # K 线盲区（研报 12 相对特征里我们缺的：上影）
    ("形态", "ush", "ush"),
    ("形态", "ush_mean20", "ts_mean(ush,20)"),
    ("形态", "lsh_mean20", "ts_mean(lsh,20)"),
    ("形态", "ush_lsh", "ts_mean(ush-lsh,20)"),
    ("形态", "rng_mean20", "ts_mean(hl,20)"),
]

ALL = BASE158 + NEWDATA

# 路线1 产出（由 opt_generate.py 写入；此处容错导入）
import json as _json  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_HERE = _Path(__file__).parent


def _load(fn: str):
    p = _HERE / fn
    if not p.exists():
        return []
    j = _json.load(open(p, encoding="utf-8"))
    return [(r.get("group", "opt"), r["name"], r["expr"]) for r in j]


OPT = _load("opt_candidates.json")
NEW = _load("new_candidates.json")
ALL = BASE158 + NEWDATA + OPT + NEW
