# 数据说明

本仓库**不分发行情原始数据**（体积与数据源条款限制）。若要在自己的数据上复现，
按下述结构准备面板即可。

## 目录约定

```
data/
├── panel/        # 面板 .npy（见下表）
├── feat/         # 可选：既有特征池（用于低相关性诊断，缺失时会自动跳过）
└── make_panel.py # 面板重建模板
```

`sf/ops.py` 默认从 `data/panel` 与 `data/feat` 读取，也可用环境变量覆盖：

```bash
export SF_BASE=/path/to/repo/data   # 基目录
export SF_PANEL=/path/to/panel      # 面板目录
export SF_FEAT=/path/to/feat        # 特征池目录
```

## 面板字段（`data/panel/*.npy`）

所有数组形状统一为 `(D, S)` = (交易日数 × 股票数)，`float32`，缺失用 `NaN`。

| 文件名 | 含义 | 是否必需 |
|---|---|---|
| `dates.npy` | 交易日，`int64`，升序，格式 `YYYYMMDD` | ✅ |
| `codes.npy` | 股票代码，`object/str`，与列索引对应 | ✅ |
| `open/high/low/close.npy` | **未复权** OHLC | ✅ |
| `pre_close.npy` | 昨收（**已做除权调整**） | ✅ |
| `factor.npy` | 复权因子（用于把 OHLC 还原成连续价） | ✅ |
| `vol.npy` / `amount.npy` | 成交量 / 成交额 | ✅ |
| `y1.npy` | **标签**：次日复权收益率 | ✅ |
| `turnover_rate_f.npy` | 换手率（自由流通） | ✅ |
| `volume_ratio.npy` | 量比 | 推荐 |
| `circ_mv.npy` | 流通市值 | 推荐 |
| `pe_ttm/pb/ps_ttm/dv_ttm.npy` | 估值 | 可选 |
| `to_up.npy` / `to_dn.npy` | 涨 / 跌停幅度（如 0.10 / 0.20 / 0.30） | 推荐 |
| `is_zt/is_dt/is_st.npy` | 涨停 / 跌停 / ST 标志（0/1） | ✅ |
| `fd_amount.npy` | 封单额 | 可选 |
| `limit_times.npy` | 连板数 | 可选 |
| `first_min.npy` / `last_min.npy` | 封板 / 末端时间（当日分钟数） | 可选 |
| `open_times.npy` | 开板次数 | 可选 |
| `up_cnt.npy` / `up_days.npy` | 连板统计 | 可选 |

> **关键口径**：`pre_close` 必须与 `close`/`factor` 同源可交叉校验，
> 即 `close_t / pre_close_t` 应等于真实的次日收益口径（本仓库自建复权时与行情源
> `pct_chg` 的一致率为 **99.90%**）。若这一条不成立，全部因子与标签都会错位。

## 重建

```bash
python data/make_panel.py --src /path/to/long_table.parquet --out data/panel
```

`--src` 需为**长表**（一行 = 一只股票 × 一个交易日）。所需列名见
`make_panel.py` 顶部的 `COLMAP`，可自行做别名映射。

## 为什么价格要复权

面板里的 OHLC 若为**未复权**价，跨股票不可比（`STD(close,20)/close` 这类因子会被
「每股价格水平」这一常数淹没）。本仓库的做法是：

1. 用 `factor` 把 OHLC 还原为**连续价** `A/H/L/O`；
2. 所有因子**只允许**由比值或收益率构成（见 `sf/ops.py::build_terminals`），
   例如 `(H-L)/O`、`turn / ts_mean(turn, 20)`、`log(turn)`；
3. 裸价格水平**永不直接进入**表达式 —— 这是复现东吴研报「去量纲」纪律的核心。
