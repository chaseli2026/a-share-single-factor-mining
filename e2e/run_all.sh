#!/usr/bin/env bash
# run_all.sh — 四臂 × 三种子，顺序执行（CPU，全部 130 列；约 25 分钟）
# 只跑 55 列量价：加 --feat X_base55.npy 并改 --tag base55
# 用法：bash run_all.sh  [PY=python3]
set -e
PY="${PY:-python}"

for arm in mse ic sharpe ret; do
  for s in 42 43 44; do
    echo "=== $arm seed=$s ==="
    "$PY" train.py --arm "$arm" --seed "$s" \
      --epochs 25 --steps 8 --block 80 --topk 80 --temp 2.0 --rebal 10
  done
done

"$PY" backtest.py
