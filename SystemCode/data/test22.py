"""FleetCare Test 21: 用验证集标签做领域自适应(交叉拟合评估)

做法:
  基线: 只用训练行训练, 在全部验证行上预测。
  自适应: 5 折切分验证车辆; 每折用 [全部训练行 + 其余 4/5 验证行(加权)] 训练,
          只预测留出的 1/5。拼起来得到每辆验证车辆的"交叉拟合"预测, 与基线在同一批行上比较。
  决策: 不再调 4 个类权重; 用一个参数的告警规则:
          按 P(非C0) 排序取前 q% 告警, 告警车辆在 C1..C4 里取概率最大的类。

依赖: 同目录下的 test_20_normalized.py (沿用其特征构造)。

运行:
  python test_21_adapt.py
  python test_21_adapt.py --val-weights 5,20
  python test_21_adapt.py --drop-blocks rate
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from test21 import (
    load, choose_rows_multi, detect_groups, build_features, spec_matrix,
    fit_predict, score_metrics, mf1, LABELS, SEED,
)


def prepare(args):
    print('Loading files...', flush=True)
    train_op = load(args.data_dir, 'train_operational_readouts.csv')
    val_op = load(args.data_dir, 'validation_operational_readouts.csv')
    train_spec = load(args.data_dir, 'train_specifications.csv')
    val_spec = load(args.data_dir, 'validation_specifications.csv')
    tte = load(args.data_dir, 'train_tte.csv')
    labels = load(args.data_dir, 'validation_labels.csv')
    train_op = train_op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    val_op = val_op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    raw_cols = [c for c in train_op if c not in ('vehicle_id', 'time_step')]

    train_sel, val_sel = choose_rows_multi(train_op, tte, val_op, labels,
                                           args.k_repair, args.k_nonrepair)
    y_train = train_sel['class_label'].to_numpy(dtype=int)
    y_val = val_sel['class_label'].to_numpy(dtype=int)
    sw = train_sel['sw'].to_numpy(dtype=float)
    print('Train rows:', len(y_train), np.bincount(y_train, minlength=5))
    print('Val   rows:', len(y_val), np.bincount(y_val, minlength=5), flush=True)

    lags = [int(x) for x in args.lags.split(',') if x]
    hist, singles = detect_groups(raw_cols)
    Xtr_f, names, blocks, _ = build_features(train_op, train_sel, raw_cols, hist, singles, lags)
    Xva_f, _, _, _ = build_features(val_op, val_sel, raw_cols, hist, singles, lags)
    del train_op, val_op
    gc.collect()

    s_tr, s_va, s_names = spec_matrix(train_spec, val_spec,
                                      train_sel['vehicle_id'].to_numpy(), val_sel['vehicle_id'].to_numpy())
    X_train = np.hstack([Xtr_f, s_tr]).astype(np.float32)
    X_val = np.hstack([Xva_f, s_va]).astype(np.float32)
    blocks = np.r_[blocks, np.array(['spec'] * len(s_names))]
    del Xtr_f, Xva_f, s_tr, s_va
    gc.collect()

    drop_blocks = [x for x in args.drop_blocks.split(',') if x]
    if drop_blocks:
        keep = ~np.isin(blocks, drop_blocks)
        X_train, X_val = X_train[:, keep], X_val[:, keep]
        print(f'剔除块 {drop_blocks}, 剩余 {X_train.shape[1]} 列')
    print('Matrices:', X_train.shape, X_val.shape, flush=True)
    return X_train, X_val, y_train, y_val, sw, val_sel['vehicle_id'].to_numpy()


def cv_with_val(X_train, y_train, sw, X_val, y_val, w_val, folds, seed):
    p = np.zeros((len(y_val), 5))
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for i, (a, b) in enumerate(skf.split(X_val, y_val), 1):
        Xtr = np.vstack([X_train, X_val[a]])
        ytr = np.r_[y_train, y_val[a]]
        swtr = np.r_[sw, np.full(len(a), float(w_val))]
        p[b] = fit_predict(Xtr, ytr, swtr, X_val[b])
        print(f'    fold {i}/{folds} done', flush=True)
        del Xtr
        gc.collect()
    return p


def alert_pred(p, rate):
    s = 1 - p[:, 0]
    k = max(1, int(round(rate * len(p))))
    idx = np.argsort(-s)[:k]
    pred = np.zeros(len(p), dtype=int)
    pred[idx] = 1 + np.argmax(p[idx, 1:], axis=1)
    return pred, idx


def report(name, y, p):
    sm = score_metrics(y, p)
    print(f'\n--- {name} ---')
    print('ROC-AUC %.4f | PR-AUC %.4f (基线 %.4f) | Recall@Top5%% %.4f | logloss %.4f' % (
        sm['roc_auc'], sm['pr_auc'], (y > 0).mean(), sm['recall_top5pct'], sm['logloss']))
    print('告警率  告警数  非C0召回  告警精度  macro-F1')
    rows = {}
    for rate in (0.02, 0.05, 0.10, 0.20):
        pred, idx = alert_pred(p, rate)
        rec = (y[idx] > 0).sum() / max((y > 0).sum(), 1)
        prec = (y[idx] > 0).mean()
        f1 = mf1(y, pred)
        rows[rate] = {'recall': float(rec), 'precision': float(prec), 'macro_f1': float(f1)}
        print(f'{rate:5.0%} {len(idx):7d}   {rec:7.3f}   {prec:7.3f}   {f1:7.4f}')
    return sm, rows


def paired_bootstrap(y, p_a, p_b, n=300, seed=0):
    """p_b - p_a 的配对 bootstrap 差值(同一批重采样行)。"""
    rng = np.random.default_rng(seed)
    d = {'pr_auc': [], 'roc_auc': [], 'recall_top5pct': []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if (y[i] > 0).sum() == 0:
            continue
        a, b = score_metrics(y[i], p_a[i]), score_metrics(y[i], p_b[i])
        for k in d:
            d[k].append(b[k] - a[k])
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument('--data-dir', type=Path, default=root / 'raw')
    ap.add_argument('--output-dir', type=Path, default=root / 'processed')
    ap.add_argument('--k-repair', type=int, default=3)
    ap.add_argument('--k-nonrepair', type=int, default=2)
    ap.add_argument('--lags', type=str, default='5,20')
    ap.add_argument('--drop-blocks', type=str, default='')
    ap.add_argument('--val-weights', type=str, default='10',
                    help='验证行样本权重, 逗号分隔, 每个值要跑一轮 5 折')
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, y_train, y_val, sw, g_va = prepare(args)

    print('\n基线: 只用训练行训练...', flush=True)
    p_base = fit_predict(X_train, y_train, sw, X_val)
    sm_b, rows_b = report('基线 (只用训练集)', y_val, p_base)

    summary = {'args': {k: str(v) for k, v in vars(args).items()},
               'baseline': {'metrics': sm_b, 'alert': {str(k): v for k, v in rows_b.items()}},
               'adapted': {}}
    preds = {'baseline': p_base}

    for w in [float(x) for x in args.val_weights.split(',') if x]:
        print(f'\n自适应: 验证行权重 {w}, {args.folds} 折交叉拟合...', flush=True)
        p = cv_with_val(X_train, y_train, sw, X_val, y_val, w, args.folds, SEED)
        sm, rows = report(f'自适应 (验证行权重={w:g}, 交叉拟合)', y_val, p)
        diff = paired_bootstrap(y_val, p_base, p)
        print('相对基线的差值 [均值, 95%CI] (区间不含 0 才算真实提升):')
        for k, (m, lo, hi) in diff.items():
            print(f'  {k:15s} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]')
        summary['adapted'][f'w={w:g}'] = {'metrics': sm, 'alert': {str(k): v for k, v in rows.items()},
                                          'diff_vs_baseline': diff}
        preds[f'adapt_w{w:g}'] = p

    out = args.output_dir
    pd.DataFrame({'vehicle_id': g_va, 'actual_class': y_val,
                  **{f'{n}_p_c{i}': p[:, i] for n, p in preds.items() for i in LABELS}}
                 ).to_csv(out / 'test_21_val_crossfit_predictions.csv', index=False)
    (out / 'test_21_summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    print('\nSaved to:', out)


if __name__ == '__main__':
    main()