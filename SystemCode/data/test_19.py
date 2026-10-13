"""FleetCare Test 18: 多行采样 + 样本权重 + 按车分组OOF + 先验校正权重调优。

要点:
 1. 每个(车,类)取 k 行，权重 1/实际行数，保证每个(车,类)总权重一致。
 2. 决策权重在训练集 OOF 上调(按车分组CV)，并重加权到验证集先验；验证集只评估，不调参。
 3. 同时报告 argmax / 贝叶斯先验校正 / OOF调优 三种决策，由 OOF 分数选择，验证集给 bootstrap 区间。
 4. 可选: 有序累积二分类 P(y>=k)，与多分类概率平均。
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from test_15_16gb import (
    load, histogram_groups, history_selected,
    add_histogram_summaries, add_specs, transformed, model,
    label_from_tte, SEED,
)

LABELS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- 采样
def take_k(df, keys, k):
    df = df.sample(frac=1, random_state=SEED)
    picked = df[df.groupby(keys, sort=False).cumcount() < k].copy()
    picked['sw'] = 1.0 / picked.groupby(keys, sort=False)['row_id'].transform('size')
    return picked


def choose_rows_multi(train_op, tte, val_op, val_labels, k_repair, k_nonrepair):
    tr = train_op[['vehicle_id', 'time_step']].merge(
        tte, on='vehicle_id', how='left', validate='many_to_one')
    if tr[['in_study_repair', 'length_of_study_time_step']].isna().any().any():
        raise ValueError('Missing TTE metadata for some training vehicles')
    remaining = tr['length_of_study_time_step'] - tr['time_step']
    repaired = tr['in_study_repair'].eq(1) & remaining.ge(0)
    safe = tr['in_study_repair'].eq(0) & remaining.gt(48)
    eligible = repaired | safe
    label = np.full(len(tr), -1, dtype=np.int8)
    label[repaired.to_numpy()] = label_from_tte(remaining[repaired].to_numpy()).astype(np.int8)
    label[safe.to_numpy()] = 0
    subset = tr.loc[eligible, ['vehicle_id']].copy()
    subset['class_label'] = label[eligible.to_numpy()]
    subset['row_id'] = np.flatnonzero(eligible.to_numpy())
    is_rep = tr.loc[eligible, 'in_study_repair'].to_numpy() == 1
    rep_pick = take_k(subset.loc[is_rep], ['vehicle_id', 'class_label'], k_repair)
    non_pick = take_k(subset.loc[~is_rep], ['vehicle_id'], k_nonrepair)
    selected = pd.concat([rep_pick, non_pick], ignore_index=True)
    selected = selected.sample(frac=1, random_state=SEED).reset_index(drop=True)
    if selected['row_id'].duplicated().any():
        raise ValueError('Duplicate training row selection')

    val_rows = val_op[['vehicle_id', 'time_step']].copy()
    val_rows['row_id'] = np.arange(len(val_rows))
    val_last = val_rows.groupby('vehicle_id', sort=False).tail(1).copy()
    val_last = val_last.merge(val_labels, on='vehicle_id', how='left', validate='one_to_one')
    if val_last['class_label'].isna().any():
        raise ValueError('Missing validation labels')
    return selected, val_last


# ---------------------------------------------------------------- 决策
def predict_weighted(p, log_w):
    w = np.exp(np.r_[0.0, log_w])
    return np.argmax(p * w[None, :], axis=1)


def mf1(y, pred, sw=None):
    return f1_score(y, pred, labels=LABELS, average='macro',
                    zero_division=0, sample_weight=sw)


def tune_weights(p, y, sw, start, rounds=3):
    """坐标上升；每个类的 log 权重在 start 附近 ×[0.1, 30] 搜索。"""
    mult = np.log(np.exp(np.linspace(np.log(0.1), np.log(30), 41)))
    lw = start.copy()
    best = mf1(y, predict_weighted(p, lw), sw)
    for _ in range(rounds):
        for j in range(4):
            for g in mult:
                t = lw.copy()
                t[j] = start[j] + g
                s = mf1(y, predict_weighted(p, t), sw)
                if s > best:
                    best, lw = s, t
    return lw, best


# ---------------------------------------------------------------- 模型
def fit_predict(Xtr, ytr, swtr, Xte, ordinal):
    """返回 (n_test, 5) 概率。"""
    m = model(binary=False)
    m.fit(Xtr, ytr, sample_weight=swtr)
    if list(m.classes_) != LABELS:
        raise ValueError(f'Unexpected class order: {m.classes_}')
    p_multi = m.predict_proba(Xte)
    del m
    if not ordinal:
        return p_multi
    ge = []
    for k in range(1, 5):
        b = model(binary=True)
        b.fit(Xtr, (ytr >= k).astype(int), sample_weight=swtr)
        ge.append(b.predict_proba(Xte)[:, 1])
        del b
    ge = np.minimum.accumulate(np.vstack(ge), axis=0)      # 保证单调 P(y>=1)>=P(y>=2)...
    p_ord = np.column_stack([1 - ge[0], ge[0] - ge[1], ge[1] - ge[2], ge[2] - ge[3], ge[3]])
    p_ord = np.clip(p_ord, 1e-9, None)
    p_ord /= p_ord.sum(axis=1, keepdims=True)
    return 0.5 * p_multi + 0.5 * p_ord


def to_matrix(X):
    if hasattr(X, 'to_numpy'):
        return X.to_numpy(dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def boot_ci(y, pred, n=500, seed=0):
    rng = np.random.default_rng(seed)
    s = [mf1(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(s, [2.5, 97.5])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument('--data-dir', type=Path, default=root / 'raw')
    ap.add_argument('--output-dir', type=Path, default=root / 'processed')
    ap.add_argument('--features', choices=['original', 'histogram'], default='original')
    ap.add_argument('--k-repair', type=int, default=3)
    ap.add_argument('--k-nonrepair', type=int, default=2)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--ordinal', action='store_true')
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    groups = train_sel['vehicle_id'].to_numpy()
    print('Train rows:', len(y_train), 'class counts:', np.bincount(y_train, minlength=5))
    print('Val   rows:', len(y_val), 'class counts:', np.bincount(y_val, minlength=5), flush=True)

    # 对抗验证: time_step 分布是否一致
    t_tr = train_op['time_step'].to_numpy()[train_sel['row_id'].to_numpy()].astype(float).reshape(-1, 1)
    t_va = val_sel[['time_step']].to_numpy(dtype=float)
    ya = np.r_[np.zeros(len(t_tr)), np.ones(len(t_va))]
    pa = cross_val_predict(xgb.XGBClassifier(max_depth=3, n_estimators=100),
                           np.vstack([t_tr, t_va]), ya, cv=5, method='predict_proba')[:, 1]
    print('time_step adversarial AUC (0.5=一致): %.3f' % roc_auc_score(ya, pa))
    print('train time_step:', pd.Series(t_tr[:, 0]).describe().round(1).to_dict())
    print('val   time_step:', pd.Series(t_va[:, 0]).describe().round(1).to_dict(), flush=True)

    train_feat = history_selected(train_op, train_sel, raw_cols, 'TRAIN')
    val_feat = history_selected(val_op, val_sel, raw_cols, 'VALIDATION')
    if args.features == 'histogram':
        g = histogram_groups(raw_cols)
        train_feat = add_histogram_summaries(train_feat, train_op, train_sel, g)
        val_feat = add_histogram_summaries(val_feat, val_op, val_sel, g)
    train_feat = add_specs(train_feat, train_sel, train_spec)
    val_feat = add_specs(val_feat, val_sel, val_spec)
    X_train, X_val = transformed(train_feat, val_feat)
    X_train, X_val = to_matrix(X_train), to_matrix(X_val)
    print('Matrices:', X_train.shape, X_val.shape, flush=True)
    del train_op, val_op, train_feat, val_feat
    gc.collect()

    # 先验: 训练(按sw加权) -> 验证
    prior_tr = np.array([sw[y_train == c].sum() for c in LABELS]); prior_tr /= prior_tr.sum()
    prior_val = np.bincount(y_val, minlength=5) / len(y_val)
    ratio = prior_val / prior_tr
    bayes_lw = np.log(ratio[1:] / ratio[0])
    shift_w = sw * ratio[y_train]            # OOF 加权: 每个(车,类)等权 + 对齐验证先验
    print('Prior train:', prior_tr.round(4), ' val:', prior_val.round(4))
    print('Bayes weights C0..C4:', np.exp(np.r_[0, bayes_lw]).round(4), flush=True)

    # OOF
    print(f'OOF {args.folds}-fold (grouped by vehicle)...', flush=True)
    oof = np.zeros((len(y_train), 5))
    cv = StratifiedGroupKFold(args.folds, shuffle=True, random_state=SEED)
    for i, (tr, va) in enumerate(cv.split(X_train, y_train, groups), 1):
        oof[va] = fit_predict(X_train[tr], y_train[tr], sw[tr], X_train[va], args.ordinal)
        print(f'  fold {i}/{args.folds} done', flush=True)

    zero = np.zeros(4)
    oof_arg = mf1(y_train, predict_weighted(oof, zero), shift_w)
    oof_bayes = mf1(y_train, predict_weighted(oof, bayes_lw), shift_w)
    tuned_lw, oof_tuned = tune_weights(oof, y_train, shift_w, bayes_lw)
    print('\n=== OOF weighted macro F1 (已对齐验证先验) ===')
    print(f'argmax={oof_arg:.4f}  bayes={oof_bayes:.4f}  tuned={oof_tuned:.4f}')
    print('Tuned weights C0..C4:', np.exp(np.r_[0, tuned_lw]).round(4), flush=True)

    # 最终模型 -> 验证集只评估
    print('\nFinal fit on all training rows...', flush=True)
    p_val = fit_predict(X_train, y_train, sw, X_val, args.ordinal)
    auc = roc_auc_score(y_val > 0, 1 - p_val[:, 0])
    print('Val AUC (C0 vs 非C0): %.4f' % auc)

    cands = {'argmax': zero, 'bayes': bayes_lw, 'tuned': tuned_lw}
    results = {}
    print('\n=== VALIDATION macro F1 (只评估一次) ===')
    for name, lw in cands.items():
        pred = predict_weighted(p_val, lw)
        lo, hi = boot_ci(y_val, pred)
        results[name] = (mf1(y_val, pred), lo, hi, pred)
        print(f'{name:7s} F1={results[name][0]:.4f}  95%CI [{lo:.3f}, {hi:.3f}]')

    chosen = max(['argmax', 'bayes', 'tuned'],
                 key=lambda n: {'argmax': oof_arg, 'bayes': oof_bayes, 'tuned': oof_tuned}[n])
    pred = results[chosen][3]
    print(f'\n按 OOF 分数选择的决策: {chosen}')
    print(classification_report(y_val, pred, labels=LABELS, zero_division=0))
    print(confusion_matrix(y_val, pred, labels=LABELS))

    tag = f"test_18_{args.features}{'_ord' if args.ordinal else ''}"
    out = args.output_dir
    pd.DataFrame({
        'vehicle_id': val_sel['vehicle_id'].to_numpy(),
        'actual_class': y_val,
        'pred_class': pred,
        **{f'p_c{i}': p_val[:, i] for i in LABELS},
    }).to_csv(out / f'{tag}_val_predictions.csv', index=False)
    (out / f'{tag}_summary.json').write_text(json.dumps({
        'args': {k: str(v) for k, v in vars(args).items()},
        'oof': {'argmax': oof_arg, 'bayes': oof_bayes, 'tuned': oof_tuned},
        'val': {n: {'f1': r[0], 'ci': [r[1], r[2]]} for n, r in results.items()},
        'val_auc_non_c0': auc,
        'chosen': chosen,
        'tuned_weights': np.exp(np.r_[0, tuned_lw]).tolist(),
        'warning': 'Validation set used in earlier development; its prior is used for OOF reweighting.',
    }, indent=2), encoding='utf-8')
    print('\nSaved to:', out)


if __name__ == '__main__':
    main()