"""FleetCare Test 18 v2

相对上一版:
 - 还原真实特征名, 打印最偏移特征在训练/验证中的取值范围与缺失率
 - 迭代剔除最偏移特征, 直到对抗 AUC <= --adv-target 或达到 --drop-shift-top 上限
 - 对抗 AUC >= 0.9 时自动跳过重要性加权(密度比不可靠)
 - 评估以无阈值指标为主: PR-AUC, ROC-AUC, Recall@Top5%, QWK, log loss

运行(放在 test_15_16gb.py 旁边):
  python test_18_v2.py                                   # 只做诊断 + 基线
  python test_18_v2.py --drop-shift-top 100 --no-iw      # 迭代剔除偏移特征
  python test_18_v2.py --drop-regex "time_step|length"   # 按名字剔除
  python test_18_v2.py --ordinal / --features histogram  # 之前测过, 无明显收益
"""
import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

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
    selected['time_step'] = train_op['time_step'].to_numpy()[selected['row_id'].to_numpy()]

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
    mult = np.linspace(np.log(0.1), np.log(30), 41)
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
    ge = np.minimum.accumulate(np.vstack(ge), axis=0)
    p_ord = np.column_stack([1 - ge[0], ge[0] - ge[1], ge[1] - ge[2], ge[2] - ge[3], ge[3]])
    p_ord = np.clip(p_ord, 1e-9, None)
    p_ord /= p_ord.sum(axis=1, keepdims=True)
    return 0.5 * p_multi + 0.5 * p_ord


def to_matrix(X):
    if hasattr(X, 'to_numpy'):
        return X.to_numpy(dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


# ---------------------------------------------------------------- 对抗验证
ADV_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.5, tree_method='hist', n_jobs=-1)


def adversarial(X_tr, X_va, g_tr, g_va):
    Xa = np.vstack([X_tr, X_va])
    ya = np.r_[np.zeros(len(X_tr)), np.ones(len(X_va))]
    ga = pd.factorize(np.r_[np.char.add('t_', g_tr.astype(str)),
                            np.char.add('v_', g_va.astype(str))])[0]
    oof = np.zeros(len(ya))
    for tr_i, te_i in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xa, ya, ga):
        am = xgb.XGBClassifier(**ADV_PARAMS).fit(Xa[tr_i], ya[tr_i])
        oof[te_i] = am.predict_proba(Xa[te_i])[:, 1]
    auc = roc_auc_score(ya, oof)
    am = xgb.XGBClassifier(**ADV_PARAMS).fit(Xa, ya)
    imp = am.feature_importances_
    del Xa, am
    gc.collect()
    return oof[:len(X_tr)], auc, imp


def importance_weights(p_adv_train, n_tr, n_va):
    p = np.clip(p_adv_train, 0.02, 0.98)
    iw = p / (1 - p) * (n_tr / n_va)
    return np.clip(iw, 0.05, 10.0)


def describe_shift(X_tr, X_va, imp, names, top=10):
    print('最偏移特征的取值范围 [min, 5%, 50%, 95%, max]:')

    def q(v):
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return 'all-NaN'
        return np.percentile(v, [0, 5, 50, 95, 100]).round(3).tolist()

    for i in np.argsort(imp)[::-1][:top]:
        a, b = X_tr[:, i], X_va[:, i]
        print(f'  {names[i]}')
        print(f'    train {q(a)}  nan={np.isnan(a).mean():.2f}')
        print(f'    val   {q(b)}  nan={np.isnan(b).mean():.2f}')


# ---------------------------------------------------------------- 评估
def recall_at_top(y, score, frac=0.05):
    k = max(1, int(round(frac * len(y))))
    idx = np.argsort(-score)[:k]
    pos = (y > 0)
    return pos[idx].sum() / max(pos.sum(), 1)


def score_metrics(y, p):
    yb = (y > 0).astype(int)
    s = 1 - p[:, 0]
    return {
        'pr_auc': average_precision_score(yb, s),
        'roc_auc': roc_auc_score(yb, s),
        'recall_top5pct': recall_at_top(y, s, 0.05),
        'logloss': log_loss(y, np.clip(p, 1e-9, 1), labels=LABELS),
    }


def decision_metrics(y, pred):
    return {
        'macro_f1': mf1(y, pred),
        'qwk': cohen_kappa_score(y, pred, weights='quadratic'),
        'within1_acc': float(np.mean(np.abs(y - pred) <= 1)),
    }


def bootstrap(y, p, pred, n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if (y[i] > 0).sum() == 0:
            continue
        r = score_metrics(y[i], p[i])
        r.update(decision_metrics(y[i], pred[i]))
        rows.append(r)
    df = pd.DataFrame(rows)
    return {c: (float(df[c].quantile(0.025)), float(df[c].quantile(0.975))) for c in df}


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
    ap.add_argument('--no-iw', action='store_true', help='不使用重要性加权')
    ap.add_argument('--drop-shift-top', type=int, default=0,
                    help='最多迭代剔除多少个最偏移特征(每轮10个)')
    ap.add_argument('--adv-target', type=float, default=0.75,
                    help='对抗AUC降到此值以下就停止剔除')
    ap.add_argument('--drop-regex', type=str, default='', help='按名字正则剔除特征')
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
    g_val = val_sel['vehicle_id'].to_numpy()
    print('Train rows:', len(y_train), 'class counts:', np.bincount(y_train, minlength=5))
    print('Val   rows:', len(y_val), 'class counts:', np.bincount(y_val, minlength=5), flush=True)

    train_feat = history_selected(train_op, train_sel, raw_cols, 'TRAIN')
    val_feat = history_selected(val_op, val_sel, raw_cols, 'VALIDATION')
    if args.features == 'histogram':
        g = histogram_groups(raw_cols)
        train_feat = add_histogram_summaries(train_feat, train_op, train_sel, g)
        val_feat = add_histogram_summaries(val_feat, val_op, val_sel, g)
    train_feat = add_specs(train_feat, train_sel, train_spec)
    val_feat = add_specs(val_feat, val_sel, val_spec)

    raw_names = list(train_feat.columns) if hasattr(train_feat, 'columns') else []
    X_train, X_val = transformed(train_feat, val_feat)
    if hasattr(X_train, 'columns'):
        feat_names = list(X_train.columns)
    elif len(raw_names) == X_train.shape[1]:
        feat_names = raw_names
    else:
        print(f'警告: 无法还原特征名 (原始列 {len(raw_names)} vs 矩阵列 {X_train.shape[1]}), '
              '改用 f{i}。可检查 transformed() 是否做了 one-hot 或列重排。')
        feat_names = [f'f{i}' for i in range(X_train.shape[1])]
    feat_names = [str(n) for n in feat_names]
    X_train, X_val = to_matrix(X_train), to_matrix(X_val)
    print('Matrices:', X_train.shape, X_val.shape, flush=True)
    del train_op, val_op, train_feat, val_feat
    gc.collect()

    dropped_names = []

    # ---- 按名字剔除
    if args.drop_regex:
        pat = re.compile(args.drop_regex)
        keep = np.array([pat.search(n) is None for n in feat_names])
        dropped_names += [n for n, k in zip(feat_names, keep) if not k]
        print(f'drop-regex 剔除 {int((~keep).sum())} 个特征', flush=True)
        X_train, X_val = X_train[:, keep], X_val[:, keep]
        feat_names = [n for n, k in zip(feat_names, keep) if k]

    # ---- 对抗验证(全特征)
    p_adv, adv_auc, imp = adversarial(X_train, X_val, groups, g_val)
    adv_auc_initial = adv_auc
    print('全特征对抗 AUC (0.5=一致): %.3f' % adv_auc)
    print('最偏移的20个特征:')
    for i in np.argsort(imp)[::-1][:20]:
        print(f'  {feat_names[i]:60s} {imp[i]:.4f}')
    describe_shift(X_train, X_val, imp, feat_names, top=10)
    sys_flush = print('', flush=True)

    # ---- 迭代剔除最偏移特征
    n_dropped = 0
    while args.drop_shift_top > 0 and adv_auc > args.adv_target and n_dropped < args.drop_shift_top:
        drop = np.argsort(imp)[::-1][:10]
        names = [feat_names[i] for i in drop]
        print('剔除:', names)
        dropped_names += names
        keep2 = np.ones(len(feat_names), dtype=bool)
        keep2[drop] = False
        X_train, X_val = X_train[:, keep2], X_val[:, keep2]
        feat_names = [n for n, k in zip(feat_names, keep2) if k]
        n_dropped += len(drop)
        p_adv, adv_auc, imp = adversarial(X_train, X_val, groups, g_val)
        print(f'已剔除 {n_dropped} 个, 剩余 {X_train.shape[1]} 个特征, 对抗 AUC: {adv_auc:.3f}', flush=True)
    if n_dropped:
        print('剔除后最偏移特征:')
        describe_shift(X_train, X_val, imp, feat_names, top=5)

    # ---- 重要性加权(仅在对抗 AUC 不退化时使用)
    iw_used = False
    if args.no_iw:
        print('跳过重要性加权 (--no-iw)')
    elif adv_auc >= 0.9:
        print('对抗 AUC >= 0.9, 密度比不可靠, 自动跳过重要性加权')
    else:
        iw = importance_weights(p_adv, len(X_train), len(X_val))
        sw = sw * iw
        sw = sw / sw.mean()
        iw_used = True
        print('重要性权重 min/median/max: %.2f / %.2f / %.2f' % (iw.min(), np.median(iw), iw.max()))

    # ---- 先验
    prior_tr = np.array([sw[y_train == c].sum() for c in LABELS])
    prior_tr /= prior_tr.sum()
    prior_val = np.bincount(y_val, minlength=5) / len(y_val)
    ratio = prior_val / prior_tr
    bayes_lw = np.log(ratio[1:] / ratio[0])
    shift_w = sw * ratio[y_train]
    print('Prior train:', prior_tr.round(4), ' val:', prior_val.round(4))
    print('Bayes weights C0..C4:', np.exp(np.r_[0, bayes_lw]).round(4), flush=True)

    # ---- OOF
    print(f'OOF {args.folds}-fold (grouped by vehicle)...', flush=True)
    oof = np.zeros((len(y_train), 5))
    cv = StratifiedGroupKFold(args.folds, shuffle=True, random_state=SEED)
    for i, (tr, va) in enumerate(cv.split(X_train, y_train, groups), 1):
        oof[va] = fit_predict(X_train[tr], y_train[tr], sw[tr], X_train[va], args.ordinal)
        print(f'  fold {i}/{args.folds} done', flush=True)

    oof_auc = roc_auc_score(y_train > 0, 1 - oof[:, 0], sample_weight=shift_w)
    oof_ap = average_precision_score(y_train > 0, 1 - oof[:, 0], sample_weight=shift_w)
    print('\nOOF (对齐验证先验) ROC-AUC: %.4f  PR-AUC: %.4f' % (oof_auc, oof_ap))

    zero = np.zeros(4)
    oof_scores = {
        'argmax': mf1(y_train, predict_weighted(oof, zero), shift_w),
        'bayes': mf1(y_train, predict_weighted(oof, bayes_lw), shift_w),
    }
    tuned_lw, oof_scores['tuned'] = tune_weights(oof, y_train, shift_w, bayes_lw)
    print('OOF weighted macro F1:', {k: round(v, 4) for k, v in oof_scores.items()})
    print('Tuned weights C0..C4:', np.exp(np.r_[0, tuned_lw]).round(4), flush=True)

    # ---- 最终模型, 验证集只评估
    print('\nFinal fit on all training rows...', flush=True)
    p_val = fit_predict(X_train, y_train, sw, X_val, args.ordinal)

    sm = score_metrics(y_val, p_val)
    print('\n=== VALIDATION: 与阈值无关的指标 (主指标) ===')
    print('ROC-AUC %.4f | PR-AUC %.4f (随机基线 %.4f) | Recall@Top5%% %.4f (随机约 0.05) | logloss %.4f' % (
        sm['roc_auc'], sm['pr_auc'], (y_val > 0).mean(), sm['recall_top5pct'], sm['logloss']))

    cands = {'argmax': zero, 'bayes': bayes_lw, 'tuned': tuned_lw}
    results, preds = {}, {}
    print('\n=== VALIDATION: 决策相关指标 (参考, 带 95%CI) ===')
    for name, lw in cands.items():
        pred = predict_weighted(p_val, lw)
        preds[name] = pred
        dm = decision_metrics(y_val, pred)
        ci = bootstrap(y_val, p_val, pred)
        results[name] = {'metrics': {**sm, **dm}, 'ci': ci}
        print(f"{name:7s} macroF1={dm['macro_f1']:.4f} [{ci['macro_f1'][0]:.3f},{ci['macro_f1'][1]:.3f}]  "
              f"QWK={dm['qwk']:.4f} [{ci['qwk'][0]:.3f},{ci['qwk'][1]:.3f}]  within1={dm['within1_acc']:.4f}")
    ci0 = results['argmax']['ci']
    print('PR-AUC 95%%CI [%.3f, %.3f]  ROC-AUC 95%%CI [%.3f, %.3f]  Recall@Top5%% 95%%CI [%.3f, %.3f]' % (
        *ci0['pr_auc'], *ci0['roc_auc'], *ci0['recall_top5pct']))

    chosen = max(oof_scores, key=oof_scores.get)
    pred = preds[chosen]
    print(f'\n按 OOF 分数选择的决策: {chosen}')
    print(classification_report(y_val, pred, labels=LABELS, zero_division=0))
    print(confusion_matrix(y_val, pred, labels=LABELS))

    tag = f"test_18v2_{args.features}{'_ord' if args.ordinal else ''}"
    out = args.output_dir
    pd.DataFrame({
        'vehicle_id': g_val,
        'actual_class': y_val,
        'pred_class': pred,
        **{f'p_c{i}': p_val[:, i] for i in LABELS},
    }).to_csv(out / f'{tag}_val_predictions.csv', index=False)
    (out / f'{tag}_summary.json').write_text(json.dumps({
        'args': {k: str(v) for k, v in vars(args).items()},
        'adversarial_auc_initial': adv_auc_initial,
        'adversarial_auc_final': adv_auc,
        'n_features_final': int(X_train.shape[1]),
        'dropped_features': dropped_names,
        'importance_weighting_used': iw_used,
        'oof': {'roc_auc': oof_auc, 'pr_auc': oof_ap, 'macro_f1': oof_scores},
        'val': results,
        'chosen': chosen,
        'tuned_weights': np.exp(np.r_[0, tuned_lw]).tolist(),
        'warning': 'Validation set used in earlier development; its prior is used for OOF reweighting.',
    }, indent=2, default=float), encoding='utf-8')
    print('\nSaved to:', out)


if __name__ == '__main__':
    main()