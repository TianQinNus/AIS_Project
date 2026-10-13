"""FleetCare Test 20: 去曝光量(累计量)特征 + 分块对抗诊断

不依赖 history_selected / transformed, 仅依赖 test_15_16gb 里的 load, label_from_tte, SEED。

特征块:
  prop  : 直方图组内分箱占比 (bin / 组总和)            -- 终身使用模式
  mix   : 最近 k 个读数的增量分箱占比                  -- 近期使用模式
  ratio : 近期速率 / 终身速率                          -- 近期是否加速
  rate  : 终身速率 (值 / (time_step+1))                -- 使用强度
  spec  : 车辆规格 one-hot
缺失值保持 NaN, 不做填补。

运行(放在 test_15_16gb.py 旁边):
  python test_20_normalized.py
  python test_20_normalized.py --drop-blocks rate          # 去掉使用强度类特征
  python test_20_normalized.py --drop-blocks rate,spec
  python test_20_normalized.py --match-time                # 训练行按验证集 time_step 分布重加权
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from test_15_16gb import load, label_from_tte, SEED

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


# ---------------------------------------------------------------- 特征
def detect_groups(cols, min_size=2, max_size=40):
    """按列名前缀(最后一个下划线之前)识别直方图组; 其余视为单列计数器。"""
    buckets = {}
    for i, c in enumerate(cols):
        c = str(c)
        key = c.rsplit('_', 1)[0] if '_' in c else c
        buckets.setdefault(key, []).append(i)
    hist = [(k, np.array(v)) for k, v in buckets.items() if min_size <= len(v) <= max_size]
    used = {i for _, v in hist for i in v}
    singles = np.array([i for i in range(len(cols)) if i not in used], dtype=int)
    return hist, singles


def vehicle_start(veh):
    change = np.r_[True, veh[1:] != veh[:-1]]
    starts = np.flatnonzero(change)
    return starts[np.cumsum(change) - 1]


def sdiv(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = a / b
    return np.where(np.isfinite(out) & (b > 0), out, np.nan)


def build_features(op, sel, raw_cols, hist, singles, lags):
    X = op[raw_cols].to_numpy(dtype=np.float32)
    veh = op['vehicle_id'].to_numpy()
    ts = op['time_step'].to_numpy(dtype=np.float64)
    vst = vehicle_start(veh)
    pos = sel['row_id'].to_numpy()
    cur = X[pos].astype(np.float64)
    t_cur = ts[pos]
    age = t_cur + 1.0
    nan_rate = np.isnan(cur).mean(axis=0)

    cols, names, blocks = [], [], []

    def add(arr, nms, block):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        cols.append(arr)
        names.extend(nms)
        blocks.extend([block] * len(nms))

    life = sdiv(cur[:, singles], age[:, None])
    add(life, [f'rate:{raw_cols[i]}' for i in singles], 'rate')
    tot_cur = {}
    for key, idx in hist:
        tot = cur[:, idx].sum(axis=1)
        tot_cur[key] = tot
        add(sdiv(cur[:, idx], tot[:, None]), [f'prop:{raw_cols[i]}' for i in idx], 'prop')
        add(sdiv(tot, age), [f'rate:{key}_total'], 'rate')

    for lag in lags:
        pl = np.maximum(pos - lag, vst[pos])
        dt = t_cur - ts[pl]
        d = cur - X[pl].astype(np.float64)
        d = np.where(d < 0, np.nan, d)
        rec = sdiv(d[:, singles], dt[:, None])
        add(sdiv(rec, life), [f'ratio{lag}:{raw_cols[i]}' for i in singles], 'ratio')
        for key, idx in hist:
            dsum = d[:, idx].sum(axis=1)
            add(sdiv(d[:, idx], dsum[:, None]), [f'mix{lag}:{raw_cols[i]}' for i in idx], 'mix')
            add(sdiv(sdiv(dsum, dt), sdiv(tot_cur[key], age)), [f'ratio{lag}:{key}_total'], 'ratio')

    del X
    gc.collect()
    return np.hstack(cols), names, np.array(blocks), nan_rate


def spec_matrix(train_spec, val_spec, train_ids, val_ids):
    a = train_spec.copy(); a['_s'] = 0
    b = val_spec.copy(); b['_s'] = 1
    both = pd.concat([a, b], ignore_index=True)
    s = both['_s'].to_numpy()
    ids = both['vehicle_id'].to_numpy()
    feat = both.drop(columns=['vehicle_id', '_s'])
    cat_cols = [c for c in feat if not pd.api.types.is_numeric_dtype(feat[c])]
    feat = pd.get_dummies(feat, columns=cat_cols, dtype=np.float32).astype(np.float32)
    ftr = feat[s == 0].set_index(ids[s == 0]).reindex(train_ids)
    fva = feat[s == 1].set_index(ids[s == 1]).reindex(val_ids)
    return ftr.to_numpy(), fva.to_numpy(), list(feat.columns)


# ---------------------------------------------------------------- 对抗验证
ADV_PARAMS = dict(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.5, tree_method='hist', n_jobs=-1)


def adv_run(Xtr, Xva, gtr, gva, folds=3, importance=False):
    Xa = np.vstack([Xtr, Xva])
    ya = np.r_[np.zeros(len(Xtr)), np.ones(len(Xva))]
    ga = pd.factorize(np.r_[np.char.add('t_', gtr.astype(str)),
                            np.char.add('v_', gva.astype(str))])[0]
    oof = np.zeros(len(ya))
    for tr_i, te_i in StratifiedGroupKFold(folds, shuffle=True, random_state=SEED).split(Xa, ya, ga):
        m = xgb.XGBClassifier(**ADV_PARAMS).fit(Xa[tr_i], ya[tr_i])
        oof[te_i] = m.predict_proba(Xa[te_i])[:, 1]
    auc = roc_auc_score(ya, oof)
    imp = None
    if importance:
        imp = xgb.XGBClassifier(**ADV_PARAMS).fit(Xa, ya).feature_importances_
    del Xa
    gc.collect()
    return auc, imp


# ---------------------------------------------------------------- 决策与评估
def predict_weighted(p, log_w):
    w = np.exp(np.r_[0.0, log_w])
    return np.argmax(p * w[None, :], axis=1)


def mf1(y, pred, sw=None):
    return f1_score(y, pred, labels=LABELS, average='macro', zero_division=0, sample_weight=sw)


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


def fit_predict(Xtr, ytr, swtr, Xte):
    m = xgb_model()
    m.fit(Xtr, ytr, sample_weight=swtr)
    if list(m.classes_) != LABELS:
        raise ValueError(f'Unexpected class order: {m.classes_}')
    return m.predict_proba(Xte)


def xgb_model():
    from test_15_16gb import model
    return model(binary=False)


def recall_at_top(y, score, frac=0.05):
    k = max(1, int(round(frac * len(y))))
    idx = np.argsort(-score)[:k]
    pos = y > 0
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


def time_match_weights(t_tr, t_va, bins=10):
    edges = np.quantile(t_va, np.linspace(0, 1, bins + 1))[1:-1]
    btr = np.digitize(t_tr, edges)
    ptr = np.bincount(btr, minlength=bins) / len(t_tr)
    w = ((1.0 / bins) / np.maximum(ptr, 1e-6))[btr]
    return np.clip(w, 0.1, 10.0)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument('--data-dir', type=Path, default=root / 'raw')
    ap.add_argument('--output-dir', type=Path, default=root / 'processed')
    ap.add_argument('--k-repair', type=int, default=3)
    ap.add_argument('--k-nonrepair', type=int, default=2)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--lags', type=str, default='5,20', help='近期窗口(读数个数), 逗号分隔')
    ap.add_argument('--drop-blocks', type=str, default='', help='例如 rate 或 rate,spec')
    ap.add_argument('--match-time', action='store_true')
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lags = [int(x) for x in args.lags.split(',') if x]
    drop_blocks = [x for x in args.drop_blocks.split(',') if x]

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
    g_tr = train_sel['vehicle_id'].to_numpy()
    g_va = val_sel['vehicle_id'].to_numpy()
    print('Train rows:', len(y_train), np.bincount(y_train, minlength=5))
    print('Val   rows:', len(y_val), np.bincount(y_val, minlength=5), flush=True)

    hist, singles = detect_groups(raw_cols)
    print(f'识别到 {len(hist)} 个直方图组 (列数 {[len(v) for _, v in hist]}), {len(singles)} 个单列计数器')
    if not hist:
        print('警告: 没有识别到直方图组(列名可能没有 前缀_序号 结构), 所有列按计数器处理')
    print('示例列名:', [str(c) for c in raw_cols[:6]], flush=True)

    Xtr_f, names, blocks, nan_tr = build_features(train_op, train_sel, raw_cols, hist, singles, lags)
    Xva_f, _, _, nan_va = build_features(val_op, val_sel, raw_cols, hist, singles, lags)
    t_train = train_sel['time_step'].to_numpy(dtype=float)
    t_val = val_sel['time_step'].to_numpy(dtype=float)
    del train_op, val_op
    gc.collect()

    print('\n原始列缺失率差异最大的10列 (训练选中行 vs 验证选中行):')
    for i in np.argsort(-np.abs(nan_tr - nan_va))[:10]:
        print(f'  {str(raw_cols[i]):20s} train NaN={nan_tr[i]:.3f}  val NaN={nan_va[i]:.3f}')

    s_tr, s_va, s_names = spec_matrix(train_spec, val_spec,
                                      train_sel['vehicle_id'].to_numpy(), val_sel['vehicle_id'].to_numpy())
    X_train = np.hstack([Xtr_f, s_tr]).astype(np.float32)
    X_val = np.hstack([Xva_f, s_va]).astype(np.float32)
    names = names + [f'spec:{n}' for n in s_names]
    blocks = np.r_[blocks, np.array(['spec'] * len(s_names))]
    del Xtr_f, Xva_f, s_tr, s_va
    gc.collect()
    print('\nMatrices:', X_train.shape, X_val.shape)
    print('特征块列数:', {b: int((blocks == b).sum()) for b in dict.fromkeys(blocks.tolist())}, flush=True)

    # ---- 分块对抗验证 (仅 C0 行, 排除类别构成差异)
    m0t, m0v = y_train == 0, y_val == 0
    print('\n=== 对抗 AUC (仅 C0 行; 0.5=分布一致) ===', flush=True)
    block_auc = {}
    for b in dict.fromkeys(blocks.tolist()):
        ix = blocks == b
        auc, _ = adv_run(X_train[m0t][:, ix], X_val[m0v][:, ix], g_tr[m0t], g_va[m0v])
        block_auc[b] = auc
        print(f'  块 {b:6s} ({int(ix.sum()):4d} 列): {auc:.3f}', flush=True)

    keep = ~np.isin(blocks, drop_blocks)
    if drop_blocks:
        print(f'剔除块 {drop_blocks}, 剩余 {int(keep.sum())} 列')
    X_train, X_val = X_train[:, keep], X_val[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    blocks = blocks[keep]
    auc_all, imp = adv_run(X_train[m0t], X_val[m0v], g_tr[m0t], g_va[m0v], importance=True)
    print(f'保留特征的对抗 AUC (仅 C0): {auc_all:.3f}')
    print('最偏移的15个特征:')
    for i in np.argsort(imp)[::-1][:15]:
        print(f'  {names[i]:55s} [{blocks[i]}] {imp[i]:.4f}', flush=True)

    if args.match_time:
        w = time_match_weights(t_train, t_val)
        sw = sw * w
        sw = sw / sw.mean()
        print('match-time 权重 min/median/max: %.2f / %.2f / %.2f' % (w.min(), np.median(w), w.max()))

    # ---- 先验
    prior_tr = np.array([sw[y_train == c].sum() for c in LABELS]); prior_tr /= prior_tr.sum()
    prior_val = np.bincount(y_val, minlength=5) / len(y_val)
    ratio = prior_val / prior_tr
    bayes_lw = np.log(ratio[1:] / ratio[0])
    shift_w = sw * ratio[y_train]
    print('\nPrior train:', prior_tr.round(4), ' val:', prior_val.round(4))

    # ---- OOF
    print(f'OOF {args.folds}-fold (grouped by vehicle)...', flush=True)
    oof = np.zeros((len(y_train), 5))
    cv = StratifiedGroupKFold(args.folds, shuffle=True, random_state=SEED)
    for i, (tr, va) in enumerate(cv.split(X_train, y_train, g_tr), 1):
        oof[va] = fit_predict(X_train[tr], y_train[tr], sw[tr], X_train[va])
        print(f'  fold {i}/{args.folds} done', flush=True)
    oof_auc = roc_auc_score(y_train > 0, 1 - oof[:, 0], sample_weight=shift_w)
    oof_ap = average_precision_score(y_train > 0, 1 - oof[:, 0], sample_weight=shift_w)
    print('\nOOF (对齐验证先验) ROC-AUC: %.4f  PR-AUC: %.4f' % (oof_auc, oof_ap))

    zero = np.zeros(4)
    oof_scores = {'argmax': mf1(y_train, predict_weighted(oof, zero), shift_w),
                  'bayes': mf1(y_train, predict_weighted(oof, bayes_lw), shift_w)}
    tuned_lw, oof_scores['tuned'] = tune_weights(oof, y_train, shift_w, bayes_lw)
    print('OOF weighted macro F1:', {k: round(v, 4) for k, v in oof_scores.items()}, flush=True)

    # ---- 最终模型
    print('\nFinal fit on all training rows...', flush=True)
    p_val = fit_predict(X_train, y_train, sw, X_val)
    sm = score_metrics(y_val, p_val)
    print('\n=== VALIDATION: 与阈值无关的指标 (主指标) ===')
    print('ROC-AUC %.4f | PR-AUC %.4f (随机基线 %.4f) | Recall@Top5%% %.4f (随机约 0.05) | logloss %.4f' % (
        sm['roc_auc'], sm['pr_auc'], (y_val > 0).mean(), sm['recall_top5pct'], sm['logloss']))
    print('(对照: 旧特征 ROC-AUC 0.653, PR-AUC 0.047, Recall@Top5%% 0.096)')

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
              f"QWK={dm['qwk']:.4f}  within1={dm['within1_acc']:.4f}")
    ci0 = results['argmax']['ci']
    print('PR-AUC 95%%CI [%.3f, %.3f]  ROC-AUC 95%%CI [%.3f, %.3f]  Recall@Top5%% 95%%CI [%.3f, %.3f]' % (
        *ci0['pr_auc'], *ci0['roc_auc'], *ci0['recall_top5pct']))

    chosen = max(oof_scores, key=oof_scores.get)
    pred = preds[chosen]
    print(f'\n按 OOF 分数选择的决策: {chosen}')
    print(classification_report(y_val, pred, labels=LABELS, zero_division=0))
    print(confusion_matrix(y_val, pred, labels=LABELS))

    tag = 'test_20_norm' + ('_' + '-'.join(drop_blocks) if drop_blocks else '') + ('_mt' if args.match_time else '')
    out = args.output_dir
    pd.DataFrame({'vehicle_id': g_va, 'actual_class': y_val, 'pred_class': pred,
                  **{f'p_c{i}': p_val[:, i] for i in LABELS}}).to_csv(out / f'{tag}_val_predictions.csv', index=False)
    (out / f'{tag}_summary.json').write_text(json.dumps({
        'args': {k: str(v) for k, v in vars(args).items()},
        'block_adv_auc_c0': block_auc,
        'adv_auc_kept_c0': auc_all,
        'n_features': int(X_train.shape[1]),
        'oof': {'roc_auc': oof_auc, 'pr_auc': oof_ap, 'macro_f1': oof_scores},
        'val': results,
        'chosen': chosen,
    }, indent=2, default=float), encoding='utf-8')
    print('\nSaved to:', out)


if __name__ == '__main__':
    main()