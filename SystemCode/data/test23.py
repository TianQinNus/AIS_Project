"""FleetCare 最终模型: 训练行 + 全部验证行(加权) 训练, 保存模型与特征元数据。

依赖: 同目录下的 test_20_normalized.py 和 test_15_16gb.py。

训练:
  python final_model.py
  python final_model.py --val-weight 10

预测(对新数据, 每辆车取最后一行):
  python final_model.py --mode predict --op-csv raw/xxx_operational_readouts.csv \
        --spec-csv raw/xxx_specifications.csv --out-csv processed/predictions.csv

输出列: vehicle_id, p_c0..p_c4, risk(=1-p_c0), risk_rank(1=风险最高), top_class(C1..C4 中概率最大的类)
告警阈值/告警率是运营决策, 脚本不替你选。
"""
import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from test21 import (
    load, choose_rows_multi, detect_groups, build_features, spec_matrix, LABELS,
)
from test_15_16gb import model


def spec_for(spec_df, ids, s_names):
    """把新数据的规格表对齐到训练时的列(缺失的 one-hot 列补 0)。"""
    feat = spec_df.set_index('vehicle_id')
    cat_cols = [c for c in feat if not pd.api.types.is_numeric_dtype(feat[c])]
    feat = pd.get_dummies(feat, columns=cat_cols, dtype=np.float32)
    feat = feat.reindex(columns=s_names, fill_value=0.0).astype(np.float32)
    return feat.reindex(ids).to_numpy()


def train(args):
    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
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
    feat_names = names + [f'spec:{n}' for n in s_names]
    del Xtr_f, Xva_f, s_tr, s_va
    gc.collect()

    X = np.vstack([X_train, X_val])
    y = np.r_[y_train, y_val]
    w = np.r_[sw, np.full(len(y_val), float(args.val_weight))]
    print('Final matrix:', X.shape, '| 验证行权重:', args.val_weight, flush=True)

    m = model(binary=False)
    print('Training final model...', flush=True)
    m.fit(X, y, sample_weight=w)
    if list(m.classes_) != LABELS:
        raise ValueError(f'Unexpected class order: {m.classes_}')

    joblib.dump(m, model_dir / 'model.joblib')
    try:
        m.save_model(str(model_dir / 'model.json'))
    except Exception as e:  # model() 不是 XGBClassifier 时跳过
        print('save_model 跳过:', e)
    meta = {
        'raw_cols': [str(c) for c in raw_cols],
        'lags': lags,
        'spec_columns': s_names,
        'feature_names': feat_names,
        'n_features': int(X.shape[1]),
        'val_weight': float(args.val_weight),
        'k_repair': args.k_repair,
        'k_nonrepair': args.k_nonrepair,
        'train_rows': int(len(y_train)),
        'val_rows': int(len(y_val)),
        'reported_performance_note': (
            '对外报告请使用 test_21 的交叉拟合结果(验证集 5 折, 验证行权重 10): '
            'ROC-AUC 0.773, PR-AUC 0.103, Recall@Top5% 0.257。'
            '本模型的验证行已进入训练, 在验证集上的指标是样本内的, 不可作为性能报告。'),
    }
    (model_dir / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved to:', model_dir)
    print(meta['reported_performance_note'])


def predict(args):
    model_dir = args.model_dir
    meta = json.loads((model_dir / 'meta.json').read_text(encoding='utf-8'))
    m = joblib.load(model_dir / 'model.joblib')
    raw_cols = meta['raw_cols']

    op = pd.read_csv(args.op_csv)
    spec = pd.read_csv(args.spec_csv)
    missing = set(raw_cols) - set(op.columns)
    if missing:
        raise ValueError(f'输入缺少 {len(missing)} 个特征列, 例如 {sorted(missing)[:5]}')
    op = op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    last = op.groupby('vehicle_id', sort=False).tail(1)
    sel = pd.DataFrame({'vehicle_id': last['vehicle_id'].to_numpy(), 'row_id': last.index.to_numpy()})

    hist, singles = detect_groups(raw_cols)
    X_f, names, _, _ = build_features(op, sel, raw_cols, hist, singles, meta['lags'])
    X_s = spec_for(spec, sel['vehicle_id'].to_numpy(), meta['spec_columns'])
    X = np.hstack([X_f, X_s]).astype(np.float32)
    if X.shape[1] != meta['n_features']:
        raise ValueError(f"特征数不一致: 推理 {X.shape[1]} vs 训练 {meta['n_features']}")

    p = m.predict_proba(X)
    risk = 1 - p[:, 0]
    out = pd.DataFrame({'vehicle_id': sel['vehicle_id'].to_numpy(),
                        **{f'p_c{i}': p[:, i] for i in LABELS},
                        'risk': risk})
    out['risk_rank'] = out['risk'].rank(ascending=False, method='first').astype(int)
    out['top_class'] = 1 + np.argmax(p[:, 1:], axis=1)
    out = out.sort_values('risk_rank')
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print('Predictions saved to:', args.out_csv, '| vehicles:', len(out))


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument('--mode', choices=['train', 'predict'], default='train')
    ap.add_argument('--data-dir', type=Path, default=root / 'raw')
    ap.add_argument('--model-dir', type=Path, default=root / 'processed' / 'final_model')
    ap.add_argument('--k-repair', type=int, default=3)
    ap.add_argument('--k-nonrepair', type=int, default=2)
    ap.add_argument('--lags', type=str, default='5,20')
    ap.add_argument('--val-weight', type=float, default=10.0)
    ap.add_argument('--op-csv', type=Path)
    ap.add_argument('--spec-csv', type=Path)
    ap.add_argument('--out-csv', type=Path, default=root / 'processed' / 'predictions.csv')
    args = ap.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        if not (args.op_csv and args.spec_csv):
            raise SystemExit('predict 模式需要 --op-csv 和 --spec-csv')
        predict(args)


if __name__ == '__main__':
    main()