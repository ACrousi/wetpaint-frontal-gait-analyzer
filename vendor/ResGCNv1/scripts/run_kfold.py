#!/usr/bin/env python
"""
5-Fold Cross-Validation 訓練腳本

使用方式：
    # 先生成 kfold 數據
    python main.py -c ./configs/resgcn_coco_2.yaml -gd --n_folds 5

    # 執行 5-fold 訓練
    python scripts/run_kfold.py --config ./configs/resgcn_coco_2.yaml --n_folds 5

    # 只跑某一個 fold
    python scripts/run_kfold.py --config ./configs/resgcn_coco_2.yaml --n_folds 5 --folds 2

    # 只收集結果（跳過訓練）
    python scripts/run_kfold.py --config ./configs/resgcn_coco_2.yaml --n_folds 5 --collect_only

輸出目錄：
    work_dir/{model_name}/run_kfold_{timestamp}/
        ├── kfold_summary.json     ← mean ± std 統計
        └── all_predictions.json   ← 合併所有 fold 的預測值
"""

import os
import sys
import json
import subprocess
import argparse
import numpy as np
from datetime import datetime


def find_fold_dirs(model_dir, n_folds):
    """在 model_dir 中找到每個 fold 最新的訓練目錄（通過 fold_results.json 辨識）"""
    fold_dirs = {}  # fold_idx -> latest dir path

    if not os.path.exists(model_dir):
        return fold_dirs

    for dir_name in sorted(os.listdir(model_dir)):
        dir_path = os.path.join(model_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        fold_file = os.path.join(dir_path, 'fold_results.json')
        if os.path.exists(fold_file):
            with open(fold_file, 'r', encoding='utf-8') as f:
                fold_info = json.load(f)
            fold_idx = fold_info.get('fold_idx')
            if fold_idx is not None and 0 <= fold_idx < n_folds:
                # sorted 保證最新的 timestamp 目錄覆蓋舊的
                fold_dirs[fold_idx] = dir_path

    return fold_dirs


def run_single_fold(config_path, fold_idx, n_folds, extra_args=None):
    """執行單個 fold 的訓練"""
    cmd = [
        sys.executable, 'main.py',
        '-c', config_path,
        '--fold_idx', str(fold_idx),
        '--n_folds', str(n_folds),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f'\n{"="*60}')
    print(f'  Starting Fold {fold_idx}/{n_folds}')
    print(f'  Command: {" ".join(cmd)}')
    print(f'{"="*60}\n')

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) + '/..')
    if result.returncode != 0:
        print(f'[ERROR] Fold {fold_idx} failed with exit code {result.returncode}')
        return False
    return True


def collect_and_summarize(model_dir, n_folds, output_dir):
    """收集所有 fold 結果並生成摘要"""
    fold_dirs = find_fold_dirs(model_dir, n_folds)

    if not fold_dirs:
        print('[ERROR] No fold results found!')
        return

    results = []
    all_predictions = []
    all_targets = []

    for fold_idx in range(n_folds):
        if fold_idx not in fold_dirs:
            print(f'[WARNING] Fold {fold_idx} not found, skipping')
            continue

        fold_dir = fold_dirs[fold_idx]

        # 讀取 fold_results.json
        results_path = os.path.join(fold_dir, 'fold_results.json')
        with open(results_path, 'r', encoding='utf-8') as f:
            fold_result = json.load(f)
        results.append(fold_result)
        print(f'Fold {fold_idx}: MAE={fold_result["best_mae"]:.4f}, '
              f'MSE={fold_result["best_mse"]:.4f}, '
              f'Spearman={fold_result["best_spearman"]:.4f}  '
              f'({os.path.basename(fold_dir)})')

        # 讀取 fold_predictions.json
        pred_path = os.path.join(fold_dir, 'fold_predictions.json')
        if os.path.exists(pred_path):
            with open(pred_path, 'r', encoding='utf-8') as f:
                fold_predictions = json.load(f)
            all_predictions.extend(fold_predictions['predictions'])
            all_targets.extend(fold_predictions['targets'])

    if not results:
        print('[ERROR] No valid fold results to summarize!')
        return

    # 計算 mean ± std
    maes = [r['best_mae'] for r in results]
    mses = [r['best_mse'] for r in results]
    spearmans = [r['best_spearman'] for r in results]

    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_folds': len(results),
        'per_fold': results,
        'fold_dirs': {str(k): os.path.basename(v) for k, v in fold_dirs.items()},
        'summary': {
            'mae': {'mean': float(np.mean(maes)), 'std': float(np.std(maes))},
            'mse': {'mean': float(np.mean(mses)), 'std': float(np.std(mses))},
            'spearman': {'mean': float(np.mean(spearmans)), 'std': float(np.std(spearmans))},
        },
        'total_eval_samples': len(all_predictions),
    }

    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    # 保存摘要
    summary_path = os.path.join(output_dir, 'kfold_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存合併的 predictions（用於畫圖）
    if all_predictions:
        merged_pred = {
            'all_predictions': all_predictions,
            'all_targets': all_targets,
        }
        pred_path = os.path.join(output_dir, 'all_predictions.json')
        with open(pred_path, 'w', encoding='utf-8') as f:
            json.dump(merged_pred, f, ensure_ascii=False, indent=2)

    # 列印論文用摘要
    print(f'\n{"="*60}')
    print(f'  {len(results)}-Fold Cross-Validation Results')
    print(f'{"="*60}')
    print(f'\n{"Fold":<6} {"MAE":>8} {"MSE":>8} {"Spearman":>10}')
    print(f'{"-"*34}')
    for r in results:
        print(f'{r["fold_idx"]:<6} {r["best_mae"]:>8.4f} {r["best_mse"]:>8.4f} {r["best_spearman"]:>10.4f}')
    print(f'{"-"*34}')
    print(f'{"Mean":<6} {np.mean(maes):>8.4f} {np.mean(mses):>8.4f} {np.mean(spearmans):>10.4f}')
    print(f'{"Std":<6} {np.std(maes):>8.4f} {np.std(mses):>8.4f} {np.std(spearmans):>10.4f}')
    print(f'\n  -> MAE:      {np.mean(maes):.4f} ± {np.std(maes):.4f}')
    print(f'  -> MSE:      {np.mean(mses):.4f} ± {np.std(mses):.4f}')
    print(f'  -> Spearman: {np.mean(spearmans):.4f} ± {np.std(spearmans):.4f}')
    print(f'\n  Output saved to: {output_dir}')
    print(f'{"="*60}\n')


def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation Training')
    parser.add_argument('--config', '-c', type=str, required=True, help='Config YAML file path')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='Specific fold indices to run (default: all)')
    parser.add_argument('--collect_only', action='store_true',
                        help='Only collect results (skip training)')
    parser.add_argument('--extra_args', type=str, nargs='*', default=None,
                        help='Extra arguments to pass to main.py')
    args = parser.parse_args()

    # 確定要執行的 fold
    folds_to_run = args.folds if args.folds else list(range(args.n_folds))

    if not args.collect_only:
        for fold_idx in folds_to_run:
            success = run_single_fold(args.config, fold_idx, args.n_folds, args.extra_args)
            if not success:
                print(f'[ERROR] Fold {fold_idx} failed. Continuing to next fold...')

    # 解析 config 來取得路徑
    import yaml
    config_path = args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    workspace_root = config.get('workspace_root', '../../outputs')
    config_dir = os.path.dirname(os.path.abspath(config_path))
    workspace_root = os.path.abspath(os.path.join(config_dir, workspace_root))

    work_dir = config.get('work_dir', 'resgcn_work_dir/')
    if not os.path.isabs(work_dir):
        work_dir = os.path.join(workspace_root, work_dir)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    model_type = config.get('model_type', 'pa-resgcn-b15-r2')
    dataset = config.get('dataset', 'coco')

    model_dir = os.path.join(work_dir, f'{config_name}_{model_type}_{dataset}')

    # 建立 run_kfold_{timestamp} 輸出資料夾
    ct = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(model_dir, f'run_kfold_{ct}')

    collect_and_summarize(model_dir, args.n_folds, output_dir)


if __name__ == '__main__':
    main()
