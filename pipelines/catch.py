# pipelines/catch.py
#
# Orchestrates the full CATCH experiment loop:
#   for each entity (channel / machine) in a dataset:
#       1. Load preprocessed numpy arrays from datasets/processed/catch/
#       2. Build CATCH model and config
#       3. Train via CATCH's detect_fit() (scaler bypass applied internally)
#       4. Infer anomaly scores on test split
#       5. Evaluate with bf_search + AUROC/AUCPR
#   Aggregate and print results across all entities.
#
# Structure mirrors pipelines/pgrf.py exactly. The evaluation block
# (bf_search, AUROC, AUCPR) is identical — this is intentional so that
# results are directly comparable across methods.
#
# Usage (from repo root):
#   python pipelines/catch.py --dataset SMAP
#   python pipelines/catch.py --dataset MSL
#   python pipelines/catch.py --dataset SMD
#   python pipelines/catch.py --dataset PSM
#
# Prerequisites:
#   methods/catch/preprocess_catch.py must have been run first.

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from methods.catch.loader import get_entity_dirs, load_entity
from methods.catch.model import build_catch_model
from methods.catch.training import train_catch_model
from methods.catch.inference import infer_catch_scores
from methods.pgrf.utils import bf_search   # shared evaluation utility

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROCESSED_ROOT = 'datasets/processed/catch'

SEQ_LEN = 192   # CATCH sliding window length (paper default)

# Training hyperparameters — paper defaults from DEFAULT_HYPER_PARAMS in model.py.
# Override per-dataset here if needed.
TRAIN_PARAMS = {
    'num_epochs': 3,
    'batch_size': 128,
    'patience':   3,
    'lr':         0.0001,
    'Mlr':        0.00001,
}

# ---------------------------------------------------------------------------
# Evaluation helpers  [identical to pgrf.py]
# ---------------------------------------------------------------------------

def _evaluate(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Run bf_search and compute AUROC / AUCPR. Returns a result dict."""
    if np.sum(labels) == 0:
        return {'F1': np.nan, 'P': np.nan, 'R': np.nan,
                'AUROC': np.nan, 'AUCPR': np.nan}

    best_metrics, _ = bf_search(scores, labels, verbose=False)
    result = {
        'F1': best_metrics[0],
        'P':  best_metrics[1],
        'R':  best_metrics[2],
    }

    if len(np.unique(labels)) > 1:
        result['AUROC'] = roc_auc_score(labels, scores)
        p, r, _ = precision_recall_curve(labels, scores)
        result['AUCPR'] = auc(r, p)
    else:
        result['AUROC'] = np.nan
        result['AUCPR'] = np.nan

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(dataset_name: str):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\nRunning CATCH on {dataset_name} | device={device}')
    print(f'seq_len={SEQ_LEN} | epochs={TRAIN_PARAMS["num_epochs"]}')

    entity_list = get_entity_dirs(PROCESSED_ROOT, dataset_name)
    print(f'Found {len(entity_list)} entities.\n')

    all_results = []

    for entity in entity_list:
        entity_id  = entity['entity_id']
        entity_dir = entity['entity_dir']

        print(f"{'='*70}")
        print(f'Entity: {entity_id}')
        print(f"{'='*70}")

        # --- Load preprocessed numpy arrays ---
        train_data, test_data, test_labels = load_entity(entity_dir)
        n_vars = train_data.shape[1]

        print(f'  vars={n_vars} | '
              f'train_timesteps={len(train_data)} | '
              f'test_timesteps={len(test_data)} | '
              f'anomaly_ratio={test_labels.mean():.2%}')

        # Guard: need at least one full window of train data
        if len(train_data) < SEQ_LEN:
            print(f'  Train data shorter than seq_len={SEQ_LEN}. Skipping.')
            continue

        # --- Compute safe batch size ---
        # CATCH's training loop computes step = min(int(n_batches / 10), 100)
        # and uses it as a modulo divisor. If n_batches < 10, step=0 and a
        # ZeroDivisionError is raised. We compute the largest power-of-2
        # batch size that keeps n_batches >= 10, falling back to skip if the
        # entity is too small even at bs=16.
        train_windows = int(len(train_data) * 0.8) - SEQ_LEN + 1
        safe_batch_size = TRAIN_PARAMS['batch_size']
        for bs_candidate in [128, 64, 32, 16]:
            if train_windows // bs_candidate >= 10:
                safe_batch_size = bs_candidate
                break
        else:
            print(f'  Entity too small for CATCH (only {train_windows} windows). Skipping.')
            continue

        if safe_batch_size != TRAIN_PARAMS['batch_size']:
            print(f'  Reduced batch_size to {safe_batch_size} '
                  f'(only {train_windows} train windows).')

        effective_params = {**TRAIN_PARAMS, 'batch_size': safe_batch_size}

        # --- Build model ---
        model, config = build_catch_model(n_vars=n_vars, seq_len=SEQ_LEN)

        # --- Train ---
        catch_instance = train_catch_model(
            model, config, train_data, **effective_params
        )

        # --- Infer anomaly scores ---
        print(f'  Inferring scores...')
        scores = infer_catch_scores(catch_instance, test_data)

        # Guard: scores must be non-trivial to evaluate
        if scores.std() == 0:
            print(f'  Scores have zero variance. Skipping evaluation.')
            continue

        # --- Evaluate ---
        result = _evaluate(scores, test_labels)
        result['entity_id'] = entity_id

        print(f"  F1={result['F1']:.4f} | "
              f"P={result['P']:.4f} | "
              f"R={result['R']:.4f} | "
              f"AUROC={result.get('AUROC', float('nan')):.4f} | "
              f"AUCPR={result.get('AUCPR', float('nan')):.4f}")

        all_results.append(result)

    # --- Aggregate ---
    if not all_results:
        print('No results to aggregate.')
        return

    results_df = pd.DataFrame(all_results)
    means = results_df.mean(numeric_only=True)

    print(f"\n{'='*70}")
    print(f'CATCH | Dataset: {dataset_name}')
    print(f'seq_len={SEQ_LEN} | epochs={TRAIN_PARAMS["num_epochs"]}')
    print(f'Entities evaluated: {len(all_results)}')
    print(f"{'='*70}")
    print(f"Mean F1:    {means['F1']:.4f}")
    print(f"Mean P:     {means['P']:.4f}")
    print(f"Mean R:     {means['R']:.4f}")
    print(f"Mean AUROC: {means['AUROC']:.4f}")
    print(f"Mean AUCPR: {means['AUCPR']:.4f}")

    out_path = f'results_catch_{dataset_name.lower()}.csv'
    results_df.to_csv(out_path, index=False)
    print(f'\nPer-entity results saved to: {out_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CATCH on a dataset.')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['SMAP', 'MSL', 'SMD', 'PSM'],
        help='Dataset to run. Must be preprocessed first.'
    )
    args = parser.parse_args()
    run(args.dataset)
