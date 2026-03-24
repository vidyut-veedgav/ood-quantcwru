# pipelines/pgrf.py
#
# Orchestrates the full PGRF-Net experiment loop:
#   for each entity (channel / machine) in a dataset:
#       1. Load preprocessed numpy arrays
#       2. Create forecast windows (X, Y, L)
#       3. Train Stage 1 (evidence extractor)
#       4. Train Stage 2 (gating fusion)
#       5. Infer anomaly scores on test split
#       6. Evaluate with bf_search + AUROC/AUCPR
#   Aggregate and print results across all entities.
#
# Usage (from repo root):
#   python pipelines/pgrf.py --dataset SMAP
#   python pipelines/pgrf.py --dataset MSL
#   python pipelines/pgrf.py --dataset SMD
#   python pipelines/pgrf.py --dataset PSM
#
# Prerequisites:
#   data/preprocessing/preprocess_pgrf.py must have been run first.

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from data.dataloaders.pgrf_loader import get_entity_dirs
from methods.pgrf.model import PGRFNet
from methods.pgrf.training import train_model_stage1, train_model_stage2
from methods.pgrf.inference import infer_scores
from methods.pgrf.utils import create_windows, bf_search

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROCESSED_ROOT = 'datasets/processed/pgrf'

WINDOW_SIZE = 60

TRAIN_PARAMS = {
    'epochs_stage1': 50,
    'lr': 1e-4,
    'batch_size': 128,
    'patience_stage1': 10,
    'epochs_stage2': 20,
    'lr_stage2': 1e-4,
    'patience_stage2': 5,
    'focal_gamma': 2.0,
    'focal_alpha': 0.5,
    'anomaly_weight': 10.0,
    'mask_reg_weight': 0.01,
    'mask_diff_weight': 0.001,
    'acyclic_penalty_weight': 1e-4,
    'lambda1': 1e-3,
    'sparsity_lambda': 1e-3,
    'context_loss_weight_stage1': 0.01,
    'spike_loss_weight_stage1': 0.01,
    'pseudo_normal_percent': 0.25,
    'gate_normal_suppress_weight': 0.1,
    'gate_entropy_weight': 0.001,
    'context_loss_weight_stage2': 0.2,
    'spike_loss_weight_stage2': 0.2,
}

# Fixed alpha for combining predictive and explanatory scores.
# alpha=0 means purely predictive; alpha=1 means purely explanatory.
FIXED_ALPHA = 0.1

NUM_PROTOS = 10

# ---------------------------------------------------------------------------
# Per-entity evaluation
# ---------------------------------------------------------------------------

def _compute_combined_score(scores_dict: dict, alpha: float) -> np.ndarray:
    """
    Combine the four PGRF evidence scores into a single anomaly score.

    Predictive score gets weight (1 - alpha).
    The three explanatory scores (structural, contextual, spike) are
    normalised to sum to 1 within each timestep, then weighted by alpha.
    """
    s_pred       = scores_dict['predictive_scores']
    s_structural = scores_dict['structural_scores']
    s_ctx        = scores_dict['contextual_scores']
    s_spike      = scores_dict['spike_scores']

    expl = np.stack([s_structural, s_ctx, s_spike], axis=-1)
    expl_sum = expl.sum(axis=1, keepdims=True) + 1e-8
    expl_norm = expl / expl_sum
    expl_score = np.sum(expl_norm * expl, axis=1)

    return (1 - alpha) * s_pred + alpha * expl_score


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\nRunning PGRF on {dataset_name} | device={device}')
    print(f'Alpha={FIXED_ALPHA} | window_size={WINDOW_SIZE}')

    # --- Get all entity directories for this dataset ---
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
        train_data  = np.load(f'{entity_dir}/train.npy')
        test_data   = np.load(f'{entity_dir}/test.npy')
        test_labels = np.load(f'{entity_dir}/test_labels.npy')

        # --- Create forecast windows (X, Y, L) ---
        # Train labels are all zero (unsupervised — train split is anomaly-free)
        train_labels = np.zeros(len(train_data), dtype=np.float32)
        X_train, Y_train, L_train = create_windows(
            train_data, train_labels, WINDOW_SIZE
        )

        if X_train.shape[0] == 0:
            print(f'  Not enough data to create windows. Skipping.')
            continue

        N_VARS = X_train.shape[2]
        print(f'  vars={N_VARS} | '
              f'train_windows={X_train.shape[0]} | '
              f'test_timesteps={len(test_data)} | '
              f'anomaly_ratio={test_labels.mean():.2%}')

        # --- Initialise model ---
        model = PGRFNet(
            num_vars=N_VARS,
            seq_len=WINDOW_SIZE,
            num_protos=NUM_PROTOS,
            num_context_protos=NUM_PROTOS,
            num_spike_protos=NUM_PROTOS,
        )
        if device == 'cuda':
            model.cuda()

        # --- Two-stage training ---
        train_model_stage1(model, X_train, Y_train, L_train, **TRAIN_PARAMS)
        train_model_stage2(model, X_train, Y_train, L_train, **TRAIN_PARAMS)

        # --- Inference ---
        print(f'  Inferring scores...')
        scores_dict = infer_scores(model, test_data, WINDOW_SIZE)

        if not scores_dict:
            print(f'  Inference returned empty scores. Skipping.')
            continue

        # --- Score combination and evaluation ---
        combined = _compute_combined_score(scores_dict, FIXED_ALPHA)
        result = _evaluate(combined, test_labels)
        result['entity_id'] = entity_id

        print(f"  F1={result['F1']:.4f} | "
              f"P={result['P']:.4f} | "
              f"R={result['R']:.4f} | "
              f"AUROC={result['AUROC']:.4f} | "
              f"AUCPR={result['AUCPR']:.4f}")

        all_results.append(result)

    # --- Aggregate results ---
    if not all_results:
        print('No results to aggregate.')
        return

    results_df = pd.DataFrame(all_results)
    means = results_df.mean(numeric_only=True)

    print(f"\n{'='*70}")
    print(f'PGRF-Net | Dataset: {dataset_name}')
    print(f'Alpha: {FIXED_ALPHA} | Window: {WINDOW_SIZE}')
    print(f'Entities evaluated: {len(all_results)}')
    print(f"{'='*70}")
    print(f"Mean F1:    {means['F1']:.4f}")
    print(f"Mean P:     {means['P']:.4f}")
    print(f"Mean R:     {means['R']:.4f}")
    print(f"Mean AUROC: {means['AUROC']:.4f}")
    print(f"Mean AUCPR: {means['AUCPR']:.4f}")

    # Save per-entity results to CSV
    out_path = f'results_pgrf_{dataset_name.lower()}.csv'
    results_df.to_csv(out_path, index=False)
    print(f'\nPer-entity results saved to: {out_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PGRF-Net on a dataset.')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['SMAP', 'MSL', 'SMD', 'PSM'],
        help='Dataset to run. Must be preprocessed first.'
    )
    args = parser.parse_args()
    run(args.dataset)