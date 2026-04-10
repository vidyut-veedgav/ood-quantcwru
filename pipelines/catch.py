# pipelines/catch.py
#
# Orchestrates the full CATCH experiment loop:
#   for each entity (channel / machine) in a dataset:
#       1. Load preprocessed numpy arrays from datasets/processed/catch/
#       2. Build CATCH model and config
#       3. Train via CATCH's detect_fit() (scaler bypass applied internally)
#       4. Infer anomaly scores on test split
#       5. Evaluate: bf_search (F1/P/R) + AUROC + AUCPR
#   Aggregate per-entity results, print a summary table, and save to
#   results/catch/<dataset>/  (per-entity CSV + summary CSV).
#
# Structure mirrors pipelines/pgrf.py exactly. The evaluation block is
# identical across both pipelines so results are directly comparable.
#
# Usage (from repo root):
#   python pipelines/catch.py --dataset SMAP
#   python pipelines/catch.py --dataset MSL
#   python pipelines/catch.py --dataset SMD
#   python pipelines/catch.py --dataset PSM
#
# Prerequisites:
#   methods/catch/preprocess_catch.py must have been run first.

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
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
RESULTS_ROOT   = 'results/catch'           # all output saved here

SEQ_LEN = 192   # CATCH sliding window length (paper default)

# Training hyperparameters — paper defaults from DEFAULT_HYPER_PARAMS in model.py.
TRAIN_PARAMS = {
    'num_epochs': 3,
    'batch_size': 128,
    'patience':   3,
    'lr':         0.0001,
    'Mlr':        0.00001,
}

# Metrics displayed in the summary table, in order.
METRICS = ['F1', 'P', 'R', 'AUROC', 'AUCPR']


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(scores: np.ndarray, labels: np.ndarray) -> dict:
    """
    Run bf_search for best-threshold F1/P/R, then compute AUROC and AUCPR.
    Returns a dict with keys: F1, P, R, AUROC, AUCPR.
    All values are NaN if there are no positive labels.
    """
    if np.sum(labels) == 0:
        return {m: np.nan for m in METRICS}

    best_metrics, _ = bf_search(scores, labels, verbose=False)
    result = {
        'F1': best_metrics[0],
        'P':  best_metrics[1],
        'R':  best_metrics[2],
    }

    if len(np.unique(labels)) > 1:
        result['AUROC'] = roc_auc_score(labels, scores)
        p, r, _         = precision_recall_curve(labels, scores)
        result['AUCPR'] = auc(r, p)
    else:
        result['AUROC'] = np.nan
        result['AUCPR'] = np.nan

    return result


# ---------------------------------------------------------------------------
# Table image saver
# ---------------------------------------------------------------------------

def _save_table_image(dataset_name: str, results_df: pd.DataFrame, out_dir: str):
    """
    Render a publication-style table image (similar to CATCH paper Table 1).

    Layout:
        Dataset  | Metric | Mean  | Min   | Max   | N
        ---------|--------|-------|-------|-------|---
        <name>   | F1     | 0.xxx | 0.xxx | 0.xxx | N
                 | P      | ...
                 | R      | ...
                 | AUROC  | ...
                 | AUCPR  | ...

    Saved to: out_dir/results_table.png
    """
    # Build rows: [dataset_label, metric, mean, min, max, n]
    rows = []
    for i, metric in enumerate(METRICS):
        col    = results_df[metric].dropna()
        rows.append([
            dataset_name if i == 0 else '',
            metric,
            f'{col.mean():.4f}' if len(col) else 'N/A',
            f'{col.min():.4f}'  if len(col) else 'N/A',
            f'{col.max():.4f}'  if len(col) else 'N/A',
            str(len(col)),
        ])

    col_labels  = ['Dataset', 'Metric', 'Mean', 'Min', 'Max', 'N']
    col_widths  = [0.18, 0.12, 0.14, 0.14, 0.14, 0.06]   # fractions of figure width

    n_rows = len(rows)
    fig_h  = 0.35 * (n_rows + 1.5)   # scale height with row count
    fig, ax = plt.subplots(figsize=(7, fig_h))
    ax.axis('off')

    fp_header = FontProperties(weight='bold', size=9)
    fp_cell   = FontProperties(size=9)

    # --- Draw header ---
    x_positions = []
    x = 0.02
    for w in col_widths:
        x_positions.append(x + w / 2)
        x += w

    y_header = 0.92
    row_h    = (y_header - 0.08) / (n_rows + 1)

    for label, xpos in zip(col_labels, x_positions):
        ax.text(xpos, y_header, label,
                ha='center', va='center',
                transform=ax.transAxes,
                fontproperties=fp_header)

    def _hline(y, lw=0.8):
        ax.plot([0.02, 0.98], [y, y], color='black', linewidth=lw,
                transform=ax.transAxes, clip_on=False)

    def _vline(x, y0, y1, lw=0.6):
        ax.plot([x, x], [y0, y1], color='black', linewidth=lw,
                transform=ax.transAxes, clip_on=False)

    # --- Header separator lines ---
    _hline(y_header + row_h * 0.6, lw=1.2)
    _hline(y_header - row_h * 0.55, lw=0.8)

    # --- Draw data rows ---
    for i, row in enumerate(rows):
        y = y_header - row_h * (i + 1.1)
        for cell, xpos in zip(row, x_positions):
            ax.text(xpos, y, cell,
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontproperties=fp_cell)

    # --- Bottom separator ---
    _hline(y_header - row_h * (n_rows + 0.6), lw=1.2)

    # --- Vertical separator after Metric column ---
    x_vsep = x_positions[1] - col_widths[1] / 2 + 0.005
    _vline(x_vsep, 0.06, 0.98)

    # --- Caption ---
    caption = (f'CATCH  |  dataset={dataset_name}  |  seq_len={SEQ_LEN}  |  '
               f'epochs={TRAIN_PARAMS["num_epochs"]}  |  '
               f'entities={len(results_df)}')
    ax.text(0.5, 0.03, caption,
            ha='center', va='center',
            transform=ax.transAxes,
            fontsize=7, style='italic', color='#444444')

    plt.tight_layout(pad=0.3)
    img_path = os.path.join(out_dir, 'results_table.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Table image saved : {img_path}')


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def _print_summary_table(dataset_name: str, results_df: pd.DataFrame):
    """
    Print a summary table in the style of the CATCH paper results table.

    Columns: Dataset | Metric | mean | min | max | n_entities
    Rows:    one row per metric (F1, P, R, AUROC, AUCPR).
    """
    col_w = 10
    header = (f"{'Dataset':<12} {'Metric':<8} "
              f"{'Mean':>{col_w}} {'Min':>{col_w}} {'Max':>{col_w}} {'N':>4}")
    sep = '-' * len(header)

    print(f'\n{sep}')
    print(header)
    print(sep)

    for i, metric in enumerate(METRICS):
        col    = results_df[metric].dropna()
        mean_v = col.mean()
        min_v  = col.min()
        max_v  = col.max()
        n      = len(col)
        # Only print dataset name on the first metric row
        ds_label = dataset_name if i == 0 else ''
        print(f'{ds_label:<12} {metric:<8} '
              f'{mean_v:>{col_w}.4f} {min_v:>{col_w}.4f} '
              f'{max_v:>{col_w}.4f} {n:>4}')

    print(sep)
    print(f'  CATCH | seq_len={SEQ_LEN} | epochs={TRAIN_PARAMS["num_epochs"]} '
          f'| entities evaluated: {len(results_df)}')
    print(sep)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(dataset_name: str):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\nRunning CATCH on {dataset_name} | device={device}')
    print(f'seq_len={SEQ_LEN} | epochs={TRAIN_PARAMS["num_epochs"]}')

    # Create output directory for this dataset
    out_dir = os.path.join(RESULTS_ROOT, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    entity_list = get_entity_dirs(PROCESSED_ROOT, dataset_name)
    print(f'Found {len(entity_list)} entities.')
    print(f'Results will be saved to: {out_dir}/\n')

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
              f'train={len(train_data)} | '
              f'test={len(test_data)} | '
              f'anomaly_ratio={test_labels.mean():.2%}')

        # Guard: need at least one full window of train data
        if len(train_data) < SEQ_LEN:
            print(f'  Train data shorter than seq_len={SEQ_LEN}. Skipping.')
            continue

        # --- Compute safe batch size ---
        # CATCH's training loop computes step = min(int(n_batches / 10), 100)
        # and uses it as a modulo divisor. If n_batches < 10, step=0 and a
        # ZeroDivisionError is raised. We find the largest batch size (from
        # the candidate list) that keeps n_batches >= 10.
        train_windows = int(len(train_data) * 0.8) - SEQ_LEN + 1
        safe_batch_size = TRAIN_PARAMS['batch_size']
        for bs_candidate in [128, 64, 32, 16]:
            if train_windows // bs_candidate >= 10:
                safe_batch_size = bs_candidate
                break
        else:
            print(f'  Entity too small for CATCH '
                  f'(only {train_windows} train windows). Skipping.')
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
        result         = _evaluate(scores, test_labels)
        result['entity_id'] = entity_id

        print(f"  F1={result['F1']:.4f} | "
              f"P={result['P']:.4f} | "
              f"R={result['R']:.4f} | "
              f"AUROC={result['AUROC']:.4f} | "
              f"AUCPR={result['AUCPR']:.4f}")

        all_results.append(result)

    # --- Aggregate and report ---
    if not all_results:
        print('No results to aggregate.')
        return

    # Column order: entity_id first, then metrics
    results_df = pd.DataFrame(all_results).reindex(columns=['entity_id'] + METRICS)

    # Print summary table
    _print_summary_table(dataset_name, results_df)

    # --- Save results ---
    # Per-entity CSV: one row per entity with all five metrics
    per_entity_path = os.path.join(out_dir, 'per_entity.csv')
    results_df.to_csv(per_entity_path, index=False, float_format='%.4f')
    print(f'\nPer-entity results : {per_entity_path}')

    # Summary CSV: one row per metric with mean/min/max/n
    summary_rows = []
    for metric in METRICS:
        col = results_df[metric].dropna()
        summary_rows.append({
            'dataset':    dataset_name,
            'method':     'CATCH',
            'metric':     metric,
            'mean':       col.mean(),
            'min':        col.min(),
            'max':        col.max(),
            'n_entities': len(col),
            'seq_len':    SEQ_LEN,
            'num_epochs': TRAIN_PARAMS['num_epochs'],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f'Summary results    : {summary_path}')

    # Table image
    _save_table_image(dataset_name, results_df, out_dir)


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
