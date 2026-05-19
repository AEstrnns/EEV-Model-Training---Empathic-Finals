import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, r2_score)

EXPRESSION_NAMES = [
    'Amusement', 'Anger', 'Awe', 'Concentration', 'Confusion',
    'Contempt', 'Contentment', 'Disappointment', 'Doubt',
    'Elation', 'Interest', 'Pain', 'Sadness', 'Surprise', 'Triumph'
]

DEFAULT_CONFIG = {
    'input_size': 15,
    'hidden_size': 64,
    'num_layers': 1,
    'output_size': 15,
    'sequence_length': 30,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 50,
    'dropout': 0.2,
}


class EmotionForecastingLSTM(nn.Module):
    def __init__(self, cfg):
        super(EmotionForecastingLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=cfg['input_size'],
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            batch_first=True,
            dropout=cfg['dropout'] if cfg['num_layers'] > 1 else 0,
        )
        self.dropout = nn.Dropout(cfg['dropout'])
        self.fc = nn.Linear(cfg['hidden_size'], cfg['output_size'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]
        dropped = self.dropout(last_timestep)
        out = self.fc(dropped)
        return self.sigmoid(out)


def detect_csv_columns(columns, csv_name=None):
    normalized = {col.strip().lower(): col for col in columns}
    video_col = next(
        (normalized[c] for c in ['video_id', 'youtube id', 'video id'] if c in normalized),
        None,
    )
    time_col = next(
        (normalized[c] for c in ['timestamp_ms', 'timestamp (milliseconds)', 'timestamp milliseconds', 'timestamp'] if c in normalized),
        None,
    )
    if video_col is None or time_col is None:
        raise KeyError(
            'Expected video and timestamp columns not found. '
            f'Found columns: {list(columns)}'
        )

    expression_cols = []
    for expr in EXPRESSION_NAMES:
        expr_key = expr.strip().lower()
        if expr_key in normalized:
            expression_cols.append(normalized[expr_key])
        else:
            raise KeyError(
                f"Missing expected emotion column '{expr}' in {csv_name or 'CSV file'}. "
                f'Found columns: {list(columns)}'
            )

    return video_col, time_col, expression_cols


def create_sequences_from_csv(csv_path: Path, seq_len: int):
    print(f'Loading {csv_path}')
    df = pd.read_csv(csv_path)
    video_col, time_col, expression_cols = detect_csv_columns(df.columns, csv_path.name)
    df = df.sort_values(by=[video_col, time_col])

    X, Y = [], []
    groups = df.groupby(video_col)
    for _, group in groups:
        values = group[expression_cols].values.astype(np.float32)
        for i in range(len(values) - seq_len):
            X.append(values[i : i + seq_len])
            Y.append(values[i + seq_len])

    if len(X) == 0:
        raise ValueError(
            f'No valid sequences found in {csv_path}. Check whether the dataset contains '
            'enough rows per video to create sequences.'
        )

    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)


class SequenceIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, csv_path, seq_len, chunksize=200_000, shuffle=False, buffer_size=20_000):
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self):
        video_col = None
        time_col = None
        expression_cols = None
        buffer_df = None
        buffer = []
        rng = np.random.default_rng()

        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunksize):
            if buffer_df is not None:
                chunk = pd.concat([buffer_df, chunk], ignore_index=True)

            if video_col is None:
                video_col, time_col, expression_cols = detect_csv_columns(chunk.columns, Path(self.csv_path).name)

            chunk = chunk.sort_values(by=[video_col, time_col])
            if chunk.empty:
                continue

            last_video_id = chunk.iloc[-1][video_col]
            complete = chunk[chunk[video_col] != last_video_id]
            buffer_df = chunk[chunk[video_col] == last_video_id]

            for _, group in complete.groupby(video_col):
                values = group[expression_cols].to_numpy(dtype=np.float32)
                for i in range(len(values) - self.seq_len):
                    item = (torch.from_numpy(values[i : i + self.seq_len]), torch.from_numpy(values[i + self.seq_len]))
                    if not self.shuffle:
                        yield item
                        continue

                    buffer.append(item)
                    if len(buffer) >= self.buffer_size:
                        idx = rng.integers(len(buffer))
                        yield buffer[idx]
                        buffer[idx] = buffer[-1]
                        buffer.pop()

        if buffer_df is not None and not buffer_df.empty:
            for _, group in buffer_df.groupby(video_col):
                values = group[expression_cols].to_numpy(dtype=np.float32)
                for i in range(len(values) - self.seq_len):
                    item = (torch.from_numpy(values[i : i + self.seq_len]), torch.from_numpy(values[i + self.seq_len]))
                    if not self.shuffle:
                        yield item
                        continue

                    buffer.append(item)
                    if len(buffer) >= self.buffer_size:
                        idx = rng.integers(len(buffer))
                        yield buffer[idx]
                        buffer[idx] = buffer[-1]
                        buffer.pop()

        if self.shuffle:
            rng.shuffle(buffer)
            for item in buffer:
                yield item


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location='cpu')
    config = checkpoint.get('config', DEFAULT_CONFIG)
    model = EmotionForecastingLSTM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def evaluate(model, dataloader, device):
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            preds = model(X)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(Y.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    return y_true, y_pred


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    metrics = {
        'expression': [],
        'mse': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'explained_variance': [],
    }

    for idx, name in enumerate(EXPRESSION_NAMES):
        y_true_dim = y_true[:, idx]
        y_pred_dim = y_pred[:, idx]
        mse = mean_squared_error(y_true_dim, y_pred_dim)
        mae = mean_absolute_error(y_true_dim, y_pred_dim)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true_dim, y_pred_dim)
        explained = explained_variance_score(y_true_dim, y_pred_dim)

        metrics['expression'].append(name)
        metrics['mse'].append(mse)
        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        metrics['r2'].append(r2)
        metrics['explained_variance'].append(explained)

    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    overall = {
        'mse': mean_squared_error(flat_true, flat_pred),
        'rmse': float(np.sqrt(mean_squared_error(flat_true, flat_pred))),
        'mae': mean_absolute_error(flat_true, flat_pred),
        'r2': r2_score(flat_true, flat_pred),
        'explained_variance': explained_variance_score(flat_true, flat_pred),
        'samples': flat_true.shape[0],
    }

    return pd.DataFrame(metrics), overall


def save_metric_plots(output_dir: Path, metrics_df: pd.DataFrame, overall: dict, y_true: np.ndarray, y_pred: np.ndarray):
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(metrics_df))

    # Combined per-expression metrics chart: errors + scores
    fig, (ax_err, ax_score) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    width = 0.25
    ax_err.bar(x - width, metrics_df['mse'], width, label='MSE')
    ax_err.bar(x, metrics_df['rmse'], width, label='RMSE')
    ax_err.bar(x + width, metrics_df['mae'], width, label='MAE')
    ax_err.set_ylabel('Error')
    ax_err.set_title('Per-Expression Error Metrics')
    ax_err.legend()
    ax_err.grid(axis='y', linestyle='--', alpha=0.5)

    ax_score.bar(x - width / 2, metrics_df['r2'], width, label='R2', alpha=0.85)
    ax_score.bar(x + width / 2, metrics_df['explained_variance'], width, label='Explained Variance', alpha=0.7)
    ax_score.set_ylabel('Score')
    ax_score.set_title('Per-Expression Score Metrics')
    ax_score.set_ylim(-1.0, 1.1)
    ax_score.legend()
    ax_score.grid(axis='y', linestyle='--', alpha=0.5)

    ax_score.set_xticks(x)
    ax_score.set_xticklabels(metrics_df['expression'], rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(output_dir / 'evaluation_per_expression_metrics.png')
    plt.close(fig)

    # Per-expression error-only chart
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, metrics_df['mse'], width, label='MSE')
    ax.bar(x, metrics_df['rmse'], width, label='RMSE')
    ax.bar(x + width, metrics_df['mae'], width, label='MAE')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['expression'], rotation=45, ha='right')
    ax.set_ylabel('Error')
    ax.set_title('Regression Error Metrics per Expression')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / 'evaluation_errors.png')
    plt.close(fig)

    # Per-expression R2 and explained variance chart
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, metrics_df['r2'], width, label='R2', alpha=0.85)
    ax.bar(x + width / 2, metrics_df['explained_variance'], width, label='Explained Variance', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['expression'], rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('R2 and Explained Variance per Expression')
    ax.set_ylim(-1.0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / 'evaluation_r2_explained_variance.png')
    plt.close(fig)

    # Overall metric summary chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metric_names = ['MSE', 'RMSE', 'MAE']
    values = [overall['mse'], overall['rmse'], overall['mae']]
    axes[0].bar(metric_names, values, color=['tab:blue', 'tab:orange', 'tab:green'])
    axes[0].set_title('Overall Error Metrics')
    axes[0].set_ylabel('Value')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    score_names = ['R2', 'Explained Variance']
    score_values = [overall['r2'], overall['explained_variance']]
    axes[1].bar(score_names, score_values, color=['tab:purple', 'tab:red'])
    axes[1].set_title('Overall Score Metrics')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / 'evaluation_overall_metrics.png')
    plt.close(fig)

    # Predicted vs actual scatter plot
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    sample_size = min(5000, flat_true.shape[0])
    sample_indices = np.random.default_rng(0).choice(flat_true.shape[0], size=sample_size, replace=False)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(flat_true[sample_indices], flat_pred[sample_indices], s=2, alpha=0.3)
    ax.plot([0.0, 1.0], [0.0, 1.0], color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Actual Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title('Predicted vs Actual Values (sampled points)')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / 'evaluation_prediction_vs_actual.png')
    plt.close(fig)

    # Save the exact sample values used for the predicted vs actual graph
    row_indices = sample_indices // len(EXPRESSION_NAMES)
    expr_indices = sample_indices % len(EXPRESSION_NAMES)
    predicted_vs_actual = pd.DataFrame({
        'sequence_index': row_indices,
        'expression': [EXPRESSION_NAMES[i] for i in expr_indices],
        'actual': flat_true[sample_indices],
        'predicted': flat_pred[sample_indices],
    })
    predicted_vs_actual.to_csv(output_dir / 'predicted_vs_actual_sample.csv', index=False, float_format='%.10f')


def print_metrics(metrics_df: pd.DataFrame, overall: dict, csv_path: Path, checkpoint_path: Path):
    print(f'Loaded evaluation file: {csv_path}')
    print(f'Loaded model checkpoint: {checkpoint_path}')
    print('\nOverall regression metrics (flattened across all expressions):')
    print(f"  - MSE: {overall['mse']:.8f}")
    print(f"  - RMSE: {overall['rmse']:.8f}")
    print(f"  - MAE: {overall['mae']:.8f}")
    print(f"  - R2: {overall['r2']:.8f}")
    print(f"  - Explained Variance: {overall['explained_variance']:.8f}")
    print(f"  - Samples: {overall['samples']}")

    print('\nPer-expression regression metrics:')
    print(metrics_df.to_string(index=False, float_format='%.8f'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the LSTM regression model on a CSV dataset and create metric graphs.'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('weights/checkpoints/best_model.pt'),
        help='Path to the model checkpoint file.',
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=Path('data/processed/val_pruned.csv'),
        help='Path to the CSV dataset to evaluate.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size used during evaluation.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/evaluation'),
        help='Directory where evaluation plots and summaries are saved.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    if not args.csv.exists():
        raise FileNotFoundError(f'CSV file not found: {args.csv}')

    model, config = load_checkpoint(args.checkpoint)
    if config['sequence_length'] != DEFAULT_CONFIG['sequence_length']:
        print('Warning: checkpoint sequence_length differs from default; using checkpoint config.')

    dataset = SequenceIterableDataset(
        args.csv,
        config['sequence_length'],
        chunksize=200_000,
        shuffle=False,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    y_true, y_pred = evaluate(model, dataloader, device)
    metrics_df, overall = compute_regression_metrics(y_true, y_pred)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_metric_plots(args.output_dir, metrics_df, overall, y_true, y_pred)
    # Save exact metric values for reproducible reporting
    metrics_csv = args.output_dir / 'per_expression_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False, float_format='%.10f')

    overall_json = args.output_dir / 'overall_metrics.json'
    with overall_json.open('w') as f:
        json.dump(overall, f, indent=2)

    # Copy or export training-curve data if available from training artifacts.
    training_json_path = Path('weights/checkpoints/training_history.json')
    processing_curve_path = Path('data/processed/training_curve.png')
    training_history_csv = args.output_dir / 'training_history.csv'
    training_history_json = args.output_dir / 'training_history.json'
    training_curve_copy = args.output_dir / 'training_curve.png'

    if training_json_path.exists():
        with training_json_path.open() as f:
            training_history = json.load(f)
        with training_history_json.open('w') as f:
            json.dump(training_history, f, indent=2)
        if 'train_loss_history' in training_history and 'val_loss_history' in training_history:
            training_df = pd.DataFrame({
                'epoch': list(range(1, len(training_history['train_loss_history']) + 1)),
                'train_loss': training_history['train_loss_history'],
                'val_loss': training_history['val_loss_history'],
            })
            training_df.to_csv(training_history_csv, index=False, float_format='%.10f')
    if processing_curve_path.exists():
        shutil.copy2(processing_curve_path, training_curve_copy)

    print_metrics(metrics_df, overall, args.csv, args.checkpoint)
    print(f"\nSaved evaluation plots to: {args.output_dir.resolve()}")
    print(f"Saved per-expression metrics CSV to: {metrics_csv.resolve()}")
    print(f"Saved overall metrics JSON to: {overall_json.resolve()}")
    if training_json_path.exists():
        print(f"Saved training history JSON to: {training_history_json.resolve()}")
        if training_history_csv.exists():
            print(f"Saved training history CSV to: {training_history_csv.resolve()}")
    if processing_curve_path.exists():
        print(f"Copied training curve image to: {training_curve_copy.resolve()}")

    # Write README for evaluation outputs
    readme_path = args.output_dir / 'README.md'
    readme_text = f'''# Evaluation Output Summary

This folder contains evaluation graphs and exact metric values generated by `src/evaluate_lstm_csv_only.py`.

## Files

- `per_expression_metrics.csv` - exact per-expression metrics for all values computed by the model.
- `overall_metrics.json` - exact overall aggregated regression metrics.
- `evaluation_per_expression_metrics.png` - per-expression chart with both error metrics (`MSE`, `RMSE`, `MAE`) and score metrics (`R2`, `Explained Variance`).
- `evaluation_errors.png` - error-only per-expression chart (`MSE`, `RMSE`, `MAE`).
- `evaluation_r2_explained_variance.png` - per-expression score chart for `R2` and `Explained Variance`.
- `evaluation_overall_metrics.png` - overall aggregated metrics chart.
- `evaluation_prediction_vs_actual.png` - predicted vs actual scatter plot for a sample of values.
- `predicted_vs_actual_sample.csv` - the exact sample values used to draw the predicted vs actual plot.
- `training_history.csv` - epoch-level training and validation loss values from training.
- `training_history.json` - the same training history in JSON format.
- `training_curve.png` - copy of the training convergence image from `data/processed/training_curve.png`.
'''
    with readme_path.open('w') as f:
        f.write(readme_text)
    print(f"Saved evaluation README to: {readme_path.resolve()}")


if __name__ == '__main__':
    main()
