import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm import tqdm

# ==========================================
# Hyperparameters
# ==========================================
CONFIG = {
    "input_size": 15,        # 15 continuous emotion values
    "hidden_size": 64,       # LSTM capacity
    "num_layers": 1,         
    "output_size": 15,       # Predicting the next 15 values
    "sequence_length": 30,   # 30 frames (5 seconds at 6Hz)
    "learning_rate": 0.001,  
    "batch_size": 256,       
    "epochs": 50,            
    "dropout": 0.2           
}

EXPRESSIONS = [
    'Amusement', 'Anger', 'Awe', 'Concentration', 'Confusion',
    'Contempt', 'Contentment', 'Disappointment', 'Doubt', 
    'Elation', 'Interest', 'Pain', 'Sadness', 'Surprise', 'Triumph'
]

# ==========================================
# LSTM ARCHITECTURE
# ==========================================
class EmotionForecastingLSTM(nn.Module):
    def __init__(self, cfg):
        super(EmotionForecastingLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            batch_first=True,
            dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0
        )
        self.dropout = nn.Dropout(cfg["dropout"])
        self.fc = nn.Linear(cfg["hidden_size"], cfg["output_size"])
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :] 
        dropped = self.dropout(last_timestep)
        out = self.fc(dropped)
        return self.sigmoid(out)

# ==========================================
# DATA PREPARATION
# ==========================================

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
            "Expected video and timestamp columns not found. "
            f"Found columns: {list(columns)}"
        )

    expression_cols = []
    for expr in EXPRESSIONS:
        expr_key = expr.strip().lower()
        if expr_key in normalized:
            expression_cols.append(normalized[expr_key])
        else:
            raise KeyError(
                f"Missing expected emotion column '{expr}' in {csv_name or 'CSV file'}. "
                f"Found columns: {list(columns)}"
            )

    return video_col, time_col, expression_cols


def create_sequences_from_csv(csv_path, seq_len):
    print(f"Loading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    video_col, time_col, expression_cols = detect_csv_columns(df.columns, csv_path.name)
    df = df.sort_values(by=[video_col, time_col])

    X, Y = [], []
    groups = df.groupby(video_col)

    for _, group in tqdm(groups, desc=f"Sequencing {csv_path.name}"):
        values = group[expression_cols].values
        for i in range(len(values) - seq_len):
            X.append(values[i : i + seq_len])
            Y.append(values[i + seq_len])

    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)


class SequenceIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, csv_path, seq_len, chunksize=200_000, shuffle=False, buffer_size=20_000):
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self):
        print(f"Streaming {Path(self.csv_path).name}...")
        buffer_df = None
        video_col = None
        time_col = None
        expression_cols = None
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


# ==========================================
# TRAINING & VALIDATION LOOP
# ==========================================
def train_model():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    train_path = project_root / "data" / "processed" / "train_pruned.csv"
    val_path = project_root / "data" / "processed" / "val_pruned.csv"
    
    train_dataset = SequenceIterableDataset(
        train_path,
        CONFIG["sequence_length"],
        shuffle=True,
        buffer_size=20_000,
    )
    val_dataset = SequenceIterableDataset(val_path, CONFIG["sequence_length"], shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False
    )
    
    model = EmotionForecastingLSTM(CONFIG)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    # ==========================================
    # CHECKPOINT SETUP
    # ==========================================
    checkpoint_dir = project_root / "weights" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    train_loss_history, val_loss_history = [], []
    best_val_loss = float('inf')
    
    print(f"Starting Training: LR={CONFIG['learning_rate']}, Batch={CONFIG['batch_size']}")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_train_loss = 0
        train_batches = 0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            train_batches += 1
            
        avg_train_loss = epoch_train_loss / train_batches if train_batches else float('nan')
        train_loss_history.append(avg_train_loss)
        
        # Validation Phase
        model.eval()
        epoch_val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                predictions = model(batch_X)
                val_loss = criterion(predictions, batch_Y)
                epoch_val_loss += val_loss.item()
                val_batches += 1
                
        avg_val_loss = epoch_val_loss / val_batches if val_batches else float('nan')
        val_loss_history.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
        
        # ==========================================
        # SAVE BEST MODEL CHECKPOINT
        # ==========================================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'config': CONFIG,
            }
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save(best_checkpoint, checkpoint_path)
            print(f"  → Saved best model to {checkpoint_path}")
        
        
    # ==========================================
    # Training Graphs
    # ==========================================
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label="Training Loss", color="blue", linewidth=2)
    plt.plot(val_loss_history, label="Validation Loss", color="orange", linewidth=2, linestyle="--")
    plt.title("LSTM Autoregressive Training Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True)
    
    out_path = project_root / "data" / "processed" / "training_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"\nGraph saved to {out_path}")
    
    # ==========================================
    # SAVE FINAL MODEL & TRAINING METADATA
    # ==========================================
    final_checkpoint = {
        'epoch': CONFIG['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'best_val_loss': best_val_loss,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'config': CONFIG,
    }
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")
    
    # Save training history as JSON for logging
    history_path = checkpoint_dir / "training_history.json"
    history_data = {
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'best_val_loss': float(best_val_loss),
        'config': CONFIG,
    }
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    train_model()