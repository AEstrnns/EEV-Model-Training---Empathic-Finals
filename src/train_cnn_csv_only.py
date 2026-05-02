import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ==========================================
# SATISFIES BULLET 1: Hyperparameters
# ==========================================
CONFIG = {
    "input_channels": 15,    
    "sequence_length": 30,   
    "output_size": 15,       
    "learning_rate": 0.001,  
    "batch_size": 256,       
    "epochs": 50,            
    "dropout": 0.3           
}

EXPRESSIONS = [
    'amusement', 'anger', 'awe', 'concentration', 'confusion',
    'contempt', 'contentment', 'disappointment', 'doubt', 
    'elation', 'interest', 'pain', 'sadness', 'surprise', 'triumph'
]

# ==========================================
# CNN ARCHITECTURE
# ==========================================
class EmotionForecastingCNN(nn.Module):
    def __init__(self, cfg):
        super(EmotionForecastingCNN, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=cfg["input_channels"], out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.flatten = nn.Flatten()
        
        self.fc_block = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(64 * 7, 128),  
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(128, cfg["output_size"]),
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = x.transpose(1, 2) 
        features = self.conv_block(x)
        flat = self.flatten(features)
        out = self.fc_block(flat)
        return out

# ==========================================
# DATA PREPARATION (The Memory-Saver Fix)
# ==========================================
class EEVSequenceDataset(Dataset):
    def __init__(self, csv_path, seq_len):
        print(f"Loading {Path(csv_path).name} into RAM (Lazy Loading enabled)...")
        self.df = pd.read_csv(csv_path)
        
        # Format columns perfectly
        self.df.columns = self.df.columns.str.strip()
        vid_col = next((col for col in self.df.columns if col.lower() in ['video_id', 'youtube_id', 'id']), 'YouTube ID')
        time_col = next((col for col in self.df.columns if col.lower() in ['timestamp_ms', 'timestamp']), 'Timestamp (milliseconds)')
        
        # Sort and reset index so the row numbers are perfectly sequential
        self.df = self.df.sort_values(by=[vid_col, time_col]).reset_index(drop=True)
        
        # Convert just the numbers to float32 immediately to save memory (~300MB instead of 17GB)
        self.features = self.df[EXPRESSIONS].values.astype(np.float32)
        self.seq_len = seq_len
        
        # Find valid starting rows (so a sequence doesn't accidentally bleed from one video into the next)
        self.valid_indices = []
        grouped = self.df.groupby(vid_col)
        
        for _, group in tqdm(grouped, desc=f"Indexing {Path(csv_path).name}"):
            start_idx = group.index[0]
            end_idx = group.index[-1]
            if (end_idx - start_idx + 1) > self.seq_len:
                self.valid_indices.extend(range(start_idx, end_idx - self.seq_len + 1))
                
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # This only builds the sequence at the EXACT moment the CNN asks for it
        start_row = self.valid_indices[idx]
        X = self.features[start_row : start_row + self.seq_len]
        Y = self.features[start_row + self.seq_len]
        return torch.tensor(X), torch.tensor(Y)

# ==========================================
# TRAINING & VALIDATION LOOP
# ==========================================
def train_model():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    train_path = project_root / "data" / "processed" / "train_pruned.csv"
    val_path = project_root / "data" / "processed" / "val_pruned.csv"
    
    # Using our new memory-saving Dataset class!
    train_dataset = EEVSequenceDataset(train_path, CONFIG["sequence_length"])
    val_dataset = EEVSequenceDataset(val_path, CONFIG["sequence_length"])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    model = EmotionForecastingCNN(CONFIG)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    train_loss_history, val_loss_history = [], []
    
    print(f"\nStarting CNN Training: LR={CONFIG['learning_rate']}, Batch={CONFIG['batch_size']}, Dropout={CONFIG['dropout']}")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_Y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} Training", leave=False):
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        # Validation Phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                predictions = model(batch_X)
                val_loss = criterion(predictions, batch_Y)
                epoch_val_loss += val_loss.item()
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
        
    # ==========================================
    # SATISFIES BULLET 5: Training Graphs
    # ==========================================
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label="CNN Training Loss", color="green", linewidth=2)
    plt.plot(val_loss_history, label="CNN Validation Loss", color="purple", linewidth=2, linestyle="--")
    plt.title("CNN Autoregressive Training Convergence")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True)
    
    out_path = project_root / "data" / "processed" / "training_curve_cnn.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"\nCNN Graph saved to {out_path}")

if __name__ == "__main__":
    train_model()