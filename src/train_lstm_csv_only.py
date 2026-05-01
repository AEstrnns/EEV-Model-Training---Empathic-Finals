import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ==========================================
# SATISFIES BULLET 1: Hyperparameters
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
# DATA PREPARATION (Sliding Window)
# ==========================================
def create_sequences_from_csv(csv_path, seq_len):
    print(f"Loading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['video_id', 'timestamp_ms'])
    
    X, Y = [], []
    groups = df.groupby('video_id')
    
    for _, group in tqdm(groups, desc=f"Sequencing {csv_path.name}"):
        values = group[EXPRESSIONS].values
        for i in range(len(values) - seq_len):
            X.append(values[i : i + seq_len])
            Y.append(values[i + seq_len])
            
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

# ==========================================
# TRAINING & VALIDATION LOOP
# ==========================================
def train_model():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    train_path = project_root / "data" / "processed" / "train_pruned.csv"
    val_path = project_root / "data" / "processed" / "val_pruned.csv"
    
    X_train, Y_train = create_sequences_from_csv(train_path, CONFIG["sequence_length"])
    X_val, Y_val = create_sequences_from_csv(val_path, CONFIG["sequence_length"])

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train), batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, Y_val), batch_size=CONFIG["batch_size"], shuffle=False
    )
    
    model = EmotionForecastingLSTM(CONFIG)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    train_loss_history, val_loss_history = [], []
    
    print(f"Starting Training: LR={CONFIG['learning_rate']}, Batch={CONFIG['batch_size']}")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_Y in train_loader:
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

if __name__ == "__main__":
    train_model()