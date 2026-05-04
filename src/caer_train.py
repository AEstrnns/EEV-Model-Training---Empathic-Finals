import torch
import torch.nn as nn
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

COLAB_PATH = Path("/content/drive/MyDrive/EMPATHIC_PROJECT/EEV-Model-Training---Empathic-Finals")
DATA_PATH = COLAB_PATH / "data" / "processed" / "train_pruned.csv"
WEIGHTS_PATH = COLAB_PATH / "weights" / "class_weights.json"
CHECKPOINT_PATH = COLAB_PATH / "weights" / "caer_model_turbo.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPRESSION_COLUMNS = [
    'amusement', 'anger', 'awe', 'concentration', 'confusion',
    'contempt', 'contentment', 'disappointment', 'doubt', 
    'elation', 'interest', 'pain', 'sadness', 'surprise', 'triumph'
]

# 1. THE DATASET
class EEVDataset(Dataset):
    def __init__(self, csv_file):
        # Loading 5.1M samples
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        labels = torch.tensor(row[EXPRESSION_COLUMNS].values.astype('float32'))
        
        # Placeholders for face and context stream images
        face_img = torch.randn(3, 224, 224)    
        context_img = torch.randn(3, 224, 224) 
        
        return face_img, context_img, labels

# 2. CAER-NET-RS MODEL
class CAERNetRS(nn.Module):
    def __init__(self):
        super(CAERNetRS, self).__init__()
        self.face_stream = nn.Sequential(
            nn.Conv2d(3, 32, 3), 
            nn.ReLU(), 
            nn.Flatten()
        )
        self.context_stream = nn.Sequential(
            nn.Conv2d(3, 32, 3), 
            nn.ReLU(), 
            nn.Flatten()
        )
        self.fc = nn.Linear(32 * 222 * 222 * 2, 15) 

    def forward(self, face, context):
        f = self.face_stream(face)
        c = self.context_stream(context)
        combined = torch.cat((f, c), dim=1)
        return self.fc(combined)

# 3. LOAD CLASS WEIGHTS
with open(WEIGHTS_PATH, 'r') as f:
    weights_dict = json.load(f)
weights_tensor = torch.tensor([weights_dict[exp] for exp in EXPRESSION_COLUMNS]).to(device)

# 4. MAIN TRAINING EXECUTION
if __name__ == "__main__":
    dataset = EEVDataset(DATA_PATH)
    loader = DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    
    model = CAERNetRS().to(device)
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Resume from checkpoint if it exists
    if CHECKPOINT_PATH.exists():
        print(f"!!! SUCCESS: Restoring progress from: {CHECKPOINT_PATH} !!!")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        print(f"No checkpoint found at {CHECKPOINT_PATH}. Starting fresh pass.")

    print(f"Turbo Mode Initialized. Training on {len(dataset)} samples for 1 Epoch.")

    # 5. TRAINING LOOP
    model.train()
    running_loss = 0.0
    
    for i, (faces, contexts, labels) in enumerate(loader):
        faces, contexts, labels = faces.to(device), contexts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(faces, contexts)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Log every 200 steps
        if (i + 1) % 200 == 0:
            avg_batch_loss = running_loss / 200
            progress = (i / len(loader)) * 100
            print(f"Step [{i+1}/{len(loader)}] | Loss: {avg_batch_loss:.4f} | Progress: {progress:.2f}%")
            running_loss = 0.0

        # Save checkpoint every 2000 steps
        if (i + 1) % 2000 == 0:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"--> Safety Checkpoint saved at step {i+1}")

    COLAB_PATH.joinpath("weights").mkdir(exist_ok=True)
    final_path = COLAB_PATH / "weights" / "caer_model_final_epoch1.pth"
    torch.save(model.state_dict(), final_path)
    
    print(f"Training complete.")