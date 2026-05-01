import torch
import torch.nn as nn
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_pruned.csv"
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "class_weights.json"

EXPRESSION_COLUMNS = [
    'amusement', 'anger', 'awe', 'concentration', 'confusion',
    'contempt', 'contentment', 'disappointment', 'doubt', 
    'elation', 'interest', 'pain', 'sadness', 'surprise', 'triumph'
]

# 1. THE DATASET BRIDGE
class EEVDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        labels = torch.tensor(row[EXPRESSION_COLUMNS].values.astype('float32'))
        
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
weights_tensor = torch.tensor([weights_dict[exp] for exp in EXPRESSION_COLUMNS])

if __name__ == "__main__":
    # 4. STARTING THE TRAINING SETUP
    dataset = EEVDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = CAERNetRS()
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Ready! Training CAER-Net-RS on {len(dataset)} pruned samples.")

    # 5. FINE-TUNING THE LEARNING RATE
    learning_rates = [0.01, 0.001, 0.0001]
    num_steps_to_test = 500

    for lr in learning_rates:
        print(f"\n--- Testing Learning Rate: {lr} ---")
        model = CAERNetRS() # Reset the model for each test
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        for i, (faces, contexts, labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(faces, contexts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f"LR: {lr} | Step [{i+1}/{num_steps_to_test}] | Loss: {loss.item():.4f}")
            
            if (i + 1) >= num_steps_to_test:
                break

print("\nFine-tuning test complete.")