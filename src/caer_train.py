import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 1. ARCHITECTURE 
class CAERNetRS(nn.Module):
    def __init__(self):
        super(CAERNetRS, self).__init__()
        self.face_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )
        self.context_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(3154176, 15)

    def forward(self, face, context):
        f = self.face_stream(face)
        c = self.context_stream(context)
        return self.fc(torch.cat((f, c), dim=1))

# 2. DATASET SETUP
class EEVDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.cols = ['amusement', 'anger', 'awe', 'concentration', 'confusion',
                     'contempt', 'contentment', 'disappointment', 'doubt',
                     'elation', 'interest', 'pain', 'sadness', 'surprise', 'triumph']
        for col in self.cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        labels = torch.tensor(self.df.iloc[idx][self.cols].values.astype(np.float32))
        return image, image, labels

# 3. UPDATED PATHS AND PARAMETERS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_PATH = '/content/drive/MyDrive/EMPATHIC_PROJECT/EEV-Model-Training---Empathic-Finals/'

TRAIN_CSV = os.path.join(BASE_PATH, 'data/processed/train_pruned.csv')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'src/caer_checkpoint.pth')
FINAL_MODEL_PATH = os.path.join(BASE_PATH, 'src/caer_model.pth')

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

BATCH_SIZE = 64
LR = 0.001
SAVE_EVERY_X_BATCHES = 500

def train():
    model = CAERNetRS().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    start_epoch = 0
    start_batch = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"--- Restoring from checkpoint: {CHECKPOINT_PATH} ---")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']
        print(f"Resuming at Epoch {start_epoch}, Batch {start_batch}")

    dataset = EEVDataset(TRAIN_CSV)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(start_epoch, 1):
        pbar = tqdm(loader, initial=start_batch, total=len(loader))

        for i, (faces, contexts, labels) in enumerate(pbar):
            if i < start_batch: continue

            faces, contexts, labels = faces.to(DEVICE), contexts.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(faces, contexts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

            if (i + 1) % SAVE_EVERY_X_BATCHES == 0:
                torch.save({
                    'epoch': epoch,
                    'batch': i + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, CHECKPOINT_PATH)

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"--- Training Complete. Final model saved in src as {FINAL_MODEL_PATH} ---")

if __name__ == "__main__":
    train()
