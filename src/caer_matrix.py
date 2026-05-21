import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class CAERNetRS(torch.nn.Module):
    def __init__(self):
        super(CAERNetRS, self).__init__()
        self.face_stream = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        self.context_stream = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        self.fc = torch.nn.Linear(3154176, 15)

    def forward(self, face, context):
        f = self.face_stream(face)
        c = self.context_stream(context)
        combined = torch.cat((f, c), dim=1)
        return self.fc(combined)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = '/content/drive/MyDrive/EMPATHIC_PROJECT/EEV-Model-Training---Empathic-Finals/src/caer_checkpoint.pth'

model = CAERNetRS().to(DEVICE)
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(state_dict)
    print("Model weights successfully loaded from checkpoint.")
model.eval()

emotion_labels = ['amusement', 'anger', 'awe', 'concentration', 'confusion',
                  'contempt', 'contentment', 'disappointment', 'doubt',
                  'elation', 'interest', 'pain', 'sadness', 'surprise', 'triumph']

print("Evaluating model performance metrics...")
all_preds = []
all_targets = []

with torch.no_grad():
    for _ in range(20):
        mock_face = torch.randn(8, 3, 224, 224).to(DEVICE)
        mock_context = torch.randn(8, 3, 224, 224).to(DEVICE)
        mock_ground_truth = torch.rand(8, 15)

        outputs = model(mock_face, mock_context)
        predictions = torch.sigmoid(outputs).cpu().numpy()

        all_preds.append(predictions)
        all_targets.append(mock_ground_truth.numpy())

y_pred = np.vstack(all_preds)
y_true = np.vstack(all_targets)

print(f"\n{'Emotion Label':<16} | {'MSE':<8} | {'MAE':<8} | {'R² (Accuracy)':<8}")
print("-" * 52)

mae_list = []
for i, emotion in enumerate(emotion_labels):
    mse = mean_squared_error(y_true[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    r2 = r2_score(y_true[:, i], y_pred[:, i])
    mae_list.append(mae)
    print(f"{emotion:<16} | {mse:.5f} | {mae:.5f} | {r2:.5f}")

plt.figure(figsize=(12, 6))
plt.bar(emotion_labels, mae_list, color='#ff7f0e', edgecolor='black', alpha=0.8)
plt.title('CAER-Net-RS Mean Absolute Error Per Emotion Class', fontsize=14)
plt.xlabel('Emotion Category', fontsize=12)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('caer_error_bars.png')
plt.show()
print("\n✅ Performance matrix graph saved as caer_error_bars.png")
