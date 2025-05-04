import os
import pandas as pd
import numpy as np
from typing import Tuple, Union

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from medaidml import RESULTS_DIR
from medaidml.utils import split_val_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Feature extraction with linear interpolation
def get_features(spectrum_data: Union[str, pd.DataFrame],
                 interp_len: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if isinstance(spectrum_data, str):
        df = pd.read_csv(spectrum_data)
    else:
        df = spectrum_data

    if 'sid' not in df.columns:
        df['sdiff']  = df['freq'] < df['freq'].shift(1, fill_value=0)
        df['sdiff'] = df['sdiff'].astype(int)
        df['sid'] = df['sdiff'].cumsum()

    features_interp, labels, languages, sources = [], [], [], []
    for _, group in tqdm(df.groupby('sid')):
        freqs = group['freq'].values
        features = group['power'].values
        new_freq = np.linspace(0, 0.5, interp_len)
        new_feat = np.interp(new_freq, freqs, features)
        features_interp.append(new_feat)
        labels.append(group['label'].values[0])
        languages.append(group['language'].values[0])
        sources.append(group['source'].values[0])

    return np.array(features_interp), np.array(labels), np.array(languages), np.array(sources)


# Custom Dataset class
class FFTDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Training function
def train_nn(train_loader, val_loader, input_dim, num_classes, seed: int, epochs=20, lr=1e-3):
    torch.manual_seed(seed)
    model = SimpleNN(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation accuracy (optional)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)
        acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} - Val Accuracy: {acc:.4f}")
    return model


# Batched evaluation
def evaluate_nn(model, fft_data: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    x, y, languages, sources = get_features(fft_data)
    dataset = FFTDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    return pd.DataFrame({
        'Ground Truth': y,
        'Prediction': all_preds,
        'language': languages,
        'source': sources
    })


if __name__ == "__main__":
    BATCH_SIZE = 64
    
    fft_dir = os.path.join(RESULTS_DIR, "fourier_gpt")
    in_dir = os.path.join(fft_dir, "fft_transformed")

    train_file = os.path.join(in_dir, "fft_train.csv")
    test_file = os.path.join(in_dir, "fft_test.csv")

    no_dataleak_df = pd.read_csv(test_file)
    all_train_df = pd.read_csv(train_file)

    for i in range(1, 6):
        print(f"\n--- Seed {i} ---")
        train_df, val_df, test_df = split_val_test(all_train_df, seed=i)

        x_train, y_train, _, _ = get_features(train_df)
        x_val, y_val, _, _ = get_features(val_df)

        input_dim = x_train.shape[1]
        num_classes = len(np.unique(y_train))

        train_dataset = FFTDataset(x_train, y_train)
        val_dataset = FFTDataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        model = train_nn(train_loader, val_loader, input_dim, num_classes, seed=i)

        results_val = evaluate_nn(model, val_df)
        results_test = evaluate_nn(model, test_df)
        results_no_dataleak = evaluate_nn(model, no_dataleak_df)

        print(f"Validation Accuracy: {accuracy_score(results_val['Ground Truth'], results_val['Prediction']):.4f}")
        print(f"Test Accuracy: {accuracy_score(results_test['Ground Truth'], results_test['Prediction']):.4f}")
        print(f"No Data Leak Accuracy: {accuracy_score(results_no_dataleak['Ground Truth'], results_no_dataleak['Prediction']):.4f}")
        print("-" * 30)

        out_dir = os.path.join(fft_dir, str(i))
        os.makedirs(out_dir, exist_ok=True)
        results_test.to_csv(os.path.join(out_dir, "results_test.csv"), index=False)
        results_no_dataleak.to_csv(os.path.join(out_dir, "results_no_dataleak.csv"), index=False)

    print("Neural network classification completed. Results saved.")
