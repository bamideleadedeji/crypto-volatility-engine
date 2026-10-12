import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Enforce single-threaded CPU execution to eliminate thread overhead on Render
torch.set_num_threads(1)


class LightweightLSTMVolatilityModel(nn.Module):
    """Memory-Optimized LSTM for Render Free Tier (512MB RAM)."""

    def __init__(self, input_size=1, hidden_layer_size=16, num_layers=1):
        super(LightweightLSTMVolatilityModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


def prepare_sequences(data: np.ndarray, seq_length: int = 15):
    """Creates lightweight rolling sequence windows for LSTM training."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : (i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm_volatility(
    returns: pd.Series, seq_length: int = 15, epochs: int = 15
):
    """
    Trains a lightweight LSTM model on absolute returns.
    
    NOTE: Removed @st.cache_resource to prevent PyTorch model objects 
    and backprop tensors from leaking RAM across Streamlit re-runs.
    """
    vol_proxy = np.abs(returns.values).astype(np.float32).reshape(-1, 1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(vol_proxy).astype(np.float32)

    X, y = prepare_sequences(scaled_data, seq_length)

    if len(X) < 10:
        # Fallback if time series is too short
        return None, vol_proxy.flatten()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = LightweightLSTMVolatilityModel(hidden_layer_size=16, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    # Inverse transform to original scale
    predicted_vol = scaler.inverse_transform(predictions).flatten().astype(np.float32)

    # Clean up PyTorch memory allocations explicitly
    del X_tensor, y_tensor, model, optimizer, criterion
    gc.collect()

    return None, predicted_vol
