import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class LSTMVolatilityModel(nn.Module):
    """LSTM Neural Network for Volatility Sequence Prediction."""

    def __init__(self, input_size=1, hidden_layer_size=64, num_layers=2):
        super(LSTMVolatilityModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


def prepare_sequences(data: np.ndarray, seq_length: int = 20):
    """Creates rolling sequence windows for LSTM training."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : (i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train_lstm_volatility(
    returns: pd.Series, seq_length: int = 20, epochs: int = 30
):
    """Trains an LSTM model on absolute/squared returns as proxies for volatility."""
    vol_proxy = np.abs(returns.values).reshape(-1, 1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(vol_proxy)

    X, y = prepare_sequences(scaled_data, seq_length)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = LSTMVolatilityModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    # Inverse transform to get original scale
    predicted_vol = scaler.inverse_transform(predictions).flatten()

    return model, predicted_vol
