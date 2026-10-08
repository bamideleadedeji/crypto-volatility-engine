import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class TransformerVolatilityModel(nn.Module):
    """Transformer Model with Multi-Head Self Attention for Time Series Volatility."""

    def __init__(
        self,
        input_size=1,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
    ):
        super(TransformerVolatilityModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        transformer_out = self.transformer_encoder(x)
        # Pooling: take the representation of the last token
        out = self.fc_out(transformer_out[:, -1, :])
        return out


def train_transformer_volatility(
    returns: pd.Series, seq_length: int = 20, epochs: int = 30
):
    """Trains a Transformer model on absolute returns to forecast daily volatility."""
    vol_proxy = np.abs(returns.values).reshape(-1, 1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(vol_proxy)

    # Re-use sequence preparation logic
    xs, ys = [], []
    for i in range(len(scaled_data) - seq_length):
        xs.append(scaled_data[i : (i + seq_length)])
        ys.append(scaled_data[i + seq_length])

    X_tensor = torch.tensor(np.array(xs), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(ys), dtype=torch.float32)

    model = TransformerVolatilityModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    predicted_vol = scaler.inverse_transform(predictions).flatten()
    return model, predicted_vol
