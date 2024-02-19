import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns


import plotly.express as px
import plotly.graph_objects as go

from non_parametric_ml.pytorch_tools.losses import PinballLoss

import plotly.io as pio

# Set the default template for plotly
pio.templates.default = 'plotly_white'

# Data creation function
def create_data(multimodal: bool):
    X = np.random.uniform(0.3, 10, 1000)
    y = np.log(X) + np.random.exponential(0.1 + X / 20.0)
    if multimodal:
        X = np.concatenate([X, np.random.uniform(5, 10, 500)])
        y = np.concatenate([y, np.random.normal(6.0, 0.3, 500)])
    return torch.tensor(X[..., None]).float(), torch.tensor(y[..., None]).float()

class BinarizedQuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(BinarizedQuantileLoss, self).__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float))
        self.qloss_low = PinballLoss(quantiles[0])
        self.qloss_high = PinballLoss(quantiles[-1])

    @staticmethod
    def find_bin_indices(y_true, q_low, q_high, n_bins):
        range = q_high - q_low
        bin_size = range / n_bins
        bin_indices = ((y_true - q_low) / bin_size).floor().long()
        bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)
        return bin_indices
        
    def forward(self, q_hat, y_true, logits):
        q_low = q_hat[:, 0:1]
        q_high = q_hat[:, -1:]
        q_interm = logits  # Use logits directly for cross-entropy

        bin_labels = self.find_bin_indices(y_true, q_low, q_high, q_interm.size(1)).squeeze()
        ce_loss = F.cross_entropy(logits, bin_labels)
        q_loss = self.qloss_low(q_low, y_true) + self.qloss_high(q_high, y_true)

        return ce_loss + q_loss

# Model definition
class NonCrossQuantReg(pl.LightningModule):
    def __init__(self, din, quantiles, loss_fn = None, name=None):
        super().__init__()
        assert len(quantiles) > 2, "There should be more than 2 quantiles"

        self.din = din
        self.loss_fn = loss_fn if loss_fn is not None else PinballLoss(quantiles)
        # self.scores_fc = nn.Linear(din, len(quantiles) - 1)
        # self.width_fc = nn.Linear(din, 1)
        # self.min_fc = nn.Linear(din, 1)

        self.scores_fc = nn.Sequential(
            nn.Linear(din, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(quantiles) - 1)
        )

        self.width_fc = nn.Sequential(
            nn.Linear(din, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.min_fc = nn.Sequential(
            nn.Linear(din, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.model_name = name if name is not None else self.__class__.__name__

    def forward(self, X):
        min: Tensor = self.min_fc(X)
        width: Tensor = F.softplus(self.width_fc(X))
        logits: Tensor = self.scores_fc(X)
        scores: Tensor = F.softmax(logits, dim=1)
        scores_with_zeros: Tensor = torch.cat((torch.zeros_like(min), scores), dim=-1)

        # Compute proportions by cumulatively summing the scores_tensor
        proportions: Tensor = torch.cumsum(scores_with_zeros, dim=-1)
        
        # Calculate final quantiles
        values: Tensor = min + width * proportions
        return values, logits
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        q_hat, logits = self.forward(X)
        probs = F.softmax(logits, dim=1)
        loss = self.loss_fn(q_hat, y, logits)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)


# Data preparation
multimodal = False
X, y = create_data(multimodal)
dataloader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

# Model instantiation and training
quantiles = np.linspace(0.01, 0.99, 10)
model = NonCrossQuantReg(din=1, quantiles=quantiles, loss_fn=BinarizedQuantileLoss(quantiles))
trainer = pl.Trainer(max_epochs=500)
trainer.fit(model, dataloader)



model.eval()
with torch.no_grad():
    q_hat, logits = model(X)
    q_low = q_hat[:, 0:1].detach()
    q_high = q_hat[:, -1:].detach()
    q_interm = q_hat[:, 1:-1].detach()
    probs = F.softmax(logits, dim=1)
    q_hat = q_hat.numpy()
    logits = logits.numpy()


probs_df = pd.DataFrame(probs, columns = quantiles[1:].round(2))
max_quantile_indices = probs_df.idxmax(axis=1)


qnames = [f'q{q:.2f}' for q in quantiles]

y_hat_df = (pd.DataFrame(q_hat, columns = qnames)
            .assign(
                X = X.numpy(),
                y = y.numpy()
            )
            .sort_values("X")
)

y_hat_df

# Assuming y_hat_df is your DataFrame and it contains columns 'X', 'q0.50', 'q0.95'
fig = px.line(y_hat_df, x='X', y=qnames, title="MQR qloss(low, high) + xentropy loss")
fig.add_trace(go.Scatter(x=y_hat_df['X'], y=y_hat_df['y'], mode='markers', name='True values'))
fig.show()

# Plotting
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=X.numpy().flatten(), y=y.numpy().flatten(), color='blue', label='True values')
# for i, quantile in enumerate(quantiles):
#     sns.lineplot(x=X.numpy().flatten(), y=q_hat[:, i], label=f'q{quantile.round(2)}')

# plt.title('Quantile Regression with Binarized Quantile Loss')
# plt.xlabel('X value')
# plt.ylabel('Y value')
# plt.legend()
# plt.show()

assert True
