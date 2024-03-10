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
from torch.utils.data import random_split
import plotly.io as pio
from non_parametric_ml.utils import set_all_random_seeds

set_all_random_seeds()

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

    needs_logits: bool = True

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

        bin_labels = self.find_bin_indices(y_true, q_low, q_high, logits.size(1)).squeeze()
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
        self.val_loss_fn = PinballLoss(quantiles)
        self.scores_fc = nn.Linear(din, len(quantiles) - 1)
        self.width_fc = nn.Linear(din, 1)
        self.min_fc = nn.Linear(din, 1)

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
        if self.loss_fn.needs_logits:
            loss = self.loss_fn(q_hat, y, logits)
        else:
            loss = self.loss_fn(q_hat, y)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        q_hat, logits = self.forward(X)
        loss = self.val_loss_fn(pred=q_hat, target=y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.LBFGS(self.parameters(), lr = 1, max_iter=50)
        optimizer = Adam(self.parameters(), lr=1e-4)
        return optimizer
        


# Data preparation
multimodal = False
X, y = create_data(multimodal)

dataset = TensorDataset(X, y)
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False) 

dataloader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=200,  # Number of epochs with no improvement after which training will be stopped
        verbose=False,
        mode='min'  # Stops training when the quantity monitored has stopped decreasing
        )

# Define ModelCheckpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',  # Monitor validation loss
    save_top_k=1,
    mode='min',
    dirpath='lightning_checkpoints/',
    filename='best-checkpoint-lbfgs'
)

# logger = TensorBoardLogger("tensorboard", name=model.model_name)

# trainer = pl.Trainer(callbacks=[early_stop_callback], max_epochs=300, logger = logger)

# Model instantiation and training
din=1
quantiles = np.linspace(0.05, 0.95, 100)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# model = NonCrossQuantReg(din=1, quantiles=quantiles, loss_fn=PinballLoss(quantiles))

model = NonCrossQuantReg(din=din, quantiles=quantiles, loss_fn=BinarizedQuantileLoss(quantiles))
trainer = pl.Trainer(max_epochs=2000, callbacks=[early_stop_callback, checkpoint_callback])
trainer.fit(model, train_loader, val_loader)

# model_best = NonCrossQuantReg.load_from_checkpoint(
#     din=din,
#     quantiles=quantiles,
#     loss_fn=BinarizedQuantileLoss(quantiles),
#     checkpoint_path="lightning_checkpoints/best-checkpoint-lbfgs.ckpt").to('cpu')


# model = model_best


model.eval()
with torch.no_grad():
    q_hat, logits = model(X)
    q_low = q_hat[:, 0:1].detach()
    q_high = q_hat[:, -1:].detach()
    q_interm = q_hat[:, 1:-1].detach()
    probs = F.softmax(logits, dim=1)
    q_hat = q_hat.numpy()
    logits = logits.numpy()


probs_df = pd.DataFrame(probs, columns = quantiles[1:].round(3))
max_quantile_indices = probs_df.idxmax(axis=1)


qnames = [f'q{q:.3f}' for q in quantiles]

y_hat_df = (pd.DataFrame(q_hat, columns = qnames)
            .assign(
                X = X.numpy(),
                y = y.numpy()
            )
            .sort_values("X")
)

y_hat_df

assert all(name in y_hat_df.columns for name in qnames), "Some quantiles in qnames do not exist in y_hat_df"
assert not y_hat_df.empty, "y_hat_df is empty"
fig = px.line(y_hat_df, x='X', y=qnames, title="MQR qloss(low, high) + xentropy loss")


# Assuming y_hat_df is your DataFrame and it contains columns 'X', 'q0.50', 'q0.95'
fig = px.line(y_hat_df, x='X', y=['q0.050', 'q0.105', 'q0.505', 'q0.705', 'q0.950'], title="MQR qloss(low, high) + xentropy loss")
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
