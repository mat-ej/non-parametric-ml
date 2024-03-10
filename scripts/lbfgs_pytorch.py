import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
pl.seed_everything(42)

class RandomDataset(Dataset):
    def __init__(self, size, num_samples):
        self.len = num_samples
        self.data = torch.randn(num_samples, size)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.len

class BoringModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)
    def forward(self, x):
        return self.layer(x)
    def loss(self, batch, prediction):
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))
    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}
    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        print("loss:", loss.item())
        return training_step_outputs
    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.LBFGS(self.parameters(), lr=0.01, max_iter=20)
        return optimizer

def main():
    ds = RandomDataset(32, 100000)
    dl = DataLoader(ds, batch_size=1024)
    model = BoringModel()
    trainer = pl.Trainer(
    )
    trainer.fit(model, dl)

if __name__ == "__main__":
    main()