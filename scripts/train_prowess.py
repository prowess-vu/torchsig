from torchsig.models.iq_models.efficientnet.efficientnet import *
from torchsig.transforms.target_transforms import DescToClassIndex
from torchsig.transforms.transforms import (ComplexTo2D, Compose)
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.utils.cm_plotter import plot_confusion_matrix
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from typing import Tuple
from torch import optim
from tqdm import tqdm
import click, torch, os, random, sys
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

enets = [efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_b8]


class ProwessNet(LightningModule):

   def __init__(self, model, data_loader, val_data_loader):
      super(ProwessNet, self).__init__()
      self.mdl: torch.nn.Module = model
      self.data_loader: DataLoader = data_loader
      self.val_data_loader: DataLoader = val_data_loader

      # Hyperparameters
      self.lr = 0.001
      self.batch_size = data_loader.batch_size

   def forward(self, x: torch.Tensor):
      return self.mdl(x.float())

   def predict(self, x: torch.Tensor):
      with torch.no_grad():
         return self.forward(x.float())

   def configure_optimizers(self):
      return optim.Adam(self.parameters(), lr=self.lr)

   def train_dataloader(self):
      return self.data_loader

   def val_dataloader(self):
      return self.val_data_loader

   def training_step(self, batch: torch.Tensor, batch_nb: int):
      x, y = batch
      y = torch.squeeze(y.to(torch.int64))
      loss = F.cross_entropy(self(x.float()), y)
      self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
      return loss

   def validation_step(self, batch: torch.Tensor, batch_nb: int):
      x, y = batch
      y = torch.squeeze(y.to(torch.int64))
      loss = F.cross_entropy(self(x.float()), y)
      self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
      return loss


@click.command()
@click.option("--id_seed", default=None, help="Unique integer seed for dataset reproducibility")
@click.option("--enet", default=0, type=int, help="EfficientNet model version to use")
@click.option("--snr", nargs=2, type=int, default=None, help="Target min/max SNR for the training dataset")
def main(id_seed: int, enet: int, snr: Tuple[int, int]):

   # List of modulation classes to include in training
   class_list = ["ook", "bpsk", "4pam", "4ask", "qpsk", "8pam", "8ask", "8psk",
                 "16qam", "16pam", "16ask", "16psk", "32qam", "32qam_cross", "32pam", "32ask", "32psk",
                 "64qam", "64pam", "64ask", "64psk", "128qam_cross", "256qam", "512qam_cross", "1024qam",
                 "2fsk", "2gfsk", "2msk", "2gmsk", "4fsk", "4gfsk", "4msk", "4gmsk", "8fsk", "8gfsk",
                 "8msk", "8gmsk", "16fsk", "16gfsk", "16msk", "16gmsk",
                 "ofdm-64", "ofdm-72", "ofdm-128", "ofdm-180", "ofdm-256", "ofdm-300",
                 "ofdm-512", "ofdm-600", "ofdm-900", "ofdm-1024", "ofdm-1200", "ofdm-2048"]
   num_classes = len(class_list)
   num_samples = num_classes * 100000
   num_validation_samples = num_classes * 1000

   # Seed the dataset instantiation for reproduceability
   pl.seed_everything(id_seed if id_seed is not None else random.randrange(sys.maxsize) % 4294967296, True)

   # Create the dataset
   transform = Compose([ComplexTo2D()])
   train_dataset = ModulationsDataset(
      classes=class_list,
      use_class_idx=True,
      level=0,
      num_iq_samples=4096,
      num_samples=num_samples,
      target_snr=snr,
      transform=transform,
      include_snr=False,
      eb_no=False)
   val_dataset = ModulationsDataset(
      classes=class_list,
      use_class_idx=True,
      level=0,
      num_iq_samples=4096,
      num_samples=num_validation_samples,
      target_snr=snr,
      transform=transform,
      include_snr=False,
      eb_no=False)
   print("EfficientNet Version: {}".format(enet))
   print("Target SNR Range: {}".format(snr if snr is not None else (100, 100)))
   print("Dataset length: {}".format(len(train_dataset)))
   print("Number of classes: {}".format(num_classes))
   print("Data shape: {}".format(train_dataset[0][0].shape))

   # Create training and validation dataloaders
   train_dataloader = DataLoader(
      dataset=train_dataset,
      batch_size=os.cpu_count(),
      num_workers=os.cpu_count() // 2,
      shuffle=True,
      drop_last=True,
      persistent_workers=True)
   val_dataloader = DataLoader(
      dataset=val_dataset,
      batch_size=os.cpu_count(),
      num_workers=os.cpu_count() // 2,
      shuffle=False,
      drop_last=True,
      persistent_workers=True)

   # Create the EfficientNet model and move it to the correct training device
   model = enets[enet](num_classes=num_classes)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   prowess_model = ProwessNet(model, train_dataloader, val_dataloader)
   prowess_model = prowess_model.to(device)

   # Setup checkpoint callbacks
   checkpoint_filename = "{}/checkpoint".format(os.getcwd())
   checkpoint_callback = ModelCheckpoint(
      filename=checkpoint_filename,
      save_top_k=True,
      monitor="val_loss",
      mode="min")

   # Create and fit trainer
   epochs = 500
   trainer = Trainer(max_epochs=epochs, callbacks=checkpoint_callback, devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu", log_every_n_steps=10)
   trainer.fit(prowess_model)

   # Load best checkpoint
   checkpoint = torch.load(checkpoint_filename + ".ckpt", map_location=lambda storage, loc: storage)
   prowess_model.load_state_dict(checkpoint["state_dict"])
   prowess_model = prowess_model.to(device=device).eval()

   # Infer results over validation set
   num_test_examples = len(val_dataset)
   y_preds = np.zeros((num_test_examples,))
   y_true = np.zeros((num_test_examples,))

   for i in tqdm(range(0, num_test_examples)):
      # Retrieve data
      data, label = val_dataset[i]
      # Infer
      data = torch.from_numpy(np.expand_dims(data, 0)).float().to(device)
      pred_tmp = prowess_model.predict(data)
      pred_tmp = pred_tmp.cpu().numpy() if torch.cuda.is_available() else pred_tmp
      # Argmax
      y_preds[i] = np.argmax(pred_tmp)
      # Store label
      y_true[i] = label

   acc = np.sum(np.asarray(y_preds) == np.asarray(y_true)) / len(y_true)
   plot_confusion_matrix(
      y_true,
      y_preds,
      classes=class_list,
      normalize=True,
      title="Example Modulations Confusion Matrix\nTotal Accuracy: {:.2f}%".format(acc * 100),
      text=False,
      rotate_x_text=90,
      figsize=(16, 9),
   )
   plt.savefig("{}/classifier_confusion.png".format(os.getcwd()))

   print("Classification Report:")
   print(classification_report(y_true, y_preds))


if __name__ == "__main__":
   main()
