from torchsig.models.iq_models.efficientnet.efficientnet import *
from torchsig.transforms.transforms import (ComplexTo2D, Compose)
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.utils.cm_plotter import plot_confusion_matrix
from sklearn.metrics import classification_report
from pytorch_lightning import LightningModule
from matplotlib import pyplot as plt
from tqdm import tqdm
import click, torch, os, sys
import numpy as np

enets = [efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_b8]


class ProwessNet(LightningModule):

   def __init__(self, model):
      super(ProwessNet, self).__init__()
      self.mdl: torch.nn.Module = model

   def forward(self, x: torch.Tensor):
      return self.mdl(x.float())

   def predict(self, x: torch.Tensor):
      with torch.no_grad():
         return self.forward(x.float())


@click.command()
@click.option("--model", "model_path", help="Path to load model checkpoint to load")
@click.option("--enet", type=int, help="EfficientNet model version to use")
@click.option("--snr", type=int, help="Target signal SNR for testing the dataset")
def main(model_path: str, enet: int, snr: int):
   
   # Validate passed in parameters
   if not model_path:
      print("Model path is required")
      sys.exit(1)
   elif not enet:
      print("EfficientNet version is required")
      sys.exit(1)
   elif not snr: 
      print("Target SNR is required")
      sys.exit(1)

   # List of modulation classes to include in testing
   class_list = ["ook", "bpsk", "4pam", "4ask", "qpsk", "8pam", "8ask", "8psk",
                 "16qam", "16pam", "16ask", "16psk", "32qam", "32qam_cross", "32pam", "32ask", "32psk",
                 "64qam", "64pam", "64ask", "64psk", "128qam_cross", "256qam", "512qam_cross", "1024qam",
                 "2fsk", "2gfsk", "2msk", "2gmsk", "4fsk", "4gfsk", "4msk", "4gmsk", "8fsk", "8gfsk",
                 "8msk", "8gmsk", "16fsk", "16gfsk", "16msk", "16gmsk",
                 "ofdm-64", "ofdm-72", "ofdm-128", "ofdm-180", "ofdm-256", "ofdm-300",
                 "ofdm-512", "ofdm-600", "ofdm-900", "ofdm-1024", "ofdm-1200", "ofdm-2048"]
   num_classes = len(class_list)
   num_validation_samples = num_classes * 10

   # Create the validation dataset
   transform = Compose([ComplexTo2D()])
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
   print("Target SNR to Test: {}".format(snr))
   print("Validation dataset length: {}".format(len(val_dataset)))
   print("Number of classes: {}".format(num_classes))
   print("Data shape: {}".format(val_dataset[0][0].shape))

   # Create the EfficientNet model and move it to the correct testing device
   model = enets[enet](num_classes=num_classes)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   prowess_model = ProwessNet(model)
   prowess_model = prowess_model.to(device)

   # Load model checkpoint
   checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
   prowess_model.load_state_dict(checkpoint["state_dict"])
   prowess_model = prowess_model.to(device=device).eval()

   # Infer results over validation set
   num_test_examples = len(val_dataset)
   y_preds = np.zeros((num_test_examples,))
   y_true = np.zeros((num_test_examples,))

   for i in tqdm(range(0, num_test_examples)):
      data, label = val_dataset[i]
      data = torch.from_numpy(np.expand_dims(data, 0)).float().to(device)
      pred_tmp = prowess_model.predict(data)
      pred_tmp = pred_tmp.cpu().numpy() if torch.cuda.is_available() else pred_tmp
      y_preds[i] = np.argmax(pred_tmp)
      y_true[i] = label

   # Print classification report and plot confusion matrix
   acc = np.sum(np.asarray(y_preds) == np.asarray(y_true)) / len(y_true)
   plot_confusion_matrix(
      y_true,
      y_preds,
      classes=class_list,
      normalize=True,
      title="EfficientNet{} Confusion Matrix\nTotal Accuracy: {:.2f}%".format(enet, acc * 100),
      text=False,
      rotate_x_text=90,
      figsize=(16, 9),
   )
   plt.savefig("{}/classifier_confusion_enet{}.png".format(os.path.dirname(model_path), enet))
   print("Classification Report:")
   print(classification_report(y_true, y_preds))


if __name__ == "__main__":
   main()
