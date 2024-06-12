from torchsig.utils.visualize import ConstellationVisualizer
from torchsig.datasets.modulations import ModulationsDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import click, sys


@click.command()
@click.option("--snr", type=int, help="Target signal SNR for testing the dataset")
@click.option("--outputs", type=int, help="Number of outputs to visualize in a single plot")
def main(snr: int, outputs: int):
   
   # Validate passed in parameters
   if snr is None:
      print("Target SNR is required")
      sys.exit(1)
   if outputs is None:
      outputs = 16

   # List of modulation classes to include in testing
   class_list = ["ook", "bpsk", "4pam", "4ask", "qpsk", "8pam", "8ask", "8psk",
                 "16qam", "16pam", "16ask", "16psk", "32qam", "32qam_cross", "32pam", "32ask", "32psk",
                 "64qam", "64pam", "64ask", "64psk", "128qam_cross", "256qam", "512qam_cross", "1024qam",
                 "2fsk", "2gfsk", "2msk", "2gmsk", "4fsk", "4gfsk", "4msk", "4gmsk", "8fsk", "8gfsk",
                 "8msk", "8gmsk", "16fsk", "16gfsk", "16msk", "16gmsk",
                 "ofdm-64", "ofdm-72", "ofdm-128", "ofdm-180", "ofdm-256", "ofdm-300",
                 "ofdm-512", "ofdm-600", "ofdm-900", "ofdm-1024", "ofdm-1200", "ofdm-2048"]
   num_samples = len(class_list) * outputs

   # Create the signal dataset
   dataset = ModulationsDataset(
      classes=class_list,
      use_class_idx=True,
      level=0,
      num_iq_samples=4096,
      num_samples=num_samples,
      iq_samples_per_symbol=1,
      target_snr=snr,
      transform=None,
      include_snr=False,
      eb_no=False)
   loader = DataLoader(dataset, batch_size=outputs, shuffle=True)

   # Generate the output visualizations
   visualizer = ConstellationVisualizer(
      data_loader=loader,
      visualize_transform=None,
      visualize_target_transform=lambda tensor: [class_list[int(tensor[idx])] for idx in range(tensor.shape[0])])
   for figure in iter(visualizer):
      figure.subplots_adjust(hspace=0.5)
      figure.set_size_inches(14, 9)
      plt.show()
      break


if __name__ == "__main__":
   main()
