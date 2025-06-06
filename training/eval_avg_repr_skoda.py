import torch
import torch as t
from model import Siamese, load_model, get_custom_CNN, jit_load, get_parametrized_model
from torchvision.transforms import Resize
from utils import get_shift, plot_samples, plot_displacement
import numpy as np
from scipy import interpolate
from torchvision.io import read_image
from matplotlib import pyplot as plt


device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")


# Set this according to your input
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 320
MODEL_PATH = "/home/zdeeno/Documents/Work/vtg/src/sensors/backends/siamese/model_tiny.pt"
IMG_PATH = "//home/zdeeno/.ros/stromovka_vtg_test/"
IMG_NAMES = ["strom_map.jpg", "strom_wet.jpg", "strom_gen_fail.jpg"]
# -------------------------------


WIDTH = 512  # - 8
PAD = 31
FRACTION = 8
OUTPUT_SIZE = WIDTH//FRACTION
CROP_SIZE = WIDTH - FRACTION
LAYER_POOL = False
FILTER_SIZE = 3
EMB_CHANNELS = 16
RESIDUALS = 0

size_frac = WIDTH / IMAGE_WIDTH
transform = Resize(int(IMAGE_HEIGHT * size_frac))
fraction_resized = int(FRACTION/size_frac)


def run_demo():

    model = get_parametrized_model(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS, RESIDUALS, PAD, device)
    model = load_model(model, MODEL_PATH)
    histograms = []

    model.eval()
    with torch.no_grad():
        for img_name in IMG_NAMES:
            source_img = transform(read_image(IMG_PATH + img_name) / 255.0).to(device)
            target_repr = np.load("/home/zdeeno/.ros/stromovka_vtg_out/1000.0.npy", allow_pickle=True).item(0)["representation"]
            target_repr = t.tensor(target_repr).to(device).unsqueeze(0)
            source_repr = model.get_repr(source_img.unsqueeze(0))
            histogram = model.conv_repr(source_repr, target_repr)
            # histogram = (histogram - t.mean(histogram)) / t.std(histogram)
            histogram = histogram[0, 0]
            # histogram = t.softmax(histogram, dim=1)

            # visualize:
            shift_hist = histogram.cpu()
            f = interpolate.interp1d(np.linspace(0, IMAGE_WIDTH - fraction_resized, 65), shift_hist, kind="cubic")
            interpolated = f(np.arange(IMAGE_WIDTH - 16))
            histograms.append(interpolated)
        # print(histograms)
        plt.figure()
        plt.title("Response to different conditions")
        for hist in histograms:
            print(hist.shape)
            # hist = hist[0, 193:437]
            hist = hist[0]
            plt.plot(np.linspace(-256, 256, np.size(hist)), hist)
        # plt.grid()
        plt.xlabel("Displacement [px]")
        plt.ylabel("Likelihood [-]")
        plt.legend(["Map", "Minor struct. change", "Major struct. change"])
        plt.show()

if __name__ == '__main__':
    run_demo()
