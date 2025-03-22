import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import sklearn
import tensorboard
import tqdm
from PIL import Image

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Matplotlib version:", plt.matplotlib.__version__)
print("PyTorch version:", torch.__version__)
print("TorchVision version:", torchvision.__version__)
print("scikit-learn version:", sklearn.__version__)
print("Tensorboard version:", tensorboard.__version__)
print("Tqdm version:", tqdm.__version__)
print("Pillow version:", Image.__version__)

print("\nAll imports successful! Environment is ready for Neural Architecture Search.")

# Test PyTorch GPU availability
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("GPU is not available, using CPU only.") 