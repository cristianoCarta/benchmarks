import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import h5py
from PIL import Image
from src.benchmarkers import *
from src.benchmarkersV2 import *
from tqdm import tqdm
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from io import BytesIO

plt.title("Resnet imagenette classification benchmark (500x500)")
file_memory_arrow = np.load(r"C:\Users\Cristiano Lavoro\Desktop\benchmarks\imagenette\imagenette2\results\resnet\file_memory_arrow.npy")
stream_memory_arrow = np.load(r"C:\Users\Cristiano Lavoro\Desktop\benchmarks\imagenette\imagenette2\results\resnet\stream_memory_arrow.npy")
file_no_memory_arrow = np.load(r"C:\Users\Cristiano Lavoro\Desktop\benchmarks\imagenette\imagenette2\results\resnet\file_no_memory_arrow.npy")
stream_no_memory_arrow = np.load(r"C:\Users\Cristiano Lavoro\Desktop\benchmarks\imagenette\imagenette2\results\resnet\stream_no_memory_arrow.npy")
parquet = np.load(r"C:\Users\Cristiano Lavoro\Desktop\benchmarks\imagenette\imagenette2\results\resnet\parquet.npy")
hdf5_core = np.load(r"C:\Users\Cristiano Lavoro\Desktop\benchmarks\imagenette\imagenette2\results\resnet\hdf5_core.npy")
hdf5_sec2 = np.load(r"C:\Users\Cristiano Lavoro\Desktop\benchmarks\imagenette\imagenette2\results\resnet\hdf5_sec2.npy")

means = [file_memory_arrow[0],stream_memory_arrow[0],file_no_memory_arrow[0],stream_no_memory_arrow[0],parquet[0],hdf5_core[0],hdf5_sec2[0]]
deviations = [file_memory_arrow[1],stream_memory_arrow[1],file_no_memory_arrow[1],stream_no_memory_arrow[1],parquet[1],hdf5_core[1],hdf5_sec2[1]]
measure_names = ["file_memory","stream_memory","file_no_memory","stream_no_memory","parquet","hdf5_core","hdf5_sec2"]

fig, ax = plt.subplots(figsize=(14, 10))


ax.bar(measure_names, means, yerr=deviations, capsize=5, color='skyblue', alpha=0.7)

# Aggiungere titoli e etichette
ax.set_ylabel('mean inference time (s)')
ax.set_xlabel('backend types')
plt.xticks(rotation=45, ha='right')


# Show the plot
plt.savefig(f"imagenette/imagenette2/results/resnet/resnet_benchmark_500.pdf")  # Save as PDF