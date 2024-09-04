import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import h5py
from src.benchmarkers import *
from src.benchmarkersV2 import *
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
np.random.seed(0)

N = [10,100,200,300,500]
dimensions = [32,64,125,192]
selected_label = 10
iterations = 100

generator = Generator()

for item in N:
    for dim in tqdm(dimensions):
        generator.create_dataset(item,f"outputs/v2/{dim}/ds_{item}",dim)
        generator.create_arrow_file(f"outputs/v2/{dim}/ds_{item}")
        generator.create_arrow_stream(f"outputs/v2/{dim}/ds_{item}")