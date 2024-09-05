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



############ PLOT ROW - WISE #################

t_load_arrow_file_memory = []
t_load_arrow_stream_memory = []
t_load_arrow_file_no_memory = []
t_load_arrow_stream_no_memory = []
t_load_arrow_parquet = []
t_load_hdf5_core = []
t_load_hdf5_sec2 = []

t_access_arrow_file_memory = []
t_access_arrow_stream_memory = []
t_access_arrow_file_no_memory = []
t_access_arrow_stream_no_memory = []
t_access_arrow_parquet = []
t_access_hdf5_core = []
t_access_hdf5_sec2 = []

t_manipulate_arrow_file_memory = []
t_manipulate_arrow_stream_memory = []
t_manipulate_arrow_file_no_memory = []
t_manipulate_arrow_stream_no_memory = []
t_manipulate_arrow_parquet = []
t_manipulate_hdf5_core = []
t_manipulate_hdf5_sec2 = []


for dim in dimensions:
    t_load_arrow_file_memory.append(np.load(f"results/v2/row_wise/{dim}/t_load_arrow_file_memory.npy"))
    t_load_arrow_stream_memory.append(np.load(f"results/v2/row_wise/{dim}/t_load_arrow_stream_memory.npy"))
    t_load_arrow_file_no_memory.append(np.load(f"results/v2/row_wise/{dim}/t_load_arrow_file_no_memory.npy"))
    t_load_arrow_stream_no_memory.append(np.load(f"results/v2/row_wise/{dim}/t_load_arrow_stream_no_memory.npy"))
    t_load_arrow_parquet.append(np.load(f"results/v2/row_wise/{dim}/t_load_arrow_parquet.npy"))
    t_load_hdf5_core.append(np.load(f"results/v2/row_wise/{dim}/t_load_hdf5_core.npy"))
    t_load_hdf5_sec2.append(np.load(f"results/v2/row_wise/{dim}/t_load_hdf5_sec2.npy"))

    t_access_arrow_file_memory.append(np.load(f"results/v2/row_wise/{dim}/t_access_arrow_file_memory.npy"))
    t_access_arrow_stream_memory.append(np.load(f"results/v2/row_wise/{dim}/t_access_arrow_stream_memory.npy"))
    t_access_arrow_file_no_memory.append(np.load(f"results/v2/row_wise/{dim}/t_access_arrow_file_no_memory.npy"))
    t_access_arrow_stream_no_memory.append(np.load(f"results/v2/row_wise/{dim}/t_access_arrow_stream_no_memory.npy"))
    t_access_arrow_parquet.append(np.load(f"results/v2/row_wise/{dim}/t_access_arrow_parquet.npy"))
    t_access_hdf5_core.append(np.load(f"results/v2/row_wise/{dim}/t_access_hdf5_core.npy"))
    t_access_hdf5_sec2.append(np.load(f"results/v2/row_wise/{dim}/t_access_hdf5_sec2.npy"))

    t_manipulate_arrow_file_memory.append(np.load(f"results/v2/row_wise/{dim}/t_manipulate_arrow_file_memory.npy"))
    t_manipulate_arrow_stream_memory.append(np.load(f"results/v2/row_wise/{dim}/t_manipulate_arrow_stream_memory.npy"))
    t_manipulate_arrow_file_no_memory.append(np.load(f"results/v2/row_wise/{dim}/t_manipulate_arrow_file_no_memory.npy"))
    t_manipulate_arrow_stream_no_memory.append(np.load(f"results/v2/row_wise/{dim}/t_manipulate_arrow_stream_no_memory.npy"))
    t_manipulate_arrow_parquet.append(np.load(f"results/v2/row_wise/{dim}/t_manipulate_arrow_parquet.npy"))
    t_manipulate_hdf5_core.append(np.load(f"results/v2/row_wise/{dim}/t_manipulate_hdf5_core.npy"))
    t_manipulate_hdf5_sec2.append(np.load(f"results/v2/row_wise/{dim}/t_manipulate_hdf5_sec2.npy"))

N_label = [str(item) for item in N]
d_label = [str(item) for item in dimensions]

################# LOAD ##################

data1 = np.array(np.array(t_load_hdf5_core)/np.array(t_load_arrow_stream_no_memory))
data2 = np.array(np.array(t_load_hdf5_sec2)/np.array(t_load_arrow_stream_memory))
data3 = np.array(np.array(t_load_hdf5_core)/np.array(t_load_arrow_parquet))
data4 = np.array(np.array(t_load_hdf5_core)/np.array(t_load_arrow_stream_memory))

data1 = np.where(data1 >= 1, 1, 0)
data2 = np.where(data2 >= 1, 1, 0)
data3 = np.where(data3 >= 1, 1, 0)
data4 = np.where(data4 >= 1, 1, 0)

fig, ax = plt.subplots(1, 4, figsize=(14, 6))
plt.suptitle("Loading benchmark")

vmin = min(data1.min(), data2.min(),data3.min(),data4.min())
vmax = max(data1.max(),  data2.max(),data3.max(),data4.max())
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# First heatmap
cax1 = ax[0].imshow(data1, cmap='inferno', interpolation='gaussian')
ax[0].set_title('core / stream_no_memory_map')
fig.colorbar(cax1, ax=ax[0], orientation='vertical')
ax[0].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[0].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Second heatmap
cax2 = ax[1].imshow(data2, cmap='inferno', interpolation='gaussian')
ax[1].set_title('sec2 / stream_memory_map')
fig.colorbar(cax2, ax=ax[1], orientation='vertical')
ax[1].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[1].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Third heatmap
cax3 = ax[2].imshow(data3, cmap='inferno', interpolation='gaussian')
ax[2].set_title('core / parquet')
fig.colorbar(cax3, ax=ax[2], orientation='vertical')
ax[2].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[2].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Third heatmap
cax3 = ax[3].imshow(data4, cmap='inferno', interpolation='gaussian')
ax[3].set_title('core / stream_memory_map')
fig.colorbar(cax3, ax=ax[3], orientation='vertical')
ax[3].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[3].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.savefig(f"results/v2/row_wise/load_heatmap.pdf")  # Save as PDF
plt.clf()

################# ACCESS ##################

data1 = np.array(np.array(t_access_hdf5_core)/np.array(t_access_arrow_stream_no_memory))
data2 = np.array(np.array(t_access_hdf5_sec2)/np.array(t_access_arrow_stream_memory))
data3 = np.array(np.array(t_access_hdf5_core)/np.array(t_access_arrow_parquet))
data4 = np.array(np.array(t_access_hdf5_core)/np.array(t_access_arrow_stream_memory))

data1 = np.where(data1 >= 1, 1, 0)
data2 = np.where(data2 >= 1, 1, 0)
data3 = np.where(data3 >= 1, 1, 0)
data4 = np.where(data4 >= 1, 1, 0)

fig, ax = plt.subplots(1, 4, figsize=(14, 6))
plt.suptitle("Access benchmark")

vmin = min(data1.min(), data2.min(),data3.min(),data4.min())
vmax = max(data1.max(),  data2.max(),data3.max(),data4.max())
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# First heatmap
cax1 = ax[0].imshow(data1, cmap='inferno', interpolation='gaussian')
ax[0].set_title('core / stream_no_memory_map')
fig.colorbar(cax1, ax=ax[0], orientation='vertical')
ax[0].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[0].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Second heatmap
cax2 = ax[1].imshow(data2, cmap='inferno', interpolation='gaussian')
ax[1].set_title('sec2 / stream_memory_map')
fig.colorbar(cax2, ax=ax[1], orientation='vertical')
ax[1].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[1].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Third heatmap
cax3 = ax[2].imshow(data3, cmap='inferno', interpolation='gaussian')
ax[2].set_title('core / parquet')
fig.colorbar(cax3, ax=ax[2], orientation='vertical')
ax[2].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[2].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Third heatmap
cax3 = ax[3].imshow(data4, cmap='inferno', interpolation='gaussian')
ax[3].set_title('core / stream_memory_map')
fig.colorbar(cax3, ax=ax[3], orientation='vertical')
ax[3].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[3].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.savefig(f"results/v2/row_wise/access_heatmap.pdf")  # Save as PDF
plt.clf()


################# MANIPULATING ##################

data1 = np.array(np.array(t_manipulate_hdf5_core)/np.array(t_manipulate_arrow_stream_no_memory))
data2 = np.array(np.array(t_manipulate_hdf5_sec2)/np.array(t_manipulate_arrow_stream_memory))
data3 = np.array(np.array(t_manipulate_hdf5_core)/np.array(t_manipulate_arrow_parquet))
data4 = np.array(np.array(t_manipulate_hdf5_core)/np.array(t_manipulate_arrow_stream_memory))

data1 = np.where(data1 >= 1, 1, 0)
data2 = np.where(data2 >= 1, 1, 0)
data3 = np.where(data3 >= 1, 1, 0)
data4 = np.where(data4 >= 1, 1, 0)

fig, ax = plt.subplots(1, 4, figsize=(14, 6))
plt.suptitle("Manipulate benchmark")

vmin = min(data1.min(), data2.min(),data3.min(),data4.min())
vmax = max(data1.max(),  data2.max(),data3.max(),data4.max())
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# First heatmap
cax1 = ax[0].imshow(data1, cmap='inferno', interpolation='gaussian')
ax[0].set_title('core / stream_no_memory_map')
fig.colorbar(cax1, ax=ax[0], orientation='vertical')
ax[0].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[0].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Second heatmap
cax2 = ax[1].imshow(data2, cmap='inferno', interpolation='gaussian')
ax[1].set_title('sec2 / stream_memory_map')
fig.colorbar(cax2, ax=ax[1], orientation='vertical')
ax[1].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[1].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Third heatmap
cax3 = ax[2].imshow(data3, cmap='inferno', interpolation='gaussian')
ax[2].set_title('core / parquet')
fig.colorbar(cax3, ax=ax[2], orientation='vertical')
ax[2].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[2].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Third heatmap
cax3 = ax[3].imshow(data4, cmap='inferno', interpolation='gaussian')
ax[3].set_title('core / stream_memory_map')
fig.colorbar(cax3, ax=ax[3], orientation='vertical')
ax[3].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[3].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.savefig(f"results/v2/row_wise/manipulating_heatmap.pdf")  # Save as PDF
plt.clf()


############# COLUMN WISE #################

t_load_arrow_file_memory = []
t_load_arrow_stream_memory = []
t_load_arrow_file_no_memory = []
t_load_arrow_stream_no_memory = []
t_load_arrow_parquet = []
t_load_hdf5_core = []
t_load_hdf5_sec2 = []

t_access_arrow_file_memory = []
t_access_arrow_stream_memory = []
t_access_arrow_file_no_memory = []
t_access_arrow_stream_no_memory = []
t_access_arrow_parquet = []
t_access_hdf5_core = []
t_access_hdf5_sec2 = []

t_manipulate_arrow_file_memory = []
t_manipulate_arrow_stream_memory = []
t_manipulate_arrow_file_no_memory = []
t_manipulate_arrow_stream_no_memory = []
t_manipulate_arrow_parquet = []
t_manipulate_hdf5_core = []
t_manipulate_hdf5_sec2 = []

for dim in dimensions:
    t_load_arrow_file_memory.append(np.load(f"results/v2/column_wise/{dim}/t_load_arrow_file_memory.npy"))
    t_load_arrow_stream_memory.append(np.load(f"results/v2/column_wise/{dim}/t_load_arrow_stream_memory.npy"))
    t_load_arrow_file_no_memory.append(np.load(f"results/v2/column_wise/{dim}/t_load_arrow_file_no_memory.npy"))
    t_load_arrow_stream_no_memory.append(np.load(f"results/v2/column_wise/{dim}/t_load_arrow_stream_no_memory.npy"))
    t_load_arrow_parquet.append(np.load(f"results/v2/column_wise/{dim}/t_load_arrow_parquet.npy"))
    t_load_hdf5_core.append(np.load(f"results/v2/column_wise/{dim}/t_load_hdf5_core.npy"))
    t_load_hdf5_sec2.append(np.load(f"results/v2/column_wise/{dim}/t_load_hdf5_sec2.npy"))

    t_access_arrow_file_memory.append(np.load(f"results/v2/column_wise/{dim}/t_access_arrow_file_memory.npy"))
    t_access_arrow_stream_memory.append(np.load(f"results/v2/column_wise/{dim}/t_access_arrow_stream_memory.npy"))
    t_access_arrow_file_no_memory.append(np.load(f"results/v2/column_wise/{dim}/t_access_arrow_file_no_memory.npy"))
    t_access_arrow_stream_no_memory.append(np.load(f"results/v2/column_wise/{dim}/t_access_arrow_stream_no_memory.npy"))
    t_access_arrow_parquet.append(np.load(f"results/v2/column_wise/{dim}/t_access_arrow_parquet.npy"))
    t_access_hdf5_core.append(np.load(f"results/v2/column_wise/{dim}/t_access_hdf5_core.npy"))
    t_access_hdf5_sec2.append(np.load(f"results/v2/column_wise/{dim}/t_access_hdf5_sec2.npy"))

    t_manipulate_arrow_file_memory.append(np.load(f"results/v2/column_wise/{dim}/t_manipulate_arrow_file_memory.npy"))
    t_manipulate_arrow_stream_memory.append(np.load(f"results/v2/column_wise/{dim}/t_manipulate_arrow_stream_memory.npy"))
    t_manipulate_arrow_file_no_memory.append(np.load(f"results/v2/column_wise/{dim}/t_manipulate_arrow_file_no_memory.npy"))
    t_manipulate_arrow_stream_no_memory.append(np.load(f"results/v2/column_wise/{dim}/t_manipulate_arrow_stream_no_memory.npy"))
    t_manipulate_arrow_parquet.append(np.load(f"results/v2/column_wise/{dim}/t_manipulate_arrow_parquet.npy"))
    t_manipulate_hdf5_core.append(np.load(f"results/v2/column_wise/{dim}/t_manipulate_hdf5_core.npy"))
    t_manipulate_hdf5_sec2.append(np.load(f"results/v2/column_wise/{dim}/t_manipulate_hdf5_sec2.npy"))

################# LOAD ##################

data1 = np.array(np.array(t_load_hdf5_core)/np.array(t_load_arrow_stream_no_memory))
data2 = np.array(np.array(t_load_hdf5_sec2)/np.array(t_load_arrow_stream_memory))
data3 = np.array(np.array(t_load_hdf5_core)/np.array(t_load_arrow_parquet))
data4 = np.array(np.array(t_load_hdf5_core)/np.array(t_load_arrow_stream_memory))

data1 = np.where(data1 >= 1, 1, 0)
data2 = np.where(data2 >= 1, 1, 0)
data3 = np.where(data3 >= 1, 1, 0)
data4 = np.where(data4 >= 1, 1, 0)


fig, ax = plt.subplots(1, 4, figsize=(14, 6))
plt.suptitle("Loading benchmark")

vmin = min(data1.min(), data2.min(),data3.min(),data4.min())
vmax = max(data1.max(),  data2.max(),data3.max(),data4.max())
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# First heatmap
cax1 = ax[0].imshow(data1, cmap='inferno', interpolation='gaussian')
ax[0].set_title('core / stream_no_memory_map')
fig.colorbar(cax1, ax=ax[0], orientation='vertical')
ax[0].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[0].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Second heatmap
cax2 = ax[1].imshow(data2, cmap='inferno', interpolation='gaussian')
ax[1].set_title('sec2 / stream_memory_map')
fig.colorbar(cax2, ax=ax[1], orientation='vertical')
ax[1].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[1].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Third heatmap
cax3 = ax[2].imshow(data3, cmap='inferno', interpolation='gaussian')
ax[2].set_title('core / parquet')
fig.colorbar(cax3, ax=ax[2], orientation='vertical')
ax[2].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[2].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Third heatmap
cax3 = ax[3].imshow(data4, cmap='inferno', interpolation='gaussian')
ax[3].set_title('core / stream_memory_map')
fig.colorbar(cax3, ax=ax[3], orientation='vertical')
ax[3].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[3].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.savefig(f"results/v2/column_wise/load_heatmap.pdf")  # Save as PDF
plt.clf()

################# ACCESS ##################

data1 = np.array(np.array(t_access_hdf5_core)/np.array(t_access_arrow_stream_no_memory))
data2 = np.array(np.array(t_access_hdf5_sec2)/np.array(t_access_arrow_stream_memory))
data3 = np.array(np.array(t_access_hdf5_core)/np.array(t_access_arrow_parquet))
data4 = np.array(np.array(t_access_hdf5_core)/np.array(t_access_arrow_stream_memory))

data1 = np.where(data1 >= 1, 1, 0)
data2 = np.where(data2 >= 1, 1, 0)
data3 = np.where(data3 >= 1, 1, 0)
data4 = np.where(data4 >= 1, 1, 0)

fig, ax = plt.subplots(1, 4, figsize=(14, 6))
plt.suptitle("Access benchmark")

vmin = min(data1.min(), data2.min(),data3.min(),data4.min())
vmax = max(data1.max(),  data2.max(),data3.max(),data4.max())
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# First heatmap
cax1 = ax[0].imshow(data1, cmap='inferno', interpolation='gaussian')
ax[0].set_title('core / stream_no_memory_map')
fig.colorbar(cax1, ax=ax[0], orientation='vertical')
ax[0].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[0].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Second heatmap
cax2 = ax[1].imshow(data2, cmap='inferno', interpolation='gaussian')
ax[1].set_title('sec2 / stream_memory_map')
fig.colorbar(cax2, ax=ax[1], orientation='vertical')
ax[1].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[1].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Third heatmap
cax3 = ax[2].imshow(data3, cmap='inferno', interpolation='gaussian')
ax[2].set_title('core / parquet')
fig.colorbar(cax3, ax=ax[2], orientation='vertical')
ax[2].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[2].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Third heatmap
cax3 = ax[3].imshow(data4, cmap='inferno', interpolation='gaussian')
ax[3].set_title('core / stream_memory_map')
fig.colorbar(cax3, ax=ax[3], orientation='vertical')
ax[3].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[3].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.savefig(f"results/v2/column_wise/access_heatmap.pdf")  # Save as PDF
plt.clf()


################# MANIPULATING ##################

data1 = np.array(np.array(t_manipulate_hdf5_core)/np.array(t_manipulate_arrow_stream_no_memory))
data2 = np.array(np.array(t_manipulate_hdf5_sec2)/np.array(t_manipulate_arrow_stream_memory))
data3 = np.array(np.array(t_manipulate_hdf5_core)/np.array(t_manipulate_arrow_parquet))
data4 = np.array(np.array(t_manipulate_hdf5_core)/np.array(t_manipulate_arrow_stream_memory))

data1 = np.where(data1 >= 1, 1, 0)
data2 = np.where(data2 >= 1, 1, 0)
data3 = np.where(data3 >= 1, 1, 0)
data4 = np.where(data4 >= 1, 1, 0)

fig, ax = plt.subplots(1, 4, figsize=(14, 6))
plt.suptitle("Manipulate benchmark")

vmin = min(data1.min(), data2.min(),data3.min(),data4.min())
vmax = max(data1.max(),  data2.max(),data3.max(),data4.max())
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# First heatmap
cax1 = ax[0].imshow(data1, cmap='inferno', interpolation='gaussian')
ax[0].set_title('core / stream_no_memory_map')
fig.colorbar(cax1, ax=ax[0], orientation='vertical')
ax[0].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[0].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Second heatmap
cax2 = ax[1].imshow(data2, cmap='inferno', interpolation='gaussian')
ax[1].set_title('sec2 / stream_memory_map')
fig.colorbar(cax2, ax=ax[1], orientation='vertical')
ax[1].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[1].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Third heatmap
cax3 = ax[2].imshow(data3, cmap='inferno', interpolation='gaussian')
ax[2].set_title('core / parquet')
fig.colorbar(cax3, ax=ax[2], orientation='vertical')
ax[2].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[2].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Third heatmap
cax3 = ax[3].imshow(data4, cmap='inferno', interpolation='gaussian')
ax[3].set_title('core / stream_memory_map')
fig.colorbar(cax3, ax=ax[3], orientation='vertical')
ax[3].set_xticks(ticks=np.arange(len(N_label)), labels=N_label,rotation=45)
ax[3].set_yticks(ticks=np.arange(len(d_label)), labels=d_label)

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.savefig(f"results/v2/column_wise/manipulating_heatmap.pdf")  # Save as PDF
plt.clf()

