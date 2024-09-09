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

for dim in dimensions:
    ########## BOUNDING BOX ##############

    print("BBOX STARTED")

    print(f"HDF5 CORE VISIT {dim}")
    hdf5_core_visit = ClockBoundingBoxConversion().benchmark_hdf5_visit_items(f"outputs/v2/{dim}/ds",N,iterations,hdf5_driver="core")
    print(f"HDF5 SEC2 VISIT {dim}")
    hdf5_sec2_visit = ClockBoundingBoxConversion().benchmark_hdf5_visit_items(f"outputs/v2/{dim}/ds",N,iterations,hdf5_driver="sec2")

    t_load_arrow_file_memory = np.load("dim")
    t_load_arrow_stream_memory = np.load("dim")
    t_load_arrow_file_no_memory = np.load("dim")
    t_load_arrow_stream_no_memory = np.load("dim")
    t_load_arrow_parquet = np.load("dim")
    t_load_hdf5_core = np.load("dim")
    t_load_hdf5_sec2 = np.load("dim")

    t_load_hdf5_core_visit = hdf5_core_visit.t_load
    t_load_hdf5_sec2_visit = hdf5_sec2_visit.t_load

    t_access_arrow_file_memory = np.load("dim")
    t_access_arrow_stream_memory = np.load("dim")
    t_access_arrow_file_no_memory = np.load("dim")
    t_access_arrow_stream_no_memory = np.load("dim")
    t_access_arrow_parquet = np.load("dim")
    t_access_hdf5_core = np.load("dim")
    t_access_hdf5_sec2 = np.load("dim")

    t_access_hdf5_core_visit = hdf5_core_visit.t_access
    t_access_hdf5_sec2_visit = hdf5_sec2_visit.t_access

    t_manipulate_arrow_file_memory = np.load("dim")
    t_manipulate_arrow_stream_memory = np.load("dim")
    t_manipulate_arrow_file_no_memory = np.load("dim")
    t_manipulate_arrow_stream_no_memory = np.load("dim")
    t_manipulate_arrow_parquet = np.load("dim")
    t_manipulate_hdf5_core = np.load("dim")
    t_manipulate_hdf5_sec2 = np.load("dim")

    t_manipulate_hdf5_core_visit = hdf5_core_visit.t_manipulate
    t_manipulate_hdf5_sec2_visit = hdf5_sec2_visit.t_manipulate

    
    np.save(f"results/v2/bounding_box/{dim}/t_load_hdf5_core_visit.npy",t_load_hdf5_core_visit)
    np.save(f"results/v2/bounding_box/{dim}/t_load_hdf5_sec2_visit.npy",t_load_hdf5_sec2_visit)

    
    np.save(f"results/v2/bounding_box/{dim}/t_access_hdf5_core_visit.npy",t_access_hdf5_core_visit)
    np.save(f"results/v2/bounding_box/{dim}/t_access_hdf5_sec2_visit.npy",t_access_hdf5_sec2_visit)

   
    np.save(f"results/v2/bounding_box/{dim}/t_manipulate_hdf5_core_visit.npy",t_manipulate_hdf5_core_visit)
    np.save(f"results/v2/bounding_box/{dim}/t_manipulate_hdf5_sec2_visit.npy",t_manipulate_hdf5_sec2_visit)

    plt.title("HDF5 vs Arrow Loading")
    plt.plot(N, t_load_arrow_file_memory, label="arrow_file_memory_map")
    plt.plot(N, t_load_arrow_stream_memory, label="arrow_stream_memory_map")
    plt.plot(N, t_load_arrow_file_no_memory, label="arrow_file_no_memory_map")
    plt.plot(N, t_load_arrow_stream_no_memory, label="arrow_stream_no_memory_map")
    plt.plot(N, t_load_arrow_parquet, label="arrow_parquet")
    plt.plot(N, t_load_hdf5_core, label="hdf5_core")
    plt.plot(N, t_load_hdf5_sec2, label="hdf5_sec2")
    plt.plot(N, t_load_hdf5_core_visit, label="hdf5_core_visit")
    plt.plot(N, t_load_hdf5_sec2_visit, label="hdf5_sec2_visit")

    # Add a legend
    plt.legend()

    # Add titles and labels
    plt.xlabel("N (number of samples)")
    plt.ylabel("t (seconds)")

    # Show the plot
    plt.savefig(f"results/v2/bounding_box/{dim}/load_{dim}.pdf")  # Save as PDF

    plt.clf()

    plt.title("HDF5 vs Arrow Access")
    plt.plot(N, t_access_arrow_file_memory, label="arrow_file_memory_map")
    plt.plot(N, t_access_arrow_stream_memory, label="arrow_stream_memory_map")
    plt.plot(N, t_access_arrow_file_no_memory, label="arrow_file_no_memory_map")
    plt.plot(N, t_access_arrow_stream_no_memory, label="arrow_stream_no_memory_map")
    plt.plot(N, t_access_arrow_parquet, label="arrow_parquet")
    plt.plot(N, t_access_hdf5_core, label="hdf5_core")
    plt.plot(N, t_access_hdf5_sec2, label="hdf5_sec2")
    plt.plot(N, t_access_hdf5_core_visit, label="hdf5_core_visit")
    plt.plot(N, t_access_hdf5_sec2_visit, label="hdf5_sec2_visit")

    # Add a legend
    plt.legend()

    # Add titles and labels
    plt.xlabel("N (number of samples)")
    plt.ylabel("t (seconds)")

    # Show the plot
    plt.savefig(f"results/v2/bounding_box/{dim}/access_{dim}.pdf")  # Save as PDF

    plt.clf()

    plt.title("HDF5 vs Arrow Manipulating")
    plt.plot(N, t_manipulate_arrow_file_memory, label="arrow_file_memory_map")
    plt.plot(N, t_manipulate_arrow_stream_memory, label="arrow_stream_memory_map")
    plt.plot(N, t_manipulate_arrow_file_no_memory, label="arrow_file_no_memory_map")
    plt.plot(N, t_manipulate_arrow_stream_no_memory, label="arrow_stream_no_memory_map")
    plt.plot(N, t_manipulate_arrow_parquet, label="arrow_parquet")
    plt.plot(N, t_manipulate_hdf5_core, label="hdf5_core")
    plt.plot(N, t_manipulate_hdf5_sec2, label="hdf5_sec2")
    plt.plot(N, t_manipulate_hdf5_core_visit, label="hdf5_core_visit")
    plt.plot(N, t_manipulate_hdf5_sec2_visit, label="hdf5_sec2_visit")

    # Add a legend
    plt.legend()

    # Add titles and labels
    plt.xlabel("N (number of samples)")
    plt.ylabel("t (seconds)")

    # Show the plot
    plt.savefig(f"results/v2/bounding_box/{dim}/manipulating_{dim}.pdf")  # Save as PDF

    plt.clf()

