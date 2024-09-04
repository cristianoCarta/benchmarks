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

for dim in dimensions:

    ########## ROW - WISE ###############

    arrow_file_memory = ClockRowWise().benchmark_arrow(f"outputs/v2/{dim}/ds",N,iterations,dim,selected_label,memory=True,stream=False)
    arrow_stream_memory = ClockRowWise().benchmark_arrow(f"outputs/v2/{dim}/ds",N,iterations,dim,selected_label,memory=True,stream=True)
    arrow_file_no_memory = ClockRowWise().benchmark_arrow(f"outputs/v2/{dim}/ds",N,iterations,dim,selected_label,memory=False,stream=False)
    arrow_stream_no_memory = ClockRowWise().benchmark_arrow(f"outputs/v2/{dim}/ds",N,iterations,dim,selected_label,memory=False,stream=True)
    arrow_parquet = ClockRowWise().benchmark_parquet(f"outputs/v2/{dim}/ds",N,iterations,selected_label,dim)
    hdf5_core = ClockRowWise().benchmark_hdf5(f"outputs/v2/{dim}/ds",N,iterations,selected_label,hdf5_driver="core")
    hdf5_sec2 = ClockRowWise().benchmark_hdf5(f"outputs/v2/{dim}/ds",N,iterations,selected_label,hdf5_driver="sec2")

    t_load_arrow_file_memory = arrow_file_memory.t_load
    t_load_arrow_stream_memory = arrow_stream_memory.t_load
    t_load_arrow_file_no_memory = arrow_file_no_memory.t_load
    t_load_arrow_stream_no_memory = arrow_stream_no_memory.t_load
    t_load_arrow_parquet = arrow_parquet.t_load
    t_load_hdf5_core = hdf5_core.t_load
    t_load_hdf5_sec2 = hdf5_sec2.t_load

    t_access_arrow_file_memory = arrow_file_memory.t_access
    t_access_arrow_stream_memory = arrow_stream_memory.t_access
    t_access_arrow_file_no_memory = arrow_file_no_memory.t_access
    t_access_arrow_stream_no_memory = arrow_stream_no_memory.t_access
    t_access_arrow_parquet = arrow_parquet.t_access
    t_access_hdf5_core = hdf5_core.t_access
    t_access_hdf5_sec2 = hdf5_sec2.t_access

    t_manipulate_arrow_file_memory = arrow_file_memory.t_manipulate
    t_manipulate_arrow_stream_memory = arrow_stream_memory.t_manipulate
    t_manipulate_arrow_file_no_memory = arrow_file_no_memory.t_manipulate
    t_manipulate_arrow_stream_no_memory = arrow_stream_no_memory.t_manipulate
    t_manipulate_arrow_parquet = arrow_parquet.t_manipulate
    t_manipulate_hdf5_core = hdf5_core.t_manipulate
    t_manipulate_hdf5_sec2 = hdf5_sec2.t_manipulate

    np.save(f"results/v2/row_wise/{dim}/t_load_arrow_file_memory.npy",t_load_arrow_file_memory)
    np.save(f"results/v2/row_wise/{dim}/t_load_arrow_stream_memory.npy",t_load_arrow_stream_memory)
    np.save(f"results/v2/row_wise/{dim}/t_load_arrow_file_no_memory.npy",t_load_arrow_file_no_memory)
    np.save(f"results/v2/row_wise/{dim}/t_load_arrow_stream_no_memory.npy",t_load_arrow_stream_no_memory)
    np.save(f"results/v2/row_wise/{dim}/t_load_arrow_parquet.npy",t_load_arrow_parquet)
    np.save(f"results/v2/row_wise/{dim}/t_load_hdf5_core.npy",t_load_hdf5_core)
    np.save(f"results/v2/row_wise/{dim}/t_load_hdf5_sec2.npy",t_load_hdf5_sec2)

    np.save(f"results/v2/row_wise/{dim}/t_access_arrow_file_memory.npy",t_access_arrow_file_memory)
    np.save(f"results/v2/row_wise/{dim}/t_access_arrow_stream_memory.npy",t_access_arrow_stream_memory)
    np.save(f"results/v2/row_wise/{dim}/t_access_arrow_file_no_memory.npy",t_access_arrow_file_no_memory)
    np.save(f"results/v2/row_wise/{dim}/t_access_arrow_stream_no_memory.npy",t_access_arrow_stream_no_memory)
    np.save(f"results/v2/row_wise/{dim}/t_access_arrow_parquet.npy",t_access_arrow_parquet)
    np.save(f"results/v2/row_wise/{dim}/t_access_hdf5_core.npy",t_access_hdf5_core)
    np.save(f"results/v2/row_wise/{dim}/t_access_hdf5_sec2.npy",t_access_hdf5_sec2)

    np.save(f"results/v2/row_wise/{dim}/t_manipulate_arrow_file_memory.npy",t_manipulate_arrow_file_memory)
    np.save(f"results/v2/row_wise/{dim}/t_manipulate_arrow_stream_memory.npy",t_manipulate_arrow_stream_memory)
    np.save(f"results/v2/row_wise/{dim}/t_manipulate_arrow_file_no_memory.npy",t_manipulate_arrow_file_no_memory)
    np.save(f"results/v2/row_wise/{dim}/t_manipulate_arrow_stream_no_memory.npy",t_manipulate_arrow_stream_no_memory)
    np.save(f"results/v2/row_wise/{dim}/t_manipulate_arrow_parquet.npy",t_manipulate_arrow_parquet)
    np.save(f"results/v2/row_wise/{dim}/t_manipulate_hdf5_core.npy",t_manipulate_hdf5_core)
    np.save(f"results/v2/row_wise/{dim}/t_manipulate_hdf5_sec2.npy",t_manipulate_hdf5_sec2)

    plt.title("HDF5 vs Arrow Loading")
    plt.plot(N, t_load_arrow_file_memory, label="arrow_file_memory_map")
    plt.plot(N, t_load_arrow_stream_memory, label="arrow_stream_memory_map")
    plt.plot(N, t_load_arrow_file_no_memory, label="arrow_file_no_memory_map")
    plt.plot(N, t_load_arrow_stream_no_memory, label="arrow_stream_no_memory_map")
    plt.plot(N, t_load_arrow_parquet, label="arrow_parquet")
    plt.plot(N, t_load_hdf5_core, label="hdf5_core")
    plt.plot(N, t_load_hdf5_sec2, label="hdf5_sec2")

    # Add a legend
    plt.legend()

    # Add titles and labels
    plt.xlabel("N (number of samples)")
    plt.ylabel("t (seconds)")

    # Show the plot
    plt.savefig(f"results/v2/row_wise/{dim}/load_{dim}.pdf")  # Save as PDF

    plt.clf()

    plt.title("HDF5 vs Arrow Access")
    plt.plot(N, t_access_arrow_file_memory, label="arrow_file_memory_map")
    plt.plot(N, t_access_arrow_stream_memory, label="arrow_stream_memory_map")
    plt.plot(N, t_access_arrow_file_no_memory, label="arrow_file_no_memory_map")
    plt.plot(N, t_access_arrow_stream_no_memory, label="arrow_stream_no_memory_map")
    plt.plot(N, t_access_arrow_parquet, label="arrow_parquet")
    plt.plot(N, t_access_hdf5_core, label="hdf5_core")
    plt.plot(N, t_access_hdf5_sec2, label="hdf5_sec2")

    # Add a legend
    plt.legend()

    # Add titles and labels
    plt.xlabel("N (number of samples)")
    plt.ylabel("t (seconds)")

    # Show the plot
    plt.savefig(f"results/v2/row_wise/{dim}/access_{dim}.pdf")  # Save as PDF

    plt.clf()

    plt.title("HDF5 vs Arrow Manipulating")
    plt.plot(N, t_manipulate_arrow_file_memory, label="arrow_file_memory_map")
    plt.plot(N, t_manipulate_arrow_stream_memory, label="arrow_stream_memory_map")
    plt.plot(N, t_manipulate_arrow_file_no_memory, label="arrow_file_no_memory_map")
    plt.plot(N, t_manipulate_arrow_stream_no_memory, label="arrow_stream_no_memory_map")
    plt.plot(N, t_manipulate_arrow_parquet, label="arrow_parquet")
    plt.plot(N, t_manipulate_hdf5_core, label="hdf5_core")
    plt.plot(N, t_manipulate_hdf5_sec2, label="hdf5_sec2")

    # Add a legend
    plt.legend()

    # Add titles and labels
    plt.xlabel("N (number of samples)")
    plt.ylabel("t (seconds)")

    # Show the plot
    plt.savefig(f"results/v2/row_wise/{dim}/manipulating_{dim}.pdf")  # Save as PDF

    plt.clf()



    ############### COLUMN - WISE #######################

    arrow_file_memory = ClockColumnWise().benchmark_arrow(f"outputs/v2/{dim}/ds",N,iterations,dim,memory=True,stream=False)
    arrow_stream_memory = ClockColumnWise().benchmark_arrow(f"outputs/v2/{dim}/ds",N,iterations,dim,memory=True,stream=True)
    arrow_file_no_memory = ClockColumnWise().benchmark_arrow(f"outputs/v2/{dim}/ds",N,iterations,dim,memory=False,stream=False)
    arrow_stream_no_memory = ClockColumnWise().benchmark_arrow(f"outputs/v2/{dim}/ds",N,iterations,dim,memory=False,stream=True)
    arrow_parquet = ClockColumnWise().benchmark_parquet(f"outputs/v2/{dim}/ds",N,iterations,dim)
    hdf5_core = ClockColumnWise().benchmark_hdf5(f"outputs/v2/{dim}/ds",N,iterations,hdf5_driver="core")
    hdf5_sec2 = ClockColumnWise().benchmark_hdf5(f"outputs/v2/{dim}/ds",N,iterations,hdf5_driver="sec2")

    t_load_arrow_file_memory = arrow_file_memory.t_load
    t_load_arrow_stream_memory = arrow_stream_memory.t_load
    t_load_arrow_file_no_memory = arrow_file_no_memory.t_load
    t_load_arrow_stream_no_memory = arrow_stream_no_memory.t_load
    t_load_arrow_parquet = arrow_parquet.t_load
    t_load_hdf5_core = hdf5_core.t_load
    t_load_hdf5_sec2 = hdf5_sec2.t_load

    t_access_arrow_file_memory = arrow_file_memory.t_access
    t_access_arrow_stream_memory = arrow_stream_memory.t_access
    t_access_arrow_file_no_memory = arrow_file_no_memory.t_access
    t_access_arrow_stream_no_memory = arrow_stream_no_memory.t_access
    t_access_arrow_parquet = arrow_parquet.t_access
    t_access_hdf5_core = hdf5_core.t_access
    t_access_hdf5_sec2 = hdf5_sec2.t_access

    t_manipulate_arrow_file_memory = arrow_file_memory.t_manipulate
    t_manipulate_arrow_stream_memory = arrow_stream_memory.t_manipulate
    t_manipulate_arrow_file_no_memory = arrow_file_no_memory.t_manipulate
    t_manipulate_arrow_stream_no_memory = arrow_stream_no_memory.t_manipulate
    t_manipulate_arrow_parquet = arrow_parquet.t_manipulate
    t_manipulate_hdf5_core = hdf5_core.t_manipulate
    t_manipulate_hdf5_sec2 = hdf5_sec2.t_manipulate

    np.save(f"results/v2/column_wise/{dim}/t_load_arrow_file_memory.npy",t_load_arrow_file_memory)
    np.save(f"results/v2/column_wise/{dim}/t_load_arrow_stream_memory.npy",t_load_arrow_stream_memory)
    np.save(f"results/v2/column_wise/{dim}/t_load_arrow_file_no_memory.npy",t_load_arrow_file_no_memory)
    np.save(f"results/v2/column_wise/{dim}/t_load_arrow_stream_no_memory.npy",t_load_arrow_stream_no_memory)
    np.save(f"results/v2/column_wise/{dim}/t_load_arrow_parquet.npy",t_load_arrow_parquet)
    np.save(f"results/v2/column_wise/{dim}/t_load_hdf5_core.npy",t_load_hdf5_core)
    np.save(f"results/v2/column_wise/{dim}/t_load_hdf5_sec2.npy",t_load_hdf5_sec2)

    np.save(f"results/v2/column_wise/{dim}/t_access_arrow_file_memory.npy",t_access_arrow_file_memory)
    np.save(f"results/v2/column_wise/{dim}/t_access_arrow_stream_memory.npy",t_access_arrow_stream_memory)
    np.save(f"results/v2/column_wise/{dim}/t_access_arrow_file_no_memory.npy",t_access_arrow_file_no_memory)
    np.save(f"results/v2/column_wise/{dim}/t_access_arrow_stream_no_memory.npy",t_access_arrow_stream_no_memory)
    np.save(f"results/v2/column_wise/{dim}/t_access_arrow_parquet.npy",t_access_arrow_parquet)
    np.save(f"results/v2/column_wise/{dim}/t_access_hdf5_core.npy",t_access_hdf5_core)
    np.save(f"results/v2/column_wise/{dim}/t_access_hdf5_sec2.npy",t_access_hdf5_sec2)

    np.save(f"results/v2/column_wise/{dim}/t_manipulate_arrow_file_memory.npy",t_manipulate_arrow_file_memory)
    np.save(f"results/v2/column_wise/{dim}/t_manipulate_arrow_stream_memory.npy",t_manipulate_arrow_stream_memory)
    np.save(f"results/v2/column_wise/{dim}/t_manipulate_arrow_file_no_memory.npy",t_manipulate_arrow_file_no_memory)
    np.save(f"results/v2/column_wise/{dim}/t_manipulate_arrow_stream_no_memory.npy",t_manipulate_arrow_stream_no_memory)
    np.save(f"results/v2/column_wise/{dim}/t_manipulate_arrow_parquet.npy",t_manipulate_arrow_parquet)
    np.save(f"results/v2/column_wise/{dim}/t_manipulate_hdf5_core.npy",t_manipulate_hdf5_core)
    np.save(f"results/v2/column_wise/{dim}/t_manipulate_hdf5_sec2.npy",t_manipulate_hdf5_sec2)

    plt.title("HDF5 vs Arrow Loading")
    plt.plot(N, t_load_arrow_file_memory, label="arrow_file_memory_map")
    plt.plot(N, t_load_arrow_stream_memory, label="arrow_stream_memory_map")
    plt.plot(N, t_load_arrow_file_no_memory, label="arrow_file_no_memory_map")
    plt.plot(N, t_load_arrow_stream_no_memory, label="arrow_stream_no_memory_map")
    plt.plot(N, t_load_arrow_parquet, label="arrow_parquet")
    plt.plot(N, t_load_hdf5_core, label="hdf5_core")
    plt.plot(N, t_load_hdf5_sec2, label="hdf5_sec2")

    # Add a legend
    plt.legend()

    # Add titles and labels
    plt.xlabel("N (number of samples)")
    plt.ylabel("t (seconds)")

    # Show the plot
    plt.savefig(f"results/v2/column_wise/{dim}/load_{dim}.pdf")  # Save as PDF

    plt.clf()

    plt.title("HDF5 vs Arrow Access")
    plt.plot(N, t_access_arrow_file_memory, label="arrow_file_memory_map")
    plt.plot(N, t_access_arrow_stream_memory, label="arrow_stream_memory_map")
    plt.plot(N, t_access_arrow_file_no_memory, label="arrow_file_no_memory_map")
    plt.plot(N, t_access_arrow_stream_no_memory, label="arrow_stream_no_memory_map")
    plt.plot(N, t_access_arrow_parquet, label="arrow_parquet")
    plt.plot(N, t_access_hdf5_core, label="hdf5_core")
    plt.plot(N, t_access_hdf5_sec2, label="hdf5_sec2")

    # Add a legend
    plt.legend()

    # Add titles and labels
    plt.xlabel("N (number of samples)")
    plt.ylabel("t (seconds)")

    # Show the plot
    plt.savefig(f"results/v2/column_wise/{dim}/access_{dim}.pdf")  # Save as PDF

    plt.clf()

    plt.title("HDF5 vs Arrow Manipulating")
    plt.plot(N, t_manipulate_arrow_file_memory, label="arrow_file_memory_map")
    plt.plot(N, t_manipulate_arrow_stream_memory, label="arrow_stream_memory_map")
    plt.plot(N, t_manipulate_arrow_file_no_memory, label="arrow_file_no_memory_map")
    plt.plot(N, t_manipulate_arrow_stream_no_memory, label="arrow_stream_no_memory_map")
    plt.plot(N, t_manipulate_arrow_parquet, label="arrow_parquet")
    plt.plot(N, t_manipulate_hdf5_core, label="hdf5_core")
    plt.plot(N, t_manipulate_hdf5_sec2, label="hdf5_sec2")

    # Add a legend
    plt.legend()

    # Add titles and labels
    plt.xlabel("N (number of samples)")
    plt.ylabel("t (seconds)")

    # Show the plot
    plt.savefig(f"results/v2/column_wise/{dim}/manipulating_{dim}.pdf")  # Save as PDF

    plt.clf()