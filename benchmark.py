import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import h5py
from src.benchmarkers import *
import time
from matplotlib import pyplot as plt
np.random.seed(0)

N = [10,100,200,300,500]
d = [(3, 32, 32), (3, 64, 64), (3, 96, 96), (3, 128, 128), (3, 192, 192), 
 (3, 256, 256), (3, 384, 384), (3, 512, 512)]
D = 5
iterations = 100
selected_label = 6

#benchmark = F1()
#
#for item in N:
#    benchmark.create_dataset(item,f"outputs/f1/core/f1_{item}","core")
#for item in N:
#    benchmark.create_dataset(item,f"outputs/f1/sec2/f1_{item}","sec2")
#
#t1,t2,t3,t4 = benchmark.benchmark("outputs/f1/core/f1",N,iterations,hdf5_driver="core",plot=True,save=True)
#t1,t2,t3,t4 = benchmark.benchmark("outputs/f1/sec2/f1",N,iterations,hdf5_driver="sec2",plot=True,save=True)
#
#benchmark = F2()
#
#for item in N:
#    benchmark.create_dataset(item,f"outputs/f2/row_wise/core/f2_{item}","core")
#for item in N:
#    benchmark.create_dataset(item,f"outputs/f2/row_wise/sec2/f2_{item}","sec2")
#
#t1,t2,t3,t4 = benchmark.benchmark_row_wise("outputs/f2/row_wise/core/f2",N,iterations,selected_label,hdf5_driver="core",plot=True,save=True)
#t1,t2,t3,t4 = benchmark.benchmark_row_wise("outputs/f2/row_wise/sec2/f2",N,iterations,selected_label,hdf5_driver="sec2",plot=True,save=True)
#
#for item in N:
#    benchmark.create_dataset(item,f"outputs/f2/bounding_boxes/core/f2_{item}","core")
#for item in N:
#    benchmark.create_dataset(item,f"outputs/f2/bounding_boxes/sec2/f2_{item}","sec2")
#
#t1,t2,t3,t4 = benchmark.benchmark_bounding_boxes_conversion("outputs/f2/bounding_boxes/core/f2",N,iterations,selected_label,hdf5_driver="core",plot=True,save=True)
#t1,t2,t3,t4 = benchmark.benchmark_bounding_boxes_conversion("outputs/f2/bounding_boxes/sec2/f2",N,iterations,selected_label,hdf5_driver="sec2",plot=True,save=True)

benchmark = F3()

for item in N:
    for res in tqdm(d):
        benchmark.create_dataset(item,res,D,f"outputs/f3/column_wise/core/f3_{item}_{res}_{D}","core")
for item in N:
    for res in tqdm(d):
        benchmark.create_dataset(item,res,D,f"outputs/f3/column_wise/sec2/f3_{item}_{res}_{D}","sec2")

t1,t2,t3,t4 = benchmark.benchmark_heatmap_column_wise("outputs/f3/column_wise/core/f3",N,d,D,iterations,hdf5_driver="core",plot=True,save=True)
t1,t2,t3,t4 = benchmark.benchmark_heatmap_column_wise("outputs/f3/column_wise/sec2/f3",N,d,D,iterations,hdf5_driver="sec2",plot=True,save=True)

selected_label = 1

for item in N:
    for res in tqdm(d):
        benchmark.create_dataset(item,res,D,f"outputs/f3/row_wise/core/f3_{item}_{res}_{D}","core")
for item in N:
    for res in tqdm(d):
        benchmark.create_dataset(item,res,D,f"outputs/f3/row_wise/sec2/f3_{item}_{res}_{D}","sec2")

t1,t2,t3,t4 = benchmark.benchmark_heatmap_row_wise("outputs/f3/row_wise/core/f3",N,d,D,iterations,selected_label,hdf5_driver="core",plot=True,save=True)
t1,t2,t3,t4 = benchmark.benchmark_heatmap_row_wise("outputs/f3/row_wise/sec2/f3",N,d,D,iterations,selected_label,hdf5_driver="sec2",plot=True,save=True)


#D = 10
#
#
#for item in N:
#    for res in tqdm(d):
#        benchmark.create_dataset(item,res,D,f"outputs/f3/column_wise/core/f3_{item}_{res}_{D}","core")
#for item in N:
#    for res in tqdm(d):
#        benchmark.create_dataset(item,res,D,f"outputs/f3/column_wise/sec2/f3_{item}_{res}_{D}","sec2")
#
#t1,t2,t3,t4 = benchmark.benchmark_heatmap_column_wise("outputs/f3/column_wise/core/f3",N,d,D,iterations,hdf5_driver="core",plot=True,save=True)
#t1,t2,t3,t4 = benchmark.benchmark_heatmap_column_wise("outputs/f3/column_wise/sec2/f3",N,d,D,iterations,hdf5_driver="sec2",plot=True,save=True)
#
#selected_label = 1
#
#for item in N:
#    for res in tqdm(d):
#        benchmark.create_dataset(item,res,D,f"outputs/f3/row_wise/core/f3_{item}_{res}_{D}","core")
#for item in N:
#    for res in tqdm(d):
#        benchmark.create_dataset(item,res,D,f"outputs/f3/row_wise/sec2/f3_{item}_{res}_{D}","sec2")
#
#t1,t2,t3,t4 = benchmark.benchmark_heatmap_row_wise("outputs/f3/row_wise/core/f3",N,d,D,iterations,selected_label,hdf5_driver="core",plot=True,save=True)
#t1,t2,t3,t4 = benchmark.benchmark_heatmap_row_wise("outputs/f3/row_wise/sec2/f3",N,d,D,iterations,selected_label,hdf5_driver="sec2",plot=True,save=True)