import numpy as np
from matplotlib import pyplot as plt

######## ACCESS ###########


t_access_arrow_parquet_row_wise_point = np.load(r"C:\Users\gianl\OneDrive\Desktop\APPOGGIO\row wise\t_access_arrow_parquet.npy")[0]
t_access_arrow_parquet_column_wise_point = np.load(r"C:\Users\gianl\OneDrive\Desktop\APPOGGIO\colum wise\t_access_arrow_parquet.npy")[0]


t_access_arrow_parquet_row_wise_old = np.load("results/v2/row_wise/192/t_access_arrow_parquet.npy")
t_access_arrow_parquet_column_wise_old = np.load("results/v2/column_wise/192/t_access_arrow_parquet.npy")


t_access_arrow_parquet_row_wise_new = t_access_arrow_parquet_row_wise_old
t_access_arrow_parquet_column_wise_new = t_access_arrow_parquet_column_wise_old



t_access_arrow_parquet_row_wise_new[4] = t_access_arrow_parquet_row_wise_point
t_access_arrow_parquet_column_wise_new[4] = t_access_arrow_parquet_column_wise_point



np.save(f"results/v2/row_wise/192/t_access_arrow_parquet.npy",t_access_arrow_parquet_row_wise_new)
np.save(f"results/v2/column_wise/192/t_access_arrow_parquet.npy",t_access_arrow_parquet_column_wise_new)



########### LOAD ###############

t_load_arrow_parquet_row_wise_point = np.load(r"C:\Users\gianl\OneDrive\Desktop\APPOGGIO\row wise\t_load_arrow_parquet.npy")[0]
t_load_arrow_parquet_column_wise_point = np.load(r"C:\Users\gianl\OneDrive\Desktop\APPOGGIO\colum wise\t_load_arrow_parquet.npy")[0]


t_load_arrow_parquet_row_wise_old = np.load("results/v2/row_wise/192/t_load_arrow_parquet.npy")
t_load_arrow_parquet_column_wise_old = np.load("results/v2/column_wise/192/t_load_arrow_parquet.npy")


t_load_arrow_parquet_row_wise_new = t_load_arrow_parquet_row_wise_old
t_load_arrow_parquet_column_wise_new = t_load_arrow_parquet_column_wise_old

t_load_arrow_parquet_row_wise_new[4] = t_load_arrow_parquet_row_wise_point
t_load_arrow_parquet_column_wise_new[4] = t_load_arrow_parquet_column_wise_point



np.save(f"results/v2/row_wise/192/t_load_arrow_parquet.npy",t_load_arrow_parquet_row_wise_new)
np.save(f"results/v2/column_wise/192/t_load_arrow_parquet.npy",t_load_arrow_parquet_column_wise_new)



############  MANIPULATING ###################

t_manipulate_arrow_parquet_row_wise_point = np.load(r"C:\Users\gianl\OneDrive\Desktop\APPOGGIO\row wise\t_manipulate_arrow_parquet.npy")[0]
t_manipulate_arrow_parquet_column_wise_point = np.load(r"C:\Users\gianl\OneDrive\Desktop\APPOGGIO\colum wise\t_manipulate_arrow_parquet.npy")[0]



t_manipulate_arrow_parquet_row_wise_old = np.load("results/v2/row_wise/192/t_manipulate_arrow_parquet.npy")
t_manipulate_arrow_parquet_column_wise_old = np.load("results/v2/column_wise/192/t_manipulate_arrow_parquet.npy")


t_manipulate_arrow_parquet_row_wise_new = t_manipulate_arrow_parquet_row_wise_old
t_manipulate_arrow_parquet_column_wise_new = t_manipulate_arrow_parquet_column_wise_old

t_manipulate_arrow_parquet_row_wise_new[4] = t_manipulate_arrow_parquet_row_wise_point
t_manipulate_arrow_parquet_column_wise_new[4] = t_manipulate_arrow_parquet_column_wise_point



np.save(f"results/v2/row_wise/192/t_manipulate_arrow_parquet.npy",t_manipulate_arrow_parquet_row_wise_new)
np.save(f"results/v2/column_wise/192/t_manipulate_arrow_parquet.npy",t_manipulate_arrow_parquet_column_wise_new)


