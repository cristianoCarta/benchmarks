from .base import SampleGenerator
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import random
from typing import Tuple, List
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

class F3(SampleGenerator):

    def create_dataset(self, N: int, d: Tuple, D: int, name : str, hdf5_driver : str):
        output = []
        with h5py.File(f"{name}_{hdf5_driver}.h5", 'w', driver="core") as f:
            
            for i in range(N):
                struct_array = []
                images = []
                images_b = []

                example = f.create_group(f'example_{i}')
                image_feature = example.create_group('image_feature')

                for j in range(D):
                    temp_dict = {}
                    im = np.random.rand(*d)
                    im_b = im.tobytes()
                    label = np.random.randint(0,3)
                    images.append(im)
                    images_b.append(im_b)

                    temp_dict["image"] = im_b
                    temp_dict["shape"] = d
                    temp_dict["class_feature"] = [{
                        "label" : label
                    }]

                    struct_array.append(temp_dict)

                    #hdf5
                    image_ds = image_feature.create_dataset(f'image_{j}', data=images[j])  
                    image_ds.attrs["label"] = label
                    image_ds.attrs["type"] = "image"

                sample = {
                    "image_feature":struct_array
                }

                output.append(sample)
        
        table = pa.Table.from_pylist(output)
        df = table.to_pandas()
        df.to_parquet(name+".parquet")

    def benchmark_heatmap_column_wise(self,
                  path : str, 
                  N : List[int],
                  d : List[Tuple],
                  D : int,
                  iterations : int,
                  hdf5_driver : str = None,
                  plot : bool = False,
                  save : bool = False):
         
        t_load_hdf5 = []
        t_manipulate_hdf5 = []

        t_load_arrow = []
        t_manipulate_arrow = []
        

        for item in N:

            tmp_load_res_arrow = []
            tmp_manipulate_res_arrow = []

            tmp_load_res_hdf5 = []
            tmp_manipulate_res_hdf5 = []

            for res in d:
                tmp_load_arrow = []
                tmp_manipulate_arrow = []

                tmp_load_hdf5 = []
                tmp_manipulate_hdf5 = []

                for j in tqdm(range(iterations)):
                    ### LOADING ###
                    st_time_arrow = time.time()
                    table = pq.read_table(f'{path}_{item}_{res}_{D}.parquet')
                    en_time_arrow = time.time()
                    tmp_load_arrow.append(en_time_arrow - st_time_arrow)
                    
                    ### MANIPULATION ###
                    start_time_arrow = time.time()
                    oggetti = np.array(table.column("image_feature").chunk(0).values.field("image").to_pylist())
                    image_numpy = np.frombuffer(oggetti, dtype=np.float64).reshape((-1,) + res)
                    new_obj = np.transpose(image_numpy,axes=(0,3,2,1))   
                    new_obj = np.square(new_obj)
                    new_obj = np.exp(new_obj)
                    new_obj = np.transpose(new_obj,axes=(0,3,2,1))
                    end_time_arrow = time.time()
                    tmp_manipulate_arrow.append(end_time_arrow-start_time_arrow)

                    ### LOADING ###
                    st_time_hdf5 = time.time()
                    with h5py.File(f'{path}_{item}_{res}_{D}_{hdf5_driver}.h5', 'r', driver="core") as f:
                        en_time_hdf5 = time.time()
                        tmp_load_hdf5.append(en_time_hdf5 - st_time_hdf5)

                        
                        ### MANIPULATION ###
                        def get_all_image_datasets():
                            image_datasets = []                
                            def visit_func(name, obj):
                                if isinstance(obj, h5py.Dataset) and obj.attrs["type"] == "image":
                                    new_obj = np.transpose(obj)   
                                    new_obj = np.square(new_obj)
                                    new_obj = np.exp(new_obj)
                                    new_obj = np.transpose(new_obj)
                                    image_datasets.append(new_obj[:])
                        
                            f.visititems(visit_func)
                            return np.array(image_datasets)
                        
                        start_time_hdf5 = time.time()
                        image_ds = get_all_image_datasets()
                        end_time_hdf5 = time.time()
                        tmp_manipulate_hdf5.append(end_time_hdf5-start_time_hdf5)
                
                tmp_load_res_arrow.append(sum(tmp_load_arrow) / len(tmp_load_arrow))
                tmp_manipulate_res_arrow.append(sum(tmp_manipulate_arrow) / len(tmp_manipulate_arrow))

                tmp_load_res_hdf5.append(sum(tmp_load_hdf5) / len(tmp_load_hdf5))
                tmp_manipulate_res_hdf5.append(sum(tmp_manipulate_hdf5) / len(tmp_manipulate_hdf5))
                    
            t_load_arrow.append(tmp_load_res_arrow)
            t_manipulate_arrow.append(tmp_manipulate_res_arrow)

            t_load_hdf5.append(tmp_load_res_hdf5)
            t_manipulate_hdf5.append(tmp_manipulate_res_hdf5)

        if plot:
            data1 = np.array(t_load_arrow)
            data2 = np.array(t_load_hdf5)
            data3 = np.array(np.array(t_load_hdf5)/np.array(t_load_arrow))

            N_label = [str(item) for item in N]
            d_label = [str(item) for item in d]

            fig, ax = plt.subplots(1, 3, figsize=(14, 6))
            plt.suptitle("Loading benchmark")

            vmin = min(data1.min(), data2.min())
            vmax = max(data1.max(), data2.max())
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            # First heatmap
            cax1 = ax[0].imshow(data1, cmap='coolwarm', interpolation='gaussian', norm=norm)
            ax[0].set_title('HDF5')
            fig.colorbar(cax1, ax=ax[0], orientation='vertical')
            ax[0].set_xticks(ticks=np.arange(len(d_label)), labels=d_label,rotation=45)
            ax[0].set_yticks(ticks=np.arange(len(N_label)), labels=N_label)

            # Second heatmap
            cax2 = ax[1].imshow(data2, cmap='coolwarm', interpolation='gaussian',norm=norm)
            ax[1].set_title('Arrow')
            fig.colorbar(cax2, ax=ax[1], orientation='vertical')
            ax[1].set_xticks(ticks=np.arange(len(d_label)), labels=d_label,rotation=45)
            ax[1].set_yticks(ticks=np.arange(len(N_label)), labels=N_label)

            # Third heatmap
            cax3 = ax[2].imshow(data3, cmap='inferno', interpolation='gaussian')
            ax[2].set_title('HDF5 / Arrrow')
            fig.colorbar(cax3, ax=ax[2], orientation='vertical')
            ax[2].set_xticks(ticks=np.arange(len(d_label)), labels=d_label,rotation=45)
            ax[2].set_yticks(ticks=np.arange(len(N_label)), labels=N_label)

            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Show the plot
            plt.savefig(f'{path}_{hdf5_driver}_{D}_loading.pdf')  # Save as PDF
            plt.show()

##################################################################################################
##################################################################################################
##################################################################################################

            data1 = np.array(t_manipulate_arrow)
            data2 = np.array(t_manipulate_hdf5)
            data3 = np.array(np.array(t_manipulate_hdf5)/np.array(t_manipulate_arrow))

            N_label = [str(item) for item in N]
            d_label = [str(item) for item in d]

            fig, ax = plt.subplots(1, 3, figsize=(14, 6))
            plt.suptitle("Manipulating benchmark")

            vmin = min(data1.min(), data2.min())
            vmax = max(data1.max(), data2.max())
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            # First heatmap
            cax1 = ax[0].imshow(data1, cmap='coolwarm', interpolation='gaussian', norm=norm)
            ax[0].set_title('HDF5')
            fig.colorbar(cax1, ax=ax[0], orientation='vertical')
            ax[0].set_xticks(ticks=np.arange(len(d_label)), labels=d_label,rotation=45)
            ax[0].set_yticks(ticks=np.arange(len(N_label)), labels=N_label)

            # Second heatmap
            cax2 = ax[1].imshow(data2, cmap='coolwarm', interpolation='gaussian',norm=norm)
            ax[1].set_title('Arrow')
            fig.colorbar(cax2, ax=ax[1], orientation='vertical')
            ax[1].set_xticks(ticks=np.arange(len(d_label)), labels=d_label,rotation=45)
            ax[1].set_yticks(ticks=np.arange(len(N_label)), labels=N_label)

            # Third heatmap
            cax3 = ax[2].imshow(data3, cmap='inferno', interpolation='gaussian')
            ax[2].set_title('HDF5 / Arrrow')
            fig.colorbar(cax3, ax=ax[2], orientation='vertical')
            ax[2].set_xticks(ticks=np.arange(len(d_label)), labels=d_label,rotation=45)
            ax[2].set_yticks(ticks=np.arange(len(N_label)), labels=N_label)

            # Adjust layout to prevent overlap
            plt.tight_layout()
            

            # Show the plot
            plt.savefig(f'{path}_{hdf5_driver}_{D}_manipulating.pdf')  # Save as PDF
            plt.show()


        if save:
            np.save(f"{path}_t_load_arrow_{D}.npy",t_load_arrow)
            np.save(f"{path}_t_manipulate_arrow_{D}.npy",t_manipulate_arrow)
            np.save(f"{path}_t_load_hdf5_{hdf5_driver}_{D}.npy",t_load_hdf5)
            np.save(f"{path}_t_manipulate_hdf5_{hdf5_driver}_{D}.npy",t_manipulate_hdf5)
            
        return t_load_arrow, t_manipulate_arrow , t_load_hdf5 , t_manipulate_hdf5
    

    def benchmark_heatmap_row_wise(self,
                  path : str, 
                  N : List[int],
                  d : List[Tuple],
                  D : int,
                  iterations : int,
                  selected_label : int,
                  hdf5_driver : str = None,
                  plot : bool = False,
                  save : bool = False):
         
        t_load_hdf5 = []
        t_manipulate_hdf5 = []

        t_load_arrow = []
        t_manipulate_arrow = []
        

        for item in N:

            tmp_load_res_arrow = []
            tmp_manipulate_res_arrow = []

            tmp_load_res_hdf5 = []
            tmp_manipulate_res_hdf5 = []

            for res in d:
                tmp_load_arrow = []
                tmp_manipulate_arrow = []

                tmp_load_hdf5 = []
                tmp_manipulate_hdf5 = []

                for j in tqdm(range(iterations)):
                    ### LOADING ###
                    st_time_arrow = time.time()
                    table = pq.read_table(f'{path}_{item}_{res}_{D}.parquet')
                    en_time_arrow = time.time()
                    tmp_load_arrow.append(en_time_arrow - st_time_arrow)
                    
                    ### MANIPULATION ###
                    start_time_arrow = time.time()
                    manipulated_images = []
                    for k in range(item):
                        if any( x == selected_label for x in table.column("image_feature").chunk(0)[k][0]["class_feature"].values.field("label").to_pylist()):  ## Indice di immagine deve variare nel framework
                            obj = table.column("image_feature").chunk(0)[k][0]["image"]
                            image_numpy = np.frombuffer(obj.as_buffer(), dtype=np.float64).reshape(res)
                            new_obj = np.transpose(image_numpy)   
                            new_obj = np.square(new_obj)
                            new_obj = np.exp(new_obj)
                            new_obj = np.transpose(new_obj)
                            manipulated_images.append(new_obj)
                    end_time_arrow = time.time()
                    tmp_manipulate_arrow.append(end_time_arrow-start_time_arrow)

                    ### LOADING ###
                    st_time_hdf5 = time.time()
                    with h5py.File(f'{path}_{item}_{res}_{D}_{hdf5_driver}.h5', 'r', driver="core") as f:
                        en_time_hdf5 = time.time()
                        tmp_load_hdf5.append(en_time_hdf5 - st_time_hdf5)

                        
                        ### MANIPULATION ###
                        def get_all_image_datasets():
                            image_datasets = []                
                            def visit_func(name, obj):
                                if isinstance(obj, h5py.Dataset) and obj.attrs["type"] == "image" and obj.attrs["label"] == selected_label :
                                    im_obj = obj[:]
                                    new_obj = np.transpose(im_obj)   
                                    new_obj = np.square(new_obj)
                                    new_obj = np.exp(new_obj)
                                    new_obj = np.transpose(new_obj)
                                    image_datasets.append(new_obj)
                            f.visititems(visit_func)
                            return image_datasets

                        start_time_hdf5 = time.time()
                        image_ds = get_all_image_datasets()
                        end_time_hdf5 = time.time()
                        tmp_manipulate_hdf5.append(end_time_hdf5-start_time_hdf5)
                
                tmp_load_res_arrow.append(sum(tmp_load_arrow) / len(tmp_load_arrow))
                tmp_manipulate_res_arrow.append(sum(tmp_manipulate_arrow) / len(tmp_manipulate_arrow))

                tmp_load_res_hdf5.append(sum(tmp_load_hdf5) / len(tmp_load_hdf5))
                tmp_manipulate_res_hdf5.append(sum(tmp_manipulate_hdf5) / len(tmp_manipulate_hdf5))
                    
            t_load_arrow.append(tmp_load_res_arrow)
            t_manipulate_arrow.append(tmp_manipulate_res_arrow)

            t_load_hdf5.append(tmp_load_res_hdf5)
            t_manipulate_hdf5.append(tmp_manipulate_res_hdf5)

        if plot:
            data1 = np.array(t_load_arrow)
            data2 = np.array(t_load_hdf5)
            data3 = np.array(np.array(t_load_hdf5)/np.array(t_load_arrow))

            N_label = [str(item) for item in N]
            d_label = [str(item) for item in d]

            fig, ax = plt.subplots(1, 3, figsize=(14, 6))
            plt.suptitle("Loading benchmark")

            vmin = min(data1.min(), data2.min())
            vmax = max(data1.max(), data2.max())
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            # First heatmap
            cax1 = ax[0].imshow(data1, cmap='coolwarm', interpolation='gaussian', norm=norm)
            ax[0].set_title('HDF5')
            fig.colorbar(cax1, ax=ax[0], orientation='vertical')
            ax[0].set_xticks(ticks=np.arange(len(d_label)), labels=d_label,rotation=45)
            ax[0].set_yticks(ticks=np.arange(len(N_label)), labels=N_label)

            # Second heatmap
            cax2 = ax[1].imshow(data2, cmap='coolwarm', interpolation='gaussian',norm=norm)
            ax[1].set_title('Arrow')
            fig.colorbar(cax2, ax=ax[1], orientation='vertical')
            ax[1].set_xticks(ticks=np.arange(len(d_label)), labels=d_label,rotation=45)
            ax[1].set_yticks(ticks=np.arange(len(N_label)), labels=N_label)

            # Third heatmap
            cax3 = ax[2].imshow(data3, cmap='inferno', interpolation='gaussian')
            ax[2].set_title('HDF5 / Arrrow')
            fig.colorbar(cax3, ax=ax[2], orientation='vertical')
            ax[2].set_xticks(ticks=np.arange(len(d_label)), labels=d_label,rotation=45)
            ax[2].set_yticks(ticks=np.arange(len(N_label)), labels=N_label)

            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Show the plot
            plt.savefig(f'{path}_{hdf5_driver}_{D}_loading.pdf')  # Save as PDF
            plt.show()

##################################################################################################
##################################################################################################
##################################################################################################

            data1 = np.array(t_manipulate_arrow)
            data2 = np.array(t_manipulate_hdf5)
            data3 = np.array(np.array(t_manipulate_hdf5)/np.array(t_manipulate_arrow))

            N_label = [str(item) for item in N]
            d_label = [str(item) for item in d]

            fig, ax = plt.subplots(1, 3, figsize=(14, 6))
            plt.suptitle("Manipulating benchmark")

            vmin = min(data1.min(), data2.min())
            vmax = max(data1.max(), data2.max())
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            # First heatmap
            cax1 = ax[0].imshow(data1, cmap='coolwarm', interpolation='gaussian', norm=norm)
            ax[0].set_title('HDF5')
            fig.colorbar(cax1, ax=ax[0], orientation='vertical')
            ax[0].set_xticks(ticks=np.arange(len(d_label)), labels=d_label,rotation=45)
            ax[0].set_yticks(ticks=np.arange(len(N_label)), labels=N_label)

            # Second heatmap
            cax2 = ax[1].imshow(data2, cmap='coolwarm', interpolation='gaussian',norm=norm)
            ax[1].set_title('Arrow')
            fig.colorbar(cax2, ax=ax[1], orientation='vertical')
            ax[1].set_xticks(ticks=np.arange(len(d_label)), labels=d_label,rotation=45)
            ax[1].set_yticks(ticks=np.arange(len(N_label)), labels=N_label)

            # Third heatmap
            cax3 = ax[2].imshow(data3, cmap='inferno', interpolation='gaussian')
            ax[2].set_title('HDF5 / Arrrow')
            fig.colorbar(cax3, ax=ax[2], orientation='vertical')
            ax[2].set_xticks(ticks=np.arange(len(d_label)), labels=d_label,rotation=45)
            ax[2].set_yticks(ticks=np.arange(len(N_label)), labels=N_label)

            # Adjust layout to prevent overlap
            plt.tight_layout()
            

            # Show the plot
            plt.savefig(f'{path}_{hdf5_driver}_{D}_manipulating.pdf')  # Save as PDF
            plt.show()


        if save:
            np.save(f"{path}_t_load_arrow_{D}.npy",t_load_arrow)
            np.save(f"{path}_t_manipulate_arrow_{D}.npy",t_manipulate_arrow)
            np.save(f"{path}_t_load_hdf5_{hdf5_driver}_{D}.npy",t_load_hdf5)
            np.save(f"{path}_t_manipulate_hdf5_{hdf5_driver}_{D}.npy",t_manipulate_hdf5)
            
        return t_load_arrow, t_manipulate_arrow , t_load_hdf5 , t_manipulate_hdf5

