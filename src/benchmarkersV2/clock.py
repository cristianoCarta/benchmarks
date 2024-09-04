from typing import List
from tqdm import tqdm
import time
import pyarrow.parquet as pq
import numpy as np
import h5py
import pyarrow as pa

class ClockColumnWise:

    def benchmark_parquet(self,
                  path : str, 
                  N : List[int],
                  iterations : int,
                  dim : int
                 ):
         
        t_load_arrow = []
        t_access_arrow = []
        t_manipulate_arrow = []

        for item in N:

            tmp_load_arrow = []
            tmp_access_arrow = []
            tmp_manipulate_arrow = []
            
            for j in tqdm(range(iterations)):
                ### LOADING ###
                st_time_arrow = time.time()
                table = pq.read_table(f'{path}_{item}.parquet')
                en_time_arrow = time.time()
                tmp_load_arrow.append(en_time_arrow - st_time_arrow)

                ### MANIPULATION ###
                start_time_arrow = time.time()
                start_access = time.time()
                oggetti = np.array(table.column("image_feature").chunk(0).values.field("boundingbox_feature").values.field("image_1_feature").values.field("image").to_pylist())
                image_numpy = np.frombuffer(oggetti, dtype=np.float64).reshape(-1, 3, dim, dim)
                end_access = time.time()
                new_obj = np.transpose(image_numpy,axes=(0,3,2,1))   
                new_obj = np.square(new_obj)
                new_obj = np.exp(new_obj)
                new_obj = np.transpose(new_obj,axes=(0,3,2,1))
                end_time_arrow = time.time()
                tmp_manipulate_arrow.append(end_time_arrow-start_time_arrow)
                tmp_access_arrow.append(end_access - start_access)

            t_load_arrow.append(sum(tmp_load_arrow) / len(tmp_load_arrow))
            t_manipulate_arrow.append(sum(tmp_manipulate_arrow) / len(tmp_manipulate_arrow))
            t_access_arrow.append(sum(tmp_access_arrow) / len(tmp_access_arrow))

        self.t_load = t_load_arrow
        self.t_manipulate = t_manipulate_arrow
        self.t_access = t_access_arrow
        return self

    def benchmark_hdf5(self,
                  path : str, 
                  N : List[int],
                  iterations : int,
                  hdf5_driver : str = None,
                  ):
         
        t_load_hdf5 = []
        t_access_hdf5 = []
        t_manipulate_hdf5 = []

        for item in N:

            tmp_load_hdf5 = []
            tmp_access_hdf5 = []
            tmp_manipulate_hdf5 = []
            
            for j in tqdm(range(iterations)):
                
                ### LOADING ###
                st_time_hdf5 = time.time()
                with h5py.File(f'{path}_{item}_{hdf5_driver}.h5', 'r', driver=hdf5_driver) as f:
                    en_time_hdf5 = time.time()
                    tmp_load_hdf5.append(en_time_hdf5 - st_time_hdf5)
                    
                    ### MANIPULATION ###
                    image_datasets = []
                    start_time_hdf5 = time.time()
                    start_access = time.time()
                    for i in range(item):
                        dataset = f.get(f"example_{i}/image_feature/image1/boundingbox_feature/bb1/image_1_feature/image")
                        if dataset:
                            obj = dataset[:]
                            end_access = time.time()
                            new_obj = np.transpose(obj)   
                            new_obj = np.square(new_obj)
                            new_obj = np.exp(new_obj)
                            new_obj = np.transpose(new_obj)
                            image_datasets.append(new_obj[:])
                    end_time_hdf5 = time.time()
                    tmp_manipulate_hdf5.append(end_time_hdf5-start_time_hdf5)
                    tmp_access_hdf5.append(end_access - start_access)

            t_load_hdf5.append(sum(tmp_load_hdf5) / len(tmp_load_hdf5))
            t_manipulate_hdf5.append(sum(tmp_manipulate_hdf5) / len(tmp_manipulate_hdf5))
            t_access_hdf5.append(sum(tmp_access_hdf5) / len(tmp_access_hdf5))

        self.t_load = t_load_hdf5
        self.t_manipulate = t_manipulate_hdf5
        self.t_access = t_access_hdf5
        return self

    def benchmark_arrow(
            self,
            path : str, 
            N : List[int],
            iterations : int,
            dim : int,
            memory : bool = False,
            stream : bool = False):
        
        t_load = []
        t_access = []
        t_manipulate = []

        for item in N:

            tmp_load = []
            tmp_access = []
            tmp_manipulate = []
            
            for j in tqdm(range(iterations)):

                
                if stream and memory:
                    st_time_load = time.time()
                    with pa.memory_map(f'{path}_{item}_stream.arrows', 'rb') as source:
                        table = pa.ipc.open_stream(source).read_all()
                        en_time_load = time.time()
                        tmp_load.append(en_time_load - st_time_load)

                    start_time = time.time()
                    start_access = time.time()
                    oggetti = np.array(table.column("image_feature").chunk(0).values.field("boundingbox_feature").values.field("image_1_feature").values.field("image").to_pylist())
                    image_numpy = np.frombuffer(oggetti, dtype=np.float64).reshape(-1, 3, dim, dim)
                    end_access = time.time()
                    new_obj = np.transpose(image_numpy,axes=(0,3,2,1))   
                    new_obj = np.square(new_obj)
                    new_obj = np.exp(new_obj)
                    new_obj = np.transpose(new_obj,axes=(0,3,2,1))
                    end_time = time.time()
                    tmp_manipulate.append(end_time-start_time)
                    tmp_access.append(end_access - start_access)

                
                elif stream and (not memory):
                    st_time_load = time.time()
                    with pa.OSFile(f'{path}_{item}_stream.arrows', 'rb') as source:
                        table = pa.ipc.open_stream(source).read_all()
                        en_time_load = time.time()
                        tmp_load.append(en_time_load - st_time_load)

                    start_time = time.time()
                    start_access = time.time()
                    oggetti = np.array(table.column("image_feature").chunk(0).values.field("boundingbox_feature").values.field("image_1_feature").values.field("image").to_pylist())
                    image_numpy = np.frombuffer(oggetti, dtype=np.float64).reshape(-1, 3, dim, dim)
                    end_access = time.time()
                    new_obj = np.transpose(image_numpy,axes=(0,3,2,1))   
                    new_obj = np.square(new_obj)
                    new_obj = np.exp(new_obj)
                    new_obj = np.transpose(new_obj,axes=(0,3,2,1))
                    end_time = time.time()
                    tmp_manipulate.append(end_time-start_time)
                    tmp_access.append(end_access - start_access)

                elif (not stream) and memory:
                    st_time_load = time.time()
                    with pa.memory_map(f'{path}_{item}_file.arrow', 'rb') as source:
                        table = pa.ipc.open_file(source).read_all()
                        en_time_load = time.time()
                        tmp_load.append(en_time_load - st_time_load)

                    start_time = time.time()
                    start_access = time.time()
                    oggetti = np.array(table.column("image_feature").chunk(0).values.field("boundingbox_feature").values.field("image_1_feature").values.field("image").to_pylist())
                    image_numpy = np.frombuffer(oggetti, dtype=np.float64).reshape(-1, 3, dim, dim)
                    end_access = time.time()
                    new_obj = np.transpose(image_numpy,axes=(0,3,2,1))   
                    new_obj = np.square(new_obj)
                    new_obj = np.exp(new_obj)
                    new_obj = np.transpose(new_obj,axes=(0,3,2,1))
                    end_time = time.time()
                    tmp_manipulate.append(end_time-start_time)
                    tmp_access.append(end_access - start_access)

                elif (not stream) and (not memory):
                    st_time_load = time.time()
                    with pa.OSFile(f'{path}_{item}_file.arrow', 'rb') as source:
                        table = pa.ipc.open_file(source).read_all()
                        en_time_load = time.time()
                        tmp_load.append(en_time_load - st_time_load)

                    start_time = time.time()
                    start_access = time.time()
                    oggetti = np.array(table.column("image_feature").chunk(0).values.field("boundingbox_feature").values.field("image_1_feature").values.field("image").to_pylist())
                    image_numpy = np.frombuffer(oggetti, dtype=np.float64).reshape(-1, 3, dim, dim)
                    end_access = time.time()
                    new_obj = np.transpose(image_numpy,axes=(0,3,2,1))   
                    new_obj = np.square(new_obj)
                    new_obj = np.exp(new_obj)
                    new_obj = np.transpose(new_obj,axes=(0,3,2,1))
                    end_time = time.time()
                    tmp_manipulate.append(end_time-start_time)
                    tmp_access.append(end_access - start_access)

                else:
                    raise NotImplementedError("something went wrong")
        
            t_load.append(sum(tmp_load) / len(tmp_load))
            t_manipulate.append(sum(tmp_manipulate) / len(tmp_manipulate))
            t_access.append(sum(tmp_access) / len(tmp_access))

        self.t_load = t_load
        self.t_manipulate = t_manipulate
        self.t_access = t_access
        return self


class ClockRowWise:

    def benchmark_parquet(self,
                  path : str, 
                  N : List[int],
                  iterations : int,
                  selected_label : int,
                  dim : int
                 ):
         
        t_load_arrow = []
        t_access_arrow = []
        t_manipulate_arrow = []

        for item in N:

            tmp_load_arrow = []
            tmp_access_arrow = []
            tmp_manipulate_arrow = []
            
            for j in tqdm(range(iterations)):
                ### LOADING ###
                st_time_arrow = time.time()
                table = pq.read_table(f'{path}_{item}.parquet')
                en_time_arrow = time.time()
                tmp_load_arrow.append(en_time_arrow - st_time_arrow)

                ### MANIPULATION ###
                manipulated_images = []
                start_time_arrow = time.time()
                start_access = time.time()
                for k in range(item):
                    if any( x == selected_label for x in table.column("image_feature").chunk(0)[k][0]["boundingbox_feature"][2]["label_feature"].values.field("label").to_pylist()) :  ## Indice di immagine deve variare nel framework
                        obj = table.column("image_feature").chunk(0)[k][0]["image"]
                        image_numpy = np.frombuffer(obj.as_buffer(), dtype=np.float64).reshape(3, dim, dim)
                        end_access = time.time()
                        new_obj = np.transpose(image_numpy)   
                        new_obj = np.square(new_obj)
                        new_obj = np.exp(new_obj)
                        new_obj = np.transpose(new_obj)
                        manipulated_images.append(new_obj)
                end_time_arrow = time.time()
                tmp_manipulate_arrow.append(end_time_arrow-start_time_arrow)
                tmp_access_arrow.append(end_access - start_access)

            t_load_arrow.append(sum(tmp_load_arrow) / len(tmp_load_arrow))
            t_manipulate_arrow.append(sum(tmp_manipulate_arrow) / len(tmp_manipulate_arrow))
            t_access_arrow.append(sum(tmp_access_arrow) / len(tmp_access_arrow))

        self.t_load = t_load_arrow
        self.t_manipulate = t_manipulate_arrow
        self.t_access = t_access_arrow
        return self
        
    def benchmark_hdf5(self,
                  path : str, 
                  N : List[int],
                  iterations : int,
                  selected_label : int,
                  hdf5_driver : str = None
                  ):
         
        t_load_hdf5 = []
        t_access_hdf5 = []
        t_manipulate_hdf5 = []

        for item in N:

            tmp_load_hdf5 = []
            tmp_access_hdf5 = []
            tmp_manipulate_hdf5 = []
            
            for j in tqdm(range(iterations)):
                
                ### LOADING ###
                st_time_hdf5 = time.time()
                with h5py.File(f'{path}_{item}_{hdf5_driver}.h5', 'r', driver=hdf5_driver) as f:
                    en_time_hdf5 = time.time()
                    tmp_load_hdf5.append(en_time_hdf5 - st_time_hdf5)
                    
                    ### MANIPULATION ###
                    image_datasets = []
                    start_time_hdf5 = time.time()
                    start_access = time.time()
                    for i in range(item):
                        dataset = f.get(f"example_{i}/image_feature/image1/boundingbox_feature/bb3/bbox")
                        if dataset and ("label" in dataset.attrs) and dataset.attrs["label"] == selected_label:
                            obj = dataset[:]
                            end_access = time.time()
                            new_obj = np.transpose(obj)   
                            new_obj = np.square(new_obj)
                            new_obj = np.exp(new_obj)
                            new_obj = np.transpose(new_obj)
                            image_datasets.append(new_obj[:])
                    end_time_hdf5 = time.time()
                    tmp_manipulate_hdf5.append(end_time_hdf5-start_time_hdf5)
                    tmp_access_hdf5.append(end_access-start_access)

            t_load_hdf5.append(sum(tmp_load_hdf5) / len(tmp_load_hdf5))
            t_manipulate_hdf5.append(sum(tmp_manipulate_hdf5) / len(tmp_manipulate_hdf5))
            t_access_hdf5.append(sum(tmp_access_hdf5) / len(tmp_access_hdf5))

        self.t_load = t_load_hdf5
        self.t_manipulate = t_manipulate_hdf5
        self.t_access = t_access_hdf5
        return self
        
    def benchmark_arrow(
            self,
            path : str, 
            N : List[int],
            iterations : int,
            dim : int,
            selected_label : int,
            memory : bool = False,
            stream : bool = False):
        
        t_load = []
        t_access = []
        t_manipulate = []

        for item in N:

            tmp_load = []
            tmp_access = []
            tmp_manipulate = []
            
            for j in tqdm(range(iterations)):

                
                if stream and memory:
                    st_time_load = time.time()
                    with pa.memory_map(f'{path}_{item}_stream.arrows', 'rb') as source:
                        table = pa.ipc.open_stream(source).read_all()
                        en_time_load = time.time()
                        tmp_load.append(en_time_load - st_time_load)

                    manipulated_images = []
                    start_time_arrow = time.time()
                    start_access = time.time()
                    for k in range(item):
                        if any( x == selected_label for x in table.column("image_feature").chunk(0)[k][0]["boundingbox_feature"][2]["label_feature"].values.field("label").to_pylist()) :  ## Indice di immagine deve variare nel framework
                            obj = table.column("image_feature").chunk(0)[k][0]["image"]
                            image_numpy = np.frombuffer(obj.as_buffer(), dtype=np.float64).reshape(3, dim, dim)
                            end_access = time.time()
                            new_obj = np.transpose(image_numpy)   
                            new_obj = np.square(new_obj)
                            new_obj = np.exp(new_obj)
                            new_obj = np.transpose(new_obj)
                            manipulated_images.append(new_obj)
                    end_time_arrow = time.time()
                    tmp_manipulate.append(end_time_arrow-start_time_arrow)
                    tmp_access.append(end_access-start_access)

                
                elif stream and (not memory):
                    st_time_load = time.time()
                    with pa.OSFile(f'{path}_{item}_stream.arrows', 'rb') as source:
                        table = pa.ipc.open_stream(source).read_all()
                        en_time_load = time.time()
                        tmp_load.append(en_time_load - st_time_load)

                    manipulated_images = []
                    start_time_arrow = time.time()
                    start_access = time.time()
                    for k in range(item):
                        if any( x == selected_label for x in table.column("image_feature").chunk(0)[k][0]["boundingbox_feature"][2]["label_feature"].values.field("label").to_pylist()) :  ## Indice di immagine deve variare nel framework
                            obj = table.column("image_feature").chunk(0)[k][0]["image"]
                            image_numpy = np.frombuffer(obj.as_buffer(), dtype=np.float64).reshape(3, dim, dim)
                            end_access = time.time()
                            new_obj = np.transpose(image_numpy)   
                            new_obj = np.square(new_obj)
                            new_obj = np.exp(new_obj)
                            new_obj = np.transpose(new_obj)
                            manipulated_images.append(new_obj)
                    end_time_arrow = time.time()
                    tmp_manipulate.append(end_time_arrow-start_time_arrow)
                    tmp_access.append(end_access-start_access)


                elif (not stream) and memory:
                    st_time_load = time.time()
                    with pa.memory_map(f'{path}_{item}_file.arrow', 'rb') as source:
                        table = pa.ipc.open_file(source).read_all()
                        en_time_load = time.time()
                        tmp_load.append(en_time_load - st_time_load)

                    manipulated_images = []
                    start_time_arrow = time.time()
                    start_access = time.time()
                    for k in range(item):
                        if any( x == selected_label for x in table.column("image_feature").chunk(0)[k][0]["boundingbox_feature"][2]["label_feature"].values.field("label").to_pylist()) :  ## Indice di immagine deve variare nel framework
                            obj = table.column("image_feature").chunk(0)[k][0]["image"]
                            image_numpy = np.frombuffer(obj.as_buffer(), dtype=np.float64).reshape(3, dim, dim)
                            end_access = time.time()
                            new_obj = np.transpose(image_numpy)   
                            new_obj = np.square(new_obj)
                            new_obj = np.exp(new_obj)
                            new_obj = np.transpose(new_obj)
                            manipulated_images.append(new_obj)
                    end_time_arrow = time.time()
                    tmp_manipulate.append(end_time_arrow-start_time_arrow)
                    tmp_access.append(end_access-start_access)

                elif (not stream) and (not memory):
                    st_time_load = time.time()
                    with pa.OSFile(f'{path}_{item}_file.arrow', 'rb') as source:
                        table = pa.ipc.open_file(source).read_all()
                        en_time_load = time.time()
                        tmp_load.append(en_time_load - st_time_load)

                    manipulated_images = []
                    start_time_arrow = time.time()
                    start_access = time.time()
                    for k in range(item):
                        if any( x == selected_label for x in table.column("image_feature").chunk(0)[k][0]["boundingbox_feature"][2]["label_feature"].values.field("label").to_pylist()) :  ## Indice di immagine deve variare nel framework
                            obj = table.column("image_feature").chunk(0)[k][0]["image"]
                            image_numpy = np.frombuffer(obj.as_buffer(), dtype=np.float64).reshape(3, dim, dim)
                            end_access = time.time()
                            new_obj = np.transpose(image_numpy)   
                            new_obj = np.square(new_obj)
                            new_obj = np.exp(new_obj)
                            new_obj = np.transpose(new_obj)
                            manipulated_images.append(new_obj)
                    end_time_arrow = time.time()
                    tmp_manipulate.append(end_time_arrow-start_time_arrow)
                    tmp_access.append(end_access-start_access)

                else:
                    raise NotImplementedError("something went wrong")
        
            t_load.append(sum(tmp_load) / len(tmp_load))
            t_manipulate.append(sum(tmp_manipulate) / len(tmp_manipulate))
            t_access.append(sum(tmp_access) / len(tmp_access))

        self.t_load = t_load
        self.t_manipulate = t_manipulate
        self.t_access = t_access
        return self