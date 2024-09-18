import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds
from PIL import Image
from src.benchmarkers import *
from src.benchmarkersV2 import *
from tqdm import tqdm
import os
import numpy as np
from io import BytesIO
from typing import *
import pickle
from collections import Counter
import framework_functions as ff

def knn(root_path_animal : str,
        root_path_species : str,
        root_path_to_merge : str,
        emb_path_animals : List[str],
        emb_path_species : List[str],
        class_path_animals : List[str],
        k : int,
        thr : int,
        ):

    def most_frequent(List):
        return max(set(List), key=List.count)
    
    
    offsets = np.load(f"{root_path_animal}/ds_offsets.npy")
    with pa.OSFile(f"{root_path_animal}/ds_{1}_of_{len(offsets)-1}_stream.arrows", 'rb') as source:
            table = pa.ipc.open_stream(source).read_all()
    for i in list(range(2,len(offsets))):
            with pa.OSFile(f"{root_path_animal}/ds_{i}_of_{len(offsets)-1}_stream.arrows", 'rb') as source:
                    table = pa.concat_tables([table,pa.ipc.open_stream(source).read_all()])
    table = table.combine_chunks()

    offsets = np.load(f"{root_path_species}/ds_offsets.npy")
    with pa.OSFile(f"{root_path_species}/ds_{1}_of_{len(offsets)-1}_stream.arrows", 'rb') as source:
            table_2 = pa.ipc.open_stream(source).read_all()
    for i in list(range(2,len(offsets))):
            with pa.OSFile(f"{root_path_species}/ds_{i}_of_{len(offsets)-1}_stream.arrows", 'rb') as source:
                    table_2 = pa.concat_tables([table_2,pa.ipc.open_stream(source).read_all()])
    table_2 = table_2.combine_chunks()

    emb_animals = np.array(ff.get_feature(table,emb_path_animals).to_pylist())
    emb_species = np.array(ff.get_feature(table_2,emb_path_species).to_pylist())
    class_animals = ff.get_feature(table,class_path_animals).to_pylist()

    image_species = np.array(ff.get_feature(table_2,["image_feature","image"]))

    d = len(emb_animals[0])
    X = emb_animals.astype("float32")
    xq = emb_species.astype("float32")
    index = faiss.IndexFlatL2(d) 
    index.add(X) 
    D, I = index.search(xq, k)

    output = []
    counter = 0

    for iv,dv in zip(I,D):
        iv = np.array(iv)
        dv = np.array(dv)
        iv = [iv[i] if dv[i]<thr else None for i in range(len(dv))]
        dv = [dv[i] if dv[i]<thr else None for i in range(len(dv))]

        found_number_of_neighbours = len([x for x in iv if x is not None])
        new_candidate_classes = [class_animals[item] for item in iv if item]
        if new_candidate_classes:
            moda = most_frequent(new_candidate_classes)
            count = 0
            for i in new_candidate_classes:
                if i==moda:
                    count = count+1
        
            sample = {
                "image_feature":[{
                    "image" : image_species[counter],
                    "class_feature":[
                        {
                            "label" : moda,
                            "dataset" : "dataset_2_species",
                            #"confidence" : count/found_number_of_neighbours
                        }
                    ],
                    "embedding_feature":[
                        {
                            "vector" : emb_species[counter]
                        }
                    ]
                }]
            }
            output.append(sample)
        counter = counter + 1

    new_table = pa.Table.from_pylist(output)
    ff.partition_dataset(new_table,root_path_to_merge)

    return pa.concat_tables([table,new_table])

