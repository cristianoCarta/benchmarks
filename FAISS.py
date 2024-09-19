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
from collections import defaultdict
import framework_functions as ff

def compute_confidence(new_candidate_classes: List,
                       dv : List,
                       k : int):
    confidence = 0
    choosen_class = None

    grouped = defaultdict(list)
    for i, d in zip(new_candidate_classes, dv):
        grouped[i].append(d)

    grouped_dict = dict(grouped)

    for cls, distncs in grouped_dict.items():
        new_confidence = new_candidate_classes.count(cls) / (k + (sum(distncs)/new_candidate_classes.count(cls)))
        if new_confidence > confidence:
              confidence = new_confidence
              choosen_class = cls
    
    return choosen_class, confidence



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

    emb_animals_norm = []
    emb_species_norm = []

    for emb_an in emb_animals:
        norm = np.linalg.norm(emb_an,ord=2)
        emb_animals_norm.append(emb_an/norm)
    
    for emb_sp in emb_species:
        norm = np.linalg.norm(emb_sp,ord=2)
        emb_species_norm.append(emb_sp/norm)

    emb_animals_norm = np.array(emb_animals_norm)
    emb_species_norm = np.array(emb_species_norm)
          

    image_species = np.array(ff.get_feature(table_2,["image_feature","image"]))

    d = len(emb_animals[0])
    X = emb_animals_norm.astype("float32")
    xq = emb_species_norm.astype("float32")
    index = faiss.IndexFlatL2(d) 
    index.add(X) 
    D, I = index.search(xq, k)

    output = []
    counter = 0

    I_t = []
    D_t = []

    for iv,dv in zip(I,D):
        iv = np.array(iv)
        dv = np.array(dv)
        #iv = [iv[i] for i in range(len(dv)) if dv[i]<thr]
        #dv = [dv[i] for i in range(len(dv)) if dv[i]<thr]
        iv = [iv[i] if dv[i]<thr else None for i in range(len(dv))]
        dv = [dv[i] if dv[i]<thr else None for i in range(len(dv))]
        #I_t.append(iv)
        #D_t.append(dv)

        new_candidate_classes = [class_animals[item] for item in iv if item]
        if new_candidate_classes:


            classe_dedotta , confidence = compute_confidence(new_candidate_classes,dv,k)
            moda = most_frequent(new_candidate_classes)
            #count = 0
            #for i in new_candidate_classes:
            #    if i==moda:
            #        count = count+1
        
            sample = {
                "image_feature":[{
                    "image" : image_species[counter],
                    "class_feature":[
                        {
                            "label" : classe_dedotta,
                            "dataset" : "dataset_2_species",
                            "confidence" : confidence*100
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

    #np.save(r"C:\Users\Cristiano Lavoro\Desktop\benchmarks\knn_outputs\resnet\to_merge\I.npy",I_t)
    #np.save(r"C:\Users\Cristiano Lavoro\Desktop\benchmarks\knn_outputs\resnet\to_merge\D.npy",D_t)

    return new_table
    #return pa.concat_tables([table,new_table])

root_path_animal = "../benchmarks/knn_outputs/resnet/all_animals"
root_path_species = "../benchmarks/knn_outputs/resnet/species"
root_path_to_merge ="../benchmarks/knn_outputs/resnet/to_merge"

merged_table = knn(root_path_animal,
        root_path_species,
        root_path_to_merge,
        ["image_feature","embedding_feature","vector"],
        ["image_feature","embedding_feature","vector"],
        ["image_feature","class_feature","label"],
        6,
        0.4)
len(merged_table)
merged_table.column("image_feature").chunk(0).values.field("image")
merged_table.column("image_feature").chunk(0).values.field("class_feature").values.field("label")
debug = [0,0,0,0,0]

