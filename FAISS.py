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

def most_frequent(List):
    return max(set(List), key=List.count)


root_path_animal = r"C:\Users\Cristiano Lavoro\Desktop\benchmarks\knn_outputs\all_animals"
root_path_species = r"C:\Users\Cristiano Lavoro\Desktop\benchmarks\knn_outputs\species"

with pa.OSFile(f"{root_path_animal}/ds_{1}_of_{18}_stream.arrows", 'rb') as source:
        table = pa.ipc.open_stream(source).read_all()
for i in list(range(2,19)):
        with pa.OSFile(f"{root_path_animal}/ds_{i}_of_{18}_stream.arrows", 'rb') as source:
                table = pa.concat_tables([table,pa.ipc.open_stream(source).read_all()])
table = table.combine_chunks()

with pa.OSFile(f"{root_path_species}/ds_{1}_of_{40}_stream.arrows", 'rb') as source:
        table_2 = pa.ipc.open_stream(source).read_all()
for i in list(range(2,41)):
        with pa.OSFile(f"{root_path_species}/ds_{i}_of_{40}_stream.arrows", 'rb') as source:
                table_2 = pa.concat_tables([table_2,pa.ipc.open_stream(source).read_all()])
table_2 = table_2.combine_chunks()

emb_animals = np.array(ff.get_feature(table,["image_feature","embedding_feature","vector"]).to_pylist())
emb_species = np.array(ff.get_feature(table_2,["image_feature","embedding_feature","vector"]).to_pylist())

class_animals = ff.get_feature(table,["image_feature","class_feature","label"]).to_pylist()
#class_species = np.array(ff.get_feature(table_2,["image_feature","class_feature","label"]).to_pylist())

image_species = np.array(ff.get_feature(table_2,["image_feature","image"]).to_pylist())

d = 512

X = emb_animals.astype("float32")
xq = emb_species.astype("float32")

index = faiss.IndexFlatL2(d) 
index.add(X) 

k = 6 
D, I = index.search(xq, k)

############# DEBUG SPACE FOR EVALUATIONS ##################
#print(I[1405])
#print(D[1405])
#prova = [10000]
#print(len(D))
############################################################
###         DA FARE MEGLIO IN ZERO COPY                  ###
thr = 35
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
        print(moda)
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
                        "confidence" : count/found_number_of_neighbours
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
ff.partition_dataset(new_table,r"C:\Users\Cristiano Lavoro\Desktop\benchmarks\knn_outputs\to_merge")
