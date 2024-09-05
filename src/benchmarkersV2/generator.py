import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
from tqdm import tqdm
import time
from matplotlib import pyplot as plt
from typing import List
import shutil

class Generator:
    


    def create_dataset(self, N: int, name : str, dim : int):
            output = []
            with h5py.File(f"{name}_core.h5", 'w') as f:
                for i in range(N):
                    text1 = "Foto di una nebulosa"
                    text2 = "Foto infrarossi di una nebulosa"
                    im1 = np.random.rand(3,dim,dim)
                    im2 = np.random.rand(3,dim,dim)
                    im3 = np.random.rand(3,dim,dim)
                    im4 = np.random.rand(3,dim,dim)
                    shape = (3,dim,dim)

                    #bb1 = np.random.rand(4)
                    #bb2 = np.random.rand(4)
                    #bb3 = np.random.rand(4)
                    #bb4 = np.random.rand(4)
                    #bb5 = np.random.rand(4)

                    bb1 = np.array([0.5,0.5,0.5,0.5,0])  #YOLO format
                                                    #Expected conversion to the other format: [0.25,0.25,0.75,0.75]
                    bb2 = np.array([0.5,0.5,0.5,0.5,0])  #YOLO format
                                                    #Expected conversion to the other format: [0.25,0.25,0.75,0.75]
                    bb3 = np.array([0.5,0.5,0.5,0.5,0])  #YOLO format
                                                    #Expected conversion to the other format: [0.25,0.25,0.75,0.75]
                    bb4 = np.array([0.0977, 0.2604, 0.8789, 0.7812,1]) #NOT YOLO format (xmin,ymin,xmax,ymax)
                                                    #Expected conversion to YOLO format: [0.4883, 0.5208, 0.7812, 0.5208]
                    bb5 = np.array([0.0977, 0.2604, 0.8789, 0.7812,1]) #NOT YOLO format (xmin,ymin,xmax,ymax)
                                                    #Expected conversion to YOLO format: [0.4883, 0.5208, 0.7812, 0.5208]
                    
                    if i == len(range(N)) - 1 :
                        label1 = 10
                    else:
                        label1 = np.random.randint(100,size=1)[0]
       
                    sample = {
                        "image_feature": [
                            {
                                "image" : im1.tobytes(),
                                "shape" : shape,
                                "boundingbox_feature":[{
                                    "bbox" : bb1,
                                    #"format" : 0,
                                    "image_1_feature": [{
                                        "image" : im4.tobytes(),
                                        "shape" : shape,
                                    }]
                                },
                                {
                                    "bbox" : bb2
                                    #"format" : 0,
                                },
                                {
                                    "bbox" : bb3,
                                    #"format" : 0,
                                    "label_feature": [{
                                        "label" : label1
                                    }]
                                }]
                            },
                            {
                                "image" : im2.tobytes(),
                                "shape" : shape,
                                "boundingbox_feature":[{
                                    "bbox" : bb4
                                    #"format" : 1,
                                },{
                                    "bbox" : bb5
                                    #"format" : 1,
                                }]
                            },
                            {
                                "image" : im3.tobytes(),
                                "shape" : shape,
                                "text_1_feature":[
                                    {
                                        "text": text2
                                    }
                                ]
                            }
                        ],
                        "text_feature" : [
                            {
                                "text": text1
                            }
                        ],

                    }

                    output.append(sample)

                    example = f.create_group(f'example_{i}')
                    image_feature = example.create_group('image_feature')
                    
                    # Image 1
                    image1 = image_feature.create_group('image1')
                    image1.create_dataset('image', data=im1)  # Replace with actual image data
                    
                    boundingbox_feature1 = image1.create_group('boundingbox_feature')
                    
                    bb1g = boundingbox_feature1.create_group('bb1')
                    bb1ds = bb1g.create_dataset('bbox', data=bb1)  # Replace with actual bbox
                    #bb1ds.attrs["format"] = 0
                    
                    image_1_feature = bb1g.create_group('image_1_feature')
                    image_1_feature_ds = image_1_feature.create_dataset('image', data=im4)  # Replace with actual image data
                    image_1_feature_ds.attrs["type"] = "leaf_image"
                    
                    bb2g = boundingbox_feature1.create_group('bb2')
                    bb2ds = bb2g.create_dataset('bbox', data=bb2)  # Replace with actual bbox
                    #bb2ds.attrs["format"] = 0
                    
                    bb3g = boundingbox_feature1.create_group('bb3')
                    bb3ds = bb3g.create_dataset('bbox', data=bb3)  # Replace with actual bbox
                    bb3ds.attrs["label"] = label1
                    #bb3ds.attrs["format"] = 0
                    
                          
                    # Image 2
                    image2 = image_feature.create_group('image2')
                    image2.create_dataset('image', data=im2)  # Replace with actual image data
                    
                    boundingbox_feature2 = image2.create_group('boundingbox_feature')
                    
                    bb4g = boundingbox_feature2.create_group('bb4')
                    bb4ds = bb4g.create_dataset('bbox', data=bb4)  # Replace with actual bbox
                    #bb4ds.attrs["format"] = 1
                    
                    bb5g = boundingbox_feature2.create_group('bb5')
                    bb5ds = bb5g.create_dataset('bbox', data=bb5)  # Replace with actual bbox
                    #bb5ds.attrs["format"] = 1
                    
                    # Image 3
                    image3 = image_feature.create_group('image3')
                    image3.create_dataset('image', data=im3)  # Replace with actual image data
                    
                    text_1_feature = image3.create_group('text_1_feature')
                    text_1_feature.create_dataset('text', data=text2)
                    
                    # Text feature
                    text_feature = example.create_group('text_feature')
                    text_feature.create_dataset('text', data=text1)

            self.table = pa.Table.from_pylist(output)
            pq.write_table(self.table, name+".parquet")
            shutil.copy(f"{name}_core.h5", f"{name}_sec2.h5")

            

    def create_arrow_stream(self, name : str):
         batches = self.table.to_batches()
         with pa.OSFile(f"{name}_stream.arrows", 'wb') as sink:  # Open the file for binary writing
            with pa.ipc.new_stream(sink, batches[0].schema) as writer:  # Create an IPC writer
                for batch in batches:
                    writer.write_batch(batch)  # Write each batch to the IPC stream

    def create_arrow_file(self, name : str):
         batches = self.table.to_batches()
         with pa.OSFile(f"{name}_file.arrow", 'wb') as sink:  # Open the file for binary writing
            with pa.ipc.new_file(sink, batches[0].schema) as writer:  # Create an IPC writer
                for batch in batches:
                    writer.write_batch(batch)  # Write each batch to the IPC stream
         


    

