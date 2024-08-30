from .base import SampleGenerator
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import random
from typing import Tuple

class F3(SampleGenerator):

    def create_dataset(self, N: int, d: Tuple, D: int, name : str = "outputs/data_hierarchy"):
        output = []
        with h5py.File(name+".h5", 'w', driver="core") as f:
            
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
