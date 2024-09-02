from .base import SampleGenerator
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
from tqdm import tqdm
import time
from matplotlib import pyplot as plt
from typing import List

class F1(SampleGenerator):

    def create_dataset(self, N: int, name : str , hdf5_driver : str):
            output = []
            for i in range(N):
                text1 = "Foto di una nebulosa"
                text2 = "Foto infrarossi di una nebulosa"
                image1 = np.random.rand(3,125,125)
                image2 = np.random.rand(3,125,125)
                image3 = np.random.rand(3,125,125)
                shape = (3,125,125)
    
                
                bb1 = np.random.rand(4)
                bb2 = np.random.rand(4)
                bb3 = np.random.rand(4)
                bb4 = np.random.rand(4)
                bb5 = np.random.rand(4)
                image4 = np.random.rand(3,125,125)
                label1 = np.random.randint(100,size=1)[0]

                im1b = image1.tobytes()
                im1b = image1.tobytes()
                im1b = image1.tobytes()
                im1b = image1.tobytes()

                # print(np.frombuffer(im1b.as_buffer(), dtype=np.float32).shape)

                sample = {
                    "image_feature": [
                        {
                            "image" : image1.tobytes(),
                            "shape" : shape,
                            "boundingbox_feature":[{
                                "bbox" : bb1,
                                "image_1_feature": [{
                                    "image" : image4.tobytes(),
                                    "shape" : shape,
                                }]
                            },
                            {
                                "bbox" : bb2
                            },
                            {
                                "bbox" : bb3,
                                "label_feature": [{
                                    "label" : label1
                                }]
                            }]
                        },
                        {
                            "image" : image2.tobytes(),
                            "shape" : shape,
                            "boundingbox_feature":[{
                                "bbox" : bb4
                            },{
                                "bbox" : bb5
                            }]
                        },
                        {
                            "image" : image3.tobytes(),
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

            with h5py.File(f"{name}_{hdf5_driver}.h5", 'w') as f:
            # Create image_feature group
                for i in range(N):
                    text1 = "Foto di una nebulosa"
                    text2 = "Foto infrarossi di una nebulosa"
                    im1 = np.random.rand(3,125,125)
                    im2 = np.random.rand(3,125,125)
                    im3 = np.random.rand(3,125,125)
                    shape = (3,125,125)
                
                    
                    bb1 = np.random.rand(4)
                    bb2 = np.random.rand(4)
                    bb3 = np.random.rand(4)
                    bb4 = np.random.rand(4)
                    bb5 = np.random.rand(4)
                    im4 = np.random.rand(3,125,125)
                    label1 = np.random.randint(100,size=1)[0]
                    example = f.create_group(f'example_{i}')
                    image_feature = example.create_group('image_feature')
                    
                    # Image 1
                    image1 = image_feature.create_group('image1')
                    image1.create_dataset('image', data=im1)  # Replace with actual image data
                    
                    boundingbox_feature1 = image1.create_group('boundingbox_feature')
                    
                    bb1 = boundingbox_feature1.create_group('bb1')
                    bb1.create_dataset('bbox', data=bb1)  # Replace with actual bbox
                    
                    image_1_feature = bb1.create_group('image_1_feature')
                    image_1_feature_ds = image_1_feature.create_dataset('image', data=im4)  # Replace with actual image data
                    image_1_feature_ds.attrs["type"] = "leaf_image"
                    
                    bb2 = boundingbox_feature1.create_group('bb2')
                    bb2.create_dataset('bbox', data=bb2)  # Replace with actual bbox
                    
                    bb3 = boundingbox_feature1.create_group('bb3')
                    bb3.create_dataset('bbox', data=bb3)  # Replace with actual bbox
                    
                    label_feature = bb3.create_group('label_feature')
                    label_feature.create_dataset('label', data=label1)
                    
                    # Image 2
                    image2 = image_feature.create_group('image2')
                    image2.create_dataset('image', data=im2)  # Replace with actual image data
                    
                    boundingbox_feature2 = image2.create_group('boundingbox_feature')
                    
                    bb4 = boundingbox_feature2.create_group('bb4')
                    bb4.create_dataset('bbox', data=bb4)  # Replace with actual bbox
                    
                    bb5 = boundingbox_feature2.create_group('bb5')
                    bb5.create_dataset('bbox', data=bb5)  # Replace with actual bbox
                    
                    # Image 3
                    image3 = image_feature.create_group('image3')
                    image3.create_dataset('image', data=im3)  # Replace with actual image data
                    
                    text_1_feature = image3.create_group('text_1_feature')
                    text_1_feature.create_dataset('text', data=text2)
                    
                    # Text feature
                    text_feature = example.create_group('text_feature')
                    text_feature.create_dataset('text', data=text1)


            table = pa.Table.from_pylist(output)
            df = table.to_pandas()
            df.to_parquet(name+".parquet")


    def benchmark(self,
                  path : str, 
                  N : List[int],
                  iterations : int,
                  hdf5_driver : str = None,
                  plot : bool = False,
                  save : bool = False):
         
        t_load_hdf5 = []
        t_manipulate_hdf5 = []

        t_load_arrow = []
        t_manipulate_arrow = []

        for item in N:

            tmp_load_hdf5 = []
            tmp_manipulate_hdf5 = []

            tmp_load_arrow = []
            tmp_manipulate_arrow = []

            
            for j in tqdm(range(iterations)):
                ### LOADING ###
                st_time_arrow = time.time()
                table = pq.read_table(f'{path}_{item}.parquet')
                en_time_arrow = time.time()
                tmp_load_arrow.append(en_time_arrow - st_time_arrow)

                ### MANIPULATION ###
                start_time_arrow = time.time()
                oggetti = np.array(table.column("image_feature").chunk(0).values.field("boundingbox_feature").values.field("image_1_feature").values.field("image").to_pylist())
                image_numpy = np.frombuffer(oggetti, dtype=np.float64).reshape(-1, 3, 125, 125)
                new_obj = np.transpose(image_numpy,axes=(0,3,2,1))   
                new_obj = np.square(new_obj)
                new_obj = np.exp(new_obj)
                new_obj = np.transpose(new_obj,axes=(0,3,2,1))
                end_time_arrow = time.time()
                tmp_manipulate_arrow.append(end_time_arrow-start_time_arrow)

                ### LOADING ###
                st_time_hdf5 = time.time()
                with h5py.File(f'{path}_{item}_{hdf5_driver}.h5', 'r', driver=hdf5_driver) as f:
                    en_time_hdf5 = time.time()
                    tmp_load_hdf5.append(en_time_hdf5 - st_time_hdf5)
                    
                    ### MANIPULATION ###
                    def get_all_image_datasets():
                        image_datasets = []                
                        def visit_func(name, obj):
                            if isinstance(obj, h5py.Dataset) and ("type" in obj.attrs) and obj.attrs["type"] == "leaf_image":
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

            t_load_arrow.append(sum(tmp_load_arrow) / len(tmp_load_arrow))
            t_manipulate_arrow.append(sum(tmp_manipulate_arrow) / len(tmp_manipulate_arrow))

            t_load_hdf5.append(sum(tmp_load_hdf5) / len(tmp_load_hdf5))
            t_manipulate_hdf5.append(sum(tmp_manipulate_hdf5) / len(tmp_manipulate_hdf5))

        if plot:
            plt.title("HDF5 vs Arrow Loading")
            plt.plot(N, t_load_hdf5, label="hdf5", color='blue')

            # Create the second plot
            plt.plot(N, t_load_arrow, label="arrow", color='red')

            # Add a legend
            plt.legend()

            # Add titles and labels
            plt.xlabel("N (number of samples)")
            plt.ylabel("t (seconds)")

            # Show the plot
            plt.savefig(f'{path}_{hdf5_driver}_load.pdf')  # Save as PDF
            #plt.show()

            plt.title("HDF5 vs Arrow Manipulating")
            plt.plot(N, t_manipulate_hdf5, label="hdf5", color='blue')

            # Create the second plot
            plt.plot(N, t_manipulate_arrow, label="arrow", color='red')

            # Add a legend
            plt.legend()

            # Add titles and labels
            plt.xlabel("N (number of samples)")
            plt.ylabel("t (seconds)")

            # Show the plot
            plt.savefig(f'{path}_{hdf5_driver}_manipulate.pdf')  # Save as PDF
            #plt.show()

        if save:
            np.save(f"{path}_t_load_arrow.npy",t_load_arrow)
            np.save(f"{path}_t_manipulate_arrow.npy",t_manipulate_arrow)
            np.save(f"{path}_t_load_hdf5_{hdf5_driver}.npy",t_load_hdf5)
            np.save(f"{path}_t_manipulate_hdf5_{hdf5_driver}.npy",t_manipulate_hdf5)
            
        return t_load_arrow, t_manipulate_arrow , t_load_hdf5 , t_manipulate_hdf5

