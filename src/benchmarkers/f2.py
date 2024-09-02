from .base import SampleGenerator
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import random
from typing import List
import time
from tqdm import tqdm
from matplotlib import pyplot as plt

# FORMATS
# 0 = YOLO
# 1 = XY

# List of words to create phrases from
nouns = ['cat', 'dog', 'tree', 'house', 'car', 'book', 'computer', 'phone', 'coffee', 'mountain']
adjectives = ['happy', 'sad', 'big', 'small', 'red', 'blue', 'green', 'clever', 'brave', 'shy']
verbs = ['runs', 'jumps', 'sleeps', 'eats', 'writes', 'reads', 'sings', 'dances', 'laughs', 'cries']

# Function to generate a random phrase
def generate_phrase():
    adj = random.choice(adjectives)
    noun = random.choice(nouns)
    verb = random.choice(verbs)
    return f"The {adj} {noun} {verb}"

def xyminmax_to_xycenterwh(bbox : List):
    if bbox[4] == 1:  ## ONLY IF XY IS FOUND
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        x_center = bbox[0] + width / 2
        y_center = bbox[1] + height / 2
        return x_center, y_center, width, height
    else:
        return [bbox[0],bbox[1],bbox[2],bbox[3]]

def xycenterwh_to_xyminmax(bbox : List):
    if bbox[4] == 0:  ## ONLY IF YOLO IS FOUND
        x_min = bbox[0] - bbox[2] / 2
        y_min = bbox[1] - bbox[3] / 2
        x_max = bbox[0] + bbox[2] / 2
        y_max = bbox[1] + bbox[3] / 2
        return [x_min, y_min, x_max, y_max]
    else:
        return [bbox[0],bbox[1],bbox[2],bbox[3]]
class F2(SampleGenerator):

    def __init__(self):
        self.texts = [generate_phrase() for i in range(10)]


    def create_dataset(self, N: int, name : str , hdf5_driver : str):
        output = []
        with h5py.File(f"{name}_{hdf5_driver}.h5", 'w', driver="core") as f:
            for i in range(N):
                text = self.texts[int(random.randint(0,9))]
                bb_class = np.random.randint(0,9)

                bb1 = np.array([0.5,0.5,0.5,0.5])  #YOLO format
                                                    #Expected conversion to the other format: [0.25,0.25,0.75,0.75]

                bb2 = np.array((0.0977, 0.2604, 0.8789, 0.7812)) #NOT YOLO format (xmin,ymin,xmax,ymax)
                                                    #Expected conversion to YOLO format: [0.4883, 0.5208, 0.7812, 0.5208]

                im = np.random.rand(3,125,125)
                im_b = im.tobytes()
                im_class = np.random.randint(0,9)
                shape = (3,125,125)

                sample = {
                    "image_feature":[{
                        "image" : im_b,
                        "shape" : shape,
                        "class_feature" : [
                            {
                                "label" : im_class
                            }
                        ],
                        "bounding_box_feature":[{
                            "bbox" : bb1,
                            "format" : 0,
                            "text_feature":[{
                                "description" : text
                            }]
                        },
                        {
                            "bbox" : bb2,
                            "format" : 1,
                            "class_feature":[
                                {
                                    "label" : bb_class
                                }
                            ]
                        }]
                    }
                    ]
                }

                output.append(sample)


                
                example = f.create_group(f'example_{i}')

                image_feature = example.create_group('image_feature')
                image = image_feature.create_group('image')
                image_ds = image.create_dataset('image', data=im)
                image_ds.attrs["type"] = "image"
                image_ds.attrs["label"] = im_class  

            
                boundingbox_feature1 = image.create_group('boundingbox_feature')
                bb1_g = boundingbox_feature1.create_group('bb1')
                bb2_g = boundingbox_feature1.create_group('bb2')
                bb1_d = bb1_g.create_dataset('bbox', data=bb1)  
                bb1_d.attrs["format"] = 0
                bb2_d = bb2_g.create_dataset('bbox', data=bb2)
                bb2_d.attrs["format"] = 1
                bb2_d.attrs["label"] = bb_class
                bb2_d.attrs["type"] = "bbox"
                

                text_feature = bb1_g.create_group("text_feature")
                text_feature.create_dataset("description",data=text)

        
        table = pa.Table.from_pylist(output)
        df = table.to_pandas()
        df.to_parquet(name+".parquet")

    def benchmark_row_wise(self,
                  path : str, 
                  N : List[int],
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
                
                manipulated_images = []
                for k in range(item):
                    if any( x == selected_label for x in table.column("image_feature").chunk(0)[k][0]["class_feature"].values.field("label").to_pylist()):  ## Indice di immagine deve variare nel framework
                        obj = table.column("image_feature").chunk(0)[k][0]["image"]
                        image_numpy = np.frombuffer(obj.as_buffer(), dtype=np.float64).reshape(3, 125, 125)
                        new_obj = np.transpose(image_numpy)   
                        new_obj = np.square(new_obj)
                        new_obj = np.exp(new_obj)
                        new_obj = np.transpose(new_obj)
                        manipulated_images.append(new_obj)
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
                            if isinstance(obj, h5py.Dataset) and ("type" in obj.attrs) and obj.attrs["type"] == "image" and obj.attrs["label"] == selected_label :
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
    
    def benchmark_bounding_boxes_conversion(self,
                                            path : str, 
                                            N : List[int],
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
                
                oggetti = np.array(table.column("image_feature").chunk(0).values.field("bounding_box_feature").values.field("bbox").to_pylist())
                formati = np.array(table.column("image_feature").chunk(0).values.field("bounding_box_feature").values.field("format").to_pylist())
                formati = formati[:,np.newaxis]
                oggetti = np.hstack((oggetti,formati))
                inverted_bboxes = np.apply_along_axis(xycenterwh_to_xyminmax, axis=1, arr=oggetti)
                end_time_arrow = time.time()
                tmp_manipulate_arrow.append(end_time_arrow-start_time_arrow)

                ### LOADING ###
                st_time_hdf5 = time.time()
                with h5py.File(f'{path}_{item}_{hdf5_driver}.h5', 'r', driver=hdf5_driver) as f:
                    en_time_hdf5 = time.time()
                    tmp_load_hdf5.append(en_time_hdf5 - st_time_hdf5)
                    
                    ### MANIPULATION ###
                    def get_all_image_datasets():
                        bbox_datasets = []                
                        def visit_func(name, obj):
                            if isinstance(obj, h5py.Dataset) and ("type" in obj.attrs) and obj.attrs["type"] == "bbox" and obj.attrs["format"] == 0:
                                bbox = obj[:]
                                x_min = bbox[0] - bbox[2] / 2
                                y_min = bbox[1] - bbox[3] / 2
                                x_max = bbox[0] + bbox[2] / 2
                                y_max = bbox[1] + bbox[3] / 2
                                bbox_datasets.append[x_min, y_min, x_max, y_max]
                        f.visititems(visit_func)
                        return np.array(bbox_datasets)
                    start_time_hdf5 = time.time()
                    bbox_ds = get_all_image_datasets()
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
            plt.show()

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
            plt.show()

        if save:
            np.save(f"{path}_t_load_arrow.npy",t_load_arrow)
            np.save(f"{path}_t_manipulate_arrow.npy",t_manipulate_arrow)
            np.save(f"{path}_t_load_hdf5_{hdf5_driver}.npy",t_load_hdf5)
            np.save(f"{path}_t_manipulate_hdf5_{hdf5_driver}.npy",t_manipulate_hdf5)
            
        return t_load_arrow, t_manipulate_arrow , t_load_hdf5 , t_manipulate_hdf5


           