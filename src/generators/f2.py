from .base import SampleGenerator
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import random

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

class F2(SampleGenerator):

    def __init__(self):
        self.texts = [generate_phrase() for i in range(10)]


    def create_dataset(self, N: int, name : str = "data_hierarchy"):
        output = []
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
                        "text_feature":[{
                            "description" : text
                        }]
                    },
                    {
                        "bbox" : bb2,
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

            with h5py.File(name+".h5", 'w', driver="core") as f:
                example = f.create_group(f'example_{i}')

                image_feature = example.create_group('image_feature')
                image = image_feature.create_group('image')
                image.create_dataset('image', data=im)  

                class_feature1 = image.create_group("class_feature")
                class1 = class_feature1.create_group("class1")
                class1.create_dataset("label",data=im_class)

                boundingbox_feature1 = image.create_group('boundingbox_feature')
                bb1 = boundingbox_feature1.create_group('bb1')
                bb2 = boundingbox_feature1.create_group('bb2')
                bb1.create_dataset('bbox', data=bb1)  
                bb2.create_dataset('bbox', data=bb2)

                text_feature = bb1.create_group("text_feature")
                text_feature.create_dataset("description",data=text)

                class_feature2 = bb2.create_group("class_feature")
                class_feature2.create_dataset("label",data=bb_class)



           