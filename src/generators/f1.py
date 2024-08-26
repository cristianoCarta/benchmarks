from .base import SampleGenerator
import numpy as np
import pyarrow as pa
import h5py

class F1(SampleGenerator):

    def create_dataset(self, N: int, name : str = "data_hierarchy.h5"):
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

            with h5py.File(name, 'w') as f:
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
                    image_1_feature.create_dataset('image', data=im4)  # Replace with actual image data
                    
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
            df.to_csv("data.csv")
            df.to_parquet("data.parquet")