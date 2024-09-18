import numpy as np
import faiss

class ThresholdedKNN():

    def __init__(self,
                 backend : str = "faiss",
                 threshold : float = 0.5,
                 distance_type : str = "euclid",
                 max_k : int = 5
                 ) -> None:
        self.backend = backend
        self.threshold = threshold
        self.distance_type = distance_type
        self.max_k = max_k



    def __call__(self,
                 dataset1,
                 dataset2,
                 emb1_column_name : str,  ### Sono i path da specificare a livello di configurazione della classe dataset
                 emb2_column_name : str,
                 classes1_column_name : str,
                 classes2_column_name : str
                 ):
        
        emb1 = dataset1[emb1_column_name,"vector"]
        emb2 = dataset2[emb1_column_name,"vector"]

        d1_classes = dataset1[classes1_column_name,"class_index"]
        d2_classes = dataset2[classes2_column_name,"class_index"]

        d1_classes_label = dataset1[classes1_column_name,"class_label"]
        d2_classes_label = dataset2[classes2_column_name,"class_label"]


        outputs = []

        for idx, query_vector in enumerate(emb2):
            output = {}
            if self.backend == "faiss":
                neighbors, classes = self.faiss_find_neighbors_for_vector(query_vector, emb1, self.threshold, self.max_k, d1_classes)
            else:
                raise NotImplementedError
            
            counts = np.bincount(classes)
            
            if neighbors:
                value = {}
                for neighbor in neighbors:
                    value[(d1_classes[neighbor],d1_classes_label[neighbor])] = counts[d1_classes[neighbor]] / len(neighbors)
                
                output[(d2_classes[idx],d2_classes_label[idx])] = value
                outputs.append(output)
            else:
                outputs.append(None)
        return outputs


    

    def faiss_find_neighbors_for_vector(self, 
                                  query_vector, 
                                  all_embeddings, 
                                  threshold, 
                                  k_neigh,
                                  d1_classes,
                                ):
        
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        all_embeddings = np.array(all_embeddings).astype('float32')
        d = all_embeddings.shape[1]
        if self.distance_type == "euclid":
            index = faiss.IndexFlatL2(d)
        else:
            raise NotImplementedError

        index.add(all_embeddings)
    
        distances, indices = index.search(query_vector, k=k_neigh)

        neighbors = []
        neighbors_classes = []
        for idx, dist in zip(indices[0], distances[0]):
            if dist <= threshold**2:
                neighbors.append(idx)
                neighbors_classes.append(d1_classes[idx])

        return neighbors, neighbors_classes
    

knn = ThresholdedKNN(threshold=5, max_k=100)

o = knn(ds2,ds1,"emb_column","emb_column","class_column","class_column")
l = [item for item in o if item]
for item in l:
    print(item)
    