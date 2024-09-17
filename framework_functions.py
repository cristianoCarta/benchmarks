import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import h5py
from PIL import Image
from src.benchmarkers import *
from src.benchmarkersV2 import *
from tqdm import tqdm
import os
import numpy as np
from io import BytesIO
from typing import *
np.random.seed(0)

## TO ADD IN CLASS CONFIGURATION BY ATTRIBUTES ##
TEST_paths = [["image_feature","boundingbox_feature","bbox"]]
TEST_indices = [[[1],[0]]]
##

def draw_tree_schema(data, indent="", is_last=True):
    output = ""
    if isinstance(data, dict):
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            connector = "└── " if is_last and i == len(items) - 1 else "├── "
            output += f"{indent}{connector}{key}\n"
            new_indent = indent + ("    " if is_last and i == len(items) - 1 else "│   ")
            output += draw_tree_schema(value, new_indent, i == len(items) - 1)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            connector = "└── " if is_last and i == len(data) - 1 else "├── "
            output += f"{indent}{connector}[{i}]\n"
            new_indent = indent + ("    " if is_last and i == len(data) - 1 else "│   ")
            output += draw_tree_schema(item, new_indent, i == len(data) - 1)
    return output

def make_offset(vector):
    return [vector[i] - vector[i - 1] for i in range(1, len(vector))]

def group_objects(objects, cardinality_list):
    result = []
    index = 0
    for count in cardinality_list:
        if count == 0:
            result.append(None)
        if count > 0:  # Only create a group if count > 0
            result.append(objects[index:index+count])
            index += count
        if index > len(objects):
            result.append(None)
    return result

################# FUNCTION TO REGISTER IN PYARROW COMPUTE ##################
def area(ctx,array):
    format = pc.list_element(array, 4)
    result = []
    for i in range(40):
        if i == 0 :
            mask = pc.equal(format,i)
            result.append(pc.if_else(mask,pc.multiply(pc.list_element(array,2),pc.list_element(array,3)),None))
        elif i == 1 :
            mask = pc.equal(format,i)
            result.append(pc.if_else(mask,pc.multiply(pc.subtract(pc.list_element(array,3),pc.list_element(array,1)),pc.subtract(pc.list_element(array,2),pc.list_element(array,0))),None))
    final_result = 0
    for i, item in enumerate(result):
        if i+1 < len(result):
            final_result = item.fill_null(result[i+1])
    
    return final_result

"""
func_doc = {}
func_doc["summary"] = "calculate area of bboxes based on various formats"
func_doc["description"] = "calculate bbox area"
func_name = "area"
in_types = {"array": pa.list_(pa.float64())}
out_type = pa.float64()
pc.register_scalar_function(area, func_name, func_doc, in_types, out_type)
"""
#############################################################################

def add_struct_to_feature(table : pa.Table,
              feature_list_path: List[str],
              feature_to_add_name : str,
              feature_to_add_values : List[List[Dict]]
              ):
    
    """
    This function adds a list of structs for every sample of the dataset. 
    Example: in a lenght 3 dataset, i can add a couple of classes for every sample
    passing a list of list of dictionaries, like this:
        [[{"label":0},{"label":1}],[{"label":1},{"label":2}],[{"label":3},{"label":4}]]

    In other words, this function adds cardinality to an existing feature of the dataset by adding one
    or more structs to it.
    """

    obj = None
    to_revert_paths = []

    for indx, feature_name in enumerate(feature_list_path):
        if indx == 0:
            obj = table.column(feature_name).chunk(0)
        else:
            obj = obj.values.field(feature_name)
        
        to_revert_paths.append(obj)

        if indx == len(feature_list_path) - 1:
            obj = obj.values.field(feature_to_add_name)
    
    new_struct = None
    field_tmp = obj.values
    cardinality_list = make_offset(obj.offsets.to_pylist())
    field_tmp = group_objects(field_tmp.to_pylist(), cardinality_list)
    new_field = []
    for i,j in zip(field_tmp,feature_to_add_values):
        if (not i) or (not j):
            new_field.append(None)
        else:
            new_field.append(i+j)
    new_field = pa.array(new_field)

    for indx,level in enumerate(list(reversed(to_revert_paths))):
        if indx!=0:
            new_field = new_struct
            feature_to_add_name = list(reversed(feature_list_path))[indx - 1]

        level_fields_values = [new_field]
        level_fields_names = [feature_to_add_name]

        for x in list(level.values.type):
            if not x.name == feature_to_add_name:
                level_fields_names.append(x.name)
                level_fields_values.append(level.values.field(x.name))    
        new_struct_no_cardinality = pa.StructArray.from_arrays(
            level_fields_values,  
            names=level_fields_names
        )
        cardinality_list = make_offset(level.offsets.to_pylist())
        new_struct = pa.array(group_objects(new_struct_no_cardinality.to_pylist(), cardinality_list))
        
        if indx == len(list(reversed(to_revert_paths))) - 1:
            return table.set_column(table.schema.get_field_index(list(reversed(feature_list_path))[indx]), list(reversed(feature_list_path))[indx], new_struct)

def add_feature(table : pa.Table,
              feature_list_path: List[str],
              new_feature_name : str,
              new_feature : List[List[Dict]],
              ):
    """
    This function permits to add a feature at the level specified by the last string contained in the attribute "feature_list_path"
    Example usage:
    add_feature(table,["image_feature","class_feature"],"boundingbox_feature",[[{"bbox":[0.5,0.5,0.5,0.5,0]},{"bbox":[0.6,0.6,0.6,0.6,1]}] for i in range(len(table))])
    """
    obj = None
    to_revert_paths = []

    for indx, feature_name in enumerate(feature_list_path):
        if indx == 0:
            obj = table.column(feature_name).chunk(0)
        else:
            obj = obj.values.field(feature_name)
        
        to_revert_paths.append(obj)

        #if indx == len(feature_list_path) - 1:
        #    obj = obj.values.field(feature_to_manipulate)

    new_field = pa.array(new_feature)

    new_struct = None

    for indx,level in enumerate(list(reversed(to_revert_paths))):
        if indx!=0:
            new_field = new_struct
            new_feature_name = list(reversed(feature_list_path))[indx - 1]
        level_fields_values = [new_field]
        level_fields_names = [new_feature_name]
        
        for x in list(level.values.type):
            if not x.name == new_feature_name:
                level_fields_names.append(x.name)
                level_fields_values.append(level.values.field(x.name))    
        new_struct_no_cardinality = pa.StructArray.from_arrays(
            level_fields_values,  
            names=level_fields_names
        )
        cardinality_list = make_offset(level.offsets.to_pylist())
        new_struct = pa.array(group_objects(new_struct_no_cardinality.to_pylist(), cardinality_list))
        
        if indx == len(list(reversed(to_revert_paths))) - 1:
            return table.set_column(table.schema.get_field_index(list(reversed(feature_list_path))[indx]), list(reversed(feature_list_path))[indx], new_struct)

def get_feature(table : pa.Table,
                feature_list_path: List[str]
              ):
    """
    This function permits to extract a feature at the location given by the feature_list_path argument
    """
    obj = None
    

    for indx, feature_name in enumerate(feature_list_path):
        if indx == 0:
            obj = table.column(feature_name).chunk(0)
        else:
            obj = obj.values.field(feature_name)

    return obj                      
        
def unary_operation_on_feature(table : pa.Table,
              feature_list_path: List[str],
              feature_to_manipulate: str,
              new_feature_name : str,
              function_name :str):
    """
    This function performs a unary operation on the specified feature and creates another feature at the same level of the specified feature
    Example: computes bounding boxes area and creates a new feature inside every struct of the list of structs under
    the "boundingbox_feature" name
    """
    obj = None
    to_revert_paths = []

    for indx, feature_name in enumerate(feature_list_path):
        if indx == 0:
            obj = table.column(feature_name).chunk(0)
        else:
            obj = obj.values.field(feature_name)
        
        to_revert_paths.append(obj)

        if indx == len(feature_list_path) - 1:
            obj = obj.values.field(feature_to_manipulate)

    new_field = pc.call_function(function_name, [obj])

    new_struct = None

    for indx,level in enumerate(list(reversed(to_revert_paths))):
        if indx!=0:
            new_field = new_struct
            new_feature_name = list(reversed(feature_list_path))[indx - 1]
        level_fields_values = [new_field]
        level_fields_names = [new_feature_name]
        
        for x in list(level.values.type):
            if not x.name == new_feature_name:
                level_fields_names.append(x.name)
                level_fields_values.append(level.values.field(x.name))    
        new_struct_no_cardinality = pa.StructArray.from_arrays(
            level_fields_values,  
            names=level_fields_names
        )
        cardinality_list = make_offset(level.offsets.to_pylist())
        new_struct = pa.array(group_objects(new_struct_no_cardinality.to_pylist(), cardinality_list))
        
        if indx == len(list(reversed(to_revert_paths))) - 1:
            return table.set_column(table.schema.get_field_index(list(reversed(feature_list_path))[indx]), list(reversed(feature_list_path))[indx], new_struct)
        
def get_sample_features(table : pa.Table,
                        sample_index : int,
                        feature_list_paths: List[List[str]],
                        feature_list_indexes : List[List[List[int]]] = None,
                ):

    sample = {}
    if feature_list_indexes:
      for item in feature_list_indexes:
        item.insert(0,[sample_index])
      for path,index in zip(feature_list_paths,feature_list_indexes):
        obj = None
        for i, feature_name in enumerate(path):
            if i == 0:
                obj = table.column(feature_name).chunk(0).take(index[i]) 
            else:
                obj = obj.values.field(feature_name).take(index[i])
        sample[str(path[-1])] = obj.to_pylist()

    elif not feature_list_indexes:
      for path in feature_list_paths :
        obj = None
        for i, feature_name in enumerate(path):
            if i == 0:
                obj = table.column(feature_name).chunk(0).take([sample_index])   
            else:
                obj = obj.values.field(feature_name)
        sample[str(path[-1])] = obj.to_pylist()
    else:
       raise TypeError("either list of indices or single index must be provided")
    return sample

def partition_dataset(
        table: pa.Table,
        save_path : str,
        max_batch_size : int = 300,
        mode : str = "stream",
        return_offsets : bool = False):
    
    batches = table.to_batches(max_batch_size)
    offsets = np.cumsum([0] + [len(b) for b in batches], dtype=np.int64)
    np.save(f"{save_path}/ds_offsets.npy",offsets)
    if mode == "stream":
        for i,batch in enumerate(batches):
            with pa.OSFile(f"{save_path}/ds_{i+1}_of_{len(batches)}_stream.arrows", 'wb') as sink:  
                with pa.ipc.new_stream(sink, batches[0].schema) as writer: 
                    writer.write_batch(batch)  
    elif mode == "file":
        for i,batch in enumerate(batches):
            with pa.OSFile(f"{save_path}/ds_{i+1}_of_{len(batches)}_file.arrow", 'wb') as sink:  
                with pa.ipc.new_file(sink, batches[0].schema) as writer: 
                    writer.write_batch(batch)
    else:
        raise NotImplementedError("mode not implemented")
    
    if return_offsets:
        return offsets
    
def interpolation_search(arr: List[int], x: int) -> int:
    """
    Return the position i of a sorted array so that arr[i] <= x < arr[i+1]

    Args:
        arr (`List[int]`): non-empty sorted list of integers
        x (`int`): query

    Returns:
        `int`: the position i so that arr[i] <= x < arr[i+1]

    Raises:
        `IndexError`: if the array is empty or if the query is outside the array values
    """
    i, j = 0, len(arr) - 1
    while i < j and arr[i] <= x < arr[j]:
        k = i + ((j - i) * (x - arr[i]) // (arr[j] - arr[i]))
        if arr[k] <= x < arr[k + 1]:
            return k
        elif arr[k] < x:
            i, j = k + 1, j
        else:
            i, j = i, k
    raise IndexError(f"Invalid query '{x}' for size {arr[-1] if len(arr) else 'none'}.")

def get_sample(root_path : str,
               sample_index : int,
               all_data_in_memory : bool = False,
               memory_map : bool = False,
               mode : str = "stream"
               ):
    
        offsets = np.load(f"{root_path}/ds_offsets.npy")

        #### ALL DATA IN MEMORY ####

        if all_data_in_memory and mode=="stream" and memory_map:
                with pa.memory_map(f"{root_path}/ds_{1}_of_{len(offsets)-1}_stream.arrows", 'rb') as source:
                        table = pa.ipc.open_stream(source).read_all()
                for i in list(range(2,len(offsets))):
                        with pa.memory_map(f"{root_path}/ds_{i}_of_{len(offsets)-1}_stream.arrows", 'rb') as source:
                                table = pa.concat_tables([table,pa.ipc.open_stream(source).read_all()])
                table = table.combine_chunks()
        
        if all_data_in_memory and mode=="stream" and not memory_map:
                with pa.OSFile(f"{root_path}/ds_{1}_of_{len(offsets)-1}_stream.arrows", 'rb') as source:
                        table = pa.ipc.open_stream(source).read_all()
                for i in list(range(2,len(offsets))):
                        with pa.OSFile(f"{root_path}/ds_{i}_of_{len(offsets)-1}_stream.arrows", 'rb') as source:
                                table = pa.concat_tables([table,pa.ipc.open_stream(source).read_all()])
                table = table.combine_chunks()

        if all_data_in_memory and mode=="file" and memory_map:
                with pa.memory_map(f"{root_path}/ds_{1}_of_{len(offsets)-1}_file.arrow", 'rb') as source:
                        table = pa.ipc.open_file(source).read_all()
                for i in list(range(2,len(offsets))):
                        with pa.memory_map(f"{root_path}/ds_{i}_of_{len(offsets)-1}_file.arrow", 'rb') as source:
                                table = pa.concat_tables([table,pa.ipc.open_file(source).read_all()])
                table = table.combine_chunks()

        if all_data_in_memory and mode=="file" and not memory_map:
                with pa.OSFile(f"{root_path}/ds_{1}_of_{len(offsets)-1}_file.arrow", 'rb') as source:
                        table = pa.ipc.open_file(source).read_all()
                for i in list(range(2,len(offsets))):
                        with pa.OSFile(f"{root_path}/ds_{i}_of_{len(offsets)-1}_file.arrow", 'rb') as source:
                                table = pa.concat_tables([table,pa.ipc.open_file(source).read_all()])
                table = table.combine_chunks()

        #### PARTITIONING ####

        index_file_to_open = interpolation_search(offsets,sample_index)

        if not all_data_in_memory and mode=="stream" and memory_map:
                with pa.memory_map(f"{root_path}/ds_{index_file_to_open+1}_of_{len(offsets)-1}_stream.arrows", 'rb') as source:
                        table = pa.ipc.open_stream(source).read_all()
        
        if not all_data_in_memory and mode=="stream" and not memory_map:
                with pa.OSFile(f"{root_path}/ds_{index_file_to_open+1}_of_{len(offsets)-1}_stream.arrows", 'rb') as source:
                        table = pa.ipc.open_stream(source).read_all()
               
        if not all_data_in_memory and mode=="file" and memory_map:
                with pa.memory_map(f"{root_path}/ds_{index_file_to_open+1}_of_{len(offsets)-1}_file.arrow", 'rb') as source:
                        table = pa.ipc.open_file(source).read_all()
               

        if not all_data_in_memory and mode=="file" and not memory_map:
                with pa.OSFile(f"{root_path}/ds_{index_file_to_open+1}_of_{len(offsets)-1}_file.arrow", 'rb') as source:
                        table = pa.ipc.open_file(source).read_all()

        if not all_data_in_memory:
                sample_index = sample_index-offsets[index_file_to_open]

        
        return get_sample_features(table,sample_index,TEST_paths,feature_list_indexes=TEST_indices)