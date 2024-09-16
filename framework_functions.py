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

################# FUNCTION TO REGISTER IN PYARROW COMPUTE ################
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

def add_feature(table : pa.Table,
              feature_list_path: List[str],
              feature_to_add_name : str,
              feature_to_add_values : List[Dict]
              ):
    
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
    new_field = pa.array([item + feature_to_add_values for item in field_tmp])

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
        
def add_struct_field(table : pa.Table,
              feature_list_path: List[str],
              feature_to_manipulate: str,
              new_feature_name : str,
              function_name :str):
    
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
                        feature_list_paths: List[List[str]],
                        feature_list_indexes : List[List[List[int]]] = None,
                        sample_index : int = None
                ):

    sample = {}
    if (not sample_index) and (feature_list_indexes):
      for path,index in zip(feature_list_paths,feature_list_indexes):
        obj = None
        for i, feature_name in enumerate(path):
            if i == 0:
                obj = table.column(feature_name).chunk(0).take(index[i]) 
            else:
                obj = obj.values.field(feature_name).take(index[i])
        sample[str(path[-1])] = obj.to_pylist()

    elif (sample_index) and (not feature_list_indexes):
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