import os
import yaml
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#-------------------- yaml related
def load_yaml(path: str) -> dict:
    with open(path, 'r') as file:
        params = yaml.safe_load(file)
    return params

#-------------------- csv related
def read_many_csv(path: str) -> pd.DataFrame:
    data = pd.DataFrame()
    for file in tqdm(os.listdir(path)):
        loaded_data = pd.read_csv(path + file)
        data = pd.concat([loaded_data, data])
    return data

def read_csv(path: str, in_folder: bool = False) -> pd.DataFrame:
    if in_folder:
        data = pd.read_csv(path)
    else:
        data = read_many_csv(path)
    return data 

#-------------------- serialization related
def serialize_data(data: any, path: str) -> None:
    joblib.dump(data, path)
    print("Serialized {}".format(path))

def deserialize_data(path: str) -> any:
    data = joblib.load(path)
    return data

#-------------------- splitting and combining dataset related
def split_predictor_target(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    data = data.copy(deep=True)
    x = data.drop(columns=params["target"]).copy()
    y = data[params["target"]].copy()
    return x, y

def combine_dataframe(list_dataframe: list, axis: int = 0) -> pd.DataFrame:
    combined_dataframe = pd.DataFrame()
    for dataframe in list_dataframe:
        combined_dataframe = pd.concat(
            [combined_dataframe, dataframe],
            axis=axis
        )
    return combined_dataframe

#-------------------- feature scaler related
def create_standard_scaler_object():
    scaler_object = StandardScaler()
    return scaler_object

def create_minmax_scaler_object():
    scaler_object = MinMaxScaler()
    return scaler_object

def fit_scaler(target_dataframe, columns_name, scaler_object):
    target_dataframe = target_dataframe.copy(deep=True)
    scaler_object.fit(target_dataframe[columns_name])
    return scaler_object

def transform_using_scaler(target_dataframe, columns_name, scaler_object_path):
    target_dataframe = target_dataframe.copy(deep=True)
    
    scaler_object = deserialize_data(scaler_object_path)
    result = scaler_object.transform(target_dataframe[columns_name])
    
    result = pd.DataFrame(
        result,
        columns=scaler_object.feature_names_in_.tolist()
    )
    result.set_index(target_dataframe.index, inplace=True)

    target_dataframe.drop(columns=columns_name, inplace=True)
    result = combine_dataframe([result, target_dataframe], axis=1)

    return result

def fit_transform_scaler(scaler_object, target_dataframe, columns_name, scaler_object_path):
    target_dataframe = target_dataframe.copy(deep=True)
    scaler_object = fit_scaler(target_dataframe, columns_name, scaler_object)
    serialize_data(scaler_object, scaler_object_path)
    target_dataframe = transform_using_scaler(
        target_dataframe,
        columns_name,
        scaler_object_path
    )

    return target_dataframe, scaler_object

#-------------------- data encoding related
def ohe_fit(fit_data: np.array) -> OneHotEncoder:
    ohe_object = OneHotEncoder(
        sparse_output = False
    )
    ohe_object.fit(fit_data)
    return ohe_object

def ohe_transform(ohe_model: OneHotEncoder, data: np.array) -> pd.DataFrame:
    transformed = ohe_model.transform(data)
    transformed = pd.DataFrame(
        transformed,
        columns=list(ohe_model.categories_[0])
    )
    return transformed

def combine_ohetransformed_to_master(master_data: pd.DataFrame, ohe_transformed: pd.DataFrame, column_name: str = None):
    master_data = master_data.copy(deep=True)
    ohe_transformed = ohe_transformed.copy(deep=True)

    ohe_transformed.set_index(master_data.index, inplace=True)
    master_data = pd.concat([ohe_transformed, master_data], axis=1)

    if column_name != None:
        master_data.drop(columns=column_name, inplace=True)

    return master_data

def ohe_transform_combine(ohe_model: OneHotEncoder, master_data: pd.DataFrame, column_name: str = None):
    master_data = master_data.copy(deep=True)

    data = np.array(master_data[column_name].to_list()).reshape(-1, 1)

    transformed = ohe_transform(ohe_model, data)
    master_data = combine_ohetransformed_to_master(master_data, transformed, column_name)

    return master_data

def le_fit(label_data: dict) -> LabelEncoder:
    le_object = LabelEncoder()
    le_object.fit(label_data)
    return le_object

def le_transform(target_label, le_object):
    target_label = target_label.copy(deep=True)
    target_label = le_object.transform(target_label)
    return target_label

#-------------------- iqr based outliers removal
def fit_transform_iqr_outliers_removal(target_dataframe, columns_name):
    target_dataframe = target_dataframe.copy(deep=True)

    iqr_data = dict()
    for column_name in columns_name:
        q1 = target_dataframe[column_name].quantile(0.25)
        q3 = target_dataframe[column_name].quantile(0.75)
        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        iqr_data[column_name] = {
            "iqr": iqr,
            "q1": q1,
            "q3": q3,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit
        }

        target_dataframe = target_dataframe[
            (target_dataframe[column_name] > lower_limit) &
            (target_dataframe[column_name] < upper_limit)
        ]
    
    return target_dataframe, iqr_data

def transform_iqr_outliers_removal(target_dataframe, iqr_data):
    target_dataframe = target_dataframe.copy(deep=True)

    for current_column_name in iqr_data:
        column_name = current_column_name
        lower_limit = iqr_data[current_column_name]["lower_limit"]
        upper_limit = iqr_data[current_column_name]["upper_limit"]

        target_dataframe = target_dataframe[
            (target_dataframe[column_name] > lower_limit) &
            (target_dataframe[column_name] < upper_limit)
        ]
    
    return target_dataframe