import os
import yaml
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def load_params(path: str) -> dict:
    with open(path, 'r') as file:
        params = yaml.safe_load(file)
    return params

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

def serialize_data(data: any, path: str) -> None:
    joblib.dump(data, path)
    print("Serialized {}".format(path))

def deserialize_data(path: str) -> any:
    data = joblib.load(path)
    return data

def split_predictor_target(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    data = data.copy(deep=True)
    x = data.drop(columns=params["target"]).copy()
    y = data[params["target"]].copy()
    return x, y

def combine_dataframe(list_dataframe: list, axis: int = 0) -> pd.DataFrame:
    combined_dataframe = pd.DataFrame()
    for dataframe in list_dataframe:
        combined_dataframe = pd.concat([combined_dataframe, dataframe], axis=axis)
    return combined_dataframe

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
    
    result = pd.DataFrame(result, columns=scaler_object.feature_names_in_.tolist())
    result.index = target_dataframe.index

    target_dataframe.drop(columns=columns_name, inplace=True)
    result = combine_dataframe([result, target_dataframe], axis=1)

    return result

def fit_transform_scaler(scaler_object, target_dataframe, columns_name, scaler_object_path):
    target_dataframe = target_dataframe.copy(deep=True)
    scaler_object = fit_scaler(target_dataframe, columns_name, scaler_object)
    serialize_data(scaler_object, scaler_object_path)
    target_dataframe = transform_using_scaler(target_dataframe, columns_name, scaler_object_path)

    return target_dataframe, scaler_object

