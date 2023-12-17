import os
import yaml
import joblib
import pandas as pd
from tqdm import tqdm

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

