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
from sklearn import pipeline
from sklearn import compose
from sklearn.base import BaseEstimator, TransformerMixin

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
    data = data.copy(deep = True)
    x = data.drop(columns = params["target"]).copy()
    y = data[params["target"]].copy()
    return x, y

def combine_dataframe(list_dataframe: list, axis: int = 0) -> pd.DataFrame:
    combined_dataframe = pd.DataFrame()
    for dataframe in list_dataframe:
        combined_dataframe = pd.concat(
            [combined_dataframe, dataframe],
            axis = axis
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
    target_dataframe = target_dataframe.copy(deep = True)
    scaler_object.fit(target_dataframe[columns_name])
    return scaler_object

def transform_using_scaler(target_dataframe, columns_name, scaler_object_path):
    target_dataframe = target_dataframe.copy(deep = True)
    
    scaler_object = deserialize_data(scaler_object_path)
    result = scaler_object.transform(target_dataframe[columns_name])
    
    result = pd.DataFrame(
        result,
        columns=scaler_object.feature_names_in_.tolist()
    )
    result.set_index(target_dataframe.index, inplace = True)

    target_dataframe.drop(columns = columns_name, inplace = True)
    result = combine_dataframe([result, target_dataframe], axis=1)

    return result

def fit_transform_scaler(scaler_object, target_dataframe, columns_name, scaler_object_path):
    target_dataframe = target_dataframe.copy(deep = True)
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
        columns = list(ohe_model.categories_[0])
    )
    return transformed

def combine_ohetransformed_to_master(master_data: pd.DataFrame, ohe_transformed: pd.DataFrame, column_name: str = None):
    master_data = master_data.copy(deep = True)
    ohe_transformed = ohe_transformed.copy(deep = True)

    ohe_transformed.set_index(master_data.index, inplace = True)
    master_data = pd.concat([ohe_transformed, master_data], axis=1)

    if column_name != None:
        master_data.drop(columns = column_name, inplace = True)

    return master_data

def ohe_transform_combine(ohe_model: OneHotEncoder, master_data: pd.DataFrame, column_name: str = None):
    master_data = master_data.copy(deep = True)

    data = np.array(master_data[column_name].to_list()).reshape(-1, 1)

    transformed = ohe_transform(ohe_model, data)
    master_data = combine_ohetransformed_to_master(master_data, transformed, column_name)

    return master_data

def le_fit(label_data: dict) -> LabelEncoder:
    le_object = LabelEncoder()
    le_object.fit(label_data)
    return le_object

def le_transform(target_label, le_object):
    target_label = target_label.copy(deep = True)
    target_label = le_object.transform(target_label)
    return target_label

#-------------------- iqr based outliers removal
def fit_transform_iqr_outliers_removal(target_dataframe, columns_name):
    target_dataframe = target_dataframe.copy(deep = True)

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
    target_dataframe = target_dataframe.copy(deep = True)

    for current_column_name in iqr_data:
        column_name = current_column_name
        lower_limit = iqr_data[current_column_name]["lower_limit"]
        upper_limit = iqr_data[current_column_name]["upper_limit"]

        target_dataframe = target_dataframe[
            (target_dataframe[column_name] > lower_limit) &
            (target_dataframe[column_name] < upper_limit)
        ]
    
    return target_dataframe








class OutliersRemoval():
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y = None):
        X = pd.DataFrame(X)

        self.iqr_data = dict()
        for column in X.columns[5:]:
            q1 = X[column].quantile(0.25)
            q3 = X[column].quantile(0.75)
            iqr = q3 - q1

            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            self.iqr_data[column] = {
                "iqr": iqr,
                "q1": q1,
                "q3": q3,
                "lower_limit": lower_limit,
                "upper_limit": upper_limit
            }

        y = pd.DataFrame(y, columns = [X.shape[1]])

        Xy = pd.concat([X, y], axis = 1)
        
        for column_name in self.iqr_data:
            Xy = Xy[
                (Xy[column_name] > self.iqr_data[column_name]["lower_limit"]) &
                (Xy[column_name] < self.iqr_data[column_name]["upper_limit"])
            ]
        
        X = Xy[Xy.columns[:-1]]
        y = Xy[Xy.columns[-1]]

        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def transform(self, X):
        X = pd.DataFrame(X) 

        for column_name in self.iqr_data:
            X = X[
                (X[column_name] > self.iqr_data[column_name]["lower_limit"]) &
                (X[column_name] < self.iqr_data[column_name]["upper_limit"])
            ]
        
        X = np.array(X)

        return X
    
class IamHere(BaseEstimator, TransformerMixin):
    def __init__(self, path):
        self.path = path

    def fit(self, X, y=None):
        y_path = self.path[:4] + "y_" + self.path[4:]
        joblib.dump(X, self.path)
        joblib.dump(y, y_path)
        return self
    
    def transform(self, X):
        return X
    
from sklearn.utils.validation import check_memory
from sklearn.utils import _print_elapsed_time
from sklearn.base import _fit_context, clone
from sklearn.pipeline import _fit_transform_one

class Pipeline(pipeline.Pipeline):
    # Estimator interface

    def _fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )

            if isinstance(X, tuple):    ###### unpack X if is tuple X = (X,y)
                X, y = X

            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X, y

    @_fit_context(
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, **fit_params):
        """Fit the model.

        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)

        if isinstance(Xt, tuple):    ###### unpack X if is tuple X = (X,y)
            Xt, y = Xt

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self