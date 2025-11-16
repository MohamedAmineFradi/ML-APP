import pytest
import numpy as np
import pandas as pd
from data_loader import (
    load_iris_data,
    get_feature_names,
    get_target_names,
    load_iris_as_dataframe,
    get_dataset_info
)
from model import IrisClassifier


class TestDataLoaderOutputs:

    def test_load_iris_data_output_types(self):
        result = load_iris_data()
        
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        X_train, X_test, y_train, y_test = result
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

    def test_load_iris_data_output_dimensions(self):
        X_train, X_test, y_train, y_test = load_iris_data()
        
        assert X_train.ndim == 2
        assert X_test.ndim == 2
        
        assert y_train.ndim == 1
        assert y_test.ndim == 1

    def test_get_feature_names_output(self):
        features = get_feature_names()
        
        assert isinstance(features, list)
        assert len(features) == 4
        
        for feature in features:
            assert isinstance(feature, str)
            assert len(feature) > 0

    def test_get_target_names_output(self):
        targets = get_target_names()
        
        assert isinstance(targets, (list, np.ndarray))
        assert len(targets) == 3
        
        for target in targets:
            assert isinstance(target, (str, np.str_))
            assert len(target) > 0

    def test_load_iris_as_dataframe_output(self):
        df = load_iris_as_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == 150
        
        assert 'target' in df.columns
        assert 'species' in df.columns

