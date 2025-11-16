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


class TestDataFormatValidation:

    def test_load_iris_data_returns_correct_types(self):
        X_train, X_test, y_train, y_test = load_iris_data()
        
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

    def test_load_iris_data_shapes(self):
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)
        
        assert X_train.shape[1] == 4
        assert X_test.shape[1] == 4
        assert len(y_train.shape) == 1
        assert len(y_test.shape) == 1
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]


    def test_load_iris_data_no_missing_values(self):
        X_train, X_test, y_train, y_test = load_iris_data()
        
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()
        assert not np.isnan(y_train).any()
        assert not np.isnan(y_test).any()
        assert not np.isinf(X_train).any()
        assert not np.isinf(X_test).any()

