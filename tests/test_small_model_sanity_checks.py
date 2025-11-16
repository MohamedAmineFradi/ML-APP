
import pytest
import numpy as np
from sklearn.metrics import accuracy_score
from data_loader import load_iris_data
from model import IrisClassifier


class TestModelSanityChecks:

    def setup_method(self):
        self.X_train, self.X_test, self.y_train, self.y_test = load_iris_data(random_state=42)
        self.classifier = IrisClassifier(random_state=42)

    def test_model_better_than_random_guessing(self):
        self.classifier.train(self.X_train, self.y_train)
        accuracy, _ = self.classifier.evaluate(self.X_test, self.y_test)
        

        assert accuracy > 0.5, "Model should perform better than random guessing"

    def test_model_perfect_on_training_data(self):
        self.classifier.train(self.X_train, self.y_train)
        train_predictions = self.classifier.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        
        assert train_accuracy > 0.9, "Model should have high training accuracy"

    def test_model_generalizes_to_test_data(self):
        self.classifier.train(self.X_train, self.y_train)
        test_predictions = self.classifier.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        
        assert test_accuracy > 0.85, "Model should generalize well to test data"

