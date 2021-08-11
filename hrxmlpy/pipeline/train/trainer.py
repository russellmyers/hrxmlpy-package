from hrxmlpy.algorithms.isolation_forest import EnhancedIsolationForest
from hrxmlpy.algorithms.sklearn_decision_forest_analyzer import SKDecisionForestAnalyser
from hrxmlpy.utils import utils

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from datetime import datetime


class Trainer:

    code_version = "0.1a"

    def __init__(self, train_params, model_config_params):
        self.train_params = train_params
        self.model_config_params = model_config_params

    def split(self, data):
        train_features = data
        if self.train_params["test_size"] > 0:
            train_features = data.sample(
                frac=1 - self.train_params["test_size"], random_state=self.train_params["random_state"]
            )
        test_features = data.drop(train_features.index)
        label = [field["title"] for field in self.model_config_params["labels"]]
        train_labels, test_labels = None, None
        if label:
            train_labels = train_features.pop(label[0])
            test_labels = test_features.pop(label[0])
        return train_features, train_labels, test_features, test_labels

    def get_scaler(self, features):
        scaler_dict = {
            "standard": preprocessing.StandardScaler(),
            "minmax": preprocessing.MinMaxScaler(),
            "robust": preprocessing.RobustScaler(),
            "normalizer": preprocessing.Normalizer(),
        }
        scaled_features, scaled_labels = None, []
        selected_scaler = scaler_dict.get(self.train_params["scaler"])
        feature_scaler = selected_scaler.fit(features)
        return feature_scaler

    def get_estimator(self):
        algorithm = self.model_config_params["learningAlgorithm"]
        estimator_dict = {
            "scikit_neural_network_regressor": MLPRegressor(),
            "scikit_isolation_forest": IsolationForest(),
            "enhanced_isolation_forest": EnhancedIsolationForest(),
        }
        estimator = estimator_dict.get(algorithm)
        return estimator

    def collect_training_metadata(self, model, filenames, metrics, train_X, train_y, test_X, test_y):

        shapes = ["N/A" if x is None else x.shape for x in [train_X, train_y, test_X, test_y]]
        shapes_info = f"train_X: {shapes[0]} train_y: {shapes[1]} test_X: {shapes[2]} test_y: {shapes[3]}"

        metadata = {
            "training_date": datetime.now(),
            "datasets_used": filenames,
            "train_params": self.train_params,
            "full_hyper_params": model.get_params(),
            "train_shapes": shapes_info,
            "metrics": metrics,
            "model_config_params": self.model_config_params,
        }
        return metadata

    def train(self, train_X, train_y=None, test_X=None, test_y=None):
        estimator = self.get_estimator()
        hyper_params = {
            param: val
            for param, val in self.train_params["hyper_params"].items()
            if not param in ["AnomalyThresholdType", "AnomalyThresholdVal"]
        }  # Exclude anomaly threshold params, used in prediction only
        estimator.set_params(**hyper_params)
        metrics = None
        predictions_test = None
        if self.model_config_params["mlType"] == "unsupervised":
            model, scaler, predictions_train = self.__unsupervised_training(estimator, train_X)
        else:
            model, scaler, predictions_train, predictions_test, metrics = self.__supervised_training(
                estimator, train_X, train_y, test_X, test_y
            )
        return model, scaler, predictions_train, predictions_test, metrics

    def __supervised_training(self, estimator, train_X, train_y, test_X, test_y):
        # feature scaling
        # feature_scaler = self.get_scaler(train_X)
        # scaled_train_X = feature_scaler.transform(train_X)
        # scaled_test_X = feature_scaler.transform(test_X)
        feature_scaler = None
        scaled_train_X = train_X.copy()
        scaled_test_X = test_X.copy()
        # estimate the function and predict
        model = estimator.fit(scaled_train_X, train_y)

        metrics = {"score": model.score(scaled_test_X, test_y), "score_train": model.score(scaled_train_X, train_y)}
        # add actual and predicted values to the test dataframe
        features, label = utils.get_features_and_labels(self.model_config_params)

        y_hat_test = model.predict(scaled_test_X)
        predictions_test = test_X.copy()
        predictions_test[label[0]] = test_y
        predictions_test[f"Predicted_{label[0]}"] = y_hat_test

        y_hat_train = model.predict(scaled_train_X)
        predictions_train = train_X.copy()
        predictions_train[label[0]] = train_y
        predictions_train[f"Predicted_{label[0]}"] = y_hat_train

        return model, feature_scaler, predictions_train, predictions_test, metrics

    def __unsupervised_training(self, estimator, train_X):
        # feature_scaler = self.get_scaler(train_X)
        # scaled_train_X = feature_scaler.transform(train_X)
        feature_scaler = None
        scaled_train_X = train_X.copy()
        # NOTE - scaled features needs to be transformed to a pandas dataframe
        # due to dataframe specific implementation of EnhancedIsolationForest
        scaled_train_X = pd.DataFrame(scaled_train_X, index=train_X.index, columns=train_X.columns)
        estimator.fit(scaled_train_X)
        # FIXME - Algorithm specific actions violates the idea of framework
        if self.model_config_params["learningAlgorithm"] == "enhanced_isolation_forest":
            importance_df = estimator.determine_important_features(scaled_train_X, train_X, 3, 20, True)
        else:
            features, _ = utils.get_features_and_labels(self.model_config_params)
            tree_analyser = SKDecisionForestAnalyser(estimator, features)
            importance_df = tree_analyser.determine_important_features(scaled_train_X, train_X, 3, 20, True)
        scores = estimator.decision_function(scaled_train_X)
        scores_df = train_X.copy()
        scores_df["score"] = scores
        scores_df = scores_df.sort_values(by="score")
        scores_df["rank"] = list(range(1, len(scores_df) + 1))
        predictions_train = importance_df.merge(scores_df)
        return estimator, feature_scaler, predictions_train


class Predicter:
    def __init__(self, train_params, model_config_params, model, feature_scaler):
        self.train_params = train_params
        self.model_config_params = model_config_params
        self.model = model
        self.feature_scaler = feature_scaler

    def split(self, data):
        predict_features = data
        predict_labels = None
        label = [field["title"] for field in self.model_config_params["labels"]]
        if label:
            predict_labels = predict_features.pop(label[0])
        return predict_features, predict_labels

    def get_scaler(self):
        return self.scaler_X

    @property
    def is_unsupervised(self):
        return self.model_config_params["mlType"] == "unsupervised"

    def _unsupervised_predict(self, predict_X_scaled, emp_ids):
        if self.model_config_params["learningAlgorithm"] == "enhanced_isolation_forest":
            importance_df = self.model.determine_important_features(predict_X_scaled, predict_X_scaled, 3, 20, True)
        else:
            features, _ = utils.get_features_and_labels(self.model_config_params)
            tree_analyser = SKDecisionForestAnalyser(self.model, features)
            importance_df = tree_analyser.determine_important_features(predict_X_scaled, predict_X_scaled, 3, 20, True)

        scores_df = importance_df.copy()
        scores = self.model.decision_function(predict_X_scaled)
        scores_df["Employee_Id"] = emp_ids
        scores_df["score"] = scores
        scores_df = scores_df.sort_values(by="score")
        scores_df["rank"] = list(range(1, len(scores_df) + 1))

        return scores_df

    def _supervised_predict(self, predict_X_scaled, predict_y, emp_ids):
        y_hat = self.model.predict(predict_X_scaled)
        metrics = {"score": self.model.score(predict_X_scaled, predict_y)}
        # add actual and predicted values to the test dataframe
        features, label = utils.get_features_and_labels(self.model_config_params)
        predictions = predict_X_scaled.copy()
        predictions["Employee_Id"] = emp_ids
        predictions[label[0]] = predict_y
        predictions[f"Predicted_{label[0]}"] = y_hat

        return predictions, metrics

    def predict(self, data, emp_ids):

        predict_X, predict_y = self.split(data)
        # predict_X_scaled = self.feature_scaler.transform(predict_X)
        predict_X_scaled = predict_X.copy()
        predict_X_scaled = pd.DataFrame(predict_X_scaled, index=predict_X.index, columns=predict_X.columns)

        metrics = None
        if self.is_unsupervised:
            predictions = self._unsupervised_predict(predict_X_scaled, emp_ids)
        else:
            predictions, metrics = self._supervised_predict(predict_X_scaled, predict_y, emp_ids)

        return predictions, metrics
