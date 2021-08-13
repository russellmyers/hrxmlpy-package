from hrxmlpy.pipeline.train.trainer import Trainer, Predicter
import hrxmlpy.algorithms.isolation_forest
import json
import pandas as pd
import sklearn
import os
import joblib

def test_trainer_twv():
    with open(TEST_DATA_DIR + '/sample_train_params_twv.json') as json_file:
        train_params = json.load(json_file)
    with open(TEST_DATA_DIR + '/sample_model_config_params_twv.json') as json_file:
        model_config_params = json.load(json_file)
    transformed_data = pd.read_csv(TEST_DATA_DIR + '/sample_transformed_training_data_twv.csv')
    t = Trainer(train_params, model_config_params)
    train_features, train_labels, test_features, test_labels = t.split(transformed_data)
    assert train_features.shape == (97,2)
    assert train_labels.shape == (97,)
    assert test_features.shape == (24,2)
    assert test_labels.shape == (24,)

    model, scaler, predictions_train, predictions_test, metrics = t.train(train_features, train_labels, test_features,
                                                                          test_labels)
    assert isinstance(model, sklearn.neural_network._multilayer_perceptron.MLPRegressor)
    assert model.max_iter == 500
    assert predictions_train.shape == (97,4)
    assert predictions_test.shape == (24,4)

def test_trainer_pad():
    with open(TEST_DATA_DIR + '/sample_train_params_pad.json') as json_file:
        train_params = json.load(json_file)
    with open(TEST_DATA_DIR + '/sample_model_config_params_pad.json') as json_file:
        model_config_params = json.load(json_file)
    transformed_data = pd.read_csv(TEST_DATA_DIR + '/sample_transformed_training_data_pad.csv')
    t = Trainer(train_params, model_config_params)
    train_features, train_labels, test_features, test_labels = t.split(transformed_data)
    assert train_features.shape == (164,6)
    assert train_labels  is None
    assert test_features.shape == (0,6)
    assert test_labels is None

    model, scaler, predictions_train, predictions_test, metrics = t.train(train_features, train_labels, test_features,
                                                                          test_labels)
    assert isinstance(model, hrxmlpy.algorithms.isolation_forest.EnhancedIsolationForest)
    assert predictions_train.shape == (164,14)
    assert predictions_test is None

def test_predicter_twv():
    with open(TEST_DATA_DIR + '/sample_train_params_twv.json') as json_file:
        train_params = json.load(json_file)
    with open(TEST_DATA_DIR + '/sample_model_config_params_twv.json') as json_file:
        model_config_params = json.load(json_file)
    transformed_prediction_data = pd.read_csv(TEST_DATA_DIR + '/sample_transformed_training_data_twv.csv')
    trained_model = joblib.load(TEST_DATA_DIR + '/sample_trained_model_twv.pkl')
    p = Predicter(train_params, model_config_params,trained_model,feature_scaler=None)
    predictions, metrics = p.predict(transformed_prediction_data.drop(["Employee_Id"], axis=1), transformed_prediction_data[['Employee_Id']])

    assert predictions.shape == (121,4)
    assert metrics["score"] == -0.5177288907075985

def test_predicter_pad():
    with open(TEST_DATA_DIR + '/sample_train_params_pad.json') as json_file:
        train_params = json.load(json_file)
    with open(TEST_DATA_DIR + '/sample_model_config_params_pad.json') as json_file:
        model_config_params = json.load(json_file)
    transformed_prediction_data = pd.read_csv(TEST_DATA_DIR + '/sample_transformed_training_data_pad.csv')
    trained_model = joblib.load(TEST_DATA_DIR + '/sample_trained_model_pad.pkl')
    p = Predicter(train_params, model_config_params,trained_model,feature_scaler=None)
    predictions, metrics = p.predict(transformed_prediction_data.drop(["Employee_Id"], axis=1), transformed_prediction_data[['Employee_Id']])

    assert predictions.shape == (164, 14)
    assert predictions.iloc[0]["Employee_Id"] == 1402
    assert round(predictions.iloc[0]["score"], 5) == -0.36439
    assert predictions.iloc[0]["rank"] == 1


TEST_DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

