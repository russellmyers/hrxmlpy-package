"""
Test prediction code using TWV ml service
"""
from hrxmlpy.utils.utils import load_sample_model_config_params, apply_anomaly_thresholds
import logging
from hrxmlpy.pipeline.train.trainer import Predicter
from hrxmlpy.interface.hrx.mlconfig import MLConfig


def predict(retrieved_model_objects, model_config_params, df):
    # X_pred = df
    label = MLConfig.get_label(model_config_params)
    anom_threshold_type, anom_threshold_val = MLConfig.get_anomaly_threshold_details(model_config_params)
    model = retrieved_model_objects["model"]
    scaler_X = retrieved_model_objects["scaler_X"]
    transformer_run_params = {}
    if model_config_params is None:
        model_config_params = load_sample_model_config_params("./parameters/model_config_params_sample_twv.conf")
    logging.info(f"ml config twv params: {model_config_params}")

    predicter = Predicter(transformer_run_params, model_config_params, model, scaler_X)
    logging.info(f"predicter {predicter}")

    emp_ids = df["Employee_Id"]
    df_pred = df.drop(["Employee_Id"], axis=1)
    logging.info(f"df_pred shape: {df_pred.shape}")
    predictions, metrics = predicter.predict(df_pred, emp_ids)
    predictions_reduced = predictions[["Employee_Id", label, f"Predicted_{label}"]]
    logging.info(f"predictions summary {predictions_reduced}")
    logging.info(f"metrics {metrics}")
    predictions["Anomaly_Flag"] = predictions.apply(
        apply_anomaly_thresholds,
        label=label,
        anom_threshold_type=anom_threshold_type,
        anom_threshold_val=anom_threshold_val,
        axis=1,
    )
    cols_to_move = ["Employee_Id", label, f"Predicted_{label}", "Anomaly_Flag"]
    predictions = predictions[cols_to_move + [col for col in predictions.columns if col not in cols_to_move]]

    return predictions

if __name__ == '__main__':
    import json
    import sklearn
    import pandas as pd
    from hrxmlpy.interface.hrx.mlconfig import MLConfig
    from hrxmlpy.pipeline.train.trainer import Trainer

    # Test TWV:
    with open('../../tests/data/sample_train_params_twv.json') as json_file:
        train_params = json.load(json_file)
    with open('../../tests/data/sample_model_config_params_twv.json') as json_file:
        model_config_params = json.load(json_file)
    transformed_data = pd.read_csv('../../tests/data/sample_transformed_training_data_twv.csv')
    t = Trainer(train_params, model_config_params)
    train_features, train_labels, test_features, test_labels = t.split(transformed_data.drop(["Employee_Id"], axis=1))
    model, scaler, predictions_train, predictions_test, metrics = t.train(train_features, train_labels, test_features,
                                                                          test_labels)
    print(t)

    p = Predicter(train_params, model_config_params, model, feature_scaler=None)
    print(p)
    transformed_prediction_data = pd.read_csv('../../tests/data/sample_transformed_training_data_twv.csv')
    res = predict({"model": model, "scaler_X": None}, model_config_params, transformed_prediction_data)
    #    predictions, metrics = p.predict(transformed_prediction_data.drop(['Employee_Id'],axis=1),transformed_prediction_data[['Employee_Id']])
    #    print(predictions, metrics)
    print(res)

