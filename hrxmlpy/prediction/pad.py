"""
Test prediction code using PAD ml service
"""
from hrxmlpy.utils.utils import load_sample_model_config_params, apply_anomaly_thresholds
import logging
from hrxmlpy.pipeline.train.trainer import Predicter
from hrxmlpy.interface.hrx.mlconfig import MLConfig


def predict(retrieved_model_objects, model_config_params, df):
    # X_pred = df
    anom_threshold_type, anom_threshold_val = MLConfig.get_anomaly_threshold_details(model_config_params)
    model = retrieved_model_objects["model"]
    scaler_X = retrieved_model_objects["scaler_X"]
    transformer_run_params = {}
    if model_config_params is None:
        model_config_params = load_sample_model_config_params("./parameters/model_config_params_sample_pad.conf")
    logging.info(f"ml config pad params: {model_config_params}")

    predicter = Predicter(transformer_run_params, model_config_params, model, scaler_X)
    logging.info(f"predicter {predicter}")

    emp_ids = df["Employee_Id"]
    df_pred = df.drop(["Employee_Id"], axis=1)
    logging.info(f"df_pred shape: {df_pred.shape}")
    predictions, metrics = predicter.predict(df_pred, emp_ids)
    predictions_reduced = predictions[["Employee_Id", "score", "rank"]]
    logging.info(f"predictions summary {predictions_reduced}")
    logging.info(f"metrics {metrics}")
    predictions["Anomaly_Flag"] = predictions.apply(
        apply_anomaly_thresholds,
        label=None,
        anom_threshold_type=anom_threshold_type,
        anom_threshold_val=anom_threshold_val,
        axis=1,
    )
    cols_to_move = ["Employee_Id", "score", "rank", "Anomaly_Flag"]
    predictions = predictions[cols_to_move + [col for col in predictions.columns if col not in cols_to_move]]
    return predictions


