"""
Test prediction code using PAD ml service
"""
from pyml.utils.utils import load_sample_model_config_params, apply_anomaly_thresholds
import logging
from pyml.pipeline.train.trainer import Predicter
from pyml.interface.hrx.mlconfig import MLConfig
from data_loader.io import EloiseSQLDatabase
import pandas as pd
import json


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


def prepare_json(row, inv_variables_map):
    json_dict = {"Rank": row["rank"], "Score": row["score"]}
    important_feature_names = [row["LF1"], row["LF2"], row["LF3"]]
    important_features = [
        {"Feature": int(inv_variables_map[feature_name]), "Value": round(row[feature_name], 3)}
        for feature_name in important_feature_names
    ]
    json_dict["Important_Features"] = important_features
    json_str = json.dumps(json_dict)
    return json_str


def output_db_results(gcc, lcc, payroll_area, scan_id, model_config_params, predictions):
    """
    Prepare results into dataframe and apply in bulk to db
    """
    ml_service = model_config_params["mlService"]
    model_code = model_config_params["code"]
    model_version = str(model_config_params["version"]).zfill(3)
    logging.info(f"Outputting to db results {gcc} {lcc} {payroll_area} {scan_id} {ml_service}")
    db = EloiseSQLDatabase()
    payrun_id = db.get_pay_group_id(gcc, payroll_area)
    logging.info(f"gcc: {gcc} payroll area: {payroll_area}, payrun_id: {payrun_id} int payrun:  {int(payrun_id)}")

    variables_map = db.get_pay_variables_map(db.get_company_group_id_from_gcc_lcc(gcc, lcc))
    inv_variables_map = {v: k for k, v in variables_map.items()}

    df = pd.DataFrame()
    df["Score"] = predictions["score"]
    df["PayRun_Id"] = int(payrun_id)
    df["Scan_Id"] = int(scan_id)
    df["Employee_Id"] = predictions["Employee_Id"]
    df["Model_Version"] = model_code + "-" + str(model_version).zfill(3)
    df["ML_Service"] = ml_service
    df["Anomaly_Flag"] = predictions["Anomaly_Flag"]
    df["Confidence"] = 0.9
    df["Json_Results"] = predictions.apply(prepare_json, inv_variables_map=inv_variables_map, axis=1)
    df["Rank"] = predictions["rank"]

    db.add_broad_spectrum_verification_results(df, scan_id, model_code, model_version, test_only=False)

    return df
