"""
Test prediction code using TWV ml service
"""
from pyml.utils.utils import load_sample_model_config_params, apply_anomaly_thresholds
import logging
from pyml.pipeline.train.trainer import Predicter
from pyml.interface.hrx.mlconfig import MLConfig

from data_loader.io import EloiseSQLDatabase
import pandas as pd
import json
from datetime import datetime


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


def prepare_json(row, label):
    json_dict = {"Predicted": row[f"Predicted_{label}"]}
    json_str = json.dumps(json_dict)
    return json_str


def format_pay_period(db, gcc, pay_group, pay_period):
    pay_period_details = db.get_period_details(gcc, pay_group, pay_period=pay_period)
    if pay_period_details.shape[0] == 0:
        return pay_period, pd.Timestamp("20000101")

    formatted_pay_period = f'{pay_group}|{pay_period_details.iloc[0]["Period_Start_Date"].strftime("%m/%d/%Y")} - {pay_period_details.iloc[0]["Period_End_Date"].strftime("%m/%d/%Y")}|Chk1'
    pay_date = pay_period_details.iloc[0]["Payroll_Paydate"]
    return formatted_pay_period, pay_date


def output_db_results(gcc, lcc, payroll_area, pay_period, scan_id, scan_type, model_config_params, predictions):
    """
    Prepare results into dataframe and apply in bulk to db
    """
    ml_service = model_config_params["mlService"]
    logging.info(f"Outputting to db results {gcc} {lcc} {payroll_area} {scan_id} {ml_service}")
    label = MLConfig.get_label(model_config_params)
    db = EloiseSQLDatabase()
    payrun_id = db.get_pay_group_id(gcc, payroll_area)
    logging.info(f"gcc: {gcc} payroll area: {payroll_area}, payrun_id: {payrun_id} int payrun:  {int(payrun_id)}")

    client_id = db.get_client_id(gcc)
    company_id = db.get_company_id_from_gcc_lcc(gcc, lcc)

    formatted_pay_period, pay_date = format_pay_period(db, gcc, payroll_area, pay_period)

    label_variable_id = db.get_variable_id_from_feature_name(gcc, lcc, label)

    df = pd.DataFrame()
    df["Employee_Id"] = predictions["Employee_Id"]
    df["Run_Id"] = int(scan_id)
    df["Model_Id"] = 1
    df["Model_Version_Id"] = 1
    df["Client_Id"] = client_id
    df["PayRun_Id"] = int(payrun_id)
    df["Company_Id"] = company_id
    df["Pay_Group"] = payroll_area
    df["Pay_Period"] = formatted_pay_period
    df["Level_Type"] = "EL"
    df["Variable_Id"] = label_variable_id
    df["Value"] = predictions[label]
    df["Center"] = 0
    df["Delta"] = 0
    df["Anomaly_Flag"] = predictions["Anomaly_Flag"]
    df["Predicted_Value"] = predictions[f"Predicted_{label}"]
    df["Json_Results"] = predictions.apply(prepare_json, label=label, axis=1)
    df["ML_Service"] = ml_service
    df["Paydate"] = pay_date
    df["Run_Type"] = scan_type

    db.add_verification_results(df, scan_id, ml_service, test_only=False)

    return df
