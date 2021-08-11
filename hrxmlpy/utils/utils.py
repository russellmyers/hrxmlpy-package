import ast
import pandas as pd
import logging
from data_loader.blobstore import EloiseDevBlobService
from os import listdir
from os.path import isfile, join
import joblib
from hrxmlpy.interface.hrx.mlconfig import MLConfig


def get_features_and_labels(model_config_params):
    features = [feature["title"] for feature in model_config_params["features"]]
    label = [field["title"] for field in model_config_params["labels"]]
    return features, label


def get_numeric_features(model_config_params):
    numeric_features = [
        feature["title"] for feature in model_config_params["features"] if feature["featureType"] == "N"
    ]
    return numeric_features


def __load_hyper_params():
    file = open("./parameters/hyper_params.conf", "r")
    contents = file.read()
    params = ast.literal_eval(contents)
    file.close()
    return params


def load_train_params():
    file = open("./parameters/train_params.conf", "r")
    contents = file.read()
    params = ast.literal_eval(contents)
    file.close()
    hyper_params = __load_hyper_params()
    params["hyper_params"] = hyper_params
    return params


def load_run_params():
    file = open("./parameters/run_params.conf", "r")
    contents = file.read()
    params = ast.literal_eval(contents)
    file.close()
    return params


def load_sample_model_config_params(filename):
    file = open(filename, "r")
    contents = file.read()
    params = ast.literal_eval(contents)
    file.close()
    return params


def conv_value_title_list_to_dict(value_title_list):
    out_dict = {}
    for row in value_title_list:
        out_dict[row["title"]] = row["value"]
    return out_dict


def pulled_json_to_df(j_data, use_value_title_format=False):

    if use_value_title_format:
        sel_dict = conv_value_title_list_to_dict(j_data["selection"])
    else:
        sel_dict = j_data["selection"]
    sel_keys = list(sel_dict.keys())
    sel_vals = list(sel_dict.values())

    num_sel_fields = len(sel_keys)

    if use_value_title_format:
        first_rec_dict = conv_value_title_list_to_dict(j_data["values"][0])
    else:
        first_rec_dict = j_data["values"][0]
    val_keys = list(first_rec_dict.keys())

    num_val_fields = len(val_keys)

    out_list = []

    for i, emp_entry in enumerate(j_data["values"]):
        # out_entry = sel_vals[:]
        out_entry = []
        if use_value_title_format:
            for row in emp_entry:

                out_entry.append(row["value"])
        else:
            for v_key in val_keys:
                out_entry.append(emp_entry[v_key])

        out_list.append(out_entry)

    # df = pd.DataFrame(out_list,columns = sel_keys + val_keys)

    df = pd.DataFrame(out_list, columns=val_keys)

    return df


def init_logger(name, level=logging.INFO):
    # TODO - Use environment variable for logging level
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging.getLogger("azure.storage.common.storageclient").setLevel(logging.WARNING)
    return logger


def collect_local_transformed_training_data(model_config_params, local_folder, specific_file_name=None):
    """
    For lab use:
    Collect all available transformed training data for ml service/model from local folder.
    If specific_file_name supplied, only retrieve specific file
    """

    ml_service_code, ml_model_code, ml_model_version = (
        model_config_params["mlService"],
        model_config_params["code"],
        model_config_params["version"],
    )

    path = f"{local_folder}/{ml_service_code}/{ml_model_code}/{str(ml_model_version).zfill(3)}/"
    filenames = [f for f in listdir(path) if isfile(join(path, f))]

    logger.info(
        f"Collecting training data for ml service: {ml_service_code}, model code: {ml_model_code}, model version: {ml_model_version}. Collecting from folder: {path}"
    )

    li = []

    if specific_file_name is not None:
        df = pd.read_csv(f"{path}/{specific_file_name}", index_col=None, header=0)
        return df, [f"{path}/{specific_file_name}"]
    for filename in filenames:
        df = pd.read_csv(f"{path}/{filename}", index_col=None, header=0)
        li.append(df)
    if len(li) == 0:
        logger.warning(f"No training data found in folder: {path}")
        return None, []
    df_all = pd.concat(li, axis=0, ignore_index=True)
    return df_all, [path + "/" + filename for filename in filenames]


def collect_transformed_training_data(model_config_params, local_input_folder=None, input_file_name=None):
    """
    For lab use:
     Collect all available transformed training data for ml service/model (or specific file only if specific_file_name supplied)
     If local_input_folder specified: collect from local folder, otherwise collect from blobstore
    """

    if local_input_folder is None:
        logger.info("Retrieving transformed data for training from blobstore")
        dev_blob_service = EloiseDevBlobService()
        (
            transformed_training_data,
            files_included,
        ) = dev_blob_service.collect_all_training_data(model_config_params, specific_file_name=input_file_name)
    else:
        logger.info(f"Retrieving transformed data for training from local folder {local_input_folder}")
        transformed_training_data, files_included = collect_local_transformed_training_data(
            model_config_params,
            local_input_folder,
            specific_file_name=input_file_name,
        )

    return transformed_training_data, files_included


def collect_model(model_config_params, local_model_folder=None):
    """
    For lab use:
     Collect model objects
     If local_model_folder specified: collect from local folder, otherwise collect from blobstore
    """
    ml_service = model_config_params["mlService"]
    model_code = model_config_params["code"]
    model_version = str(model_config_params["version"]).zfill(3)
    model_id = ml_service + "_" + model_code + "_" + model_version

    if local_model_folder is None:
        dev_blob_service = EloiseDevBlobService()
        all_model_objects = dev_blob_service.retrieve_model_objects(ml_service, model_code, model_version)
    else:
        full_path_in = join(local_model_folder, model_id + "_MODEL.joblib")
        try:
            model_objects = joblib.load(full_path_in)
        except:
            model_objects = {"model": None, "model_metadata": None}
        try:
            full_path_in = join(local_model_folder, model_id + "_SCALER.joblib")
            scaler = joblib.load(full_path_in)
        except:
            scaler = None
        try:
            full_path_in = join(local_model_folder, model_id + "_ENCODER.joblib")
            encoder = joblib.load(full_path_in)
        except:
            encoder = None

        all_model_objects = {
            "model": model_objects["model"],
            "model_metadata": model_objects["model_metadata"],
            "scaler_X": scaler,
            "encoder": encoder,
        }

    return all_model_objects


def apply_anomaly_thresholds(row, label=None, anom_threshold_type=None, anom_threshold_val=None):
    if label is not None:
        act_col = label
        pred_col = f"Predicted_{label}"

    anom_flag = 0
    if anom_threshold_type == MLConfig.ANOMALY_THRESHOLD_TYPE_NONE:
        pass
    elif anom_threshold_type == MLConfig.ANOMALY_THRESHOLD_TYPE_ABSOLUTE:
        if label is None:
            anom_flag = 0
        else:
            anom_flag = 1 if abs(row[pred_col] - row[act_col]) > anom_threshold_val else 0
    elif anom_threshold_type == MLConfig.ANOMALY_THRESHOLD_TYPE_PERCENTAGE:
        if label is None:
            anom_flag = 0
        else:
            perc_diff = 0 if row[act_col] == 0 else abs((row[pred_col] - row[act_col]) / row[act_col]) * 100.0
            anom_flag = 1 if perc_diff > anom_threshold_val else 0
    elif anom_threshold_type == MLConfig.ANOMALY_THRESHOLD_TYPE_RANK:
        if "rank" in row:
            anom_flag = 1 if row["rank"] <= anom_threshold_val else 0
        else:
            anom_flag = 0
    return anom_flag


logger = init_logger(__name__)
