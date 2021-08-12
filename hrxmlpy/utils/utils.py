import ast
import pandas as pd
import logging
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
