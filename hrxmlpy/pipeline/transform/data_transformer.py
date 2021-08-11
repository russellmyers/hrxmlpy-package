import numpy as np
import pandas as pd
from sklearn import preprocessing
from hrxmlpy.utils import utils
import logging

# logging.getLogger(__name__)
logger = utils.init_logger(__name__)


class DataTransformer:

    code_version = "0.1a"

    def __init__(self, run_params, model_config_params, scaler=None, encoder=None):
        self.run_params = run_params
        self.model_config_params = model_config_params
        self.scaler = scaler
        self.encoder = encoder

    def get_scaler(self, data):
        if self.scaler is not None:
            logger.info(f"Transformer - Scaler already exists. Using existing scaler: {self.scaler}")
            return self.scaler

        scaler_dict = {
            "standard": preprocessing.StandardScaler(),
            "minmax": preprocessing.MinMaxScaler(),
            "robust": preprocessing.RobustScaler(),
            "normalizer": preprocessing.Normalizer(),
        }
        selected_scaler = scaler_dict.get(self.run_params["scaler_type"])
        numeric_feature_names = utils.get_numeric_features(self.model_config_params)
        found_numeric_feature_names = []
        for feature in numeric_feature_names:
            if feature in data.columns:
                found_numeric_feature_names.append(feature)
            else:
                logger.warning(f"Scaling fit - missing numeric feature: {feature}")

        numeric_features = data[found_numeric_feature_names]
        feature_scaler = selected_scaler.fit(numeric_features)
        self.scaler = feature_scaler
        logger.info(f"Transformer - No previous scaler exists. New scaler fit: {self.scaler}")
        return feature_scaler

    def _get_categorical_features(self, data):
        all_categorical_features = [
            feature["title"] for feature in self.model_config_params["features"] if feature["featureType"] == "C"
        ]
        found_categorical_features = []
        for feature in all_categorical_features:
            if feature in data.columns:
                found_categorical_features.append(feature)
            else:
                logger.warning(f"Transformer - Missing categorical feature: {feature}")

        return all_categorical_features, found_categorical_features

    def get_encoder(self, data):
        if self.encoder is not None:
            logger.info(f"Transformer - One Hot Encoder already exists. Using existing encoder: {self.encoder}")
            return self.encoder

        all_categorical_features, found_categorical_features = self._get_categorical_features(data)

        if len(found_categorical_features) > 0:
            encoder = preprocessing.OneHotEncoder(handle_unknown="ignore")
            encoder.fit(data[found_categorical_features])
            logger.info(f"Transformer - No previous One Hot Encoder exists. New encoder has been fit: {encoder}")
        else:
            logger.info("Transformer - No categorical features found => no OHE encoder created")
            encoder = None
        self.encoder = encoder
        return encoder

    def remove_retros(self, data):
        # if ("Period" in data.columns) and ("ForPeriod" in data.columns):
        if "Retro_Period" in data.columns:
            non_retro_data = data[data["Retro_Period"].isna()]
            num_retros = len(data) - len(non_retro_data)
            logger.info(f"Num retros removed: {num_retros}")
            if num_retros > 0:
                data = non_retro_data
        else:
            logger.info("Retro_Period column not found. No retros removed")
        return data

    def _parse_exclusions(self, data):
        exclusions = {}
        for exclusion_dict in self.model_config_params["exclusions"]:
            rec_num = exclusion_dict["exclusionrecordnum"]
            if rec_num not in exclusions:
                exclusions[rec_num] = []
            if (exclusion_dict["sourcefieldname"] in data.columns) and (
                data[exclusion_dict["sourcefieldname"]].dtype == np.int64
            ):
                field_val = int(exclusion_dict["exclusionvalue"])
            else:
                field_val = exclusion_dict["exclusionvalue"]

            exclusions[rec_num].append((exclusion_dict["sourcefieldname"], field_val))
        return exclusions

    def _apply_exclusion(self, data, exclusion):
        num_recs_before = data.shape[0]
        if len(exclusion) == 1:
            excl_field_name, excl_field_val = exclusion[0]
            if excl_field_name in data.columns:
                data = data[~(data[excl_field_name] == excl_field_val)]
        elif len(exclusion) == 2:
            excl_field_name1, excl_field_val1 = exclusion[0]
            excl_field_name2, excl_field_val2 = exclusion[1]
            if excl_field_name1 in data.columns and excl_field_name2 in data.columns:
                data = data[~(data[excl_field_name1] == excl_field_val1) | ~(data[excl_field_name2] == excl_field_val2)]
        else:
            logger.warning(f"Complex exclusion not supported: {exclusion} - ignoring")
        num_recs_after = data.shape[0]
        logger.info(
            f"Transformer - exclusion {exclusion} applied. Num records removed: {num_recs_before - num_recs_after}"
        )

        return data

    def remove_exclusions(self, data):
        exclusions = self._parse_exclusions(data)
        for exclusion_rec_num, exclusion in exclusions.items():
            data = self._apply_exclusion(data, exclusion)
        return data

    def convert_string_fields(self, data):
        indexes = data.isna().sum().index
        values = data.isna().sum().values
        for k, v in zip(indexes, values):
            if v > 0:
                data[k] = data[k].fillna(0)
                data[k] = data[k].astype(str)
        return data

    def filter_unwanted_columns(self, data):
        features, labels = utils.get_features_and_labels(self.model_config_params)
        id_ = ["Employee_Id"]

        found_features = []
        for feature in features:
            if feature in data.columns:
                found_features.append(feature)
            else:
                logger.warning(f"Missing feature: {feature}")

        found_labels = []
        for label in labels:
            if label in data.columns:
                found_labels.append(label)
            else:
                logger.warning(f"Missing label: {label}")

        data = data[id_ + found_features + found_labels]
        return data

    def auto_select_features(self, data):
        # TODO: Scheduled for next release
        pass

    def one_hot_encode(self, data):

        if self.encoder is None:
            return data

        if self.model_config_params["learningAlgorithm"] == "enhanced_isolation_forest":
            logger.info(
                "Transformer - Bypassing One Hot Encoding: Enhanced Isolation Forest caters for Categorical features directly"
            )
            return data

        all_categorical_features, found_categorical_features = self._get_categorical_features(data)
        if len(found_categorical_features) < len(all_categorical_features):
            missing_categorical_features = [x for x in all_categorical_features if x not in found_categorical_features]
            for missing_feature in missing_categorical_features:
                logger.warning(f"Transformer - Missing categorical feature: {missing_feature}")

        if len(found_categorical_features) > 0:
            self.encoder.transform(data[found_categorical_features])
            cat_feat_names = self.encoder.get_feature_names()
            cat_feat_names_expanded = []
            for cat_feat_name in cat_feat_names:
                feat_num, feat_val = cat_feat_name[1:].split("_")
                cat_feat_names_expanded.append(found_categorical_features[int(feat_num)] + "_" + feat_val)
            ohe_data = pd.DataFrame(
                self.encoder.transform(data[found_categorical_features]).toarray(), columns=cat_feat_names_expanded
            )
            _, labels = utils.get_features_and_labels(self.model_config_params)
            non_label_columns = [x for x in data.columns if x not in labels]
            ohe_data = pd.concat([data[non_label_columns].reset_index(drop=True), ohe_data], axis=1)
            ohe_data.drop(found_categorical_features, axis=1, inplace=True)
            ohe_data[labels] = data[labels].copy()  # Ensure label stays at end
        else:
            ohe_data = data

        return ohe_data

    def scale(self, data):
        scaled_data = data.copy()
        numeric_feature_names = utils.get_numeric_features(self.model_config_params)
        found_numeric_feature_names = []
        for feature in numeric_feature_names:
            if feature in data.columns:
                found_numeric_feature_names.append(feature)
            else:
                logger.warning(f"Scaling transform - missing numeric feature: {feature}")

        if len(found_numeric_feature_names) == 0:
            return data

        numeric_features = data[found_numeric_feature_names]
        scaled_values = self.scaler.transform(numeric_features)
        for i, feature in enumerate(numeric_features.columns):
            scaled_data[feature] = scaled_values[:, i]
        return scaled_data

    def transform(self, data):
        transformer_dict = {
            "PAD": PayrollAnomalyDataTransformer(
                self.run_params, self.model_config_params, scaler=self.scaler, encoder=self.encoder
            ),
            "TWV": TaxAnomalyDataTransformer(
                self.run_params, self.model_config_params, scaler=self.scaler, encoder=self.encoder
            ),
        }
        selected_transformer = transformer_dict.get(self.run_params["ml_service_code"])
        data = selected_transformer.transform(data)
        self.scaler = selected_transformer.scaler
        self.encoder = selected_transformer.encoder
        return data


class TaxAnomalyDataTransformer(DataTransformer):
    def __init__(self, run_params, model_config_params, scaler=None, encoder=None):
        super(TaxAnomalyDataTransformer, self).__init__(run_params, model_config_params, scaler=scaler, encoder=encoder)

    def transform(self, data):
        data = self.remove_retros(data)
        data = self.remove_exclusions(data)
        data = self.filter_unwanted_columns(data)
        data = self.convert_string_fields(data)
        scaler = self.get_scaler(data)
        logger.info(f"fitted scaler: {scaler}")
        data = self.scale(data)
        encoder = self.get_encoder(data)
        data = self.one_hot_encode(data)
        return data


class PayrollAnomalyDataTransformer(DataTransformer):
    def __init__(self, run_params, model_config_params, scaler=None, encoder=None):
        super(PayrollAnomalyDataTransformer, self).__init__(
            run_params, model_config_params, scaler=scaler, encoder=encoder
        )

    def transform(self, data):
        data = self.remove_retros(data)
        data = self.remove_exclusions(data)
        data = self.filter_unwanted_columns(data)
        data = self.convert_string_fields(data)
        scaler = self.get_scaler(data)
        logger.info(f"fitted scaler: {scaler}")
        data = self.scale(data)
        encoder = self.get_encoder(data)
        data = self.one_hot_encode(data)
        return data
