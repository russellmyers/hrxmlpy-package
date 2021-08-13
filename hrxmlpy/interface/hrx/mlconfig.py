import os
import requests
import json
import pandas as pd
import logging


class MLConfig:
    ANOMALY_THRESHOLD_TYPE_NONE = None
    ANOMALY_THRESHOLD_TYPE_ABSOLUTE = "A"
    ANOMALY_THRESHOLD_TYPE_PERCENTAGE = "P"
    ANOMALY_THRESHOLD_TYPE_RANK = "R"

    DEFAULT_CUSTOMER_ID = "ZZZ"
    DEFAULT_COMPANY_ID = "Z99"
    DEFAULT_AUTH_ENDPOINT = "/auth/tokens"
    DEFAULT_ML_SERVICES_ENDPOINT = "/api/V2/ml-services"
    DEFAULT_MODELS_ENDPOINT = "/api/V2/model-info"
    DEFAULT_CUSTOMER_MODELS_ENDPOINT = "/api/V2/customer-models"

    def __init__(self):
        pass

    @property
    def auth_token_header(self):
        auth_token_header = None
        data = {
            "clientId": os.environ["MLCONFIG_API_CLIENT_ID"],
            "clientSecret": os.environ["MLCONFIG_API_CLIENT_SECRET"],
            "customerId": os.environ.get("MLCONFIG_API_CUSTOMER_ID", MLConfig.DEFAULT_CUSTOMER_ID),
            "companyId": os.environ.get("MLCONFIG_API_COMPANY_ID", MLConfig.DEFAULT_COMPANY_ID)
        }
        url = os.environ["MLCONFIG_API_BASE_URL"] + os.environ.get("MLCONFIG_API_AUTH_ENDPOINT", MLConfig.DEFAULT_AUTH_ENDPOINT)
        try:
            response = requests.post(url, data=data, headers=None)
        except Exception as e:
            logging.error(
                f"{__class__.__name__} - ML Config failed to reach url: {url}. Is ML config api active? Error: {e}"
            )
            raise

        if response.status_code == 200:
            token = eval(response.text)["token"]
            bearer = "Bearer " + token
            auth_token_header = {"Authorization": bearer}
        else:
            logging.error(f'{__class__.__name__} - Auth error for clientId: {os.environ["MLCONFIG_API_CLIENT_ID"]}')
        return auth_token_header

    @staticmethod
    def get(endpoint):
        token = MLConfig().auth_token_header
        response_json_data = {}
        if token:
            request_url = os.environ["MLCONFIG_API_BASE_URL"] + endpoint
            response = requests.get(request_url, headers=token)
            if response.status_code == 200:
                response_json_data = json.loads(response.text)
            else:
                print(f"{__class__.__name__} - {response.status_code} error for endpoint: {endpoint}")
        return response_json_data

    @staticmethod
    def add_ml_service_technical_details(ml_service_details):
        ml_service_tech_details_all = MLConfig().get(os.environ.get("MLCONFIG_API_ML_SERVICES_ENDPOINT", MLConfig.DEFAULT_ML_SERVICES_ENDPOINT))
        for model_config_params in ml_service_details:
            ml_service_tech_details = [
                x for x in ml_service_tech_details_all["items"] if x["code"] == model_config_params["mlService"]
            ][0]
            model_config_params["mlType"] = ml_service_tech_details["mlType"]
            model_config_params["learningAlgorithm"] = ml_service_tech_details["learningAlgorithm"]

    @staticmethod
    def get_ml_model_details(gcc, lcc, payroll_area, country):
        token = MLConfig().auth_token_header
        model_info_endpoint = os.environ.get("MLCONFIG_API_MODELS_ENDPOINT", MLConfig.DEFAULT_MODELS_ENDPOINT)
        response_json_data = {}
        if token:
            request_url = os.environ["MLCONFIG_API_BASE_URL"] + model_info_endpoint
            response = requests.get(
                request_url, headers=token, params={"gcc": gcc, "lcc": lcc, "payrollArea": payroll_area}
            )
            if response.status_code == 200:
                response_json_data = json.loads(response.text)
                MLConfig.add_ml_service_technical_details(response_json_data)
            else:
                logging.error(
                    f"{__class__.__name__} - {response.status_code} error for endpoint: {model_info_endpoint}"
                )
        return response_json_data

    @staticmethod
    def get_all_customers_using_model(ml_service, model_code):
        token = MLConfig().auth_token_header
        customer_models_endpoint = os.environ.get("MLCONFIG_API_CUSTOMER_MODELS_ENDPOINT", MLConfig.DEFAULT_CUSTOMER_MODELS_ENDPOINT)
        response_json_data = {}

        customers = []
        if token:
            request_url = os.environ["MLCONFIG_API_BASE_URL"] + customer_models_endpoint
            response = requests.get(request_url, headers=token)
            if response.status_code == 200:
                response_json_data = json.loads(response.text)
                for cust_model in response_json_data["items"]:
                    if (cust_model["mlService"]["code"] == ml_service) and (cust_model["modelCode"] == model_code):
                        customers.append(
                            {
                                "gcc": cust_model["customer"]["gcc"],
                                "lcc": cust_model["customer"]["lcc"],
                                "payroll_areas": cust_model["payrollAreas"],
                            }
                        )
            else:
                logging.error(
                    f"{__class__.__name__} - {response.status_code} error for endpoint: {customer_models_endpoint}"
                )
        return customers

    @staticmethod
    def get_ml_model_details_by_model(ml_service, model_code):
        all_custs = MLConfig.get_all_customers_using_model(ml_service, model_code)
        if len(all_custs) == 0:
            return {}

        cust = all_custs[0]  # Use first customer to retrieve model info
        model_details = MLConfig.get_ml_model_details(cust["gcc"], cust["lcc"], cust["payroll_areas"][0], "")
        for model_config_params in model_details:
            if model_config_params["mlService"] == ml_service:
                return model_config_params

        return {}

    @staticmethod
    def get_ml_service_details(ml_service_code):
        # TODO: To be updated once the ml-services endpoint is amended
        ml_service_id_map = {"PAD": 1, "TWV": 2}
        token = MLConfig().auth_token_header
        ml_services_endpoint = f"{os.environ.get('MLCONFIG_API_ML_SERVICES_ENDPOINT', MLConfig.DEFAULT_ML_SERVICES_ENDPOINT)}/{ml_service_id_map[ml_service_code]}"
        if token:
            request_url = os.environ["MLCONFIG_API_BASE_URL"] + ml_services_endpoint
            response = requests.get(request_url, headers=token)
            if response.status_code == 200:
                response_json_data = json.loads(response.text)
            else:
                logging.error(
                    f"{__class__.__name__} - {response.status_code} error for endpoint: {ml_services_endpoint}"
                )
        return response_json_data

    @staticmethod
    def get_label(model_config_params):
        """ Get label name"""
        label = None
        try:
            label = model_config_params["labels"][0]["title"]
        except Exception as e:
            logging.error("No label specified in ML Config -  message: {e}")

        return label

    @staticmethod
    def get_anomaly_threshold_details(model_config_params):
        """ Get anom threshold type and anom threshold value"""
        try:
            anom_threshold_type = model_config_params["hyperparameters"]["AnomalyThresholdType"]
            anom_threshold_val = model_config_params["hyperparameters"]["AnomalyThresholdVal"]
        except Exception as e:
            logging.info(f"No anom threshold config specified - all anomaly flags will be set to false. Message: {e}")
            anom_threshold_type = MLConfig.ANOMALY_THRESHOLD_TYPE_NONE
            anom_threshold_val = 0

        return anom_threshold_type, anom_threshold_val


if __name__ == '__main__':
    os.environ["MLCONFIG_API_CLIENT_ID"] = "d7947bf7-3240-435d-8a79-5b2fc05248c7"
    os.environ["MLCONFIG_API_CLIENT_SECRET"] = "DaTbzvauLJyFqDjpmE5nD5Q7t887TBHdZozxxagBsvQsHePXawgpwoTHuWSAydLj"
    os.environ["MLCONFIG_API_BASE_URL"] = "http://localhost:5000"
    ml_config = MLConfig()
    token_header = ml_config.auth_token_header
    print(token_header)

    ml_service_details = ml_config.get_ml_service_details("TWV")
    print(ml_service_details)

    model_dets = ml_config.get_ml_model_details("ZZZ", "Z99", "Z9", "GL")
    print(model_dets)
