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

    def __init__(self, api_base_url, api_client_id, api_client_secret):
        self.api_base_url = api_base_url
        self.api_client_id = api_client_id
        self.api_client_secret = api_client_secret

        self.api_customer_id = MLConfig.DEFAULT_CUSTOMER_ID
        self.api_company_id  = MLConfig.DEFAULT_COMPANY_ID
        self.auth_endpoint   = MLConfig.DEFAULT_AUTH_ENDPOINT
        self.ml_services_endpoint = MLConfig.DEFAULT_ML_SERVICES_ENDPOINT
        self.models_endpoint = MLConfig.DEFAULT_MODELS_ENDPOINT
        self.customer_models_endpoint = MLConfig.DEFAULT_CUSTOMER_MODELS_ENDPOINT


    def override_api_customer_company(self, api_customer_id, api_company_id):
        """ Optional: used to override api customer_id and company_id  used in auth tokens if rqd"""
        self.api_customer_id = api_customer_id
        self.api_company_id  = api_company_id

    def override_default_endpoints(self, auth_endpoint=None, ml_services_endpoint=None, models_endpoint=None, customer_models_endpoint=None):
        """ Optional: used to override default endpoints if rqd """
        if auth_endpoint is not None:
           self.auth_endpoint = auth_endpoint
        if ml_services_endpoint is not None:
            self.ml_services_endpoint = ml_services_endpoint
        if models_endpoint is not None:
            self.models_endpoint = models_endpoint
        if customer_models_endpoint is not None:
           self.customer_models_endpoint = customer_models_endpoint


    @property
    def auth_token_header(self):
        auth_token_header = None
        data = {
            "clientId": self.api_client_id,
            "clientSecret": self.api_client_secret,
            "customerId": self.api_customer_id,
            "companyId": self.api_company_id,
        }
        url = self.api_base_url + self.auth_endpoint
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
            logging.error(f'{__class__.__name__} - Auth error for clientId: {self.api_client_id}')
        return auth_token_header

    def get(self, endpoint):
        token = self.auth_token_header
        response_json_data = {}
        if token:
            request_url = self.api_base_url + endpoint
            response = requests.get(request_url, headers=token)
            if response.status_code == 200:
                response_json_data = json.loads(response.text)
            else:
                print(f"{__class__.__name__} - {response.status_code} error for endpoint: {request_url}")
        return response_json_data


    def add_ml_service_technical_details(self, ml_service_details):
        ml_service_tech_details_all = self.get(self.ml_services_endpoint)
        for model_config_params in ml_service_details:
            ml_service_tech_details = [
                x for x in ml_service_tech_details_all["items"] if x["code"] == model_config_params["mlService"]
            ][0]
            model_config_params["mlType"] = ml_service_tech_details["mlType"]
            model_config_params["learningAlgorithm"] = ml_service_tech_details["learningAlgorithm"]

    def get_ml_model_details(self, gcc, lcc, payroll_area, country):
        token = self.auth_token_header
        model_info_endpoint = self.models_endpoint
        response_json_data = {}
        if token:
            request_url = self.api_base_url + model_info_endpoint
            response = requests.get(
                request_url, headers=token, params={"gcc": gcc, "lcc": lcc, "payrollArea": payroll_area}
            )
            if response.status_code == 200:
                response_json_data = json.loads(response.text)
                self.add_ml_service_technical_details(response_json_data)
            else:
                logging.error(
                    f"{__class__.__name__} - {response.status_code} error for endpoint: {request_url}"
                )
        return response_json_data

    def get_all_customers_using_model(self, ml_service, model_code):
        token = self.auth_token_header
        customer_models_endpoint = self.customer_models_endpoint
        response_json_data = {}

        customers = []
        if token:
            request_url = self.api_base_url + customer_models_endpoint
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
                    f"{__class__.__name__} - {response.status_code} error for endpoint: {request_url}"
                )
        return customers

    def get_ml_model_details_by_model(self, ml_service, model_code):
        all_custs = self.get_all_customers_using_model(ml_service, model_code)
        if len(all_custs) == 0:
            return {}

        cust = all_custs[0]  # Use first customer to retrieve model info
        model_details = self.get_ml_model_details(cust["gcc"], cust["lcc"], cust["payroll_areas"][0], "")
        for model_config_params in model_details:
            if model_config_params["mlService"] == ml_service:
                return model_config_params

        return {}

    def get_ml_service_details(self, ml_service_code):
        # TODO: To be updated once the ml-services endpoint is amended to allow for ml service codes, not just ids
        ml_service_id_map = {"PAD": 1, "TWV": 2}
        token = self.auth_token_header
        ml_services_endpoint = f"{self.ml_services_endpoint}/{ml_service_id_map[ml_service_code]}"
        if token:
            request_url = self.api_base_url + ml_services_endpoint
            response = requests.get(request_url, headers=token)
            if response.status_code == 200:
                response_json_data = json.loads(response.text)
            else:
                logging.error(
                    f"{__class__.__name__} - {response.status_code} error for endpoint: {request_url}"
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
    ml_config = MLConfig(api_base_url= "http://localhost:5000", api_client_id="d7947bf7-3240-435d-8a79-5b2fc05248c7", api_client_secret="DaTbzvauLJyFqDjpmE5nD5Q7t887TBHdZozxxagBsvQsHePXawgpwoTHuWSAydLj")
    token_header = ml_config.auth_token_header
    print(token_header)

    ml_service_details = ml_config.get_ml_service_details("TWV")
    print(ml_service_details)

    model_dets = ml_config.get_ml_model_details("ZZZ", "Z99", "Z9", "GL")
    print(model_dets)
