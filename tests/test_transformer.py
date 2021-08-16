import os
import json
import pandas as pd
from hrxmlpy.pipeline.transform.data_transformer import DataTransformer

def test_transform_pad():
    with open(TEST_DATA_DIR + '/sample_run_params_pad.json') as json_file:
        run_params = json.load(json_file)
    with open(TEST_DATA_DIR + '/sample_model_config_params_pad.json') as json_file:
        model_config_params = json.load(json_file)
    raw_data = pd.read_csv(TEST_DATA_DIR + '/sample_raw_training_data_pad.csv')
    t = DataTransformer(run_params=run_params, model_config_params=model_config_params)
    transformed_data = t.transform(raw_data)

    assert raw_data.shape == (164,21)
    assert transformed_data.shape == (164, 6)
    assert list(transformed_data.columns) == ['Employee_Id', '/101_Gross', '/560_PaymentAmount', '/559_NettPay', '0010_BasicSalary', '/700_WagePlusERCont']
    assert raw_data.iloc[0]['Employee_Id'] == 50009617
    assert raw_data.iloc[0]['/101_Gross'] == 2500.00
    assert transformed_data.iloc[0]['Employee_Id'] == 50009617
    assert round(transformed_data.iloc[0]['/101_Gross'],5) == -0.29117

def test_transform_twv():
    with open(TEST_DATA_DIR + '/sample_run_params_twv.json') as json_file:
        run_params = json.load(json_file)
    with open(TEST_DATA_DIR + '/sample_model_config_params_twv.json') as json_file:
        model_config_params = json.load(json_file)
    raw_data = pd.read_csv(TEST_DATA_DIR + '/sample_raw_training_data_twv.csv')
    t = DataTransformer(run_params=run_params, model_config_params=model_config_params)
    transformed_data = t.transform(raw_data)

    assert raw_data.shape == (121, 184)
    assert transformed_data.shape == (121, 3)
    assert list(transformed_data.columns) == ['Employee_Id', 'Gross_Pay', 'Federal Withholding [USA]-W_FW-Federal: Federal']
    assert raw_data.iloc[0]['Employee_Id'] == 1374
    assert round(raw_data.iloc[0]['Gross_Pay'],2) == 10944.03
    assert transformed_data.iloc[0]['Employee_Id'] == 1374
    assert round(transformed_data.iloc[0]['Gross_Pay'],5) == 0.14038



TEST_DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

if __name__ == '__main__':
    test_transform_pad()
    test_transform_twv()