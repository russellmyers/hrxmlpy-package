#hrxmlpy
 
The hrxmlpy package provides functionality to perform data transforms, ML training and ML predictions for models configured in the MLConfig API.

Currently, the following model types are available:
- Tax Withholding Verification
- Broad Spectrum Anomaly Detection

## Package structure
The package has the following structure:

- algorithms
  - Proprietary ML algorithms (Enhanced Isolation Forest). Note: standard scikit-learn algorithms can also used

- interface
  -  MLConfig class to provide access to the MLConfig API
  
- pipeline
  -   classes to perform ML training and transformation
  
- prediction
  - prediction functionality  
  
- utils
  - helper functions
  
## Environment variables

The hrxmlpy package relies on the following environment variables to be set for access to MLConfig API:

MLCONFIG_API_CLIENT_ID eg "d7947bf7-3240-435d-8a79-5b2fc05248c7"
MLCONFIG_API_CLIENT_SECRET eg "DaTbzvauLJyFqDjpmE5nD5Q7t887TBHdZozxxagBsvQsHePXawgpwoTHuWSAydLj"
MLCONFIG_API_BASE_URL eg "http://localhost:5000"

The following environment variables are not required to be provided (ie they have defaults), but can be provided as overrides if desired:
MLCONFIG_API_CUSTOMER_ID (default "ZZZ")
MLCONFIG_API_COMPANY_ID (default "Z99")
MLCONFIG_API_AUTH_ENDPOINT (default "/auth/tokens")
MLCONFIG_API_CUSTOMER_MODELS_ENDPOINT (default "/api/V2/customer-models")
MLCONFIG_API_MODELS_ENDPOINT (default "/api/V2/model-info")
MLCONFIG_API_ML_SERVICES_ENDPOINT (default "/api/V2/ml-services")


(*TODO*: replace all environment parameters with ordinary parameters? ie rely on calling programs to pass the parameters based on their own environment)



