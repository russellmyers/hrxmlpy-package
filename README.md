#hrxmlpy
 
The hrxmlpy package provides functionality to perform data transforms, ML training and ML predictions for models configured in the MLConfig API.

Currently, the following model types are available:
- Tax Withholding Verification
- Broad Spectrum Anomaly Detection

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
  


