# hrxmlpy
 
The hrxmlpy package provides core functionality to perform data transforms, ML training and ML predictions for models configured in the MLConfig API.

Main classes provided in the package are:
-  **DataTransformer**
-  **Trainer**
-  **Predicter**
-  **MLConfig**

Training of customer models is performed using one of the following learning algorithms, based on the "learningAlgorithm" parameter specified when defining the ml_service used by the customer model in the ml config api (see Trainer class for more information):

| ml_service learningAlgorithm selected in ML Config API | class used | type  | example use case |
|---|---|---|---|
| "scikit_neural_network_regressor" |  scikit-learn's ```sklearn.neural_network.MLPRegressor``` class | Supervised, regression |eg use for TWV |
|"scikit_isolation_forest"  |scikit-learn's standard Isolation Forest class (```sklearn.ensemble.IsolationForest```) |  Unsupervised | No longer used |
|"enhanced_isolation_forest" |Alight's enhanced Isolation Forest algorithm contained in this package (```hrxmlpy.algorithms.isolation_forest.EnhancedIsolationForest```) | Unsupervised |eg use for PAD broad spectrum payroll anomaly detection |
 can be defined using learning algorithms 

Currently, the following ml_services are specifically supported with additional functionality used by Eloise (although the ml config api has been designed to be generally applicable to other ml_services):
>- Tax Withholding Verification (TWV)
>- Broad Spectrum Anomaly Detection (PAD)

Training flow:  


![Training flow diagram](./img_train_flow.PNG?raw=true "Training flow")


Prediction flow:  

![Prediction flow diagram](./img_predn_flow.PNG?raw=true "Prediction flow")


## Package structure
The package has the following structure:

- algorithms
  - Contains any proprietary ML algorithms developed by Alight for use within hrxmlpy (currently includes "Enhanced Isolation Forest").  
   *Note: Various standard scikit-learn algorithms can also be used "as-is" within hrxmlpy. See logic in train method of Trainer class below for more details*

- interface
  -  Contains an "MLConfig" class to provide wrapper access to the MLConfig API
  
- pipeline
  -  Contains various classes to perform data transform and ML training
  
- prediction
  - Contains ML prediction functionality  
  
- utils
  - Contains general helper functions
  
  
## MLConfig Class
`hrxmlpy.interface.hrx.mlconfig.MLConfig`  
Wrapper class for access to ML Config API to retrieve registered customer models

Main methods:
* \_\_**init**\_\_(api_base_url, api_client_id, api_client_secret)   
  Instantiate instance of class, passing Base MLConfig API URL, Client Id and Client Secret  
  Sample:
   * Base URL: "http://localhost:5000"
   * Client Id: "d7947bf7-3240-435d-8a79-5b2fc05248c7"
   * Client Secret: "DaTbzvauLJyFqDjpmE5nD5Q7t887TBHdZozxxagBsvQsHePXawgpwoTHuWSAydLj"
  
* **get_ml_model_details**(gcc, lcc, payroll_area, country)  
  Main method to retrieve ml_config_params for all customer models registered for a gcc/lcc/payroll_area, Uses the ML Config API *model-info* endpoint   
  *Returns*: list of model_config_params dictionaries (ie one model_config_params dict for each customer model registered for a gcc/lcc/payroll_area)
 
Other methods available:  

* auth_token_header  
  Retrieve authorisation token required to populate request header within API calls. Called implicitly from most other methods
* get(endpoint)  
  General method to get any endpoint from ML Config API (result returned as json)
* get_all_customers_using_model(ml_service, model_code)
* get_ml_model_details_by_model(ml_service, model_code)
* get_ml_service_details(ml_service_code)
* get_label(model_config_params)  *(static method)*
* get_anomaly_threshold_details(model_config_params)  *(static method)
    
## Trainer Class
`hrxmlpy.pipeline.train.trainer.Trainer`  
Class to manage training of a ML Model, based on configuration specified in model_config_params and train_params

Main methods:
* \_\_**init**\_\_(train_params, model_config_params)   
  Instantiate instance of Trainer, passing train_params and model_config_params.
  
  Example of train_params:
  ```json
  {
  "random_state": 42,
  "test_size": 0.2,
  "scaler": "standard",
  "hyper_params": {
    "max_iter": 500,
    "AnomalyThresholdType": "A",
    "AnomalyThresholdVal": 20
    }
  }
  ``` 
  
  model_config_params:  
  - Dictionary containing configuration for a single model registered for a customer (ie for a gcc/lcc/payroll_area/ml_service).  
  Retrieved using get_ml_model_details method of MLConfig class.  
  (Note: get_ml_model_details returns a *list* of registered model dictionaries, ie one for each ml_service registered for a gcc/lcc/payroll_area. Trainer is then instantiated with a single dict from this list - ie for a specified ml_service)
 
  
* **split**(data)  
  Input:
  - pandas dataframe containing (already transformed) training data
  
  Splits  data into a train and test set according to "test_size" parameter in train_params (eg 0.2 = 80/20 split).  
  Note: For unsupervised models, 0 should be used as test_size (ie empty test set)
  
  Note:
  Id column (eg "Employee_Id") should be dropped prior to calling this method, ie "data" should not contain id column
   
  Returns:
  - train_features (eg "X_train"), note: only contains columns from data marked as features in ML config
  - train_labels (eg "y_train"), note: only contains column from data marked as label  in ML config. "None" for unsupervised models
  - test_features (eg "X_test"), note: only contains columns from data marked as features in ML config
  - test_labels (eg "y_test"),note: only contains column from data marked as label  in ML config. "None" for unsupervised models
  
* **train**(train_X, train_y, test_X, test_y)   

  Input:
  * train_X, train_y, test_X, test_y (from split method).  
   Note: train_y, test_X and test_y are not rqd for unsupervised models (ie are set to "None")  
  
  Logic:  
  * Determine learning algorithm to use (based on "learningAlgorithm" parameter in model_config_params).  
  Current choices:
    * "scikit_neural_network_regressor" (use scikit-learn MLPRegressor class), eg use for TWV
    * "scikit_isolation_forest" (use standard scikit-learn Isolation Forest class)
    * "enhanced_isolation_forest" (use Alight enhanced Isolation Forest algorithm), eg use for broad spectrum payroll anomaly detection

  * Set hyper parameters of learning algorithm based on "hyper_params" supplied within train_params
  
  * fit (train) model using selected learning algorithm and supplied training data
  
  * Determine metrics
    * For supervised models:  
      => Returns overall Training and Test scores (ie train/test losses)
      
    * For unsupervised models:  
      => Returns None
  
  * Perform predictions on training/test set 
   
    Predictions are returned as pandas dataframe(s), with one record per example
    * For supervised models:  
      => Calculate predicted labels for each training and test example and return training and test prediction dataframes
       
    * For unsupervised models:  
      => Determine Anomaly score, Rank and Important Features for each training example, and return training prediction dataframe
      
      
  Notes:
  > Training data supplied to the Trainer class is expected to be already transformed (ie scaled, one hot encoded etc).  This transformation should be performed  prior to training (or prediction) using the DataTransformer class/subclasses.
  
  > Predictions made during training within the Trainer class are provided to assist with tuning hyperparameters prior to finalising the trained model. For actual predictions (ie on trained model), use Predicter class 
 
## Predicter Class
`hrxmlpy.pipeline.train.trainer.Predicter`  
Class to perform predictions for a trained ML Model

Main methods:
* \_\_**init**\_\_(train_params, model_config_params, model, feature_scaler)   
  Instantiate instance of Predicter, passing train_params , model_config_params and trained model    
  Note: See Trainer class for example of input train_params and model_config_params.    
  
  "model" is the trained model object to be used for prediction (ie returned from the Trainer class "train" method) 
  

* **predict**(data, emp_ids)   

  Input:
  * pandas dataframe containing (already transformed) data, ready for prediction  
  
  Logic:  
  * Call split method (see below) to separate out features from label (if label column is supplied)
   
  * Perform predictions on prediction data provided 
   
    Predictions are returned as pandas dataframe(s) - one record per example
    * For supervised models:  
      => Calculate predicted labels for each example and return prediction results dataframe
       
    * For unsupervised models:  
      => Determine Anomaly score, Rank and Important Features for each prediction example, and return prediction results dataframe
      
  Notes:
  > Data supplied to the Predicter class for prediction is expected to be already transformed (ie scaled, one hot encoded etc).  Transformation can be performed using the DataTransformer class/subclasses
 
* **split**(data)   
  (Called from predict method - see above)   
  
  Input:
  * pandas dataframe containing (already transformed) data, ready for prediction
  
  Splits input data into features and labels (based on label specified in ml config).  
 
  Notes:
   * label column is not supplied for unsupervised training
   * label column is optional for supervised training (if supplied, comparison can be made of actuals vs predictions - eg for TWV)
   * Id column (eg "Employee_Id") should be dropped prior to calling this method, ie "data" should not contain id column
   
  Returns:
  * predict_features (eg "X_predict")
  * predict_labels (eg "y_predict")  - if relevant, otherwise None

## DataTransformer Class

```
hrxmlpy.pipeline.transform.data_transformer.DataTransformer
```  
Class to manage data transformation prior to training/prediction (ie  includes methods for scaling, one hot encoding etc)

Main methods:
* \_\_**init**\_\_(run_params, model_config_params, scaler, encoder)   
  Instantiate instance of DataTransformer, passing train_params, model_config_params, already-trained scaler model (optional) and already-trained encoder model(optional)
  
  **model_config_params**  
  See Trainer class for description of model_config_params
  
  **run_params**  
  Example:  
  
  ```json
  {
  "scaler_type": "standard",
  "ml_service_code": "TWV"
   }
  ``` 
  Options for scaler_type (used for training a scaler model if pre-trained scaler is not supplied):
  
  | scaler_type | Scikit-learn method used |
  |---|---|
  | standard |  preprocessing.StandardScaler()|
  | minmax | preprocessing.MinMaxScaler()|
  | robust |  preprocessing.RobustScaler() |
  | normalizer | preprocessing.Normalizer()|
  
  **scaler**  
  Already-trained scaler model to be used in transforms (optional. If not passed, then a new scaler model will be trained)
  
  **encoder**  
  Already-trained one hot encoder model to be used in transforms (optional. If not passed, then a new one hot encoder model will be trained)  
  

* **transform**(data)  
  Perform transformations on raw data provided in pandas dataframe.  

  Instantiates relevant DataTransformer subclass (based on run_params "ml_service_code") and calls transform method  of subclass to perform transformations.  
  
  Subclasses perform relevant transformations by calling selected relevant main class methods below.
  

* **remove_retros**(data)
  
  Remove any rows from data with a non-null value in the "Retro_Period" column.  
  
  Note: If a "Retro_Period" column does not exist in data, this method does nothing  


* **remove_exclusions**(data)

  Remove any rows from data with values matching values specified in model_config_params["exclusions"] dictionary 

* **filter_unwanted_columns**(data)  

  Filter data to only include id column (eg "Employee_Id") plus any columns listed as features or labels in model_config_params.  
  (ie exclude any extraneous columns in data which are not used by ml_service model)
 
* **get_scaler**(data)  

  If an already-trained scaler model was provided to constructor:  
  Just  return this already trained scaler model
  
  Otherwise:   
  Create and train (fit) a new scaler model based on the "scaler_type" supplied in run_params (and all *numeric* feature fields in data), and return the trained scaler model

* **scale**(data)  

  Scale numeric feature columns in data using the scaler model returned by get_scaler method 

* **get_encoder**(data)  

  If an already-trained one hot encoder model was provided to constructor:  
  Just  return this already trained one hot encoder model
  
  Otherwise:   
  Create and train (fit) a new one hot encoder model using sci-kit learn ```preprocessing.OneHotEncoder(handle_unknown="ignore")``` (and all *categorical* feature fields in data), and return the trained one hot encoder model

* **one_hot_encode**(data)  

  One hot encode categorical feature columns in data using the one hot encoder model returned by get_encoder method.  
  
  Note:  if the ML service model being used is "enhanced_isolation_forest" (ie ```model_config_params["learningAlgorithm"] == "enhanced_isolation_forest"```), this method is by-passed since enhanced isolation forest can cater for categorical features directly. 

## TaxAnomalyDataTransformer Class
#### (subclass of DataTransformer)
```
hrxmlpy.pipeline.transform.data_transformer.TaxAnomalyDataTransformer
```  
Class to manage data transformation specifically for TWV ml_service-based models.  
Instantiated automatically from parent class when "ml_service_code" of "TWV" is supplied within run_params

Main methods:
 
* **transform**(data)  
  Perform transformations on raw data provided in pandas dataframe, by calling the following methods of the parent class in sequence:
  
  - remove_retros(data)
  - remove_exclusions(data)
  - filter_unwanted_columns(data)
  - convert_string_fields(data)
  - get_scaler(data)
  - scale(data)
  - get_encoder(data)
  - one_hot_encode(data)

## PayAnomalyDataTransformer Class
#### (subclass of DataTransformer)
```
hrxmlpy.pipeline.transform.data_transformer.PayAnomalyDataTransformer
```  
Class to manage data transformation specifically for PAD ml_service-based models.  
Instantiated automatically from parent class, when  "ml_service_code" of "PAD" is supplied within run_params


Main methods:
 
* **transform**(data)  
  Perform transformations on raw data provided in pandas dataframe, by calling the following methods of the parent class in sequence:
  
  - remove_retros(data)
  - remove_exclusions(data)
  - filter_unwanted_columns(data)
  - convert_string_fields(data)
  - get_scaler(data)
  - scale(data)
  - get_encoder(data)
  - one_hot_encode(data)

  


