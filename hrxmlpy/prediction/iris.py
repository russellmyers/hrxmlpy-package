"""
Test prediction code using Iris model
"""

import logging
import pandas as pd


def predict(retrieved_model_objects, X):
    X_pred = X
    model = retrieved_model_objects["model"]
    feature_names = retrieved_model_objects["feature_names"]
    target_names = retrieved_model_objects["target_names"]
    svm_predictions = model.predict(X_pred)
    pred_in = [[feature_names[i] + ":" + str(feat) for i, feat in enumerate(x)] for x in X]
    pred_out = [target_names[y] for y in svm_predictions]
    for i, p_in in enumerate(pred_in):
        logging.info(f"Prediction for: {p_in} => {pred_out[i]}")
    return pd.DataFrame(pred_out, columns=["Predicted class"])
