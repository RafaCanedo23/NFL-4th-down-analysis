import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score
import xgboost as xgb
import joblib
import config
import logging
import os

logging.basicConfig(filename=config.LOGGING_PATH + 'model_training.log', level=logging.INFO)

def load_preprocessed_data(file_path):
    """
    Load preprocessed data from a pickle file.
    """
    try:
        data = pd.read_pickle(file_path)
        logging.info(f"Preprocessed data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading preprocessed data: {e}")
        raise

def split_data(data, target_variable):
    """
    Split the data into training and testing sets.
    """
    try:
        X = data.drop(target_variable, axis=1)
        y = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
        y_train = y_train.replace({'Not converted' : 0, 'Converted' : 1})
        y_test = y_test.replace({'Not converted' : 0, 'Converted' : 1})
        logging.info("Data split into training and testing sets")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def train_model(model_name, X_train, y_train, params):
    """
    Train a machine learning model.
    """
    try:
        if model_name == 'LogisticRegression':
            model = LogisticRegression(**params)
        elif model_name == 'RandomForestClassifier':
            model = RandomForestClassifier(**params)
        elif model_name == 'XGBClassifier':
            model = xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train, y_train)
        logging.info(f"{model_name} trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error training {model_name}: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a machine learning model.
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        logging.info(f"Model evaluation completed with Accuracy: {accuracy}, Recall: {recall}, F1 Score: {f1}")
        return accuracy, recall, f1
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_model(model, model_name):
    """
    Save the trained model to a file.
    """
    try:
        joblib.dump(model, f"{config.SAVED_MODEL_PATH}{model_name}.pkl")
        logging.info(f"{model_name} saved successfully")
    except Exception as e:
        logging.error(f"Error saving {model_name}: {e}")
        raise

if __name__ == "__main__":
    os.makedirs(config.SAVED_MODEL_PATH, exist_ok=True)
    preprocessed_data = load_preprocessed_data(config.DATA_PATH.replace('.pkl', '_preprocessed.pkl'))
    target_variable = config.DATA_PARAMS['target_variable']
    X_train, X_test, y_train, y_test = split_data(preprocessed_data, target_variable)

    # Train and evaluate Logistic Regression
    lr_params = config.MODEL_PARAMS['LogisticRegression']
    lr_model = train_model('LogisticRegression', X_train, y_train, lr_params)
    lr_accuracy, lr_recall, lr_f1 = evaluate_model(lr_model, X_test, y_test)
    save_model(lr_model, 'LogisticRegression')

    # Train and evaluate Random Forest
    rf_params = config.MODEL_PARAMS['RandomForestClassifier']
    rf_model = train_model('RandomForestClassifier', X_train, y_train, rf_params)
    rf_accuracy, rf_recall, rf_f1 = evaluate_model(rf_model, X_test, y_test)
    save_model(rf_model, 'RandomForestClassifier')

    # Train and evaluate XGBoost
    xgb_params = config.MODEL_PARAMS['XGBClassifier']
    xgb_model = train_model('XGBClassifier', X_train, y_train, xgb_params)
    xgb_accuracy, xgb_recall, xgb_f1 = evaluate_model(xgb_model, X_test, y_test)
    save_model(xgb_model, 'XGBClassifier')