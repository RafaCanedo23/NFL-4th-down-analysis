import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import config
import logging
import os

logging.basicConfig(filename=config.LOGGING_PATH + 'data_preparation.log', level=logging.INFO)

def load_data(file_path):
    """
    Load data from a pickle file.
    """
    try:
        data = pd.read_pickle(file_path).drop(['yardline_group', 'play_type'], axis = 1)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """
    Preprocess the data by encoding categorical variables and scaling numerical variables.
    """
    try:
        categorical_variables = config.DATA_PARAMS['categorical_variables']
        numerical_variables = config.DATA_PARAMS['numerical_variables']

        # Encode categorical variables
        encoder = OneHotEncoder(sparse_output=False)
        encoded_data = encoder.fit_transform(data[categorical_variables])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_variables))
        data = data.drop(categorical_variables, axis=1)
        data = pd.concat([data, encoded_df], axis=1)

        # Scale numerical variables
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data[numerical_variables])
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_variables)
        data[numerical_variables] = scaled_df

        logging.info("Data preprocessing completed successfully")
        return data
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise

def feature_importance_reduction(data):
    """
    Delete not important features according to regression parameters
    """
    try:
        raws_to_drop = ['no_huddle', 'half_seconds_remaining', 'play_location_right', 'play_location_left', 
                        'play_location_middle', 'shotgun', 'posteam_type_away', 'posteam_type_home']
        data = data.drop(raws_to_drop, axis = 1)

        logging.info("Feature importance completed successfully")
        return data
    except Exception as e:
        logging.error(f"Error in feature deleting: {e}")
        raise

def save_preprocessed_data(data, file_path):
    """
    Save preprocessed data to a pickle file.
    """
    try:
        data.to_pickle(file_path)
        logging.info(f"Preprocessed data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving preprocessed data: {e}")
        raise

if __name__ == "__main__":
    os.makedirs(config.LOGGING_PATH, exist_ok=True)
    raw_data = load_data(config.DATA_PATH)
    preprocessed_data = preprocess_data(raw_data)
    feat_importance_data = feature_importance_reduction(preprocessed_data)
    save_preprocessed_data(feat_importance_data, config.DATA_PATH.replace('.pkl', '_preprocessed.pkl'))