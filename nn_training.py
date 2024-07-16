import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
import config
import logging
import os

logging.basicConfig(filename=config.LOGGING_PATH + 'neural_network_training.log', level=logging.INFO)

def load_preprocessed_data(file_path):
    """
    Load preprocessed data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
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

def train_neural_network(X_train, y_train, params):
    """
    Train a neural network model.
    """
    try:
        model = Sequential()
        model.add(Dense(128, input_dim=params['input_dim'], activation='relu', kernel_regularizer=regularizers.L2(0.01)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.05))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.05))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall()])
        model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'])

        logging.info("Neural network trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error training neural network: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a neural network model.
    """
    try:
        loss, accuracy, recall = model.evaluate(X_test, y_test)
        logging.info(f"Model evaluation completed with Loss: {loss}, Accuracy: {accuracy}, Recall: {recall}")
        return loss, accuracy, recall
    except Exception as e:
        logging.error(f"Error evaluating neural network: {e}")
        raise

def save_model(model, model_name):
    """
    Save the trained model to a file.
    """
    try:
        model.save(f"{config.SAVED_MODEL_PATH}{model_name}.h5")
        logging.info(f"{model_name} saved successfully")
    except Exception as e:
        logging.error(f"Error saving {model_name}: {e}")
        raise

if __name__ == "__main__":
    os.makedirs(config.SAVED_MODEL_PATH, exist_ok=True)
    preprocessed_data = load_preprocessed_data(config.DATA_PATH.replace('.csv', '_preprocessed.csv'))
    target_variable = config.DATA_PARAMS['target_variable']
    X_train, X_test, y_train, y_test = split_data(preprocessed_data, target_variable)

    nn_params = config.MODEL_PARAMS['NeuralNetwork']
    nn_params['input_dim'] = X_train.shape[1]
    nn_model = train_neural_network(X_train, y_train, nn_params)
    nn_loss, nn_accuracy, nn_recall = evaluate_model(nn_model, X_test, y_test)
    save_model(nn_model, 'NeuralNetwork')
