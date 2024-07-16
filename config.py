# Configuration file for paths and parameters

DATA_PATH = "data/fourth_down_data.csv"
SAVED_MODEL_PATH = "models/"
LOGGING_PATH = "logs/"

# Parameters for model training
MODEL_PARAMS = {
    "LogisticRegression": {
        "penalty": "l2",
        "C": 0.25,
        "solver": "liblinear",
        "random_state": 123
    },
    "RandomForestClassifier": {
        "n_estimators": 50,
        "max_depth": 4,
        "min_samples_split": 0.1,
        "criterion": "gini",
        "bootstrap": True,
        "random_state": 123
    },
    "XGBClassifier": {
        "learning_rate": 0.25,
        "max_depth": 4,
        "n_estimators": 100,
        "random_state": 123
    },
    "NeuralNetwork": {
        "epochs": 50,
        "batch_size": 32,
        "input_dim": None  # to be set dynamically
    }
}

# Parameters for data processing
DATA_PARAMS = {
    "categorical_variables": ['posteam_type', 'game_half', 'play_location', 'play_subtype'],
    "numerical_variables": ['yardline_100', 'half_seconds_remaining', 'ydstogo', 'ydsnet', 'score_differential'],
    "binary_variables": ['goal_to_go', 'shotgun', 'no_huddle', 'qb_dropback'],
    "target_variable": 'fourth_down_converted'
}
