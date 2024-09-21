import nfl_data_py as nfl
import pandas as pd
import numpy as np

# Ejecutamos la búsqueda
years = [2019, 2020, 2021, 2022, 2023]
data = nfl.import_pbp_data(years=years)

# Seleccionamos las observaciones pertinentes al proyecto
fourth_down_data_row = data[
    (data['down'] == 4) &
    (data['play_type'].isin(['pass', 'run'])) &
    (data['replay_or_challenge'] == 0) &
    (data['penalty'] == 0)
]

# Definimos las variables de interés
scoped_variables = [
    'yardline_100', 'posteam_type', 'half_seconds_remaining', 'game_half', 
    'goal_to_go', 'ydstogo', 'ydsnet', 'play_type', 'shotgun', 'no_huddle', 
    'qb_dropback', 'pass_length', 'pass_location', 'run_location', 'run_gap', 
    'score_differential', 'fourth_down_converted'
]

# Nos quedamos con el scope final (row data)
fourth_down_data = fourth_down_data_row[scoped_variables]

# Combinamos las columnas de run & pass location (porque son los mismos valores únicos)
fourth_down_data['play_location'] = np.where(
    fourth_down_data['play_type'] == 'pass',
    fourth_down_data['pass_location'],
    fourth_down_data['run_location']
)
fourth_down_data = fourth_down_data.dropna(subset=['play_location'])

# Quitamos las variables de pass & run locations
fourth_down_data = fourth_down_data.drop(columns=['pass_location', 'run_location'])

# Combinamos las columnas de run & pass location (porque son los mismos valores únicos)
fourth_down_data['play_subtype'] = np.where(
    fourth_down_data['play_type'] == 'pass',
    'Pass: ' + fourth_down_data['pass_length'],
    'Run: ' + fourth_down_data['run_gap']
)
fourth_down_data = fourth_down_data.dropna(subset=['play_subtype'])

# Quitamos las variables de pass & run locations
fourth_down_data = fourth_down_data.drop(columns=['pass_length', 'run_gap'])

# Generamos una nueva columna con rangos para la variable yardline_100
bins = list(range(0, 101, 10))
labels = [f"{i}-{i+9}" for i in bins[:-1]]
fourth_down_data['yardline_group'] = pd.cut(
    fourth_down_data['yardline_100'], bins=bins, labels=labels, right=False
)

# Asignamos etiquetas a la variable y
fourth_down_data['fourth_down_converted'] = fourth_down_data[
    'fourth_down_converted'
].replace({0: 'Not converted', 1: 'Converted'})

fourth_down_data.to_pickle('data/fourth_down_data.pkl')
