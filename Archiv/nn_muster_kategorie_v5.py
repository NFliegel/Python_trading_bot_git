import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sqlalchemy


import random
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU deaktivieren
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['TF_NUM_INTEROP_THREADS'] = '20'  # Anzahl der Threads für parallele Operationen
os.environ['TF_NUM_INTRAOP_THREADS'] = '20'  # Anzahl der Threads für interne Operationen

import tensorflow as tf
# import tensorflow_hub as hub
# import tensorflow_text as text  # Wichtig für BERT Preprocessing

import pandas as pd
import optuna
import optuna.visualization as vis
from optuna.integration import TFKerasPruningCallback
import joblib


import mysql.connector
from mysql.connector import Error
import functools
import json

from sqlalchemy import create_engine
from difflib import SequenceMatcher
from tqdm import tqdm
import traceback
import matplotlib.pyplot as plt
import time
# import datetime
import math
import re
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
# from keras.preprocessing.sequence import TimeseriesGenerator
# from tensorflow.keras.models import load_model


seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)


connection_config = {
    'database': 'trading_bot',
    'host': 'localhost',
    'user': 'root',
    'password': '',
}


# SETTINGSDATA
#  =====================================================================================================================

y_target_data = "Trend"
# y_target_data = "total_change"

# sample_weight_column = "price_quality"


categorical_str_cols_v3 = []
# vectorized_column = "modell_opt"  # V3

categorial_int_cols_v3 = []
# categorial_int_cols_v3 = ['lead_created_year', 'lead_created_month']
# ==> ACHTUNG!!! Sehr schlecht, da bei einer Abfrage, die nicht genau die nötigen Kategorien hat ein Durchschnittspreis geliefert wird. So als ob das Rad nicht gefunden wurde.


# baujahr_spalte = "baujahr_jahr"

# numerical_cols_v3 = ['laufleistung_approximately', 'alter_jahre', 'baujahr_jahr', 'zustand_motorrad', 'lead_created_year', 'lead_created_month']
# numerical_cols_v3 = ['laufleistung_approximately', 'alter_jahre', 'baujahr_jahr', 'zustand_motorrad']



# SETTINGSDATA
#  =====================================================================================================================


# SETTINGSTRAIN
#  =====================================================================================================================
#  Initialize the callbacks
#  ReduceLROnPlateau
tuning_reduce_lr_factor = 0.1
# tuning_reduce_lr_factor = 0.01  # test ob man mit einer hohen lr starten kann und durch kleinere sprünge im decay lr die hohen "falschen" gewichte ausreichen zurückstellen kann

# tuning_reduce_lr_patience = 1
tuning_reduce_lr_patience = 20

tuning_reduce_lr_min_delta = 0  # --> muss eine minimale Verbesserung gegeben haben
tuning_reduce_lr_min_lr = 1e-10
tuning_reduce_lr_cooldown = 0

#  EarlyStopping

# tuning_early_stopping_patience = 5
tuning_early_stopping_patience = 80

# tuning_early_stopping_min_delta = 0.001  # es muss sich mindestens so viel verbessert haben um als Verbesserung zu gelten
tuning_early_stopping_min_delta = 0.0


#  Optimizer Adam
tuning_adam_learning_rate = 0.01  # --> hohe Lernrate, ist dieser Wert zu hoch werden die Gewichtungen zu stark und können bei Reduzierung der lr
# !!!!!! Die Kurve ist maßgeblich, ob die lr richtig eingestellt ist !!!

# nicht mehr in die richtige Richtung umgelenkt werden

# SETTINGSTRAIN
#  =====================================================================================================================


# SETTINGSBUILD
#  =====================================================================================================================
#  Initialize the callbacks
#  ReduceLROnPlateau
building_reduce_lr_factor = 0.1
building_reduce_lr_patience = 50
building_reduce_lr_min_delta = 0  # --> muss eine minimale Verbesserung gegeben haben
building_reduce_lr_min_lr = 1e-20
building_reduce_lr_cooldown = 0

#  EarlyStopping
building_early_stopping_patience = 200  # es muss sich mindestens so viel verbessert haben um als Verbesserung zu gelten
building_early_stopping_min_delta = 0  # --> muss eine minimale Verbesserung gegeben haben

#  Optimizer Adam
# building_adam_learning_rate = 0.001  # ist dieser Wert zu hoch werden die Gewichtungen zu stark und können bei Reduzierung der lr
# nicht mehr in die richtige Richtung umgelenkt werden
building_adam_learning_rate = 0.001
# !!!!!! Die Kurve ist maßgeblich, ob die lr richtig eingestellt ist !!!
# SETTINGSBUILD
#  =====================================================================================================================


def calculate_neurons_and_print_code(features_shape):
    input_features = features_shape[1]  # Anzahl der Eingabefeatures
    output_units = 1  # Für Regressionsprobleme üblicherweise 1 Ausgabeeinheit

    # Regel 1: Geometrisches Mittel
    neurons_rule1 = int(math.sqrt(input_features * output_units))

    # Regel 2: Obergrenze
    neurons_rule2 = 2 * input_features

    print(f"n_units = trial.suggest_int('n_units', {neurons_rule1}, {min(neurons_rule2, 2000)}/{input_features})  # Passen Sie die Obergrenze nach Bedarf an")


def load_data(database_name, table_name):

    source_config = f'mysql+mysqlconnector://{connection_config["user"]}:{connection_config["password"]}@{connection_config["host"]}/{database_name}'
    engine = create_engine(source_config, echo=False)

    dataframe = pd.read_sql(f'SELECT * FROM {table_name}', con=engine)

    # print("Entferne die Spalte daily_open_diff")
    # columns_to_drop = ['daily_open_diff']
    # columns_to_drop = [col for col in columns_to_drop if col in dataframe.columns]
    # dataframe = dataframe.drop(columns_to_drop, axis=1)


    print(dataframe.head())
    print(f'dataframe_len:{len(dataframe)}')

    dataframe.dropna(inplace=True)


    return dataframe




def create_database(host, user, password, db_name):
    try:
        # Verbindung zur MySQL-Datenbank herstellen
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password
        )
        if connection.is_connected():
            cursor = connection.cursor()
            # Versuchen, die Datenbank zu erstellen
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name};")
            print(f"Verwendete Datenbank: {db_name}")

            cursor.close()
            connection.close()

    except Error as e:
        print(f"Fehler: {e}")


def load_study(database_name_optuna):

    storage_name_str = f"mysql+mysqlconnector://{connection_config['user']}:{connection_config['password']}@{connection_config['host']}/optuna_{database_name_optuna}"
    # storage_name = f"mysql+mysqlconnector://{connection_config['user']}:{connection_config['password']}@{connection_config['host']}:{connection_config['port']}/{database_name}"
    study_name_str = "study"

    create_database(host=connection_config['host'], user=connection_config['user'], password=connection_config['password'], db_name=f'optuna_{database_name_optuna}')
    study = optuna.create_study(study_name=study_name_str, storage=storage_name_str, direction="minimize", load_if_exists=True)
    # study = optuna.create_study(study_name=study_name_str, storage=storage_name_str, direction="maximize", load_if_exists=True)

    try:
        print(f'best_params : {study.best_params}')
        print(f'best_value : {study.best_value}')
    except:
        print(f'best_params : no Data yet')
        print(f'best_value : no Data yet')

    return study




def train_model_v3(tuning=None, n_trials=None, n_jobs=None, database_name=None, table_name=None, database_name_optuna=None, show_progression=False, verbose=False):

    if verbose: print(f'load_data()')
    db_wkdm = load_data(database_name=database_name, table_name=table_name)
    # db_wkdm = db_wkdm[:100]

    X_orig_df = db_wkdm[db_wkdm.columns[:-1]].values  # Werte der Feature-Spalten
    Y_orig_df = db_wkdm["Trend"].values  # Werte der Ziel-Spalte

    # scaler = StandardScaler()
    # scaler.fit(X_orig_df)
    #
    # label_encoder = LabelEncoder()
    # label_encoder.fit(Y_orig_df)

    scalers = {col: StandardScaler() for col in db_wkdm.columns[:-1]}
    for col in db_wkdm.columns[:-1]:
        scalers[col].fit(db_wkdm[col].values.reshape(-1, 1))

    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoder.fit(Y_orig_df.reshape(-1, 1))

    if tuning:
        train_df, test_df = split_data_v3(db_wkdm)
    else:
        train_df = db_wkdm

    sequence_length = 12

    if tuning:
        if verbose: print(f'split_data_v3')

        if verbose: print(f'Transforming data to datasets...')
          # This should be dynamically determined or passed as a parameter based on your data preparation
        # train_ds = prepare_data_for_nn(train_df, feature_cols=train_df.columns[:-1], target_col="Trend", sequence_length=sequence_length, scaler=scaler, label_encoder=label_encoder)
        # test_ds  = prepare_data_for_nn(test_df, feature_cols=test_df.columns[:-1], target_col="Trend", sequence_length=sequence_length, scaler=scaler, label_encoder=label_encoder)

        train_ds, _, _ = prepare_data_for_nn(train_df, feature_cols=train_df.columns[:-1], target_col="Trend", sequence_length=sequence_length, scalers=scalers, onehot_encoder=onehot_encoder)
        test_ds, _, _ = prepare_data_for_nn(test_df, feature_cols=test_df.columns[:-1], target_col="Trend", sequence_length=sequence_length, scalers=scalers, onehot_encoder=onehot_encoder)

        # print(f'train_ds:{train_ds.head()}')
        # print(f'train_ds[0].shape[0]:{train_ds[0].shape[0]}')
        # print(f'train_ds[0].shape[1]:{train_ds[0].shape[1]}')
        # print(f'train_ds[0].shape[2]:{train_ds[0].shape[2]}')
        #
        # print(f'train_ds[1].shape[0]:{train_ds[1].shape[0]}')
        # print(f'train_ds[1].shape[1]:{train_ds[1].shape[1]}')
        # print(f'train_ds[1].shape[2]:{train_ds[1].shape[2]}')

        print(f'train_ds:{len(train_ds)}, test_ds:{len(test_ds)}')
        # exit()

        if verbose: print(f'load_study')
        study = load_study(database_name_optuna)

        if verbose: print(f'build_objective_with_data')
        objective_with_data = functools.partial(train_or_tune_model_v3, train_ds=train_ds, test_ds=test_ds, tuning=tuning, verbose=verbose, show_progression=show_progression)

        start_study = time.time()

        if verbose: print(f'start_study')
        study.optimize(objective_with_data, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

        best_params = study.best_params
        best_loss = study.best_value
        print(f"Best hyperparameters: {best_params}")
        print(f"Best loss: {best_loss}")

        end_study = time.time()
        print(f'Study: {round(end_study - start_study, ndigits=2)} Seconds')


    else:
        if verbose: print(f'transform_data_train_ds')
        # train_ds = df_to_dataset(db_wkdm, batch_size=32, shuffle=True)
        # train_ds, normalizer = normalize_target(train_ds, label_name='percent_change')

        train_ds, _, _ = prepare_data_for_nn(train_df, feature_cols=train_df.columns[:-1], target_col="Trend", sequence_length=sequence_length, scalers=scalers, onehot_encoder=onehot_encoder)

        study = load_study(database_name_optuna)
        best_params = study.best_params
        best_loss = study.best_value

        print(f"Best hyperparameters: {best_params}")
        print(f"Best loss: {best_loss}")

        if verbose: print(f'train_or_tune_model_v3()...')
        # model = train_or_tune_model_v3(train_ds=train_ds, test_ds=None, num_units=best_params['n_units'], dropout_rate=best_params['dropout_rate'], batch_size=best_params['batch_size'], epochs=10, verbose=verbose)
        model = train_or_tune_model_v3(best_params=best_params, train_ds=train_ds, tuning=tuning, verbose=verbose, show_progression=show_progression)


        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")

        model_save_path = f"saved_models/nn_model_{database_name_optuna}"

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        model.save(f'{model_save_path}/nn_model_{database_name_optuna}.keras')

        # Save scalers and onehot_encoder
        scalers_save_path = f"{model_save_path}/scalers.pkl"
        onehot_encoder_save_path = f"{model_save_path}/onehot_encoder.pkl"

        joblib.dump(scalers, scalers_save_path)
        joblib.dump(onehot_encoder, onehot_encoder_save_path)


# def prepare_data_for_nn(df, feature_cols, target_col, sequence_length, scaler=None, label_encoder=None):
#     X = df[feature_cols].values
#     y = df[target_col].values
#
#     X_scaled = scaler.transform(X)
#     y_encoded = label_encoder.transform(y)
#
#     # Erzeuge die Zeitreihendaten
#     generator = TimeseriesGenerator(X_scaled, y_encoded, length=sequence_length, batch_size=1)
#
#     # X_seq = np.array([x for x, _ in generator])
#     # y_seq = np.array([y for _, y in generator])
#
#     X_seq = np.array([x[0] for x in generator])
#     y_seq = np.array([y for _, y in generator])
#
#     y_seq = np.squeeze(y_seq, axis=-1)
#
#
#     # Überprüfe die Form, um sicherzustellen, dass sie korrekt ist
#     print(f"X_seq shape: {X_seq.shape}")  # Sollte (num_samples, sequence_length, num_features) sein
#     print(f"y_seq shape: {y_seq.shape}")  # Sollte (num_samples,) sein
#
#     return (X_seq, y_seq)


# def prepare_data_for_nn(df, feature_cols, target_col, sequence_length, scalers=None, onehot_encoder=None):
#     X = df[feature_cols].values
#     y = df[target_col].values
#
#     # Vorbereitung der Feature-Skalierer
#     if scalers is None:
#         scalers = {col: StandardScaler() for col in feature_cols}
#
#     # Features skalieren
#     features_scaled = np.zeros(X.shape)
#     for i, col in enumerate(feature_cols):
#         if scalers[col] is None:
#             scalers[col] = StandardScaler()
#             features_scaled[:, i] = scalers[col].fit_transform(df[col].values.reshape(-1, 1)).flatten()
#         else:
#             features_scaled[:, i] = scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()
#
#     X_scaled = features_scaled
#
#     # Kodierung der Zielwerte
#     # if onehot_encoder is None:
#     #     onehot_encoder = OneHotEncoder(sparse_output=False)
#     #     y_encoded = onehot_encoder.fit_transform(y.reshape(-1, 1))
#     # else:
#     #     y_encoded = onehot_encoder.transform(y.reshape(-1, 1))
#     y_encoded = onehot_encoder.transform(y.reshape(-1, 1))
#     # y_encoded = onehot_encoder.transform(y)
#
#     # Erzeuge die Zeitreihendaten
#     generator = TimeseriesGenerator(X_scaled, y_encoded, length=sequence_length, batch_size=1)
#
#     X_seq = np.array([x[0] for x, _ in generator])  # Hier sicherstellen, dass die Form (sequence_length, num_features) ist
#     y_seq = np.array([y for _, y in generator])
#
#     # Überprüfe die Form, um sicherzustellen, dass sie korrekt ist
#     print(f"X_seq shape: {X_seq.shape}")  # Sollte (num_samples, sequence_length, num_features) sein
#     print(f"y_seq shape: {y_seq.shape}")  # Sollte (num_samples, num_classes) sein
#
#     return (X_seq, y_seq), scalers, onehot_encoder


def prepare_data_for_nn(df, feature_cols, target_col, sequence_length, scalers=None, onehot_encoder=None):
    X = df[feature_cols].values
    y = df[target_col].values

    # Preparation of feature scalers
    if scalers is None:
        scalers = {col: StandardScaler() for col in feature_cols}

    # Scale features
    features_scaled = np.zeros(X.shape)
    for i, col in enumerate(feature_cols):
        if scalers[col] is None:
            scalers[col] = StandardScaler()
            features_scaled[:, i] = scalers[col].fit_transform(df[col].values.reshape(-1, 1)).flatten()
        else:
            features_scaled[:, i] = scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()

    X_scaled = features_scaled

    # Encode target values
    if onehot_encoder is None:
        onehot_encoder = OneHotEncoder(sparse_output=False)
        y_encoded = onehot_encoder.fit_transform(y.reshape(-1, 1))
    else:
        y_encoded = onehot_encoder.transform(y.reshape(-1, 1))

    # Create time series data
    # generator = TimeseriesGenerator(X_scaled, y_encoded, length=sequence_length, batch_size=1)
    #
    # X_seq = np.array([x[0] for x, _ in generator])  # Ensure shape is (sequence_length, num_features)
    # y_seq = np.array([y for _, y in generator])
    #
    # y_seq = y_seq.reshape(-1, y_encoded.shape[1])  # Ensure shape is (num_samples, num_classes)

    batch_size = 32
    generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_scaled, y_encoded, length=sequence_length, batch_size=batch_size)

    # Check if the generator is producing data
    num_samples = len(generator)
    print(f"Number of samples in generator: {num_samples}")
    if num_samples == 0:
        raise ValueError("The TimeseriesGenerator is empty. Check the sequence_length and the size of your data.")

    # Initialize lists for sequences and targets
    X_seq = []
    y_seq = []

    # Extract sequences and targets with progress bar
    for i in tqdm(range(num_samples), desc="Extracting sequences and targets"):
        x, y = generator[i]
        X_seq.append(x)
        y_seq.append(y)

    # Convert lists to numpy arrays
    X_seq = np.vstack(X_seq)
    y_seq = np.vstack(y_seq)

    # Check shapes to ensure they are correct
    print(f"X_seq shape: {X_seq.shape}")  # Should be (num_samples, sequence_length, num_features)
    print(f"y_seq shape: {y_seq.shape}")  # Should be (num_samples, num_classes)

    return (X_seq, y_seq), scalers, onehot_encoder


def split_data_v3(dataframe, test_size=0.2):
    train_size = int(len(dataframe) * (1 - test_size))
    train_df = dataframe[:train_size]
    test_df = dataframe[train_size:]
    return train_df, test_df


def df_to_dataset(dataframe, batch_size=32, shuffle=True, num_timesteps=None):
    # Assuming that the last column is the target
    data = dataframe.values
    X = data[:, :-1]  # all data except last column
    y = data[:, -1]  # last column as target
    X = X.reshape((X.shape[0], num_timesteps, 1))  # Reshape for LSTM, 12 timesteps each with 1 feature

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def normalize_target(dataset):
    # Annahme, dass die Labels als zweites Element in jedem Tupel im Dataset vorliegen
    labels = dataset.map(lambda features, label: label)

    # Erstellen einer Normalisierungsschicht
    normalizer = tf.keras.layers.Normalization(axis=None)
    normalizer.adapt(labels)

    # Anwendung der Normalisierung auf die Labels
    def apply_normalization(features, label):
        return features, normalizer(label)

    normalized_ds = dataset.map(apply_normalization)
    return normalized_ds, normalizer



def preprocess_data_v3(train_df=None, train_ds=None, verbose=False):

    encoded_features = []
    all_inputs = []


    numerical_cols = train_df.loc[:, ~train_df.columns.str.contains(y_target_data)].columns.tolist()

    # Numerical Columns
    if verbose: print(f'Preparing Numerical Columns')
    for header in tqdm(numerical_cols, total=len(numerical_cols), desc="Numerical Columns"):
        try:
            numeric_col = tf.keras.Input(shape=(1,), name=header)
            normalization_layer = get_normalization_layer_v3(header, train_ds)
            encoded_numeric_col = normalization_layer(numeric_col)
            all_inputs.append(numeric_col)
            encoded_features.append(encoded_numeric_col)
        except:
            print(traceback.format_exc())
            print(f'header:{header}')
            exit()


    if verbose: print(f'Concatenate all_features')
    all_features = tf.keras.layers.concatenate(encoded_features)

    return all_inputs, all_features



def get_normalization_layer_v3(name, dataset):
    normalizer = tf.keras.layers.Normalization(axis=None)
    feature_ds = dataset.map(lambda x, y: x[name])
    normalizer.adapt(feature_ds)
    return normalizer



def df_to_dataset1(dataframe, mode=None):
    df = dataframe.copy()

    labels = df.pop(y_target_data)

    df_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in df.items()}
    # print(df_dict)
    ds = tf.data.Dataset.from_tensor_slices((df_dict, labels))
    ds = ds.prefetch(1)

    return ds



def df_to_dataset1_predict(dataframe):
    df = dataframe.copy()

    df_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in df.items()}
    # print(df_dict)
    ds = tf.data.Dataset.from_tensor_slices((df_dict))
    ds = ds.prefetch(1)

    return ds



def train_or_tune_model_v3(trial=None, train_ds=None, test_ds=None, all_features=None, best_params=None, tuning=True, verbose=0, show_progression=False, num_timesteps=None, num_features=None):

    try:
        # Hyperparameter space
        if tuning:

                #TODO
                # min_neurons = int(math.sqrt(all_features.shape[1]))
                # max_neurons = all_features.shape[1]
                # min_neurons = int(num_features / 2)  # Basierend auf der Anzahl der Features
                # max_neurons = num_features * 2

                # print(f"n_units = trial.suggest_int('n_units', {min_neurons}, {max_neurons}, log=False)")

                activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
                # n_layers = trial.suggest_int('n_layers', 1, 3)
                # n_units = trial.suggest_int('n_units', min_neurons, max_neurons, log=False)
                n_units = trial.suggest_int('n_units', 1, 2000, log=False)
                # n_units = trial.suggest_int('n_units', min_neurons, 100, log=False)
                # n_units = trial.suggest_int('n_units', 1, 12, log=False)


                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.9, log=False)
                batch_size = trial.suggest_int('batch_size', 8, 1000, log=False)
                optimizer = trial.suggest_categorical('optimizer', ['adam'])

                if optimizer == 'adam':
                    optimizer = tf.keras.optimizers.Adam(learning_rate=tuning_adam_learning_rate)

        else:
            # n_layers = best_params['n_layers']
            n_units = best_params['n_units']
            activation = best_params['activation']
            optimizer = tf.keras.optimizers.Adam(learning_rate=building_adam_learning_rate)
            dropout_rate = best_params['dropout_rate']
            batch_size = best_params['batch_size']

        # print(f'n_layers:{n_layers},n_units:{n_units},dropout_rate:{dropout_rate},batch_size:{batch_size}')


        class CustomCallback(tf.keras.callbacks.Callback):
            def __init__(self, tuning=False):
                super().__init__()
                self.tuning = tuning

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()

            def on_epoch_end(self, epoch, logs=None):
                epoch_time = time.time() - self.epoch_start_time
                # current_lr = self.model.optimizer.lr.numpy()  # Zugriff auf die aktuelle Lernrate
                current_lr = self.model.optimizer.learning_rate.numpy()  # Zugriff auf die aktuelle Lernrate

                scale_value = 1
                if self.tuning:
                    print(f"Epoch: {epoch + 1:2d}, loss: {logs['loss'] / scale_value:15.3f}, Val loss: {logs.get('val_loss', 'N/A') / scale_value:15.3f}, Zeit: {epoch_time:6.2f} Sek., LR: {current_lr:.2e}")
                else:
                    print(f"Epoch: {epoch + 1:2d}, loss: {logs['loss'] / scale_value:15.3f}, Zeit: {epoch_time:6.2f} Sek., LR: {current_lr:.2e}")


        callbacks = []
        # if tuning:
        #     reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=tuning_reduce_lr_factor, min_delta=tuning_reduce_lr_min_delta, patience=tuning_reduce_lr_patience, min_lr=tuning_reduce_lr_min_lr, cooldown=tuning_reduce_lr_cooldown)
        #     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=tuning_early_stopping_patience, min_delta=tuning_early_stopping_min_delta, restore_best_weights=True)
        #     pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
        #     callbacks.extend([early_stopping, reduce_lr, pruning_callback, CustomCallback(tuning=tuning)])
        #
        # else:
        #     reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=building_reduce_lr_factor, min_delta=building_reduce_lr_min_delta, patience=building_reduce_lr_patience, min_lr=building_reduce_lr_min_lr, cooldown=building_reduce_lr_cooldown)
        #     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=building_early_stopping_patience, min_delta=building_early_stopping_min_delta, restore_best_weights=True)
        #     callbacks.extend([early_stopping, reduce_lr, CustomCallback(tuning=tuning)])

        if tuning:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=tuning_reduce_lr_factor, min_delta=tuning_reduce_lr_min_delta, patience=tuning_reduce_lr_patience, min_lr=tuning_reduce_lr_min_lr, cooldown=tuning_reduce_lr_cooldown)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=tuning_early_stopping_patience, min_delta=tuning_early_stopping_min_delta, restore_best_weights=True)
            pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')
            callbacks.extend([early_stopping, reduce_lr, pruning_callback, CustomCallback(tuning=tuning)])

        else:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=building_reduce_lr_factor, min_delta=building_reduce_lr_min_delta, patience=building_reduce_lr_patience, min_lr=building_reduce_lr_min_lr, cooldown=building_reduce_lr_cooldown)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=building_early_stopping_patience, min_delta=building_early_stopping_min_delta, restore_best_weights=True)
            callbacks.extend([early_stopping, reduce_lr, CustomCallback(tuning=tuning)])


        input_shape = (train_ds[0].shape[1], train_ds[0].shape[2])
        # input_shape = (train_ds.shape[1], train_ds.shape[2])

        print(f"Input shape: {input_shape}")

        # Model building
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(n_units, activation=activation, input_shape=input_shape, return_sequences=True))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.LSTM(n_units, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        # model.add(tf.keras.layers.Dense(1))
        # model.add(tf.keras.layers.Dense(3, activation='softmax'))  # Assuming 3 classes for one-hot encoding
        model.add(tf.keras.layers.Dense(train_ds[1].shape[1], activation='softmax'))  # Assuming 3 classes for one-hot encoding

        # Y_train.shape[1]

        
        # model.add(tf.keras.layers.Dense(1, activation='softmax'))  # Assuming 3 classes for one-hot encoding

        # model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        train_dataset = tf.data.Dataset.from_tensor_slices((train_ds[0], train_ds[1])).batch(batch_size)

        if tuning:
            test_dataset = tf.data.Dataset.from_tensor_slices((test_ds[0], test_ds[1])).batch(batch_size)
        # train_dataset = train_ds
        # test_dataset = test_ds

        #TODO
        fit_params = {
            'x': train_dataset,
            # 'epochs': 1,
            'epochs': 999999,
            'verbose': verbose,
            'callbacks': callbacks,
            'batch_size': batch_size,
            # 'workers': 32,
            # 'use_multiprocessing': True
        }

        try:

            # Train the model
            if tuning:
                fit_params['validation_data'] = test_dataset
                if show_progression:
                    history = model.fit(**fit_params)  # Speichern Sie die Trainingshistorie

                    try:
                        # Plot loss und val_loss
                        plt.plot(history.history['loss'], label='Training loss')
                        plt.plot(history.history['val_loss'], label='Validation loss')

                        # Hinzufügen von Titel und Beschriftungen
                        plt.title('Training and Validation loss Over Epochs')
                        plt.ylabel('loss')
                        plt.xlabel('Epoch')

                        # Legende anzeigen
                        plt.legend()

                        # Zeigen Sie das Diagramm an
                        plt.show()
                    except:
                        pass
                else:
                    model.fit(**fit_params)  # Speichern Sie die Trainingshistorie


                if test_dataset:
                    loss, mae = model.evaluate(test_dataset, verbose=verbose)
                    return loss  # Rückgabe nur des Verlustes
            else:
                if show_progression:
                    history = model.fit(**fit_params)  # Speichern Sie die Trainingshistorie

                    try:
                        # Plot the validation loss
                        plt.plot(history.history['loss'])
                        plt.title('Validation loss Over Epochs')
                        plt.ylabel('loss')
                        plt.xlabel('Epoch')
                        plt.show()
                    except:
                        pass
                else:
                    model.fit(**fit_params)  # Speichern Sie die Trainingshistorie


        except optuna.exceptions.TrialPruned as e:
            # print(traceback.format_exc())
            # print(f"Trial pruned: {e} with params : {{'dropout_rate': {dropout_rate}, 'activation': '{activation}', 'optimizer': 'adam', 'n_layers': {n_layers}, 'n_units': {n_units}, 'batch_size': {batch_size}}}")
            return float('inf')  # oder was immer Optuna erwartet, um die Beschneidung zu signalisieren

        except AttributeError as e:
            # Handle the AttributeError
            print(traceback.format_exc())
            print(f"Es trat ein Fehler auf: {e}")
            return float('inf')  # oder passende Fehlerbehandlung

        except Exception as e:
            # Fängt alle anderen Ausnahmen, die nicht spezifisch abgefangen wurden
            print(traceback.format_exc())
            print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
            return float('inf')  # oder passende Fehlerbehandlung

        return model

    except Exception:
        print(traceback.format_exc())



def generate_stock_prices(open_price, noise_duration, noise_strength, pattern_prices, trend_changes, num_cycles, start_date):
    noise_steps = noise_duration // 5
    pattern_length = len(pattern_prices)
    trend_length = len(trend_changes)

    total_steps = (noise_steps + pattern_length + trend_length) * num_cycles + noise_steps
    prices = np.zeros(total_steps)

    datetimes = []
    current_datetime = datetime.strptime(start_date, "%m/%d/%Y")

    step = 0
    for _ in range(num_cycles):
        if step + noise_steps > total_steps:
            break
        noise = np.random.normal(open_price, noise_strength, noise_steps)
        prices[step:step + noise_steps] = noise
        for _ in range(noise_steps):
            datetimes.append(current_datetime)
            current_datetime += timedelta(minutes=5)
        step += noise_steps

        if step + pattern_length > total_steps:
            break
        prices[step:step + pattern_length] = pattern_prices
        for _ in range(pattern_length):
            datetimes.append(current_datetime)
            current_datetime += timedelta(minutes=5)
        step += pattern_length

        if step + trend_length > total_steps:
            break
        last_price = prices[step - 1]
        prices[step:step + trend_length] = [last_price + change for change in trend_changes]
        for _ in range(trend_length):
            datetimes.append(current_datetime)
            current_datetime += timedelta(minutes=5)
        step += trend_length

    if step < total_steps:
        final_noise = np.random.normal(open_price, noise_strength, total_steps - step)
        prices[step:total_steps] = final_noise
        for _ in range(total_steps - step):
            datetimes.append(current_datetime)
            current_datetime += timedelta(minutes=5)

    df = pd.DataFrame({
        "Datetime": datetimes,
        "Close": prices
    })

    # Neue Spalte hinzufügen: Differenz zum vorherigen Preis
    df["Close_Diff"] = df["Close"].diff().fillna(0)


    return df


def calculate_averages(df, history_steps, future_steps):
    df[f"Close_FMA"] = df["Close"].shift(-future_steps).rolling(window=future_steps).mean()
    df[f"Close_FEMA"] = df["Close"].shift(-future_steps).ewm(span=future_steps, adjust=False).mean()

    df[f"Close_Diff_MA"] = df["Close_Diff"].rolling(window=history_steps).mean()

    df[f"Close_Diff_FMA"] = df["Close_Diff"].shift(-future_steps).rolling(window=future_steps).mean()
    df[f"Close_Diff_FMA"] = df["Close_Diff"].shift(-future_steps).rolling(window=future_steps).mean()
    df[f"Close_Diff_FEMA"] = df["Close_Diff"].shift(-future_steps).ewm(span=future_steps, adjust=False).mean()


def classify_trend(df, future_steps):
    threshhold = 5

    df["Future_Close"] = df["Close"].shift(-future_steps)
    df["Close_FMA_minus_Close"] = df["Close_FMA"] - df["Close"]
    df["Close_FEMA_minus_Close"] = df["Close_FEMA"] - df["Close"]
    df["Future_Close_Diff"] = df["Future_Close"] - df["Close"]

    # df[f"Future_Close_FMA"] = df[f"Close_FMA"].shift(-future_steps)
    # df[f"Future_Close_Diff_FMA"] = df[f"Close_Diff_FMA"].shift(-future_steps)

    df["Trend_Close"] = (df["Future_Close"] - df["Close"]).apply(lambda x: "Up" if x >= threshhold else ("Down" if x <= -threshhold else "Stable"))
    df["Trend_Close_FMA"] = (df["Close_FMA"] - df["Close"]).apply(lambda x: "Up" if x >= threshhold else ("Down" if x <= -threshhold else "Stable"))
    df["Trend_Close_FEMA"] = (df["Close_FEMA"] - df["Close"]).apply(lambda x: "Up" if x >= threshhold else ("Down" if x <= -threshhold else "Stable"))

    df["Trend_Close_Diff_FMA"] = df["Close_Diff_FMA"].apply(lambda x: "Up" if x >= threshhold else ("Down" if x <= -threshhold else "Stable"))
    df["Trend_Close_Diff_FEMA"] = df["Close_Diff_FEMA"].apply(lambda x: "Up" if x >= threshhold else ("Down" if x <= -threshhold else "Stable"))


    # df["Trend_Close_FMA"] = df[f"Close_FMA"].diff().apply(lambda x: "Up" if x >= threshhold else ("Down" if x < -threshhold else "Stable"))
    # df["Trend_Close_Diff_FMA"] = df[f"Close_Diff_FMA"].diff().apply(lambda x: "Up" if x >= threshhold else ("Down" if x < -threshhold else "Stable"))

    # df["Trend_FUTURE_Close_FMA"] = (df[f"Future_Close_FMA"] - df[f"Close_FMA"]).apply(lambda x: "Up" if x >= threshhold else ("Down" if x < -threshhold else "Stable"))
    # df["Trend_FUTURE_Close_Diff_FMA"] = (df[f"Future_Close_Diff_FMA"] - df[f"Close_Diff_FMA"]).apply(lambda x: "Up" if x >= threshhold else ("Down" if x < -threshhold else "Stable"))

    df.dropna(inplace=True)


def save_to_db(dataframe, to_table, db):

    # Erstellt eine Verbindung zur Datenbank
    engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://root:@localhost/{db}', echo=False)
    conn = engine.connect()
    dataframe.to_sql(con=conn, name=f'{to_table}', if_exists='replace', index=False)
    conn.close()
    print(f"Daten erfolgreich in Tabelle '{to_table}' gespeichert.")



def plot_stock_prices(df):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["Datetime"], df["Close"], label='Aktienkurs', color='blue')


    # trend = "Trend_Close"
    # trend_streangth = "Future_Close_Diff"
    trend = "Trend_Close_FMA"
    trend_streangth = "Close_FMA_minus_Close"
    # trend = "Trend_Close_FEMA"
    # trend_streangth = "Close_FEMA_minus_Close"
    # trend = "Trend_Close_Diff_FMA"
    # trend_streangth = "Close_Diff_FMA"
    # trend = "Trend_Close_Diff_FEMA"
    # trend_streangth = "Close_Diff_FEMA"

    for i in range(len(df)):
        if df[trend].iloc[i] == "Up":
            ax1.plot(df["Datetime"].iloc[i], df["Close"].iloc[i], marker='^', color='green', markersize=10,
                     label='Up' if i == 0 else "")
        elif df[trend].iloc[i] == "Down":
            ax1.plot(df["Datetime"].iloc[i], df["Close"].iloc[i], marker='v', color='red', markersize=10,
                     label='Down' if i == 0 else "")
        else:
            ax1.plot(df["Datetime"].iloc[i], df["Close"].iloc[i], marker='o', color='grey', markersize=5,
                     label='Stable' if i == 0 else "")

    # Balkendiagramm hinzufügen
    ax2 = ax1.twinx()
    ax2.set_frame_on(False)  # Versteckt den Rahmen der zweiten Achse
    bar_width = 0.01  # Breite der Balken

    colors = df[trend].apply(lambda x: 'green' if x == 'Up' else ('red' if x == 'Down' else 'grey'))
    ax2.bar(df["Datetime"], df[trend_streangth], width=bar_width, color=colors, alpha=0.3, label='Trend Strength')

    ax1.set_title('Simulierte Aktienkurse mit Rauschen, Muster und Trendphasen')
    ax1.set_xlabel('Datum und Zeit')
    ax1.set_ylabel('Kurs')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    fig.autofmt_xdate()  # Automatische Formatierung der x-Achse
    plt.show()


def predict_and_integrate(df, model_path, scalers_path, onehot_encoder_path, sequence_length=12, target_col="Trend"):
    # Laden des Modells und der Transformationsobjekte
    model = tf.keras.models.load_model(model_path)
    scalers = joblib.load(scalers_path)
    onehot_encoder = joblib.load(onehot_encoder_path)

    feature_cols = [col for col in df.columns if col != target_col]

    # Vorbereitung der Daten
    X = df[feature_cols].values
    y = df[target_col].values

    # Skalieren der Features
    features_scaled = np.zeros(X.shape)
    for i, col in enumerate(feature_cols):
        features_scaled[:, i] = scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()

    X_scaled = features_scaled

    #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Erstellung von Zeitreihen-Daten
    generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_scaled, y, length=sequence_length, batch_size=1)

    X_seq = np.array([x[0] for x, _ in generator])  # Sicherstellen, dass die Form korrekt ist

    # Vorhersagen treffen
    y_pred_encoded = model.predict(X_seq)

    # Rücktransformation der One-Hot-Encoded-Vorhersagen
    y_pred = onehot_encoder.inverse_transform(y_pred_encoded)

    # Die Vorhersagen in das ursprüngliche DataFrame integrieren
    prediction_col = target_col + "_predicted"
    df[prediction_col] = np.nan

    # Stellen Sie sicher, dass die Vorhersagenspalte den richtigen Datentyp hat
    df[prediction_col] = df[prediction_col].astype(object)
    df[prediction_col].iloc[sequence_length:] = y_pred.flatten()

    return df


def query_database(db, table_name, columns=None, conditions=None, engine_kwargs={}, query_kwargs={}):
    # Erstellen der Verbindungs-URL für die Datenbank
    connection_url = f"mysql+mysqlconnector://root:@localhost/{db}"

    # Erstellen eines SQLAlchemy Engine-Objekts
    engine = sqlalchemy.create_engine(connection_url, **engine_kwargs)

    # Bestimmen, welche Spalten abgefragt werden sollen
    if columns:
        columns_str = ", ".join(columns)  # Umwandeln der Spaltenliste in einen String
    else:
        columns_str = "*"

    # Aufbauen der Abfrage
    query = f"SELECT {columns_str} FROM {table_name}"
    if conditions:
        query += f" WHERE {conditions}"

    # Ausführen der Abfrage und Schließen der Verbindung
    with engine.connect() as conn:
        dataframe = pd.read_sql(query, con=conn, **query_kwargs)

    return dataframe

if __name__ == "__main__":
    # open_Close = 500
    # noise_duration = 60  # 60 Minuten
    # noise_strength = 1  # Stärke des Rauschens
    # pattern_prices = [500 + x for x in [10, 20, 15, 30, 10, 20, 5]]
    # trend_changes = [-10, -20, -15, -30, -10, -25, -5]  # Abwärtstrend nach einem Head and Shoulders Muster
    #
    # num_cycles = 10  # Anzahl der Wiederholungen
    # start_date = "03/28/2022"  # Startdatum
    #
    # df = generate_stock_prices(open_Close, noise_duration, noise_strength, pattern_prices, trend_changes, num_cycles, start_date)
    #
    # history_steps = 12
    # future_steps = 6
    # calculate_averages(df, history_steps=history_steps, future_steps=future_steps)
    # classify_trend(df, future_steps=future_steps)
    #
    # print(df.head(50))  # Zeige die ersten 50 Zeilen des DataFrames
    # plot_stock_prices(df)
    #
    # # df = df[["Close_Diff", "Close_FMA_minus_Close", "Trend_Close_FMA"]]
    # # df = df[["Close_Diff", "Trend_Close_FMA"]]
    # df = df[["Datetime", "Close", "Close_Diff", "Close_FMA_minus_Close", "Trend_Close_FMA"]]
    #
    # df = df.rename(columns={'Trend_Close_FMA': 'Trend'})
    #
    # print(df.head(50))  # Zeige die ersten 50 Zeilen des DataFrames
    #
    # # save_to_db(dataframe=df, to_table="muster_5min_dataframe_db", db="trading_bot")
    # save_to_db(dataframe=df, to_table="muster_5min_dataframe_db_close", db="trading_bot")
    #
    # exit()


    database_name = "trading_bot"
    table_dataset = "muster_5min_dataframe_db"

    # database_name_optuna = f"test_{2}"  # num_cycles = 10 history_steps = 12 future_steps = 6 ["Close_Diff", "Close_FMA_minus_Close", "Trend_Close_FMA"]
    # database_name_optuna = f"test_{3}"  # num_cycles = 10 history_steps = 12 future_steps = 6 ["Close_Diff", "Trend_Close_FMA"] keine spürbare verbesserung mit weniger spalten # muss gelöscht werden, falsche output dense
    # database_name_optuna = f"test_{4}"  # num_cycles = 20 history_steps = 12 future_steps = 6 ["Close_Diff", "Trend_Close_FMA"] keine spürbare verbesserung mit mehr daten # muss gelöscht werden, falsche output dense
    # database_name_optuna = f"test_{54}"  # num_cycles = 10 history_steps = 12 future_steps = 6 ["Close_Diff", "Trend_Close_FMA"] keine spürbare verbesserung mit mehr daten # muss gelöscht werden, falsche output dense
    # database_name_optuna = f"test_{6}"  # num_cycles = 10 history_steps = 12 future_steps = 6 ["Close_Diff", "Trend_Close_FMA"] keine spürbare verbesserung mit mehr daten
    # database_name_optuna = f"test_{91}"  # num_cycles = 10 history_steps = 12 future_steps = 6 ["Close_Diff", "Trend_Close_FMA"] keine spürbare verbesserung mit mehr daten
    database_name_optuna = f"test_{912}"  # num_cycles = 10 history_steps = 12 future_steps = 6 ["Close_Diff", "Trend_Close_FMA"] keine spürbare verbesserung mit mehr daten

    workers = 10
    # train_model_v3(tuning=True, n_trials=workers * 2, n_jobs=workers, database_name="trading_bot", table_name=table_dataset, database_name_optuna=database_name_optuna, show_progression=False, verbose=True)
    train_model_v3(tuning=False, n_trials=workers * 1, n_jobs=workers, database_name="trading_bot", table_name=table_dataset, database_name_optuna=database_name_optuna, show_progression=False, verbose=True)
    exit()

    model_path = f"saved_models/nn_model_{database_name_optuna}"
    scalers_save_path = f"{model_path}/scalers.pkl"
    onehot_encoder_save_path = f"{model_path}/onehot_encoder.pkl"

    df = query_database(db=database_name, table_name=table_dataset)
    original_df = query_database(db=database_name, table_name="muster_5min_dataframe_db_close")

    print(df.tail())
    df_with_predictions = predict_and_integrate(df=df, model_path=model_path, scalers_path=scalers_save_path, onehot_encoder_path=onehot_encoder_save_path)
    df_with_predictions["Close"] = original_df["Close"]
    df_with_predictions["Datetime"] = original_df["Datetime"]
    df_with_predictions["Close_FMA_minus_Close"] = original_df["Close_FMA_minus_Close"]


    # print(df_with_predictions)
    print(df_with_predictions.head())


    df_with_predictions = df_with_predictions[["Datetime", "Close", "Close_FMA_minus_Close", "Trend_predicted"]]
    df_with_predictions = df_with_predictions.rename(columns={'Trend_predicted': 'Trend_Close_FMA'})
    print(df_with_predictions.head())


    original_df = original_df.rename(columns={'Trend': 'Trend_Close_FMA'})

    plot_stock_prices(original_df[-50:])
    plot_stock_prices(df_with_predictions[-50:])
    #
    #
    # trend = "Trend_Close_FMA"
    # trend_streangth = "Close_FMA_minus_Close"