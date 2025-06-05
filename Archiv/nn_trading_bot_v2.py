# ======================================================================================================================
# Info
# ======================================================================================================================
# Date: 11.03.2024
# Name:
# Aufgabe:
# Version: V4.0
# Author: Niklas Fliegel
# ======================================================================================================================
# libraries
# ======================================================================================================================

# C:\Users\Nik\AppData\Local\Programs\Python\Python39\python.exe -m pip install
# C:\Users\Nik\AppData\Local\Programs\Python\Python39\python.exe -m pip install tensorflow==2.10.0
import custom_functions_fov_v1 as cf
# import identify_model_v3

import random
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU deaktivieren
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Wichtig für BERT Preprocessing

import pandas as pd
import optuna
import optuna.visualization as vis
from optuna.integration import TFKerasPruningCallback


import mysql.connector
from mysql.connector import Error
import functools

from sqlalchemy import create_engine
from difflib import SequenceMatcher
from tqdm import tqdm
import traceback
import matplotlib.pyplot as plt
import time
import datetime
import math
import re
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential


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
# ======================================================================================================================
# Custom_functions
# ======================================================================================================================
def calculate_neurons_and_print_code(features_shape):
    input_features = features_shape[1]  # Anzahl der Eingabefeatures
    output_units = 1  # Für Regressionsprobleme üblicherweise 1 Ausgabeeinheit

    # Regel 1: Geometrisches Mittel
    neurons_rule1 = int(math.sqrt(input_features * output_units))

    # Regel 2: Obergrenze
    neurons_rule2 = 2 * input_features

    print(f"n_units = trial.suggest_int('n_units', {neurons_rule1}, {min(neurons_rule2, 2000)}/{input_features})  # Passen Sie die Obergrenze nach Bedarf an")


def df_to_dataset(dataframe, mode=None):
    df = dataframe.copy()

    labels = df.pop(y_target_data)
    # sample_weight = df.pop(sample_weight_column)

    df_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in df.items()}
    # print(df_dict)
    # ds = tf.data.Dataset.from_tensor_slices((df_dict, labels, sample_weight))
    ds = tf.data.Dataset.from_tensor_slices((df_dict, labels))

    ds = ds.prefetch(1)

    return ds


def df_to_dataset_predict(dataframe, mode=None):
    df = dataframe.copy()

    # labels = df.pop(y_target_data)
    # sample_weight = df.pop(sample_weight_column)

    df_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in df.items()}
    # print(df_dict)
    ds = tf.data.Dataset.from_tensor_slices(df_dict)
    ds = ds.prefetch(1)

    return ds





def load_data(database_name, table_name):

    source_config = f'mysql+mysqlconnector://{connection_config["user"]}:{connection_config["password"]}@{connection_config["host"]}/{database_name}'
    engine = create_engine(source_config, echo=False)

    dataframe = pd.read_sql(f'SELECT * FROM {table_name}', con=engine)

    print(f'dataframe_len:{len(dataframe)}')
    dataframe.dropna(inplace=True)
    return dataframe


def random_selection_from_db(marke=None, db_wkdm=None, price_quality_threshold=None):

    # Wählen Sie zufällige Daten aus der Datenbank für die gegebene Marke
    if marke != "":
        filtered_data = db_wkdm[(db_wkdm['marke_opt'] == marke)
                                & (db_wkdm['price_quality'] >= price_quality_threshold)
                                ]
    else:
        filtered_data = db_wkdm[db_wkdm['price_quality'] >= price_quality_threshold]

    if filtered_data.empty:
        raise ValueError(f"Keine Daten für die Marke {marke} gefunden.")

    random_row = filtered_data.sample(1).iloc[0]


    # TODO
    # Alter berechnen
    # random_row['alter_jahre'] = calculate_age(random_row[baujahr_spalte])


    return random_row



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

    storage_name = f"mysql+mysqlconnector://{connection_config['user']}:{connection_config['password']}@{connection_config['host']}/optuna_{database_name_optuna}"
    # storage_name = f"mysql+mysqlconnector://{connection_config['user']}:{connection_config['password']}@{connection_config['host']}:{connection_config['port']}/{database_name}"
    study_name = "study"

    create_database(host=connection_config['host'], user=connection_config['user'], password=connection_config['password'], db_name=f'optuna_{database_name_optuna}')
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)

    try:
        print(f'best_params : {study.best_params}')
        print(f'best_value : {study.best_value}')
    except:
        print(f'best_params : no Data yet')
        print(f'best_value : no Data yet')

    return study


def show_study(database_name):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = load_study(database_name)
    vis.plot_optimization_history(study).show()
    vis.plot_parallel_coordinate(study).show()
    vis.plot_slice(study).show()


def show_study_params(database_name, n_last_trials):
    study = load_study(database_name)
    df = study.trials_dataframe()
    df = df.loc[~(df["value"] == float('inf')) & ~pd.isna(df["value"])]

    sorted_df = df.sort_values(by='value', ascending=False)

    # Wählen Sie die letzten n Trials
    last_n_trials = sorted_df.tail(n_last_trials)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    print(last_n_trials)


def calculate_age(baujahr_jahr):
    current_year = pd.Timestamp.now().year
    return current_year - int(baujahr_jahr)


def string_match_ratio(s1, s2):
    return SequenceMatcher(None, s1.lower().replace(" ", ""), s2.lower().replace(" ", "")).ratio()



def calculate_model_size_and_recommendation(n_layers, n_units, dropout_rate, batch_size, input_features):
    # Berechnung der Gesamtanzahl der Parameter im Modell
    total_params = 0
    for i in range(n_layers):
        if i == 0:
            # Parameter für die erste Schicht (Eingabeschicht)
            total_params += input_features * n_units
        else:
            # Parameter für nachfolgende Schichten
            total_params += n_units * n_units
        # Berücksichtigung der Biases
        total_params += n_units

    # Schätzung der Modellgröße
    model_size = total_params * (1 - dropout_rate)

    # Empfehlung für CPU oder GPU basierend auf der Modellgröße und Batch-Größe
    if model_size < 500000 and batch_size < 100:  # Grenzwerte können angepasst werden
        recommendation = "CPU"
    else:
        recommendation = "GPU"

    return model_size, recommendation


# ======================================================================================================================
# V_3
# ======================================================================================================================
# Step 1
########################################################################################################################

# def train_model_v3(vectorize=None, n_trials=None, n_jobs=None, database_name=None, table_name=None, show_progression=False, tune_with_best=False, load_preprocess_data=False, verbose=False):
#
#     if verbose: print(f'load_data()')
#     db_wkdm = load_data(database_name=database_name, table_name=table_name)
#
#
#     if verbose: print(f'split_data_v3')
#     train_df, test_df = split_data_v3(db_wkdm)
#
#     if verbose: print(f'transform_data_train_ds')
#     train_ds = df_to_dataset(train_df)
#
#     if verbose: print(f'transform_data_test_ds')
#     test_ds = df_to_dataset(test_df)
#
#     all_inputs, all_features = preprocess_data_v3(train_df=train_df, train_ds=train_ds, verbose=verbose)
#
#     print(f'all_features shape: {all_features.shape}')
#     calculate_neurons_and_print_code(all_features.shape)
#
#     if verbose: print(f'load_study')
#     study = load_study(database_name)
#     best_params = None
#
#     if verbose: print(f'build_objective_with_data')
#     objective_with_data = functools.partial(train_or_tune_model_v3, train_ds=train_ds, all_inputs=all_inputs, all_features=all_features, best_params=best_params, tune_with_best=tune_with_best, test_ds=test_ds, tuning=True, verbose=0, show_progression=show_progression)
#
#     start_study = time.time()
#
#     if verbose: print(f'start_study')
#     study.optimize(objective_with_data, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
#
#
#     best_params = study.best_params
#     best_loss = study.best_value
#     print(f"Best hyperparameters: {best_params}")
#     print(f"Best loss: {best_loss}")
#
#     end_study = time.time()
#     print(f'Study: {round(end_study - start_study, ndigits=2)} Seconds')
def train_model_v3(n_trials=None, n_jobs=None, database_name=None, database_name_optuna=None, table_name=None, show_progression=False, tune_with_best=False, load_preprocess_data=False, verbose=False):
    if verbose: print(f'load_data()')
    db_wkdm = load_data(database_name=database_name, table_name=table_name)

    # print(db_wkdm.head())
    # exit()

    if verbose: print(f'split_data_v3')
    train_df, test_df = split_data_v3(db_wkdm)

    if verbose: print(f'Preparing data for training')
    X_train, Y_train, scaler_train = prepare_data(train_df, target_column_name="Close", history_points=60)
    X_test, Y_test, scaler_test = prepare_data(test_df, target_column_name="Close", history_points=60)


    if verbose: print(f'load_study')
    study = load_study(database_name_optuna)
    best_params = None

    if verbose: print(f'build_objective_with_data')
    objective_with_data = functools.partial(train_or_tune_model_v3, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, best_params=best_params, tuning=True, verbose=0, show_progression=show_progression)

    start_study = time.time()

    if verbose: print(f'start_study')
    study.optimize(objective_with_data, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    best_params = study.best_params
    best_loss = study.best_value
    print(f"Best hyperparameters: {best_params}")
    print(f"Best loss: {best_loss}")

    end_study = time.time()
    print(f'Study: {round(end_study - start_study, ndigits=2)} Seconds')
########################################################################################################################

def prepare_data1(df, history_points=60):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    features = data_scaled[:, :-1]
    target = data_scaled[:, -1]  # Angenommen, 'Close' ist die letzte Spalte

    generator = TimeseriesGenerator(features, target, length=history_points, batch_size=len(features))
    X, Y = [], []
    for i in range(len(generator)):
        x, y = generator[i]
        X.append(x)
        Y.append(y)

    # Die Zeile, die das Problem verursacht hat, wurde entfernt.
    # `X` und `Y` sind bereits in der korrekten Form.
    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)
    return X, Y, scaler


def prepare_data(df, target_column_name, history_points=60):
    target = df[target_column_name].values
    features = df.drop(target_column_name, axis=1).values

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    generator = TimeseriesGenerator(features_scaled, target, length=history_points, batch_size=len(features_scaled))

    X, Y = [], []
    for i in range(len(generator)):
        x, y = generator[i]
        X.append(x)
        Y.append(y)

    X = np.array(X).squeeze()  # <--- Versuch, die Form anzupassen
    Y = np.array(Y).reshape(-1, 1)

    # Überprüfung der Form der Daten
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    return X, Y, scaler


# Step 2
########################################################################################################################

def split_data_v3(dataframe, test_size=0.2):
    train_size = int(len(dataframe) * (1 - test_size))
    train_df = dataframe[:train_size]
    test_df = dataframe[train_size:]
    return train_df, test_df


# Step 2_2
########################################################################################################################
def preprocess_data_v3(train_df=None, train_ds=None, verbose=False):

    encoded_features = []
    all_inputs = []

    # categorical_str_cols = categorical_str_cols_v3
    # categorial_int_cols = categorial_int_cols_v3
    # numerical_cols = numerical_cols_v3

    such_string = 'input'
    numerical_cols = train_df.filter(like=such_string).columns.tolist()

    # # Categorical String Columns
    # if verbose: print(f'Preparing Categorical String Columns')
    # for header in tqdm(categorical_str_cols, total=len(categorical_str_cols), desc="Categorical String Columns"):
    #     num_unique = len(train_df[header].unique().tolist()) + 1
    #     categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    #     encoding_layer = get_category_encoding_layer_v3(name=header,
    #                                                     dataset=train_ds,
    #                                                     dtype='string',
    #                                                     max_tokens=num_unique)
    #     encoded_categorical_col = encoding_layer(categorical_col)
    #     all_inputs.append(categorical_col)
    #     encoded_features.append(encoded_categorical_col)


    # # Categorical Number Columns
    # if verbose: print(f'Preparing Categorical Number Columns')
    # for header in tqdm(categorial_int_cols, total=len(categorial_int_cols), desc="Categorical Number Columns"):
    #     # print(f'header:{header}')
    #     # print(f'train_df[header]:{train_df[header]}')
    #     num_unique = len(train_df[header].unique().tolist()) + 1
    #     abs_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')
    #     encoding_layer = get_category_encoding_layer_v3(name=header,
    #                                                  dataset=train_ds,
    #                                                  dtype='int64',
    #                                                  max_tokens=num_unique)
    #     encoded_abs_col = encoding_layer(abs_col)
    #     all_inputs.append(abs_col)
    #     encoded_features.append(encoded_abs_col)


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


# Step 2_2_1
########################################################################################################################
def get_normalization_layer_v3(name, dataset):
    normalizer = tf.keras.layers.Normalization(axis=None)
    feature_ds = dataset.map(lambda x, y: x[name])
    normalizer.adapt(feature_ds)
    return normalizer


def get_category_encoding_layer_v3(name, dataset, dtype, max_tokens=None):
    if dtype == 'string':
        index = tf.keras.layers.StringLookup(max_tokens=max_tokens)
    else:
        index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)
    feature_ds = dataset.map(lambda x, y, z: x[name])
    index.adapt(feature_ds)
    encoder = tf.keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())
    return lambda feature: encoder(index(feature))


# Step 3
########################################################################################################################
def train_or_tune_model_v33(trial=None, train_ds=None, all_inputs=None, all_features=None, best_params=None, tune_with_best=None, test_ds=None, tuning=True, verbose=0, show_progression=False):

    try:
        # Hyperparameter space
        if tuning:
            #norm TODO
            min_neurons = int(math.sqrt(all_features.shape[1]))
            max_neurons = all_features.shape[1]
            print(f"n_units = trial.suggest_int('n_units', {min_neurons}, {max_neurons}, log=False)")

            activation = trial.suggest_categorical('activation', ['relu'])
            n_layers = trial.suggest_int('n_layers', 1, 1)
            n_units = trial.suggest_int('n_units', min_neurons, max_neurons, log=False)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.9, log=False)
            batch_size = trial.suggest_int('batch_size', 8, 1000, log=False)
            optimizer = trial.suggest_categorical('optimizer', ['adam'])

            if optimizer == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=tuning_adam_learning_rate)

        else:
            n_layers = best_params['n_layers']
            n_units = best_params['n_units']
            activation = best_params['activation']
            optimizer = tf.keras.optimizers.Adam(learning_rate=building_adam_learning_rate)
            dropout_rate = best_params['dropout_rate']
            batch_size = best_params['batch_size']


        class CustomCallback(tf.keras.callbacks.Callback):
            def __init__(self, tuning=False):
                super().__init__()
                self.tuning = tuning

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()

            def on_epoch_end(self, epoch, logs=None):
                epoch_time = time.time() - self.epoch_start_time
                current_lr = self.model.optimizer.lr.numpy()  # Zugriff auf die aktuelle Lernrate
                scale_value = 1
                if self.tuning:
                    print(f"Epoch: {epoch + 1:2d}, Loss: {logs['loss'] / scale_value:15.3f}, Val Loss: {logs.get('val_loss', 'N/A') / scale_value:15.3f}, Zeit: {epoch_time:6.2f} Sek., LR: {current_lr:.2e}")
                else:
                    print(f"Epoch: {epoch + 1:2d}, Loss: {logs['loss'] / scale_value:15.3f}, Zeit: {epoch_time:6.2f} Sek., LR: {current_lr:.2e}")


        callbacks = []
        if tuning:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=tuning_reduce_lr_factor, min_delta=tuning_reduce_lr_min_delta,
                                          patience=tuning_reduce_lr_patience, min_lr=tuning_reduce_lr_min_lr, cooldown=tuning_reduce_lr_cooldown)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=tuning_early_stopping_patience,
                                           min_delta=tuning_early_stopping_min_delta, restore_best_weights=True)

            if not tune_with_best:
                pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
                callbacks.extend([early_stopping, reduce_lr, pruning_callback, CustomCallback(tuning=tuning)])
            else:
                callbacks.extend([early_stopping, reduce_lr, CustomCallback(tuning=tuning)])

        else:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=building_reduce_lr_factor, min_delta=building_reduce_lr_min_delta,
                                          patience=building_reduce_lr_patience, min_lr=building_reduce_lr_min_lr, cooldown=building_reduce_lr_cooldown)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=building_early_stopping_patience,
                                           min_delta=building_early_stopping_min_delta, restore_best_weights=True)
            callbacks.extend([early_stopping, reduce_lr, CustomCallback(tuning=tuning)])


        x = all_features
        #############################################
        for _ in range(n_layers):
            x = tf.keras.layers.Dense(n_units, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        #############################################

        model = tf.keras.Model(inputs=all_inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss='mse', weighted_metrics=[])


        train_ds = train_ds.batch(batch_size)
        if tuning:
            test_ds = test_ds.batch(batch_size)  # Falls Sie einen Test-DS haben

        #TODO
        fit_params = {
            'x': train_ds,
            'epochs': 999999,
            'verbose': verbose,
            'callbacks': callbacks,
            'batch_size': batch_size,
            'workers': 32,
            # 'use_multiprocessing': True
        }

        try:

            # Train the model
            if tuning:
                fit_params['validation_data'] = test_ds
                if show_progression:
                    history = model.fit(**fit_params)  # Speichern Sie die Trainingshistorie

                    try:
                        # Plot loss und val_loss
                        plt.plot(history.history['loss'], label='Training Loss')
                        plt.plot(history.history['val_loss'], label='Validation Loss')

                        # Hinzufügen von Titel und Beschriftungen
                        plt.title('Training and Validation Loss Over Epochs')
                        plt.ylabel('Loss')
                        plt.xlabel('Epoch')

                        # Legende anzeigen
                        plt.legend()

                        # Zeigen Sie das Diagramm an
                        plt.show()
                    except:
                        pass
                else:
                    model.fit(**fit_params)  # Speichern Sie die Trainingshistorie

                # Evaluate and report the intermediate result
                if test_ds:
                    loss = model.evaluate(test_ds, verbose=verbose)
                    return loss
            else:
                if show_progression:
                    history = model.fit(**fit_params)  # Speichern Sie die Trainingshistorie

                    try:
                        # Plot the validation loss
                        plt.plot(history.history['loss'])
                        plt.title('Validation Loss Over Epochs')
                        plt.ylabel('Loss')
                        plt.xlabel('Epoch')
                        plt.show()
                    except:
                        pass
                else:
                    model.fit(**fit_params)  # Speichern Sie die Trainingshistorie


        except optuna.exceptions.TrialPruned as e:
            print(f"Trial pruned: {e} with params : {{'dropout_rate': {dropout_rate}, 'activation': '{activation}', 'optimizer': 'adam', 'n_layers': {n_layers}, 'n_units': {n_units}, 'batch_size': {batch_size}}}")
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

        if not tune_with_best:
            return model
    except Exception:
        print(traceback.format_exc())




def train_or_tune_model_v3(trial, X_train, Y_train, X_test, Y_test, tuning=True, verbose=0, best_params=None, show_progression=False):

    # Modellparameter
    # norm TODO
    if tuning:
        min_neurons = int(math.sqrt(X_train.shape[1]))
        max_neurons = X_train.shape[1]
        print(f"n_units = trial.suggest_int('n_units', {min_neurons}, {max_neurons}, log=False)")

        n_layers = trial.suggest_int('n_layers', 1, 3)
        n_units = trial.suggest_int('n_units', min_neurons, max_neurons, log=False)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        batch_size = trial.suggest_int('batch_size', 8, 1000, log=False)

    else:
        n_layers = best_params['n_layers']
        n_units = best_params['n_units']
        dropout_rate = best_params['dropout_rate']
        batch_size = best_params['batch_size']



    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self, tuning=False):
            super().__init__()
            self.tuning = tuning

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            current_lr = self.model.optimizer.lr.numpy()  # Zugriff auf die aktuelle Lernrate
            scale_value = 1
            if self.tuning:
                print(
                    f"Epoch: {epoch + 1:2d}, Loss: {logs['loss'] / scale_value:15.3f}, Val Loss: {logs.get('val_loss', 'N/A') / scale_value:15.3f}, Zeit: {epoch_time:6.2f} Sek., LR: {current_lr:.2e}")
            else:
                print(
                    f"Epoch: {epoch + 1:2d}, Loss: {logs['loss'] / scale_value:15.3f}, Zeit: {epoch_time:6.2f} Sek., LR: {current_lr:.2e}")



    callbacks = []
    if tuning:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=tuning_reduce_lr_factor,
                                                         min_delta=tuning_reduce_lr_min_delta,
                                                         patience=tuning_reduce_lr_patience,
                                                         min_lr=tuning_reduce_lr_min_lr,
                                                         cooldown=tuning_reduce_lr_cooldown)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=tuning_early_stopping_patience,
                                                          min_delta=tuning_early_stopping_min_delta,
                                                          restore_best_weights=True)

        pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
        callbacks.extend([early_stopping, reduce_lr, pruning_callback, CustomCallback(tuning=tuning)])

    else:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=building_reduce_lr_factor,
                                                         min_delta=building_reduce_lr_min_delta,
                                                         patience=building_reduce_lr_patience,
                                                         min_lr=building_reduce_lr_min_lr,
                                                         cooldown=building_reduce_lr_cooldown)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=building_early_stopping_patience,
                                                          min_delta=building_early_stopping_min_delta,
                                                          restore_best_weights=True)
        callbacks.extend([early_stopping, reduce_lr, CustomCallback(tuning=tuning)])





    # Modellaufbau
    model = Sequential()
    model.add(LSTM(n_units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=n_layers > 1))
    model.add(Dropout(dropout_rate))

    for i in range(1, n_layers):
        if i == n_layers - 1:  # letzter Layer
            model.add(LSTM(n_units, activation='relu', return_sequences=False))
        else:  # Zwischenlayer
            model.add(LSTM(n_units, activation='relu', return_sequences=True))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=tuning_adam_learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    # Callbacks für das Tuning
    # callbacks = [TFKerasPruningCallback(trial, 'val_loss')]
    # if tuning:
    #     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    #     callbacks.append(early_stopping)

    # Training
    model.fit(X_train, Y_train, epochs=999999, batch_size=batch_size, validation_data=(X_test, Y_test), callbacks=callbacks, verbose=verbose)

    # Evaluation
    loss = model.evaluate(X_test, Y_test, verbose=0)
    return loss


# Step 4
########################################################################################################################
def build_model_v3(vectorize=None, database_name=None, table_name=None, model_name=None, show_progression=False, load_preprocess_data=False, verbose=False):
    train_df = load_data(database_name=database_name, table_name=table_name)



    if verbose: print(f'df_to_dataset()...')
    train_ds = df_to_dataset(train_df)


    if verbose: print(f'preprocess_data_v3()...')
    all_inputs, all_features = preprocess_data_v3(train_df=train_df, train_ds=train_ds, verbose=verbose)

    study = load_study(database_name)
    # best_params = study.best_params

    valid_trials = [trial for trial in study.trials if trial.value is not None]
    sorted_trials = sorted(valid_trials, key=lambda trial: trial.value, reverse=False)
    best_params = sorted_trials[1].params
    print(f'best_params : {best_params}')


    if verbose: print(f'train_or_tune_model_v3()...')
    model = train_or_tune_model_v3(best_params=best_params, train_ds=train_ds, all_inputs=all_inputs, all_features=all_features, verbose=2, tuning=False, show_progression=show_progression)

    # Setze das Arbeitsverzeichnis auf das Verzeichnis, in dem sich das Skript befindet
    os.chdir(os.path.dirname(__file__))
    # Speichern des trainierten Modells

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    model_save_path = f"saved_models/nn_model_{model_name}"
    model.save(model_save_path)

    # with open(f'saved_models/vectorizer_{model_name}.pkl', 'wb') as f:
    #     pickle.dump(fitted_vectorizer_tfidf, f)



# Step 2
########################################################################################################################

def predict_v3(vectorize=None, model_name=None, brand=None, number_of_predicts=None, matrix_predict=False, mode="random",  price_quality_threshold=0, string_match_threshold=0.92, bike_data=None):

    if bike_data is None:
        bike_data = {}
    results_list = []


    print("Loading Model...")
    os.chdir(os.path.dirname(__file__))
    model, loaded_vectorizer_tfidf = load_trained_model_v3(vectorize=vectorize, model_name=model_name)
    print("Finished loading")


    if vectorize:
        modell_spalte = vectorized_column
        print("Loading BERT...")
        bert_preprocessor = hub.KerasLayer(r"nlp_models\bert\preprocessor")
        bert_encoder = hub.KerasLayer(r"nlp_models\bert\bert-en-uncased-l-2-h-128-a-2")
        print("Finished loading")


    else:
        modell_spalte = "modell_opt"
        bert_preprocessor = None
        bert_encoder = None


    if mode == "random":
        db_wkdm = load_data(vectorize=vectorize, predict=True, brand=brand)
        # print(f'db_wkdm_len:{len(db_wkdm)}')
        # db_wkdm_filtered = db_wkdm[-1000:].copy()
        # db_wkdm_filtered = db_wkdm[db_wkdm[modell_spalte].str.contains("ADV",)*(db_wkdm[baujahr_spalte] == 2007)]

        # for i in tqdm(range(number_of_predicts)):
        for index, row in tqdm(db_wkdm.iterrows(), total=(len(db_wkdm))):
            try:
                # bike_data = random_selection_from_db(marke=brand, db_wkdm=db_wkdm, price_quality_threshold=price_quality_threshold)
                bike_data = row
                # print(bike_data.to_dict())
                # exit()


                counterproposals_av = bike_data.pop("counterproposals_av")
                post_id = bike_data.pop("post_id")
                counterproposal_num = bike_data.pop("counterproposal_num")

                actual_price = bike_data[y_target_data]

                predicted_price, loss = predict_price_v3(vectorize=vectorize, data=bike_data, model=model, bert_preprocessor=bert_preprocessor, bert_encoder=bert_encoder, show_loss=True)

                # Summe der "price_quality"-Werte für ähnliche Modelle
                quality_baujahr, quality_marke = sum_values_based_on_string_match(vectorize=vectorize, df=db_wkdm, target_marke=brand, target_model=bike_data[modell_spalte], target_baujahr=bike_data[baujahr_spalte], threshold=string_match_threshold)

                diff_value = predicted_price - actual_price

                results_list.append({
                    "post_id": post_id,
                    "Marke": bike_data["marke_opt"],
                    "Modell": bike_data[modell_spalte],
                    "Modell_opt": bike_data[modell_spalte],
                    "Baujahr": bike_data[baujahr_spalte],
                    "Laufleistung": bike_data['laufleistung_approximately'],
                    "Zustand": bike_data['zustand_motorrad'],
                    "DB_€": actual_price,
                    "NN_€": predicted_price,
                    "gv_av": counterproposals_av,
                    "gv_num": counterproposal_num,
                    "DIFF_€": diff_value,
                    "Q_baujahr": quality_baujahr,
                    "Q_marke": quality_marke,
                })

            except:
                # print(f'ERROR @:{row}')
                pass

        # Convert the results list to a DataFrame
        df = pd.DataFrame(results_list)
        print(df)
        print("\n")

        try:
            df.to_excel(f"predicts_{model_name}.xlsx")
        except:
            pass

        # Print min, max, median and mean values for DIFF, Q and loss
        for col in ['DIFF_€', 'Q_baujahr', 'Q_marke']:
            print(f"{col} Min: {df[col].min():.2f}, {col} Max: {df[col].max():.2f}, {col} Median: {df[col].median():.2f}, {col} Mean: {df[col].mean():.2f}")

    if mode == "km":
        db_wkdm = load_data(vectorize=vectorize, predict=True)

        bike_data = random_selection_from_db(marke=brand, db_wkdm=db_wkdm, price_quality_threshold=price_quality_threshold)
        counterproposals_av = bike_data.pop("counterproposals_av")
        post_id = bike_data.pop("post_id")
        counterproposal_num = bike_data.pop("counterproposal_num")
        # actual_price = bike_data[y_target_data]

        # Summe der "price_quality"-Werte für ähnliche Modelle
        quality_baujahr, quality_marke = sum_values_based_on_string_match(vectorize=vectorize, df=db_wkdm, target_marke=brand, target_model=bike_data[vectorized_column], target_baujahr=bike_data[baujahr_spalte], threshold=string_match_threshold)

        results = []
        for km in tqdm(range(0, 5000, 100)):
            bike_data["laufleistung_bereinigt"] = km
            data = bike_data.copy()
            data["predicted_price"], data["loss"] = predict_price_v3(vectorize=vectorize, data=bike_data, model=model, loaded_vectorizer_tfidf=loaded_vectorizer_tfidf, show_loss=True)
            results.append(dict(data))

        df = pd.DataFrame(results)

        print(df)
        print("\n")
        print(f'Q_baujahr:{quality_baujahr:.2f}')
        print(f'Q_m:{quality_marke:.2f}')


    if mode == "lead":

        if not matrix_predict:
            db_wkdm = load_data(vectorize=vectorize, predict=False)

            for lead in bike_data:
                lead = pd.Series(lead)
                predicted_price, loss = predict_price_v3(vectorize=vectorize, data=lead, model=model, bert_preprocessor=bert_preprocessor, bert_encoder=bert_encoder, show_loss=True)
                quality_baujahr, quality_marke = sum_values_based_on_string_match(vectorize=vectorize, df=db_wkdm, target_marke=brand, target_model=lead[modell_spalte], target_baujahr=lead[baujahr_spalte], threshold=string_match_threshold)
                print(f"\n{lead[modell_spalte]}")
                print(f'Q_baujahr:{quality_baujahr:.2f}, Q_marke:{quality_marke:.2f},P:{predicted_price:.0f}, loss:{loss:.0f}')
        else:

            if vectorize:
                modell_column = "modell"
                hubraum_column = "hubraum_bereinigt"
                leistung_column = "leistung_bereinigt"

            else:
                modell_column = "modell_opt"
                hubraum_column = "hubraum_ermittelt"
                leistung_column = "leistung_ermittelt"

            # bike_data = bike_data[0]
            bike_data_orig = bike_data

            #TODO

            laufleistung_range = {'start': 0, 'end': 125000, 'step': 1000}
            baujahr_range = {'start': 2004, 'end': 2019, 'step': 1}
            zustand_range = {'start': 1, 'end': 5, 'step': 1}
            #
            # laufleistung_range = {'start': 0, 'end': 125000, 'step': 1000}
            # baujahr_range = {'start': 2004, 'end': 2004, 'step': 1}
            # zustand_range = {'start': 1, 'end': 1, 'step': 1}

            results_list = []

            laufleistung_range_len = ((laufleistung_range['end'] + laufleistung_range['step']) - laufleistung_range['start']) / laufleistung_range['step']

            baujahr_range_len = int(((baujahr_range['end'] + baujahr_range['step']) - baujahr_range['start']) / baujahr_range['step'])
            baujahr_range_len_counter = 1

            zustand_range_len = int(((zustand_range['end'] + zustand_range['step']) - zustand_range['start']) / zustand_range['step'])

            orig_model_name_input = ""

            if vectorize:
                bike_data = bike_data_orig.copy()

                try:
                    # del bike_data['date']
                    del bike_data['erstzulassung']
                    del bike_data['laufleistung']
                    del bike_data['laufleistung_approximately']
                except:
                    pass

                bike_data["marke_opt"] = bike_data.pop("marke")
                bike_data["abs_"] = bike_data.pop("abs")
                bike_data["hubraum_bereinigt"] = int(bike_data.pop("hubraum"))
                bike_data["leistung_bereinigt"] = int(bike_data.pop("leistung"))

                orig_model_name_input = bike_data['modell_opt']

                bike_data['modell_opt'] = cf.apply_opt_v3(bike_data, column='modell_opt')


            for baujahr in tqdm(range(baujahr_range['start'], baujahr_range['end'] + baujahr_range['step'], baujahr_range['step']), disable=True):

                if vectorize:
                    column = "leistung_bereinigt"
                else:
                    column = "leistung"

                if 2004 <= baujahr <= 2007:
                    bike_data[column] = 72
                if 2008 <= baujahr <= 2009:
                    bike_data[column] = 77
                if 2010 <= baujahr <= 2012:
                    bike_data[column] = 81
                if 2013 <= baujahr <= 2019:
                    bike_data[column] = 92

                if not vectorize:
                    bike_data_orig["baujahr_jahr"] = str(baujahr)
                    bike_data_orig["erstzulassung"] = str(baujahr)

                    # bike_data = identify_model_v3.identify_model(bike_data_orig)

                    del bike_data['date']
                    del bike_data['erstzulassung']
                    del bike_data['erstzulassung_bereinigt']
                    del bike_data['date_date']

                else:
                    bike_data["baujahr_auswahl"] = baujahr


                zustand_range_len_counter = 1
                for zustand in tqdm(range(zustand_range['start'], zustand_range['end'] + zustand_range['step'], zustand_range['step']), disable=True):

                    for laufleistung in tqdm(range(laufleistung_range['start'], laufleistung_range['end'] + laufleistung_range['step'], laufleistung_range['step']), total=laufleistung_range_len,
                            desc=f'Baujahr {baujahr_range_len_counter}/{baujahr_range_len}, Zustand {zustand_range_len_counter}/{zustand_range_len}, Laufleistung'):

                        bike_data["zustand_motorrad"] = zustand
                        bike_data["laufleistung_bereinigt"] = laufleistung
                        bike_data["alter_jahre"] = 2023 - baujahr

                        predicted_price, loss = predict_price_v3(vectorize=vectorize, data=bike_data, model=model,
                                                                 bert_preprocessor=bert_preprocessor,
                                                                 bert_encoder=bert_encoder, show_loss=True)

                        # print(f'bike_data:{bike_data}')

                        if vectorize:
                            results_dict = {'marke_opt': bike_data["marke_opt"],
                                            'modell': orig_model_name_input,
                                            'modell_bereinigt': bike_data[modell_column],
                                            'hubraum_bereinigt': bike_data[hubraum_column],
                                            'leistung_bereinigt': bike_data[leistung_column],
                                            'baujahr_auswahl': baujahr,
                                            'zustand_motorrad': zustand,
                                            'laufleistung_bereinigt': laufleistung,
                                            'alter_jahre': bike_data["alter_jahre"],
                                            'predicted_price': predicted_price,
                                            }
                        else:

                            results_dict = {
                                            'marke': bike_data["marke"],
                                            'marke_ermittelt': bike_data["marke_ermittelt"],
                                            'marke_opt': bike_data["marke_opt"],
                                            'marke_ermittelt_qualität': bike_data["marke_ermittelt_qualität"],
                                            'modell': bike_data["modell"],
                                            'modell_ermittelt': bike_data["modell_ermittelt"],
                                            'modell_opt': bike_data["modell_opt"],
                                            'modell_ermittelt_qualität': bike_data["modell_ermittelt_qualität"],
                                            'predicted_price': predicted_price,
                                            'abw_hubraum': bike_data["abw_hubraum"],
                                            'abw_baujahr': bike_data["abw_baujahr"],
                                            'score': bike_data["score"],
                                            'rating': '',
                                            'leistung': bike_data["leistung"],
                                            'hubraum': bike_data["hubraum"],
                                            'laufleistung': bike_data["laufleistung_bereinigt"],
                                            'laufleistung_approximately': bike_data["laufleistung_bereinigt"],
                                            'baujahr_jahr': bike_data["baujahr_jahr"],
                                            'baujahr_auswahl': bike_data["baujahr_auswahl"],
                                            'baujahr_ermittelt': bike_data["baujahr_ermittelt"],
                                            'alter_jahre': bike_data["alter_jahre"],
                                            'abs': bike_data["abs"],
                                            'zustand_motorrad': bike_data["zustand_motorrad"],
                                            'timestamp': bike_data["timestamp"],
                                            'laufleistung_bereinigt': bike_data["laufleistung_bereinigt"],
                                            'baujahr_jahr_bereinigt': bike_data["baujahr_jahr_bereinigt"],
                                            'leistung_bereinigt': bike_data["leistung_bereinigt"],
                                            'leistung_ermittelt': bike_data["leistung_ermittelt"],
                                            'hubraum_bereinigt': bike_data["hubraum_bereinigt"],
                                            'hubraum_ermittelt': bike_data["hubraum_ermittelt"],
                                            'abw_leistung': bike_data["abw_leistung"],
                                            'datenqualitaet': bike_data["datenqualitaet"],
                                            'abw_hubraum_proz': bike_data["abw_hubraum_proz"]
                                            }



                        results_list.append(results_dict)
                    zustand_range_len_counter += 1
                baujahr_range_len_counter += 1

            results_df = pd.DataFrame(results_list)

            # print(f'results_df:{results_df}')

            if vectorize:
                text_vec = "vec"

            else:
                text_vec = "cat"

            zeitstempel = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            if vectorize:
                results_df.to_excel(rf"test_results\test_results_{text_vec}_{orig_model_name_input}_{model_name}_{zeitstempel}.xlsx")
            else:
                results_df.to_excel(rf"test_results\test_results_{text_vec}_{bike_data['modell']}_{model_name}_{zeitstempel}.xlsx")

            print(results_df)

def predict_price_v3(vectorize=None, data=None, model=None, bert_preprocessor=None, bert_encoder=None, show_loss=None):

    # Umwandeln von pd.serien zu dataframe
    df = pd.DataFrame([data])
    # print(df.to_dict(orient='records'))
    # exit()
    if vectorize:
        df, bert_feature_names_df = transform_data_with_bert(data_df=df, bert_preprocessor=bert_preprocessor, bert_encoder=bert_encoder, verbose=False)

    # Erstellen des Datasets
    ds = df_to_dataset_predict(df)

    # Vorhersage mit dem Modell
    predicted_price = model.predict(ds, verbose=0)

    # Wenn ein tatsächlicher Wert gegeben ist, berechnen Sie den Loss
    if show_loss is not None:
        loss_fn = tf.keras.losses.MeanSquaredError()
        loss = loss_fn(show_loss, predicted_price).numpy()
        return predicted_price[0][0], loss
    else:
        return predicted_price[0][0]


def load_trained_model_v3(vectorize=None, model_name=None):
    # Setze das Arbeitsverzeichnis auf das Verzeichnis, in dem sich das Skript befindet
    os.chdir(os.path.dirname(__file__))

    # Pfad zum gespeicherten Modell
    model_path = f"saved_models/nn_model_{model_name}"
    model = tf.keras.models.load_model(model_path)

    if vectorize:
        # Laden des Modells mit Keras
        # with open(f'saved_models/vectorizer_{model_name}.pkl', 'rb') as f:
        #     loaded_vectorizer = pickle.load(f)
        loaded_vectorizer = None
    else:
        loaded_vectorizer = None

    return model, loaded_vectorizer
########################################################################################################################



# SETTINGSDATA
#  =====================================================================================================================

y_target_data = "target_value"
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
building_adam_learning_rate = 0.0001
# !!!!!! Die Kurve ist maßgeblich, ob die lr richtig eingestellt ist !!!
# SETTINGSBUILD
#  =====================================================================================================================


# n_layers_range_from = 4
# n_layers_range_to = 4

if __name__ == "__main__":
    start_total = time.time()
    database_name = "trading_bot"

    # database_name_v3 = f"db_test{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # database_name_v3 = "bot_v1"  # nlayers 1-6
    # database_name_v3 = "bot_v2"  # nlayers 1-3
    database_name_v3 = "bot_v3"  # nlayers 1-1

    table_name = ""
    workers = 32
    train_model_v3(n_trials=workers*10, n_jobs=workers, database_name=database_name, database_name_optuna=database_name_v3, table_name=table_name, show_progression=False, tune_with_best=False, load_preprocess_data=False, verbose=True)
    # show_study_params(database_name=database_name_v3, n_last_trials=5)
    # show_study(database_name=database_name_v3)
    build_model_v3(database_name=database_name_v3, table_name=table_name, model_name=database_name_v3, show_progression=True)
    
    # predict_v3(vectorize=vectorize, model_name=database_name_v3, brand="ktm", mode="random", number_of_predicts=100, matrix_predict=True, price_quality_threshold=0, string_match_threshold=0.92)
    # mode[km , random, lead]


    end_total = time.time()
    print(f'Total: {round(end_total - start_total, ndigits=2)} Seconds')



