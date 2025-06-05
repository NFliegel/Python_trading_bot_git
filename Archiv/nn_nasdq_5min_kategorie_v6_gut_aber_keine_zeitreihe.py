import numpy as np
import pandas as pd
import pandas_ta as ta
import schedule
import threading

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sqlalchemy
from sqlalchemy.exc import SQLAlchemyError
import optuna
# import optuna.visualization as vis
from optuna.integration.tfkeras import TFKerasPruningCallback
import joblib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import random
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # GPU deaktivieren
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mysql.connector
from mysql.connector import Error
import functools
import json

from sqlalchemy import create_engine, text

from difflib import SequenceMatcher
from tqdm import tqdm
import traceback
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

import time
# import datetime
import math
import re
import pickle
import faulthandler
import yfinance as yf

from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA

import tensorflow as tf
import keras
import tensorflow_hub as hub
import tensorflow_text as text  # Wichtig für BERT Preprocessing
from keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from tensorflow.keras import backend as K



seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
# tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

os.environ['TF_NUM_INTEROP_THREADS'] = '10'  # Anzahl der Threads für parallele Operationen
os.environ['TF_NUM_INTRAOP_THREADS'] = '10'  # Anzahl der Threads für interne Operationen


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
tuning_early_stopping_patience = 40

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
building_reduce_lr_patience = 30
building_reduce_lr_min_delta = 0  # --> muss eine minimale Verbesserung gegeben haben
building_reduce_lr_min_lr = 1e-9
building_reduce_lr_cooldown = 0

#  EarlyStopping
building_early_stopping_patience = 60  # es muss sich mindestens so viel verbessert haben um als Verbesserung zu gelten
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




def train_model_v3(tuning=None, n_trials=None, n_jobs=None, time_series_sequence_length=None, database_name_optuna=None, show_progression=False, verbose=False, db_wkdm=None):

    if verbose: print(f'load_data()')
    # db_wkdm = load_data(database_name=database_name, table_name=table_name)
    # db_wkdm = db_wkdm[:1000]

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


    if tuning:
        if verbose: print(f'split_data_v3')

        if verbose: print(f'Transforming data to datasets...')
          # This should be dynamically determined or passed as a parameter based on your data preparation
        # train_ds = prepare_data_for_nn(train_df, feature_cols=train_df.columns[:-1], target_col="Trend", sequence_length=sequence_length, scaler=scaler, label_encoder=label_encoder)
        # test_ds  = prepare_data_for_nn(test_df, feature_cols=test_df.columns[:-1], target_col="Trend", sequence_length=sequence_length, scaler=scaler, label_encoder=label_encoder)

        print(train_df.head())

        train_ds, _, _ = prepare_data_for_nn(train_df, feature_cols=train_df.columns[:-1], target_col="Trend", sequence_length=time_series_sequence_length, scalers=scalers, onehot_encoder=onehot_encoder)
        test_ds, _, _ = prepare_data_for_nn(test_df, feature_cols=test_df.columns[:-1], target_col="Trend", sequence_length=time_series_sequence_length, scalers=scalers, onehot_encoder=onehot_encoder)

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
        objective_with_data = functools.partial(train_or_tune_model_v3, train_ds=train_ds, test_ds=test_ds, tuning=tuning, n_jobs=n_jobs, verbose=verbose, show_progression=show_progression)

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

        train_ds, _, _ = prepare_data_for_nn(train_df, feature_cols=train_df.columns[:-1], target_col="Trend", sequence_length=time_series_sequence_length, scalers=scalers, onehot_encoder=onehot_encoder)

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
        model_name = f"nn_model_{database_name_optuna}"


        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        model.save(f'{model_save_path}/{model_name}.keras')

        scalers_save_path = f"{model_save_path}/scalers.pkl"
        onehot_encoder_save_path = f"{model_save_path}/onehot_encoder.pkl"

        joblib.dump(scalers, scalers_save_path)
        joblib.dump(onehot_encoder, onehot_encoder_save_path)


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
    # generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_scaled, y_encoded, length=sequence_length, batch_size=1)
    # generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_scaled, y_encoded, length=sequence_length, batch_size=128)
    # generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_scaled, y_encoded, length=sequence_length, batch_size=32)

    # print(f'generator_len:{len(generator)}')
    #
    # for x, _ in generator:
    #     print(x)
    #
    #     if not x.any():
    #         exit()
    #
    # X_seq = np.array([x[0] for x, _ in generator])  # Ensure shape is (sequence_length, num_features)
    # # X_seq = np.array([x for x, _ in tqdm(generator)])
    # # X_seq = X_seq.reshape((len(generator), sequence_length, len(feature_cols)))
    #
    # y_seq = np.array([y for _, y in tqdm(generator)])
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



def train_or_tune_model_v3(trial=None, train_ds=None, test_ds=None, all_features=None, best_params=None, tuning=True, n_jobs=1, verbose=0, show_progression=False, num_timesteps=None, num_features=None):

    try:
        # Hyperparameter space
        if tuning:

                #TODO
                # min_neurons = int(math.sqrt(all_features.shape[1]))
                # max_neurons = all_features.shape[1]
                # min_neurons = int(num_features / 2)  # Basierend auf der Anzahl der Features
                # max_neurons = num_features * 2

                # print(f"n_units = trial.suggest_int('n_units', {min_neurons}, {max_neurons}, log=False)")

                # activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
                # activation = trial.suggest_categorical('activation', ['relu'])
                activation = trial.suggest_categorical('activation', ['tanh'])

                # n_layers = trial.suggest_int('n_layers', 1, 3)
                # n_units = trial.suggest_int('n_units', min_neurons, max_neurons, log=False)
                # n_units = trial.suggest_int('n_units', 1, 300, log=False)
                n_units = trial.suggest_int('n_units', 100, 100, log=False)

                # n_units = trial.suggest_int('n_units', min_neurons, 100, log=False)
                # n_units = trial.suggest_int('n_units', 1, 12, log=False)


                dropout_rate = trial.suggest_float('dropout_rate', 0.4, 0.7, log=False)
                batch_size = trial.suggest_int('batch_size', 128, 2000, log=False)
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
        print(f'activation:{activation}, n_units:{n_units},dropout_rate:{dropout_rate},batch_size:{batch_size}')


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


        class MinLossStoppingCallback(tf.keras.callbacks.Callback):
            def __init__(self, min_loss):
                super(MinLossStoppingCallback, self).__init__()
                self.min_loss = min_loss

            def on_epoch_end(self, epoch, logs=None):
                current_loss = logs.get("loss")
                if current_loss is not None and current_loss < self.min_loss:
                    print(f"\nEpoch {epoch + 1}: loss is below the minimum threshold of {self.min_loss}, stopping training.")
                    self.model.stop_training = True

        min_loss_threshold = 0.01

        class MaxAccuracyStoppingCallback(tf.keras.callbacks.Callback):
            def __init__(self, max_accuracy):
                super(MaxAccuracyStoppingCallback, self).__init__()
                self.max_accuracy = max_accuracy

            def on_epoch_end(self, epoch, logs=None):
                current_accuracy = logs.get("f1_score")
                if current_accuracy is not None and current_accuracy > self.max_accuracy:
                    print(f"\nEpoch {epoch + 1}: loss is below the minimum threshold of {self.max_accuracy}, stopping training.")
                    self.model.stop_training = True

        max_f1_score_threshold = 0.99



        def f1_score(y_true, y_pred):
            def recall(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + K.epsilon())
                return recall

            def precision(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            precision_value = precision(y_true, y_pred)
            recall_value = recall(y_true, y_pred)
            return 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))


        if tuning:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score', factor=tuning_reduce_lr_factor, min_delta=tuning_reduce_lr_min_delta, patience=tuning_reduce_lr_patience, min_lr=tuning_reduce_lr_min_lr, cooldown=tuning_reduce_lr_cooldown)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=tuning_early_stopping_patience, min_delta=tuning_early_stopping_min_delta, restore_best_weights=True, mode='max')
            pruning_callback = optuna.integration.tfkeras.TFKerasPruningCallback(trial, 'val_f1_score')

            callbacks.extend([early_stopping, reduce_lr, pruning_callback, CustomCallback(tuning=tuning),
                              # MinLossStoppingCallback(min_loss_threshold),
                              # MaxAccuracyStoppingCallback(max_f1_score_threshold)
                              ])

        else:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='f1_score', factor=building_reduce_lr_factor, min_delta=building_reduce_lr_min_delta, patience=building_reduce_lr_patience, min_lr=building_reduce_lr_min_lr, cooldown=building_reduce_lr_cooldown)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='f1_score', patience=building_early_stopping_patience, min_delta=building_early_stopping_min_delta, restore_best_weights=True, mode='max')
            callbacks.extend([early_stopping, reduce_lr, CustomCallback(tuning=tuning),
                              # MinLossStoppingCallback(min_loss_threshold),
                              MaxAccuracyStoppingCallback(max_f1_score_threshold)
                             ])




        input_shape = (train_ds[0].shape[1], train_ds[0].shape[2])
        # input_shape = (train_ds.shape[1], train_ds.shape[2])

        print(f"Input shape: {input_shape}")
        print(f"Output shape: {train_ds[1].shape[1]}")

        # # Model building
        # model = tf.keras.Sequential()
        # # model.add(tf.keras.layers.LSTM(n_units, activation=activation, input_shape=input_shape, return_sequences=True))
        # model.add(tf.keras.layers.LSTM(n_units, activation=activation, input_shape=input_shape, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        # model.add(tf.keras.layers.Dropout(dropout_rate))
        # # model.add(tf.keras.layers.LSTM(n_units, activation=activation))
        # model.add(tf.keras.layers.LSTM(n_units, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        # model.add(tf.keras.layers.Dropout(dropout_rate))
        # model.add(tf.keras.layers.Dense(train_ds[1].shape[1], activation='softmax'))  # Assuming 3 classes for one-hot encoding
        # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[f1_score, 'accuracy'])


        #TODO

        model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, activation=activation, input_shape=input_shape, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))))
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, activation=activation, input_shape=input_shape, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, activation=activation, input_shape=input_shape, return_sequences=True)))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.BatchNormalization())

        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, activation=activation, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))))
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, activation=activation)))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dense(train_ds[1].shape[1], activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[f1_score, 'accuracy'])




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
            # 'workers': int(32/n_jobs),
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
                    loss  = model.evaluate(test_dataset, verbose=verbose)
                    return loss[0]  # Rückgabe nur des Verlustes
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




def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))



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


def calculate_slope(series, window):
    slopes = [0] * len(series)
    for i in range(window, len(series)):
        y = series[i-window:i]
        x = np.arange(window)
        if len(np.unique(y)) == 1:  # Avoids issue if all y values are the same
            slope = 0
        else:
            slope = np.polyfit(x, y, 1)[0]
        slopes[i] = slope
    return slopes


def calculate_future_slope(series, window):
    slopes = [np.nan] * len(series)
    for i in range(len(series) - window):
        y = series[i:i+window]
        x = np.arange(window)
        if len(np.unique(y)) == 1:  # Vermeidet Probleme bei gleichen y-Werten
            slope = 0
        else:
            slope = np.polyfit(x, y, 1)[0]
        slopes[i] = slope
    return slopes



def save_to_db(dataframe, to_table, db):

    # Erstellt eine Verbindung zur Datenbank
    engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://root:@localhost/{db}', echo=False)
    conn = engine.connect()
    dataframe.to_sql(con=conn, name=f'{to_table}', if_exists='replace', index=False)
    conn.close()
    print(f"Daten erfolgreich in Tabelle '{to_table}' gespeichert.")



def plot_stock_prices(df, test=False, x_interval_min=60, y_interval_dollar=25, additional_lines=None, secondary_y_scale=1.0, time_stay=999999):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["Datetime"], df["Close"], label='Aktienkurs', color='blue')

    trend = "Trend"

    # Erstelle Dummy-Marker für die Legende
    up_marker, = ax1.plot([], [], marker='^', color='green', markersize=10, linestyle='None', label='Up')
    down_marker, = ax1.plot([], [], marker='v', color='red', markersize=10, linestyle='None', label='Down')
    stable_marker, = ax1.plot([], [], marker='o', color='grey', markersize=5, linestyle='None', label='Stable')

    for i in range(len(df)):
        if df[trend].iloc[i] == "Up":
            ax1.plot(df["Datetime"].iloc[i], df["Close"].iloc[i], marker='^', color='green', markersize=8)
        elif df[trend].iloc[i] == "Down":
            ax1.plot(df["Datetime"].iloc[i], df["Close"].iloc[i], marker='v', color='red', markersize=8)
        else:
            ax1.plot(df["Datetime"].iloc[i], df["Close"].iloc[i], marker='o', color='grey', markersize=2)

    # Zusätzliche Liniendiagramme hinzufügen
    if additional_lines:
        ax2 = ax1.twinx()
        ax2.set_frame_on(False)  # Versteckt den Rahmen der zweiten Achse
        additional_lines_handles = []
        for column, multiplier in additional_lines:
            if column in df.columns:
                line, = ax2.plot(df["Datetime"], df[column] * multiplier, alpha=0.7, label=f'{column} (x{multiplier})')
                additional_lines_handles.append(line)
        # Skalieren der zweiten Y-Achse
        ax2.set_ylim(ax2.get_ylim()[0] * secondary_y_scale, ax2.get_ylim()[1] * secondary_y_scale)

    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=x_interval_min))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(y_interval_dollar))

    ax1.set_title('Kurs, Trendstärke, Signale')
    ax1.set_xlabel('Datum und Zeit')
    ax1.set_ylabel('Kurs')

    # Legende für beide Achsen erstellen
    handles, labels = ax1.get_legend_handles_labels()
    if additional_lines:
        handles += additional_lines_handles
    handles += [up_marker, down_marker, stable_marker]
    labels = [h.get_label() for h in handles]
    unique_handles_labels = dict(zip(labels, handles))
    ax1.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc='upper left')

    ax1.grid(True)
    fig.autofmt_xdate()  # Automatische Formatierung der x-Achse
    # plt.show()

    # if time_stay:
    #     time.sleep(time_stay)
    #     plt.close(fig)

    if not test:
        def close_plot():
            time.sleep(time_stay)
            plt.close(fig)

        close_thread = threading.Thread(target=close_plot)
        close_thread.start()

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
        if scalers.get(col) is None:
            scalers[col] = StandardScaler()
            features_scaled[:, i] = scalers[col].fit_transform(df[col].values.reshape(-1, 1)).flatten()
        else:
            features_scaled[:, i] = scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()

    X_scaled = features_scaled

    # Encode target values
    y_encoded = onehot_encoder.transform(y.reshape(-1, 1))

    # Erstellung von Zeitreihen-Daten
    batch_size = 32
    generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_scaled, y_encoded, length=sequence_length, batch_size=batch_size)

    # Überprüfen, ob der Generator Daten produziert
    num_samples = len(generator)
    print(f"Anzahl der Samples im Generator: {num_samples}")
    if num_samples == 0:
        raise ValueError("Der TimeseriesGenerator ist leer. Überprüfen Sie die sequence_length und die Größe Ihrer Daten.")

    # Initialisieren von Listen für Sequenzen und Ziele
    X_seq = []
    y_seq = []

    # Extrahieren von Sequenzen und Zielen mit Fortschrittsbalken
    for i in tqdm(range(num_samples), desc="Extrahieren von Sequenzen und Zielen"):
        x, y = generator[i]
        X_seq.append(x)
        y_seq.append(y)

    # Umwandeln der Listen in Numpy-Arrays
    X_seq = np.vstack(X_seq)
    y_seq = np.vstack(y_seq)

    # Sicherstellen, dass die Formen korrekt sind
    print(f"X_seq shape: {X_seq.shape}")  # Sollte (num_samples, sequence_length, num_features) sein
    print(f"y_seq shape: {y_seq.shape}")  # Sollte (num_samples, num_classes) sein

    # Vorhersagen treffen
    y_pred_encoded = model.predict(X_seq)

    # Rücktransformation der One-Hot-Encoded-Vorhersagen
    y_pred = onehot_encoder.inverse_transform(y_pred_encoded)

    # Die Vorhersagen in das ursprüngliche DataFrame integrieren
    prediction_col = target_col + "_predicted"
    df[prediction_col] = np.nan

    # Stellen Sie sicher, dass die Vorhersagenspalte den richtigen Datentyp hat
    df[prediction_col] = df[prediction_col].astype(object)
    df[prediction_col].iloc[sequence_length:sequence_length+len(y_pred)] = y_pred.flatten()

    return df


def predict_and_integrate_live(df, database_name_optuna, time_series_sequence_length=12):

    model_save_path = f"saved_models/nn_model_{database_name_optuna}"
    model_name = f"nn_model_{database_name_optuna}"
    model_path = f'{model_save_path}/{model_name}.keras'
    scalers_path = f"{model_save_path}/scalers.pkl"
    onehot_encoder_path = f"{model_save_path}/onehot_encoder.pkl"

    # Laden des Modells und der Transformationsobjekte
    # model = tf.keras.models.load_model(model_path)
    # model = tf.keras.models.load_model(model_path, custom_objects={'f1_score': f1_score})
    model = tf.keras.models.load_model(model_path, custom_objects={'f1_score': f1_score})

    scalers = joblib.load(scalers_path)
    onehot_encoder = joblib.load(onehot_encoder_path)

    feature_cols = df.columns

    df = df.dropna()

    # Vorbereitung der Daten
    X = df[feature_cols].values

    # Skalieren der Features
    features_scaled = np.zeros(X.shape)
    for i, col in enumerate(feature_cols):
        if scalers.get(col) is None:
            scalers[col] = StandardScaler()
            features_scaled[:, i] = scalers[col].fit_transform(df[col].values.reshape(-1, 1)).flatten()
        else:
            features_scaled[:, i] = scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()

    X_scaled = features_scaled

    # Erstellung von Zeitreihen-Daten
    generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_scaled, X_scaled, length=time_series_sequence_length, batch_size=32)

    # Überprüfen, ob der Generator Daten produziert
    num_samples = len(generator)
    print(f"Anzahl der Samples im Generator: {num_samples}")
    if num_samples == 0:
        raise ValueError("Der TimeseriesGenerator ist leer. Überprüfen Sie die sequence_length und die Größe Ihrer Daten.")

    # Initialisieren von Listen für Sequenzen
    X_seq = []

    # Extrahieren von Sequenzen mit Fortschrittsbalken
    for i in tqdm(range(num_samples), desc="Extrahieren von Sequenzen"):
        x, _ = generator[i]
        X_seq.append(x)

    # Umwandeln der Liste in ein Numpy-Array
    X_seq = np.vstack(X_seq)

    # Sicherstellen, dass die Form korrekt ist
    print(f"X_seq shape: {X_seq.shape}")  # Sollte (num_samples, sequence_length, num_features) sein

    # Vorhersagen treffen
    y_pred_encoded = model.predict(X_seq)

    # Rücktransformation der One-Hot-Encoded-Vorhersagen
    y_pred = onehot_encoder.inverse_transform(y_pred_encoded)

    # Die Vorhersagen in das ursprüngliche DataFrame integrieren
    prediction_col = "Trend"
    df[prediction_col] = np.nan

    # Stellen Sie sicher, dass die Vorhersagenspalte den richtigen Datentyp hat
    df[prediction_col] = df[prediction_col].astype(object)
    df[prediction_col].iloc[time_series_sequence_length:time_series_sequence_length+len(y_pred)] = y_pred.flatten()

    return df



def train_and_save_model_random_forest(df, target_column, database_name_optuna):

    model_save_path = f"saved_models/nn_model_{database_name_optuna}"
    model_name = f"rf_model_{database_name_optuna}"
    model_path = f'{model_save_path}/{model_name}'

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Features und Zielspalte trennen
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Random Forest Modell trainieren
    rf = RandomForestClassifier(n_estimators=200, random_state=42, verbose=1)
    rf.fit(X_train, y_train)

    # Modell speichern
    joblib.dump(rf, model_path)
    print(f"Modell gespeichert als {model_name}")


def load_model_and_predict_random_forest(prediction_data, database_name_optuna):

    model_save_path = f"saved_models/nn_model_{database_name_optuna}"
    model_name = f"rf_model_{database_name_optuna}"
    model_path = f'{model_save_path}/{model_name}'

    # Modell laden
    rf = joblib.load(model_path)
    print(f"Modell geladen aus {model_name}")

    # Vorhersagen für die übergebenen Daten machen
    predictions = rf.predict(prediction_data)

    # Vorhersagen als neue Spalte "Trend" zum DataFrame hinzufügen
    prediction_data['Trend'] = predictions

    return prediction_data



def query_database(db, table, columns=None, conditions=None, engine_kwargs={}, query_kwargs={}):
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
    query = f"SELECT {columns_str} FROM {table}"
    if conditions:
        query += f" WHERE {conditions}"

    # Ausführen der Abfrage und Schließen der Verbindung
    with engine.connect() as conn:
        dataframe = pd.read_sql(query, con=conn, **query_kwargs)

    return dataframe


def load_txt_to_mysql(filename, table_name, database_name):

    script_path = os.path.dirname(os.path.abspath(__file__))

    # Vollständiger Pfad zur Datei
    file_path = os.path.join(script_path, filename)

    """
    Lädt eine TXT-Datei in eine MySQL-Datenbank.

    :param filename: Der Pfad zur TXT-Datei.
    :param table_name: Der Name der Zieltabelle in der MySQL-Datenbank.
    :param database_name: Der Name der MySQL-Datenbank.
    """
    # Lese die TXT-Datei ein. Achte darauf, das korrekte Trennzeichen anzugeben.
    df = pd.read_csv(fr'{file_path}', sep=",")

    # Speichere den DataFrame in der Datenbank
    save_to_db(df, table_name, database_name)





def get_stock_data(ticker, start=None, end=None, interval='5m', prepost=False, actions=True, auto_adjust=True, back_adjust=False, proxy=None, rounding=False, progress=True, hours=8):
    """
    Fetch stock data in specified intervals and return as a DataFrame.

    Parameters:
    - ticker (str): The ticker symbol of the stock.
    - start (str, optional): The start date for fetching data (e.g., '2020-01-01').
    - end (str, optional): The end date for fetching data (e.g., '2021-01-01').
    - interval (str, optional): The interval for fetching data ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo').
    - prepost (bool, optional): Whether to include pre/post market hours data. Default is False.
    - actions (bool, optional): Whether to include corporate actions (dividends, splits). Default is True.
    - auto_adjust (bool, optional): Whether to auto-adjust prices. Default is True.
    - back_adjust (bool, optional): Whether to back-adjust prices. Default is False.
    - proxy (str, optional): Proxy URL scheme to use when downloading data.
    - rounding (bool, optional): Whether to round prices to 2 decimal places. Default is False.
    - progress (bool, optional): Whether to show progress bar during download. Default is True.

    Returns:
    - pd.DataFrame: DataFrame containing stock data with specified interval.
    """
    # Create the ticker object
    stock = yf.Ticker(ticker)

    # Calculate the start date as the current date and time minus the specified number of hours
    # start = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d')
    # Set end date to the current date plus one day, only using the date part
    # end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    # Fetch the data
    data = stock.history(
        start=start,
        end=end,
        interval=interval,
        prepost=prepost,
        actions=actions,
        auto_adjust=auto_adjust,
        back_adjust=back_adjust,
        proxy=proxy,
        rounding=rounding,
        # progress=progress
    )
    try:
        data.index = data.index.tz_convert('Europe/Berlin')
    except:
        pass

    try:
        data.index = data.index.tz_localize(None)
    except:
        pass
    # Remove the timezone information from the Datetime index

    # Reset index to convert Datetime index to a column
    data.reset_index(inplace=True)

    return data


def generate_stock_pattern():
    open_Close = 500
    noise_duration = 60  # 60 Minuten
    noise_strength = 1  # Stärke des Rauschens
    pattern_prices = [500 + x for x in [10, 20, 15, 30, 10, 20, 5]]
    trend_changes = [-10, -20, -15, -30, -10, -25, -5]  # Abwärtstrend nach einem Head and Shoulders Muster

    num_cycles = 10  # Anzahl der Wiederholungen
    start_date = "03/28/2022"  # Startdatum

    df = generate_stock_prices(open_Close, noise_duration, noise_strength, pattern_prices, trend_changes, num_cycles, start_date)



# def calculate_averages(df, history_steps, future_steps, threshhold):
#
#     #TODO
#     # ####
#     # ####
#     # ####
#     # ####
#
#     # Timetable
#     if "Date" in list(df.columns) and "Time" in list(df.columns):
#         df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
#
#     if "Up" in list(df.columns) and "Down" in list(df.columns):
#         df['Volume'] = df['Up'] + df['Down']
#
#
#     # History Close Diff
#     df["Close_Diff"] = df["Close"].diff().fillna(0)
#
#
#
#     # History Mouving Average
#     df[f"Close_Diff_MA12"] = df["Close_Diff"].rolling(window=12).mean()
#     df[f"Close_Diff_MA24"] = df["Close_Diff"].rolling(window=24).mean()
#     df[f"Close_Diff_MA36"] = df["Close_Diff"].rolling(window=36).mean()
#     df[f"Close_Diff_MA48"] = df["Close_Diff"].rolling(window=48).mean()
#     df[f"Close_Diff_MA100"] = df["Close_Diff"].rolling(window=100).mean()
#     df[f"Close_Diff_MA200"] = df["Close_Diff"].rolling(window=200).mean()
#
#     # Future Mouving Average
#     df[f"Close_FMA"] = df["Close"].shift(-future_steps).rolling(window=future_steps).mean()
#     # df["Close_FMA_minus_Close"] = df["Close_FMA"] - df["Close"]
#     # df[f"Close_Diff_FMA"] = df["Close_Diff"].shift(-future_steps).rolling(window=future_steps).mean()
#
#     # df[f"Close_Diff_FMA12"] = df["Close_Diff"].shift(-12).rolling(window=12).mean()
#     # df[f"Close_Diff_FMA24"] = df["Close_Diff"].shift(-24).rolling(window=24).mean()
#     # df[f"Close_Diff_FMA36"] = df["Close_Diff"].shift(-36).rolling(window=36).mean()
#     # df[f"Close_Diff_FMA48"] = df["Close_Diff"].shift(-48).rolling(window=48).mean()
#     # df[f"Close_Diff_FMA100"] = df["Close_Diff"].shift(-100).rolling(window=100).mean()
#     # df[f"Close_Diff_FMA200"] = df["Close_Diff"].shift(-200).rolling(window=200).mean()
#
#
#
#     # EMA (Exponential Moving Average)
#     df['Close_Diff_EMA12'] = ta.ema(df['Close_Diff'], length=12)
#     df['Close_Diff_EMA24'] = ta.ema(df['Close_Diff'], length=24)
#     df['Close_Diff_EMA36'] = ta.ema(df['Close_Diff'], length=36)
#     df['Close_Diff_EMA48'] = ta.ema(df['Close_Diff'], length=48)
#     df['Close_Diff_EMA100'] = ta.ema(df['Close_Diff'], length=100)
#     df['Close_Diff_EMA200'] = ta.ema(df['Close_Diff'], length=200)
#
#
#     # Future Exponential Mouving Average
#     # df[f"Close_FEMA"] = df["Close"].shift(-future_steps).ewm(span=future_steps, adjust=False).mean()
#     # df["Close_FEMA_minus_Close"] = df["Close_FEMA"] - df["Close"]
#
#     # df['Close_Diff_FEMA12'] = df['Close_Diff'].shift(-12).ewm(span=12, adjust=False).mean()
#     # df['Close_Diff_FEMA24'] = df['Close_Diff'].shift(-24).ewm(span=24, adjust=False).mean()
#     # df['Close_Diff_FEMA36'] = df['Close_Diff'].shift(-36).ewm(span=36, adjust=False).mean()
#     # df['Close_Diff_FEMA48'] = df['Close_Diff'].shift(-48).ewm(span=48, adjust=False).mean()
#     # df['Close_Diff_FEMA100'] = df['Close_Diff'].shift(-100).ewm(span=100, adjust=False).mean()
#     # df['Close_Diff_FEMA200'] = df['Close_Diff'].shift(-200).ewm(span=200, adjust=False).mean()
#
#
#
#     df['RSI12'] = ta.rsi(df['Close'], length=12)
#     df['RSI24'] = ta.rsi(df['Close'], length=24)
#     df['RSI36'] = ta.rsi(df['Close'], length=36)
#     df['RSI48'] = ta.rsi(df['Close'], length=48)
#     df['RSI100'] = ta.rsi(df['Close'], length=100)
#     df['RSI200'] = ta.rsi(df['Close'], length=200)
#
#
#
#     # SMA (Simple Moving Average)
#     df['SMA12'] = ta.sma(df['Close'], length=12)
#     df['SMA24'] = ta.sma(df['Close'], length=24)
#     df['SMA36'] = ta.sma(df['Close'], length=36)
#     df['SMA48'] = ta.sma(df['Close'], length=48)
#     df['SMA100'] = ta.sma(df['Close'], length=100)
#     df['SMA200'] = ta.sma(df['Close'], length=200)
#
#
#
#     df['EMA12'] = ta.ema(df['Close'], length=12)
#     df['EMA24'] = ta.ema(df['Close'], length=24)
#     df['EMA36'] = ta.ema(df['Close'], length=36)
#     df['EMA48'] = ta.ema(df['Close'], length=48)
#     df['EMA100'] = ta.ema(df['Close'], length=100)
#     df['EMA200'] = ta.ema(df['Close'], length=200)
#
#
#     # MACD (Moving Average Convergence Divergence)
#     macd = ta.macd(df['Close'])
#     df['MACD'] = macd['MACD_12_26_9']
#     df['MACD_Signal'] = macd['MACDs_12_26_9']
#     df['MACD_Hist'] = macd['MACDh_12_26_9']
#
#     # Bollinger Bands
#     bb_period = 20
#     bb_std_dev = 2
#     bb = ta.bbands(df['Close'], length=bb_period, std=bb_std_dev)
#     df['BB_upper'] = bb['BBU_20_2.0']
#     df['BB_middle'] = bb['BBM_20_2.0']
#     df['BB_lower'] = bb['BBL_20_2.0']
#
#
#     # Bollinger Bands
#     bb_period = 20
#     bb_std_dev = 2
#     bb_diff = ta.bbands(df['Close_Diff'], length=bb_period, std=bb_std_dev)
#     df['BB_upper_diff'] = bb_diff['BBU_20_2.0']
#     df['BB_middle_diff'] = bb_diff['BBM_20_2.0']
#     df['BB_lower_diff'] = bb_diff['BBL_20_2.0']
#
#
#     # ATR (Average True Range)
#     df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=20)
#
#     # ADX (Average Directional Index)
#     adx = ta.adx(df['High'], df['Low'], df['Close'], length=20)
#     df['ADX'] = adx[f'ADX_{20}']
#     df['DMP'] = adx[f'DMP_{20}']
#     df['DMN'] = adx[f'DMN_{20}']
#
#     # Stochastic Oscillator
#     stoch = ta.stoch(df['High'], df['Low'], df['Close'], length=20)
#     df['Stoch_K'] = stoch['STOCHk_14_3_3']
#     df['Stoch_D'] = stoch['STOCHd_14_3_3']
#
#     # OBV (On-Balance Volume)
#     # df['OBV'] = ta.obv(df['Close'], df['Volume'])
#
#     # CCI (Commodity Channel Index)
#     df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
#
#     # Williams %R
#     df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'], length=20)
#
#     # MFI (Money Flow Index)
#     # df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=history_steps)
#
#     # Calculating Slopes
#     df['Slope_MA12'] = calculate_slope(df['SMA12'], 12)
#     df['Slope_MA24'] = calculate_slope(df['SMA24'], 24)
#     df['Slope_MA36'] = calculate_slope(df['SMA36'], 36)
#     df['Slope_MA48'] = calculate_slope(df['SMA48'], 48)
#     df['Slope_MA100'] = calculate_slope(df['SMA100'], 100)
#     df['Slope_MA200'] = calculate_slope(df['SMA200'], 200)
#
#
#     df['Slope_EMA12'] = calculate_slope(df['EMA12'], 12)
#     df['Slope_EMA24'] = calculate_slope(df['EMA24'], 24)
#     df['Slope_EMA36'] = calculate_slope(df['EMA36'], 36)
#     df['Slope_EMA48'] = calculate_slope(df['EMA48'], 48)
#     df['Slope_EMA100'] = calculate_slope(df['EMA100'], 100)
#     df['Slope_EMA200'] = calculate_slope(df['EMA200'], 200)
#
#
#     df['Slope_FMA'] = calculate_slope(df[f"Close_FMA"], future_steps)
#     # df['Slope_FEMA'] = calculate_slope(df[f"Close_FEMA"], future_steps)
#
#
#     # TODO
#     # ####
#     # ####
#     # ####
#     # ####
#
#     # df["Trend"] = (df["Close_FMA"] - df["Close"]).apply(lambda x: "Up" if x >= threshhold else ("Down" if x <= -threshhold else "Stable"))
#     # df["Trend"] = df["Slope_FEMA"].apply(lambda x: "Up" if x >= threshhold else ("Down" if x <= -threshhold else "Stable"))
#     df["Trend"] = df["Slope_FMA"].apply(lambda x: "Up" if x >= threshhold else "Stable")
#     # df["Trend"] = (df["Close_FEMA"] - df["Close"]).apply(lambda x: "Up" if x >= threshhold else ("Down" if x <= -threshhold else "Stable"))
#
#     # df["Trend_str"] = df["Close_FEMA"] - df["Close"]
#     # df["Trend_str"] = df["Close_FMA"] - df["Close"]
#     # df["Trend"] = df["Trend_str"].apply(lambda x: "Up" if x >= threshhold else "Stable")
#
#
#
#     return df


def calculate_fma(df, window, column):
    fma = [np.nan] * len(df)
    for i in range(len(df) - window + 1):
        fma[i] = df[column][i:i+window].mean()
    return fma


def calculate_fema(df, span, column):
    fema = [np.nan] * len(df)
    for i in range(len(df) - span + 1):
        fema[i] = df[column][i:i+span].ewm(span=span, adjust=False).mean().iloc[-1]
    return fema



def calculate_averages(live, df, history_steps, future_steps, threshhold, test=False):

    #TODO
    # ####
    # ####
    # ####
    # ####


    #################################################################### GENERAL
    # Timetable
    if "Date" in list(df.columns) and "Time" in list(df.columns):
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    if "Up" in list(df.columns) and "Down" in list(df.columns):
        df['Volume'] = df['Up'] + df['Down']

    # History Close Diff
    df["Close_Diff"] = df["Close"].diff().fillna(0)

    if not test:
        #################################################################### HISTORY

        # History Mouving Average
        df[f"Close_Diff_MA12"] = df["Close_Diff"].rolling(window=12).mean()
        df[f"Close_Diff_MA24"] = df["Close_Diff"].rolling(window=24).mean()
        df[f"Close_Diff_MA36"] = df["Close_Diff"].rolling(window=36).mean()
        df[f"Close_Diff_MA48"] = df["Close_Diff"].rolling(window=48).mean()
        df[f"Close_Diff_MA100"] = df["Close_Diff"].rolling(window=100).mean()
        df[f"Close_Diff_MA200"] = df["Close_Diff"].rolling(window=200).mean()

        # EMA (Exponential Moving Average)
        df['Close_Diff_EMA12'] = ta.ema(df['Close_Diff'], length=12)
        df['Close_Diff_EMA24'] = ta.ema(df['Close_Diff'], length=24)
        df['Close_Diff_EMA36'] = ta.ema(df['Close_Diff'], length=36)
        df['Close_Diff_EMA48'] = ta.ema(df['Close_Diff'], length=48)
        df['Close_Diff_EMA100'] = ta.ema(df['Close_Diff'], length=100)
        df['Close_Diff_EMA200'] = ta.ema(df['Close_Diff'], length=200)

        df['RSI12'] = ta.rsi(df['Close'], length=12)
        df['RSI24'] = ta.rsi(df['Close'], length=24)
        df['RSI36'] = ta.rsi(df['Close'], length=36)
        df['RSI48'] = ta.rsi(df['Close'], length=48)
        df['RSI100'] = ta.rsi(df['Close'], length=100)
        df['RSI200'] = ta.rsi(df['Close'], length=200)

        # SMA (Simple Moving Average)
        df['SMA12'] = ta.sma(df['Close'], length=12)
        df['SMA24'] = ta.sma(df['Close'], length=24)
        df['SMA36'] = ta.sma(df['Close'], length=36)
        df['SMA48'] = ta.sma(df['Close'], length=48)
        df['SMA100'] = ta.sma(df['Close'], length=100)
        df['SMA200'] = ta.sma(df['Close'], length=200)

        df['EMA12'] = ta.ema(df['Close'], length=12)
        df['EMA24'] = ta.ema(df['Close'], length=24)
        df['EMA36'] = ta.ema(df['Close'], length=36)
        df['EMA48'] = ta.ema(df['Close'], length=48)
        df['EMA100'] = ta.ema(df['Close'], length=100)
        df['EMA200'] = ta.ema(df['Close'], length=200)

        # MACD (Moving Average Convergence Divergence)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']

        # Bollinger Bands
        bb_period = 20
        bb_std_dev = 2
        bb = ta.bbands(df['Close'], length=bb_period, std=bb_std_dev)
        df['BB_upper'] = bb['BBU_20_2.0']
        df['BB_middle'] = bb['BBM_20_2.0']
        df['BB_lower'] = bb['BBL_20_2.0']

        # Bollinger Bands
        bb_period = 20
        bb_std_dev = 2
        bb_diff = ta.bbands(df['Close_Diff'], length=bb_period, std=bb_std_dev)
        df['BB_upper_diff'] = bb_diff['BBU_20_2.0']
        df['BB_middle_diff'] = bb_diff['BBM_20_2.0']
        df['BB_lower_diff'] = bb_diff['BBL_20_2.0']

        # ATR (Average True Range)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=20)

        # ADX (Average Directional Index)
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=20)
        df['ADX'] = adx[f'ADX_{20}']
        df['DMP'] = adx[f'DMP_{20}']
        df['DMN'] = adx[f'DMN_{20}']

        # Stochastic Oscillator
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], length=20)
        df['Stoch_K'] = stoch['STOCHk_14_3_3']
        df['Stoch_D'] = stoch['STOCHd_14_3_3']

        # OBV (On-Balance Volume)
        # df['OBV'] = ta.obv(df['Close'], df['Volume'])

        # CCI (Commodity Channel Index)
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)

        # Williams %R
        df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'], length=20)

        # MFI (Money Flow Index)
        # df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=history_steps)

        # Calculating Slopes
        df['Slope_MA12'] = calculate_slope(df['SMA12'], 12)
        df['Slope_MA24'] = calculate_slope(df['SMA24'], 24)
        df['Slope_MA36'] = calculate_slope(df['SMA36'], 36)
        df['Slope_MA48'] = calculate_slope(df['SMA48'], 48)
        df['Slope_MA100'] = calculate_slope(df['SMA100'], 100)
        df['Slope_MA200'] = calculate_slope(df['SMA200'], 200)

        df['Slope_EMA12'] = calculate_slope(df['EMA12'], 12)
        df['Slope_EMA24'] = calculate_slope(df['EMA24'], 24)
        df['Slope_EMA36'] = calculate_slope(df['EMA36'], 36)
        df['Slope_EMA48'] = calculate_slope(df['EMA48'], 48)
        df['Slope_EMA100'] = calculate_slope(df['EMA100'], 100)
        df['Slope_EMA200'] = calculate_slope(df['EMA200'], 200)



    #################################################################### FUTURE
    if not live:
        ##### FMA
        # for i in [12, 24, 36, 48, 100, 200]:
        #     df[f"Close_FMA{i}"] = df["Close"].shift(-i).rolling(window=i).mean()
        #     df[f'Close_FMA{i}'] = calculate_fma(df, i, column="Close")

        # for i in [12, 24, 36, 48, 100, 200]:
        #     # df[f"Close_Diff_FMA{i}"] = df["Close_Diff"].shift(-i).rolling(window=i).mean()
        #     df[f'Close_Diff_FMA{i}'] = calculate_fma(df, i, column="Close_Diff")



        df[f"Close_FMA"] = calculate_fma(df, future_steps, column="Close")
        # df['Slope_FMA'] = calculate_future_slope(df[f"Close_FMA"], future_steps)
        # df['Slope_FMA'] = calculate_slope(df[f"Close_FMA"], future_steps)
        # df[f"Close_Diff_FMA"] = calculate_fma(df, future_steps, column="Close_Diff")

        df["Close_FMA_minus_Close"] = df["Close_FMA"] - df["Close"]
        # df['Slope_Close_FMA_minus_Close'] = calculate_future_slope(df[f"Close_FMA_minus_Close"], future_steps)


        ##### FEMA
        # for i in [12, 24, 36, 48, 100, 200]:
        #     df[f'Close_FEMA{i}'] = df['Close'].shift(-i).ewm(span=i, adjust=False).mean()
        #     df[f"Close_FEMA{i}"] = calculate_fema(df, i, column="Close")

        # for i in [12, 24, 36, 48, 100, 200]:
        #     # df[f'Close_Diff_FEMA{i}'] = df['Close_Diff'].shift(-i).ewm(span=i, adjust=False).mean()
        #     df[f'Close_Diff_FEMA{i}'] = calculate_fema(df, i, column="Close_Diff")


        # df['Close_Diff_FEMA'] = calculate_fema(df, future_steps, column="Close_Diff")
        # df["Close_FEMA_minus_Close"] = df["Close_FEMA"] - df["Close"]


        # df["Close_FEMA"] = calculate_fema(df, future_steps, column="Close")
        # df['Slope_FEMA'] = calculate_future_slope(df[f"Close_FEMA"], future_steps)
        # df['Slope_FEMA'] = calculate_slope(df[f"Close_FEMA"], future_steps)


        # TODO
        # ################################################################### ZIELWERTE

        # df["Trend"] = X.apply(lambda x: "Up" if x >= threshhold else ("Down" if x <= -threshhold else "Stable"))


        # df["Trend"] = df["Slope_Close_FMA_minus_Close"].apply(lambda x: "Up" if x >= threshhold else "Stable")
        df["Trend"] = df["Close_FMA_minus_Close"].apply(lambda x: "Up" if x >= threshhold else "Stable")

        # df["Trend"] = df["Slope_FMA"].apply(lambda x: "Up" if x >= threshhold else "Stable")
        # df["Trend"] = df["Slope_FEMA"].apply(lambda x: "Up" if x >= threshhold else "Stable")

        # df["Trend"] = (df["Close_FEMA"] - df["Close"]).apply(lambda x: "Up" if x >= threshhold else ("Down" if x <= -threshhold else "Stable"))

        # df["Trend_str"] = df["Close_FEMA"] - df["Close"]
        # df["Trend_str"] = df["Close_FMA"] - df["Close"]
        # df["Trend"] = df["Trend_str"].apply(lambda x: "Up" if x >= threshhold else "Stable")

        # pivot_points = pd.DataFrame(index=df.index)
        # pivot_points['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        # pivot_points['Support1'] = 2 * pivot_points['Pivot'] - df['High']
        # pivot_points['Resistance1'] = 2 * pivot_points['Pivot'] - df['Low']
        # pivot_points['Support2'] = pivot_points['Pivot'] - (df['High'] - df['Low'])
        # pivot_points['Resistance2'] = pivot_points['Pivot'] + (df['High'] - df['Low'])
        # pivot_points['Support3'] = pivot_points['Support2'] - (df['High'] - df['Low'])
        # pivot_points['Resistance3'] = pivot_points['Resistance2'] + (df['High'] - df['Low'])
        # df.join(pivot_points)

        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Support1'] = 2 * df['Pivot'] - df['High']
        df['Resistance1'] = 2 * df['Pivot'] - df['Low']
        df['Support2'] = df['Pivot'] - (df['High'] - df['Low'])
        df['Resistance2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['Support3'] = df['Support2'] - (df['High'] - df['Low'])
        df['Resistance3'] = df['Resistance2'] + (df['High'] - df['Low'])

    return df



def prepare_stock_data(live, db, from_stock=False, from_table=None, to_table=None, history_steps=None, future_steps=None, signal_grenzwert=None, df_stock_orig=None, test=False):

    if not from_stock:
        df_stock_orig = query_database(db=db, table=from_table)
        # df_stock_orig = df_stock_orig[-500:]

    df_stock_orig = calculate_averages(live, test=test, df=df_stock_orig, history_steps=history_steps, future_steps=future_steps, threshhold=signal_grenzwert)
    # print(df_stock_orig.head())

    columns = df_stock_orig.columns

    # for column in columns:
    #     print(f"'{column}',")

    # df_stock_filtered = df_stock_orig[columns]


    if not live:
        print(df_stock_orig.head())
        print(df_stock_orig.tail())

        # x_interval_min = int(future_steps*5)
        # plot_stock_prices(df_stock_orig[-1440:], test=test, secondary_y_scale=1, x_interval_min=x_interval_min, y_interval_dollar=25, additional_lines=[
        #     # ('Slope_FMA', 1),
        #     # ('Slope_FEMA', 1),
        #     # ('Slope_Close_FMA_minus_Close', 1),
        #     ('Close_FMA_minus_Close', 1),
        #
        #     # ('Pivot', 1),
        #     # ('Support1', 1),
        #     # ('Resistance1', 1),
        #     # ('Support2', 1),
        #     # ('Resistance2', 1),
        #     # ('Support3', 1),
        #     # ('Resistance3', 1),
        #
        # ])



    # Berechnung der prozentualen Anteile der Unique-Werte in der Spalte "Trend"
    # trend_counts = df_stock_orig['Trend'].value_counts(normalize=True) * 100
    # print("Prozentualer Anteil der Unique-Werte in der Spalte 'Trend':")
    # print(trend_counts)

    df_stock_orig.dropna(inplace=True)

    if not test:
        if not live:
            save_to_db(dataframe=df_stock_orig, to_table=to_table, db="trading_bot")
        else:
            return df_stock_orig
    else:
        exit()

def copy_missing_columns(df_with_predictions, original_df):
    # Iteriere über alle Spalten im original_df
    for column in original_df.columns:
        # Überprüfe, ob die Spalte nicht im df_with_predictions vorhanden ist
        if column not in df_with_predictions.columns:
            # Kopiere die Spalte vom original_df in den df_with_predictions
            df_with_predictions[column] = original_df[column]
    return df_with_predictions


def evaluate_trend_prediction(df, trend_col='Trend_soll', predicted_trend_col='Trend', ignore_label='Stable'):
    # Entfernen von Zeilen mit NaN-Werten in den relevanten Spalten
    df_clean = df.dropna(subset=[trend_col, predicted_trend_col])

    # Entfernen von Zeilen, bei denen die Vorhersage 'Stable' ist
    df_clean = df_clean[df_clean[predicted_trend_col] != ignore_label]

    # Berechnung der Übereinstimmung
    correct_predictions = (df_clean[trend_col] == df_clean[predicted_trend_col]).sum()
    total_predictions = len(df_clean)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Berechnung der falsch positiven Vorhersagen
    false_positives_up = ((df_clean[trend_col] != 'Up') & (df_clean[predicted_trend_col] == 'Up')).sum()
    false_positives_down = ((df_clean[trend_col] != 'Down') & (df_clean[predicted_trend_col] == 'Down')).sum()

    # Ausgabe der Ergebnisse
    print(f"Anzahl der Vorhersagen (ohne '{ignore_label}'): {total_predictions}")
    print(f"Korrekte Vorhersagen: {correct_predictions}")
    print(f"Genauigkeit: {accuracy:.2%}")
    print(f"Falsch positive 'Up' Vorhersagen: {false_positives_up}")
    print(f"Falsch positive 'Down' Vorhersagen: {false_positives_down}")

    return accuracy, false_positives_up, false_positives_down


def calculate_feature_importance_and_predict(df, target_column, prediction_data=None, return_predictions=False):

    # Features und Zielspalte trennen
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Random Forest Modell trainieren
    rf = RandomForestClassifier(n_estimators=200, random_state=42, verbose=1)
    rf.fit(X_train, y_train)

    # Vorhersagen für die übergebenen Daten machen, falls erforderlich
    predictions = rf.predict(prediction_data)
    return predictions



def combined_feature_selection(df, target_column, num_features=10):
    """
    Kombiniert mehrere Methoden zur Feature-Auswahl und wählt die wichtigsten Features basierend auf den Ergebnissen aller Methoden aus.

    Parameters:
    df (pd.DataFrame): Der Datensatz mit Features und Zielspalte.
    target_column (str): Der Name der Zielspalte.
    num_features (int): Die Anzahl der wichtigsten Features, die ausgewählt werden sollen.

    Returns:
    list: Liste der wichtigsten Features.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 1. Korrelationsanalyse
    correlation_matrix = df.corr()
    corr_target = abs(correlation_matrix[target_column])
    corr_features = corr_target.drop(target_column).sort_values(ascending=False).head(num_features).index.tolist()

    # 2. Univariate Feature Selection (ANOVA F-Test)
    univariate_selector = SelectKBest(score_func=f_classif, k=num_features)
    univariate_selector.fit(X, y)
    univariate_features = X.columns[univariate_selector.get_support()].tolist()

    # 3. Recursive Feature Elimination (RFE)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe_selector = RFE(model, n_features_to_select=num_features)
    rfe_selector.fit(X, y)
    rfe_features = X.columns[rfe_selector.get_support()].tolist()

    # 4. Feature-Wichtigkeit aus Random Forest
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    rf_features = feature_importances.sort_values(by='Importance', ascending=False).head(num_features)['Feature'].tolist()

    # 5. L1-basierte Feature Selection (Lasso)
    lasso = Lasso(alpha=0.01)
    lasso.fit(X, y)
    lasso_model = SelectFromModel(lasso, prefit=True)
    lasso_features = X.columns[lasso_model.get_support()].tolist()

    # 6. PCA (Principal Component Analysis) (Optional)
    # pca = PCA(n_components=num_features)
    # principalComponents = pca.fit_transform(X)
    # pca_features = [f'PC{i}' for i in range(1, num_features+1)]

    # Zusammenführung der Ergebnisse
    all_features = corr_features + univariate_features + rfe_features + rf_features + lasso_features
    feature_counts = pd.Series(all_features).value_counts()

    # Auswahl der häufigsten Features
    selected_features = feature_counts.head(num_features).index.tolist()

    return selected_features


def replace_following_trend(df, column_name, search_word, min_count):
    """
    Ersetzt aufeinanderfolgende 'search_word' Werte in einer Spalte durch 'Stable' ab einer bestimmten Menge.

    Parameters:
    df (pd.DataFrame): Der DataFrame mit den Daten.
    column_name (str): Der Name der Spalte, die überprüft werden soll.
    search_word (str): Das Suchwort, das überprüft werden soll (z.B. 'Up').
    min_count (int): Die Mindestanzahl an aufeinanderfolgenden Suchwörtern, ab der ersetzt werden soll.

    Returns:
    pd.DataFrame: Der bearbeitete DataFrame.
    """
    trend_series = df[column_name]
    count = 0
    start_idx = -1

    for i in range(len(trend_series)):
        if trend_series.iloc[i] == search_word:
            if count == 0:
                start_idx = i
            count += 1
        else:
            if count >= min_count:
                trend_series.iloc[start_idx + 1:start_idx + count] = "Stable"
            else:
                trend_series.iloc[start_idx:start_idx + count] = "Stable"
            count = 0
            start_idx = -1

    # Überprüfe die letzte Serie, falls sie am Ende des DataFrames endet
    if count >= min_count:
        trend_series.iloc[start_idx + 1:start_idx + count] = "Stable"
    else:
        trend_series.iloc[start_idx:start_idx + count] = "Stable"

    df[column_name] = trend_series
    return df



if __name__ == "__main__":
    faulthandler.enable()

    # database_name_optuna = f"nasdq_neu_{50}"  # ohne regularizers.l2     #  history_steps = 36, future_steps = 12, signal_grenzwert = 25 !! keine absoluten Werte
    # database_name_optuna = f"nasdq_neu_{51}"  # ohne regularizers.l2     #  history_steps = 36, future_steps = 36, signal_grenzwert = 25 !! alle Spalten
    # database_name_optuna = f"nasdq_neu_{52}"  # ohne regularizers.l2     #  history_steps = 36, future_steps = 36, signal_grenzwert = 25 !! alle Spalten, Trend FMA Genauigkeit: 51.27%
    # database_name_optuna = f"nasdq_neu_{53}"  # ohne regularizers.l2     #  history_steps = 36, future_steps = 12, signal_grenzwert = 25 !! alle Spalten, Trend FMA Genauigkeit:
    # database_name_optuna = f"nasdq_neu_{99991}"  # ohne regularizers.l2     #  history_steps = 36, future_steps = 12, signal_grenzwert = 25 !! alle Spalten, Trend FMA Genauigkeit:
    database_name_optuna = f"nasdq_neu_{5311}"  # ohne regularizers.l2     #  history_steps = 36, future_steps = 12, signal_grenzwert = 25 !! alle Spalten, Trend FMA Genauigkeit:

    db = "trading_bot"
    orig_stock_data_table = 'nasdq_5min'
    df_all_data_enriched_data = f"{orig_stock_data_table}_enriched_data_{database_name_optuna}"

    # history_steps = 36
    # future_steps = 12
    # signal_grenzwert = 0.5

    history_steps = 36
    future_steps = 2
    # future_steps = 6
    # future_steps = 12
    # future_steps = 24
    # future_steps = 36
    # future_steps = 48
    # future_steps = 60
    # future_steps = 72
    # future_steps = 84
    # future_steps = 96
    # future_steps = 108
    # future_steps = 120
    # future_steps = 200




    signal_grenzwert = 0

    time_series_sequence_length = 12  # DIE LETZTEN 3 STUNDEN!!!!



    list_of_additional_columns = [
            "Datetime",
            'Close_FMA',
            'Close_FEMA',
            'Close_Diff_FMA',
            "Close_FMA_minus_Close",
            "Close_FEMA_minus_Close",
            'Slope_FEMA',
            'Slope_FMA',
            "Close",
            # "Trend_str"

        ]

    list_of_predicting_columns = [
        'Volume',
        'Close_Diff',
        "Close",  # not diff
        'Close_Diff_MA12',
        'Close_Diff_MA24',
        'Close_Diff_MA36',
        'Close_Diff_MA48',
        'Close_Diff_MA100',
        'Close_Diff_MA200',
        'Close_Diff_EMA12',
        'Close_Diff_EMA24',
        'Close_Diff_EMA36',
        'Close_Diff_EMA48',
        'Close_Diff_EMA100',
        'Close_Diff_EMA200',
        'RSI12',
        'RSI24',
        'RSI36',
        'RSI48',
        'RSI100',
        'RSI200',

        'SMA12',  # not diff
        'SMA24',  # not diff
        'SMA36',  # not diff
        'SMA48',  # not diff
        'SMA100',  # not diff
        'SMA200',  # not diff
        'EMA12',  # not diff
        'EMA24',  # not diff
        'EMA36',  # not diff
        'EMA48',  # not diff
        'EMA100',  # not diff
        'EMA200',  # not diff

        'MACD',
        'MACD_Signal',
        'MACD_Hist',

        'BB_upper',  # not diff
        'BB_middle',  # not diff
        'BB_lower',  # not diff

        'BB_upper_diff',
        'BB_middle_diff',
        'BB_lower_diff',

        'ATR',
        'ADX',
        'DMP',
        'DMN',
        'Stoch_K',
        'Stoch_D',
        'CCI',
        'Williams_R',
        'Slope_MA12',
        'Slope_MA24',
        'Slope_MA36',
        'Slope_MA48',
        'Slope_MA100',
        'Slope_MA200',
        'Slope_EMA12',
        'Slope_EMA24',
        'Slope_EMA36',
        'Slope_EMA48',
        'Slope_EMA100',
        'Slope_EMA200',
    ]

    list_of_tuning_columns = list_of_predicting_columns + ["Trend"]

    all_columns = list_of_additional_columns + list_of_tuning_columns
    # print(all_columns)
    # exit()





    ########################### LOAD TXT TO DB
    # load_txt_to_mysql(database_name="trading_bot", filename='nasdq_5min.txt', table_name="nasdq_5min")


    ########################### prepare_stock_data
    prepare_stock_data(test=False, live=False, db=db, from_table=orig_stock_data_table, to_table=df_all_data_enriched_data, history_steps=history_steps, future_steps=future_steps, signal_grenzwert=signal_grenzwert)
    # prepare_stock_data(test=True, live=False, db=db, from_table=orig_stock_data_table, to_table=df_all_data_enriched_data, history_steps=history_steps, future_steps=future_steps, signal_grenzwert=signal_grenzwert)
    # exit()



    ########################### check importance
    # df_all_data = query_database(db=db, table=df_all_data_enriched_data)
    # df_all_data_filtered_for_tuning = df_all_data[list_of_tuning_columns]
    # importance_df = calculate_feature_importance_and_predict(df=df_all_data_filtered_for_tuning, target_column="Trend")
    # print(importance_df)
    # df_all_data_filtered_for_tuning['Trend'] = df_all_data_filtered_for_tuning['Trend'].map({'Stable': 0, 'Up': 1})
    # selected_features = combined_feature_selection(df=df_all_data_filtered_for_tuning, target_column="Trend", num_features=62)
    # print("Wichtige Features:", selected_features)



    ########################### NEURONALES TUNING
    # #TODO
    # # TUNING
    # df_all_data = query_database(db=db, table=df_all_data_enriched_data)
    # workers = 2
    # while True:
    #     try:
    #         train_model_v3(tuning=True, n_trials=workers * 5, n_jobs=workers, time_series_sequence_length=time_series_sequence_length, database_name_optuna=database_name_optuna, show_progression=False, verbose=True, db_wkdm=df_all_data[list_of_tuning_columns])
    #         break
    #     except:
    #         pass
    # df_all_data_splitted_filtered_for_training, _ = split_data_v3(df_all_data[list_of_tuning_columns], test_size=0.1)
    # train_model_v3(tuning=False, n_trials=workers * 1, n_jobs=workers, time_series_sequence_length=time_series_sequence_length, database_name_optuna=database_name_optuna, show_progression=False, verbose=True, db_wkdm=df_all_data_splitted_filtered_for_training)
    # exit()




    ########################### RANDOM FOREST TUNING
    df_all_data = query_database(db=db, table=df_all_data_enriched_data)
    df_all_data_splitted_filtered_for_training, df_all_data_splitted_filtered_for_testing = split_data_v3(df_all_data, test_size=0.1)

    df_all_data_splitted_filtered_for_training = df_all_data_splitted_filtered_for_training[list_of_tuning_columns]

    original_df = df_all_data_splitted_filtered_for_testing.copy()
    df_all_data_splitted_filtered_for_testing = df_all_data_splitted_filtered_for_testing[list_of_tuning_columns]


    ### SAVE MODEL
    train_and_save_model_random_forest(df=df_all_data_splitted_filtered_for_training, target_column="Trend", database_name_optuna=database_name_optuna)
    ## PREDICTION
    df_with_predictions_rf = load_model_and_predict_random_forest(prediction_data=df_all_data_splitted_filtered_for_testing[list_of_predicting_columns], database_name_optuna=database_name_optuna)

    df_with_predictions_rf["Close"] = original_df["Close"]
    df_with_predictions_rf["Datetime"] = original_df["Datetime"]

    # plot_stock_prices(df_with_predictions_rf[-1440:], test=True, secondary_y_scale=1, x_interval_min=60, y_interval_dollar=25, additional_lines=[
    #     # ('Slope_FEMA', 1),
    #     # ('Slope_FMA', 1),
    # ])
    # exit()



    ########################### PREDICTING
    # Query Data
    # df_all_data = query_database(db=db, table=df_all_data_enriched_data)
    #
    # # Filter Data by Columns
    # # df_all_data = df_all_data[all_columns]
    #
    # # Split Filtered Data
    # df_all_data_splitted_filtered_for_training, df_all_data_splitted_filtered_for_testing = split_data_v3(df_all_data, test_size=0.1)
    # df_all_data_splitted_filtered_for_training = df_all_data_splitted_filtered_for_training[list_of_tuning_columns]
    #
    # original_df = df_all_data_splitted_filtered_for_testing.copy()
    #
    # # Copy Data for later use
    # df_pred = df_all_data_splitted_filtered_for_testing.copy()
    # df_pred = df_pred[list_of_predicting_columns]
    #
    # df_with_predictions = predict_and_integrate_live(df=df_pred, database_name_optuna=database_name_optuna, time_series_sequence_length=time_series_sequence_length)
    #
    # # Copy Data for analysis
    # df_with_predictions["Close"] = original_df["Close"]
    # df_with_predictions["Datetime"] = original_df["Datetime"]
    # df_with_predictions = df_with_predictions[["Datetime", "Close", "Trend_predicted"]]
    # df_with_predictions = df_with_predictions.rename(columns={'Trend_predicted': 'Trend'})
    # df_with_predictions["Trend_soll"] = original_df["Trend"]
    #
    # # copy_missing_columns(df_with_predictions, original_df)
    #
    # accuracy = evaluate_trend_prediction(df_with_predictions, trend_col='Trend_soll', predicted_trend_col='Trend')
    #
    #
    # # plot_stock_prices(original_df, secondary_y_scale=1, x_interval_min=60, y_interval_dollar=25, additional_lines=[
    # #     ('Slope_FEMA', 1),
    # #     ('Slope_FMA', 1),
    # # ])
    #
    # # df_with_predictions["Trend_orig"] = df_with_predictions["Trend"]
    # # df_with_predictions = replace_following_trend(df=df_with_predictions, column_name="Trend", search_word="Up", min_count=6)
    #
    # plot_stock_prices(df_with_predictions[-500:], secondary_y_scale=1, x_interval_min=60, y_interval_dollar=25, additional_lines=[
    #     ('Slope_FEMA', 1),
    #     ('Slope_FMA', 1),
    # ])
    #
    #
    # importance_df, predictions = calculate_feature_importance_and_predict(df=df_all_data_splitted_filtered_for_training, target_column='Trend', prediction_data=df_pred, return_predictions=True)
    # df_pred['Trend_soll'] = original_df["Trend"]
    # df_pred['Trend_predicted'] = predictions
    # df_pred['Trend_predicted_nn'] = df_with_predictions["Trend"]
    #
    # df_pred['Datetime'] = df_with_predictions["Datetime"]
    # df_pred['Close'] = df_with_predictions["Close"]
    #
    # df_pred['Slope_FMA'] = df_with_predictions["Slope_FMA"]
    # df_pred['Slope_FEMA'] = df_with_predictions["Slope_FEMA"]
    #
    # df_pred = df_pred[["Datetime", "Close", "Trend_soll", "Trend_predicted", "Trend_predicted_nn", "Slope_FMA", "Slope_FEMA"]]
    # df_pred.dropna(inplace=True)
    # df_pred.reset_index(drop=True, inplace=True)
    #
    # df_pred["Possible_buy"] = "Stable"
    # df_pred["Trend"] = "Stable"
    #
    #
    #
    # for i in range(2, len(df_pred) - 2):
    #     if (df_pred.loc[i, "Trend_predicted"] == "Up" and df_pred.loc[i - 1, "Trend_predicted"] == "Up" and
    #             df_pred.loc[i, "Trend_predicted_nn"] == "Up" and df_pred.loc[i - 1, "Trend_predicted_nn"] == "Up"):
    #         # Zusätzliche Bedingungen um falsch positive Ergebnisse zu minimieren
    #         if ((df_pred.loc[i + 1, "Trend_predicted"] == "Up" and df_pred.loc[i + 1, "Trend_predicted_nn"] == "Up") or
    #             (df_pred.loc[i - 2, "Trend_predicted"] == "Up" and df_pred.loc[i - 2, "Trend_predicted_nn"] == "Up")) and \
    #                 ((df_pred.loc[i + 2, "Trend_predicted"] == "Up" and df_pred.loc[i + 2, "Trend_predicted_nn"] == "Up") or
    #                  (df_pred.loc[i - 3, "Trend_predicted"] == "Up" and df_pred.loc[i - 3, "Trend_predicted_nn"] == "Up")):
    #             df_pred.loc[i, "Trend"] = "Up"
    #
    #
    # # Berechnung der Genauigkeit der Spalte "bought"
    # total_bought = df_pred["Trend"].value_counts().get("Up", 0)
    # correct_bought = len(df_pred[(df_pred["Trend"] == "Up") & (df_pred["Trend_soll"] == "Up")])
    #
    # if total_bought > 0:
    #     accuracy = correct_bought / total_bought * 100
    # else:
    #     accuracy = 0
    #
    #
    # df_pred = replace_following_trend(df=df_pred, column_name="Trend", search_word="Up", min_count=3)
    #
    # plot_stock_prices(df_pred[-500:], secondary_y_scale=1, x_interval_min=60, y_interval_dollar=25, additional_lines=[
    #     ('Slope_FMA', 1),
    #     ('Slope_FEMA', 1),
    #     # ('Close_Diff_FMA', 1),
    #     ])
    #
    # print(f"Anzahl der gemachten Trades (bought): {total_bought}")
    # print(f"Anzahl der richtigen Trades (bought): {correct_bought}")
    # print(f"Genauigkeit der Spalte 'bought': {accuracy:.2f}%")
    #
    # # print(importance_df)
    # # print(predictions)
    # # print(df_pred)
    # df_pred.to_excel("df_pred.xlsx")
    # exit()
    ########################### PREDICTING


    def job(time_stay):
        df = get_stock_data(
            ticker='NQ=F',
            # ticker='RHM.DE',
            # ticker='NQM24.CME',

            start='2024-05-13',
            end='2024-06-14',
            # end=end,
            interval='5m',
            prepost=True,
            actions=True,
            auto_adjust=True,
            back_adjust=False,
            proxy=None,
            rounding=False,
        )

        # print(df.tail())

        df_all_data = prepare_stock_data(live=True, db=db, from_stock=True, df_stock_orig=df, to_table="yf_nasdq", history_steps=history_steps, future_steps=future_steps, signal_grenzwert=signal_grenzwert)

        # df_all_data = query_database(db=db, table="yf_nasdq")
        original_df = df_all_data.copy()
        df_pred = df_all_data[list_of_predicting_columns].copy()


        # df_with_predictions = predict_and_integrate_live(df=df_pred, database_name_optuna=database_name_optuna, time_series_sequence_length=time_series_sequence_length)
        # df_with_predictions["Close"] = original_df["Close"]
        # df_with_predictions["Datetime"] = original_df["Datetime"]
        # df_with_predictions = df_with_predictions[["Datetime", "Close", "Trend"]]


        df_with_predictions_rf = load_model_and_predict_random_forest(prediction_data=df_pred, database_name_optuna=database_name_optuna)
        df_with_predictions_rf["Close"] = original_df["Close"]
        df_with_predictions_rf["Datetime"] = original_df["Datetime"]

        df_with_predictions_rf["BB_upper"] = original_df["BB_upper"]
        df_with_predictions_rf["BB_middle"] = original_df["BB_middle"]
        df_with_predictions_rf["BB_lower"] = original_df["BB_lower"]


        df_with_predictions_rf = df_with_predictions_rf[["Datetime", "Close", "Trend", "BB_upper", "BB_middle", "BB_lower"]]
        # print(df_with_predictions_rf.tail())

        # df_with_predictions["Trend_soll"] = original_df["Trend"]
        # copy_missing_columns(df_with_predictions, original_df)

        # df_with_predictions = replace_following_trend(df=df_with_predictions, column_name="Trend", search_word="Up", min_count=4)
        # plot_stock_prices(df_with_predictions[-600:], test=False, secondary_y_scale=1, x_interval_min=30, y_interval_dollar=25, additional_lines=[
        #     ("BB_upper", 1),
        #     ("BB_middle", 1),
        #     ("BB_lower", 1)],
        #     time_stay=time_stay)

        hours = 48
        points = hours * 12

        plot_stock_prices(df_with_predictions_rf[-points:], test=False, secondary_y_scale=1, x_interval_min=30, y_interval_dollar=10,
                          additional_lines=[("BB_upper", 1),
                                            ("BB_middle", 1),
                                            ("BB_lower", 1)],
                          time_stay=time_stay)


    time_stay = 120
    # job(time_stay)

    while True:
        try:
            job(time_stay)
        except:
            time.sleep(10)
        # schedule.run_pending()
        # time.sleep(1)
