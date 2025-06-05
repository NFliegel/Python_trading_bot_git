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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.preprocessing.sequence import TimeseriesGenerator

from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from joblib import dump, load


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
    # study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True)


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

def train_model_v3(n_trials=None, n_jobs=None, database_name=None, database_name_optuna=None, table_name=None, show_progression=False, verbose=False):

    if verbose: print(f'load_data()')
    db_wkdm = load_data(database_name=database_name, table_name=table_name)

    # print(db_wkdm.head())
    # exit()

    if verbose: print(f'split_data_v3')
    train_df, test_df = split_data_v3(db_wkdm)

    if verbose: print(f'Preparing data for training')
    # X_train, Y_train, scaler_train = prepare_data(train_df, target_column_name=y_target_data, history_points=60)
    # X_test, Y_test, scaler_test = prepare_data(test_df, target_column_name=y_target_data, history_points=60)

    X_train, Y_train, scaler_train, encoder_train = prepare_data(train_df, target_column_name='target_category')
    X_test, Y_test, scaler_test, encoder_test = prepare_data(test_df, target_column_name='target_category')

    if verbose: print(f'load_study')
    study = load_study(database_name_optuna)

    if verbose: print(f'build_objective_with_data')
    objective_with_data = functools.partial(train_or_tune_model_v3, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, tuning=True, verbose=0, show_progression=show_progression)

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

def prepare_data(df, target_column_name):
    # Isolieren des Ziels und Umwandeln in One-Hot-Form
    target = df[target_column_name].values
    onehot_encoder = OneHotEncoder(sparse_output=False)
    target_encoded = onehot_encoder.fit_transform(target.reshape(-1, 1))

    # Vorbereitung der Feature-Skalierer
    feature_columns = df.drop(target_column_name, axis=1).columns
    scalers = {col: MinMaxScaler() for col in feature_columns}

    # Features skalieren
    features_scaled = np.zeros((df.shape[0], len(feature_columns)))
    for i, col in enumerate(feature_columns):
        features_scaled[:, i] = scalers[col].fit_transform(df[col].values.reshape(-1, 1)).flatten()

    X = features_scaled
    Y = target_encoded

    X = features_scaled.reshape(df.shape[0], X.shape[1], 1)  # Add channel dimension (1)

    # Reshape target if necessary (assuming multiple classes)
    if Y.shape[1] > 1:
        Y = Y.reshape(-1, Y.shape[1])  # Reshape to match number of classes


    # Überprüfung der Form der Daten
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    return X, Y, scalers, onehot_encoder


# Step 2
########################################################################################################################

def split_data_v3(dataframe, test_size=0.2):
    train_size = int(len(dataframe) * (1 - test_size))
    train_df = dataframe[:train_size]
    test_df = dataframe[train_size:]
    return train_df, test_df


# Step 3
########################################################################################################################

def train_or_tune_model_v3(trial=None, X_train=None, Y_train=None, X_test=None, Y_test=None, tuning=True, verbose=0, best_params=None, show_progression=False):

    # Modellparameter
    # norm TODO
    if tuning:
        min_neurons = int(math.sqrt(X_train.shape[1]))
        max_neurons = X_train.shape[1]
        print(f"n_units = trial.suggest_int('n_units', {min_neurons}, {max_neurons}, log=False)")

        n_layers = trial.suggest_int('n_layers', 1, 3)
        n_units = trial.suggest_int('n_units', min_neurons, max_neurons, log=False)
        # n_units = trial.suggest_int('n_units', min_neurons, 200, log=False)


        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        batch_size = trial.suggest_int('batch_size', 8, 1000, log=False)


    else:
        n_layers = best_params['n_layers']
        n_units = best_params['n_units']
        dropout_rate = best_params['dropout_rate']
        batch_size = best_params['batch_size']


    print(f'n_layers:{n_layers},n_units:{n_units},dropout_rate:{dropout_rate},batch_size:{batch_size}')

    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self, tuning=False):
            super().__init__()
            self.tuning = tuning

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        # def on_epoch_end(self, epoch, logs=None):
        #     epoch_time = time.time() - self.epoch_start_time
        #     current_lr = self.model.optimizer.lr.numpy()  # Zugriff auf die aktuelle Lernrate
        #     scale_value = 1
        #     if self.tuning:
        #         print(
        #             f"Epoch: {epoch + 1:2d}, Loss: {logs['loss'] / scale_value:15.3f}, Val Loss: {logs.get('val_loss', 'N/A') / scale_value:15.3f}, Zeit: {epoch_time:6.2f} Sek., LR: {current_lr:.2e}")
        #     else:
        #         print(
        #             f"Epoch: {epoch + 1:2d}, Loss: {logs['loss'] / scale_value:15.3f}, Zeit: {epoch_time:6.2f} Sek., LR: {current_lr:.2e}")
        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            current_lr = self.model.optimizer.lr.numpy()  # Zugriff auf die aktuelle Lernrate
            scale_value = 1
            if self.tuning:
                print(
                    f"Epoch: {epoch + 1:2d}, accuracy: {logs['accuracy'] / scale_value:15.3f}, Val accuracy: {logs.get('val_accuracy', 'N/A') / scale_value:15.3f}, Zeit: {epoch_time:6.2f} Sek., LR: {current_lr:.2e}")
            else:
                print(
                    f"Epoch: {epoch + 1:2d}, accuracy: {logs['accuracy'] / scale_value:15.3f}, Zeit: {epoch_time:6.2f} Sek., LR: {current_lr:.2e}")


    callbacks = []
    if tuning:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=tuning_reduce_lr_factor,
                                                         min_delta=tuning_reduce_lr_min_delta,
                                                         patience=tuning_reduce_lr_patience,
                                                         min_lr=tuning_reduce_lr_min_lr,
                                                         cooldown=tuning_reduce_lr_cooldown)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=tuning_early_stopping_patience,
                                                          min_delta=tuning_early_stopping_min_delta,
                                                          restore_best_weights=True)

        pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')
        callbacks.extend([early_stopping, reduce_lr, pruning_callback, CustomCallback(tuning=tuning)])

    else:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=building_reduce_lr_factor,
                                                         min_delta=building_reduce_lr_min_delta,
                                                         patience=building_reduce_lr_patience,
                                                         min_lr=building_reduce_lr_min_lr,
                                                         cooldown=building_reduce_lr_cooldown)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=building_early_stopping_patience,
                                                          min_delta=building_early_stopping_min_delta,
                                                          restore_best_weights=True)
        callbacks.extend([early_stopping, reduce_lr, CustomCallback(tuning=tuning)])


    # Modellaufbau
    print("Shape of X_train:", X_train.shape)



    model = Sequential()
    model.add(LSTM(n_units, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=n_layers > 1))
    model.add(Dropout(dropout_rate))
    for _ in range(1, n_layers):
        model.add(LSTM(n_units, activation='tanh', return_sequences=True))
        model.add(Dropout(dropout_rate))

    model.add(Dense(Y_train.shape[1], activation='softmax'))  # Anzahl der Klassen

    optimizer = tf.keras.optimizers.Adam(learning_rate=tuning_adam_learning_rate)  # Standard-Lernrate verwenden
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    # Training
    # model.fit(X_train, Y_train, epochs=100, batch_size=batch_size, validation_data=(X_test, Y_test), callbacks=callbacks, verbose=verbose)

    if tuning:
        model.fit(X_train, Y_train, epochs=999999, batch_size=batch_size, validation_data=(X_test, Y_test), callbacks=callbacks, verbose=1)

        loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
        return accuracy  # oder eine andere Metrik, je nach Anforderung
    else:
        model.fit(X_train, Y_train, epochs=999999, batch_size=batch_size, callbacks=callbacks, verbose=1)

        return model


    # Evaluation
    # loss = model.evaluate(X_test, Y_test, verbose=0)
    # return loss


# Step 4
########################################################################################################################
def build_model_v3(database_name=None, database_name_optuna=None, table_name=None, show_progression=False, verbose=False):

    if verbose: print(f'load_data()')
    train_df = load_data(database_name=database_name, table_name=table_name)

    print(train_df.head())

    if verbose: print(f'df_to_dataset()...')
    # train_ds = df_to_dataset(train_df)


    if verbose: print(f'Preparing data for training')
    X_train, Y_train, scaler_train, encoder_train = prepare_data(train_df, target_column_name='target_category', history_points=12)


    if verbose: print(f'load_study')
    study = load_study(database_name_optuna)
    # best_params = study.best_params

    valid_trials = [trial for trial in study.trials if trial.value is not None]
    sorted_trials = sorted(valid_trials, key=lambda trial: trial.value, reverse=False)
    best_params = sorted_trials[1].params
    print(f'best_params : {best_params}')


    if verbose: print(f'train_or_tune_model_v3()...')
    model = train_or_tune_model_v3(best_params=best_params, X_train=X_train, Y_train=Y_train, tuning=False, verbose=0, show_progression=show_progression)

    # Setze das Arbeitsverzeichnis auf das Verzeichnis, in dem sich das Skript befindet
    os.chdir(os.path.dirname(__file__))
    # Speichern des trainierten Modells

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    model_save_path = f"saved_models/nn_model_{database_name_optuna}"
    model.save(model_save_path)

    dump(scaler_train, f"saved_models/nn_model_{database_name_optuna}/scaler.joblib")
    dump(encoder_train, f"saved_models/nn_model_{database_name_optuna}/encoder.joblib")

    # with open(f'saved_models/vectorizer_{model_name}.pkl', 'wb') as f:
    #     pickle.dump(fitted_vectorizer_tfidf, f)



# Step 2
########################################################################################################################



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

y_target_data = "target_category"
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
    database_name_v3 = "bot_v4_test1"  # nlayers 1-1

    table_name = "nasdq_5min_dataset"
    workers = 1
    train_model_v3(n_trials=workers*1, n_jobs=workers, database_name=database_name, database_name_optuna=database_name_v3, table_name=table_name, show_progression=False, verbose=True)
    # show_study_params(database_name=database_name_v3, n_last_trials=5)
    # show_study(database_name=database_name_v3)
    # build_model_v3(database_name=database_name_v3, table_name=table_name, model_name=database_name_v3, show_progression=True)
    
    # predict_v3(vectorize=vectorize, model_name=database_name_v3, brand="ktm", mode="random", number_of_predicts=100, matrix_predict=True, price_quality_threshold=0, string_match_threshold=0.92)
    # mode[km , random, lead]


    end_total = time.time()
    print(f'Total: {round(end_total - start_total, ndigits=2)} Seconds')



