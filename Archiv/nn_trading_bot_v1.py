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
import json

from sqlalchemy import create_engine
from difflib import SequenceMatcher
from tqdm import tqdm
import traceback
import matplotlib.pyplot as plt
import time
import datetime
import math
import re
import pickle


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




def df_to_dataset(dataframe, label_indexer, label_encoder):
    df = dataframe.copy()
    labels = df.pop(y_target_data)

    # Anwendung der vorhandenen Kodierer
    indexed_labels = label_indexer(labels)
    labels_tensor = tf.convert_to_tensor(indexed_labels, dtype=tf.int64)
    one_hot_labels = label_encoder(labels_tensor)

    df_dict = {key: value.to_numpy()[:, np.newaxis] for key, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices((df_dict, one_hot_labels))
    ds = ds.prefetch(1)
    return ds


def df_to_dataset_predict(dataframe):
    df = dataframe.copy()

    df_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in df.items()}
    ds = tf.data.Dataset.from_tensor_slices(df_dict)
    ds = ds.prefetch(1)
    return ds


def load_data(database_name, table_name):

    source_config = f'mysql+mysqlconnector://{connection_config["user"]}:{connection_config["password"]}@{connection_config["host"]}/{database_name}'
    engine = create_engine(source_config, echo=False)

    dataframe = pd.read_sql(f'SELECT * FROM {table_name}', con=engine)

    print("Entferne die Spalte daily_open_diff")
    columns_to_drop = ['daily_open_diff']
    columns_to_drop = [col for col in columns_to_drop if col in dataframe.columns]
    dataframe = dataframe.drop(columns_to_drop, axis=1)


    print(dataframe.head())
    print(f'dataframe_len:{len(dataframe)}')

    dataframe.dropna(inplace=True)


    dataframe = dataframe.loc[dataframe["target_category"] != "kein_ereignis"]

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
    study = optuna.create_study(study_name=study_name_str, storage=storage_name_str, direction="maximize", load_if_exists=True)

    try:
        print(f'best_params : {study.best_params}')
        print(f'best_value : {study.best_value}')
    except:
        print(f'best_params : no Data yet')
        print(f'best_value : no Data yet')

    return study


# ======================================================================================================================
# V_3
# ======================================================================================================================
# Step 1
########################################################################################################################

def train_model_v3(tuning=None, n_trials=None, n_jobs=None, database_name=None, table_name=None, database_name_optuna=None, show_progression=False, verbose=False):

    if verbose: print(f'load_data()')
    db_wkdm = load_data(database_name=database_name, table_name=table_name)

    # db_wkdm = db_wkdm[:10]


    # Initialisieren der Kodierer
    label_indexer = tf.keras.layers.StringLookup()
    label_indexer.adapt(db_wkdm[y_target_data])
    label_encoder = tf.keras.layers.CategoryEncoding(num_tokens=label_indexer.vocabulary_size(), output_mode='one_hot')

    if tuning:
        if verbose: print(f'split_data_v3')
        train_df, test_df = split_data_v3(db_wkdm)

        if verbose: print(f'transform_data_train_ds')
        train_ds = df_to_dataset(train_df, label_indexer, label_encoder)

        if verbose: print(f'transform_data_test_ds')
        test_ds = df_to_dataset(test_df, label_indexer, label_encoder)

        all_inputs, all_features = preprocess_data_v3(train_df=train_df, train_ds=train_ds, verbose=verbose)

        print(f'all_features shape: {all_features.shape}')
        calculate_neurons_and_print_code(all_features.shape)

        if verbose: print(f'load_study')
        study = load_study(database_name_optuna)
        best_params = None

        if verbose: print(f'build_objective_with_data')
        objective_with_data = functools.partial(train_or_tune_model_v3, train_ds=train_ds, all_inputs=all_inputs, all_features=all_features, best_params=best_params, test_ds=test_ds, tuning=tuning, verbose=verbose, show_progression=show_progression, label_indexer=label_indexer)

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
        train_ds = df_to_dataset(db_wkdm, label_indexer, label_encoder)
        all_inputs, all_features = preprocess_data_v3(train_df=db_wkdm, train_ds=train_ds, verbose=verbose)

        study = load_study(database_name_optuna)
        best_params = study.best_params
        best_loss = study.best_value

        print(f"Best hyperparameters: {best_params}")
        print(f"Best loss: {best_loss}")

        if verbose: print(f'train_or_tune_model_v3()...')
        # model = train_or_tune_model_v3(best_params=best_params, train_ds=train_ds, all_inputs=all_inputs, all_features=all_features, verbose=2, tuning=False, show_progression=show_progression)
        model = train_or_tune_model_v3(best_params=best_params, train_ds=train_ds, all_inputs=all_inputs, all_features=all_features, tuning=tuning, verbose=verbose, show_progression=show_progression, label_indexer=label_indexer)

        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")

        model_save_path = f"saved_models/nn_model_{database_name_optuna}"
        model.save(model_save_path)

        label_indexer_vocab = label_indexer.get_vocabulary()
        with open(f"saved_models/nn_model_{database_name_optuna}/label_indexer_vocab.json", "w") as f:
            json.dump(label_indexer_vocab, f)

        # Speichern der Konfiguration von label_encoder
        label_encoder_config = label_encoder.get_config()
        with open(f"saved_models/nn_model_{database_name_optuna}/label_encoder_config.json", "w") as f:
            json.dump(label_encoder_config, f)


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


    numerical_cols = train_df.loc[:, ~train_df.columns.str.contains('target_category')].columns.tolist()

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
def train_or_tune_model_v3(trial=None, train_ds=None, all_inputs=None, all_features=None, best_params=None, tune_with_best=None, test_ds=None, tuning=True, verbose=0, show_progression=False, label_indexer=None):

    try:
        # Hyperparameter space
        if tuning:

            if not tune_with_best:
                #norm TODO
                min_neurons = int(math.sqrt(all_features.shape[1]))
                max_neurons = all_features.shape[1]
                print(f"n_units = trial.suggest_int('n_units', {min_neurons}, {max_neurons}, log=False)")

                activation = trial.suggest_categorical('activation', ['relu', 'softmax'])
                # n_layers = trial.suggest_int('n_layers', n_layers_range_from, n_layers_range_to)
                n_layers = trial.suggest_int('n_layers', 1, 6)
                n_units = trial.suggest_int('n_units', min_neurons, max_neurons, log=False)
                # n_units = trial.suggest_int('n_units', min_neurons, 100, log=False)

                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.9, log=False)
                batch_size = trial.suggest_int('batch_size', 8, 1000, log=False)
                optimizer = trial.suggest_categorical('optimizer', ['adam'])
                if optimizer == 'adam':
                    optimizer = tf.keras.optimizers.Adam(learning_rate=tuning_adam_learning_rate)

            else:
                activation = trial.suggest_categorical('activation', ['relu'])
                n_layers = trial.suggest_int('n_layers', best_params['n_layers'], best_params['n_layers'])
                n_units = trial.suggest_int('n_units', best_params['n_units'], best_params['n_units'], log=False)
                dropout_rate = trial.suggest_float('dropout_rate', best_params['dropout_rate'], best_params['dropout_rate'], log=False)
                batch_size = trial.suggest_int('batch_size', best_params['batch_size'], best_params['batch_size'], log=False)
                optimizer = trial.suggest_categorical('optimizer', ['adam'])
                if optimizer == 'adam':
                    optimizer = tf.keras.optimizers.Adam(learning_rate=tuning_adam_learning_rate)

                # n_layers = best_params['n_layers']
                # n_units = best_params['n_units']
                # activation = best_params['activation']
                # optimizer = tf.keras.optimizers.Adam(learning_rate=tuning_adam_learning_rate)
                # dropout_rate = best_params['dropout_rate']
                # batch_size = best_params['batch_size']

        else:
            n_layers = best_params['n_layers']
            n_units = best_params['n_units']
            activation = best_params['activation']
            optimizer = tf.keras.optimizers.Adam(learning_rate=building_adam_learning_rate)
            dropout_rate = best_params['dropout_rate']
            batch_size = best_params['batch_size']

        print(f'n_layers:{n_layers},n_units:{n_units},dropout_rate:{dropout_rate},batch_size:{batch_size}')


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
                    print(f"Epoch: {epoch + 1:2d}, accuracy: {logs['accuracy'] / scale_value:15.3f}, Val accuracy: {logs.get('val_accuracy', 'N/A') / scale_value:15.3f}, Zeit: {epoch_time:6.2f} Sek., LR: {current_lr:.2e}")
                else:
                    print(f"Epoch: {epoch + 1:2d}, accuracy: {logs['accuracy'] / scale_value:15.3f}, Zeit: {epoch_time:6.2f} Sek., LR: {current_lr:.2e}")


        callbacks = []
        if tuning:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=tuning_reduce_lr_factor, min_delta=tuning_reduce_lr_min_delta,
                                          patience=tuning_reduce_lr_patience, min_lr=tuning_reduce_lr_min_lr, cooldown=tuning_reduce_lr_cooldown)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=tuning_early_stopping_patience,
                                           min_delta=tuning_early_stopping_min_delta, restore_best_weights=True)

            if not tune_with_best:
                pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')
                callbacks.extend([early_stopping, reduce_lr, pruning_callback, CustomCallback(tuning=tuning)])
            else:
                callbacks.extend([early_stopping, reduce_lr, CustomCallback(tuning=tuning)])

        else:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=building_reduce_lr_factor, min_delta=building_reduce_lr_min_delta,
                                          patience=building_reduce_lr_patience, min_lr=building_reduce_lr_min_lr, cooldown=building_reduce_lr_cooldown)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=building_early_stopping_patience,
                                           min_delta=building_early_stopping_min_delta, restore_best_weights=True)
            callbacks.extend([early_stopping, reduce_lr, CustomCallback(tuning=tuning)])


        num_classes = label_indexer.vocabulary_size()  # Stellen Sie sicher, dass dies die korrekte Anzahl der Klassen widerspiegelt.


        x = all_features
        #############################################
        for _ in range(n_layers):
            x = tf.keras.layers.Dense(n_units, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        # outputs = tf.keras.layers.Dense(1)(x)
        # outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        outputs = tf.keras.layers.Dense(label_indexer.vocabulary_size(), activation=activation)(x)

        #############################################

        model = tf.keras.Model(inputs=all_inputs, outputs=outputs)
        # model.compile(optimizer=optimizer, loss='mse', weighted_metrics=[])
        # model.compile(optimizer=optimizer, loss='accuracy', weighted_metrics=[])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        train_ds = train_ds.batch(batch_size)
        if tuning:
            test_ds = test_ds.batch(batch_size)  # Falls Sie einen Test-DS haben

        #TODO
        fit_params = {
            'x': train_ds,
            'epochs': 99999,
            'verbose': verbose,
            'callbacks': callbacks,
            'batch_size': batch_size,
            'workers': 1,
            # 'use_multiprocessing': True
        }

        try:

            # Train the model
            if tuning:
                fit_params['validation_data'] = test_ds
                if show_progression:
                    history = model.fit(**fit_params)  # Speichern Sie die Trainingshistorie

                    try:
                        # Plot accuracy und val_accuracy
                        plt.plot(history.history['accuracy'], label='Training accuracy')
                        plt.plot(history.history['val_accuracy'], label='Validation accuracy')

                        # Hinzufügen von Titel und Beschriftungen
                        plt.title('Training and Validation accuracy Over Epochs')
                        plt.ylabel('accuracy')
                        plt.xlabel('Epoch')

                        # Legende anzeigen
                        plt.legend()

                        # Zeigen Sie das Diagramm an
                        plt.show()
                    except:
                        pass
                else:
                    model.fit(**fit_params)  # Speichern Sie die Trainingshistorie


                if test_ds:
                    loss, accuracy = model.evaluate(test_ds, verbose=verbose)
                    return accuracy  # Rückgabe nur des Verlustes
            else:
                if show_progression:
                    history = model.fit(**fit_params)  # Speichern Sie die Trainingshistorie

                    try:
                        # Plot the validation accuracy
                        plt.plot(history.history['accuracy'])
                        plt.title('Validation accuracy Over Epochs')
                        plt.ylabel('accuracy')
                        plt.xlabel('Epoch')
                        plt.show()
                    except:
                        pass
                else:
                    model.fit(**fit_params)  # Speichern Sie die Trainingshistorie


        except optuna.exceptions.TrialPruned as e:
            # print(traceback.format_exc())
            print(f"Trial pruned: {e} with params : {{'dropout_rate': {dropout_rate}, 'activation': '{activation}', 'optimizer': 'adam', 'n_layers': {n_layers}, 'n_units': {n_units}, 'batch_size': {batch_size}}}")
            return float('-inf')  # oder was immer Optuna erwartet, um die Beschneidung zu signalisieren

        except AttributeError as e:
            # Handle the AttributeError
            print(traceback.format_exc())
            print(f"Es trat ein Fehler auf: {e}")
            return float('-inf')  # oder passende Fehlerbehandlung

        except Exception as e:
            # Fängt alle anderen Ausnahmen, die nicht spezifisch abgefangen wurden
            print(traceback.format_exc())
            print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
            return float('-inf')  # oder passende Fehlerbehandlung

        if not tune_with_best:
            return model
    except Exception:
        print(traceback.format_exc())



def load_model_and_predict(input_df, database_name_optuna, additional_cols=None):
    """
    Lädt ein trainiertes Modell und die zugehörigen Label-Kodierer, bereitet den DataFrame zeilenweise vor und macht Vorhersagen.

    Args:
        input_df (pd.DataFrame): DataFrame, der die Eingabedaten für Vorhersagen enthält.
        database_name_optuna (str): Eindeutiger Bezeichner für das Modell und die Präprozessoren.
        additional_cols (list): Liste von Spaltennamen, die vor der Vorhersage entfernt und danach wieder angefügt werden.

    Returns:
        pd.DataFrame: DataFrame mit den Vorhersagen und den zusätzlichen Spalten.
    """
    # Laden des Modells
    model_save_path = f"saved_models/nn_model_{database_name_optuna}"
    model = tf.keras.models.load_model(model_save_path)

    # Laden der Label-Kodierer
    with open(f"{model_save_path}/label_indexer_vocab.json", "r") as f:
        label_indexer_vocab = json.load(f)
    label_indexer = tf.keras.layers.StringLookup(vocabulary=label_indexer_vocab, mask_token=None, invert=True)

    # Entfernen zusätzlicher Spalten und Speichern für später
    if additional_cols:
        additional_data = input_df[additional_cols]
        input_df = input_df.drop(columns=additional_cols)
    else:
        additional_data = None

    # Vorhersagen für jede Zeile einzeln durchführen
    predicted_labels = []
    for index, row in tqdm(input_df.iterrows(), total=len(input_df)):
        input_dict = {name: np.array([value]) for name, value in row.items()}
        prediction = model.predict(input_dict, verbose=0)
        predicted_index = np.argmax(prediction, axis=1)
        predicted_label = label_indexer(predicted_index).numpy()
        predicted_labels.append(predicted_label)

    # Erstellen eines Ergebnis-DataFrames
    results_df = pd.DataFrame(predicted_labels, columns=['target_category'])
    results_df['target_category'] = results_df['target_category'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    if additional_data is not None:
        results_df = pd.concat([results_df, additional_data.reset_index(drop=True)], axis=1)

    return results_df


def load_model_and_predict2(input_df, database_name_optuna, additional_cols=None):
    """
    Lädt ein trainiertes Modell und die zugehörigen Label-Kodierer, bereitet den DataFrame zeilenweise vor und macht Vorhersagen.

    Args:
        input_df (pd.DataFrame): DataFrame, der die Eingabedaten für Vorhersagen enthält.
        database_name_optuna (str): Eindeutiger Bezeichner für das Modell und die Präprozessoren.
        additional_cols (list): Liste von Spaltennamen, die vor der Vorhersage entfernt und danach wieder angefügt werden.

    Returns:
        pd.DataFrame: DataFrame mit den Vorhersagen und den zusätzlichen Spalten.
    """
    # Laden des Modells
    model_save_path = f"saved_models/nn_model_{database_name_optuna}"
    model = tf.keras.models.load_model(model_save_path)

    # Laden der Label-Kodierer
    with open(f"{model_save_path}/label_indexer_vocab.json", "r") as f:
        label_indexer_vocab = json.load(f)
    label_indexer = tf.keras.layers.StringLookup(vocabulary=label_indexer_vocab, mask_token=None, invert=True)

    # Entfernen zusätzlicher Spalten und Speichern für später
    if additional_cols:
        additional_data = input_df[additional_cols]
        input_df = input_df.drop(columns=additional_cols)
    else:
        additional_data = None

    # Verwenden der Vorverarbeitungsfunktion
    input_dataset = df_to_dataset_predict(input_df)

    # Vorhersagen machen
    predictions = model.predict(input_dataset.batch(1), verbose=1)
    predicted_indices = np.argmax(predictions, axis=1)

    # Verwendung des label_indexer, um die vorhergesagten Indizes in lesbare Labels umzuwandeln
    predicted_labels = label_indexer(predicted_indices).numpy()

    # Erstellen eines Ergebnis-DataFrames
    results_df = pd.DataFrame(predicted_labels, columns=['target_category'])
    results_df['target_category'] = results_df['target_category'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    if additional_data is not None:
        results_df = pd.concat([results_df, additional_data.reset_index(drop=True)], axis=1)

    return results_df

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


    # database_name = "trading_bot"
    # # database_name_v3 = f"db_test{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # database_name_v3 = "trading_bot_test"
    # database_name_v3 = f'{database_name_v3}13'  # threshold_change=50, time_window=60
    #
    # table_dataset = "nasdq_5min_dataset_db"
    # workers = 1
    # # nn3.train_model_v3(n_trials=workers * 1, n_jobs=workers, database_name=database_name, database_name_optuna=database_name_v3, table_name=table_dataset, show_progression=False, verbose=True)
    # # nn.build_model_v3(database_name=database_name, table_name=table_dataset, database_name_optuna=database_name_v3, show_progression=True)
    # # # exit()
    #
    # # nn1.train_model_v3(tuning=True, n_trials=workers * 1, n_jobs=workers, database_name=database_name, table_name=table_dataset, database_name_optuna=database_name_v3, show_progression=False, verbose=2)
    # train_model_v3(tuning=False, n_trials=workers * 1, n_jobs=workers, database_name=database_name, table_name=table_dataset, database_name_optuna=database_name_v3, show_progression=True, verbose=2)
    #
    # # show_study_params(database_name=database_name_v3, n_last_trials=5)
    # # show_study(database_name=database_name_v3)
    # build_model_v3(database_name=database_name_v3, table_name=table_name, model_name=database_name_v3, show_progression=True)
    
    # predict_v3(vectorize=vectorize, model_name=database_name_v3, brand="ktm", mode="random", number_of_predicts=100, matrix_predict=True, price_quality_threshold=0, string_match_threshold=0.92)
    # mode[km , random, lead]


    end_total = time.time()
    print(f'Total: {round(end_total - start_total, ndigits=2)} Seconds')



