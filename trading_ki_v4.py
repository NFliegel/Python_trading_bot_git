# Standardbibliotheken
import os
import sys
import random
import re
import math
import time
import logging
import traceback
import threading
import functools
from datetime import datetime
import psutil
import shutil
from multiprocessing import Pool
from statistics import median

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, handlers=[
    logging.FileHandler("debug.log"),
    logging.StreamHandler()
])

cores = psutil.cpu_count(logical=True)


# Umgebungsvariablen für TensorFlow und Umgebung setzen (vor dem Import von TensorFlow!)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ''    # GPU deaktivieren
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_ENABLE_MKL_NATIVE_FORMAT'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = str(cores)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(cores)

os.environ['OMP_NUM_THREADS'] = str(cores)
os.environ['KMP_BLOCKTIME'] = '0'
os.environ["MKL_NUM_THREADS"] = str(cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(cores)

# Datenverarbeitung und Analyse
import numpy as np
import pandas as pd
import ta
import joblib
import pickle

# Datenbanken und ORM
import sqlalchemy
# from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR, Float, Integer, DateTime
import mysql.connector
from mysql.connector import Error

# Maschinelles Lernen und Optimierung
import optuna
from optuna.study import StudyDirection
from optuna.samplers import TPESampler
# from optuna.pruners import MedianPruner
from optuna.integration import TFKerasPruningCallback

# from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import Lasso
# from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import linregress
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator, WilliamsRIndicator, StochRSIIndicator

# TensorFlow und Erweiterungen (nach dem Setzen der Env-Variablen!)
import tensorflow as tf
# print(tf.config.list_physical_devices('CPU'))

from tensorflow.keras import backend as K
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import Layer, Lambda
import tensorflow_addons as tfa

# Weitere Bibliotheken
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
# import mpld3


from tqdm import tqdm
import yfinance as yf
import faulthandler
import subprocess
# import multiprocessing
# from sqlalchemy.orm import scoped_session, sessionmaker

# CPU- und Thread-Konfiguration
physical_cores = psutil.cpu_count(logical=False)
logical_cores = psutil.cpu_count(logical=True)

tf.config.threading.set_intra_op_parallelism_threads(physical_cores)
tf.config.threading.set_inter_op_parallelism_threads(logical_cores)

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    # histogram_freq=1,
    profile_batch=2
)


# from tensorflow.python.profiler import profiler_v2

# Dein Trainingscode hier


# Seed für Reproduzierbarkeit setzen
def set_seed(seed=42):
    """
    Setzt Samen für Python, NumPy und TensorFlow, um die Reproduzierbarkeit zu gewährleisten.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Begrenzen der Threads (optional, aber empfohlen für deterministische Ergebnisse)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

set_seed(42)

# Pandas-Anzeigeoptionen setzen
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

# Datenbankkonfiguration
connection_config = {
    'database': 'trading_bot',
    'host': 'localhost',
    'user': 'root',
    'password': '',
}

# Windows-spezifische CPU-Last anzeigen
# os.system("wmic cpu get loadpercentage")

# Programm beenden
# exit()




# SETTINGSDATA
#  =====================================================================================================================

y_target_data = "Trend"
# sample_weight_column = "price_quality"
categorical_str_cols_v3 = []
categorial_int_cols_v3 = []
numerical_cols_v3 = []

# SETTINGSDATA
#  =====================================================================================================================


# SETTINGSTRAIN
#  =====================================================================================================================
#  Initialize the callbacks
#  ReduceLROnPlateau

# ReduceLROnPlateau
tuning_reduce_lr_factor = 0.5  # Lernrate halbieren
tuning_reduce_lr_patience = 10  # Geduld von 10 Epochen
tuning_reduce_lr_min_delta = 0.001  # Minimaler Delta für Verbesserung
tuning_reduce_lr_min_lr = 1e-6  # Mindestlernrate
tuning_reduce_lr_cooldown = 5  # Cooldown von 5 Epochen

# EarlyStopping
tuning_early_stopping_patience = 20  # Geduld von 20 Epochen
tuning_early_stopping_min_delta = 0.0  # Minimaler Delta für Verbesserung

# SETTINGSTRAIN
#  =====================================================================================================================



# SETTINGSBUILD
#  =====================================================================================================================
#  Initialize the callbacks

# ReduceLROnPlateau
building_reduce_lr_factor = 0.5  # Lernrate halbieren
building_reduce_lr_patience = 30  # Geduld von 30 Epochen
building_reduce_lr_min_delta = 0.001  # Minimaler Delta für Verbesserung
building_reduce_lr_min_lr = 1e-6  # Mindestlernrate
building_reduce_lr_cooldown = 10  # Cooldown von 10 Epochen

# EarlyStopping
building_early_stopping_patience = 60  # Geduld von 60 Epochen
building_early_stopping_min_delta = 0.0  # Minimaler Delta für Verbesserung


# SETTINGSBUILD
#  =====================================================================================================================


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


def load_study(database_name_optuna, direction):

    storage_name_str = f"mysql+mysqlconnector://{connection_config['user']}:{connection_config['password']}@{connection_config['host']}/optuna_{database_name_optuna}"
    # storage_name = f"mysql+mysqlconnector://{connection_config['user']}:{connection_config['password']}@{connection_config['host']}:{connection_config['port']}/{database_name}"
    study_name_str = "study"

    create_database(host=connection_config['host'], user=connection_config['user'], password=connection_config['password'], db_name=f'optuna_{database_name_optuna}')
    # study = optuna.create_study(study_name=study_name_str, storage=storage_name_str, direction="minimize", load_if_exists=True)
    # study = optuna.create_study(study_name=study_name_str, storage=storage_name_str, direction="maximize", load_if_exists=True)
    # study = optuna.create_study(study_name=study_name_str, storage=storage_name_str, direction=direction, load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=42))

    if isinstance(direction, list):
        directions = [StudyDirection.MAXIMIZE if d == "maximize" else StudyDirection.MINIMIZE for d in direction]
        study = optuna.create_study(study_name=study_name_str, storage=storage_name_str, directions=directions, load_if_exists=True, sampler=TPESampler())

        try:
            print("Pareto-optimal Solutions:")
            for i, t in enumerate(study.best_trials):
                print(f"  Trial #{t.number}: values={t.values}, params={t.params}")

        except:
            print(f'best_params : no Data yet')
            print(f'best_value : no Data yet')



    else:
        study = optuna.create_study(study_name=study_name_str, storage=storage_name_str, direction=direction, load_if_exists=True, sampler=TPESampler())

        try:
            print(f'best_params : {study.best_params}')
            print(f'best_value : {study.best_value}')
        except:
            print(f'best_params : no Data yet')
            print(f'best_value : no Data yet')

    return study


def load_study_multy(database_name_optuna, direction):

    storage_name_str = f"mysql+mysqlconnector://{connection_config['user']}:{connection_config['password']}@{connection_config['host']}/optuna_{database_name_optuna}"
    study_name_str = "study"

    create_database(host=connection_config['host'], user=connection_config['user'], password=connection_config['password'], db_name=f'optuna_{database_name_optuna}')

    if isinstance(direction, list):
        directions = [StudyDirection.MAXIMIZE if d == "maximize" else StudyDirection.MINIMIZE for d in direction]
        study = optuna.create_study(study_name=study_name_str, storage=storage_name_str, directions=directions, load_if_exists=True, sampler=TPESampler())
    else:
        study = optuna.create_study(study_name=study_name_str, storage=storage_name_str, direction=direction, load_if_exists=True, sampler=TPESampler())

    try:
        print(f'best_params : {study.best_params}')
        print(f'best_value : {study.best_value}')
    except:
        print(f'best_params : no Data yet')
        print(f'best_value : no Data yet')

    return study



def query_database1(db, table, columns=None, conditions=None, limit=None, engine_kwargs={}, query_kwargs={}):
    """
    Führt eine Abfrage auf einer Datenbanktabelle durch.

    :param db: Name der Datenbank
    :param table: Name der Tabelle
    :param columns: Liste der Spalten, die abgefragt werden sollen (Standard: alle)
    :param conditions: Bedingungen für die WHERE-Klausel (optional)
    :param limit: Maximale Anzahl der zurückgegebenen Zeilen (optional)
    :param engine_kwargs: Zusätzliche Argumente für sqlalchemy.create_engine
    :param query_kwargs: Zusätzliche Argumente für pd.read_sql
    :return: DataFrame mit den abgefragten Daten
    """
    # Erstellen der Verbindungs-URL für die Datenbank
    connection_url = f"mysql+mysqlconnector://root:@localhost/{db}"

    # Erstellen eines SQLAlchemy Engine-Objekts
    engine = sqlalchemy.create_engine(connection_url, **engine_kwargs)

    # Bestimmen, welche Spalten abgefragt werden sollen
    columns_str = ", ".join(columns) if columns else "*"

    # Aufbauen der Abfrage
    query = f"SELECT {columns_str} FROM {table}"
    if conditions:
        query += f" WHERE {conditions}"
    if limit:
        query += f" LIMIT {limit}"

    # Ausführen der Abfrage und Schließen der Verbindung
    with engine.connect() as conn:
        dataframe = pd.read_sql(query, con=conn, **query_kwargs)

    return dataframe

def query_database(
    db, table, columns=None, conditions=None, limit=None,
    order_column=None, order_direction="ASC", date_as_text=False, engine_kwargs={}, query_kwargs={}
):
    """
    Führt eine Abfrage auf einer Datenbanktabelle durch, mit Unterstützung für die Sortierung von Text-Datumswerten.

    :param db: Name der Datenbank
    :param table: Name der Tabelle
    :param columns: Liste der Spalten, die abgefragt werden sollen (Standard: alle)
    :param conditions: Bedingungen für die WHERE-Klausel (optional)
    :param limit: Maximale Anzahl der zurückgegebenen Zeilen (optional)
    :param order_column: Spalte, nach der sortiert werden soll (optional)
    :param order_direction: Sortierrichtung ("ASC" oder "DESC", Standard: "ASC")
    :param date_as_text: Gibt an, ob die Sortierung einer Datumsspalte als Text erfolgt (Standard: False)
    :param engine_kwargs: Zusätzliche Argumente für sqlalchemy.create_engine
    :param query_kwargs: Zusätzliche Argumente für pd.read_sql
    :return: DataFrame mit den abgefragten Daten
    """
    import pandas as pd
    import sqlalchemy

    # Erstellen der Verbindungs-URL für die Datenbank
    connection_url = f"mysql+mysqlconnector://root:@localhost/{db}"

    # Erstellen eines SQLAlchemy Engine-Objekts
    engine = sqlalchemy.create_engine(connection_url, **engine_kwargs)

    # Bestimmen, welche Spalten abgefragt werden sollen
    columns_str = ", ".join(columns) if columns else "*"

    # Aufbauen der Abfrage
    query = f"SELECT {columns_str} FROM {table}"
    if conditions:
        query += f" WHERE {conditions}"
    if order_column:
        if date_as_text:
            # Sortierung als echtes Datum behandeln
            query += f" ORDER BY STR_TO_DATE({order_column}, '%d/%m/%Y') {order_direction}"
        else:
            # Normale Sortierung
            query += f" ORDER BY {order_column} {order_direction}"
    if limit:
        query += f" LIMIT {limit}"

    # Ausführen der Abfrage und Schließen der Verbindung
    with engine.connect() as conn:
        dataframe = pd.read_sql(query, con=conn, **query_kwargs)

    return dataframe



def optimize_dataframe(df_optimized):
    """
    Optimiert die Datentypen eines DataFrames, um den Speicherverbrauch zu reduzieren.

    Parameters:
    df (pd.DataFrame): Der zu optimierende DataFrame.

    Returns:
    pd.DataFrame: Der optimierte DataFrame.
    """
    # df_optimized = df.copy()

    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype

        # if col_type == 'object':
        #     num_unique_values = len(df_optimized[col].unique())
        #     num_total_values = len(df_optimized[col])
        #     if num_unique_values / num_total_values < 0.5:
        #         df_optimized[col] = df_optimized[col].astype('category')

        if col_type == 'int64':
            if df_optimized[col].min() > np.iinfo(np.int32).min and df_optimized[col].max() < np.iinfo(np.int32).max:
                df_optimized[col] = df_optimized[col].astype(np.int32)

        elif col_type == 'float64':
            df_optimized[col] = df_optimized[col].astype(np.float32)

        elif col_type == 'int32':
            if df_optimized[col].min() > np.iinfo(np.int16).min and df_optimized[col].max() < np.iinfo(np.int16).max:
                df_optimized[col] = df_optimized[col].astype(np.int16)

        elif col_type == 'float32':
            if df_optimized[col].min() > np.finfo(np.float16).min and df_optimized[col].max() < np.finfo(np.float16).max:
                df_optimized[col] = df_optimized[col].astype(np.float16)

    return df_optimized



def fetch_stock_data(ticker, interval='5m', period='60d', output_currency='USD'):
    """
    Ruft Aktienkursdaten ab und gibt sie in der gewünschten Währung aus (USD oder EUR).

    Parameters:
    - ticker (str): Das Ticker-Symbol der Aktie.
    - interval (str): Das Intervall der Daten (z.B. '5m', '1d').
    - period (str): Der Zeitraum der Daten (z.B. '60d', '1mo').
    - output_currency (str): Die gewünschte Ausgabewährung ('USD' oder 'EUR').
    - from_table (optional): Zusätzlicher Parameter, falls benötigt.

    Returns:
    - pd.DataFrame: DataFrame mit den Aktienkursdaten in der gewünschten Währung.
    """

    # Validierung der Ausgabe-Währung
    if output_currency not in ['USD', 'EUR']:
        raise ValueError("Parameter 'output_currency' muss entweder 'USD' oder 'EUR' sein.")


    # Ausgabeparameter
    print(f"Abrufen der Daten für Ticker: {ticker}")

    # Validierung des Intervalls
    if interval != '5m':
        print(f"Warnung: Für diese Anwendung wird ein 5-Minuten-Intervall empfohlen. Das gewählte Intervall ist: {interval}")

    # Validierung des Zeitraums
    if period is None:
        raise ValueError("Parameter 'period' muss angegeben werden, wenn from_stock=True.")

    print(f"Daten mit Intervall '{interval}' und Periode '{period}' werden abgerufen.")

    # Abrufen der Aktienkursdaten von yfinance
    df_stock_orig = yf.download(tickers=ticker, period=period, interval=interval, progress=False, multi_level_index=False)

    if df_stock_orig.empty:
        raise ValueError(f"Keine Daten für Ticker '{ticker}' im Zeitraum '{period}' gefunden.")

    print(f"Daten für Ticker '{ticker}' erfolgreich abgerufen. Anzahl der Zeilen: {len(df_stock_orig)}")

    # Überprüfung und Zurücksetzen des Index, falls notwendig
    if isinstance(df_stock_orig.index, pd.DatetimeIndex):
        df_stock_orig = df_stock_orig.reset_index()
        df_stock_orig['Datetime'] = pd.to_datetime(df_stock_orig['Datetime'])
        print("Index 'Datetime' zurückgesetzt und in eine Spalte umgewandelt.")
    else:
        print("Warnung: 'Datetime' Index nicht gefunden oder kein DatetimeIndex. 'Datetime' Spalte wird nicht erstellt.")

    # Wenn die Ausgabe in EUR gewünscht ist, Umrechnung durchführen
    if output_currency == 'EUR':
        print("Umrechnung der Aktienkurse von USD nach EUR.")

        # Abrufen des EUR/USD-Wechselkurses
        eur_usd_ticker = "EURUSD=X"
        df_eur_usd = yf.download(tickers=eur_usd_ticker, period=period, interval=interval, progress=False, multi_level_index=False)

        if df_eur_usd.empty:
            raise ValueError(f"Keine Wechselkursdaten für Ticker '{eur_usd_ticker}' im Zeitraum '{period}' gefunden.")

        print(f"Wechselkursdaten für '{eur_usd_ticker}' erfolgreich abgerufen. Anzahl der Zeilen: {len(df_eur_usd)}")

        # Überprüfung und Zurücksetzen des Index für Wechselkursdaten
        if isinstance(df_eur_usd.index, pd.DatetimeIndex):
            df_eur_usd = df_eur_usd.reset_index()
            df_eur_usd['Datetime'] = pd.to_datetime(df_eur_usd['Datetime'])
        else:
            print("Warnung: 'Datetime' Index für Wechselkursdaten nicht gefunden oder kein DatetimeIndex.")

        # Zusammenführen der Aktien- und Wechselkursdaten basierend auf dem Datum und der Uhrzeit
        # Da Wechselkurse im '5m'-Intervall möglicherweise nicht verfügbar sind, werden wir auf Tagesbasis umrechnen
        # Daher extrahieren wir das Datum und mergen basierend auf dem Datum

        # Extrahieren des Datums aus der Datetime-Spalte
        df_stock_orig['Date'] = df_stock_orig['Datetime'].dt.date
        df_eur_usd['Date'] = df_eur_usd['Datetime'].dt.date

        # Aggregieren des Wechselkurses pro Tag (letzter verfügbaren Kurs des Tages)
        df_eur_usd_daily = df_eur_usd.groupby('Date')['Close'].last().reset_index().rename(columns={'Close': 'EURUSD'})

        # Zusammenführen der Aktien- und Wechselkursdaten basierend auf dem Datum
        df_combined = pd.merge(df_stock_orig, df_eur_usd_daily, on='Date', how='left')

        # Überprüfung auf fehlende Wechselkurse
        if df_combined['EURUSD'].isnull().any():
            print("Warnung: Einige Wechselkurse fehlen. Es wird versucht, fehlende Werte aufzufüllen.")
            df_combined['EURUSD'].fillna(method='ffill', inplace=True)

        # Umrechnung der Aktienkurse in EUR
        for col in ['Open', 'High', 'Low', 'Close']:
            df_combined[col] = df_combined[col] / df_combined['EURUSD']

        # Entfernen der zusätzlichen 'Date' und 'EURUSD' Spalten
        df_result = df_combined.drop(columns=['Date', 'EURUSD'])

        print("Umrechnung abgeschlossen. Aktienkurse sind jetzt in EUR.")

    else:
        # Ausgabe in USD, keine Umrechnung erforderlich
        df_result = df_stock_orig.copy()
        print("Ausgabe erfolgt in USD. Keine Umrechnung notwendig.")


    return df_result


def summarize_trend_distribution(df_stock_orig, trend_column='Trend', base_category='Stable'):
    """
    Analysiert die Verteilung der Kategorien in der angegebenen Trend-Spalte und zeigt
    die Anzahl, den Prozentsatz und das Verhältnis jeder Kategorie zur Basis-Kategorie an.

    Args:
        df_stock_orig (pd.DataFrame): DataFrame mit den historischen Aktienkursdaten. Muss eine Spalte für den Trend enthalten.
        trend_column (str, optional): Name der Spalte, die die Trend-Kategorien enthält. Standard ist 'Trend'.
        base_category (str, optional): Die Basis-Kategorie, zu der das Verhältnis berechnet wird. Standard ist 'Stable'.

    Returns:
        pd.DataFrame: Ein DataFrame mit den Spalten 'Count', 'Percentage (%)' und 'Ratio (1:x)' für jede Trend-Kategorie.
    """
    # Überprüfen, ob die Trend-Spalte vorhanden ist
    if trend_column not in df_stock_orig.columns:
        raise ValueError(f"Der DataFrame muss eine '{trend_column}'-Spalte enthalten.")

    # Berechnung der Counts und Prozentsätze
    trend_counts = df_stock_orig[trend_column].value_counts()
    trend_percentages = df_stock_orig[trend_column].value_counts(normalize=True) * 100

    # Erstellung des Zusammenfassungs-DataFrames
    trend_summary = pd.DataFrame({
        'Count': trend_counts,
        'Percentage (%)': trend_percentages.round(2)
    })

    # Auswahl der Basis-Kategorie für das Verhältnis
    if base_category not in trend_summary.index:
        raise ValueError(f"Die Basis-Kategorie '{base_category}' ist nicht in den Trend-Daten vorhanden.")

    # Basis-Kategorie Count
    base_count = trend_summary.loc[base_category, 'Count']

    # Funktion zur Berechnung des Verhältnisses im Format '1:x'
    def compute_ratio(row, base_count, base_category):
        if row.name == base_category:
            return "1:1"
        else:
            # Berechnung des Verhältnisses: 1:x, wobei x = base_count / current_count
            ratio = base_count / row['Count']
            # Um das Verhältnis als ganze Zahl darzustellen, runden wir auf die nächste ganze Zahl
            ratio = int(round(ratio))
            return f"1:{ratio}"

    # Hinzufügen der Ratio-Spalte zum Zusammenfassungs-DataFrame
    trend_summary['Ratio (1:x)'] = trend_summary.apply(
        lambda row: compute_ratio(row, base_count, base_category), axis=1
    )

    # Optional: Sortieren nach Count absteigend
    trend_summary = trend_summary.sort_values(by='Count', ascending=False)

    # Anzeige der Ergebnisse
    print("\nVerteilung der Kategorien in der Spalte 'Trend':")
    print(trend_summary)

    return trend_summary


def prepare_stock_data(
        database_name_optuna=None,
        analyze=False,
        use_create_lagged_features=False,
        db=None, save_in_db=False,
        cut_rows=None,
        trendfunc=None,
        test_size=None,
        db_lines=None,
        nutzungszeitraum=None,
        db_bereich=None,
        from_stock=False,
        from_table=None,
        to_table=None,
        future_steps=None,
        time_series_sequence_length=None,
        threshold_high=1,
        threshold_low=1,
        threshold_high_pct=0.000,
        threshold_low_pct=0.000,
        use_percentage=False,
        min_cum_return=0.000,
        df_stock_orig=None,
        test=False,
        lookback_steps=0,
        lookback_threshold=0.0,
        up_signal_mode="none",
        use_lookback_check=False,
        require_double_ups=False,
        offset_after_lowest=0,
        use_lookforward_check=False,
        look_forward_threshold=0.0,
        forward_steps=0,
        indicators=None,
        consecutive_negatives_lookback_steps=None,
        max_consecutive_negatives_lookback=None,
        consecutive_negatives_forward_steps=None,
        max_consecutive_negatives_forward=None,
        backwarts_shift_labels=0,
        consecutive_positives_lookback_steps=None,
        max_consecutive_positives_lookback=None,
        consecutive_positives_forward_steps=None,
        max_consecutive_positives_forward=None
        ):

# IMPORTANT
    """
    Bereitet die Aktienkursdaten vor, berechnet technische Indikatoren, erstellt Lagged Features und speichert oder analysiert die Daten.

    Parameter:
    - db: Datenbankverbindung
    - save_in_db: Speichern der Daten in der Datenbank
    - cut_rows: Anzahl der letzten Zeilen, die entfernt werden sollen
    - test_size: Anteil der Daten für Tests
    - from_stock: Datenquelle
    - from_table: Quelltabelle
    - to_table: Zieltabelle
    - history_steps: Historische Schritte für Indikatoren
    - future_steps: Zukünftige Schritte für Trendberechnung
    - time_series_sequence_length: Länge der Zeitreihen-Sequenz
    - signal_grenzwert: Schwellenwert für Signale
    - df_stock_orig: Ursprünglicher DataFrame
    - test: Testmodus
    - check_features: Feature-Analyse
    """

    # Daten laden
    if not from_stock:
        df_stock_orig = query_database1(db=db, table=from_table, limit=None)  # 351623
        # df_stock_orig = query_database1(db=db, table=from_table, limit=db_lines)
        # df_stock_orig = query_database(db=db, table=from_table, limit=db_lines, order_column="Date", order_direction="ASC")  # DESC ASC
        print("Daten aus der Datenbank geladen.")

        if db_lines and db_bereich and not df_stock_orig.empty:
            print("Daten Filtern mittels db_lines")
            # Begrenze db_lines auf die Länge des DataFrames
            db_lines = min(len(df_stock_orig), max(0, db_lines))
            print(f'db_lines:{db_lines}, df_stock_orig:{len(df_stock_orig)}')
            if db_bereich == "alte_daten":
                df_stock_orig = df_stock_orig[:db_lines].copy()  # Nimm die ersten db_lines Zeilen
            elif db_bereich == "neue_daten":
                df_stock_orig = df_stock_orig[-db_lines:].copy()  # Nimm die letzten db_lines Zeilen
            else:
                raise ValueError("Unbekannter db_bereich-Wert")

            df_stock_orig.reset_index(drop=True, inplace=True)

        else:
            if nutzungszeitraum:
                df_stock_orig = df_stock_orig[nutzungszeitraum[0]: nutzungszeitraum[1]].copy()
                df_stock_orig.reset_index(drop=True, inplace=True)
            else:
                print("Kein Nutzungszeitraum angegeben")
                exit()
    else:

        # ticker = "AAPL"  # Beispiel: Apple Inc.
        # ticker = "^NDX"
        ticker = "NQ=F"
        interval = '5m'  # 5-Minuten-Intervall
        period = '1mo'
        # output_currency = 'EUR'  # Gewünschte Währung: 'EUR' oder 'USD'
        output_currency = 'USD'  # Gewünschte Währung: 'EUR' oder 'USD'

        # Daten abrufen
        try:
            df_stock_orig = fetch_stock_data(ticker, interval, period, output_currency)
            print(df_stock_orig.head())
        except ValueError as e:
            print(f"Fehler: {e}")
            exit()

    # Technische Indikatoren berechnen
    print("Berechne technische Indikatoren...")
    df_stock_orig = set_indicators_2(
        test=test,
        df=df_stock_orig,
        trendfunc=trendfunc,
        future_steps=future_steps,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        lookback_steps=lookback_steps,
        lookback_threshold=lookback_threshold,
        threshold_high_pct=threshold_high_pct,
        threshold_low_pct=threshold_low_pct,
        min_cum_return=min_cum_return,
        use_percentage=use_percentage,
        database_name_optuna=database_name_optuna,
        up_signal_mode=up_signal_mode,
        use_lookback_check=use_lookback_check,
        require_double_ups=require_double_ups,
        offset_after_lowest=offset_after_lowest,
        use_lookforward_check=use_lookforward_check,
        look_forward_threshold=look_forward_threshold,
        forward_steps=forward_steps,
        indicators=indicators,
        consecutive_negatives_lookback_steps=consecutive_negatives_lookback_steps,
        max_consecutive_negatives_lookback=max_consecutive_negatives_lookback,
        consecutive_negatives_forward_steps=consecutive_negatives_forward_steps,
        max_consecutive_negatives_forward=max_consecutive_negatives_forward,
        backwarts_shift_labels=backwarts_shift_labels,
        consecutive_positives_lookback_steps=consecutive_positives_lookback_steps,
        max_consecutive_positives_lookback=max_consecutive_positives_lookback,
        consecutive_positives_forward_steps=consecutive_positives_forward_steps,
        max_consecutive_positives_forward=max_consecutive_positives_forward
        )
    # df_stock_orig = set_indicators_3(test=test, df=df_stock_orig, future_steps=future_steps, threshold_high=threshold_high, threshold_low=threshold_low)


    # Letzte Zeilen abschneiden, falls angegeben
    if cut_rows:
        print(f"Schneide die letzten {cut_rows} Zeilen ab...")
        print("Vor dem Schneiden:")
        print(df_stock_orig["Datetime"].tail())
        print(f'Datenlänge vor dem Schneiden: {len(df_stock_orig)}')

        df_stock_orig = df_stock_orig.iloc[:-cut_rows].copy()

        print("Nach dem Schneiden:")
        print(df_stock_orig["Datetime"].tail())
        print(f'Datenlänge nach dem Schneiden: {len(df_stock_orig)}')
    else:
        print("Keine Zeilen abgeschnitten.")

    print(f'Datenlänge nach dem Schneiden (falls durchgeführt): {len(df_stock_orig)}')

    # DataFrame optimieren
    print("Optimiere DataFrame...")
    df_stock_orig = optimize_dataframe(df_stock_orig)

    # Dynamisch Features auswählen
    # Definieren Sie Muster, die auf Indikatoren hinweisen
    #TODO

    # Bisher beste für lag und dazu alle aus der indicator erzeugung
    # Vorlage
    # indicator_patterns = [  # Für 400 bisher beste
    #     '^SMA',
    #     '^EMA',
    #     '^RSI',
    #     # '^MACD',
    #     '^BB',
    #     # '^OBV',
    #     '^ROC',
    #     # '^CCI',
    #     '^MFI',
    #     # 'Close_Diff',
    #     # 'High_Diff',
    #     # 'Low_Diff',
    #     'Volume',
    #     'Close_Pct_Change',
    #     # 'High_Diff_Pct',
    #     # 'Low_Diff_Pct',
    #     '^Slope',
    #     # '^Slope_',
    #     "^ATR",
    #     # "^ADX",
    #     # "^STOCH",  # Stochastic Oscillator
    #     # "^WILLR",  # Williams %R
    #     # "^STOCH_RSI",  # Stoch RSI (optional)
    # ]


    indicator_patterns = [  # test
        '^SMA',  #
        '^EMA',  #
        '^RSI',  #
        '^MACD',  #
        '^BB',  #
        '^OBV',
        '^ROC',  #
        '^CCI', #
        '^MFI',  #
        'Close_Diff',
        'High_Diff',
        'Low_Diff',
        'Volume',  #
        'Close_Pct_Change',
        'High_Diff_Pct',
        'Low_Diff_Pct',

        '^Slope_Close_Pct_Change',
        '^Slope_Close_',
        '^Slope_SMA',
        '^Slope_EMA',
        '^Slope_ROC_SMA',
        '^Slope_ROC_EMA',

        "^ATR",
        "^ADX",
        "^STOCH",
        "^WILLR",
        "^STOCH_RSI",
    ]


    # indicator_patterns = ['^SMA', '^RSI', '^BB', '^ROC', '^MFI', 'Volume', '^Slope_Close_Pct_Change', '^Slope_Close_', '^Slope_SMA', '^Slope_ROC_SMA', '^ATR']
    # train f1_score: 0.5439, F1-Score für 'Up': 0.0847

    # indicator_patterns = ['^SMA', '^RSI', '^BB', '^ROC', '^MFI', 'Volume', '^Slope_Close_Pct_Change', '^Slope_Close_', '^Slope_SMA', '^Slope_EMA', '^Slope_ROC_SMA', '^ATR']
    # train f1_score: 0.5429, F1-Score für 'Up': 0.0763

    # indicator_patterns = ['^SMA', '^RSI', '^BB', '^ROC', '^MFI', 'Volume', '^Slope_Close_Pct_Change', '^Slope_Close_', '^Slope_SMA', '^Slope_EMA', '^Slope_ROC_SMA', '^Slope_ROC_EMA', '^ATR']
    # train f1_score: 0.5435, F1-Score für 'Up': 0.0927

    # indicator_patterns = ['^SMA', '^RSI', '^BB', '^ROC', '^MFI', 'Volume', '^Slope_SMA', '^Slope_ROC_SMA', '^Slope_ROC_EMA', '^ATR']
    # train f1_score: 0.5399, F1-Score für 'Up': 0.0970

    # indicator_patterns = ['^SMA', '^RSI', '^BB', '^ROC', '^MFI', 'Volume', '^Slope_SMA', '^Slope_ROC_SMA', '^ATR']
    # train f1_score: 0.5438, F1-Score für 'Up': 0.0721

    # indicator_patterns = ['^SMA', '^EMA', '^RSI', '^BB', '^ROC', '^MFI', 'Volume', '^Slope_SMA', '^Slope_EMA', '^Slope_ROC_SMA', '^Slope_ROC_EMA', '^ATR']
    # train f1_score: 0.5493, F1-Score für 'Up': 0.0756

    # bisher benutzt
    # indicator_patterns = ['^SMA', '^EMA', '^RSI', '^BB', '^ROC', '^MFI', 'Volume', '^Slope_Close_Pct_Change', '^Slope_Close_', '^Slope_SMA', '^Slope_EMA', '^Slope_ROC_SMA', '^Slope_ROC_EMA', '^ATR']
    # train f1_score: 0.5508, F1-Score für 'Up': 0.0481

    # indicator_patterns = ['^SMA', '^EMA', '^RSI', '^MACD', '^BB', '^ROC', '^MFI', 'Volume', '^Slope_Close_Pct_Change', '^Slope_Close_', '^Slope_SMA', '^Slope_EMA', '^Slope_ROC_SMA', '^Slope_ROC_EMA', '^ATR']
    # train f1_score: 0.5483, F1-Score für 'Up': 0.0355

    # indicator_patterns = ['^SMA', '^EMA', '^RSI', '^BB', '^OBV', '^ROC', '^MFI', 'Volume', '^Slope_Close_Pct_Change', '^Slope_Close_', '^Slope_SMA', '^Slope_EMA', '^Slope_ROC_SMA', '^Slope_ROC_EMA', '^ATR']
    # train f1_score: 0.5472, F1-Score für 'Up': 0.0786

    # indicator_patterns = ['^SMA', '^EMA', '^RSI', '^MACD', '^BB', '^OBV', '^ROC', '^MFI', 'Volume', '^Slope_Close_Pct_Change', '^Slope_Close_', '^Slope_SMA', '^Slope_EMA', '^Slope_ROC_SMA', '^Slope_ROC_EMA', '^ATR']
    # train f1_score: 0.5447, F1-Score für 'Up': 0.0000

    # indicator_patterns = ['^SMA', '^EMA', '^RSI', '^BB', '^OBV', '^ROC', '^CCI', '^MFI', 'Volume', '^Slope_Close_Pct_Change', '^Slope_Close_', '^Slope_SMA', '^Slope_EMA', '^Slope_ROC_SMA', '^Slope_ROC_EMA', '^ATR']
    # train f1_score: 0.5479, F1-Score für 'Up': 0.0566

    # indicator_patterns = ['^SMA', '^EMA', '^RSI', '^BB', '^OBV', '^ROC', '^MFI', 'Volume', 'Close_Pct_Change', '^Slope_Close_Pct_Change', '^Slope_Close_', '^Slope_SMA', '^Slope_EMA', '^Slope_ROC_SMA', '^Slope_ROC_EMA', '^ATR']
    # train f1_score: 0.5488, F1-Score für 'Up': 0.0525

    print(f'indicator_patterns: \nindicator_patterns = {indicator_patterns}')


    # Kombinieren Sie die Muster zu einem einzigen regulären Ausdruck
    combined_pattern = '|'.join(indicator_patterns)

    # Schließen Sie nicht-indikatorische Spalten aus
    non_feature_columns = ['Date', 'Time', 'Datetime', 'Trend']

    # Wählen Sie alle Spalten, die den Indikatormustern entsprechen und nicht in non_feature_columns sind
    features = [
        col for col in df_stock_orig.columns
        if re.match(combined_pattern, col) and col not in non_feature_columns
    ]

    # Optional: Fügen Sie Basisspalten hinzu, die als Features dienen sollen
    base_features = [
        # 'Open',
        # 'High',
        # 'Low',
        # 'Close',
        # 'Volume',
        # 'Up',
        # 'Down'
    ]
    preserve_features = [
        # 'Close'
    ]
    features = base_features + features

    # Überprüfen, ob alle Features im DataFrame vorhanden sind
    missing_features = [feature for feature in features if feature not in df_stock_orig.columns]
    if missing_features:
        print(f"Warnung: Die folgenden Features fehlen im DataFrame und werden entfernt: {missing_features}")
        features = [feature for feature in features if feature in df_stock_orig.columns]

    # Lagged Features erstellen, falls nicht im Testmodus
    if use_create_lagged_features:
        if not test:
            print("Erstelle Lagged Features...")
            df_stock_orig = create_lagged_features(
                df_stock_orig,
                features=features,
                preserve_features=preserve_features,
                window_size=time_series_sequence_length
            )



    # Trend-Statistiken anzeigen
    unique_trend_count = df_stock_orig['Trend'].nunique()
    print("Anzahl der einzigartigen Kategorien in der Spalte 'Trend':")
    print(unique_trend_count)

    # trend_counts = df_stock_orig['Trend'].value_counts()
    # print("\nVerteilung der Kategorien in der Spalte 'Trend':")
    # print(trend_counts)

    # Analyse durchführen
    trend_summary = summarize_trend_distribution(
        df_stock_orig=df_stock_orig,
        trend_column='Trend',
        base_category='Stable'
    )


    if analyze and test:
        percentiles = []
        # percentiles = [5, 95]

        for i in range(1, 101, 1):
            percentiles.append(i)

        df_stock_orig, thresholds, significant_movements = analyze_stock_signals(
            df=df_stock_orig,
            future_window=future_steps,
            abs_threshold_up=threshold_high,
            abs_threshold_down=-threshold_high,
            percentiles=percentiles,
            plot=True,
            upper_percentile=93,
            lower_percentile=5,
            save_csv=True,
            output_csv_path='stock_analysis_with_signals.xlsx'
        )


    # Plotten im Testmodus
    if test:

        # Beispielausgabe der Signale
        # print("\nBeispiel der generierten Signale:")
        # print(df_stock_orig[['Close', 'Future_Close', 'Pct_Change', 'Signal']].head(10))


        x_interval_min = int(future_steps * 5)
        # x_interval_min = 1440
        # x_interval_min = 60

        print("Plotte Aktienkurse für Testmodus...")
        plot_stock_prices(
            df_stock_orig[-10000:],
            test=test,
            secondary_y_scale=1,
            x_interval_min=x_interval_min,
            y_interval_dollar=threshold_high,
            additional_lines=[
                # ['Slope_Close_Pct_Change10_lag_1', 1],
                # ['Slope_Close10_lag_1', 1],
                # ['SlopeEMA10_10_lag_1', 1],
                # ['ATR16', 1],
                # ['ATR32', 1],
                # ['ATR48', 1],
                # ['ATR64', 1],

            ],
            trend="Trend"
        )
        exit()


    df_stock_orig.dropna(inplace=True)

    # Daten speichern oder zurückgeben
    if not test:
        if save_in_db:
            print("Speichere Daten in der Datenbank...")
            list_of_tuning_columns = get_columns_by_mode(df=df_stock_orig, mode="training")
            df_to_save = df_stock_orig[["Datetime", "Close"] + list_of_tuning_columns]
            print("Letzte Zeilen der zu speichernden Daten:")
            print(df_to_save.tail())
            save_to_db(dataframe=df_to_save, to_table=to_table, db="trading_bot")
        else:
            print("Gebe den optimierten DataFrame zurück.")
            # exit()
            return df_stock_orig, indicator_patterns
    else:
        exit()



def calculate_neurons_and_print_code(features_shape):
    input_features = features_shape[1]  # Anzahl der Eingabefeatures
    output_units = 1  # Für Regressionsprobleme üblicherweise 1 Ausgabeeinheit

    # Regel 1: Geometrisches Mittel
    neurons_rule1 = int(math.sqrt(input_features * output_units))

    # Regel 2: Obergrenze
    neurons_rule2 = 2 * input_features

    print(f"n_units = trial.suggest_int('n_units', {neurons_rule1}, {min(neurons_rule2, 2000)}/{input_features})  # Passen Sie die Obergrenze nach Bedarf an")


def df_to_dataset(dataframe, feature_cols, target_cols, shuffle=True, batch_size=32):
    """
    Konvertiert einen Pandas DataFrame in ein TensorFlow Dataset mit Features als Dictionary.

    Args:
        dataframe (pd.DataFrame): Eingabe-DataFrame.
        feature_cols (list): Liste der Feature-Spalten.
        target_cols (list): Liste der Target-Spalten.
        shuffle (bool, optional): Ob die Daten gemischt werden sollen. Standard ist True.
        batch_size (int, optional): Größe der Batches. Standard ist 32.

    Returns:
        tf.data.Dataset: TensorFlow Dataset.
    """
    dataframe = dataframe.copy()
    labels = dataframe[target_cols].values
    features = {col: dataframe[col].values for col in feature_cols}
    ds = tf.data.Dataset.from_tensor_slices((features, labels))

    return ds



def df_to_dataset_stratified(dataframe, feature_cols, target_cols, batch_size=32, shuffle=True, seed=42):
    """
    Konvertiert einen Pandas DataFrame in ein TensorFlow Dataset mit stratifizierten Batches.

    Args:
        dataframe (pd.DataFrame): Eingabe-DataFrame.
        feature_cols (list): Liste der Feature-Spalten.
        target_cols (list): Liste der Target-Spalten.
        batch_size (int): Größe der Batches.
        shuffle (bool): Ob die Daten gemischt werden sollen.
        seed (int): Zufallssaat für das Mischen.

    Returns:
        tf.data.Dataset: TensorFlow Dataset.
    """
    # Features und Labels trennen
    X = dataframe[feature_cols].values
    y = dataframe[target_cols].values
    y_labels = np.argmax(y, axis=1)

    # DataFrame mit Features und Labels erstellen
    df = pd.DataFrame(X, columns=feature_cols)
    df['label'] = y_labels

    # DataFrame mischen
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Anzahl der Proben pro Klasse
    class_counts = df['label'].value_counts().sort_index()
    class_proportions = class_counts / class_counts.sum()
    samples_per_class_per_batch = (class_proportions * batch_size).astype(int)

    # Rundungsfehler korrigieren
    deficit = batch_size - samples_per_class_per_batch.sum()
    if deficit > 0:
        samples_per_class_per_batch.iloc[0] += deficit  # Defizit zur ersten Klasse hinzufügen
    elif deficit < 0:
        samples_per_class_per_batch.iloc[0] += deficit  # Überschuss von der ersten Klasse abziehen

    # Separate Datasets pro Klasse erstellen
    class_datasets = []
    for class_idx, count in class_counts.items():  # Ändere iteritems() zu items()
        class_df = df[df['label'] == class_idx]
        class_ds = tf.data.Dataset.from_tensor_slices((
            {col: class_df[col].values for col in feature_cols},
            class_df['label'].values
        ))
        if shuffle:
            class_ds = class_ds.shuffle(buffer_size=len(class_df), seed=seed)
        class_datasets.append(class_ds)

    # Batches pro Klasse erstellen
    batched_class_datasets = []
    for ds, count in zip(class_datasets, samples_per_class_per_batch):
        if count > 0:
            batched_ds = ds.batch(count)
            batched_class_datasets.append(batched_ds)

    # Die batched Datasets kombinieren
    zipped_ds = tf.data.Dataset.zip(tuple(batched_class_datasets))

    # Flache die gebündelten Batches
    def flatten_batch(*batch):
        features = {key: tf.concat([b[0][key] for b in batch], axis=0) for key in batch[0][0].keys()}
        labels = tf.concat([b[1] for b in batch], axis=0)
        return features, labels

    final_ds = zipped_ds.map(flatten_batch)

    # Prefetch für Performance
    final_ds = final_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return final_ds


def preprocess_and_balance_smoteenn(dataframe, target_col='Trend', random_state=42):
    """
    Balanciert die Klassen in einem DataFrame durch Kombination von SMOTE (Überabtastung)
    und ENN (Unterabtastung), wobei 'Trend' die Zielspalte ist und alle anderen Spalten als Features betrachtet werden.

    Args:
        dataframe (pd.DataFrame): Eingabe-DataFrame mit 'Trend' als Zielspalte und allen anderen Spalten als Features.
        target_col (str, optional): Name der Zielspalte. Standard ist 'Trend'.
        random_state (int, optional): Zufallszustand für die Reproduzierbarkeit. Standard ist 42.

    Returns:
        pd.DataFrame: Das resampelte DataFrame mit ausgeglichenen Klassen.
    """
    # Überprüfen, ob die Zielspalte vorhanden ist
    if target_col not in dataframe.columns:
        raise ValueError(f"Die Zielspalte '{target_col}' ist nicht im DataFrame vorhanden.")

    # Definiere die Feature- und Zielspalten
    feature_cols = [col for col in dataframe.columns if col != target_col]
    target_col = target_col

    # Extrahiere Features und Zielvariable
    X = dataframe[feature_cols].values
    y = dataframe[target_col].values

    # Initialisiere SMOTEENN
    smoteenn = SMOTEENN(random_state=random_state)

    # Wende SMOTEENN an, um die Klassen auszugleichen
    X_res, y_res = smoteenn.fit_resample(X, y)

    # Zurückkonvertieren zu einem DataFrame
    df_resampled = pd.DataFrame(X_res, columns=feature_cols)
    df_resampled[target_col] = y_res

    return df_resampled


def preprocess_and_under_sample(dataframe, target_col='Trend', random_state=42):
    """
    Balanciert die Klassen in einem DataFrame durch Unterabtastung der Mehrheitsklasse,
    wobei 'Trend' die Zielspalte ist und alle anderen Spalten als Features betrachtet werden.

    Args:
        dataframe (pd.DataFrame): Eingabe-DataFrame mit 'Trend' als Zielspalte und allen anderen Spalten als Features.
        target_col (str, optional): Name der Zielspalte. Standard ist 'Trend'.
        random_state (int, optional): Zufallszustand für die Reproduzierbarkeit. Standard ist 42.

    Returns:
        pd.DataFrame: Das resampelte DataFrame mit ausgeglichenen Klassen.
    """
    # Überprüfen, ob die Zielspalte vorhanden ist
    if target_col not in dataframe.columns:
        raise ValueError(f"Die Zielspalte '{target_col}' ist nicht im DataFrame vorhanden.")

    # Definiere die Feature- und Zielspalten
    feature_cols = [col for col in dataframe.columns if col != target_col]
    target_col = target_col

    # Extrahiere Features und Zielvariable
    X = dataframe[feature_cols].values
    y = dataframe[target_col].values

    # Initialisiere den RandomUnderSampler
    rus = RandomUnderSampler(random_state=random_state)

    # Wende den RandomUnderSampler an, um die Mehrheitsklasse zu unterabtasten
    X_res, y_res = rus.fit_resample(X, y)

    # Zurückkonvertieren zu einem DataFrame
    df_resampled = pd.DataFrame(X_res, columns=feature_cols)
    df_resampled[target_col] = y_res

    return df_resampled

def copy_this_script(destination_path: str) -> None:
    """
    Erstellt eine Kopie des Skripts, in dem diese Funktion definiert ist,
    im angegebenen Zielverzeichnis (destination_path).

    Parameter:
    -----------
    destination_path : str
        Pfad zu dem Verzeichnis, in das die Kopie erstellt werden soll.
    """

    # Absoluten Pfad dieser Datei (Skript) ermitteln
    current_script = os.path.abspath(__file__)

    # Falls das Zielverzeichnis nicht existiert, erstellen wir es
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Zielpfad setzen (Skriptname + Zielverzeichnis)
    target_file = os.path.join(destination_path, os.path.basename(current_script))

    # Kopie erstellen
    shutil.copy2(current_script, target_file)

    print(f"Die Kopie von '{current_script}' wurde erfolgreich nach '{target_file}' erstellt.")



class Tee:
    """
    Leitet alle geschriebenen Daten (z.B. von print) an mehrere Streams weiter.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

# Globale Variablen für Original-Stdout und Datei-Handle
_original_stdout = None
_log_file = None

def start_printout_capture(file_path: str):
    """
    Leitet ab sofort sämtliche print-Ausgaben
    sowohl in die Konsole (PyCharm) als auch in die angegebene Textdatei um.
    """
    global _original_stdout, _log_file

    if _original_stdout is not None:
        # Falls bereits ein Logging läuft, machen wir nichts
        return

    # Original stdout sichern und Datei öffnen
    _original_stdout = sys.stdout
    _log_file = open(file_path, "w", encoding="utf-8")

    # Tee-Objekt, das in Konsole UND Datei schreibt
    tee = Tee(_original_stdout, _log_file)
    sys.stdout = tee  # Ab jetzt gehen alle prints in beide Streams

def stop_printout_capture():
    """
    Stoppt die Umleitung des Printouts und schließt die Datei.
    Danach gehen print-Ausgaben nur noch an die Konsole (PyCharm).
    """
    global _original_stdout, _log_file

    if _original_stdout is None:
        # Falls kein Logging aktiv ist
        return

    # sys.stdout wieder zurücksetzen
    sys.stdout = _original_stdout
    _original_stdout = None

    # Datei schließen
    if _log_file is not None:
        _log_file.close()
        _log_file = None



def shuffle_data(data, random_state=None):
    """
    Shuffelt die übergebenen Daten.
    - Bei einem Pandas DataFrame oder Series werden die Zeilen gemischt.
    - Bei einem NumPy-Array werden die Zeilen (erste Dimension) gemischt.
    - Bei einer Liste wird die Reihenfolge der Elemente gemischt.

    Falls ein random_state angegeben wird, erfolgt das Shuffling reproduzierbar.

    Parameters:
        data: pandas.DataFrame, pandas.Series, numpy.ndarray oder list
        random_state (int, optional): Seed für reproduzierbares Shuffling. Standardmäßig None.

    Returns:
        Die geshuffelten Daten in der gleichen Datenstruktur wie das Original.
    """
    # Für Pandas DataFrame oder Series
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Für NumPy-Array
    elif isinstance(data, np.ndarray):
        # Erstelle einen Generator mit oder ohne Seed
        rng = np.random.default_rng(random_state) if random_state is not None else np.random.default_rng()
        indices = rng.permutation(data.shape[0])
        return data[indices]

    # Für Listen
    elif isinstance(data, list):
        data_copy = data[:]  # Kopie erstellen, um das Original nicht zu verändern
        if random_state is not None:
            rnd = random.Random(random_state)
            rnd.shuffle(data_copy)
        else:
            random.shuffle(data_copy)
        return data_copy

    else:
        raise ValueError("Data type not supported for shuffling")



def train_model_v4(tuning=None, n_trials=None, n_jobs=None, database_name_optuna=None, max_epochs=999999, show_progression=False, verbose=False, db_wkdm_orig=None, config=None):
    db_wkdm = db_wkdm_orig.copy()

    # Konfiguriere Logging
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    # Daten vorbereiten
    logging.info('Daten werden vorbereitet...')
    Y_orig_df = db_wkdm["Trend"].values

    # One-Hot Encoder initialisieren und fitten
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoder.fit(Y_orig_df.reshape(-1, 1))

    # Daten aufteilen, falls Tuning aktiviert ist
    if tuning:
        test_size = 0.2
        train_df, test_df = train_test_split(
            db_wkdm,
            test_size=test_size,
            stratify=Y_orig_df,
            random_state=42,  # Für Reproduzierbarkeit
            shuffle=True  # Daten werden gemischt
            # shuffle = False  # Daten werden gemischt

        )
        # print(train_df.head())
        # print(test_df.head())
        # exit()
        # train_df = preprocess_and_balance(train_df)
        # train_df = preprocess_and_under_sample(dataframe=train_df, target_col='Trend', random_state=42)
        # train_df = preprocess_and_balance_smoteenn(dataframe=train_df, target_col='Trend', random_state=42)

    else:

        # train_df = db_wkdm
        train_df = shuffle_data(data=db_wkdm, random_state=42)
        test_df = None

    # One-Hot Encoding der Zielvariable in train_df
    train_encoded = onehot_encoder.transform(train_df[['Trend']].values)
    trend_columns = [f"Trend_{category}" for category in onehot_encoder.categories_[0]]
    train_trend_df = pd.DataFrame(train_encoded, columns=trend_columns, index=train_df.index)
    train_df = pd.concat([train_df.drop("Trend", axis=1), train_trend_df], axis=1)
    # print(f'train_df:\n{train_df.head()}')
    # exit()

    # One-Hot Encoding der Zielvariable in test_df, falls vorhanden
    if tuning and test_df is not None:
        test_encoded = onehot_encoder.transform(test_df[['Trend']].values)
        test_trend_df = pd.DataFrame(test_encoded, columns=trend_columns, index=test_df.index)
        test_df = pd.concat([test_df.drop("Trend", axis=1), test_trend_df], axis=1)

    # Trenne Features und Targets
    feature_cols = [col for col in train_df.columns if col not in trend_columns]
    target_cols = trend_columns

    if verbose:
        print('Transformiere Trainingsdaten zu TensorFlow Dataset')
    train_ds = df_to_dataset(train_df, feature_cols=feature_cols, target_cols=target_cols)

    # for features, labels in train_ds.take(5):  # Zeigt die ersten 5 Beispiele an
    #     print("Features:")
    #     print(features)  # Merkmale (Inputs)
    #     print("Labels:")
    #     print(labels)  # Zielwerte (Outputs)    exit()
    # exit()


    if tuning:
        if verbose:
            print('Transformiere Testdaten zu TensorFlow Dataset')
        test_ds = df_to_dataset(test_df, feature_cols=feature_cols, target_cols=target_cols)
        # test_ds = df_to_dataset_stratified(dataframe=test_df, feature_cols=feature_cols, target_cols=target_cols, batch_size=batch_size, shuffle=True)

    else:
        test_ds = None

    all_inputs, all_features_combined, normalization_layers, encoding_layers = preprocess_data2(
        train_df=train_df,
        train_ds=train_ds,
        # numerical_cols=feature_cols,
        trend_columns=trend_columns,
        verbose=verbose
    )

    calculate_neurons_and_print_code(all_features_combined.shape)

    if verbose: print(f'load_study')
    study = load_study(database_name_optuna,
                       # direction="minimize"
                       direction="maximize"
                      )

    # study = load_study_multy(database_name_optuna, direction=["minimize", "maximize"])




    # Extrahiere die Zielvariable aus dem Trainingsset
    y_train = train_df[target_cols].values
    y_train_labels = np.argmax(y_train, axis=1)

    # Berechne die Klassengewichte
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )

    # Umwandlung der Klassengewichte in ein Dictionary
    class_weights = dict(enumerate(class_weights))

    # Normalisierung der Klassengewichte (Summe der Gewichte = 1)
    sum_weights = sum(class_weights.values())
    normalized_class_weights = {k: v / sum_weights for k, v in class_weights.items()}

    if verbose:
        print(f"Normalisierte Klassengewichte: {normalized_class_weights}")
    # exit()



    if tuning:

        saved_models_tuning_data = f"saved_models_tuning_data/{database_name_optuna}"
        copy_this_script(destination_path=saved_models_tuning_data)

        # Objective-Funktion mit Daten erstellen
        logging.info('Objective-Funktion wird erstellt...')
        objective_with_data = functools.partial(
            train_or_tune_model,
            train_ds=train_ds,
            test_ds=test_ds,
            all_inputs=all_inputs,
            all_features_combined=all_features_combined,
            tuning=True,
            verbose=0,
            n_jobs=n_jobs,
            max_epochs=max_epochs,
            trend_columns=trend_columns,
            class_weights=normalized_class_weights  # Normalisierte Klassengewichte hinzufügen

        )

        # Optimierung starten
        logging.info('Optimierung der Studie startet...')
        start_study = time.time()
        study.optimize(objective_with_data, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)  # n_jobs=1 wegen Optuna-Beschränkungen
        end_study = time.time()
        logging.info(f'Studie abgeschlossen in {round(end_study - start_study, 2)} Sekunden')

        best_params = study.best_params
        best_loss = study.best_value
        logging.info(f"Beste Hyperparameter: {best_params}")
        logging.info(f"Bester Verlust: {best_loss}")

    else:
        # Modell und Skalierer speichern
        model_save_dir = f"saved_models/nn_model_{database_name_optuna}"
        model_save_dir_config = f"saved_models/nn_model_{database_name_optuna}_{config}"

        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(model_save_dir_config, exist_ok=True)


        model_save_path = os.path.join(model_save_dir, f"nn_model_{database_name_optuna}.keras")
        model_save_path_config = os.path.join(model_save_dir_config, f"nn_model_{database_name_optuna}.keras")

        printout_save_path = os.path.join(model_save_dir, f"nn_model_{database_name_optuna}_printout.txt")
        start_printout_capture(printout_save_path)



        best_params = study.best_params
        best_loss = study.best_value
        logging.info(f"Beste Hyperparameter: {best_params}")
        logging.info(f"Bester Verlust: {best_loss}")

        # Modell trainieren
        logging.info('Modell wird trainiert...')
        model = train_or_tune_model(
            best_params=best_params,
            train_ds=train_ds,
            all_inputs=all_inputs,
            all_features_combined=all_features_combined,
            tuning=tuning,
            verbose=2,
            n_jobs=n_jobs,
            max_epochs=max_epochs,
            trend_columns=trend_columns,
            class_weights=normalized_class_weights  # Normalisierte Klassengewichte hinzufügen

        )


        model.save(model_save_path)

        try:
            model.save(model_save_path_config)
            copy_this_script(destination_path=model_save_dir_config)

        except:
            print(traceback.print_exc())
            pass

        copy_this_script(destination_path=model_save_dir)

        stop_printout_capture()
        print("Logging ist jetzt beendet.")

def preprocess_and_balance(dataframe):
    """
    Balanciert die Klassen in einem DataFrame unter Verwendung von SMOTE, wobei 'Trend' die Zielspalte ist
    und alle anderen Spalten als Features betrachtet werden.

    Args:
        dataframe (pd.DataFrame): Eingabe-DataFrame mit 'Trend' als Zielspalte und allen anderen Spalten als Features.

    Returns:
        pd.DataFrame: Das resampelte DataFrame mit ausgeglichenen Klassen.
    """
    # Definiere die Feature- und Zielspalten
    feature_cols = [col for col in dataframe.columns if col != 'Trend']
    target_col = 'Trend'

    # Extrahiere Features und Zielvariable
    X = dataframe[feature_cols].values
    y = dataframe[target_col].values

    # Initialisiere SMOTE
    smote = SMOTE(random_state=42)

    # Wende SMOTE an, um die Klassen auszugleichen
    X_res, y_res = smote.fit_resample(X, y)

    # Zurückkonvertieren zu einem DataFrame
    df_resampled = pd.DataFrame(X_res, columns=feature_cols)
    df_resampled[target_col] = y_res

    return df_resampled



# class FeatureWeightingLayer(Layer):
#     def __init__(self, num_features, **kwargs):
#         super(FeatureWeightingLayer, self).__init__(**kwargs)
#         self.num_features = num_features
#
#     def build(self, input_shape):
#         # Initialisiere die Gewichte mit 1.0
#         self.feature_weights = self.add_weight(
#             shape=(self.num_features,),
#             initializer='ones',
#             trainable=True,
#             name='feature_weights',
#             regularizer=tf.keras.regularizers.l2(0.01)  # Beispiel für L2-Regularisierung
#         )
#         super(FeatureWeightingLayer, self).build(input_shape)
#
#     def call(self, inputs):
#         return inputs * self.feature_weights
#
#     def get_config(self):
#         config = super(FeatureWeightingLayer, self).get_config()
#         config.update({"num_features": self.num_features})
#         return config


# class FeatureWeightingLayer(Layer):
#     def __init__(self, num_features, l2_reg=0.01, **kwargs):
#         super(FeatureWeightingLayer, self).__init__(**kwargs)
#         self.num_features = num_features
#         self.l2_reg = l2_reg  # L2-Regularisierungsstärke als Parameter
#
#     def build(self, input_shape):
#         # Initialisiere die Gewichte mit 1.0 und setze die L2-Regularisierung
#         self.feature_weights = self.add_weight(
#             shape=(self.num_features,),
#             initializer='ones',
#             trainable=True,
#             name='feature_weights',
#             regularizer=tf.keras.regularizers.l2(self.l2_reg)
#         )
#         super(FeatureWeightingLayer, self).build(input_shape)
#
#     def call(self, inputs):
#         return inputs * self.feature_weights
#
#     def get_config(self):
#         config = super(FeatureWeightingLayer, self).get_config()
#         config.update({
#             "num_features": self.num_features,
#             "l2_reg": self.l2_reg
#         })
#         return config


class FeatureWeightingLayer(tf.keras.layers.Layer):
    def __init__(self, l2_reg=0.01, **kwargs):
        super(FeatureWeightingLayer, self).__init__(**kwargs)
        self.l2_reg = l2_reg  # L2-Regularisierungsstärke als Parameter

    def build(self, input_shape):
        # Dynamische Ableitung der Anzahl der Features aus der Eingabeform
        self.num_features = input_shape[-1]
        self.feature_weights = self.add_weight(
            shape=(self.num_features,),
            initializer='ones',
            trainable=True,
            name='feature_weights',
            regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )
        super(FeatureWeightingLayer, self).build(input_shape)

    def call(self, inputs):
        return inputs * self.feature_weights

    def get_config(self):
        config = super(FeatureWeightingLayer, self).get_config()
        config.update({
            "l2_reg": self.l2_reg
        })
        return config





# def preprocess_data1(train_df=None, train_ds=None, trend_columns=None, verbose=False):
#     """
#     Vorverarbeitung der Daten für das neuronale Netz, einschließlich der Integration von FastText- und BERT-Features.
#
#     Args:
#         train_df (pd.DataFrame): Trainings-DataFrame.
#         train_ds (tf.data.Dataset): Trainings-Dataset.
#         trend_columns (list): Liste der Trend-Spalten nach One-Hot-Encoding.
#         verbose (bool, optional): Anzeige von Fortschrittsbalken. Standard ist False.
#
#     Returns:
#         tuple: (all_inputs, all_features_combined, normalization_layers, encoding_layers)
#     """
#     encoded_features = []
#     all_inputs = []
#     normalization_layers = {}  # Speichere Normalisierungsschichten
#     encoding_layers = {}        # Speichere Kategorisierungsschichten
#
#     ###################### Numerische Spalten (ohne Trend-Spalten)
#     if verbose:
#         print(f'Preparing Numerical Columns')
#
#     numerical_cols = [col for col in train_df.columns if col not in trend_columns]
#     for header in tqdm(numerical_cols, total=len(numerical_cols), desc="Numerical Columns", disable=not verbose):
#         try:
#             numeric_col = tf.keras.Input(shape=(1,), name=header, dtype='float32')
#             normalization_layer = get_normalization_layer(header, train_ds)
#             encoded_numeric_col = normalization_layer(numeric_col)
#             all_inputs.append(numeric_col)
#             encoded_features.append(encoded_numeric_col)
#             normalization_layers[header] = normalization_layer  # Speichere die Schicht
#         except Exception as e:
#             print(f"Fehler beim Verarbeiten der numerischen Spalte '{header}': {e}")
#             print(traceback.format_exc())
#             exit()
#
#     ###################### Kombinieren der Features
#     if verbose:
#         print(f'Concatenate all_features')
#     if encoded_features:
#         all_features_combined = tf.keras.layers.concatenate(encoded_features)
#     else:
#         all_features_combined = None
#
#     ###################### Feature Weighting
#     if verbose:
#         print(f'Applying FeatureWeightingLayer')
#     if all_features_combined is not None:
#         all_features_combined = FeatureWeightingLayer(num_features=all_features_combined.shape[-1], name='feature_weighting')(all_features_combined)
#     else:
#         all_features_combined = None
#
#     return all_inputs, all_features_combined, normalization_layers, encoding_layers
#     # return all_inputs, all_features_combined, normalization_layers, encoding_layers


def preprocess_data2(train_df=None, train_ds=None, trend_columns=None, verbose=False):
    """
    Optimierte Vorverarbeitung der Daten für das neuronale Netz, einschließlich der Integration von FastText- und BERT-Features.

    Args:
        train_df (pd.DataFrame): Trainings-DataFrame.
        train_ds (tf.data.Dataset): Trainings-Dataset.
        trend_columns (list): Liste der Trend-Spalten nach One-Hot-Encoding.
        verbose (bool, optional): Anzeige von Fortschrittsbalken. Standard ist False.

    Returns:
        tuple: (all_inputs, all_features_combined, normalization_layers, encoding_layers)
    """
    encoded_features = []
    all_inputs = []
    normalization_layers = {}  # Speichere Normalisierungsschichten
    encoding_layers = {}  # Speichere Kategorisierungsschichten

    ###################### Numerische Spalten (ohne Trend-Spalten)
    if verbose:
        print('Preparing Numerical Columns')

    numerical_cols = [col for col in train_df.columns if col not in trend_columns]

    # Vorab Berechnung von Mittelwert und Standardabweichung für alle numerischen Spalten
    stats = train_df[numerical_cols].agg(['mean', 'std']).to_dict()

    for header in tqdm(numerical_cols, total=len(numerical_cols), desc="Numerical Columns", disable=not verbose):
        try:
            numeric_col = tf.keras.Input(shape=(1,), name=header, dtype='float32')

            # Manuelle Normalisierung mittels Lambda
            mean = stats[header]['mean']
            std = stats[header]['std']
            normalization_layer = Lambda(lambda x, m=mean, s=std: (x - m) / s, name=f'normalization_{header}')

            # normalization_layer = get_normalization_layer(header, train_ds)

            encoded_numeric_col = normalization_layer(numeric_col)

            all_inputs.append(numeric_col)
            encoded_features.append(encoded_numeric_col)
            normalization_layers[header] = normalization_layer  # Speichere die Schicht
        except Exception as e:
            print(f"Fehler beim Verarbeiten der numerischen Spalte '{header}': {e}")
            print(traceback.format_exc())
            exit()

    ###################### Kombinieren der Features
    if verbose:
        print('Concatenate all_features')
    if encoded_features:
        all_features_combined = tf.keras.layers.concatenate(encoded_features)
    else:
        all_features_combined = None

    ###################### Feature Weighting
    if verbose: print('Applying FeatureWeightingLayer')
    # if all_features_combined is not None:
    # all_features_combined = FeatureWeightingLayer(num_features=all_features_combined.shape[-1], name='feature_weighting')(all_features_combined)
    # else:
    #     all_features_combined = None

    # return all_inputs, all_features_combined, normalization_layers, encoding_layers
    return all_inputs, all_features_combined, normalization_layers, encoding_layers





def preprocess_data3(train_ds, numerical_cols, trend_columns, verbose=False):
    """
    Optimierte Vorverarbeitung der Daten direkt aus einem TensorFlow Dataset.

    Args:
        train_ds (tf.data.Dataset): TensorFlow Dataset für das Training.
        numerical_cols (list): Liste der numerischen Spalten.
        trend_columns (list): Liste der Trend-Spalten nach One-Hot-Encoding.
        verbose (bool, optional): Anzeige von Fortschrittsbalken. Standard ist False.

    Returns:
        tuple: (all_inputs, all_features_combined, normalization_layers, encoding_layers)
    """
    encoded_features = []
    all_inputs = []
    normalization_layers = {}
    encoding_layers = {}

    if verbose:
        print("Berechne Mittelwert und Standardabweichung für numerische Spalten...")

    # Berechne Mittelwert und Standardabweichung aus train_ds
    stats = {}
    for col in tqdm(numerical_cols):
        feature_ds = train_ds.map(lambda x, _: x[col], num_parallel_calls=tf.data.AUTOTUNE)
        mean = tf.reduce_mean(list(feature_ds.as_numpy_iterator()))
        std = tf.math.reduce_std(list(feature_ds.as_numpy_iterator()))
        stats[col] = {'mean': mean.numpy(), 'std': std.numpy() + 1e-8}

    # Normalisierungsschichten erstellen
    for col in tqdm(numerical_cols, disable=not verbose):
        numeric_input = tf.keras.Input(shape=(1,), name=col, dtype=tf.float32)
        mean, std = stats[col]['mean'], stats[col]['std']
        normalization_layer = Lambda(lambda x, m=mean, s=std: (x - m) / s, name=f"normalization_{col}")
        encoded_numeric_col = normalization_layer(numeric_input)

        all_inputs.append(numeric_input)
        encoded_features.append(encoded_numeric_col)
        normalization_layers[col] = normalization_layer

    # Kombinieren der Features
    if encoded_features:
        all_features_combined = tf.keras.layers.concatenate(encoded_features)
    else:
        all_features_combined = None

    return all_inputs, all_features_combined, normalization_layers, encoding_layers



def get_normalization_layer2(name, dataset):
    normalizer = tf.keras.layers.Normalization(axis=None)

    # feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = dataset.map(lambda x, y: x[name], num_parallel_calls=tf.data.AUTOTUNE)
    feature_ds = feature_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    normalizer.adapt(feature_ds)

    return normalizer


def get_normalization_layer(name, dataset):
    """
    Erstellt eine Normalisierungsschicht für ein bestimmtes Feature und passt sie an die Daten an.

    Args:
        name (str): Der Name des Features, das normalisiert werden soll.
        dataset (tf.data.Dataset): Das TensorFlow Dataset, das die Daten enthält.

    Returns:
        tf.keras.layers.Normalization: Die angepasste Normalisierungsschicht.
    """
    normalizer = tf.keras.layers.Normalization(axis=None)

    # Extrahiere das spezifische Feature mit paralleler Verarbeitung
    feature_ds = dataset.map(lambda x, y: x[name], num_parallel_calls=tf.data.AUTOTUNE)
    # feature_ds = dataset.map(lambda x, y: x[name], num_parallel_calls=32)

    # Passe die Normalisierungsschicht an die extrahierten Feature-Daten an
    normalizer.adapt(feature_ds)

    return normalizer


# Definieren der F1-Score-Metrik als benutzerdefinierte Keras-Metrik
# class F1Score(tf.keras.metrics.Metric):
#     def __init__(self, name='f1_score', **kwargs):
#         super(F1Score, self).__init__(name=name, **kwargs)
#         self.precision = tf.keras.metrics.Precision()
#         self.recall = tf.keras.metrics.Recall()
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # Konvertiere y_pred zu Klassenlabels
#         y_pred_labels = tf.argmax(y_pred, axis=1)
#         y_true_labels = tf.argmax(y_true, axis=1)
#         self.precision.update_state(y_true_labels, y_pred_labels, sample_weight)
#         self.recall.update_state(y_true_labels, y_pred_labels, sample_weight)
#
#     def result(self):
#         precision = self.precision.result()
#         recall = self.recall.result()
#         return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
#
#     def reset_state(self):
#         self.precision.reset_state()
#         self.recall.reset_state()


# class F1Score(tf.keras.metrics.Metric):
#     def __init__(self, name='f1_score', **kwargs):
#         super(F1Score, self).__init__(name=name, **kwargs)
#         self.precision = tf.keras.metrics.Precision()
#         self.recall = tf.keras.metrics.Recall()
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred_labels = tf.argmax(y_pred, axis=1)
#         y_true_labels = tf.argmax(y_true, axis=1)
#         self.precision.update_state(y_true_labels, y_pred_labels, sample_weight)
#         self.recall.update_state(y_true_labels, y_pred_labels, sample_weight)
#
#     def result(self):
#         precision = self.precision.result()
#         recall = self.recall.result()
#         return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
#
#     def reset_states(self):
#         self.precision.reset_states()
#         self.recall.reset_states()


# class F1Score(tf.keras.metrics.Metric):
#     def __init__(self, name='f1_score', average='macro', num_classes=2, **kwargs):
#         super(F1Score, self).__init__(name=name, **kwargs)
#         self.average = average
#         self.num_classes = num_classes
#
#         # Initialisiere Zähler für True Positives, False Positives und False Negatives pro Klasse
#         self.tp = self.add_weight(name='tp', shape=(self.num_classes,), initializer='zeros')
#         self.fp = self.add_weight(name='fp', shape=(self.num_classes,), initializer='zeros')
#         self.fn = self.add_weight(name='fn', shape=(self.num_classes,), initializer='zeros')
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # Konvertiere y_true und y_pred in Label-Form (nicht one-hot)
#         if y_true.shape[-1] > 1:
#             y_true = tf.argmax(y_true, axis=-1)
#         if y_pred.shape[-1] > 1:
#             y_pred = tf.argmax(y_pred, axis=-1)
#
#         y_true = tf.cast(y_true, tf.int32)
#         y_pred = tf.cast(y_pred, tf.int32)
#
#         # Erstelle One-Hot Encodings
#         y_true_one_hot = tf.one_hot(y_true, self.num_classes)
#         y_pred_one_hot = tf.one_hot(y_pred, self.num_classes)
#
#         # Berechne True Positives, False Positives und False Negatives
#         tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
#         fp = tf.reduce_sum((1 - y_true_one_hot) * y_pred_one_hot, axis=0)
#         fn = tf.reduce_sum(y_true_one_hot * (1 - y_pred_one_hot), axis=0)
#
#         # Aktualisiere die Zähler
#         self.tp.assign_add(tp)
#         self.fp.assign_add(fp)
#         self.fn.assign_add(fn)
#
#     def result(self):
#         precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
#         recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
#         f1 = 2 * precision * recall / (precision + recall + K.epsilon())
#
#         if self.average == 'macro':
#             return tf.reduce_mean(f1)
#         elif self.average == 'micro':
#             return 2 * tf.reduce_sum(self.tp) / (2 * tf.reduce_sum(self.tp) + tf.reduce_sum(self.fp) + tf.reduce_sum(self.fn))
#         elif self.average == 'weighted':
#             support = self.tp + self.fn
#             return tf.reduce_sum(f1 * support) / tf.reduce_sum(support)
#         else:
#             return tf.reduce_mean(f1)
#
#     def reset_states(self):
#         for var in self.variables:
#             var.assign(tf.zeros_like(var))


class CustomF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super(CustomF1Score, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        return 2 * precision * recall / (precision + recall + 1e-7)

    # def reset_states(self):
    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

def train_or_tune_model(trial=None, train_ds=None, test_ds=None, all_inputs=None, all_features_combined=None, best_params=None, tuning=True, verbose=0, n_jobs=1, max_epochs=999999, trend_columns=None, class_weights=None, use_class_weights=False):
    """
    Trainiert oder optimiert das neuronale Netz-Modell basierend auf den bereitgestellten Parametern.

    Args:
        trial (optuna.trial.Trial, optional): Optuna Trial-Objekt für Hyperparameter-Tuning.
        train_ds (tf.data.Dataset): Trainings-Dataset.
        test_ds (tf.data.Dataset, optional): Test-Dataset.
        all_inputs (list): Liste der Input-Layer.
        all_features_combined (Tensor): Gewichtete Features nach dem FeatureWeightingLayer.
        best_params (dict, optional): Beste Hyperparameter aus der Studie.
        tuning (bool, optional): Flag zur Aktivierung des Tunings. Standard ist True.
        verbose (int, optional): Verbosity-Level. Standard ist 0.
        trend_columns (list, optional): Liste der Trend-Spalten nach One-Hot-Encoding.

    Returns:
        float or tf.keras.Model: Verlustwert für Optuna oder das trainierte Modell.
    """
    try:

        workers = max(1, 32 // n_jobs)

        if n_jobs > 1:
            use_multiprocessing = True
        else:
            use_multiprocessing = False

        # Dynamische Bestimmung der Neuronenzahl basierend auf der Feature-Anzahl
        min_neurons = int(math.sqrt(all_features_combined.shape[-1]))
        num_neurons = all_features_combined.shape[-1]
        max_neurons = all_features_combined.shape[-1] * 2  # Beispiel: doppelte Anzahl der Features

        # Hyperparameterraum definieren
        if tuning:


            # IMPORTANT
            # Vorschlagen von Hyperparametern
            activation = trial.suggest_categorical('activation', [
                'relu',
                # 'leaky_relu',
                # 'elu',
                # 'selu',
                # 'tanh',
                # 'sigmoid'
            ])
            # activation = 'relu'

            kernel_initializer = trial.suggest_categorical('kernel_initializer', [
                'he_normal',
                # 'he_uniform',
                # 'glorot_normal',
                # 'glorot_uniform',
                # 'lecun_normal'
            ])
            # kernel_initializer = 'he_normal'
            # kernel_initializer = 'he_uniform'

            n_layers = trial.suggest_int('n_layers', 1, 1, log=False)
            # n_layers = trial.suggest_int('n_layers', 2, 2, log=False)


            layer_units = []
            prev_units = max_neurons
            # prev_units = num_neurons
            for i in range(n_layers):
                units = trial.suggest_int(f'n_units_l{i}', min_neurons, prev_units, log=False)
                # units = trial.suggest_int(f'n_units_l{i}', num_neurons, prev_units, log=False)
                # units = trial.suggest_int(f'n_units_l{i}', min_neurons, num_neurons, log=False)
                # units = trial.suggest_int(f'n_units_l{i}', 407, 407, log=False)
                layer_units.append(units)
                prev_units = units  # Sicherstellen, dass nächste Schicht <= vorherige


            layer_dropout_rates = []
            for i in range(n_layers):
                # dropout = trial.suggest_int(f'dropout_rate_l{i}', 0.3, 0.7, log=False)
                dropout = trial.suggest_float(f'dropout_rate_l{i}', 0.4, 0.6, log=False)
                layer_dropout_rates.append(dropout)


            # batch_size = trial.suggest_int('batch_size', 1, 30, log=False)
            batch_size = trial.suggest_int('batch_size', 16, 16, log=False)


            initial_learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=False)
            # initial_learning_rate = trial.suggest_int('learning_rate', 0.001, 0.002, log=False)


            optimizer_name = trial.suggest_categorical('optimizer', [
                'adam',
                # 'nadam',
                # 'rmsprop'
            ])
            # optimizer_name = 'adam'


            if optimizer_name == 'adam':
                # beta_1 = trial.suggest_int('adam_beta_1', 0.8, 0.99, step=0.01, log=False)
                beta_1 = trial.suggest_float('adam_beta_1', 0.9, 0.99, log=False)

                # beta_2 = trial.suggest_int('adam_beta_2', 0.9, 0.9999, step=0.0001, log=False)
                beta_2 = trial.suggest_float('adam_beta_2', 0.9, 0.99, log=False)

                epsilon = trial.suggest_float('adam_epsilon', 1e-8, 1e-6, log=False)
                # epsilon = trial.suggest_int('adam_epsilon', 0.0000008665, 0.0000008665, log=False)  # test 273

                # Vorschlag für Clipping-Methoden
                # clipping_method = trial.suggest_categorical('clipping_method', [None, 'clipnorm', 'clipvalue'])
                # if clipping_method == 'clipnorm':
                #     clipnorm = trial.suggest_int('clipnorm', 0.5, 5.0, log=True)
                #     clipvalue = None
                # elif clipping_method == 'clipvalue':
                #     clipvalue = trial.suggest_int('clipvalue', 0.1, 5.0, log=True)
                #     clipnorm = None
                # else:
                #     clipnorm = None
                #     clipvalue = None


                optimizer = tf.keras.optimizers.Adam(
                                                    learning_rate=initial_learning_rate,
                                                    beta_1=beta_1,
                                                    beta_2=beta_2,
                                                    epsilon=epsilon,
                                                    # clipvalue=clipnorm,  # ggfs. entfernen, ACHTUNG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                                    # clipnorm=clipvalue  # ggfs. entfernen, ACHTUNG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                                    )

            # elif optimizer_name == 'nadam':
            #     optimizer = tf.keras.optimizers.Nadam(learning_rate=initial_learning_rate)
            # elif optimizer_name == 'rmsprop':
            #     optimizer = tf.keras.optimizers.RMSprop(learning_rate=initial_learning_rate)
            # else:
            #     raise ValueError(f"Unbekannter Optimierer: {optimizer_name}")

            # Vorschlagen der L2-Regularisierungsstärke für FeatureWeightingLayer
            # feature_weighting_l2 = trial.suggest_int('feature_weighting_l2', 0.001, 0.1, log=False)
            feature_weighting_l2 = trial.suggest_float('feature_weighting_l2', 0.01, 0.1, log=False)

            # Vorschlagen der L2-Regularisierungsstärke für Dense-Schichten
            # dense_l2_reg = trial.suggest_int('dense_l2_reg', 0.001, 0.1, log=False)
            dense_l2_reg = trial.suggest_float('dense_l2_reg', 0.01, 0.1, log=False)


            average = trial.suggest_categorical('average', [
                'macro',
                # 'micro',
                # 'weighted'
                ])

        else:
            # Verwenden der besten Hyperparameter aus der Studie
            n_layers = best_params['n_layers']
            # n_layers = 2

            # layer_units = [best_params[f'n_units_l{i}'] for i in range(n_layers)]
            # layer_units = [min_neurons]
            layer_units = [int(num_neurons/2)]
            # layer_units = [num_neurons]
            # layer_units = [num_neurons*10]
            # layer_units = [num_neurons/2, num_neurons/4]


            activation = best_params['activation']
            # activation = 'relu'

            kernel_initializer = best_params['kernel_initializer']
            # kernel_initializer = "he_normal"

            layer_dropout_rates = [best_params[f'dropout_rate_l{i}'] for i in range(n_layers)]
            # layer_dropout_rates = [0.5322357247372841, 0.5322357247372841]
            # layer_dropout_rates = [0.6]


            # batch_size = best_params['batch_size']
            batch_size = 8
            # batch_size = 4
            # batch_size = 2
            # batch_size = 1

            initial_learning_rate = best_params['learning_rate']
            # initial_learning_rate = 0.001

            optimizer_name = best_params['optimizer']
            # optimizer_name = 'adam'

            feature_weighting_l2 = best_params['feature_weighting_l2']
            # feature_weighting_l2 = 0.09

            dense_l2_reg = best_params['dense_l2_reg']
            # dense_l2_reg = 0.06

            # average = best_params['average']
            # average = 'micro'
            average = 'macro'
            # average = 'weighted'



            if optimizer_name == 'adam':
                beta_1 = best_params['adam_beta_1']
                # beta_1 = 0.9110356784

                beta_2 = best_params['adam_beta_2']
                # beta_2 = 0.9526585946

                epsilon = best_params['adam_epsilon']
                # epsilon = 0.0000000263

                # Vorschlag für Clipping-Methoden
                # clipping_method = best_params['clipping_method']
                # if clipping_method == 'clipnorm':
                #     clipnorm = clipping_method = best_params['clipnorm']
                #     clipvalue = None
                # elif clipping_method == 'clipvalue':
                #     clipvalue = clipping_method = best_params['clipvalue']
                #     clipnorm = None
                # else:
                #     clipnorm = None
                #     clipvalue = None

                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=initial_learning_rate,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    epsilon=epsilon,
                    # clipvalue=clipnorm,  # ggfs. entfernen, ACHTUNG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # clipnorm=clipvalue  # ggfs. entfernen, ACHTUNG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                )
            elif optimizer_name == 'nadam':
                optimizer = tf.keras.optimizers.Nadam(learning_rate=initial_learning_rate)
            elif optimizer_name == 'rmsprop':
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=initial_learning_rate)
            else:
                raise ValueError(f"Unbekannter Optimierer: {optimizer_name}")

        try:
            print(f'\nlayer_units:{layer_units}, layer_dropout_rates:{layer_dropout_rates}, initial_learning_rate:{initial_learning_rate}, batch_size:{batch_size}, feature_weighting_l2:{feature_weighting_l2}, dense_l2_reg:{dense_l2_reg}, beta_1:{beta_1}, beta_2:{beta_2}, epsilon:{epsilon}\n')
        except:
            try:
                print(f'\nlayer_units:{layer_units}, layer_dropout_rates:{layer_dropout_rates}, initial_learning_rate:{initial_learning_rate}, batch_size:{batch_size}, feature_weighting_l2:{feature_weighting_l2}, dense_l2_reg:{dense_l2_reg}\n')
            except:
                pass



        class CustomCallback(tf.keras.callbacks.Callback):
            def __init__(self, tuning=False):
                super().__init__()
                self.tuning = tuning

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()

            def on_epoch_end(self, epoch, logs=None):
                epoch_time = time.time() - self.epoch_start_time
                current_lr = self.model.optimizer.lr.numpy()
                if self.tuning:
                    print(f"Epoch: {epoch + 1:2d}, Loss: {logs['loss']:.4f}, Val Loss: {logs.get('val_loss', 'N/A')}, Zeit: {epoch_time:.2f} Sek., LR: {current_lr:.2e}")
                else:
                    print(f"Epoch: {epoch + 1:2d}, Loss: {logs['loss']:.4f}, Zeit: {epoch_time:.2f} Sek., LR: {current_lr:.2e}")

        class CustomCallback2(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print(f"Epoch {epoch + 1}: val_f1_score = {logs.get('val_f1_score')}")

        import warnings
        class ReduceLROnPlateauWithRestore(tf.keras.callbacks.Callback):
            def __init__(self, monitor='val_loss', factor=0.1, patience=10, verbose=0,
                         mode='auto', min_delta=1e-4, cooldown=0, min_lr=0.0, restore_best_weights=False):
                super().__init__()
                self.monitor = monitor
                self.factor = factor
                if factor >= 1.0:
                    raise ValueError('Der Faktor muss kleiner als 1.0 sein.')
                self.patience = patience
                self.verbose = verbose
                self.cooldown = cooldown
                self.cooldown_counter = 0  # Zähler für Cooldown-Perioden.
                self.min_lr = min_lr
                self.restore_best_weights = restore_best_weights
                self.best_weights = None
                self.wait = 0
                self.min_delta = min_delta

                # Bestimme, ob wir nach "min" oder "max" optimieren
                if mode not in ['auto', 'min', 'max']:
                    warnings.warn(f'Mode {mode} unbekannt, nutze auto.', RuntimeWarning)
                    mode = 'auto'
                self.mode = mode
                if self.mode == 'min':
                    self.monitor_op = lambda current, best: current < best - self.min_delta
                    self.best = np.Inf
                elif self.mode == 'max':
                    self.monitor_op = lambda current, best: current > best + self.min_delta
                    self.best = -np.Inf
                else:
                    # Bei "auto" versuchen wir anhand des Namens zu entscheiden
                    if 'acc' in self.monitor or self.monitor.startswith('f1'):
                        self.monitor_op = lambda current, best: current > best + self.min_delta
                        self.best = -np.Inf
                    else:
                        self.monitor_op = lambda current, best: current < best - self.min_delta
                        self.best = np.Inf

            def on_train_begin(self, logs=None):
                # Reset der internen Zustände
                self.cooldown_counter = 0
                self.wait = 0
                # self.best wurde bereits in __init__ initialisiert

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        f'ReduceLROnPlateau: Der überwachte Wert `{self.monitor}` steht nicht zur Verfügung. '
                        f'Verfügbare Keys: {",".join(list(logs.keys()))}',
                        RuntimeWarning
                    )
                    return

                # Liegt gerade eine Cooldown-Phase vor?
                if self.in_cooldown:
                    self.cooldown_counter -= 1
                    self.wait = 0

                # Verbesserung prüfen und Bestwert (sowie Gewichte) aktualisieren
                if self.monitor_op(current, self.best):
                    self.best = current
                    self.best_weights = self.model.get_weights()
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        old_lr = float(K.get_value(self.model.optimizer.lr))
                        if old_lr > self.min_lr:
                            new_lr = max(old_lr * self.factor, self.min_lr)
                            K.set_value(self.model.optimizer.lr, new_lr)
                            if self.verbose > 0:
                                print(f'\nEpoch {epoch + 1:05d}: Reduziere Lernrate von {old_lr:.6e} auf {new_lr:.6e}.')
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                            if self.restore_best_weights and self.best_weights is not None:
                                print(f'Epoch {epoch + 1:05d}: Beste Gewichte werden wiederhergestellt.')
                                self.model.set_weights(self.best_weights)
                # Sicherstellen, dass der Cooldown-Zähler nicht negativ wird
                if self.in_cooldown:
                    self.cooldown_counter = max(0, self.cooldown_counter)

            @property
            def in_cooldown(self):
                return self.cooldown_counter > 0

        class ReduceLROnPlateauWithFullRestore(tf.keras.callbacks.Callback):
            def __init__(self, monitor='val_loss', factor=0.1, patience=10, verbose=1,
                         mode='auto', min_delta=1e-4, cooldown=0, min_lr=0.0,
                         restore_best_weights=False):
                super().__init__()
                self.monitor = monitor
                self.factor = factor
                if factor >= 1.0:
                    raise ValueError('Der Faktor muss kleiner als 1.0 sein.')
                self.patience = patience
                self.verbose = verbose
                self.cooldown = cooldown
                self.cooldown_counter = 0  # Zähler für die Cooldown-Phase.
                self.min_lr = min_lr
                self.restore_best_weights = restore_best_weights
                self.best_weights = None  # Hier werden die bisher besten Modellgewichte gespeichert.
                self.wait = 0
                self.min_delta = min_delta

                # Bestimme, ob nach "min" oder "max" optimiert werden soll.
                if mode not in ['auto', 'min', 'max']:
                    warnings.warn(f'Mode {mode} unbekannt, nutze auto.', RuntimeWarning)
                    mode = 'auto'
                self.mode = mode
                if self.mode == 'min':
                    self.monitor_op = lambda current, best: current < best - self.min_delta
                    self.best = np.Inf
                elif self.mode == 'max':
                    self.monitor_op = lambda current, best: current > best + self.min_delta
                    self.best = -np.Inf
                else:
                    # Bei "auto" wird anhand des Namens entschieden.
                    if 'acc' in self.monitor or self.monitor.startswith('f1'):
                        self.monitor_op = lambda current, best: current > best + self.min_delta
                        self.best = -np.Inf
                    else:
                        self.monitor_op = lambda current, best: current < best - self.min_delta
                        self.best = np.Inf

            def on_train_begin(self, logs=None):
                # Initialisiere interne Zustände
                self.cooldown_counter = 0
                self.wait = 0

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        f'ReduceLROnPlateau: Der überwachte Wert `{self.monitor}` steht nicht zur Verfügung. '
                        f'Verfügbare Keys: {", ".join(list(logs.keys()))}',
                        RuntimeWarning
                    )
                    return

                # Falls in einer Cooldown-Phase, setze den Warteszähler zurück.
                if self.in_cooldown:
                    self.cooldown_counter -= 1
                    self.wait = 0

                # Bei Verbesserung: aktualisiere den Bestwert und speichere die Gewichte.
                if self.monitor_op(current, self.best):
                    self.best = current
                    self.best_weights = self.model.get_weights()
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        old_lr = float(K.get_value(self.model.optimizer.lr))
                        if old_lr > self.min_lr:
                            new_lr = max(old_lr * self.factor, self.min_lr)
                            # Setze die neue Lernrate
                            K.set_value(self.model.optimizer.lr, new_lr)
                            if self.verbose > 0:
                                print(f'\nEpoch {epoch + 1:05d}: Reduziere Lernrate von {old_lr:.6e} auf {new_lr:.6e}.')
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                            # Falls aktiviert, best weights und Optimizer-Zustand zurücksetzen:
                            if self.restore_best_weights and self.best_weights is not None:
                                print(f'Epoch {epoch + 1:05d}: Beste Gewichte werden wiederhergestellt und Optimizer-Zustand zurückgesetzt.')
                                # Setze die Modellgewichte zurück
                                self.model.set_weights(self.best_weights)
                                # Erzeuge einen neuen Optimizer anhand der Konfiguration des aktuellen Optimizers.
                                optimizer_config = self.model.optimizer.get_config()
                                new_optimizer = self.model.optimizer.__class__.from_config(optimizer_config)
                                # Setze die Lernrate des neuen Optimizers explizit, um Konsistenz zu gewährleisten.
                                K.set_value(new_optimizer.lr, new_lr)
                                # Weise den neuen Optimizer dem Modell zu.
                                self.model.optimizer = new_optimizer

            @property
            def in_cooldown(self):
                return self.cooldown_counter > 0



        #TODO
        callbacks = []
        if tuning:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score', factor=0.5, patience=1, min_delta=0, min_lr=1e-6, cooldown=0, mode="max")
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=10, min_delta=0, restore_best_weights=True, verbose=1, mode="max")
            pruning_callback = TFKerasPruningCallback(trial, 'val_f1_score')
            callbacks.extend([early_stopping, reduce_lr, pruning_callback,
                              # tensorboard_callback,
                              # CustomCallback(tuning=True)
                              ])


        else:
            # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='f1_score', factor=0.5, patience=50, min_delta=0, min_lr=1e-10, cooldown=10)
            # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='f1_score', patience=300, min_delta=1e-10, restore_best_weights=True, verbose=1)

            # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='f1_score', factor=0.5, patience=1, min_delta=0, min_lr=1e-20, cooldown=0, mode="max")
            # reduce_lr = ReduceLROnPlateauWithRestore(monitor='f1_score', factor=0.5, patience=1, min_delta=0, min_lr=1e-20, cooldown=0, mode="max", restore_best_weights=True, verbose=1)
            reduce_lr = ReduceLROnPlateauWithRestore(monitor='f1_score', factor=0.5, patience=5, min_delta=0, min_lr=1e-20, cooldown=0, mode="max", restore_best_weights=True, verbose=1)
            # reduce_lr = ReduceLROnPlateauWithFullRestore(monitor='f1_score', factor=0.5, patience=2, min_delta=0, min_lr=1e-20, cooldown=0, mode="max", restore_best_weights=True, verbose=1)

            # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='f1_score', patience=6, min_delta=0, restore_best_weights=True, verbose=1, mode="max")
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='f1_score', patience=10, min_delta=0, restore_best_weights=True, verbose=1, mode="max")

            callbacks.extend([early_stopping, reduce_lr,
                              # tensorboard_callback,
                              # CustomCallback(tuning=False)
                              ])

        # def build_model(
        #         all_inputs,
        #         all_features_combined,
        #         n_layers,
        #         layer_units,
        #         layer_dropout_rates,
        #         activation,
        #         optimizer,
        #         kernel_initializer,
        #         num_classes,
        #         feature_weighting_l2,
        #         dense_l2_reg,  # Übergabe des neuen Parameters
        #         average
        # ):
        #     """
        #     Baut das neuronale Netz-Modell mit integrierten gewichteten Features.
        #
        #     Args:
        #         all_inputs (list): Liste der Input-Layer.
        #         all_features_combined (Tensor): Gewichtete Features nach dem FeatureWeightingLayer.
        #         n_layers (int): Anzahl der Dense-Schichten.
        #         layer_units (list): Liste der Neuronenzahlen für jede Schicht.
        #         layer_dropout_rates (list): Liste der Dropout-Raten für jede Schicht.
        #         activation (str): Aktivierungsfunktion.
        #         optimizer (tf.keras.optimizers.Optimizer): Optimierer.
        #         kernel_initializer (str): Initialisierungsfunktion für die Gewichte.
        #         num_classes (int): Anzahl der Klassen für die Ausgangsschicht.
        #
        #     Returns:
        #         tf.keras.Model: Das erstellte Modell.
        #     """
        #
        #     # x = all_features_combined
        #
        #     x = FeatureWeightingLayer(
        #         num_features=all_features_combined.shape[-1],
        #         l2_reg=feature_weighting_l2,
        #         # l2_reg=0.01,
        #         name='feature_weighting'
        #     )(all_features_combined)
        #
        #     # Weiterverarbeitung durch Dense Layers
        #     for i in range(n_layers):
        #         x = tf.keras.layers.Dense(
        #             layer_units[i],
        #             activation=activation,
        #             kernel_regularizer=tf.keras.regularizers.l2(dense_l2_reg),
        #             kernel_initializer=kernel_initializer,
        #             name=f'dense_{i}'
        #         )(x)
        #         x = tf.keras.layers.BatchNormalization(name=f'batch_norm_{i}')(x)
        #         x = tf.keras.layers.Dropout(layer_dropout_rates[i], name=f'dropout_{i}')(x)
        #
        #     # Ausgangsschicht mit korrekter Anzahl an Klassen und Aktivierungsfunktion
        #     outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)  # für >= 2 klassen
        #     # outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid', name='output')(x)  # für binäre klassen also nur eine
        #
        #
        #     # Modell erstellen
        #     model = tf.keras.Model(inputs=all_inputs, outputs=outputs)
        #     model.compile(
        #         optimizer=optimizer,
        #         loss='categorical_crossentropy',  # bei one-hot-labels
        #         # loss=tf.keras.metrics.categorical_crossentropy,
        #         # loss='binary_crossentropy',
        #         # loss='sparse_categorical_crossentropy',
        #
        #         # metrics=[F1Score(), 'accuracy']
        #         # metrics=[F1Score(average=average, num_classes=num_classes), 'accuracy']
        #         # metrics=[tfa.metrics.F1Score(num_classes=num_classes, average='macro'), 'accuracy']
        #         metrics=[tfa.metrics.F1Score(num_classes=num_classes, average=average, name="f1_score"), 'accuracy']
        #         # metrics = [CustomF1Score(name='f1_score'), 'accuracy']
        #     )
        #
        #     return model

        def build_model(
                all_inputs,
                all_features_combined,
                n_layers,
                layer_units,
                layer_dropout_rates,
                activation,
                optimizer,
                kernel_initializer,
                num_classes,
                feature_weighting_l2,
                dense_l2_reg,  # Übergabe des neuen Parameters
                average
        ):
            """
            Baut das neuronale Netz-Modell mit integrierten gewichteten Features.

            Args:
                all_inputs (list): Liste der Input-Layer.
                all_features_combined (Tensor): Gewichtete Features nach dem FeatureWeightingLayer.
                n_layers (int): Anzahl der Dense-Schichten.
                layer_units (list): Liste der Neuronenzahlen für jede Schicht.
                layer_dropout_rates (list): Liste der Dropout-Raten für jede Schicht.
                activation (str): Aktivierungsfunktion.
                optimizer (tf.keras.optimizers.Optimizer): Optimierer.
                kernel_initializer (str): Initialisierungsfunktion für die Gewichte.
                num_classes (int): Anzahl der Klassen für die Ausgangsschicht.
                feature_weighting_l2 (float): L2-Regularisierung für den FeatureWeightingLayer.
                dense_l2_reg (float): L2-Regularisierung für Dense-Schichten.
                average (str): Durchschnittsmodus für den F1-Score.

            Returns:
                tf.keras.Model: Das erstellte Modell.
            """
            # Optional: Normalisierung oder Preprocessing direkt im Modell
            # normalized_features = tf.keras.layers.LayerNormalization(name="feature_normalization")(all_features_combined)
            # x = normalized_features

            # IMPORTANT

            # Feature Weighting Layer
            # x = all_features_combined
            # x = FeatureWeightingLayer(
            #     num_features=all_features_combined.shape[-1],
            #     l2_reg=feature_weighting_l2,
            #     name='feature_weighting'
            # )(all_features_combined)

            x = FeatureWeightingLayer(l2_reg=feature_weighting_l2, name='feature_weighting')(all_features_combined)

            # Verarbeitung durch Dense Layers
            for i in range(n_layers):
                x = tf.keras.layers.Dense(
                    layer_units[i],
                    kernel_regularizer=tf.keras.regularizers.l2(dense_l2_reg),
                    kernel_initializer=kernel_initializer,
                    name=f'dense_{i}'
                )(x)
                x = tf.keras.layers.BatchNormalization(name=f'batch_norm_{i}')(x)
                x = tf.keras.layers.Activation(activation, name=f'activation_{i}')(x)  # Aktivierung außerhalb der Dense-Schicht
                x = tf.keras.layers.Dropout(layer_dropout_rates[i], name=f'dropout_{i}')(x)

            # Ausgangsschicht
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)  # Für >= 2 Klassen
            # outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid', name='output')(x)  # Für binäre Klassen

            # Modell erstellen
            model = tf.keras.Model(inputs=all_inputs, outputs=outputs)

            # Verlustfunktion (je nach Label-Typ)
            loss = 'categorical_crossentropy'  # Für One-Hot-Labels
            # loss = 'sparse_categorical_crossentropy'  # Für Integer-Labels

            # Zusätzliche Metriken (optional)
            # metrics = [tf.keras.metrics.AUC(name='auc'), 'accuracy']
            metrics = [tfa.metrics.F1Score(num_classes=num_classes, average=average, name="f1_score"), 'accuracy']

            # Modell kompilieren
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )

            return model



        def build_model_func(all_inputs, all_features_combined, n_layers, layer_units,
                             layer_dropout_rates,
                             activation, optimizer, kernel_initializer, num_classes,
                             average
                             ):
            # Baue das Modell mit allen Features
            model = build_model(
                all_inputs=all_inputs,
                all_features_combined=all_features_combined,
                n_layers=n_layers,
                layer_units=layer_units,
                layer_dropout_rates=layer_dropout_rates,
                activation=activation,
                optimizer=optimizer,
                kernel_initializer=kernel_initializer,
                num_classes=num_classes,
                feature_weighting_l2=feature_weighting_l2,
                dense_l2_reg=dense_l2_reg,  # Übergabe des neuen Parameters
                average=average
            )
            return model

        try:
            # if tuning:
            #     # Batch-Größe und Prefetching anpassen
            #     train_ds = train_ds.batch(batch_size)
            #     train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            #
            #     if test_ds is not None:
            #         test_ds = test_ds.batch(batch_size)
            #         test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            if tuning:
                # Trainingsdaten vorbereiten
                train_ds = train_ds.batch(batch_size)  # Dynamische Batchgröße
                train_ds = train_ds.cache()  # Zwischenspeicherung nach Batching
                train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Pipeline entkoppeln

                # Testdaten vorbereiten, falls vorhanden
                if test_ds is not None:
                    test_ds = test_ds.batch(batch_size)  # Dynamische Batchgröße
                    test_ds = test_ds.cache()  # Zwischenspeicherung nach Batching
                    test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Pipeline entkoppeln

                # Modell erstellen
                model = build_model_func(
                    all_inputs=all_inputs,
                    all_features_combined=all_features_combined,
                    n_layers=n_layers,
                    layer_units=layer_units,
                    layer_dropout_rates=layer_dropout_rates,
                    activation=activation,
                    optimizer=optimizer,
                    kernel_initializer=kernel_initializer,
                    num_classes=len(trend_columns),
                    average=average
                )

                # Modell trainieren
                model.fit(
                    train_ds,
                    validation_data=test_ds,
                    epochs=max_epochs,  # Maximale Anzahl der Epochen
                    # verbose=verbose,
                    verbose=1,
                    callbacks=callbacks,
                    class_weight=class_weights,  # Normalisierte Klassengewichte hinzufügen
                    workers=workers,
                    # workers=32,
                    use_multiprocessing=use_multiprocessing
                )

                if test_ds:
                    loss = model.evaluate(test_ds, verbose=verbose)
                    # return loss[0]
                    return loss[1]
                    # return loss[2]

            else:
                # # Batch-Größe und Prefetching anpassen
                # train_ds = train_ds.batch(batch_size)
                # train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

                # Trainingsdaten vorbereiten
                train_ds = train_ds.batch(batch_size)  # Dynamische Batchgröße
                train_ds = train_ds.cache()  # Zwischenspeicherung nach Batching
                train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Pipeline entkoppeln

                # Modell erstellen
                model = build_model_func(
                    all_inputs=all_inputs,
                    all_features_combined=all_features_combined,
                    n_layers=n_layers,
                    layer_units=layer_units,
                    layer_dropout_rates=layer_dropout_rates,
                    activation=activation,
                    optimizer=optimizer,
                    kernel_initializer=kernel_initializer,
                    num_classes=len(trend_columns),
                    average=average
                )

                # Modell trainieren
                model.fit(
                    train_ds,
                    epochs=max_epochs,  # Maximale Anzahl der Epochen
                    verbose=1,
                    callbacks=callbacks,
                    class_weight=class_weights,  # Normalisierte Klassengewichte hinzufügen
                    workers=workers,
                    use_multiprocessing=use_multiprocessing
                )

                return model

        except optuna.exceptions.TrialPruned as e:
            # print(f"Trial pruned: {e} with params : {{'activation': '{activation}', 'optimizer': '{optimizer_name}', 'n_layers': {n_layers}, 'layer_units': {layer_units}, 'layer_dropout_rates': {layer_dropout_rates}, 'batch_size': {batch_size}}}")
            # return 999999  # Signalisieren der Pruning-Bedingung
            return -999999  # Signalisieren der Pruning-Bedingung

        except AttributeError as e:
            # Handle the AttributeError
            print(traceback.format_exc())
            print(f"Es trat ein Fehler auf: {e}")
            # return 999999  # Signalisieren eines Fehlers
            return -999999  # Signalisieren der Pruning-Bedingung


    except Exception as e:
        # Fängt alle anderen Ausnahmen, die nicht spezifisch abgefangen wurden
        print(traceback.format_exc())
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        # return 999999  # Signalisieren eines Fehler
        return -999999  # Signalisieren der Pruning-Bedingung


def save_to_db(dataframe, to_table, db):
    # Mapping der Pandas-Datentypen auf die SQLAlchemy-Datentypen
    dtype_mapping = {
        'object': VARCHAR(255),
        'float64': Float,
        'float32': Float,
        'int64': Integer,
        'datetime64[ns]': DateTime
    }

    # Erstellen eines Dictionarys, das die Spaltennamen und ihre SQLAlchemy-Datentypen enthält
    dtype = {col: dtype_mapping[str(dataframe[col].dtype)] for col in dataframe.columns}

    # Erstellt eine Verbindung zur Datenbank
    engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://root:@localhost/{db}', echo=False)
    conn = engine.connect()
    trans = conn.begin()

    try:
        dataframe.to_sql(con=conn, name=f'{to_table}', if_exists='replace', index=False, chunksize=1000, dtype=dtype)
        trans.commit()  # Commit nur, wenn alles erfolgreich war
    except Exception as e:
        print(f"Ein Fehler trat auf: {e}")
        trans.rollback()  # Rollback im Fehlerfall
    finally:
        conn.close()

    print(f"Daten erfolgreich in Tabelle '{to_table}' gespeichert.")



def plot_stock_prices1(df, trend="Trend", test=False, x_interval_min=60, y_interval_dollar=25, additional_lines=None, secondary_y_scale=1.0, time_stay=999999):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["Datetime"], df["Close_orig"], label='Aktienkurs', color='blue')

    # trend = "Trend"

    """ '.' : point
    ',' : pixel
    'o' : circle
    'v' : triangle_down
    '^' : triangle_up
    '<' : triangle_left
    '>' : triangle_right
    '1' : tri_down
    '2' : tri_up
    '3' : tri_left
    '4' : tri_right
    's' : square
    'p' : pentagon
    '*' : star
    'h' : hexagon1
    'H' : hexagon2
    '+' : plus
    'x' : x
    'D' : diamond
    'd' : thin_diamond
    '|' : vline
    '_' : hline
    """

    # Marker für bekannte Trends setzen
    known_trends = {
        "Up": {'marker': '^', 'color': 'green', 'label': 'Up', 'markersize': 8},
        "Down": {'marker': 'v', 'color': 'red', 'label': 'Down', 'markersize': 8},
        "Stable": {'marker': 'o', 'color': 'grey', 'label': 'Stable', 'markersize': 3},
        "Fail": {'marker': '.', 'color': 'orange', 'label': 'Fail', 'markersize': 12}

    }

    # Dummy-Marker für die Legende erstellen
    markers_for_legend = {}
    for key, value in known_trends.items():
        marker, = ax1.plot([], [], marker=value['marker'], color=value['color'], markersize=value['markersize'], linestyle='None', label=value['label'])
        markers_for_legend[key] = marker

    # Automatisch Marker für unbekannte Trends setzen
    for trend_value in df[trend].unique():
        if trend_value not in known_trends:
            color = plt.cm.tab20(len(markers_for_legend) % 20)  # Automatische Farbzuweisung
            marker, = ax1.plot([], [], marker='x', color=color, markersize=6, linestyle='None', label=trend_value)
            markers_for_legend[trend_value] = marker

    # Marker für die Datenpunkte setzen
    for i in range(len(df)):
        trend_value = df[trend].iloc[i]
        if trend_value in known_trends:
            marker_info = known_trends[trend_value]
            ax1.plot(df["Datetime"].iloc[i], df["Close_orig"].iloc[i], marker=marker_info['marker'], color=marker_info['color'], markersize=marker_info['markersize'])
        else:
            ax1.plot(df["Datetime"].iloc[i], df["Close_orig"].iloc[i], marker='x', color=markers_for_legend[trend_value].get_color(), markersize=6)

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
    handles += list(markers_for_legend.values())
    labels = [h.get_label() for h in handles]
    unique_handles_labels = dict(zip(labels, handles))
    ax1.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc='upper left')

    ax1.grid(True)
    fig.autofmt_xdate()  # Automatische Formatierung der x-Achse

    if not test:
        def close_plot():
            time.sleep(time_stay)
            plt.close(fig)

        close_thread = threading.Thread(target=close_plot)
        close_thread.start()

    plt.show()


def plot_stock_prices2(df, trend="Trend", test=False, x_interval_min=60, y_interval_dollar=25, additional_lines=None, secondary_y_scale=1.0, time_stay=999999):
    """
    Plottet Aktienkurse mit Trendmarkern auf einer nicht-kontinuierlichen Zeitachse.
    Es werden nur die vorhandenen Datenpunkte gezeigt und alle vorhandenen Punkte sind durch eine Linie verbunden.
    """
    # Sortiere das DataFrame nach Datum, falls nicht bereits geschehen
    df = df.sort_values('Datetime').reset_index(drop=True)

    # Erzeuge diskrete X-Werte für jeden Datenpunkt
    x_values = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot der Datenpunkte mit durchgehender Linie
    # Da wir eine diskrete Achse nutzen, gibt es keine leeren Zeiträume auf der x-Achse
    ax1.plot(x_values, df["Close_orig"], label='Aktienkurs', color='blue', marker='o', markersize=3)

    # Bekannte Trend-Marker
    known_trends = {
        "Up": {'marker': '^', 'color': 'green', 'label': 'Up', 'markersize': 8},
        "Down": {'marker': 'v', 'color': 'red', 'label': 'Down', 'markersize': 8},
        "Stable": {'marker': 'o', 'color': 'grey', 'label': 'Stable', 'markersize': 3},
        "Fail": {'marker': '.', 'color': 'orange', 'label': 'Fail', 'markersize': 12}
    }

    # Dummy-Marker für Legende
    markers_for_legend = {}
    for key, value in known_trends.items():
        marker, = ax1.plot([], [], marker=value['marker'], color=value['color'], markersize=value['markersize'], linestyle='None', label=value['label'])
        markers_for_legend[key] = marker

    # Automatische Marker für unbekannte Trends
    unique_trends = df[trend].unique()
    for trend_value in unique_trends:
        if trend_value not in known_trends:
            color = plt.cm.tab20(len(markers_for_legend) % 20)
            marker, = ax1.plot([], [], marker='x', color=color, markersize=6, linestyle='None', label=trend_value)
            markers_for_legend[trend_value] = marker

    # Marker für die Datenpunkte je nach Trend setzen
    for i in range(len(df)):
        trend_value = df[trend].iloc[i]
        y_val = df["Close_orig"].iloc[i]
        if trend_value in known_trends:
            marker_info = known_trends[trend_value]
            ax1.plot(x_values[i], y_val, marker=marker_info['marker'], color=marker_info['color'], markersize=marker_info['markersize'])
        else:
            ax1.plot(x_values[i], y_val, marker='x', color=markers_for_legend[trend_value].get_color(), markersize=6)

    # Zusätzliche Linien auf zweiter Y-Achse
    additional_lines_handles = []
    if additional_lines:
        ax2 = ax1.twinx()
        ax2.set_frame_on(False)
        for column, multiplier in additional_lines:
            if column in df.columns:
                line, = ax2.plot(x_values, df[column] * multiplier, alpha=0.7, label=f'{column} (x{multiplier})')
                additional_lines_handles.append(line)
        ax2.set_ylim(ax2.get_ylim()[0] * secondary_y_scale, ax2.get_ylim()[1] * secondary_y_scale)

    ax1.set_title('Kurs, Trendstärke, Signale')
    ax1.set_xlabel('Datenpunkte (indexbasiert)')
    ax1.set_ylabel('Kurs')

    # Manuelle Tick-Labels für die X-Achse: Nur vorhandene Zeitpunkte anzeigen
    # Wähle ein sinnvolles Intervall für die Beschriftungen
    interval = max(1, len(df)//10)
    ax1.set_xticks(x_values[::interval])
    ax1.set_xticklabels(df["Datetime"].iloc[::interval].dt.strftime('%Y-%m-%d %H:%M'), rotation=45)

    # Y-Achse anpassen
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(y_interval_dollar))

    # Legende erstellen
    handles, labels = ax1.get_legend_handles_labels()
    handles += additional_lines_handles
    handles += list(markers_for_legend.values())
    labels = [h.get_label() for h in handles]
    unique_handles_labels = dict(zip(labels, handles))
    ax1.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc='upper left')

    ax1.grid(True)
    plt.tight_layout()

    if not test:
        def close_plot():
            time.sleep(time_stay)
            plt.close(fig)
        close_thread = threading.Thread(target=close_plot)
        close_thread.start()

    plt.show()


def plot_stock_prices(df, config=None, trend="Trend", test=False, x_interval_min=60,
                      y_interval_dollar=25, additional_lines=None, secondary_y_scale=1.0,
                      time_stay=999999,
                      save_plot_as_picture=False, folder=r"Backtest"):
    """
    Plottet Aktienkurse mit Trendmarkern auf einer nicht-kontinuierlichen Zeitachse.
    Zusätzlich werden Kauf-/Verkaufsignale sowie Stop-Loss- und Trailing-Stop-Linien dargestellt, falls im DataFrame vorhanden.
    """


    # Sortiere das DataFrame nach Datum, falls nicht bereits geschehen
    df = df.sort_values('Datetime').reset_index(drop=True)

    # Erzeuge diskrete X-Werte für jeden Datenpunkt
    x_values = np.arange(len(df))

    # fig, ax1 = plt.subplots(figsize=(10, 5))
    # fig, ax1 = plt.subplots(figsize=(20, 10))
    fig, ax1 = plt.subplots(figsize=(30, 15))
    # fig, ax1 = plt.subplots(figsize=(40, 20))
    # fig, ax1 = plt.subplots(figsize=(50, 25))
    # fig, ax1 = plt.subplots(figsize=(60, 30))
    # fig, ax1 = plt.subplots(figsize=(70, 35))
    # fig, ax1 = plt.subplots(figsize=(80, 40))

    # Plot der Datenpunkte mit durchgehender Linie
    ax1.plot(x_values, df["Close_orig"], label='Aktienkurs', color='blue', marker='.', markersize=1)

    # Bekannte Trend-Marker

    markersize = 8
    known_trends = {
        "Up": {'marker': '^', 'color': 'green', 'label': 'Up', 'markersize': markersize},
        "Down": {'marker': 'v', 'color': 'red', 'label': 'Down', 'markersize': markersize},
        # "Stable": {'marker': '.', 'color': 'grey', 'label': 'Stable', 'markersize': 0},
        "Fail": {'marker': '.', 'color': 'orange', 'label': 'Fail', 'markersize': markersize},
        "Sold": {'marker': 'v', 'color': 'orange', 'label': 'Sold', 'markersize': markersize},
        "Up|Sold": {'marker': '^', 'color': 'orange', 'label': 'Up|Sold', 'markersize': markersize}



    }

    # Dummy-Marker für Legende
    markers_for_legend = {}
    for key, value in known_trends.items():
        marker, = ax1.plot([], [], marker=value['marker'], color=value['color'], markersize=value['markersize'], linestyle='None', label=value['label'])
        markers_for_legend[key] = marker

    # Automatische Marker für unbekannte Trends
    # unique_trends = df[trend].unique()
    # for trend_value in unique_trends:
    #     if trend_value not in known_trends:
    #         color = plt.cm.tab20(len(markers_for_legend) % 20)
    #         marker, = ax1.plot([], [], marker='x', color=color, markersize=6, linestyle='None', label=trend_value)
    #         markers_for_legend[trend_value] = marker

    # Marker für die Datenpunkte je nach Trend setzen
    for i in range(len(df)):
        trend_value = df[trend].iloc[i]
        y_val = df["Close_orig"].iloc[i]
        if trend_value in known_trends:
            marker_info = known_trends[trend_value]
            ax1.plot(x_values[i], y_val, marker=marker_info['marker'], color=marker_info['color'], markersize=marker_info['markersize'])
        # else:
        #     ax1.plot(x_values[i], y_val, marker='x', color=markers_for_legend[trend_value].get_color(), markersize=6)

    # Zusätzliche Linien auf zweiter Y-Achse
    additional_lines_handles = []
    if additional_lines:
        ax2 = ax1.twinx()
        ax2.set_frame_on(False)
        for column, multiplier in additional_lines:
            if column in df.columns:
                line, = ax2.plot(x_values, df[column] * multiplier, alpha=0.7, label=f'{column} (x{multiplier})')
                additional_lines_handles.append(line)
        ax2.set_ylim(ax2.get_ylim()[0] * secondary_y_scale, ax2.get_ylim()[1] * secondary_y_scale)

    # -------------------------
    # Buy-/Sell-Signale plotten
    # -------------------------
    if "Position" in df.columns:
        # Finde Kaufpunkte (Position wechselt von 0 auf 1)
        buy_points = df.index[(df["Position"].shift(1, fill_value=0) == 0) & (df["Position"] == 1)].tolist()
        # Finde Verkaufspunkte (Position wechselt von 1 auf 0)
        sell_points = df.index[(df["Position"].shift(1, fill_value=0) == 1) & (df["Position"] == 0)].tolist()

        # Plot Buy Signale
        for idx in buy_points:
            # Kauf erfolgt zu Entry_Price (falls vorhanden), sonst Close_orig
            if "Entry_Price" in df.columns and not np.isnan(df["Entry_Price"].iloc[idx]):
                buy_price = df["Entry_Price"].iloc[idx]
            else:
                buy_price = df["Close_orig"].iloc[idx]
            ax1.plot(x_values[idx], buy_price, marker='^', color='lime', markersize=markersize, label='Buy' if idx == buy_points[0] else "")

        # Plot Sell Signale
        for idx in sell_points:
            # Verkauf zum aktuellen Kurs, da hier der Trade geschlossen wurde
            sell_price = df["Close_orig"].iloc[idx]
            ax1.plot(x_values[idx], sell_price, marker='v', color='red', markersize=markersize, label='Sell' if idx == sell_points[0] else "")

    # -------------------------------
    # Stop-Loss und Trailing-Stop plotten
    # -------------------------------
    # Wir plotten diese Linien nur, wenn sie existieren und wenn Position offen ist.
    # Stop-Loss Linie plotten
    # if "Stop_Loss" in df.columns:
    #     # Kopie erstellen, außerhalb der offenen Positionen NaN setzen
    #     stop_loss_line = df["Stop_Loss"].copy()
    #     stop_loss_line.loc[df["Position"] == 0] = np.nan
    #     # Forward-Fill, damit zwischen Kauf und Verkauf ein kontinuierlicher Wert vorhanden ist
    #     stop_loss_line = stop_loss_line.ffill()
    #     if not stop_loss_line.isnull().all():
    #         ax1.plot(x_values, stop_loss_line, '--', color='red', linewidth=1.5, label='Stop Loss', zorder=10)

    # Trailing-Stop Linie plotten
    if "Trailing_Stop" in df.columns:
        trailing_stop_line = df["Trailing_Stop"].copy()
        trailing_stop_line.loc[df["Position"] == 0] = np.nan
        trailing_stop_line = trailing_stop_line.ffill()
        if not trailing_stop_line.isnull().all():
            ax1.plot(x_values, trailing_stop_line, '--', color='orange', linewidth=1.5, label='Trailing Stop', zorder=10)

    ax1.set_title('Kurs, Trendstärke, Signale, Stop-Loss & Trailing-Stop')
    ax1.set_xlabel('Datenpunkte (indexbasiert)')
    ax1.set_ylabel('Kurs')

    # Manuelle Tick-Labels für die X-Achse: Nur vorhandene Zeitpunkte anzeigen
    interval = max(1, len(df)//10)
    if "Datetime" in df.columns:
        ax1.set_xticks(x_values[::interval])
        ax1.set_xticklabels(df["Datetime"].iloc[::interval].dt.strftime('%Y-%m-%d %H:%M'), rotation=45)

    # Y-Achse anpassen
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(y_interval_dollar))

    # Legende erstellen
    handles, labels = ax1.get_legend_handles_labels()
    handles += additional_lines_handles
    # Marker aus known_trends wurden bereits am Anfang dummy-haft angelegt
    # Füge nun Buy/Sell etc. hinzu
    # Duplikate entfernen
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    ax1.legend(unique.values(), unique.keys(), loc='upper left')
    ax1.grid(True)
    plt.tight_layout()

    if not test:
        def close_plot():
            time.sleep(time_stay)
            plt.close(fig)
        close_thread = threading.Thread(target=close_plot)
        close_thread.start()

    # plt.savefig("mein_plot.svg")
    # plt.savefig(f"{config}.svg", format="svg")
    if save_plot_as_picture:
        # plt.savefig(rf"Backtest\{config}.png", dpi=300)
        plt.savefig(rf"{folder}\{config}.png", dpi=300)

    # plt.savefig(f"{config}_600.png", dpi=600)
    # plt.savefig(f"{config}_900.png", dpi=900)
    # plt.savefig(f"{config}.pdf")

    # html_str = mpld3.fig_to_html(plt.gcf())
    # with open(f"{config}.html", "w") as f:
    #     f.write(html_str)

    # manager = plt.get_current_fig_manager()
    # try:
    #     manager.window.state('zoomed')
    # except AttributeError:
    #     try:
    #         manager.window.showMaximized()
    #     except AttributeError:
    #         print("Vollbildmodus wird von diesem Backend nicht unterstützt.")

    # IMPORTANT
    if test:
        plt.show()


def create_lagged_features(df, features, window_size=3, preserve_features=None, hide=False):
    """
    Erstellt verzögerte (lagged) Features für die angegebenen Merkmale und entfernt die ursprünglichen Merkmale,
    außer denjenigen, die in `preserve_features` angegeben sind.

    Parameter:
    - df (pd.DataFrame): Der ursprüngliche DataFrame.
    - features (list of str): Liste der Spaltennamen, für die Lagged Features erstellt werden sollen.
    - window_size (int): Anzahl der Verzögerungen (lags) pro Feature.
    - preserve_features (list of str, optional): Liste der Spaltennamen, die nach der Erstellung der Lagged Features
                                                nicht entfernt werden sollen. Standard ist None.
    - hide (bool, optional): Wenn True, wird die Fortschrittsanzeige von tqdm deaktiviert. Standard ist False.

    Rückgabe:
    - pd.DataFrame: Der erweiterte DataFrame mit Lagged Features und den beibehaltenen Spalten.
    """
    new_df = df.copy()
    lagged_data = {}

    # Erstellung der Lagged Features
    for feature in tqdm(features, disable=hide, desc="Erstelle Lagged Features"):
        for i in range(1, window_size + 1):
            lagged_col_name = f'{feature}_lag_{i}'
            lagged_data[lagged_col_name] = df[feature].shift(i)

    # Umwandlung der Lagged Features in einen DataFrame
    lagged_df = pd.DataFrame(lagged_data)

    # Hinzufügen der Lagged Features zum neuen DataFrame
    new_df = pd.concat([new_df, lagged_df], axis=1)

    # Entfernen von Zeilen mit NaN-Werten, die durch das Verschieben entstanden sind
    new_df.dropna(inplace=True)

    # Wenn keine preserve_features angegeben sind, setze sie auf eine leere Liste
    if preserve_features is None:
        preserve_features = []

    # Bestimme, welche Spalten entfernt werden sollen
    # Entferne nur die Features, die nicht in preserve_features sind
    columns_to_drop = [col for col in features if col not in preserve_features]

    # Entferne die ausgewählten Spalten
    new_df.drop(columns=columns_to_drop, inplace=True)

    return new_df


def slope_of_series(series):
    """
    Berechnet die Steigung einer linearen Regression auf den übergebenen Werten.
    series: Pandas Series oder NumPy-Array von Länge n
    return: float (die Steigung)
    """
    x = np.arange(len(series))
    y = series.values  # oder series.to_numpy()
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope


def compute_rolling_slope(series, window):
    x = np.arange(window)
    sum_x = x.sum()
    sum_x2 = (x ** 2).sum()
    denominator = window * sum_x2 - sum_x ** 2

    # Berechnung der rollenden Summen
    sum_y = series.rolling(window).sum()
    sum_xy = series.rolling(window).apply(lambda y: np.dot(x, y), raw=True)

    # Berechnung der Steigung
    slope = (window * sum_xy - sum_x * sum_y) / denominator
    return slope


def compute_rate_of_change(series, window):
    """Berechnet die Rate of Change (ROC) einer Zeitreihe über ein gegebenes Fenster."""
    return series.pct_change(periods=window) * 100




def set_indicators_2(
        df,
        future_steps,
        threshold_high,
        threshold_low,
        database_name_optuna,
        lookback_steps,
        lookback_threshold,
        threshold_high_pct,
        threshold_low_pct,
        use_percentage,
        trendfunc,
        min_cum_return,
        test,
        up_signal_mode,
        use_lookback_check,
        require_double_ups,
        offset_after_lowest,
        use_lookforward_check,
        look_forward_threshold,
        forward_steps,
        indicators,
        consecutive_negatives_lookback_steps,
        max_consecutive_negatives_lookback,
        consecutive_negatives_forward_steps,
        max_consecutive_negatives_forward,
        backwarts_shift_labels,
        consecutive_positives_lookback_steps,
        max_consecutive_positives_lookback,
        consecutive_positives_forward_steps,
        max_consecutive_positives_forward
        ):
    # Allgemeine Verarbeitung

    #TODO
    try:
        df.columns = [col.capitalize() for col in df.columns]
    except:
        pass

    print("Erstelle Close_orig")
    df["Close_orig"] = df["Close"].copy()

    if "Date" in df.columns and "Time" in df.columns:
        print("Erstelle Datetime")
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    if "Date" not in df.columns and "Time" in df.columns:
        print("Erstelle Datetime")
        df['Datetime'] = pd.to_datetime(df['Time'])

    df = df.sort_values(by='Datetime', ascending=True)

    print(f'db_head:\n{df["Datetime"].head(1).values[0]}')
    print(f'db_tail:\n{df["Datetime"].tail(1).values[0]}')


    if not test:
        # base_value_close = "Close"
        # base_value_high = "High"
        # base_value_low = "Low"

        # base_value_close = "Close_Diff"
        # base_value_high = "High_Diff"
        # base_value_low = "Low_Diff"

        base_value_close = "Close_Pct_Change"
        base_value_high = "High_Diff_Pct"
        base_value_low = "Low_Diff_Pct"


        print("Erstelle Close_Pct_Change")
        df["Close_Pct_Change"] = df["Close"].pct_change().fillna(0)

        # indicators = [
        #     "Volume",
        #
        #     "Close_Diff",
        #     "High_Diff",
        #     "Low_Diff",
        #
        #     'Close_Pct_Change',
        #     'High_Diff_Pct',  # keine Ergebnisse, mit lag kommen viele signale
        #     'Low_Diff_Pct',  # keine Ergebnisse, mit lag kommen viele signale
        #     "SMA",
        #     "EMA",
        #     "RSI",
        #     "MACD",  # scheint wichtig zu sein, bring auch als lag noch mehr signale
        #     "BB",
        #     "OBV",  # nicht unwichtig aber sehr wenige ergebnisse, selbes ergebnis wie ohne lag
        #     "ROC",
        #     "CCI",  # erzeugt signale aber evtl die falschen? mitlag werden garkeien ergebnisse mehr erzeugt
        #     "MFI",
        #
        #     "Slope_Close_Pct_Change",
        #     "Slope_Close",
        #     "Slope_SMA",
        #     "Slope_EMA",
        #     "Slope_ROC_SMA",
        #     "Slope_ROC_EMA",
        #
        #     "ATR",
        #     "ADX",  # bringt alleine keine Ergebnisse, auch mit lag nicht
        #     "STOCH",  # im lag gute ergebnisse
        #     "WILLR",  # zeigt recht viele Signale, auch in lag
        #     "STOCH_RSI",  # mittel viele Signale und gute Stellen, auch in lag
        #
        #     ]



        # print(df.columns)

        if "Volume" in indicators:
            print("Erstelle Volume")
            if "Up" in df.columns and "Down" in df.columns:
                df['Volume'] = df['Up'] + df['Down']

        if "Close_Diff" in indicators:
            print("Erstelle Close_Diff")
            df["Close_Diff"] = df["Close"].diff().fillna(0)

        # if "Close_Pct_Change" in indicators:
        #     print("Erstelle Close_Pct_Change")
        #     df["Close_Pct_Change"] = df["Close"].pct_change().fillna(0)

        if "High_Diff" in indicators:
            print("Erstelle High_Diff")
            df["High_Diff"] = df["High"] - df["Close"]

        if "Low_Diff" in indicators:
            print("Erstelle Low_Diff")
            df["Low_Diff"] = df["Close"] - df["Low"]

        if "High_Diff_Pct" in indicators:
            print("Erstelle High_Diff_Pct")
            df["High_Diff_Pct"] = (df["High"] - df["Close"]) / df["Close"].shift(1)
            df["High_Diff_Pct"] = df["High_Diff_Pct"].fillna(0)

        if "Low_Diff_Pct" in indicators:
            print("Erstelle Low_Diff_Pct")
            df["Low_Diff_Pct"] = (df["Close"] - df["Low"]) / df["Close"].shift(1)
            df["Low_Diff_Pct"] = df["Low_Diff_Pct"].fillna(0)




        window = [5, 10, 15, 20, 50, 100, 200]
        # window = [5, 10, 20, 50, 100, 200]


        sma_windows = [10, 50, 100]
        # sma_windows = [50]
        # sma_windows = window

        ema_windows = [20, 50, 100]
        # ema_windows = [20]
        # ema_windows = window

        rsi_windows = [7, 14]
        # rsi_windows = [14]
        # rsi_windows = window

        macd_fast = 12
        macd_slow = 26
        macd_signal = 9


        bb_windows = [10, 20]
        # bb_windows = [20]
        # bb_windows = window

        roc_windows = [7, 14]
        # roc_windows = [14]
        # roc_windows = window

        # cci_windows = [10, 20]
        cci_windows = [20]
        # cci_windows = window

        mfi_windows = [7, 14]
        # mfi_windows = [14]
        # mfi_windows = window

        Slope_Close_Pct_Change_windows = [5]
        # Slope_Close_Pct_Change_windows = window

        Slope_Close_windows = [5]
        # Slope_Close_windows = window

        SlopeSMA_windows = [5]
        # SlopeSMA_windows = window

        SlopeEMA_windows = [5]
        # SlopeEMA_windows = window

        Slope_ROC_SMA_windows = [5]

        Slope_ROC_EMA_windows = [5]

        atr_windows = [14, 20]
        # atr_windows = [14]
        # atr_windows = window

        adx_windows = [14, 20]
        # adx_windows = [14]
        # adx_windows = window

        stoch_windows = [(14, 3), (20, 3)]  # (Hauptfenster, Glättung)
        # stoch_windows = [(14, 3)]  # (Hauptfenster, Glättung)

        willr_windows = [14, 20]
        # willr_windows = [14]
        # willr_windows = window

        stoch_rsi_windows = [14]  # Stoch RSI
        # stoch_rsi_windows = window  # Stoch RSI


        # 1. Gleitende Durchschnitte (SMA)
        if "SMA" in indicators:
            print("Erstelle SMA")
            for window in sma_windows:
                # df[f'SMA{window}'] = ta.trend.sma_indicator(df[base_value_close], window=window)
                # df[f'SMA{window}'] = ta.trend.sma_indicator(df['Close'], window=window)
                df[f'SMA{window}'] = ta.trend.sma_indicator(df['Close_Pct_Change'], window=window)  # --> Empfehlung Prozent, da nomiert die Daten und macht sie vergleichbar


        # 2. Gleitende Durchschnitte (EMA)
        if "EMA" in indicators:
            print("Erstelle EMA")
            for window in ema_windows:
                # df[f'EMA{window}'] = ta.trend.ema_indicator(df[base_value_close], window=window)
                # df[f'EMA{window}'] = ta.trend.ema_indicator(df['Close'], window=window)
                df[f'EMA{window}'] = ta.trend.ema_indicator(df['Close_Pct_Change'], window=window)  # --> Empfehlung Prozent, da nomiert die Daten und macht sie vergleichbar


        # 3. Relative Stärke Index (RSI)
        if "RSI" in indicators:
            print("Erstelle RSI")
            for window in rsi_windows:
                # df[f'RSI{window}'] = ta.momentum.RSIIndicator(df[base_value_close], window=window).rsi()
                # df[f'RSI{window}'] = ta.momentum.RSIIndicator(df['Close'], window=window).rsi()
                df[f'RSI{window}'] = ta.momentum.RSIIndicator(df['Close_Pct_Change'], window=window).rsi()  # --> Empfehlung Prozent, da nomiert die Daten und macht sie vergleichbar


        # 4. MACD (Moving Average Convergence Divergence)
        if "MACD" in indicators:
            print("Erstelle MACD")
            # macd = ta.trend.MACD(df[base_value_close], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
            # macd = ta.trend.MACD(df['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)  # --> Empfehlung, benötigt absolute Veränderung
            macd = ta.trend.MACD(df['Close_Pct_Change'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)  # vorher war es dieser

            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()


        # 5. Bollinger Bänder
        if "BB" in indicators:
            print("Erstelle BB")
            for window in bb_windows:
                # bb = ta.volatility.BollingerBands(df[base_value_close], window=window, window_dev=2)
                bb = ta.volatility.BollingerBands(df['Close'], window=window, window_dev=2)  # --> Empfehlung, benötigt absolute Veränderung
                # bb = ta.volatility.BollingerBands(df['Close_Pct_Change'], window=window, window_dev=2)


                df[f'BB_upper{window}'] = bb.bollinger_hband()
                df[f'BB_middle{window}'] = bb.bollinger_mavg()
                df[f'BB_lower{window}'] = bb.bollinger_lband()


        # 6. On-Balance Volume (OBV)
        if "OBV" in indicators:
            print("Erstelle OBV")
            # df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df[base_value_close], df['Volume']).on_balance_volume()
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()  # --> Empfehlung, benötigt absolute Veränderung
            # df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close_Pct_Change'], df['Volume']).on_balance_volume()


        # 7. Rate of Change (ROC)
        if "ROC" in indicators:
            print("Erstelle ROC")
            for window in roc_windows:
                df[f'ROC{window}'] = ta.momentum.ROCIndicator(df['Close'], window=window).roc()  # --> Empfehlung, benötigt absolute Veränderung
                # df[f'ROC{window}'] = ta.momentum.ROCIndicator(df['Close_Pct_Change'], window=window).roc()


        # 8. Commodity Channel Index (CCI)
        if "CCI" in indicators:
            print("Erstelle CCI")
            for window in cci_windows:
                # df[f'CCI{window}'] = ta.trend.CCIIndicator(df[base_value_high], df[base_value_low], df[base_value_close], window=window).cci()
                df[f'CCI{window}'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=window).cci()  # --> Empfehlung, benötigt absolute Veränderung
                # df[f'CCI{window}'] = ta.trend.CCIIndicator(df['High_Diff_Pct'], df['Low_Diff_Pct'], df['Close_Pct_Change'], window=window).cci()


        # 9. Money Flow Index (MFI)
        if "MFI" in indicators:
            print("Erstelle MFI")
            for window in mfi_windows:
                # df[f'MFI{window}'] = ta.volume.MFIIndicator(df[base_value_high], df[base_value_low], df[base_value_close], df['Volume'], window=window).money_flow_index()
                df[f'MFI{window}'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=window).money_flow_index()  # --> Empfehlung, benötigt absolute Veränderung
                # df[f'MFI{window}'] = ta.volume.MFIIndicator(df['High_Diff_Pct'], df['Low_Diff_Pct'], df['Close_Pct_Change'], df['Volume'], window=window).money_flow_index()



        if "Slope_Close_Pct_Change" in indicators:
            print("Erstelle Slope_Close_Pct_Change")
            for window in Slope_Close_Pct_Change_windows:
                df[f'Slope_{base_value_close}_{window}'] = compute_rolling_slope(df['Close_Pct_Change'], window)


        if "Slope_Close" in indicators:
            print("Erstelle Slope_Close")
            for window in Slope_Close_windows:
                df[f'Slope_Close_{window}'] = compute_rolling_slope(df['Close'], window)



        if "Slope_SMA" in indicators:
            print("Erstelle Slope_SMA")
            for window in SlopeSMA_windows:
                # df[f'Slope_SMA10_{window}'] = compute_rolling_slope(df['SMA10'], window)
                # df[f'Slope_SMA20_{window}'] = compute_rolling_slope(df['SMA20'], window)
                df[f'Slope_SMA50_{window}'] = compute_rolling_slope(df['SMA50'], window)
                # df[f'Slope_SMA100_{window}'] = compute_rolling_slope(df['SMA100'], window)


        if "Slope_EMA" in indicators:
            print("Erstelle Slope_EMA")
            for window in SlopeEMA_windows:
                # df[f'Slope_EMA10_{window}'] = compute_rolling_slope(df['EMA10'], window)
                df[f'Slope_EMA20_{window}'] = compute_rolling_slope(df['EMA20'], window)
                # df[f'Slope_EMA50_{window}'] = compute_rolling_slope(df['EMA50'], window)
                # df[f'Slope_EMA100_{window}'] = compute_rolling_slope(df['EMA100'], window)


        if "Slope_ROC_SMA" in indicators:
            print("Erstelle ROC der Slope Indikatoren")
            for window in Slope_ROC_SMA_windows:
                df[f'Slope_ROC_SMA50_{window}'] = compute_rate_of_change(df['Slope_SMA50_5'], window=window)


        # print(df['Slope_ROC_SMA50_5'])
        # exit()

        if "Slope_ROC_EMA" in indicators:
            print("Erstelle ROC der Slope Indikatoren")
            for window in Slope_ROC_EMA_windows:
                df[f'Slope_ROC_EMA20_{window}'] = compute_rate_of_change(df['Slope_EMA20_5'], window=window)


        # 11. ATR
        if "ATR" in indicators:
            print("Erstelle ATR")
            for window in atr_windows:
                # atr = AverageTrueRange(high=df[base_value_high], low=df[base_value_low], close=df[base_value_close], window=window)
                atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=window)  # --> Empfehlung, benötigt absolute Veränderung
                # atr = AverageTrueRange(high=df['High_Diff_Pct'], low=df['Low_Diff_Pct'], close=df['Close_Pct_Change'], window=window)

                df[f'ATR{window}'] = atr.average_true_range()

        # 12. ADX
        if "ADX" in indicators:
            print("Erstelle ADX")
            for window in adx_windows:
                # adx = ADXIndicator(high=df[base_value_high], low=df[base_value_low], close=df[base_value_close], window=window)
                adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=window)  # --> Empfehlung, benötigt absolute Veränderung
                # adx = ADXIndicator(high=df['High_Diff_Pct'], low=df['Low_Diff_Pct'], close=df['Close_Pct_Change'], window=window)

                df[f'ADX{window}'] = adx.adx()
                df[f'ADX_posDI{window}'] = adx.adx_pos()
                df[f'ADX_negDI{window}'] = adx.adx_neg()

        # 13. Stochastic Oscillator
        if "STOCH" in indicators:
            print("Erstelle StochOsc")
            for (window, smooth_w) in stoch_windows:
                stoch = StochasticOscillator(
                    # high=df[base_value_high],
                    # low=df[base_value_low],
                    # close=df[base_value_close],

                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],  # --> Empfehlung, benötigt absolute Veränderung

                    # high=df['High_Diff_Pct'],
                    # low=df['Low_Diff_Pct'],
                    # close=df['Close_Pct_Change'],

                    window=window,
                    smooth_window=smooth_w
                )
                df[f'STOCHk_{window}_{smooth_w}'] = stoch.stoch()
                df[f'STOCHd_{window}_{smooth_w}'] = stoch.stoch_signal()

        # 14. Williams %R
        if "WILLR" in indicators:
            print("Erstelle Williams%R")
            for window in willr_windows:
                # willr = WilliamsRIndicator(high=df[base_value_close], low=df[base_value_low], close=df[base_value_close], lbp=window)
                willr = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=window)  # --> Empfehlung, benötigt absolute Veränderung
                # willr = WilliamsRIndicator(high=df['High_Diff_Pct'], low=df['Low_Diff_Pct'], close=df['Close_Pct_Change'], lbp=window)


                df[f'WILLR{window}'] = willr.williams_r()

        # 15. Stochastic RSI (optional)
        if "STOCH_RSI" in indicators:
            print("Erstelle StochRSI")
            for window in stoch_rsi_windows:
                # Standard: window=14, smooth1=3, smooth2=3
                stoch_rsi = StochRSIIndicator(
                    # close=df[base_value_close],
                    # close=df['Close'], # vorher war es dieser
                    close=df['Close_Pct_Change'],   # --> Empfehlung Prozent, da nomiert die Daten und macht sie vergleichbar
                    window=window,
                    smooth1=3,
                    smooth2=3
                )
                # df[f'STOCH_RSIk_{window}'] = stoch_rsi.stoch_rsi_k()
                # df[f'STOCH_RSId_{window}'] = stoch_rsi.stoch_rsi_d()
                df[f'STOCH_RSIk_{window}'] = stoch_rsi.stochrsi_k()  # statt stoch_rsi_k()
                df[f'STOCH_RSId_{window}'] = stoch_rsi.stochrsi_d()  # statt stoch_rsi_d()



        excludes = ['Open',
                    'High',
                    'Low',
                    # 'Close',
                    'Up',
                    'Down',
                    'Close_Pct_Change',
                    ]
        columns_to_delete = [x for x in excludes if x not in indicators and x in df.columns]
        df = df.drop(columns=columns_to_delete)

        # try:
        # df = df.drop(columns=['Close_Pct_Change'])


        print(f'indicators:\n {indicators}')
        #TODO
    #################################################################### Trend


    def calculate_trend_future_only1(df, threshold_high, threshold_low, future_steps):

        def max_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_max = df['Close'][index:end_index].max()
            return future_max

        def min_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_min = df['Close'][index:end_index].min()
            return future_min

        def future_close_condition(index, df, steps_into_future, current_close):
            end_index = index + steps_into_future
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_closes = df['Close'][index:end_index]
            # return all(future_closes >= current_close)
            #TODO
            return all(future_closes > current_close)


        def calculate_trend_with_threshold_low(row, df, future_steps, threshold_high, threshold_low):
            future_max = max_future_close(row.name, df, future_steps)
            future_min = min_future_close(row.name, df, future_steps)
            current_close = row['Close']
            # if (future_max - current_close) >= threshold_high and future_min >= threshold_low and future_close_condition(row.name, df, steps_into_future=12, current_close=current_close):
            # if (future_max - current_close) >= threshold_high and future_min >= threshold_low and future_close_condition(row.name, df, steps_into_future=future_steps, current_close=current_close):
            if (future_max - current_close) >= threshold_high and future_close_condition(row.name, df, steps_into_future=future_steps, current_close=current_close):
            # if (future_max - current_close) >= threshold_high and future_min >= (current_close - threshold_low):

                return "Up"
            else:
                return "Stable"

        # Anwendung der Funktion auf den DataFrame
        df['Trend'] = df.apply(lambda row: calculate_trend_with_threshold_low(row, df, future_steps, threshold_high, threshold_low), axis=1)

        return df

    def calculate_trend_future_only2(df, threshold_high, threshold_low, future_steps):
        """
        Berechnet die Trend-Spalte basierend auf zukünftigen Schlusskursen.

        Args:
            df (pd.DataFrame): DataFrame mit mindestens einer 'Close'-Spalte.
            threshold_high (float): Mindestanstieg des maximalen zukünftigen Close.
            threshold_low (float): Maximale Abwärtsbewegung des minimalen zukünftigen Close.
            future_steps (int): Anzahl der Schritte in die Zukunft.

        Returns:
            pd.DataFrame: DataFrame mit zusätzlicher 'Trend'-Spalte.
        """
        trends = []
        closes = df['Close'].values
        n = len(closes)

        for i in range(n):
            current_close = closes[i]
            end = min(i + future_steps, n)
            future_closes = closes[i:end]

            # Suche nach dem ersten Zeitpunkt, an dem Close >= current_close + threshold_high
            target = current_close + threshold_high
            try:
                target_index = np.where(future_closes >= target)[0][0] + i
                # Überprüfe, ob vor diesem Zeitpunkt Close nie < current_close - threshold_low
                if np.all(closes[i:target_index] >= (current_close - threshold_low)):
                    trends.append("Up")
                else:
                    trends.append("Stable")
            except IndexError:
                # threshold_high wird innerhalb der future_steps nicht erreicht
                trends.append("Stable")

        df['Trend'] = trends
        return df

    def calculate_trend_future_only3(df, threshold_high_percent, threshold_low_percent, future_steps):
        """
        Berechnet die Trend-Spalte basierend auf prozentualen zukünftigen Schlusskursen.

        Args:
            df (pd.DataFrame): DataFrame mit mindestens einer 'Close'-Spalte.
            threshold_high_percent (float): Mindestanstieg in Prozent (z.B. 0.01 für 1%).
            threshold_low_percent (float): Maximaler Rückgang in Prozent (z.B. 0.005 für 0.5%).
            future_steps (int): Anzahl der Schritte in die Zukunft.

        Returns:
            pd.DataFrame: DataFrame mit zusätzlicher 'Trend'-Spalte.
        """
        trends = []
        closes = df['Close'].values
        n = len(closes)

        for i in range(n):
            current_close = closes[i]
            end = min(i + future_steps, n)
            future_closes = closes[i:end]

            # Berechne prozentuale Schwellenwerte
            threshold_high = current_close * threshold_high_percent
            threshold_low = current_close * threshold_low_percent

            # Definiere das Ziel: current_close + threshold_high
            target = current_close + threshold_high

            # Finde den ersten Zeitpunkt, an dem der Schlusskurs >= target erreicht
            target_indices = np.where(future_closes >= target)[0]
            if target_indices.size > 0:
                target_index = target_indices[0] + i
                # Überprüfe, ob vor dem Erreichen des Targets der Schlusskurs nie < current_close - threshold_low gefallen ist
                if np.all(closes[i:target_index] >= (current_close - threshold_low)):
                    trends.append("Up")
                else:
                    trends.append("Stable")
            else:
                trends.append("Stable")

        df['Trend'] = trends
        return df

    def calculate_trend_future_only4(df, threshold_high, threshold_low, future_steps, lookback_steps, lookback_threshold):
        def max_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_max = df['Close'][index:end_index].max()
            return future_max

        def min_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_min = df['Close'][index:end_index].min()
            return future_min

        def future_close_condition(index, df, steps_into_future, current_close):
            end_index = index + steps_into_future
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_closes = df['Close'][index:end_index]
            return all(future_closes >= current_close)

        def violates_past_condition(index, df, current_close, lookback_steps, lookback_threshold):
            # Prüfe die letzten lookback_steps Kerzen
            start_index = max(0, index - lookback_steps)  # Falls nicht genug Vergangenheit vorhanden ist
            past_closes = df['Close'][start_index:index]
            # Wenn einer der historischen Kurse > current_close + lookback_threshold ist,
            # dann soll "Up" nicht erfüllt sein
            return (past_closes > (current_close + lookback_threshold)).any()

        def calculate_trend_with_threshold_low(row, df, future_steps, threshold_high, threshold_low, lookback_steps, lookback_threshold):
            future_max = max_future_close(row.name, df, future_steps)
            future_min = min_future_close(row.name, df, future_steps)
            current_close = row['Close']

            # Neue zusätzliche Bedingung prüfen
            if violates_past_condition(row.name, df, current_close, lookback_steps, lookback_threshold):
                return "Stable"  # Falls die Bedingung verletzt wird, kein "Up"

            # Vorhandene Bedingungen
            # if (future_max - current_close) >= threshold_high and future_min >= threshold_low and future_close_condition(row.name, df, steps_into_future=future_steps, current_close=current_close):
            # if (future_max - current_close) >= threshold_high and future_min >= (current_close - threshold_low) and future_close_condition(row.name, df, steps_into_future=future_steps, current_close=current_close):
            if (future_max - current_close) >= threshold_high and future_min >= (current_close - threshold_low):

                return "Up"
            else:
                return "Stable"

        # Anwendung der Funktion auf den DataFrame
        df['Trend'] = df.apply(lambda row: calculate_trend_with_threshold_low(row, df, future_steps, threshold_high, threshold_low, lookback_steps, lookback_threshold), axis=1)

        return df

    def calculate_trend_future_only5(df, threshold_high_percent, threshold_low_percent, future_steps, lookback_steps, lookback_threshold_percent):
        def max_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_max = df['Close'][index:end_index].max()
            return future_max

        def min_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_min = df['Close'][index:end_index].min()
            return future_min

        def future_close_condition(index, df, steps_into_future, current_close):
            end_index = index + steps_into_future
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_closes = df['Close'][index:end_index]
            return all(future_closes >= current_close)

        def violates_past_condition(index, df, current_close, lookback_steps, lookback_threshold):
            # Prüfe die letzten lookback_steps Kerzen
            start_index = max(0, index - lookback_steps)  # Falls nicht genug Vergangenheit vorhanden ist
            past_closes = df['Close'][start_index:index]
            # Wenn einer der historischen Kurse > current_close + lookback_threshold ist,
            # dann soll "Up" nicht erfüllt sein
            return (past_closes > (current_close + lookback_threshold)).any()

        def calculate_trend_with_threshold_low(row, df, future_steps, threshold_high_percent, threshold_low_percent, lookback_steps, lookback_threshold_percent):
            current_close = row['Close']

            # Berechne Schwellenwerte in absoluten Werten basierend auf Prozenten
            threshold_high = current_close * threshold_high_percent
            threshold_low = current_close * threshold_low_percent
            lookback_threshold = current_close * lookback_threshold_percent

            future_max = max_future_close(row.name, df, future_steps)
            future_min = min_future_close(row.name, df, future_steps)

            # Neue zusätzliche Bedingung prüfen
            if violates_past_condition(row.name, df, current_close, lookback_steps, lookback_threshold):
                return "Stable"  # Falls die Bedingung verletzt wird, kein "Up"

            # Vorhandene Bedingungen
            if (future_max - current_close) >= threshold_high and future_min >= (current_close - threshold_low) and future_close_condition(row.name, df, steps_into_future=future_steps, current_close=current_close):
                return "Up"
            else:
                return "Stable"

        # Anwendung der Funktion auf den DataFrame
        df['Trend'] = df.apply(lambda row: calculate_trend_with_threshold_low(row, df, future_steps, threshold_high_percent, threshold_low_percent, lookback_steps, lookback_threshold_percent), axis=1)

        return df

    def calculate_trend_future_only6(df, threshold_high, threshold_low, future_steps, lookback_steps, lookback_threshold):
        def max_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_max = df['Close'][index:end_index].max()
            return future_max

        def min_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_min = df['Close'][index:end_index].min()
            return future_min

        def future_close_condition(index, df, steps_into_future, current_close):
            """
            Diese Funktion wird in diesem Beispiel nicht genutzt,
            könnte aber weiterverwendet oder entfernt werden.
            """
            end_index = index + steps_into_future
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_closes = df['Close'][index:end_index]
            return all(future_closes >= current_close)

        def violates_past_condition(index, df, lookback_steps, lookback_threshold):
            """
            Prüfe die letzten `lookback_steps` Kerzen:
            - Ermittele die Kursveränderungen (deltas).
            - Zähle, wie viele davon positiv bzw. negativ sind.
            - Wenn der Anteil negativer deltas > lookback_threshold,
              soll 'Up' nicht erfüllt sein (=> "Stable").

            Beispiel:
            - lookback_threshold = 0.6 => bedeutet:
              Wenn in den letzten lookback_steps mehr als 60% der Kursänderungen negativ waren,
              dann markieren wir diesen Punkt als 'Stable'.
            """
            start_index = max(0, index - lookback_steps)
            # Wir nehmen die Kurse von `start_index` bis `index` (inklusive index),
            # damit wir die Veränderungen in diesem Bereich anschauen können.
            relevant_closes = df['Close'].iloc[start_index:index + 1]

            # Wenn nicht genug Punkte vorhanden sind oder nur ein einzelner Schlusskurs,
            # können wir keine Veränderungen berechnen => keine Verletzung.
            if len(relevant_closes) < 2:
                return False

            # Kursänderungen (Delta) berechnen
            # diff() erzeugt den Unterschied: deltas[i] = Close[i] - Close[i-1]
            deltas = relevant_closes.diff().dropna()
            # dropna() entfernt das erste NaN durch die Differenzbildung

            # Anzahl positiver vs. negativer Deltas
            pos_count = (deltas > 0).sum()
            neg_count = (deltas < 0).sum()
            total_changes = pos_count + neg_count

            # Falls es durch Zufall nur 0 oder 1 Delta gibt, kann total_changes = 0 sein
            if total_changes == 0:
                return False  # Keine Änderungen => keine Verletzung

            # Anteil negativer Änderungen
            ratio_neg = neg_count / total_changes

            # Liegt der Anteil negativer Kerzen über unserem Schwellwert?
            return ratio_neg > lookback_threshold

        def calculate_trend_with_threshold_low(row, df, future_steps, threshold_high, threshold_low,
                                               lookback_steps, lookback_threshold):
            future_max = max_future_close(row.name, df, future_steps)
            future_min = min_future_close(row.name, df, future_steps)
            current_close = row['Close']

            # Neue Logik: prüfe das Verhältnis von pos/neg Kursänderungen
            if violates_past_condition(row.name, df, lookback_steps, lookback_threshold):
                return "Stable"  # Falls der Anteil negativer Deltas zu hoch war => kein "Up"

            # Vorhandene Bedingungen (Beispiel für zukünftigen Trendcheck)
            # Hier: Steigt future_max mind. threshold_high über current_close?
            #       Fällt future_min nicht unter (current_close - threshold_low)?
            if (future_max - current_close) >= threshold_high and future_min >= (current_close - threshold_low):
                return "Up"
            else:
                return "Stable"

        # Anwendung der Funktion auf den DataFrame
        df['Trend'] = df.apply(
            lambda row: calculate_trend_with_threshold_low(
                row, df, future_steps, threshold_high, threshold_low,
                lookback_steps, lookback_threshold
            ),
            axis=1
        )

        return df

    def calculate_trend_future_only7(
            df,
            threshold_high,  # absoluter Schwellwert für future_max
            threshold_low,  # absoluter Schwellwert für future_min
            threshold_high_pct,  # prozentualer Schwellwert für future_max
            threshold_low_pct,  # prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            use_percentage=False
    ):
        """
        Berechnet den Trend pro Zeile ('Up' oder 'Stable').

        Parameter:
        -----------
        df : pandas.DataFrame
            DataFrame mit mindestens einer 'Close'-Spalte.
        threshold_high : float
            Absoluter Wert, um festzulegen, wie hoch future_max über current_close liegen muss.
        threshold_low : float
            Absoluter Wert, wie stark future_min maximal unter current_close liegen darf.
        threshold_high_pct : float
            Prozentualer Wert, z. B. 0.1 (= 10%), wie hoch future_max über current_close liegen muss.
        threshold_low_pct : float
            Prozentualer Wert, z. B. 0.05 (= 5%), wie stark future_min maximal unter current_close liegen darf.
        future_steps : int
            Wie viele Kerzen (Zeilen) in die Zukunft geschaut wird.
        lookback_steps : int
            Wie viele Kerzen (Zeilen) in die Vergangenheit geschaut wird.
        lookback_threshold : float
            Schwellwert für den Anteil negativer Deltas, ab dem ein "Up" ausgeschlossen wird.
        use_percentage : bool
            False => threshold_high, threshold_low werden genutzt (absolute Werte).
            True  => threshold_high_pct, threshold_low_pct werden genutzt (prozentuale Werte).
        """

        def max_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_max = df['Close'][index:end_index].max()
            return future_max

        def min_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_min = df['Close'][index:end_index].min()
            return future_min

        def violates_past_condition(index, df, lookback_steps, lookback_threshold):
            """
            Prüfe die letzten `lookback_steps` Kerzen:
            - Ermittele die Kursveränderungen (deltas).
            - Zähle, wie viele davon positiv bzw. negativ sind.
            - Wenn der Anteil negativer deltas > lookback_threshold,
              soll 'Up' nicht erfüllt sein (=> "Stable").
            """
            start_index = max(0, index - lookback_steps)
            relevant_closes = df['Close'].iloc[start_index:index + 1]

            # Zu wenige Daten => Keine Aussage, also kein Verstoß
            if len(relevant_closes) < 2:
                return False

            # Differenzen berechnen (Delta[i] = Close[i] - Close[i-1])
            deltas = relevant_closes.diff().dropna()
            pos_count = (deltas > 0).sum()
            neg_count = (deltas < 0).sum()
            total_changes = pos_count + neg_count

            if total_changes == 0:
                return False  # keine Kursbewegungen => kein Verstoß

            ratio_neg = neg_count / total_changes
            return ratio_neg > lookback_threshold

        def calculate_trend(row):
            future_max = max_future_close(row.name, df, future_steps)
            future_min = min_future_close(row.name, df, future_steps)
            current_close = row['Close']

            # 1) Vergangenheits-Prüfung (Anteil negativer Kursänderungen)
            if violates_past_condition(row.name, df, lookback_steps, lookback_threshold):
                return "Stable"

            # 2) Zukunfts-Bedingungen
            if use_percentage:
                # => Schwellwerte prozentual interpretieren:
                #    (future_max - current_close) / current_close >= threshold_high_pct
                #    future_min >= current_close * (1 - threshold_low_pct)
                condition_high = ((future_max - current_close) / current_close) >= threshold_high_pct
                condition_low = (future_min >= current_close * (1 - threshold_low_pct))
            else:
                # => Schwellwerte absolut interpretieren:
                #    (future_max - current_close) >= threshold_high
                #    future_min >= (current_close - threshold_low)
                condition_high = (future_max - current_close) >= threshold_high
                condition_low = (future_min >= (current_close - threshold_low))

            if condition_high and condition_low:
                return "Up"
            else:
                return "Stable"

        # Anwendung der Funktion auf den DataFrame
        df['Trend'] = df.apply(lambda row: calculate_trend(row), axis=1)

        return df



    def calculate_trend_future_only8(
            df,
            threshold_high,  # absoluter Schwellwert für future_max
            threshold_low,  # absoluter Schwellwert für future_min
            threshold_high_pct,  # prozentualer Schwellwert für future_max
            threshold_low_pct,  # prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            use_percentage=False
    ):
        """
        Berechnet den Trend pro Zeile ('Up' oder 'Stable') -- aber so,
        dass nur der *erste* Tag innerhalb eines Fensters als "Up" markiert wird
        und wir Downtrends in der Vergangenheit ausschließen.

        Parameter:
        -----------
        df : pandas.DataFrame
            DataFrame mit mindestens einer 'Close'-Spalte.
        threshold_high : float
            Absoluter Wert, um festzulegen, wie hoch future_max über current_close liegen muss.
        threshold_low : float
            Absoluter Wert, wie stark future_min maximal unter current_close liegen darf.
        threshold_high_pct : float
            Prozentualer Wert, z. B. 0.1 (= 10%), wie hoch future_max über current_close liegen muss.
        threshold_low_pct : float
            Prozentualer Wert, z. B. 0.05 (= 5%), wie stark future_min maximal unter current_close liegen darf.
        future_steps : int
            Wie viele Kerzen (Zeilen) in die Zukunft geschaut wird.
        lookback_steps : int
            Wie viele Kerzen (Zeilen) in die Vergangenheit geschaut wird.
        lookback_threshold : float
            Schwellwert für den Anteil negativer Deltas, ab dem ein "Up" ausgeschlossen wird.
        use_percentage : bool
            False => threshold_high, threshold_low werden genutzt (absolute Werte).
            True  => threshold_high_pct, threshold_low_pct werden genutzt (prozentuale Werte).
        """

        # Hilfsfunktionen ---------------------------------------------------------

        def max_future_close(index, period):
            end_index = min(index + period, len(df))
            return df['Close'].iloc[index:end_index].max()

        def min_future_close(index, period):
            end_index = min(index + period, len(df))
            return df['Close'].iloc[index:end_index].min()

        def violates_past_condition(index, lookback_steps, lookback_threshold):
            """
            Prüfe die letzten `lookback_steps` Kerzen:
            - Ermittele die Kursveränderungen (deltas).
            - Zähle, wie viele davon positiv bzw. negativ sind.
            - Wenn der Anteil negativer deltas > lookback_threshold,
              soll 'Up' nicht erfüllt sein (=> "Stable").
            """
            start_index = max(0, index - lookback_steps)
            relevant_closes = df['Close'].iloc[start_index:index + 1]

            # Zu wenige Daten => Keine Aussage, also kein Verstoß
            if len(relevant_closes) < 2:
                return False

            deltas = relevant_closes.diff().dropna()
            pos_count = (deltas > 0).sum()
            neg_count = (deltas < 0).sum()
            total_changes = pos_count + neg_count
            if total_changes == 0:
                return False  # keine Kursbewegungen => kein Verstoß

            ratio_neg = neg_count / total_changes
            return ratio_neg > lookback_threshold

        def meets_future_condition(index, future_steps):
            """
            Prüfe, ob (future_max - current_close) >= threshold_high (oder threshold_high_pct)
            und future_min >= (current_close - threshold_low) (oder prozentual).
            """
            current_close = df['Close'].iloc[index]
            f_max = max_future_close(index, future_steps)
            f_min = min_future_close(index, future_steps)

            if use_percentage:
                # prozentual
                cond_high = ((f_max - current_close) / current_close) >= threshold_high_pct
                cond_low = f_min >= current_close * (1 - threshold_low_pct)
            else:
                # absolut
                cond_high = (f_max - current_close) >= threshold_high
                cond_low = (f_min >= current_close - threshold_low)

            return cond_high and cond_low

        # ------------------------------------------------------------------------
        # Hauptlogik: Nur den *ersten* Tag innerhalb eines Fensters als Up labeln
        # ------------------------------------------------------------------------

        # Wir erstellen zuerst eine Spalte "Trend" mit "Stable"
        df['Trend'] = "Stable"

        i = 0
        last_up_index = -9999  # fiktiver Startwert, sehr kleiner Index

        while i < len(df):
            # Prüfe, ob wir erst ab hier wieder labeln dürfen
            # (vermeidet mehrfache "Up"-Signale in kurzer Zeit).
            if i < last_up_index + future_steps:
                # Wir haben gerade ein Signal gegeben -> für die nächsten future_steps kein neues Signal
                df.at[i, 'Trend'] = "Stable"
                i += 1
                continue

            # 1) Prüfen: Vergangenheitsbedingung (keine starke Downphase)
            if violates_past_condition(i, lookback_steps, lookback_threshold):
                df.at[i, 'Trend'] = "Stable"
                i += 1
                continue

            # 2) Prüfen: meets_future_condition
            if meets_future_condition(i, future_steps):
                # -> Label als "Up"
                df.at[i, 'Trend'] = "Up"
                last_up_index = i
            else:
                df.at[i, 'Trend'] = "Stable"

            i += 1

        return df

    def calculate_trend_future_only9(
            df,
            threshold_high,  # Absoluter Schwellwert für future_max
            threshold_low,  # Absoluter Schwellwert für future_min
            threshold_high_pct,  # Prozentualer Schwellwert für future_max
            threshold_low_pct,  # Prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            use_percentage=False,
            up_signal_mode="none",  # NEUER Parameter für Up-Signal-Mode
            use_lookback_check=True  # NEUER Parameter, um die Lookback-Prüfung ein-/auszuschalten
    ):
        """
        Berechnet den Trend pro Zeile ('Up' oder 'Stable') und passt danach ggf. die 'Up'-Signale an.

        Parameter:
        -----------
        df : pandas.DataFrame
            DataFrame mit mindestens einer 'Close'-Spalte.
        threshold_high : float
            Absoluter Wert, um festzulegen, wie hoch future_max über current_close liegen muss.
        threshold_low : float
            Absoluter Wert, wie stark future_min maximal unter current_close liegen darf.
        threshold_high_pct : float
            Prozentualer Wert, z. B. 0.1 (= 10%), wie hoch future_max über current_close liegen muss.
        threshold_low_pct : float
            Prozentualer Wert, z. B. 0.05 (= 5%), wie stark future_min maximal unter current_close liegen darf.
        future_steps : int
            Wie viele Kerzen (Zeilen) in die Zukunft geschaut wird.
        lookback_steps : int
            Wie viele Kerzen (Zeilen) in die Vergangenheit geschaut wird.
        lookback_threshold : float
            Schwellwert für den Anteil negativer Deltas, ab dem ein "Up" ausgeschlossen wird.
        use_percentage : bool
            False => threshold_high, threshold_low werden genutzt (absolute Werte).
            True  => threshold_high_pct, threshold_low_pct werden genutzt (prozentuale Werte).
        up_signal_mode : str
            Steuerung, wie zusammenhängende "Up"-Signale nachträglich angepasst werden:
              - "none": Keine Änderung.
              - "first_only": Nur das erste 'Up' in einer Kette bleibt bestehen.
              - "lowest_only": Nur das 'Up' mit dem tiefsten Kurs in einer Kette bleibt bestehen.
              - "lowest_plus_one": Nur das 'Up' direkt nach dem tiefsten Kurs (wenn Kurs dort höher).
              - "all_after_lowest": Alle 'Up'-Signale nach dem tiefsten Kurs bleiben erhalten
                                    (sofern die nächste Kerze höher ist als das Tief).
        use_lookback_check : bool
            Wenn True, wird die Prüfung mit lookback_steps/lookback_threshold ausgeführt.
            Wenn False, wird diese Prüfung übersprungen.
        """

        def max_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_max = df['Close'][index:end_index].max()
            return future_max

        def min_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_min = df['Close'][index:end_index].min()
            return future_min

        def violates_past_condition(index, df, lookback_steps, lookback_threshold):
            """
            Prüfe die letzten `lookback_steps` Kerzen:
            - Ermittele die Kursveränderungen (deltas).
            - Zähle, wie viele davon positiv bzw. negativ sind.
            - Wenn der Anteil negativer deltas > lookback_threshold,
              soll 'Up' nicht erfüllt sein (=> "Stable").
            """
            start_index = max(0, index - lookback_steps)
            relevant_closes = df['Close'].iloc[start_index:index + 1]

            # Zu wenige Daten => Keine Aussage, also kein Verstoß
            if len(relevant_closes) < 2:
                return False

            # Differenzen berechnen (Delta[i] = Close[i] - Close[i-1])
            deltas = relevant_closes.diff().dropna()
            pos_count = (deltas > 0).sum()
            neg_count = (deltas < 0).sum()
            total_changes = pos_count + neg_count

            if total_changes == 0:
                return False  # keine Kursbewegungen => kein Verstoß

            ratio_neg = neg_count / total_changes
            return ratio_neg > lookback_threshold

        def calculate_trend(row):
            future_max = max_future_close(row.name, df, future_steps)
            future_min = min_future_close(row.name, df, future_steps)
            current_close = row['Close']

            # (A) Vergangenheits-Prüfung (nur wenn use_lookback_check == True)
            if use_lookback_check:
                if violates_past_condition(row.name, df, lookback_steps, lookback_threshold):
                    return "Stable"

            # (B) Zukunfts-Bedingungen
            if use_percentage:
                # => Schwellwerte prozentual interpretieren:
                condition_high = ((future_max - current_close) / current_close) >= threshold_high_pct
                condition_low = (future_min >= current_close * (1 - threshold_low_pct))
            else:
                # => Schwellwerte absolut interpretieren:
                condition_high = (future_max - current_close) >= threshold_high
                condition_low = (future_min >= (current_close - threshold_low))

            if condition_high and condition_low:
                return "Up"
            else:
                return "Stable"

        # 1) Hauptberechnung: Spalte 'Trend'
        df['Trend'] = df.apply(calculate_trend, axis=1)

        # 2) Nachträgliche Anpassung zusammenhängender "Up"-Signale basierend auf up_signal_mode
        if up_signal_mode != "none":
            up_chains = []
            current_chain = []

            # Finde Ketten aufeinander folgender "Up"-Signale
            for i in range(len(df)):
                if df['Trend'].iloc[i] == 'Up':
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            # Falls am Ende noch eine Kette "offen" ist
            if current_chain:
                up_chains.append(current_chain)

            def get_min_close_index(chain_indices):
                min_price = df['Close'].iloc[chain_indices[0]]
                min_idx = chain_indices[0]
                for idx in chain_indices:
                    price = df['Close'].iloc[idx]
                    if price < min_price:
                        min_price = price
                        min_idx = idx
                return min_idx

            # Bearbeite jede Kette nach den gewünschten Regeln
            for chain in up_chains:
                # Hat die Kette nur ein einziges Up-Signal?
                if len(chain) == 1:
                    # => Keine Änderung
                    continue

                if up_signal_mode == "first_only":
                    # Nur den ersten Up behalten, Rest "Stable"
                    first = chain[0]
                    for idx in chain[1:]:
                        df.at[idx, 'Trend'] = "Stable"

                elif up_signal_mode == "lowest_only":
                    # Nur den Up mit dem tiefsten Kurs behalten
                    min_idx = get_min_close_index(chain)
                    for idx in chain:
                        if idx != min_idx:
                            df.at[idx, 'Trend'] = "Stable"

                elif up_signal_mode == "lowest_plus_one":
                    # Nur das Up-Signal direkt NACH dem Tiefpunkt (wenn Kurs dort höher)
                    min_idx = get_min_close_index(chain)
                    chain_pos_of_min = chain.index(min_idx)
                    next_pos = chain_pos_of_min + 1

                    if next_pos < len(chain):
                        next_idx = chain[next_pos]
                        # Prüfen, ob der Kurs beim nächsten Signal höher ist als am Tief
                        if df['Close'].iloc[next_idx] > df['Close'].iloc[min_idx]:
                            # Nur diesen 'Up' behalten
                            for idx in chain:
                                if idx != next_idx:
                                    df.at[idx, 'Trend'] = "Stable"
                        else:
                            # Kein höherer Kurs => gesamte Kette verwerfen
                            for idx in chain:
                                df.at[idx, 'Trend'] = "Stable"
                    else:
                        # Kein Folgesignal nach Tiefpunkt => verwerfen
                        for idx in chain:
                            df.at[idx, 'Trend'] = "Stable"

                elif up_signal_mode == "all_after_lowest":
                    """
                    Behalte alle Signale NACH dem Tag mit dem tiefsten Kurs in der Kette,
                    sofern die nächste Candle nach dem Tief auch wirklich höher liegt.
                    Entferne alle Signale bis (einschl.) zum tiefsten Kurs.
                    """
                    min_idx = get_min_close_index(chain)
                    chain_pos_of_min = chain.index(min_idx)

                    if chain_pos_of_min < len(chain) - 1:
                        # Prüfe, ob Kurs beim Signal nach dem Tief > Tief-Kurs
                        next_idx = chain[chain_pos_of_min + 1]
                        if df['Close'].iloc[next_idx] > df['Close'].iloc[min_idx]:
                            # Entferne Signale bis zum Tief (einschl.)
                            for pos_in_chain, idx in enumerate(chain):
                                if pos_in_chain <= chain_pos_of_min:
                                    df.at[idx, 'Trend'] = "Stable"
                                # alle danach bleiben Up
                        else:
                            # Kein höherer Kurs => verwerfe gesamte Kette
                            for idx in chain:
                                df.at[idx, 'Trend'] = "Stable"
                    else:
                        # Tiefpunkt ist am Ende => keine Folge-Signale => verwerfen
                        for idx in chain:
                            df.at[idx, 'Trend'] = "Stable"

        return df

    def calculate_trend_future_only10(
            df,
            threshold_high,  # Absoluter Schwellwert für future_max
            threshold_low,  # Absoluter Schwellwert für future_min
            threshold_high_pct,  # Prozentualer Schwellwert für future_max
            threshold_low_pct,  # Prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            use_percentage=False,
            up_signal_mode="none",  # Steuerung für die nachträgliche "Up"-Anpassung
            use_lookback_check=True,
            require_double_ups=False,  # NEUER Parameter: Einzelne 'Up's -> 'Stable'?
            offset_after_lowest=0  # NEUER Parameter für all_after_lowest_offset
    ):
        """
        Berechnet den Trend pro Zeile ('Up' oder 'Stable') und nimmt danach ggf. eine nachträgliche
        Anpassung der 'Up'-Signale vor.

        Parameter:
        -----------
        df : pandas.DataFrame
            DataFrame mit mindestens einer 'Close'-Spalte.
        threshold_high : float
            Absoluter Wert, wie hoch future_max über current_close liegen muss.
        threshold_low : float
            Absoluter Wert, wie stark future_min maximal unter current_close liegen darf.
        threshold_high_pct : float
            Prozentualer Wert, wie hoch future_max über current_close liegen muss.
        threshold_low_pct : float
            Prozentualer Wert, wie stark future_min maximal unter current_close liegen darf.
        future_steps : int
            Wie viele Kerzen (Zeilen) in die Zukunft geschaut wird.
        lookback_steps : int
            Wie viele Kerzen (Zeilen) in die Vergangenheit geschaut wird.
        lookback_threshold : float
            Schwellwert für den Anteil negativer Deltas, ab dem ein "Up" ausgeschlossen wird.
        use_percentage : bool
            False => threshold_high, threshold_low werden genutzt (absolute Werte).
            True  => threshold_high_pct, threshold_low_pct werden genutzt (prozentuale Werte).
        up_signal_mode : str
            Steuerung, wie zusammenhängende "Up"-Signale nachträglich angepasst werden:
              - "none": Keine Änderung.
              - "first_only": Nur das erste 'Up' in einer Kette bleibt bestehen.
              - "lowest_only": Nur das 'Up' mit dem tiefsten Kurs in einer Kette bleibt bestehen.
              - "lowest_plus_one": Nur das 'Up' direkt nach dem tiefsten Kurs (wenn Kurs dort höher).
              - "all_after_lowest": Alle 'Up'-Signale nach dem tiefsten Kurs bleiben erhalten.
              - "all_after_lowest_offset": Wie 'all_after_lowest', aber abzüglich einer
                                           bestimmten Anzahl Signale nach dem Tiefpunkt.
        use_lookback_check : bool
            Wenn True, wird die Prüfung mit lookback_steps/lookback_threshold ausgeführt.
            Wenn False, wird diese Prüfung übersprungen.
        require_double_ups : bool
            Wenn True, werden einzelne Up-Signale (Kettenlänge == 1) in "Stable" umgewandelt.
            Mindestens zwei aufeinanderfolgende Up-Signale bleiben "Up".
        offset_after_lowest : int
            Wird nur im Modus "all_after_lowest_offset" genutzt. Gibt an, wie viele Signale
            ab dem tiefsten Signal in einer Kette noch auf "Stable" gesetzt werden sollen,
            bevor alle weiteren auf "Up" belassen werden.
        """

        # --- Validierung neuer Parameter ---
        if offset_after_lowest < 0:
            raise ValueError("offset_after_lowest muss >= 0 sein.")

        def max_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_max = df['Close'][index:end_index].max()
            return future_max

        def min_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)  # Verhindert Indexfehler am DataFrame-Ende
            future_min = df['Close'][index:end_index].min()
            return future_min

        def violates_past_condition(index, df, lookback_steps, lookback_threshold):
            """
            Prüfe die letzten `lookback_steps` Kerzen:
            - Ermittele die Kursveränderungen (deltas).
            - Zähle, wie viele davon positiv bzw. negativ sind.
            - Wenn der Anteil negativer deltas > lookback_threshold,
              soll 'Up' nicht erfüllt sein (=> "Stable").
            """
            start_index = max(0, index - lookback_steps)
            relevant_closes = df['Close'].iloc[start_index:index + 1]

            # Zu wenige Daten => Keine Aussage, also kein Verstoß
            if len(relevant_closes) < 2:
                return False

            # Differenzen berechnen (Delta[i] = Close[i] - Close[i-1])
            deltas = relevant_closes.diff().dropna()
            pos_count = (deltas > 0).sum()
            neg_count = (deltas < 0).sum()
            total_changes = pos_count + neg_count

            if total_changes == 0:
                return False  # keine Kursbewegungen => kein Verstoß

            ratio_neg = neg_count / total_changes
            return ratio_neg > lookback_threshold

        def calculate_trend(row):
            future_max = max_future_close(row.name, df, future_steps)
            future_min = min_future_close(row.name, df, future_steps)
            current_close = row['Close']

            # (A) Vergangenheits-Prüfung (nur wenn use_lookback_check == True)
            if use_lookback_check:
                if violates_past_condition(row.name, df, lookback_steps, lookback_threshold):
                    return "Stable"

            # (B) Zukunfts-Bedingungen
            if use_percentage:
                # => Schwellwerte prozentual interpretieren:
                condition_high = ((future_max - current_close) / current_close) >= threshold_high_pct
                condition_low = (future_min >= current_close * (1 - threshold_low_pct))
            else:
                # => Schwellwerte absolut interpretieren:
                condition_high = (future_max - current_close) >= threshold_high
                condition_low = (future_min >= (current_close - threshold_low))

            if condition_high and condition_low:
                return "Up"
            else:
                return "Stable"

        # 1) Hauptberechnung: Spalte 'Trend'
        df['Trend'] = df.apply(calculate_trend, axis=1)

        # 2) Nachträgliche Anpassung zusammenhängender "Up"-Signale basierend auf up_signal_mode
        if up_signal_mode != "none":
            up_chains = []
            current_chain = []

            # Finde Ketten aufeinanderfolgender "Up"-Signale
            for i in range(len(df)):
                if df['Trend'].iloc[i] == 'Up':
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            # Falls am Ende noch eine Kette "offen" ist
            if current_chain:
                up_chains.append(current_chain)

            def get_min_close_index(chain_indices):
                """Finde den Index mit dem tiefsten 'Close' innerhalb der Kette."""
                min_price = df['Close'].iloc[chain_indices[0]]
                min_idx = chain_indices[0]
                for idx in chain_indices:
                    price = df['Close'].iloc[idx]
                    if price < min_price:
                        min_price = price
                        min_idx = idx
                return min_idx

            for chain in up_chains:
                # Hat die Kette nur ein einziges Up-Signal?
                if len(chain) == 1:
                    # Für fast alle Modi ändert das nichts (das Signal bleibt Up),
                    # außer später ggf. durch require_double_ups.
                    continue

                # Haben wir Ketten mit >= 2 Signalen, wende den gewünschten Modus an
                if up_signal_mode == "first_only":
                    # Nur den ersten Up behalten, Rest "Stable"
                    first = chain[0]
                    for idx in chain[1:]:
                        df.at[idx, 'Trend'] = "Stable"

                elif up_signal_mode == "lowest_only":
                    # Nur den Up mit dem tiefsten Kurs behalten
                    min_idx = get_min_close_index(chain)
                    for idx in chain:
                        if idx != min_idx:
                            df.at[idx, 'Trend'] = "Stable"

                elif up_signal_mode == "lowest_plus_one":
                    # Nur das Up-Signal direkt NACH dem Tiefpunkt (wenn Kurs dort höher)
                    min_idx = get_min_close_index(chain)
                    chain_pos_of_min = chain.index(min_idx)
                    next_pos = chain_pos_of_min + 1

                    if next_pos < len(chain):
                        next_idx = chain[next_pos]
                        # Prüfen, ob der Kurs an next_idx > Kurs an min_idx
                        if df['Close'].iloc[next_idx] > df['Close'].iloc[min_idx]:
                            # Nur diesen 'Up' behalten
                            for idx in chain:
                                if idx != next_idx:
                                    df.at[idx, 'Trend'] = "Stable"
                        else:
                            # Kein höherer Kurs => gesamte Kette verwerfen
                            for idx in chain:
                                df.at[idx, 'Trend'] = "Stable"
                    else:
                        # Kein Folgesignal => gesamte Kette verwerfen
                        for idx in chain:
                            df.at[idx, 'Trend'] = "Stable"

                elif up_signal_mode == "all_after_lowest":
                    """
                    Alle Signale vor und inkl. tiefstem Signal => Stable.
                    Alles danach => Up (sofern next_idx noch existiert und ggf. höher ist).
                    Hier laut ursprünglicher Logik oft mit Kursprüfung verbunden,
                    aber wenn du das nicht mehr willst, kannst du die Prüfung entfernen.
                    """
                    min_idx = get_min_close_index(chain)
                    chain_pos_of_min = chain.index(min_idx)
                    # Rest wie bisher:
                    # => Signale bis zum Tiefpunkt inkl. auf Stable
                    for pos_in_chain, idx in enumerate(chain):
                        if pos_in_chain <= chain_pos_of_min:
                            df.at[idx, 'Trend'] = "Stable"
                        else:
                            df.at[idx, 'Trend'] = "Up"

                elif up_signal_mode == "all_after_lowest_offset":
                    """
                    Wie 'all_after_lowest', aber mit zusätzlichem Offset.

                    - offset_after_lowest = 0:
                        Alle Signale ab dem tiefsten (einschließlich Tiefpunkt) bleiben Up.
                    - offset_after_lowest = 1:
                        Der Tiefpunkt wird Stable,
                        ab dem 1. Signal nach dem Tiefpunkt bleiben alle Up.
                    - offset_after_lowest = 2:
                        Tiefpunkt und das erste Signal danach => Stable,
                        ab dem 2. Signal nach dem Tiefpunkt => Up.
                    - usw.
                    """
                    min_idx = get_min_close_index(chain)
                    chain_pos_of_min = chain.index(min_idx)

                    for pos_in_chain, idx in enumerate(chain):
                        if pos_in_chain < chain_pos_of_min:
                            # Signale VOR dem Tiefpunkt => Stable
                            df.at[idx, 'Trend'] = "Stable"
                        else:
                            # Wir sind ab dem Tiefpunkt
                            # Bis "tiefster + offset" => Stable
                            if pos_in_chain < chain_pos_of_min + offset_after_lowest:
                                df.at[idx, 'Trend'] = "Stable"
                            else:
                                # Ab chain_pos_of_min + offset_after_lowest => Up
                                df.at[idx, 'Trend'] = "Up"

        # 3) Zweite nachträgliche Anpassung: require_double_ups
        #    => Falls aktiviert, werden einzelne Up-Signale in "Stable" umgewandelt.
        if require_double_ups:
            # Neue Suche nach Up-Ketten (weil sich durch obige Modi ggf. etwas geändert haben kann)
            up_chains = []
            current_chain = []
            for i in range(len(df)):
                if df['Trend'].iloc[i] == 'Up':
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            # Falls am Ende noch eine Kette "offen" ist
            if current_chain:
                up_chains.append(current_chain)

            # Ketten mit nur 1 Up-Signal => Stable
            for chain in up_chains:
                if len(chain) == 1:
                    df.at[chain[0], 'Trend'] = "Stable"

        return df

    def calculate_trend_robust_11(
            df,
            threshold_high,  # Absoluter Schwellwert für future_max
            threshold_low,  # Absoluter Schwellwert für future_min
            threshold_high_pct,  # Prozentualer Schwellwert für future_max
            threshold_low_pct,  # Prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            min_cum_return,  # Minimaler kumulativer Return im Lookback-Zeitraum
            use_percentage=False,
            up_signal_mode="all_after_lowest_offset",
            use_lookback_check=True,
            require_double_ups=True,
            offset_after_lowest=2
    ):
        """
        Diese Funktion markiert Punkte als "Up", sofern sowohl die historische als auch die zukünftige
        Dynamik auf einen signifikanten Aufwärtstrend hinweist. Es werden folgende Kriterien berücksichtigt:

        1. Historische Prüfung:
           - Es wird ein Lookback-Zeitraum betrachtet, in dem der Anteil negativer Kursbewegungen
             (z. B. über 'lookback_steps') nicht zu hoch sein darf (definiert über 'lookback_threshold').
           - Zusätzlich muss der kumulative Return im betrachteten Zeitraum mindestens 'min_cum_return'
             betragen. Damit wird sichergestellt, dass der Punkt nicht lediglich eine kurzfristige Erholung
             innerhalb eines dominanten Abwärtstrends darstellt.

        2. Zukunftsprüfung:
           - Innerhalb von 'future_steps' muss der Kurs einen signifikanten Anstieg verzeichnen,
             gemessen an absoluten oder prozentualen Schwellenwerten.

        3. Nachträgliche Signal-Nachbearbeitung:
           - Zusammenhängende Up-Signale werden anhand verschiedener Modi (z. B. "all_after_lowest_offset")
             weiter gefiltert, sodass nur die relevanten Signale – typischerweise jene, die nach einem Tiefpunkt
             und mit einem bestimmten Offset auftreten – als "Up" belassen werden.
           - Zusätzlich werden isolierte Up-Signale (bei denen nur ein einzelner Punkt als Up markiert ist)
             verworfen, sofern 'require_double_ups' aktiviert ist.

        Rückgabe:
           Der DataFrame mit einer zusätzlichen Spalte 'Trend', in der jeder Punkt entweder "Up" oder "Stable" ist.
        """
        if offset_after_lowest < 0:
            raise ValueError("offset_after_lowest muss >= 0 sein.")

        def max_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)
            return df['Close'][index:end_index].max()

        def min_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)
            return df['Close'][index:end_index].min()

        def check_historical_context(index, df, lookback_steps, lookback_threshold, min_cum_return):
            """
            Überprüft, ob der Punkt an der gegebenen Stelle in einem stabilen historischen Kontext liegt.
            Es werden zwei Kriterien geprüft:
              - Der Anteil negativer Kursbewegungen im Lookback-Zeitraum darf den Schwellenwert 'lookback_threshold'
                nicht überschreiten.
              - Der kumulative Return im Lookback-Zeitraum muss mindestens 'min_cum_return' betragen.
            """
            start_index = max(0, index - lookback_steps)
            relevant_closes = df['Close'].iloc[start_index:index + 1]

            if len(relevant_closes) < 2:
                return True  # Nicht genügend Daten => keine Einschränkung

            # Berechnung der Differenzen und des negativen Anteils
            deltas = relevant_closes.diff().dropna()
            pos_count = (deltas > 0).sum()
            neg_count = (deltas < 0).sum()
            total_changes = pos_count + neg_count
            ratio_neg = neg_count / total_changes if total_changes > 0 else 0

            # Berechnung des kumulativen Returns
            cum_return = (relevant_closes.iloc[-1] - relevant_closes.iloc[0]) / relevant_closes.iloc[0]

            if ratio_neg > lookback_threshold or cum_return < min_cum_return:
                return False
            return True

        def calculate_trend_for_row(row):
            current_close = row['Close']
            idx = row.name

            # Historische Prüfung: Stelle sicher, dass der Punkt nicht aus einer reinen Erholungsphase stammt
            if use_lookback_check:
                if not check_historical_context(idx, df, lookback_steps, lookback_threshold, min_cum_return):
                    return "Stable"

            # Zukunftsprüfung: Berechne zukünftige Maximal- und Minimalwerte
            future_max = max_future_close(idx, df, future_steps)
            future_min = min_future_close(idx, df, future_steps)

            if use_percentage:
                condition_high = ((future_max - current_close) / current_close) >= threshold_high_pct
                condition_low = (future_min >= current_close * (1 - threshold_low_pct))
            else:
                condition_high = (future_max - current_close) >= threshold_high
                condition_low = (future_min >= current_close - threshold_low)

            if condition_high and condition_low:
                return "Up"
            else:
                return "Stable"

        # 1) Markiere jeden Punkt basierend auf den historischen und zukünftigen Kriterien
        df['Trend'] = df.apply(calculate_trend_for_row, axis=1)

        # 2) Nachträgliche Anpassung der zusammenhängenden Up-Signale
        if up_signal_mode != "none":
            up_chains = []
            current_chain = []
            for i in range(len(df)):
                if df['Trend'].iloc[i] == "Up":
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            if current_chain:
                up_chains.append(current_chain)

            def get_min_close_index(chain_indices):
                min_idx = chain_indices[0]
                min_price = df['Close'].iloc[min_idx]
                for idx in chain_indices:
                    if df['Close'].iloc[idx] < min_price:
                        min_price = df['Close'].iloc[idx]
                        min_idx = idx
                return min_idx

            for chain in up_chains:
                # Bei einer einzigen Markierung (isoliertes Up-Signal) erfolgt später ggf. eine Anpassung
                if len(chain) == 1:
                    continue
                if up_signal_mode == "first_only":
                    for idx in chain[1:]:
                        df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "lowest_only":
                    min_idx = get_min_close_index(chain)
                    for idx in chain:
                        if idx != min_idx:
                            df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "lowest_plus_one":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    if pos_min + 1 < len(chain):
                        keep_idx = chain[pos_min + 1]
                        for idx in chain:
                            if idx != keep_idx:
                                df.at[idx, 'Trend'] = "Stable"
                    else:
                        for idx in chain:
                            df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "all_after_lowest":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    for pos, idx in enumerate(chain):
                        if pos <= pos_min:
                            df.at[idx, 'Trend'] = "Stable"
                        else:
                            df.at[idx, 'Trend'] = "Up"
                elif up_signal_mode == "all_after_lowest_offset":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    for pos, idx in enumerate(chain):
                        if pos < pos_min:
                            df.at[idx, 'Trend'] = "Stable"
                        else:
                            if pos < pos_min + offset_after_lowest:
                                df.at[idx, 'Trend'] = "Stable"
                            else:
                                df.at[idx, 'Trend'] = "Up"

        # 3) Verwerfe isolierte Up-Signale, sofern require_double_ups aktiviert ist
        if require_double_ups:
            up_chains = []
            current_chain = []
            for i in range(len(df)):
                if df['Trend'].iloc[i] == "Up":
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            if current_chain:
                up_chains.append(current_chain)
            for chain in up_chains:
                if len(chain) == 1:
                    df.at[chain[0], 'Trend'] = "Stable"

        return df

    def calculate_trend_robust_12(
            df,
            threshold_high,  # Absoluter Schwellwert für future_max
            threshold_low,  # Absoluter Schwellwert für future_min
            threshold_high_pct,  # Prozentualer Schwellwert für future_max
            threshold_low_pct,  # Prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            min_cum_return,  # Minimaler kumulativer Return im Lookback-Zeitraum
            use_percentage=False,
            up_signal_mode="all_after_lowest_offset",
            use_lookback_check=True,
            require_double_ups=True,
            offset_after_lowest=2,
            use_lookforward_check=False,  # Neuer Parameter: Prüfe zukünftigen Kontext
            look_forward_threshold=0.0,  # Schwellenwert für negative Kursbewegungen in Zukunft
            forward_steps=5  # Anzahl der zukünftigen Kerzen für die Lookforward-Prüfung
    ):
        """
        Diese Funktion markiert Punkte als "Up", sofern sowohl die historische als auch die zukünftige
        Dynamik auf einen signifikanten Aufwärtstrend hinweist. Es werden folgende Kriterien berücksichtigt:

        1. Historische Prüfung:
           - Es wird ein Lookback-Zeitraum betrachtet, in dem der Anteil negativer Kursbewegungen
             (z. B. über 'lookback_steps') nicht zu hoch sein darf (definiert über 'lookback_threshold').
           - Zusätzlich muss der kumulative Return im betrachteten Zeitraum mindestens 'min_cum_return'
             betragen. Damit wird sichergestellt, dass der Punkt nicht lediglich eine kurzfristige Erholung
             innerhalb eines dominanten Abwärtstrends darstellt.

        2. Zukunftsprüfung:
           - Innerhalb von 'future_steps' muss der Kurs einen signifikanten Anstieg verzeichnen,
             gemessen an absoluten oder prozentualen Schwellenwerten.
           - Neu: Zusätzlich wird geprüft, ob der zukünftige Kontext (innerhalb von 'forward_steps')
             keine überproportionalen negativen Bewegungen aufweist – analog zum historischen Lookback.

        3. Nachträgliche Signal-Nachbearbeitung:
           - Zusammenhängende Up-Signale werden anhand verschiedener Modi (z. B. "all_after_lowest_offset")
             weiter gefiltert, sodass nur die relevanten Signale – typischerweise jene, die nach einem Tiefpunkt
             und mit einem bestimmten Offset auftreten – als "Up" belassen werden.
           - Zusätzlich werden isolierte Up-Signale (bei denen nur ein einzelner Punkt als Up markiert ist)
             verworfen, sofern 'require_double_ups' aktiviert ist.

        Rückgabe:
           Der DataFrame mit einer zusätzlichen Spalte 'Trend', in der jeder Punkt entweder "Up" oder "Stable" ist.
        """
        if offset_after_lowest < 0:
            raise ValueError("offset_after_lowest muss >= 0 sein.")

        def max_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)
            return df['Close'][index:end_index].max()

        def min_future_close(index, df, period):
            end_index = index + period
            if end_index > len(df):
                end_index = len(df)
            return df['Close'][index:end_index].min()

        def check_historical_context(index, df, lookback_steps, lookback_threshold, min_cum_return):
            """
            Überprüft, ob der Punkt an der gegebenen Stelle in einem stabilen historischen Kontext liegt.
            Es werden zwei Kriterien geprüft:
              - Der Anteil negativer Kursbewegungen im Lookback-Zeitraum darf den Schwellenwert 'lookback_threshold'
                nicht überschreiten.
              - Der kumulative Return im Lookback-Zeitraum muss mindestens 'min_cum_return' betragen.
            """
            start_index = max(0, index - lookback_steps)
            relevant_closes = df['Close'].iloc[start_index:index + 1]

            if len(relevant_closes) < 2:
                return True  # Nicht genügend Daten => keine Einschränkung

            deltas = relevant_closes.diff().dropna()
            pos_count = (deltas > 0).sum()
            neg_count = (deltas < 0).sum()
            total_changes = pos_count + neg_count
            ratio_neg = neg_count / total_changes if total_changes > 0 else 0

            cum_return = (relevant_closes.iloc[-1] - relevant_closes.iloc[0]) / relevant_closes.iloc[0]

            if ratio_neg > lookback_threshold or cum_return < min_cum_return:
                return False
            return True

        def check_forward_context(index, df, forward_steps, look_forward_threshold):
            """
            Überprüft, ob der Punkt in der Zukunft (innerhalb von forward_steps) in einem stabilen Kontext liegt.
            Hier wird analog zur historischen Prüfung der Anteil negativer Kursbewegungen betrachtet.
            Ist der Anteil negativer Bewegungen im Forward-Zeitraum zu hoch, wird der Kontext als instabil bewertet.
            """
            end_index = index + forward_steps
            if end_index > len(df) - 1:
                end_index = len(df) - 1
            relevant_closes = df['Close'].iloc[index:end_index + 1]

            if len(relevant_closes) < 2:
                return True  # Nicht genügend Daten => keine Einschränkung

            deltas = relevant_closes.diff().dropna()
            pos_count = (deltas > 0).sum()
            neg_count = (deltas < 0).sum()
            total_changes = pos_count + neg_count
            ratio_neg = neg_count / total_changes if total_changes > 0 else 0

            if ratio_neg > look_forward_threshold:
                return False
            return True

        def calculate_trend_for_row(row):
            current_close = row['Close']
            idx = row.name

            # Historische Prüfung: Sicherstellen, dass der Punkt nicht aus einer reinen Erholungsphase stammt
            if use_lookback_check:
                if not check_historical_context(idx, df, lookback_steps, lookback_threshold, min_cum_return):
                    return "Stable"

            # Zukunftsprüfung: Berechne zukünftige Maximal- und Minimalwerte im unmittelbaren Future-Fenster
            future_max = max_future_close(idx, df, future_steps)
            future_min = min_future_close(idx, df, future_steps)

            if use_percentage:
                condition_high = ((future_max - current_close) / current_close) >= threshold_high_pct
                condition_low = (future_min >= current_close * (1 - threshold_low_pct))
            else:
                condition_high = (future_max - current_close) >= threshold_high
                condition_low = (future_min >= current_close - threshold_low)

            if not (condition_high and condition_low):
                return "Stable"

            # Neu: Zukünftiger Kontext-Check (Lookforward):
            # Sicherstellen, dass im weiteren zukünftigen Fenster (forward_steps) keine überproportional negativen Bewegungen auftreten.
            if use_lookforward_check:
                if not check_forward_context(idx, df, forward_steps, look_forward_threshold):
                    return "Stable"

            return "Up"

        # 1) Markiere jeden Punkt basierend auf den historischen und unmittelbaren zukünftigen Kriterien
        df['Trend'] = df.apply(calculate_trend_for_row, axis=1)

        # 2) Nachträgliche Anpassung der zusammenhängenden Up-Signale
        if up_signal_mode != "none":
            up_chains = []
            current_chain = []
            for i in range(len(df)):
                if df['Trend'].iloc[i] == "Up":
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            if current_chain:
                up_chains.append(current_chain)

            def get_min_close_index(chain_indices):
                min_idx = chain_indices[0]
                min_price = df['Close'].iloc[min_idx]
                for idx in chain_indices:
                    if df['Close'].iloc[idx] < min_price:
                        min_price = df['Close'].iloc[idx]
                        min_idx = idx
                return min_idx

            for chain in up_chains:
                if len(chain) == 1:
                    continue
                if up_signal_mode == "first_only":
                    for idx in chain[1:]:
                        df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "lowest_only":
                    min_idx = get_min_close_index(chain)
                    for idx in chain:
                        if idx != min_idx:
                            df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "lowest_plus_one":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    if pos_min + 1 < len(chain):
                        keep_idx = chain[pos_min + 1]
                        for idx in chain:
                            if idx != keep_idx:
                                df.at[idx, 'Trend'] = "Stable"
                    else:
                        for idx in chain:
                            df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "all_after_lowest":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    for pos, idx in enumerate(chain):
                        if pos <= pos_min:
                            df.at[idx, 'Trend'] = "Stable"
                        else:
                            df.at[idx, 'Trend'] = "Up"
                elif up_signal_mode == "all_after_lowest_offset":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    for pos, idx in enumerate(chain):
                        if pos < pos_min:
                            df.at[idx, 'Trend'] = "Stable"
                        else:
                            if pos < pos_min + offset_after_lowest:
                                df.at[idx, 'Trend'] = "Stable"
                            else:
                                df.at[idx, 'Trend'] = "Up"

        # 3) Verwerfe isolierte Up-Signale, sofern require_double_ups aktiviert ist
        if require_double_ups:
            up_chains = []
            current_chain = []
            for i in range(len(df)):
                if df['Trend'].iloc[i] == "Up":
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            if current_chain:
                up_chains.append(current_chain)
            for chain in up_chains:
                if len(chain) == 1:
                    df.at[chain[0], 'Trend'] = "Stable"

        return df


    def calculate_trend_robust_13(
            df,
            threshold_high,  # Absoluter Schwellwert für future_max
            threshold_low,  # Absoluter Schwellwert für future_min
            threshold_high_pct,  # Prozentualer Schwellwert für future_max
            threshold_low_pct,  # Prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            min_cum_return,  # Minimaler kumulativer Return im Lookback-Zeitraum
            use_percentage=False,
            up_signal_mode="all_after_lowest_offset",
            use_lookback_check=True,
            require_double_ups=True,
            offset_after_lowest=2,
            use_lookforward_check=False,  # Neuer Parameter: Prüfe zukünftigen Kontext
            look_forward_threshold=0.0,  # Schwellenwert für negative Kursbewegungen in Zukunft
            forward_steps=5  # Anzahl der zukünftigen Kerzen für die Lookforward-Prüfung
    ):
        """
        Diese Funktion markiert Punkte als "Up", sofern sowohl die historische als auch die zukünftige
        Dynamik auf einen signifikanten Aufwärtstrend hinweist. Es werden folgende Kriterien berücksichtigt:

        1. Historische Prüfung:
           - Es wird ein Lookback-Zeitraum betrachtet, in dem der Anteil negativer Kursbewegungen
             (z. B. über 'lookback_steps') nicht zu hoch sein darf (definiert über 'lookback_threshold').
           - Zusätzlich muss der kumulative Return im betrachteten Zeitraum mindestens 'min_cum_return'
             betragen. Damit wird sichergestellt, dass der Punkt nicht lediglich eine kurzfristige Erholung
             innerhalb eines dominanten Abwärtstrends darstellt.

        2. Zukunftsprüfung:
           - Innerhalb von 'future_steps' muss der Kurs einen signifikanten Anstieg verzeichnen,
             gemessen an absoluten oder prozentualen Schwellenwerten.
           - Neu: Zusätzlich wird geprüft, ob der zukünftige Kontext (innerhalb von 'forward_steps')
             keine überproportionalen negativen Bewegungen aufweist – analog zum historischen Lookback.

        3. Nachträgliche Signal-Nachbearbeitung:
           - Zusammenhängende Up-Signale werden anhand verschiedener Modi (z. B. "all_after_lowest_offset")
             weiter gefiltert, sodass nur die relevanten Signale – typischerweise jene, die nach einem Tiefpunkt
             und mit einem bestimmten Offset auftreten – als "Up" belassen werden.
           - Zusätzlich werden isolierte Up-Signale (bei denen nur ein einzelner Punkt als Up markiert ist)
             verworfen, sofern 'require_double_ups' aktiviert ist.

        Rückgabe:
           Der DataFrame mit einer zusätzlichen Spalte 'Trend', in der jeder Punkt entweder "Up" oder "Stable" ist.
        """
        if offset_after_lowest < 0:
            raise ValueError("offset_after_lowest muss >= 0 sein.")

        # Umrechnung der 'Close'-Werte in ein numpy Array
        close = df['Close'].to_numpy()
        n = len(close)

        # 1) Berechnung der zukünftigen Maxima/Minima:
        # Wir nutzen hier einen Trick mit dem umgekehrten Series-Objekt und rolling,
        # um für jedes Element das Maximum/Minimum im Intervall [i, i+future_steps) zu erhalten.
        s = pd.Series(close)
        future_max = s[::-1].rolling(window=future_steps, min_periods=1).max()[::-1].to_numpy()
        future_min = s[::-1].rolling(window=future_steps, min_periods=1).min()[::-1].to_numpy()

        # 2) Historische Kontextprüfung über kumulative Summen:
        # Berechne zunächst die Differenzen und bestimme, ob sie positiv oder negativ sind.
        d = np.diff(close, prepend=close[0])
        pos = (d > 0).astype(int)
        neg = (d < 0).astype(int)
        cum_pos = np.cumsum(pos)
        cum_neg = np.cumsum(neg)
        indices = np.arange(n)
        start_idx = np.maximum(0, indices - lookback_steps)

        # Initialisiere Arrays für die historische Bedingung und den kumulativen Return
        hist_condition = np.ones(n, dtype=bool)
        cum_return = np.empty(n, dtype=float)
        cum_return[0] = 0.0  # Für i==0 wird immer True zurückgegeben

        for i in range(1, n):
            start = start_idx[i]
            window_length = i - start  # Anzahl der Änderungen im Fenster
            if window_length < 1:
                hist_condition[i] = True
                cum_return[i] = 0.0
            else:
                pos_count = cum_pos[i] - cum_pos[start]
                neg_count = cum_neg[i] - cum_neg[start]
                total = pos_count + neg_count
                ratio_neg = neg_count / total if total > 0 else 0
                base = close[start] if close[start] != 0 else 1
                cum_return[i] = (close[i] - close[start]) / base
                hist_condition[i] = (ratio_neg <= lookback_threshold) and (cum_return[i] >= min_cum_return)

        # 3) Forward Kontextprüfung (falls aktiviert)
        forward_condition = np.ones(n, dtype=bool)
        if use_lookforward_check:
            for i in range(n):
                end = min(n - 1, i + forward_steps)
                window_length = end - i  # Anzahl der Änderungen im forward Fenster
                if window_length < 1:
                    forward_condition[i] = True
                else:
                    pos_count = cum_pos[end] - cum_pos[i]
                    neg_count = cum_neg[end] - cum_neg[i]
                    total = pos_count + neg_count
                    ratio_neg = neg_count / total if total > 0 else 0
                    forward_condition[i] = (ratio_neg <= look_forward_threshold)

        # 4) Berechnung der Preisbewegungs-Bedingungen
        if use_percentage:
            condition_high = ((future_max - close) / close) >= threshold_high_pct
            condition_low = future_min >= close * (1 - threshold_low_pct)
        else:
            condition_high = (future_max - close) >= threshold_high
            condition_low = future_min >= (close - threshold_low)

        overall_condition = condition_high & condition_low
        if use_lookback_check:
            overall_condition = overall_condition & hist_condition
        if use_lookforward_check:
            overall_condition = overall_condition & forward_condition

        # Setze Trend: "Up" wenn alle Bedingungen erfüllt sind, ansonsten "Stable"
        trend = np.where(overall_condition, "Up", "Stable")
        df['Trend'] = trend

        # 5) Nachträgliche Signal-Nachbearbeitung
        if up_signal_mode != "none":
            up_chains = []
            current_chain = []
            for i in range(n):
                if df['Trend'].iloc[i] == "Up":
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            if current_chain:
                up_chains.append(current_chain)

            def get_min_close_index(chain_indices):
                min_idx = chain_indices[0]
                min_price = df['Close'].iloc[min_idx]
                for idx in chain_indices:
                    if df['Close'].iloc[idx] < min_price:
                        min_price = df['Close'].iloc[idx]
                        min_idx = idx
                return min_idx

            for chain in up_chains:
                if len(chain) == 1:
                    continue
                if up_signal_mode == "first_only":
                    for idx in chain[1:]:
                        df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "lowest_only":
                    min_idx = get_min_close_index(chain)
                    for idx in chain:
                        if idx != min_idx:
                            df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "lowest_plus_one":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    if pos_min + 1 < len(chain):
                        keep_idx = chain[pos_min + 1]
                        for idx in chain:
                            if idx != keep_idx:
                                df.at[idx, 'Trend'] = "Stable"
                    else:
                        for idx in chain:
                            df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "all_after_lowest":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    for pos, idx in enumerate(chain):
                        if pos <= pos_min:
                            df.at[idx, 'Trend'] = "Stable"
                        else:
                            df.at[idx, 'Trend'] = "Up"
                elif up_signal_mode == "all_after_lowest_offset":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    for pos, idx in enumerate(chain):
                        if pos < pos_min:
                            df.at[idx, 'Trend'] = "Stable"
                        else:
                            if pos < pos_min + offset_after_lowest:
                                df.at[idx, 'Trend'] = "Stable"
                            else:
                                df.at[idx, 'Trend'] = "Up"

        # 6) Verwerfe isolierte Up-Signale, falls require_double_ups aktiviert ist
        if require_double_ups:
            up_chains = []
            current_chain = []
            for i in range(n):
                if df['Trend'].iloc[i] == "Up":
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            if current_chain:
                up_chains.append(current_chain)
            for chain in up_chains:
                if len(chain) == 1:
                    df.at[chain[0], 'Trend'] = "Stable"

        return df

    def calculate_trend_robust_14(
            df,
            threshold_high,  # Absoluter Schwellwert für future_max
            threshold_low,  # Absoluter Schwellwert für future_min
            threshold_high_pct,  # Prozentualer Schwellwert für future_max
            threshold_low_pct,  # Prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            min_cum_return,  # Minimaler kumulativer Return im Lookback-Zeitraum
            use_percentage=False,
            up_signal_mode="all_after_lowest_offset",
            use_lookback_check=True,
            require_double_ups=True,
            offset_after_lowest=2,
            use_lookforward_check=False,  # Neuer Parameter: Prüfe zukünftigen Kontext
            look_forward_threshold=0.0,  # Schwellenwert für negative Kursbewegungen in Zukunft
            forward_steps=5,  # Anzahl der zukünftigen Kerzen für die Lookforward-Prüfung
            consecutive_negatives_lookback_steps=None,  # Parameter A: Anzahl der Schritte in die Vergangenheit
            max_consecutive_negatives_lookback=None,  # Parameter B: maximal erlaubte aufeinanderfolgende negative Schritte
            consecutive_negatives_forward_steps=None,  # Parameter A (Zukunft): Anzahl der Schritte in die Zukunft
            max_consecutive_negatives_forward=None  # Parameter B (Zukunft): maximal erlaubte aufeinanderfolgende negative Schritte
    ):
        """
        Diese Funktion markiert Punkte als "Up", sofern sowohl die historische als auch die zukünftige
        Dynamik auf einen signifikanten Aufwärtstrend hinweist. Es werden folgende Kriterien berücksichtigt:

        1. Historische Prüfung:
           - Es wird ein Lookback-Zeitraum betrachtet, in dem der Anteil negativer Kursbewegungen
             (z. B. über 'lookback_steps') nicht zu hoch sein darf (definiert über 'lookback_threshold').
           - Zusätzlich muss der kumulative Return im betrachteten Zeitraum mindestens 'min_cum_return'
             betragen.
           - Neu: Es wird zusätzlich geprüft, ob innerhalb eines separaten Fensters (consecutive_negatives_lookback_steps)
             nicht mehr als 'max_consecutive_negatives_lookback' aufeinanderfolgende negative Kursbewegungen auftreten.

        2. Zukunftsprüfung:
           - Innerhalb von 'future_steps' muss der Kurs einen signifikanten Anstieg verzeichnen,
             gemessen an absoluten oder prozentualen Schwellenwerten.
           - Neu: Zusätzlich wird im zukünftigen Kontext (innerhalb von forward_steps) sowohl ein
             Ratio-Check als auch ein Check auf maximal erlaubte aufeinanderfolgende negative Bewegungen
             (innerhalb von consecutive_negatives_forward_steps und max_consecutive_negatives_forward) durchgeführt.

        3. Nachträgliche Signal-Nachbearbeitung:
           - Zusammenhängende Up-Signale werden anhand verschiedener Modi (z. B. "all_after_lowest_offset")
             weiter gefiltert, sodass nur die relevanten Signale – typischerweise jene, die nach einem Tiefpunkt
             und mit einem bestimmten Offset auftreten – als "Up" belassen werden.
           - Zusätzlich werden isolierte Up-Signale (bei denen nur ein einzelner Punkt als Up markiert ist)
             verworfen, sofern 'require_double_ups' aktiviert ist.

        Rückgabe:
           Der DataFrame mit einer zusätzlichen Spalte 'Trend', in der jeder Punkt entweder "Up" oder "Stable" ist.
        """
        if offset_after_lowest < 0:
            raise ValueError("offset_after_lowest muss >= 0 sein.")

        import numpy as np
        import pandas as pd

        # Hilfsfunktion: Bestimme die maximale Anzahl an aufeinanderfolgenden negativen Werten in einem Array
        def max_consecutive_negatives(arr):
            max_count = 0
            count = 0
            for value in arr:
                if value < 0:
                    count += 1
                    if count > max_count:
                        max_count = count
                else:
                    count = 0
            return max_count

        # Umrechnung der 'Close'-Werte in ein numpy Array
        close = df['Close'].to_numpy()
        n = len(close)

        # 1) Berechnung der zukünftigen Maxima/Minima:
        s = pd.Series(close)
        future_max = s[::-1].rolling(window=future_steps, min_periods=1).max()[::-1].to_numpy()
        future_min = s[::-1].rolling(window=future_steps, min_periods=1).min()[::-1].to_numpy()

        # 2) Historische Kontextprüfung über kumulative Summen:
        d = np.diff(close, prepend=close[0])
        pos = (d > 0).astype(int)
        neg = (d < 0).astype(int)
        cum_pos = np.cumsum(pos)
        cum_neg = np.cumsum(neg)
        indices = np.arange(n)
        start_idx = np.maximum(0, indices - lookback_steps)

        hist_condition = np.ones(n, dtype=bool)
        cum_return = np.empty(n, dtype=float)
        cum_return[0] = 0.0

        for i in range(1, n):
            start = start_idx[i]
            window_length = i - start
            if window_length < 1:
                hist_condition[i] = True
                cum_return[i] = 0.0
            else:
                pos_count = cum_pos[i] - cum_pos[start]
                neg_count = cum_neg[i] - cum_neg[start]
                total = pos_count + neg_count
                ratio_neg = neg_count / total if total > 0 else 0
                base = close[start] if close[start] != 0 else 1
                cum_return[i] = (close[i] - close[start]) / base
                hist_condition[i] = (ratio_neg <= lookback_threshold) and (cum_return[i] >= min_cum_return)

            # Zusätzlicher Check: Historischer Fenster-Check auf aufeinanderfolgende negative Schritte
            if (consecutive_negatives_lookback_steps is not None) and (max_consecutive_negatives_lookback is not None):
                start_for_neg = max(0, i - consecutive_negatives_lookback_steps + 1)
                window_d = d[start_for_neg:i + 1]
                if max_consecutive_negatives(window_d) > max_consecutive_negatives_lookback:
                    hist_condition[i] = False

        # 3) Forward Kontextprüfung (falls aktiviert)
        forward_condition = np.ones(n, dtype=bool)
        if use_lookforward_check:
            for i in range(n):
                end = min(n - 1, i + forward_steps)
                window_length = end - i
                if window_length < 1:
                    forward_condition[i] = True
                else:
                    pos_count = cum_pos[end] - cum_pos[i]
                    neg_count = cum_neg[end] - cum_neg[i]
                    total = pos_count + neg_count
                    ratio_neg = neg_count / total if total > 0 else 0
                    forward_condition[i] = (ratio_neg <= look_forward_threshold)
                # Zusätzlicher Check: Zukunfts-Fenster-Check auf aufeinanderfolgende negative Schritte
                if (consecutive_negatives_forward_steps is not None) and (max_consecutive_negatives_forward is not None):
                    end_for_neg = min(n, i + consecutive_negatives_forward_steps + 1)
                    window_d_forward = d[i + 1:end_for_neg]
                    if max_consecutive_negatives(window_d_forward) > max_consecutive_negatives_forward:
                        forward_condition[i] = False

        # 4) Berechnung der Preisbewegungs-Bedingungen
        if use_percentage:
            condition_high = ((future_max - close) / close) >= threshold_high_pct
            condition_low = future_min >= close * (1 - threshold_low_pct)
        else:
            condition_high = (future_max - close) >= threshold_high
            condition_low = future_min >= (close - threshold_low)

        overall_condition = condition_high & condition_low
        if use_lookback_check:
            overall_condition = overall_condition & hist_condition
        if use_lookforward_check:
            overall_condition = overall_condition & forward_condition

        trend = np.where(overall_condition, "Up", "Stable")
        df['Trend'] = trend

        # 5) Nachträgliche Signal-Nachbearbeitung
        if up_signal_mode != "none":
            up_chains = []
            current_chain = []
            for i in range(n):
                if df['Trend'].iloc[i] == "Up":
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            if current_chain:
                up_chains.append(current_chain)

            def get_min_close_index(chain_indices):
                min_idx = chain_indices[0]
                min_price = df['Close'].iloc[min_idx]
                for idx in chain_indices:
                    if df['Close'].iloc[idx] < min_price:
                        min_price = df['Close'].iloc[idx]
                        min_idx = idx
                return min_idx

            for chain in up_chains:
                if len(chain) == 1:
                    continue
                if up_signal_mode == "first_only":
                    for idx in chain[1:]:
                        df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "lowest_only":
                    min_idx = get_min_close_index(chain)
                    for idx in chain:
                        if idx != min_idx:
                            df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "lowest_plus_one":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    if pos_min + 1 < len(chain):
                        keep_idx = chain[pos_min + 1]
                        for idx in chain:
                            if idx != keep_idx:
                                df.at[idx, 'Trend'] = "Stable"
                    else:
                        for idx in chain:
                            df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "all_after_lowest":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    for pos, idx in enumerate(chain):
                        if pos <= pos_min:
                            df.at[idx, 'Trend'] = "Stable"
                        else:
                            df.at[idx, 'Trend'] = "Up"
                elif up_signal_mode == "all_after_lowest_offset":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    for pos, idx in enumerate(chain):
                        if pos < pos_min:
                            df.at[idx, 'Trend'] = "Stable"
                        else:
                            if pos < pos_min + offset_after_lowest:
                                df.at[idx, 'Trend'] = "Stable"
                            else:
                                df.at[idx, 'Trend'] = "Up"

        # 6) Verwerfe isolierte Up-Signale, falls require_double_ups aktiviert ist
        if require_double_ups:
            up_chains = []
            current_chain = []
            for i in range(n):
                if df['Trend'].iloc[i] == "Up":
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            if current_chain:
                up_chains.append(current_chain)
            for chain in up_chains:
                if len(chain) == 1:
                    df.at[chain[0], 'Trend'] = "Stable"

        return df

    def calculate_trend_robust_15(
            df,
            threshold_high,  # Absoluter Schwellwert für future_max
            threshold_low,  # Absoluter Schwellwert für future_min
            threshold_high_pct,  # Prozentualer Schwellwert für future_max
            threshold_low_pct,  # Prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            min_cum_return,  # Minimaler kumulativer Return im Lookback-Zeitraum
            use_percentage=False,
            up_signal_mode="all_after_lowest_offset",
            use_lookback_check=True,
            require_double_ups=True,
            offset_after_lowest=2,
            use_lookforward_check=False,  # Prüfe zukünftigen Kontext
            look_forward_threshold=0.0,  # Schwellenwert für negative Kursbewegungen in Zukunft
            forward_steps=5,  # Anzahl der zukünftigen Kerzen für die Lookforward-Prüfung
            # Parameter für negative Bewegungen im historischen Kontext:
            consecutive_negatives_lookback_steps=None,  # Anzahl der Schritte in die Vergangenheit
            max_consecutive_negatives_lookback=None,  # maximal erlaubte aufeinanderfolgende negative Schritte
            # Parameter für negative Bewegungen im zukünftigen Kontext:
            consecutive_negatives_forward_steps=None,  # Anzahl der Schritte in die Zukunft
            max_consecutive_negatives_forward=None,  # maximal erlaubte aufeinanderfolgende negative Schritte
            # Neue Parameter für positive Bewegungen im historischen Kontext:
            consecutive_positives_lookback_steps=None,  # Anzahl der Schritte in die Vergangenheit
            max_consecutive_positives_lookback=None,  # maximal erlaubte aufeinanderfolgende positive Schritte
            # Neue Parameter für positive Bewegungen im zukünftigen Kontext:
            consecutive_positives_forward_steps=None,  # Anzahl der Schritte in die Zukunft
            max_consecutive_positives_forward=None  # maximal erlaubte aufeinanderfolgende positive Schritte
    ):
        """
        Diese Funktion markiert Punkte als "Up", sofern sowohl die historische als auch die zukünftige
        Dynamik auf einen signifikanten Aufwärtstrend hinweist. Es werden folgende Kriterien berücksichtigt:

        1. Historische Prüfung:
           - Es wird ein Lookback-Zeitraum betrachtet, in dem der Anteil negativer Kursbewegungen
             (z. B. über 'lookback_steps') nicht zu hoch sein darf (definiert über 'lookback_threshold').
           - Zusätzlich muss der kumulative Return im betrachteten Zeitraum mindestens 'min_cum_return'
             betragen.
           - Neu: Es wird zusätzlich geprüft, ob innerhalb eines separaten Fensters (consecutive_negatives_lookback_steps)
             nicht mehr als 'max_consecutive_negatives_lookback' aufeinanderfolgende negative Kursbewegungen auftreten.
           - Neu: Analog dazu wird geprüft, ob innerhalb eines separaten Fensters (consecutive_positives_lookback_steps)
             nicht mehr als 'max_consecutive_positives_lookback' aufeinanderfolgende positive Kursbewegungen auftreten.

        2. Zukunftsprüfung:
           - Innerhalb von 'future_steps' muss der Kurs einen signifikanten Anstieg verzeichnen,
             gemessen an absoluten oder prozentualen Schwellenwerten.
           - Neu: Zusätzlich wird im zukünftigen Kontext (innerhalb von forward_steps) sowohl ein
             Ratio-Check als auch ein Check auf maximal erlaubte aufeinanderfolgende negative Bewegungen
             (innerhalb von consecutive_negatives_forward_steps und max_consecutive_negatives_forward) durchgeführt.
           - Neu: Ebenso wird geprüft, ob im Forward-Fenster nicht mehr als 'max_consecutive_positives_forward'
             aufeinanderfolgende positive Schritte auftreten, sofern 'consecutive_positives_forward_steps' definiert ist.

        3. Nachträgliche Signal-Nachbearbeitung:
           - Zusammenhängende Up-Signale werden anhand verschiedener Modi (z. B. "all_after_lowest_offset")
             weiter gefiltert, sodass nur die relevanten Signale – typischerweise jene, die nach einem Tiefpunkt
             und mit einem bestimmten Offset auftreten – als "Up" belassen werden.
           - Zusätzlich werden isolierte Up-Signale (bei denen nur ein einzelner Punkt als Up markiert ist)
             verworfen, sofern 'require_double_ups' aktiviert ist.

        Rückgabe:
           Der DataFrame mit einer zusätzlichen Spalte 'Trend', in der jeder Punkt entweder "Up" oder "Stable" ist.
        """
        if offset_after_lowest < 0:
            raise ValueError("offset_after_lowest muss >= 0 sein.")

        import numpy as np
        import pandas as pd

        # Hilfsfunktion: Bestimme die maximale Anzahl an aufeinanderfolgenden negativen Werten in einem Array
        def max_consecutive_negatives(arr):
            max_count = 0
            count = 0
            for value in arr:
                if value < 0:
                    count += 1
                    if count > max_count:
                        max_count = count
                else:
                    count = 0
            return max_count

        # Hilfsfunktion: Bestimme die maximale Anzahl an aufeinanderfolgenden positiven Werten in einem Array
        def max_consecutive_positives(arr):
            max_count = 0
            count = 0
            for value in arr:
                if value > 0:
                    count += 1
                    if count > max_count:
                        max_count = count
                else:
                    count = 0
            return max_count

        # Umrechnung der 'Close'-Werte in ein numpy Array
        close = df['Close'].to_numpy()
        n = len(close)

        # 1) Berechnung der zukünftigen Maxima/Minima:
        s = pd.Series(close)
        future_max = s[::-1].rolling(window=future_steps, min_periods=1).max()[::-1].to_numpy()
        future_min = s[::-1].rolling(window=future_steps, min_periods=1).min()[::-1].to_numpy()

        # 2) Historische Kontextprüfung über kumulative Summen:
        d = np.diff(close, prepend=close[0])
        pos = (d > 0).astype(int)
        neg = (d < 0).astype(int)
        cum_pos = np.cumsum(pos)
        cum_neg = np.cumsum(neg)
        indices = np.arange(n)
        start_idx = np.maximum(0, indices - lookback_steps)

        hist_condition = np.ones(n, dtype=bool)
        cum_return = np.empty(n, dtype=float)
        cum_return[0] = 0.0

        for i in range(1, n):
            start = start_idx[i]
            window_length = i - start
            if window_length < 1:
                hist_condition[i] = True
                cum_return[i] = 0.0
            else:
                pos_count = cum_pos[i] - cum_pos[start]
                neg_count = cum_neg[i] - cum_neg[start]
                total = pos_count + neg_count
                ratio_neg = neg_count / total if total > 0 else 0
                base = close[start] if close[start] != 0 else 1
                cum_return[i] = (close[i] - close[start]) / base
                hist_condition[i] = (ratio_neg <= lookback_threshold) and (cum_return[i] >= min_cum_return)

            # Zusätzlicher Check: Historischer Fenster-Check auf aufeinanderfolgende negative Schritte
            if (consecutive_negatives_lookback_steps is not None) and (max_consecutive_negatives_lookback is not None):
                start_for_neg = max(0, i - consecutive_negatives_lookback_steps + 1)
                window_d = d[start_for_neg:i + 1]
                if max_consecutive_negatives(window_d) > max_consecutive_negatives_lookback:
                    hist_condition[i] = False

            # Neuer Check: Historischer Fenster-Check auf aufeinanderfolgende positive Schritte
            if (consecutive_positives_lookback_steps is not None) and (max_consecutive_positives_lookback is not None):
                start_for_pos = max(0, i - consecutive_positives_lookback_steps + 1)
                window_d_pos = d[start_for_pos:i + 1]
                if max_consecutive_positives(window_d_pos) > max_consecutive_positives_lookback:
                    hist_condition[i] = False

        # 3) Forward Kontextprüfung (falls aktiviert)
        forward_condition = np.ones(n, dtype=bool)
        if use_lookforward_check:
            for i in range(n):
                end = min(n - 1, i + forward_steps)
                window_length = end - i
                if window_length < 1:
                    forward_condition[i] = True
                else:
                    pos_count = cum_pos[end] - cum_pos[i]
                    neg_count = cum_neg[end] - cum_neg[i]
                    total = pos_count + neg_count
                    ratio_neg = neg_count / total if total > 0 else 0
                    forward_condition[i] = (ratio_neg <= look_forward_threshold)
                # Zusätzlicher Check: Forward-Fenster-Check auf aufeinanderfolgende negative Schritte
                if (consecutive_negatives_forward_steps is not None) and (max_consecutive_negatives_forward is not None):
                    end_for_neg = min(n, i + consecutive_negatives_forward_steps + 1)
                    window_d_forward = d[i + 1:end_for_neg]
                    if max_consecutive_negatives(window_d_forward) > max_consecutive_negatives_forward:
                        forward_condition[i] = False
                # Neuer Check: Forward-Fenster-Check auf aufeinanderfolgende positive Schritte
                if (consecutive_positives_forward_steps is not None) and (max_consecutive_positives_forward is not None):
                    end_for_pos = min(n, i + consecutive_positives_forward_steps + 1)
                    window_d_forward_pos = d[i + 1:end_for_pos]
                    if max_consecutive_positives(window_d_forward_pos) > max_consecutive_positives_forward:
                        forward_condition[i] = False

        # 4) Berechnung der Preisbewegungs-Bedingungen
        if use_percentage:
            condition_high = ((future_max - close) / close) >= threshold_high_pct
            condition_low = future_min >= close * (1 - threshold_low_pct)
        else:
            condition_high = (future_max - close) >= threshold_high
            condition_low = future_min >= (close - threshold_low)

        overall_condition = condition_high & condition_low
        if use_lookback_check:
            overall_condition = overall_condition & hist_condition
        if use_lookforward_check:
            overall_condition = overall_condition & forward_condition

        trend = np.where(overall_condition, "Up", "Stable")
        df['Trend'] = trend

        # 5) Nachträgliche Signal-Nachbearbeitung
        if up_signal_mode != "none":
            up_chains = []
            current_chain = []
            for i in range(n):
                if df['Trend'].iloc[i] == "Up":
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            if current_chain:
                up_chains.append(current_chain)

            def get_min_close_index(chain_indices):
                min_idx = chain_indices[0]
                min_price = df['Close'].iloc[min_idx]
                for idx in chain_indices:
                    if df['Close'].iloc[idx] < min_price:
                        min_price = df['Close'].iloc[idx]
                        min_idx = idx
                return min_idx

            for chain in up_chains:
                if len(chain) == 1:
                    continue
                if up_signal_mode == "first_only":
                    for idx in chain[1:]:
                        df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "lowest_only":
                    min_idx = get_min_close_index(chain)
                    for idx in chain:
                        if idx != min_idx:
                            df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "lowest_plus_one":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    if pos_min + 1 < len(chain):
                        keep_idx = chain[pos_min + 1]
                        for idx in chain:
                            if idx != keep_idx:
                                df.at[idx, 'Trend'] = "Stable"
                    else:
                        for idx in chain:
                            df.at[idx, 'Trend'] = "Stable"
                elif up_signal_mode == "all_after_lowest":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    for pos, idx in enumerate(chain):
                        if pos <= pos_min:
                            df.at[idx, 'Trend'] = "Stable"
                        else:
                            df.at[idx, 'Trend'] = "Up"
                elif up_signal_mode == "all_after_lowest_offset":
                    min_idx = get_min_close_index(chain)
                    pos_min = chain.index(min_idx)
                    for pos, idx in enumerate(chain):
                        if pos < pos_min:
                            df.at[idx, 'Trend'] = "Stable"
                        else:
                            if pos < pos_min + offset_after_lowest:
                                df.at[idx, 'Trend'] = "Stable"
                            else:
                                df.at[idx, 'Trend'] = "Up"

        # 6) Verwerfe isolierte Up-Signale, falls require_double_ups aktiviert ist
        if require_double_ups:
            up_chains = []
            current_chain = []
            for i in range(n):
                if df['Trend'].iloc[i] == "Up":
                    current_chain.append(i)
                else:
                    if current_chain:
                        up_chains.append(current_chain)
                        current_chain = []
            if current_chain:
                up_chains.append(current_chain)
            for chain in up_chains:
                if len(chain) == 1:
                    df.at[chain[0], 'Trend'] = "Stable"

        return df

    # IMPORTANT

    # print(df.head())

    print("Erstelle Trend")

    if trendfunc == "v1":
        """
            Markiert 'Up', wenn der Kurs innerhalb der angegebenen Anzahl von `future_steps` um mindestens `threshold_high` steigt, 
            nicht unter den um `threshold_low` reduzierten aktuellen Schlusskurs fällt. Andernfalls wird 'Stable' markiert.
        """
        df = calculate_trend_future_only1(df, threshold_high, threshold_low, future_steps)  # sehr strend - alle Werte in der Zuk


    elif trendfunc == "v4":
        """
            1. Der Schlusskurs steigt innerhalb der angegebenen Anzahl von `future_steps` um mindestens `threshold_high`.
            2. Der Mindestschlusskurs innerhalb der `future_steps` fällt nicht unter `current_close - threshold_low`.
            3. Alle zukünftigen Schlusskurse innerhalb des Zeitfensters von `future_steps` sind größer oder gleich dem aktuellen Schlusskurs.
            4. Kein Schlusskurs in den letzten `lookback_steps` überschreitet den Wert `current_close + lookback_threshold`.
            
            Falls eine dieser Bedingungen nicht erfüllt wird, wird 'Stable' markiert.
        """
        df = calculate_trend_future_only4(df, threshold_high, threshold_low, future_steps, lookback_steps, lookback_threshold)


    elif trendfunc == "v7":
        calculate_trend_future_only7(
            df,
            threshold_high,  # absoluter Schwellwert für future_max
            threshold_low,  # absoluter Schwellwert für future_min
            threshold_high_pct,  # prozentualer Schwellwert für future_max
            threshold_low_pct,  # prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            use_percentage)

    elif trendfunc == "v8":
        calculate_trend_future_only8(
            df,
            threshold_high,  # absoluter Schwellwert für future_max
            threshold_low,  # absoluter Schwellwert für future_min
            threshold_high_pct,  # prozentualer Schwellwert für future_max
            threshold_low_pct,  # prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            use_percentage)

    elif trendfunc == "v9":

        calculate_trend_future_only9(
            df,
            threshold_high,  # absoluter Schwellwert für future_max
            threshold_low,  # absoluter Schwellwert für future_min
            threshold_high_pct,  # prozentualer Schwellwert für future_max
            threshold_low_pct,  # prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            use_percentage,
            up_signal_mode,
            use_lookback_check
        )

    elif trendfunc == "v10":

        calculate_trend_future_only10(
            df,
            threshold_high,  # absoluter Schwellwert für future_max
            threshold_low,  # absoluter Schwellwert für future_min
            threshold_high_pct,  # prozentualer Schwellwert für future_max
            threshold_low_pct,  # prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            use_percentage,
            up_signal_mode,
            use_lookback_check,
            require_double_ups,
            offset_after_lowest

        )

    elif trendfunc == "v11":
        calculate_trend_robust_11(
            df,
            threshold_high,  # Absoluter Schwellwert für future_max
            threshold_low,  # Absoluter Schwellwert für future_min
            threshold_high_pct,  # Prozentualer Schwellwert für future_max
            threshold_low_pct,  # Prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            min_cum_return,  # Minimaler kumulativer Return im Lookback-Zeitraum
            use_percentage,
            up_signal_mode,
            use_lookback_check,
            require_double_ups,
            offset_after_lowest
        )

    elif trendfunc == "v12":
        calculate_trend_robust_12(
            df,
            threshold_high,  # Absoluter Schwellwert für future_max
            threshold_low,  # Absoluter Schwellwert für future_min
            threshold_high_pct,  # Prozentualer Schwellwert für future_max
            threshold_low_pct,  # Prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            min_cum_return,  # Minimaler kumulativer Return im Lookback-Zeitraum
            use_percentage,
            up_signal_mode,
            use_lookback_check,
            require_double_ups,
            offset_after_lowest,
            use_lookforward_check,
            look_forward_threshold,
            forward_steps
        )

    elif trendfunc == "v13":
        calculate_trend_robust_13(
            df,
            threshold_high,  # Absoluter Schwellwert für future_max
            threshold_low,  # Absoluter Schwellwert für future_min
            threshold_high_pct,  # Prozentualer Schwellwert für future_max
            threshold_low_pct,  # Prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            min_cum_return,  # Minimaler kumulativer Return im Lookback-Zeitraum
            use_percentage,
            up_signal_mode,
            use_lookback_check,
            require_double_ups,
            offset_after_lowest,
            use_lookforward_check,
            look_forward_threshold,
            forward_steps
        )

    elif trendfunc == "v14":
        calculate_trend_robust_14(
            df,
            threshold_high,  # Absoluter Schwellwert für future_max
            threshold_low,  # Absoluter Schwellwert für future_min
            threshold_high_pct,  # Prozentualer Schwellwert für future_max
            threshold_low_pct,  # Prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            min_cum_return,  # Minimaler kumulativer Return im Lookback-Zeitraum
            use_percentage,
            up_signal_mode,
            use_lookback_check,
            require_double_ups,
            offset_after_lowest,
            use_lookforward_check,
            look_forward_threshold,
            forward_steps,
            consecutive_negatives_lookback_steps,
            max_consecutive_negatives_lookback,
            consecutive_negatives_forward_steps,
            max_consecutive_negatives_forward,
        )

    elif trendfunc == "v15":
        calculate_trend_robust_15(
            df,
            threshold_high,  # Absoluter Schwellwert für future_max
            threshold_low,  # Absoluter Schwellwert für future_min
            threshold_high_pct,  # Prozentualer Schwellwert für future_max
            threshold_low_pct,  # Prozentualer Schwellwert für future_min
            future_steps,
            lookback_steps,
            lookback_threshold,
            min_cum_return,  # Minimaler kumulativer Return im Lookback-Zeitraum
            use_percentage,
            up_signal_mode,
            use_lookback_check,
            require_double_ups,
            offset_after_lowest,
            use_lookforward_check,
            look_forward_threshold,
            forward_steps,
            consecutive_negatives_lookback_steps,
            max_consecutive_negatives_lookback,
            consecutive_negatives_forward_steps,
            max_consecutive_negatives_forward,
            consecutive_positives_lookback_steps,
            max_consecutive_positives_lookback,
            consecutive_positives_forward_steps,
            max_consecutive_positives_forward
        )


    df['Trend'] = df['Trend'].shift(-backwarts_shift_labels)

    df.dropna(inplace=True)

    print(f'df_columns:\n {df.columns}')

    return df


def get_columns_by_mode(df, mode):
        # Liste der Spalten, die ausgeschlossen werden sollen

        excluded_columns_training = [
            'Dividends', 'Stock Splits', 'Date', 'Time', 'Comm', 'NonComm', 'Spec', 'MidPoint', 'High.1', 'Low.1', '100', '0', '120', '-20', 'Up', 'Down',

            "Datetime",
            'Volume',
            'Open',
            'High',
            'Low',
            'Close',
            "Close_orig"
        ]

        excluded_columns_predicting = [
            'Dividends', 'Stock Splits', 'Date', 'Time', 'Comm', 'NonComm', 'Spec', 'MidPoint', 'High.1', 'Low.1', '100', '0', '120', '-20', 'Up', 'Down',

            # "Datetime",
            'Volume',
            'Open',
            'High',
            'Low',
            'Close',
            # "Close_orig"
        ]

        # Alle Spalten des DataFrames ermitteln
        all_columns = df.columns.tolist()

        if mode == "all":
            # Rückgabe aller Spalten
            return all_columns
        elif mode == "training":
            # Training mode: Alle Spalten minus ausgeschlossene, aber inklusive 'Trend'
            training_columns = [col for col in all_columns if col not in excluded_columns_training]
            return training_columns
        elif mode == "predicting":
            # Predicting mode: Alle Spalten minus ausgeschlossene und minus 'Trend'
            predicting_columns = [col for col in all_columns if col not in excluded_columns_predicting
                                  # and col != 'Trend'
                                  ]
            # print(f'predicting_columns:{predicting_columns}')
            return predicting_columns
        else:
            raise ValueError("Invalid mode specified. Use 'all', 'training', or 'predicting'.")


def split_data_v0(dataframe, test_size=0.2):
    train_size = int(len(dataframe) * (1 - test_size))
    train_df = dataframe[:train_size]
    test_df = dataframe[train_size:]
    return train_df, test_df



# def predict_and_plot2(
#         df_new_data,
#         database_name_optuna,
#         plot_length=500,  # Anzahl der letzten Datenpunkte, die geplottet werden sollen
#         additional_lines=None,
#         secondary_y_scale=1.0,
#         x_interval_min=60,
#         y_interval_dollar=80,
#         predicting_columns=None  # Liste der Vorhersagespalten
# ):
#     """
#     Lädt ein trainiertes Modell, führt Vorhersagen auf neuen Daten durch und plottet die Ergebnisse.
#
#     Parameter:
#     - df_new_data (pd.DataFrame): Neue Daten für Vorhersagen.
#     - database_name_optuna (str): Name der Datenbank/Studie für das Modell.
#     - plot_length (int): Anzahl der letzten Datenpunkte, die geplottet werden sollen.
#     - additional_lines (list, optional): Zusätzliche Linien für den Plot.
#     - secondary_y_scale (float): Skalierungsfaktor für die sekundäre Y-Achse.
#     - x_interval_min (int): Intervall für die X-Achse in Minuten.
#     - y_interval_dollar (int): Intervall für die Y-Achse in Dollar.
#     - predicting_columns (list of str, optional): Liste der Spalten, die für die Vorhersage verwendet werden.
#     """
#
#     model_save_path = f"saved_models/nn_model_{database_name_optuna}"
#     model_name = f"nn_model_{database_name_optuna}"
#     scalers_save_path = f"{model_save_path}/scalers.pkl"  # Skalierer Pfad (falls verwendet)
#
#     # 1. Modell laden
#     print("Lade das trainierte Modell...")
#     try:
#         model = tf.keras.models.load_model(
#             os.path.join(model_save_path, model_name + '.keras'),
#             custom_objects={
#                 # 'F1Score': F1Score,
#                 'F1Score': tfa.metrics.F1Score(num_classes=2, average='macro'),
#                 'FeatureWeightingLayer': FeatureWeightingLayer
#             }
#         )
#         print("Modell erfolgreich geladen.")
#     except Exception as e:
#         print(f"Fehler beim Laden des Modells: {e}")
#         traceback.print_exc()
#         return
#
#     # 2. Scaler laden und Daten skalieren (falls Skalierer separat gespeichert wurden)
#     # Hinweis: Wenn die Normalisierungsschichten Teil des Modells sind, ist dies möglicherweise nicht erforderlich.
#     # Entfernen Sie den folgenden Block, wenn die Normalisierung vollständig im Modell integriert ist.
#
#     # Uncomment the following block if you have separate scalers
#     """
#     print("Lade die Scaler und skaliere die Daten...")
#     try:
#         scalers = joblib.load(scalers_save_path)
#         # Skalierung der Features
#         for col in predicting_columns:
#             if col in scalers:
#                 df_new_data[col] = scalers[col].transform(df_new_data[col].values.reshape(-1, 1)).flatten()
#             else:
#                 print(f"Warnung: Kein Scaler für die Spalte '{col}' gefunden. Diese Spalte wird nicht skaliert.")
#     except Exception as e:
#         print(f"Fehler beim Laden der Scaler: {e}")
#         traceback.print_exc()
#         return
#     """
#
#     # 3. Daten vorbereiten
#     print("Bereite die Daten für die Vorhersage vor...")
#
#     # Identifiziere die Zeitspalten (ohne 'Close_orig' und 'Trend')
#     time_columns = ['Date', 'Time', 'Datetime', 'Close_orig']  # 'Trend' ist die Zielvariable
#     present_time_columns = [col for col in time_columns if col in df_new_data.columns]
#     if present_time_columns:
#         # Extrahiere die Zeitspalten für das Plotten
#         time_data = df_new_data[present_time_columns].copy()
#     else:
#         print("Keine Zeitspalten gefunden. Stelle sicher, dass die Spaltennamen korrekt sind.")
#         time_data = pd.DataFrame()
#
#     # Entferne die Zeitspalten und die Zielspalte 'Trend' für die Vorhersage
#     if predicting_columns is None:
#         # Verwende alle Spalten außer 'Trend' und Zeitspalten
#         predicting_columns = [col for col in df_new_data.columns if col not in ['Trend'] + present_time_columns]
#     else:
#         # Exkludiere Zeitspalten und 'Trend' aus 'predicting_columns'
#         predicting_columns = [col for col in predicting_columns if col not in present_time_columns and col != 'Trend']
#
#     print(f"Predicting columns: {predicting_columns}")
#
#     # Sicherstellen, dass 'Trend' nicht in den Features ist
#     if 'Trend' in predicting_columns:
#         predicting_columns.remove('Trend')
#
#     X_new = df_new_data[predicting_columns].copy()
#
#     # 4. Vorbereitung der Eingaben für das Modell
#     print("Bereite die Eingaben für das Modell vor...")
#     input_dict = {}
#     for col in predicting_columns:
#         # Jede Eingabe muss eine 2D-Array mit Shape (num_samples, 1) sein
#         input_dict[col] = X_new[col].values.reshape(-1, 1)
#
#     # 5. Vorhersagen machen
#     print("Mache Vorhersagen...")
#     try:
#         predictions = model.predict(input_dict)
#         print("Vorhersagen erfolgreich durchgeführt.")
#     except Exception as e:
#         print("Fehler bei der Vorhersage:")
#         print(traceback.format_exc())
#         return
#
#     # 6. Anpassen der Vorhersagen ohne One-Hot-Encoding
#     print("Verarbeite die Vorhersagen...")
#     if predictions.shape[1] == 1:
#         # Sigmoid-Ausgabe (für binäre Klassifikation)
#         predicted_labels = (predictions > 0.5).astype(int).flatten()
#         label_mapping = {0: 'Stable', 1: 'Up'}
#         predicted_labels = [label_mapping[label] for label in predicted_labels]
#     elif predictions.shape[1] > 1:
#         # Softmax-Ausgabe (für Mehrklassenklassifikation)
#         predicted_classes = np.argmax(predictions, axis=1)
#         # Annahme: Die Reihenfolge der Kategorien entspricht der Reihenfolge im OneHotEncoder
#         # Beispiel: {0: 'Stable', 1: 'Up'}
#         label_mapping = {0: 'Stable', 1: 'Up'}
#         predicted_labels = [label_mapping.get(class_, 'Unknown') for class_ in predicted_classes]
#     else:
#         print("Fehler: Unerwartete Anzahl von Ausgängen im Modell.")
#         return
#
#     # 7. Vorhersagen dem DataFrame hinzufügen
#     print("Füge die Vorhersagen dem DataFrame hinzu...")
#     # Sicherstellen, dass die Anzahl der Vorhersagen mit der Anzahl der Daten übereinstimmt
#     if len(predicted_labels) != len(df_new_data):
#         print("Warnung: Die Anzahl der Vorhersagen stimmt nicht mit der Anzahl der Daten überein.")
#         df_new_data = df_new_data.iloc[:len(predicted_labels)].copy()
#     else:
#         df_new_data = df_new_data.copy()
#
#     # Füge die Zeitspalten wieder ein, falls vorhanden
#     if not time_data.empty:
#         df_new_data[present_time_columns] = time_data.iloc[:len(predicted_labels)].values
#
#     # Füge die Vorhersagen hinzu
#     df_new_data["Predicted_Trend"] = predicted_labels
#
#     # Stellen Sie sicher, dass die 'Datetime'-Spalte vorhanden ist
#     if 'Datetime' not in df_new_data.columns:
#         print("Warnung: 'Datetime' Spalte fehlt. Stelle sicher, dass die Zeitspalten korrekt sind.")
#
#     # Auswahl der relevanten Spalten für das Excel-Export und Plotting
#     if 'Datetime' in df_new_data.columns:
#         df_new_data = df_new_data[['Datetime', 'Close_orig'] + ['Trend'] + ['Predicted_Trend']]
#     else:
#         df_new_data = df_new_data[['Close_orig', 'Trend', 'Predicted_Trend']]
#
#     # 8. Exportiere die Ergebnisse nach Excel
#     print("Exportiere die Ergebnisse nach Excel...")
#     try:
#         df_new_data.to_excel("df_new_data.xlsx", index=False)
#         print("Daten erfolgreich nach 'df_new_data.xlsx' exportiert.")
#     except Exception as e:
#         print(f"Fehler beim Exportieren der Daten nach Excel: {e}")
#         traceback.print_exc()
#
#     # 9. Plotten der Ergebnisse
#     print("Plotten der Ergebnisse...")
#     try:
#         plot_stock_prices(
#             df_new_data.tail(plot_length),
#             test=True,
#             trend="Predicted_Trend",
#             secondary_y_scale=secondary_y_scale,
#             x_interval_min=x_interval_min,
#             y_interval_dollar=y_interval_dollar,
#             additional_lines=additional_lines
#         )
#         print("Vorhersage und Plot abgeschlossen.")
#     except Exception as e:
#         print(f"Fehler beim Plotten der Ergebnisse: {e}")
#         traceback.print_exc()


def backtest_strategy1(
    df,
    initial_capital=10000.0,
    shares_per_order=100,
    initial_stop_loss=1.0,       # Absoluter Abstand, z.B. 1.0 = 1 Dollar/Euro unter Einstiegskurs
    use_trailing_stop=False,
    trailing_stop_distance=1.0,  # Absoluter Abstand für den Trailing Stop
    transaction_costs=0.0,       # Absolut, z.B. 5.0 = 5 Dollar pro Order
    slippage=0.0,                # Absoluter Betrag, z.B. 0.05 = 5 Cent
    allow_short=False
):
    """
    Führt ein einfaches Backtesting basierend auf den Vorhersagesignalen durch.

    Parameter:
    - df (pd.DataFrame): DataFrame mit 'Close_orig' und 'Predicted_Trend'
    - initial_capital (float): Startkapital
    - shares_per_order (int): Anzahl Aktien pro Trade
    - initial_stop_loss (float): Absoluter Abstand des Stop-Loss zum Einstiegspreis
    - use_trailing_stop (bool): Ob ein Trailing Stop eingesetzt werden soll
    - trailing_stop_distance (float): Absoluter Abstand des Trailing Stops
    - transaction_costs (float): Absolute Kosten pro Order (Kauf oder Verkauf)
    - slippage (float): Absoluter Wert der Slippage pro Trade (wird auf den Kaufkurs addiert und vom Verkaufskurs abgezogen)
    - allow_short (bool): Ob Short-Positionen erlaubt sind (hier nicht implementiert)

    Rückgabe:
    - results (dict): Dictionary mit Performance-Kennzahlen
    - df (pd.DataFrame): DataFrame mit zusätzlichen Spalten (Equity, Trades, etc.)
    """

    df = df.copy()
    df["Position"] = 0      # 1 = Long Position offen, 0 = keine Position
    df["Entry_Price"] = np.nan
    df["Stop_Loss"] = np.nan
    df["Trailing_Stop"] = np.nan
    df["Trade_PnL"] = 0.0
    df["Capital"] = 0.0

    capital = initial_capital
    position_open = False
    entry_price = np.nan
    stop_loss_price = np.nan
    trailing_stop_price = np.nan

    for i in range(len(df)):
        current_price = df["Close_orig"].iloc[i]
        signal = df["Predicted_Trend"].iloc[i]

        if not position_open:
            if signal == "Up":
                # Kauf mit Berücksichtigung von Slippage
                buy_price = current_price + slippage
                cost = buy_price * shares_per_order + transaction_costs
                if cost <= capital:
                    position_open = True
                    entry_price = buy_price
                    # initial_stop_loss ist eine absolute Differenz, die vom Kaufkurs abgezogen wird
                    stop_loss_price = entry_price - initial_stop_loss
                    if use_trailing_stop:
                        trailing_stop_price = entry_price - trailing_stop_distance
                    capital -= cost
                    df.at[df.index[i], "Position"] = 1
                    df.at[df.index[i], "Entry_Price"] = entry_price
                    df.at[df.index[i], "Stop_Loss"] = stop_loss_price
                    df.at[df.index[i], "Trailing_Stop"] = trailing_stop_price

        else:
            # Position ist offen
            # Stop-Loss Check
            if current_price <= stop_loss_price:
                # Ausgestoppt, Verkauf mit Slippage
                sell_price = stop_loss_price - slippage if (stop_loss_price - slippage) > 0 else 0.0
                revenue = sell_price * shares_per_order - transaction_costs
                trade_pnl = revenue - (entry_price * shares_per_order + transaction_costs)
                capital += revenue
                position_open = False
                df.at[df.index[i], "Trade_PnL"] = trade_pnl
                df.at[df.index[i], "Position"] = 0
                entry_price = np.nan
                stop_loss_price = np.nan
                trailing_stop_price = np.nan

            else:
                # Prüfen ob Signal "Stable" => Verkauf
                if signal == "Stable":
                    sell_price = current_price - slippage if (current_price - slippage) > 0 else 0.0
                    revenue = sell_price * shares_per_order - transaction_costs
                    trade_pnl = revenue - (entry_price * shares_per_order + transaction_costs)
                    capital += revenue
                    position_open = False
                    df.at[df.index[i], "Trade_PnL"] = trade_pnl
                    df.at[df.index[i], "Position"] = 0
                    entry_price = np.nan
                    stop_loss_price = np.nan
                    trailing_stop_price = np.nan
                else:
                    # Signal bleibt "Up", eventuell Trailing Stop anpassen
                    if use_trailing_stop:
                        # Höheres Kursniveau, evtl. trailing_stop_price anheben
                        # Angenommen, wir passen den Trailing Stop an, wenn der Kurs höher als der Entry ist
                        if current_price > entry_price:
                            # Neuer Trailing Stop - wir orientieren uns am aktuellen Preis
                            new_trailing_stop = current_price - trailing_stop_distance
                            if new_trailing_stop > trailing_stop_price:
                                trailing_stop_price = new_trailing_stop
                            # Der Stop-Loss darf nicht tiefer als der Trailing Stop sein:
                            stop_loss_price = max(stop_loss_price, trailing_stop_price)

        df.at[df.index[i], "Capital"] = capital

    # Falls am Ende noch eine Position offen ist, glattstellen:
    if position_open:
        final_sell_price = df["Close_orig"].iloc[-1] - slippage
        if final_sell_price < 0:
            final_sell_price = 0.0
        revenue = final_sell_price * shares_per_order - transaction_costs
        trade_pnl = revenue - (entry_price * shares_per_order + transaction_costs)
        capital += revenue
        df.at[df.index[-1], "Trade_PnL"] = trade_pnl
        df.at[df.index[-1], "Position"] = 0

    final_capital = capital
    df["Equity"] = df["Capital"].fillna(method='ffill')

    # Performance Metriken
    trades = df[df["Trade_PnL"] != 0].copy()
    wins = trades[trades["Trade_PnL"] > 0].shape[0]
    losses = trades[trades["Trade_PnL"] < 0].shape[0]
    total_trades = trades.shape[0]
    win_rate = wins / total_trades if total_trades > 0 else 0
    profit_factor = (trades[trades["Trade_PnL"] > 0]["Trade_PnL"].sum() /
                     abs(trades[trades["Trade_PnL"] < 0]["Trade_PnL"].sum())) if losses > 0 else np.inf
    max_drawdown = (df["Equity"].cummax() - df["Equity"]).max()

    results = {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "net_profit": final_capital - initial_capital,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown
    }

    return results, df


def backtest_strategy(
        df,
        initial_capital=10000.0,
        shares_per_order=100,
        initial_stop_loss=1.0,  # Absoluter Abstand unter Einstiegspreis (für Long)
        profit_target=2.0,  # Absoluter Gewinnabstand über Einstiegspreis (für Long)
        use_trailing_stop=False,
        trailing_stop_distance=1.0,  # Absoluter Abstand für den Trailing Stop
        transaction_costs=0.0,
        slippage=0.0,
        allow_short=False
):
    """
    Backtestingfunktion mit Trailing-Stop-Logik nach dem Vorbild des TradingView-Skripts.

    Handelslogik:
    - Kauf, wenn Signal "Up" und Position nicht offen.
    - Verkauf, wenn Stop-Loss (initial oder trailing) oder Profit-Target erreicht.
    - Trailing Stop (Long):
        Nach Entry:
         - max_price_during_trade = entry_price
         - Wenn Kurs steigt, max_price_during_trade = max(max_price_during_trade, current_price)
         - Wenn max_price_during_trade > entry_price, trailing_stop = max_price_during_trade - trailing_stop_distance
         - Verkauf, wenn current_price <= trailing_stop
    """

    df = df.copy()
    df["Position"] = 0
    df["Entry_Price"] = np.nan
    df["Stop_Loss"] = np.nan
    df["Trailing_Stop"] = np.nan
    df["Trade_PnL"] = 0.0
    df["Capital"] = 0.0

    capital = initial_capital
    position_open = False
    entry_price = np.nan
    stop_loss_price = np.nan
    trailing_stop_price = np.nan
    max_price_during_trade = np.nan
    min_price_during_trade = np.nan  # Für Shorts, falls benötigt

    for i in range(len(df)):
        current_price = df["Close_orig"].iloc[i]
        signal = df["Predicted_Trend"].iloc[i]

        if not position_open:
            # Kaufbedingung: wenn Signal = "Up"
            if signal == "Up":
                buy_price = current_price + slippage
                cost = buy_price * shares_per_order + transaction_costs
                if cost <= capital:
                    position_open = True
                    entry_price = buy_price
                    # Initial Stop-Loss
                    stop_loss_price = entry_price - initial_stop_loss
                    trailing_stop_price = np.nan
                    max_price_during_trade = entry_price  # Für Trailing Stop (Long)
                    min_price_during_trade = np.nan

                    capital -= cost
                    df.at[df.index[i], "Position"] = 1
                    df.at[df.index[i], "Entry_Price"] = entry_price
                    df.at[df.index[i], "Stop_Loss"] = stop_loss_price
                    df.at[df.index[i], "Trailing_Stop"] = trailing_stop_price
        else:
            # Position offen (Long)
            # Aktualisiere max_price_during_trade für Long
            if not np.isnan(max_price_during_trade):
                if current_price > max_price_during_trade:
                    max_price_during_trade = current_price

            # Trailing Stop berechnen, wenn use_trailing_stop=True
            # Trailing Stop wird nur gesetzt, wenn der Kurs über den Einstiegspreis gestiegen ist
            if use_trailing_stop and not np.isnan(max_price_during_trade) and max_price_during_trade > entry_price:
                trailing_stop_price = max_price_during_trade - trailing_stop_distance
            else:
                trailing_stop_price = np.nan

            # Aktuellen effektiven Stop-Loss berechnen
            # Der effektive Stop-Loss ist immer der höhere von initial_stop_loss und trailing_stop (falls vorhanden),
            # ABER in diesem Beispiel orientieren wir uns streng an der Logik des Skripts:
            # Dort wird der initiale Stop-Loss gesetzt, und wenn Trailing greift, überschreibt er den Stop-Loss.
            # Wir folgen hier der Logik: initial_stop_loss ist nur bis Trailing greift. Sobald Trailing greift,
            # ist dieser maßgeblich.

            effective_stop_loss = stop_loss_price
            if use_trailing_stop and not np.isnan(trailing_stop_price):
                # Wenn Trailing Stop aktiv ist, überschreibt er den initialen Stop-Loss
                effective_stop_loss = trailing_stop_price

            # Verkaufsbedingungen:
            # 1. Stop-Loss/Trailing-Stop unterschritten
            # 2. Profit-Target erreicht (current_price >= entry_price + profit_target)
            if current_price <= effective_stop_loss or (current_price >= entry_price + profit_target):
                # Position schließen
                sell_price = current_price - slippage
                if sell_price < 0:
                    sell_price = 0.0
                revenue = sell_price * shares_per_order - transaction_costs
                trade_pnl = revenue - (entry_price * shares_per_order + transaction_costs)
                capital += revenue
                position_open = False
                df.at[df.index[i], "Trade_PnL"] = trade_pnl
                df.at[df.index[i], "Position"] = 0
                entry_price = np.nan
                stop_loss_price = np.nan
                trailing_stop_price = np.nan
                max_price_during_trade = np.nan
                min_price_during_trade = np.nan
            else:
                # Position weiter halten
                df.at[df.index[i], "Position"] = 1
                df.at[df.index[i], "Entry_Price"] = entry_price
                df.at[df.index[i], "Stop_Loss"] = stop_loss_price
                df.at[df.index[i], "Trailing_Stop"] = trailing_stop_price

        df.at[df.index[i], "Capital"] = capital

    # Falls am Ende noch eine Position offen ist, glattstellen
    if position_open:
        final_sell_price = max(df["Close_orig"].iloc[-1] - slippage, 0.0)
        revenue = final_sell_price * shares_per_order - transaction_costs
        trade_pnl = revenue - (entry_price * shares_per_order + transaction_costs)
        capital += revenue
        df.at[df.index[-1], "Trade_PnL"] = trade_pnl
        df.at[df.index[-1], "Position"] = 0

    final_capital = capital
    df["Equity"] = df["Capital"].fillna(method='ffill')

    # Performance Metriken
    trades = df[df["Trade_PnL"] != 0].copy()
    wins = trades[trades["Trade_PnL"] > 0].shape[0]
    losses = trades[trades["Trade_PnL"] < 0].shape[0]
    total_trades = trades.shape[0]
    win_rate = wins / total_trades if total_trades > 0 else 0
    profit_factor = (trades[trades["Trade_PnL"] > 0]["Trade_PnL"].sum() /
                     abs(trades[trades["Trade_PnL"] < 0]["Trade_PnL"].sum())) if losses > 0 else np.inf
    max_drawdown = (df["Equity"].cummax() - df["Equity"]).max()

    results = {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "net_profit": final_capital - initial_capital,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown
    }

    print(f'results:{results}')

    return results, df


def backtest_strategy_multiple_positions1(
    df,
    shares_per_order=1,
    initial_stop_loss=1.0,
    profit_target=2.0,
    use_trailing_stop=False,
    trailing_stop_distance=1.0
):
    """
    Backtesting-Funktion, bei der jede "Up"-Vorhersage ("Predicted_Trend")
    eine neue Long-Position eröffnet. Es gibt:
      - Kein Startkapital (Thema 'unbegrenztes' Kaufen).
      - Keine Transaktionskosten.
      - Keine Slippage.
      - Keine Short-Positionen.

    Parameter:
    -----------
    df : pd.DataFrame
        Muss mindestens die Spalten ["Close_orig", "Predicted_Trend"] enthalten.
    shares_per_order : int
        Anzahl an gekauften Stücken pro Signal.
    initial_stop_loss : float
        Absoluter Abstand unter dem Einstiegspreis, bei dem verkauft wird (Stop-Loss).
    profit_target : float
        Absoluter Abstand oberhalb des Einstiegspreises, bei dem verkauft wird (Take Profit).
    use_trailing_stop : bool
        Ob ein Trailing Stop genutzt wird.
    trailing_stop_distance : float
        Ab welcher Distanz unter dem bisherigen Maximum der Kurs verkauft wird.

    Rückgabe:
    -----------
    results : dict
        Verschiedene Kennzahlen (z.B. total_profit, average_profit, win_rate).
    df : pd.DataFrame
        DataFrame mit zusätzlichen Spalten zu den Trades.

    Hinweise:
    -----------
    - Da hier kein Kapital limitiert, kann ein Signal immer ein weiteres Mal
      eine Position eröffnen, selbst wenn bereits Positionen offen sind.
    - Am Ende des Durchlaufs werden alle offenen Positionen zum letzten
      Kurs glattgestellt.
    """

    # Kopie des DataFrames, um keine Seiteneffekte zu erzeugen
    df = df.copy()

    # Wir speichern jeden offenen Trade als Dictionary in einer Liste "open_trades"
    open_trades = []
    closed_trades = []

    # Erstellen von Spalten für Debugging/Analyse (optional)
    df["Open_Positions_Count"] = 0
    df["Closed_Trades"] = 0
    df["PnL_Closed"] = 0.0

    for i in range(len(df)):
        current_price = df["Close_orig"].iloc[i]
        signal = df["Predicted_Trend"].iloc[i]

        # 1) Neue Long-Position für jedes "Up"-Signal eröffnen:
        if signal == "Up":
            open_trades.append({
                "entry_index": i,
                "entry_price": current_price,
                "shares": shares_per_order,
                "stop_loss_price": current_price - initial_stop_loss,
                "profit_target_price": current_price + profit_target,
                "max_price_during_trade": current_price  # Für Trailing Stop
            })

        # 2) Jede offene Position überprüfen, ob sie geschlossen werden muss:
        trades_to_close = []
        for trade in open_trades:
            # Trailing Stop:
            if use_trailing_stop:
                # Aktualisiere das Maximum
                if current_price > trade["max_price_during_trade"]:
                    trade["max_price_during_trade"] = current_price

                # trailing_stop wird aktiv, sobald Kurs über entry_price gestiegen ist
                if trade["max_price_during_trade"] > trade["entry_price"]:
                    trailing_stop_price = trade["max_price_during_trade"] - trailing_stop_distance
                else:
                    trailing_stop_price = trade["stop_loss_price"]  # Fallback auf initialen Stop

                effective_stop_loss = max(trade["stop_loss_price"], trailing_stop_price)
            else:
                effective_stop_loss = trade["stop_loss_price"]

            # Verkaufsbedingungen
            #  a) Kurs <= Stop-Loss
            #  b) Kurs >= Profit-Target
            if current_price <= effective_stop_loss or current_price >= trade["profit_target_price"]:
                # Schließe diesen Trade
                exit_price = current_price
                trade_pnl = (exit_price - trade["entry_price"]) * trade["shares"]
                closed_trades.append({
                    "entry_index": trade["entry_index"],
                    "exit_index": i,
                    "entry_price": trade["entry_price"],
                    "exit_price": exit_price,
                    "shares": trade["shares"],
                    "pnl": trade_pnl
                })
                trades_to_close.append(trade)

        # 3) Geschlossene Trades entfernen
        for trade in trades_to_close:
            open_trades.remove(trade)

        # 4) Optional: Für Analysezwecke in DataFrame schreiben
        df.at[df.index[i], "Open_Positions_Count"] = len(open_trades)
        df.at[df.index[i], "Closed_Trades"] = len(trades_to_close)
        df.at[df.index[i], "PnL_Closed"] = sum([t["pnl"] for t in closed_trades]) if closed_trades else 0.0

    # 5) Falls am Ende noch Positionen offen sind, schließen wir sie zum letzten Kurs
    if len(open_trades) > 0:
        final_price = df["Close_orig"].iloc[-1]
        for trade in open_trades:
            trade_pnl = (final_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "entry_index": trade["entry_index"],
                "exit_index": len(df) - 1,
                "entry_price": trade["entry_price"],
                "exit_price": final_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
        open_trades.clear()

    # 6) Kennzahlen berechnen
    all_pnls = [t["pnl"] for t in closed_trades]
    total_profit = np.sum(all_pnls)
    total_trades = len(all_pnls)
    wins = np.sum([1 for pnl in all_pnls if pnl > 0])
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    average_profit = np.mean(all_pnls) if total_trades > 0 else 0.0

    results = {
        "total_profit": total_profit,
        "average_profit_per_trade": average_profit,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses
    }

    return results, df


def backtest_strategy_multiple_positions2(
    df,
    shares_per_order=1,
    initial_stop_loss=1.0,
    profit_target=2.0,
    use_trailing_stop=False,
    trailing_stop_distance=1.0
):
    """
    Backtesting-Funktion, bei der jede "Up"-Vorhersage ("Predicted_Trend")
    eine neue Long-Position eröffnet. Es gibt:
      - Kein Startkapital (d. h. Kapital ist nicht limitiert).
      - Keine Transaktionskosten.
      - Keine Slippage.
      - Keine Short-Positionen.
      - Mehrere parallele Trades sind möglich.

    Zusätzlich wird in einer neuen Spalte "Equity" der jeweilige
    "Kontostand" (mark-to-market) am Ende jedes Bars gespeichert.
    """

    # Kopie des DataFrames, um keine Seiteneffekte zu erzeugen
    df = df.copy()

    # Wir speichern jeden offenen Trade als Dictionary in einer Liste "open_trades"
    open_trades = []
    closed_trades = []

    # Spalten, die wir für Debugging/Analyse anlegen
    df["Open_Positions_Count"] = 0      # Wie viele Positionen aktuell offen
    df["Closed_Trades"] = 0            # Wie viele Positionen wurden in diesem Bar geschlossen
    df["PnL_Closed"] = 0.0             # Realisierter Gewinn/Verlust bis zu diesem Zeitpunkt
    df["Equity"] = 0.0                 # Realisierter + unrealiserter Gewinn/Verlust (alle offenen Trades)

    for i in range(len(df)):
        current_price = df["Close_orig"].iloc[i]
        signal = df["Predicted_Trend"].iloc[i]

        # 1) Neue Long-Position für jedes "Up"-Signal eröffnen:
        if signal == "Up":
            open_trades.append({
                "entry_index": i,
                "entry_price": current_price,
                "shares": shares_per_order,
                "stop_loss_price": current_price - initial_stop_loss,
                "profit_target_price": current_price + profit_target,
                "max_price_during_trade": current_price  # Für Trailing Stop (wenn aktiviert)
            })

        # 2) Jede offene Position überprüfen, ob sie geschlossen werden muss:
        trades_to_close = []
        for trade in open_trades:
            # Trailing Stop:
            if use_trailing_stop:
                # Aktualisiere das Maximum
                if current_price > trade["max_price_during_trade"]:
                    trade["max_price_during_trade"] = current_price

                # trailing_stop wird aktiv, sobald Kurs über entry_price gestiegen ist
                if trade["max_price_during_trade"] > trade["entry_price"]:
                    trailing_stop_price = trade["max_price_during_trade"] - trailing_stop_distance
                else:
                    trailing_stop_price = trade["stop_loss_price"]  # Fallback auf initialen Stop

                effective_stop_loss = max(trade["stop_loss_price"], trailing_stop_price)
            else:
                effective_stop_loss = trade["stop_loss_price"]

            # Verkaufsbedingungen:
            #  a) Kurs <= Stop-Loss
            #  b) Kurs >= Profit-Target
            if current_price <= effective_stop_loss or current_price >= trade["profit_target_price"]:
                # Schließe diesen Trade
                exit_price = current_price
                trade_pnl = (exit_price - trade["entry_price"]) * trade["shares"]
                closed_trades.append({
                    "entry_index": trade["entry_index"],
                    "exit_index": i,
                    "entry_price": trade["entry_price"],
                    "exit_price": exit_price,
                    "shares": trade["shares"],
                    "pnl": trade_pnl
                })
                trades_to_close.append(trade)

        # 3) Geschlossene Trades entfernen
        for trade in trades_to_close:
            open_trades.remove(trade)

        # 4) Realisierten PnL und Equity (realisierter + unrealisierter PnL) berechnen
        #    a) Realisiert: Summe aller geschlossenen Trades
        realized_pnl = sum(t["pnl"] for t in closed_trades)

        #    b) Unrealisiert: Summe des (aktuellen Kurses - Einstiegspreis) * Stück für alle offenen Trades
        #       (mark to market)
        unrealized_pnl = sum((current_price - t["entry_price"]) * t["shares"] for t in open_trades)

        #    c) Equity = realized + unrealized
        equity = realized_pnl + unrealized_pnl

        # 5) Werte ins DataFrame schreiben
        df.at[df.index[i], "Open_Positions_Count"] = len(open_trades)
        df.at[df.index[i], "Closed_Trades"] = len(trades_to_close)
        df.at[df.index[i], "PnL_Closed"] = realized_pnl
        df.at[df.index[i], "Equity"] = equity

    # 6) Falls am Ende noch Positionen offen sind, schließen wir sie zum letzten Kurs
    #    (oder du lässt sie einfach weiterlaufen, je nach Bedarf).
    if len(open_trades) > 0:
        final_price = df["Close_orig"].iloc[-1]
        for trade in open_trades:
            trade_pnl = (final_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "entry_index": trade["entry_index"],
                "exit_index": len(df) - 1,
                "entry_price": trade["entry_price"],
                "exit_price": final_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
        open_trades.clear()

        # Letztes Bar updaten, da wir jetzt alles glattgestellt haben
        realized_pnl = sum(t["pnl"] for t in closed_trades)
        unrealized_pnl = 0.0  # keine offenen Trades mehr
        df.at[df.index[-1], "PnL_Closed"] = realized_pnl
        df.at[df.index[-1], "Equity"] = realized_pnl

    # 7) Finale Kennzahlen
    all_pnls = [t["pnl"] for t in closed_trades]
    total_profit = np.sum(all_pnls)
    total_trades = len(all_pnls)
    wins = np.sum([1 for pnl in all_pnls if pnl > 0])
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    average_profit = np.mean(all_pnls) if total_trades > 0 else 0.0

    results = {
        "total_profit": total_profit,
        "average_profit_per_trade": average_profit,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses
    }

    return results, df


def backtest_strategy_multiple_positions3(
    df,
    shares_per_order=1,
    initial_stop_loss=1.0,
    profit_target=2.0,
    use_trailing_stop=False,
    trailing_stop_distance=1.0,
    use_max_holding_period=False,
    max_holding_period=10,
    use_nth_signal=False,
    nth_signal=2
):
    """
    Backtesting-Funktion, bei der jede "Up"-Vorhersage ("Predicted_Trend")
    (unter bestimmten Bedingungen) eine neue Long-Position eröffnet. Folgende Optionen sind möglich:

      - Kein Startkapital (d. h. Kapital ist nicht limitiert).
      - Keine Transaktionskosten, keine Slippage.
      - Keine Short-Positionen, nur Long.
      - Mehrere parallele Trades sind möglich.
      - Optional: maximale Haltezeit (Bars), nach der Trades zwangsweise geschlossen werden.
      - Optional: Kaufen erst nach dem n-ten aufeinanderfolgenden "Up"-Signal mit jeweils ansteigendem Kurs.

    Parameter:
    -----------
    df : pd.DataFrame
        Muss mindestens die Spalten ["Close_orig", "Predicted_Trend"] enthalten.
    shares_per_order : int
        Anzahl an gekauften Stücken pro Signal.
    initial_stop_loss : float
        Absoluter Abstand unter dem Einstiegspreis, bei dem verkauft wird (Stop-Loss).
    profit_target : float
        Absoluter Abstand oberhalb des Einstiegspreises, bei dem verkauft wird (Take Profit).
    use_trailing_stop : bool
        Ob ein Trailing Stop genutzt wird.
    trailing_stop_distance : float
        Ab welcher Distanz unter dem bisherigen Maximum der Kurs verkauft wird.
    use_max_holding_period : bool
        Ob Trades nach einer bestimmten Anzahl Bars automatisch geschlossen werden sollen.
    max_holding_period : int
        Anzahl Bars (Kerzen), nach denen ein Trade automatisch geschlossen wird.
    use_nth_signal : bool
        Ob erst nach n aufeinanderfolgenden "Up"-Signalen mit ansteigendem Kurs gekauft wird.
    nth_signal : int
        Anzahl der aufeinanderfolgenden "Up"-Signale mit steigendem Kurs, nach denen erst ein Kauf erfolgt.

    Rückgabe:
    -----------
    results : dict
        Verschiedene Kennzahlen (z.B. total_profit, average_profit, win_rate).
    df : pd.DataFrame
        DataFrame mit zusätzlichen Spalten zu den Trades.
    """

    # Kopie des DataFrames, um keine Seiteneffekte zu erzeugen
    df = df.copy()

    # Wir speichern jeden offenen Trade als Dictionary in einer Liste "open_trades"
    open_trades = []
    closed_trades = []

    # Für die "nth_signal"-Logik: Zähler für aufeinanderfolgende Up-Signale mit steigendem Kurs
    consecutive_up_count = 0

    # Spalten, die wir für Debugging/Analyse anlegen
    df["Open_Positions_Count"] = 0      # Wie viele Positionen aktuell offen
    df["Closed_Trades"] = 0            # Wie viele Positionen wurden in diesem Bar geschlossen
    df["PnL_Closed"] = 0.0             # Realisierter Gewinn/Verlust bis zu diesem Zeitpunkt
    df["Equity"] = 0.0                 # Realisierter + unrealiserter Gewinn/Verlust

    previous_price = None  # Für Vergleich, ob Kurs aufsteigt

    for i in range(len(df)):
        current_price = df["Close_orig"].iloc[i]
        signal = df["Predicted_Trend"].iloc[i]

        # 1) Logik für "nth_signal": Zählen wir erst Up-Signale + steigenden Kurs?
        if use_nth_signal:
            # Prüfen, ob das aktuelle Signal "Up" ist und der Kurs gegenüber dem letzten Bar steigt
            # (Achtung: i > 0, sonst gibt es keinen Vortageskurs)
            if i > 0 and signal == "Up" and previous_price is not None and current_price > previous_price:
                consecutive_up_count += 1
            else:
                consecutive_up_count = 0

            # Wenn der Zähler das nth_signal erreicht hat, öffnen wir eine Position
            if consecutive_up_count == nth_signal:
                open_trades.append({
                    "entry_index": i,
                    "entry_price": current_price,
                    "shares": shares_per_order,
                    "stop_loss_price": current_price - initial_stop_loss,
                    "profit_target_price": current_price + profit_target,
                    "max_price_during_trade": current_price  # für Trailing Stop
                })
                # Nach dem Kauf Zähler zurücksetzen (damit die Trades nicht in jedem Bar getriggert werden)
                consecutive_up_count = 0
        else:
            # Standardfall: bei jedem "Up"-Signal kaufen
            if signal == "Up":
                open_trades.append({
                    "entry_index": i,
                    "entry_price": current_price,
                    "shares": shares_per_order,
                    "stop_loss_price": current_price - initial_stop_loss,
                    "profit_target_price": current_price + profit_target,
                    "max_price_during_trade": current_price
                })

        # 2) Jede offene Position überprüfen, ob sie geschlossen werden muss:
        trades_to_close = []
        for trade in open_trades:

            # a) Trailing Stop (falls aktiv)
            if use_trailing_stop:
                # Aktualisiere das Maximum
                if current_price > trade["max_price_during_trade"]:
                    trade["max_price_during_trade"] = current_price

                # trailing_stop wird aktiv, sobald Kurs über entry_price gestiegen ist
                if trade["max_price_during_trade"] > trade["entry_price"]:
                    trailing_stop_price = trade["max_price_during_trade"] - trailing_stop_distance
                else:
                    trailing_stop_price = trade["stop_loss_price"]  # Fallback auf initialen Stop

                effective_stop_loss = max(trade["stop_loss_price"], trailing_stop_price)
            else:
                effective_stop_loss = trade["stop_loss_price"]

            # b) Check Stop-Loss / Take Profit
            if (current_price <= effective_stop_loss) or (current_price >= trade["profit_target_price"]):
                # Schließe diesen Trade
                exit_price = current_price
                trade_pnl = (exit_price - trade["entry_price"]) * trade["shares"]
                closed_trades.append({
                    "entry_index": trade["entry_index"],
                    "exit_index": i,
                    "entry_price": trade["entry_price"],
                    "exit_price": exit_price,
                    "shares": trade["shares"],
                    "pnl": trade_pnl
                })
                trades_to_close.append(trade)
                continue  # nach dem Schließen nicht weiter prüfen

            # c) Check maximale Haltezeit
            if use_max_holding_period:
                # Falls die Anzahl der Bars seit dem Einstieg >= max_holding_period ist, schließen
                holding_period = i - trade["entry_index"]
                if holding_period >= max_holding_period:
                    exit_price = current_price
                    trade_pnl = (exit_price - trade["entry_price"]) * trade["shares"]
                    closed_trades.append({
                        "entry_index": trade["entry_index"],
                        "exit_index": i,
                        "entry_price": trade["entry_price"],
                        "exit_price": exit_price,
                        "shares": trade["shares"],
                        "pnl": trade_pnl
                    })
                    trades_to_close.append(trade)

        # 3) Geschlossene Trades entfernen
        for trade in trades_to_close:
            open_trades.remove(trade)

        # 4) Realisierten PnL und Equity (realisierter + unrealisierter PnL) berechnen
        #    a) Realisiert: Summe aller geschlossenen Trades
        realized_pnl = sum(t["pnl"] for t in closed_trades)

        #    b) Unrealisiert: Summe des (aktuellen Kurses - Einstiegspreis) * Stück für alle offenen Trades
        unrealized_pnl = sum((current_price - t["entry_price"]) * t["shares"] for t in open_trades)

        #    c) Equity = realized + unrealized
        equity = realized_pnl + unrealized_pnl

        # 5) Werte ins DataFrame schreiben
        df.at[df.index[i], "Open_Positions_Count"] = len(open_trades)
        df.at[df.index[i], "Closed_Trades"] = len(trades_to_close)
        df.at[df.index[i], "PnL_Closed"] = realized_pnl
        df.at[df.index[i], "Equity"] = equity

        # Update previous_price für nächsten Bar
        previous_price = current_price

    # 6) Falls am Ende noch Positionen offen sind, schließen wir sie zum letzten Kurs
    if len(open_trades) > 0:
        final_price = df["Close_orig"].iloc[-1]
        for trade in open_trades:
            trade_pnl = (final_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "entry_index": trade["entry_index"],
                "exit_index": len(df) - 1,
                "entry_price": trade["entry_price"],
                "exit_price": final_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
        open_trades.clear()

        # Letztes Bar updaten, da jetzt alles glattgestellt
        realized_pnl = sum(t["pnl"] for t in closed_trades)
        unrealized_pnl = 0.0  # keine offenen Trades mehr
        df.at[df.index[-1], "PnL_Closed"] = realized_pnl
        df.at[df.index[-1], "Equity"] = realized_pnl

    # 7) Finale Kennzahlen
    all_pnls = [t["pnl"] for t in closed_trades]
    total_profit = np.sum(all_pnls)
    total_trades = len(all_pnls)
    wins = np.sum([1 for pnl in all_pnls if pnl > 0])
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    average_profit = np.mean(all_pnls) if total_trades > 0 else 0.0

    results = {
        "total_profit": total_profit,
        "average_profit_per_trade": average_profit,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses
    }

    return results, df


def backtest_strategy_multiple_positions4(
        df,
        shares_per_order=1,
        initial_stop_loss=1.0,
        profit_target=2.0,
        use_trailing_stop=False,
        trailing_stop_distance=1.0,
        use_max_holding_period=False,
        max_holding_period=10,
        use_nth_signal=False,
        nth_signal=2
):
    """
    Backtesting-Funktion, bei der Long-Positionen auf Basis von "Up"-Signalen eröffnet werden.

    Erweiterungen:
      - Jeder Trade erhält eine eindeutige ID (trade_id).
      - Neue Spalten zeigen, welche Trades pro Bar geöffnet/geschlossen wurden.
      - Gewinn/Verlust jedes Trades wird klar ausgewiesen.

    Parameter:
    -----------
    df : pd.DataFrame
        Muss mindestens die Spalten ["Close_orig", "Predicted_Trend"] enthalten.
    shares_per_order : int
        Anzahl an gekauften Stücken pro Signal.
    initial_stop_loss : float
        Absoluter Abstand unter dem Einstiegspreis, bei dem verkauft wird (Stop-Loss).
    profit_target : float
        Absoluter Abstand oberhalb des Einstiegspreises, bei dem verkauft wird (Take Profit).
    use_trailing_stop : bool
        Ob ein Trailing Stop genutzt wird.
    trailing_stop_distance : float
        Abstand unterhalb des bisherigen Maximums (Kurs - trailing_stop_distance),
        bei dem ein Verkauf ausgelöst wird.
    use_max_holding_period : bool
        Ob Trades nach einer bestimmten Anzahl Bars (Kerzen) automatisch geschlossen werden sollen.
    max_holding_period : int
        Anzahl Bars, nach denen ein Trade automatisch geschlossen wird (wenn aktiviert).
    use_nth_signal : bool
        Ob erst nach n aufeinanderfolgenden "Up"-Signalen mit steigendem Kurs gekauft wird.
    nth_signal : int
        Anzahl der aufeinanderfolgenden "Up"-Signale + steigender Kurs, nach denen erst ein Kauf erfolgt.

    Rückgabe:
    -----------
    results : dict
        Kennzahlen der Strategie (z.B. total_profit, average_profit_per_trade, win_rate).
    df : pd.DataFrame
        DataFrame mit zusätzlichen Spalten für Analyse und Nachvollziehbarkeit.
    """

    # Kopie, um das Original nicht zu verändern
    df = df.copy()

    # Liste für offene Trades, Liste für abgeschlossene Trades
    open_trades = []
    closed_trades = []

    # Eindeutiger Zähler für trade_id
    trade_id_counter = 0

    # Zähler für "n hintereinanderfolgende Up-Signale + steigender Kurs"
    consecutive_up_count = 0

    # Neue Spalten anlegen, um Käufe/Verkäufe transparent zu machen
    df["Open_Positions_Count"] = 0  # Wie viele Positionen sind aktuell offen
    df["Closed_Trades_Count"] = 0  # Wie viele Positionen wurden in diesem Bar geschlossen
    df["Opened_Trade_IDs"] = [[] for _ in range(len(df))]  # Liste der pro Bar geöffneten trade_ids
    df["Closed_Trade_IDs"] = [[] for _ in range(len(df))]  # Liste der pro Bar geschlossenen trade_ids
    df["Closed_Trade_PnLs"] = [[] for _ in range(len(df))]  # Liste der PnLs der in diesem Bar geschlossenen Trades
    df["PnL_Closed"] = 0.0  # Summe realisierter Gewinne/Verluste bis zu diesem Zeitpunkt
    df["Equity"] = 0.0  # Realisierter + unrealisierter Gewinn/Verlust

    previous_price = None

    for i in range(len(df)):
        current_price = df["Close_orig"].iloc[i]
        signal = df["Predicted_Trend"].iloc[i]

        # --- 1) Logik für nth_signal: erst nach n aufeinanderfolgenden Up+steigender Kurs kaufen ---
        if use_nth_signal:
            if i > 0 and signal == "Up" and previous_price is not None and current_price > previous_price:
                consecutive_up_count += 1
            else:
                consecutive_up_count = 0

            if consecutive_up_count == nth_signal:
                # => Neuen Trade öffnen
                trade_id_counter += 1
                open_trades.append({
                    "trade_id": trade_id_counter,
                    "entry_index": i,
                    "entry_price": current_price,
                    "shares": shares_per_order,
                    "stop_loss_price": current_price - initial_stop_loss,
                    "profit_target_price": current_price + profit_target,
                    "max_price_during_trade": current_price
                })
                # Trade-ID in DataFrame eintragen
                df.at[df.index[i], "Opened_Trade_IDs"].append(trade_id_counter)

                # Reset, damit nicht jeder Folgetag einen weiteren Kauf triggert
                consecutive_up_count = 0
        else:
            # --- Standardfall: bei jedem "Up"-Signal direkt kaufen ---
            if signal == "Up":
                trade_id_counter += 1
                open_trades.append({
                    "trade_id": trade_id_counter,
                    "entry_index": i,
                    "entry_price": current_price,
                    "shares": shares_per_order,
                    "stop_loss_price": current_price - initial_stop_loss,
                    "profit_target_price": current_price + profit_target,
                    "max_price_during_trade": current_price
                })
                df.at[df.index[i], "Opened_Trade_IDs"].append(trade_id_counter)

        # --- 2) Alle offenen Positionen prüfen, ob sie geschlossen werden müssen ---
        trades_to_close = []
        for trade in open_trades:

            # a) Effektiven Stop (Trailing oder initial)
            if use_trailing_stop:
                if current_price > trade["max_price_during_trade"]:
                    trade["max_price_during_trade"] = current_price
                # trailing_stop wird aktiv, wenn der Kurs über den entry_price gestiegen ist
                if trade["max_price_during_trade"] > trade["entry_price"]:
                    trailing_stop_price = trade["max_price_during_trade"] - trailing_stop_distance
                else:
                    trailing_stop_price = trade["stop_loss_price"]
                effective_stop_loss = max(trade["stop_loss_price"], trailing_stop_price)
            else:
                effective_stop_loss = trade["stop_loss_price"]

            # b) Prüfen, ob Stop-Loss oder Take-Profit erreicht
            if current_price <= effective_stop_loss or current_price >= trade["profit_target_price"]:
                trades_to_close.append(trade)
                continue

            # c) Prüfen, ob maximale Haltezeit überschritten
            if use_max_holding_period:
                holding_period = i - trade["entry_index"]
                if holding_period >= max_holding_period:
                    trades_to_close.append(trade)

        # --- 3) Trades schließen, die in diesem Bar fällig sind ---
        closed_this_bar = []
        for trade in trades_to_close:
            exit_price = current_price
            trade_pnl = (exit_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "trade_id": trade["trade_id"],
                "entry_index": trade["entry_index"],
                "exit_index": i,
                "entry_price": trade["entry_price"],
                "exit_price": exit_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })

            # Für DataFrame-Spalten
            closed_this_bar.append((trade["trade_id"], trade_pnl))
            open_trades.remove(trade)

        # Liste der in diesem Bar geschlossenen trade_ids + PnLs
        if closed_this_bar:
            df.at[df.index[i], "Closed_Trade_IDs"] = [x[0] for x in closed_this_bar]
            df.at[df.index[i], "Closed_Trade_PnLs"] = [x[1] for x in closed_this_bar]

        # --- 4) Realisierten PnL und Equity aktualisieren ---
        realized_pnl = sum(t["pnl"] for t in closed_trades)
        unrealized_pnl = sum((current_price - t["entry_price"]) * t["shares"] for t in open_trades)
        equity = realized_pnl + unrealized_pnl

        df.at[df.index[i], "Closed_Trades_Count"] = len(closed_this_bar)
        df.at[df.index[i], "Open_Positions_Count"] = len(open_trades)
        df.at[df.index[i], "PnL_Closed"] = realized_pnl
        df.at[df.index[i], "Equity"] = equity

        # Price-Merkhilfe
        previous_price = current_price

    # --- 5) Am Ende alle offenen Positionen zum letzten Kurs schließen (optional) ---
    if len(open_trades) > 0:
        final_price = df["Close_orig"].iloc[-1]
        closed_this_bar = []
        for trade in open_trades:
            trade_pnl = (final_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "trade_id": trade["trade_id"],
                "entry_index": trade["entry_index"],
                "exit_index": len(df) - 1,
                "entry_price": trade["entry_price"],
                "exit_price": final_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
            closed_this_bar.append((trade["trade_id"], trade_pnl))

        open_trades.clear()
        # Letztes Bar updaten (PnL und Equity)
        realized_pnl = sum(t["pnl"] for t in closed_trades)
        df.at[df.index[-1], "PnL_Closed"] = realized_pnl
        df.at[df.index[-1], "Closed_Trade_IDs"] = [x[0] for x in closed_this_bar]
        df.at[df.index[-1], "Closed_Trade_PnLs"] = [x[1] for x in closed_this_bar]
        df.at[df.index[-1], "Closed_Trades_Count"] = len(closed_this_bar)
        df.at[df.index[-1], "Equity"] = realized_pnl  # kein unrealized PnL mehr

    # --- 6) Finale Kennzahlen berechnen ---
    all_pnls = [t["pnl"] for t in closed_trades]
    total_profit = np.sum(all_pnls)
    total_trades = len(all_pnls)
    wins = np.sum([1 for pnl in all_pnls if pnl > 0])
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    average_profit = np.mean(all_pnls) if total_trades > 0 else 0.0

    results = {
        "total_profit": total_profit,
        "average_profit_per_trade": average_profit,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses
    }

    return results, df


def backtest_strategy_multiple_positions5(
    df,
    shares_per_order=1,
    initial_stop_loss=1.0,
    profit_target=2.0,
    use_trailing_stop=False,
    trailing_stop_distance=1.0,
    use_max_holding_period=False,
    max_holding_period=10,
    use_nth_signal=False,
    nth_signal=2
):
    """
    Backtesting-Funktion nach dem alten Muster (d. h. die Logik, wann gekauft/verkauft wird,
    bleibt unverändert). Wir verwenden die Original-Signale für "Up"/"Stable"/... aus der Spalte
    'Predicted_Trend', damit dieselbe Trading-Logik greift wie zuvor (somit gleicher Profit).

    Anschließend überschreiben wir in der Ausgabe-DataFrame
    aber die Spalte 'Predicted_Trend':
      - 'Up' nur, wenn tatsächlich gekauft wird
      - 'Sold' nur, wenn verkauft wird
      - 'Up|Sold', wenn beides in derselben Zeile passiert
      - 'Stable' sonst.

    Damit hast du eine konsistente Anzeige in 'Predicted_Trend', ohne das ursprüngliche
    Trading-Verhalten zu verändern.
    """

    # 1) DataFrame kopieren und Original-Signale retten
    df = df.copy()
    df["Original_Signal"] = df["Predicted_Trend"]  # So bleibt die Kauf-/Verkaufslogik identisch
    df["Predicted_Trend"] = "Stable"               # Wir überschreiben später, je nach Ereignis

    # Listen für offene / geschlossene Trades
    open_trades = []
    closed_trades = []

    # Eindeutiger Zähler für Trade-IDs
    trade_id_counter = 0

    # Zähler für "n hintereinanderfolgende Up-Signale mit steigendem Kurs"
    consecutive_up_count = 0

    # Zusätzliche Spalten anlegen
    df["Open_Positions_Count"] = 0
    df["Closed_Trades_Count"] = 0
    df["Opened_Trade_IDs"] = [[] for _ in range(len(df))]
    df["Closed_Trade_IDs"] = [[] for _ in range(len(df))]
    df["Closed_Trade_PnLs"] = [[] for _ in range(len(df))]
    df["PnL_Closed"] = 0.0
    df["Equity"] = 0.0

    previous_price = None

    # -------------------------------------------------------
    # 2) Schleife über alle Bars: Trading-Logik bleibt WIE BISHER
    # -------------------------------------------------------
    for i in range(len(df)):
        current_price = df["Close_orig"].iloc[i]
        signal = df["Original_Signal"].iloc[i]  # Wichtig: Original-Signal für Kauf-/Verkaufslogik

        opened_this_bar = []
        closed_this_bar = []

        # --- a) Bestimmen, ob wir heute kaufen ---
        if use_nth_signal:
            # (1) Aufeinanderfolgende "Up"-Signale + steigender Kurs
            if i > 0 and previous_price is not None and signal == "Up" and current_price > previous_price:
                consecutive_up_count += 1
            else:
                consecutive_up_count = 0

            # (2) Wenn Zähler erreicht, Trade eröffnen
            if consecutive_up_count == nth_signal:
                trade_id_counter += 1
                open_trades.append({
                    "trade_id": trade_id_counter,
                    "entry_index": i,
                    "entry_price": current_price,
                    "shares": shares_per_order,
                    "stop_loss_price": current_price - initial_stop_loss,
                    "profit_target_price": current_price + profit_target,
                    "max_price_during_trade": current_price
                })
                opened_this_bar.append(trade_id_counter)
                consecutive_up_count = 0
        else:
            # Standard: Wenn Original-Signal == "Up", kaufen
            if signal == "Up":
                trade_id_counter += 1
                open_trades.append({
                    "trade_id": trade_id_counter,
                    "entry_index": i,
                    "entry_price": current_price,
                    "shares": shares_per_order,
                    "stop_loss_price": current_price - initial_stop_loss,
                    "profit_target_price": current_price + profit_target,
                    "max_price_during_trade": current_price
                })
                opened_this_bar.append(trade_id_counter)

        # --- b) Offene Positionen checken -> verkaufen? ---
        trades_to_close = []
        for trade in open_trades:
            # Trailing Stop
            if use_trailing_stop:
                if current_price > trade["max_price_during_trade"]:
                    trade["max_price_during_trade"] = current_price
                if trade["max_price_during_trade"] > trade["entry_price"]:
                    trailing_stop_price = trade["max_price_during_trade"] - trailing_stop_distance
                else:
                    trailing_stop_price = trade["stop_loss_price"]
                effective_stop_loss = max(trade["stop_loss_price"], trailing_stop_price)
            else:
                effective_stop_loss = trade["stop_loss_price"]

            # Stop-Loss / Take-Profit
            if current_price <= effective_stop_loss or current_price >= trade["profit_target_price"]:
                trades_to_close.append(trade)
                continue

            # Max. Haltezeit
            if use_max_holding_period:
                holding_period = i - trade["entry_index"]
                if holding_period >= max_holding_period:
                    trades_to_close.append(trade)

        # Positionen schließen
        for trade in trades_to_close:
            exit_price = current_price
            trade_pnl = (exit_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "trade_id": trade["trade_id"],
                "entry_index": trade["entry_index"],
                "exit_index": i,
                "entry_price": trade["entry_price"],
                "exit_price": exit_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
            closed_this_bar.append((trade["trade_id"], trade_pnl))
            open_trades.remove(trade)

        # --- c) Equity, Realized PnL
        realized_pnl = sum(t["pnl"] for t in closed_trades)
        unrealized_pnl = sum((current_price - t["entry_price"]) * t["shares"] for t in open_trades)
        equity = realized_pnl + unrealized_pnl

        df.at[df.index[i], "Open_Positions_Count"] = len(open_trades)
        df.at[df.index[i], "Closed_Trades_Count"] = len(closed_this_bar)
        df.at[df.index[i], "PnL_Closed"] = realized_pnl
        df.at[df.index[i], "Equity"] = equity
        df.at[df.index[i], "Opened_Trade_IDs"] = opened_this_bar
        df.at[df.index[i], "Closed_Trade_IDs"] = [x[0] for x in closed_this_bar]
        df.at[df.index[i], "Closed_Trade_PnLs"] = [x[1] for x in closed_this_bar]

        # Price-History
        previous_price = current_price

        # -------------------------------------------------------
        # 3) NACHDEM Kauf/Verkauf entschieden -> Spalte Predicted_Trend manipulieren
        # -------------------------------------------------------
        # a) Falls wir Käufe hatten, 'Up'
        if len(opened_this_bar) > 0:
            df.at[df.index[i], "Predicted_Trend"] = "Up"
        # b) Falls wir Verkäufe hatten, 'Sold' oder 'Up|Sold'
        if len(closed_this_bar) > 0:
            if df.at[df.index[i], "Predicted_Trend"] == "Up":
                df.at[df.index[i], "Predicted_Trend"] = "Up|Sold"
            else:
                df.at[df.index[i], "Predicted_Trend"] = "Sold"

    # -------------------------------------------------------
    # 4) Am Ende alle offenen Positionen schließen (so wie im alten Code)
    # -------------------------------------------------------
    if len(open_trades) > 0:
        final_price = df["Close_orig"].iloc[-1]
        closed_this_bar = []
        for trade in open_trades:
            trade_pnl = (final_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "trade_id": trade["trade_id"],
                "entry_index": trade["entry_index"],
                "exit_index": len(df) - 1,
                "entry_price": trade["entry_price"],
                "exit_price": final_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
            closed_this_bar.append((trade["trade_id"], trade_pnl))

        open_trades.clear()

        # Letztes Bar updaten
        df.at[df.index[-1], "Closed_Trade_IDs"] = [x[0] for x in closed_this_bar]
        df.at[df.index[-1], "Closed_Trade_PnLs"] = [x[1] for x in closed_this_bar]
        df.at[df.index[-1], "Closed_Trades_Count"] = len(closed_this_bar)

        # 'Sold' oder 'Up|Sold'
        if df.at[df.index[-1], "Predicted_Trend"] == "Up":
            df.at[df.index[-1], "Predicted_Trend"] = "Up|Sold"
        elif df.at[df.index[-1], "Predicted_Trend"] == "Stable":
            df.at[df.index[-1], "Predicted_Trend"] = "Sold"
        elif df.at[df.index[-1], "Predicted_Trend"] == "Up|Sold":
            pass  # Bleibt so
        else:
            df.at[df.index[-1], "Predicted_Trend"] = "Sold"

        realized_pnl = sum(t["pnl"] for t in closed_trades)
        df.at[df.index[-1], "PnL_Closed"] = realized_pnl
        df.at[df.index[-1], "Equity"] = realized_pnl  # keine offenen Trades mehr

    # -------------------------------------------------------
    # 5) Finale Kennzahlen
    # -------------------------------------------------------
    all_pnls = [t["pnl"] for t in closed_trades]
    total_profit = np.sum(all_pnls)
    total_trades = len(all_pnls)
    wins = np.sum([1 for pnl in all_pnls if pnl > 0])
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    average_profit = np.mean(all_pnls) if total_trades > 0 else 0.0

    results = {
        "total_profit": total_profit,
        "average_profit_per_trade": average_profit,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses
    }

    return results, df


def backtest_strategy_multiple_positions6(
    df,
    shares_per_order=1,
    initial_stop_loss=1.0,
    profit_target=2.0,
    use_trailing_stop=False,
    trailing_stop_distance=1.0,
    use_max_holding_period=False,
    max_holding_period=10,
    use_nth_signal=False,
    nth_signal=2,
    leverage=1.0
):
    """
    Backtesting-Funktion nach dem alten Muster (d. h. die Logik, wann gekauft/verkauft wird,
    bleibt unverändert). Wir verwenden die Original-Signale für "Up"/"Stable"/... aus der Spalte
    'Predicted_Trend', damit dieselbe Trading-Logik greift wie zuvor (somit gleicher Profit).

    Anschließend überschreiben wir in der Ausgabe-DataFrame
    aber die Spalte 'Predicted_Trend':
      - 'Up' nur, wenn tatsächlich gekauft wird
      - 'Sold' nur, wenn verkauft wird
      - 'Up|Sold', wenn beides in derselben Zeile passiert
      - 'Stable' sonst.

    NEU:
      - Hebel (leverage): Multiplikation des Kaufwertes wird durch den Hebel geteilt,
        um den real erforderlichen Kapitalbedarf zu berechnen (Kapitalbedarf = Kaufwert / Hebel).
      - capital_usage: aufsummierte Kapitalnutzung pro Bar über den ganzen Backtest.
      - peak_capital_usage: höchster gleichzeitiger Kapitalbedarf (Maximum der Capital_Usage pro Bar).
    """

    # 1) DataFrame kopieren und Original-Signale retten
    df = df.copy()
    df["Original_Signal"] = df["Predicted_Trend"]  # So bleibt die Kauf-/Verkaufslogik identisch
    df["Predicted_Trend"] = "Stable"               # Wir überschreiben später, je nach Ereignis

    # Listen für offene / geschlossene Trades
    open_trades = []
    closed_trades = []

    # Eindeutiger Zähler für Trade-IDs
    trade_id_counter = 0

    # Zähler für "n hintereinanderfolgende Up-Signale mit steigendem Kurs"
    consecutive_up_count = 0

    # Zusätzliche Spalten anlegen
    df["Open_Positions_Count"] = 0
    df["Closed_Trades_Count"] = 0
    df["Opened_Trade_IDs"] = [[] for _ in range(len(df))]
    df["Closed_Trade_IDs"] = [[] for _ in range(len(df))]
    df["Closed_Trade_PnLs"] = [[] for _ in range(len(df))]
    df["PnL_Closed"] = 0.0
    df["Equity"] = 0.0

    # -------------------- NEU: Kapitalnutzung --------------------
    df["Capital_Usage"] = 0.0  # Kapitalbedarf pro Bar
    peak_capital_usage = 0.0   # Maximale gleichzeitige Kapitalnutzung
    capital_usage = 0.0        # Aufsummierter Kapitalbedarf

    previous_price = None

    # -------------------------------------------------------
    # 2) Schleife über alle Bars: Trading-Logik bleibt WIE BISHER
    # -------------------------------------------------------
    for i in range(len(df)):
        current_price = df["Close_orig"].iloc[i]
        signal = df["Original_Signal"].iloc[i]  # Wichtig: Original-Signal für Kauf-/Verkaufslogik

        opened_this_bar = []
        closed_this_bar = []

        # --- a) Bestimmen, ob wir heute kaufen ---
        if use_nth_signal:
            # (1) Aufeinanderfolgende "Up"-Signale + steigender Kurs
            if i > 0 and previous_price is not None and signal == "Up" and current_price > previous_price:
                consecutive_up_count += 1
            else:
                consecutive_up_count = 0

            # (2) Wenn Zähler erreicht, Trade eröffnen
            if consecutive_up_count == nth_signal:
                trade_id_counter += 1
                open_trades.append({
                    "trade_id": trade_id_counter,
                    "entry_index": i,
                    "entry_price": current_price,
                    "shares": shares_per_order,
                    "stop_loss_price": current_price - initial_stop_loss,
                    "profit_target_price": current_price + profit_target,
                    "max_price_during_trade": current_price
                })
                opened_this_bar.append(trade_id_counter)
                consecutive_up_count = 0
        else:
            # Standard: Wenn Original-Signal == "Up", kaufen
            if signal == "Up":
                trade_id_counter += 1
                open_trades.append({
                    "trade_id": trade_id_counter,
                    "entry_index": i,
                    "entry_price": current_price,
                    "shares": shares_per_order,
                    "stop_loss_price": current_price - initial_stop_loss,
                    "profit_target_price": current_price + profit_target,
                    "max_price_during_trade": current_price
                })
                opened_this_bar.append(trade_id_counter)

        # --- b) Offene Positionen checken -> verkaufen? ---
        trades_to_close = []
        for trade in open_trades:
            # Trailing Stop
            if use_trailing_stop:
                if current_price > trade["max_price_during_trade"]:
                    trade["max_price_during_trade"] = current_price
                if trade["max_price_during_trade"] > trade["entry_price"]:
                    trailing_stop_price = trade["max_price_during_trade"] - trailing_stop_distance
                else:
                    trailing_stop_price = trade["stop_loss_price"]
                effective_stop_loss = max(trade["stop_loss_price"], trailing_stop_price)
            else:
                effective_stop_loss = trade["stop_loss_price"]

            # Stop-Loss / Take-Profit
            if current_price <= effective_stop_loss or current_price >= trade["profit_target_price"]:
                trades_to_close.append(trade)
                continue

            # Max. Haltezeit
            if use_max_holding_period:
                holding_period = i - trade["entry_index"]
                if holding_period >= max_holding_period:
                    trades_to_close.append(trade)

        # Positionen schließen
        for trade in trades_to_close:
            exit_price = current_price
            trade_pnl = (exit_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "trade_id": trade["trade_id"],
                "entry_index": trade["entry_index"],
                "exit_index": i,
                "entry_price": trade["entry_price"],
                "exit_price": exit_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
            closed_this_bar.append((trade["trade_id"], trade_pnl))
            open_trades.remove(trade)

        # --- c) Equity, Realized PnL
        realized_pnl = sum(t["pnl"] for t in closed_trades)
        unrealized_pnl = sum((current_price - t["entry_price"]) * t["shares"] for t in open_trades)
        equity = realized_pnl + unrealized_pnl

        df.at[df.index[i], "Open_Positions_Count"] = len(open_trades)
        df.at[df.index[i], "Closed_Trades_Count"] = len(closed_this_bar)
        df.at[df.index[i], "PnL_Closed"] = realized_pnl
        df.at[df.index[i], "Equity"] = equity
        df.at[df.index[i], "Opened_Trade_IDs"] = opened_this_bar
        df.at[df.index[i], "Closed_Trade_IDs"] = [x[0] for x in closed_this_bar]
        df.at[df.index[i], "Closed_Trade_PnLs"] = [x[1] for x in closed_this_bar]

        # -------------------- d) Kapitalbedarf (NEU) --------------------
        # "Eingesetztes Kapital" = Summe (entry_price * shares / leverage) für alle offenen Trades
        # => Sofern du den Kurs in "Euro" oder "USD" hast, und shares die Stückzahl.
        # => Falls du Mark-to-Market-Kapitalbindung abbilden willst, kannst du
        #    (current_price * shares) / leverage nehmen.
        capital_usage_now = 0.0
        for t in open_trades:
            # Kaufwert / leverage
            capital_usage_now += (t["entry_price"] * t["shares"]) / leverage

        df.at[df.index[i], "Capital_Usage"] = capital_usage_now

        # Aktualisiere peak_capital_usage
        if capital_usage_now > peak_capital_usage:
            peak_capital_usage = capital_usage_now

        # Aufsummieren in capital_usage (Summe über alle Bars)
        capital_usage += capital_usage_now

        # Price-History
        previous_price = current_price

        # -------------------------------------------------------
        # 3) NACHDEM Kauf/Verkauf entschieden -> Spalte Predicted_Trend manipulieren
        # -------------------------------------------------------
        # a) Falls wir Käufe hatten, 'Up'
        if len(opened_this_bar) > 0:
            df.at[df.index[i], "Predicted_Trend"] = "Up"
        # b) Falls wir Verkäufe hatten, 'Sold' oder 'Up|Sold'
        if len(closed_this_bar) > 0:
            if df.at[df.index[i], "Predicted_Trend"] == "Up":
                df.at[df.index[i], "Predicted_Trend"] = "Up|Sold"
            else:
                df.at[df.index[i], "Predicted_Trend"] = "Sold"

    # -------------------------------------------------------
    # 4) Am Ende alle offenen Positionen schließen
    # -------------------------------------------------------
    if len(open_trades) > 0:
        final_price = df["Close_orig"].iloc[-1]
        closed_this_bar = []
        for trade in open_trades:
            trade_pnl = (final_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "trade_id": trade["trade_id"],
                "entry_index": trade["entry_index"],
                "exit_index": len(df) - 1,
                "entry_price": trade["entry_price"],
                "exit_price": final_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
            closed_this_bar.append((trade["trade_id"], trade_pnl))

        open_trades.clear()

        # Letztes Bar updaten
        df.at[df.index[-1], "Closed_Trade_IDs"] = [x[0] for x in closed_this_bar]
        df.at[df.index[-1], "Closed_Trade_PnLs"] = [x[1] for x in closed_this_bar]
        df.at[df.index[-1], "Closed_Trades_Count"] = len(closed_this_bar)

        # 'Sold' oder 'Up|Sold'
        if df.at[df.index[-1], "Predicted_Trend"] == "Up":
            df.at[df.index[-1], "Predicted_Trend"] = "Up|Sold"
        elif df.at[df.index[-1], "Predicted_Trend"] == "Stable":
            df.at[df.index[-1], "Predicted_Trend"] = "Sold"
        elif df.at[df.index[-1], "Predicted_Trend"] == "Up|Sold":
            pass  # Bleibt so
        else:
            df.at[df.index[-1], "Predicted_Trend"] = "Sold"

        realized_pnl = sum(t["pnl"] for t in closed_trades)
        df.at[df.index[-1], "PnL_Closed"] = realized_pnl
        df.at[df.index[-1], "Equity"] = realized_pnl  # keine offenen Trades mehr

    # -------------------------------------------------------
    # 5) Finale Kennzahlen
    # -------------------------------------------------------
    all_pnls = [t["pnl"] for t in closed_trades]
    total_profit = np.sum(all_pnls)
    total_trades = len(all_pnls)
    wins = np.sum([1 for pnl in all_pnls if pnl > 0])
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    average_profit = np.mean(all_pnls) if total_trades > 0 else 0.0

    rendite = total_profit / peak_capital_usage

    results = {
        "total_profit": total_profit,
        "average_profit_per_trade": average_profit,
        "rendite": rendite,

        "total_trades": total_trades,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        # Neu hinzugefügt:
        "peak_capital_usage": peak_capital_usage,
        "capital_usage": capital_usage,
        "leverage": leverage
    }

    return results, df

# funktioniert bisher bis auf die fehlerhafte kummulierung von dem gesamt genutzt kapital und evtl auch max_drawdown_pct_equity war falsch
def backtest_strategy_multiple_positions_7(
    df,
    shares_per_order=1,
    initial_stop_loss=1.0,
    profit_target=2.0,
    use_trailing_stop=False,
    trailing_stop_distance=1.0,
    use_max_holding_period=False,
    max_holding_period=10,
    use_nth_signal=False,
    nth_signal=2,
    leverage=1.0
):
    """
    Backtesting-Funktion nach dem alten Muster (d. h. die Logik, wann gekauft/verkauft wird,
    bleibt unverändert). Wir verwenden die Original-Signale für "Up"/"Stable"/... aus der Spalte
    'Predicted_Trend', damit dieselbe Trading-Logik greift wie zuvor (somit gleicher Profit).

    Anschließend überschreiben wir in der Ausgabe-DataFrame
    aber die Spalte 'Predicted_Trend':
      - 'Up' nur, wenn tatsächlich gekauft wird
      - 'Sold' nur, wenn verkauft wird
      - 'Up|Sold', wenn beides in derselben Zeile passiert
      - 'Stable' sonst.

    NEU:
      - Hebel (leverage): Multiplikation des Kaufwertes wird durch den Hebel geteilt,
        um den real erforderlichen Kapitalbedarf zu berechnen (Kapitalbedarf = Kaufwert / Hebel).
      - capital_usage: aufsummierte Kapitalnutzung pro Bar über den ganzen Backtest.
      - peak_capital_usage: höchster gleichzeitiger Kapitalbedarf (Maximum der Capital_Usage pro Bar).
      - max_drawdown: maximaler Drawdown der Equity-Kurve (absolut).
      - max_drawdown_pct_equity: maximaler prozentualer Drawdown (klassisch über Equity-Kurve).
      - calmar_ratio: Rendite / max_drawdown_pct_equity.
    """

    # 1) DataFrame kopieren und Original-Signale retten
    df = df.copy()
    df["Original_Signal"] = df["Predicted_Trend"]  # So bleibt die Kauf-/Verkaufslogik identisch
    df["Predicted_Trend"] = "Stable"               # Wir überschreiben später, je nach Ereignis

    # Listen für offene / geschlossene Trades
    open_trades = []
    closed_trades = []

    # Eindeutiger Zähler für Trade-IDs
    trade_id_counter = 0

    # Zähler für "n hintereinanderfolgende Up-Signale mit steigendem Kurs"
    consecutive_up_count = 0

    # Zusätzliche Spalten anlegen
    df["Open_Positions_Count"] = 0
    df["Closed_Trades_Count"] = 0
    df["Opened_Trade_IDs"] = [[] for _ in range(len(df))]
    df["Closed_Trade_IDs"] = [[] for _ in range(len(df))]
    df["Closed_Trade_PnLs"] = [[] for _ in range(len(df))]
    df["PnL_Closed"] = 0.0
    df["Equity"] = 0.0

    # -------------------- NEU: Kapitalnutzung --------------------
    df["Capital_Usage"] = 0.0  # Kapitalbedarf pro Bar
    peak_capital_usage = 0.0   # Maximale gleichzeitige Kapitalnutzung
    capital_usage = 0.0        # Aufsummierter Kapitalbedarf

    previous_price = None

    # -------------------------------------------------------
    # 2) Schleife über alle Bars: Trading-Logik bleibt WIE BISHER
    # -------------------------------------------------------
    for i in range(len(df)):
        current_price = df["Close_orig"].iloc[i]
        signal = df["Original_Signal"].iloc[i]  # Wichtig: Original-Signal für Kauf-/Verkaufslogik

        opened_this_bar = []
        closed_this_bar = []

        # --- a) Bestimmen, ob wir heute kaufen ---
        if use_nth_signal:
            # (1) Aufeinanderfolgende "Up"-Signale + steigender Kurs
            if i > 0 and previous_price is not None and signal == "Up" and current_price > previous_price:
                consecutive_up_count += 1
            else:
                consecutive_up_count = 0

            # (2) Wenn Zähler erreicht, Trade eröffnen
            if consecutive_up_count == nth_signal:
                trade_id_counter += 1
                open_trades.append({
                    "trade_id": trade_id_counter,
                    "entry_index": i,
                    "entry_price": current_price,
                    "shares": shares_per_order,
                    "stop_loss_price": current_price - initial_stop_loss,
                    "profit_target_price": current_price + profit_target,
                    "max_price_during_trade": current_price
                })
                opened_this_bar.append(trade_id_counter)
                consecutive_up_count = 0
        else:
            # Standard: Wenn Original-Signal == "Up", kaufen
            if signal == "Up":
                trade_id_counter += 1
                open_trades.append({
                    "trade_id": trade_id_counter,
                    "entry_index": i,
                    "entry_price": current_price,
                    "shares": shares_per_order,
                    "stop_loss_price": current_price - initial_stop_loss,
                    "profit_target_price": current_price + profit_target,
                    "max_price_during_trade": current_price
                })
                opened_this_bar.append(trade_id_counter)

        # --- b) Offene Positionen checken -> verkaufen? ---
        trades_to_close = []
        for trade in open_trades:
            # Trailing Stop
            if use_trailing_stop:
                if current_price > trade["max_price_during_trade"]:
                    trade["max_price_during_trade"] = current_price
                if trade["max_price_during_trade"] > trade["entry_price"]:
                    trailing_stop_price = trade["max_price_during_trade"] - trailing_stop_distance
                else:
                    trailing_stop_price = trade["stop_loss_price"]
                effective_stop_loss = max(trade["stop_loss_price"], trailing_stop_price)
            else:
                effective_stop_loss = trade["stop_loss_price"]

            # Stop-Loss / Take-Profit
            if current_price <= effective_stop_loss or current_price >= trade["profit_target_price"]:
                trades_to_close.append(trade)
                continue

            # Max. Haltezeit
            if use_max_holding_period:
                holding_period = i - trade["entry_index"]
                if holding_period >= max_holding_period:
                    trades_to_close.append(trade)

        # Positionen schließen
        for trade in trades_to_close:
            exit_price = current_price
            trade_pnl = (exit_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "trade_id": trade["trade_id"],
                "entry_index": trade["entry_index"],
                "exit_index": i,
                "entry_price": trade["entry_price"],
                "exit_price": exit_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
            closed_this_bar.append((trade["trade_id"], trade_pnl))
            open_trades.remove(trade)

        # --- c) Equity, Realized PnL
        realized_pnl = sum(t["pnl"] for t in closed_trades)
        unrealized_pnl = sum((current_price - t["entry_price"]) * t["shares"] for t in open_trades)
        equity = realized_pnl + unrealized_pnl

        df.at[df.index[i], "Open_Positions_Count"] = len(open_trades)
        df.at[df.index[i], "Closed_Trades_Count"] = len(closed_this_bar)
        df.at[df.index[i], "PnL_Closed"] = realized_pnl
        df.at[df.index[i], "Equity"] = equity
        df.at[df.index[i], "Opened_Trade_IDs"] = opened_this_bar
        df.at[df.index[i], "Closed_Trade_IDs"] = [x[0] for x in closed_this_bar]
        df.at[df.index[i], "Closed_Trade_PnLs"] = [x[1] for x in closed_this_bar]

        # -------------------- d) Kapitalbedarf (NEU) --------------------
        capital_usage_now = 0.0
        for t in open_trades:
            capital_usage_now += (t["entry_price"] * t["shares"]) / leverage

        df.at[df.index[i], "Capital_Usage"] = capital_usage_now

        if capital_usage_now > peak_capital_usage:
            peak_capital_usage = capital_usage_now

        capital_usage += capital_usage_now

        previous_price = current_price

        # -------------------------------------------------------
        # 3) NACHDEM Kauf/Verkauf entschieden -> Spalte Predicted_Trend manipulieren
        # -------------------------------------------------------
        if len(opened_this_bar) > 0:
            df.at[df.index[i], "Predicted_Trend"] = "Up"
        if len(closed_this_bar) > 0:
            if df.at[df.index[i], "Predicted_Trend"] == "Up":
                df.at[df.index[i], "Predicted_Trend"] = "Up|Sold"
            else:
                df.at[df.index[i], "Predicted_Trend"] = "Sold"

    # -------------------------------------------------------
    # 4) Am Ende alle offenen Positionen schließen
    # -------------------------------------------------------
    if len(open_trades) > 0:
        final_price = df["Close_orig"].iloc[-1]
        closed_this_bar = []
        for trade in open_trades:
            trade_pnl = (final_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "trade_id": trade["trade_id"],
                "entry_index": trade["entry_index"],
                "exit_index": len(df) - 1,
                "entry_price": trade["entry_price"],
                "exit_price": final_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
            closed_this_bar.append((trade["trade_id"], trade_pnl))

        open_trades.clear()

        df.at[df.index[-1], "Closed_Trade_IDs"] = [x[0] for x in closed_this_bar]
        df.at[df.index[-1], "Closed_Trade_PnLs"] = [x[1] for x in closed_this_bar]
        df.at[df.index[-1], "Closed_Trades_Count"] = len(closed_this_bar)

        if df.at[df.index[-1], "Predicted_Trend"] == "Up":
            df.at[df.index[-1], "Predicted_Trend"] = "Up|Sold"
        elif df.at[df.index[-1], "Predicted_Trend"] == "Stable":
            df.at[df.index[-1], "Predicted_Trend"] = "Sold"
        elif df.at[df.index[-1], "Predicted_Trend"] == "Up|Sold":
            pass
        else:
            df.at[df.index[-1], "Predicted_Trend"] = "Sold"

        realized_pnl = sum(t["pnl"] for t in closed_trades)
        df.at[df.index[-1], "PnL_Closed"] = realized_pnl
        df.at[df.index[-1], "Equity"] = realized_pnl  # keine offenen Trades mehr

    # -------------------------------------------------------
    # 5) Finale Kennzahlen
    # -------------------------------------------------------
    all_pnls = [t["pnl"] for t in closed_trades]
    total_profit = np.sum(all_pnls)
    total_trades = len(all_pnls)
    wins = np.sum([1 for pnl in all_pnls if pnl > 0])
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    average_profit = np.mean(all_pnls) if total_trades > 0 else 0.0

    # peak_capital_usage könnte 0 sein, also absichern:
    if peak_capital_usage != 0.0:
        rendite = total_profit / peak_capital_usage
    else:
        rendite = 0.0

    # -------------------- (1) Absoluter max Drawdown (im Code bereits vorhanden) --------------------
    # Falls du den schon als "max_drawdown" haben willst (z. B. differenziert vom equity-basierten):
    equity_array = df["Equity"].values
    if len(equity_array) == 0:
        max_drawdown = 0.0
    else:
        running_max_equity = np.maximum.accumulate(equity_array)
        drawdown_abs = running_max_equity - equity_array
        max_drawdown = np.max(drawdown_abs)

    # -------------------- (2) Klassischer, equity-basierter prozentualer Drawdown --------------------
    # Variante 1 (wie gewünscht):
    if len(equity_array) == 0:
        max_drawdown_pct_equity = 0.0
    else:
        # running_max_equity = np.maximum.accumulate(equity_array)
        # drawdowns = (running_max_equity - equity_array) / running_max_equity
        # max_drawdown_pct_equity = np.max(drawdowns)

        running_max_equity = np.maximum.accumulate(equity_array)

        # Sicherstellen, dass wir nicht durch 0 teilen:
        running_max_equity_safe = np.where(running_max_equity == 0, 1e-9, running_max_equity)
        drawdowns = (running_max_equity_safe - equity_array) / running_max_equity_safe

        max_drawdown_pct_equity = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    # -------------------- (3) Calmar Ratio (Rendite / max_drawdown_pct_equity) ----------------------
    if max_drawdown_pct_equity == 0.0:
        calmar_ratio = np.inf
    else:
        calmar_ratio = rendite / max_drawdown_pct_equity


    results = {
        "total_profit": round(total_profit, 0),
        "average_profit_per_trade": round(average_profit, 0),
        "rendite": round(rendite, 4),
        "total_trades": total_trades,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "peak_capital_usage": round(peak_capital_usage, 0),
        "capital_usage": round(capital_usage, 0),
        "leverage": leverage,

        # (1) Absoluter max Drawdown (bereits vorhanden)
        "max_drawdown_abs": round(max_drawdown, 2),
        # (2) Der klassische prozentuale Drawdown über die Equity-Kurve
        "max_drawdown_pct_equity": max_drawdown_pct_equity,
        # (3) Calmar Ratio
        "calmar_ratio": calmar_ratio
    }

    return results, df


import numpy as np
import pandas as pd


def backtest_strategy_multiple_positions_old(
        df,
        shares_per_order=1,
        initial_stop_loss=1.0,
        profit_target=2.0,
        use_trailing_stop=False,
        trailing_stop_distance=1.0,
        use_max_holding_period=False,
        max_holding_period=10,
        use_nth_signal=False,
        nth_signal=2,
        leverage=1.0
):
    """
    Optimierte Backtesting-Funktion ohne initial_capital.
    Kapitalnutzung wird nicht kumuliert, sondern basierend auf den Kaufkosten der geöffneten Trades berechnet.
    """

    # 1) DataFrame kopieren und Original-Signale retten
    df = df.copy()
    df["Original_Signal"] = df["Predicted_Trend"]  # So bleibt die Kauf-/Verkaufslogik identisch
    df["Predicted_Trend"] = "Stable"  # Wir überschreiben später, je nach Ereignis

    # Listen für offene / geschlossene Trades
    open_trades = []
    closed_trades = []

    # Eindeutiger Zähler für Trade-IDs
    trade_id_counter = 0

    # Zähler für "n hintereinanderfolgende Up-Signale mit steigendem Kurs"
    consecutive_up_count = 0

    # Zusätzliche Spalten anlegen
    df["Open_Positions_Count"] = 0
    df["Closed_Trades_Count"] = 0
    df["Opened_Trade_IDs"] = [[] for _ in range(len(df))]
    df["Closed_Trade_IDs"] = [[] for _ in range(len(df))]
    df["Closed_Trade_PnLs"] = [[] for _ in range(len(df))]
    df["PnL_Closed"] = 0.0
    df["Equity"] = 0.0  # Start ohne initial_capital

    # -------------------- NEU: Kapitalnutzung --------------------
    df["Capital_Usage"] = 0.0  # Kapitalbedarf pro Bar
    peak_capital_usage = 0.0  # Maximale gleichzeitige Kapitalnutzung

    previous_price = None

    # -------------------------------------------------------
    # 2) Schleife über alle Bars: Trading-Logik bleibt WIE BISHER
    # -------------------------------------------------------
    for i in range(len(df)):
        current_price = df["Close_orig"].iloc[i]
        signal = df["Original_Signal"].iloc[i]  # Wichtig: Original-Signal für Kauf-/Verkaufslogik

        opened_this_bar = []
        closed_this_bar = []

        # --- a) Bestimmen, ob wir heute kaufen ---
        if use_nth_signal:
            # (1) Aufeinanderfolgende "Up"-Signale + steigender Kurs
            if i > 0 and previous_price is not None and signal == "Up" and current_price > previous_price:
                consecutive_up_count += 1
            else:
                consecutive_up_count = 0

            # (2) Wenn Zähler erreicht, Trade eröffnen
            if consecutive_up_count == nth_signal:
                trade_id_counter += 1
                open_trades.append({
                    "trade_id": trade_id_counter,
                    "entry_index": i,
                    "entry_price": current_price,
                    "shares": shares_per_order,
                    "stop_loss_price": current_price - initial_stop_loss,
                    "profit_target_price": current_price + profit_target,
                    "max_price_during_trade": current_price
                })
                opened_this_bar.append(trade_id_counter)
                consecutive_up_count = 0
        else:
            # Standard: Wenn Original-Signal == "Up", kaufen
            if signal == "Up":
                trade_id_counter += 1
                open_trades.append({
                    "trade_id": trade_id_counter,
                    "entry_index": i,
                    "entry_price": current_price,
                    "shares": shares_per_order,
                    "stop_loss_price": current_price - initial_stop_loss,
                    "profit_target_price": current_price + profit_target,
                    "max_price_during_trade": current_price
                })
                opened_this_bar.append(trade_id_counter)

        # --- b) Offene Positionen checken -> verkaufen? ---
        trades_to_close = []
        for trade in open_trades:
            # Trailing Stop
            if use_trailing_stop:
                if current_price > trade["max_price_during_trade"]:
                    trade["max_price_during_trade"] = current_price
                if trade["max_price_during_trade"] > trade["entry_price"]:
                    trailing_stop_price = trade["max_price_during_trade"] - trailing_stop_distance
                else:
                    trailing_stop_price = trade["stop_loss_price"]
                effective_stop_loss = max(trade["stop_loss_price"], trailing_stop_price)
            else:
                effective_stop_loss = trade["stop_loss_price"]

            # Stop-Loss / Take-Profit
            if current_price <= effective_stop_loss or current_price >= trade["profit_target_price"]:
                trades_to_close.append(trade)
                continue

            # Max. Haltezeit
            if use_max_holding_period:
                holding_period = i - trade["entry_index"]
                if holding_period >= max_holding_period:
                    trades_to_close.append(trade)

        # Positionen schließen
        for trade in trades_to_close:
            exit_price = current_price
            trade_pnl = (exit_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "trade_id": trade["trade_id"],
                "entry_index": trade["entry_index"],
                "exit_index": i,
                "entry_price": trade["entry_price"],
                "exit_price": exit_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
            closed_this_bar.append((trade["trade_id"], trade_pnl))
            open_trades.remove(trade)

        # --- c) Equity, Realized PnL ---
        realized_pnl = sum(t["pnl"] for t in closed_trades)
        unrealized_pnl = sum((current_price - t["entry_price"]) * t["shares"] for t in open_trades)
        equity = realized_pnl + unrealized_pnl

        df.at[df.index[i], "Open_Positions_Count"] = len(open_trades)
        df.at[df.index[i], "Closed_Trades_Count"] = len(closed_this_bar)
        df.at[df.index[i], "PnL_Closed"] = realized_pnl
        df.at[df.index[i], "Equity"] = equity
        df.at[df.index[i], "Opened_Trade_IDs"] = opened_this_bar
        df.at[df.index[i], "Closed_Trade_IDs"] = [x[0] for x in closed_this_bar]
        df.at[df.index[i], "Closed_Trade_PnLs"] = [x[1] for x in closed_this_bar]

        # -------------------- d) Kapitalbedarf (NEU) --------------------
        capital_usage_now = 0.0
        for t in open_trades:
            capital_usage_now += (t["entry_price"] * t["shares"]) / leverage

        df.at[df.index[i], "Capital_Usage"] = capital_usage_now

        # Aktualisiere peak_capital_usage basierend auf der aktuellen Kapitalnutzung
        if capital_usage_now > peak_capital_usage:
            peak_capital_usage = capital_usage_now

        previous_price = current_price

        # -------------------------------------------------------
        # 3) NACHDEM Kauf/Verkauf entschieden -> Spalte Predicted_Trend manipulieren
        # -------------------------------------------------------
        if len(opened_this_bar) > 0:
            df.at[df.index[i], "Predicted_Trend"] = "Up"
        if len(closed_this_bar) > 0:
            if df.at[df.index[i], "Predicted_Trend"] == "Up":
                df.at[df.index[i], "Predicted_Trend"] = "Up|Sold"
            else:
                df.at[df.index[i], "Predicted_Trend"] = "Sold"

    # -------------------------------------------------------
    # 4) Am Ende alle offenen Positionen schließen
    # -------------------------------------------------------
    if len(open_trades) > 0:
        final_price = df["Close_orig"].iloc[-1]
        closed_this_bar = []
        for trade in open_trades:
            trade_pnl = (final_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "trade_id": trade["trade_id"],
                "entry_index": trade["entry_index"],
                "exit_index": len(df) - 1,
                "entry_price": trade["entry_price"],
                "exit_price": final_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
            closed_this_bar.append((trade["trade_id"], trade_pnl))

        open_trades.clear()

        df.at[df.index[-1], "Closed_Trade_IDs"] = [x[0] for x in closed_this_bar]
        df.at[df.index[-1], "Closed_Trade_PnLs"] = [x[1] for x in closed_this_bar]
        df.at[df.index[-1], "Closed_Trades_Count"] = len(closed_this_bar)

        if df.at[df.index[-1], "Predicted_Trend"] == "Up":
            df.at[df.index[-1], "Predicted_Trend"] = "Up|Sold"
        elif df.at[df.index[-1], "Predicted_Trend"] == "Stable":
            df.at[df.index[-1], "Predicted_Trend"] = "Sold"
        elif df.at[df.index[-1], "Predicted_Trend"] == "Up|Sold":
            pass
        else:
            df.at[df.index[-1], "Predicted_Trend"] = "Sold"

        realized_pnl = sum(t["pnl"] for t in closed_trades)
        df.at[df.index[-1], "PnL_Closed"] = realized_pnl
        df.at[df.index[-1], "Equity"] = realized_pnl  # keine offenen Trades mehr

    # -------------------------------------------------------
    # 5) Finale Kennzahlen
    # -------------------------------------------------------
    all_pnls = [t["pnl"] for t in closed_trades]
    total_profit = np.sum(all_pnls)
    total_trades = len(all_pnls)
    wins = np.sum([1 for pnl in all_pnls if pnl > 0])
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    average_profit = np.mean(all_pnls) if total_trades > 0 else 0.0

    # Berechnung der Rendite
    rendite = total_profit / peak_capital_usage if peak_capital_usage > 0.0 else 0.0

    # -------------------- (1) Absoluter max Drawdown (im Code bereits vorhanden) --------------------
    equity_array = df["Equity"].values
    if len(equity_array) == 0:
        max_drawdown = 0.0
    else:
        running_max_equity = np.maximum.accumulate(equity_array)
        drawdown_abs = running_max_equity - equity_array
        max_drawdown = np.max(drawdown_abs)

    # -------------------- (2) Klassischer, equity-basierter prozentualer Drawdown --------------------
    if len(equity_array) == 0:
        max_drawdown_pct_equity = 0.0
    else:
        # Laufendes Maximum der Equity-Kurve ohne Begrenzung
        running_max_equity = np.maximum.accumulate(equity_array)

        # Sicherstellen, dass wir nicht durch 0 teilen
        running_max_equity_safe = np.where(running_max_equity == 0, 1e-9, running_max_equity)

        # Berechnung der prozentualen Drawdowns
        drawdowns = (running_max_equity_safe - equity_array) / running_max_equity_safe

        # Maximaler Drawdown
        max_drawdown_pct_equity = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    # -------------------- (3) Calmar Ratio (Rendite / max_drawdown_pct_equity) ----------------------
    if peak_capital_usage > 0.0 and max_drawdown_pct_equity > 0.0:
        calmar_ratio = rendite / max_drawdown_pct_equity
    elif peak_capital_usage > 0.0 and max_drawdown_pct_equity == 0.0:
        # Trades wurden durchgeführt und kein Drawdown
        calmar_ratio = 1e6  # Sehr hoher, aber endlicher Wert
    else:
        # Keine Trades durchgeführt
        calmar_ratio = 0.0

    results = {
        "total_profit": round(total_profit, 0),
        "average_profit_per_trade": round(average_profit, 0),
        "rendite": round(rendite, 4),
        "total_trades": total_trades,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "peak_capital_usage": round(peak_capital_usage, 0),
        "capital_usage": round(peak_capital_usage, 0),  # Einheitlich auf peak_capital_usage gesetzt
        "leverage": leverage,

        # (1) Absoluter max Drawdown (bereits vorhanden)
        "max_drawdown_abs": round(max_drawdown, 2),
        # (2) Der klassische prozentuale Drawdown über die Equity-Kurve
        "max_drawdown_pct_equity": max_drawdown_pct_equity,
        # (3) Calmar Ratio
        "calmar_ratio": calmar_ratio
    }

    return results, df


import numpy as np

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def backtest_strategy_multiple_positions(
    df,
    shares_per_order,
    initial_stop_loss,
    profit_target,
    use_trailing_stop,
    trailing_stop_distance,
    use_max_holding_period,
    max_holding_period,
    use_nth_signal,
    nth_signal,
    use_chain_signals,
    max_signals_per_chain,
    leverage=1.0
):
    """
    Backtesting-Funktion mit erweiterter Ketten- und n-th-Signal-Logik.

    KORREKTUR:
    ----------
    - Die n-th-Signal-Logik wird nur dann aufgerufen, wenn das
      Bar-Signal (Original_Signal) == "Up" ist.

    1) Ketten-Logik (Chain Signals):
       - Nur wenn direkt nacheinander mehrere Up-Signale auftreten, liegt eine Kette vor.
         (Kein Kurs-Anstiegs-Check für das Definieren der Kette selbst.)

    2) n-th-Signal-Logik:
       - Prüft, ob die letzten n Bars strictly aufwärts (Close[i] > Close[i-1]) waren.
       - Nur gültig, wenn das aktuelle Signal == "Up".
       - nth_signal=1 => 1 Aufwärtsbewegung (der letzte Bar stieg).
       - nth_signal=2 => die letzten 2 Bar-Schritte stiegen, usw.

    Andere Parameter (Stop-Loss, Trailing Stop, max. Haltezeit usw.) bleiben wie gehabt.
    """

    # ------------------- Hilfsfunktion -------------------
    def is_n_bars_rising(dataframe, bar_index, n):
        """
        Prüft, ob die letzten n 'Schritte' des Kurses strikt aufwärts gingen,
        d. h. Close[i] > Close[i-1], Close[i-1] > Close[i-2], ...

        Beispiel: n=2 => die letzten 2 Bars stiegen hintereinander.
        """
        if bar_index < n:
            # Zu wenige Bars, um n Anstiege zu prüfen
            return False

        for step in range(n):
            # step=0 => vergleicht i mit i-1
            # step=1 => vergleicht i-1 mit i-2
            # ...
            if dataframe["Close_orig"].iloc[bar_index - step] <= dataframe["Close_orig"].iloc[bar_index - step - 1]:
                return False
        return True

    # 1) DataFrame kopieren und Original-Signale retten
    df = df.copy()
    df["Original_Signal"] = df["Predicted_Trend"]
    df["Predicted_Trend"] = "Stable"  # Wird später je nach Trade-Events überschrieben

    # Listen für offene / geschlossene Trades
    open_trades = []
    closed_trades = []

    # Eindeutiger Zähler für Trade-IDs
    trade_id_counter = 0

    # Variablen für die Ketten-Logik
    consecutive_up_count = 0
    in_chain = False
    opened_signals_in_current_chain = 0

    # Zusätzliche Spalten anlegen
    df["Open_Positions_Count"] = 0
    df["Closed_Trades_Count"] = 0
    df["Opened_Trade_IDs"] = [[] for _ in range(len(df))]
    df["Closed_Trade_IDs"] = [[] for _ in range(len(df))]
    df["Closed_Trade_PnLs"] = [[] for _ in range(len(df))]
    df["PnL_Closed"] = 0.0
    df["Equity"] = 0.0  # Start ohne initial_capital

    # Kapitalnutzung
    df["Capital_Usage"] = 0.0
    peak_capital_usage = 0.0

    previous_price = None

    # -------------------------------------------------------
    # 2) Schleife über alle Bars
    # -------------------------------------------------------
    for i in range(len(df)):
        current_price = df["Close_orig"].iloc[i]
        signal = df["Original_Signal"].iloc[i]

        opened_this_bar = []
        closed_this_bar = []

        # ------------------- A) Chain-Logik -------------------
        if use_chain_signals:
            # Kette = mehrere aufeinanderfolgende "Up"-Signale
            if signal == "Up":
                if not in_chain:
                    # Starte neue Kette
                    in_chain = True
                    opened_signals_in_current_chain = 0
                consecutive_up_count += 1
            else:
                # Kette endet, wenn kein Up-Signal
                in_chain = False
                consecutive_up_count = 0
        else:
            # Keine Ketten-Logik -> ggf. alte Logik oder Reset
            if i > 0 and previous_price is not None and signal == "Up" and current_price > previous_price:
                consecutive_up_count += 1
            else:
                consecutive_up_count = 0

        # -------------- B) Kauf-Logik (Trades eröffnen) --------------
        if use_chain_signals and in_chain:
            # Ketten-basierter Handel -> nur solange in_chain==True und signal=="Up"
            if signal == "Up":
                if use_nth_signal:
                    # Nur prüfen, wenn n-th-Signal aktiv ist + Signal == "Up"
                    if is_n_bars_rising(df, i, nth_signal):
                        if opened_signals_in_current_chain < max_signals_per_chain:
                            trade_id_counter += 1
                            open_trades.append({
                                "trade_id": trade_id_counter,
                                "entry_index": i,
                                "entry_price": current_price,
                                "shares": shares_per_order,
                                "stop_loss_price": current_price - initial_stop_loss,
                                "profit_target_price": current_price + profit_target,
                                "max_price_during_trade": current_price
                            })
                            opened_this_bar.append(trade_id_counter)
                            opened_signals_in_current_chain += 1

                        # Reset, damit nicht sofort im nächsten Bar wieder ein Kauf erfolgt:
                        consecutive_up_count = 0
                else:
                    # Wenn use_nth_signal=False => Bei JEDEM "Up" in der Kette
                    if opened_signals_in_current_chain < max_signals_per_chain:
                        trade_id_counter += 1
                        open_trades.append({
                            "trade_id": trade_id_counter,
                            "entry_index": i,
                            "entry_price": current_price,
                            "shares": shares_per_order,
                            "stop_loss_price": current_price - initial_stop_loss,
                            "profit_target_price": current_price + profit_target,
                            "max_price_during_trade": current_price
                        })
                        opened_this_bar.append(trade_id_counter)
                        opened_signals_in_current_chain += 1

        elif not use_chain_signals:
            # Keine Ketten-Logik (Standard-Fall)
            if signal == "Up":
                if use_nth_signal:
                    # Nur wenn Bar-Signal "Up" ist, prüfen wir die n-th-Signal-Logik
                    if is_n_bars_rising(df, i, nth_signal):
                        trade_id_counter += 1
                        open_trades.append({
                            "trade_id": trade_id_counter,
                            "entry_index": i,
                            "entry_price": current_price,
                            "shares": shares_per_order,
                            "stop_loss_price": current_price - initial_stop_loss,
                            "profit_target_price": current_price + profit_target,
                            "max_price_during_trade": current_price
                        })
                        opened_this_bar.append(trade_id_counter)
                        consecutive_up_count = 0
                else:
                    # Standard: Bei jedem "Up" (ohne n-th-Signal-Logik)
                    trade_id_counter += 1
                    open_trades.append({
                        "trade_id": trade_id_counter,
                        "entry_index": i,
                        "entry_price": current_price,
                        "shares": shares_per_order,
                        "stop_loss_price": current_price - initial_stop_loss,
                        "profit_target_price": current_price + profit_target,
                        "max_price_during_trade": current_price
                    })
                    opened_this_bar.append(trade_id_counter)

        # -------------- C) Offene Positionen prüfen -> verkaufen? --------------
        trades_to_close = []
        for trade in open_trades:
            # Trailing Stop
            if use_trailing_stop:
                if current_price > trade["max_price_during_trade"]:
                    trade["max_price_during_trade"] = current_price

                if trade["max_price_during_trade"] > trade["entry_price"]:
                    trailing_stop_price = trade["max_price_during_trade"] - trailing_stop_distance
                else:
                    trailing_stop_price = trade["stop_loss_price"]
                effective_stop_loss = max(trade["stop_loss_price"], trailing_stop_price)
            else:
                effective_stop_loss = trade["stop_loss_price"]

            # Stop-Loss oder Take-Profit
            if current_price <= effective_stop_loss or current_price >= trade["profit_target_price"]:
                trades_to_close.append(trade)
                continue

            # Max. Haltezeit
            if use_max_holding_period:
                holding_period = i - trade["entry_index"]
                if holding_period >= max_holding_period:
                    trades_to_close.append(trade)

        # Positionen schließen
        for trade in trades_to_close:
            exit_price = current_price
            trade_pnl = (exit_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "trade_id": trade["trade_id"],
                "entry_index": trade["entry_index"],
                "exit_index": i,
                "entry_price": trade["entry_price"],
                "exit_price": exit_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
            closed_this_bar.append((trade["trade_id"], trade_pnl))
            open_trades.remove(trade)

        # -------------- D) Equity, Realized PnL --------------
        realized_pnl = sum(t["pnl"] for t in closed_trades)
        unrealized_pnl = sum((current_price - t["entry_price"]) * t["shares"] for t in open_trades)
        equity = realized_pnl + unrealized_pnl

        df.at[df.index[i], "Open_Positions_Count"] = len(open_trades)
        df.at[df.index[i], "Closed_Trades_Count"] = len(closed_this_bar)
        df.at[df.index[i], "PnL_Closed"] = realized_pnl
        df.at[df.index[i], "Equity"] = equity
        df.at[df.index[i], "Opened_Trade_IDs"] = opened_this_bar
        df.at[df.index[i], "Closed_Trade_IDs"] = [x[0] for x in closed_this_bar]
        df.at[df.index[i], "Closed_Trade_PnLs"] = [x[1] for x in closed_this_bar]

        # -------------- E) Kapitalbedarf --------------
        capital_usage_now = 0.0
        for t in open_trades:
            capital_usage_now += (t["entry_price"] * t["shares"]) / leverage

        df.at[df.index[i], "Capital_Usage"] = capital_usage_now
        if capital_usage_now > peak_capital_usage:
            peak_capital_usage = capital_usage_now

        previous_price = current_price

        # -------------- F) Nach dem Kauf/Verkauf -> Predicted_Trend setzen --------------
        if len(opened_this_bar) > 0:
            df.at[df.index[i], "Predicted_Trend"] = "Up"
        if len(closed_this_bar) > 0:
            if df.at[df.index[i], "Predicted_Trend"] == "Up":
                df.at[df.index[i], "Predicted_Trend"] = "Up|Sold"
            else:
                df.at[df.index[i], "Predicted_Trend"] = "Sold"

    # -------------------------------------------------------
    # 3) Am Ende alle offenen Positionen schließen
    # -------------------------------------------------------
    if len(open_trades) > 0:
        final_price = df["Close_orig"].iloc[-1]
        closed_this_bar = []
        for trade in open_trades:
            trade_pnl = (final_price - trade["entry_price"]) * trade["shares"]
            closed_trades.append({
                "trade_id": trade["trade_id"],
                "entry_index": trade["entry_index"],
                "exit_index": len(df) - 1,
                "entry_price": trade["entry_price"],
                "exit_price": final_price,
                "shares": trade["shares"],
                "pnl": trade_pnl
            })
            closed_this_bar.append((trade["trade_id"], trade_pnl))

        open_trades.clear()

        df.at[df.index[-1], "Closed_Trade_IDs"] = [x[0] for x in closed_this_bar]
        df.at[df.index[-1], "Closed_Trade_PnLs"] = [x[1] for x in closed_this_bar]
        df.at[df.index[-1], "Closed_Trades_Count"] = len(closed_this_bar)

        if df.at[df.index[-1], "Predicted_Trend"] == "Up":
            df.at[df.index[-1], "Predicted_Trend"] = "Up|Sold"
        elif df.at[df.index[-1], "Predicted_Trend"] == "Stable":
            df.at[df.index[-1], "Predicted_Trend"] = "Sold"
        elif df.at[df.index[-1], "Predicted_Trend"] == "Up|Sold":
            pass
        else:
            df.at[df.index[-1], "Predicted_Trend"] = "Sold"

        realized_pnl = sum(t["pnl"] for t in closed_trades)
        df.at[df.index[-1], "PnL_Closed"] = realized_pnl
        df.at[df.index[-1], "Equity"] = realized_pnl  # keine offenen Trades mehr

    # -------------------------------------------------------
    # 4) Finale Kennzahlen berechnen
    # -------------------------------------------------------
    all_pnls = [t["pnl"] for t in closed_trades]
    total_profit = np.sum(all_pnls)
    total_trades = len(all_pnls)
    wins = np.sum([1 for pnl in all_pnls if pnl > 0])
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    average_profit = np.mean(all_pnls) if total_trades > 0 else 0.0

    # Berechnung der Rendite
    rendite = total_profit / peak_capital_usage if peak_capital_usage > 0.0 else 0.0

    # Absoluter Max Drawdown
    equity_array = df["Equity"].values
    if len(equity_array) == 0:
        max_drawdown = 0.0
    else:
        running_max_equity = np.maximum.accumulate(equity_array)
        drawdown_abs = running_max_equity - equity_array
        max_drawdown = np.max(drawdown_abs)

    # Prozentualer Drawdown über die Equity-Kurve
    if len(equity_array) == 0:
        max_drawdown_pct_equity = 0.0
    else:
        running_max_equity_safe = np.where(running_max_equity == 0, 1e-9, running_max_equity)
        drawdowns = (running_max_equity_safe - equity_array) / running_max_equity_safe
        max_drawdown_pct_equity = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    # Calmar Ratio
    if peak_capital_usage > 0.0 and max_drawdown_pct_equity > 0.0:
        calmar_ratio = rendite / max_drawdown_pct_equity
    elif peak_capital_usage > 0.0 and max_drawdown_pct_equity == 0.0:
        calmar_ratio = 1e6  # Sehr hoch, wenn kein Drawdown entstand
    else:
        calmar_ratio = 0.0

    results = {
        "total_profit": round(total_profit, 0),
        "average_profit_per_trade": round(average_profit, 0),
        "rendite": round(rendite, 4),
        "total_trades": total_trades,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "peak_capital_usage": round(peak_capital_usage, 0),
        "capital_usage": round(peak_capital_usage, 0),
        "leverage": leverage,
        "max_drawdown_abs": round(max_drawdown, 2),
        "max_drawdown_pct_equity": max_drawdown_pct_equity,
        "calmar_ratio": calmar_ratio
    }

    return results, df





def auswertung_mit_mehreren_trials1(study, df_data_test, direction, top_n_trials=1, ausgabe_datei="backtests.xlsx"):
    """
    Führt Backtests für entweder den besten Trial, die x besten Trials oder
    alle Trials aus und speichert die Ergebnisse in einer Excel-Datei.

    Parameter:
    -----------
    study : optuna.study.Study
        Das Study-Objekt von Optuna.
    df_data_test : pd.DataFrame
        Deine Marktdaten (DataFrame).
    direction : str oder list
        Wenn list, dann Multi-Objective; sonst Single-Objective.
    top_n_trials : int
        - 1 -> Nur bester Trial (Default).
        - x (x>1) -> x beste Trials
        - -1 -> Alle Trials
    ausgabe_datei : str
        Dateiname der Excel-Ausgabe.
    """

    # -------------------------------------------------------------
    # Hilfsfunktionen zum Sortieren
    # -------------------------------------------------------------
    def sort_single_objective_trials(study_obj):
        """Sortiere alle abgeschlossenen Trials bei Single-Objective."""
        completed_trials = [t for t in study_obj.trials if t.state.is_finished()]

        # Prüfen, ob maximiert oder minimiert wird
        if study_obj.direction.name == "MAXIMIZE":
            completed_trials.sort(key=lambda t: t.value, reverse=True)
        else:
            completed_trials.sort(key=lambda t: t.value, reverse=False)

        return completed_trials

    def sort_multi_objective_trials(study_obj):
        """
        Sortiere alle abgeschlossenen Trials bei Multi-Objective.
        Beispiel: Sortierung nur nach t.values[0] (Calmar Ratio) absteigend.
        """
        completed_trials = [t for t in study_obj.trials if t.state.is_finished()]
        completed_trials.sort(key=lambda t: t.values[0], reverse=True)

        return completed_trials

    # -------------------------------------------------------------
    # Sortierte Trials abhängig vom Objective-Typ
    # -------------------------------------------------------------
    if isinstance(direction, list):
        # Multi-Objective
        sorted_trials = sort_multi_objective_trials(study)
    else:
        # Single-Objective
        sorted_trials = sort_single_objective_trials(study)

    # -------------------------------------------------------------
    # Auswahl der Trials abhängig von top_n_trials
    # -------------------------------------------------------------
    if top_n_trials == 1:
        best_trials = [sorted_trials[0]]
    elif top_n_trials == -1:
        best_trials = sorted_trials
    else:
        best_trials = sorted_trials[:top_n_trials]

    # -------------------------------------------------------------
    # Liste zum Sammeln der Ergebnisse
    # -------------------------------------------------------------
    results_list = []

    # -------------------------------------------------------------
    # Schleife über die ausgewählten Trials
    # -------------------------------------------------------------
    for idx, trial in tqdm(enumerate(best_trials, start=1)):

        # -------------------------------
        # Multi- oder Single-Objective?
        # -------------------------------
        if isinstance(direction, list):
            # Multi-Objective
            print(f"\nTrial #{trial.number} - Values: {trial.values}")
            selected_params = trial.params
            # Falls du z.B. Calmar Ratio explizit ausgeben willst:
            print(f"Calmar Ratio (values[0]) = {trial.values[0]}")
        else:
            # Single-Objective
            print(f"\nTrial #{trial.number} - Value: {trial.value}")
            selected_params = trial.params

        # -------------------------------
        # Eingangs-Parameter aus dem Trial
        # -------------------------------
        shares_per_order = selected_params['shares_per_order']
        profit_target = selected_params['profit_target']
        initial_stop_loss = selected_params['initial_stop_loss']

        use_trailing_stop = selected_params['use_trailing_stop']
        trailing_stop_distance = (selected_params['trailing_stop_distance']
                                  if use_trailing_stop else 0)

        use_max_holding_period = selected_params['use_max_holding_period']
        max_holding_period = (selected_params['max_holding_period']
                              if use_max_holding_period else 0)

        use_nth_signal = selected_params['use_nth_signal']
        nth_signal = (selected_params['nth_signal']
                      if use_nth_signal else 0)

        # -------------------------------
        # Backtest ausführen
        # -------------------------------
        # results = dict mit Kennzahlen
        # df_with_backtest = DataFrame mit Detaildaten
        results, df_with_backtest = backtest_strategy_multiple_positions(
            df_data_test,
            shares_per_order=shares_per_order,
            profit_target=profit_target,
            initial_stop_loss=initial_stop_loss,
            use_trailing_stop=use_trailing_stop,
            trailing_stop_distance=trailing_stop_distance,
            use_nth_signal=use_nth_signal,
            nth_signal=nth_signal,
            use_max_holding_period=use_max_holding_period,
            max_holding_period=max_holding_period,
            leverage=1,
        )

        # Beispielhafte Ausgabe
        print(f'Zeitraum Anfang:\n{df_with_backtest["Datetime"].head(1).values[0]}')
        print(f'Zeitraum Ende:\n{df_with_backtest["Datetime"].tail(1).values[0]}')
        for k, v in results.items():
            print(f"{k}: {v}")

        # -------------------------------
        # Zusammenführen: Input-Parameter + Ergebnis-Kennzahlen
        # -------------------------------
        row_dict = {
            "trial_number": trial.number,
            "shares_per_order": shares_per_order,
            "profit_target": profit_target,
            "initial_stop_loss": initial_stop_loss,
            "use_trailing_stop": use_trailing_stop,
            "trailing_stop_distance": trailing_stop_distance,
            "use_max_holding_period": use_max_holding_period,
            "max_holding_period": max_holding_period,
            "use_nth_signal": use_nth_signal,
            "nth_signal": nth_signal,
        }

        # Nun alle Werte aus results anhängen
        for k, v in results.items():
            row_dict[k] = v

        # row_dict in die results_list übernehmen
        results_list.append(row_dict)

    # -------------------------------------------------------------
    # DataFrame mit allen Ergebnissen erstellen
    # -------------------------------------------------------------
    df_results = pd.DataFrame(results_list)

    # -------------------------------------------------------------
    # Export nach Excel
    # -------------------------------------------------------------
    df_results.to_excel(ausgabe_datei, index=False)
    print(f"\nErgebnisse wurden in '{ausgabe_datei}' gespeichert.\n")


def process_single_trial(args):
    """
    Führt den Backtest für einen Trial durch und gibt das Ergebnis
    als Dictionary zurück, das sowohl Input-Parameter als auch
    Backtest-Kennzahlen enthält.

    Parameter:
    -----------
    trial : optuna.trial.FrozenTrial
        Der Trial, der verarbeitet werden soll.
    direction : str oder list
        Die Richtung des Optimierungsziels.
    df_data_test : pd.DataFrame
        Die Marktdaten für den Backtest.
    """
    trial, direction, df_data_test = args

    # -------------------------------
    # Multi- oder Single-Objective?
    # -------------------------------
    if isinstance(direction, list):
        # Multi-Objective
        print(f"\nTrial #{trial.number} - Values: {trial.values}")
        selected_params = trial.params
        # Beispielhafte Ausgabe (Calmar Ratio etc.)
        print(f"Calmar Ratio (values[0]) = {trial.values[0]}")
    else:
        # Single-Objective
        print(f"\nTrial #{trial.number} - Value: {trial.value}")
        selected_params = trial.params

    # -------------------------------
    # Eingangs-Parameter aus dem Trial
    # -------------------------------
    shares_per_order = selected_params.get('shares_per_order', 1)
    profit_target = selected_params.get('profit_target', 0.0)
    initial_stop_loss = selected_params.get('initial_stop_loss', 0.0)

    use_trailing_stop = selected_params.get('use_trailing_stop', False)
    trailing_stop_distance = (
        selected_params.get('trailing_stop_distance', 0.0)
        if use_trailing_stop else 0.0
    )

    use_max_holding_period = selected_params.get('use_max_holding_period', False)
    max_holding_period = (
        selected_params.get('max_holding_period', 0)
        if use_max_holding_period else 0
    )

    use_nth_signal = selected_params.get('use_nth_signal', False)
    nth_signal = (
        selected_params.get('nth_signal', 0)
        if use_nth_signal else 0
    )

    # -------------------------------
    # Backtest ausführen
    # -------------------------------
    # Hier rufst du deine Backtest-Funktion auf
    results, df_with_backtest = backtest_strategy_multiple_positions(
        df_data_test,
        shares_per_order=shares_per_order,
        profit_target=profit_target,
        initial_stop_loss=initial_stop_loss,
        use_trailing_stop=use_trailing_stop,
        trailing_stop_distance=trailing_stop_distance,
        use_nth_signal=use_nth_signal,
        nth_signal=nth_signal,
        use_max_holding_period=use_max_holding_period,
        max_holding_period=max_holding_period,
        leverage=1,
    )

    # Beispielhafte Ausgabe
    print(f'Zeitraum Anfang:\n{df_with_backtest["Datetime"].iloc[0]}')
    print(f'Zeitraum Ende:\n{df_with_backtest["Datetime"].iloc[-1]}')
    for k, v in results.items():
        print(f"{k}: {v}")

    # -------------------------------
    # Zusammenführen: Input-Parameter + Ergebnis-Kennzahlen
    # -------------------------------
    row_dict = {
        "trial_number": trial.number,
        "shares_per_order": shares_per_order,
        "profit_target": profit_target,
        "initial_stop_loss": initial_stop_loss,
        "use_trailing_stop": use_trailing_stop,
        "trailing_stop_distance": trailing_stop_distance,
        "use_max_holding_period": use_max_holding_period,
        "max_holding_period": max_holding_period,
        "use_nth_signal": use_nth_signal,
        "nth_signal": nth_signal,
    }

    # Alle Werte aus results anhängen (z.B. total_profit, calmar_ratio, etc.)
    for k, v in results.items():
        row_dict[k] = v

    return row_dict

def auswertung_mit_mehreren_trials(
    study,
    df_data_test,
    direction,
    top_n_trials=1,
    ausgabe_datei="ergebnisse.xlsx",
    n_cpus=1
):
    """
    Führt Backtests für entweder den besten Trial, die x besten Trials oder
    alle Trials aus und speichert die Ergebnisse in einer Excel-Datei.
    Optional werden mehrere Prozesse genutzt, um die Auswertung zu beschleunigen.

    Parameter:
    -----------
    study : optuna.study.Study
        Das Study-Objekt von Optuna.
    df_data_test : pd.DataFrame
        Deine Marktdaten (DataFrame).
    direction : str oder list
        Wenn list, dann Multi-Objective; sonst Single-Objective.
    top_n_trials : int
        - 1 -> Nur bester Trial (Default).
        - x (x>1) -> x beste Trials
        - -1 -> Alle Trials
    ausgabe_datei : str
        Dateiname der Excel-Ausgabe.
    n_cpus : int
        Anzahl der Prozesse (CPUs). Ist n_cpus=1, läuft alles sequentiell.
    """

    # -------------------------------------------------------------
    # Hilfsfunktionen zum Sortieren
    # -------------------------------------------------------------
    def sort_single_objective_trials(study_obj):
        """Sortiere alle abgeschlossenen Trials bei Single-Objective."""
        completed_trials = [t for t in study_obj.trials if t.state.is_finished()]

        # Prüfen, ob maximiert oder minimiert wird
        if study_obj.direction.name == "MAXIMIZE":
            completed_trials.sort(key=lambda t: t.value, reverse=True)
        else:
            completed_trials.sort(key=lambda t: t.value, reverse=False)

        return completed_trials

    def sort_multi_objective_trials(study_obj):
        """
        Sortiere alle abgeschlossenen Trials bei Multi-Objective.
        Beispiel: Sortierung nur nach t.values[0] (Calmar Ratio) absteigend.
        """
        completed_trials = [t for t in study_obj.trials if t.state.is_finished()]
        completed_trials.sort(key=lambda t: t.values[0], reverse=True)

        return completed_trials

    # -------------------------------------------------------------
    # Sortierte Trials abhängig vom Objective-Typ
    # -------------------------------------------------------------
    if isinstance(direction, list):
        # Multi-Objective
        sorted_trials = sort_multi_objective_trials(study)
    else:
        # Single-Objective
        sorted_trials = sort_single_objective_trials(study)

    # -------------------------------------------------------------
    # Auswahl der Trials abhängig von top_n_trials
    # -------------------------------------------------------------
    if top_n_trials == 1:
        best_trials = [sorted_trials[0]]
    elif top_n_trials == -1:
        best_trials = sorted_trials
    else:
        best_trials = sorted_trials[:top_n_trials]

    # -------------------------------------------------------------
    # Vorbereitung der Argumente für multiprocessing
    # -------------------------------------------------------------
    list_of_args = [(trial, direction, df_data_test) for trial in best_trials]

    # -------------------------------------------------------------
    # Multiprocessing (optional) oder sequentiell
    # -------------------------------------------------------------
    if n_cpus > 1:
        print(f"Starte parallele Verarbeitung mit {n_cpus} Prozessen...")
        with Pool(processes=n_cpus) as pool:
            results_list = pool.map(process_single_trial, list_of_args)
    else:
        print("Starte sequentielle Verarbeitung...")
        results_list = [process_single_trial(args) for args in list_of_args]

    # -------------------------------------------------------------
    # DataFrame mit allen Ergebnissen erstellen
    # -------------------------------------------------------------
    df_results = pd.DataFrame(results_list)

    # -------------------------------------------------------------
    # Export nach Excel
    # -------------------------------------------------------------
    df_results.to_excel(ausgabe_datei, index=False)
    print(f"\nErgebnisse wurden in '{ausgabe_datei}' gespeichert.\n")


def objective1(trial, df):
    """
    In dieser Funktion wird festgelegt, wie die Parameter für jeden Optuna-Trial
    gewählt werden, und welche Zielmetrik optimiert wird.

    df,
    initial_capital=10000.0,
    shares_per_order=100,
    initial_stop_loss=1.0,  # Absoluter Abstand unter Einstiegspreis (für Long)
    profit_target=2.0,  # Absoluter Gewinnabstand über Einstiegspreis (für Long)
    use_trailing_stop=False,
    trailing_stop_distance=1.0,  # Absoluter Abstand für den Trailing Stop
    transaction_costs=0.0,
    slippage=0.0,
    allow_short=False
    """

    shares_per_order = trial.suggest_int('shares_per_order', 1, 1, step=1)
    initial_stop_loss = trial.suggest_int('initial_stop_loss', 1.0, 500.0, step=1.0)
    profit_target = trial.suggest_int('profit_target', 1.0, 500.0, step=1.0)

    use_trailing_stop = trial.suggest_categorical('use_trailing_stop', [True, False])
    if use_trailing_stop:
        trailing_stop_distance = trial.suggest_int('trailing_stop_distance', 1.0, 100.0, step=1.0)
    else:
        trailing_stop_distance = 0.0


    use_max_holding_period = trial.suggest_categorical('use_max_holding_period', [True, False])
    # use_max_holding_period = trial.suggest_categorical('use_max_holding_period', [False, False])

    if use_max_holding_period:
        max_holding_period = trial.suggest_int('max_holding_period', 1.0, 1000.0, step=1.0)
    else:
        max_holding_period = 0.0


    use_nth_signal = trial.suggest_categorical('use_nth_signal', [True, False])
    # use_nth_signal = trial.suggest_categorical('use_nth_signal', [True, True])  # Steht außer Frage, wenn es zu viele Signale in Abwärtstrends gibt.
    # use_nth_signal = trial.suggest_categorical('use_nth_signal', [False, False])

    if use_nth_signal:
        nth_signal = trial.suggest_int('nth_signal', 1.0, 10.0, step=1.0)
        # nth_signal = trial.suggest_int('nth_signal', 2.0, 2.0, step=1.0)
    else:
        nth_signal = 0.0


    use_chain_signals = trial.suggest_categorical('use_chain_signals', [True, False])
    if use_chain_signals:
        max_signals_per_chain = trial.suggest_int('max_signals_per_chain', 1.0, 10.0, step=1.0)
        # nth_signal = trial.suggest_int('nth_signal', 2.0, 2.0, step=1.0)
    else:
        max_signals_per_chain = 0.0


    try:
        results, df_with_backtest = backtest_strategy_multiple_positions(
            df=df,
            shares_per_order=shares_per_order,
            initial_stop_loss=initial_stop_loss,
            profit_target=profit_target,
            use_trailing_stop=use_trailing_stop,
            trailing_stop_distance=trailing_stop_distance,
            # use_max_holding_period=False,
            # max_holding_period=10,
            # use_nth_signal=False,
            # nth_signal=2
            use_max_holding_period=use_max_holding_period,
            max_holding_period=max_holding_period,
            use_nth_signal=use_nth_signal,
            nth_signal=nth_signal,
            use_chain_signals=use_chain_signals,
            max_signals_per_chain=max_signals_per_chain
        )

        total_profit = results["total_profit"]
        average_profit_per_trade = results["average_profit_per_trade"]
        rendite = results["rendite"]
        total_trades = results["total_trades"]
        win_rate = results["win_rate"]
        calmar_ratio = results["calmar_ratio"]


    except:
        total_profit = 0
        average_profit_per_trade = 0
        rendite = 0
        total_trades = 0
        win_rate = 0
        calmar_ratio = 0


    # min_trades = 150
    # penalty_factor = 0.5  # je nach Gusto
    # # Berechne Abweichung vom Min-Trade-Ziel
    # shortfall = max(0, min_trades - total_trades)
    # # Penalty: je mehr wir unter min_trades liegen, desto stärker
    # penalty = penalty_factor * shortfall
    # # Kombiniere nun
    # score = calmar_ratio - penalty

    # min_trades = 401
    # max_trades = 500
    # penalty_factor = 1000
    #
    # if total_trades < min_trades:
    #     # Abweichung nach unten
    #     shortfall = min_trades - total_trades
    #     penalty = penalty_factor * shortfall
    # elif total_trades > max_trades:
    #     # Abweichung nach oben
    #     shortfall = total_trades - max_trades
    #     penalty = penalty_factor * shortfall
    # else:
    #     # Liegt innerhalb der Range -> kein Penalty
    #     penalty = 0
    #
    # score = calmar_ratio - penalty


    return calmar_ratio
    # return calmar_ratio, total_trades
    # return score
    # return total_profit, average_profit_per_trade, rendite, total_trades, win_rate, calmar_ratio


def split_data_into_n_parts(df: pd.DataFrame, n: int):
    """
    Teilt den DataFrame df in n gleich große Zeitsegmente.
    Returnt eine Liste von DataFrames [df_part_1, df_part_2, ..., df_part_n].
    """
    # Beispielhafter Ansatz (einfache, gleiche Längenaufteilung):
    df_list = []
    chunk_size = len(df) // n
    for i in range(n):
        start_idx = i * chunk_size
        # Beim letzten Chunk bis zum Ende, damit keine Daten "übrig" bleiben:
        end_idx = (i + 1) * chunk_size if i < n - 1 else len(df)
        df_part = df.iloc[start_idx:end_idx].copy()
        df_list.append(df_part)
    return df_list


def objective_n_splits(trial, df_list):
    """
    Objective-Funktion, die einen Satz an Backtest-Parametern optimiert.
    Für jedes DataFrame in df_list wird ein Backtest durchgeführt.
    Die Zielmetrik wird gemittelt (z. B. 'calmar_ratio').
    """

    # 1. Parameter, die du mit Optuna optimieren möchtest
    shares_per_order = trial.suggest_int('shares_per_order', 1, 1, step=1)

    # initial_stop_loss = trial.suggest_int('initial_stop_loss', 1, 500, step=1)
    # initial_stop_loss = trial.suggest_int('initial_stop_loss', 1, 100, step=1)
    initial_stop_loss = trial.suggest_int('initial_stop_loss', 1, 50, step=1)


    profit_target = trial.suggest_int('profit_target', 1, 100, step=1)
    # profit_target = trial.suggest_int('profit_target', 1, 100, step=1)


    # use_trailing_stop = trial.suggest_categorical('use_trailing_stop', [True, False])
    use_trailing_stop = trial.suggest_categorical('use_trailing_stop', [True, True])
    # use_trailing_stop = trial.suggest_categorical('use_trailing_stop', [False, False])
    if use_trailing_stop:
        trailing_stop_distance = trial.suggest_int('trailing_stop_distance', 1, 12, step=1)
    else:
        trailing_stop_distance = 0


    # use_max_holding_period = trial.suggest_categorical('use_max_holding_period', [True, False])
    use_max_holding_period = trial.suggest_categorical('use_max_holding_period', [True, True])
    # use_max_holding_period = trial.suggest_categorical('use_max_holding_period', [False, False])
    if use_max_holding_period:
        # max_holding_period = trial.suggest_int('max_holding_period', 1, 100, step=1)
        max_holding_period = trial.suggest_int('max_holding_period', 12, 12, step=1)

    else:
        max_holding_period = 0

    ######### Prüft, wie viele Signale vor dem Up-Signal schon positiv sein müssen. Entspricht einer Trendbestätigung
    use_nth_signal = trial.suggest_categorical('use_nth_signal', [True, False])
    # use_nth_signal = trial.suggest_categorical('use_nth_signal', [True, True])
    # use_nth_signal = trial.suggest_categorical('use_nth_signal', [False, False])
    if use_nth_signal:
        nth_signal = trial.suggest_int('nth_signal', 1, 3, step=1)
        # nth_signal = trial.suggest_int('nth_signal', 1, 10, step=1)
    else:
        nth_signal = 0


    ######### False damit die Anzahl an Käufen nicht begrenzt ist
    # use_chain_signals = trial.suggest_categorical('use_chain_signals', [True, False])
    # use_chain_signals = trial.suggest_categorical('use_chain_signals', [True, True])
    use_chain_signals = trial.suggest_categorical('use_chain_signals', [False, False])

    if use_chain_signals:
        max_signals_per_chain = trial.suggest_int('max_signals_per_chain', 1, 20, step=1)
        # max_signals_per_chain = trial.suggest_int('max_signals_per_chain', 1, 4, step=1)
    else:
        max_signals_per_chain = 0



    # IMPORTANT
    # 2. Für jede Teilmenge einen Backtest durchführen und die Kennzahl aufsummieren
    calmar_sum = 0.0
    calmar_results = []

    for df_part in df_list:
        try:
            results, df_with_backtest = backtest_strategy_multiple_positions(
                df=df_part,
                shares_per_order=shares_per_order,
                initial_stop_loss=initial_stop_loss,
                profit_target=profit_target,
                use_trailing_stop=use_trailing_stop,
                trailing_stop_distance=trailing_stop_distance,
                use_max_holding_period=use_max_holding_period,
                max_holding_period=max_holding_period,
                use_nth_signal=use_nth_signal,
                nth_signal=nth_signal,
                use_chain_signals=use_chain_signals,
                max_signals_per_chain=max_signals_per_chain
            )
            # calmar_sum += results["calmar_ratio"]
            calmar_results.append(results["calmar_ratio"])
            # calmar_results.append(results["rendite"])



        except:
            # Falls es irgendwo crasht, z. B. keine Trades
            # calmar_sum += 0
            calmar_results.append(0)

    # 3. Gemittelte Metrik zurückgeben
    # if calmar_sum != 0:
    #     avg_calmar = calmar_sum / len(df_list)
    # else:
    #     avg_calmar = 0

    try:
        avg_calmar = median(calmar_results)
    except:
        avg_calmar = 0


    return avg_calmar




def predict_and_plot(
        config,
        df_data_test,
        df_data_val,
        database_name_optuna,

        plot_length_test=500,  # Anzahl der letzten Datenpunkte, die geplottet werden sollen
        plot_length_val=500,  # Anzahl der letzten Datenpunkte, die geplottet werden sollen

        additional_lines=None,
        secondary_y_scale=1.0,
        x_interval_min=60,
        y_interval_dollar=80,
        predicting_columns=None,  # Liste der Vorhersagespalten
        show_plot=False,
        from_stock=False,

        backtest=False,
        backtest_tuning=False,
        backtest_tuning_with_n_splits=False,
        n_splits=1,

        backtest_study_name="",
        backtest_tuning_trials=1,
        backtest_tuning_parallel_trials=1,
        best_backtest=False,
        use_val_in_final_backtest=False,
        save_plot_as_picture=None,
        model_folder=None,
):

    """
    Lädt ein trainiertes Modell, führt Vorhersagen auf neuen Daten durch und plottet die Ergebnisse.

    Parameter:
    - df_data_test (pd.DataFrame): Neue Daten für Vorhersagen (inkl. 'Trend').
    - database_name_optuna (str): Name der Datenbank/Studie für das Modell.
    - plot_length (int): Anzahl der letzten Datenpunkte, die geplottet werden sollen.
    - additional_lines (list, optional): Zusätzliche Linien für den Plot.
    - secondary_y_scale (float): Skalierungsfaktor für die sekundäre Y-Achse.
    - x_interval_min (int): Intervall für die X-Achse in Minuten.
    - y_interval_dollar (int): Intervall für die Y-Achse in Dollar.
    - predicting_columns (list of str, optional): Liste der Spalten, die für die Vorhersage verwendet werden.
    """


    # model_save_path = f"saved_models/nn_model_{database_name_optuna}"
    # model_name = f"nn_model_{database_name_optuna}"
    # scalers_save_path = f"{model_save_path}/scalers.pkl"  # Skalierer Pfad (falls verwendet)

    if model_folder != "":
        model_save_path = f"saved_models/{model_folder}"
        model_name = f"nn_model_{database_name_optuna}"
    else:
        model_save_path = f"saved_models/nn_model_{database_name_optuna}"
        model_name = f"nn_model_{database_name_optuna}"

    # 1. Modell laden
    print("Lade das trainierte Modell...")
    try:
        model = tf.keras.models.load_model(
            os.path.join(model_save_path, model_name + '.keras'),
            custom_objects={
                # 'F1Score': F1Score,
                'F1Score': tfa.metrics.F1Score(num_classes=2, average='macro'),
                # 'CustomF1Score': CustomF1Score,
                'FeatureWeightingLayer': FeatureWeightingLayer
            },
            safe_mode=False  # Erlaubt das Laden von Lambda-Schichten
        )
        print("Modell erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        traceback.print_exc()
        return


    def process_data(df_data_test, predicting_columns):
        # 3. Daten vorbereiten
        print("Bereite die Daten für die Vorhersage vor...")
        # Identifiziere die Zeitspalten (ohne 'Close_orig' und 'Trend')
        time_columns = ['Date', 'Time', 'Datetime', 'Close_orig']  # 'Trend' ist die Zielvariable
        present_time_columns = [col for col in time_columns if col in df_data_test.columns]
        if present_time_columns:
            # Extrahiere die Zeitspalten für das Plotten
            time_data = df_data_test[present_time_columns].copy()
        else:
            print("Keine Zeitspalten gefunden. Stelle sicher, dass die Spaltennamen korrekt sind.")
            time_data = pd.DataFrame()

        # Entferne die Zeitspalten für die Vorhersage, aber behalte 'Trend' für die Metrikberechnung
        if predicting_columns is None:
            # Verwende alle Spalten außer Zeitspalten und 'Trend'
            predicting_columns = [col for col in df_data_test.columns if col not in present_time_columns + ['Trend']]
        else:
            # Exkludiere Zeitspalten und 'Trend' aus 'predicting_columns'
            predicting_columns = [col for col in predicting_columns if col not in present_time_columns + ['Trend']]

        # print(f"Predicting columns: {predicting_columns}")

        # Sicherstellen, dass 'Trend' nicht in den Features ist
        if 'Trend' in predicting_columns:
            predicting_columns.remove('Trend')

        X_new = df_data_test[predicting_columns].copy()

        # 4. Vorbereitung der Eingaben für das Modell
        print("Bereite die Eingaben für das Modell vor...")
        input_dict = {}
        for col in predicting_columns:
            # Jede Eingabe muss eine 2D-Array mit Shape (num_samples, 1) sein
            input_dict[col] = X_new[col].values.reshape(-1, 1)

        # 5. Vorhersagen machen
        print("Mache Vorhersagen...")
        try:
            predictions = model.predict(input_dict)
            print("Vorhersagen erfolgreich durchgeführt.")
        except Exception as e:
            print("Fehler bei der Vorhersage:")
            print(traceback.format_exc())
            return

        # 6. Anpassen der Vorhersagen ohne One-Hot-Encoding
        print("Verarbeite die Vorhersagen...")
        if predictions.shape[1] == 1:
            # Sigmoid-Ausgabe (für binäre Klassifikation)
            predicted_labels = (predictions > 0.5).astype(int).flatten()
            label_mapping = {0: 'Stable', 1: 'Up'}
            predicted_labels = [label_mapping[label] for label in predicted_labels]
        elif predictions.shape[1] > 1:
            # Softmax-Ausgabe (für Mehrklassenklassifikation)
            predicted_classes = np.argmax(predictions, axis=1)
            # Annahme: Die Reihenfolge der Kategorien entspricht der Reihenfolge im OneHotEncoder
            # Beispiel: {0: 'Stable', 1: 'Up'}
            label_mapping = {0: 'Stable', 1: 'Up'}
            predicted_labels = [label_mapping.get(class_, 'Unknown') for class_ in predicted_classes]
        else:
            print("Fehler: Unerwartete Anzahl von Ausgängen im Modell.")
            return

        # 7. Vorhersagen dem DataFrame hinzufügen
        print("Füge die Vorhersagen dem DataFrame hinzu...")
        # Sicherstellen, dass die Anzahl der Vorhersagen mit der Anzahl der Daten übereinstimmt
        if len(predicted_labels) != len(df_data_test):
            print("Warnung: Die Anzahl der Vorhersagen stimmt nicht mit der Anzahl der Daten überein.")
            df_data_test = df_data_test.iloc[:len(predicted_labels)].copy()
        else:
            df_data_test = df_data_test.copy()

        # Füge die Zeitspalten wieder ein, falls vorhanden
        if not time_data.empty:
            df_data_test[present_time_columns] = time_data.iloc[:len(predicted_labels)].values

        # Füge die Vorhersagen hinzu
        df_data_test["Predicted_Trend"] = predicted_labels


        # Stellen Sie sicher, dass die 'Datetime'-Spalte vorhanden ist
        if 'Datetime' not in df_data_test.columns:
            print("Warnung: 'Datetime' Spalte fehlt. Stelle sicher, dass die Zeitspalten korrekt sind.")

        # Auswahl der relevanten Spalten für das Excel-Export und Plotting
        if 'Datetime' in df_data_test.columns:
            df_data_test = df_data_test[['Datetime', 'Close_orig', 'Trend', 'Predicted_Trend']]
        else:
            df_data_test = df_data_test[['Close_orig', 'Trend', 'Predicted_Trend']]

        # print(f'df_data_test:{df_data_test.head()}')


        # 8. Berechnung und Print der Genauigkeiten
        print("Berechne die Genauigkeiten...")
        try:
            # Umwandlung der Labels in numerische Werte für die Berechnung (falls notwendig)
            label_mapping = {'Stable': 0, 'Up': 1}
            df_data_test['Trend_Mapped'] = df_data_test['Trend'].map(label_mapping)
            df_data_test['Predicted_Trend_Mapped'] = df_data_test['Predicted_Trend'].map(label_mapping)

            # Überprüfung der gemappten Labels
            print("Verteilung der tatsächlichen Labels:")
            print(df_data_test['Trend_Mapped'].value_counts())
            print("Verteilung der vorhergesagten Labels:")
            print(df_data_test['Predicted_Trend_Mapped'].value_counts())

            # Entfernen von Einträgen, die nicht gemappt werden konnten
            df_valid = df_data_test.dropna(subset=['Trend_Mapped', 'Predicted_Trend_Mapped'])

            y_true = df_valid['Trend_Mapped'].astype(int).values
            y_pred = df_valid['Predicted_Trend_Mapped'].astype(int).values

            # Gesamtgenauigkeit
            overall_accuracy = accuracy_score(y_true, y_pred)

            # Berechnung der Confusion Matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0  # Fallback falls die Matrix nicht 2x2 ist

            # Präzision und Recall für "Up" (Klasse 1)
            precision_up = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_up = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # F1-Score für "Up"
            if precision_up + recall_up > 0:
                f1_up = 2 * precision_up * recall_up / (precision_up + recall_up)
            else:
                f1_up = 0.0

            # Optional: Ausgabe eines vollständigen Classification Reports
            report = classification_report(y_true, y_pred, target_names=['Stable', 'Up'])
            print("\nClassification Report:")
            print(report)

            # Ausgabe der berechneten Metriken
            print(f"Gesamtgenauigkeit: {overall_accuracy:.4f}")
            print(f"Genauigkeit der 'Up'-Vorhersagen (Präzision): {precision_up:.4f}")
            print(f"Genauigkeit der 'Up'-Vorhersagen (Recall): {recall_up:.4f}")
            print(f"F1-Score für 'Up': {f1_up:.4f}")
        except Exception as e:
            print(f"Fehler bei der Berechnung der Genauigkeiten: {e}")
            traceback.print_exc()

        # 9. Exportiere die Ergebnisse nach Excel
        # print("Exportiere die Ergebnisse nach Excel...")
        # try:
        #     df_data_test.to_excel("df_data_test.xlsx", index=False)
        #     print("Daten erfolgreich nach 'df_data_test.xlsx' exportiert.")
        # except Exception as e:
        #     # print(f"Fehler beim Exportieren der Daten nach Excel: {e}")
        #     # traceback.print_exc()
        #     pass


        # return df_data_test, f1_up
        return df_data_test, round(f1_up, 4)

    if not from_stock:
        df_data_test, f1_up_test = process_data(df_data_test, predicting_columns)

        if use_val_in_final_backtest:
            df_data_val, f1_up_val = process_data(df_data_val, predicting_columns)
        else:
            df_data_val = df_data_test
            f1_up_val = f1_up_test
    else:
        df_data_test, f1_up_test = process_data(df_data_test, predicting_columns)
        df_data_val = df_data_test



    if show_plot:
        # 10. Plotten der Ergebnisse
        print("Plotten der Testergebnisse...")
        try:
            plot_stock_prices(
                config="test_" + str(config) + f"_f1_{f1_up_test}",
                folder=r"Backtest",
                df=df_data_test.head(plot_length_test),
                test=True,
                trend="Predicted_Trend",
                secondary_y_scale=secondary_y_scale,
                x_interval_min=x_interval_min,
                y_interval_dollar=y_interval_dollar,
                additional_lines=additional_lines,
                save_plot_as_picture=save_plot_as_picture
            )
            print("Vorhersage und Plot abgeschlossen.")
        except Exception as e:
            print(f"Fehler beim Plotten der Ergebnisse: {e}")
            traceback.print_exc()


        if not from_stock and use_val_in_final_backtest:
            # 10. Plotten der Ergebnisse
            print("Plotten der Validierungsergebnisse...")
            try:
                plot_stock_prices(
                    config="val_" + str(config) + f"_f1_{f1_up_val}",
                    folder=r"Backtest\val",
                    df=df_data_val.head(plot_length_val),
                    test=True,
                    trend="Predicted_Trend",
                    secondary_y_scale=secondary_y_scale,
                    x_interval_min=x_interval_min,
                    y_interval_dollar=y_interval_dollar,
                    additional_lines=additional_lines,
                    save_plot_as_picture=save_plot_as_picture
                )
                print("Vorhersage und Plot abgeschlossen.")
            except Exception as e:
                print(f"Fehler beim Plotten der Ergebnisse: {e}")
                traceback.print_exc()


    if backtest:

        # direction = "minimize"
        direction = "maximize"
        # direction = ["maximize", "maximize"]
        # direction = ["maximize", "maximize", "maximize", "maximize", "maximize", "maximize"]

        study = load_study(backtest_study_name, direction=direction)


        if backtest_tuning:
            # 4. Erstelle eine Study und starte das Optimieren
            #    Richte die Study so ein, dass der Nettogewinn maximiert wird:



            if backtest_tuning_with_n_splits:
                # n_splits = 3
                df_splits = split_data_into_n_parts(df_data_test, n_splits)

                objective_with_data = functools.partial(
                    objective_n_splits,
                    df_list=df_splits
                )

            else:
                objective_with_data = functools.partial(
                    objective1,
                    df=df_data_test
                )

            study.optimize(objective_with_data, n_trials=backtest_tuning_trials, n_jobs=backtest_tuning_parallel_trials)

            # 5. Ausgabe der besten gefundenen Parameter
            print("Best Trial:")
            best_trial = study.best_trial
            print(f"  Value (net profit): {best_trial.value:.2f}")
            print("  Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")



        # if not backtest_tuning:

        if isinstance(direction, list):

            # Multi-Objective
            selected_params = max(study.best_trials, key=lambda t: t.values[0])  # Calmar Ratio
            # selected_params = max(study.best_trials, key=lambda t: t.values[1])  # Total Trades

            print("\nTrial mit dem höchsten Calmar Ratio:")
            print(f"  Trial #{selected_params.number}")
            print(f"  Calmar Ratio (values[0]) = {selected_params.values[0]}")
            print(f"  Andere Metrik (values[1]) = {selected_params.values[1]}")
            print(f"  Params = {selected_params.params}")
            selected_params = selected_params.params

        else:
            # Single-Objective
            selected_params = study.best_params
            print(f'Using the best trial with params: {selected_params}')


        shares_per_order = selected_params['shares_per_order']
        profit_target = selected_params['profit_target']
        initial_stop_loss = selected_params['initial_stop_loss']
        # initial_stop_loss = 10

        use_trailing_stop = selected_params['use_trailing_stop']
        if use_trailing_stop:
            trailing_stop_distance = selected_params['trailing_stop_distance']
        else:
            trailing_stop_distance = 0

        use_max_holding_period = selected_params['use_max_holding_period']
        if use_max_holding_period:
            max_holding_period = selected_params['max_holding_period']
        else:
            max_holding_period = 0

        use_nth_signal = selected_params['use_nth_signal']
        if use_nth_signal:
            nth_signal = selected_params['nth_signal']
        else:
            nth_signal = 0


        use_chain_signals = selected_params['use_chain_signals']
        if use_chain_signals:
            max_signals_per_chain = selected_params['max_signals_per_chain']
        else:
            max_signals_per_chain = 0

        # IMPORTANT

        # shares_per_order = 1
        # profit_target = 1000
        # initial_stop_loss = 50
        # use_trailing_stop = False
        # trailing_stop_distance = 5
        # use_nth_signal = True
        # nth_signal = 2
        # use_max_holding_period = False
        # max_holding_period = 233
        # use_chain_signals = True
        # max_signals_per_chain = 10

        # for profit_target in [1, 10, 25, 50, 100, 150, 200, 300, 400, 500]:
        #     for initial_stop_loss in [1, 10, 25, 50, 100, 150, 200, 300, 400, 500]:

        # results, df_with_backtest = backtest_strategy_multiple_positions_7(
        results, df_with_backtest = backtest_strategy_multiple_positions(
            df=df_data_test,
            shares_per_order=shares_per_order,
            profit_target=profit_target,
            initial_stop_loss=initial_stop_loss,
            use_trailing_stop=use_trailing_stop,
            trailing_stop_distance=trailing_stop_distance,
            use_nth_signal=use_nth_signal,
            nth_signal=nth_signal,
            use_max_holding_period=use_max_holding_period,
            max_holding_period=max_holding_period,
            use_chain_signals=use_chain_signals,
            max_signals_per_chain=max_signals_per_chain,
        )

        print(f'Zeitraum Anfang:\n{df_with_backtest["Datetime"].head(1).values[0]}')
        print(f'Zeitraum Ende:\n{df_with_backtest["Datetime"].tail(1).values[0]}')

        # print("Backtesting abgeschlossen. Ergebnisse:")
        for k, v in results.items():
            print(f"{k}: {v}")

        # Optional: Exportieren
        # try:
        #     df_with_backtest.to_excel("df_with_backtest.xlsx", index=False)
        #     print("Backtesting-Daten erfolgreich nach 'df_with_backtest.xlsx' exportiert.")
        # except Exception as e:
        #     print(f"Fehler beim Export der Backtest-Daten: {e}")

        # Plot der Equity-Kurve
        plt.figure(figsize=(12, 6))
        plt.plot(df_with_backtest["Equity"], label='Equity')
        plt.title('Equity-Kurve')
        plt.xlabel('Bar')
        plt.ylabel('Equity')
        plt.legend()
        plt.show()

        # 10. Plotten der Ergebnisse
        print("Plotten der Ergebnisse...")
        try:
            plot_stock_prices(
                config="test_trade" + str(config) + f"_f1_{f1_up_test}",
                folder=r"Backtest",
                df=df_with_backtest.tail(plot_length_test),
                test=True,
                trend="Predicted_Trend",
                secondary_y_scale=secondary_y_scale,
                x_interval_min=x_interval_min,
                y_interval_dollar=y_interval_dollar,
                additional_lines=additional_lines,
                save_plot_as_picture=False
            )
            print("Vorhersage und Plot abgeschlossen.")
        except Exception as e:
            print(f"Fehler beim Plotten der Ergebnisse: {e}")
            traceback.print_exc()

        if use_val_in_final_backtest and not from_stock:

            # results, df_with_backtest = backtest_strategy_multiple_positions_7(
            results, df_with_backtest = backtest_strategy_multiple_positions(
                df=df_data_val,
                shares_per_order=shares_per_order,
                profit_target=profit_target,
                initial_stop_loss=initial_stop_loss,
                use_trailing_stop=use_trailing_stop,
                trailing_stop_distance=trailing_stop_distance,
                use_nth_signal=use_nth_signal,
                nth_signal=nth_signal,
                use_max_holding_period=use_max_holding_period,
                max_holding_period=max_holding_period,
                use_chain_signals=use_chain_signals,
                max_signals_per_chain=max_signals_per_chain
            )

            print(f'Zeitraum Anfang:\n{df_with_backtest["Datetime"].head(1).values[0]}')
            print(f'Zeitraum Ende:\n{df_with_backtest["Datetime"].tail(1).values[0]}')

            # print("Backtesting abgeschlossen. Ergebnisse:")
            for k, v in results.items():
                print(f"{k}: {v}")

            # Optional: Exportieren
            # try:
            #     df_with_backtest.to_excel("df_with_backtest.xlsx", index=False)
            #     print("Backtesting-Daten erfolgreich nach 'df_with_backtest.xlsx' exportiert.")
            # except Exception as e:
            #     print(f"Fehler beim Export der Backtest-Daten: {e}")


            # Plot der Equity-Kurve
            plt.figure(figsize=(12, 6))
            plt.plot(df_with_backtest["Equity"], label='Equity')
            plt.title('Equity-Kurve')
            plt.xlabel('Bar')
            plt.ylabel('Equity')
            plt.legend()
            plt.show()

            # 10. Plotten der Ergebnisse
            print("Plotten der Ergebnisse...")
            try:
                plot_stock_prices(
                    config="val_trade" + str(config) + f"_f1_{f1_up_val}",
                    folder=r"Backtest\val",
                    df=df_with_backtest.tail(plot_length_val),
                    test=True,
                    trend="Predicted_Trend",
                    secondary_y_scale=secondary_y_scale,
                    x_interval_min=x_interval_min,
                    y_interval_dollar=y_interval_dollar,
                    additional_lines=additional_lines,
                    save_plot_as_picture=False
                )
                print("Vorhersage und Plot abgeschlossen.")
            except Exception as e:
                print(f"Fehler beim Plotten der Ergebnisse: {e}")
                traceback.print_exc()


            # auswertung_mit_mehreren_trials(study, df_data_test, direction, top_n_trials=1000, ausgabe_datei="backtests.xlsx", n_cpus=5)

    return f1_up_test


def upgrade_optuna_storage(database_name_optuna):
    """
    Führt ein Upgrade des Optuna Storage Schemas durch, um die Kompatibilität mit der aktuellen Optuna-Version sicherzustellen.

    Args:
        storage_url (str): Die URL des Optuna-Storages, z.B. "mysql+mysqlconnector://user:password@host/database"

    Raises:
        RuntimeError: Wenn das Upgrade fehlschlägt.
    """
    try:

        # storage_url = f"mysql+mysqlconnector://{user}:{password}@{host}/optuna_{database_name_optuna}"
        storage_url = f"mysql+mysqlconnector://{connection_config['user']}:{connection_config['password']}@{connection_config['host']}/optuna_{database_name_optuna}"

        print(f"Führe Optuna Storage Upgrade für die Storage-URL aus: {storage_url}")

        # Führe den Optuna Storage Upgrade Befehl aus
        result = subprocess.run(
            ['optuna', 'storage', 'upgrade', '--storage', storage_url],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("Optuna Storage Upgrade erfolgreich durchgeführt.")
        print(result.stdout)

    except FileNotFoundError:
        print("Fehler: Optuna CLI wurde nicht gefunden. Stellen Sie sicher, dass Optuna korrekt installiert ist.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("Fehler beim Ausführen des Optuna Storage Upgrade Befehls:")
        print(e.stderr)
        raise RuntimeError("Optuna Storage Upgrade fehlgeschlagen.") from e


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




def analyze_stock_signals(
        df,
        future_window,
        abs_threshold_up,
        abs_threshold_down,
        percentiles,
        plot=True,
        upper_percentile=95,
        lower_percentile=5,
        save_csv=False,
        output_csv_path='stock_analysis_with_signals.xlsx',
        significant_pct_threshold=0,  # Prozentuale Schwelle für signifikant
        extreme_pct_threshold=3,        # Prozentuale Schwelle für extrem
        duration_percentiles=[10, 20, 30, 40, 50, 60, 70, 80, 90]  # Perzentile für die Dauer
):
    """
    Analysiert signifikante prozentuale und absolute Schwankungen in Aktienkursen und generiert Up-, Down- und Stable-Signale.
    Zusätzlich werden signifikante Bewegungen hinsichtlich Größe und Dauer analysiert und die Perzentilverteilung der Dauer berechnet.

    Args:
        df (pd.DataFrame): DataFrame mit historischen Aktienkursdaten. Muss eine 'Close'-Spalte enthalten.
        future_window (int): Anzahl der Intervalle, die in die Zukunft geschaut werden sollen.
        abs_threshold_up (float): Schwellenwert für Up-Signale basierend auf absoluten Veränderungen.
        abs_threshold_down (float): Schwellenwert für Down-Signale basierend auf absoluten Veränderungen.
        percentiles (list, optional): Liste der Perzentile zur Berechnung der Schwellenwerte. Beispiel: [5, 95].
        plot (bool, optional): Wenn True, werden Visualisierungen erstellt. Standard ist True.
        upper_percentile (int, optional): Das obere Perzentil für Up-Signale. Standard ist 95.
        lower_percentile (int, optional): Das untere Perzentil für Down-Signale. Standard ist 5.
        save_csv (bool, optional): Wenn True, werden die Ergebnisse in einer Excel-Datei gespeichert. Standard ist False.
        output_csv_path (str, optional): Pfad zur Excel-Datei. Standard ist 'stock_analysis_with_signals.xlsx'.
        significant_pct_threshold (float, optional): Prozentuale Schwelle für signifikante Bewegungen. Standard ist 0.5%.
        extreme_pct_threshold (float, optional): Prozentuale Schwelle für extreme Bewegungen. Standard ist 1.0%.
        duration_percentiles (list, optional): Liste der Perzentile zur Berechnung der Dauerverteilung. Beispiel: [10, 50, 90].

    Returns:
        tuple: (df_signals (pd.DataFrame), thresholds (dict), summary (pd.DataFrame))
    """

    # Überprüfen, ob die 'Close'-Spalte vorhanden ist
    if 'Close' not in df.columns:
        raise ValueError("Der DataFrame muss eine 'Close'-Spalte enthalten.")

    # Sicherstellen, dass der Index datetime ist
    if not np.issubdtype(df.index.dtype, np.datetime64):
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
        else:
            raise ValueError("Der DataFrame muss einen datetime Index haben oder eine 'Datetime'-Spalte enthalten.")

    # Berechnung der zukünftigen Schlusskurse
    df = df.copy()
    df['Future_Close'] = df['Close'].shift(-future_window)

    # Berechnung der prozentualen Veränderungen
    df['Pct_Change'] = (df['Future_Close'] - df['Close']) / df['Close'] * 100

    # Berechnung der absoluten Veränderungen
    df['Abs_Change'] = df['Future_Close'] - df['Close']

    # Entfernen von NaN-Werten, die durch das Shift entstehen
    df.dropna(subset=['Future_Close', 'Pct_Change', 'Abs_Change'], inplace=True)

    # Berechnung der Perzentile für prozentuale Veränderungen
    pct_stats = {}
    for p in percentiles:
        pct_stats[p] = np.percentile(df['Pct_Change'], p)

    # Erstellen eines DataFrame mit den Prozentwerten, mit Perzentilen als Index
    df_pct_stats = pd.DataFrame({
        'Pct_Value': [pct_stats[p] for p in percentiles],
        'Count': [
            np.sum(df['Pct_Change'] >= pct_stats[p]) if p >= upper_percentile else
            np.sum(df['Pct_Change'] <= pct_stats[p]) if p <= lower_percentile else
            0
            for p in percentiles
        ]
    }, index=percentiles)
    df_pct_stats.index.name = 'Percentile'
    print(f'pct_stats:\n{df_pct_stats}')

    # Berechnung der Perzentile für absolute Veränderungen
    abs_stats = {}
    for p in percentiles:
        abs_stats[p] = np.percentile(df['Abs_Change'], p)

    # Erstellen eines DataFrame mit den absoluten Perzentilen, mit Perzentilen als Index
    df_abs_stats = pd.DataFrame({
        'Abs_Value': [abs_stats[p] for p in percentiles],
        'Count': [
            np.sum(df['Abs_Change'] >= abs_stats[p]) if p >= upper_percentile else
            np.sum(df['Abs_Change'] <= abs_stats[p]) if p <= lower_percentile else
            0
            for p in percentiles
        ]
    }, index=percentiles)
    df_abs_stats.index.name = 'Percentile'
    print(f'abs_stats:\n{df_abs_stats}')

    # Festlegung der Schwellenwerte für prozentuale Veränderungen
    threshold_up = pct_stats.get(upper_percentile, np.percentile(df['Pct_Change'], upper_percentile))
    threshold_down = pct_stats.get(lower_percentile, np.percentile(df['Pct_Change'], lower_percentile))

    thresholds = {
        'upper_threshold': threshold_up,
        'lower_threshold': threshold_down
    }

    # Generierung der Signale basierend auf prozentualen Veränderungen
    df['Trend'] = np.where(df['Pct_Change'] >= threshold_up, 'Up',
                           np.where(df['Pct_Change'] <= threshold_down, 'Down', 'Stable'))

    # Berechnung der Anzahl der jeweiligen Signale
    signal_counts = df['Trend'].value_counts()
    print("\nAnzahl der jeweiligen Signale:")
    for signal in ['Up', 'Down', 'Stable']:
        count = signal_counts.get(signal, 0)
        print(f"{signal}: {count}")

    # ------------------------------
    # Analyse Signifikante Bewegungen mit dynamischer Dauerermittlung
    # ------------------------------
    print("\nAnalysiere signifikante Bewegungen mit dynamischer Dauerermittlung...")

    # Definition der signifikanten Bewegungen basierend auf prozentualen Schwellenwerten
    df['Significant'] = np.where(df['Pct_Change'] >= extreme_pct_threshold, 'Extreme',
                                 np.where(df['Pct_Change'] >= significant_pct_threshold, 'Significant', 'Normal'))

    # Fokussierung auf 'Significant' und 'Extreme' Bewegungen
    significant_df = df[df['Significant'].isin(['Significant', 'Extreme'])].copy()

    # Sort the significant_df by index to ensure chronological order
    significant_df = significant_df.sort_index()

    # Gruppieren der Events basierend auf dem Zeitabstand zwischen ihnen
    # Wenn der Zeitabstand zwischen zwei Signalen größer als der erwartete Zeitabstand ist, starten wir eine neue Gruppe
    # Angenommen, die erwartete Zeitabstand ist future_window * 5 Minuten

    time_gap = pd.Timedelta(minutes=5*future_window)

    significant_df['Time_Diff'] = significant_df.index.to_series().diff()

    # Neue Gruppe starten, wenn die Zeitdifferenz größer als die erwartete ist oder der Typ sich ändert
    significant_df['New_Group'] = (significant_df['Time_Diff'] > time_gap) | (significant_df['Significant'] != significant_df['Significant'].shift())

    significant_df['Group_ID'] = significant_df['New_Group'].cumsum()

    # Gruppieren und aggregieren
    grouped = significant_df.groupby('Group_ID')

    movements = []

    for name, group in grouped:
        start_time = group.index.min()
        end_time = group.index.max() + pd.Timedelta(minutes=5*future_window)
        duration = int((end_time - start_time).total_seconds() / 60)  # Dauer in Minuten
        start_price = group.iloc[0]['Close']
        peak_price = group.iloc[-1]['Future_Close']
        pct_change = group.iloc[-1]['Pct_Change']
        abs_change = group.iloc[-1]['Abs_Change']
        event_type = group.iloc[-1]['Significant']

        movements.append({
            'Start_Time': start_time,
            'Start_Price': start_price,
            'Peak_Price': peak_price,
            'Peak_Time': end_time,
            'Duration_Minutes': duration,
            'Pct_Change': pct_change,
            'Abs_Change': abs_change,
            'Type': event_type
        })

    # Erstellen eines DataFrames aus den Bewegungen
    significant_movements = pd.DataFrame(movements)

    print(f"\nAnzahl signifikanter Bewegungen: {len(significant_movements)}")

    # ------------------------------
    # Erstellung einer Zusammenfassung
    # ------------------------------
    print("\nErstelle Zusammenfassung der signifikanten Bewegungen...")

    summary = {}

    if not significant_movements.empty:
        # Gesamtanzahl
        summary['Total_Significant_Movements'] = len(significant_movements)
        summary['Total_Extreme_Movements'] = len(significant_movements[significant_movements['Type'] == 'Extreme'])
        summary['Total_Normal_Significant_Movements'] = len(significant_movements[significant_movements['Type'] == 'Significant'])

        # Prozentuale Veränderungen
        summary['Average_Pct_Change'] = significant_movements['Pct_Change'].mean()
        summary['Max_Pct_Change'] = significant_movements['Pct_Change'].max()
        summary['Min_Pct_Change'] = significant_movements['Pct_Change'].min()
        summary['Median_Pct_Change'] = significant_movements['Pct_Change'].median()
        summary['Std_Pct_Change'] = significant_movements['Pct_Change'].std()

        # Absolute Veränderungen
        summary['Average_Abs_Change'] = significant_movements['Abs_Change'].mean()
        summary['Max_Abs_Change'] = significant_movements['Abs_Change'].max()
        summary['Min_Abs_Change'] = significant_movements['Abs_Change'].min()
        summary['Median_Abs_Change'] = significant_movements['Abs_Change'].median()
        summary['Std_Abs_Change'] = significant_movements['Abs_Change'].std()

        # Dauer der Bewegungen
        summary['Average_Duration_Minutes'] = significant_movements['Duration_Minutes'].mean()
        summary['Max_Duration_Minutes'] = significant_movements['Duration_Minutes'].max()
        summary['Min_Duration_Minutes'] = significant_movements['Duration_Minutes'].min()
        summary['Median_Duration_Minutes'] = significant_movements['Duration_Minutes'].median()
        summary['Std_Duration_Minutes'] = significant_movements['Duration_Minutes'].std()

        # Perzentilverteilung der Dauer
        duration_values = significant_movements['Duration_Minutes']
        for p in duration_percentiles:
            percentile_value = np.percentile(duration_values, p)
            summary[f'Duration_{p}th_Percentile_Minutes'] = percentile_value

        # Zusammenfassung als DataFrame
        summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
        print(summary_df)
    else:
        print("Keine signifikanten Bewegungen gefunden.")
        summary_df = pd.DataFrame()

    # ------------------------------
    # Optional: Visualisierung der Signifikanten Bewegungen
    # ------------------------------
    if plot:
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['Close'], label='Close Price', color='blue')

        # Markiere signifikante Anstiege
        plt.scatter(significant_movements['Start_Time'],
                    significant_movements['Start_Price'],
                    color='orange', label='Significant Increase', marker='^')

        # Markiere extreme Anstiege
        extreme_movements = significant_movements[significant_movements['Type'] == 'Extreme']
        plt.scatter(extreme_movements['Start_Time'],
                    extreme_movements['Start_Price'],
                    color='red', label='Extreme Increase', marker='v')

        plt.xlabel('Zeit')
        plt.ylabel('Preis')
        plt.title('Signifikante Kursbewegungen auf 5-Minuten-Charts')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Zusätzliche Visualisierungen für die Zusammenfassung
        if not significant_movements.empty:
            # Histogramm der prozentualen Veränderungen
            plt.figure(figsize=(12, 6))
            sns.histplot(significant_movements['Pct_Change'], bins=30, kde=True, color='purple', edgecolor='black')
            plt.title('Verteilung der Prozentualen Veränderungen bei Signifikanten Bewegungen')
            plt.xlabel('Prozentuale Veränderung (%)')
            plt.ylabel('Anzahl')
            plt.grid(True)
            plt.show()

            # Boxplot der prozentualen Veränderungen nach Bewegungstyp
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Type', y='Pct_Change', data=significant_movements, palette={'Significant': 'orange', 'Extreme': 'red'})
            plt.title('Boxplot der Prozentualen Veränderungen nach Bewegungstyp')
            plt.xlabel('Bewegungstyp')
            plt.ylabel('Prozentuale Veränderung (%)')
            plt.grid(True)
            plt.show()

            # Boxplot der absoluten Veränderungen nach Bewegungstyp
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Type', y='Abs_Change', data=significant_movements, palette={'Significant': 'orange', 'Extreme': 'red'})
            plt.title('Boxplot der Absoluten Veränderungen nach Bewegungstyp')
            plt.xlabel('Bewegungstyp')
            plt.ylabel('Absolute Veränderung ($)')
            plt.grid(True)
            plt.show()

            # Histogramm der Dauer der Bewegungen
            plt.figure(figsize=(12, 6))
            sns.histplot(significant_movements['Duration_Minutes'], bins=30, kde=True, color='green', edgecolor='black')
            plt.title('Verteilung der Dauer der Signifikanten Bewegungen')
            plt.xlabel('Dauer (Minuten)')
            plt.ylabel('Anzahl')
            plt.grid(True)
            plt.show()

            # Boxplot der Dauer der Bewegungen nach Bewegungstyp
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Type', y='Duration_Minutes', data=significant_movements, palette={'Significant': 'orange', 'Extreme': 'red'})
            plt.title('Boxplot der Dauer der Bewegungen nach Typ')
            plt.xlabel('Bewegungstyp')
            plt.ylabel('Dauer (Minuten)')
            plt.grid(True)
            plt.show()

    # ------------------------------
    # Optional: Speichern der Ergebnisse in einer Excel-Datei
    # ------------------------------
    if save_csv:
        with pd.ExcelWriter(output_csv_path) as writer:
            df.to_excel(writer, sheet_name='Signals')
            df_pct_stats.to_excel(writer, sheet_name='Pct_Stats')
            df_abs_stats.to_excel(writer, sheet_name='Abs_Stats')
            significant_movements.to_excel(writer, sheet_name='Significant_Movements')
            if not summary_df.empty:
                summary_df.to_excel(writer, sheet_name='Summary')
        print(f"Ergebnisse wurden in '{output_csv_path}' gespeichert.")

    return df, thresholds, summary_df








def inspect_model(database_name_optuna, feature_names=None):
    """
    Lädt ein gespeichertes Keras-Modell und gibt detaillierte Informationen darüber aus,
    einschließlich der wichtigsten Features und der Modellarchitektur.

    Args:
        database_name_optuna (str): Name der Optuna-Datenbank zur Konstruktion des Modellpfads.
        feature_names (list, optional): Liste der Feature-Namen, um die Feature-Wichtigkeiten zu beschriften.
                                        Wenn None, werden sie als 'Feature 1', 'Feature 2', ... bezeichnet.

    Returns:
        None
    """

    model_save_dir = f"saved_models/nn_model_{database_name_optuna}"
    model_save_path = os.path.join(model_save_dir, f"nn_model_{database_name_optuna}.keras")

    # Prüfen, ob das Modell existiert
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Das Modell unter dem Pfad '{model_save_path}' wurde nicht gefunden.")

    # Laden des Modells mit den erforderlichen benutzerdefinierten Objekten
    try:
        model = tf.keras.models.load_model(
            model_save_path,
            custom_objects={
                'F1Score': tfa.metrics.F1Score(num_classes=2, average='macro'),
                'FeatureWeightingLayer': FeatureWeightingLayer  # Stellen Sie sicher, dass diese Klasse definiert ist
            },
            safe_mode=False  # Erlaubt das Laden von Lambda-Schichten
        )
    except Exception as e:
        print("Fehler beim Laden des Modells. Stellen Sie sicher, dass alle benutzerdefinierten Schichten definiert sind.")
        raise e

    # Modellzusammenfassung anzeigen
    print("\nModellzusammenfassung:")
    model.summary()

    # Informationen zu den Schichten sammeln
    layer_info = []
    for layer in model.layers:
        layer_dict = {
            'Layer Name': layer.name,
            'Layer Type': layer.__class__.__name__,
            'Output Shape': layer.output_shape
        }
        if isinstance(layer, tf.keras.layers.Dense):
            layer_dict['Units'] = layer.units
            layer_dict['Activation'] = layer.activation.__name__
            layer_dict['Kernel Initializer'] = layer.kernel_initializer.__class__.__name__
            layer_dict['Kernel Regularizer'] = layer.kernel_regularizer.__class__.__name__ if layer.kernel_regularizer else 'None'
        elif isinstance(layer, tf.keras.layers.Dropout):
            layer_dict['Rate'] = layer.rate
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer_dict['Momentum'] = layer.momentum
        elif 'FeatureWeightingLayer' in layer.__class__.__name__:
            # Spezifische Informationen für die FeatureWeightingLayer
            layer_dict['L2 Regularization'] = layer.l2_reg if hasattr(layer, 'l2_reg') else 'N/A'
        # Weitere Schichttypen können hier hinzugefügt werden
        layer_info.append(layer_dict)

    # Informationen als DataFrame darstellen
    layers_df = pd.DataFrame(layer_info)
    print("\nSchichtinformationen:")
    print(layers_df)

    # Feature-Wichtigkeiten extrahieren
    feature_weighting_layer = None
    for layer in model.layers:
        if 'feature_weighting' in layer.name.lower():
            feature_weighting_layer = layer
            break

    if feature_weighting_layer is None:
        print("\nFeatureWeightingLayer wurde im Modell nicht gefunden.")
    else:
        # Gewichte der FeatureWeightingLayer extrahieren
        weights = feature_weighting_layer.get_weights()[0].flatten()  # Annahme: Die Gewichte sind in der ersten Position

        # Überprüfen der Länge der feature_names
        num_features = len(weights)
        if feature_names is None:
            feature_names = [f"Feature {i + 1}" for i in range(num_features)]
            print("Keine Feature-Namen übergeben. Verwende generische Namen.")
        elif len(feature_names) != num_features:
            print("Warnung: Die Länge von 'feature_names' stimmt nicht mit der Anzahl der Features im Modell überein.")
            print(f"Anzahl der übergebenen Feature-Namen: {len(feature_names)}, Anzahl der Features im Modell: {num_features}")
            feature_names = [f"Feature {i + 1}" for i in range(num_features)]
            print("Verwende generische Feature-Namen.")
        else:
            print("Feature-Namen erfolgreich zugeordnet.")

        feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': weights
        })

        # Sortieren nach Wichtigkeit
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)

        print("\nFeature-Wichtigkeiten:")
        print(feature_importances)

        # Visualisierung der Feature-Wichtigkeiten
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
        plt.title('Feature Wichtigkeiten')
        plt.xlabel('Gewicht')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    # Weitere Informationen können hier hinzugefügt werden, z.B. Modellmetriken, Layer-Gewichte visualisieren etc.

    # Beispiel: Gesamte Anzahl der Parameter
    total_params = model.count_params()
    print(f"\nGesamtanzahl der Parameter im Modell: {total_params}")

    # Beispiel: Trainierbare und nicht-trainierbare Parameter
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    print(f"Trainierbare Parameter: {trainable_params}")
    print(f"Nicht-trainierbare Parameter: {non_trainable_params}")

    # Beispiel: Visualisierung des Netzwerks (optional, erfordert pydot und graphviz)
    try:
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model_architecture.png')
        print("\nModellarchitektur wurde als 'model_architecture.png' gespeichert.")
    except Exception as e:
        print("\nFehler beim Visualisieren der Modellarchitektur. Stellen Sie sicher, dass 'pydot' und 'graphviz' installiert sind.")
        print(e)


def sql_lite(save_or_load=None, df=None, database="", table=""):
    try:

        if save_or_load == "save":
                # In SQLite speichern
                engine = sqlalchemy.create_engine(f'sqlite:///{database}.db')
                df.to_sql(table, con=engine, index=False, if_exists='replace')

        elif save_or_load == "load":
            engine = sqlalchemy.create_engine(f'sqlite:///{database}.db', echo=False)
            conn = engine.connect()
            df = pd.read_sql(f'SELECT * FROM {table}', con=conn)
            conn.close()
            return df

        else:
            print("AUSWAHL FEHLT!!!!!!")
            exit()

    except:
        print(traceback.format_exc())



def berechne_zeitraeume(df_length, length_month, distance_month, training_month, backtest_month, validation_month):
    total_timeframe = training_month + backtest_month + validation_month
    total_timeframe_end = df_length - (distance_month * length_month)
    total_timeframe_beginning = total_timeframe_end - (total_timeframe * length_month)

    zeitraeume = {
        'nutzungszeitraum': [total_timeframe_beginning, total_timeframe_end],
        'modell_trainings_zeitraum': [1, training_month * length_month],
        'backtest_zeitraum': [
            training_month * length_month,
            training_month * length_month + backtest_month * length_month
        ],
        'bewertungszeitraum': [
            training_month * length_month + backtest_month * length_month,
            training_month * length_month + backtest_month * length_month + validation_month * length_month
        ]
    }

    # Validierung der Indizes
    for key, (start, end) in zeitraeume.items():
        if start < 0 or end > df_length:
            raise ValueError(f"Die Indizes für {key} liegen außerhalb der DataFrame-Grenzen.")
        if start >= end:
            raise ValueError(f"Der Startindex ist größer oder gleich dem Endindex für {key}.")

    return zeitraeume


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for f in self.files:
            f.write(message)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def analyze_course_state1(db,
                         table,
                         lookback_steps=12,
                         lookforward_steps=12,
                         threshold_high=0.002,  # Mindestprozentualer Anstieg, z.B. 0.2%
                         threshold_low=0.002,  # Maximal erlaubter Rückgang, z.B. 0.2%
                         use_adjacent_check=False,
                         adjacent_pos_threshold=0,
                         adjacent_neg_threshold=0):
    """
    Analysiert die Close-Werte eines DataFrames und klassifiziert den Marktzustand
    an jedem betrachteten Aktienpunkt basierend auf zwei Kriterien:

      1. Stärke der Kursbewegung im Lookforward-Fenster:
         - Es wird der maximale prozentuale Anstieg (pct_increase) und der maximale
           prozentuale Rückgang (pct_drop) innerhalb der nächsten Kerzen berechnet.
         - Zustand:
             * Trend Up: pct_increase >= threshold_high und pct_drop <= threshold_low
             * Trend Down: pct_drop >= threshold_low und pct_increase < threshold_high
             * Neutral: Alle anderen Fälle.

      2. Länge der zusammenhängenden positiven/negativen Bewegungen (Run-Längen)
         - Für Lookback- und Lookforward-Fenster werden die maximalen Run-Längen (Anzahl
           aufeinanderfolgender Kerzen, in denen der Kurs steigt bzw. fällt) ermittelt.

    Zusätzlich gibt die Funktion einen übersichtlichen Report aus – inkl. der verwendeten
    Parameter, statistischer Kennzahlen (State Counts, Mittelwert/Median der prozentualen
    Veränderungen sowie der Run-Längen) – und liefert Empfehlungen für die Einstellung der
    folgenden Parameter:

      - consecutive_negatives_lookback_steps / max_consecutive_negatives_lookback
      - consecutive_positives_lookback_steps / max_consecutive_positives_lookback
      - consecutive_negatives_forward_steps / max_consecutive_negatives_forward
      - consecutive_positives_forward_steps / max_consecutive_positives_forward

    Der erweiterte DataFrame wird als "analyze_course_state.xlsx" gespeichert.
    """
    import numpy as np
    import pandas as pd

    # Lade den DataFrame aus der Datenbank
    df = query_database1(db=db, table=table, limit=None)
    close = df['Close'].to_numpy()
    n = len(close)

    # Arrays zur Speicherung der prozentualen Veränderungen
    pct_increase_arr = np.full(n, np.nan)
    pct_drop_arr = np.full(n, np.nan)
    # Arrays für Run-Längen (Anzahl aufeinanderfolgender positiver/negativer Bewegungen)
    max_cons_pos_lookback_arr = np.full(n, np.nan)
    max_cons_neg_lookback_arr = np.full(n, np.nan)
    max_cons_pos_lookforward_arr = np.full(n, np.nan)
    max_cons_neg_lookforward_arr = np.full(n, np.nan)

    # Array zur Speicherung des Zustands
    state = np.full(n, "Undefined", dtype=object)

    # Hilfsfunktion: maximale Anzahl aufeinanderfolgender Elemente, die eine Bedingung erfüllen
    def max_run(arr, condition):
        max_count = 0
        count = 0
        for val in arr:
            if condition(val):
                count += 1
                if count > max_count:
                    max_count = count
            else:
                count = 0
        return max_count

    # Für jeden Punkt (Index i) mit vollständigem Lookforward-Fenster
    for i in range(lookback_steps, n - lookforward_steps):
        start_val = close[i]
        # Lookforward-Fenster: Von i bis i + lookforward_steps (inklusive)
        future_window = close[i:i + lookforward_steps + 1]
        max_future = np.max(future_window)
        min_future = np.min(future_window)
        pct_increase = (max_future - start_val) / start_val
        pct_drop = (start_val - min_future) / start_val

        pct_increase_arr[i] = pct_increase
        pct_drop_arr[i] = pct_drop

        # Run-Längen im Lookback-Fenster
        past_window = close[i - lookback_steps:i + 1]  # inklusive aktueller Punkt
        past_diff = np.diff(past_window)
        max_pos_lb = max_run(past_diff, lambda x: x > 0)
        max_neg_lb = max_run(past_diff, lambda x: x < 0)
        max_cons_pos_lookback_arr[i] = max_pos_lb
        max_cons_neg_lookback_arr[i] = max_neg_lb

        # Run-Längen im Lookforward-Fenster
        future_diff = np.diff(future_window)
        max_pos_lf = max_run(future_diff, lambda x: x > 0)
        max_neg_lf = max_run(future_diff, lambda x: x < 0)
        max_cons_pos_lookforward_arr[i] = max_pos_lf
        max_cons_neg_lookforward_arr[i] = max_neg_lf

        # Klassifikation anhand der prozentualen Veränderungen
        if pct_increase >= threshold_high and pct_drop <= threshold_low:
            state[i] = "Trend Up"
        elif pct_drop >= threshold_low and pct_increase < threshold_high:
            state[i] = "Trend Down"
        else:
            state[i] = "Neutral"

        # Optionale Prüfung direkt angrenzender Kerzen
        if use_adjacent_check and (i - 1 >= 0 and i + 1 < n):
            prev_diff = close[i] - close[i - 1]
            next_diff = close[i + 1] - close[i]
            if prev_diff > adjacent_pos_threshold and next_diff > adjacent_pos_threshold:
                if state[i] == "Trend Up":
                    state[i] = state[i] + " (adjacent confirmed)"
            if prev_diff < adjacent_neg_threshold and next_diff < adjacent_neg_threshold:
                if state[i] == "Trend Down":
                    state[i] = state[i] + " (adjacent confirmed)"

    # Füge die berechneten Werte in den DataFrame ein
    df['pct_increase_lookforward'] = pct_increase_arr
    df['pct_drop_lookforward'] = pct_drop_arr
    df['max_cons_pos_lookback'] = max_cons_pos_lookback_arr
    df['max_cons_neg_lookback'] = max_cons_neg_lookback_arr
    df['max_cons_pos_lookforward'] = max_cons_pos_lookforward_arr
    df['max_cons_neg_lookforward'] = max_cons_neg_lookforward_arr
    df['State'] = state

    # ---------------------------
    # Statistische Analyse der Ergebnisse
    # ---------------------------
    analysis_results = {}
    analysis_results['state_counts'] = df['State'].value_counts().to_dict()
    analysis_results['mean_pct_increase'] = df.groupby('State')['pct_increase_lookforward'].mean().to_dict()
    analysis_results['mean_pct_drop'] = df.groupby('State')['pct_drop_lookforward'].mean().to_dict()
    analysis_results['median_pct_increase'] = df.groupby('State')['pct_increase_lookforward'].median().to_dict()
    analysis_results['median_pct_drop'] = df.groupby('State')['pct_drop_lookforward'].median().to_dict()

    analysis_results['mean_max_cons_pos_lookback'] = df.groupby('State')['max_cons_pos_lookback'].mean().to_dict()
    analysis_results['mean_max_cons_neg_lookback'] = df.groupby('State')['max_cons_neg_lookback'].mean().to_dict()
    analysis_results['mean_max_cons_pos_lookforward'] = df.groupby('State')['max_cons_pos_lookforward'].mean().to_dict()
    analysis_results['mean_max_cons_neg_lookforward'] = df.groupby('State')['max_cons_neg_lookforward'].mean().to_dict()

    analysis_results['median_max_cons_pos_lookback'] = df.groupby('State')['max_cons_pos_lookback'].median().to_dict()
    analysis_results['median_max_cons_neg_lookback'] = df.groupby('State')['max_cons_neg_lookback'].median().to_dict()
    analysis_results['median_max_cons_pos_lookforward'] = df.groupby('State')['max_cons_pos_lookforward'].median().to_dict()
    analysis_results['median_max_cons_neg_lookforward'] = df.groupby('State')['max_cons_neg_lookforward'].median().to_dict()

    # ---------------------------
    # Ausgabe als übersichtlicher Report
    # ---------------------------
    print("\n=== Analyse Report ===")
    print("Verwendete Parameter:")
    print(f"  Database: {db}")
    print(f"  Table: {table}")
    print(f"  Lookback Steps: {lookback_steps}")
    print(f"  Lookforward Steps: {lookforward_steps}")
    print(f"  Threshold High (min. prozent. Anstieg): {threshold_high}")
    print(f"  Threshold Low (max. prozent. Rückgang): {threshold_low}")
    print(f"  Use Adjacent Check: {use_adjacent_check}")
    if use_adjacent_check:
        print(f"    Adjacent Pos Threshold: {adjacent_pos_threshold}")
        print(f"    Adjacent Neg Threshold: {adjacent_neg_threshold}")

    print("\n--- Statistische Zusammenfassung ---")
    print("\nState Counts:")
    for state_key, count in analysis_results['state_counts'].items():
        print(f"  {state_key}: {count}")

    print("\nDurchschnittliche prozentuale Veränderungen (Lookforward):")
    for key, val in analysis_results['mean_pct_increase'].items():
        print(f"  {key} - Mean Increase: {val:.5f}")
    for key, val in analysis_results['mean_pct_drop'].items():
        print(f"  {key} - Mean Drop: {val:.5f}")

    print("\nMedian der prozentualen Veränderungen (Lookforward):")
    for key, val in analysis_results['median_pct_increase'].items():
        print(f"  {key} - Median Increase: {val:.5f}")
    for key, val in analysis_results['median_pct_drop'].items():
        print(f"  {key} - Median Drop: {val:.5f}")

    print("\nDurchschnittliche Run-Längen (Lookback):")
    for key, val in analysis_results['mean_max_cons_pos_lookback'].items():
        print(f"  {key} - Mean Pos Run (LB): {val:.2f}")
    for key, val in analysis_results['mean_max_cons_neg_lookback'].items():
        print(f"  {key} - Mean Neg Run (LB): {val:.2f}")

    print("\nMedian Run-Längen (Lookback):")
    for key, val in analysis_results['median_max_cons_pos_lookback'].items():
        print(f"  {key} - Median Pos Run (LB): {val}")
    for key, val in analysis_results['median_max_cons_neg_lookback'].items():
        print(f"  {key} - Median Neg Run (LB): {val}")

    print("\nDurchschnittliche Run-Längen (Lookforward):")
    for key, val in analysis_results['mean_max_cons_pos_lookforward'].items():
        print(f"  {key} - Mean Pos Run (LF): {val:.2f}")
    for key, val in analysis_results['mean_max_cons_neg_lookforward'].items():
        print(f"  {key} - Mean Neg Run (LF): {val:.2f}")

    print("\nMedian Run-Längen (Lookforward):")
    for key, val in analysis_results['median_max_cons_pos_lookforward'].items():
        print(f"  {key} - Median Pos Run (LF): {val}")
    for key, val in analysis_results['median_max_cons_neg_lookforward'].items():
        print(f"  {key} - Median Neg Run (LF): {val}")

    # ---------------------------
    # Empfehlung zur Parameter-Einstellung basierend auf den neutralen Zuständen
    # ---------------------------
    # Wir verwenden hier die Median-Werte aus dem Zustand "Neutral" als Basis.
    neutral_pos_lb = analysis_results['median_max_cons_pos_lookback'].get('Neutral', None)
    neutral_neg_lb = analysis_results['median_max_cons_neg_lookback'].get('Neutral', None)
    neutral_pos_lf = analysis_results['median_max_cons_pos_lookforward'].get('Neutral', None)
    neutral_neg_lf = analysis_results['median_max_cons_neg_lookforward'].get('Neutral', None)

    # Empfehlung:
    # - Für die Lookback-Parameter verwenden wir das gesamte Fenster (lookback_steps) und setzen den max-Parameter auf den neutralen Median.
    # - Analog für den Lookforward-Bereich.
    print("\n=== Empfehlung zur Parameter-Einstellung ===")
    if neutral_pos_lb is not None and neutral_neg_lb is not None:
        print("Lookback:")
        print(f"  - Empfohlen: consecutive_positives_lookback_steps = {lookback_steps} (ganzer Lookback) und")
        print(f"                max_consecutive_positives_lookback ≈ {neutral_pos_lb}")
        print(f"  - Empfohlen: consecutive_negatives_lookback_steps = {lookback_steps} (ganzer Lookback) und")
        print(f"                max_consecutive_negatives_lookback ≈ {neutral_neg_lb}")
    if neutral_pos_lf is not None and neutral_neg_lf is not None:
        print("Lookforward:")
        print(f"  - Empfohlen: consecutive_positives_forward_steps = {lookforward_steps} (ganzer Lookforward) und")
        print(f"                max_consecutive_positives_forward ≈ {neutral_pos_lf}")
        print(f"  - Empfohlen: consecutive_negatives_forward_steps = {lookforward_steps} (ganzer Lookforward) und")
        print(f"                max_consecutive_negatives_forward ≈ {neutral_neg_lf}")
    print("Hinweis: Diese Empfehlungen basieren auf den Medianwerten in neutralen Phasen. Anpassungen können nötig sein, wenn Trends früher erkannt werden sollen.")
    print("======================\n")


def analyze_course_state(db,
                         table,
                         lookback_steps=12,
                         lookforward_steps=12,
                         threshold_high=0.002,  # Mindestprozentualer Anstieg, z.B. 0.2%
                         threshold_low=0.002,  # Maximal erlaubter Rückgang, z.B. 0.2%
                         use_adjacent_check=False,
                         adjacent_pos_threshold=0,
                         adjacent_neg_threshold=0):
    """
    Analysiert die Close-Werte eines DataFrames und klassifiziert den Marktzustand
    an jedem betrachteten Aktienpunkt basierend auf zwei Kriterien:

      1. Stärke der Kursbewegung im Lookforward-Fenster:
         - Es wird der maximale prozentuale Anstieg (pct_increase) und der maximale
           prozentuale Rückgang (pct_drop) innerhalb der nächsten Kerzen berechnet.
         - Zustand:
             * Trend Up: pct_increase >= threshold_high und pct_drop <= threshold_low
             * Trend Down: pct_drop >= threshold_low und pct_increase < threshold_high
             * Neutral: Alle anderen Fälle.

      2. Länge der zusammenhängenden positiven/negativen Bewegungen (Run-Längen)
         - Für Lookback- und Lookforward-Fenster werden die maximalen Run-Längen (Anzahl
           aufeinanderfolgender Kerzen, in denen der Kurs steigt bzw. fällt) ermittelt.

    Neu: Zusätzlich werden direkt angrenzende Kursbewegungen analysiert. Für jeden Punkt
    werden folgende Metriken berechnet:
         - adjacent_pos_run_lookback: Anzahl der unmittelbar hintereinander folgenden
           positiven Kursbewegungen vor dem Punkt.
         - adjacent_neg_run_lookback: Analog für negative Bewegungen vor dem Punkt.
         - adjacent_pos_run_lookforward: Anzahl der unmittelbar hintereinander folgenden
           positiven Kursbewegungen nach dem Punkt.
         - adjacent_neg_run_lookforward: Analog für negative Bewegungen nach dem Punkt.

    Die Funktion gibt einen übersichtlichen Report aus – inkl. der verwendeten Parameter,
    statistischer Kennzahlen (State Counts, Mittelwert/Median der prozentualen Veränderungen,
    der Run-Längen sowie der angrenzenden Run-Längen) – und liefert Empfehlungen für die
    Filterung von Labeln basierend auf den angrenzenden Bewegungen in neutralen Phasen.

    Der erweiterte DataFrame wird als "analyze_course_state.xlsx" gespeichert.
    """
    import numpy as np
    import pandas as pd

    # Lade den DataFrame aus der Datenbank
    df = query_database1(db=db, table=table, limit=None)
    close = df['Close'].to_numpy()
    n = len(close)

    # Arrays zur Speicherung der prozentualen Veränderungen
    pct_increase_arr = np.full(n, np.nan)
    pct_drop_arr = np.full(n, np.nan)
    # Arrays für Run-Längen in Lookback und Lookforward
    max_cons_pos_lookback_arr = np.full(n, np.nan)
    max_cons_neg_lookback_arr = np.full(n, np.nan)
    max_cons_pos_lookforward_arr = np.full(n, np.nan)
    max_cons_neg_lookforward_arr = np.full(n, np.nan)
    # Arrays für direkt angrenzende Run-Längen
    adjacent_pos_run_lookback = np.full(n, np.nan)
    adjacent_neg_run_lookback = np.full(n, np.nan)
    adjacent_pos_run_lookforward = np.full(n, np.nan)
    adjacent_neg_run_lookforward = np.full(n, np.nan)

    # Array zur Speicherung des Zustands
    state = np.full(n, "Undefined", dtype=object)

    # Hilfsfunktion: maximale Anzahl aufeinanderfolgender Elemente, die eine Bedingung erfüllen
    def max_run(arr, condition):
        max_count = 0
        count = 0
        for val in arr:
            if condition(val):
                count += 1
                if count > max_count:
                    max_count = count
            else:
                count = 0
        return max_count

    # Berechne prozentuale Veränderungen und Run-Längen
    # Für jeden Punkt, bei dem ein vollständiges Lookforward-Fenster vorliegt:
    for i in range(lookback_steps, n - lookforward_steps):
        start_val = close[i]
        # Lookforward-Fenster
        future_window = close[i:i + lookforward_steps + 1]
        max_future = np.max(future_window)
        min_future = np.min(future_window)
        pct_increase = (max_future - start_val) / start_val
        pct_drop = (start_val - min_future) / start_val
        pct_increase_arr[i] = pct_increase
        pct_drop_arr[i] = pct_drop

        # Run-Längen im Lookback-Fenster (inkl. aktueller Punkt)
        past_window = close[i - lookback_steps:i + 1]
        past_diff = np.diff(past_window)
        max_pos_lb = max_run(past_diff, lambda x: x > 0)
        max_neg_lb = max_run(past_diff, lambda x: x < 0)
        max_cons_pos_lookback_arr[i] = max_pos_lb
        max_cons_neg_lookback_arr[i] = max_neg_lb

        # Run-Längen im Lookforward-Fenster
        future_diff = np.diff(future_window)
        max_pos_lf = max_run(future_diff, lambda x: x > 0)
        max_neg_lf = max_run(future_diff, lambda x: x < 0)
        max_cons_pos_lookforward_arr[i] = max_pos_lf
        max_cons_neg_lookforward_arr[i] = max_neg_lf

        # Klassifikation basierend auf prozentualen Veränderungen
        if pct_increase >= threshold_high and pct_drop <= threshold_low:
            state[i] = "Trend Up"
        elif pct_drop >= threshold_low and pct_increase < threshold_high:
            state[i] = "Trend Down"
        else:
            state[i] = "Neutral"

    # Berechnung der direkt angrenzenden Run-Längen (für alle i von 1 bis n-2)
    for i in range(1, n - 1):
        # Lookback: Zähle positive Bewegungen unmittelbar vor i
        count = 0
        j = i - 1
        while j >= 0 and (close[j + 1] - close[j] > 0):
            count += 1
            j -= 1
        adjacent_pos_run_lookback[i] = count

        count = 0
        j = i - 1
        while j >= 0 and (close[j + 1] - close[j] < 0):
            count += 1
            j -= 1
        adjacent_neg_run_lookback[i] = count

        # Lookforward: Zähle positive Bewegungen unmittelbar nach i
        count = 0
        j = i
        while j < n - 1 and (close[j + 1] - close[j] > 0):
            count += 1
            j += 1
        adjacent_pos_run_lookforward[i] = count

        count = 0
        j = i
        while j < n - 1 and (close[j + 1] - close[j] < 0):
            count += 1
            j += 1
        adjacent_neg_run_lookforward[i] = count

    # Füge alle berechneten Werte in den DataFrame ein
    df['pct_increase_lookforward'] = pct_increase_arr
    df['pct_drop_lookforward'] = pct_drop_arr
    df['max_cons_pos_lookback'] = max_cons_pos_lookback_arr
    df['max_cons_neg_lookback'] = max_cons_neg_lookback_arr
    df['max_cons_pos_lookforward'] = max_cons_pos_lookforward_arr
    df['max_cons_neg_lookforward'] = max_cons_neg_lookforward_arr
    df['adjacent_pos_run_lookback'] = adjacent_pos_run_lookback
    df['adjacent_neg_run_lookback'] = adjacent_neg_run_lookback
    df['adjacent_pos_run_lookforward'] = adjacent_pos_run_lookforward
    df['adjacent_neg_run_lookforward'] = adjacent_neg_run_lookforward
    df['State'] = state

    # ---------------------------
    # Statistische Analyse der Ergebnisse
    # ---------------------------
    analysis_results = {}
    analysis_results['state_counts'] = df['State'].value_counts().to_dict()
    analysis_results['mean_pct_increase'] = df.groupby('State')['pct_increase_lookforward'].mean().to_dict()
    analysis_results['mean_pct_drop'] = df.groupby('State')['pct_drop_lookforward'].mean().to_dict()
    analysis_results['median_pct_increase'] = df.groupby('State')['pct_increase_lookforward'].median().to_dict()
    analysis_results['median_pct_drop'] = df.groupby('State')['pct_drop_lookforward'].median().to_dict()

    analysis_results['mean_max_cons_pos_lookback'] = df.groupby('State')['max_cons_pos_lookback'].mean().to_dict()
    analysis_results['mean_max_cons_neg_lookback'] = df.groupby('State')['max_cons_neg_lookback'].mean().to_dict()
    analysis_results['mean_max_cons_pos_lookforward'] = df.groupby('State')['max_cons_pos_lookforward'].mean().to_dict()
    analysis_results['mean_max_cons_neg_lookforward'] = df.groupby('State')['max_cons_neg_lookforward'].mean().to_dict()

    analysis_results['median_max_cons_pos_lookback'] = df.groupby('State')['max_cons_pos_lookback'].median().to_dict()
    analysis_results['median_max_cons_neg_lookback'] = df.groupby('State')['max_cons_neg_lookback'].median().to_dict()
    analysis_results['median_max_cons_pos_lookforward'] = df.groupby('State')['max_cons_pos_lookforward'].median().to_dict()
    analysis_results['median_max_cons_neg_lookforward'] = df.groupby('State')['max_cons_neg_lookforward'].median().to_dict()

    # Zusätzlich: Analyse der direkt angrenzenden Run-Längen (z. B. für neutrale Zustände)
    analysis_results['mean_adjacent_pos_run_lookback'] = df.groupby('State')['adjacent_pos_run_lookback'].mean().to_dict()
    analysis_results['mean_adjacent_neg_run_lookback'] = df.groupby('State')['adjacent_neg_run_lookback'].mean().to_dict()
    analysis_results['mean_adjacent_pos_run_lookforward'] = df.groupby('State')['adjacent_pos_run_lookforward'].mean().to_dict()
    analysis_results['mean_adjacent_neg_run_lookforward'] = df.groupby('State')['adjacent_neg_run_lookforward'].mean().to_dict()

    analysis_results['median_adjacent_pos_run_lookback'] = df.groupby('State')['adjacent_pos_run_lookback'].median().to_dict()
    analysis_results['median_adjacent_neg_run_lookback'] = df.groupby('State')['adjacent_neg_run_lookback'].median().to_dict()
    analysis_results['median_adjacent_pos_run_lookforward'] = df.groupby('State')['adjacent_pos_run_lookforward'].median().to_dict()
    analysis_results['median_adjacent_neg_run_lookforward'] = df.groupby('State')['adjacent_neg_run_lookforward'].median().to_dict()

    # ---------------------------
    # Ausgabe als übersichtlicher Report
    # ---------------------------
    print("\n=== Analyse Report ===")
    print("Verwendete Parameter:")
    print(f"  Database: {db}")
    print(f"  Table: {table}")
    print(f"  Lookback Steps: {lookback_steps}")
    print(f"  Lookforward Steps: {lookforward_steps}")
    print(f"  Threshold High (min. prozent. Anstieg): {threshold_high}")
    print(f"  Threshold Low (max. prozent. Rückgang): {threshold_low}")
    print(f"  Use Adjacent Check: {use_adjacent_check}")
    if use_adjacent_check:
        print(f"    Adjacent Pos Threshold: {adjacent_pos_threshold}")
        print(f"    Adjacent Neg Threshold: {adjacent_neg_threshold}")

    print("\n--- Statistische Zusammenfassung ---")
    print("\nState Counts:")
    for state_key, count in analysis_results['state_counts'].items():
        print(f"  {state_key}: {count}")

    print("\nDurchschnittliche prozentuale Veränderungen (Lookforward):")
    for key, val in analysis_results['mean_pct_increase'].items():
        print(f"  {key} - Mean Increase: {val:.5f}")
    for key, val in analysis_results['mean_pct_drop'].items():
        print(f"  {key} - Mean Drop: {val:.5f}")

    print("\nMedian der prozentualen Veränderungen (Lookforward):")
    for key, val in analysis_results['median_pct_increase'].items():
        print(f"  {key} - Median Increase: {val:.5f}")
    for key, val in analysis_results['median_pct_drop'].items():
        print(f"  {key} - Median Drop: {val:.5f}")

    print("\nDurchschnittliche Run-Längen (Lookback):")
    for key, val in analysis_results['mean_max_cons_pos_lookback'].items():
        print(f"  {key} - Mean Pos Run (LB): {val:.2f}")
    for key, val in analysis_results['mean_max_cons_neg_lookback'].items():
        print(f"  {key} - Mean Neg Run (LB): {val:.2f}")

    print("\nMedian Run-Längen (Lookback):")
    for key, val in analysis_results['median_max_cons_pos_lookback'].items():
        print(f"  {key} - Median Pos Run (LB): {val}")
    for key, val in analysis_results['median_max_cons_neg_lookback'].items():
        print(f"  {key} - Median Neg Run (LB): {val}")

    print("\nDurchschnittliche Run-Längen (Lookforward):")
    for key, val in analysis_results['mean_max_cons_pos_lookforward'].items():
        print(f"  {key} - Mean Pos Run (LF): {val:.2f}")
    for key, val in analysis_results['mean_max_cons_neg_lookforward'].items():
        print(f"  {key} - Mean Neg Run (LF): {val:.2f}")

    print("\nMedian Run-Längen (Lookforward):")
    for key, val in analysis_results['median_max_cons_pos_lookforward'].items():
        print(f"  {key} - Median Pos Run (LF): {val}")
    for key, val in analysis_results['median_max_cons_neg_lookforward'].items():
        print(f"  {key} - Median Neg Run (LF): {val}")

    print("\nDurchschnittliche angrenzende Run-Längen:")
    for key, val in analysis_results['mean_adjacent_pos_run_lookback'].items():
        print(f"  {key} - Mean Adjacent Pos Run (LB): {val:.2f}")
    for key, val in analysis_results['mean_adjacent_neg_run_lookback'].items():
        print(f"  {key} - Mean Adjacent Neg Run (LB): {val:.2f}")
    for key, val in analysis_results['mean_adjacent_pos_run_lookforward'].items():
        print(f"  {key} - Mean Adjacent Pos Run (LF): {val:.2f}")
    for key, val in analysis_results['mean_adjacent_neg_run_lookforward'].items():
        print(f"  {key} - Mean Adjacent Neg Run (LF): {val:.2f}")

    print("\nMedian angrenzender Run-Längen:")
    for key, val in analysis_results['median_adjacent_pos_run_lookback'].items():
        print(f"  {key} - Median Adjacent Pos Run (LB): {val}")
    for key, val in analysis_results['median_adjacent_neg_run_lookback'].items():
        print(f"  {key} - Median Adjacent Neg Run (LB): {val}")
    for key, val in analysis_results['median_adjacent_pos_run_lookforward'].items():
        print(f"  {key} - Median Adjacent Pos Run (LF): {val}")
    for key, val in analysis_results['median_adjacent_neg_run_lookforward'].items():
        print(f"  {key} - Median Adjacent Neg Run (LF): {val}")

    # ---------------------------
    # Empfehlung zur Parameter-Einstellung basierend auf neutralen Zuständen
    # ---------------------------
    # Wir verwenden hier die Median-Werte aus dem Zustand "Neutral" als Basis.
    neutral_pos_lb = analysis_results['median_max_cons_pos_lookback'].get('Neutral', None)
    neutral_neg_lb = analysis_results['median_max_cons_neg_lookback'].get('Neutral', None)
    neutral_pos_lf = analysis_results['median_max_cons_pos_lookforward'].get('Neutral', None)
    neutral_neg_lf = analysis_results['median_max_cons_neg_lookforward'].get('Neutral', None)
    neutral_adj_pos_lb = analysis_results['median_adjacent_pos_run_lookback'].get('Neutral', None)
    neutral_adj_neg_lb = analysis_results['median_adjacent_neg_run_lookback'].get('Neutral', None)
    neutral_adj_pos_lf = analysis_results['median_adjacent_pos_run_lookforward'].get('Neutral', None)
    neutral_adj_neg_lf = analysis_results['median_adjacent_neg_run_lookforward'].get('Neutral', None)

    print("\n=== Empfehlung zur Parameter-Einstellung ===")
    if neutral_pos_lb is not None and neutral_neg_lb is not None:
        print("Lookback (Run-Längen aus gesamtem Fenster):")
        print(f"  - Empfohlen: consecutive_positives_lookback_steps = {lookback_steps} und max_consecutive_positives_lookback ≈ {neutral_pos_lb}")
        print(f"  - Empfohlen: consecutive_negatives_lookback_steps = {lookback_steps} und max_consecutive_negatives_lookback ≈ {neutral_neg_lb}")
    if neutral_pos_lf is not None and neutral_neg_lf is not None:
        print("Lookforward (Run-Längen aus gesamtem Fenster):")
        print(f"  - Empfohlen: consecutive_positives_forward_steps = {lookforward_steps} und max_consecutive_positives_forward ≈ {neutral_pos_lf}")
        print(f"  - Empfohlen: consecutive_negatives_forward_steps = {lookforward_steps} und max_consecutive_negatives_forward ≈ {neutral_neg_lf}")
    if neutral_adj_pos_lb is not None and neutral_adj_neg_lb is not None:
        print("Direkt angrenzende Bewegungen (Lookback):")
        print(f"  - Empfohlen: Adjacent Positives (Lookback) ≈ {neutral_adj_pos_lb}")
        print(f"  - Empfohlen: Adjacent Negatives (Lookback) ≈ {neutral_adj_neg_lb}")
    if neutral_adj_pos_lf is not None and neutral_adj_neg_lf is not None:
        print("Direkt angrenzende Bewegungen (Lookforward):")
        print(f"  - Empfohlen: Adjacent Positives (Lookforward) ≈ {neutral_adj_pos_lf}")
        print(f"  - Empfohlen: Adjacent Negatives (Lookforward) ≈ {neutral_adj_neg_lf}")
    print("Hinweis: Diese Empfehlungen basieren auf den Medianwerten in neutralen Phasen. Je nach gewünschter Sensitivität können diese Werte angepasst werden.")
    print("======================\n")




###################################################################################################################################
# NOTIZEN #
###################################################################################################################################

"""
    Erkenntnisse:
    - wenn future_steps zu groß entstehen signale zu weit weg vom Event und vor dem Muster
    - Die Signale müssen zwischen Muster und Up Trend liegen
    - Möglichst viele Signale an der richtigen Stelle --> mit Lookback legt man Ausschlusskriterien fest wodurch muster ausgeschlossen werden, die man nicht kennt
    - Um das Handeln von Up Signalen auf einem Abwärtstrend zu verhindern muss nth_signale verwendet werden. Dies kann erst wegfallen, wenn das Problem der Up-Vorhersagen auf fallenden Kursen beseitigt wurde
    - threshold_low_pct muss 0 sein. So sind Events zu definieren. So macht mand as im Trading.
    - all_after_lowest führt zu dem vermuteten Effekt.
    
    Fragen:
    - weniger Indikatoren, aber die richtigen - bessere Ergebnisse?
    - Future Steps = 12 zu lang?
    - threshold_high_pct muss signifikant sein. Nur durch testen ermittelbar?
    - all after lowest plus n, damit die Signale noch weiter in das Event geschoben werden?
    

    
    Idee:
    - Backtest:
        - Ma 10/20 Crossover oder Crossunder Ja/Nein muss zutreffen? Wahlweise andere Kombinationen. Winkel wichtig?
        
        
        
    
"""

###################################################################################################################################
# NOTIZEN #
###################################################################################################################################


# ########################## LOAD TXT TO DB
# load_txt_to_mysql(database_name="trading_bot", filename='nasdq_5min.txt', table_name="nasdq_5min")
# load_txt_to_mysql(database_name="trading_bot", filename='5 Years 5 Minute Chart Nasdaq Für Tobi Clean.txt', table_name="nasdq_5min_5y")
# exit()




# IMPORTANT
if __name__ == "__main__":

    # profiler_v2.stop()

    # profiler_v2.start(logdir='logs/fit/')

    faulthandler.enable()

    db = "trading_bot"
    orig_stock_data_table = 'nasdq_5min_5y'
    # orig_stock_data_table = 'NASDAQ_DLY_NDX_5'


    # ########################## Analysiere Volatilität

    # for i in [1,2,3,4,5]:


    # analyze_course_state(
    #     db=db,
    #     table=orig_stock_data_table,
    #     lookback_steps=12,
    #     lookforward_steps=12,
    #     threshold_high=0.002,  # Mindestprozentualer Anstieg, z.B. 0.2%
    #     threshold_low=0.002,  # Maximal erlaubter Rückgang (relativ zum Start), z.B. 0.1%
    #     use_adjacent_check=False,
    #     adjacent_pos_threshold=0,  # Falls direkt angrenzende positive Bewegungen extra geprüft werden sollen
    #     adjacent_neg_threshold=0
    # )
    #
    # exit()


    # ########################## LOAD TXT TO DB
    # load_txt_to_mysql(database_name="trading_bot", filename='nasdq_5min.txt', table_name="nasdq_5min")
    # load_txt_to_mysql(database_name="trading_bot", filename='NASDAQ_DLY_NDX_5.csv', table_name="NASDAQ_DLY_NDX_5")
    # exit()

    # future_steps = 48
    # future_steps = 36
    # future_steps = 24
    future_steps = 12
    # future_steps = 6
    # future_steps = 3

    threshold_high = 100
    threshold_low = 0

    # threshold_high_pct = 0.005
    # threshold_high_pct = 0.0045
    # threshold_high_pct = 0.004
    # threshold_high_pct = 0.0036
    # threshold_high_pct = 0.0035  # bisher benutzt
    # threshold_high_pct = 0.0034
    # threshold_high_pct = 0.0033
    # threshold_high_pct = 0.0032
    # threshold_high_pct = 0.0031
    threshold_high_pct = 0.002
    # threshold_high_pct = 0.0025
    # threshold_high_pct = 0.002
    # threshold_high_pct = 0.0015
    # threshold_high_pct = 0.001
    # threshold_high_pct = 0.0005
    # threshold_high_pct = 0.0001
    # threshold_high_pct = 0.00001
    # threshold_high_pct_l = [0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050]
    threshold_high_pct_l = [0.0020, 0.0030, 0.0040, 0.0050]

    # threshold_low_pct = 0.002
    threshold_low_pct = 0.000

    # threshold_low_pct = 1.0

    # use_percentage = False
    use_percentage = True

    time_series_sequence_length = 18

    # trendfunc = "v1"
    # trendfunc = "v4"
    # trendfunc = "v7"
    # trendfunc = "v8"
    # trendfunc = "v9"
    # trendfunc = "v10"
    # trendfunc = "v11"
    # trendfunc = "v12"
    # trendfunc = "v13"
    # trendfunc = "v14"
    trendfunc = "v15"

    # v9
    # up_signal_mode = "none"
    # up_signal_mode = "first_only"
    # up_signal_mode = "lowest_only"
    # up_signal_mode = "lowest_plus_one"
    up_signal_mode = "all_after_lowest"
    # up_signal_mode = "all_after_lowest_offset"

    require_double_ups = False
    offset_after_lowest = 0


    lookback_steps = 0
    lookback_threshold = 0.0
    lookback_threshold_l = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    use_lookback_check = True
    lookback_steps_l = [0, 1, 2, 3, 4, 5]
    # lookback_steps_l = [3, 4, 5]


    forward_steps = 0
    look_forward_threshold = 0.0
    look_forward_threshold_l = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    use_lookforward_check = True
    forward_steps_l = [0, 1, 2, 3,]


    min_cum_return = 0.0000
    min_cum_return_l = [0.0000, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030]

    # negativ
    consecutive_negatives_lookback_steps = 4
    max_consecutive_negatives_lookback = 0

    consecutive_negatives_forward_steps = 0
    max_consecutive_negatives_forward = 0

    # positive
    consecutive_positives_lookback_steps = 0
    max_consecutive_positives_lookback = 0

    consecutive_positives_forward_steps = 0
    max_consecutive_positives_forward = 0



    consecutive_negatives_lookback_steps_l = [1,2,3,4,5,6]
    max_consecutive_negatives_lookback_l = [0,1,2]

    consecutive_negatives_forward_steps_l = [1,2,3,4,5,6]
    max_consecutive_negatives_forward_l = [0,1,2]

    consecutive_positives_lookback_steps_l = [1,2,3,4,5,6]
    max_consecutive_positives_lookback_l = [0,1,2]

    consecutive_positives_forward_steps_l = [1,2,3,4,5,6]
    max_consecutive_positives_forward_l = [0,1,2]


    # !!!!!!!!!!!!!!!!!
    backwarts_shift_labels = 0


    database_name_optuna = "trading_ki_v2_kopie_test_400"  # noch Paras notieren, sowie Erkenntnisse bezüglich der Callbacks


    # nutzungszeitraum = [207000, 339000]  # Abschnitt -3
    # nutzungszeitraum = [213000, 345000]  # Abschnitt -2
    # nutzungszeitraum = [219000, 351000]  # Abschnitt -1


    # Parameterdefinition
    length_month = 6000  # Zeilen pro Monat
    df_lines = 351000  # Gesamtanzahl der Zeilen im DataFrame

    # Zeiträume in Monaten definieren
    timeframe_distance = 2  # Abstand vom Ende in Monaten
    # timeframe_distance = 3  # test

    # timeframe_training = 6  # Trainingszeitraum in Monaten # total 24 Monate
    # timeframe_training = 8  # Trainingszeitraum in Monaten # total 24 Monate
    # timeframe_training = 10  # Trainingszeitraum in Monaten # total 24 Monate
    # timeframe_training = 12  # Trainingszeitraum in Monaten # total 24 Monate
    # timeframe_training = 18  # Trainingszeitraum in Monaten # total 24 Monate
    # timeframe_training = 20  # Trainingszeitraum in Monaten # total 24 Monate
    # timeframe_training = 22  # Trainingszeitraum in Monaten # total 24 Monate
    # timeframe_training = 24  # Trainingszeitraum in Monaten # total 24 Monate
    # timeframe_training = 26  # Trainingszeitraum in Monaten # total 24 Monate
    # timeframe_training = 28  # Trainingszeitraum in Monaten # total 24 Monate
    # timeframe_training = 30  # Trainingszeitraum in Monaten # total 24 Monate
    # timeframe_training = 36  # Trainingszeitraum in Monaten # total 42 Monate
    # timeframe_training = 40  # Trainingszeitraum in Monaten # total 24 Monate
    timeframe_training = 44  # Mehr geht nicht danach error

    timeframe_backtest = 3  # Backtestzeitraum in Monaten
    # timeframe_backtest = 2  # test
    timeframe_validation = 1  # Bewertungszeitraum in Monaten


    # Verwendung der Funktion
    try:
        zeitraeume = berechne_zeitraeume(
            df_length=df_lines,
            length_month=length_month,
            distance_month=timeframe_distance,
            training_month=timeframe_training,
            backtest_month=timeframe_backtest,
            validation_month=timeframe_validation
        )
        print("Berechnete Zeiträume:", zeitraeume)
    except ValueError as e:
        print("Fehler bei der Berechnung der Zeiträume:", e)
        exit()

    nutzungszeitraum = [zeitraeume['nutzungszeitraum'][0], zeitraeume['nutzungszeitraum'][1]]  # Abschnitt TEST!!!!!!!
    modell_trainings_zeitraum = [zeitraeume['modell_trainings_zeitraum'][0], zeitraeume['modell_trainings_zeitraum'][1]]
    backtest_zeitraum = [zeitraeume['backtest_zeitraum'][0], zeitraeume['backtest_zeitraum'][1]]
    bewertungs_zeitraum = [zeitraeume['bewertungszeitraum'][0], zeitraeume['bewertungszeitraum'][1]]

    # config = f'fs_{future_steps}_thp_{threshold_high_pct}_tlp_{threshold_low_pct}_tssl_{time_series_sequence_length}_rdu_{require_double_ups}_oal_{offset_after_lowest}_lbs_{lookback_steps}_lbt_{lookback_threshold}_fws_{forward_steps}_lft_{look_forward_threshold}_mcr_{min_cum_return}'
    # print(config)
    # exit()

    # nutzungszeitraum = None
    db_lines = None
    # db_lines = 351623
    # db_lines = 200000
    # db_lines = 100000
    # db_lines = 20000
    # db_lines = 2000

    # db_bereich = "alte_daten"
    db_bereich = "neue_daten"


    # from_stock = True
    from_stock = False

    tuning = False
    training = True
    # training = False


    def evaluate(indicators):

        df_all_data, indicator_patterns = prepare_stock_data(
            test=False, analyze=False, nutzungszeitraum=nutzungszeitraum,
            from_stock=from_stock, save_in_db=False, db=db, trendfunc=trendfunc,
            db_bereich=db_bereich, from_table=orig_stock_data_table, to_table=f"{orig_stock_data_table}_{database_name_optuna}",
            future_steps=future_steps, time_series_sequence_length=time_series_sequence_length,
            use_percentage=use_percentage, min_cum_return=min_cum_return, threshold_high_pct=threshold_high_pct,
            threshold_low_pct=threshold_low_pct, threshold_high=threshold_high, threshold_low=threshold_low,
            database_name_optuna=database_name_optuna, db_lines=db_lines, use_create_lagged_features=True,
            up_signal_mode=up_signal_mode, require_double_ups=require_double_ups, offset_after_lowest=offset_after_lowest,
            lookback_steps=lookback_steps, lookback_threshold=lookback_threshold, use_lookback_check=use_lookback_check,
            use_lookforward_check=use_lookforward_check, look_forward_threshold=look_forward_threshold, forward_steps=forward_steps, indicators=indicators,
            consecutive_negatives_lookback_steps=consecutive_negatives_lookback_steps,
            max_consecutive_negatives_lookback=max_consecutive_negatives_lookback,
            consecutive_negatives_forward_steps=consecutive_negatives_forward_steps,
            max_consecutive_negatives_forward=max_consecutive_negatives_forward,
            backwarts_shift_labels=backwarts_shift_labels,
            consecutive_positives_lookback_steps=consecutive_positives_lookback_steps,
            max_consecutive_positives_lookback=max_consecutive_positives_lookback,
            consecutive_positives_forward_steps=consecutive_positives_forward_steps,
            max_consecutive_positives_forward=max_consecutive_positives_forward
            )
        # sql_lite(save_or_load="save", df=df_all_data, database=database_name_optuna, table=database_name_optuna)
        # exit()
        # df_all_data = sql_lite(save_or_load="load", database=database_name_optuna, table=database_name_optuna)

        # config_params = f'fs_{future_steps}_bwsl_{backwarts_shift_labels}_thp_{threshold_high_pct}_tlp_{threshold_low_pct}_tssl_{time_series_sequence_length}_rdu_{require_double_ups}_oal_{offset_after_lowest}_lbs_{lookback_steps}_lbt_{lookback_threshold}_fws_{forward_steps}_lft_{look_forward_threshold}_mcr_{min_cum_return}'
        # print(f'config_params:{config_params}')

        config_params_d = {
            "db": db,
            "orig_stock_data_table": orig_stock_data_table,
            "future_steps": future_steps,
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
            "threshold_high_pct": threshold_high_pct,
            "threshold_low_pct": threshold_low_pct,
            "use_percentage": use_percentage,
            "trendfunc": trendfunc,
            "up_signal_mode": up_signal_mode,
            "require_double_ups": require_double_ups,
            "offset_after_lowest": offset_after_lowest,
            "lookback_steps": lookback_steps,
            "lookback_threshold": lookback_threshold,
            "use_lookback_check": use_lookback_check,
            "forward_steps": forward_steps,
            "look_forward_threshold": look_forward_threshold,
            "use_lookforward_check": use_lookforward_check,
            "min_cum_return": min_cum_return,

            "consecutive_negatives_lookback_steps": consecutive_negatives_lookback_steps,
            "max_consecutive_negatives_lookback": max_consecutive_negatives_lookback,

            "consecutive_negatives_forward_steps": consecutive_negatives_forward_steps,
            "max_consecutive_negatives_forward": max_consecutive_negatives_forward,

            "consecutive_positives_lookback_steps": consecutive_positives_lookback_steps,
            "max_consecutive_positives_lookback": max_consecutive_positives_lookback,

            "consecutive_positives_forward_steps": consecutive_positives_forward_steps,
            "max_consecutive_positives_forward": max_consecutive_positives_forward,

            "backwarts_shift_labels": backwarts_shift_labels,
            "database_name_optuna": database_name_optuna,
            "length_month": length_month,
            "df_lines": df_lines,
            "timeframe_distance": timeframe_distance,
            "timeframe_training": timeframe_training,
            "timeframe_backtest": timeframe_backtest,
            "timeframe_validation": timeframe_validation,
            "db_lines": db_lines,
            "db_bereich": db_bereich,
            "from_stock": from_stock,
        }


        # print(f'config_params_d:{config_params_d}')

        for key, value in config_params_d.items():
            print(f'{key}: {value}')

        # config = "Test"
        config = time.time()

        config_params_d_str = '\n'.join([f"{key:<40}: {value}" for key, value in config_params_d.items()])

        # ############## GET COLUMNS
        list_of_tuning_columns = get_columns_by_mode(df=df_all_data, mode="training")
        list_of_predicting_columns = get_columns_by_mode(df=df_all_data, mode="predicting")
        print(f'list_of_tuning_columns:\n{list_of_tuning_columns}')
        print(f'list_of_predicting_columns:\n{list_of_predicting_columns}')
        # exit()

        config_txt = f'params = {config_params_d}\n\n' + f'{config_params_d_str}\n\n' + f'indicators = {indicators}\n\n' + f'indicator_patterns = {indicator_patterns}\n\n' + f'list_of_tuning_columns:\n{list_of_tuning_columns}\n\n' + f'list_of_predicting_columns:\n{list_of_predicting_columns}'

        with open(rf"Backtest\test_{config}.txt", 'w', encoding='utf-8') as datei:
            datei.write(config_txt)

        # exit()

        if db_lines:

            # test_size = 4000 / db_lines
            test_size = 18000 / db_lines

            # df_all_data_splitted_filtered_for_training, df_all_data_splitted_filtered_for_testing = split_data_v0(df_all_data, test_size=0.2)
            # df_all_data_splitted_filtered_for_training, df_all_data_splitted_filtered_for_testing = split_data_v0(df_all_data, test_size=0.1)
            df_all_data_splitted_filtered_for_training, df_all_data_splitted_filtered_for_testing = split_data_v0(df_all_data, test_size=test_size)
            # df_all_data_splitted_filtered_for_training, df_all_data_splitted_filtered_for_testing = split_data_v0(df_all_data, test_size=1.0)

        else:

            if nutzungszeitraum:
                df_all_data_splitted_filtered_for_training = df_all_data[modell_trainings_zeitraum[0]:modell_trainings_zeitraum[1]].copy()
                df_all_data_splitted_filtered_for_testing = df_all_data[backtest_zeitraum[0]:backtest_zeitraum[1]].copy()
                df_all_data_splitted_filtered_for_validation = df_all_data[bewertungs_zeitraum[0]:bewertungs_zeitraum[1]].copy()

        original_stdout = sys.stdout
        with open(rf"Backtest\test_{config}.txt", 'a', encoding='utf-8') as datei:
            # sys.stdout = Tee(original_stdout, datei)


            print(f'train:{len(df_all_data_splitted_filtered_for_training)}, test:{len(df_all_data_splitted_filtered_for_testing)}')
            print(f'train_head:\n{df_all_data_splitted_filtered_for_training["Datetime"].head(1).values[0]}')
            print(f'train_tail:\n{df_all_data_splitted_filtered_for_training["Datetime"].tail(1).values[0]}')

            # print(f'train_head:\n{df_all_data_splitted_filtered_for_training.head(100)}')
            # print(f'train_tail:\n{df_all_data_splitted_filtered_for_training.tail(100)}')
            # exit()
            if not from_stock:
                print(f'test_head:\n{df_all_data_splitted_filtered_for_testing["Datetime"].head(1).values[0]}')
                print(f'test_tail:\n{df_all_data_splitted_filtered_for_testing["Datetime"].tail(1).values[0]}')

            if nutzungszeitraum and not df_all_data_splitted_filtered_for_validation.empty:
                print(f'val_head:\n{df_all_data_splitted_filtered_for_validation["Datetime"].head(1).values[0]}')
                print(f'val_tail:\n{df_all_data_splitted_filtered_for_validation["Datetime"].tail(1).values[0]}')

            # exit()

            ############# TUNING #
            # train_model_v4(database_name_optuna=database_name_optuna, show_progression=False, verbose=True,
            #                tuning=True,
            #                n_trials=999999, n_jobs=5,
            #                max_epochs=10,
            #                db_wkdm_orig=df_all_data_splitted_filtered_for_training[list_of_tuning_columns],
            #                )

            ############# TRAINING #
            try:
                if training:
                    train_model_v4(database_name_optuna=database_name_optuna, show_progression=True, verbose=True,
                                   tuning=False,
                                   n_trials=1, n_jobs=1,
                                   max_epochs=999999,
                                   db_wkdm_orig=df_all_data_splitted_filtered_for_training[list_of_tuning_columns],
                                   config=config,
                                   )
            except:
                print(traceback.print_exc())
                sys.stdout = original_stdout
                return
            # exit()

            sys.stdout = Tee(original_stdout, datei)


            backtest_study_name = "test_20250410_3"


            if from_stock:
                df_data_test = df_all_data[list_of_predicting_columns]
                df_data_val = df_data_test

            else:
                # df_data_test = df_all_data_splitted_filtered_for_training[list_of_predicting_columns]  # ACHTUNG TEST !!!!!!!!!!!!!!!!!!!
                df_data_test = df_all_data_splitted_filtered_for_testing[list_of_predicting_columns]
                # df_data_test = df_all_data_splitted_filtered_for_testing[list_of_predicting_columns][:len(df_all_data_splitted_filtered_for_testing) // 3]  # ACHTUNG TEST !!!!!!!!!!!!!!!!!!!

                # df_data_test = df_all_data_splitted_filtered_for_validation[list_of_predicting_columns]  # ACHTUNG TEST !!!!!!!!!!!!!!!!!!!

                if nutzungszeitraum:
                    # df_data_val = df_all_data_splitted_filtered_for_training[list_of_predicting_columns]  # ACHTUNG TEST !!!!!!!!!!!!!!!!!!!
                    # df_data_val = df_all_data_splitted_filtered_for_testing[list_of_predicting_columns]  # ACHTUNG TEST !!!!!!!!!!!!!!!!!!!
                    # df_data_val = df_all_data_splitted_filtered_for_testing[list_of_predicting_columns][:len(df_all_data_splitted_filtered_for_testing) // 3]  # ACHTUNG TEST !!!!!!!!!!!!!!!!!!!
                    df_data_val = df_all_data_splitted_filtered_for_validation[list_of_predicting_columns]

                else:
                    # df_data_val = None

                    # df_data_val = df_all_data_splitted_filtered_for_training[list_of_predicting_columns]  # ACHTUNG TEST !!!!!!!!!!!!!!!!!!!
                    # df_data_val = df_all_data_splitted_filtered_for_testing[list_of_predicting_columns]  # ACHTUNG TEST !!!!!!!!!!!!!!!!!!!
                    df_data_val = df_all_data_splitted_filtered_for_validation[list_of_predicting_columns]

            """
            Backtest erweitern:
            - kürzere Testzeiträume oder wieder split?
            - only buy on chains with right coniditions
            """

            if training:
                model_folder = ""
            else:
                # model_folder = "nn_model_trading_ki_v2_kopie_test_400_1743425060.0295973"
                # model_folder = "nn_model_trading_ki_v2_kopie_test_400_1744041982.9029934"  # test_1744041982.9029934_f1_0.0169.png
                # model_folder = "nn_model_trading_ki_v2_kopie_test_400_1744010156.558374"  # test_1744010156.558374_f1_0.0578.png
                # model_folder = "nn_model_trading_ki_v2_kopie_test_400_1744136995.856982"  # test_1744136995.856982_f1_0.1102.png

                # model_folder = "nn_model_trading_ki_v2_kopie_test_400_1744358059.0109985"
                # model_folder = "nn_model_trading_ki_v2_kopie_test_400_1744359385.6022143"
                # model_folder = "nn_model_trading_ki_v2_kopie_test_400_1744360659.1109953"

                # model_folder = "nn_model_trading_ki_v2_kopie_test_400_1744365887.074534"
                model_folder = "nn_model_trading_ki_v2_kopie_test_400_1744362600.0868964"



            # HIER WIRD DIE LEISTUNG DES ELEMENTKOMBINATION GETESTET
            f1_up_test = predict_and_plot(
                config=config,
                plot_length_test=len(df_data_test),
                # plot_length_test=2000,
                plot_length_val=len(df_data_val),
                # plot_length_val=2000,

                df_data_test=df_data_test,
                df_data_val=df_data_val,

                database_name_optuna=database_name_optuna,
                predicting_columns=list_of_predicting_columns,  # Übergebe die Vorhersagespalten

                show_plot=True,
                # show_plot=False,

                save_plot_as_picture=True,

                from_stock=from_stock,

                # backtest=True,
                backtest=False,

                backtest_tuning=True,
                # backtest_tuning=False,



                backtest_tuning_with_n_splits=True,
                n_splits=1,
                best_backtest=True,
                # best_backtest=False,
                use_val_in_final_backtest=True,
                # use_val_in_final_backtest=False,
                backtest_tuning_trials=1000,
                backtest_tuning_parallel_trials=10,
                backtest_study_name=backtest_study_name,
                model_folder=model_folder
            )


            """
            bankrolle?
            - längerer testzeitraum von 3 Monaten
            - mindestens 50 Trades maximal 1000 --> mindestens 60 (Maximal 500?) über 3 Monate
            """

            # list_of_predicting_columns_filtered = [col for col in list_of_predicting_columns if col not in ['Close_orig', 'Datetime', 'Trend']]
            # inspect_model(database_name_optuna=database_name_optuna, feature_names=list_of_predicting_columns_filtered)

            sys.stdout = original_stdout

            return f1_up_test



    indicators = [
        "Volume",
        'Close_Pct_Change',
        "SMA",
        "EMA",
        "RSI",
        "Slope_Close_Pct_Change",
        "Slope_Close",
        "Slope_SMA",
        "Slope_EMA",
        "Slope_ROC_SMA",
        "Slope_ROC_EMA",

        # "Close_Diff",
        # "High_Diff",
        # "Low_Diff",

        # 'High_Diff_Pct',  # keine Ergebnisse, mit lag kommen viele signale
        # 'Low_Diff_Pct',  # keine Ergebnisse, mit lag kommen viele signale
        # "MACD",  # scheint wichtig zu sein, bring auch als lag noch mehr signale
        "BB",
        # "OBV",  # nicht unwichtig aber sehr wenige ergebnisse, selbes ergebnis wie ohne lag
        "ROC",
        # "CCI",  # erzeugt signale aber evtl die falschen? mitlag werden garkeien ergebnisse mehr erzeugt
        "MFI",

        "ATR",
        # "ADX",  # bringt alleine keine Ergebnisse, auch mit lag nicht
        # "STOCH",  # im lag gute ergebnisse
        # "WILLR",  # zeigt recht viele Signale, auch in lag
        # "STOCH_RSI",  # mittel viele Signale und gute Stellen, auch in lag

    ]

    indicators_candidate = indicators.copy()
    evaluate(indicators=indicators_candidate)



    # for threshold_high_pct in tqdm(threshold_high_pct_l, desc="threshold_high_pct_l"):
    #     for lookback_steps in tqdm(lookback_steps_l, desc="lookback_steps_l"):
    #         for forward_steps in tqdm(forward_steps_l, desc="forward_steps_l"):
    #             for min_cum_return in tqdm(min_cum_return_l, desc="min_cum_return_l"):



    # for lookback_threshold in tqdm(lookback_threshold_l, desc="lookback_threshold_l"):
    #     for look_forward_threshold in tqdm(look_forward_threshold_l, desc="look_forward_threshold_l"):
    #
    #         indicators_candidate = indicators.copy()
    #         evaluate(indicators=indicators_candidate)






    # for consecutive_negatives_lookback_steps in tqdm(consecutive_negatives_lookback_steps_l, desc="consecutive_negatives_lookback_steps_l"):
    #     for consecutive_positives_lookback_steps in tqdm(consecutive_positives_lookback_steps_l, desc="consecutive_positives_lookback_steps_l"):
    #
    #         for max_consecutive_negatives_lookback in tqdm(max_consecutive_negatives_lookback_l, desc="max_consecutive_negatives_lookback_l"):
    #
    #         # for consecutive_negatives_forward_steps in tqdm(consecutive_negatives_forward_steps_l, desc="consecutive_negatives_forward_steps_l"):
    #         #     for max_consecutive_negatives_forward in tqdm(max_consecutive_negatives_forward_l, desc="max_consecutive_negatives_forward_l"):
    #
    #             for max_consecutive_positives_lookback in tqdm(max_consecutive_positives_lookback_l, desc="max_consecutive_positives_lookback_l"):
    #                 #
    #                 #         for consecutive_positives_forward_steps in tqdm(consecutive_positives_forward_steps_l, desc="consecutive_positives_forward_steps_l"):
    #                 #             for max_consecutive_positives_forward in tqdm(max_consecutive_positives_forward_l, desc="max_consecutive_positives_forward_l"):
    #
    #
    #                 consecutive_negatives_forward_steps = consecutive_negatives_lookback_steps
    #                 max_consecutive_negatives_forward = max_consecutive_negatives_lookback
    #
    #                 consecutive_positives_lookback_steps = consecutive_positives_lookback_steps
    #                 max_consecutive_positives_lookback = max_consecutive_positives_lookback
    #
    #                 consecutive_positives_forward_steps = consecutive_positives_lookback_steps
    #                 max_consecutive_positives_forward = max_consecutive_positives_lookback
    #
    #
    #
    #                 if max_consecutive_negatives_lookback > consecutive_negatives_lookback_steps:
    #                     continue
    #
    #                 if max_consecutive_negatives_forward > consecutive_negatives_forward_steps:
    #                     continue
    #
    #                 if max_consecutive_positives_lookback > consecutive_positives_lookback_steps:
    #                     continue
    #
    #                 if max_consecutive_positives_forward > consecutive_positives_forward_steps:
    #                     continue
    #
    #
    #                 try:
    #                     indicators_candidate = indicators.copy()
    #                     evaluate(indicators=indicators_candidate)
    #                 except:
    #                     print(traceback.print_exc())
    #                     continue



