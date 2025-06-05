import yfinance as yf
from datetime import timedelta
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MultipleLocator
import json

import nn_trading_bot_v3 as nn3
import nn_trading_bot_v1 as nn1

import datetime
import os
from joblib import dump, load

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

def save_to_db(dataframe, to_table, db):

    # Erstellt eine Verbindung zur Datenbank
    engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://root:@localhost/{db}', echo=False)
    conn = engine.connect()
    dataframe.to_sql(con=conn, name=f'{to_table}', if_exists='replace', index=False)
    conn.close()
    print(f"Daten erfolgreich in Tabelle '{to_table}' gespeichert.")


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


def enrich_data(df_orig):
    df = df_orig.copy()

    # Sicherstellen, dass die benötigten Spalten vorhanden sind
    if not {'Open', 'High', 'Low', 'Close', 'Up', 'Down'}.issubset(df.columns):
        raise ValueError("Required columns are missing from the DataFrame.")

    # Berechnung der neuen Spalten
    # df['Range'] = df['High'] - df['Low']
    # df['Midpoint'] = (df['High'] + df['Low']) / 2
    df['Change'] = df['Close'] - df['Open']
    # df['Relative Change (%)'] = (df['Change'] / df['Open']) * 100
    df['Total Volume'] = df['Up'] + df['Down']
    # df['Directional Strength'] = df['Up'] - df['Down']

    """
    Range: Differenz zwischen dem höchsten und dem niedrigsten Kurs.
    Midpoint: Durchschnitt aus dem höchsten und dem niedrigsten Kurs.
    Change: Differenz zwischen Schlusskurs und Eröffnungskurs.
    Relative Change (%): Prozentuale Veränderung vom Eröffnungskurs zum Schlusskurs.
    Total Volume: Gesamtzahl der Transaktionen.
    Directional Strength: Unterschied zwischen der Anzahl von Transaktionen, die zu einem Preisanstieg und denen, die zu einem Preisrückgang führten.
    """

    # Zurückgeben des erweiterten DataFrame
    return df

def plot_stock_data(df, num_rows=None):

    # Überprüfen, ob das DataFrame die erforderlichen Spalten enthält
    # required_columns = {'Open', 'High', 'Low', 'Close', 'Range', 'Midpoint', 'Change', 'Relative Change (%)', 'Total Volume', 'Directional Strength'}
    required_columns = {'Open', 'High', 'Low', 'Close', 'Change', 'Total Volume'}


    if not required_columns.issubset(df.columns):
        raise ValueError("The DataFrame does not contain all required columns.")

    # Begrenzung der Anzahl der Zeilen, falls angegeben
    if num_rows is not None:
        df = df.head(num_rows)

    # Erstellen eines Plots
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Hauptachsen für die Aktienkurse
    df[['Open', 'High', 'Low', 'Close']].plot(ax=ax1, title='NASDAQ Stock Metrics Over Time')
    ax1.set_ylabel('Price')

    # Zweitachse für die zusätzlichen Metriken
    ax2 = ax1.twinx()
    # df[['Range', 'Midpoint', 'Change', 'Relative Change (%)', 'Total Volume', 'Directional Strength']].plot(ax=ax2, linestyle='--')
    df[['Change', 'Total Volume']].plot(ax=ax2, linestyle='--')

    ax2.set_ylabel('Metrics')

    # Verbesserung der Legende
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # Anzeigen des Plots
    plt.show()



def create_dataset_3(original_df, history_time_window, future_time_window, threshold_change, mode):
    df = original_df.copy()

    # df = df[:1000]
    # Bereinigung und Vorbereitung des DataFrames



    if mode == "yf":
        df.reset_index(inplace=True)
        df.rename(columns={'Datetime': 'DateTime'}, inplace=True)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    elif mode == "db":
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Datum und Zeit extrahieren
    df['Date'] = df['DateTime'].dt.date
    df['Time'] = df['DateTime'].dt.time

    # Erste Öffnungszeiten und tägliche Differenzen berechnen
    first_opens = df.groupby('Date')['Open'].first()
    # df['daily_open_diff'] = first_opens.diff()
    # df['daily_open_diff'] = df['Date'].map(df['daily_open_diff']).fillna(0)


    #TODO
    window_size = int(history_time_window / 5)
    if mode == "db":
        # start bei 13 bis len(df)
        for i in tqdm(range(window_size + 1, len(df) + 1)):
            df_part = df[[
                'Open',
                'High',
                'Low',
                'Close',
                'Total Volume',
                'Change'
            ]][i-window_size-1:i]
            # print(df_part)

            for j in range(1, len(df_part)):
                df.loc[i - 1, f'close_diff_{j}'] = df_part.iloc[j, 0] - df_part.iloc[j - 1, 0]  # Differenz der 'Close'-Werte
                # df.loc[i - 1, f'total_volume_{j}'] = df_part.iloc[j, 1]  # Differenz der 'Total Volume'-Werte
                # df.loc[i - 1, f'change_{j}'] = df_part.iloc[j, 2]  # Differenz der 'Change'-Werte

                df.loc[i - 1, f'open_{j}'] = df_part.iloc[j, 0]
                df.loc[i - 1, f'high_{j}'] = df_part.iloc[j, 1]
                df.loc[i - 1, f'low_{j}'] = df_part.iloc[j, 2]
                df.loc[i - 1, f'close_{j}'] = df_part.iloc[j, 3]
                df.loc[i - 1, f'total_volume_{j}'] = df_part.iloc[j, 4]


    elif mode == "yf":
        for i in tqdm(range(window_size + 1, len(df) + 1)):
            df_part = df[['Close', 'Volume', 'Change']][i-window_size-1:i]
            # print(df_part)

            for j in range(1, len(df_part)):
                df.loc[i - 1, f'close_diff_{j}'] = df_part.iloc[j, 0] - df_part.iloc[j - 1, 0]  # Differenz der 'Close'-Werte
                df.loc[i - 1, f'total_volume_{j}'] = df_part.iloc[j, 1] - df_part.iloc[j - 1, 1]  # Differenz der 'Total Volume'-Werte
                df.loc[i - 1, f'change_{j}'] = df_part.iloc[j, 2] - df_part.iloc[j - 1, 2]  # Differenz der 'Change'-Werte

    future_window_size = int(future_time_window / 5)  # 30 Minuten, angenommen jedes Intervall ist 5 Minuten

    for index in range(len(df) - future_window_size):
        current_close = df.loc[index, 'Close']
        future_closes = df.loc[index + 1: index + future_window_size, 'Close']
        category = categorize_change_2(current_close, future_closes.tolist(), threshold_change, future_time_window)
        df.loc[index, 'target_category'] = category


    start = 1
    end = window_size +1
    #TODO
    # Spalten auswählen und DataFrame bereinigen
    # columns_of_interest = ['Date', 'Time', 'Open', 'Close', 'daily_open_diff'] + \
    columns_of_interest = (['Date', 'Time', 'Open', 'Close'] +
                          [f'close_diff_{i}' for i in range(start, end)] +
                          [f'total_volume_{i}' for i in range(start, end)] +
                          [f'change_{i}' for i in range(start, end)] +
                          ['target_category'])

    # columns_of_interest = (['Date', 'Time', 'Open', 'Close'] +
    #                        [f'open_{i}' for i in range(start, end)] +
    #                        [f'high_{i}' for i in range(start, end)] +
    #                        [f'low_{i}' for i in range(start, end)] +
    #                        [f'close_{i}' for i in range(start, end)] +
    #                        [f'total_volume_{i}' for i in range(start, end)] +
    #                        ['target_category'])


    # print(df[columns_of_interest].tail())
    df['target_category'] = df['target_category'].replace('nan', None).replace(np.nan, None)
    # print(df[columns_of_interest].tail())

    return df[columns_of_interest].dropna()


def categorize_change_2(current_price, future_prices, threshold_change, window_length):
    """
    Kategorisiert die Änderung basierend auf einem Array von zukünftigen Preisen im Vergleich zum aktuellen Preis.
    current_price: Aktueller Close-Preis.
    future_prices: Liste der zukünftigen Close-Preise innerhalb des Fensters.
    threshold_change: Schwellenwert für eine signifikante Änderung.
    window_length: Länge des Zeitfensters in Minuten.
    """
    # Überprüfen, ob der maximale absolute Unterschied den Schwellenwert überschreitet
    max_increase = max(future_prices) - current_price
    max_decrease = current_price - min(future_prices)

    if max_increase >= threshold_change:
        return f'plus_{threshold_change}p_{window_length}min'
    elif max_decrease >= threshold_change:
        return f'minus_{threshold_change}p_{window_length}min'
    else:
        return 'kein_ereignis'


def correct_time_format(time_value):
    # Überprüft, ob der Wert ein Timedelta ist und konvertiert diesen entsprechend.
    if isinstance(time_value, pd.Timedelta):
        # Umwandlung von Timedelta in Uhrzeit-String (HH:MM:SS)
        total_seconds = int(time_value.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
    else:
        # Wenn es kein Timedelta ist, nehmen wir an, dass es sich um einen regulären Zeit-String handelt
        time_str = time_value.strftime('%H:%M:%S')
    return time_str

def plot_stock_data_with_categories(df, date_col='Date', time_col='Time', close_col='Close', cat_col='target_category', show_labels=True, label_angle=90, y_achse_gitter_punkte=100, x_achse_gitter_min=60):
    # Um SettingWithCopyWarning zu vermeiden, arbeiten wir auf einer expliziten Kopie des DataFrames.
    df = df.copy()

    try:
        df['DateTime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))
    except:
        df[time_col] = df[time_col].apply(correct_time_format)
        df['DateTime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))

    # Beginne mit der Visualisierung
    fig, ax = plt.subplots(figsize=(14, 7))

    # Zeichne den Linienplot für den Aktienkurs
    ax.plot(df['DateTime'], df[close_col], label='Close Price', color='blue', linewidth=2)

    # Bereite die Markierungen für die Kategorien vor
    categories = df[cat_col].unique()

    # Verarbeiten der Kategorien, um sicherzustellen, dass sie korrekt geparst werden
    plus_categories = [cat for cat in categories if 'plus' in cat]
    minus_categories = [cat for cat in categories if 'minus' in cat]

    # Sortiere die Kategorien nach Zeitkomponente (Annahme: Kategorieformat 'plus_50p_5min')
    plus_categories.sort(key=lambda x: int(x.split('_')[-1].replace('min', '')))
    minus_categories.sort(key=lambda x: int(x.split('_')[-1].replace('min', '')))

    # Farbpaletten erstellen
    green_blue = LinearSegmentedColormap.from_list("green_blue", ["limegreen", "darkblue"], N=len(plus_categories))
    red_yellow = LinearSegmentedColormap.from_list("red_orange", ["red", "orange"], N=len(minus_categories))

    # Zuordnung der Farben zu Kategorien
    color_map = {cat: green_blue(i) for i, cat in enumerate(plus_categories)}
    color_map.update({cat: red_yellow(i) for i, cat in enumerate(minus_categories)})

    # Zeichne Markierungen für jede Kategorie
    for category, color in color_map.items():
        # Datenpunkte dieser Kategorie
        mask = df[cat_col] == category
        points = ax.scatter(df['DateTime'][mask], df[close_col][mask], color=color, label=category, s=50, edgecolors='k')
        if show_labels:
            for (x, y), label in zip(points.get_offsets(), df[cat_col][mask]):
                ax.text(x, y, label, color=color, rotation=label_angle, ha='right', va='bottom')

    # Formatiere das Diagramm
    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Setzt die Hauptticks auf jede Stunde
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, x_achse_gitter_min]))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Datums- und Zeitformat

    ax.yaxis.set_major_locator(MultipleLocator(y_achse_gitter_punkte))  # Setzt die Hauptticks alle 50 Einheiten auf der Y-Achse

    ax.set_title('Aktienkurs mit Kategorien-Markierungen')
    ax.set_xlabel('Datum und Uhrzeit')
    ax.set_ylabel('Close Preis')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def download_data_with_adjustment(ticker, interval, last_n_days):

    # start_date, end_date = calculate_start_end_dates(interval, max_data_points)

    try:
        data = yf.download(ticker, period=last_n_days, interval=interval)
        return data
    except:
        print(f"Keine Daten für {ticker} gefunden.")
        return None


    # # start_date und end_date sollten hier bereits datetime Objekte sein
    # while start_date < end_date:
    #     try:
    #         # data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)
    #         data = yf.download(ticker, period="1d", interval=interval)
    #
    #         if not data.empty:
    #             print(f"start_date:{start_date.strftime('%Y-%m-%d')}")
    #             print(f"end_date:{end_date.strftime('%Y-%m-%d')}")
    #
    #             return data
    #     except Exception as e:
    #         print(f"Fehler beim Herunterladen der Daten für {ticker} von {start_date.strftime('%Y-%m-%d')} bis {end_date.strftime('%Y-%m-%d')}: {e}")
    #
    #     start_date += timedelta(days=1)
    #     # end_date -= timedelta(days=1)
    #
    # print(f"Keine Daten für {ticker} gefunden.")
    # return None

def calculate_start_end_dates(interval, max_data_points=None):
    """
    Berechne die Start- und Enddaten basierend auf dem gewählten Intervall und der optionalen maximalen Anzahl von Datenpunkten.
    """
    now = datetime.datetime.now()

    # Standardmäßige Zeiträume festlegen
    if interval == '1m':
        days_back = 7
    elif interval in ['5m', '15m', '30m', '1h']:
        days_back = 60
    else:
        days_back = 365  # Standardmäßig ein Jahr zurück für tägliche, wöchentliche und monatliche Daten

    start_date = now - timedelta(days=days_back)
    end_date = now

    # Anpassung des Startdatums basierend auf der Anzahl der Datenpunkte
    if max_data_points:
        estimated_interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60,
                                      '1d': 1440, '1wk': 10080, '1mo': 43800}
        minutes_back = max_data_points * estimated_interval_minutes[interval]
        start_date = now - timedelta(minutes=minutes_back)

    return start_date, end_date  # Gibt datetime Objekte zurück





def prepare_data_for_prediction(df, scaler, encoder, history_points=12):
    df = df.drop("target_category", axis=1)
    features = df.values
    features_scaled = scaler.transform(features)  # Verwenden Sie nur transform()

    # Erzeugen von Sequenzen für die Modellvorhersage
    X = []
    for i in range(len(features_scaled) - history_points + 1):
        X.append(features_scaled[i:i + history_points])
    X = np.array(X)

    return X

def make_predictions(model, X, encoder):
    predictions = model.predict(X)
    # Annahme: predictions sind kategorisch und encoder kann sie umkehren
    predicted_category = encoder.inverse_transform(predictions)
    # Erwartet, dass jede Vorhersage ein Array von Kategorien ist
    return [pred[0] for pred in predicted_category]


if __name__ == "__main__":
    db = "trading_bot"
    table_name = 'nasdq_5min'

    # load_txt_to_mysql(database_name=db, filename=r'K:\OneDrive\Coding\Python_trading_bot\nasdq_5min.txt', table_name=table_name)
    # load_txt_to_mysql(database_name=db, filename='nasdq_5min.txt', table_name=table_name)



    ####################################################################################################################
    # columns_to_retrieve = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Up', 'Down']
    # nasdq_5min = query_database(db=db, table_name=table_name, columns=columns_to_retrieve)
    # nasdq_5min_enriched = enrich_data(nasdq_5min)
    # save_to_db(dataframe=nasdq_5min_enriched, to_table="nasdq_5min_db_enriched", db=db)
    # print(nasdq_5min_enriched.head())
    # plot_stock_data(nasdq_5min_enriched, num_rows=100)  # Zeichnet die ersten 100 Zeilen des DataFrame
    # exit()
    ####################################################################################################################



    # dataset from db
    ####################################################################################################################
    nasdq_5min_enriched = query_database(db=db, table_name="nasdq_5min_db_enriched")
    dataset_db_with_time = create_dataset_3(nasdq_5min_enriched, threshold_change=25, history_time_window=60, future_time_window=30, mode="db")
    save_to_db(dataframe=dataset_db_with_time, to_table="nasdq_5min_dataset_db_with_time", db=db)
    ####################################################################################################################


    # plot dataset
    ####################################################################################################################
    # dataset_db_with_time = query_database(db=db, table_name="nasdq_5min_dataset_db_with_time")
    # plot_stock_data_with_categories(dataset_db_with_time[-500:], y_achse_gitter_punkte=50, x_achse_gitter_min=60)
    # exit()
    ####################################################################################################################



    # save_to_db
    ####################################################################################################################
    columns_to_drop = ['Date', 'Time', 'Open', 'Close']
    columns_to_drop = [col for col in columns_to_drop if col in dataset_db_with_time.columns]
    dataset_db = dataset_db_with_time.drop(columns_to_drop, axis=1)
    save_to_db(dataframe=dataset_db, to_table="nasdq_5min_dataset_db", db=db)
    # exit()
    ####################################################################################################################



    # nn_training
    ####################################################################################################################
    # database_name = "trading_bot"
    # table_dataset = "nasdq_5min_dataset_db"
    #
    # # database_name_v3 = f"db_test{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # database_name_v3 = f'trading_bot_test{22}'  # threshold_change=50, history_time_window=60, future_time_window=60 accuracy_max=0.748 ==> Erkennung funktioniert nicht, close_diff, total_volume, change
    # database_name_v3 = f'trading_bot_test{23}'  # threshold_change=50, history_time_window=120, future_time_window=60 accuracy_max=0.748 ==> Erkennung funktioniert nicht, close_diff, total_volume, change
    # database_name_v3 = f'trading_bot_test{24}'  # threshold_change=50, history_time_window=60, future_time_window=30 accuracy_max=0.874 ==> Erkennung funktioniert nicht, close_diff, total_volume, change
    # database_name_v3 = f'trading_bot_test{25}'  # threshold_change=50, history_time_window=60, future_time_window=60 accuracy_max=0.7526 ==> Erkennung funktioniert nicht, open, high, low, close, total_volume
    # database_name_v3 = f'trading_bot_test{26}'  # threshold_change=50, history_time_window=30, future_time_window=30 accuracy_max=0.874 ==> Erkennung funktioniert nicht, close_diff, total_volume, change
    # database_name_v3 = f'trading_bot_test{27}'  # threshold_change=50, history_time_window=30, future_time_window=30 accuracy_max= , activation überall variabel ermittelt
    # database_name_v3 = f'trading_bot_test{28}'  # threshold_change=50, history_time_window=180, future_time_window=60 accuracy_max=0.559 , activation überall variabel ermittelt beste Erkennung bisher auch wenn oft fragwürdig
    # database_name_v3 = f'trading_bot_test{29}'  # threshold_change=25, history_time_window=60, future_time_window=60 accuracy_max=0.559 , activation überall variabel ermittelt beste Erkennung bisher auch wenn oft fragwürdig könnte einen tick besser sein
    # database_name_v3 = f'trading_bot_test{30}'  # threshold_change=25, history_time_window=60, future_time_window=30 accuracy_max=0.67, auch nicht schlecht aber er kann nicht unterscheiden zwischen plus und minus, bestes bisher?
    # database_name_v3 = f'trading_bot_test{31}'  # threshold_change=25, history_time_window=30, future_time_window=30 accuracy_max=0.661 etwas schlechter siehe plot
    database_name_v3 = f'trading_bot_test{32}'  # threshold_change=25, history_time_window=60, future_time_window=30 accuracy_max=     ["target_category"] != "kein_ereignis"]  whrs sehr falsch aber interessant. plot nochmal analysieren!!!

    #
    workers = 10
    nn1.train_model_v3(tuning=True, n_trials=workers * 3, n_jobs=workers, database_name="trading_bot", table_name="nasdq_5min_dataset_db", database_name_optuna=database_name_v3, show_progression=False, verbose=0)
    nn1.train_model_v3(tuning=False, n_trials=workers * 1, n_jobs=workers, database_name="trading_bot", table_name="nasdq_5min_dataset_db", database_name_optuna=database_name_v3, show_progression=True, verbose=0)
    # exit()
    ####################################################################################################################



    # nn_predict functional model
    ####################################################################################################################
    nasdq_5min_dataset_db_with_time = query_database(db=db, table_name="nasdq_5min_dataset_db_with_time")
    nasdq_5min_dataset_db_with_time["target_category_soll"] = nasdq_5min_dataset_db_with_time["target_category"]

    nasdq_5min_dataset_db_with_time = nasdq_5min_dataset_db_with_time[:1000]
    # plot_stock_data_with_categories(nasdq_5min_dataset_db_with_time, y_achse_gitter_punkte=50, x_achse_gitter_min=30)


    columns_to_drop = ['target_category']
    columns_to_drop = [col for col in columns_to_drop if col in nasdq_5min_dataset_db_with_time.columns]
    nasdq_5min_dataset_db_with_time = nasdq_5min_dataset_db_with_time.drop(columns_to_drop, axis=1)


    nasdq_5min_dataset_db_with_time_predicts = nn1.load_model_and_predict2(input_df=nasdq_5min_dataset_db_with_time, additional_cols=['target_category_soll', 'Date', 'Time', 'Open', 'Close'], database_name_optuna=database_name_v3)
    # print(nasdq_5min_dataset_db_with_time_predicts)
    #
    plot_stock_data_with_categories(nasdq_5min_dataset_db_with_time_predicts, y_achse_gitter_punkte=50, x_achse_gitter_min=30)
    ####################################################################################################################



    # dataset from dyf
    ####################################################################################################################
    # """
    # 1d: Tägliche Daten
    # 1wk: Wöchentliche Daten
    # 1mo: Monatliche Daten
    # 1h: Stündliche Daten (nur für die letzten 60 Tage verfügbar)
    # 30m: Daten alle 30 Minuten (nur für die letzten 60 Tage verfügbar)
    # 15m: Daten alle 15 Minuten (nur für die letzten 60 Tage verfügbar)
    # 5m: Daten alle 5 Minuten (nur für die letzten 60 Tage verfügbar)
    # 1m: Daten jede Minute (nur für die letzten 7 Tage verfügbar)
    # """

    # Definiere das Intervall für die Datenabfrage
    # interval = '1m'
    # interval = '5m'
    # # interval = '15m'
    # # interval = '30m'
    # # interval = '1h'
    # # interval = '1d'
    # # interval = '1wk'
    # # interval = '1mo'
    #
    # # last_n_intervals = 13
    # # last_n_intervals = 12
    # # last_n_intervals = None
    #
    # # Setze den Ticker
    # # ticker_symbol = 'BOSCHLTD.BO'
    # # ticker_symbol = 'TSLA'
    # ticker_symbol = 'NQ=F'
    #
    # data_yf = download_data_with_adjustment(ticker_symbol, interval, last_n_days="60d")
    # print(data_yf[-5:])
    # print(data_yf.tail())

    # dataset_yf_with_time = create_dataset_yf(data_yf, threshold_change=50, time_window_steps_num=12)  # 6=30min 12=60min 36=3h
    # dataset_yf_with_time = create_dataset_3(data_yf, threshold_change=50, time_window=30, mode="yf")  # 6=30min 12=60min 36=3h

    # print(dataset_yf_with_time.tail())
    # plot_stock_data_with_categories(dataset_yf_with_time[-1000:], y_achse_gitter_punkte=50, x_achse_gitter_min=30)
    # save_to_db(dataframe=dataset_yf_with_time, to_table="nasdq_5min_dataset_yf_with_time", db=db)
    # # exit()
    ####################################################################################################################


    # save_to_db
    ####################################################################################################################
    # columns_to_drop = ['Date', 'Time', 'Open', 'Close']
    # columns_to_drop = [col for col in columns_to_drop if col in dataset_yf_with_time.columns]
    # dataset_yf = dataset_yf_with_time.drop(columns_to_drop, axis=1)
    # save_to_db(dataframe=dataset_yf, to_table="nasdq_5min_dataset_yf", db=db)
    # exit()
    ####################################################################################################################


    # query_db yf_with_time
    ####################################################################################################################
    # # table_name = "nasdq_5min_dataset_yf_with_time"
    # table_name = "nasdq_5min_dataset_db_with_time"
    #
    # nasdq_5min_dataset_yf_with_time = query_database(db=db, table_name=table_name)
    # # print(nasdq_5min_dataset_yf_with_time.head(1))
    # plot_stock_data_with_categories(nasdq_5min_dataset_yf_with_time[-1000:])
    #
    # results = load_model_and_predict(db=db, table_name=table_name, model_name="trading_bot_test1")
    # # print(results)
    # plot_stock_data_with_categories(results[-1000:])
    # # plot_stock_data_with_categories(results[:1000])

    ####################################################################################################################


    # data = yf.download("SPY", period="60d", interval="5m")
    # print(data.tail())
