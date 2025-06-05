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
import talib

import nn_trading_bot_v3 as nn3
import nn_trading_bot_v1 as nn1
import nn_trading_bot_v4 as nn4

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



def create_dataset_4(original_df, history_time_window, future_time_window, mode):
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


    #TODO
    window_size = int(history_time_window / 5)

    def calculate_indicators(df):
        # EMA (Exponentiell Gleitender Durchschnitt)
        df['EMA_Close'] = talib.EMA(df['Close'].values, timeperiod=10)

        # ATR (Average True Range)
        df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)

        df['SAR'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)

        df['OBV'] = talib.OBV(df['Close'], df['Total Volume'])


        # Annahme: Berechnung der verschiedenen Ichimoku-Komponenten:
        nine_period_high = df['High'].rolling(window=9).max()
        nine_period_low = df['Low'].rolling(window=9).min()
        df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

        thirty_period_high = df['High'].rolling(window=26).max()
        thirty_period_low = df['Low'].rolling(window=26).min()
        df['kijun_sen'] = (thirty_period_high + thirty_period_low) / 2


        df['doji'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])


        # Initialisierung der SuperTrend-Komponenten
        hl2 = (df['High'] + df['Low']) / 2
        df['final_upperband'] = hl2 + (3 * df['ATR'])
        df['final_lowerband'] = hl2 - (3 * df['ATR'])
        df['SuperTrend'] = 0

        for i in range(1, len(df)):
            if df['Close'][i] > df['final_upperband'][i - 1]:
                df['SuperTrend'][i] = df['final_lowerband'][i]
            elif df['Close'][i] < df['final_lowerband'][i - 1]:
                df['SuperTrend'][i] = df['final_upperband'][i]
            else:
                df['SuperTrend'][i] = df['SuperTrend'][i - 1]

        # Stdev (Standardabweichung der Schlusskurse)
        df['Stdev'] = df['Close'].rolling(window=10).std()

        return df




    def prepare_data(df, window_size, columns):
        # Erstellen der neuen Features im bestehenden DataFrame durch direktes Hinzufügen
        df = calculate_indicators(df)

        for j in tqdm(range(1, window_size + 1)):
            for col in columns:
                shifted_col = df[col].shift(j)
                if col == 'Close':
                    # Berechnung der prozentualen Änderung für Close und direktes Hinzufügen zum DataFrame
                    df[f'close_diff_{j}'] = (shifted_col - df[col]) / df[col] * 100
                    # df[f'close_diff_{j}'] = shifted_col - df[col]

                elif col == "Total Volume":
                    df[f'{col.lower().replace(" ", "_")}_{j}'] = shifted_col

                # else:
                #     Kopieren der verschobenen Werte direkt in den DataFrame
                    # df[f'{col.lower().replace(" ", "_")}_{j}'] = shifted_col

        # Entfernen von Zeilen mit NaN, die durch das Shiften entstehen können
        return df.dropna()

    def process_stock_data(df, window_size, mode):
        if mode == "db":
            column_names = ['Open', 'High', 'Low', 'Close', 'Total Volume', 'Change']
        elif mode == "yf":
            column_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']  # Angenommen 'Volume' steht für 'Total Volume' bei YF

        # Bereiten Sie die Daten vor, ohne die ursprünglichen Spalten zu entfernen
        prepared_df = prepare_data(df.copy(), window_size, column_names)
        return prepared_df

    # Annahme: `df` ist der DataFrame, der die Daten enthält
    df = process_stock_data(df, window_size, mode)
    # df['rolling_mean_close'] = df['Close'].rolling(window=future_window_size).mean()

    future_window_size = int(future_time_window / 5)  # 30 Minuten, angenommen jedes Intervall ist 5 Minuten
    df['future_close'] = df['Close'].shift(-future_window_size)
    df['total_change'] = df['future_close'] - df['Close']
    df['percent_change'] = (df['total_change'] / df['Close']) * 100

    df.dropna(subset=['future_close', 'total_change', 'percent_change'], inplace=True)

    # print(df.head())

    start = 1
    end = window_size +1
    #TODO
    # Spalten auswählen und DataFrame bereinigen

    columns_of_interest = (['Date', 'Time', 'Open', 'Close'] +
                           [f'close_diff_{i}' for i in range(start, end)] +
                           [f'total_volume_{i}' for i in range(start, end)] +
                           # [f'change_{i}' for i in range(start, end)] +
                           # [f'open_{i}' for i in range(start, end)] +
                           # [f'high_{i}' for i in range(start, end)] +
                           # [f'low_{i}' for i in range(start, end)] +

                           ['EMA_Close'] +
                           ['ATR'] +
                           ['SAR'] +
                           ['OBV'] +
                           # ['tenkan_sen'] +
                           # ['kijun_sen'] +
                           # ['doji'] +
                           # ['SuperTrend'] +

                           # ['future_close'] +
                           # ['total_change'])
                           ['percent_change'])


    # print(df[columns_of_interest].tail())
    df['percent_change'] = df['percent_change'].replace('nan', None).replace(np.nan, None)
    # print(df[columns_of_interest].tail())

    return df[columns_of_interest].dropna()





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



def plot_stock_data_with_percent_change(df, date_col='Date', time_col='Time', close_col='Close', percent_change_col='percent_change', y_achse_gitter_punkte=100, x_achse_gitter_min=60, shift_minutes=0):
    # Erstelle eine Kopie des DataFrames, um sicherzustellen, dass keine Änderungen am Original vorgenommen werden
    df = df.copy()

    # Konvertiere Datum und Uhrzeit in ein DateTime-Objekt für das Plotten
    try:
        df['DateTime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))
    except Exception as e:
        print("Fehler bei der Konvertierung von Datum und Uhrzeit:", e)
        df[time_col] = df[time_col].apply(correct_time_format)  # Funktion correct_time_format muss definiert sein
        df['DateTime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))

    # Berechne den zukünftigen Close-Preis
    df['future_close'] = df[close_col] * (1 + df[percent_change_col] / 100)

    # Verschiebe die future_close Daten
    shift_steps = int(shift_minutes / 5)
    df['DateTime_shifted'] = df['DateTime'] + pd.Timedelta(minutes=shift_steps * 5)

    # Start der Visualisierung
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Zeichne den Linienplot für den Close-Kurs
    ax1.plot(df['DateTime'], df[close_col], label='Close Price', color='blue', linewidth=2)

    # Zeichne den Linienplot für den zukünftigen Close-Kurs mit Verschiebung
    ax1.plot(df['DateTime_shifted'], df['future_close'], label='Future Close', color='red', linewidth=2, linestyle='--')

    # Formatiere das Diagramm
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, x_achse_gitter_min]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Datums- und Zeitformat
    ax1.yaxis.set_major_locator(MultipleLocator(y_achse_gitter_punkte))

    ax1.set_xlabel('Datum und Uhrzeit')
    ax1.set_ylabel('Close Preis')

    ax1.set_title('Aktienkurs und zukünftige Preisänderung')
    plt.xticks(rotation=90)

    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    ax1.grid(True)
    plt.tight_layout()
    plt.show()


def plot_stock_data_with_total_change(df, date_col='Date', time_col='Time', close_col='Close', total_change_col='total_change', y_achse_gitter_punkte=100, x_achse_gitter_min=60, shift_minutes=0):
    # Erstelle eine Kopie des DataFrames, um sicherzustellen, dass keine Änderungen am Original vorgenommen werden
    df = df.copy()

    # Konvertiere Datum und Uhrzeit in ein DateTime-Objekt für das Plotten
    try:
        df['DateTime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))
    except Exception as e:
        # print("Fehler bei der Konvertierung von Datum und Uhrzeit:", e)
        df[time_col] = df[time_col].apply(correct_time_format)  # Funktion correct_time_format muss definiert sein
        df['DateTime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))

    # Berechne den zukünftigen Close-Preis
    df['future_close'] = df[close_col] + df[total_change_col]

    # Verschiebe die future_close Daten
    # shift_steps = int(shift_minutes / 5)
    df['DateTime_shifted'] = df['DateTime'] + pd.Timedelta(minutes=shift_minutes)

    # Start der Visualisierung
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Zeichne den Linienplot für den Close-Kurs
    ax1.plot(df['DateTime'], df[close_col], label='Close Price', color='blue', linewidth=2)

    # Zeichne den Linienplot für den zukünftigen Close-Kurs mit Verschiebung
    ax1.plot(df['DateTime_shifted'], df['future_close'], label='Future Close', color='red', linewidth=2, linestyle='--')

    print(df)

    # Formatiere das Diagramm
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, x_achse_gitter_min]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Datums- und Zeitformat
    ax1.yaxis.set_major_locator(MultipleLocator(y_achse_gitter_punkte))

    ax1.set_xlabel('Datum und Uhrzeit')
    ax1.set_ylabel('Close Preis')

    ax1.set_title('Aktienkurs und zukünftige Preisänderung')
    plt.xticks(rotation=90)

    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    ax1.grid(True)
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
    # exit()


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
    dataset_db_with_time = create_dataset_4(nasdq_5min_enriched, history_time_window=60, future_time_window=30, mode="db")
    print(dataset_db_with_time.head(1000))
    # save_to_db(dataframe=dataset_db_with_time, to_table="nasdq_5min_dataset_db_with_time", db=db)
    # exit()
    # ####################################################################################################################
    #
    #
    # # plot dataset
    # ####################################################################################################################
    # dataset_db_with_time = query_database(db=db, table_name="nasdq_5min_dataset_db_with_time")
    # dataset_db_with_time = dataset_db_with_time[:1000]
    # plot_stock_data_with_total_change(dataset_db_with_time, y_achse_gitter_punkte=100, x_achse_gitter_min=60, shift_minutes=5)
    # plot_stock_data_with_percent_change(dataset_db_with_time, y_achse_gitter_punkte=50, x_achse_gitter_min=60, shift_minutes=30)
    # exit()
    # ####################################################################################################################
    #
    #
    #
    # # save_to_db
    # ####################################################################################################################
    columns_to_drop = ['Date', 'Time', 'Open', 'Close']
    columns_to_drop = [col for col in columns_to_drop if col in dataset_db_with_time.columns]
    dataset_db = dataset_db_with_time.drop(columns_to_drop, axis=1)
    save_to_db(dataframe=dataset_db, to_table="nasdq_5min_dataset_db", db=db)
    # exit()
    ####################################################################################################################



    # nn_training
    ####################################################################################################################

    # ALT Functional Kategorisch
    # database_name_v3 = f"db_test{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    # database_name_v3 = f'trading_bot_test{32}'  # threshold_change=25, history_time_window=60, future_time_window=30 accuracy_max=     ["target_category"] != "kein_ereignis"]  whrs sehr falsch aber interessant. plot nochmal analysieren!!!


    # NEU Functional Regression close_diff in Totalen Werten
    # database_name_v3 = "test_36"  # history_time_window=60,    future_time_window=5   loss= 192   ,val_loss=117, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_37"  # history_time_window=30,    future_time_window=5   loss= 192   ,val_loss=117, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_38"  # history_time_window=15,    future_time_window=5   loss= 192   ,val_loss=117, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_39"  # history_time_window=10,    future_time_window=5   loss= 192   ,val_loss=117, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_40"  # history_time_window=5,     future_time_window=5   loss= 192   ,val_loss=117, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_41"  # history_time_window=90,    future_time_window=5   loss= 192   ,val_loss=117, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_42"  # history_time_window=120,   future_time_window=5   loss= 191   ,val_loss=117, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_43"  # history_time_window=60,    future_time_window=10  loss= 385   ,val_loss=238, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_44"  # history_time_window=60,    future_time_window=20  loss= 744   ,val_loss=587, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_45"  # history_time_window=60,    future_time_window=30  loss= 1174  ,val_loss=702, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_46"  # history_time_window=60,    future_time_window=60  loss= 2336  ,val_loss=1386, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_47"  # history_time_window=90,    future_time_window=60  loss= 2328  ,val_loss=1385, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_48"  # history_time_window=120,   future_time_window=60  loss= 2318  ,val_loss=1388, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_49"  # history_time_window=150,   future_time_window=60  loss= 2332  ,val_loss=1386, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_50"  # history_time_window=180,   future_time_window=60  loss= 2313  ,val_loss= 1384, absolute Werte, diff_close, future_diff_close
    # database_name_v3 = "test_51"  # history_time_window=240,   future_time_window=60  loss= 2320  ,val_loss= 1383, absolute Werte, diff_close, future_diff_close


    # NEU Functional Regression close_diff in prozentualen Werten
    # database_name_v3 = "test_53"  # history_time_window=60,    future_time_window=30   loss=0.0805, val_loss=0.0461, prozentuale Werte, diff_close, future_diff_close


    # NEU Functional Regression close_diff in prozentualen Werten + weitere
    # database_name_v3 = "test_54"  # history_time_window=60,    future_time_window=30   loss=0.0805, val_loss=0.0461, prozentuale Werte, diff_close, future_diff_close, Volumen, ATR, SAR, OBW, tenkan_sen, kijun_sen, doji, super_trend  ###### keine Verbesserung

    database_name_v3 = "test_55"

    # train_ds: 57042, test_ds: 14261



    database_name = "trading_bot"
    table_dataset = "nasdq_5min_dataset_db"
    workers = 1
    nn4.train_model_v3(tuning=True, n_trials=workers * 1, n_jobs=workers, database_name="trading_bot", table_name="nasdq_5min_dataset_db", database_name_optuna=database_name_v3, show_progression=False, verbose=1)
    # nn4.train_model_v3(tuning=False, n_trials=workers * 1, n_jobs=workers, database_name="trading_bot", table_name="nasdq_5min_dataset_db", database_name_optuna=database_name_v3, show_progression=True, verbose=0)
    # exit()
    ####################################################################################################################



    # nn_predict functional model
    ####################################################################################################################
    # nasdq_5min_dataset_db_with_time = query_database(db=db, table_name="nasdq_5min_dataset_db_with_time")
    # nasdq_5min_dataset_db_with_time["total_change_soll"] = nasdq_5min_dataset_db_with_time["total_change"]
    # nasdq_5min_dataset_db_with_time["percent_change_soll"] = nasdq_5min_dataset_db_with_time["percent_change"]

    # nasdq_5min_dataset_db_with_time = nasdq_5min_dataset_db_with_time[:1000]
    # plot_stock_data_with_total_change(nasdq_5min_dataset_db_with_time, total_change_col='total_change', y_achse_gitter_punkte=100, x_achse_gitter_min=60, shift_minutes=5)
    # plot_stock_data_with_percent_change(nasdq_5min_dataset_db_with_time, total_change_col='percent_change', y_achse_gitter_punkte=100, x_achse_gitter_min=60, shift_minutes=30)
    # print(nasdq_5min_dataset_db_with_time)


    # columns_to_drop = ['total_change']
    # columns_to_drop = ['percent_change']
    # columns_to_drop = [col for col in columns_to_drop if col in nasdq_5min_dataset_db_with_time.columns]
    # nasdq_5min_dataset_db_with_time = nasdq_5min_dataset_db_with_time.drop(columns_to_drop, axis=1)


    # nasdq_5min_dataset_db_with_time_predicts = nn4.load_model_and_predict2(input_df=nasdq_5min_dataset_db_with_time, additional_cols=['total_change_soll', 'Date', 'Time', 'Open', 'Close'], database_name_optuna=database_name_v3)
    # nasdq_5min_dataset_db_with_time_predicts = nn4.load_model_and_predict2(input_df=nasdq_5min_dataset_db_with_time, change_column="percent_change", additional_cols=['percent_change_soll', 'Date', 'Time', 'Open', 'Close'], database_name_optuna=database_name_v3)
    # print(nasdq_5min_dataset_db_with_time_predicts)

    # plot_stock_data_with_total_change(nasdq_5min_dataset_db_with_time_predicts, total_change_col='total_change', y_achse_gitter_punkte=100, x_achse_gitter_min=60, shift_minutes=30)
    # plot_stock_data_with_percent_change(nasdq_5min_dataset_db_with_time_predicts, percent_change_col='percent_change', y_achse_gitter_punkte=100, x_achse_gitter_min=60, shift_minutes=5)
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
