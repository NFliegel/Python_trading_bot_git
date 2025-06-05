import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.stats import linregress
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import trading_bot
import nn_trading_bot_v2 as nn_training

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)



def calculate_start_end_dates(interval):
    """
    Berechne die Start- und Enddaten basierend auf dem gewählten Intervall.
    """
    now = datetime.now()

    if interval == '1m':
        start_date = now - timedelta(days=7)
    elif interval in ['5m', '15m', '30m', '1h']:
        start_date = now - timedelta(days=60)
    else:
        start_date = datetime(now.year, 1, 1)

    end_date = now

    return start_date, end_date  # Gibt datetime Objekte zurück


def plot_stock_data(df, ticker_symbol):
    """
    Zeichne die Schlusskurse für das gegebene DataFrame.

    Args:
    - df: Ein DataFrame mit den Aktiendaten, das mindestens eine 'Close' Spalte enthält.
    - ticker_symbol: Der Ticker der Aktie als String, der im Plot-Titel verwendet wird.
    """
    plt.figure(figsize=(10, 6))  # Setze die Größe des Plots
    plt.plot(df.index, df['Close'], label='Schlusskurse')  # Zeichne die Schlusskurse
    plt.title(f'Schlusskurse von {ticker_symbol}')  # Titel des Plots
    plt.xlabel('Datum')  # X-Achsen-Beschriftung
    plt.ylabel('Preis')  # Y-Achsen-Beschriftung
    plt.legend()  # Zeige Legende
    plt.xticks(rotation=45)  # Drehe die Datumsangaben, damit sie lesbar sind
    plt.tight_layout()  # Stelle sicher, dass nichts abgeschnitten wird
    plt.show()  # Zeige den Plot an



def add_trendline_and_sma_with_volume(df):
    # Berechne die Tage als numerische X-Werte
    # df['days'] = np.arange(len(df))
    df.loc[:, 'days'] = np.arange(len(df))

    # Lineare Trendlinie für Close-Preise
    slope, intercept, _, _, _ = linregress(df['days'], df['Close'])
    # df['trendline_close'] = intercept + slope * df['days']
    df.loc[:, 'trendline_close'] = intercept + slope * df['days']

    # Lineare Trendlinie für das Volumen
    slope_vol, intercept_vol, _, _, _ = linregress(df['days'], df['Volume'])
    df['trendline_volume'] = intercept_vol + slope_vol * df['days']

    # Einfacher gleitender Durchschnitt (SMA) für Close-Preise, z.B. 20-Tage SMA
    df['sma20'] = df['Close'].rolling(window=20).mean()

    return df



def plot_stock_data_full(df, ticker_symbol, columns_to_plot, last_n_intervals=None):
    df.index = pd.to_datetime(df.index)

    if last_n_intervals is not None:
        df = df.iloc[-last_n_intervals:]

    colors = {
        'Open': 'blue', 'High': 'green', 'Low': 'red', 'Close': 'black', 'Adj Close': 'cyan',
        'Volume': 'purple', 'Price Change': 'magenta', 'Price Change %': 'yellow',
        'Intra-Minute Volatility': 'orange', 'Volume Change': 'gray', 'Volume Change %': 'pink',
        'SMA 5 Close': 'lime', 'SMA 20 Close': 'orange', 'SMA 5 Volume': 'purple'
    }

    # Generiere den Titel mit Spaltennamen und Farben
    title_parts = [f'{ticker_symbol} - Kerzendiagramm mit ausgewählten Metriken:']
    for column in columns_to_plot:
        if column in colors:
            title_parts.append(f"{column} ({colors[column]})")
    title = ', '.join(title_parts)

    apds = []
    for column in columns_to_plot:
        if column in df:
            secondary_y = 'auto' if column in ['Volume', 'Volume Change', 'Volume Change %', 'SMA 5 Volume'] else False
            apds.append(mpf.make_addplot(df[column], color=colors.get(column, 'blue'), width=2, secondary_y=secondary_y))

    mpf.plot(df, type='candle', style='charles', title=title,
             ylabel='Preis', ylabel_lower='Volumen', volume=True, figratio=(12, 8),
             show_nontrading=False, addplot=apds)

def adjust_timezone(df, to_zone='Europe/Berlin'):
    """
    Konvertiere die Zeitzone eines DataFrame zu einer anderen, wenn nötig.

    Args:
    - df: pandas DataFrame mit einem DateTimeIndex.
    - to_zone: Die Zielzeitzone. Für Deutschland 'Europe/Berlin'.

    Returns:
    - df: Der DataFrame mit angepasster Zeitzone.
    """
    # Überprüfe, ob der DataFrame bereits eine Zeitzone hat
    if df.index.tz is not None:
        # Konvertiere direkt, wenn eine Zeitzone vorhanden ist
        df.index = df.index.tz_convert(to_zone)
    else:
        # Lokalisiere und konvertiere, wenn keine Zeitzone vorhanden ist
        df.index = df.index.tz_localize('UTC').tz_convert(to_zone)

    return df



def plot_with_trendlines(df, ticker_symbol, last_n_intervals=None):
    """
    Zeichnet für jede der Spalten 'Open', 'High', 'Low', 'Close' und 'Volume' eines DataFrame
    separate Diagramme mit Trendlinien.
    """
    # Wenn last_n_intervals definiert ist, beschränke den DataFrame auf die letzten n Datenpunkte
    if last_n_intervals is not None:
        df = df.iloc[-last_n_intervals:]

    # Berechne die Tage seit dem Startdatum als X-Werte für die Regression
    df['days'] = (df.index - df.index[0]).days

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 20), sharex=True)
    fig.suptitle(f'{ticker_symbol} - Trendlinien')

    for ax, column in zip(axes, ['Open', 'High', 'Low', 'Close', 'Volume']):
        # Verwende lineare Regression auf die aktuelle Spalte
        if column != 'Volume':  # Volumen wird gesondert behandelt
            slope, intercept, r_value, p_value, std_err = linregress(df['days'], df[column])
            trendline = intercept + slope * df['days']
            ax.plot(df.index, df[column], label=column)
            ax.plot(df.index, trendline, label=f'{column} Trend', linestyle='--')
        else:
            ax.bar(df.index, df[column], label=column, color='orange')

        ax.set_ylabel(column)
        ax.legend()

    plt.xlabel('Datum')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Anpassen, um Platz für den Gesamttitel zu schaffen
    plt.show()


def add_calculated_metrics(df):
    """
    Ergänzt den DataFrame mit berechneten Metriken: Absolute und prozentuale Preisänderungen,
    intra-Minuten-Volatilität, Volumenänderungen, gleitende Durchschnitte für Schlusskurse
    und Volumen.

    Args:
    - df: DataFrame mit den Spalten 'Open', 'High', 'Low', 'Close', 'Volume'.

    Returns:
    - DataFrame mit zusätzlichen berechneten Metriken.
    """
    # Stelle sicher, dass der Index als datetime formatiert ist
    df.index = pd.to_datetime(df.index)

    # Absolute Preisänderung
    df['Price Change'] = df['Close'].diff()

    # Prozentuale Preisänderung
    df['Price Change %'] = df['Close'].pct_change() * 100

    # Intra-Minuten-Volatilität
    df['Intra-Minute Volatility'] = df['High'] - df['Low']

    # Volumenänderungen
    df['Volume Change'] = df['Volume'].diff()

    # Prozentuale Volumenänderung
    df['Volume Change %'] = df['Volume'].pct_change() * 100

    # Gleitende Durchschnitte
    df['SMA 5 Close'] = df['Close'].rolling(window=5).mean()
    df['SMA 20 Close'] = df['Close'].rolling(window=20).mean()
    df['SMA 5 Volume'] = df['Volume'].rolling(window=5).mean()

    # Bereinige NaN Werte, die durch die Berechnungen entstehen können
    df.fillna(0, inplace=True)

    return df


# Stelle sicher, dass deine download_data_with_adjustment Funktion bereit ist, datetime Objekte zu handhaben:
def download_data_with_adjustment(ticker, start_date, end_date, interval):

    # start_date und end_date sollten hier bereits datetime Objekte sein
    while start_date < end_date:
        try:
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)
            if not data.empty:
                print(f"start_date:{start_date.strftime('%Y-%m-%d')}")
                print(f"end_date:{end_date.strftime('%Y-%m-%d')}")

                return data
        except Exception as e:
            print(f"Fehler beim Herunterladen der Daten für {ticker} von {start_date.strftime('%Y-%m-%d')} bis {end_date.strftime('%Y-%m-%d')}: {e}")

        start_date += timedelta(days=1)
        # end_date -= timedelta(days=1)


    print(f"Keine Daten für {ticker} gefunden.")
    return None


# def prepare_data(df, history_points=60):
#     """
#     Bereitet die Daten für das neuronale Netz vor.
#     """
#     # Schritt 1: Normalisierung der Daten
#     scaler = MinMaxScaler()
#     print(f'df_head:{df.head()}')
#     print(f'df_len:{len(df)}')
#     data_scaled = scaler.fit_transform(df)
#     print(f'data_scaled_len:{len(data_scaled)}\n')
#
#     # Schritt 2: Erstellung von Sequenzen
#     X, Y = [], []
#     for i in tqdm(range(len(data_scaled) - history_points), total=len(data_scaled) - history_points):
#         X.append(data_scaled[i:i + history_points])
#         Y.append(data_scaled[i + history_points, 3])  # Wir verwenden den 'Close'-Wert als Vorhersageziel
#
#     X, Y = np.array(X), np.array(Y)
#
#     # Schritt 3: Aufteilung in Trainings- und Testsets
#     split = int(0.8 * len(X))
#     X_train, X_test = X[:split], X[split:]
#     Y_train, Y_test = Y[:split], Y[split:]
#
#     return X_train, Y_train, X_test, Y_test, scaler
def prepare_data(df, history_points=60):
    # Schritt 0: Bereinigung
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Ersetze inf durch NaN
    df.fillna(df.mean(), inplace=True)  # Ersetze NaN durch den Mittelwert

    # Schritt 1: Normalisierung der Daten
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # Schritt 2: Erstellung von Sequenzen
    X, Y = [], []
    for i in tqdm(range(len(data_scaled) - history_points), total=len(data_scaled) - history_points):
        X.append(data_scaled[i:i + history_points])
        Y.append(data_scaled[i + history_points, 3])  # Verwendung von 'Close' als Vorhersageziel

    X, Y = np.array(X), np.array(Y)

    # Schritt 3: Aufteilung in Trainings- und Testsets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    return X_train, Y_train, X_test, Y_test, scaler


if __name__ == "__main__":
    db = "trading_bot"


    """
    1d: Tägliche Daten
    1wk: Wöchentliche Daten
    1mo: Monatliche Daten
    1h: Stündliche Daten (nur für die letzten 60 Tage verfügbar)
    30m: Daten alle 30 Minuten (nur für die letzten 60 Tage verfügbar)
    15m: Daten alle 15 Minuten (nur für die letzten 60 Tage verfügbar)
    5m: Daten alle 5 Minuten (nur für die letzten 60 Tage verfügbar)
    1m: Daten jede Minute (nur für die letzten 7 Tage verfügbar)
    """

    # Definiere das Intervall für die Datenabfrage
    # interval = '1m'
    interval = '5m'
    # interval = '15m'
    # interval = '30m'
    # interval = '1h'
    # interval = '1d'
    # interval = '1wk'
    # interval = '1mo'

    # last_n_intervals = 13
    last_n_intervals = 1000
    # last_n_intervals = None

    # Setze den Ticker
    # ticker_symbol = 'BOSCHLTD.BO'
    # ticker_symbol = 'TSLA'
    ticker_symbol = 'NQ=F'


    start_date, end_date = calculate_start_end_dates(interval)

    data = download_data_with_adjustment(ticker_symbol, start_date, end_date, interval)

    print(data.head(60))
    exit()

    if data is not None and not data.empty:
        data_adjusted = adjust_timezone(data)

        data_adjusted = add_calculated_metrics(data_adjusted)
        # Zeige die ersten Zeilen der heruntergeladenen Daten an
        # print(data_adjusted)
        print(f'data_adjusted_len:{len(data_adjusted)}')
        print(f'days:{len(data_adjusted)/13}')  # für 30m

        columns_to_plot = [
            'Open',
            'High',
            'Low',
            'Close',
            # 'Adj Close',
            # 'Volume',
            # 'Price Change',
            # 'Price Change %',
            # 'Intra-Minute Volatility',
            # 'Volume Change',
            # 'Volume Change %',
            # 'SMA 5 Close',
            # 'SMA 20 Close',
            # 'SMA 5 Volume'
        ]
        plot_stock_data_full(data_adjusted, ticker_symbol, columns_to_plot, last_n_intervals=last_n_intervals)

        # X_train, Y_train, X_test, Y_test, scaler = prepare_data(data_adjusted, history_points=60)
        # print(X_train, Y_train, X_test, Y_test, scaler)

        # trading_bot.save_to_db(dataframe=data_adjusted, to_table=f'{ticker_symbol}_{interval}', db=db)

    else:
        print("Keine Daten abgerufen.")



    # 4. Graph aus NN Vorhersagen
    ####################################################################################################################
    # database_name = "trading_bot"
    # database_name_optuna = "bot_yf_v3"  # nlayers 1-1
    #
    # table_name = f'{ticker_symbol}_{interval}'
    # workers = 8
    # nn_training.train_model_v3(n_trials=workers * 200, n_jobs=workers, database_name=database_name, database_name_optuna=database_name_optuna, table_name=table_name, show_progression=False, tune_with_best=False, load_preprocess_data=False, verbose=True)
    # nn_training.build_model_v3(database_name=database_name_v3, table_name=table_name, model_name=database_name_v3, show_progression=True)
    ####################################################################################################################
