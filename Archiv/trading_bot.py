import traceback

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy
from tqdm import tqdm
import tensorflow as tf
import os
import nn_trading_bot_v1 as nn_training
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

def query_database(db, table_name, conditions=None, engine_kwargs={}, query_kwargs={}):

    # Erstellen der Verbindungs-URL für die Datenbank
    connection_url = f"mysql+mysqlconnector://root:@localhost/{db}"

    # Erstellen eines SQLAlchemy Engine-Objekts
    engine = sqlalchemy.create_engine(connection_url, **engine_kwargs)

    # Aufbauen der Abfrage
    query = f"SELECT * FROM {table_name}"
    if conditions:
        query += f" WHERE {conditions}"

    # Ausführen der Abfrage und Schließen der Verbindung
    with engine.connect() as conn:
        dataframe = pd.read_sql(query, con=conn, **query_kwargs)

    return dataframe

def berechne_sinuskurve(punkte, schrittweite):

    x_werte = np.arange(0, punkte * schrittweite, schrittweite)
    y_werte = np.sin(x_werte)

    return x_werte, y_werte


def zeichne_kurve(x_werte, y_werte):

    plt.figure(figsize=(10, 6)) # Setzt die Größe des Diagramms
    plt.plot(x_werte, y_werte, label='Sinuskurve') # Zeichnet die Kurve
    plt.title('Sinuskurve') # Setzt den Titel des Diagramms
    plt.xlabel('X-Achse') # Beschriftung der X-Achse
    plt.ylabel('Y-Achse') # Beschriftung der Y-Achse
    plt.grid(True) # Zeigt das Gitternetz
    plt.legend() # Zeigt die Legende
    plt.show() # Zeigt das Diagramm an


def save_to_db(dataframe, to_table, db):

    # Erstellt eine Verbindung zur Datenbank
    engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://root:@localhost/{db}', echo=False)
    conn = engine.connect()
    dataframe.to_sql(con=conn, name=f'{to_table}', if_exists='replace', index=False)
    conn.close()
    print(f"Daten erfolgreich in Tabelle '{to_table}' gespeichert.")


def transformiere_daten(shift_size=0, df=None, anzahl_y=None, sprung=None):

    neue_zeilen = []

    for start in range(shift_size, len(df) - anzahl_y - sprung + 1):
        # Extrahiere die aktuelle Sequenz von y-Werten
        aktuelle_sequenz = df.iloc[start:start + anzahl_y]['y'].values
        # Berechne die Differenzen innerhalb der Sequenz
        differenzen = np.diff(aktuelle_sequenz).tolist()
        # Berechne die zusätzliche Differenz
        if start + anzahl_y + sprung - 1 < len(df):
            zusatz_diff = df.iloc[start + anzahl_y + sprung - 1]['y'] - df.iloc[start + anzahl_y - 1]['y']
        else:
            zusatz_diff = np.nan  # Falls kein weiterer Wert existiert
        # Füge die zusätzliche Differenz zu den Differenzen hinzu
        differenzen.append(zusatz_diff)
        # Füge die berechnete Zeile zu den neuen Zeilen hinzu
        neue_zeilen.append(differenzen)

    # Erstelle ein neues DataFrame aus den neuen Zeilen
    neues_df = pd.DataFrame(neue_zeilen)

    # Benenne die Spalten um
    spalten_namen = ['input_' + str(i) for i in range(neues_df.shape[1] - 1)] + ['target_value']
    neues_df.columns = spalten_namen

    return neues_df


def berechne_y_differenzen(df):
    df['target_value'] = df['y'].diff()

    return df


def zeichne_y_diff_graf(df, schrittgroesse=0.01):
    """
    Zeichnet einen Graphen basierend auf den Differenzen der 'y'-Werte,
    die in der Spalte 'y_diff' eines DataFrames gegeben sind.

    Parameters:
    df (pd.DataFrame): Der DataFrame, der die 'y_diff'-Werte enthält.
    """
    # Berechne die kumulative Summe der y-Differenzen, um die y-Werte zu rekonstruieren.
    # Wir starten mit dem ersten 'y'-Wert aus dem DataFrame als Ausgangspunkt.
    df['target_value'].iloc[0] = 0

    y_rekonstruiert = df['target_value'].cumsum()

    # Setze den ersten y-Wert auf den tatsächlichen ersten 'y'-Wert aus dem DataFrame,
    # da die erste Differenz NaN ist und die kumulative Summe mit dem zweiten 'y'-Wert beginnt.
    # y_rekonstruiert.iloc[0] = df['y'].iloc[0]
    # y_rekonstruiert.iloc[0] = 0



    x_werte = np.arange(0, schrittgroesse * len(y_rekonstruiert), schrittgroesse)
    x_werte = x_werte[:len(y_rekonstruiert)]



    # Zeichne den Graphen
    plt.figure(figsize=(10, 6))
    # plt.plot(df['x'], y_rekonstruiert, marker='.', linestyle='-', color='blue')
    plt.plot(x_werte, y_rekonstruiert, marker='.', linestyle='-', color='blue')

    plt.title('Graph der rekonstruierten y-Werte basierend auf y_diff')
    plt.xlabel('x')
    plt.ylabel('Rekonstruierte y-Werte')
    plt.grid(True)
    plt.show()

    # print(df['target_value'])
    # print(y_rekonstruiert)


def df_to_dataset_predict(dataframe, mode=None):

    dataframe = pd.DataFrame([dataframe])

    df = dataframe.copy()
    # y_target_data = "target_value"
    # labels = df.pop(y_target_data)
    # sample_weight = df.pop(sample_weight_column)
    # print(dataframe)

    # print(dataframe)

    df_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in df.items()}
    # print(df_dict)
    # ds = tf.data.Dataset.from_tensor_slices((df_dict, labels, sample_weight))
    ds = tf.data.Dataset.from_tensor_slices(df_dict)

    ds = ds.prefetch(1)

    return ds




def predict_v3(db, from_table, model_name=None):

    os.chdir(os.path.dirname(__file__))

    model_path = f"saved_models/nn_model_{model_name}"
    model = tf.keras.models.load_model(model_path)

    sinuswerte_as_dataset = query_database(db=db, table_name=from_table, conditions='')

    # sinuswerte_as_dataset = sinuswerte_as_dataset[-100:]

    future_value_list = []
    for index, row in tqdm(sinuswerte_as_dataset.iterrows(), total=(len(sinuswerte_as_dataset))):

        try:
            ds = df_to_dataset_predict(row)
            future_value = model.predict(ds, verbose=0)
            future_value = future_value[0][0]
            future_value_dict = {'x': index, 'y': row["target_value"], 'target_value': future_value}
            future_value_list.append(future_value_dict)

        except:
            print(traceback.format_exc())
            exit()

    future_value_df = pd.DataFrame(future_value_list)

    return future_value_df


def berechne_schritte(schrittgroesse):
    """
    Berechnet die Anzahl der Schritte, die erforderlich sind, um genau einen Zyklus
    einer Sinuswelle mit einer gegebenen Schrittgröße zu durchlaufen.
    """
    return math.ceil(2 * math.pi / schrittgroesse)


if __name__ == "__main__":

    db = "trading_bot"
    sinuswerte_tabelle = "sinuskurvenwerte"
    sinuswerte_tabelle_erweitert = "sinuskurvenwerte_erweitert"
    sinuswerte_as_dataset = "sinuskurvenwerte_datensaetze"
    sinuswerte_as_dataset_shifted = f'{sinuswerte_as_dataset}_shifted'

    sinus_x_schrittgroesse = 0.01
    sinus_x_schritte = 2 * berechne_schritte(sinus_x_schrittgroesse)  # bot_v2/3
    # sinus_x_schritte = 2000 * berechne_schritte(sinus_x_schrittgroesse)

    y_werte_pro_datensatz = 100
    # x_distanz_zum_zielwert = 5  # bot_v2
    x_distanz_zum_zielwert = 1  # bot_v3


    # 1. Koordinaten einer Sinuskurve berechnen
    ####################################################################################################################
    x, y = berechne_sinuskurve(sinus_x_schritte, sinus_x_schrittgroesse)
    df_sinus = pd.DataFrame({'x': x, 'y': y})
    zeichne_kurve(x, y)
    save_to_db(dataframe=df_sinus, to_table=sinuswerte_tabelle, db=db)
    exit()
    ####################################################################################################################



    # 2. Graph aus Differenzwerten der Sinuswertetabelle
    ####################################################################################################################
    y_koordinaten = query_database(db=db, table_name=sinuswerte_tabelle, conditions='')
    y_koordinaten_differenzen = berechne_y_differenzen(y_koordinaten)
    save_to_db(dataframe=y_koordinaten_differenzen, to_table=sinuswerte_tabelle_erweitert, db=db)
    zeichne_y_diff_graf(y_koordinaten_differenzen)
    ####################################################################################################################



    # 3. Datensätze aus Koordinaten bilden und zeichnen
    ####################################################################################################################
    sinuswerte = query_database(db=db, table_name=sinuswerte_tabelle, conditions='')
    sinuswerte_dataset = transformiere_daten(shift_size=0, df=sinuswerte, anzahl_y=y_werte_pro_datensatz, sprung=x_distanz_zum_zielwert)
    save_to_db(dataframe=sinuswerte_dataset, to_table=sinuswerte_as_dataset, db=db)

    # Zeichnen
    sinuswerte_dataset = query_database(db=db, table_name=sinuswerte_as_dataset, conditions='')
    zeichne_y_diff_graf(sinuswerte_dataset, schrittgroesse=sinus_x_schrittgroesse)
    ####################################################################################################################



    # 4. Datensätze verschoben aus Koordinaten bilden und zeichnen
    ####################################################################################################################
    sinuswerte = query_database(db=db, table_name=sinuswerte_tabelle, conditions='')
    sinuswerte_dataset_shifted = transformiere_daten(shift_size=50, df=sinuswerte, anzahl_y=y_werte_pro_datensatz, sprung=x_distanz_zum_zielwert)
    save_to_db(dataframe=sinuswerte_dataset_shifted, to_table=sinuswerte_as_dataset_shifted, db=db)

    # Zeichnen
    sinuswerte_dataset_shifted = query_database(db=db, table_name=sinuswerte_as_dataset_shifted, conditions='')
    zeichne_y_diff_graf(sinuswerte_dataset_shifted, schrittgroesse=sinus_x_schrittgroesse)
    ####################################################################################################################


    # 4. Graph aus NN Vorhersagen
    ####################################################################################################################
    # database_name_v3 = f"db_test{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # database_name_v3 = "bot_v1"  # nlayers 1-6
    # database_name_v3 = "bot_v2"  # nlayers 1-3
    database_name_v3 = "bot_v3"  # nlayers 1-1

    table_name = ""
    workers = 32
    nn_training.train_model_v3(n_trials=workers*10, n_jobs=workers, database_name=database_name_v3, table_name=table_name, show_progression=False, tune_with_best=False, load_preprocess_data=False, verbose=True)
    nn_training.build_model_v3(database_name=database_name_v3, table_name=table_name, model_name=database_name_v3, show_progression=True)


    # sinuswerte_differenzen = predict_v3(db=db, from_table=sinuswerte_as_dataset, model_name="bot_v3")
    sinuswerte_differenzen = predict_v3(db=db, from_table=sinuswerte_as_dataset_shifted, model_name="bot_v3")

    zeichne_y_diff_graf(sinuswerte_differenzen)
    ####################################################################################################################














