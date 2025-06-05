import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
def compare_stock_data(db, table1, table2, detailed=True, columns=None, conditions=None, engine_kwargs={}, query_kwargs={}):
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

    # Laden der DataFrames
    df1 = query_database(db, table1, columns, conditions, engine_kwargs, query_kwargs)
    df2 = query_database(db, table2, columns, conditions, engine_kwargs, query_kwargs)

    # Sicherstellen, dass die DataFrames dieselben Primärschlüssel haben
    if not set(['Date', 'Time']).issubset(df1.columns) or not set(['Date', 'Time']).issubset(df2.columns):
        raise ValueError("Beide Tabellen müssen 'Date' und 'Time' Spalten enthalten.")

    df1.set_index(['Date', 'Time'], inplace=True)
    df2.set_index(['Date', 'Time'], inplace=True)

    # Ausrichten der Indizes, um sicherzustellen, dass sie identisch sind
    df1, df2 = df1.align(df2, join='inner', axis=0)

    # Vergleich der DataFrames
    comparison_result = {
        "shape_match": df1.shape == df2.shape,
        "dtypes_match": df1.dtypes.equals(df2.dtypes),
        "value_differences": None
    }

    # Vergleichen der Datentypen und der Werte
    if comparison_result["dtypes_match"]:
        differences = []
        for col in df1.columns:
            mask = (df1[col] != df2[col]) | (df1[col].isna() != df2[col].isna())
            if mask.any():
                if detailed:
                    diff_details = pd.DataFrame({
                        'Date': df1.index.get_level_values('Date')[mask],
                        'Time': df1.index.get_level_values('Time')[mask],
                        f'{col}_table1': df1[col][mask],
                        f'{col}_table2': df2[col][mask]
                    })
                    # Berechnung der Differenz und Hinzufügen einer Spalte für die totale Differenz
                    diff_details['Total Difference'] = (diff_details[f'{col}_table1'] - diff_details[f'{col}_table2']).abs()
                    differences.append((col, diff_details))
                else:
                    differences.append(col)

        comparison_result["value_differences"] = differences

    return comparison_result

def save_differences_to_excel(differences, filename="differences.xlsx"):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for col, diff in differences:
            diff.to_excel(writer, sheet_name=col, index=False)





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

def plot_stock_data_comparison(db, table1, table2, start_date, end_date):
    # Bedingung für die Datumsspanne
    conditions = f"Date >= '{start_date}' AND Date <= '{end_date}'"

    # Laden der DataFrames
    df1 = query_database(db, table1, columns=['Date', 'Time', 'Close'], conditions=conditions)
    df2 = query_database(db, table2, columns=['Date', 'Time', 'Close'], conditions=conditions)

    # Konvertieren der 'Date'-Spalte in datetime-Format
    df1['Date'] = pd.to_datetime(df1['Date'], format='%m/%d/%Y')
    df2['Date'] = pd.to_datetime(df2['Date'], format='%m/%d/%Y')

    # Konvertieren der 'Time'-Spalte in Zeit-Format
    df1['Time'] = pd.to_datetime(df1['Time'], format='%H:%M').dt.time
    df2['Time'] = pd.to_datetime(df2['Time'], format='%H:%M').dt.time

    # Kombinieren von 'Date' und 'Time' zu einem datetime-Index
    df1['DateTime'] = pd.to_datetime(df1['Date'].astype(str) + ' ' + df1['Time'].astype(str))
    df2['DateTime'] = pd.to_datetime(df2['Date'].astype(str) + ' ' + df2['Time'].astype(str))

    df1.set_index('DateTime', inplace=True)
    df2.set_index('DateTime', inplace=True)

    # Ausrichten der Indizes, um sicherzustellen, dass sie identisch sind
    df1, df2 = df1.align(df2, join='inner', axis=0)

    # Plotten der Close Kurse
    plt.figure(figsize=(14, 7))
    plt.plot(df1.index, df1['Close'], label=f'{table1} Close')
    plt.plot(df2.index, df2['Close'], label=f'{table2} Close', linestyle='--')
    plt.xlabel('DateTime')
    plt.ylabel('Close Price')
    plt.title('Stock Close Price Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


    

if __name__ == "__main__":
    # Nutzung der Funktion
    db = "trading_bot"
    table1 = 'nasdq_5min'
    table2 = 'nasdq_5min_5y'

    # # Detaillierter Vergleich
    # detailed_result = compare_stock_data(db, table1, table2, detailed=True)
    # print("Detaillierter Vergleich:")
    # if detailed_result["shape_match"] and detailed_result["dtypes_match"]:
    #     if detailed_result["value_differences"]:
    #         for col, diff in detailed_result["value_differences"]:
    #             print(f"Unterschiede in Spalte: {col}")
    #             print(diff)
    #         save_differences_to_excel(detailed_result["value_differences"])
    #     else:
    #         print("Keine Unterschiede in den Werten gefunden.")
    # else:
    #     print("Die Form oder die Datentypen der DataFrames stimmen nicht überein.")
    #     print(f"Shape match: {detailed_result['shape_match']}")
    #     print(f"Dtypes match: {detailed_result['dtypes_match']}")
    #
    # # Zusammenfassender Vergleich
    # summary_result = compare_stock_data(db, table1, table2, detailed=False)
    # print("\nZusammenfassender Vergleich:")
    # if summary_result["shape_match"] and summary_result["dtypes_match"]:
    #     if summary_result["value_differences"]:
    #         print("Unterschiede gefunden in den Spalten:", summary_result["value_differences"])
    #     else:
    #         print("Keine Unterschiede in den Werten gefunden.")
    # else:
    #     print("Die Form oder die Datentypen der DataFrames stimmen nicht überein.")
    #     print(f"Shape match: {summary_result['shape_match']}")
    #     print(f"Dtypes match: {summary_result['dtypes_match']}")

    start_date = '01/03/2023'
    end_date = '25/03/2023'

    plot_stock_data_comparison(db, table1, table2, start_date, end_date)
