import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import os


def load_and_analyze(filename):
    # Lade die Excel-Datei in ein DataFrame
    df = pd.read_excel(filename)

    # Berechne die Korrelation jeder Spalte mit 'percentage_change'
    correlation = df.corr()['percentage_change'].drop('percentage_change')
    print("Korrelationen der Spalten mit 'percentage_change':")
    print(correlation.sort_values(ascending=False))

    # Plotte die Korrelationen
    correlation.sort_values().plot(kind='barh', title='Einfluss der Spalten auf percentage_change')
    plt.xlabel('Korrelationskoeffizient')
    plt.show()

    # Berechne die beste Kombination von Werten
    best_value = None
    best_combination = None
    for length in range(1, len(df.columns)):
        for combo in combinations([col for col in df.columns if col != 'percentage_change'], length):
            subset = df[list(combo) + ['percentage_change']]
            filtered = subset.dropna()
            if not filtered.empty:
                current_value = filtered['percentage_change'].max()
                if best_value is None or current_value > best_value:
                    best_value = current_value
                    best_combination = combo

    print("Beste Kombination von Spaltenwerten führt zu:")
    print(f"Wert: {best_value}")
    print("Spalten:", best_combination)

    # Optional: Anzeigen der Zeilen mit dem besten Wert
    best_rows = df[df['percentage_change'] == best_value]
    print("Zeilen mit dem höchsten 'percentage_change'-Wert:")
    print(best_rows)


if __name__ == "__main__":

    # Verwenden der Funktion
    script_dir = os.path.dirname(__file__)  # Pfad zum Skriptverzeichnis
    filename = os.path.join(script_dir, 'data_df_study.xlsx')  # Name der Excel-Datei
    load_and_analyze(filename)
