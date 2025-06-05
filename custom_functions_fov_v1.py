# ======================================================================================================================
# Info
# ======================================================================================================================
# Date: 14.06.2022
# Name:
# Aufgabe:
# Version: V0.0
# Author: Niklas Fliegel
# ======================================================================================================================
# libraries
# ======================================================================================================================
import sys
import os
from pathlib import Path
import difflib
from tqdm.auto import tqdm
import datetime
import re
import sqlalchemy
from collections import Counter
import numpy as np
import pandas as pd
import json
from phpserialize import unserialize
from phpserialize import phpobject
import numbers
import mysql.connector
from sqlalchemy import create_engine

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import pyautogui
import time
import requests
import pygetwindow as gw
import win32gui
import win32con
import subprocess

tqdm.pandas()

import smtplib
from email.mime.text import MIMEText
# ======================================================================================================================
# Custom_functions
# ======================================================================================================================


class TextColor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def apply_return_str(s, column, value):
    if isnan(s[column]) or \
            s[column] == "None" or \
            s[column] == "NaN" or \
            s[column] == "nan" or \
            s[column] == "" or \
            s[column] == " " or \
            s[column] == "NULL" or \
            pd.isna(s[column]) or \
            pd.isnull(s[column]) or \
            s[column] == float("nan"):

        # print(f'true= {s[column]}')
        return value
    else:
        return s[column]


def return_str(value, string):
    if isnan(value) or \
            value == "None" or \
            value == "NaN" or \
            value == "nan" or \
            value == "" or \
            value == " " or \
            value == "NULL" or \
            pd.isna(value) or \
            pd.isnull(value) or \
            value == float("nan"):

        # print(f'true= {value}')
        return string
    else:
        return value


def apply_opt(s, column, case):
    if s[column]:
        # Umwandeln der Umlaute in ihre zweibuchstabigen Äquivalente
        cleaned_string = str(s[column])
        replacements = {
            'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
            'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
            'ß': 'ss'
        }
        for k, v in replacements.items():
            cleaned_string = cleaned_string.replace(k, v)

        # Entfernt Leerzeichen und wendet Groß-/Kleinschreibung an
        cleaned_string = cleaned_string.replace(" ", "").lower() if case == "lower" else cleaned_string.replace(" ", "").upper()

        # Entfernt alles außer Buchstaben, Zahlen und Punkten
        cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', cleaned_string)

        return cleaned_string
    else:
        return s[column]


# def apply_opt_v3(s, column):
#     if s[column]:
#         # Umwandeln der Umlaute in ihre zweibuchstabigen Äquivalente
#         cleaned_string = str(s[column])
#         replacements = {
#             'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
#             'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
#             'ß': 'ss'
#         }
#
#         for k, v in replacements.items():
#             cleaned_string = cleaned_string.replace(k, v)
#
#         cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', cleaned_string).lower()
#         cleaned_string = re.sub(re.escape(s["marke_opt"]), '', cleaned_string, flags=re.IGNORECASE)
#
#         return cleaned_string
#     else:
#         return s[column]


def apply_opt_v3(s, column):
    if s[column]:
        # Umwandeln der Umlaute in ihre zweibuchstabigen Äquivalente
        cleaned_string = str(s[column])
        replacements = {
            'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
            'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
            'ß': 'ss'
        }

        for k, v in replacements.items():
            cleaned_string = cleaned_string.replace(k, v)

        cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', cleaned_string).lower()

        if not "marke" in column:
            cleaned_string = re.sub(re.escape(s["marke_opt"]), '', cleaned_string, flags=re.IGNORECASE)

        return cleaned_string
    else:
        return s[column]




def apply_abweichung(s, c1, user_eingabe):
    try:
        if isnan(user_eingabe) or user_eingabe == "None" or user_eingabe == "NaN" or user_eingabe == "nan" or user_eingabe == "" or user_eingabe == "NULL"  \
                or isnan(s[c1]) or s[c1] == "None" or s[c1] == "NaN" or s[c1] == "nan" or s[c1] == "" or s[c1] == "NULL":
            abweichung = 0
            return abweichung
        else:
            abweichung = abs(float(s[c1]) - float(user_eingabe))
            return abweichung
    except:
        abweichung = 0
        return abweichung


def apply_string_fragments(s, column, user_eingabe):
    fragments = 0
    if not isnan(s[column]) and not isnan(user_eingabe):
        user_eingabe = str(user_eingabe).upper().replace(" ", "")
        katalog = str(s[column]).upper()

        for char in range(len(user_eingabe)):
            end = char
            while end <= len(user_eingabe):
                check_string = user_eingabe[char:end]
                if check_string in katalog:
                    fragments += 1
                end += 1

    return fragments


def apply_number_matches(s, column, modell):
    matches = 0
    if not isnan(s[column]) and not isnan(modell):
        user_eingabe = re.findall(r'\d+', modell)
        katalog = re.findall(r'\d+', s[column])

        for number in user_eingabe:
            if number in katalog:
                matches += 1

    return matches


def apply_backcheck_sections(s, column, user_eingabe):
    matches = 0
    if not isnan(s[column]) and not isnan(user_eingabe):
        user_eingabe = str(user_eingabe).upper().replace(" ", "")
        katalog = str(s[column]).upper()
        katalog = re.split('[;, -]', katalog)

        for word in katalog:
            if word in user_eingabe:
                matches += 1

    return matches


def apply_len_diff(s, modell):
    diff = float("nan")
    if not isnan(s['modell_opt']) and not isnan(modell):
        user_eingabe = modell
        katalog = str(s['modell_opt'])
        diff = abs(len(katalog) - len(user_eingabe))
    return diff


def apply_rating(s):
    if isnan(s['modell_ermittelt_qualität']) or isnan(s['abw_leistung']) or isnan(s['abw_hubraum']) or isnan(s['abw_baujahr']) or isnan(s['datenqualitaet']):
        return float("nan")
    else:
        rating = (s['modell_ermittelt_qualität'] -
                  (s['abw_leistung'] / 10) -
                  (s['abw_hubraum'] / 10) -
                  (s['abw_baujahr'] * 10))
        return rating

    # Datenqualität passt dort nicht rein weil es hierbei nicht um die gesamtqualität geht sondern darum wie gut das modell ermittelt wurde



def result_to_mysql(dataframe, database, table):
    engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://root:@localhost/{database}', echo=False)
    dataframe.to_sql(con=engine, name=f'{table}', if_exists='replace', index=False)


def apply_str_to_int(s, c1):
    if s[c1] and not isnan(s[c1]):
        return int(s[c1])
    else:
        return 0


def apply_convert_to_number(s, c1):
    if s[c1] and not isnan(s[c1]):
        number = str(s[c1])
        if "." in number:
            decimals = len(number) - number.find(".") - 1
            if decimals == 0:
                number = int(number)
            elif decimals == 1:
                number = int(number.split(".", 1)[0])
            elif decimals == 2:
                number = int(number.split(".", 1)[0])
            elif decimals >= 3:
                number = int(number.replace(".", ""))
        elif "," in number:
            decimals = len(number) - number.find(",") - 1
            if decimals == 0:
                number = int(number)
            elif decimals == 1:
                number = int(number.split(",", 1)[0])
            elif decimals == 2:
                number = int(number.split(",", 1)[0])
            elif decimals >= 3:
                number = int(number.replace(",", ""))

        else:
            number = float(number)
    else:
        number = float("nan")
    return number


def average(lst):
    if len(lst) == 0:
        return
    elif len(lst) > 0:
        res = 0.0
        n = 0
        for i in lst:
            if i > 100:
                n += 1
                res = res + i
        if n == 0:
            return
        else:
            res = res / n
            return res


def apply_date_finder1(s, c1):
    date_to_find = float("nan")
    if s['date_date'] != 0 and s['date_date'] != 0.0 and not isnan(s['date_date']):
        year_now = s['date_date'].year

        try:
            if s[c1] != "" and s[c1] != " " and not isnan(s[c1]):
                date_to_find = str(s[c1])

                date_to_find = re.sub(r"[^\d.]", "", date_to_find)

                if date_to_find != "" and date_to_find != " " and not isnan(date_to_find):

                    if "." not in date_to_find[-4:] and date_to_find[-4:].isdigit():
                        date_to_find = int(date_to_find[-4:])
                    elif "." not in date_to_find[-2:] and date_to_find[-2:].isdigit():
                        date_to_find = date_to_find[-2:]
                        up_dist = abs(100 - int(date_to_find))
                        dw_dist = abs(00 - int(date_to_find))
                        if up_dist < dw_dist:
                            date_to_find = int("19" + str(date_to_find))
                        elif up_dist > dw_dist:
                            date_to_find = int("20" + str(date_to_find))
                        elif up_dist == dw_dist:
                            date_to_find = int("19" + str(date_to_find))
                    elif "." in date_to_find[-2:] and date_to_find[-1:].isdigit():
                        date_to_find = int("20" + str(date_to_find)[-1:])
                    elif "." in date_to_find[-1:]:
                        date_to_find = float("nan")
                    elif "." not in date_to_find and date_to_find[-1:].isdigit():
                        date_to_find = int("20" + str(date_to_find)[-1:])

                    if date_to_find != "" and date_to_find != " " and not isnan(date_to_find):
                        date_to_find = int(date_to_find)
                        if int(date_to_find) < 1900 and int(date_to_find) < 0:
                            date_to_find = str(date_to_find)[-2:]
                            up_dist = abs(100 - int(date_to_find))
                            dw_dist = abs(00 - int(date_to_find))
                            if up_dist < dw_dist:
                                date_to_find = int("19" + str(date_to_find))
                            elif up_dist > dw_dist:
                                date_to_find = int("20" + str(date_to_find))
                            elif up_dist == dw_dist:
                                date_to_find = int("19" + str(date_to_find))

                        if date_to_find > year_now:
                            date_to_find = int("20" + str(date_to_find)[-2:])

                        if date_to_find < 1945:
                            date_to_find = int("20" + str(date_to_find)[-2:])

                        if date_to_find > year_now:
                            date_to_find = int("19" + str(date_to_find)[-2:])
                            if date_to_find < 1945:
                                date_to_find = float("nan")
                    else:
                        date_to_find = float("nan")
                else:
                    date_to_find = float("nan")

        except:
            date_to_find = float("nan")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

        if date_to_find == "" or date_to_find == " " or isnan(date_to_find):
            date_to_find = float("nan")

    else:
        date_to_find = float("nan")

    return date_to_find


def apply_date_finder2(s, c1):
    if s[c1]:
        date = s[c1]

        if date != "" and date != " " and not isnan(date):
            try:
                date = datetime.datetime.strptime(date, "%d.%m.%Y").date()
            except:
                try:
                    date = datetime.datetime.strptime(date, "%d.%m.%y").date()
                except:
                    try:
                        date = datetime.datetime.strptime(date, "%m/%d/%Y").date()
                    except:
                        try:
                            date = re.sub(r"\D.", "", date)
                            date = datetime.datetime.strptime(date, "%d.%m.%Y").date()
                        except:
                            try:
                                date = re.sub(r"\D.", "", date)
                                date = datetime.datetime.strptime(date, "%d.%m.%y").date()
                            except:
                                try:
                                    date = re.sub(r"\D.", "", date)
                                    date = datetime.datetime.strptime(date, "%Y").date()
                                except:
                                    try:
                                        if s['erstzulassung_jahr'] and not isnan(s['erstzulassung_jahr']):
                                            date = datetime.datetime.strptime(s['erstzulassung_jahr'], "%Y").date()
                                        else:
                                            date = float("nan")

                                    except:
                                        date = float("nan")
                                        exc_type, exc_obj, exc_tb = sys.exc_info()
                                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                        print(exc_type, fname, exc_tb.tb_lineno)
            try:
                if date.year < 1900:
                    try:
                        if s['erstzulassung_jahr']:
                            date = datetime.datetime.strptime(s['erstzulassung_jahr'], "%Y").date()
                        else:
                            date = float("nan")

                    except:
                        date = float("nan")
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
            except:
                pass
        else:
            date = float("nan")
    else:
        date = float("nan")
    return date


def apply_date_auswahl(s):
    auswahl = float("nan")
    limit = 1900
    if s['date_date'] != 0 and s['date_date'] != 0.0 and not isnan(s['date_date']):
        year_inserat = s['date_date'].year

        if isnan(s['erstzulassung_bereinigt']):
            erstzulassung_bereinigt_jahr = 0
        else:
            erstzulassung_bereinigt_jahr = s['erstzulassung_bereinigt'].year
            if s['erstzulassung_bereinigt'] > s['date_date']:
                erstzulassung_bereinigt_jahr = 0

        if isnan(s['baujahr_jahr_bereinigt']):
            baujahr_jahr_bereinigt = 0
        else:
            baujahr_jahr_bereinigt = int(s['baujahr_jahr_bereinigt'])
            if baujahr_jahr_bereinigt > year_inserat:
                baujahr_jahr_bereinigt = 0

        if baujahr_jahr_bereinigt == erstzulassung_bereinigt_jahr and erstzulassung_bereinigt_jahr != 0:
            auswahl = baujahr_jahr_bereinigt
        elif baujahr_jahr_bereinigt == 0 and erstzulassung_bereinigt_jahr == 0:
            auswahl = float("nan")
        elif baujahr_jahr_bereinigt == 0 and erstzulassung_bereinigt_jahr != 0:
            if erstzulassung_bereinigt_jahr > limit:
                auswahl = erstzulassung_bereinigt_jahr
            else:
                auswahl = float("nan")
        elif baujahr_jahr_bereinigt != 0 and erstzulassung_bereinigt_jahr == 0:
            if baujahr_jahr_bereinigt > limit:
                auswahl = baujahr_jahr_bereinigt
            else:
                auswahl = float("nan")
        elif baujahr_jahr_bereinigt != 0 and erstzulassung_bereinigt_jahr != 0:
            if erstzulassung_bereinigt_jahr < baujahr_jahr_bereinigt:
                if baujahr_jahr_bereinigt > limit:
                    auswahl = baujahr_jahr_bereinigt
                else:
                    auswahl = float("nan")
            elif erstzulassung_bereinigt_jahr > baujahr_jahr_bereinigt:
                if erstzulassung_bereinigt_jahr > limit:
                    if (erstzulassung_bereinigt_jahr - baujahr_jahr_bereinigt) <= 2:
                        auswahl = erstzulassung_bereinigt_jahr
                    else:
                        if baujahr_jahr_bereinigt > limit:
                            auswahl = baujahr_jahr_bereinigt
                        else:
                            auswahl = float("nan")
                else:
                    auswahl = float("nan")
    else:
        auswahl = float("nan")
        # auswahl = min(filter(lambda x: x > 1945, [s[c1], s[c2], erstzulassung_jahr]), default=None)

    return auswahl


def apply_clean_laufleistung(s, c1, c2):
    try:
        if s[c1] != "" and not isnan(s[c1]):
            laufleistung = str(s[c1])
            if "." in laufleistung:
                decimals = len(laufleistung) - laufleistung.find(".") - 1
                laufleistung = re.sub(r"[^\d.]", "", laufleistung)

                if decimals == 0:
                    laufleistung = int(laufleistung)
                elif decimals == 1:
                    laufleistung = int(laufleistung.split(".", 1)[0])
                elif decimals == 2:
                    laufleistung = int(laufleistung.split(".", 1)[0])
                elif decimals >= 3:
                    laufleistung = int(laufleistung.replace(".", ""))
            else:
                laufleistung = re.sub(r"\D", "", laufleistung)
                if laufleistung:
                    laufleistung = int(laufleistung)
                else:
                    laufleistung = float("nan")
        else:
            laufleistung = float("nan")
    except:
        laufleistung = float("nan")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    if s[c2] != "" and not isnan(s[c2]):
        try:
            if int(s[c2]) > 1000 and int(laufleistung) < 1000:
                if laufleistung != 0:
                    laufleistung = abs(laufleistung)
                    while laufleistung <= int(s[c2]):
                        laufleistung *= 10
                    laufleistung /= 10
                elif laufleistung != "" or isnan(laufleistung):
                    laufleistung = int(s[c2])
                else:
                    laufleistung = s[c2]

            elif int(s[c2]) > 1000 and int(laufleistung) > int(s[c2]):
                while laufleistung > int(s[c2]):
                    laufleistung /= 10
                # laufleistung *= 10

            elif int(s[c2]) == 1000 and int(laufleistung) < int(s[c2]):
                laufleistung = laufleistung

            elif laufleistung != "" or isnan(laufleistung):
                laufleistung = int(s[c2])

        except:
            laufleistung = float("nan")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    return laufleistung


def apply_clean_preisvorstellung(s, c1):
    s[c1] = str(s[c1])

    try:
        if s[c1] != "" and not isnan(s[c1]):
            preisvorstellung = str(s[c1])

            if preisvorstellung != "" and preisvorstellung != " " and preisvorstellung != "nan" and not isnan(preisvorstellung):
                if "." in preisvorstellung:
                    if ".000" in preisvorstellung:
                        preisvorstellung = preisvorstellung.replace(".", "")

                    if ".00" in preisvorstellung:
                        preisvorstellung = preisvorstellung.replace(".00", "")

                    if ".0" in preisvorstellung:
                        preisvorstellung = preisvorstellung.replace(".0", "")

                elif "," in preisvorstellung:
                    if ",000" in preisvorstellung:
                        preisvorstellung = preisvorstellung.replace(",", "")

                    if ",00" in preisvorstellung:
                        preisvorstellung = preisvorstellung.replace(",00", "")

                    if ",0" in preisvorstellung:
                        preisvorstellung = preisvorstellung.replace(",0", "")

                elif ":" in preisvorstellung:
                    if ":000" in preisvorstellung:
                        preisvorstellung = preisvorstellung.replace(":", "")

                    if ":00" in preisvorstellung:
                        preisvorstellung = preisvorstellung.replace(":00", "")

                    if ":0" in preisvorstellung:
                        preisvorstellung = preisvorstellung.replace(":0", "")
                try:
                    if preisvorstellung[::-1].find('.') >= 3:
                        preisvorstellung = preisvorstellung.replace(".", "")
                except:
                    pass
                preisvorstellung = preisvorstellung.replace("Tsd", "000").replace(" ", "")

                preisvorstellung = average(list(map(float, re.findall(r'\d+', preisvorstellung))))
                if (isinstance(preisvorstellung, float) or isinstance(preisvorstellung, int)) and (preisvorstellung < 100 or preisvorstellung > 100000):
                    preisvorstellung = float("nan")
                elif not (isinstance(preisvorstellung, float) or not isinstance(preisvorstellung, int)):
                    preisvorstellung = float("nan")

        else:
            preisvorstellung = float("nan")

    except:
        preisvorstellung = float("nan")
        print(traceback.format_exc())

    if preisvorstellung == "" or preisvorstellung == " " or preisvorstellung == "nan" or isnan(preisvorstellung):
        preisvorstellung = float("nan")

    return preisvorstellung


def apply_clean_leistung(s, c1):
    if s[c1] != "" and not isnan(s[c1]):
        leistung = str(s[c1])
        leistung = re.sub(r"\D.", "", leistung)

        if leistung != "":
            if "." in leistung:
                decimals = leistung[::-1].find('.')
                if decimals == 0:
                    leistung = int(leistung.replace(".", ""))
                elif decimals == 1:
                    leistung = float(leistung)
                elif decimals >= 2:
                    if leistung.split(".", 1)[0] == "0":
                        leistung = float(leistung)
                        while leistung < 1:
                            leistung *= 10
                    else:
                        leistung = float(leistung.replace(".", ""))
            else:
                leistung = re.sub(r"\D", "", leistung)
                if leistung:
                    leistung = float(leistung)
                else:
                    leistung = float("nan")
        else:
            leistung = float("nan")

        return leistung


def apply_clean_hubraum(s, c1):
    if s[c1] != "" and not isnan(s[c1]):
        hubraum = str(s[c1])
        hubraum = re.sub(r"\D.", "", hubraum)

        if hubraum != "":
            if "." in hubraum:
                decimals = hubraum[::-1].find('.')
                if decimals == 1 or decimals == 2:
                    hubraum = float(hubraum)
                elif decimals > 2:
                    hubraum = hubraum.replace(".", "")
                    hubraum = float(hubraum)
            else:
                hubraum = re.sub(r"\D", "", hubraum)
                if hubraum:
                    hubraum = float(hubraum)
                else:
                    hubraum = float("nan")
        else:
            hubraum = float("nan")
        return hubraum


def apply_read_date(s):
    try:
        date = datetime.datetime.strptime((str(s["date"]).split(",", 1)[0]), "%d.%m.%Y").date()
    except:
        try:
            date = datetime.datetime.strptime((str(s["date"]).split(",", 1)[0]), "%m/%d/%Y").date()
        except:
            date = float("nan")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    return date


def apply_check_sh_ratings(s):
    if s["offer"] != "" and s["offer"] != " " and not isnan(s["offer"]):
        return True
    else:
        return False


def apply_check_ek_externalpricesuggestions_wkdm(s):
    value = s["EK_ExternalPriceSuggestions_WKDM"]

    if value != "" and value != " " and not isnan(value):
        EK_ExternalPriceSuggestions_WKDM_auto = 1

        if "{" in value:

            # Suche nach den Positionen des JSON-Strings innerhalb des gesamten Strings
            start_pos = value.find('{')
            end_pos = value.rfind('}') + 1

            # Extrahiere den JSON-String aus dem gesamten String
            json_string = value[start_pos:end_pos]

            # Parsen Sie den inneren JSON-String
            data = json.loads(json_string)

            if "request_success" in data:
                if data["request_success"]:

                    if "modell_ermittelt_qualitaet" in data:
                        if data["modell_ermittelt_qualitaet"] <= 0.5:
                            EK_ExternalPriceSuggestions_WKDM_auto = 0

                    if "abw_hubraum" in data:
                        if data["abw_hubraum"] > 46:
                            EK_ExternalPriceSuggestions_WKDM_auto = 0

                    if "abw_baujahr" in data:
                        if data["abw_baujahr"] >= 2:
                            EK_ExternalPriceSuggestions_WKDM_auto = 0

                    if "modell_preisqualitaet" in data:
                        if data["modell_preisqualitaet"] <= 0.6:
                            EK_ExternalPriceSuggestions_WKDM_auto = 0

                    if "unfall" in data:
                        if data["unfall"] == "ja":
                            EK_ExternalPriceSuggestions_WKDM_auto = 0

                else:
                    EK_ExternalPriceSuggestions_WKDM_auto = 0
            else:
                EK_ExternalPriceSuggestions_WKDM_auto = 0

            return EK_ExternalPriceSuggestions_WKDM_auto

        else:
            return float("nan")
    else:
        return float("nan")


def most_frequent(liste):
    occurence_count = Counter(liste)
    return occurence_count.most_common(1)[0][1]


def apply_calculate_price_quality(s):
    quality = 0
    post_type = 0

    counterproposal = s['counterproposal']
    counterproposal_num = s['counterproposal_num']
    bike_sold = s['EK_WKDM_Sold']
    offer_accepted = s['EK_WKDM']

    scratches = int(s['scratches'])

    umkipper = s['umkipper']
    teschinsche_optische_maengel = s['teschinsche_optische_maengel']

    sh_rated = s['sh_rated']
    user_offer = s['user_offer']
    user_offer_num = s['user_offer_num']

    # Typ 1: Bikes mit Gegenvorschlag --------------------------------------------------------------------------------------------------------------------------
    if counterproposal and not isnan(counterproposal):
        post_type = 1

        # Gegenvorschläge
        if counterproposal_num >= 3:
            quality += 10
        elif counterproposal_num == 2:
            quality += 9
        elif counterproposal_num == 1:
            quality += 7

        # Bike verkauft
        if bike_sold != "" and bike_sold != " " and not isnan(bike_sold) and bike_sold == "1":
            quality += 1


    # Typ 2: Bikes mit SH Bewertung, kein Gegenvorschlag(Überschneidung???) und keine externe Händlerbewertung, Verkauf regulär über Shop ----------------------
    elif sh_rated and not isnan(sh_rated) and (not counterproposal or isnan(counterproposal)) and (not user_offer or isnan(user_offer)):
        post_type = 2

        # VK hat angenommen
        if offer_accepted != "" and offer_accepted != " " and not isnan(offer_accepted) and offer_accepted == "1":
            quality += 1

        # Bike verkauft
        if bike_sold != "" and bike_sold != " " and not isnan(bike_sold) and bike_sold == "1":
            quality += 2


    # Typ 3: Bikes die nur von externen Händlern und nicht SH bewertet wurden  Externes Händlergebot ---------------------------------------------------------
    elif user_offer and not isnan(user_offer) and (not sh_rated or isnan(sh_rated)):
        post_type = 3

        if user_offer_num and not isnan(user_offer_num):
            # Gegenvorschläge
            if user_offer_num >= 3:
                quality += 10
            elif user_offer_num == 2:
                quality += 9
            elif user_offer_num == 1:
                quality += 5


    # Für alle Fälle gleich ------------------------------------------------------------------------------------------------------------------------------------
    if post_type == 1 or post_type == 2 or post_type == 3:

        # Zustand
        if umkipper != "" and umkipper != " " and not isnan(umkipper) and umkipper == "ja":
            quality -= 1
        if scratches != "" and scratches != " " and not isnan(scratches):
            # keine Kratzer
            if scratches == "1":
                quality += 1
            # Kratzer deutlich zu erkennen
            elif scratches == "3":
                quality -= 1
        if teschinsche_optische_maengel != "" and teschinsche_optische_maengel != " " and not isnan(teschinsche_optische_maengel):
            if teschinsche_optische_maengel == "ja":
                quality -= 1


        if quality <= 0:
            if post_type == 1:
                quality = 0.0625
            elif post_type == 2:
                quality = 0.05
            elif post_type == 3:
                quality = 0.0625

        # Punktabzug abhängig vom Alter des Inserates
        try:
            current_date = datetime.datetime.today().date()
            inserat_date = s['date_date']

            if current_date < inserat_date:
                diff = current_date - inserat_date
                if diff.days > 365:
                    quality = quality * 0.75 ** diff
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    return quality


def apply_calculate_final_price(s):
    price = 0

    counterproposal = s['counterproposal']
    counterproposal_num = s['counterproposal_num']
    counterproposals_av = s['counterproposals_av']
    EK_WKDM_Sold_differenzbetrag = s['EK_WKDM_Sold_differenzbetrag']
    vk_offer_number = s['vk_offer_number']

    sh_rated = s['sh_rated']
    user_offer = s['user_offer']

    # Wurde verkauft
    EK_WKDM_Sold = s['EK_WKDM_Sold']
    # Verkäufer hat Angebot angenommen(abgegeben)
    EK_WKDM = s['EK_WKDM']


    # Typ 1: Bikes mit Gegenvorschlag --------------------------------------------------------------------------------------------------------------------------
    if counterproposal and not isnan(counterproposal):
        # Gegenvorschläge
        if counterproposal_num >= 2:
            price = counterproposals_av

        elif counterproposal_num == 1:
            price = counterproposals_av

    # Typ 2: Bikes mit SH Bewertung, kein Gegenvorschlag(Überschneidung???) und keine externe Händlerbewertung, Verkauf regulär über Shop ----------------------
    elif sh_rated and not counterproposal and not user_offer:
        if vk_offer_number != "" and vk_offer_number != " " and vk_offer_number != 0 and not isnan(vk_offer_number) and EK_WKDM_Sold_differenzbetrag != "" and EK_WKDM_Sold_differenzbetrag != " " and not isnan(EK_WKDM_Sold_differenzbetrag):
            price = int(vk_offer_number) - int(EK_WKDM_Sold_differenzbetrag)
        elif vk_offer_number != "" and vk_offer_number != " " and vk_offer_number != 0 and not isnan(vk_offer_number) and isnan(EK_WKDM_Sold_differenzbetrag):
            price = vk_offer_number - 70

        # a. WURDE VERKAUFT Verkauf an Händler durch SH
        # if EK_WKDM_Sold:
        #     price = price

        # b. WURDE NICHT VERKAUFT ABER VK hat ABGEGEBEN --> Preis war zu hoch
        if not EK_WKDM_Sold and EK_WKDM:
            price = price * 0.9

        # c.	WURDE NICHT VERKAUFT WEIL VK NICHT ABGEGEBEN – bis zur automatisierten Preisbestimmung Datum
        # 1.1.1.	Automatisch verschickt  nicht rein – Ampel Grenzwert, Sendefunktion prüfen --> post_type 0
        # 1.1.1.	nicht automatisch verschickt – Manuell  rein
        # if not EK_WKDM:
        #     price = price


    if price < 0:
        price = float("nan")

    return price


def apply_determine_post_type(s):
    post_type = 0

    counterproposal = s['counterproposal']
    sh_rated = s['sh_rated']
    user_offer = s['user_offer']

    abgegeben = s['EK_WKDM']
    # ampel_gruen = s['ampel_gruen']

    # Typ 1: Bikes mit Gegenvorschlag --------------------------------------------------------------------------------------------------------------------------
    if counterproposal and not isnan(counterproposal):
        post_type = 1

    # Typ 2: Bikes mit SH Bewertung, kein Gegenvorschlag und keine externe Händlerbewertung, Verkauf regulär über Shop ----------------------
    elif sh_rated and not isnan(sh_rated) and (not counterproposal or isnan(counterproposal)) and (not user_offer or isnan(user_offer)):
        # if ampel_gruen and not isnan(ampel_gruen) and isnan(abgegeben):
        if isnan(abgegeben) or not abgegeben:
            post_type = 0
        else:
            post_type = 2
        # post_type = 2


    # Typ 3: Bikes die nur von externen Händlern und nicht SH bewertet wurden  Externes Händlergebot ---------------------------------------------------------
    elif user_offer and not isnan(user_offer) and (not sh_rated or isnan(sh_rated)):
        post_type = 3

    return post_type


def apply_convert_to_timestamp(s):
    try:
        date_ = datetime.datetime.strptime(str(s['date_date']), "%Y-%m-%d %H:%M:%S")
        time_stamp = datetime.datetime.timestamp(date_)
    except:
        try:
            date_ = datetime.datetime.strptime(str(s['date_date']), "%Y-%m-%d")
            time_stamp = datetime.datetime.timestamp(date_)
        except:
            time_stamp = float("nan")

    return time_stamp


# def apply_correct_final_price(s, price_column, mode):
#
#     gesamtabzug = 0
#     gesamtabzug_proz = 0.00
#     normalized_price = float("nan")
#
#     if s[price_column] != "" and not isnan(s[price_column]):
#         normalized_price = int(s[price_column])
#
#         # Zustand der Verschleißteile
#         if s['zustand_verschleissteile'] != "" and s['zustand_verschleissteile'] != " " and not isnan(s['zustand_verschleissteile']) and s['zustand_verschleissteile']:
#             zustand_verschleissteile = int(s['zustand_verschleissteile'])
#             if zustand_verschleissteile == 2:
#                 gesamtabzug_proz += 0.01
#             elif zustand_verschleissteile == 3:
#                 gesamtabzug_proz += 0.02
#
#         # Zustand & Alter Reifen
#         if s['zustand_reifen'] != "" and s['zustand_reifen'] != " " and not isnan(s['zustand_reifen']) and s['zustand_reifen']:
#             zustand_reifen = int(s['zustand_reifen'])
#             if zustand_reifen == 2:
#                 if normalized_price < 1500:
#                     gesamtabzug += 25
#                 elif normalized_price >= 1500:
#                     gesamtabzug += 50
#             elif zustand_reifen == 3:
#                 if normalized_price < 1500:
#                     gesamtabzug += 50
#                 elif normalized_price >= 1500:
#                     gesamtabzug += 100
#             elif zustand_reifen == 4:
#                 if normalized_price < 1500:
#                     gesamtabzug += 100
#                 elif normalized_price >= 1500:
#                     gesamtabzug += 200
#             elif zustand_reifen == 5:
#                 if normalized_price < 1500:
#                     gesamtabzug += 150
#                 elif normalized_price >= 1500:
#                     gesamtabzug += 300
#
#         # Kratzer Vorhanden
#         if s['scratches'] != "" and not isnan(s['scratches']):
#             scratches = int(s['scratches'])
#             if scratches == 2:
#                 if normalized_price < 1500:
#                     gesamtabzug += 50
#                 elif 1501 <= normalized_price <= 6000:
#                     gesamtabzug += 100
#                 elif normalized_price >= 6000:
#                     gesamtabzug += 150
#             elif scratches == 3:
#                 if normalized_price < 1500:
#                     gesamtabzug += 100
#                 elif 1501 <= normalized_price <= 6000:
#                     gesamtabzug += 200
#                 elif normalized_price >= 6000:
#                     gesamtabzug += 300
#
#     if mode == "basispreis":
#         normalized_price = normalized_price + gesamtabzug + (normalized_price * gesamtabzug_proz)
#
#     elif mode == "gebrauchtpreis":
#         normalized_price = normalized_price - gesamtabzug - (normalized_price * gesamtabzug_proz)
#
#     return normalized_price

def apply_correct_final_price(s, price_column, mode):
    gesamtabzug = 0
    gesamtabzug_proz = 0.00
    normalized_price = float("nan")

    if s[price_column] != "" and not isnan(s[price_column]):
        normalized_price = int(s[price_column])

        # Zustand Motorrad
        if s.get('zustand_motorrad', None):
            zustand_motorrad = int(s['zustand_motorrad'])
            if zustand_motorrad == 2:
                gesamtabzug_proz += 0.01
            elif zustand_motorrad == 3:
                gesamtabzug_proz += 0.02
            elif zustand_motorrad == 4:
                gesamtabzug_proz += 0.05
            elif zustand_motorrad == 5:
                gesamtabzug_proz += 0.40

        # Zustand der Verschleißteile
        if s.get('zustand_verschleissteile', None):
            zustand_verschleissteile = int(s['zustand_verschleissteile'])
            if zustand_verschleissteile == 2:
                gesamtabzug_proz += 0.01
            elif zustand_verschleissteile == 3:
                gesamtabzug_proz += 0.03
            elif zustand_verschleissteile == 4:
                gesamtabzug_proz += 0.05
            elif zustand_verschleissteile == 5:
                gesamtabzug_proz += 0.10

        # Zustand & Alter Reifen
        if s.get('zustand_reifen', None):
            zustand_reifen = int(s['zustand_reifen'])
            if zustand_reifen == 2:
                gesamtabzug += 40 if normalized_price >= 1500 else 20
            elif zustand_reifen == 3:
                gesamtabzug += 60 if normalized_price >= 1500 else 30
            elif zustand_reifen == 4:
                gesamtabzug += 80 if normalized_price >= 1500 else 40
            elif zustand_reifen == 5:
                gesamtabzug += 120 if normalized_price >= 1500 else 60

        # # Kratzer Vorhanden (Unverändert)
        # if s['scratches'] != "" and not isnan(s['scratches']):
        #     scratches = int(s['scratches'])
        #     if scratches == 2:
        #         if normalized_price < 1500:
        #             gesamtabzug += 50
        #         elif 1501 <= normalized_price <= 6000:
        #             gesamtabzug += 100
        #         elif normalized_price >= 6000:
        #             gesamtabzug += 150
        #     elif scratches == 3:
        #         if normalized_price < 1500:
        #             gesamtabzug += 100
        #         elif 1501 <= normalized_price <= 6000:
        #             gesamtabzug += 200
        #         elif normalized_price >= 6000:
        #             gesamtabzug += 300

    if mode == "basispreis":
        normalized_price = normalized_price + gesamtabzug + (normalized_price * gesamtabzug_proz)
    elif mode == "gebrauchtpreis":
        normalized_price = normalized_price - gesamtabzug - (normalized_price * gesamtabzug_proz)

    return normalized_price


# This is the updated function based on the new conditions provided.


def apply_sm(s, c1, c2):
    value = difflib.SequenceMatcher(None, s[c1], s[c2]).ratio()
    return value


def apply_sm_marke(s, c1, user_eingabe):
    value = difflib.SequenceMatcher(None, s[c1], user_eingabe).ratio()
    return value


def make_hyperlink(s, c1, c2):
    url = "https://www.google.com/search?q={}"
    return '=HYPERLINK("%s", "%s")' % (url.format(str(s[c1])+"+"+str(s[c2])), "XXX")


def add_column(database, existing_column, new_column):
    if new_column not in database:
        position = database.columns.get_loc(existing_column) + 1
        database.insert(position, new_column, "")


def apply_string_sections(s, column, user_eingabe):
    sections = 0
    if not isnan(s[column]) and not isnan(user_eingabe):
        user_eingabe = str(user_eingabe).lower()
        user_eingabe = re.split('[;, -]', user_eingabe)

        katalog = str(s[column]).lower()
        katalog = re.split('[;, -]', katalog)

        for section in user_eingabe:
            if section in katalog:
                sections += 1

    return sections


def apply_score_modell(s, u_hubraum, u_baujahr, gewicht_ratio, gewicht_hubraum, gewicht_baujahr, gewicht_sfragments, gewicht_ssections, gewicht_backcheck, gewicht_nrmatches, gewicht_len_diff):
    if isnan(s['ratio']) or s['ratio'] == "None" or s['ratio'] == "NaN" or s['ratio'] == "nan" or s['ratio'] == "":
        s_ratio = 0
    else:
        s_ratio = s['ratio']

    if isnan(s['hubraum']) or s['hubraum'] == "None" or s['hubraum'] == "NaN" or s['hubraum'] == "nan" or s['hubraum'] == "":
        s_hubraum = 0
    else:
        s_hubraum = float(s['hubraum'])

    if isnan(s['baujahr']) or s['baujahr'] == "None" or s['baujahr'] == "NaN" or s['baujahr'] == "nan" or s['baujahr'] == "":
        s_baujahr = 0
    else:
        s_baujahr = int(s['baujahr'])

    if isnan(s['string_fragments']) or s['string_fragments'] == "None" or s['string_fragments'] == "NaN" or s['string_fragments'] == "nan" or s['string_fragments'] == "":
        s_string_fragments = 0
    else:
        s_string_fragments = s['string_fragments']

    if isnan(s['number_matches']) or s['number_matches'] == "None" or s['number_matches'] == "NaN" or s['number_matches'] == "nan" or s['number_matches'] == "":
        s_number_matches = 0
    else:
        s_number_matches = s['number_matches']

    if isnan(s['string_sections']) or s['string_sections'] == "None" or s['string_sections'] == "NaN" or s['string_sections'] == "nan" or s['string_sections'] == "":
        s_string_sections = 0
    else:
        s_string_sections = s['string_sections']

    if isnan(s['string_backcheck']) or s['string_backcheck'] == "None" or s['string_backcheck'] == "NaN" or s['string_backcheck'] == "nan" or s['string_backcheck'] == "":
        s_string_backcheck = 0
    else:
        s_string_backcheck = s['string_backcheck']

    if isnan(u_hubraum) or u_hubraum == "None" or u_hubraum == "NaN" or u_hubraum == "nan" or u_hubraum == "":
        u_hubraum = float(s['hubraum'])
    else:
        u_hubraum = float(u_hubraum)

    if isnan(u_baujahr) or u_baujahr == "None" or u_baujahr == "NaN" or u_baujahr == "nan" or u_baujahr == "":
        u_baujahr = float(s['baujahr'])
    else:
        u_baujahr = float(u_baujahr)

    if isnan(s['len_diff']) or s['len_diff'] == "None" or s['len_diff'] == "NaN" or s['len_diff'] == "nan" or s['len_diff'] == "":
        len_diff = 0
    else:
        len_diff = s['len_diff']

    # print("############################################################")
    # print(f's_hubraum={s_hubraum}, s["hubraum"]={s["hubraum"]}')
    # print(f'u_hubraum={u_hubraum}')
    # print(f's_baujahr={s_baujahr}')
    # print(f'u_baujahr={u_baujahr}')
    # print(f's_string_fragments={s_string_fragments}')
    # print(f's_number_matches={s_number_matches}')
    # print(f's_string_sections={s_string_sections}')
    # print(f's_string_backcheck={s_string_backcheck}')
    # print(f'len_diff={len_diff}')
    # print("############################################################")


    score = (1 - s_ratio) * gewicht_ratio + \
        abs(s_hubraum - u_hubraum) * gewicht_hubraum + \
        abs(s_baujahr - u_baujahr) * gewicht_baujahr + \
        (s_string_fragments * (-1)) * gewicht_sfragments + \
        (s_number_matches * (-1)) * gewicht_nrmatches + \
        (s_string_sections * (-1)) * gewicht_ssections + \
        (s_string_backcheck * (-1)) * gewicht_backcheck + \
        len_diff * gewicht_len_diff
    return score


def apply_score_marke(s, gewicht_ratio, gewicht_sfragments, gewicht_ssections, gewicht_backcheck):
    if isnan(s['ratio']) or s['ratio'] == "None" or s['ratio'] == "NaN" or s['ratio'] == "nan" or s['ratio'] == "":
        s_ratio = 0
    else:
        s_ratio = s['ratio']

    if isnan(s['string_fragments']) or s['string_fragments'] == "None" or s['string_fragments'] == "NaN" or s['string_fragments'] == "nan" or s['string_fragments'] == "":
        s_string_fragments = 0
    else:
        s_string_fragments = s['string_fragments']

    if isnan(s['string_sections']) or s['string_sections'] == "None" or s['string_sections'] == "NaN" or s['string_sections'] == "nan" or s['string_sections'] == "":
        s_string_sections = 0
    else:
        s_string_sections = s['string_sections']

    if isnan(s['string_backcheck']) or s['string_backcheck'] == "None" or s['string_backcheck'] == "NaN" or s['string_backcheck'] == "nan" or s['string_backcheck'] == "":
        s_string_backcheck = 0
    else:
        s_string_backcheck = s['string_backcheck']

    score = (1 - s_ratio) * gewicht_ratio + \
        (s_string_fragments * (-1)) * gewicht_sfragments + \
        (s_string_sections * (-1)) * gewicht_ssections + \
        (s_string_backcheck * (-1)) * gewicht_backcheck
    return score


def apply_to_int(s, column):
    val = s[column]
    if isinstance(val, np.generic):
        return val.item()
    return val


def apply_model_opt(s, column):
    if s[column] != "" and not isnan(s[column]):
        model = s[column].upper().replace(" ", "")

        return model


def isnan(num):
    return num != num


def apply_encode(s, column):
    if s[column] != "" and not isnan(s[column]):
        if s[column] == "ja":
            value = 1
        else:
            value = 0

        return value


def apply_age(s):
    if s["baujahr_jahr"] != "" and not isnan(s["baujahr_jahr"]) and s["timestamp"] != "" and not isnan(s["timestamp"]):
        age = datetime.datetime.fromtimestamp(s["timestamp"]).year - s["baujahr_jahr"]
        return age

# def apply_age(s):
#     if s["baujahr_jahr"] != "" and not isnan(s["baujahr_jahr"]) and s["timestamp"] != "" and not isnan(s["timestamp"]):
#
#         try:
#             # age = datetime.datetime.fromtimestamp(s["timestamp"]).year - int(s["baujahr_jahr"])
#
#             timestamp = s["timestamp"]
#             age = datetime.datetime.fromtimestamp(timestamp).year - int(s["baujahr_jahr"])
#
#
#             if age < 0.25:
#                 age = calc_part_of_year_lead_created(timestamp) + 0.25
#
#             elif age == 1:
#                 age = calc_part_of_year_lead_created(timestamp) + 0.5
#
#             else:
#                 age = age - 1 + calc_part_of_year_lead_created(timestamp) + 0.5
#
#         except:
#             age = 0
#
#         return age


def calc_part_of_year_lead_created(timestamp):
    # Datum basierend auf dem Zeitstempel ermitteln
    datum = datetime.datetime.fromtimestamp(timestamp)

    # Ersten Tag des Jahres ermitteln, basierend auf dem Jahr des Zeitstempels
    jahresanfang = datetime.datetime(datum.year, 1, 1)

    # Letzten Tag des Jahres ermitteln (31. Dezember), basierend auf dem Jahr des Zeitstempels
    jahresende = datetime.datetime(datum.year, 12, 31)

    # Tage im Jahr berechnen
    tage_im_jahr = (jahresende - jahresanfang).days + 1

    # Vergangene Tage seit Jahresbeginn bis zum Datum berechnen
    vergangene_tage = (datum - jahresanfang).days

    # Vergangene Tage in eine Kommazahl zwischen 0 und 1 umrechnen
    anteil_vergangenes_jahr = vergangene_tage / tage_im_jahr

    return anteil_vergangenes_jahr



def is_json(myjson):
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True


def log_message(level, client_address, kind, message, language="dev", source="dev"):
    today = datetime.datetime.now().date()
    here = os.path.dirname(os.path.realpath(__file__))
    Path(here + f"/logs/{language}/").mkdir(parents=True, exist_ok=True)

    message = f'[{level}] [{datetime.datetime.now()}] [Client[{client_address}], Language[{language}], Source[{source}]] [{kind}] [{message}];\n'

    with open(f'{here}/logs/{language}/log_{language}_{source}_{today}.txt', "a", encoding="utf-8") as file_object:
        file_object.write(message)
    print(message)



def send_email(subject, body):
    # Einstellungen für den E-Mail-Server (in diesem Beispiel Gmail)
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    from_email = "py.status.automail@gmail.com"
    to_email = "niklas.fliegel@gmail.com"
    password = "hjqw ehrt wtgi tsnf"

    # Erstellen der E-Mail
    msg = MIMEText(body)
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Verbindung zum E-Mail-Server und Senden der E-Mail
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(from_email, password)
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()


def extract_and_update_brand(row):
    """
    Updated function that checks if the 'marke' column is not 'Andere' and updates the 'modell'
    column by removing the brand name if it is present in the 'modell'. If 'marke' is 'Andere',
    it attempts to extract the brand name from the 'modell' column. It updates the 'marke' column
    with the extracted or existing brand name and the 'modell' column with the remaining part.
    This function handles case-insensitivity and removes additional spaces.

    Parameters:
    row (pd.Series): A row of the DataFrame

    Returns:
    pd.Series: Updated row with the extracted brand name and modified model name, or original values in case of errors
    """
    try:
        if 'marke' in row and 'modell' in row and isinstance(row['marke'], str) and isinstance(row['modell'], str):
            marke = row['marke'].strip().lower()
            modell = row['modell'].strip()

            if marke != 'andere':
                # Case-insensitive removal of brand from model
                modell_cleaned = ' '.join(filter(lambda x: marke not in x.lower(), modell.split()))
                return pd.Series([row['marke'], modell_cleaned])
            else:
                words = modell.split()
                if words:
                    extracted_brand = words[0]
                    remaining_model = ' '.join(words[1:])
                    return pd.Series([extracted_brand, remaining_model])
        return pd.Series([row['marke'], row['modell']])
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.Series([row['marke'], row['modell']])




def shuffle_and_save_db(source_database, source_table, target_table):
    # Konfiguration
    config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': source_database
    }

    source_config = f'mysql+mysqlconnector://{config["user"]}:{config["password"]}@{config["host"]}/{config["database"]}'
    target_config = source_config  # Da die Ziel-Datenbank dieselbe wie die Quell-Datenbank ist

    # SQLAlchemy-Engine für die Quell-Datenbank erstellen
    source_engine = create_engine(source_config)

    # Daten aus der Quell-Datenbank laden
    query = f"SELECT * FROM {source_table}"
    df = pd.read_sql(query, source_engine)


    # unnötige Spalten entfernen
    zu_entfernende_strings = ['mobile', 'counter', 'user']
    spalten_zu_behalten = [spalte for spalte in df.columns if not any(s in spalte for s in zu_entfernende_strings)]
    df = df[spalten_zu_behalten]

    df = df[df['modell'].str.strip() != '']

    # Filter
    df = df[
        (df['post_type'] != 0) &
        (df['marke_opt'] != "") &
        (df['modell_opt'] != "") &
        (df['baujahr_jahr'] != "")
        ]

    # Daten mischen
    df = df.sample(frac=1).reset_index(drop=True)

    # SQLAlchemy-Engine für die Ziel-Datenbank erstellen
    target_engine = create_engine(target_config)

    # Gemischte Daten in der Ziel-Datenbank speichern
    df.to_sql(target_table, target_engine, if_exists='replace', index=False)



def transform_data_with_bert(data_df, bert_preprocessor, bert_encoder, batch_size=32, verbose=True):

    bert_outputs = []

    if verbose:
        disable = False
    else:
        disable = True

    for i in tqdm(range(0, len(data_df), batch_size), desc="Verarbeite Daten mit BERT", disable=disable):
        batch_texts = data_df['modell'].iloc[i:i+batch_size].tolist()
        preprocessed_texts = bert_preprocessor(batch_texts)
        bert_output = bert_encoder(preprocessed_texts)['pooled_output']
        bert_outputs.extend(bert_output.numpy())

    # Benenne die BERT-Features
    bert_feature_names = [f'bert_feature_{i}' for i in range(bert_outputs[0].shape[-1])]
    bert_output_df = pd.DataFrame(bert_outputs, columns=bert_feature_names, index=data_df.index)

    data_df = data_df.drop(columns=['modell'])
    data_df = pd.concat([data_df, bert_output_df], axis=1)

    return data_df, bert_feature_names


def df_to_dataset_predict(dataframe, mode=None):
    df = dataframe.copy()

    # labels = df.pop(y_target_data)
    # sample_weight = df.pop(sample_weight_column)

    df_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in df.items()}
    # print(df_dict)
    ds = tf.data.Dataset.from_tensor_slices(df_dict)
    ds = ds.prefetch(1)

    return ds


def get_bert():
    # Laden des BERT-Vorverarbeitungsmodells und des BERT-Encoders
    bert_preprocessor = hub.KerasLayer("nlp_models/bert/preprocessor")
    bert_encoder = hub.KerasLayer("nlp_models/bert/bert-en-uncased-l-2-h-128-a-2")

    return bert_preprocessor, bert_encoder

def vectorize_and_save_db(source_database, source_table, target_table):
    # Konfiguration
    config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': source_database
    }

    # Laden des BERT-Vorverarbeitungsmodells und des BERT-Encoders
    # bert_preprocessor = hub.KerasLayer("nlp_models/bert/preprocessor")
    # bert_encoder = hub.KerasLayer("nlp_models/bert/bert-en-uncased-l-2-h-128-a-2")
    bert_preprocessor, bert_encoder = get_bert()

    source_config = f'mysql+mysqlconnector://{config["user"]}:{config["password"]}@{config["host"]}/{config["database"]}'
    engine = create_engine(source_config)


    try:
        # Daten aus der Datenbank und Tabelle lesen
        query = f'SELECT marke_opt, modell, modell_opt, baujahr_jahr, price_quality FROM {source_table} WHERE post_type != 0'
        data_df = pd.read_sql(query, con=engine)


        bert_outputs = []
        for i in tqdm(range(0, len(data_df), 32), desc="Verarbeite Daten mit BERT", disable=False):
            batch_texts = data_df['modell_opt'].iloc[i:i + 32].tolist()
            preprocessed_texts = bert_preprocessor(batch_texts)
            bert_output = bert_encoder(preprocessed_texts)['pooled_output']
            bert_outputs.extend(bert_output.numpy())

        # Benenne die BERT-Features
        bert_feature_names = [f'bert_vector_{i}' for i in range(bert_outputs[0].shape[-1])]
        bert_output_df = pd.DataFrame(bert_outputs, columns=bert_feature_names, index=data_df.index)
        df = pd.concat([data_df[['marke_opt', 'modell', 'modell_opt', 'baujahr_jahr', 'price_quality']], bert_output_df], axis=1)


        try:
            # Schreiben der DataFrame in die MySQL-Datenbank
            df.to_sql(target_table, con=engine, if_exists='replace', index=False)
            print("Daten erfolgreich in die Datenbank geschrieben.")

        except Exception as e:
            print(traceback.format_exc())
            print(f"Fehler beim Schreiben in die Datenbank: {e}")

    except Exception as e:
        print(traceback.format_exc())
        print(f"Fehler bei der Verarbeitung der Daten: {e}")



def evaluate_price_accuracy(post_meta_vectorized, marke_opt, model_user_input, baujahr_jahr, baujahr_range, bert_preprocessor, bert_encoder, threshold=0.92):
    try:
        df = post_meta_vectorized[
            (post_meta_vectorized['baujahr_jahr'] >= baujahr_jahr - baujahr_range) &
            (post_meta_vectorized['baujahr_jahr'] <= baujahr_jahr + baujahr_range) &
            (post_meta_vectorized['marke_opt'] == marke_opt)
        ].copy()

        # Vektorisierung des Eingabetextes
        preprocessed_text = bert_preprocessor([model_user_input])
        outputs = bert_encoder(preprocessed_text)
        input_vec = np.array(outputs['pooled_output'])[0]

        # Berechnung der Kosinusähnlichkeit für jeden Eintrag
        vector_columns = [col for col in df.columns if col.startswith('bert_vector_')]

        df['similarity'] = df.apply(lambda row: cosine_similarity([input_vec], [np.array([row[col] for col in vector_columns])])[0][0], axis=1)

        # Filtern der Einträge basierend auf dem Schwellenwert und Sortieren nach Ähnlichkeit
        similar_data = df[df['similarity'] > threshold].sort_values(by='similarity', ascending=False)

        if not similar_data.empty:
            aggregated_quality = similar_data['price_quality'].sum()
            # similar_models = pd.unique(similar_data['modell'])[:3]
            similar_models = similar_data[['modell', 'similarity']].drop_duplicates(subset=['modell']).head(3).values.tolist()

            print(f'model_user_input = {model_user_input}')
            print(f'threshold = {threshold}')
            print("Aggregierte Qualität:", aggregated_quality)
            print("Ähnliche similar_models:", similar_models)

            return aggregated_quality, similar_models
        else:
            print("Keine ähnlichen Datenpunkte gefunden.")
            return 0, []

    except Exception as e:
        print(traceback.format_exc())
        print(f"Ein Fehler ist aufgetreten: {e}")
        return 0, []

def finde_ahnlichstes_modell(post_meta_vectorized, marke_opt, model_user_input, baujahr_jahr, baujahr_range):

    df = post_meta_vectorized[
        (post_meta_vectorized['baujahr_jahr'] >= baujahr_jahr - baujahr_range) & (
                    post_meta_vectorized['baujahr_jahr'] <= baujahr_jahr + baujahr_range)
        & (post_meta_vectorized['marke_opt'] == marke_opt)
        ].copy()

    # Sicherstellen, dass modell_opt als String behandelt wird
    df['modell_opt'] = df['modell_opt'].astype(str)

    # Liste aller möglichen Modelle extrahieren
    modelle = df['modell_opt'].tolist()

    # Finden des am ähnlichsten Modells zur Eingabe
    ahnlichstes_modell = difflib.get_close_matches(model_user_input, modelle, n=1, cutoff=0.0)

    # Prüfen, ob ein ähnliches Modell gefunden wurde
    if ahnlichstes_modell:
        return ahnlichstes_modell[0]
    else:
        return model_user_input


def calc_part_of_this_year():
    # Aktuelles Datum ermitteln
    heute = datetime.datetime.now()

    # Ersten Tag des Jahres ermitteln
    jahresanfang = datetime.datetime(heute.year, 1, 1)

    # Letzten Tag des Jahres ermitteln (31. Dezember)
    jahresende = datetime.datetime(heute.year, 12, 31)

    # Tage im Jahr berechnen
    tage_im_jahr = (jahresende - jahresanfang).days + 1

    # Vergangene Tage seit Jahresbeginn berechnen
    vergangene_tage = (heute - jahresanfang).days

    # Vergangene Tage in eine Kommazahl zwischen 0 und 1 umrechnen
    anteil_vergangenes_jahr = vergangene_tage / tage_im_jahr

    return anteil_vergangenes_jahr





def get_new_ip():

    # nl_random_area
    # verbindung_trennen_nl_hintergrund
    # schnellverbinden
    # verbinden_button
    # profil_erstellen_button
    # connection_screen

    def center_and_activate_window(title, application_path=r"C:\Program Files\Proton\VPN\v3.2.9\ProtonVPN.exe"):
        def enum_windows_handler(hwnd, resultList):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) != "":
                resultList.append((hwnd, win32gui.GetWindowText(hwnd)))

        def get_window_by_title(title):
            windows = []
            win32gui.EnumWindows(enum_windows_handler, windows)
            for hwnd, windowTitle in windows:
                if windowTitle.lower().find(title.lower()) != -1:
                    return hwnd
            return None

        while True:
            try:
                hwnd = get_window_by_title(title)
                if hwnd:
                    # Wiederherstellen, falls minimiert, und in den Vordergrund bringen
                    # win32gui.ShowWindow(hwnd, 9)  # Verwende den direkten Wert für SW_RESTORE
                    win32gui.ShowWindow(hwnd, win32con.SW_NORMAL)

                    win32gui.SetForegroundWindow(hwnd)

                    # Zentriere das Fenster (optional)
                    screenWidth, screenHeight = pyautogui.size()
                    rect = win32gui.GetWindowRect(hwnd)
                    windowWidth = rect[2] - rect[0]
                    windowHeight = rect[3] - rect[1]
                    new_x = int((screenWidth - windowWidth) / 2)
                    new_y = int((screenHeight - windowHeight) / 2)
                    win32gui.MoveWindow(hwnd, new_x, new_y, windowWidth, windowHeight, True)
                else:
                    print(f"Kein Fenster mit dem Titel '{title}' gefunden.")
                    if application_path:
                        print("Versuche, die Anwendung zu starten...")
                        subprocess.Popen(application_path)
                        time.sleep(5)  # Warte einen Moment, damit die Anwendung starten kann
                        hwnd = get_window_by_title(title)
                        if hwnd:
                            center_and_activate_window(title)  # Versuche erneut, das Fenster zu zentrieren und zu aktivieren
                        else:
                            print(
                                "Die Anwendung konnte nicht gestartet werden oder das Fenster ist immer noch nicht auffindbar.")

                return

            except:
                print(traceback.format_exc())
                time.sleep(2)
                pass



    def check_ip(old_ip=None):
        if old_ip:
            print(f"Old ip is {old_ip}")

        while True:
            try:
                print("Sending request to api.ipify...")
                response = requests.get('https://api.ipify.org?format=json')
                ip = response.json().get('ip')
                time.sleep(2)
            except:
                print(traceback.format_exc())
                time.sleep(2)
                pass

            if ip != old_ip:
                break
            else:
                print("New ip is same as old ip. Trying again...")
                pass

        print(f"New ip is {ip}")
        return ip

    while True:

        try:
            print("Getting ip...")
            old_ip = check_ip()
            new_ip = old_ip

            print("Moving mouse to de random...")
            position = pyautogui.locateOnScreen(r'auto_gui_images\de_random_area.png', confidence=0.8)
            pyautogui.moveTo(pyautogui.center(position), duration=0)

            # time.sleep(2)

            print("Connect to new ip...")
            start_button_location = pyautogui.locateOnScreen(r'auto_gui_images\verbinden_button.png', confidence=0.8)
            pyautogui.click(start_button_location)

            # time.sleep(2)

            print("Waiting for new conneciton...")
            try:
                while pyautogui.locateOnScreen(r'auto_gui_images\connection_screen.png', confidence=0.8) is not None:
                    print("Verbindung wird hergestellt. Warte...")
                    time.sleep(2)  # Kurze Pause, um endlose schnelle Abfragen zu vermeiden
            except:
                pass


            print("Moving mouse to waiting position...")
            position = pyautogui.locateOnScreen(r'auto_gui_images\profil_erstellen_button.png', confidence=0.8)
            pyautogui.moveTo(pyautogui.center(position), duration=0)

            print("Checking for ip change...")
            while new_ip == old_ip:
                new_ip = check_ip(old_ip)
                time.sleep(2)
            break

        except:
            print(traceback.format_exc())
            center_and_activate_window("Proton VPN")
            # time.sleep(2)
            pass



def save_to_db(data, db_name, table_name):

    create_database(host='localhost', user='root', password='', db_name=db_name)
    engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://root:@localhost/{db_name}', echo=False)
    data.to_sql(con=engine, name=f'{table_name}', if_exists='replace', index=False)


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

    except:
        print(traceback.format_exc())


# Funktion, um den Monat als Zahl zu extrahieren
def extract_month(date):
    return date.month

# Funktion, um das Jahr als Zahl zu extrahieren
def extract_year(date):
    return date.year



def convert_columns_to_most_frequent_dtype(df):
    for column in df.columns:
        # Ersetze exakte 'nan' Strings durch np.nan ohne Beeinflussung anderer Strings
        df[column] = df[column].apply(lambda x: np.nan if x == 'nan' else x)

        # Bestimme den häufigsten Datentyp in der Spalte, ignoriere dabei np.nan Werte
        non_na_values = df[column].dropna()
        if non_na_values.empty:
            continue  # Wenn alle Werte NaN sind, überspringe die Konvertierung

        most_common_type = non_na_values.apply(lambda x: type(x)).mode()[0]

        # Versuche, die gesamte Spalte zu diesem Typ zu konvertieren
        try:
            if most_common_type == type(""):
                df[column] = df[column].astype(str)
            elif most_common_type == type(1.0):
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif most_common_type == type(1):
                df[column] = pd.to_numeric(df[column], downcast='integer', errors='coerce')
            elif most_common_type == type(True):
                df[column] = df[column].astype(bool)
            # Füge hier bei Bedarf weitere Datentypkonvertierungen hinzu
        except Exception as e:
            print(f"Konvertierung der Spalte {column} fehlgeschlagen: {e}")

    return df


def apply_sm_v3(df, column, user_eingabe):
    try:
        value = difflib.SequenceMatcher(None, df[column], user_eingabe).ratio()
        return value
    except:
        return 0


def apply_score_modell_v3(s, results_df_baujahr_column, u_baujahr, gewicht_ratio, gewicht_baujahr, gewicht_sfragments, gewicht_ssections, gewicht_backcheck, gewicht_nrmatches, gewicht_len_diff):
    if isnan(s['ratio']) or s['ratio'] == "None" or s['ratio'] == "NaN" or s['ratio'] == "nan" or s['ratio'] == "":
        s_ratio = 0
    else:
        s_ratio = s['ratio']


    if isnan(s[results_df_baujahr_column]) or s[results_df_baujahr_column] == "None" or s[results_df_baujahr_column] == "NaN" or s[results_df_baujahr_column] == "nan" or s[results_df_baujahr_column] == "":
        s_baujahr = 0
    else:
        s_baujahr = int(s[results_df_baujahr_column])

    if isnan(s['string_fragments']) or s['string_fragments'] == "None" or s['string_fragments'] == "NaN" or s['string_fragments'] == "nan" or s['string_fragments'] == "":
        s_string_fragments = 0
    else:
        s_string_fragments = s['string_fragments']

    if isnan(s['number_matches']) or s['number_matches'] == "None" or s['number_matches'] == "NaN" or s['number_matches'] == "nan" or s['number_matches'] == "":
        s_number_matches = 0
    else:
        s_number_matches = s['number_matches']

    if isnan(s['string_sections']) or s['string_sections'] == "None" or s['string_sections'] == "NaN" or s['string_sections'] == "nan" or s['string_sections'] == "":
        s_string_sections = 0
    else:
        s_string_sections = s['string_sections']

    if isnan(s['string_backcheck']) or s['string_backcheck'] == "None" or s['string_backcheck'] == "NaN" or s['string_backcheck'] == "nan" or s['string_backcheck'] == "":
        s_string_backcheck = 0
    else:
        s_string_backcheck = s['string_backcheck']



    if isnan(u_baujahr) or u_baujahr == "None" or u_baujahr == "NaN" or u_baujahr == "nan" or u_baujahr == "":
        u_baujahr = float(s[results_df_baujahr_column])
    else:
        u_baujahr = float(u_baujahr)

    if isnan(s['len_diff']) or s['len_diff'] == "None" or s['len_diff'] == "NaN" or s['len_diff'] == "nan" or s['len_diff'] == "":
        len_diff = 0
    else:
        len_diff = s['len_diff']


    score = (1 - s_ratio) * gewicht_ratio + \
        abs(s_baujahr - u_baujahr) * gewicht_baujahr + \
        (s_string_fragments * (-1)) * gewicht_sfragments + \
        (s_number_matches * (-1)) * gewicht_nrmatches + \
        (s_string_sections * (-1)) * gewicht_ssections + \
        (s_string_backcheck * (-1)) * gewicht_backcheck + \
        len_diff * gewicht_len_diff
    return score


def apply_ampel(df):

    ampel = 1

    if "ratio" in df:
        if df["ratio"] <= 0.7:
            ampel = 0

    if "abw_baujahr" in df:
        if df["abw_baujahr"] >= 2:
            ampel = 0

    if "unfall" in df:
        if df["unfall"] == "ja":
            ampel = 0

    return ampel


if __name__ == "__main__":

    get_new_ip()
