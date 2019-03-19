import os
import csv
import math
import pickle
import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np

def date(text):
    return datetime.datetime.strptime(text, "%Y-%m-%d").date()
def ll(text):
    return int(float(text)*100)/100
def precip(text):
    TRACE = 1e-3
    # .strip() 양쪽 공백을 지워준다.
    text = text.strip()
    if text =="M":
        return None
    if text =="-":
        return None
    if text =="T":
        return TRACE
    return float(text)

def impute_missing_weather_station_values(weather):
    for k, v in weather.items():
        if v[0] is None:
            v[0] = v[1]
        elif v[1] is None:
            v[1] = v[0]
        for k1 in v[0]:
            if v[0][k1] is None:
                v[0][k1] = v[1][k1]
        for k1 in v[1]
            if v[1][k1] is None:
                v[1][k1] = v[0][k1]

    for line in csv.DictReader(open("dataPath")):
        for name, converter in featuer_dict:
            line[name] = converter(line[name])

def load_weather():
    weather = {}
    for line in csv.DictReader(open("west_nile/weater.csv")):
        for name, converter in {"Date" : date,
                                "Tmax" : float, "Tmin" : float, "Tavg" : float,
                                "DewPoint" : float, "WebBulb" : float,
                                "PrecipTotal" : precip, "Sunrise" : precip, "Sunset" : precip,
                                "Depart" : float, "Heat" : precip, "Cool" : precip,
                                "ResultSpeed" : float, "ResultDir" : float, "AvgSpeed" : float,
                                "StnPressure" : float, "SeaLevel" : float}.items():
            x = line[name].strip()
            line[name] = converter(x) if (x != "M") else None

        station = int(line["Station"]) - 1

        dt = line["Date"]
        if dt not in weather:
            weather[dt] = [None, None]

        weather[dt][station] = line
    impute_missing_weather_station_values(weather)
    return weather

def load_train():
    training = []
    for line in csv.DictReader(open("west_nile/train.csv")):
        for name, converter in {"Date" : date,
                                "Latitude" : ll, "Longitude" : ll,
                                "NumMosquitos" : int, "WnvPresent" : int}.items():
            line[name] = converter(line[name])
        training.append(line)
    return training

def load_test():
    training = []
    for line in csv.DictReader(open("west_nile/test.csv")):
        for name, converter in {"Date" : date,
                                "Latitude" : ll, "Longitude" : ll}.items():
            line[name] = converter(line[name])
        training.append(line)
    return training

def normalize(X, mean=None, std=None):
    count = X.shape[1]
    if mean is None:
        #nanmean 의 경우 None 값을 무시하고 평균을 계산한다. ex [1, nan] 의 평균은 mean의 경우 0.5 nanmean의 경우 1
        mean = np.nanmean(X, axis=0)
    for i in range(count):
        #np.isnan은 리턴값이 bool형이다.
        X[np.isnan(X[:,i]), i] = mean[i]
    if std is None:
        #axis = 0 이면 열끼리 비교하고 1이면 행끼리 비교한다.
        std = np.std(X, axis=0)
    for i in range(count):
        X[:, i] = (X[:, i] - mean[i]) / std[i]
    return mean, std

def scaled_count(record):
    SCALE = 9.0
    if "NumMosquitos" not in record:
        return 1
    return int(np.ceil(record["NumMosquitos"] / SCALE))

def get_closet_station(lat, long):
    stations = np.array([[41.995, -87.933],
                         [41.786, -87.752]])
    loc = np.array([lat, long])
    deltas = stations - loc[None, :]
    dist2 = (deltas**2).sum(1) #여기서의 sum안의 숫자는 axis임.
    return np.argmin(dist2)

species_map = {'CULEX RESTUANS' : "100000",
              'CULEX TERRITANS' : "010000",
              'CULEX PIPIENS'   : "001000",
              'CULEX PIPIENS/RESTUANS' : "101000",
              'CULEX ERRATICUS' : "000100",
              'CULEX SALINARIUS': "000010",
              'CULEX TARSALIS' :  "000001",
              'UNSPECIFIED CULEX': "001100"}

def assemble_X(base, weather):
    X = []
    for b in base:
        date = b["Date"]
        date2 = np.sin((2 * math.pi * date.day) / 365 * 24)
        date3 = np.cos((2 * math.pi * date.day) / 365 * 24)
        date4 = np.sin((2 * math.pi * date.month) / 365)

        lat, longi = b["Latitude"], b["Longitude"]
        case = [date.year, date.month, date4, date.day, date.weekday(), date2, date3, lat, longi]

        for days_ago in [1,2,3,5,8,12]:
            day = datetime.timedelta(days=days_ago)
            for obs in ["Tmax", "Tmin", "Tavg", "DewPoint", "WetBulb",
                        "PrecipTotal", "Depart", "Sunrise", "Sunset",
                        "Heat", "Cool", "ResultSpeed", "ResultDir"]:
                station = get_closet_station(lat, longi)
                case.append(weather[day][station][obs])
        species_vector = [float(x) for x in species_map[b["Species"]]]
        case.extend(species_vector)

        for repeat in range(scaled_count(b)):
            X.append(case)
    X = np.asarray(X, dtype=np.float32)
    return X

def assemble_Y(base):
    y = []
    for b in base:
        present = b["WnvPresent"]
        for repeat in range(scaled_count(b)):
            y.append(present)
    return np.asarray(y, dtype=np.float32).reshape(-1, 1)

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data




