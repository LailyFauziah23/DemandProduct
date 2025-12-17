import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ===============================
# LOAD & PREPROCESS DATA
# ===============================
def load_and_prepare_data(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop_duplicates()

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])

    data = data[(data['Date'] >= '2012-01-01') &
                (data['Date'] <= '2016-12-31')]

    data['Order_Demand'] = pd.to_numeric(data['Order_Demand'], errors='coerce')

    le = LabelEncoder()
    data['Product_Code'] = le.fit_transform(data['Product_Code'])
    data['Warehouse'] = le.fit_transform(data['Warehouse'])
    data['Product_Category'] = le.fit_transform(data['Product_Category'])

    data.set_index('Date', inplace=True)

    monthly_data = data['Order_Demand'].resample('M').sum()
    return monthly_data


# ===============================
# TRAIN MODEL
# ===============================
def train_sarima(data):
    model = SARIMAX(
        data,
        order=(3,1,0),
        seasonal_order=(1,1,0,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(disp=False)
    return result


# ===============================
# SAVE & LOAD MODEL
# ===============================
def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)


# ===============================
# FORECAST
# ===============================
def forecast(model, steps=12):
    future = model.get_forecast(steps=steps)
    mean = future.predicted_mean
    ci = future.conf_int()
    return mean, ci
