import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Crop_recommendation.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

len(X_test)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

model_rf = RandomForestClassifier(n_estimators=200)
model_rf.fit(X_train_sc, y_train)
pred_y = model_rf.predict(X_test_sc)


# input_data = [[78	,37	,22	,25.342171	,63.318020	,6.330554	,74.520820]]

# val = scaler.transform(input_data)

# prediction = model_rf.predict(val)

# print("Predicted Crop:", prediction)

def crop_prediction(N,P,K,temperature,humidity,ph,rainfall):
    data = pd.DataFrame([[
        N,P,K,temperature,humidity,ph,rainfall
    ]])

    data_scaled = scaler.transform(data)
    return model_rf.predict(data_scaled)

