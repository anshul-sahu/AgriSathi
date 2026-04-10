import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('fertilizer_recommendation.csv')

df.head()

X = df.iloc[:, : -1]
y = df.iloc[:, -1].values

X = pd.get_dummies(X, dtype='int')
train_columns=X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

X_train_sc[2]

model_ran = RandomForestClassifier()
model_ran.fit(X_train_sc, y_train)
y_pred_ran = model_ran.predict(X_test_sc)

accuracy_score(y_test,y_pred_ran)*100

from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train_sc, y_train)
y_predict_lr = model_lr.predict(X_test_sc)


def predict_fertilizer(Temperature,Humidity,Moisture,Soil_Type,Crop_Type,Nitrogen,Potassium,Phosphorous):
    data = pd.DataFrame([{
        'Temparature': Temperature,
        'Humidity': Humidity,
        'Moisture': Moisture,
        'Soil_Type': Soil_Type,
        'Crop_Type': Crop_Type,
        'Nitrogen': Nitrogen,
        'Potassium': Potassium,
        'Phosphorous': Phosphorous
    }])
    data = pd.get_dummies(data)
    data = data.reindex(columns=train_columns, fill_value=0)
    
    data_scaled = scaler.transform(data)
    
    prediction = model_ran.predict(data_scaled)
    
    return prediction[0]

