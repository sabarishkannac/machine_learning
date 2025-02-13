import requests
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor 
from sklearn.metrics import mean_squared_error
import datetime as dt
from sklearn.model_selection import GridSearchCV

url = "https://api.eia.gov/v2/electricity/rto/daily-region-sub-ba-data/data/?frequency=daily&data[0]=value&facets[subba][]=4004&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key=hJQMpZOgaq4zMOKjy5NY1pTN9wqPchbSKxBRQXpV"
response = requests.get(url)
data = response.json()
with open('data.json', 'w') as f:
    json.dump(data, f, indent=4)
df1=pd.read_json(r"C:\Users\sabar\.vscode\extensions\ms-python.python-2024.22.2-win32-x64\python_files\.vscode\data.json")
data=df1['response']['data']
with open(r"C:\Users\sabar\Downloads\data69.json", 'w') as f:
    json.dump(data, f, indent=4)
df=pd.read_json(r"C:\Users\sabar\Downloads\data69.json")
df['period'] = pd.to_datetime(df['period']) 
df = df.groupby('period')['value'].sum().reset_index()
df = df.sort_values('period')
df['period'] = pd.to_datetime(df['period'], errors='coerce') 
df = df.dropna(subset=['period']).reset_index(drop=True)
df['day_of_year']= df['period'].dt.day_of_year
x= df['day_of_year']
y=df['value']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

X_train = X_train.values.reshape(-1, 1)

model = XGBRegressor(objective="reg:squarederror", random_state=42)
param_grid = {
    "n_estimators": [10,20,30,40,50,60,70,80,90,100],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    "subsample": [0.8, 1.0],
}

grid_search= GridSearchCV(estimator=model,param_grid=param_grid,scoring="neg_mean_squared_error",
    cv=3,n_jobs=-1)
grid_search.fit(X_train, Y_train)
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:", mse)
