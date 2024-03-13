import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

print("import done")
data = pd.read_csv('"E:/front UI/london_energy.csv"')

print("stage 1 done")

#  data.head()
data['Date'] = pd.to_datetime(data['Date'] , dayfirst= True)
data['day_of_week'] = data['Date'].dt.dayofweek
data['month'] = data['Date'].dt.month
data['season'] = (data['Date'].dt.month%12 + 3)//3
data['Date'] = data['Date'].map(datetime.datetime.toordinal)

print("stage 2 done")




X = data[['House_number', 'day_of_week', 'month', 'season' ,"Date"]]
y = data['Energy Demand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x1 = np.array(["Date"])
y1 = np.array(["Energy Demand"])
plt.plot(x1, y1)
plt.show()




# Initialize the model
model = RandomForestRegressor()



# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error on Test Set:", mae)



pickle.dump(model , open("model.pkl" , "wb"))


