## Actividad 03: Regresión
### Sebastián Hernández Mantilla

## Predicción del precio de la vivienda en Taiwan
####  El ejercicio de predicción fue desarrollado en python con ayuda de las librerías de Sickit Learn y Keras. La medida de error usada fue el error cuadratico medio (MSE).
#### El impacto de las variables para cada modelo se obtuvo de ScikitLearn y para la visualización de esto se tomó el valor absoluto. En primer lugar se hace un breve análisis visual de los datos y luego se presenta, para cada modelo, el código usado, el resultado del error y una gráfica con el impacto de las variables.


```python
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_excel('Real estate valuation data set.xlsx',index_col='No')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1 transaction date</th>
      <th>X2 house age</th>
      <th>X3 distance to the nearest MRT station</th>
      <th>X4 number of convenience stores</th>
      <th>X5 latitude</th>
      <th>X6 longitude</th>
      <th>Y house price of unit area</th>
    </tr>
    <tr>
      <th>No</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2012.916667</td>
      <td>32.0</td>
      <td>84.87882</td>
      <td>10</td>
      <td>24.98298</td>
      <td>121.54024</td>
      <td>37.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012.916667</td>
      <td>19.5</td>
      <td>306.59470</td>
      <td>9</td>
      <td>24.98034</td>
      <td>121.53951</td>
      <td>42.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013.583333</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>47.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013.500000</td>
      <td>13.3</td>
      <td>561.98450</td>
      <td>5</td>
      <td>24.98746</td>
      <td>121.54391</td>
      <td>54.8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2012.833333</td>
      <td>5.0</td>
      <td>390.56840</td>
      <td>5</td>
      <td>24.97937</td>
      <td>121.54245</td>
      <td>43.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
average = df['Price'].mean()
print(average)

med = df['Price'].median()
print(med)

standard_deviation = df['Price'].std()
print(standard_deviation)
```


```python
sns.pairplot(df)
```




    <seaborn.axisgrid.PairGrid at 0x7f92b1b98c90>




    
![png](Actividad3_nueva_files/Actividad3_nueva_4_1.png)
    



```python
average = df['Y house price of unit area'].mean()
print("Precio promedio")
print(average)
print('\n')

standard_deviation = df['Y house price of unit area'].std()
print("Desviación estandar del precio")
print(standard_deviation)
```

    Precio promedio
    37.98019323671498
    
    
    Desviación estandar del precio
    13.606487697735314
    

### Regresión linear clásica





```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x = df.drop(columns=['Y house price of unit area'],axis=1)
y = df['Y house price of unit area']

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=42)
reg = LinearRegression().fit(X_train,y_train)
y_pred = reg.predict(X_test)

print('Error MSE: ')
print(mean_squared_error(y_test, y_pred))
reg_coef = reg.coef_
```

    Error MSE: 
    53.50225236117862
    


```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0,0,1,1])
variables = df.columns[0:-1]
importancia = np.absolute(reg.coef_)
ax.barh(variables,importancia)
plt.show()
```


    
![png](Actividad3_nueva_files/Actividad3_nueva_8_0.png)
    


### Regresión lineal Elastic Net


```python
from sklearn.linear_model import ElasticNet
reg_elast = ElasticNet()
reg_elast.fit(X_train,y_train)
y_pred = reg_elast.predict(X_test)
print('Error MSE: ')
print(mean_squared_error(y_test, y_pred))
reg_elast_coef = reg_elast.coef_
```

    Error MSE: 
    59.59722132086818
    


```python
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0,0,1,1])
variables = df.columns[0:-1]
importancia = np.absolute(reg_elast.coef_)
ax.barh(variables,importancia)
plt.show()
```


    
![png](Actividad3_nueva_files/Actividad3_nueva_11_0.png)
    


### Random Forest


```python
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print('Error MSE: ')
print(mean_squared_error(y_test, y_pred))

```

    Error MSE: 
    32.98104331708834
    


```python
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0,0,1,1])
variables = df.columns[0:-1]
importancia = np.absolute(random_forest.feature_importances_)
ax.barh(variables,importancia)
plt.show()
```


    
![png](Actividad3_nueva_files/Actividad3_nueva_14_0.png)
    


### XG Boost


```python
from xgboost import XGBRegressor
xg = XGBRegressor(n_estimators=100, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=0.8)

xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)

print('Error MSE: ')
print(mean_squared_error(y_test, y_pred))
xg_coef = xg.feature_importances_
```

    [18:59:17] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    Error MSE: 
    34.70455433976067
    


```python
xg_coef
```




    array([0.0334417 , 0.08472455, 0.22568797, 0.10473609, 0.25279075,
           0.29861888], dtype=float32)




```python
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0,0,1,1])
variables = df.columns[0:-1]
importancia = np.absolute(xg.feature_importances_)
ax.barh(variables,importancia)
plt.show()
```


    
![png](Actividad3_nueva_files/Actividad3_nueva_18_0.png)
    


### Support Vectors Machines

#### Para SVM y la red neuronal se estandarizó la data usando la herramienta StandardScaler de sklearn, la ecuación usada en esta operación es:
y = (x – mean) / standard_deviation


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_y = df['Y house price of unit area']
scaled_x = scaler.fit_transform(df.drop(columns=['Y house price of unit area'],axis=1))

X_train,X_test,y_train,y_test = train_test_split(scaled_x,scaled_y,test_size = 0.2, random_state=42)
```


```python
from sklearn.svm import SVR
svm = SVR(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print('Error MSE: ')
print(mean_squared_error(y_test, y_pred))
#svm.coef_
```

    Error MSE: 
    54.98634955567108
    


```python
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0,0,1,1])
variables = df.columns[0:-1]
importancia = np.absolute(svm.coef_[0])
ax.barh(variables,importancia)
plt.show()
```


    
![png](Actividad3_nueva_files/Actividad3_nueva_23_0.png)
    


### Red Neuronal
#### Se uso el MSE y el MAE como funciones de perdida


```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=6, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
#model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
hist = model.fit(X_train, y_train, epochs=75, batch_size=50,  verbose=1, validation_split=0.2)
y_pred= model.predict(X_test)

```

    Epoch 1/75
    6/6 [==============================] - 14s 100ms/step - loss: 1577.2102 - mse: 1577.2102 - mae: 37.6826 - val_loss: 1922.5105 - val_mse: 1922.5105 - val_mae: 40.6614
    Epoch 2/75
    6/6 [==============================] - 0s 5ms/step - loss: 1637.0549 - mse: 1637.0549 - mae: 38.3193 - val_loss: 1920.5093 - val_mse: 1920.5093 - val_mae: 40.6372
    Epoch 3/75
    6/6 [==============================] - 0s 5ms/step - loss: 1621.2965 - mse: 1621.2965 - mae: 38.1823 - val_loss: 1918.6138 - val_mse: 1918.6138 - val_mae: 40.6152
    Epoch 4/75
    6/6 [==============================] - 0s 6ms/step - loss: 1631.2268 - mse: 1631.2268 - mae: 38.3148 - val_loss: 1916.6847 - val_mse: 1916.6847 - val_mae: 40.5936
    Epoch 5/75
    6/6 [==============================] - 0s 5ms/step - loss: 1516.6758 - mse: 1516.6758 - mae: 36.8403 - val_loss: 1914.6504 - val_mse: 1914.6504 - val_mae: 40.5705
    Epoch 6/75
    6/6 [==============================] - 0s 5ms/step - loss: 1570.4711 - mse: 1570.4711 - mae: 37.4584 - val_loss: 1912.3846 - val_mse: 1912.3846 - val_mae: 40.5446
    Epoch 7/75
    6/6 [==============================] - 0s 5ms/step - loss: 1643.1599 - mse: 1643.1599 - mae: 38.5446 - val_loss: 1909.9133 - val_mse: 1909.9133 - val_mae: 40.5160
    Epoch 8/75
    6/6 [==============================] - 0s 5ms/step - loss: 1639.8094 - mse: 1639.8094 - mae: 38.2113 - val_loss: 1907.1481 - val_mse: 1907.1481 - val_mae: 40.4835
    Epoch 9/75
    6/6 [==============================] - 0s 6ms/step - loss: 1532.4689 - mse: 1532.4689 - mae: 37.0131 - val_loss: 1904.0405 - val_mse: 1904.0405 - val_mae: 40.4465
    Epoch 10/75
    6/6 [==============================] - 0s 6ms/step - loss: 1511.6144 - mse: 1511.6144 - mae: 36.8023 - val_loss: 1900.5474 - val_mse: 1900.5474 - val_mae: 40.4051
    Epoch 11/75
    6/6 [==============================] - 0s 6ms/step - loss: 1628.4517 - mse: 1628.4517 - mae: 38.2201 - val_loss: 1896.6300 - val_mse: 1896.6300 - val_mae: 40.3595
    Epoch 12/75
    6/6 [==============================] - 0s 5ms/step - loss: 1573.7469 - mse: 1573.7469 - mae: 37.6211 - val_loss: 1892.2559 - val_mse: 1892.2559 - val_mae: 40.3096
    Epoch 13/75
    6/6 [==============================] - 0s 6ms/step - loss: 1594.1605 - mse: 1594.1605 - mae: 37.8468 - val_loss: 1887.3477 - val_mse: 1887.3477 - val_mae: 40.2541
    Epoch 14/75
    6/6 [==============================] - 0s 6ms/step - loss: 1581.0973 - mse: 1581.0973 - mae: 37.6869 - val_loss: 1881.8853 - val_mse: 1881.8853 - val_mae: 40.1923
    Epoch 15/75
    6/6 [==============================] - 0s 6ms/step - loss: 1594.5844 - mse: 1594.5844 - mae: 37.6802 - val_loss: 1875.9023 - val_mse: 1875.9025 - val_mae: 40.1244
    Epoch 16/75
    6/6 [==============================] - 0s 6ms/step - loss: 1587.0309 - mse: 1587.0309 - mae: 37.6380 - val_loss: 1869.3091 - val_mse: 1869.3091 - val_mae: 40.0493
    Epoch 17/75
    6/6 [==============================] - 0s 6ms/step - loss: 1509.3265 - mse: 1509.3265 - mae: 36.6738 - val_loss: 1862.0115 - val_mse: 1862.0115 - val_mae: 39.9662
    Epoch 18/75
    6/6 [==============================] - 0s 6ms/step - loss: 1564.1695 - mse: 1564.1695 - mae: 37.4548 - val_loss: 1853.7883 - val_mse: 1853.7883 - val_mae: 39.8726
    Epoch 19/75
    6/6 [==============================] - 0s 6ms/step - loss: 1545.1098 - mse: 1545.1098 - mae: 37.2836 - val_loss: 1844.4956 - val_mse: 1844.4956 - val_mae: 39.7672
    Epoch 20/75
    6/6 [==============================] - 0s 6ms/step - loss: 1490.6424 - mse: 1490.6424 - mae: 36.6034 - val_loss: 1834.2013 - val_mse: 1834.2013 - val_mae: 39.6502
    Epoch 21/75
    6/6 [==============================] - 0s 35ms/step - loss: 1482.9896 - mse: 1482.9896 - mae: 36.4567 - val_loss: 1822.6345 - val_mse: 1822.6345 - val_mae: 39.5188
    Epoch 22/75
    6/6 [==============================] - 0s 6ms/step - loss: 1580.0046 - mse: 1580.0046 - mae: 37.7946 - val_loss: 1809.9149 - val_mse: 1809.9149 - val_mae: 39.3742
    Epoch 23/75
    6/6 [==============================] - 0s 5ms/step - loss: 1488.2504 - mse: 1488.2504 - mae: 36.3955 - val_loss: 1796.2639 - val_mse: 1796.2639 - val_mae: 39.2181
    Epoch 24/75
    6/6 [==============================] - 0s 6ms/step - loss: 1506.8801 - mse: 1506.8801 - mae: 36.6884 - val_loss: 1781.2892 - val_mse: 1781.2892 - val_mae: 39.0464
    Epoch 25/75
    6/6 [==============================] - 0s 5ms/step - loss: 1410.7891 - mse: 1410.7891 - mae: 35.7006 - val_loss: 1765.2538 - val_mse: 1765.2538 - val_mae: 38.8615
    Epoch 26/75
    6/6 [==============================] - 0s 5ms/step - loss: 1405.7910 - mse: 1405.7910 - mae: 35.4958 - val_loss: 1747.4834 - val_mse: 1747.4834 - val_mae: 38.6562
    Epoch 27/75
    6/6 [==============================] - 0s 5ms/step - loss: 1480.6095 - mse: 1480.6095 - mae: 36.3905 - val_loss: 1728.2297 - val_mse: 1728.2297 - val_mae: 38.4324
    Epoch 28/75
    6/6 [==============================] - 0s 5ms/step - loss: 1402.3091 - mse: 1402.3091 - mae: 35.3446 - val_loss: 1708.1509 - val_mse: 1708.1509 - val_mae: 38.1967
    Epoch 29/75
    6/6 [==============================] - 0s 7ms/step - loss: 1402.0879 - mse: 1402.0878 - mae: 35.3867 - val_loss: 1686.2305 - val_mse: 1686.2305 - val_mae: 37.9381
    Epoch 30/75
    6/6 [==============================] - 0s 6ms/step - loss: 1421.5835 - mse: 1421.5835 - mae: 35.5776 - val_loss: 1662.7784 - val_mse: 1662.7784 - val_mae: 37.6588
    Epoch 31/75
    6/6 [==============================] - 0s 7ms/step - loss: 1334.2460 - mse: 1334.2460 - mae: 34.4830 - val_loss: 1638.2253 - val_mse: 1638.2253 - val_mae: 37.3632
    Epoch 32/75
    6/6 [==============================] - 0s 5ms/step - loss: 1310.0889 - mse: 1310.0889 - mae: 34.1979 - val_loss: 1611.6237 - val_mse: 1611.6237 - val_mae: 37.0405
    Epoch 33/75
    6/6 [==============================] - 0s 5ms/step - loss: 1301.3905 - mse: 1301.3905 - mae: 34.0179 - val_loss: 1582.6656 - val_mse: 1582.6656 - val_mae: 36.6858
    Epoch 34/75
    6/6 [==============================] - 0s 6ms/step - loss: 1285.2217 - mse: 1285.2217 - mae: 33.8480 - val_loss: 1551.5410 - val_mse: 1551.5410 - val_mae: 36.3010
    Epoch 35/75
    6/6 [==============================] - 0s 5ms/step - loss: 1241.2321 - mse: 1241.2321 - mae: 33.2763 - val_loss: 1519.3320 - val_mse: 1519.3320 - val_mae: 35.8980
    Epoch 36/75
    6/6 [==============================] - 0s 6ms/step - loss: 1242.3117 - mse: 1242.3117 - mae: 33.2504 - val_loss: 1485.7518 - val_mse: 1485.7518 - val_mae: 35.4710
    Epoch 37/75
    6/6 [==============================] - 0s 6ms/step - loss: 1228.2000 - mse: 1228.2000 - mae: 33.1250 - val_loss: 1449.8408 - val_mse: 1449.8407 - val_mae: 35.0100
    Epoch 38/75
    6/6 [==============================] - 0s 6ms/step - loss: 1195.7595 - mse: 1195.7595 - mae: 32.5910 - val_loss: 1412.2771 - val_mse: 1412.2771 - val_mae: 34.5199
    Epoch 39/75
    6/6 [==============================] - 0s 7ms/step - loss: 1153.8736 - mse: 1153.8736 - mae: 32.0324 - val_loss: 1373.2656 - val_mse: 1373.2656 - val_mae: 34.0025
    Epoch 40/75
    6/6 [==============================] - 0s 7ms/step - loss: 1144.6868 - mse: 1144.6868 - mae: 32.0172 - val_loss: 1333.6190 - val_mse: 1333.6190 - val_mae: 33.4668
    Epoch 41/75
    6/6 [==============================] - 0s 6ms/step - loss: 1113.5187 - mse: 1113.5187 - mae: 31.4213 - val_loss: 1292.8317 - val_mse: 1292.8317 - val_mae: 32.9051
    Epoch 42/75
    6/6 [==============================] - 0s 9ms/step - loss: 1045.8590 - mse: 1045.8591 - mae: 30.6729 - val_loss: 1251.4166 - val_mse: 1251.4166 - val_mae: 32.3224
    Epoch 43/75
    6/6 [==============================] - 0s 8ms/step - loss: 1033.9918 - mse: 1033.9918 - mae: 30.3736 - val_loss: 1208.0333 - val_mse: 1208.0333 - val_mae: 31.7024
    Epoch 44/75
    6/6 [==============================] - 0s 7ms/step - loss: 924.7930 - mse: 924.7930 - mae: 28.6585 - val_loss: 1163.4725 - val_mse: 1163.4725 - val_mae: 31.0511
    Epoch 45/75
    6/6 [==============================] - 0s 6ms/step - loss: 965.6540 - mse: 965.6540 - mae: 29.1911 - val_loss: 1117.2272 - val_mse: 1117.2272 - val_mae: 30.3614
    Epoch 46/75
    6/6 [==============================] - 0s 6ms/step - loss: 873.4442 - mse: 873.4442 - mae: 27.8515 - val_loss: 1071.9730 - val_mse: 1071.9730 - val_mae: 29.6650
    Epoch 47/75
    6/6 [==============================] - 0s 6ms/step - loss: 802.1700 - mse: 802.1700 - mae: 26.6741 - val_loss: 1027.1002 - val_mse: 1027.1002 - val_mae: 28.9537
    Epoch 48/75
    6/6 [==============================] - 0s 6ms/step - loss: 795.9877 - mse: 795.9877 - mae: 26.3455 - val_loss: 981.6653 - val_mse: 981.6653 - val_mae: 28.2125
    Epoch 49/75
    6/6 [==============================] - 0s 5ms/step - loss: 758.2457 - mse: 758.2457 - mae: 25.9220 - val_loss: 937.0184 - val_mse: 937.0184 - val_mae: 27.4638
    Epoch 50/75
    6/6 [==============================] - 0s 6ms/step - loss: 749.8727 - mse: 749.8727 - mae: 25.5979 - val_loss: 892.7395 - val_mse: 892.7395 - val_mae: 26.6929
    Epoch 51/75
    6/6 [==============================] - 0s 6ms/step - loss: 713.5735 - mse: 713.5735 - mae: 25.0219 - val_loss: 849.1143 - val_mse: 849.1143 - val_mae: 25.9051
    Epoch 52/75
    6/6 [==============================] - 0s 5ms/step - loss: 660.0206 - mse: 660.0206 - mae: 23.9744 - val_loss: 806.4897 - val_mse: 806.4897 - val_mae: 25.1080
    Epoch 53/75
    6/6 [==============================] - 0s 6ms/step - loss: 623.9130 - mse: 623.9131 - mae: 23.3186 - val_loss: 764.9446 - val_mse: 764.9446 - val_mae: 24.3037
    Epoch 54/75
    6/6 [==============================] - 0s 5ms/step - loss: 616.4174 - mse: 616.4174 - mae: 22.9764 - val_loss: 724.9099 - val_mse: 724.9099 - val_mae: 23.4957
    Epoch 55/75
    6/6 [==============================] - 0s 6ms/step - loss: 568.1575 - mse: 568.1575 - mae: 21.8359 - val_loss: 686.3093 - val_mse: 686.3093 - val_mae: 22.6789
    Epoch 56/75
    6/6 [==============================] - 0s 6ms/step - loss: 526.4719 - mse: 526.4719 - mae: 21.1226 - val_loss: 648.4824 - val_mse: 648.4824 - val_mae: 21.8448
    Epoch 57/75
    6/6 [==============================] - 0s 6ms/step - loss: 482.7170 - mse: 482.7170 - mae: 20.2673 - val_loss: 612.9055 - val_mse: 612.9055 - val_mae: 21.0217
    Epoch 58/75
    6/6 [==============================] - 0s 5ms/step - loss: 462.1342 - mse: 462.1342 - mae: 19.4795 - val_loss: 578.8438 - val_mse: 578.8438 - val_mae: 20.1961
    Epoch 59/75
    6/6 [==============================] - 0s 5ms/step - loss: 435.6995 - mse: 435.6995 - mae: 18.7669 - val_loss: 546.7598 - val_mse: 546.7598 - val_mae: 19.3764
    Epoch 60/75
    6/6 [==============================] - 0s 6ms/step - loss: 386.2355 - mse: 386.2355 - mae: 17.6107 - val_loss: 516.1667 - val_mse: 516.1667 - val_mae: 18.5604
    Epoch 61/75
    6/6 [==============================] - 0s 5ms/step - loss: 382.3008 - mse: 382.3008 - mae: 17.3543 - val_loss: 486.5733 - val_mse: 486.5733 - val_mae: 17.7527
    Epoch 62/75
    6/6 [==============================] - 0s 6ms/step - loss: 358.2125 - mse: 358.2125 - mae: 16.7830 - val_loss: 459.7442 - val_mse: 459.7442 - val_mae: 16.9752
    Epoch 63/75
    6/6 [==============================] - 0s 6ms/step - loss: 346.5051 - mse: 346.5051 - mae: 16.1378 - val_loss: 435.0261 - val_mse: 435.0261 - val_mae: 16.2531
    Epoch 64/75
    6/6 [==============================] - 0s 7ms/step - loss: 308.4983 - mse: 308.4983 - mae: 15.2908 - val_loss: 412.1986 - val_mse: 412.1986 - val_mae: 15.5558
    Epoch 65/75
    6/6 [==============================] - 0s 5ms/step - loss: 303.7688 - mse: 303.7688 - mae: 15.0019 - val_loss: 391.4041 - val_mse: 391.4041 - val_mae: 14.9073
    Epoch 66/75
    6/6 [==============================] - 0s 6ms/step - loss: 264.9798 - mse: 264.9798 - mae: 13.8620 - val_loss: 371.9313 - val_mse: 371.9313 - val_mae: 14.3072
    Epoch 67/75
    6/6 [==============================] - 0s 5ms/step - loss: 242.4536 - mse: 242.4536 - mae: 13.1912 - val_loss: 353.8891 - val_mse: 353.8891 - val_mae: 13.7393
    Epoch 68/75
    6/6 [==============================] - 0s 5ms/step - loss: 226.3466 - mse: 226.3466 - mae: 12.5506 - val_loss: 337.8478 - val_mse: 337.8478 - val_mae: 13.2134
    Epoch 69/75
    6/6 [==============================] - 0s 6ms/step - loss: 220.1114 - mse: 220.1114 - mae: 12.2916 - val_loss: 323.7257 - val_mse: 323.7257 - val_mae: 12.7524
    Epoch 70/75
    6/6 [==============================] - 0s 5ms/step - loss: 196.1500 - mse: 196.1500 - mae: 11.2749 - val_loss: 311.0445 - val_mse: 311.0445 - val_mae: 12.3633
    Epoch 71/75
    6/6 [==============================] - 0s 5ms/step - loss: 199.4631 - mse: 199.4631 - mae: 11.5428 - val_loss: 299.2138 - val_mse: 299.2138 - val_mae: 12.0124
    Epoch 72/75
    6/6 [==============================] - 0s 7ms/step - loss: 189.8267 - mse: 189.8267 - mae: 11.1262 - val_loss: 288.7536 - val_mse: 288.7536 - val_mae: 11.7177
    Epoch 73/75
    6/6 [==============================] - 0s 6ms/step - loss: 180.7771 - mse: 180.7771 - mae: 10.8704 - val_loss: 279.2018 - val_mse: 279.2018 - val_mae: 11.4507
    Epoch 74/75
    6/6 [==============================] - 0s 6ms/step - loss: 166.2245 - mse: 166.2245 - mae: 10.4279 - val_loss: 270.7699 - val_mse: 270.7699 - val_mae: 11.2053
    Epoch 75/75
    6/6 [==============================] - 0s 6ms/step - loss: 166.2076 - mse: 166.2076 - mae: 10.3119 - val_loss: 263.1284 - val_mse: 263.1284 - val_mae: 11.0076
    

#### Para encontrar la importancia de las variables dada por la red neuronal se usó un array de zeros con un 1 en la posición donde se quería obtener el coeficiente y se predijo este array con el modelo anteriormente entrenado. 


```python
RN_coef=np.zeros((1,6))
for i in range(0,6):
  inputs = np.zeros((1,6))
  inputs[0][i] = 1
  tmp_coef = model.predict(inputs)
  RN_coef[0][i] = tmp_coef
  #print(inputs)
RN_coef
```




    array([[22.00492668, 14.76357937,  8.26590824, 23.52750969, 27.50346375,
            29.1417942 ]])




```python
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0,0,1,1])
variables = df.columns[0:-1]
importancia = np.absolute(RN_coef[0])
ax.barh(variables,importancia)
plt.show()
```


    
![png](Actividad3_nueva_files/Actividad3_nueva_28_0.png)
    



```python
print('Error MSE: ')
print(mean_squared_error(y_test, y_pred))
```

    Error MSE: 
    178.25906258042215
    

### Preguntas adicionales

### **¿Qué variables tienen el mayor impacto en el precio de la vivienda? ¿Cómo aporta cada modelo al conocimiento de este impacto?**

#### Para la regresión lineal clásica se obtuvo que las variables más importante fueron las de ubicación, latitud y longitud, junto con la fecha de la transacción.
#### La regresión lineal Elastic le dio mas importancia a la cantidad de tiendas cercanas y la edad de la vivienda
#### En el random forest las variables con más peso fueron la distancia de la casa al transporte masivo, la edad de la casa y la ubicación, en comparación el XG Boost tuvo mas en cuenta la ubicación, la distancia de la casa al transporte masivo y la cantidad de tiendas cercanas
#### Para el entrenamiento de la red neuronal y el SVM se usó estandarizaron los datos, los resultados fueron:
#### En SVM, similar al random forest, la variable más importante fue la distancia de la casa al transporte masivo, la latitud y la edad de la casa 

#### La red neuronal tuvo más en cuenta la ubicación (longitud y latitud) y la cantidad de tiendas cercanas.  






### **¿Cuál es el mejor modelo entre los usados para resolver este problema? ¿Qué criterios se pueden utilizar para responder a esta pregunta?**

#### En el ejercicio se dividió el data set en datos de entrenamiento y de testeo en un ratio de 80-20, la medida de error usada fue el MSE, según esto, los modelos que menor error tuvieron fueron los arreglos de arboles:
* Random forest, MSE = 33.7
* XG Boost, MSE = 34.7

#### Otras medidas comunmente usadas para medir el nivel de error en modelos de regresión son RMSE y MAE, (muy parecidos al error MSE usado en la actividad), y el coeficiente de determinación (R²)

