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
sns.pairplot(df)
```

![](graficaRelacion.png)

### Variable Y, (Precio/U de area):
- Se observa que a mayor número de tiendas cercanas el precio tiende a crecer ligeramente
- La distancia a la estación de transporte masivo mas cercana y el precio muestra una relación inversa donde las casas con mayor precio estan mas cerca a una estación.                                                                                  
- Para la longitud y latitud el precio crece a mayor sean los valores de las 2 variables

### Relación entre variables predictivas:

- Se observa que la variable X3, (distancia a la estación de transporte masivo más cercano), tiende a valores muy bajos para ciertos valores de X5 y X6, (longitud y latitud), de esto se infiere que la mayor cantidad de estaciones están centradas en una ubicación especifica.
- Hay mayor cantidad de tiendas alrededor de las casas más cercanas a una estación de transporte masivo



```python
import matplotlib.pyplot as plt
df_=df.rename(columns={"X1 transaction date":"X1",'X2 house age':'X2','X3 distance to the nearest MRT station':'X3',
                       'X4 number of convenience stores':'X4','X5 latitude':'X5','X6 longitude':'X6',
                       'Y house price of unit area':'Y'})
corrMatrix = df_.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()
```

![](imagenes\correlacion.png)

### De la matriz de correlación:
- X3 tiene los valores más altos de correlación con las variables predictivas X4,X5,X6 y la variable Y a predecir.
- X1 tiene los valores más bajos con el resto de variables, por lo que no se tendrá en cuenta para el entrenamiento de los modelos


### Descripción del data set


```python
df.drop(['X1 transaction date'],axis=1,inplace=True)
df.describe()
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
      <th>X2 house age</th>
      <th>X3 distance to the nearest MRT station</th>
      <th>X4 number of convenience stores</th>
      <th>X5 latitude</th>
      <th>X6 longitude</th>
      <th>Y house price of unit area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>414.000000</td>
      <td>414.000000</td>
      <td>414.000000</td>
      <td>414.000000</td>
      <td>414.000000</td>
      <td>414.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>17.712560</td>
      <td>1083.885689</td>
      <td>4.094203</td>
      <td>24.969030</td>
      <td>121.533361</td>
      <td>37.980193</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.392485</td>
      <td>1262.109595</td>
      <td>2.945562</td>
      <td>0.012410</td>
      <td>0.015347</td>
      <td>13.606488</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>23.382840</td>
      <td>0.000000</td>
      <td>24.932070</td>
      <td>121.473530</td>
      <td>7.600000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.025000</td>
      <td>289.324800</td>
      <td>1.000000</td>
      <td>24.963000</td>
      <td>121.528085</td>
      <td>27.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>16.100000</td>
      <td>492.231300</td>
      <td>4.000000</td>
      <td>24.971100</td>
      <td>121.538630</td>
      <td>38.450000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>28.150000</td>
      <td>1454.279000</td>
      <td>6.000000</td>
      <td>24.977455</td>
      <td>121.543305</td>
      <td>46.600000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>43.800000</td>
      <td>6488.021000</td>
      <td>10.000000</td>
      <td>25.014590</td>
      <td>121.566270</td>
      <td>117.500000</td>
    </tr>
  </tbody>
</table>
</div>



### Preparación de la data


```python
# Se inicializa la herramienta para escalar los datos
X = df.drop(['Y house price of unit area'],axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```


```python
from sklearn.model_selection import train_test_split

# Todas las variables
scaled_x = scaler.fit_transform(X)
# Se dividen los datos en 80-20 para el set de entrenamiento y de testeo
X_train1,X_test1,y_train1,y_test1 = train_test_split(scaled_x,df['Y house price of unit area'],test_size = 0.2, random_state=42)

################################################################################################################################

X1 = X.drop(['X5 latitude','X6 longitude'],axis=1)
scaled_x = scaler.fit_transform(X1)
X_train,X_test,y_train,y_test = train_test_split(scaled_x,df['Y house price of unit area'],test_size = 0.2, random_state=42)

X_train2,X_test2,y_train2,y_test2 = train_test_split(df.iloc[:, 3:5],df['Y house price of unit area'],
                                                 test_size = 0.2, random_state=42)
```

### Regresión lineal clásica
##### Para ver como afecta el uso de las variables X5 y X6, solo en regresión clásica, se quiere usar un modelo con todas las variables , otro solo con latitud y longitud y otro con todas las variables menos latitud y longitud


```python
reg3 = LinearRegression().fit(X_train1,y_train1)
y_pred = reg.predict(X_test1)

print('Error MSE: ')
print(mean_squared_error(y_test1, y_pred))
reg_coef = reg.coef_
```

    Error MSE: 
    54.58094520086212
    


```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 3))
ax = fig.add_axes([0,0,1,1])
variables = X.columns
importancia = np.absolute(reg3.coef_)
ax.barh(variables,importancia)
plt.show()
```

![](imagenes\regrelinealclasica.png)


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg = LinearRegression().fit(X_train,y_train)
y_pred = reg.predict(X_test)

print('Error MSE: ')
print(mean_squared_error(y_test, y_pred))
reg_coef = reg.coef_
```

    Error MSE: 
    58.88825128983576
    


```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 3))
ax = fig.add_axes([0,0,1,1])
variables = X1.columns
importancia = np.absolute(reg.coef_)
ax.barh(variables,importancia)
plt.show()
```

![](imagenes\regclasica2.png)


```python
reg2 = LinearRegression().fit(X_train2,y_train2)
y_pred = reg2.predict(X_test2)

print('Error MSE: ')
print(mean_squared_error(y_test2, y_pred))
reg_coef = reg.coef_
```

    Error MSE: 
    82.08014010115268
    

### Regresión lineal Elastic Net


```python
from sklearn.linear_model import ElasticNet
reg_elast = ElasticNet()
reg_elast.fit(X_train1,y_train1)
y_pred = reg_elast.predict(X_test1)
print('Error MSE: ')
print(mean_squared_error(y_test, y_pred))
reg_elast_coef = reg_elast.coef_
```

    Error MSE: 
    61.35399330573307
    


```python
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0,0,1,1])
variables = df.columns[0:-1]
importancia = np.absolute(reg_elast.coef_)
ax.barh(variables,importancia)
plt.show()
```

![](imagenes\regreElastic.png)

### Random Forest


```python
print(importancia)
```

    [0.24628009 0.69727733 0.05644258]
    


```python
# Debido a que los arboles no son afectados por la escala de los datos, en este caso, se usan todas las variables
X_train,X_test,y_train,y_test = train_test_split(X,df['Y house price of unit area'],test_size = 0.2, random_state=42)
```


```python
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print('Error MSE: ')
print(mean_squared_error(y_test, y_pred))

```

    Error MSE: 
    36.28159186986518
    


```python
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0,0,1,1])
variables = X.columns
importancia = np.absolute(random_forest.feature_importances_)
ax.barh(variables,importancia)
plt.show()
```

![](imagenes\randomforest.png)

### XG Boost


```python
from xgboost import XGBRegressor
#xg = XGBRegressor(n_estimators=100, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=0.8)
xg = XGBRegressor()

xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)

print('Error MSE: ')
print(mean_squared_error(y_test, y_pred))
xg_coef = xg.feature_importances_
```

    Error MSE: 
    39.8139344587423
    


```python
xg_coef
```




    array([0.0463699 , 0.664023  , 0.06194649, 0.12445176, 0.1032089 ],
          dtype=float32)




```python
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0,0,1,1])
variables = df.columns[0:-1]
importancia = np.absolute(xg.feature_importances_)
ax.barh(variables,importancia)
plt.show()
```

![](imagenes\xgBoost.png)

### Support Vectors Machines


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
svm.fit(X_train1,y_train1)
y_pred = svm.predict(X_test1)
print('Error MSE: ')
print(mean_squared_error(y_test1, y_pred))
#svm.coef_
```

    Error MSE: 
    60.725841689226584
    


```python
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0,0,1,1])
variables = X.columns
importancia = np.absolute(svm.coef_[0])
ax.barh(variables,importancia)
plt.show()
```

![](imagenes\SVM.png)

### Red Neuronal
#### Se usó el MSE y el MAE como funciones de pérdida


```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
#model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
hist = model.fit(X_train1, y_train1, epochs=75, batch_size=50,  verbose=1, validation_split=0.2)
y_pred= model.predict(X_test1)

```

    Epoch 1/75
    6/6 [==============================] - 1s 148ms/step - loss: 1212.7402 - mse: 1212.7402 - mae: 30.3724 - val_loss: 1508.1959 - val_mse: 1508.1959 - val_mae: 33.4689
    Epoch 2/75
    6/6 [==============================] - 0s 4ms/step - loss: 1156.5328 - mse: 1156.5327 - mae: 29.8880 - val_loss: 1448.9918 - val_mse: 1448.9917 - val_mae: 32.6836
    Epoch 3/75
    6/6 [==============================] - 0s 3ms/step - loss: 1088.1287 - mse: 1088.1287 - mae: 28.7321 - val_loss: 1385.5951 - val_mse: 1385.5951 - val_mae: 31.8925
    Epoch 4/75
    6/6 [==============================] - 0s 4ms/step - loss: 1030.1753 - mse: 1030.1753 - mae: 27.7805 - val_loss: 1318.3425 - val_mse: 1318.3424 - val_mae: 30.9418
    Epoch 5/75
    6/6 [==============================] - 0s 4ms/step - loss: 967.2819 - mse: 967.2819 - mae: 26.8653 - val_loss: 1257.0819 - val_mse: 1257.0819 - val_mae: 30.0582
    Epoch 6/75
    6/6 [==============================] - 0s 4ms/step - loss: 912.9451 - mse: 912.9451 - mae: 26.0239 - val_loss: 1196.6714 - val_mse: 1196.6714 - val_mae: 29.2425
    Epoch 7/75
    6/6 [==============================] - 0s 4ms/step - loss: 861.3686 - mse: 861.3686 - mae: 25.0831 - val_loss: 1140.1206 - val_mse: 1140.1206 - val_mae: 28.4989
    Epoch 8/75
    6/6 [==============================] - 0s 4ms/step - loss: 804.3870 - mse: 804.3870 - mae: 24.1502 - val_loss: 1073.4906 - val_mse: 1073.4906 - val_mae: 27.3333
    Epoch 9/75
    6/6 [==============================] - 0s 4ms/step - loss: 745.1896 - mse: 745.1896 - mae: 23.1150 - val_loss: 1004.1925 - val_mse: 1004.1925 - val_mae: 26.3241
    Epoch 10/75
    6/6 [==============================] - 0s 3ms/step - loss: 683.8279 - mse: 683.8279 - mae: 21.9718 - val_loss: 933.9156 - val_mse: 933.9157 - val_mae: 25.3008
    Epoch 11/75
    6/6 [==============================] - 0s 3ms/step - loss: 624.5649 - mse: 624.5649 - mae: 20.8687 - val_loss: 860.8370 - val_mse: 860.8370 - val_mae: 24.1244
    Epoch 12/75
    6/6 [==============================] - 0s 4ms/step - loss: 557.9302 - mse: 557.9302 - mae: 19.4996 - val_loss: 785.3895 - val_mse: 785.3895 - val_mae: 22.4912
    Epoch 13/75
    6/6 [==============================] - 0s 3ms/step - loss: 511.1060 - mse: 511.1060 - mae: 18.4252 - val_loss: 714.1479 - val_mse: 714.1479 - val_mae: 21.1519
    Epoch 14/75
    6/6 [==============================] - 0s 4ms/step - loss: 452.7346 - mse: 452.7346 - mae: 17.2093 - val_loss: 652.6400 - val_mse: 652.6400 - val_mae: 20.4823
    Epoch 15/75
    6/6 [==============================] - 0s 4ms/step - loss: 396.1524 - mse: 396.1524 - mae: 15.7465 - val_loss: 599.9890 - val_mse: 599.9890 - val_mae: 18.7513
    Epoch 16/75
    6/6 [==============================] - 0s 3ms/step - loss: 339.1245 - mse: 339.1245 - mae: 14.2348 - val_loss: 505.2301 - val_mse: 505.2301 - val_mae: 17.3552
    Epoch 17/75
    6/6 [==============================] - 0s 3ms/step - loss: 287.6193 - mse: 287.6193 - mae: 13.2720 - val_loss: 432.3472 - val_mse: 432.3472 - val_mae: 14.9051
    Epoch 18/75
    6/6 [==============================] - 0s 3ms/step - loss: 238.8875 - mse: 238.8875 - mae: 11.4377 - val_loss: 372.6318 - val_mse: 372.6318 - val_mae: 13.3508
    Epoch 19/75
    6/6 [==============================] - 0s 3ms/step - loss: 199.4809 - mse: 199.4809 - mae: 10.6895 - val_loss: 323.3661 - val_mse: 323.3661 - val_mae: 12.4982
    Epoch 20/75
    6/6 [==============================] - 0s 4ms/step - loss: 167.6119 - mse: 167.6119 - mae: 9.6848 - val_loss: 288.8970 - val_mse: 288.8970 - val_mae: 11.0588
    Epoch 21/75
    6/6 [==============================] - 0s 4ms/step - loss: 152.3924 - mse: 152.3924 - mae: 9.0761 - val_loss: 259.4858 - val_mse: 259.4858 - val_mae: 10.3438
    Epoch 22/75
    6/6 [==============================] - 0s 4ms/step - loss: 139.8603 - mse: 139.8603 - mae: 8.8187 - val_loss: 243.4812 - val_mse: 243.4812 - val_mae: 9.7622
    Epoch 23/75
    6/6 [==============================] - 0s 3ms/step - loss: 128.1980 - mse: 128.1980 - mae: 8.4389 - val_loss: 229.9206 - val_mse: 229.9206 - val_mae: 9.5462
    Epoch 24/75
    6/6 [==============================] - 0s 4ms/step - loss: 120.1926 - mse: 120.1926 - mae: 8.2730 - val_loss: 220.9747 - val_mse: 220.9747 - val_mae: 9.0796
    Epoch 25/75
    6/6 [==============================] - 0s 4ms/step - loss: 115.6447 - mse: 115.6447 - mae: 8.0475 - val_loss: 209.6136 - val_mse: 209.6136 - val_mae: 8.9749
    Epoch 26/75
    6/6 [==============================] - 0s 3ms/step - loss: 109.5338 - mse: 109.5338 - mae: 8.0707 - val_loss: 199.7238 - val_mse: 199.7238 - val_mae: 8.6979
    Epoch 27/75
    6/6 [==============================] - 0s 4ms/step - loss: 105.1437 - mse: 105.1437 - mae: 7.8212 - val_loss: 192.1225 - val_mse: 192.1225 - val_mae: 8.4734
    Epoch 28/75
    6/6 [==============================] - 0s 4ms/step - loss: 103.0124 - mse: 103.0124 - mae: 7.8702 - val_loss: 186.8308 - val_mse: 186.8308 - val_mae: 8.3661
    Epoch 29/75
    6/6 [==============================] - 0s 3ms/step - loss: 99.0677 - mse: 99.0677 - mae: 7.6386 - val_loss: 183.1266 - val_mse: 183.1266 - val_mae: 8.2957
    Epoch 30/75
    6/6 [==============================] - 0s 3ms/step - loss: 95.9980 - mse: 95.9980 - mae: 7.5307 - val_loss: 180.3674 - val_mse: 180.3674 - val_mae: 8.3912
    Epoch 31/75
    6/6 [==============================] - 0s 4ms/step - loss: 96.1197 - mse: 96.1197 - mae: 7.6992 - val_loss: 177.7075 - val_mse: 177.7075 - val_mae: 8.1528
    Epoch 32/75
    6/6 [==============================] - 0s 4ms/step - loss: 92.7237 - mse: 92.7237 - mae: 7.3846 - val_loss: 174.8457 - val_mse: 174.8457 - val_mae: 8.0994
    Epoch 33/75
    6/6 [==============================] - 0s 3ms/step - loss: 91.0746 - mse: 91.0746 - mae: 7.3354 - val_loss: 173.1532 - val_mse: 173.1532 - val_mae: 8.0826
    Epoch 34/75
    6/6 [==============================] - 0s 4ms/step - loss: 89.3025 - mse: 89.3025 - mae: 7.2392 - val_loss: 170.4680 - val_mse: 170.4680 - val_mae: 7.9745
    Epoch 35/75
    6/6 [==============================] - 0s 4ms/step - loss: 88.0378 - mse: 88.0378 - mae: 7.1602 - val_loss: 168.5618 - val_mse: 168.5618 - val_mae: 8.1053
    Epoch 36/75
    6/6 [==============================] - 0s 4ms/step - loss: 88.7968 - mse: 88.7968 - mae: 7.3800 - val_loss: 168.9274 - val_mse: 168.9274 - val_mae: 7.9343
    Epoch 37/75
    6/6 [==============================] - 0s 4ms/step - loss: 87.6458 - mse: 87.6458 - mae: 7.1339 - val_loss: 165.9505 - val_mse: 165.9505 - val_mae: 7.9599
    Epoch 38/75
    6/6 [==============================] - 0s 4ms/step - loss: 86.2968 - mse: 86.2968 - mae: 7.2276 - val_loss: 164.7000 - val_mse: 164.7000 - val_mae: 7.8123
    Epoch 39/75
    6/6 [==============================] - 0s 3ms/step - loss: 83.2896 - mse: 83.2896 - mae: 6.9685 - val_loss: 163.4578 - val_mse: 163.4578 - val_mae: 7.7986
    Epoch 40/75
    6/6 [==============================] - 0s 3ms/step - loss: 82.1016 - mse: 82.1016 - mae: 6.8947 - val_loss: 162.4143 - val_mse: 162.4143 - val_mae: 7.8079
    Epoch 41/75
    6/6 [==============================] - 0s 5ms/step - loss: 81.1080 - mse: 81.1080 - mae: 6.9230 - val_loss: 161.2779 - val_mse: 161.2779 - val_mae: 7.7309
    Epoch 42/75
    6/6 [==============================] - 0s 3ms/step - loss: 82.1855 - mse: 82.1855 - mae: 6.9478 - val_loss: 160.0356 - val_mse: 160.0356 - val_mae: 7.7496
    Epoch 43/75
    6/6 [==============================] - 0s 3ms/step - loss: 78.9993 - mse: 78.9993 - mae: 6.7508 - val_loss: 161.2702 - val_mse: 161.2702 - val_mae: 7.8497
    Epoch 44/75
    6/6 [==============================] - 0s 3ms/step - loss: 79.0845 - mse: 79.0845 - mae: 6.7707 - val_loss: 161.4147 - val_mse: 161.4147 - val_mae: 7.7440
    Epoch 45/75
    6/6 [==============================] - 0s 3ms/step - loss: 78.1878 - mse: 78.1878 - mae: 6.6334 - val_loss: 158.7108 - val_mse: 158.7108 - val_mae: 7.7743
    Epoch 46/75
    6/6 [==============================] - 0s 3ms/step - loss: 77.8328 - mse: 77.8328 - mae: 6.8242 - val_loss: 157.0415 - val_mse: 157.0415 - val_mae: 7.6541
    Epoch 47/75
    6/6 [==============================] - 0s 3ms/step - loss: 78.1141 - mse: 78.1141 - mae: 6.7235 - val_loss: 155.5665 - val_mse: 155.5665 - val_mae: 7.7219
    Epoch 48/75
    6/6 [==============================] - 0s 3ms/step - loss: 77.5874 - mse: 77.5874 - mae: 6.7993 - val_loss: 154.8470 - val_mse: 154.8470 - val_mae: 7.5927
    Epoch 49/75
    6/6 [==============================] - 0s 4ms/step - loss: 77.5250 - mse: 77.5250 - mae: 6.7166 - val_loss: 153.3896 - val_mse: 153.3896 - val_mae: 7.6103
    Epoch 50/75
    6/6 [==============================] - 0s 3ms/step - loss: 74.2661 - mse: 74.2661 - mae: 6.5168 - val_loss: 156.1951 - val_mse: 156.1951 - val_mae: 7.5617
    Epoch 51/75
    6/6 [==============================] - 0s 3ms/step - loss: 73.7535 - mse: 73.7535 - mae: 6.4062 - val_loss: 154.2850 - val_mse: 154.2850 - val_mae: 7.5910
    Epoch 52/75
    6/6 [==============================] - 0s 3ms/step - loss: 72.4371 - mse: 72.4371 - mae: 6.3942 - val_loss: 152.9173 - val_mse: 152.9173 - val_mae: 7.5059
    Epoch 53/75
    6/6 [==============================] - 0s 3ms/step - loss: 73.0068 - mse: 73.0068 - mae: 6.4795 - val_loss: 152.8561 - val_mse: 152.8561 - val_mae: 7.4724
    Epoch 54/75
    6/6 [==============================] - 0s 4ms/step - loss: 72.2864 - mse: 72.2864 - mae: 6.3589 - val_loss: 151.9648 - val_mse: 151.9648 - val_mae: 7.6463
    Epoch 55/75
    6/6 [==============================] - 0s 3ms/step - loss: 71.8967 - mse: 71.8967 - mae: 6.4284 - val_loss: 152.4530 - val_mse: 152.4530 - val_mae: 7.4281
    Epoch 56/75
    6/6 [==============================] - 0s 3ms/step - loss: 71.3249 - mse: 71.3249 - mae: 6.2631 - val_loss: 151.7546 - val_mse: 151.7546 - val_mae: 7.4711
    Epoch 57/75
    6/6 [==============================] - 0s 3ms/step - loss: 70.8216 - mse: 70.8216 - mae: 6.2683 - val_loss: 150.8410 - val_mse: 150.8410 - val_mae: 7.4227
    Epoch 58/75
    6/6 [==============================] - 0s 3ms/step - loss: 70.3462 - mse: 70.3462 - mae: 6.2673 - val_loss: 149.8798 - val_mse: 149.8798 - val_mae: 7.6622
    Epoch 59/75
    6/6 [==============================] - 0s 3ms/step - loss: 74.6380 - mse: 74.6380 - mae: 6.6826 - val_loss: 150.7230 - val_mse: 150.7230 - val_mae: 7.3515
    Epoch 60/75
    6/6 [==============================] - 0s 3ms/step - loss: 72.1362 - mse: 72.1362 - mae: 6.3175 - val_loss: 147.8234 - val_mse: 147.8234 - val_mae: 7.5338
    Epoch 61/75
    6/6 [==============================] - 0s 3ms/step - loss: 69.9010 - mse: 69.9010 - mae: 6.3147 - val_loss: 147.8737 - val_mse: 147.8737 - val_mae: 7.2461
    Epoch 62/75
    6/6 [==============================] - 0s 3ms/step - loss: 69.3424 - mse: 69.3424 - mae: 6.1699 - val_loss: 148.0349 - val_mse: 148.0349 - val_mae: 7.2277
    Epoch 63/75
    6/6 [==============================] - 0s 3ms/step - loss: 70.5425 - mse: 70.5425 - mae: 6.1615 - val_loss: 146.9137 - val_mse: 146.9137 - val_mae: 7.3851
    Epoch 64/75
    6/6 [==============================] - 0s 3ms/step - loss: 71.0949 - mse: 71.0949 - mae: 6.4059 - val_loss: 147.3223 - val_mse: 147.3223 - val_mae: 7.1706
    Epoch 65/75
    6/6 [==============================] - 0s 3ms/step - loss: 70.1304 - mse: 70.1304 - mae: 6.1219 - val_loss: 146.1427 - val_mse: 146.1427 - val_mae: 7.5245
    Epoch 66/75
    6/6 [==============================] - 0s 3ms/step - loss: 68.8421 - mse: 68.8421 - mae: 6.2080 - val_loss: 145.6959 - val_mse: 145.6959 - val_mae: 7.1576
    Epoch 67/75
    6/6 [==============================] - 0s 3ms/step - loss: 66.6550 - mse: 66.6550 - mae: 6.0684 - val_loss: 145.6769 - val_mse: 145.6769 - val_mae: 7.2306
    Epoch 68/75
    6/6 [==============================] - 0s 3ms/step - loss: 66.4812 - mse: 66.4812 - mae: 5.9677 - val_loss: 145.8913 - val_mse: 145.8913 - val_mae: 7.1324
    Epoch 69/75
    6/6 [==============================] - 0s 3ms/step - loss: 65.8769 - mse: 65.8769 - mae: 5.9188 - val_loss: 144.5307 - val_mse: 144.5307 - val_mae: 7.2631
    Epoch 70/75
    6/6 [==============================] - 0s 3ms/step - loss: 66.3151 - mse: 66.3151 - mae: 6.0146 - val_loss: 145.0665 - val_mse: 145.0665 - val_mae: 7.0394
    Epoch 71/75
    6/6 [==============================] - 0s 3ms/step - loss: 65.0614 - mse: 65.0614 - mae: 5.8630 - val_loss: 143.7738 - val_mse: 143.7738 - val_mae: 7.4482
    Epoch 72/75
    6/6 [==============================] - 0s 3ms/step - loss: 68.2404 - mse: 68.2404 - mae: 6.1850 - val_loss: 146.0652 - val_mse: 146.0652 - val_mae: 7.1530
    Epoch 73/75
    6/6 [==============================] - 0s 4ms/step - loss: 66.9548 - mse: 66.9548 - mae: 5.9719 - val_loss: 142.4100 - val_mse: 142.4100 - val_mae: 7.1678
    Epoch 74/75
    6/6 [==============================] - 0s 3ms/step - loss: 65.5326 - mse: 65.5326 - mae: 5.9223 - val_loss: 143.8892 - val_mse: 143.8892 - val_mae: 7.0672
    Epoch 75/75
    6/6 [==============================] - 0s 4ms/step - loss: 65.3446 - mse: 65.3446 - mae: 5.9192 - val_loss: 142.0236 - val_mse: 142.0236 - val_mae: 7.0361
    

#### Para encontrar la importancia de las variables dada por la red neuronal se usó un array de zeros con un 1 en la posición donde se quería obtener el coeficiente y se predijo este array con el modelo anteriormente entrenado. 


```python
RN_coef=np.zeros((1,5))
for i in range(0,5):
  inputs = np.zeros((1,5))
  inputs[0][i] = 1
  tmp_coef = model.predict(inputs)
  RN_coef[0][i] = tmp_coef
  #print(inputs)
RN_coef
```




    array([[0.51342887, 0.62517893, 0.93927634, 0.9477129 , 0.88648379]])




```python
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0,0,1,1])
variables = df.columns[0:-1]
importancia = np.absolute(RN_coef[0])
ax.barh(variables,importancia)
plt.show()
```

![](imagenes\RedNeuronal.png)


```python
print('Error MSE: ')
print(mean_squared_error(y_test1, y_pred))
```

    Error MSE: 
    53.195676395431434
    

### Preguntas adicionales

### **¿Qué variables tienen el mayor impacto en el precio de la vivienda? ¿Cómo aporta cada modelo al conocimiento de este impacto?**

- Para la regresión lineal clásica y elastic y los modelos de ensamble se obtuvo que la variable más importantes fue la distancia la estación de transporte masivo más cercana
- En máquinas de soporte vectorial la variable mas importante fue X4 (cantidad de tiendas cercanas), y al contrario de los primeros modelos X3 no tuvo mucho peso
- La red neuronal le dio mucha importancia a la ubicación (latitud y longitud) y la cantidad de tiendas cercanas

### **¿Cuál es el mejor modelo entre los usados para resolver este problema? ¿Qué criterios se pueden utilizar para responder a esta pregunta?**

#### En el ejercicio se dividió el data set en datos de entrenamiento y de testeo en un ratio de 80-20, la medida de error usada fue el MSE, según esto, los modelos que menor error tuvieron fueron los arreglos de arboles:
* Random forest, MSE = 36.28
* XG Boost, MSE = 39.81

#### Otras medidas comunmente usadas para medir el nivel de error en modelos de regresión son RMSE y MAE, (muy parecidos al error MSE usado en la actividad), y el coeficiente de determinación (R²)

#### En la regresión lineal clásica para entender mejor el impacto de las variables de ubicación en los modelos, se usaron 3 data sets, uno con todas las variables estandarizadas, otro sin las variables de latitud y longitud y otro solo con latitud y longitud. De estos modelos el mejor fue el que se entrenó con todas las variables estandarizadas 

#### Estandarización :
y = (x – mean) / standard_deviation


```python

```
