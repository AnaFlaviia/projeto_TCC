#Suavizando a curva histórica - em virtude da grande quantidade de outliers

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

df = pd.read_csv('base_cafe_exporta.csv', sep=';', parse_dates=['Data'])
df.set_index('Data', inplace=True)
df.rename(columns={'Quilograma': 'valor'}, inplace=True)
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[['valor']])
df['outlier'] = df['anomaly'] == -1
df['valor_corrigido'] = df['valor']
df.loc[df['outlier'], 'valor_corrigido'] = np.nan
df['valor_corrigido'] = df['valor_corrigido'].interpolate()
print(df.columns)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['valor'], label='Original')
plt.plot(df.index, df['valor_corrigido'], label='Corrigido', linestyle='--')
plt.scatter(df.index[df['outlier']], df['valor'][df['outlier']], color='red', label='Outliers', zorder=5)
plt.legend()
plt.title('Detecção e Correção de Outliers com Isolation Forest')
plt.grid(True)
plt.show()

#df = pd.read_csv('base_cafe_exporta.csv')
#print(df.columns)

#Com auto arima usando os valores corrigidos
import pmdarima as pm


modelo = pm.auto_arima(df['valor_corrigido'],seasonal=True,m=12, stepwise=True,suppress_warnings=True,trace=True)
print(modelo.summary())

#Previsão de resultados
previsao = modelo_auto.predict(n_periods=36)
plt.figure(figsize=(10,5))
plt.plot(df['valor_corrigido'], label='Original')
plt.plot(pd.date_range(df.index[-1], periods=36, freq='M'), previsao, label='Previsão', color='purple')
plt.legend()
plt.show()


#Medindo a qualidade do modelo com Isolation Forest

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = df['valor_corrigido'][-36:].values
y_pred = previsao
    
#MAE,MSE,RMSE
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# MAPE (%)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# R²
r2 = r2_score(y_true, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²:   {r2:.4f}")

--------------------------------------------------------------------------------------------------------------------
#Com auto arima e sazonalidade - sem usar os valores corrigidos

pip install pmdarima
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
#from statsmodels.tsa.stattools import adfuller
df= pd.read_csv("base_cafe_exporta.csv", sep=';')
#print(df.columns)
df['Data'] = pd.to_datetime(df['Data'])
#print(df['Data'].head())
df.set_index('Data', inplace=True)
df = df.sort_index()
modelo_auto = auto_arima(df['Quilograma'],seasonal=True,m=12,trace=True,stepwise=True)
print(modelo_auto.summary())
previsao = modelo_auto.predict(n_periods=36)
plt.figure(figsize=(10,5))
plt.plot(df['Quilograma'], label='Original')
plt.plot(pd.date_range(df.index[-1], periods=36, freq='M'), previsao, label='Previsão', color='purple')
plt.legend()
plt.show()

#Métrica para avaliar a qualidade do modelo preditivo

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = df['Quilograma'][-36:].values
y_pred = previsao
    
#MAE,MSE,RMSE
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# MAPE (%)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# R²
r2 = r2_score(y_true, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²:   {r2:.4f}")

----------------------------------------------------------------------------------------------------------------------------
#Com auto arima e sem sazonalidade

pip install pmdarima
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
df= pd.read_csv("base_cafe_exporta.csv", sep=';')
#print(df.columns)
df['Data'] = pd.to_datetime(df['Data'])
#print(df['Data'].head())
df.set_index('Data', inplace=True)
df = df.sort_index()
modelo_auto = auto_arima(df['Quilograma'],seasonal=False,trace=True,stepwise=True)
print(modelo_auto.summary())
previsao = modelo_auto.predict(n_periods=12)
plt.figure(figsize=(10,5))
plt.plot(df['Quilograma'], label='Original')
plt.plot(pd.date_range(df.index[-1], periods=12, freq='M'), previsao, label='Previsão', color='purple')
plt.legend()
plt.show()

#p=0/d=1/q=4 pelo autoarima
-----------------------------------------------------------------------------------------------------------------------------
#Rodando o ARIMA com os p/d/q encontrados no Autoarima

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
df= pd.read_csv("base_cafe_exporta.csv", sep=';')
df['Data'] = pd.to_datetime(df['Data'])
df.set_index('Data', inplace=True)
df = df.sort_index()
print(df.head())
modelo = ARIMA(df['Quilograma'], order=(2,1,1))
resultado= modelo.fit()

print(resultado.summary())

#Previsões
previsao = resultado.get_forecast(steps=12)
media_previsao = previsao.predicted_mean
intervalo = previsao.conf_int()

# Plotar resultados
plt.figure(figsize=(10,5))
plt.plot(df['Quilograma'], label='Original')
plt.plot(media_previsao, label='Previsão', color='orange')
plt.fill_between(media_previsao.index, intervalo.iloc[:, 0], intervalo.iloc[:, 1], color='orange', alpha=0.3)
plt.legend()
plt.show()


# Resíduos do modelo
residuos = resultado.resid

# Plotar os resíduos
plt.figure(figsize=(10,5))
plt.plot(residuos)
plt.title("Resíduos do Modelo ARIMA")
plt.show()

#resíduos não parecem aleatórios, indicando que pode haver sazonalidade, usar o SARIMA também

----------------------------------------------------------------------------------------------------------------------------

#Código para saber se a série é estacionária

from statsmodels.tsa.stattools import adfuller
df= pd.read_csv("base_cafe_exporta.csv", sep=';')

# Teste de Dickey-Fuller
resultado = adfuller(df['Quilograma'])

# Exibir resultados
print(f"Estatística do teste: {resultado[0]}")
print(f"Valor p: {resultado[1]}")
print(f"Valor crítico: {resultado[4]}")

#O que significa o valor p:
#O valor p é 0.3769, que é maior que 0,05. Isso indica que não vamos rejeitar a hipótese nula. Ou seja, não há evidências suficientes para concluir que a série é estacionária.
#Comparando a Estatística do teste com os valores críticos:
#A estatística do teste (-1.8071) é maior que o valor crítico de 5% (-2.8864).
#Isso também confirma que não podemos rejeitar a hipótese nula de que a série possui uma raiz unitária.
#Dado que a série é não estacionaria, podemos usar o método ARIMA

------------------------------------------------------------------------------------------------------------------------
#Código para decompor série em três componentes principais - Tendência / Sazonalidade / Resíduo


import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
# Carregar os dados
df= pd.read_csv("base_cafe_exporta.csv", sep=';')
# Converter a coluna de data
df['Data'] = pd.to_datetime(df['Data'])
# Colocar a coluna de data como índice
df.set_index('Data', inplace=True)
# Certificar-se de que os dados estão ordenados por data
df = df.sort_index()
# Aplicar STL
decomposition = STL(df['Quilograma'], period=12).fit()
# Plotar os componentes
decomposition.plot()
plt.show()
--------------------------------------------------------------------------------------------------------------------------
#Código para rodar o SARIMA - coms os (p, d, q) e (P, D, Q, s) encontrados no autoarima

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
df= pd.read_csv("base_cafe_exporta.csv", sep=';')
df['Data'] = pd.to_datetime(df['Data'])
df.set_index('Data', inplace=True)
df = df.sort_index()
print(df.head())
modelo_sarima = SARIMAX(df['Quilograma'], order=(2,1,1), seasonal_order=(1,0,1,12))
resultado_sarima = modelo_sarima.fit()
print(resultado_sarima.summary())
previsao = resultado_sarima.get_forecast(steps=12)
media_previsao = previsao.predicted_mean
intervalo = previsao.conf_int()
indice_previsao = pd.date_range(df.index[-1] + pd.Timedelta(1, unit='M'), periods=12, freq='M')


# Plotar os resultados
plt.figure(figsize=(10,5))
plt.plot(df['Quilograma'], label='Original')
plt.plot(media_previsao.index, media_previsao, label='Previsão', color='orange')
plt.fill_between(media_previsao.index, intervalo.iloc[:, 0], intervalo.iloc[:, 1], color='orange', alpha=0.3)
plt.legend()
plt.show()

------------------------------------------------------------------------------------------------------------------
#Usando resultado SARIMA para modelo de ML

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np

# Carregar as séries temporais de diferentes arquivos
df_cafe = pd.read_csv('base_cafe_exporta.csv', sep=';', parse_dates=['Data'], dayfirst=True)
df_preco_coca = pd.read_csv('serie_coca_cola.csv', sep=';', parse_dates=['Data'], dayfirst=True)
df_preco_nestle= pd.read_csv('serie_nestle.csv', sep=';', parse_dates=['Data'], dayfirst=True)
df_preco_starbucks= pd.read_csv('serie_Starbucks.csv', sep=';', parse_dates=['Data'], dayfirst=True)

# Definir a coluna 'Data' como índice
df_cafe.set_index('Data', inplace=True)
df_preco_coca.set_index('Data', inplace=True)
df_preco_nestle.set_index('Data', inplace=True)
df_preco_starbucks.set_index('Data', inplace=True)

# Exibir as primeiras linhas de cada DataFrame para verificar os dados
print(df_cafe.head())
print(df_preco_coca.head())
print(df_preco_nestle.head())
print(df_preco_starbucks.head())


# Mesclar os DataFrames com base no índice (Data)
df_completo = df_cafe.join([df_preco_coca, df_preco_nestle, df_preco_starbucks], how='inner')

# Verifique o DataFrame mesclado
print(df_completo.head())

#tirar espaços extras
print(df_completo.columns.tolist())
df_completo.columns = df_completo.columns.str.strip()
df_completo['Último Coca'].str.replace("R$","",regex=False).str.replace(',','.',regex=False).astype(float)
df_completo['Último Coca']=df_completo['Último Coca'].str.replace("R$","",regex=False).str.replace(',','.',regex=False).astype(float)
df_completo['Último Nestle'].str.replace("R$","",regex=False).str.replace(',','.',regex=False).astype(float)
df_completo['Último Nestle']=df_completo['Último Nestle'].str.replace("R$","",regex=False).str.replace(',','.',regex=False).astype(float)
df_completo['Último Star'].str.replace("R$","",regex=False).str.replace(',','.',regex=False).astype(float)
df_completo['Último Star']=df_completo['Último Star'].str.replace("R$","",regex=False).str.replace(',','.',regex=False).astype(float)



Y = df_completo[['Quilograma']]
X = df_completo[['Último Coca', 'Último Nestle', 'Último Star']].astype(float)

print(Y.columns)
print(X.columns)

#forçar que as colunas como objeto convertam em numéricas
#f_completo['Último Coca'] = pd.to_numeric(df_completo['Último Coca'], errors='coerce')
#df_completo['Último Nestle'] = pd.to_numeric(df_completo['Último Nestle'], errors='coerce')
#df_completo['Último Star'] = pd.to_numeric(df_completo['Último Star'], errors='coerce')
#print(X.isnull().sum()) 


# Ajustar o modelo ARIMA à série 'cafe'
modelo_sarima = SARIMAX(Y, exog=X, order=(2,1,1),seasonal_order=(1,0,1,12))
resultado = modelo_sarima.fit()

# Prever os próximos 12 períodos
n_periods = 120

# X_futuro: valores das variáveis exógenas para o futuro (de mesmo tamanho que n_periods)
forecast = resultado.get_forecast(steps=n_periods, exog=X)

# Obter valores previstos e intervalo de confiança
y_pred = forecast.predicted_mean
conf_int = forecast.conf_int()
datas_futuras = pd.date_range(start=Y.index[-1] + pd.Timedelta(1, unit='D'), periods=n_periods, freq='M')


# Plotar
plt.figure(figsize=(12,6))
plt.plot(Y, label='Histórico')
plt.plot(datas_futuras, y_pred, label='Previsão', color='purple')
plt.fill_between(datas_futuras, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.3)
plt.legend()
plt.title('Previsão SARIMAX com variáveis exógenas')
plt.grid(True)
plt.show()


print(resultado.summary())

print(df_completo.dtypes)


