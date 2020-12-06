# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd, numpy as np
import math
import statistics
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('jogadores.csv',';')
data = data.loc[:, ~data.columns.isin(['Id', "Idade"])]
data["Classe"] = pd.Categorical(data['Classe']).codes

#2-multicolinearidade
#sns.pairplot(data)
#sns.heatmap(data.corr(), cmap="Blues", annot=True)

validacao = pd.read_csv('validacao.csv',';')
validacao = validacao.loc[:, ~validacao.columns.isin(['Id', "Idade"])]


print(data['Classe'].value_counts())

X = data.drop(["Classe"], axis=1).values
X = MaxAbsScaler().fit_transform(X)


y = data["Classe"].values

X_val = MaxAbsScaler().fit_transform(validacao)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
#y_pred2 = np.where(y_pred > 0.4, 1, 0)
#y_pred2 = np.select([y_pred <= .2, y_pred>.8], [np.zeros_like(y_pred), np.ones_like(y_pred)])
y_pred2 = (y_pred > .5).astype(int)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = math.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
acc = accuracy_score(y_true=y, y_pred=y_pred2)

"""from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y, y_pred2)
sns.heatmap(data.corr(), cmap="Blues", annot=True)
"""
print("MAE:{}".format(mae))
print("MSE:{}".format(mse))
print("RMSE:{}".format(rmse))
print("R2:{}".format(r2))
print("ACC:{}".format(acc))
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

val_pred = model.predict(X_val)
val_pred = (val_pred > .5).astype(int)

#3-distribuição
"""sns.distplot((y-y_pred), bins = 50)
#plt.hist(y-y_pred)
#plt.ylabel('Distribuição Residual')
"""

#1-linearidade

ax = sns.regplot(x=y_pred, y=y, lowess=True, line_kws={'color': 'red'})
ax.set_title('Valores Observados vs Valores Preditos', fontsize=16)
ax.set(xlabel='Predito', ylabel='Observado')

#4-Homocedasticidade
"""
ax = sns.regplot(x=y_pred, y=(y-y_pred), lowess=True, line_kws={'color': 'red'})
ax.set_title('Resíduos vs Valores Preditos', fontsize=16)
ax.set(xlabel='Predito', ylabel='Resíduos')
"""
