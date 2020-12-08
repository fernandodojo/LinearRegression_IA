# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd, numpy as np
import math
import statistics
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


data = pd.read_csv('jogadores.csv',';')                 #carrega base jogadores
data = data.loc[:, ~data.columns.isin(['Id', "Idade"])] #exclui as colunas id e idade da base jogadores pre carregada
print(data['Classe'].value_counts())                    #printa a contagem de cada classe atacantes e defensores
data["Classe"] = pd.Categorical(data['Classe']).codes   #categoriza atacante como 0, e defensor como 1
print(data['Classe'].value_counts())                    #print a contagem porem selecionando 0 e 1 após categorização

#2-multicolinearidade
sns.pairplot(data)
sns.heatmap(data.corr(), cmap="Blues", annot=True)


validacao = pd.read_csv('validacao.csv',';')                            #carrega a base de validação
validacao = validacao.loc[:, ~validacao.columns.isin(['Id', "Idade"])]  #exclui as colunas id e idade

X = data.drop(["Classe"], axis=1).values                #Codigo para seleção apenas das caracteríticas, ou seja exclui as classe
X = MaxAbsScaler().fit_transform(X)                     #Normalização dos valores das caracteristicas de x para a faixa de -1 e 1
y = data["Classe"].values                               #Seleciona a coluna classes para ser o y
X_val = MaxAbsScaler().fit_transform(validacao)         #Normaliza os valores de X da base de validação para a faixa de -1 e 1


model = LinearRegression()                              #Instanciamento da classe de regressão linear

kf = KFold(shuffle=True, n_splits=5, random_state=0)    #Tecnica de validação cruzada (kfold) com 5 divições utilizando shuffle

scoring = ('neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2') #lista de métricas a serem calculadas para regressão
scores = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=True) #Aplicação do treinamento e geração das métricas

###Métricas de avalização de regressão###
mae = scores['test_neg_mean_absolute_error'].mean()*-1
mse = scores['test_neg_mean_squared_error'].mean()*-1
rmse = scores['test_neg_root_mean_squared_error'].mean()*-1
r2 = scores['test_r2'].mean()

print("MAE:{}".format(mae))
print("MSE:{}".format(mse))
print("RMSE:{}".format(rmse))
print("R2:{}".format(r2))
print("\n")
###Métricas de avalização de regressão###


#precição de resultados para posterior comparação
y_pred = cross_val_predict(model, X, y, cv=kf)  #predição do valores para cada jogador de forma a permitir o arredondamento
y_pred2 = np.where(y_pred > 0.4, 1, 0)          #arredondamento do dos valores preditos para cada jogador considerando threasold de 0.5+ para defensor e 0.4- atacante
#y_pred2 = np.select([y_pred <= .2, y_pred>.8], [np.zeros_like(y_pred), np.ones_like(y_pred)]) #formas alternativas de arredondamento
#y_pred2 = (y_pred > .5).astype(int)                                                           #formas alternativas de arredondamento                                         


###Métricas de avaliação de classificação###
acc = accuracy_score(y_true=y, y_pred=y_pred2)
f1 = f1_score(y_true=y, y_pred=y_pred2)
precision = precision_score(y_true=y, y_pred=y_pred2)
recall = recall_score(y_true=y, y_pred=y_pred2)
cnf_matrix = confusion_matrix(y, y_pred2)
print("ACC:{}".format(acc))
print("F1:{}".format(f1))
print("PRECISION:{}".format(precision))
print("RECALL:{}".format(recall))
print("Matrix de Confusao:{}".format(cnf_matrix))
###Métricas de avaliação de classificação###


model.fit(X, y)  #Outra aplicação do modelo alternativa de para predição dos 30 jogadores desconhecidos, além do calculo do intercept e coeficientes
print('Coefficients:', model.coef_)     #Coeficientes
print('Intercept:', model.intercept_)   #Intercept, ou inclinação da reta


val_pred = model.predict(X_val)             #Predição dos jogadores com base no modelo model acima
val_pred = np.where(val_pred > 0.4, 1, 0)   #arredondamento do dos valores preditos para cada jogador considerando threasold de 0.5+ para defensor e 0.4- atacante
#val_pred = (val_pred > .4).astype(int)     #formas alternativas de arredondamento


###Conversão para legenda atacante e defensor; exportação para csv###
val_pred = np.where(val_pred == 0, 'Atacante', 'Defensor')
val_pred = pd.DataFrame(val_pred, columns=["Classe"])
val_pred.to_csv('Jog_Desconhecidos.csv', index=False)
###Conversão para legenda atacante e defensor; exportação para csv###



#3-distribuição
"""
ax = sns.distplot((y-y_pred), bins = 50)
ax.set(xlabel='Distribuição dos Resíduos')
#plt.hist(y-y_pred)
#plt.ylabel('Distribuição Residual')
"""

#1-linearidade
"""
ax = sns.regplot(x=y_pred, y=y, lowess=True, line_kws={'color': 'red'})
ax.set_title('Valores Observados vs Valores Preditos', fontsize=16)
ax.set(xlabel='Predito', ylabel='Observado')
"""

#4-Homocedasticidade
"""
ax = sns.regplot(x=y_pred, y=(y-y_pred), lowess=True, line_kws={'color': 'red'})
ax.set_title('Resíduos vs Valores Preditos', fontsize=16)
ax.set(xlabel='Predito', ylabel='Resíduos')
"""

#5-Independencia de erros
"""import statsmodels.tsa.api as smt
acf = smt.graphics.plot_acf((y-y_pred), lags=40 , alpha=0.05)
acf.show()
"""






