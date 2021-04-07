# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 18:21:51 2021

@author: BazanJuanCarlos
"""

#crea un arbol de desicion cart
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

dataset=pd.read_csv('master.csv')


copia=dataset
print(copia.shape)
sns.set()
#sns.pairplot(copia,hue='sex',size=1.5) 
#copia.info()
x=copia.loc[:,['country','year','age','suicides_no','population']].values
y=copia.loc[:,['sex']].values

print(x.shape)
print(y.shape)

#from sklearn.impute import SimpleImputer
#imputer=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0) 
#imputer=imputer.fit(x[:,5:])
#x[:,5:]=imputer.transform(x[:,5:])


from sklearn.preprocessing import LabelEncoder

labelEncoder_x=LabelEncoder()

x[:,0]=labelEncoder_x.fit_transform(x[:,0])

x[:,2]=labelEncoder_x.fit_transform(x[:,2])

labelEncoder_y=LabelEncoder()
y=labelEncoder_y.fit_transform(y)


#seleccionamos el porcentaje de entrenamiento y testeo
from sklearn import tree
clasificador=tree.DecisionTreeClassifier(criterion="entropy")
#clasificador.fit(x,y)
#prediccion=clasificador.predict([2,1,8,33],[30,1,7,40])
#print(prediccion)

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3)
print(x_train)
print(x_test)
print(len(x_train) )
print(len(x_test))
clasificador.fit(x_train,y_train)
prediccion=clasificador.predict(x_test)
print(prediccion)
print(y_test)
confu=confusion_matrix(y_test,prediccion)
print(confu)


#reperit 10 veces 

for i in range(10):
    x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3)
    clasificador.fit(x_train,y_train)
    prediccion=clasificador.predict(x_test)
    print("arbol de confucion de la iteraccion",i)
    confu=confusion_matrix(y_test,prediccion)
    print(confu)
    print("============================================")
    print("estimacion de exactitud")
    print(metrics.accuracy_score(y_test,prediccion))
    print("_______________________________________________")
sns.set()
sns.heatmap(confu,square=True,annot=True,cbar=False)
plt.xlabel('Valores de prediccion')
plt.ylabel("valores de verdad")

#tree.plot_tree(clasificador)
#fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
#fig.savefig('tree.png')








