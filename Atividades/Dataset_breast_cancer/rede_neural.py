'''
Dataset breast_cancer_wisconsin_diagnostic

Atividade
a) Converta a variável Diagnóstico em 2 variáveis binárias (one hot encode)

b) Divida os dados em 70% de treino e 30% de teste

c) Criar uma RNA em Keras para classificar o dataset

d) Apresente a matriz de confusão para os dados de teste
'''

# Bibliotecas
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.losses import mean_squared_error
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# Importanto o Dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# Devinindo as Entradas e Saidas
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets


# Convertendo a variavel Diagnostico em 0 para benigno e 1 maligno
y["Diagnosis"] =  y['Diagnosis'].astype('category').cat.codes


# Dividindo dados para teste e treinamento
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

# Normalizando os dados com media 0 e desvio padrão 1 e valores entre 0 e 1
scala = StandardScaler()
X_treino = scala.fit_transform(X_treino)
X_teste = scala.transform(X_teste)

# Criando o Modelo com Keras
model = Sequential()
model.add(Input(shape=(30,))) # Quantidade de entradas

# Camadas Ocultas
model.add(Dense(60, activation=LeakyReLU(alpha=0.1), kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(30, activation=LeakyReLU(alpha=0.1), kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(15, activation=LeakyReLU(alpha=0.1), kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))

# Camada de Saida
model.add(Dense(1, activation="sigmoid"))

# Definção do otimizador e função de custo
model.compile(optimizer=Adam(0.0001), loss="binary_crossentropy", metrics=['accuracy'])

# Epocas
epocas = 200

# Treinamento
H = model.fit(X_treino, y_treino, 
                batch_size=32, 
                epochs=epocas, 
                verbose=2,
                validation_data=(X_teste, y_teste))

# Matriz de Confusao
y_pred = model.predict(X_teste)
y_pred = (y_pred > 0.5).astype(int)

cm = confusion_matrix(y_teste, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno 0', 'Maligno 1'], yticklabels=['Benigno 0', 'Maligno 1'])
plt.title('Matriz de Confusão')
plt.xlabel('Predição')
plt.ylabel('Real')
plt.savefig('Dataset_breast_cancer/matriz_de_confusão.pdf')
plt.close()

# Grafico do Erro
plt.plot(H.history['loss'], label='Loss')
plt.plot(H.history['val_loss'], label='Val Loss')
plt.title('Gráfico de Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Dataset_breast_cancer/evolucao_erro.pdf')
plt.close()

# Grafico da Acuracia
plt.plot(H.history['accuracy'], label='Acurácia')
plt.plot(H.history['val_accuracy'], label='Val  Acurácia')
plt.title('Gráfico de Acurácia')
plt.xlabel('Epochs')
plt.ylabel('Acurácia')
plt.legend()
plt.savefig('Dataset_breast_cancer/evolucao_acuracia.pdf')
plt.close()