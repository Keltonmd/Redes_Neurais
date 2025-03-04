'''
Dataset MNIST
Principais campos

digits.images: Dados da imagem em pixels (matriz 8x8 para visualização)
digits.data: Dados da imagem em linha (vetor de 64 posições)
digits.target: Classse (de 0 à 9)

Atividade
a) Divida os dados em 70% de treino e 30% de teste

c) Criar uma RNA em Keras para classificar o dataset (64 entradas e 10 saídas)

d) Apresente a matriz de confusão para os dados de teste
'''

# Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# Carregando o dataset
digitos = load_digits()

# Criando uma figura com subplots
fig, ax = plt.subplots(5, 2, figsize=(10, 10))  
plt.gray()  # Definindo a escala de cores para tons de cinza

# Exibindo as imagens
for i in range(5):
    for j in range(2):
        # Exibe a imagem no subplot correspondente
        ax[i, j].matshow(digitos.images[j + i * 2])
        
        # Adicionando título para identificar o dígito
        ax[i, j].set_title(f"Label: {digitos.target[j + i * 2]}")
        
        # Removendo os eixos para melhorar a visualização
        ax[i, j].axis('off')

# Salvando a figura
plt.savefig("Dataset_MNIST/digitos.pdf", bbox_inches='tight')
plt.close()


# Definindo as Entradas e Saidas
X = np.zeros((digitos.target.shape[0], 64))
Y = np.zeros((digitos.target.shape[0], 10))

for i in range(digitos.target.shape[0]):
  X[i] = digitos.images[i].reshape((64))
  Y[i][digitos.target[i]] = 1

# Dividindo dados para teste e treinamento
X_treino, X_teste, y_treino, y_teste = train_test_split(X, Y, test_size=0.3, random_state=0)

# Criando o Modelo com Keras
model = Sequential()
model.add(Input(shape=(64,))) # Quantidade de entradas

# Camadas Ocultas
model.add(Dense(128, activation=LeakyReLU(alpha=0.01), kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(64, activation=LeakyReLU(alpha=0.01), kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(32, activation=LeakyReLU(alpha=0.01), kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))

# Camada de saida
model.add(Dense(10, activation="softmax"))

# Definção do otimizador e função de custo
model.compile(optimizer=Adam(0.0005), loss="categorical_crossentropy", metrics=['accuracy'])

# Epocas
epocas = 250

# Treinamento
H = model.fit(X_treino, y_treino, 
              batch_size=32, 
              epochs=epocas, 
              verbose=2,
              validation_data=(X_teste, y_teste))

# Realizando a predição
y_pred = model.predict(X_teste)

# Convertendo as predições e os rótulos reais para classes (índices)
y_pred_classes = np.argmax(y_pred, axis=1)
y_teste_classes = np.argmax(y_teste, axis=1)

# Calculando a matriz de confusão
cm = confusion_matrix(y_teste_classes, y_pred_classes)

# Plotando a matriz de confusão
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[str(i) for i in range(10)], 
            yticklabels=[str(i) for i in range(10)],
            annot_kws={'size': 14})  # Aumentando o tamanho da fonte das anotações

# Adicionando título e labels
plt.title('Matriz de Confusão', fontsize=16)
plt.xlabel('Predição', fontsize=14)
plt.ylabel('Real', fontsize=14)

# Salvando a matriz de confusão
plt.savefig("Dataset_MNIST/Matriz_de_Confusao.pdf", bbox_inches='tight')
plt.close()

# Grafico do Erro
plt.plot(H.history['loss'], label='Loss')
plt.plot(H.history['val_loss'], label='Val Loss')
plt.title('Gráfico de Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Dataset_MNIST/evolucao_erro.pdf')
plt.close()

# Grafico da Acuracia
plt.plot(H.history['accuracy'], label='Acurácia')
plt.plot(H.history['val_accuracy'], label='Val  Acurácia')
plt.title('Gráfico de Acurácia')
plt.xlabel('Epochs')
plt.ylabel('Acurácia')
plt.legend()
plt.savefig('Dataset_MNIST/evolucao_acuracia.pdf')
plt.close()