'''
Utilizando o dataset IRIS, já empregado no notebook sobre MLP, crie sua própria RNA usando o framework Keras para classificar o dataset.
Requisitos:
Particione os dados em 70% para treino e 30% para teste
Monte um modelo com pelo menos 95% de acurácia
'''

# Bibliotecas
import numpy as np
import pandas as pd
import matplotlib as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense, Input, LeakyReLU, Dropout
from keras.optimizers import SGD, Adam
from keras.regularizers import l2


# Importando o Dataset
dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                      names=["sepal_length","sepal_width", "petal_length", "petal_width", "class"])

# Adicionando os campos para separar os dados
dataset['Iris_Setosa'] = dataset['class'] == 'Iris-setosa'
dataset['Iris_Versicolor'] = dataset['class'] == 'Iris-versicolor'
dataset['Iris_Virginica'] = dataset['class'] == 'Iris-virginica'

# Criando um dicionário para mapeamento de classes e cores
class_map = {
    "Iris-setosa": ("blue", "Setosa"),
    "Iris-versicolor": ("red", "Versicolor"),
    "Iris-virginica": ("green", "Virginica")
}

# Definição das combinações de eixos para os gráficos
plot_configs = [
    ("sepal_length", "sepal_width", "Sepal Length x Width"),
    ("sepal_length", "petal_width", "Sepal Length x Petal Width"),
    ("sepal_width", "petal_length", "Sepal Width x Petal Length"),
    ("petal_length", "petal_width", "Petal Length x Width")
]

# Criando a figura e os subplots
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))

# Iterando sobre os subplots e plotando os dados
for (x_feature, y_feature, title), subplot in zip(plot_configs, ax.flatten()):
    for iris_class, (color, label) in class_map.items():
        subset = dataset[dataset["class"] == iris_class]
        subplot.scatter(subset[x_feature], subset[y_feature], c=color, label=label, alpha=0.7)
    
    subplot.set_title(title)
    subplot.set_xlabel(x_feature.replace("_", " ").title())
    subplot.set_ylabel(y_feature.replace("_", " ").title())
    subplot.legend()

# Ajustando layout e salvando o gráfico
plt.tight_layout()
plt.savefig("Dataset_Iris/grafico_iris.pdf")
plt.close()  # Fechar para evitar sobreposição

# Definindo X e Y
X = dataset[['sepal_length','sepal_width','petal_length','petal_width']].values
Y = dataset[['Iris_Setosa','Iris_Versicolor','Iris_Virginica']].values
Y = np.where(Y == True, 1, 0)

# Normalizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividindo os dados de teste e treino
X_treino, X_teste, y_treino, y_teste = train_test_split(X, Y, test_size=0.3, random_state=0)

''' LeakyReLu
alpha = 0.01 → Valor padrão mais comum.
alpha = 0.1 → Mantém mais valores negativos, pode ser útil para redes profundas.
alpha = 0.001 → Mantém menos valores negativos, mais parecido com ReLU.
'''

# Criando o Modelo com Keras
model = Sequential()
model.add(Input(shape=(4,))) # Quantidade de entradas

model.add(Dense(30, activation=LeakyReLU(alpha=0.1), kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))

model.add(Dense(15, activation=LeakyReLU(alpha=0.1), kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))

model.add(Dense(9, activation=LeakyReLU(alpha=0.1), kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))

model.add(Dense(3, activation="softmax")) # Camada de saida com 3 neuronios

model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=['accuracy']) # Definindo a função de custo e otimizador

# Definindo a quantidade de epocas
epocas = 150

# Iniciando o treinamento
H = model.fit(X_treino, y_treino, 
        batch_size=16, 
        epochs=epocas, 
        verbose=2,
        validation_data=(X_teste, y_teste))

# Predição
y_pred = np.argmax(model.predict(X_teste), axis=1)

# Convertendo os dados de y_Teste para o mesmo formato do y_pred
y_teste_convertido = np.argmax(y_teste, axis=1)


# Criando a matriz de confusão
cm = confusion_matrix(y_teste_convertido, y_pred)

# Exibindo a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Setosa", "Versicolor", "Virginica"])
disp.plot(cmap=plt.cm.Blues)
plt.savefig('Dataset_Iris/matriz_de_confusão.pdf')
plt.close()

# Grafico do Erro
plt.plot(H.history['loss'], label='Loss')
plt.plot(H.history['val_loss'], label='Val Loss')
plt.title('Gráfico de Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Dataset_Iris/evolucao_erro.pdf')
plt.close()

# Grafico da Acuracia
plt.plot(H.history['accuracy'], label='Acurácia')
plt.plot(H.history['val_accuracy'], label='Val  Acurácia')
plt.title('Gráfico de Acurácia')
plt.xlabel('Epochs')
plt.ylabel('Acurácia')
plt.legend()
plt.savefig('Dataset_Iris/evolucao_acuracia.pdf')
plt.close()