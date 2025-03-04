# 🧠 Repositório de Redes Neurais com Keras

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Keras](https://img.shields.io/badge/Keras-2.12.0-red.svg)](https://keras.io)

Repositório contendo implementações avançadas de redes neurais usando Keras para a disciplina de Redes Neurais do Prof. Petronio Candido. Inclui soluções para problemas de classificação com técnicas modernas de regularização e visualização de dados.

## 📚 Conteúdo
- [Projetos](#-projetos)
- [Técnicas Destacadas](#-técnicas-destacadas)
- [Instalação](#-instalação)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)
- [Resultados](#-resultados)
- [Contribuição](#-contribuição)
- [Licença](#-licença)
- [Referências](#-referências)

---

## 🚀 Projetos

### 1. Classificação de Dígitos MNIST
**Objetivo**: Reconhecimento de dígitos manuscritos (0-9) com 99%+ de acurácia  
**Destaques**:
- Arquitetura profunda com camadas densas (128-64-32 neurônios)
- Uso de `LeakyReLU` e `Dropout` para regularização
- Visualização de matriz de confusão e evolução do treinamento
- Dataset: [MNIST via scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

### 2. Diagnóstico de Câncer de Mama
**Objetivo**: Classificação binária de tumores (benigno/maligno)  
**Destaques**:
- Pré-processamento com `StandardScaler`
- Camadas ocultas com regularização L2
- Curvas ROC e análise de métricas de desempenho
- Dataset: [Breast Cancer Wisconsin](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

### 3. Classificação de Flores Iris
**Objetivo**: Classificação multiclasse com 95%+ de acurácia  
**Destaques**:
- Visualização 3D das características morfológicas
- Normalização de dados por `StandardScaler`
- Modelo otimizado com `Adam` e early stopping
- Dataset: [Iris UCI](https://archive.ics.uci.edu/dataset/53/iris)

---

## 🛠️ Técnicas Destacadas
| Técnica               | Aplicação                          | Benefício                             |
|-----------------------|------------------------------------|---------------------------------------|
| **LeakyReLU**         | Ativação em camadas ocultas        | Evita neurônios "mortos"              |
| **Dropout (30%)**     | Regularização entre camadas        | Prevenção de overfitting              |
| **Regularização L2**  | Penalização de pesos               | Controle de complexidade do modelo    |
| **Adam Optimizer**    | Otimização adaptativa              | Convergência mais rápida e estável   |
| **Data Augmentation** | Visualização de dados              | Análise exploratória avançada        |

---

## ⚙️ Instalação

### Pré-requisitos
- Python 3.9+
- pip

### Passo a Passo
1. Clonar repositório:
```bash
git clone https://github.com/seu-usuario/redes-neurais-keras.git
cd redes-neurais-keras
```
## 🛠️ Como Configurar o Ambiente

### Criar Ambiente Virtual (Venv)
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

### Instalar Bibliotecas

#### **TensorFlow**  
**Instalação**: `pip install tensorflow==2.12.0`  
**Utilização**: Backend para execução das redes neurais e integração com o Keras.

#### **Keras**  
**Instalação**: `pip install keras==2.12.0`  
**Utilização**: Construção e treinamento das arquiteturas de redes neurais.

#### **NumPy**  
**Instalação**: `pip install numpy==1.24.3`  
**Utilização**: Manipulação numérica eficiente de arrays para pré-processamento.

#### **Matplotlib**  
**Instalação**: `pip install matplotlib==3.7.1`  
**Utilização**: Geração de gráficos (evolução do treinamento, visualização de dados).

#### **scikit-learn**  
**Instalação**: `pip install scikit-learn==1.2.2`  
**Utilização**: Divisão de dados (`train_test_split`), normalização (`StandardScaler`) e métricas (`confusion_matrix`).

#### **Pandas**  
**Instalação**: `pip install pandas==2.0.2`  
**Utilização**: Manipulação de datasets em formato tabular (ex: dataset Iris).

#### **Seaborn**  
**Instalação**: `pip install seaborn==0.12.2`  
**Utilização**: Visualização de matrizes de confusão estilizadas.

#### **ucimlrepo**  
**Instalação**: `pip install ucimlrepo==0.0.3`  
**Utilização**: Download direto de datasets da UCI (ex: Breast Cancer Wisconsin).

---

### Instalação Rápida (via requirements.txt)
pip install -r requirements.txt
```

> **requirements.txt**:
> ```txt
> tensorflow==2.12.0
> keras==2.12.0
> numpy==1.24.3
> matplotlib==3.7.1
> scikit-learn==1.2.2
> pandas==2.0.2
> seaborn==0.12.2
> ucimlrepo==0.0.3
> ```
```