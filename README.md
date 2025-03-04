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