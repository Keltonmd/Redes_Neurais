# 🧠 Repositório de Redes Neurais com Keras

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Keras](https://img.shields.io/badge/Keras-2.12.0-red.svg)](https://keras.io)

Repositório contendo implementações avançadas de redes neurais usando Keras para a disciplina de Redes Neurais do Prof. Petronio Candido. Inclui soluções para problemas de classificação com técnicas modernas de regularização e visualização de dados.

## 📚 Conteúdo
- [Projetos](#-projetos)
- [Técnicas Destacadas](#️-técnicas-destacadas)
- [Configuração do Ambiente](#️-configuração-do-ambiente)
- [Instalar Dependências](#-instalar-dependências)

---

# 🚀 Projetos

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

# 🛠️ Técnicas Destacadas
| Técnica               | Aplicação                          | Benefício                             |
|-----------------------|------------------------------------|---------------------------------------|
| **LeakyReLU**         | Ativação em camadas ocultas        | Evita neurônios "mortos"              |
| **Dropout (30%)**     | Regularização entre camadas        | Prevenção de overfitting              |
| **Regularização L2**  | Penalização de pesos               | Controle de complexidade do modelo    |
| **Adam Optimizer**    | Otimização adaptativa              | Convergência mais rápida e estável   |
| **Data Augmentation** | Visualização de dados              | Análise exploratória avançada        |

---

# ⚙️ Configuração do Ambiente

### Pré-requisitos
- Python 3.9+ instalado
- Git para clonar o repositório (opcional)

### Passo a Passo

1. **Clonar o Repositório**
```bash
git clone https://github.com/seu-usuario/redes-neurais-keras.git
cd redes-neurais-keras
```

2. **Criar e Ativar Ambiente Virtual**
```bash
# Criar ambiente
python -m venv .venv

# Ativar ambiente
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

# 📦 Instalar Dependências
```bash
pip install -r Atividades/requirements.txt
```

### Lista Completa de Bibliotecas

| Biblioteca       | Versão   | Função Principal                          |
|-------------------|----------|-------------------------------------------|
| TensorFlow        | 2.18.0   | Backend para computação numérica          |
| Keras             | 3.8.0    | Construção de modelos de redes neurais    |
| NumPy             | 2.2.3    | Manipulação de arrays multidimensionais   |
| Matplotlib        | 3.10.1   | Geração de gráficos e visualizações       |
| scikit-learn      | 1.6.1    | Ferramentas de ML e pré-processamento     |
| Pandas            | 2.2.3    | Manipulação de dados tabulares            |
| Seaborn           | 0.13.2   | Visualizações estatísticas avançadas      |
| ucimlrepo         | 0.0.7    | Acesso a datasets da UCI ML Repository    |
| PyQt6             | 6.8.1    | Interface gráfica para visualizações      |
| pip               | 25.0.1   | Gerenciador de pacotes Python             |

> 📁 **Atividades/requirements.txt**  
> ```txt
> pip==25.0.1
> PyQt6==6.8.1
> scikit-learn==1.6.1
> seaborn==0.13.2
> tensorflow==2.18.0
> ucimlrepo==0.0.7
> keras==3.8.0
> matplotlib==3.10.1
> numpy==2.2.3
> pandas==2.2.3
> ```

### ✅ Verificação da Instalação
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import keras; print(f'Keras: {keras.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
```

### ⚠️ Notas Importantes
1. A versão do pip será atualizada automaticamente durante a instalação
2. O PyQt6 é necessário para algumas funcionalidades gráficas do Matplotlib
3. A estrutura do Keras 3.x é compatível com TensorFlow 2.18.0
4. Use sempre o ambiente virtual ativado para executar os projetos