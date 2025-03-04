```markdown
# Repositório de Redes Neurais com Keras

Este repositório contém implementações de redes neurais utilizando a biblioteca Keras para resolver atividades da disciplina de Redes Neurais, ministrada pelo professor Petronio Candido de Lima e Silva. Os códigos abordam diferentes problemas de classificação, como a predição de dígitos, diagnóstico de câncer de mama e classificação de flores Iris.

## Projetos

### 1. Predição de Dígitos (MNIST)
- **Dataset**: MNIST (dígitos manuscritos)
- **Objetivo**: Classificar dígitos de 0 a 9.
- **Técnicas Utilizadas**:
  - Divisão dos dados em 70% para treino e 30% para teste.
  - Criação de uma rede neural com Keras.
  - Apresentação da matriz de confusão para os dados de teste.

### 2. Predição de Câncer de Mama
- **Dataset**: Breast Cancer Wisconsin Diagnostic
- **Objetivo**: Classificar tumores como benignos ou malignos.
- **Técnicas Utilizadas**:
  - Conversão da variável de diagnóstico em variáveis binárias (one-hot encode).
  - Divisão dos dados em 70% para treino e 30% para teste.
  - Criação de uma rede neural com Keras.
  - Apresentação da matriz de confusão para os dados de teste.

### 3. Classificação de Flores Iris
- **Dataset**: Iris
- **Objetivo**: Classificar flores em três categorias: Setosa, Versicolor e Virginica.
- **Técnicas Utilizadas**:
  - Divisão dos dados em 70% para treino e 30% para teste.
  - Criação de uma rede neural com Keras.
  - Atingir pelo menos 95% de acurácia.

## Bibliotecas Utilizadas

- **NumPy**: Para manipulação de arrays e operações matemáticas.
- **Matplotlib**: Para visualização de dados e criação de gráficos.
- **Scikit-learn**: Para pré-processamento de dados, divisão de datasets e métricas de avaliação.
- **Keras**: Para construção e treinamento de redes neurais.
- **Seaborn**: Para visualização de matrizes de confusão.
- **Pandas**: Para manipulação de dados em formato de tabela.
- **UCIMLRepo**: Para carregar datasets da UCI Machine Learning Repository.

## Instalação

Para executar os códigos deste repositório, você precisará instalar as seguintes bibliotecas:

```bash
pip install numpy matplotlib scikit-learn keras seaborn pandas ucimlrepo
```

## Executando os Códigos

Cada projeto está organizado em um script Python separado. Para executar um dos projetos, basta rodar o script correspondente:

```bash
python predicao_digitos.py
python predicao_cancer_mama.py
python classificacao_iris.py
```

## Estrutura do Repositório

- `predicao_digitos.py`: Código para a predição de dígitos.
- `predicao_cancer_mama.py`: Código para a predição de câncer de mama.
- `classificacao_iris.py`: Código para a classificação de flores Iris.
- `Dataset_MNIST/`: Diretório contendo gráficos e resultados do projeto de predição de dígitos.
- `Dataset_breast_cancer/`: Diretório contendo gráficos e resultados do projeto de predição de câncer de mama.
- `Dataset_Iris/`: Diretório contendo gráficos e resultados do projeto de classificação de flores Iris.

## Resultados

Cada projeto gera gráficos e matrizes de confusão que são salvos em seus respectivos diretórios. Esses resultados ajudam a visualizar o desempenho do modelo e a entender melhor o processo de classificação.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorar este repositório.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
```

### Explicação do README

1. **Título e Descrição**: O título e a descrição fornecem uma visão geral do repositório e dos projetos contidos nele.
2. **Projetos**: Cada projeto é descrito brevemente, incluindo o dataset utilizado, o objetivo e as técnicas aplicadas.
3. **Bibliotecas Utilizadas**: Lista as bibliotecas necessárias para executar os códigos.
4. **Instalação**: Fornece o comando para instalar as dependências necessárias.
5. **Executando os Códigos**: Explica como executar cada script.
6. **Estrutura do Repositório**: Descreve a organização dos arquivos e diretórios.
7. **Resultados**: Explica onde os resultados (gráficos e matrizes de confusão) são salvos.
8. **Contribuições**: Convida contribuições da comunidade.
9. **Licença**: Informa sobre a licença do projeto.
