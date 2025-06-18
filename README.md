# 📊 Projeto: Inteligência Artificial para Recrutamento - Decision Datathon 🚀

## 📍 Sobre o Projeto

Este projeto foi desenvolvido como parte do **Datathon Decision - Pós Tech FIAP**, com o objetivo de aplicar Inteligência Artificial para otimizar o processo de recrutamento e seleção de candidatos(as) na empresa **Decision**, especializada em serviços de bodyshop e alocação de profissionais de TI.

O foco principal é criar um sistema de **machine learning** que avalie o "match" entre candidatos e vagas com base em dados históricos, oferecendo uma solução escalável, testável e pronta para deployment via API.

---

## 🏢 Contexto do Problema

A **Decision** enfrenta os seguintes desafios:

* **Falta de padronização nas entrevistas**, causando perda de informações importantes.
* **Dificuldade em identificar o real engajamento** dos candidatos.
* **Baixa precisão no match entre perfis e vagas**, aumentando o tempo de contratação.

O objetivo da solução é reduzir o tempo de seleção, aumentar a assertividade nas contratações e fornecer suporte com base em dados concretos.

---

## 🛠️ Arquitetura da Solução

### 📌 Pipeline de Machine Learning:

1. **Data Collection:**
   Importação de dados de **candidatos**, **prospects** e **vagas**.

2. **Pré-processamento:**
   Limpeza de dados, tratamento de nulos, transformação de tipos e garantia de integridade de schema.

3. **Engenharia de Features:**
   Criação de novas variáveis (idade, experiência, faixa salarial, etc.), codificação de categorias e escalonamento de variáveis numéricas.

4. **Treinamento do Modelo:**
   Uso de um **MLP (Perceptron Multi-Camadas)** implementado em **Keras/TensorFlow**.
   O modelo e o scaler são salvos para uso posterior na API.

5. **Avaliação do Modelo:**
   Geração de relatórios com métricas como **Accuracy**, **F1-Score** e **Matriz de Confusão**.

---

## 🌐 API - Deployment

### ✅ Ferramentas utilizadas:

* **FastAPI** para construção da API.
* **Docker** para empacotamento e deploy do serviço.
* **Joblib / Pickle** para serialização do modelo e scaler.

### ✅ Endpoints disponíveis:

| Método | Endpoint   | Função                                                     |
| ------ | ---------- | ---------------------------------------------------------- |
| POST   | `/predict` | Recebe dados de um candidato e retorna a previsão de match |

### ✅ Teste da API:

Testado localmente via **Postman** e com **testes automatizados** com `pytest`.

---

## 🧪 Testes Unitários

Os seguintes componentes possuem testes:

* ✅ Pré-processamento
* ✅ Engenharia de Features
* ✅ Treinamento do Modelo
* ✅ API (endpoint `/predict`)

Os testes garantem a qualidade e robustez da solução.

---

## 🐳 Docker

### Build da imagem:

```bash
docker build -t decision-ml-api .
```

### Rodando o container:

```bash
docker run -p 8000:8000 decision-ml-api
```

A API ficará disponível localmente em:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 📂 Estrutura do Projeto

```
fiap-posmlops-fase5-datathon-decision
│
├── .github/                             # Pipeline do projeto
│   ├── workflows
│       ├── pipeline.yaml                
│
├── data/                                # Dados brutos e pré-processados
│   ├── applicants
│	│	├── applicants.json
│   │
│	├── prospects
│   │   ├── prospects.json
│   │
│   └── vagas
│       ├── vagas.json
│
├── metrics/                             # Relatórios de avaliação do modelo
│   └── metric_report.html
│
├── models/                              # Modelos treinados
│
├── logs/                                # Logs do projeto
│   
├── app/                                 # Módulos do projeto
│   ├── api.py
│   ├── data_collector.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── metric_report.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── tests/                               # Testes unitários da API e modelo
│   ├── test_api.py                      
│   ├── test_model.py 
│   ├── test_data_preprocessing.py
│   └── test_feature_engineering.py                   
│
├── main.py                              # Script principal
├── requirements.txt                     # Bibliotecas de instalação
├── Dockerfile                           # Container de execução
├── .gitignore                           # Arquivo e pastas ignorados pelo git
├── README.md                            # Este arquivo
└── Documentacao_Decision_IA.pdf         # Documentação do Projeto IA
```

---

## 📈 Resultados Esperados

A API permite que a Decision envie os dados de novos candidatos e receba **previsões automatizadas de match**, otimizando o processo de triagem de currículos e entrevistas.

---

## ✅ Entregáveis

✔️ Código-fonte documentado
✔️ API funcionando e testada
✔️ Dockerfile para deployment
✔️ Testes unitários
✔️ Relatórios de métricas
✔️ Documentação técnica

---

## 🎥 Vídeo Explicativo

O vídeo com a explicação da solução pode ser acessado em:
👉 \[Inserir link do vídeo após upload]

---

## 🚀 Tecnologias Utilizadas

* Python
* FastAPI
* TensorFlow / Keras
* Scikit-learn
* Pandas / Numpy
* Docker
* Pytest

---

## 📌 Autoria

Projeto desenvolvido por:
**\[Seu Nome]**
Para o **Datathon Pós Tech FIAP – Decision**

---

