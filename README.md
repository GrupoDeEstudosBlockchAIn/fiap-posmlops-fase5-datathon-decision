
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
│   └── test_model.py                    
│
├── main.py                              # Script principal
├── requirements.txt                     # Bibliotecas de instalação
├── Dockerfile                           # Container de execução
├── .gitignore                           # Arquivo e pastas ignorados pelo git
├── README.md                            # Este arquivo
└── Documentacao_Decision_IA.pdf         # Documentação do Projeto IA