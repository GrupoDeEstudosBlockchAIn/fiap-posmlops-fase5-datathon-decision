import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "dataset_processado.csv")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "features_treinamento.pkl")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")
FEATURE_PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "feature_pipeline.pkl")

CONST_DATABASE_VAGAS = 'vagas.json'
CONST_DATABASE_APPLICANTS = 'applicants.json'
CONST_DATABASE_PROSPECTS = 'prospects.json'