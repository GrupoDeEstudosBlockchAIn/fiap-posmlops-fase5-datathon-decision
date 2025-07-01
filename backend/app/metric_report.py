import os
import joblib
import pandas as pd
import datetime
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from app.feature_engineering import FeatureEngineer
from app.model_utils import construir_dataframe_supervisionado

# Configuração de logging
logger = logging.getLogger(__name__)

def gerar_metric_report():
    MODEL_PATH = "models/model.pkl"
    PIPELINE_PATH = "models/feature_pipeline.pkl"
    OUTPUT_DIR = "metric_reports"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Carregando dados e reconstruindo pipeline de features...")
    df = construir_dataframe_supervisionado()
    X = df[['cv', 'nivel_ingles', 'area_atuacao']]
    y = df['match']

    fe = FeatureEngineer()
    fe.pipeline = joblib.load(PIPELINE_PATH)
    X_transformed = fe.transform(X)

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_transformed)
    y_proba = model.predict_proba(X_transformed)[:, 1]

    acc = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)
    relatorio = classification_report(y, y_pred, output_dict=True)
    matriz_conf = confusion_matrix(y, y_pred)

    logger.info("Gerando visualizações de métricas...")

    plt.figure(figsize=(6, 4))
    sns.heatmap(matriz_conf, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    matriz_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(matriz_path)
    plt.close()
    logger.info(f"Matriz de confusão salva em {matriz_path}")

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"model_metric_report_{now}.html"
    filepath = os.path.join(OUTPUT_DIR, filename)

    logger.info("Salvando relatório HTML com métricas do modelo...")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="pt-br">
<head><meta charset="UTF-8"><title>Relatório de Métricas - Modelo</title>
<style>body {{ font-family: Arial; background-color: #f9f9f9; margin: 40px; color: #333; }}
.metric-box {{ background: #e6f2ff; padding: 10px; border-left: 5px solid #005a9c; margin-bottom: 10px; }}
table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}</style>
</head><body>
<h1>Relatório de Métricas do Modelo</h1><p>Data: {now}</p>
<div class="metric-box"><strong>Acurácia:</strong> {acc:.4f}</div>
<div class="metric-box"><strong>ROC AUC:</strong> {roc_auc:.4f}</div>
<h2>Relatório de Classificação</h2>
<table><tr><th>Classe</th><th>Precisão</th><th>Revocação</th><th>F1-score</th><th>Suporte</th></tr>""")

        for label, metrics in relatorio.items():
            if label in ["accuracy", "macro avg", "weighted avg"]:
                continue
            f.write(f"<tr><td>{label}</td><td>{metrics['precision']:.2f}</td><td>{metrics['recall']:.2f}</td><td>{metrics['f1-score']:.2f}</td><td>{metrics['support']}</td></tr>")

        f.write(f"""</table><h2>Matriz de Confusão</h2>
<img src="confusion_matrix.png" alt="Matriz de Confusão" width="400"/>
<hr/><p>Relatório gerado automaticamente pelo Decision AI.</p></body></html>""")

    logger.info(f"Relatório HTML salvo com sucesso: {filepath}")
