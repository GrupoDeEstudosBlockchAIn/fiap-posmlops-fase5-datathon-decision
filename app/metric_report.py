# app/metric_report.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from datetime import datetime
import numpy as np
import os
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_proba, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Curva ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_metric_report(y_true, y_pred, y_proba, output_dir='metric_reports'):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    html_filename = f"metric_report_{timestamp}.html"
    html_path = os.path.join(output_dir, html_filename)

    # Paths para os gráficos
    conf_matrix_img = os.path.join(output_dir, f"conf_matrix_{timestamp}.png")
    roc_curve_img = os.path.join(output_dir, f"roc_curve_{timestamp}.png")

    # Plotar e salvar gráficos
    plot_confusion_matrix(y_true, y_pred, conf_matrix_img)
    plot_roc_curve(y_true, y_proba, roc_curve_img)

    # Gerar tabela de métricas (classification report)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_html_table = report_df.to_html(classes='table table-striped table-bordered', border=0)

    # Gerar AUC
    auc = roc_auc_score(y_true, y_proba)

    # HTML Final
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Métricas - {timestamp}</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {{ padding: 20px; }}
            h1, h2 {{ margin-top: 20px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>📊 Relatório de Métricas - Datathon Decision</h1>
        <p><strong>Data de geração:</strong> {timestamp}</p>

        <h2>🔎 Classification Report</h2>
        {report_html_table}

        <h2>🧱 Matriz de Confusão</h2>
        <img src="{os.path.basename(conf_matrix_img)}" alt="Matriz de Confusão">

        <h2>📈 Curva ROC (AUC: {auc:.2f})</h2>
        <img src="{os.path.basename(roc_curve_img)}" alt="Curva ROC">

        <footer class="mt-4">
            <p>Projeto Datathon Decision - Desenvolvido por [Seu Nome]</p>
        </footer>
    </body>
    </html>
    """

    # Salvar o HTML
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Relatório salvo em: {html_path}")
    return html_path
