from app.data_collector import DataCollector
from app.data_preprocessing import preprocess_data
from app.feature_engineering import engineer_features
from app.model_training import train_model
from app.model_evaluation import evaluate_model
from app.metric_report import generate_metric_report  # <-- Incluído

def main():
    # 1️⃣ Coleta de dados
    data_collector = DataCollector()
    applicants_df = data_collector.load_applicants_data()
    prospects_df = data_collector.load_prospects_data()
    vagas_df = data_collector.load_vagas_data()

    # 2️⃣ Pré-processamento
    preprocessed_data = preprocess_data(applicants_df, prospects_df, vagas_df)

    # 3️⃣ Engenharia de features
    features, labels = engineer_features(preprocessed_data)

    # 4️⃣ Treinamento do modelo
    model, scaler, (X_test, y_test) = train_model(features, labels)

    # 5️⃣ Avaliação do modelo (usando o conjunto de teste)
    predictions, y_proba = evaluate_model(model, (X_test, y_test))

    # 6️⃣ Geração de relatório de métricas
    generate_metric_report(y_test, predictions, y_proba)

    print("🏁 Processo concluído com sucesso!")

if __name__ == "__main__":
    main()
