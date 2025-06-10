# main.py

from app.data_collector import DataCollector
from app.data_preprocessing import preprocess_data
from app.feature_engineering import engineer_features
from app.model_training import train_model
from app.model_evaluation import evaluate_model

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
    model = train_model(features, labels)

    # 5️⃣ Avaliação do modelo
    evaluation_report = evaluate_model(model, features, labels)
    
    print("🏁 Processo concluído com sucesso!")
    print(evaluation_report)

if __name__ == "__main__":
    main()
