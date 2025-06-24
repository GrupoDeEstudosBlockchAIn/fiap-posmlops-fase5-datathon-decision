import json
import pandas as pd
from pathlib import Path

def carregar_json(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        return json.load(f)

def coletar_dados(applicants_path, prospects_path, vagas_path):
    
    try:

        print(f"Carregando dados de {applicants_path}, {prospects_path} e {vagas_path}")

        # Carregar os JSONs
        applicants = carregar_json(applicants_path)
        prospects = carregar_json(prospects_path)
        vagas = carregar_json(vagas_path)

        dados = []

        for vaga_id, vaga_data in prospects.items():
            for candidato in vaga_data.get("prospects", []):
                codigo_candidato = candidato.get("codigo")
                status = candidato.get("situacao_candidado")
                contratado = 1 if "contratado" in status.lower() else 0

                dados_applicant = applicants.get(codigo_candidato, {})
                if not dados_applicant:
                    continue  # Skip candidatos que não estão no applicants.json

                cv = dados_applicant.get("cv_pt", "")
                ingles = dados_applicant.get("formacao_e_idiomas", {}).get("nivel_ingles", "")
                area = dados_applicant.get("informacoes_profissionais", {}).get("area_atuacao", "")
                nome = dados_applicant.get("informacoes_pessoais", {}).get("nome", "")

                dados.append({
                    "codigo": codigo_candidato,
                    "nome": nome,
                    "nivel_ingles": ingles,
                    "area_atuacao": area,
                    "cv": cv,
                    "contratado": contratado
                })

        print(f"[OK] Dados coletados com sucesso! Total de candidatos: {len(dados)}")
        # Criar DataFrame
        return pd.DataFrame(dados)
    
    except FileNotFoundError as e:
        print(f"Erro ao carregar arquivos JSON: {e}")
        return pd.DataFrame()  # Retorna um DataFrame vazio em caso de erro


