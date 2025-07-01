import streamlit as st
import requests

st.set_page_config(page_title="Decision AI - Match de Candidatos", layout="centered")
st.title("🤖 Otimizador de Entrevistas - Decision AI")

st.markdown("""
Preencha os dados abaixo para avaliar o nível de compatibilidade com as vagas da Decision.
""")

# Formulário
with st.form("form_candidato"):
    nome = st.text_input("Nome completo")
    email = st.text_input("Email de contato")
    area = st.selectbox("Área de atuação", ["TI", "Financeiro", "RH", "Vendas", "Outros"])
    ingles = st.selectbox("Nível de Inglês", ["Básico", "Intermediário", "Avançado", "Fluente"])
    cv_texto = st.text_area("Cole aqui o texto do seu currículo (copie de um PDF ou Word)", height=300)
    enviar = st.form_submit_button("🔍 Avaliar Match")

# Enviar dados para API
if enviar:
    if not cv_texto.strip():
        st.error("⚠️ O campo de currículo não pode estar vazio.")
    else:
        st.info("Enviando dados para avaliação...")
        try:
            payload = {
                "cv": cv_texto,
                "nivel_ingles": ingles,
                "area_atuacao": area
            }
            response = requests.post("http://localhost:8000/predict", json=payload)
            if response.status_code == 200:
                resultado = response.json()
                if resultado["match"]:
                    st.success(f"✅ Candidato compatível com {resultado['score']*100:.1f}% de match!")
                else:
                    st.warning(f"❌ Baixa compatibilidade ({resultado['score']*100:.1f}%)")
                st.markdown(f"**Perfil Recomendado:** {resultado['perfil_recomendado']}")
            else:
                st.error(f"Erro {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Erro na comunicação com a API: {str(e)}")
