from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_match_endpoint():
    sample_input = {
        "id_vaga": "5185",
        "nome": "Maria Souza",
        "cv": "Desenvolvedora Python com conhecimento em APIs, banco de dados e Git.",
        "nivel_ingles": "Avançado",
        "area_atuacao": "Tecnologia da Informação"
    }

    response = client.post("/match", json=sample_input)
    assert response.status_code == 200

    json_response = response.json()
    assert "match" in json_response
    assert "score" in json_response
    assert "perfil_recomendado" in json_response
