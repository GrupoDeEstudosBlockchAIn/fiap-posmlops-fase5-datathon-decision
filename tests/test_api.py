from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_predict_endpoint():
    # Dados fake seguindo o input esperado
    sample_input = {
        "idade": 30,
        "experiencia": 5,
        "salario": 7000,
        "cargo": "Dev"
    }

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200

    json_response = response.json()
    assert "prediction" in json_response
    assert json_response["prediction"] in [0, 1]  # Output esperado binário
