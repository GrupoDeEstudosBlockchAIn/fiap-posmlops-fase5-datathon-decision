import pandas as pd
from app.model_training import train_model
import numpy as np

def test_train_model():
    # Criando dados fake
    features = pd.DataFrame(np.random.rand(100, 3), columns=['f1', 'f2', 'f3'])
    labels = np.random.randint(0, 2, size=100)

    model, scaler, _ = train_model(features, labels)

    # Testa se o modelo é um objeto Keras
    assert model is not None
    assert hasattr(model, 'predict')

    # Testa se o scaler é do sklearn
    from sklearn.preprocessing import StandardScaler
    assert isinstance(scaler, StandardScaler)
