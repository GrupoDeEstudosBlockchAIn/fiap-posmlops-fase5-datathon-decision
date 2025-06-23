# app/model_training.py

import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_model(features, labels, model_save_path='saved_model/mlp_model.keras'):
    """
    Treina um modelo MLP com TensorFlow/Keras e salva o modelo treinado.
    O pré-processamento (escalonamento e encoding) deve ser feito antes de chamar esta função.
    """

    # Garantir que os labels estão em formato NumPy array
    if isinstance(labels, (list, pd.Series)):
        labels = np.array(labels)

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    logging.info(f"✅ Shape Treino: {X_train.shape} | Shape Teste: {X_test.shape}")

    # Definindo arquitetura da MLP
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Callback para salvar o melhor modelo
    checkpoint_path = model_save_path.replace('.keras', '_best.keras')
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True)

    # Treinamento
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint_cb]
    )

    # Criar diretório se não existir
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Salvar o modelo final
    model.save(model_save_path)
    logging.info(f"✅ Modelo final salvo em: {model_save_path}")

    return model, history, (X_test, y_test)
