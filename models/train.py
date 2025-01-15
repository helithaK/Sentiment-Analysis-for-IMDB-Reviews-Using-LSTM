import os
from models.model import create_lstm_model
from data.preprocess_data import load_and_preprocess_data

def train_model(data_path, save_model_path):
    X, y, tokenizer = load_and_preprocess_data(data_path)
    model = create_lstm_model(input_dim=len(tokenizer.word_index) + 1)

    history = model.fit(
        X, y,
        batch_size=32,
        epochs=10,
        validation_split=0.2
    )

    model.save(save_model_path)
    return history