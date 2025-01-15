from tensorflow.keras.models import load_model
from data.preprocess_data import load_and_preprocess_data
import numpy as np

def run_inference_pipeline(model_path, tokenizer, input_texts, max_len=100):
    model = load_model(model_path)
    sequences = tokenizer.texts_to_sequences(input_texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    predictions = model.predict(padded_sequences)
    return ["Positive" if pred > 0.5 else "Negative" for pred in predictions]
