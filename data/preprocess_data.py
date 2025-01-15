import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_preprocess_data(data_path, num_words=10000, max_len=100):
    with open(data_path, 'r', encoding='utf-8') as f:
        texts = []
        labels = []
        for line in f:
            label, text = line.strip().split('\t')
            texts.append(text)
            labels.append(int(label))

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    labels = np.array(labels)