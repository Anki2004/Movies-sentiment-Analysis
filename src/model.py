from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM


def create_model(max_words, max_len, embedding_dim = 128, lstm_units = 64):
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length = max_len),
        LSTM(lstm_units, return_sequences=True),
        LSTM(lstm_units // 2),
        Dense(1, activation = 'sigmoid')
    ]   
    )
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model