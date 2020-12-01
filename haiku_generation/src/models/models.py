from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    SimpleRNN, 
    LSTM,
    GRU,
    Embedding,
    Dropout,
    Dense
)


def get_lstm_model(num_words, X, embedding_dim=20,
                   num_units=100, dropout=0.1, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, input_length=len(X[0])))
    model.add(LSTM(num_units))
    model.add(Dropout(dropout))

    # softmax for probability of most likely *next* word.
    model.add(Dense(num_words, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def get_rnn_model(num_words, X, embedding_dim=20,
                  num_units=100, dropout=0.1, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, input_length=len(X[0])))
    model.add(SimpleRNN(num_units))
    model.add(Dropout(dropout))

    # softmax for probability of most likely *next* word.
    model.add(Dense(num_words, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def get_gru_model(num_words, X, embedding_dim=20,
                  num_units=100, dropout=0.1, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, input_length=len(X[0])))
    model.add(GRU(num_units))
    model.add(Dropout(dropout))

    # softmax for probability of most likely *next* word.
    model.add(Dense(num_words, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
