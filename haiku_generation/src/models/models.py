from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    SimpleRNN, 
    LSTM,
    GRU,
    Embedding,
    Dropout,
    Dense
)


def get_optimizer(name, lr):
    if name == 'adam':
        return keras.optimizers.Adam(learning_rate=lr)
    elif name == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=lr)
    elif name == 'sgd':
        return keras.optimizers.SGD(learning_rate=lr)
    elif name == 'adadelta':
        return keras.optimizers.Adadelta(learning_rate=lr)
    elif name == 'nadam':
        return keras.optimizers.Nadam(learning_rate=lr)


def get_lstm_model(num_words, X, embedding_dim=20,
                   num_units=100, dropout=0.1, optimizer_name='adam',
                   lr=0.001):
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, input_length=len(X[0])))
    model.add(LSTM(num_units))
    model.add(Dropout(dropout))

    # softmax for probability of most likely *next* word.
    model.add(Dense(num_words, activation='softmax'))
    model.summary()
    optimizer = get_optimizer(optimizer_name, lr)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def get_rnn_model(num_words, X, embedding_dim=20,
                  num_units=100, dropout=0.1, optimizer_name='adam',
                  lr=0.001):
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, input_length=len(X[0])))
    model.add(SimpleRNN(num_units))
    model.add(Dropout(dropout))

    # softmax for probability of most likely *next* word.
    model.add(Dense(num_words, activation='softmax'))
    model.summary()
    optimizer = get_optimizer(optimizer_name, lr)


    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def get_gru_model(num_words, X, embedding_dim=20,
                  num_units=100, dropout=0.1, optimizer_name='adam',
                  lr=0.001):
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, input_length=len(X[0])))
    model.add(GRU(num_units))
    model.add(Dropout(dropout))

    # softmax for probability of most likely *next* word.
    model.add(Dense(num_words, activation='softmax'))
    model.summary()
    optimizer = get_optimizer(optimizer_name, lr)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
