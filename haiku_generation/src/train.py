import keras.utils as utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from haiku_generation.src.models.haiku_lstm import get_lstm_model
from haiku_generation.src.dataloaders.haikus_dataloader import HaikuDataset

# sample haiku data
# haiku_data = ['rectory roofers their ladder take them higher',
#             'summer cabin the ants do the dish',
#             'lagoon at sunrise the shadow chase its pelican',
#             'barren trees even the tiniest twig embraced by the mist',
#             'windfall apples bees tango to a waltz',
#             'that foghorn bawl calf separated from its mother',
#             'spray art the cherry knows no bounds',
#             'cold night the escalating heat of my old notebook',
#             'fly fishing my thoughts untangle',
#             'arrows of geese the puppy chases leaves on the wind',
#             'absently choking on a sliver of lemon near total eclipse',
#             'twilight a full moon between fractured branches',
#             'at the tulip festival she holds a bouquet of dandelions',
#             'random rain splatters on the sidewalk polka dots',
#             'yukon hike water sloshes to the beat of the bear bell',
#             'spring rain the dog coat finer than mine',
#             'winter solstice a furlong of a dream curls inside me',
#             'autumn sky swaying wheat field shaping the wind',
#             'blue sky morning the hospital huffs the only clouds',
#             'still life lilies residents sitting in the nursing home lounge',
#             'years end only the sound of mouse clicks from every desk',
#             'high tide wild blackberry canes overtake the shore',
#             'ocean swim not knowing whats beneath me',
#             'stocking empty to the toe the spray of citrus',
#             'autumns last month still awaiting my inheritance',
#             'after the rain only a few petals left with the rose',
#             'empty restaurant all the tables candlelit',
#             'spring again a songbirds paean at dawn',
#             'the busker buttons his collar',
#             'rolling thunder the bass on the car stereo passing by']


def get_n_gram_text_sequences(tokenizer, text_data):
    tokenizer.fit_on_texts(text_data)

    sequences = []

    for haiku in text_data:
        haiku_token = tokenizer.texts_to_sequences([haiku])[0]

        for j in range(1, len(haiku_token)):
            sequences.append(haiku_token[:j+1])

    return sequences, len(tokenizer.word_index)+1


def pad_text_sequences(num_words, sequences):
    input_sequences = pad_sequences(sequences)

    texts, labels = input_sequences[:,:-1], input_sequences[:,-1]
    labels = utils.to_categorical(labels, num_classes=num_words)

    return texts, labels


def haiku_generation(start_word, total_words, tokenizer, X, lstm_model):
    while len(start_word.split()) != total_words:
        tokens = pad_sequences([tokenizer.texts_to_sequences([start_word])[0]], maxlen=len(X[0]))

        pred_ind = lstm_model.predict_classes(tokens, verbose=1)

        for key, value in tokenizer.word_index.items():
            if value == pred_ind:
                start_word = start_word + ' ' + key
                break

    return start_word


if __name__ == '__main__':
    haikus = HaikuDataset('/home/peasant98/Documents/haiku-generation/haiku_generation/datasets/all_haiku.csv', is_kaggle=True)
    haiku_data = haikus.get_all_poems()[:1000]
    tokenizer = Tokenizer()

    sequences, num_words = get_n_gram_text_sequences(tokenizer, haiku_data)

    X, Y = pad_text_sequences(num_words, sequences)

    lstm_model = get_lstm_model(num_words=num_words, X=X)
    lstm_model.fit(X, Y, epochs=200, batch_size=32, verbose=1)

    haiku = haiku_generation("wind", 9, tokenizer, X, lstm_model)
    print(haiku)
