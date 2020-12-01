import pickle
from matplotlib.pyplot import plot
# this is needed to avoid an error, even though the import isn't used
# (don't delete the below line)
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import collections
import nltk
import keras.utils as utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from tensorflow import keras

from haiku_generation.src.models.models import (
    get_rnn_model,
    get_lstm_model,
    get_gru_model
)
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
    total_words = len(tokenizer.word_index)+1
    sequences = []

    for haiku in text_data:
        haiku_token = tokenizer.texts_to_sequences([haiku])[0]

        for j in range(1, len(haiku_token)):
            sequences.append(haiku_token[:j+1])

    # [2, 31, 5, 62, 33]
    # sequences = [[2],
    #             [2, 31],
    #             [2, 31, 5],
    #             [2, 31, 5, 62],
    #             [2, 31, 5, 62, 33], ...]

    return sequences, total_words


def pad_text_sequences(num_words, sequences):
    input_sequences = pad_sequences(sequences)

    # [2, 31, 5, 62, 33]
    # sequences = [[0, 0, 0, 0, 2],
    #             [0, 0, 0, 2, 31],
    #             [0, 0, 2, 31, 5],
    #             [0, 2, 31, 5, 62],
    #             [2, 31, 5, 62, 33], ...]

    # array[:, :-1] for all rows, keep all columns excpet the last column
    # array[:, -1] for all rows, keep only the last column
    texts, labels = input_sequences[:, :-1], input_sequences[:, -1]
    labels = utils.to_categorical(labels, num_classes=num_words)

    # text [0, 0, 0, 0] label [2]
    # text [0, 0, 0, 2] label [31]
    # text [0, 0, 2, 31] label [5]
    # text [0, 2, 31, 5] label [62]
    # text [2, 31, 5, 62] label [33]

    return texts, labels


def haiku_generation(haiku, total_words, tokenizer, X, model):
    while len(haiku.split()) != total_words:
        tokens = pad_sequences([tokenizer.texts_to_sequences([haiku])[0]], maxlen=len(X[0]))

        # predict the next word's index based on the current haiku
        pred_ind = model.predict_classes(tokens, verbose=1)

        # get the word corresponding to the predicted index and add to the current haiku
        for key, value in tokenizer.word_index.items():
            if value == pred_ind:
                haiku = haiku + ' ' + key
                break

    return haiku

def evaluation(haiku_data, generated_haiku):
    #using preplexity to evaluate
    #evaluation code from https://stackoverflow.com/questions/33266956/nltk-package-to-estimate-the-unigram-perplexity
    tokens_list = []
    for haiku in haiku_data:
        tokens_list.append(nltk.word_tokenize(haiku))
    tokens = [item for sublist in tokens_list for item in sublist]
    # here you construct the unigram language model
    def unigram(tokens):
        model = collections.defaultdict(lambda: 0.01)
        for f in tokens:
            try:
                model[f] += 1
            except KeyError:
                model[f] = 1
                continue
        N = float(sum(model.values()))
        for word in model:
            model[word] = model[word] / N
        return model

    # computes perplexity of the unigram model on a testset
    def perplexity(testset, model):
        testset = testset.split()
        perplexity = 1
        N = 0
        for word in testset:
            N += 1
            perplexity = perplexity * (1 / model[word])
        perplexity = pow(perplexity, 1 / float(N))
        return perplexity

    model = unigram(tokens)
    print("Perplexity score:")
    print(perplexity(generated_haiku, model))

def run_ablative_experiments(haiku_data, tokenizer,
                             model_type='lstm', total_training_amount=200,
                             num_epochs=200, batch_size=32,
                             embedding_dim=20, num_units=100, dropout=0.1,
                             optimizer='adam', lr=0.001, validation_split=0.2,
                             train_test_split_val=0.2, dict_label='standard'):
    """
    run ablative experiments with various parameters.
    """
    haiku_data = haiku_data[:total_training_amount]

    sequences_full, num_words_full = get_n_gram_text_sequences(tokenizer, haiku_data)

    X_full, Y_full = pad_text_sequences(num_words_full, sequences_full)

    X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=train_test_split_val)
    if model_type == 'rnn':
        model = get_rnn_model(num_words=num_words_full, X=X_full,
                              embedding_dim=embedding_dim, num_units=num_units,
                              dropout=dropout, optimizer_name=optimizer,
                              lr=lr)

    elif model_type == 'lstm':
        model = get_lstm_model(num_words=num_words_full, X=X_full,
                               embedding_dim=embedding_dim, num_units=num_units,
                               dropout=dropout, optimizer_name=optimizer,
                               lr=lr)

    elif model_type == 'gru':
        model = get_gru_model(num_words=num_words_full, X=X_full,
                              embedding_dim=embedding_dim, num_units=num_units,
                              dropout=dropout, optimizer_name=optimizer,
                              lr=lr)
    else:
        return [], [], []

    history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=validation_split)
    eval_results = model.evaluate(X_test, Y_test, verbose=1)
    final_eval_result_dict = dict(zip(model.metrics_names, eval_results))
    history.history['label'] = dict_label
    return final_eval_result_dict, history.history, X_full, model, haiku_data


def explainability_fnc():
    pass


def get_tokenized_data(haiku_data, total_training_amount=200, train_test_split_val=0.2):
    haiku_data = haiku_data[:total_training_amount]

    sequences_full, num_words_full = get_n_gram_text_sequences(tokenizer, haiku_data)

    X_full, Y_full = pad_text_sequences(num_words_full, sequences_full)

    return X_full


def get_embedding_layer(start_word, total_words, model, X):
    embeddings_list = []
    words_list = [start_word]
    while len(start_word.split()) != total_words:
        tokens = pad_sequences([tokenizer.texts_to_sequences([start_word])[0]], maxlen=len(X[0]))

        inp = model.input
        outputs = [layer.output for layer in model.layers]
        functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
        pred_ind = model.predict_classes(tokens, verbose=1)

        for key, value in tokenizer.word_index.items():
            if value == pred_ind:
                start_word = start_word + ' ' + key
                words_list.append(key)
                break
        # Testing
        layer_outs = [func([tokens, 1.]) for func in functors]
        embedding_layer = layer_outs[0]
        layer_shape = embedding_layer[0].shape
        embedding_layer = embedding_layer[0].reshape((layer_shape[1], layer_shape[2]))
        embeddings_list.append(embedding_layer[-1])
    X_embedded = TSNE(n_components=2).fit_transform(np.array(embeddings_list))

    return X_embedded, words_list[:-1]


def plot_embeddings(vectors, words_list):
    if vectors.shape[-1] == 2:
        for idx, vector in enumerate(vectors):
            plt.scatter(x=vector[0], y=vector[1], label=words_list[idx])
        plt.legend()
        plt.show()
    elif vectors.shape[-1] == 3:
        pass    
    else:
        # don't plot anything here.
        pass


def plot_losses(train_dict_list, test_dict_list):
    """
    plots the losses from the keras training history
    """
    for train_dict in train_dict_list:
        train_loss = train_dict['loss']
        train_acc = train_dict['acc']

        val_loss = train_dict['val_loss']
        val_acc = train_dict['val_acc']

        plt.plot(train_loss, label=train_dict['label'])
    plt.title('Plot of training losses for Sequence based models')
    plt.xlabel('Epoch')
    plt.ylabel('CE Loss')
    plt.legend()
    plt.show()

    for train_dict in train_dict_list:
        train_acc = train_dict['acc']
        plt.plot(train_acc, label=train_dict['label'])

    plt.title('Plot of training accuracy for Sequence based models')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    haikus = HaikuDataset('/home/peasant98/Documents/haiku-generation/haiku_generation/datasets/all_haiku.csv', is_kaggle=True)
    haiku_data = haikus.get_all_poems()
    np.random.shuffle(haiku_data)
    tokenizer = Tokenizer()

    # experiment time.
    # compare different model types
    # train_dict_list = []
    # test_dict_list = []
    # for model_type in ["rnn", "lstm", "gru"]:
    #     test, train, tokenized_data, model = run_ablative_experiments(haiku_data=haiku_data, tokenizer=tokenizer,
    #                                                                   model_type=model_type, dict_label=model_type)
    #     train_dict_list.append(train)
    #     test_dict_list.append(test)
    # plot_losses(train_dict_list, test_dict_list)
    # vary batch size
    train_dict_list = []
    test_dict_list = []
    # for batch_size in [8, 16, 32, 64]:
    #     test, train, tokenized_data, model = run_ablative_experiments(haiku_data=haiku_data, tokenizer=tokenizer,
    #                                                                   batch_size=batch_size,
    #                                                                   dict_label=f'Batch size {batch_size}')
    #     train_dict_list.append(train)
    #     test_dict_list.append(test)
    # plot_losses(train_dict_list, test_dict_list)

    # for embedding_dim in [3, 10, 20, 50]:
    #     test, train, tokenized_data, model = run_ablative_experiments(haiku_data=haiku_data, tokenizer=tokenizer,
    #                                                                   embedding_dim=embedding_dim,
    #                                                                   dict_label=f'Embedding {embedding_dim}')
    #     train_dict_list.append(train)
    #     test_dict_list.append(test)

    # for num_units in [50, 100, 150, 200]:
    #     test, train, tokenized_data, model = run_ablative_experiments(haiku_data=haiku_data, tokenizer=tokenizer,
    #                                                                   num_units=num_units,
    #                                                                   dict_label=f'Num units {num_units}')
    #     train_dict_list.append(train)
    #     test_dict_list.append(test)

    # for dropout in [0.1, 0.3, 0.5, 0.9]:
    #     test, train, tokenized_data, model = run_ablative_experiments(haiku_data=haiku_data, tokenizer=tokenizer,
    #                                                                   dropout=dropout,
    #                                                                   dict_label=f'Dropout {dropout}')
    #     train_dict_list.append(train)
    #     test_dict_list.append(test)
    # for optimizer in ['adam', 'rmsprop', 'sgd', 'adadelta', 'nadam']:
    #     test, train, tokenized_data, model = run_ablative_experiments(haiku_data=haiku_data, tokenizer=tokenizer,
    #                                                                   optimizer=optimizer,
    #                                                                   dict_label=f'Optimizer {optimizer}')
    #     train_dict_list.append(train)
    #     test_dict_list.append(test)

    # for lr in [0.0001, 0.001, 0.01, 0.1, 1]:
    #     test, train, tokenized_data, model = run_ablative_experiments(haiku_data=haiku_data, tokenizer=tokenizer,
    #                                                                   lr=lr,
    #                                                                   dict_label=f'Learning Rate {lr}')
    #     train_dict_list.append(train)
    #     test_dict_list.append(test)
    # plot_losses(train_dict_list, test_dict_list)
    test, train, tokenized_data, model, haiku_data = run_ablative_experiments(haiku_data=haiku_data, tokenizer=tokenizer)

    # # save
    # model.save('haiku_lstm')
    # with open('tokenized_data.npy', 'wb') as f:
    #     np.save(f, np.array(tokenized_data))

    # pickle.dump(haiku_data, open( "haiku_data.p","wb"))

    # load
    model = keras.models.load_model('haiku_lstm')
    with open('tokenized_data.npy', 'rb') as f:
        tokenized_data = np.load(f)

    haiku_data = pickle.load(open( "haiku_data.p","rb"))
    tokenizer.fit_on_texts(haiku_data)

    start_words = ['night', 'morning', 'twilight', 'water', 'snow', 'I', 'love', 'king']
    for word in start_words:
        vecs, wl = get_embedding_layer(word, 9, model, tokenized_data)
        plot_embeddings(vecs, wl)
        haiku = haiku_generation(word, 9, tokenizer, tokenized_data, model)
        print('Generated Haiku:', haiku)
    # for word in start_words:
    #     haiku = haiku_generation(word, 9, tokenizer, tokenized_data, model)
    #     print(haiku)
    #     evaluation(haiku_data, haiku)
        print(haiku)
        evaluation(haiku_data, haiku)
