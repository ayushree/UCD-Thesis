import pandas as pd
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import backend as K

input_fp = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/cleaned_month_addresses.csv"
input_test_fp = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/cleaned_month_addresses_test.csv"

start_symbol = '*'
stop_symbol = 'STOP'
rare_symbol = '_RARE_'
smoothing_factor_lambda = 0.1
log_prob_of_zero = -1000.0
valid_tags = ['PAON', 'street', 'city', 'district', 'county']
smoothing_factor_v = len(valid_tags)

tags = {}

bigram_tags_count = {}
trigram_tags_count = {}
transitions_probability = {}
state_obs_pair_dict = {}
tag_count = {}
complete_unique_words_dict = {}
start_tag = '<start>'
end_tag = '<end>'
feature_vector_labels = []
word_to_tag_dict = {}

words_vocab = set([])
tags_vocab = set([])


def read_data(file):
    data = pd.read_csv(file)
    return data


def get_word_corpus(word):
    count = 0
    for c in word:
        if c == 'D':
            count += 1
    if count != len(word):
        if word not in complete_unique_words_dict.keys():
            complete_unique_words_dict[word] = 1.0
        else:
            complete_unique_words_dict[word] += 1.0


def tag_train_data(data):
    subset_data = data[['PAON', 'street', 'city', 'district', 'county']]
    for col in subset_data.columns:
        tagged_col = []
        curr_col_list = subset_data[col].values.flatten()
        for i in range(len(curr_col_list)):

            if isinstance(curr_col_list[i], str):
                parts = curr_col_list[i].split(" ")
                tagged_part = ""
                for word in parts:
                    if re.findall("\d+", word):
                        length = len(word)
                        word = 'D' * length
                    get_word_corpus(word)
                    # if 'D'*len(word) != len(word):
                    #     if word
                    tagged_word = word + "/" + col
                    tagged_part = tagged_part + " " + tagged_word
                    if word not in tags.keys() and isinstance(word, str):
                        tags[word] = []
                        tags[word].append(col)
                    elif col not in tags[word]:
                        tags[word].append(col)
                tagged_col.append(tagged_part)
        i = 0
        for index, address in subset_data[col].iteritems():
            if not pd.isna(subset_data.iloc[index][col]):
                subset_data.iloc[index][col] = tagged_col[i]
                i += 1
    return subset_data


def join_cols(df):
    subset_df = df[valid_tags]
    tagged_complete_address = []

    for i in range(len(subset_df)):
        new_address = ""
        for j in range(len(subset_df.columns)):
            if isinstance(subset_df.iloc[i][j], str):
                new_address += subset_df.iloc[i][j]
        new_address = new_address[1:]
        tagged_complete_address.append(new_address)
    tagged_complete_address = pd.DataFrame(tagged_complete_address)
    return tagged_complete_address


def separate_words_and_tags(address):
    address = address.split(" ")
    words = []
    tags = []
    for part in address:
        # print(part)
        word = part.split("/")[0]
        # print(word)
        tag = part.split("/")[1]
        words.append(word)
        tags.append(tag)
    sentences.append(np.array(words))
    sentence_tags.append(np.array(tags))


def build_vocab():
    for sen in train_sentences:
        for word in sen:
            words_vocab.add(word)
    for sen_tag in train_tags:
        for tag in sen_tag:
            tags_vocab.add(tag)


def embed_training_words():
    for sen in train_sentences:
        sen_int = []
        for word in sen:
            try:
                sen_int.append(word2index[word])
            except KeyError:
                sen_int.append(word2index['-OOV-'])
        train_sentences_input.append(sen_int)

    for s in train_tags:
        train_sentences_output.append([tag2index[t] for t in s])


def embed_test_words():
    for sen in test_sentences:
        sen_int = []
        for word in sen:
            try:
                sen_int.append(word2index[word])
            except KeyError:
                sen_int.append(word2index['-OOV-'])
        test_sentences_input.append(sen_int)

    for s in test_tags:
        test_sentences_output.append([tag2index[t] for t in s])


def encode_one_hot(sequences, classes):
    categorical_seq = []
    for s in sequences:
        categories = []
        for item in s:
            categories.append(np.zeros(classes))
            categories[-1][item] = 1.0
        categorical_seq.append(categories)
    return np.array(categorical_seq)


def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return ignore_accuracy


input_train_data = read_data(input_fp)
tagged_data = tag_train_data(input_train_data)
joined_data = join_cols(tagged_data)

# print(joined_train_data)
sentences, sentence_tags = [], []

for index, address in joined_data[0].iteritems():
    separate_words_and_tags(address)

(train_sentences,
 test_sentences,
 train_tags,
 test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)

build_vocab()

word2index = {w: i + 2 for i, w in enumerate(list(words_vocab))}
word2index['-PAD-'] = 0
word2index['-OOV-'] = 1

tag2index = {t: i + 1 for i, t in enumerate(list(tags_vocab))}
tag2index['-PAD-'] = 0  # The special value used to padding

train_sentences_input = []
train_sentences_output = []
test_sentences_input = []
test_sentences_output = []

embed_training_words()
embed_test_words()

# print(sentences_input, sentences_output)
# print(sentences, sentence_tags)

len_longest_sen = len(max(train_sentences_input, key=len))
print(len_longest_sen)

train_sentences_input = pad_sequences(train_sentences_input, maxlen=len_longest_sen, padding='post')
train_sentences_output = pad_sequences(train_sentences_output, maxlen=len_longest_sen, padding='post')
test_sentences_input = pad_sequences(test_sentences_input, maxlen=len_longest_sen, padding='post')
test_sentences_output = pad_sequences(test_sentences_output, maxlen=len_longest_sen, padding='post')

model = Sequential()
model.add(InputLayer(input_shape=(len_longest_sen,)))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy', ignore_class_accuracy(0)])

model.summary()

categorical_sentences_output = encode_one_hot(train_sentences_output, len(tag2index))

model.fit(train_sentences_input, encode_one_hot(train_sentences_output, len(tag2index)), epochs=5,
          validation_split=0.2)

scores = model.evaluate(test_sentences_input, encode_one_hot(test_sentences_output, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")
