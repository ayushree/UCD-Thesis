import pandas as pd
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import backend as K

input_fp = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/cleaned_month_addresses.csv"
input_test_fp = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/cleaned_month_addresses_test.csv"

start_symbol = '*'
stop_symbol = 'STOP'
new_symbol = 'NEW'
pad_symbol = 'PAD'
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

words_vocab = {}
tags_vocab = {}

word_encode_key = {}
tag_encode_key = {}


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
            words_vocab[word] = 0.0
    for sen_tag in train_tags:
        for tag in sen_tag:
            tags_vocab[tag] = 0.0


def create_embedding_dict():
    i = 2
    word_encode_key[pad_symbol] = 0
    word_encode_key[new_symbol] = 1
    for word in words_vocab.keys():
        word_encode_key[word] = i
        i += 1
    j = 1
    tag_encode_key[pad_symbol] = 0
    for tag in tags_vocab.keys():
        tag_encode_key[tag] = j
        j += 1


def embed_training_words():
    for sen in train_sentences:
        sen_int = []
        for word in sen:
            if word in word_encode_key.keys():
                sen_int.append(word_encode_key[word])
            else:
                sen_int.append(word_encode_key[new_symbol])

        train_sentences_input.append(sen_int)
    for tag_sen in train_tags:
        sen_int = []
        for tag in tag_sen:
            sen_int.append(tag_encode_key[tag])
        train_sentences_output.append(sen_int)


def embed_test_words():
    for sen in test_sentences:
        sen_int = []
        for word in sen:
            if word in word_encode_key.keys():
                sen_int.append(word_encode_key[word])
            else:
                sen_int.append(word_encode_key[new_symbol])
        test_sentences_input.append(sen_int)

    for tag_sen in test_tags:
        sen_int = []
        for tag in tag_sen:
            sen_int.append(tag_encode_key[tag])
        test_sentences_output.append(sen_int)


def ignore_class_accuracy(to_ignore=0):
    def true_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return true_accuracy


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
create_embedding_dict()

train_sentences_input = []
train_sentences_output = []
test_sentences_input = []
test_sentences_output = []

embed_training_words()
embed_test_words()

len_longest_sen = len(max(train_sentences_input, key=len))
print(len_longest_sen)
print("train input")
print(len(train_sentences_input))
print("train output")
print(len(train_sentences_output))
print("test input")
print(len(test_sentences_input))
print("test output")
print(len(test_sentences_output))
train_sentences_input = pad_sequences(train_sentences_input, maxlen=len_longest_sen, padding='post')
train_sentences_output = pad_sequences(train_sentences_output, maxlen=len_longest_sen, padding='post')
test_sentences_input = pad_sequences(test_sentences_input, maxlen=len_longest_sen, padding='post')
test_sentences_output = pad_sequences(test_sentences_output, maxlen=len_longest_sen, padding='post')

model = Sequential()
model.add(InputLayer(input_shape=(len_longest_sen,)))
model.add(Embedding(len(word_encode_key), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag_encode_key))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

model.summary()
# print("using inbuilt one hot encoding")
# print(to_categorical(train_sentences_output, num_classes=len(tag_encode_key)))
# print("using diff function")
# test = to_categorical2(train_sentences_output, len(tag_encode_key))
# print(test)

model.fit(train_sentences_input, to_categorical(train_sentences_output, num_classes=len(tag_encode_key)), epochs=1,
          validation_split=0.2)

scores = model.evaluate(test_sentences_input, to_categorical(test_sentences_output, num_classes=len(tag_encode_key)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")
