import pandas as pd
import numpy as np
import re
import operator
import random
from collections import defaultdict, deque
from itertools import combinations_with_replacement, permutations

# defining relevant symbols for HMM trigram model
# start symbol is appended to the beginning of addresses for the trigram model
start_symbol = '*'
# stop symbol signifies the end of an address string
stop_symbol = 'STOP'
rare_symbol = '_RARE_'
# smoothing factor defined to account for new words seen in the test data that were not part of
# the train data vocabulary
smoothing_factor_lambda = 0.1
# defining log probability of zero, to be used in computing the score in the viterbi algorithm
log_prob_of_zero = -1000.0
# defining the set of valid tags
valid_tags = ['PAON', 'street', 'city', 'district', 'county']
smoothing_factor_v = len(valid_tags)

# specifying paths for input train and test data
input_train_fp = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/cleaned_month_addresses_train.csv"
input_test_fp = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/cleaned_month_addresses_test.csv"

tags = {}
# dictionary to keep track of frequency of bigram combinations of tags in the train data
bigram_tags_count = {}

# dictionary to keep track of frequency of trigram combinations of tags in the train data
trigram_tags_count = {}

# dictionary to store transition probabilities from state i to state j for all i,j
transitions_probability = {}

# dictionary to store emission probabilities of observations and their respective states
state_obs_pair_dict = {}

# dictionary that keeps track of the frequency of each tag for all addresses
tag_count = {}

# dictionary to store all the unique words and their counts extracted from the addresses
complete_unique_words_dict = {}
start_tag = '<start>'
end_tag = '<end>'

# list keeps track of cell labels for feature vectors
# to be used while encoding features
feature_vector_labels = []
word_to_tag_dict = {}


# function to read a csv file and convert it to a pandas dataframe and return that
def read_data(file):
    data = pd.read_csv(file)
    return data


# function that builds vocabulary of unique words from addresses
# takes a word as an input
def get_word_corpus(word):
    count = 0
    # check for if the word is a number whose digits have been converted to uppercase D
    # and if it's not, only then consider it a unique word to be added to the vocabulary
    for c in word:
        if c == 'D':
            count += 1
    if count != len(word):
        if word not in complete_unique_words_dict.keys():
            complete_unique_words_dict[word] = 1.0
        else:
            complete_unique_words_dict[word] += 1.0


# function to assign tags to individual words for all addresses
def tag_train_data(data):
    # extracts the relevant subset of data
    subset_data = data[['PAON', 'street', 'city', 'district', 'county']]
    for col in subset_data.columns:
        tagged_col = []
        curr_col_list = subset_data[col].values.flatten()
        for i in range(len(curr_col_list)):

            if isinstance(curr_col_list[i], str):
                # splitting the address into an array of individual words
                parts = curr_col_list[i].split(" ")
                tagged_part = ""
                for word in parts:
                    # using regex to check if the word has atleast one digit
                    if re.findall("\d+", word):
                        length = len(word)
                        # if the condition is satisfied, the characters of the word are converted to uppercase D
                        word = 'D' * length
                    get_word_corpus(word)
                    # tagging each word
                    tagged_word = word + "/" + col
                    tagged_part = tagged_part + " " + tagged_word
                    # storing the words and their respective tags in a dictionary called tags
                    if word not in tags.keys() and isinstance(word, str):
                        tags[word] = []
                        tags[word].append(col)
                    elif col not in tags[word]:
                        tags[word].append(col)
                tagged_col.append(tagged_part)
        i = 0
        for index, address in subset_data[col].iteritems():
            if not pd.isna(subset_data.iloc[index][col]):
                # rewriting the tagged address in the original dataframe
                subset_data.iloc[index][col] = tagged_col[i]
                i += 1
    return subset_data


# function computes the prior probability of each tag
def compute_prior_prob(tag_dict):
    prior_prob = {}
    total_num = 0
    # initialising the prior probability of each tag in the dictionary as 0
    for tag in valid_tags:
        prior_prob[tag] = 0.0
    # for each word in the tag dictionary
    for word in tag_dict.keys():
        # for each tag that is associated with the given word
        for tag in tag_dict[word]:
            # occurrence of the tag in the prior_prob dictionary is incremented by 1
            prior_prob[tag] += 1.0
            # keeping track of total number of tags
            total_num += 1.0
    for tag in valid_tags:
        # storing the frequency of each tag for all possible addresses
        tag_count[tag] = prior_prob[tag]
        # computing the prior probability of each tag
        prior_prob[tag] = prior_prob[tag] / total_num

    return prior_prob


# this function joins the different tagged columns into one address string where each word is appropriately tagged
def join_cols(df):
    # extracting valid columns
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


# function to initialise the bigram dictionary
# assigns a value of 0 to all possible bigram combination of tags
def init_bigram_dict():
    for i in range(len(valid_tags)):
        for j in range(len(valid_tags)):
            tag_1 = valid_tags[i] + "-" + valid_tags[j]
            tag_2 = valid_tags[j] + "-" + valid_tags[i]
            if tag_1 not in bigram_tags_count.keys():
                bigram_tags_count[tag_1] = 0
            if tag_2 not in bigram_tags_count.keys():
                bigram_tags_count[tag_2] = 0


# function to initialise the trigram dictionary using the bigram dictionary
# assigns a value of 0 to all possible trigram combination of tags
def init_trigram_dict(bigram_dict):
    for key, value in bigram_dict.items():
        for tag in valid_tags:
            tag_1 = key + "-" + tag
            tag_2 = tag + "-" + key
            if tag_1 not in trigram_tags_count.keys():
                trigram_tags_count[tag_1] = 0
            if tag_2 not in trigram_tags_count.keys():
                trigram_tags_count[tag_2] = 0


# function to compute the count of word-tag pairs, i.e., the numerator of emission probabilities
# for each word-tag pair
def emission_prob_count(state_obs_pair):
    if state_obs_pair not in state_obs_pair_dict.keys():
        state_obs_pair_dict[state_obs_pair] = 1.0
    else:
        state_obs_pair_dict[state_obs_pair] += 1.0


# function to compute the frequency of each bigram and trigram combination of tags
def count_bigram_trigram(curr_tag_list):
    for i in range(len(curr_tag_list) - 2):
        for j in range(i + 1, len(curr_tag_list) - 1):
            bigram_tag = curr_tag_list[i] + "-" + curr_tag_list[j]
            trigram_tag = bigram_tag + "-" + curr_tag_list[j + 1]
            bigram_tags_count[bigram_tag] += 1
            trigram_tags_count[trigram_tag] += 1
    last_bigram_tag = curr_tag_list[len(curr_tag_list) - 2] + "-" + curr_tag_list[len(curr_tag_list) - 1]
    bigram_tags_count[last_bigram_tag] += 1


# function to delete those bigram and trigrams whose frequency is 0
def clean_transition_count_dict():
    for key in list(bigram_tags_count):
        if bigram_tags_count[key] == 0:
            bigram_tags_count.pop(key)
    for key in list(trigram_tags_count):
        if trigram_tags_count[key] == 0:
            trigram_tags_count.pop(key)


def transitions_count(data):
    data_list = pd.Series.tolist(data)
    for i in range(len(data_list)):
        curr_tags = []
        curr_address = (data_list[i][0].strip()).split(" ")
        for part in curr_address:
            emission_prob_count(part)
            tag = part.split("/")[1]
            curr_tags.append(tag)
        count_bigram_trigram(curr_tags)
    # clean_transition_count_dict()


def init_transitions_prob_dict():
    for key in list(trigram_tags_count):
        if key not in transitions_probability.keys():
            transitions_probability[key] = float(trigram_tags_count[key])


def transitions_prob():
    init_transitions_prob_dict()
    for key in list(transitions_probability):
        split_key = key.split("-")
        bigram_tag = split_key[0] + "-" + split_key[1]
        bigram_tag_count = bigram_tags_count[bigram_tag]

        transitions_probability[key] = float(transitions_probability[key] + smoothing_factor_lambda) / float(
            bigram_tag_count + smoothing_factor_lambda * smoothing_factor_v)


def emission_probability(prior_prob):
    for key in list(state_obs_pair_dict):
        tag = key.split("/")[1]
        state_obs_pair_dict[key] = float(state_obs_pair_dict[key] + smoothing_factor_lambda) / float(
            prior_prob[tag] + smoothing_factor_lambda * smoothing_factor_v)


def create_start_link():
    for tag in valid_tags:
        curr_tag = start_symbol + "-" + start_symbol + "-" + tag
        feature_vector_labels.append(curr_tag)
    for tag1 in valid_tags:
        for tag2 in valid_tags:
            second_tag = start_symbol + "-" + tag1 + "-" + tag2
            feature_vector_labels.append(second_tag)


def create_end_link():
    for tag1 in valid_tags:
        for tag2 in valid_tags:
            curr_tag = tag1 + "-" + tag2 + "-" + stop_symbol
            feature_vector_labels.append(curr_tag)


def create_trigram_link():
    for trigram in trigram_tags_count.keys():
        feature_vector_labels.append(trigram)


def create_words_link():
    for word in unique_words_dict_subset.keys():
        curr_tags = tags[word]
        for tag in curr_tags:
            curr_word_tag = word + "/" + tag
            feature_vector_labels.append(curr_word_tag)


def get_tag_seq(address):
    first_tag = start_symbol + "-" + start_symbol + "-" + (address[0].split("/"))[1]
    second_tag = start_symbol + "-" + (address[0].split("/"))[1] + "-" + (address[1].split("/"))[1]
    last_tag = (address[-2].split("/"))[1] + "-" + (address[-1].split("/"))[1] + "-" + stop_symbol
    tag_seq = [first_tag, second_tag]
    for i in range(len(address) - 2):
        curr_tag = (address[i].split("/"))[1]
        next_tag = (address[i + 1].split("/"))[1]
        next_to_next_tag = (address[i + 2].split("/"))[1]
        curr_tag_seq = curr_tag + "-" + next_tag + "-" + next_to_next_tag
        tag_seq.append(curr_tag_seq)
    tag_seq.append(last_tag)
    return tag_seq


def encode_features(address):
    # print("in encode_features, address is: ", address)
    tag_seq = get_tag_seq(address)

    feature_dict = {k: 0 for k in feature_vector_labels}
    # print("feature dict:", feature_dict)
    # length = len(feature_dict)
    # print(len(feature_dict))
    for tag in tag_seq:
        feature_dict[tag] += 1
    for add in address:
        if add in feature_dict.keys():
            feature_dict[add] += 1
    return feature_dict


def compute_start_prob_unigram(data):
    start_p = {'PAON': 0.0, 'street': 0.0, 'city': 0.0, 'district': 0.0, 'county': 0.0}
    for index, address in data[0].iteritems():
        tag = (address.split("/")[1]).split(" ")[0]
        start_p[tag] += 1.0
    for k in start_p.keys():
        transitions_probability[start_symbol + "-" + start_symbol + "-" + k] = start_p[k] / len(data)


def compute_start_prob_bigram(data):
    start_p = {}
    for tag1 in valid_tags:
        for tag2 in valid_tags:
            start_p[tag1 + "-" + tag2] = 0.0

    for index, address in data[0].iteritems():
        tag1 = (address.split("/")[1]).split(" ")[0]
        tag2 = (address.split("/")[2]).split(" ")[0]
        bigram = tag1 + "-" + tag2
        start_p[bigram] += 1.0
    for k in start_p.keys():
        transitions_probability[start_symbol + "-" + k] = start_p[k] / len(data)


def compute_end_prob_bigram(data):
    end_p = {}
    for tag1 in valid_tags:
        for tag2 in valid_tags:
            end_p[tag1 + "-" + tag2] = 0.0
    for index, address in data[0].iteritems():
        address = address.split(" ")
        tag1 = address[-2].split("/")[1]
        tag2 = address[-1].split("/")[1]
        bigram = tag1 + "-" + tag2
        end_p[bigram] += 1.0
    for k in end_p.keys():
        transitions_probability[k + "-" + stop_symbol] = end_p[k] / len(data)


def viterbi(address, taglist, known_words, q_values, e_values, weights_vec):
    tagged = []

    # pi[(k, u, v)]: max probability of a tag sequence ending in tags u, v at position k
    # bp[(k, u, v)]: backpointers to recover the argmax of pi[(k, u, v)]
    pi = defaultdict(float)
    bp = {}
    # Initialization
    pi[(0, start_symbol, start_symbol)] = 1.0

    for u in taglist:
        pi[(0, start_symbol, u)] = 0.0
        for v in taglist:
            pi[(0, u, v)] = 0.0

    # Define tagsets S(k)
    def S(k):
        if k in (-1, 0):
            return {start_symbol}
        else:
            return taglist

        # The Viterbi algorithm

    words = [word if word in known_words else rare_symbol for word in address]

    n = len(words)

    for k in range(1, n + 1):
        for u in S(k - 1):
            for v in S(k):
                max_score = float('-Inf')
                max_tag = None
                for w in S(k - 2):

                    # if e_values.get((words[k - 1] + "/" + v), 0) != 0:
                    # print("entering loop")
                    # print("pi: ", pi.get((k - 1, w, u), log_prob_of_zero))
                    # print("q: ", q_values[w + "-" + u + "-" + v])
                    # print("e: ", e_values[words[k - 1] + "/" + v])
                    # print("testing:")
                    # if (k - 1, w, u) in pi.keys():
                    #     print("yes, val: ", pi[(k - 1, w, u)])
                    # else:
                    #     print("should be -1000")
                    if (words[k - 1] + "/" + v) in e_values.keys():
                        e_val = e_values[(words[k - 1] + "/" + v)]
                        e_val_wt = weights_vec[words[k - 1] + "/" + v]
                    else:
                        e_val = 0.001
                        e_val_wt = 0.0000000000000001
                    # print("k-1,w,u:", k - 1, w, u)

                    if pi[k - 1, w, u] == float('-Inf'):
                        pi_val = 0.0
                    else:
                        pi_val = pi[k - 1, w, u]
                    # print("pi score: ", pi_val)
                    score = pi_val + \
                            (weights_vec[w + "-" + u + "-" + v] * q_values[w + "-" + u + "-" + v]) + \
                            (e_val * e_val_wt)
                    # print("score", score)
                    if score > max_score:
                        max_score = score
                        max_tag = w
                pi[(k, u, v)] = max_score
                bp[(k, u, v)] = max_tag

    max_score = float('-Inf')
    u_max, v_max = None, None
    tags = deque()
    for u in S(n - 1):
        for v in S(n):
            # if u + "-" + v + "-" + stop_symbol in q_values.keys():
            # print(pi.get((n, u, v), -1000))
            # print(q_values[u + "-" + v + "-" + stop_symbol])
            score = pi.get((n, u, v), 0.0) + \
                    q_values[u + "-" + v + "-" + stop_symbol]
            # print("Score: ", score)
            # print("u: ", u)
            # print("v: ", v)
            if score > max_score:
                max_score = score
                u_max = u
                v_max = v

    tags.append(v_max)
    tags.append(u_max)
    # print(tags)
    for i, k in enumerate(range(n - 2, 0, -1)):
        if (k + 2, tags[i + 1], tags[i]) in bp.keys():
            tags.append(bp[(k + 2, tags[i + 1], tags[i])])
    tags.reverse()

    tagged_sentence = []
    for j in range(0, n):
        tagged_sentence.append(address[j] + '/' + tags[j])
    # print(tagged_sentence)
    # tagged_sentence.append('\n')

    return tagged_sentence


# def viterbi()
def structured_perceptron(data, weights_vec):
    # weights_vec = {k: 0 for k in feature_vector_labels}
    print("training!")
    for index, address in data[0].iteritems():
        if index % 10000 == 0:
            print(index)
        address = address.split(" ")
        word_seq = []
        for part in address:
            part = part.split("/")[0]
            word_seq.append(part)
        # print("actual address: ", address)
        encoded_actual_address = encode_features(address)
        # print("encoded actual address: ", encoded_actual_address)

        result = viterbi(word_seq, valid_tags, known_words, transitions_probability, state_obs_pair_dict, weights_vec)
        # print("predicted tag seq: ", result)
        encoded_predicted_address = encode_features(result)
        # print("encoded predicted tag seq: ", encoded_predicted_address)
        if len(encoded_predicted_address) == len(encoded_actual_address):
            for key in encoded_actual_address.keys():
                diff = encoded_actual_address[key] - encoded_predicted_address[key]
                if diff != 0:
                    weights_vec[key] += diff
        else:
            print("lengths don't match")
    # weights_vec += diff
    # print("\n")
    return weights_vec


# def dot(word_list, predicted_tag_tuple, weights_vec):
#     address = ['']
#     for i in range(len(word_list)):
#         tagged_word = word_list[i] + "/" + predicted_tag_tuple[i]
#         address.append(tagged_word)
#     encoded_vec = phi(address)
#     dot_product = 0
#     for i in range(len(encoded_vec)):
#         dot_product += encoded_vec[i] * weights_vec[i]
#     return (dot_product, encoded_vec, address)
#
#
# def actual_structured_perceptron(data):
#     weights_vec = []
#     zero_vec = []
#
#     for i in range(len(data)):
#         weights_vec.append(0)
#         zero_vec.append(0)
#
#     for index, address in data[0].iteritems():
#         print(index)
#         max_score = float('-Inf')
#         most_likely_tag_seq = []
#         set_of_tag_seq = {}
#         address = address.split(" ")
#         print(address)
#         phi_of_actual_output = phi(address)
#         phi_of_predicted_vec = []
#         word_seq = []
#         actual_tag_seq = []
#         for part in address:
#             if part != '':
#                 word = part.split("/")[0]
#                 tag = part.split("/")[1]
#                 word_seq.append(word)
#                 actual_tag_seq.append(tag)
#         n = len(word_seq)
#         combinations = combinations_with_replacement(valid_tags, n)
#         for i in list(combinations):
#             permutations_of_i = permutations(list(i), n)
#             for j in list(permutations_of_i):
#                 if j not in set_of_tag_seq.keys():
#                     set_of_tag_seq[j] = 0
#         for key in set_of_tag_seq.keys():
#             result = dot(word_seq, key, weights_vec)
#             score = result[0]
#             if score > max_score:
#                 max_score = score
#                 most_likely_tag_seq = result[1]
#                 phi_of_predicted_vec = result[2]
#         print(most_likely_tag_seq)
#         diff = phi_of_actual_output - phi_of_predicted_vec
#         if diff != zero_vec:
#             weights_vec += diff
#     return weights_vec

def predict(df, weights):
    print("testing!!")
    matches0 = 0
    matches1 = 0
    matches2 = 0
    matches3 = 0
    for index, address in df[0].iteritems():
        diff = 0
        if index % 1000 == 0:
            print(index)
        address = address.split(" ")
        tag_seq = []
        word_seq = []
        for part in address:
            word = part.split("/")[0]
            tag = part.split("/")[1]
            word_seq.append(word)
            tag_seq.append(tag)

        result = viterbi(word_seq, valid_tags, known_words, transitions_probability, state_obs_pair_dict, weights)
        actual_encoded_add = encode_features(address)
        predicted_encoded_add = encode_features(result)
        for key in actual_encoded_add.keys():
            if actual_encoded_add[key] != predicted_encoded_add[key]:
                diff += 1
        if diff == 0:
            matches0 += 1
        if diff <= 1:
            matches1 += 1
        if diff <= 2:
            matches2 += 1
        if diff <= 3:
            matches3 += 1
    return [matches0 / len(df), matches1 / len(df), matches2 / len(df), matches3 / len(df)]

    # print("actual test vec:", address)
    # print("test predicted vec:", result)


df = read_data(input_train_fp)
test_df = read_data(input_test_fp)
# print("original df", df)
tagged_train_data = tag_train_data(df)
tagged_test_data = tag_train_data(test_df)

joined_test_data = join_cols(tagged_test_data)

unique_words_dict_subset = {key: value for key, value in complete_unique_words_dict.items() if value > 5.0}
known_words = [k for k, v in unique_words_dict_subset.items()]
# print("\nknown words", known_words)

prior = compute_prior_prob(tags)
default_tag = max(prior.items(), key=operator.itemgetter(1))[0]
new_df = join_cols(tagged_train_data)

init_bigram_dict()
init_trigram_dict(bigram_tags_count)
transitions_count(new_df)
transitions_prob()
emission_probability(tag_count)

compute_start_prob_unigram(new_df)
compute_start_prob_bigram(new_df)
compute_end_prob_bigram(new_df)

# print("transitions_probability", transitions_probability)
# create_words_link()
# print(feature_vector_labels)
# print("emission prob", state_obs_pair_dict)
# for index, address in new_df[0].iteritems():
#     phi(address)

create_start_link()
create_end_link()
create_trigram_link()
create_words_link()
# print(feature_vector_labels)
# print(len(feature_vector_labels))
epochs = 20
training_weights = {k: 0 for k in feature_vector_labels}
for epoch in range(epochs):
    print(epoch)
    training_weights = structured_perceptron(new_df, training_weights)
    print(training_weights)

accuracy_vec = predict(joined_test_data, training_weights)
print("exact accuracy:", accuracy_vec[0])
print("one wrong tag accuracy:", accuracy_vec[1])
print("two wrong tags accuracy:", accuracy_vec[2])
print("three wrong tags accuracy", accuracy_vec[3])
