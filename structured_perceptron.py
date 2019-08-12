import pandas as pd
import re
import operator
from collections import defaultdict, deque

# defining relevant symbols for HMM trigram model
# start symbol is appended to the beginning of addresses for the trigram model
start_symbol = '*'
# stop symbol signifies the end of an address string
stop_symbol = 'STOP'
new_symbol = 'NEW'
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


# function that appropriately calls functions to compute
# 1. numerators of emission probabilities
# 2. bigram tag combinations count
# 3. trigram tag combinations count
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


# function to initialise the dictionary that will store the transition probabilities
# for our model, we are considering a trigram HMM
# thus we will initialise the dictionary with the frequency of trigram combinations of tags
def init_transitions_prob_dict():
    for key in list(trigram_tags_count):
        if key not in transitions_probability.keys():
            transitions_probability[key] = float(trigram_tags_count[key])


# function to compute transition probabilities
# transition probability of state w given state u and state v (in that order) =
# count(<state u, state v, state w>)/count(<state u, state v>)
def transitions_prob():
    init_transitions_prob_dict()
    for key in list(transitions_probability):
        split_key = key.split("-")
        print(split_key)
        trigram_tag = split_key[0] + "-" + split_key[1] + "-" + split_key[2]
        trigram_tag_count = trigram_tags_count[trigram_tag]
        # smoothing factor is also used to account for cases where the transition probability might be zero
        # or the denominator while computing the probability might be 0
        transitions_probability[key] = float(transitions_probability[key] + smoothing_factor_lambda) / float(
            trigram_tag_count + smoothing_factor_lambda * smoothing_factor_v)


# function to calculate emission probability of each word-tag pair
def emission_probability(prior_prob):
    for key in list(state_obs_pair_dict):
        tag = key.split("/")[1]
        # emission probability is computed as the number of times a word, w was tagged with tag t
        # out of all the words that have been tagged as t
        # similar to transition probability, a smoothing factor has been used
        state_obs_pair_dict[key] = float(state_obs_pair_dict[key] + smoothing_factor_lambda) / float(
            prior_prob[tag] + smoothing_factor_lambda * smoothing_factor_v)


# this function is part of a set of functions that have been used in feature encoding
# more specifically to create the labels that each byte of the feature vector represents
# this function creates the starting trigram combinations that have not been taken into account previously
# i.e. tags of type: *-*-tag and *-tag-tag
def create_start_link():
    for tag in valid_tags:
        curr_tag = start_symbol + "-" + start_symbol + "-" + tag
        feature_vector_labels.append(curr_tag)
    for tag1 in valid_tags:
        for tag2 in valid_tags:
            second_tag = start_symbol + "-" + tag1 + "-" + tag2
            feature_vector_labels.append(second_tag)


# similar to the previous function, this function creates previously unaccounted for end trigram combinations
# of the form tag-tag-STOP
def create_end_link():
    for tag1 in valid_tags:
        for tag2 in valid_tags:
            curr_tag = tag1 + "-" + tag2 + "-" + stop_symbol
            feature_vector_labels.append(curr_tag)


# this function creates all trigram combinations out of the given set of valid tags
# and adds them to the set of feature labels
def create_trigram_link():
    for trigram in trigram_tags_count.keys():
        feature_vector_labels.append(trigram)


# this function creates word-tag pairs and adds them to the set of feature labels
# it contains words taken from the vocabulary created earlier
def create_words_link():
    for word in unique_words_dict_subset.keys():
        curr_tags = tags[word]
        for tag in curr_tags:
            curr_word_tag = word + "/" + tag
            feature_vector_labels.append(curr_word_tag)


# for each tagged address, this function gets the trigram tag sequence:
# for example if the address is word1/tag1 word2/tag2 word3/tag3,
# then the tagged sequence will be
# [*-*-tag1, *-tag1-tag2, tag1-tag2-tag3, tag2-tag3-STOP]
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


# this function encodes each address in terms of the features created above
# it extracts the trigram tag sequences and increments the value of the respective bytes in the resultant feature vector
# additionally, it extracts word-tag pair and increments the value of the respective bytes in the resultant feature vector
def encode_features(address):
    tag_seq = get_tag_seq(address)

    feature_dict = {k: 0 for k in feature_vector_labels}
    for tag in tag_seq:
        feature_dict[tag] += 1
    for add in address:
        if add in feature_dict.keys():
            feature_dict[add] += 1
    return feature_dict


# for each valid tag, this function computes the probability that an address has of starting with it
# it adds each probability measure to the transitions probability dictionary
def compute_start_prob_unigram(data):
    start_p = {'PAON': 0.0, 'street': 0.0, 'city': 0.0, 'district': 0.0, 'county': 0.0}
    for index, address in data[0].iteritems():
        tag = (address.split("/")[1]).split(" ")[0]
        start_p[tag] += 1.0
    for k in start_p.keys():
        transitions_probability[start_symbol + "-" + start_symbol + "-" + k] = start_p[k] / len(data)


# this function computes the probability of an address starting with a particular tag bigram sequence
# and adds each new probability measure to the transitions probability dictionary
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


# this function computes the probability of an address ending with a particular tag bigram sequence
# and adds each new probability measure to the transitions probability dictionary
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


def set_of_tags(k):
    if k == -1 or k == 0:
        return {start_symbol}
    else:
        return valid_tags


# this is the viterbi algorithm
# it is a dynamic programming algorithm that computes the most optimal tag sequence
def viterbi(address, taglist, known_words, q_values, e_values, weights_vec):
    pi = defaultdict(float)
    # keys of this dictionary are in the form of a tuple: (k,u,v)
    # it stores the maximum probability of a k-length sequence ending in tags u and v (in that order)
    bp = {}
    # keys of this dictionary are in the form of a tuple: (k,u,v)
    # it stores the backpointers so that the entire tag sequence can be extracted easily

    # we initialise the dictionary such that a 0 length sequence ending in tag sequence *-* has probability 1
    pi[(0, start_symbol, start_symbol)] = 1.0
    # initialising the probability of a 0 length sequence ending in tag sequence *-tag or tag-tag is 0
    for u in taglist:
        pi[(0, start_symbol, u)] = 0.0
        for v in taglist:
            pi[(0, u, v)] = 0.0
    # for the given address, we check if each word in the address belongs to our vocabulary
    # if it does then it is stored as it is and if it doesn't, then we store 'NEW' instead signalling the word
    # doesn't belong to our vocabulary and will thus not have any transition or emission probability associated with it
    words = [word if word in known_words else new_symbol for word in address]

    n = len(words)
    # we have broken down our problem from an n-length sequence to a k-length sequence where k goes from 0:n
    for k in range(1, n + 1):
        for u in set_of_tags(k - 1):
            for v in set_of_tags(k):
                # default max score
                max_score = float('-Inf')
                # default tag for max score
                max_tag = None
                for w in set_of_tags(k - 2):
                    # check if word-tag pair is in emission probability dictionary
                    if (words[k - 1] + "/" + v) in e_values.keys():
                        # if it is, we extract the emission probability
                        e_val = e_values[(words[k - 1] + "/" + v)]
                        # and the current weight associated with it
                        e_val_wt = weights_vec[words[k - 1] + "/" + v]
                    else:
                        # if it's not, then we assign a very small arbitrary emission probability to it
                        e_val = 0.001
                        # and make the weight associated with it also arbitrarily small
                        e_val_wt = 0.0000000000000001
                    # now we check the max probability of the k-1 length tag sequence ending
                    # in w and u is the default score
                    if pi[k - 1, w, u] == float('-Inf'):
                        # if it is then we take the value to be 0 so that it doesn't contribute to the final score
                        pi_val = 0.0
                    else:
                        # if it is not, then we take the value as it is
                        pi_val = pi[k - 1, w, u]
                    # computing the score for the k length sequence using
                    # weighted transition probabilities
                    # weighted emission probabilities
                    # max probability of k-1 length tag sequence
                    score = pi_val + \
                            (weights_vec[w + "-" + u + "-" + v] * q_values[w + "-" + u + "-" + v]) + \
                            (e_val * e_val_wt)
                    # extracting the maximum score
                    if score > max_score:
                        max_score = score
                        max_tag = w
                # storing the maximum score and tag for the k-length sequence
                pi[(k, u, v)] = max_score
                bp[(k, u, v)] = max_tag
    # executing the same process as above for the last bigram tag sequence: tag-tag-STOP
    max_score = float('-Inf')
    u_max, v_max = None, None
    tags = deque()
    for u in set_of_tags(n - 1):
        for v in set_of_tags(n):
            score = pi.get((n, u, v), 0.0) + \
                    q_values[u + "-" + v + "-" + stop_symbol]
            if score > max_score:
                max_score = score
                u_max = u
                v_max = v

    tags.append(v_max)
    tags.append(u_max)
    # traversing the back pointers to get the reversed tag sequence
    for i, k in enumerate(range(n - 2, 0, -1)):
        if (k + 2, tags[i + 1], tags[i]) in bp.keys():
            tags.append(bp[(k + 2, tags[i + 1], tags[i])])
    tags.reverse()

    # recreating the address with each part tagged according to the back pointers
    tagged_sentence = []
    for j in range(0, n):
        tagged_sentence.append(address[j] + '/' + tags[j])

    return tagged_sentence


# structured perceptron algorithm
# computes the most probable tag sequence using a weighted version of the viterbi algorithm during training
def structured_perceptron(data, weights_vec):
    print("training!")
    # iterating over the data
    for index, address in data[0].iteritems():
        if index % 10000 == 0:
            print(index)
        address = address.split(" ")
        # extracting the word sequence from the address
        word_seq = []
        for part in address:
            part = part.split("/")[0]
            word_seq.append(part)
        # encoding the actual tagged address
        encoded_actual_address = encode_features(address)
        # calling the viterbi algorithm with the current set of weights
        result = viterbi(word_seq, valid_tags, known_words, transitions_probability, state_obs_pair_dict, weights_vec)
        # encoding the predicted tagged address
        encoded_predicted_address = encode_features(result)
        # comparing the feature encodings of the actual and predicted tagged addresses
        if len(encoded_predicted_address) == len(encoded_actual_address):
            for key in encoded_actual_address.keys():
                # if the predicted and actual tag sequences are the same, the weights are left unchanged
                diff = encoded_actual_address[key] - encoded_predicted_address[key]
                if diff != 0:
                    # if they are not the same, then the weights are updated to account for misclassification
                    weights_vec[key] += diff
        else:
            print("lengths don't match")

    return weights_vec


# function used to predict most probable tag sequence for test data
# requires final set of weights computed during training by the structured perceptron
def predict(df, weights):
    print("testing!!")
    matches = {}
    total_actual = {}
    total_predicted = {}
    accuracy_numerator = 0
    for tag in valid_tags:
        matches[tag] = 0
        total_actual[tag] = 0
        total_predicted[tag] = 0
    for index, address in df[0].iteritems():
        if index % 1000 == 0:
            print(index)
        # extracting the word and tag sequence from the test data
        address = address.split(" ")
        actual_tag_seq = []
        predicted_tag_seq = []
        word_seq = []
        for part in address:
            word = part.split("/")[0]
            tag = part.split("/")[1]
            word_seq.append(word)
            actual_tag_seq.append(tag)
        # using the viterbi algorithm with the final set of weights to predict the most probable tag sequence
        result = viterbi(word_seq, valid_tags, known_words, transitions_probability, state_obs_pair_dict, weights)
        for part in result:
            predicted_tag_seq.append((part.split("/"))[1])
        # encoding both the actual and predicted tagged addresses
        # actual_encoded_add = encode_features(address)
        # predicted_encoded_add = encode_features(result)
        # computing test accuracy

        # for key in actual_encoded_add.keys():
        #     if actual_encoded_add[key] != predicted_encoded_add[key]:
        #         diff += 1
        # if diff == 0:
        #     matches0 += 1
        diff = 0
        for i in range(len(actual_tag_seq)):
            total_actual[actual_tag_seq[i]] += 1
            total_predicted[predicted_tag_seq[i]] += 1
            if actual_tag_seq[i] == predicted_tag_seq[i]:
                matches[actual_tag_seq[i]] += 1
            else:
                diff = 1
        if diff == 0:
            accuracy_numerator += 1

    return [accuracy_numerator / len(df), matches, total_predicted, total_actual]


# reading the train and test data from file
df = read_data(input_train_fp)
test_df = read_data(input_test_fp)
# tagging both sets of data
tagged_train_data = tag_train_data(df)
tagged_test_data = tag_train_data(test_df)

# joining tagged columns into a single string for both train and test data
new_df = join_cols(tagged_train_data)
joined_test_data = join_cols(tagged_test_data)

# computing a subset of the entire vocabulary in the addresses
# the distribution of frequency of each word was considered and the threshold was chosen to be 5
unique_words_dict_subset = {key: value for key, value in complete_unique_words_dict.items() if value > 5.0}
# storing the chosen subset of vocabulary in a list
known_words = [k for k, v in unique_words_dict_subset.items()]

# computing the prior probability of tags
prior = compute_prior_prob(tags)
# computing the most frequent tag and letting that be the default tag
default_tag = max(prior.items(), key=operator.itemgetter(1))[0]

# initialising the bigram and trigram dictionaries
init_bigram_dict()
init_trigram_dict(bigram_tags_count)

# computing transition probabilities for each trigram tag sequence
transitions_count(new_df)
transitions_prob()

# computing emission probability of each word-tag pair
# where the word is in the new vocabulary
emission_probability(tag_count)

# computing previously unaccounted for transition probabilities
compute_start_prob_unigram(new_df)
compute_start_prob_bigram(new_df)
compute_end_prob_bigram(new_df)

# computing feature encoding labels
create_start_link()
create_end_link()
create_trigram_link()
create_words_link()

# number of epochs during training
epochs = 60
# initialising weights vector
training_weights = {k: 0 for k in feature_vector_labels}
for epoch in range(epochs):
    print("epoch", epoch + 1)
    # training such that the weights from the previous epoch is fed to the next one
    training_weights = structured_perceptron(new_df, training_weights)
    print(training_weights)
# predicting on test data and storing accuracy
precision = {}
recall = {}
accuracy_vec = predict(joined_test_data, training_weights)
print("exact accuracy:", accuracy_vec[0])
# print("correctly predicted tags:", accuracy_vec[1])
# print("total predicted per tag:", accuracy_vec[2])
# print("total actual per tag:", accuracy_vec[3])
for key in accuracy_vec[1].keys():
    precision[key] = accuracy_vec[1][key] / accuracy_vec[2][key]
    recall[key] = accuracy_vec[1][key] / accuracy_vec[3][key]

print("precision: ", precision)
print("recall", recall)
