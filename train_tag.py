import pandas as pd
import numpy as np
import re
import operator

input_fp = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/cleaned_month_addresses.csv"

tags = {}

valid_tags = ['PAON', 'street', 'city', 'district', 'county']
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


def compute_prior_prob(tag_dict):
    prior_prob = {}
    total_num = 0
    for tag in valid_tags:
        prior_prob[tag] = 0.0
    for word in tag_dict.keys():
        for tag in tag_dict[word]:
            prior_prob[tag] += 1.0
            total_num += 1.0
    for tag in valid_tags:
        tag_count[tag] = prior_prob[tag]
        prior_prob[tag] = prior_prob[tag] / total_num

    return prior_prob


def join_cols(df):
    subset_df = df[valid_tags]
    tagged_complete_address = []

    for i in range(len(subset_df)):
        new_address = ""
        for j in range(len(subset_df.columns)):
            if isinstance(subset_df.iloc[i][j], str):
                new_address += subset_df.iloc[i][j]
        tagged_complete_address.append(new_address)
    tagged_complete_address = pd.DataFrame(tagged_complete_address)
    return tagged_complete_address


def init_bigram_dict():
    for i in range(len(valid_tags)):
        for j in range(len(valid_tags)):
            tag_1 = valid_tags[i] + "-" + valid_tags[j]
            tag_2 = valid_tags[j] + "-" + valid_tags[i]
            if tag_1 not in bigram_tags_count.keys():
                bigram_tags_count[tag_1] = 0
            if tag_2 not in bigram_tags_count.keys():
                bigram_tags_count[tag_2] = 0


def init_trigram_dict(bigram_dict):
    for key, value in bigram_dict.items():
        for tag in valid_tags:
            tag_1 = key + "-" + tag
            tag_2 = tag + "-" + key
            if tag_1 not in trigram_tags_count.keys():
                trigram_tags_count[tag_1] = 0
            if tag_2 not in trigram_tags_count.keys():
                trigram_tags_count[tag_2] = 0


def emission_prob_count(state_obs_pair):
    if state_obs_pair not in state_obs_pair_dict.keys():
        state_obs_pair_dict[state_obs_pair] = 1.0
    else:
        state_obs_pair_dict[state_obs_pair] += 1.0


def count_bigram_trigram(curr_tag_list):
    for i in range(len(curr_tag_list) - 2):
        for j in range(i + 1, len(curr_tag_list) - 1):
            bigram_tag = curr_tag_list[i] + "-" + curr_tag_list[j]
            trigram_tag = bigram_tag + "-" + curr_tag_list[j + 1]
            bigram_tags_count[bigram_tag] += 1
            trigram_tags_count[trigram_tag] += 1
    last_bigram_tag = curr_tag_list[len(curr_tag_list) - 2] + "-" + curr_tag_list[len(curr_tag_list) - 1]
    bigram_tags_count[last_bigram_tag] += 1


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
    clean_transition_count_dict()


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
        transitions_probability[key] = transitions_probability[key] / bigram_tag_count


def emission_probability(prior_prob):
    for key in list(state_obs_pair_dict):
        tag = key.split("/")[1]
        state_obs_pair_dict[key] = float(state_obs_pair_dict[key]) / float(prior_prob[tag])


def create_start_link():
    for tag in valid_tags:
        curr_tag = start_tag + "-" + tag
        feature_vector_labels.append(curr_tag)


def create_end_link():
    for tag in valid_tags:
        curr_tag = tag + "-" + end_tag
        feature_vector_labels.append(curr_tag)


def create_bigram_link():
    for bigram in bigram_tags_count.keys():
        feature_vector_labels.append(bigram)


def create_words_link():
    for word in unique_words_dict_subset.keys():
        curr_tags = tags[word]
        for tag in curr_tags:
            curr_word_tag = word + "/" + tag
            feature_vector_labels.append(curr_word_tag)


def get_tag_seq(address):
    first_tag = start_tag + "-" + (address[0].split("/"))[1]
    last_tag = (address[-1].split("/"))[1] + "-" + end_tag
    tag_seq = [first_tag]
    for i in range(len(address) - 1):
        curr_tag = (address[i].split("/"))[1]
        next_tag = (address[i + 1].split("/"))[1]
        curr_tag_seq = curr_tag + "-" + next_tag
        tag_seq.append(curr_tag_seq)
    tag_seq.append(last_tag)
    return tag_seq


def create_feature_vectors():
    create_start_link()
    create_end_link()
    create_bigram_link()
    create_words_link()
    vectors_df = pd.DataFrame()
    for index, address in new_df[0].iteritems():
        address = address.split(" ")[1:]
        tag_seq = get_tag_seq(address)

        feature_dict = {k: 0 for k in feature_vector_labels}
        # length = len(feature_dict)
        # print(len(feature_dict))
        for tag in tag_seq:
            feature_dict[tag] = 1
        for add in address:
            if add in feature_dict.keys():
                feature_dict[add] = 1
        feature_list = [feature_dict[key] for key in feature_dict.keys()]
        feature_list = pd.Series(feature_list)
        # print(type(feature_list))
        # print(feature_list)
        vectors_df[index] = feature_list
        print(vectors_df[index])
    return vectors_df


df = read_data(input_fp)
tagged_train_data = tag_train_data(df)
unique_words_dict_subset = {key: value for key, value in complete_unique_words_dict.items() if value > 5.0}

prior = compute_prior_prob(tags)
default_tag = max(prior.items(), key=operator.itemgetter(1))[0]
new_df = join_cols(tagged_train_data)

init_bigram_dict()
init_trigram_dict(bigram_tags_count)
transitions_count(new_df)
transitions_prob()
emission_probability(tag_count)
feature_vectors_df = create_feature_vectors()
feature_vectors_df.to_csv("/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/featurevectors.csv")
# print(feature_vectors_df)

# print(len(feature_vector_labels))
# print(new_df)
