import pandas as pd
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np

# storing all english stop words in a variable called stpwrds
stpwrds = stopwords.words('english')

# specifying path for the input file
input_file_path = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/addresses_month.csv"


# function that reads a csv file containing appropriate housing data
# and returns a dataframe containing relevant columns
def read_data(file):
    col_names = ['UID', 'price', 'transfer_date', 'postcode', 'property_type', 'age', 'duration', 'PAON', 'SAON',
                 'street',
                 'locality', 'city', 'district', 'county', 'PPD_type', 'record_status']
    data = pd.read_csv(file, header=None)
    data.columns = col_names

    df = data[['UID', 'price', 'postcode', 'PAON', 'SAON', 'street', 'locality', 'city', 'district', 'county']]
    return df


# function to write a dataframe to a csv file
def write_csv(file, df):
    df.to_csv(file, index=False)


# data normalisation function 1: converts every instance of a string passed as an argument to lowercase
def lowercase(address):
    try:
        if isinstance(address, str):
            return address.lower()
        else:
            return address
    except:
        print("error in function lowercase with address: " + str(address))


# data normalisation function 2: removes unnecessary spaces in the string passed as argument
def rem_space(address):
    try:
        if isinstance(address, str):
            # removing leading and trailing spaces
            address = address.strip()
            address = address.replace('\n', '')
            address = address.replace('\t', '')
            # removing more than one space
            address = re.sub(" +", " ", address)
    except:
        print("Exception in rem_space: address = " + str(address))
    return address


# data normalisation function 3: removes punctuation from the string passed as argument
def rem_punc(address):
    try:
        if isinstance(address, str):
            address = re.sub(r'[^\w\s]', " ", address)
    except:
        print("Exception in rem_punc: address = " + str(address))
    return address


# data normalisation function 4: removes stop words from the string passed as argument
def rem_stopwords(address):
    # new_address = []
    try:

        if isinstance(address, str):
            final_address = ""
            add_parts = address.split(" ")
            new_address = [add for add in add_parts if add not in stpwrds]
            for part in new_address:
                final_address = final_address + part + " "
            final_address = final_address[:-1]
            return final_address
        else:
            return address
    except:
        print("Exception in rem_stopwords: address = " + str(address))


# data normalisation function 5: removes all non-ascii characters from an input string
# requires an array of strings (tokenised) as its input
def rem_non_ascii(address):
    new_add = []
    try:
        address = str(address)
        for part in address:
            new_part = ""
            for letter in part:
                if ord(letter) < 128:
                    new_part = new_part + str(letter)
            new_add.append(new_part)
    except:
        print("Exception in rem_non_ascii: address = " + str(address))
    return new_add


# main function calling all other normalisation functions
# takes an address as input and returns a fully normalised version of the address
def normalise(address):
    new_add = lowercase(address)
    new_add = rem_punc(new_add)
    new_add = rem_space(new_add)
    new_add = rem_stopwords(new_add)
    # new_add = rem_non_ascii(new_add)
    # final_add = "".join(new_add)
    return new_add


# specifying output csv file path for both test and train data
output_train_file_path = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/cleaned_month_addresses_train.csv"
output_test_file_path = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/cleaned_month_addresses_test.csv"
output_validation_file_path = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/cleaned_month_addresses_validation.csv"


# driver function that takes the path for the input file, output train file and output test file as input
# iterates through the relevant columns of the dataframe and calls the main normalising function
# replaces the old dataframe addresses with the new normalised addresses
# splits the new dataframe into train and test data and writes them to the appropriate csv files
def standardise_data(input_path, output_train_path, output_test_path):
    df = read_data(input_path)

    col_count = 0
    for col in df.columns:
        if col != 'UID' and col != 'SAON' and col != 'locality' and col != 'price' and col != 'postcode':
            add_col = df[col]
            count = 0
            col_count += 1
            for index, address in add_col.iteritems():
                final_address = normalise(address)
                add_col[index] = final_address
                count += 1
                print(col_count, "--", count)
            df[col] = add_col
    # postcode_col = df['postcode']
    # for index, address in postcode_col.iteritems():
    #     final_address = str(address[0]) + "-" + str(address[1])
    #     postcode_col[index] = final_address
    # df['postcode'] = postcode_col
    df_copy = df
    train_df = df_copy.sample(frac=0.75, random_state=0)
    test_df = df_copy.drop(train_df.index)
    train_df_copy = train_df
    final_train_df = train_df_copy.sample(frac=0.8, random_state=0)
    validation_df = train_df_copy.drop(final_train_df.index)
    print("train df", final_train_df)
    print("test df", test_df)
    print("validation df", validation_df)
    write_csv(output_train_path, final_train_df)
    write_csv(output_test_path, test_df)
    write_csv(output_validation_file_path, validation_df)
    print("written to file!!!!!")


def missing_data(df):
    for col in df.columns:
        # print(df.columns)
        print(col)
        print(df[col].isnull().sum().sum() / len(df[col]))


def unique_vals(df):
    unique_vals_dict = {}
    labels = []
    values = []
    for col in df.columns:
        if col != 'UID' and col != 'price' and col != 'postcode' and col != 'SAON' and col != 'locality':
            unique_vals_dict[col] = len(df[col].unique()) / len(df)
            labels.append(col)
            values.append(len(df[col].unique()) / len(df))
            if col == 'county':
                print(len(df[col].unique()))
    index = np.arange(len(labels))
    plt.bar(index, values)
    plt.xlabel('Column Names', fontsize=5)
    plt.ylabel('Percentage of Unique Values', fontsize=5)
    plt.xticks(index, labels, fontsize=5, rotation=30)
    plt.title('Column wise Unique Values (%)')
    plt.savefig('unique_vals_per_column.png')


def price_summary(df):
    county_freq = pd.DataFrame(df['county'].value_counts())
    most_freq_county = list(county_freq.index)[:5]
    county_wise_price_dict = {}
    for county in most_freq_county:
        county_wise_price_dict[county] = []
    for index, price in df['price'].iteritems():
        if df.iloc[index]['county'] in most_freq_county:
            county = df.iloc[index]['county']
            county_wise_price_dict[county].append(price)
    print(county_wise_price_dict)


def EDA(input_path):
    df = read_data(input_path)
    missing_data(df)
    unique_vals(df)
    price_summary(df)
    # dropping SAON and locality


# calling driver function
standardise_data(input_file_path, output_train_file_path, output_test_file_path)
# EDA(input_file_path)
