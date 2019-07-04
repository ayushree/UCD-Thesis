import pandas as pd
import re
from nltk.corpus import stopwords
from num2words import num2words

stpwrds = stopwords.words('english')

input_file_path = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/addresses_month.csv"


def read_data(file):
    col_names = ['UID', 'price', 'transfer_date', 'postcode', 'property_type', 'age', 'duration', 'PAON', 'SAON',
                 'street',
                 'locality', 'city', 'district', 'county', 'PPD_type', 'record_status']
    data = pd.read_csv(file, header=None)
    data.columns = col_names

    df = data[['UID', 'price', 'postcode', 'PAON', 'SAON', 'street', 'locality', 'city', 'district', 'county']]
    return df


def write_csv(file, df):
    df.to_csv(file, index=False)


def lowercase(address):
    try:
        if isinstance(address, str):
            return address.lower()
        else:
            return address
    except:
        print("error in function lowercase with address: " + str(address))


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


def rem_punc(address):
    try:
        if isinstance(address, str):
            address = re.sub(r'[^\w\s]', " ", address)
    except:
        print("Exception in rem_punc: address = " + str(address))
    return address


def rem_stopwords(address):
    new_address = []
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


# this function removes all non-ascii characters from an input string
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


def normalise(address):
    new_add = lowercase(address)
    new_add = rem_punc(new_add)
    new_add = rem_space(new_add)
    new_add = rem_stopwords(new_add)
    # new_add = rem_non_ascii(new_add)
    # final_add = "".join(new_add)
    return new_add


output_file_path = "/Users/ayushree/Desktop/ResearchProject/StatisticalAnalysisUsingH2O/cleaned_month_addresses.csv"


def standardise_data(input_path, output_path):
    df = read_data(input_path)
    count = 0
    for col in df.columns:
        if col != 'UID' and col != 'SAON' and col != 'locality':
            add_col = df[col]
            for index, address in add_col.iteritems():
                final_address = normalise(address)
                add_col[index] = final_address
                count += 1
                print(count)
            df[col] = add_col
    # postcode_col = df['postcode']
    # for index, address in postcode_col.iteritems():
    #     final_address = str(address[0]) + "-" + str(address[1])
    #     postcode_col[index] = final_address
    # df['postcode'] = postcode_col
    write_csv(output_path, df)


df = read_data(input_file_path)
for col in df.columns:
    # print(df.columns)
    print(col)
    print(df[col].isnull().sum().sum() / len(df[col]))
    # dropping SAON and locality

standardise_data(input_file_path, output_file_path)
