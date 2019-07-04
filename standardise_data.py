# Geowox: Technical Assignment for Data Engineer-Graduate Position
# written by Ayushree Gangal

# importing the required libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from num2words import num2words

# please specify the default input file path here (the path for properties.csv)
# currently it's the directory containing the python file
input_fp = "./properties.csv"
# please specify the default output file path here (the new csv file is called cleaned_properties.csv)
# currently it's the directory containing the python file
output_fp = "./cleaned_properties.csv"

# loading the stopwords list using nltk
stpwrds = stopwords.words('english')


# this function requires an input file path (as specified above)
# it reads the csv file and converts it into a pandas dataframe and returns the created dataframe
def read_csv(filepath):
    df = pd.read_csv(filepath)
    return df


# this function requires a string input and it converts the string to lowercase and returns the new string
def lowercase(address):
    try:
        return address.lower()
    except:
        print("Exception in lowercase: address = " + address)


# this function requires a string input and aims to remove unwanted spaces, tabs and newlines
# requires string as its input
def rem_space(address):
    try:
        # removing leading and trailing spaces
        address = address.strip()
        # removing more than one space
        address = re.sub(" +", " ", address)
        # replacing new lines with empty space
        address = address.replace('\n', '')
        # returns new string
    except:
        print("Exception in rem_space: address = " + address)
    return address


# this function replaces all punctuation marks with spaces using regex and returns the new string
# requires string as its input
def rem_punc(address):
    try:
        address = re.sub(r'[^\w\s]', " ", address)
    except:
        print("Exception in rem_punc: address = " + address)
    return address


# this function removes stopwords by using the nltk package
# requires string as its input
def rem_stopwords(address):
    new_address = []
    try:
        # tokenising the string 'address'
        add_parts = address.split(" ")
        # removing stopwords based on the stpwrds list declared above
        new_address = [add for add in add_parts if add not in stpwrds]
        # returns an array with words from the input string after removing the stop words
    except:
        print("Exception in rem_stopwords: address = " + address)
    return new_address


# this function removes all non-ascii characters from an input string
# requires an array of strings (tokenised) as its input
def rem_non_ascii(address):
    # new_add will store the strings after all non-ascii characters have been removed
    new_add = []
    try:
        # for each string in the input array
        for part in address:
            # each string is deconstructed into letters
            # all non-ascii letters are removed
            # remaining letters are joined to form the new string
            # the new string is stored in new_part
            new_part = ""
            for letter in part:
                # ord() will return the unicode code of each letter and for ascii characters, this should be less than 128
                if ord(letter) < 128:
                    # if the letter is an ascii character, it is added to the new_part string
                    new_part = new_part + letter
            # appending the new array with each new string
            new_add.append(new_part)
        # returning the new array
    except:
        print("Exception in rem_non_ascii: address = " + address)
    return new_add


# this function aims to remove duplicates from the address (later occurrences)
# requires an array of strings (tokenised) as its input
def rem_dup(address):
    # a dictionary is used for fast lookup
    # each string in the array is stored as a key with its first index as the value
    dict = {}
    # rem_index keeps track of which indices need to be removed
    rem_index = []
    try:
        # iterating over the input array
        for i in range(len(address)):
            if address[i] not in dict.keys():
                # if the string is not already a part of the keys of the dictionary, it is added
                # its index is added as the corresponding value
                dict[address[i]] = i
            else:
                # else the string has already occurred at some point (say at index i)
                # and is being repeated right now (say at index j) such that i<j
                # thus, index j represents a duplicate element and is thus added to the rem_index array
                rem_index.append(i)
        # traversing the rem_index array from the end to the start (so as to not change the indices of the duplicates)
        for i in reversed(rem_index):
            # the appropriate elements are removed from the input array
            address.pop(i)
    except:
        print("Exception in rem_dup: address = " + address)
    return address


# this function converts ordinals in numbers to their equivalent in words
# requires an array of strings (tokenised) as its input
def convert_ordinals(address):
    # new_add will store the final strings after they have been transformed
    new_add = []
    # this array stores the suffixes for valid ordinals
    valid_suffix = ["th", "st", "rd", "nd"]
    try:
        # for each string in the input array
        for str in address:
            # regex is used to determine if it's starting with a digit followed by
            # either more digits and then letters. This is stored in 'ordinal'
            # note here that strings entered incorrectly such as "403angelica" will not be
            # treated as an ordinal and will be left unchanged
            ordinal = re.findall('^\d+[a-z]+', str)
            # strings not satisfying the specified regex are appended to the new address array without any change
            if ordinal == []:
                new_add.append(str)
            # similarly, strings that don't end in a valid prefix or don't have their third last element as a number
            # are considered invalid ordinals and are added to the new array without any changes
            elif ordinal[0][-2:] not in valid_suffix or not ordinal[0][-3].isdigit():
                new_add.append(str)
            # if the above conditions are not satisfied, then the string is considered to be a valid ordinal
            # and is thus converted to its equivalent word
            else:
                # regex is first used to remove the trailing suffix
                # subsequently, num2words converts the resulting number to an ordinal
                new_str = num2words(re.findall('^\d+', ordinal[0])[0], to='ordinal')
                # now that the ordinal has been expressed in words, it needs to be normalised too
                new_str = normalise(new_str)
                # At this point, the result is an array of strings containing each word of the ordinal
                # the next few steps convert this array back to a single string
                final_str = "".join(new_str)
                # finally, the transformed ordinal is appended to the new address array
                new_add.append(final_str)
    except:
        print("Exception in convert_ordinals: address = " + address)
    return new_add


# this function is the master function that calls all the other functions in the appropriate order
# it requires a string input
def normalise(address):
    # calling all other functions
    new_add = lowercase(address)
    new_add = rem_punc(new_add)
    new_add = rem_space(new_add)
    new_add = rem_stopwords(new_add)
    new_add = rem_non_ascii(new_add)
    new_add = rem_dup(new_add)
    new_add = convert_ordinals(new_add)
    # the result at this point is an array of strings
    # the next step converts this array back to a single string
    final_add = " ".join(new_add)
    # the transformed address is returned
    return final_add


# this function writes a pandas dataframe to csv
# it requires an output filepath and the dataframe as input
def write_csv(filepath, df):
    df.to_csv(filepath, index=False)


def standardise_data(input_path, output_path):
    try:
        # checking to see if the user specified file path is valid
        file = open(input_path, 'r')
        inp_file_path = input_path
    except FileNotFoundError:
        # if the user specified path is not valid, then the harcoded input file path is used instead
        print("Input path specified by user is invalid. Using default path instead.")
        inp_file_path = input_fp
    # read data using a function called read_csv that requires the input file path as its argument
    data = read_csv(inp_file_path)
    # extracting the address column from the data
    add_col = data['Address']
    # normalising the address column
    for index, address in add_col.iteritems():
        final_add = normalise(address)
        add_col[index] = final_add
    # overwriting the original address column in the data
    data['Address'] = add_col
    # writing new data to a csv file (requires an output file path)
    try:
        # checking if the user specified output path is valid
        file = open(output_path, 'w+')
        out_file_path = output_path
    except FileNotFoundError:
        # if the user specified output path is invalid, then the hardcoded default output path is used
        print("Output path specified by user is invalid. Using default path instead.")
        out_file_path = output_fp
    write_csv(out_file_path, data)


# to run the program, the user is asked to specify the path to the data file
# the path will be validated and if found invalid, it will use a default hardcoded file path
user_inp_path = input("Please enter a file path to the data file: ")
user_out_path = input("Please enter a file path where the output file should be created: ")
# calling the main function that requires a file path as input
standardise_data(user_inp_path, user_out_path)
