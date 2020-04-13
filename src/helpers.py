import re
import pandas as pd
import operator 
import numpy as np
import os

# Add 'datatype' column that indicates if the record is original wiki answer as 0, training data 1, test data 2, onto 
# the dataframe - uses stratified random sampling (with seed) to sample by task & plagiarism amount 

# Use function to label datatype for training 1 or test 2 
def create_datatype(df, train_value, test_value, datatype_var, compare_dfcolumn, operator_of_compare, value_of_compare,
                    sampling_number, sampling_seed):
    # Subsets dataframe by condition relating to statement built from:
    # 'compare_dfcolumn' 'operator_of_compare' 'value_of_compare'
    df_subset = df[operator_of_compare(df[compare_dfcolumn], value_of_compare)]
    df_subset = df_subset.drop(columns = [datatype_var])
    
    # Prints counts by task and compare_dfcolumn for subset df
    #print("\nCounts by Task & " + compare_dfcolumn + ":\n", df_subset.groupby(['Task', compare_dfcolumn]).size().reset_index(name="Counts") )
    
    # Sets all datatype to value for training for df_subset
    df_subset.loc[:, datatype_var] = train_value
    
    # Performs stratified random sample of subset dataframe to create new df with subset values 
    df_sampled = df_subset.groupby(['Task', compare_dfcolumn], group_keys=False).apply(lambda x: x.sample(min(len(x), sampling_number), random_state = sampling_seed))
    df_sampled = df_sampled.drop(columns = [datatype_var])
    # Sets all datatype to value for test_value for df_sampled
    df_sampled.loc[:, datatype_var] = test_value
    
    # Prints counts by compare_dfcolumn for selected sample
    #print("\nCounts by "+ compare_dfcolumn + ":\n", df_sampled.groupby([compare_dfcolumn]).size().reset_index(name="Counts") )
    #print("\nSampled DF:\n",df_sampled)
    
    # Labels all datatype_var column as train_value which will be overwritten to 
    # test_value in next for loop for all test cases chosen with stratified sample
    for index in df_sampled.index: 
        # Labels all datatype_var columns with test_value for straified test sample
        df_subset.loc[index, datatype_var] = test_value

    #print("\nSubset DF:\n",df_subset)
    # Adds test_value and train_value for all relevant data in main dataframe
    for index in df_subset.index:
        # Labels all datatype_var columns in df with train_value/test_value based upon 
        # stratified test sample and subset of df
        df.loc[index, datatype_var] = df_subset.loc[index, datatype_var]

    # returns nothing because dataframe df already altered 
    
def train_test_dataframe(random_seed=100):
    def inner(clean_df): 
        new_df = clean_df.copy()

        # Initialize datatype as 0 initially for all records - after function 0 will remain only for original wiki answers
        new_df.loc[:,'Datatype'] = 0

        # Creates test & training datatypes for plagiarized answers (1,2,3)
        create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.gt, 0, 1, random_seed)

        # Creates test & training datatypes for NON-plagiarized answers (0)
        create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.eq, 0, 2, random_seed)
        
        # creating a dictionary of categorical:numerical mappings for plagiarsm categories
        mapping = {0:'orig', 1:'train', 2:'test'} 

        # traversing through dataframe and replacing categorical data
        new_df.Datatype = [mapping[item] for item in new_df.Datatype] 
        return new_df

    return inner


# helper function for pre-processing text given a file
def process_file(file):
    # put text in all lower case letters 
    all_text = file.read().lower()

    # remove all non-alphanumeric chars
    all_text = re.sub(r"[^a-zA-Z0-9]", " ", all_text)
    # remove newlines/tabs, etc. so it's easier to match phrases, later
    all_text = re.sub(r"\t", " ", all_text)
    all_text = re.sub(r"\n", " ", all_text)
    all_text = re.sub("  ", " ", all_text)
    all_text = re.sub("   ", " ", all_text)
    
    return all_text


def create_text_column(df, file_directory='./input/data/'):
    '''Reads in the files, listed in a df and returns that df with an additional column, `Text`. 
       :param df: A dataframe of file information including a column for `File`
       :param file_directory: the main directory where files are stored
       :return: A dataframe with processed text '''
   
    # create copy to modify
    text_df = df.copy()
    
    # store processed text
    text = []
    
    # for each file (row) in the df, read in the file 
    for row_i in df.index:
        filename = df.iloc[row_i]['File']
        #print(filename)
        file_path = file_directory + filename
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:

            # standardize text using helper function
            file_text = process_file(file)
            # append processed text to list
            text.append(file_text)
    
    # add column to the copied dataframe
    text_df['Text'] = text
    
    return text_df

def create_containment_features(df, n, column_name=None):

    containment_values = []

    if(column_name==None):
        column_name = 'c_'+str(n) # c_1, c_2, .. c_n

    # iterates through dataframe rows
    for i in df.index:
        file = df.loc[i, 'File']
        # Computes features using calculate_containment function
        if df.loc[i,'Category'] > -1:
            c = calculate_containment(df, n, file)
            containment_values.append(c)
        # Sets value to -1 for original tasks
        else:
            containment_values.append(-1)

    print(str(n)+'-gram containment features created!')
    return containment_values

def create_lcs_features(df, column_name='lcs_word'):

    lcs_values = []

    # iterate through files in dataframe
    for i in df.index:
        # Computes LCS_norm words feature using function above for answer tasks
        if df.loc[i,'Category'] > -1:
            # get texts to compare
            answer_text = df.loc[i, 'Text']
            task = df.loc[i, 'Task']
            # we know that source texts have Class = -1
            orig_rows = df[(df['Class'] == -1)]
            orig_row = orig_rows[(orig_rows['Task'] == task)]
            source_text = orig_row['Text'].values[0]

            # calculate lcs
            lcs = lcs_norm_word(answer_text, source_text)
            lcs_values.append(lcs)
        # Sets to -1 for original tasks
        else:
            lcs_values.append(-1)

    print('LCS features created!')
    return lcs_values

def lcs_norm_word(answer_text, source_text):
    '''Computes the longest common subsequence of words in two texts; returns a normalized value.
       :param answer_text: The pre-processed text for an answer text
       :param source_text: The pre-processed text for an answer's associated source text
       :return: A normalized LCS value'''
    a = answer_text.split()
    s = source_text.split()
    a_count = len(a)
    s_count = len(s)
    m = np.zeros(shape=(s_count + 1, a_count + 1), dtype=int)

    for s_idx, s_word in enumerate(s, 1):
        for a_idx, a_word in enumerate(a, 1):
            if s_word == a_word:
                m[s_idx][a_idx] = m[s_idx-1][a_idx-1] + 1
            else:
                m[s_idx][a_idx] = max(m[s_idx-1][a_idx], m[s_idx][a_idx-1])
    lcs = m[s_count][a_count]
    normalized = lcs / a_count
    return normalized

from sklearn.feature_extraction.text import CountVectorizer
# Calculate the ngram containment for one answer file/source file pair in a df
def calculate_containment(df, n, answer_filename):
    '''Calculates the containment between a given answer text and its associated source text.
       This function creates a count of ngrams (of a size, n) for each text file in our data.
       Then calculates the containment by finding the ngram count for a given answer text,
       and its associated source text, and calculating the normalized intersection of those counts.
       :param df: A dataframe with columns,
           'File', 'Task', 'Category', 'Class', 'Text', and 'Datatype'
       :param n: An integer that defines the ngram size
       :param answer_filename: A filename for an answer text in the df, ex. 'g0pB_taskd.txt'
       :return: A single containment value that represents the similarity
           between an answer text and its source text.
    '''
    answer = df.copy().loc[df['File'] == answer_filename, :]
    answer_task = str(answer['Task'].values[0])
    answer_text = str(answer['Text'].values[0])
    source = df.loc[(df['Task'] == answer_task) & (df['Datatype'] == 'orig'), :]
    source_text = str(source['Text'].values[0])

    count_ngram = CountVectorizer(analyzer='word', ngram_range=(n,n))
    ngram_array = count_ngram.fit_transform([answer_text, source_text]).toarray()
    intersection_xs = np.amin(ngram_array, axis=0)
    intersection = np.sum(intersection_xs)
    answer_cnt = np.sum(ngram_array[0])
    return intersection / answer_cnt


# Takes in dataframes and a list of selected features (column names)
# and returns (train_x, train_y), (test_x, test_y)
def train_test_data(complete_df, features_df, selected_features):
    '''Gets selected training and test features from given dataframes, and
       returns tuples for training and test features and their corresponding class labels.
       :param complete_df: A dataframe with all of our processed text data, datatypes, and labels
       :param features_df: A dataframe of all computed, similarity features
       :param selected_features: An array of selected features that correspond to certain columns in `features_df`
       :return: training and test features and labels: (train_x, train_y), (test_x, test_y)'''

    # get the training features
    df = pd.concat([complete_df, features_df[selected_features]], axis=1)
    train_df = df[df['Datatype'] == 'train']
    test_df = df[df['Datatype'] == 'test']

    # And training class labels (0 or 1)
    train_x = train_df.loc[:, selected_features].values
    train_y = train_df['Class'].values

    # get the test features and labels
    test_x = test_df.loc[:, selected_features].values
    test_y = test_df['Class'].values

    return (train_x, train_y), (test_x, test_y)

def make_csv(x, y, filename, data_dir):
    '''Merges features and labels and converts them into one csv file with labels in the first column.
       :param x: Data features
       :param y: Data labels
       :param file_name: Name of csv file, ex. 'train.csv'
       :param data_dir: The directory where files will be saved
       '''
    # make data dir, if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # your code here
    pd.concat([pd.DataFrame(y), pd.DataFrame(x)], axis=1) \
        .to_csv(os.path.join(data_dir, filename), header=False, index=False)


    # nothing is returned, but a print statement indicates that the function has run
    print('Path created: '+str(data_dir)+'/'+str(filename))
