import pandas as pd
import numpy as np
import os
from functools import reduce
from helpers import create_text_column, train_test_dataframe, create_containment_features, create_lcs_features, calculate_containment, lcs_norm_word, train_test_data, make_csv

pipe = lambda fns: lambda x : reduce(lambda v, f: f(v), fns, x)

def load_df(path):
    return pd.read_csv(path)

def make_class(df):
    return df.assign(Class=df.Category.map({
            'non': 0, 'orig': -1, 'heavy': 1, 'light': 1, 'cut': 1}))

def make_category(df):
    return df.assign(Category=df.Category.map({
            'non': 0, 'heavy': 1, 'light': 2, 'cut': 3, 'orig': -1}))

def create_features_df(complete_df):
    ngram_range = range(1,7)
    features_list = []
    all_features = np.zeros((len(ngram_range)+1, len(complete_df)))
    i = 0
    for n in ngram_range:
        column_name = 'c_'+str(n)
        features_list.append(column_name)
        all_features[i]=np.squeeze(create_containment_features(complete_df, n))
        i+=1 
    features_list.append('lcs_word')
    all_features[i]= np.squeeze(create_lcs_features(complete_df))
    return (complete_df, pd.DataFrame(np.transpose(all_features), columns=features_list))

def split_data(df):
    complete_df, features_df = df
    selected_features = ['c_1', 'c_6', 'lcs_word']
    return train_test_data(complete_df, features_df, selected_features)

def save_csv(data):
    (train_x, train_y), (test_x, test_y) = data
    make_csv(train_x, train_y, filename='train.csv', data_dir='../models')
    make_csv(test_x, test_y, filename='test.csv', data_dir='../models')

program = pipe([
        make_class,
        make_category,
        create_text_column,
        train_test_dataframe(1),
        create_features_df,
        split_data,
        ])

main = pipe([
        load_df,
        program,
        save_csv
        ])


if __name__ == '__main__':
    main('../input/data/test_info.csv')
        
