# import libraries
import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine 

# load messages dataset
messages = pd.read_csv('messages.csv')

# load categories dataset
categories = pd.read_csv('categories.csv')

# merge datasets
df = messages.merge(categories, how='outer', on=['id'])

# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';', expand=True, n=36)

# select the first row of the categories dataframe
row = categories[1:2]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = row.apply(lambda x : x.str.slice(0,-2))

#converting index from multi to single 
category_colnames_list = category_colnames.values.tolist()
categories.columns = category_colnames_list

category_col_clean = []

for cat in categories:
    category_col_clean.append(cat[0])

categories.columns = category_col_clean

categories_clean = categories.copy()

#Convert category values to just numbers 0 or 1

for column in category_colnames_list:
    # set each value to be the last character of the string
    categories_clean[column] = categories_clean[column].apply(lambda x : x.str.slice(-1))
    
    # convert column from string to numeric
    categories_clean[column] = categories_clean[column].astype(int)

# drop the original categories column from `df`
df_clean_one = df.drop(columns=['categories'])

# concatenate the original dataframe with the new `categories` dataframe
df_clean_two = pd.concat([df_clean_one, categories_clean],join='inner', axis=1)

# drop duplicates
df_clean_two['message'] = df_clean_two['message'].drop_duplicates()

#save the clean dataset into an sqlite database.
engine = create_engine('sqlite:///disaster-data.db')
df_clean_two.to_sql('messages', engine, index=False)



