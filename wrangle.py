from env import host, user, password, get_db_url
import pandas as pd 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def acquire():
    '''
    This function takes in no arguments, and acquires data from a locally stored csv file and creates a dataframe.
    '''
    df = pd.read_csv('Life Expectancy Data.csv')
    return df

def attribute_nulls(df):
    nulls = df.isnull().sum()
    rows = len(df)
    percent_missing = nulls / rows 
    dataframe = pd.DataFrame({'rows_missing': nulls, 'percent_missing': percent_missing})
    return dataframe

def column_nulls(df):
    new_df = pd.DataFrame(df.isnull().sum(axis=1), columns = ['cols_missing']).reset_index()\
    .groupby('cols_missing').count().reset_index().\
    rename(columns = {'index': 'rows'})
    new_df['percent_missing'] = new_df.cols_missing/df.shape[1]
    return new_df
    
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def split_data(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    return train, validate, test

#remove outliers

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

# Scale data after splitting with MinMax
def scale_data_mvp(train, validate, test, return_scaler=False):
    '''
    Scales the 3 data splits.
    
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    
    If return_scaler is true, the scaler object will be returned as well.
    '''
    columns_to_scale = ['bedrooms', 'bathrooms', 'square_feet']
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

def visualize_scaler(scaler, df, target_columns, bins=10):
    fig, axs = plt.subplots(len(target_columns), 2, figsize=(16, 9))
    df_scaled = df.copy()
    df_scaled[target_columns] = scaler.fit_transform(df[target_columns])
    for (ax1, ax2), col in zip(axs, target_columns):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()
    return fig, axs

## dictionary to be used in imputing_missing_values function
columns_strategy = {
'mean' : [
       'calculatedfinishedsquarefeet',
       'finishedsquarefeet12',
     'structuretaxvaluedollarcnt',
        'taxvaluedollarcnt',
        'landtaxvaluedollarcnt',
        'taxamount'
    ],
    'most_frequent' : [
        'calculatedbathnbr',
         'fullbathcnt',
        'regionidcity',
         'regionidzip',
         'yearbuilt'
     ],
     'median' : [
         'censustractandblock'
     ]
 }

def impute_missing_values(df, columns_strategy):
    train, validate, test = split_data(df)
    
    for strategy, columns in columns_strategy.items():
        imputer = SimpleImputer(strategy = strategy)
        imputer.fit(train[columns])

        train[columns] = imputer.transform(train[columns])
        validate[columns] = imputer.transform(validate[columns])
        test[columns] = imputer.transform(test[columns])
    
    return train, validate, test



def prepare_who(df):
    '''Prepare who for data exploration.'''
    df = get_single_units(df)
    df = handle_missing_values(df)
    train, validate, test = impute_missing_values(df, columns_strategy)
    return train, validate, test
    
def wrangle_who():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_who(acquire())
    
    return train, validate, test
    
