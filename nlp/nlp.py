from NLP._processing import load_text
from NLP._stores import stores as st

from collections import defaultdict
import pandas as pd
import numpy as np

def initialize(filenames: list, parser=None):
    for file in filenames:
        load_text(file, parser=parser)


def index_commonality_df():
    """Groups by the indices of the processed data frames and sums"""
    dfs = defaultdict(pd.DataFrame)
    for _, store in st.stores.items():
        for feature, df in store.items():
            dfs[feature] = pd.concat([dfs[feature], df])
    for feature, df in dfs.items():
        dfs[feature] = df.groupby(df.index).sum()
    return dfs

def sentiment_progression(smoothing = True):
    """Takes sentiment of each text store over each line optionally smoothing them
    then finds aggregate mean of all of them by index"""
    feature = "sentiment"
    df = pd.DataFrame({})
    for _, store in st.stores.items():
        if smoothing:
            # guass smoothing
            y = store[feature][feature].values
            kernel = np.exp(-np.linspace(-2, 2, 11)**2)
            smooth_y = np.convolve(y, kernel, mode='same') / np.sum(kernel) 
            store[feature][feature] = smooth_y
        df = pd.concat([df, store[feature]])   

    df = df.groupby(df.index).mean()            
    df = df[df["sentiment"] != 0 ]     # if sent
    return df    
   
def get_stores():
    return st.stores
