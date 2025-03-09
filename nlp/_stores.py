"""

File: _stores.py
Description: Store text data for NLP processing.

"""
from collections import defaultdict
import pandas as pd
import numpy as np

class Stores:
    def __init__(self, as_df=True) -> None:
        self.store = self._df_store if as_df else self._dict_store
       
        self.stores = defaultdict(dict)
        self.text = ""

    def _df_store(self, name, feature, col:list|np.ndarray, idx:list|np.ndarray, val:list|np.ndarray, data:pd.DataFrame=None):
        """Store data as a dataframe in stores"""
        if data == None:
           data = pd.DataFrame(val, idx, col)


        self.stores[name][feature] = data

    def _dict_store(self):
        pass     
        


stores = Stores()