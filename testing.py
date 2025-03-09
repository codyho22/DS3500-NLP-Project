from NLP import index_commonality_df
import os

# Specify the path to the folder
folder_path = '/data'
files_in_folder = os.listdir(os.getcwd()+folder_path)
files = []
for file in files_in_folder:
    files.append(folder_path[1:] + "/" + file)

print(index_commonality_df(files)["wordcount"])


def wordcount_sankey(self, word_list=None, k=5):
    # df in the jupyter is the wordcount dataframe
     sorted_df = df.sort_values('wordcount', ascending=False) 
     

