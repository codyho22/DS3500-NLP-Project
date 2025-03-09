"""

File: graphics.py
Description: Generate Graphics for NLP Processing Results

"""
#import necessary libraries
from NLP import index_commonality_df, is_stop_word, get_stores, initialize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from NLP._stores import stores as st
import plotly.graph_objects as go
from wordcloud import WordCloud
import os

#establish path
folder_path = '/data'
files_in_folder = os.listdir(os.getcwd()+folder_path)
files = []
for file in files_in_folder:
    files.append(folder_path[1:] + "/" + file)

initialize(files)
commonality = index_commonality_df()

df = commonality["wordcount"]
df["index"] = df.index.astype(str)
df = df[df["index"].apply(is_stop_word) == False]
df = df[df["index"] != "Jolene"]
df = df[df["index"] != " "]
df = df.dropna()

#assign data variable from get_stores(), as data variable will be used in last two vizs
data = get_stores()
lst_files = []
for i in list(data.keys()):
    lst_files.append(i.split('/')[-1])



def bar_plot(df: pd.DataFrame, title, x_label, y_label, limit=10):
    """Graphs a bar plot with the index as the x labels"""
    df = df.nlargest(limit)
    print(df)
    fig = plt.figure(figsize=(10, 5))
    plt.bar(df.index, df.values, color="blue", width=0.4)

    plt.xlabel(x_label)
    ylabel = df.columns[0] if y_label == None else y_label
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def sentiment_distributions():
    """Compare Sentiment Distributions"""
    feature = "sentiment"
    names = [key[key.find("/") + 1 : key.find(".")] for key in list(st.stores.keys())]
    fig_size = len(names)
    fig, axs = plt.subplots(int(fig_size/2), 2, figsize=(15, 10), sharey=True)
    axs = axs.flatten()
    fig.suptitle("Sentiment Progression")
    order = list(range(fig_size))[::2] + list(range(fig_size))[1::2]
    for i, (_, store) in enumerate(st.stores.items()):
        axs[order[i]].plot(list(range(store[feature].shape[0])), store[feature], color = "blue" if order[i] % 2 else "orange" )
        axs[order[i]].set_title(f"Sentiment {names[i]}; Mean:{float(store[feature].mean()):.2f}")    
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def make_sankey(df, src, targ):
    """
        :param df: Input dataframe
        :param src: the column in df used for source for sankey
        :param targ:  the column in df used for target for sankey
        :return: Plotly Figure object
        """

    #Store unique labels from src and targ, to remove repeating after groupby
    source_labels = list(df[src].unique())
    target_labels = list(df[targ].unique())

    # map labels to indexes
    link = {
        'source': df[src].map(lambda x: source_labels.index(x)),
        'target': df[targ].map(lambda x: len(source_labels) + target_labels.index(x)),
        'value': df['value']
    }

    node = {'label': source_labels + target_labels}
    #make sankey
    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)
    fig.show()

def sankey_wordcount(word_list=None, k=None):
    """
    Generates a Sankey diagram to visualize word counts across songs

    Args:
        word_list (list, optional): Specific set of words to include.
        k (int, optional): Number of most frequent words to include.
    """
    if k:
        # if k is passed, use the top k words in terms of total count
        #df refers to the dataframe with total word counts from all files
        sorted_df = df.sort_values(by='wordcount', ascending=False)
        word_list = list(sorted_df.head(k).index)

    # init empty list for creating a df
    data_list = []

    # Iterate over each word and file to fill in the data list with (name, word, count)
    for word in word_list:
        for file_name in lst_files:
            try:
                #get the count for that song and word
                count = data['data/' + file_name]["wordcount"].loc[word].wordcount
            except KeyError:
                # if keyerror, just input 0, keyerror means no occurance
                count = 0
            
            # Append data as tuple (filename, word, count) to the list
            data_list.append((file_name, word, count))

    # Create DataFrame from the list, set cols
    filtered_df = pd.DataFrame(data_list, columns=['filename', 'word', 'value'])

    #call make_sankey() using the filtered_df built, and src/targ cols from filtered_df
    make_sankey(filtered_df, 'filename', 'word')


def songs_wordcloud():
    """
    Makes and displays a wordcloud for each song
    """
    wordcloud = WordCloud(width=600, height=400, background_color='white')
    fig = plt.figure()
    #filter out stop words
    df = df[df["index"].apply(is_stop_word) == False]

    for i, file_name in enumerate(lst_files, 1):
        if i <= 9:
            #get the df returned from passing in the filename to data
            song_df = data['data/'+file_name]['wordcount']
            #remove any empty strings, as sometimes stopword filtering fails at removing these
            song_df.index.drop('')

            #filter out the stopwords
            song_df = song_df[~song_df.index.map(is_stop_word)]


            #add column to the df so that there is a 'word' column instead of using index
            song_df['word'] = song_df.index
            #turn it into a dict for word cloud processing
            song_dict = dict(zip(song_df['word'], song_df['wordcount']))
            song_dict
            
            cloud = wordcloud.generate_from_frequencies(song_dict)
            fig.add_subplot(4, 3, i)
            plt.imshow(cloud)
            plt.title(file_name, fontsize=10)
            plt.axis("off")

    plt.show()


def main():
    """ Runs the viz functions """

    sentiment_distributions()
    sankey_wordcount(word_list=['day', 'stand', 'love', 'Oh', 'days', 'boy', 'die', 'wait'])
    songs_wordcloud()
    

#run main
main()
        