from matplotlib import pyplot as plt
import numpy as np

import pandas as pd

from . import utils
from . import config

CATEGORIES = ['left', 'mainstream', 'right']
ENGAGEMENT_COLUMNS = ['reaction_count','comment_count','share_count']


def count_group(df, by, kind='bar', figsize=(8, 8), rot=0, hspace=.85):
    df['count'] = 1
    group = df.groupby(by)
    s = group.count()['count']
    s = s.unstack(level=0).sort(ascending=0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    s.plot(kind='bar', subplots=True, ax=ax, rot=rot)
    fig.subplots_adjust(hspace=hspace)
    plt.tight_layout()
    del df['count']


def barplot_count_posts_per_page(df):
    post_count = df['page'].value_counts().to_frame()
    post_count['order'] = range(len(post_count))
    post_count['category'] = [config.categories[p] for p in post_count.index]
    post_count['page_name'] = [config.page_name_map[p] for p in post_count.index]
    fig, ax = plt.subplots(1,1, figsize=(12,3))
    for cat in CATEGORIES:
        mask = post_count['category'] == cat
        ax.bar(post_count[mask]['order']+0.25, post_count[mask]['page'], width=0.5, 
               lw=0, label=cat, color=config.get_color(cat))
    ax.set_xticks(post_count['order']+0.6)
    ax.set_xticklabels(post_count['page_name'], rotation=60, ha='right')
    ax.tick_params(axis='x', which='both',length=0)
    ax.set_title('Number of articles published per page')
    ax.set_ylabel('Number of articles published')
    ax.set_xlabel('Page name')
    ax.legend()


def barplot_engagement(df, agg_func='median'):
    data = df.groupby('page').agg(agg_func)
    data = data.reindex(index=config.page_index)
    data = data.reset_index()
    data['order'] = data.index
    data['category'] = [config.categories[p] for p in data['page']]
    data['page_name'] = [config.page_name_map[p] for p in data['page']]
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    for ax, col in zip(axs, ENGAGEMENT_COLUMNS):
        for cat in CATEGORIES:
            mask = data['category'] == cat
            bar = ax.bar(data[mask]['order']+0.15, data[mask][col], width=0.8, lw=0, 
                         label=cat, color=config.get_color(cat))
            ax.set_title('{} {}'.format(col[:-6].capitalize(), str(agg_func)))
            utils.autolabel(bar, ax)
        ax.set_ylim(top=np.ceil((data[col].max()*1.15)/1e3)*1e3)
        ax.set_xticks(data['order']+0.95)
        ax.set_xticklabels(data['page_name'], rotation=60, ha='right')
        ax.tick_params(axis='x', which='both',length=0)
    axs[1].legend(ncol=3, prop={'size':11}, loc='upper center')
        

def aggregate_message_length(df, by, agg_funcs='median', message_len_col='Message length'):
    assert(message_len_col in df.columns, "{} not in dataframe".format(message_len_col))
    by_cleaned = by.replace('_', ' ').capitalize()
    data = df[message_len_col].to_frame()
    data[by_cleaned] = df[by]
    data = data[data[message_len_col] > 0]
    data = data[[by_cleaned, message_len_col]].groupby(by_cleaned).agg(agg_funcs)
    return data


def count_words(X, vectorizer):
    """
    Count the word occurences in the matrix `X` based on the vocabulary in 
    `vectorizer`
    """
    counts = np.array(X.sum(axis=0)).squeeze()
    serie = pd.Series(counts, index=vectorizer.inverse_vocabulary_)
    serie.sort(ascending=False)
    return serie