import pandas as pd

from lib.config import palette, color_idx

def load_dataset(filepath='data/dataset.csv'):
    df = pd.read_csv(filepath, index_col=0, 
        dtype={'rating': str}).reset_index(drop=True)
    df['created_time'] = pd.to_datetime(df['created_time'])
    return df

def get_colors(iterable, by='categories'):
    return [palette[color_idx[getattr(config, by)[e]]] for e in iterable]

def autolabel(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
