import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


def vectorize_messages(iterable, stop_words='english', min_df=10, 
    vectorizerObj=CountVectorizer, **kwargs):
    """
    Vectorize an iterable of text documents into a sparse matrix counting the 
    occurences of each word in each document. This process is done using the
    `CountVectorizer` from `scikit-learn`. Returns the matrix `X` and its
    corresponding `vectorizer`
    """
    vectorizer = vectorizerObj(stop_words=stop_words, min_df=min_df, **kwargs)
    X = vectorizer.fit_transform(iterable)
    # Build the inverse dictionary that maps indices to words
    indices = list(vectorizer.vocabulary_.values())
    keys = np.array(list(vectorizer.vocabulary_.keys()))
    args = np.argsort(indices)
    vectorizer.inverse_vocabulary_ = list(keys[args])
    return X, vectorizer


