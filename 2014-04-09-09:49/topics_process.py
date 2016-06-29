from __future__ import division
from __future__ import print_function
from time import time
import re
import json
import operator
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import bigrams
from collections import defaultdict
import string
import vincent
import pandas
from datetime import datetime, timedelta
import math
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pyLDAvis
 
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']
 
def tokenize(s):
    #return word_tokenize(tweet)
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def terms_all(terms): return [term for term in terms]
def terms_stop(terms): return [term for term in terms if term not in stop]
def terms_bigram(terms): return bigrams(terms)

# Count terms only once, equivalent to Document Frequency
def terms_single(terms): return set(terms)

# Count hashtags only
def terms_hash(terms): return [term for term in terms if term.startswith('#') and len(term) > 1]

# Count terms only (no hashtags, no mentions)
def terms_only(terms):
    # mind the ((double brackets))
    # startswith() takes a tuple (not a list) if 
    # we pass a list of inputs
    return [term for term in terms if term not in stop and not term.startswith(('#', '@'))] 

def terms_nolink(terms):
    return [term for term in terms if 'http' not in term] 

def terms_manual(terms, remove_words = ['I']):
    return [term for term in terms if term not in remove_words]

def terms_filter(tokens):
    #terms = terms_hash(tokens)
    terms = terms_nolink(terms_stop(terms_only(tokens)))
    terms = terms_manual(terms, [u'\ud83d', u'I', u'\ud83c', u"I'm", u'w', u'\ufe0f', u'3', u'\u2764', u'\u2665', u'2', u'The'])
    #terms = terms_bigram(terms)
    return terms

def getText(tweet):
    if tweet.get('text', None):
        return tweet.get('text', None).strip()

def getTime(tweet):
    tweetDate = tweet.get('created_at', None)
    utcOffset = tweet.get('user').get('utc_offset')

    if utcOffset is not None:
        tweetCorrectedDate = datetime.strptime(tweetDate, '%a %b %d %H:%M:%S +0000 %Y') + timedelta(seconds=int(utcOffset))
    else:
        tweetCorrectedDate = datetime.strptime(tweetDate, '%a %b %d %H:%M:%S +0000 %Y')

    return tweetCorrectedDate

def preprocess_text(text):
    tokens = preprocess(text, True)
    terms = terms_nolink(terms_only(terms_stop(tokens)))
    sample = ' '.join(terms)
    #sample = text
    return sample, len(terms)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def nmf_topic(data_samples, n_features, n_topics, n_top_words):
    n_samples = len(data_samples)

    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                       stop_words='english')
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))

    # Fit the NMF model
    print("Fitting the NMF model with tf-idf features,"
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    t0 = time()
    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
    #exit()
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

def lda_topic(data_samples, n_features, n_topics, n_top_words):
    n_samples = len(data_samples)

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                    stop_words='english')
    #t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    # print("done in %0.3fs." % (time() - t0))

    # print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..." % (n_samples, n_features))
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0)
    #t0 = time()
    lda.fit(tf)
    # print("done in %0.3fs." % (time() - t0))

    # print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)


def lda_viz(docs, lengths, n_features, n_topics, n_top_words):
    n_samples = len(docs)

    norm = lambda data: pandas.DataFrame(data).div(data.sum(1),axis=0).values
    
    vect = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                    stop_words='english')
    vected = vect.fit_transform(docs)
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0)
    doc_topic_dists = norm(lda.fit_transform(vected))
    
    prepared = pyLDAvis.prepare(
                        doc_lengths = lengths,
                        vocab = vect.get_feature_names(),
                        term_frequency = vected.sum(axis=0).tolist()[0],
                        topic_term_dists = norm(lda.components_),
                        doc_topic_dists = doc_topic_dists,
                        )

    #print(doc_topic_dists)
    #print(n_samples)
    return prepared, doc_topic_dists

def time_series(dates, period='1Min', counts=None):
    # a list of "1" to count the hashtags
    if counts is None: counts = [1]*len(dates)
    # the index of the series
    idx = pandas.DatetimeIndex(dates)
    # the actual series (at series of 1s for the moment)
    series = pandas.Series(counts, index=idx)
    per_minute = series.resample(period, how='sum').fillna(0)
    return per_minute

def main(fn):
    with open(fn + '.json', 'r') as f:
        data_samples = []
        doc_lengths = []
        lb = datetime(2014, 04, 25) #datetime(2014, 01, 01)
        ub = datetime(2014, 04, 27) #datetime(2014, 12, 12)

        for line in f:
            tweet = json.loads(line)
            if getTime(tweet) < lb or getTime(tweet) > ub:
                continue
            
            sample, length = preprocess_text(getText(tweet))
            data_samples.append(sample)
            doc_lengths.append(length)

    n_features = 1000
    n_topics = 10
    n_top_words = 20

    #lda_topic(data_samples, n_features, n_topics, n_top_words)
    data_viz, _ = lda_viz(data_samples, doc_lengths, n_features, n_topics, n_top_words)
    #data_viz = pyLDAvis.prepare(**data_viz)
    #pyLDAvis.show(data_viz)
    pyLDAvis.save_html(data_viz, 'topics.html')

def main2(fn):
    with open(fn + '.json', 'r') as f:
        data_samples = []
        doc_lengths = []
        dates = []

        for line in f:
            tweet = json.loads(line)
            sample, length = preprocess_text(getText(tweet))
            data_samples.append(sample)
            doc_lengths.append(length)
            dates.append(getTime(tweet))

    n_features = 1000
    n_topics = 10
    n_top_words = 20

    #lda_topic(data_samples, n_features, n_topics, n_top_words)
    data_viz, doc_topic_dists = lda_viz(data_samples, doc_lengths, n_features, n_topics, n_top_words)
    #print(type(data_viz))

    counts = []
    for i in range(0, len(dates)):
        #count = sum(1 for topic_prob in doc_topic_dists[i] if topic_prob >= 0.5)
        count = 1 if doc_topic_dists[i][9] >= 0.5 else 0
        #rint(count)
        counts.append(count)

    per_minute = time_series(dates, '1D', counts)
    per_minute.to_csv(fn + '.topic.csv', sep='\t', encoding='utf-8')

    # and now the plotting
    time_chart = vincent.Line(per_minute)
    time_chart.axis_titles(x='Time', y='Freq')
    time_chart.to_json(fn + '.topic.time_chart.json')

if __name__ == '__main__':
    #fn = 'test.sample.en'
    fn = 'test.en'
    main(fn)
    #main2(fn)