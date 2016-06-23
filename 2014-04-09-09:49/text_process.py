from __future__ import division
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
import numpy as np
import matplotlib.pyplot as plt
 
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
 
def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: http://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857 
    - http://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # resize 
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(40, 20))

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
def terms_single(terms): return list(set(terms))

# Count hashtags only
def terms_hash(terms): return [term for term in terms if term.startswith('#')]

# Count terms only (no hashtags, no mentions)
def terms_only(terms):
    # mind the ((double brackets))
    # startswith() takes a tuple (not a list) if 
    # we pass a list of inputs
    return [term for term in terms if term not in stop and not term.startswith(('#', '@'))] 

def terms_nolink(terms):
    return [term for term in terms if 'http' not in term] 

def terms_filter(tokens):
    #terms = terms_hash(tokens)
    terms = terms_nolink(terms_stop(terms_only(tokens)))
    #terms = terms_bigram(terms)
    return terms

def co_occurrences(terms, com):
    if com is None: com = defaultdict(lambda : defaultdict(int))

    for i in range(len(terms)-1):            
        for j in range(i+1, len(terms)):
            w1, w2 = sorted([terms[i], terms[j]])                
            if w1 != w2:
                com[w1][w2] += 1

def contains_search_word(terms, search_word):
    if not search_word or search_word in terms:
        return terms
    else:
        return []

def getHashtags(tweet):
    hashtags = []
    if tweet.get('entities', None) is not None:
            if tweet.get('entities', None).get('hashtags', None) is not None:
                hashtagsList = tweet.get('entities', None).get('hashtags', None)

                for hashtagObj in hashtagsList:
                    #hashtags += "#" + hashtagObj.get('text', None).strip() + ","
                    hashtags.append("#" + hashtagObj.get('text', None).strip())

    return hashtags

def getTime(tweet):
    tweetDate = tweet.get('created_at', None)
    utcOffset = tweet.get('user').get('utc_offset')

    if utcOffset is not None:
        tweetCorrectedDate = datetime.strptime(tweetDate, '%a %b %d %H:%M:%S +0000 %Y') + timedelta(seconds=int(utcOffset))
    else:
        tweetCorrectedDate = datetime.strptime(tweetDate, '%a %b %d %H:%M:%S +0000 %Y')

    return tweetCorrectedDate

# For each term, look for the most common co-occurrent terms
def terms_max(com):
    com_max = []
    for t1 in com:
        t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
        for t2, t2_count in t1_max_terms:
            com_max.append(((t1, t2), t2_count))
    # Get the most frequent co-occurrences
    terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
    return terms_max

def time_series(dates, period='1Min', counts=None):
    # a list of "1" to count the hashtags
    if counts is None: counts = [1]*len(dates)
    # the index of the series
    idx = pandas.DatetimeIndex(dates)
    # the actual series (at series of 1s for the moment)
    series = pandas.Series(counts, index=idx)
    per_minute = series.resample(period, how='sum').fillna(0)
    return per_minute

def sentiment(count, com, n_docs, pos_file = None, neg_file = None):
    if pos_file is None:
        positive_vocab = [
            'good', 'nice', 'great', 'awesome', 'outstanding',
            'fantastic', 'terrific', ':)', ':-)', 'like', 'love',
            # shall we also include game-specific terms?
            # 'triumph', 'triumphal', 'triumphant', 'victory', etc.
        ]
    else:
        positive_vocab = [line.strip() for line in open(pos_file)]
    
    if neg_file is None:
        negative_vocab = [
            'bad', 'terrible', 'crap', 'useless', 'hate', ':(', ':-(',
            # 'defeat', etc.
        ]
    else:
        negative_vocab = [line.strip() for line in open(neg_file)]

    # n_docs is the total n. of tweets
    p_t = {}
    p_t_com = defaultdict(lambda : defaultdict(int))
     
    for term, n in count.items():
        p_t[term] = n / n_docs
        for t2 in com[term]:
            p_t_com[term][t2] = com[term][t2] / n_docs

    pmi = defaultdict(lambda : defaultdict(int))
    for t1 in p_t:
        for t2 in com[t1]:
            denom = p_t[t1] * p_t[t2]
            pmi[t1][t2] = math.log(p_t_com[t1][t2] / denom, 2)
     
    semantic_orientation = {}
    for term, n in p_t.items():
        positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
        negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
        semantic_orientation[term] = positive_assoc - negative_assoc

    semantic_sorted = sorted(semantic_orientation.items(), 
                         key=operator.itemgetter(1), 
                         reverse=True)
    return semantic_sorted

def main(fn):
    with open(fn + '.json', 'r') as f:
        count_all = Counter()
        com = defaultdict(lambda : defaultdict(int))
        search_word = ''

        for line in f:
            tweet = json.loads(line)
            tokens = preprocess(tweet['text'])
            
            #terms = terms_filter(tokens)
            terms = terms_single(terms_hash(tokens) + getHashtags(tweet))

            co_occurrences(terms, com)

            terms = contains_search_word(terms, search_word)
            count_all.update(terms)

            #print tweet['text']
            #print ' '.join(terms)
            #print tokens
    print(count_all.most_common(10))        
    print(terms_max(com)[:10])

def main2(fn):
    with open(fn + '.json', 'r') as f:
        count_all = Counter()
        search_word = ''

        for line in f:
            tweet = json.loads(line)
            tokens = preprocess(tweet['text'])
            
            #terms = terms_filter(tokens)
            terms = terms_single(terms_hash(tokens) + getHashtags(tweet))

            terms = contains_search_word(terms, search_word)
            count_all.update(terms)

    word_freq = count_all.most_common(10)
    print(word_freq)
    labels, freq = zip(*word_freq)
    data = {'data': freq, 'x': labels}
    bar = vincent.Bar(data, iter_idx='x')
    bar.to_json(fn + '.term_freq.json')

def main3(fn):
    with open(fn + '.json', 'r') as f:
        count_all = Counter()
        datess = []
        search_hashes = ['#Endomondo','#MexicoNeedsWWATour']

        for line in f:
            tweet = json.loads(line)
            tokens = preprocess(tweet['text'])
            
            terms = terms_hash(tokens)
            count_all.update(terms)

            for i in range(0,len(search_hashes)):
                datess.append([])
                if search_hashes[i] in terms:
                    datess[i].append(getTime(tweet))

    print(count_all.most_common(10)) 

    per_minutes = []
    for dates in datess:
        per_minute = time_series(dates)
        per_minutes.append(per_minute)

    keys = search_hashes
    values = per_minutes

    # all the data together
    match_data = dict(zip(keys,values))
    # we need a DataFrame, to accommodate multiple series
    all_matches = pandas.DataFrame(data=match_data,
                                   index=values[0].index)
    # Resampling as above
    all_matches = all_matches.resample('1Min', how='sum').fillna(0)
    #all_matches = all_matches.resample('1D', how='sum').fillna(0)

    # and now the plotting
    time_chart = vincent.Line(all_matches[keys])
    time_chart.axis_titles(x='Time', y='Freq')
    time_chart.legend(title='Matches')
    time_chart.to_json(fn + '.time_chart.json')
     
    # time_chart = vincent.Line(per_minute_1)
    # time_chart.axis_titles(x='Time', y='Freq')
    # time_chart.to_json('time_chart.json')

def main4(fn):
    with open(fn + '.json', 'r') as f:
        count_all = Counter()
        com = defaultdict(lambda : defaultdict(int))
        n_docs = 0
        docs = []

        for line in f:
            n_docs += 1
            tweet = json.loads(line)
            tokens = preprocess(tweet['text'], True)
            
            terms = terms_nolink(terms_stop(terms_only(tokens)))
            docs.append(terms)

            co_occurrences(terms, com)
            count_all.update(terms)

    #print(count_all.most_common(10)) 
    semantic_sorted = sentiment(count_all, com, n_docs, 'positive-words.txt', 'negative-words.txt')
    top_pos = semantic_sorted[:10]
    top_neg = semantic_sorted[-10:]
    print(top_pos)
    print(top_neg)
    print semantic_sorted

    semantic_sorted = dict(semantic_sorted)

    polarity = [0,0,0]
    for doc in docs:
        score = sum(semantic_sorted.get(term, 0) for term in doc)
        if len(doc) > 0: score = score / len(doc)
        #print (' '.join(doc), score)
        if score > 10: polarity[0] += 1
        elif score < -10: polarity[1] += 1
        else: polarity[2] += 1
    print polarity

def main5(fn):
    with open(fn + '.json', 'r') as f:
        count_all = Counter()
        dates = []
        search_hashes = '#WWAT'

        for line in f:
            tweet = json.loads(line)
            tokens = preprocess(tweet['text'])
            
            terms = terms_single(terms_hash(tokens) + getHashtags(tweet))

            count_all.update(terms)

            time = getTime(tweet)
            if time is not None and search_hashes in terms: dates.append(time)

    print(count_all.most_common(10)) 

    per_minute = time_series(dates, '1D')
    per_minute.to_csv(fn + '.series.csv', sep='\t', encoding='utf-8')

    # and now the plotting
    time_chart = vincent.Line(per_minute)
    time_chart.axis_titles(x='Time', y='Freq')
    time_chart.to_json(fn + '.count.time_chart.json')

def main6():
    # Generate data: 5 labels, 10 examples, binary.
    label_headers = 'Alice Bob Carol Dave Eve'.split(' ')
    label_data = np.random.randint(0,2,(10,5)) # binary here but could be any integer.
    print(label_data)

    # Compute cooccurrence matrix 
    cooccurrence_matrix = np.dot(label_data.transpose(),label_data)
    print(cooccurrence_matrix) 

    # Compute cooccurrence matrix in percentage
    # FYI: http://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    #      http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero/32106804#32106804
    cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))
    print('\ncooccurrence_matrix_percentage:\n{0}'.format(cooccurrence_matrix_percentage))

    # Add count in labels
    label_header_with_count = [ '{0} ({1})'.format(label_header, cooccurrence_matrix_diagonal[label_number]) for label_number, label_header in enumerate(label_headers)]  
    print('\nlabel_header_with_count: {0}'.format(label_header_with_count))

    # Plotting
    x_axis_size = cooccurrence_matrix_percentage.shape[0]
    y_axis_size = cooccurrence_matrix_percentage.shape[1]
    title = "Co-occurrence matrix\n"
    xlabel= ''#"Labels"
    ylabel= ''#"Labels"
    xticklabels = label_header_with_count
    yticklabels = label_header_with_count
    heatmap(cooccurrence_matrix_percentage, title, xlabel, ylabel, xticklabels, yticklabels)
    plt.savefig('image_output.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
    #plt.show()

if __name__ == '__main__':
    #fn = 'test.sample.en'
    fn = 'test.en'
    #fn = 'test'
    #main(fn)
    #main2(fn)
    #main3(fn)
    #main4(fn)
    main5(fn)
    #main6(fn)