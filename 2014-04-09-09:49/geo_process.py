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

def main(fn):
    with open(fn + '.json', 'r') as f:
        geo_data = {
            "type": "FeatureCollection",
            "features": []
        }

        for line in f:
            tweet = json.loads(line)
            if tweet['coordinates']:
                geo_json_feature = {
                    "type": "Feature",
                    "geometry": tweet['coordinates'],
                    "properties": {
                        "text": tweet['text'],
                        "created_at": tweet['created_at']
                    }
                }
                geo_data['features'].append(geo_json_feature)

    # Save geo data
    with open(fn + '.geo.json', 'w') as fout:
        fout.write(json.dumps(geo_data, indent=4))

if __name__ == '__main__':
    #fn = 'test.sample.en'
    fn = 'test.en'
    main(fn)