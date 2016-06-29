from __future__ import print_function
import json
import pandas
import matplotlib.pyplot as plt
import math

def getLang(tweet):
    return tweet.get('lang', '')

def main():
	fn = 'test'
	#fn = 'test.sample'
	f = open(fn + '.json')
	tweets = []
	for line in f:
	    tweet = json.loads(line)
	    tweets += [tweet]

	fw = open(fn + '.en.json','w')
	for tweet in tweets:
		if getLang(tweet) == 'en':
			 print(json.dumps(tweet), file=fw)

	fw.close()

def main2():
	fn = 'test'
	#fn = 'test.sample'
	f = open(fn + '.json')

	langs = dict()
	n_tweets = 0

	for line in f:
		n_tweets += 1
		tweet = json.loads(line)
		lang = getLang(tweet)
		if lang:
			count = langs.get(lang, 0) + 1
			langs[lang] = count
	
	print(n_tweets)
	print(sum(langs.values()))
	print(langs)
	df = pandas.DataFrame(langs.values(), index=langs.keys(), columns=[''])
	df.plot.pie(subplots=True, figsize=(8, 6), legend=False)
	plt.savefig(fn + '.plot.pdf', dpi=300)
	#plt.show()

if __name__ == '__main__':
	#main()
	main2()