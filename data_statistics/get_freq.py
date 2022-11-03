import json
import csv
import sys
import pickle
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from topic_modeling import get_texts

def get_hashtag_freq(data):
    hashtags_cnts = {}
    for datum in data:
        hashtags = datum['entities']['hashtags']
        for tag_st in hashtags:
            tag = tag_st['text'].lower()
            if tag in hashtags_cnts:
                hashtags_cnts[tag] += 1
            else:
                hashtags_cnts[tag] = 1

        #[{'text': 'Coronavirus', 'indices': [21, 33]}]
    sorted_cnts = {k: v for k, v in sorted(hashtags_cnts.items(), 
            key=lambda item: item[1], reverse=True)}
    print(sorted_cnts)
    with open('hashtags_cnts.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['hashtag', 'cnt'])
        for k, v in sorted_cnts.items():
            writer.writerow([k,v])

def get_loc_freq(data):
    loc_cnts = {}
    for datum in data:
        loc = datum['bio_location'] #['country']
        if loc == None: 
            continue
        print(loc)
        #loc = '%s, %s' % (loc['full_name'], loc['country']) 
        #loc = loc['country']
        if loc in loc_cnts:
            loc_cnts[loc] += 1
        else:
            loc_cnts[loc] = 1

    sorted_cnts = {k: v for k, v in sorted(loc_cnts.items(), 
            key=lambda item: item[1], reverse=True)}
    print(sorted_cnts)
    with open('place_cnts.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['place', 'cnt'])
        for k, v in sorted_cnts.items():
            writer.writerow([k,v])

def get_lang_freq(data):
    lang_cnts = {}
    for datum in data:
        lang = datum['lang'] #['country']
        if lang == None: 
            continue
        if lang in lang_cnts:
            lang_cnts[lang] += 1
        else:
            lang_cnts[lang] = 1

    sorted_cnts = {k: v for k, v in sorted(lang_cnts.items(), 
            key=lambda item: item[1], reverse=True)}
    print(sorted_cnts)
    with open('lang_cnts.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['lang', 'cnt'])
        for k, v in sorted_cnts.items():
            writer.writerow([k,v])

def get_stopwords(filename):
    with open(filename, 'r') as f:
        stopwords = []
        for line in f:
            w = line.strip()
            stopwords.append(w)
        return stopwords

def get_word_freq(data):
    stopwords = get_stopwords('stopwords.txt')
    texts = get_texts(input_file)
    word_cnts = {}
    for text in texts:
        #text = datum['text']
        tokens = text.split(' ')
        for tok in tokens:
            tok = tok.strip()
            if tok in stopwords:
                continue
            if tok in word_cnts:
                word_cnts[tok] += 1
            else:
                word_cnts[tok] = 1
    sorted_cnts = {k: v for k, v in sorted(word_cnts.items(), 
            key=lambda item: item[1], reverse=True)}
    print(sorted_cnts)
    with open('token_cnts_nostopwords.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['token', 'cnt'])
        for k, v in sorted_cnts.items():
            writer.writerow([k,v])


if __name__ == '__main__':
    input_file = 'covid.jsonl'
    data = parse(input_file)
    #get_hashtag_freq(data)
    #get_loc_freq(data)
    #get_lang_freq(data)
    get_word_freq(input_file)
