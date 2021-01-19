import json
import csv
import sys
import pickle
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from topic_modeling import get_texts

def parse(input_file):
    data = []
    with open(input_file) as rf:
        for line in rf:
            data_json = json.loads(line)
            data.append(data_json)
    return data

def run(texts, num_topics, method='NMF'):
    # parse JSON

    df = pd.DataFrame(texts, columns=['Text'])
    print(df)
    # NMF is able to use tf-idf
    #    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(df['Text'])
    print('tfidf feature table size: %s', tfidf.shape)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
#    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf_vectorizer = CountVectorizer(stop_words='english')
    tf = tf_vectorizer.fit_transform(df['Text'])
    print('tf feature table size: %s', tfidf.shape)
    tf_feature_names = tf_vectorizer.get_feature_names()

    model = None
    feature_names = None
    if method == 'NMF':  
        model = NMF(n_components=num_topics).fit(tfidf)
        feature_names = tfidf_feature_names
    #   model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(data)
    elif method == 'LDA':
        model = LatentDirichletAllocation(n_components=num_topics).fit(tf)
    #   model = LatentDirichle tAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(data)
        feature_names = tf_feature_names
    
    output_file = '%s_%d.txt' % (method, num_topics)
    display_topics(model, feature_names, 50, output_file)

    return model


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
