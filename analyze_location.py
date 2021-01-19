import json
import gzip
import joblib
from pathlib import Path
import csv
import sys
import pickle
from collections import defaultdict, OrderedDict
import preprocessor as p
from datetime import datetime

from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer


def binning(input_prefix, input_num):
    cnt = 0
    data = []
    for i in range(input_num):
        input_file = '%s_%d.txt' % (input_prefix, i)
        with open(input_file, 'r') as rf:
            for line in rf:
                # date conversion
                data_json = json.loads(line)
                time_expr = data_json['created_at']
                text = data_json['full_text']
                clean_text = p.clean(text)
                data.append(clean_text)
                cnt += 1
    return data

def get_topic_dist(data, method, model_file, output_file):
    # load LDA/NMF model
    model, feature_names, voca = joblib.load(model_file)
   

    # for each bin, get mean probability
    corpus_topic_dist = None

    doc_topic_dist = None
    if method == 'LDA':
        tf_vectorizer = CountVectorizer(stop_words='english', 
                vocabulary=voca)
        tf = tf_vectorizer.fit_transform(data)
        doc_topic_dist = model.transform(tf)
    elif method == 'NMF':
        tfidf_transformer = TfidfTransformer()
        loaded_vec = CountVectorizer(stop_words='english',
                vocabulary=voca)
        tfidf = tfidf_transformer.fit_transform(loaded_vec.fit_transform(data))
        doc_topic_dist = model.transform(tfidf)
    else:
        print('wrong method:', method)
    
    # compute average
    avg = doc_topic_dist.mean(axis=0)
    corpus_topic_dist = avg.tolist()

    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(corpus_topic_dist)


def run(input_prefix, input_num, method, model_file, output_file):
    data = binning(input_prefix, input_num)
    
    get_topic_dist(data, method, model_file, output_file)

if __name__ == '__main__':
    input_loc = 'canada'
    input_prefix = 'tweets_%s_en' % input_loc# _0.txt'
    method = 'LDA'
    model_loc = 'canada_us'
    model_file = '%s_%s_20.model' % (model_loc, method) # model could be different from input e.g., doesn't need to be canada
    input_num = 5 # 5 files
    output_file = 'loc_%s.csv' % input_loc 
    run(input_prefix, input_num, method, model_file, output_file)

